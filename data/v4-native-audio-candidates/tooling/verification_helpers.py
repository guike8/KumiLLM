#!/usr/bin/env python3
"""Offline Studio-only tokenization/loss gate for reviewed recovery candidates.

This script never trains, downloads models, reads heldout data, or changes MLX.
It needs the Studio's existing transformers, mlx, mlx-lm and Qwen tokenizer.
It intentionally imports no workspace helper, so this directory is portable.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
CAP = 32768
RESEARCH_CAP = 65536


class RenderError(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_json(text):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "duplicate_json_key")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("nonfinite_json")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def read_candidates(data_path, manifest_path):
    raw = data_path.read_bytes()
    manifest = load_json(manifest_path.read_text())
    require(manifest["format_version"] == 1 and manifest["kind"] == "reviewed_recovery_candidates", "unsupported_candidate_manifest")
    require(digest(raw) == manifest["candidate_file_sha256"], "candidate_file_hash_mismatch")
    require(manifest["exact_native_schema_capture_missing"] == 0, "exact_native_tool_schema_capture_missing")
    require(manifest["max_seq_length"] == CAP and manifest["mask_prompt_required"] is True, "unsafe_candidate_contract")
    records = [load_json(line) for line in raw.decode().splitlines() if line.strip()]
    require(len(records) == manifest["record_count"] == len(manifest["records"]), "candidate_count_mismatch")
    require(records, "no_candidates")
    for record, receipt in zip(records, manifest["records"]):
        require(digest(canonical(record).encode()) == receipt["record_sha256"], "candidate_record_hash_mismatch")
        for field, value in (("tools_sha256", record["tools"]), ("prefix_sha256", record["messages"][:-1]),
                             ("target_sha256", record["messages"][-1])):
            require(digest(canonical(value).encode()) == receipt[field], field + "_mismatch")
        require(record["metadata"]["source_sha256"] == receipt["source_sha256"] == manifest["source_sha256"], "candidate_source_mismatch")
        require(record["metadata"]["call_index"] == receipt["call_index"], "candidate_call_index_mismatch")
        require(record["metadata"]["mask_all_history_required"] is True
                and record["metadata"]["train_on_final_assistant_only_required"] is True, "candidate_history_mask_not_required")
        require(record["messages"][-1]["role"] == "assistant", "candidate_not_final_assistant")
        names = []
        for tool in record["tools"]:
            require(tool["type"] == "function" and isinstance(tool["function"]["parameters"], dict), "candidate_tool_schema_missing")
            names.append(tool["function"]["name"])
        require(len(names) == len(set(names)), "candidate_duplicate_tool_schema")
        target = record["messages"][-1]
        if receipt["target_kind"] == "tool_only":
            require(target.get("content") in (None, "") and target.get("tool_calls"), "candidate_not_tool_only")
            require(all(c["function"]["name"] in names and isinstance(c["function"]["arguments"], dict)
                        for c in target["tool_calls"]), "candidate_inactive_or_unparsed_tool")
        elif receipt["target_kind"] in ("clarification", "checked_public_response"):
            require(not target.get("tool_calls") and isinstance(target.get("content"), str) and target["content"], "candidate_not_text_only")
            terminal = receipt.get("terminal_validation", {})
            require(terminal.get("contract_validated") is True and terminal.get("native_response_exact_match") is True,
                    "text_terminal_contract_or_observed_response_missing")
            if receipt["target_kind"] == "clarification":
                require(terminal.get("intent_reviewed") is True and terminal.get("status") == "awaiting_user", "clarification_intent_review_missing")
        else:
            raise ValueError("unsupported_candidate_target_kind")
    return records, manifest


def check_tool_render(suffix, wanted):
    bodies = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", suffix, flags=re.S)
    require(len(bodies) == len(wanted), "qwen_rendered_tool_count_mismatch")
    for body, expected in zip(bodies, wanted):
        if body.startswith("{"):
            require(load_json(body) == expected, "qwen_rendered_json_arguments_not_exact")
            continue
        # Qwen Coder templates may use function/parameter delimiters instead of
        # JSON call envelopes. Compare every typed parameter, without executing
        # strings or accepting undocumented formats.
        function = re.fullmatch(r"<function=([^>\n]+)>\n?(.*?)\n?</function>", body, flags=re.S)
        require(function is not None and function[1] == expected["name"], "unsupported_or_mismatched_qwen_function")
        parameter_body = function[2]
        matches = list(re.finditer(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", parameter_body, flags=re.S))
        require(re.sub(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", "", parameter_body, flags=re.S).strip() == "",
                "unexpected_qwen_parameter_text")
        require(len(matches) == len(expected["arguments"]), "qwen_parameter_count")
        seen = set()
        for match in matches:
            name, value = match[1], match[2]
            require(name in expected["arguments"] and name not in seen, "qwen_unknown_or_duplicate_parameter")
            seen.add(name)
            original = expected["arguments"][name]
            if isinstance(original, str):
                require(value == original, "qwen_string_parameter_not_exact")
            else:
                # The Studio's captured Coder template uses Jinja |string for
                # nonstrings, yielding Python-style dict/list representations.
                # literal_eval parses those values without executing code.
                try:
                    parsed = load_json(value)
                except (ValueError, json.JSONDecodeError):
                    parsed = ast.literal_eval(value)

                def json_value(item):
                    if isinstance(item, dict):
                        return all(isinstance(k, str) and json_value(v) for k, v in item.items())
                    if isinstance(item, list):
                        return all(json_value(v) for v in item)
                    return item is None or type(item) in (str, bool, int, float)

                require(json_value(parsed) and canonical(parsed) == canonical(original), "qwen_typed_parameter_not_exact")


def inspect_render(record, receipt, tokenizer, chat_dataset_class, max_seq_length, *, hard_cap=CAP):
    options = {"tools": record["tools"], "tokenize": True, "return_dict": False}
    prompt = tokenizer.apply_chat_template(record["messages"][:-1], add_generation_prompt=True, **options)
    full = tokenizer.apply_chat_template(record["messages"], add_generation_prompt=False, **options)
    require(isinstance(full, list) and isinstance(prompt, list)
            and all(type(token) is int for token in prompt + full), "tokenizer_ids_not_flat_integers")
    counts = {"full_tokens": len(full), "prompt_tokens": len(prompt), "target_tokens": len(full)-len(prompt),
              "max_seq_length": max_seq_length, "production_cap": CAP,
              "over_production_cap_tokens": max(0, len(full)-CAP),
              "over_selected_cap_tokens": max(0, len(full)-max_seq_length),
              "prefix_exact": full[:len(prompt)] == prompt}
    for condition, reason in ((counts["prefix_exact"], "prompt_is_not_exact_full_token_prefix"),
                              (len(prompt) > 0, "empty_prompt"), (len(full) > len(prompt), "empty_final_target"),
                              (1 <= max_seq_length <= hard_cap, "requested_cap_not_authorized")):
        if not condition:
            raise RenderError(reason, counts)
    # This executes the installed trainer dataset path, not a reimplementation.
    native_tokens, native_offset = chat_dataset_class([record], tokenizer, mask_prompt=True).process(record)
    counts.update(native_mask_offset=native_offset, native_full_tokens=len(native_tokens),
                  native_chatdataset_exact=native_tokens == full and native_offset == len(prompt))
    if not counts["native_chatdataset_exact"]:
        raise RenderError("native_mlx_chatdataset_mismatch", counts)
    suffix = tokenizer.decode(full[len(prompt):], skip_special_tokens=False)
    if receipt["target_kind"] == "tool_only":
        wanted = [{"name": call["function"]["name"], "arguments": call["function"]["arguments"]}
                  for call in record["messages"][-1]["tool_calls"]]
        check_tool_render(suffix, wanted)
    else:
        require(record["messages"][-1]["content"] in suffix, "rendered_text_target_missing")
    counts["rendered_target_exact"] = True
    if len(full) > max_seq_length:
        raise RenderError("full_context_exceeds_selected_sequence_cap", counts)
    return {**counts,
            "native_mask_offset": native_offset, "full_token_ids_sha256": digest(canonical(full).encode()),
            "prefix_exact": True, "native_chatdataset_exact": True, "rendered_target_exact": True,
            "no_truncation": True}


def probe_loss(loss_function):
    """Observe native gradients at every position of a tiny padded two-row batch.

    Each position has its own independent probe logit. A nonzero gradient
    therefore identifies exactly which target positions the loss supervises.
    No model weights are loaded and no training step occurs.
    """
    import mlx.core as mx

    batch = mx.ones((2, 10), dtype=mx.int32)
    lengths = mx.array([[3, 7], [2, 9]], dtype=mx.int32)

    class Probe:
        def __init__(self, weights):
            self.weights = weights

        def __call__(self, inputs):
            return mx.stack([self.weights, mx.zeros_like(self.weights)], axis=-1)

    def objective(weights):
        return loss_function(Probe(weights), batch, lengths)[0]

    weights = mx.zeros((2, 9))
    gradient = mx.grad(objective)(weights)
    _, count = loss_function(Probe(weights), batch, lengths)
    mx.eval(gradient, count)
    active = [[index + 1 for index, value in enumerate(row) if abs(value) > 1e-9] for row in gradient.tolist()]
    expected = [list(range(3, 7)), list(range(2, 9))]
    return {"ok": active == expected and int(count.item()) == 11, "actual_target_positions": active,
            "expected_target_positions": expected, "supervised_token_count": int(count.item()),
            "history_and_padding_have_zero_gradient": active == expected}


def tokenizer_fingerprint(path):
    names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
             "chat_template.jinja", "vocab.json", "merges.txt", "config.json")
    files = {name: digest((path / name).read_bytes()) for name in names if (path / name).is_file()}
    templates = path / "chat_templates"
    if templates.is_dir():
        files.update({str(p.relative_to(path)): digest(p.read_bytes()) for p in sorted(templates.glob("*.jinja"))})
    require("tokenizer_config.json" in files and ("tokenizer.json" in files or "vocab.json" in files), "tokenizer_files_incomplete")
    return files


def run(args):
    output = args.output.resolve()
    require(output.is_relative_to(HERE), "output_must_remain_in_recovery_directory")
    require(not output.exists() or not any(output.iterdir()), "use_a_fresh_output_directory")
    output.mkdir(parents=True, exist_ok=True)
    report = {"format_version": 1, "ok": False, "max_seq_length": args.max_seq_length,
              "current_production_cap": CAP, "research_context_only": args.research_context_only,
              "local_files_only": True, "trust_remote_code": False, "mask_prompt_required": True,
              "loss_contract": args.loss_contract, "tokenizer_masking_verified": False,
              "ready_for_training_ingestion": False, "accepted_records": 0, "records": [], "errors": []}
    accepted = []
    try:
        effective_cap = RESEARCH_CAP if args.research_context_only else CAP
        require(1 <= args.max_seq_length <= effective_cap,
                "sequence_cap_requires_research_context_only_above_32768")
        require(args.model.is_dir(), "model_must_be_an_existing_local_directory")
        records, manifest = read_candidates(args.candidates, args.candidate_manifest)
        report["candidate_file_sha256"] = manifest["candidate_file_sha256"]
        report["source_sha256"] = manifest["source_sha256"]
        report["model"] = str(args.model.resolve())
        report["tokenizer_files_sha256"] = tokenizer_fingerprint(args.model)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoTokenizer
        from mlx_lm.tuner.datasets import ChatDataset
        from mlx_lm.tuner.trainer import default_loss
        tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()), local_files_only=True, trust_remote_code=False)
        report["versions"] = {name: importlib.metadata.version(name) for name in ("transformers", "mlx", "mlx-lm")}
        report["python"] = sys.version.split()[0]
        report["chat_template_sha256"] = digest(canonical(tokenizer.chat_template).encode())
        report["native_chatdataset_source_sha256"] = digest(inspect.getsource(ChatDataset).encode())
        report["native_loss_source_sha256"] = digest(inspect.getsource(default_loss).encode())
        report["native_loss_probe"] = probe_loss(default_loss)
        selected_loss = default_loss
        if args.loss_contract == "recovery-final-target":
            from final_target_loss import loss
            selected_loss = loss
        report["selected_loss_source_sha256"] = digest(inspect.getsource(selected_loss).encode())
        report["selected_loss_probe"] = probe_loss(selected_loss)
        for record, receipt in zip(records, manifest["records"]):
            result = {"call_index": receipt["call_index"], "target_kind": receipt["target_kind"], "ok": False}
            try:
                result.update(inspect_render(record, receipt, tokenizer, ChatDataset, args.max_seq_length, hard_cap=effective_cap))
                require(report["selected_loss_probe"]["ok"], "selected_loss_supervises_history_or_padding")
                result["ok"] = True
                prepared = copy.deepcopy(record)
                prepared["metadata"].update(tokenizer_masking_verified=True,
                    ready_for_training_ingestion=not args.research_context_only,
                    verified_candidate_file_sha256=manifest["candidate_file_sha256"], max_seq_length=args.max_seq_length,
                    required_loss_contract=args.loss_contract, required_loss_source_sha256=report["selected_loss_source_sha256"],
                    tokenizer_chat_template_sha256=report["chat_template_sha256"])
                accepted.append(prepared)
            except Exception as error:
                result.update(getattr(error, "diagnostics", {}))
                result["error"] = f"{type(error).__name__}: {error}"
            report["records"].append(result)
        report["accepted_records"] = 0 if args.research_context_only else len(accepted)
        report["withheld_records"] = len(records) - report["accepted_records"]
        report["research_passed_records"] = len(accepted) if args.research_context_only else 0
        report["ok"] = bool(accepted) and len(accepted) == len(records)
        report["tokenizer_masking_verified"] = bool(accepted)
        report["ready_for_training_ingestion"] = bool(accepted) and not args.research_context_only
        report["verification_scope"] = "future_context_diagnostic_only" if args.research_context_only else "prepared_file_only"
        if accepted and not args.research_context_only:
            prepared_dir = output / "prepared-data"
            prepared_dir.mkdir()
            data = "".join(canonical(record) + "\n" for record in accepted)
            (prepared_dir / "train.jsonl").write_text(data)
            report["prepared_train_sha256"] = digest(data.encode())
            report["required_training_configuration"] = {"mask_prompt": True, "max_seq_length": args.max_seq_length,
                "loss_contract": args.loss_contract, "loss_source_sha256": report["selected_loss_source_sha256"],
                "template_kwargs": {}, "test": False,
                "note": "The selected loss must be used by the new trainer. This script does not start or change training."}
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    report["verifier_script_sha256"] = digest(Path(__file__).read_bytes())
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, default=HERE / "artifacts/candidates/train.jsonl")
    parser.add_argument("--candidate-manifest", type=Path, default=HERE / "artifacts/candidates/manifest.json")
    parser.add_argument("--output", type=Path, default=HERE / "artifacts/studio-tokenizer-check")
    parser.add_argument("--max-seq-length", type=int, default=CAP)
    parser.add_argument("--loss-contract", choices=("native", "recovery-final-target"), default="native")
    parser.add_argument("--research-context-only", action="store_true",
                        help="Allow diagnostics up to 65536; never emits trainable data or grants current-trainer admission")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Check MLX final-assistant masks using an already-downloaded HF tokenizer.

All model/tokenizer loading is local-only. No network calls are performed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from validate_dataset import reject_nonfinite, validate_record


def inspect_record(record: dict, tokenizer, max_seq_length: int, template_kwargs: dict | None = None) -> dict:
    errors = validate_record(record)
    if errors:
        raise ValueError("structural validation failed: " + "; ".join(errors))
    # Mirror MLX ChatDataset explicitly: recent Transformers versions can
    # otherwise return BatchEncoding instead of the flat IDs used for masking.
    options = {"tools": record["tools"], "tokenize": True, "return_dict": False, **(template_kwargs or {})}
    prompt = tokenizer.apply_chat_template(record["messages"][:-1], add_generation_prompt=True, **options)
    full = tokenizer.apply_chat_template(record["messages"], add_generation_prompt=False, **options)
    if not isinstance(prompt, list) or not isinstance(full, list) or not all(isinstance(token, int) for token in prompt + full):
        raise ValueError("tokenizer did not return flat token-ID lists")
    if full[:len(prompt)] != prompt:
        divergence = next((index for index, (left, right) in enumerate(zip(prompt, full)) if left != right), min(len(prompt), len(full)))
        raise ValueError(f"rendered prompt tokens are not an exact prefix of the full example (first divergence at token {divergence})")
    target_count = len(full) - len(prompt)
    if target_count <= 0:
        raise ValueError("supervised target has no tokens")
    if len(full) > max_seq_length:
        raise ValueError(f"example has {len(full)} tokens, exceeding max_seq_length={max_seq_length}; truncation could remove the target")
    return {"full": len(full), "prompt": len(prompt), "target": target_count}


def check_files(data: Path, tokenizer, max_seq_length: int, template_kwargs: dict | None = None, max_errors: int = 30) -> dict:
    errors: list[str] = []
    error_count = 0
    splits: dict[str, dict] = {}

    def add_error(error: str):
        nonlocal error_count
        error_count += 1
        if len(errors) < max_errors:
            errors.append(error)

    for split in ("train", "valid", "test"):
        path = data / f"{split}.jsonl"
        if not path.exists():
            if split == "train":
                add_error("train.jsonl is missing")
            continue
        statistics = {"records": 0, "checked": 0, "tokens": {key: {"min": None, "max": None} for key in ("full", "prompt", "target")}}
        splits[split] = statistics
        try:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    statistics["records"] += 1
                    try:
                        record = json.loads(line, parse_constant=reject_nonfinite)
                        counts = inspect_record(record, tokenizer, max_seq_length, template_kwargs)
                    except Exception as error:
                        add_error(f"{split}:{line_number}: {type(error).__name__}: {error}")
                        continue
                    statistics["checked"] += 1
                    for key, value in counts.items():
                        previous = statistics["tokens"][key]
                        previous["min"] = value if previous["min"] is None else min(value, previous["min"])
                        previous["max"] = value if previous["max"] is None else max(value, previous["max"])
        except (OSError, UnicodeError) as error:
            add_error(f"{split}: cannot read UTF-8 JSONL ({type(error).__name__})")
        if statistics["records"] == 0:
            add_error(f"{split}: no examples found")
    return {"ok": error_count == 0, "max_seq_length": max_seq_length, "splits": splits, "error_count": error_count, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Existing local directory with tokenizer files")
    parser.add_argument("--data", required=True, type=Path, help="Directory containing train.jsonl and optional valid/test.jsonl")
    parser.add_argument("--max-seq-length", required=True, type=int)
    parser.add_argument("--chat-template-kwargs", default="{}", help="JSON object of deployment-matching template options")
    args = parser.parse_args()
    if not args.model.is_dir() or not args.data.is_dir():
        parser.error("--model and --data must be existing local directories")
    if args.max_seq_length < 1:
        parser.error("--max-seq-length must be positive")
    try:
        template_kwargs = json.loads(args.chat_template_kwargs)
    except ValueError:
        parser.error("--chat-template-kwargs must be a JSON object")
    if not isinstance(template_kwargs, dict) or set(template_kwargs).intersection({"tools", "tokenize", "add_generation_prompt", "return_dict", "return_tensors"}):
        parser.error("template options must be an object and may not override tools, tokenization or generation boundaries")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()), local_files_only=True, trust_remote_code=False)
        report = check_files(args.data, tokenizer, args.max_seq_length, template_kwargs)
        report["model"] = str(args.model.resolve())
        report["local_files_only"] = True
        report["trust_remote_code"] = False
    except Exception as error:
        report = {"ok": False, "error": {"type": type(error).__name__, "message": str(error)}, "local_files_only": True, "trust_remote_code": False}
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

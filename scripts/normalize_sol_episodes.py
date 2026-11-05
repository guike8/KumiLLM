#!/usr/bin/env python3
"""Admit clean, execution-verified Sol Ultra Kumi demonstrations into corpus v2.

The teacher identity is declared by the collection orchestrator. This gate
checks its exact model/effort/transport tuple and preserves that provenance;
it does not independently authenticate a model provider.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile
from urllib.parse import urlsplit

SPLITS = ("train", "valid", "test")
SCOPE = "executed_synthetic_kumi"
TEACHER = "gpt-6.1-sol-ultra"
TEACHER_MODEL = "gpt-6.1-sol"
TEACHER_REASONING_EFFORT = "ultra"
TEACHER_TRANSPORT = "codex_agent_file_queue"
CORPUS_VERSION = "v2-sol-ultra-pilot"
TEACHER_PROVENANCE = {
    "teacher": TEACHER,
    "teacher_model": TEACHER_MODEL,
    "teacher_reasoning_effort": TEACHER_REASONING_EFFORT,
    "teacher_transport": TEACHER_TRANSPORT,
}
PRIVATE_REASONING_FIELDS = frozenset(("analysis", "reasoning", "chain_of_thought", "private_reasoning", "thoughts"))
REFERENCE_FIELDS = ("reference_manifest_sha256", "reference_source_ids")


def default_validator_path():
    return Path(__file__).resolve().parent / "validate_dataset.py"


class Rejected(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise Rejected(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def reject_constant(value):
    raise Rejected("nonfinite_json_number")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def load_json(text):
    def finite_float(value):
        parsed = float(value)
        require(math.isfinite(parsed), "nonfinite_json_number")
        return parsed
    return json.loads(text, parse_constant=reject_constant, parse_float=finite_float, object_pairs_hook=unique_object)


def identifier(value, label):
    require(isinstance(value, str) and bool(value.strip()), label + "_missing_or_invalid")
    return value


def checked_corpus_version(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", value) is not None,
            "invalid_corpus_version")
    return value


def valid_source_id(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,127}", value) is not None


def valid_sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9A-Fa-f]{64}", value) is not None


def https_source_url(value):
    if not isinstance(value, str) or any(character.isspace() for character in value):
        return False
    try:
        url = urlsplit(value)
        return url.scheme == "https" and bool(url.hostname) and url.username is None and url.password is None
    except ValueError:
        return False


def load_reference_manifest(path):
    """Keep only source identities/provenance; a bad manifest quarantines its episodes.

    The actual input bytes, rather than a reserialized document, define its hash.
    Errors are fixed labels so private input paths or article text cannot enter
    the output manifest through an exception message.
    """
    reference = {"sha256": None, "sources": {}, "error": None}
    try:
        path = Path(path)
        require(not path.is_symlink(), "reference_manifest_symlink_not_allowed")
        require(path.is_file(), "reference_manifest_missing_or_not_file")
        raw = path.read_bytes()
        reference["sha256"] = digest(raw)
        try:
            document = load_json(raw.decode("utf-8"))
        except (ValueError, UnicodeError):
            raise Rejected("invalid_reference_manifest_json") from None
        require(isinstance(document, dict), "reference_manifest_must_be_object")
        require(type(document.get("format_version")) is int and document["format_version"] == 1,
                "unsupported_reference_manifest_format")
        sources = document.get("sources")
        require(isinstance(sources, list) and sources, "reference_manifest_sources_missing")
        for source in sources:
            require(isinstance(source, dict), "invalid_reference_source")
            source_id = source.get("id")
            require(valid_source_id(source_id), "invalid_reference_source_id")
            require(source_id not in reference["sources"], "duplicate_reference_source_id")
            reference["sources"][source_id] = {key: source.get(key) for key in ("url", "source_hash", "status")}
    except Rejected as error:
        reference["error"] = str(error)
        reference["sources"] = {}
    except OSError:
        reference["error"] = "reference_manifest_unreadable"
        reference["sources"] = {}
    return reference


def check_references(episode, reference):
    if reference is None:
        require(not any(key in episode for key in REFERENCE_FIELDS), "reference_manifest_not_provided")
        return {}
    require(reference["error"] is None, reference["error"])
    declared_hash = episode.get("reference_manifest_sha256")
    require(valid_sha256(declared_hash), "missing_or_invalid_reference_manifest_sha256")
    require(declared_hash.lower() == reference["sha256"], "reference_manifest_sha256_mismatch")
    source_ids = episode.get("reference_source_ids")
    require(isinstance(source_ids, list) and source_ids, "reference_source_ids_missing_or_empty")
    require(all(valid_source_id(source_id) for source_id in source_ids), "invalid_episode_reference_source_id")
    require(len(set(source_ids)) == len(source_ids), "duplicate_episode_reference_source_id")
    for source_id in source_ids:
        require(source_id in reference["sources"], "unknown_reference_source_id")
        source = reference["sources"][source_id]
        require(source["status"] == "success", "reference_source_not_successful")
        require(https_source_url(source["url"]), "reference_source_url_must_be_https")
        require(valid_sha256(source["source_hash"]), "invalid_reference_source_hash")
    return {"reference_manifest_sha256": declared_hash, "reference_source_ids": list(source_ids)}


def check_teacher_provenance(episode):
    for key, expected in TEACHER_PROVENANCE.items():
        require(episode.get(key) == expected, "wrong_or_missing_" + key)
    require(not PRIVATE_REASONING_FIELDS.intersection(episode), "private_reasoning_field_not_allowed")
    require(not episode.get("teacher_usage"), "teacher_usage_unavailable_for_file_queue")


def check_unreported_usage(parts):
    """The Codex file queue supplies no provider token-accounting receipt."""
    for part in parts:
        if not isinstance(part, dict) or part.get("type") != "finish":
            continue
        usage = part.get("usage", {})
        require(isinstance(usage, dict), "invalid_teacher_usage")
        require(set(usage).issubset({"inputTokens", "outputTokens", "raw"}), "unsupported_teacher_usage")
        for key in ("inputTokens", "outputTokens"):
            counters = usage.get(key, {})
            require(isinstance(counters, dict), "invalid_teacher_usage")
            require(all(value is None for value in counters.values()), "teacher_usage_unavailable_for_file_queue")
        require(usage.get("raw") is None, "teacher_usage_unavailable_for_file_queue")


def no_provider_options(value):
    require(not value.get("providerOptions") and not value.get("provider_options"), "unsupported_provider_options")
    require(not value.get("providerMetadata") and not value.get("provider_metadata"), "unsupported_provider_metadata")


def result_error(value):
    """Reject explicit operational failures, while leaving returned tool schemas alone."""
    if isinstance(value, list):
        return any(result_error(item) for item in value)
    if not isinstance(value, dict):
        return False
    if value.get("ok") is False or value.get("isError") is True or value.get("is_error") is True:
        return True
    if any(value.get(key) not in (None, False, "", [], {}) for key in ("error", "errors", "refused")):
        return True
    if value.get("state") in ("refused", "failed", "cancelled", "canceled", "denied"):
        return True
    if value.get("status") in ("error", "failed", "cancelled", "canceled", "denied"):
        return True
    schema_keys = {"inputSchema", "parameters", "properties", "$defs", "definitions", "schema"}
    return any(result_error(child) for key, child in value.items() if key not in schema_keys)


def normalize_output(output):
    require(isinstance(output, dict), "invalid_tool_output")
    no_provider_options(output)
    kind = output.get("type")
    require(kind in ("text", "json", "content"), "tool_result_error_or_unsupported_output")
    if kind == "content":
        values = output.get("value")
        require(isinstance(values, list), "invalid_tool_content")
        strings = []
        for item in values:
            require(isinstance(item, dict) and item.get("type") == "text", "unsupported_tool_media")
            no_provider_options(item)
            require(isinstance(item.get("text"), str), "invalid_tool_text")
            # Multiple JSON documents cease to be valid JSON when joined. Check
            # each source block as well as the combined text before flattening.
            try:
                block = load_json(item["text"])
            except json.JSONDecodeError:
                block = None
            require(not result_error(block), "tool_result_reports_failure")
            strings.append(item["text"])
        text = "\n".join(strings)
        try:
            decoded = load_json(text)
        except json.JSONDecodeError:
            decoded = None
        require(not result_error(decoded), "tool_result_reports_failure")
        return text
    value = output.get("value")
    if kind == "json":
        require(not result_error(value), "tool_result_reports_failure")
        return canonical(value)
    require(isinstance(value, str), "invalid_tool_text")
    try:
        decoded = load_json(value)
    except json.JSONDecodeError:
        decoded = None
    require(not result_error(decoded), "tool_result_reports_failure")
    return value


def normalize_call(part, *, stream=False):
    no_provider_options(part)
    require(part.get("providerExecuted") in (None, False), "unsupported_provider_executed_call")
    call_id = identifier(part.get("toolCallId"), "tool_call_id")
    name = identifier(part.get("toolName"), "tool_name")
    arguments = part.get("input")
    if stream:
        require(isinstance(arguments, str), "stream_tool_input_must_be_json_text")
        try:
            arguments = load_json(arguments)
        except ValueError as error:
            raise Rejected("invalid_stream_tool_input") from error
    require(isinstance(arguments, dict), "tool_arguments_must_be_object")
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def normalize_prompt(prompt):
    require(isinstance(prompt, list) and prompt, "missing_request_prompt")
    messages = []
    for message in prompt:
        require(isinstance(message, dict), "invalid_native_message")
        no_provider_options(message)
        role, content = message.get("role"), message.get("content")
        require(role in ("system", "user", "assistant", "tool"), "unsupported_native_role")
        if role == "system":
            require(isinstance(content, str), "invalid_system_text")
            messages.append({"role": role, "content": content})
            continue
        require(isinstance(content, list), "native_content_must_be_multipart")
        text, calls = [], []
        for part in content:
            require(isinstance(part, dict), "invalid_native_part")
            no_provider_options(part)
            kind = part.get("type")
            if kind == "text" and role != "tool":
                require(isinstance(part.get("text"), str), "invalid_native_text")
                text.append(part["text"])
            elif kind == "tool-call" and role == "assistant":
                calls.append(normalize_call(part))
            elif kind == "tool-result" and role == "tool":
                messages.append({"role": "tool", "tool_call_id": identifier(part.get("toolCallId"), "tool_result_id"),
                                 "name": identifier(part.get("toolName"), "tool_result_name"),
                                 "content": normalize_output(part.get("output"))})
            else:
                raise Rejected("unsupported_native_part:" + str(kind))
        if role != "tool":
            normalized = {"role": role, "content": "".join(text)}
            if calls:
                normalized["tool_calls"] = calls
            messages.append(normalized)
    return messages


def normalize_tools(tools):
    require(isinstance(tools, list), "request_tools_must_be_list")
    normalized = []
    names = set()
    for tool in tools:
        require(isinstance(tool, dict) and tool.get("type", "function") == "function", "unsupported_tool_definition")
        no_provider_options(tool)
        name = identifier(tool.get("name"), "tool_definition_name")
        require(name not in names, "duplicate_tool_definition")
        names.add(name)
        require(isinstance(tool.get("inputSchema"), dict), "missing_tool_input_schema")
        function = {"name": name, "parameters": tool["inputSchema"]}
        for field in ("description", "inputExamples", "strict"):
            if tool.get(field) is not None:
                function[field] = tool[field]
        normalized.append({"type": "function", "function": function})
    return normalized


def normalize_response(parts):
    require(isinstance(parts, list) and parts, "missing_stream_response")
    text, calls, finishes = [], [], []
    streamed = {}
    for index, part in enumerate(parts):
        require(isinstance(part, dict), "invalid_stream_part")
        no_provider_options(part)
        kind = part.get("type")
        if kind == "text-delta":
            require(isinstance(part.get("delta"), str), "invalid_text_delta")
            text.append(part["delta"])
        elif kind == "tool-call":
            calls.append(normalize_call(part, stream=True))
        elif kind == "finish":
            require(index == len(parts) - 1, "finish_is_not_last_stream_part")
            finishes.append(part)
        elif kind == "tool-input-start":
            call_id = identifier(part.get("id"), "stream_input_id")
            require(call_id not in streamed, "duplicate_stream_input_start")
            streamed[call_id] = {"name": identifier(part.get("toolName"), "stream_tool_name"), "input": "", "ended": False}
        elif kind == "tool-input-delta":
            entry = streamed.get(part.get("id"))
            require(entry is not None and not entry["ended"] and isinstance(part.get("delta"), str), "orphan_stream_input_delta")
            entry["input"] += part["delta"]
        elif kind == "tool-input-end":
            entry = streamed.get(part.get("id"))
            require(entry is not None and not entry["ended"], "orphan_stream_input_end")
            entry["ended"] = True
        elif kind == "stream-start":
            require(not part.get("warnings"), "stream_warnings")
        elif kind not in ("text-start", "text-end", "response-metadata"):
            raise Rejected("error_or_unsupported_stream_part:" + str(kind))
    require(len(finishes) == 1, "missing_or_duplicate_finish")
    reason = finishes[0].get("finishReason", {}).get("unified")
    require(reason in ("stop", "tool-calls"), "incomplete_or_error_finish")
    require((reason == "tool-calls") == bool(calls), "finish_reason_call_mismatch")
    by_id = {call["id"]: call for call in calls}
    require(len(by_id) == len(calls), "duplicate_response_tool_call_id")
    for call_id, entry in streamed.items():
        require(entry["ended"] and call_id in by_id, "unfinished_streamed_tool_input")
        call = by_id[call_id]
        require(entry["name"] == call["function"]["name"], "streamed_tool_name_mismatch")
        try:
            assembled = load_json(entry["input"])
        except ValueError as error:
            raise Rejected("invalid_streamed_tool_input") from error
        require(assembled == call["function"]["arguments"], "streamed_tool_input_mismatch")
    assistant = {"role": "assistant", "content": "".join(text)}
    if calls:
        assistant["tool_calls"] = calls
    require(bool(calls) or bool(assistant["content"].strip()), "empty_assistant_target")
    return assistant


def normalize_episode(episode, validate_record, *, corpus_version=CORPUS_VERSION, reference=None, tool_actions_only=False):
    corpus_version = checked_corpus_version(corpus_version)
    require(isinstance(episode, dict), "episode_must_be_object")
    require(type(episode.get("format_version")) is int and episode["format_version"] == 1, "unsupported_format_version")
    for key in ("episode_id", "project_id", "project_family_id", "source_revision"):
        identifier(episode.get(key), key)
    require(episode.get("scope") == SCOPE, "wrong_scope")
    check_teacher_provenance(episode)
    reference_metadata = check_references(episode, reference)
    require(episode.get("split") in SPLITS, "invalid_split")
    require(isinstance(episode.get("goals"), dict) and episode["goals"], "missing_goals")
    verification = episode.get("verification")
    require(isinstance(verification, dict) and verification.get("passed") is True, "verification_failed")
    checks = verification.get("checks")
    require(isinstance(checks, dict) and checks and all(value is True for value in checks.values()), "verification_checks_not_all_true")
    require(all(isinstance(verification.get(key), dict) and verification[key] for key in ("before", "after")), "missing_state_evidence")
    outcome = episode.get("outcome")
    require(isinstance(outcome, dict) and outcome.get("passed") is True and outcome.get("error") is None, "outcome_failed")
    calls = episode.get("calls")
    require(isinstance(calls, list) and calls, "missing_calls")
    records = []
    prefixes = []
    for index, call in enumerate(calls):
        require(isinstance(call, dict) and call.get("finished") is True, "unfinished_call")
        require(not PRIVATE_REASONING_FIELDS.intersection(call), "private_reasoning_field_not_allowed")
        request = call.get("request")
        require(isinstance(request, dict), "missing_call_request")
        require(not PRIVATE_REASONING_FIELDS.intersection(request), "private_reasoning_field_not_allowed")
        require(not request.get("headers"), "request_headers_not_allowed")
        no_provider_options(request)
        prefix = normalize_prompt(request.get("prompt"))
        tools = normalize_tools(request.get("tools"))
        assistant = normalize_response(call.get("response"))
        check_unreported_usage(call["response"])
        active = {tool["function"]["name"] for tool in tools}
        require(all(item["function"]["name"] in active for item in assistant.get("tool_calls", [])), "target_tool_not_in_active_schema")
        metadata = {key: episode[key] for key in ("episode_id", "project_id", "project_family_id", "scope", "source_revision", "split")}
        metadata.update(TEACHER_PROVENANCE)
        metadata.update(reference_metadata)
        metadata.update(corpus_version=corpus_version, call_index=index,
                        verification="executed_synthetic_state_predicate_passed", real_live_verified=False,
                        teacher_usage_reported=False)
        record = {"messages": prefix + [assistant], "tools": tools, "metadata": metadata}
        errors = validate_record(record)
        require(not errors, "normalized_record_invalid:" + (errors[0] if errors else ""))
        records.append(record)
        prefixes.append(prefix)
    for index, record in enumerate(records):
        for target in record["messages"][-1].get("tool_calls", []):
            seen = False
            for later in prefixes[index + 1:]:
                historical = [item for message in later for item in message.get("tool_calls", []) if item.get("id") == target["id"]]
                results = [message for message in later if message.get("role") == "tool" and message.get("tool_call_id") == target["id"]]
                require(all(item == target for item in historical), "tool_call_history_changed")
                if results:
                    require(bool(historical), "result_without_captured_target_history")
                    require(all(result.get("name") == target["function"]["name"] for result in results), "tool_result_name_mismatch")
                    seen = True
            require(seen, "missing_subsequent_successful_tool_result")
    # All native calls, results, state checks and response envelopes above must
    # pass before target selection. Dropping a prose target must never hide a
    # failed action or an invalid later response from whole-episode admission.
    if tool_actions_only:
        tool_records = []
        for record in records:
            target = record["messages"][-1]
            if target.get("tool_calls"):
                require(target.get("content") in (None, ""), "tool_action_target_contains_public_text")
                record["metadata"]["target_policy"] = "verified_tool_calls_only"
                tool_records.append(record)
        require(bool(tool_records), "episode_has_no_tool_action_targets")
        records = tool_records
    return records


def load_validator(path):
    spec = importlib.util.spec_from_file_location("kumi_existing_dataset_validator", path)
    require(spec is not None and spec.loader is not None, "validator_not_loadable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(callable(getattr(module, "validate_record", None)), "validator_missing_validate_record")
    return module


def collect(raw_dir, output_dir, validator_path, *, overwrite=False, corpus_version=CORPUS_VERSION, reference_manifest=None,
            tool_actions_only=False):
    corpus_version = checked_corpus_version(corpus_version)
    raw_dir, output_dir, validator_path = Path(raw_dir).resolve(), Path(output_dir).resolve(), Path(validator_path).resolve()
    require(raw_dir.is_dir(), "raw_directory_missing")
    require(raw_dir != output_dir and raw_dir not in output_dir.parents and output_dir not in raw_dir.parents, "raw_and_output_directories_must_not_overlap")
    require(not output_dir.exists() or output_dir.is_dir(), "output_is_not_directory")
    require(overwrite or not output_dir.exists() or not any(output_dir.iterdir()), "output_directory_not_empty_use_explicit_overwrite")
    reference_path = Path(reference_manifest) if reference_manifest is not None else None
    if reference_path is not None:
        require(reference_path.resolve() != output_dir and output_dir not in reference_path.resolve().parents,
                "reference_manifest_must_not_be_inside_output_directory")
    reference = load_reference_manifest(reference_path) if reference_path is not None else None
    validator = load_validator(validator_path)
    sources = sorted(raw_dir.rglob("*.json"))
    require(bool(sources), "no_raw_episode_json_files")
    entries = []
    for path in sources:
        entry = {"source": str(path.relative_to(raw_dir)), "reasons": [], "records": []}
        try:
            require(not path.is_symlink(), "source_symlink_not_allowed")
            raw = path.read_bytes()
            entry.update(source_sha256=digest(raw), source_bytes=len(raw))
            episode = load_json(raw.decode("utf-8"))
            require(isinstance(episode, dict), "episode_must_be_object")
            entry["identity"] = {key: episode.get(key) for key in ("episode_id", "project_id", "project_family_id", "split", "source_revision", *TEACHER_PROVENANCE)}
            entry["records"] = normalize_episode(episode, validator.validate_record, corpus_version=corpus_version,
                                                 reference=reference, tool_actions_only=tool_actions_only)
        except (Rejected, ValueError, UnicodeError, TypeError, KeyError) as error:
            entry["reasons"].append(str(error) if isinstance(error, Rejected) else "invalid_raw_episode:" + type(error).__name__)
        entries.append(entry)
    groups = defaultdict(list)
    examples = defaultdict(list)
    for index, entry in enumerate(entries):
        identity = entry.get("identity", {})
        for key in ("episode_id", "project_id", "project_family_id"):
            value = identity.get(key)
            if isinstance(value, str) and value.strip():
                groups[(key, value)].append(index)
        for record in entry["records"]:
            examples[digest(canonical({key: record[key] for key in ("messages", "tools")}).encode())].append(index)
    for (key, _), members in groups.items():
        splits = {entries[index].get("identity", {}).get("split") for index in members}
        reason = "duplicate_episode_id" if key == "episode_id" and len(members) > 1 else "group_shared_across_splits:" + key if len(splits) > 1 else None
        if reason:
            for index in members:
                entries[index]["reasons"].append(reason)
    for members in examples.values():
        if len(members) > 1:
            for index in members:
                entries[index]["reasons"].append("duplicate_normalized_example")
    accepted = {split: [] for split in SPLITS}
    manifest_entries = []
    for entry in entries:
        reasons = sorted(set(entry["reasons"]))
        if not reasons:
            for record in entry["records"]:
                record["metadata"].update(raw_source=entry["source"], raw_source_sha256=entry["source_sha256"])
            accepted[entry["identity"]["split"]].extend(entry["records"])
        manifest_entries.append({key: value for key, value in entry.items() if key not in ("records", "reasons")} | {
            "status": "quarantined" if reasons else "accepted", "reasons": reasons,
            "accepted_records": 0 if reasons else len(entry["records"]),
        })
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".kumi-normalized-", dir=output_dir.parent) as temporary:
        stage = Path(temporary) / "dataset"
        stage.mkdir()
        output_hashes = {}
        for split in SPLITS:
            path = stage / (split + ".jsonl")
            path.write_text("".join(canonical(record) + "\n" for record in accepted[split]), encoding="utf-8")
            output_hashes[path.name] = digest(path.read_bytes())
        records_by_split = {split: len(records) for split, records in accepted.items()}
        empty_splits = [split for split, count in records_by_split.items() if count == 0]
        manifest = {"format_version": 2, "raw_format_version": 1, "corpus_version": corpus_version,
                    "scope": SCOPE, **TEACHER_PROVENANCE, "real_live_verified": False,
                    "teacher_usage_reported": False,
                    "teacher_provenance_check": "exact declared model/effort/transport; not independent provider authentication",
                    "normalizer_sha256": digest(Path(__file__).read_bytes()), "validator_sha256": digest(validator_path.read_bytes()),
                    "raw_directory": str(raw_dir), "raw_files_preserved": True,
                    "source_episodes": len(entries), "accepted_episodes": sum(not entry["reasons"] for entry in entries),
                    "quarantined_episodes": sum(bool(entry["reasons"]) for entry in entries),
                    "records_by_split": records_by_split,
                    "dataset_readiness": {"empty_splits": empty_splits,
                                          "all_training_validation_test_splits_nonempty": not empty_splits,
                                          "pilot_only": bool(empty_splits),
                                          "generalization_evaluation_established": False},
                    "output_sha256": output_hashes, "episodes": manifest_entries,
                    "limitations": ["Verification is execution against Kumi's synthetic bridge, not Ableton Live or audio verification.",
                                    "Teacher actions were supplied through a Codex-agent file queue; this is not a direct OpenAI API sampling log.",
                                    "Provider token usage and private reasoning are not supplied by this transport and are not training data.",
                                    "Train-only pilots have no validation/test coverage and must not be treated as a complete evaluation corpus.",
                                    "Successful teacher examples do not establish a competitive success rate; all held-out evaluation attempts must be scored separately.",
                                    "Token lengths, prompt masks and musical quality require separate checks."]}
        if tool_actions_only:
            manifest["target_policy"] = "verified_tool_calls_only"
            manifest["target_policy_note"] = (
                "Only verified tool-call targets with no accompanying public text are emitted. Text-only responses remain in raw "
                "evidence and any later prompt context; that context must remain loss-masked by the trainer.")
        if reference is not None:
            used_sources = {source_id for records in accepted.values() for record in records
                            for source_id in record["metadata"]["reference_source_ids"]}
            manifest.update(reference_manifest_sha256=reference["sha256"],
                            reference_sources={source_id: reference["sources"][source_id] for source_id in sorted(used_sources)})
            if reference["error"] is not None:
                manifest["reference_manifest_error"] = reference["error"]
        for entry in entries:
            if "source_sha256" not in entry:
                continue  # Rejected links were deliberately never opened or hashed.
            source = raw_dir / entry["source"]
            require(not source.is_symlink(), "raw_source_changed_during_collection")
            require(digest(source.read_bytes()) == entry["source_sha256"], "raw_source_changed_during_collection")
        if reference is not None and reference["sha256"] is not None:
            require(not reference_path.is_symlink(), "reference_manifest_changed_during_collection")
            require(digest(reference_path.read_bytes()) == reference["sha256"], "reference_manifest_changed_during_collection")
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        stage.rename(output_dir)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validator", type=Path, default=default_validator_path())
    parser.add_argument("--corpus-version", default=CORPUS_VERSION, help="ASCII corpus label, up to 96 letters/digits/dots/underscores/hyphens; starts with a letter or digit")
    parser.add_argument("--reference-manifest", type=Path, help="Research source manifest required for raw reference provenance claims")
    parser.add_argument("--tool-actions-only", action="store_true", help="Emit only verified tool-call targets with no accompanying prose, after validating the entire episode")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing output dataset directory")
    args = parser.parse_args()
    try:
        manifest = collect(args.raw_dir, args.output_dir, args.validator, overwrite=args.overwrite,
                           corpus_version=args.corpus_version, reference_manifest=args.reference_manifest,
                           tool_actions_only=args.tool_actions_only)
    except (Rejected, OSError, ImportError) as error:
        print(json.dumps({"ok": False, "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, "accepted_episodes": manifest["accepted_episodes"],
                      "quarantined_episodes": manifest["quarantined_episodes"], "records_by_split": manifest["records_by_split"],
                      "manifest": str(args.output_dir.resolve() / "manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

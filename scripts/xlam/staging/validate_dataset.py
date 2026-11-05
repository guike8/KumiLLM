#!/usr/bin/env python3
"""Validate normalized MLX tool-use JSONL; never prints conversation contents.

Each record is a request prefix ending at one target assistant action, with
top-level messages and tools. Tokenization and target-token masking must also
be checked separately using the exact training tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


GROUP_KEYS = ("episode_id", "project_id", "project_family_id", "song_id", "source_audio_id")


def validate_record(record: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(record, dict):
        return ["record must be an object"]
    tools = record.get("tools")
    if not isinstance(tools, list):
        errors.append("tools must be a list (use an empty list when no tools are available)")
    else:
        names: set[str] = set()
        for index, tool in enumerate(tools):
            definition = tool.get("function", tool) if isinstance(tool, dict) else None
            if not isinstance(definition, dict) or not isinstance(definition.get("name"), str) or not definition["name"].strip():
                errors.append(f"tools[{index}] must have a nonempty function name")
                continue
            if definition["name"] in names:
                errors.append(f"tools[{index}] has a duplicate function name")
            names.add(definition["name"])
            if not isinstance(definition.get("parameters"), dict):
                errors.append(f"tools[{index}].parameters must be an object")
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return errors + ["messages must be a nonempty list"]
    if not isinstance(messages[-1], dict) or messages[-1].get("role") != "assistant":
        errors.append("final message must be the target assistant action")
    pending: dict[str, str] = {}
    seen_ids: set[str] = set()
    for index, message in enumerate(messages):
        location = f"messages[{index}]"
        if not isinstance(message, dict):
            errors.append(f"{location} must be an object")
            continue
        role = message.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            errors.append(f"{location} has an unsupported role")
            continue
        content = message.get("content")
        if role == "tool":
            if not isinstance(content, str):
                errors.append(f"{location}.content must be a normalized string")
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                errors.append(f"{location} has no matching pending tool call")
            else:
                name = pending.pop(call_id)
                if "name" in message and message["name"] != name:
                    errors.append(f"{location}.name does not match its tool call")
            continue
        if pending:
            errors.append(f"{location} occurs before all preceding tool calls have results")
        if content is not None and not isinstance(content, str):
            errors.append(f"{location}.content must be a string or null; normalize multipart messages first")
        calls = message.get("tool_calls", [])
        if not isinstance(calls, list):
            errors.append(f"{location}.tool_calls must be a list")
            calls = []
        if role != "assistant" and calls:
            errors.append(f"{location} only assistant messages may contain tool_calls")
        if role == "assistant" and not calls and not (isinstance(content, str) and content.strip()):
            errors.append(f"{location} assistant action has no text or tool calls")
        for call_index, call in enumerate(calls):
            call_location = f"{location}.tool_calls[{call_index}]"
            if not isinstance(call, dict):
                errors.append(f"{call_location} must be an object")
                continue
            if call.get("type", "function") != "function":
                errors.append(f"{call_location}.type must be function")
            function = call.get("function")
            if not isinstance(function, dict):
                errors.append(f"{call_location}.function must be an object")
                continue
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                errors.append(f"{call_location}.function.name must be a nonempty string")
            if not isinstance(function.get("arguments"), dict):
                errors.append(f"{call_location}.function.arguments must be an object, not serialized JSON")
            call_id = call.get("id")
            if not isinstance(call_id, str) or not call_id.strip():
                errors.append(f"{call_location}.id must be a nonempty string")
            elif call_id in seen_ids:
                errors.append(f"{call_location}.id duplicates an earlier call")
            else:
                seen_ids.add(call_id)
                pending[call_id] = name
    # Pending calls are expected when the final assistant is the supervised call.
    if pending and (not isinstance(messages[-1], dict) or messages[-1].get("role") != "assistant" or not messages[-1].get("tool_calls")):
        errors.append("unanswered tool calls are only permitted in the final target assistant action")
    return errors


def reject_nonfinite(value: str):
    raise ValueError(f"non-finite JSON number {value} is unsupported")


def validate_files(files: dict[str, Path], max_errors: int = 50) -> dict:
    errors: list[str] = []
    error_count = 0
    counts: dict[str, int] = {}
    group_splits: dict[tuple[str, str], str] = {}
    seen_records: dict[str, str] = {}
    without_metadata = 0
    exact_duplicates = 0

    def add_error(message: str):
        nonlocal error_count
        error_count += 1
        if len(errors) < max_errors:
            errors.append(message)

    for split, path in files.items():
        counts[split] = 0
        try:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    counts[split] += 1
                    location = f"{split}:{line_number}"
                    try:
                        record = json.loads(line, parse_constant=reject_nonfinite)
                    except (ValueError, json.JSONDecodeError):
                        add_error(f"{location}: invalid JSON")
                        continue
                    record_errors = validate_record(record)
                    for error in record_errors:
                        add_error(f"{location}: {error}")
                    if not isinstance(record, dict):
                        continue
                    digest = hashlib.sha256(json.dumps({key: record.get(key) for key in ("messages", "tools")}, sort_keys=True).encode()).hexdigest()
                    old_split = seen_records.get(digest)
                    if old_split is not None and old_split != split:
                        add_error(f"{location}: exact training example is shared with {old_split}")
                    elif old_split is not None:
                        exact_duplicates += 1
                    seen_records[digest] = split
                    metadata = record.get("metadata", record.get("_meta", {}))
                    if not isinstance(metadata, dict):
                        add_error(f"{location}: metadata must be an object")
                        metadata = {}
                    groups_found = False
                    for key in GROUP_KEYS:
                        value = metadata.get(key, record.get(key))
                        if value is None:
                            continue
                        groups_found = True
                        if isinstance(value, bool) or not isinstance(value, (str, int)) or str(value).strip() == "":
                            add_error(f"{location}: {key} must be a nonempty string or integer")
                            continue
                        group = (key, str(value))
                        old_split = group_splits.get(group)
                        if old_split is not None and old_split != split:
                            add_error(f"{location}: {key} is shared with {old_split}; split original groups before expanding prefixes")
                        else:
                            group_splits[group] = split
                    if not groups_found:
                        without_metadata += 1
        except (OSError, UnicodeError) as error:
            add_error(f"{split}: could not read UTF-8 JSONL ({type(error).__name__})")
        if counts[split] == 0:
            add_error(f"{split}: no examples found")
    return {
        "ok": error_count == 0,
        "records": counts,
        "error_count": error_count,
        "errors": errors,
        "same_split_exact_duplicates": exact_duplicates,
        "records_without_group_metadata": without_metadata,
        "limitations": [
            "Token lengths, exact rendered prompt prefixes, and positive target-token masks require the selected tokenizer and are not checked here.",
            "Semantic success, audio quality, and group leakage without identifiers require independent verification.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "valid", "test"):
        parser.add_argument(f"--{split}", type=Path)
    parser.add_argument("--max-errors", type=int, default=50)
    args = parser.parse_args()
    files = {split: getattr(args, split) for split in ("train", "valid", "test") if getattr(args, split) is not None}
    if not files:
        parser.error("provide at least one of --train, --valid, --test")
    if args.max_errors < 1:
        parser.error("--max-errors must be positive")
    report = validate_files(files, args.max_errors)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

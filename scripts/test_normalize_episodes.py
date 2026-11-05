import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from normalize_episodes import Rejected, collect, default_validator_path, load_validator, normalize_episode


VALIDATOR = default_validator_path()
VALIDATE_RECORD = load_validator(VALIDATOR).validate_record


def finish(reason):
    return {"type": "finish", "usage": {"inputTokens": {}, "outputTokens": {}}, "finishReason": {"unified": reason}}


def episode(number=1, split="train", family=None):
    tool = {"type": "function", "name": "inspect_value", "description": "Inspect a value.",
            "inputSchema": {"type": "object", "properties": {"level": {"anyOf": [{"type": "number"}, {"const": "-inf", "type": "string"}]}}}}
    system = {"role": "system", "content": "Use the supplied tools."}
    user = {"role": "user", "content": [{"type": "text", "text": "Inspect item " + str(number)}]}
    assistant = {"role": "assistant", "content": [{"type": "tool-call", "toolCallId": "c1", "toolName": "inspect_value", "input": {"level": -3}}]}
    result = {"role": "tool", "content": [{"type": "tool-result", "toolCallId": "c1", "toolName": "inspect_value",
                                            "output": {"type": "json", "value": {"value": number}}}]}
    return {"format_version": 1, "episode_id": "episode-" + str(number), "project_id": "project-" + str(number),
            "project_family_id": family or "family-" + str(number), "split": split, "scope": "executed_synthetic_kumi",
            "teacher": "scripted_oracle_v1", "source_revision": "a" * 40, "goals": {"observe": number},
            "verification": {"passed": True, "checks": {"observed": True, "unchanged": True}, "before": {"value": number}, "after": {"value": number}},
            "outcome": {"passed": True, "error": None},
            "calls": [{"finished": True, "request": {"prompt": [system, user], "tools": [tool]},
                       "response": [{"type": "tool-call", "toolCallId": "c1", "toolName": "inspect_value", "input": '{"level":-3}'}, finish("tool-calls")]},
                      {"finished": True, "request": {"prompt": [system, user, assistant, result], "tools": [tool]},
                       "response": [{"type": "text-delta", "id": "answer", "delta": "Observed " + str(number) + "."}, finish("stop")]}]}


class NormalizationTests(unittest.TestCase):
    def test_native_conversion_preserves_schema_numeric_arguments_and_result(self):
        raw = episode()
        original = copy.deepcopy(raw)
        records = normalize_episode(raw, VALIDATE_RECORD)
        self.assertEqual(raw, original)
        self.assertEqual(len(records), 2)
        target = records[0]["messages"][-1]["tool_calls"][0]
        self.assertEqual(target["function"]["arguments"], {"level": -3})
        self.assertEqual(records[0]["tools"][0]["function"]["parameters"], raw["calls"][0]["request"]["tools"][0]["inputSchema"])
        self.assertEqual(records[1]["messages"][-2], {"role": "tool", "tool_call_id": "c1", "name": "inspect_value", "content": '{"value":1}'})
        self.assertFalse(records[0]["metadata"]["real_live_verified"])
        self.assertEqual([VALIDATE_RECORD(record) for record in records], [[], []])

    def test_explicit_stream_error_quarantines_even_when_outcome_passed(self):
        raw = episode()
        raw["calls"][0]["response"].insert(0, {"type": "error", "error": {"message": "failed"}})
        with self.assertRaisesRegex(Rejected, "unsupported_stream_part:error"):
            normalize_episode(raw, VALIDATE_RECORD)

    def test_missing_subsequent_result_is_rejected(self):
        raw = episode()
        raw["calls"] = raw["calls"][:1]
        with self.assertRaisesRegex(Rejected, "missing_subsequent_successful_tool_result"):
            normalize_episode(raw, VALIDATE_RECORD)

    def test_error_tool_result_and_refused_steps_are_rejected(self):
        for output in ({"type": "error-text", "value": "rejected"},
                       {"type": "text", "value": '{"done":[],"refused":[{"step":1,"error":"bad"}]}'},
                       {"type": "json", "value": {"ok": False, "error": "bad"}}):
            with self.subTest(output=output):
                raw = episode()
                raw["calls"][1]["request"]["prompt"][-1]["content"][0]["output"] = output
                with self.assertRaises(Rejected):
                    normalize_episode(raw, VALIDATE_RECORD)

    def test_returned_tool_schema_error_property_is_not_execution_error(self):
        raw = episode()
        raw["calls"][1]["request"]["prompt"][-1]["content"][0]["output"] = {
            "type": "json", "value": {"tools": [{"name": "check", "inputSchema": {"properties": {"error": {"type": "string"}}}}]}}
        self.assertEqual(len(normalize_episode(raw, VALIDATE_RECORD)), 2)

    def test_invalid_structured_text_cannot_bypass_failure_checks(self):
        texts = ['{"error":"rejected","error":null}', '{"value":NaN}', '{"value":1e999}']
        for text in texts:
            for kind in ("text", "content"):
                with self.subTest(text=text, kind=kind):
                    raw = episode()
                    output = {"type": kind, "value": text if kind == "text" else [{"type": "text", "text": text}]}
                    raw["calls"][1]["request"]["prompt"][-1]["content"][0]["output"] = output
                    with self.assertRaises(Rejected):
                        normalize_episode(raw, VALIDATE_RECORD)

    def test_separate_json_content_blocks_are_each_checked_for_errors(self):
        raw = episode()
        result = raw["calls"][1]["request"]["prompt"][-1]["content"][0]
        result["output"] = {"type": "content", "value": [
            {"type": "text", "text": '{"ok":true}'},
            {"type": "text", "text": '{"error":"rejected"}'},
        ]}
        with self.assertRaisesRegex(Rejected, "tool_result_reports_failure"):
            normalize_episode(raw, VALIDATE_RECORD)
        result["output"]["value"][1]["text"] = '{"value":42}'
        self.assertEqual(len(normalize_episode(raw, VALIDATE_RECORD)), 2)

    def test_ordinary_plain_tool_text_remains_supported(self):
        raw = episode()
        result = raw["calls"][1]["request"]["prompt"][-1]["content"][0]
        result["output"] = {"type": "text", "value": "The inspected value is available."}
        self.assertEqual(len(normalize_episode(raw, VALIDATE_RECORD)), 2)

    def test_incomplete_and_unsupported_content_rejected(self):
        changes = [lambda r: r["calls"][0].update(finished=False),
                   lambda r: r["calls"][0]["response"][-1]["finishReason"].update(unified="length"),
                   lambda r: r["calls"][1]["request"]["prompt"][2]["content"].append({"type": "reasoning", "text": "hidden"}),
                   lambda r: r["calls"][0]["response"].insert(0, {"type": "file", "data": "x"}),
                   lambda r: r["verification"]["checks"].update(unchanged=False)]
        for change in changes:
            raw = episode()
            change(raw)
            with self.assertRaises(Rejected):
                normalize_episode(raw, VALIDATE_RECORD)

    def test_tool_call_must_be_in_current_schema(self):
        raw = episode()
        raw["calls"][0]["request"]["tools"] = []
        with self.assertRaisesRegex(Rejected, "target_tool_not_in_active_schema"):
            normalize_episode(raw, VALIDATE_RECORD)

    def test_streamed_input_must_agree_with_complete_call(self):
        raw = episode()
        raw["calls"][0]["response"][:0] = [{"type": "tool-input-start", "id": "c1", "toolName": "inspect_value"},
                                              {"type": "tool-input-delta", "id": "c1", "delta": '{"level":-3}'},
                                              {"type": "tool-input-end", "id": "c1"}]
        self.assertEqual(len(normalize_episode(raw, VALIDATE_RECORD)), 2)
        raw["calls"][0]["response"][1]["delta"] = '{"level":6}'
        with self.assertRaisesRegex(Rejected, "streamed_tool_input_mismatch"):
            normalize_episode(raw, VALIDATE_RECORD)

    def collect_sources(self, directory, episodes):
        source = Path(directory) / "raw"
        source.mkdir()
        for index, raw in enumerate(episodes):
            (source / (str(index) + ".json")).write_text(json.dumps(raw))
        output = Path(directory) / "dataset"
        return source, output, collect(source, output, VALIDATOR)

    def test_group_split_leakage_quarantines_whole_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            _, output, manifest = self.collect_sources(directory, [episode(1, "train", "shared"), episode(2, "valid", "shared")])
            self.assertEqual(manifest["accepted_episodes"], 0)
            self.assertEqual(manifest["quarantined_episodes"], 2)
            self.assertTrue(all("group_shared_across_splits:project_family_id" in row["reasons"] for row in manifest["episodes"]))
            self.assertEqual((output / "train.jsonl").read_text(), "")

    def test_duplicate_content_detected_across_different_episode_ids(self):
        a, b = episode(), episode()
        b.update(episode_id="other-episode", project_id="other-project", project_family_id="other-family", split="test")
        with tempfile.TemporaryDirectory() as directory:
            _, _, manifest = self.collect_sources(directory, [a, b])
            self.assertEqual(manifest["accepted_episodes"], 0)
            self.assertTrue(all("duplicate_normalized_example" in row["reasons"] for row in manifest["episodes"]))

    def test_admitted_source_hashes_output_records_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output, manifest = self.collect_sources(directory, [episode(1), episode(2, "valid"), episode(3, "test")])
            before = {p.name: p.read_bytes() for p in source.iterdir()}
            self.assertEqual(manifest["accepted_episodes"], 3)
            self.assertEqual(manifest["records_by_split"], {"train": 2, "valid": 2, "test": 2})
            for row in manifest["episodes"]:
                self.assertEqual(row["source_sha256"], hashlib.sha256(before[row["source"]]).hexdigest())
            with self.assertRaisesRegex(Rejected, "output_directory_not_empty"):
                collect(source, output, VALIDATOR)
            self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})
            self.assertEqual(collect(source, output, VALIDATOR, overwrite=True)["accepted_episodes"], 3)

    def test_source_symlink_rejected_without_reading_or_hashing_target(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / 'raw'
            source.mkdir()
            unrelated = base / 'unrelated.json'
            unrelated.write_text(json.dumps(episode()))
            link = source / 'linked.json'
            link.symlink_to(unrelated)
            original_read = Path.read_bytes

            def guarded_read(path):
                if path == link or path == unrelated:
                    raise AssertionError('The source symlink target must never be opened')
                return original_read(path)

            with mock.patch.object(Path, 'read_bytes', guarded_read):
                manifest = collect(source, base / 'dataset', VALIDATOR)
            self.assertEqual(manifest['accepted_episodes'], 0)
            self.assertEqual(manifest['quarantined_episodes'], 1)
            rejected = manifest['episodes'][0]
            self.assertEqual(rejected['reasons'], ['source_symlink_not_allowed'])
            self.assertNotIn('source_sha256', rejected)
            self.assertNotIn('source_bytes', rejected)


if __name__ == "__main__":
    unittest.main()

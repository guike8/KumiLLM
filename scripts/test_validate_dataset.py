import copy
import json
import tempfile
import unittest
from pathlib import Path

from validate_dataset import validate_files, validate_record


def example():
    return {
        "tools": [{"type": "function", "function": {"name": "read_value", "parameters": {"type": "object"}}}],
        "messages": [
            {"role": "user", "content": "Read the current value."},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_value", "arguments": {}}}]},
        ],
    }


class DatasetValidationTests(unittest.TestCase):
    def test_target_call_may_be_unanswered(self):
        self.assertEqual(validate_record(example()), [])

    def test_prior_call_result_and_next_target(self):
        record = example()
        record["messages"] += [
            {"role": "tool", "tool_call_id": "call_1", "content": "42"},
            {"role": "assistant", "content": "The value is 42."},
        ]
        self.assertEqual(validate_record(record), [])

    def test_serialized_arguments_rejected(self):
        record = example()
        record["messages"][-1]["tool_calls"][0]["function"]["arguments"] = "{}"
        self.assertTrue(any("arguments must be an object" in error for error in validate_record(record)))

    def test_orphan_or_out_of_order_results_rejected(self):
        record = example()
        record["messages"].insert(1, {"role": "tool", "tool_call_id": "call_1", "content": "42"})
        self.assertTrue(any("no matching pending" in error for error in validate_record(record)))

    def test_new_assistant_before_tool_result_rejected(self):
        record = example()
        record["messages"].append({"role": "assistant", "content": "Done."})
        self.assertTrue(any("before all preceding" in error for error in validate_record(record)))

    def test_duplicate_calls_rejected(self):
        record = example()
        record["messages"][-1]["tool_calls"].append(copy.deepcopy(record["messages"][-1]["tool_calls"][0]))
        self.assertTrue(any("duplicates an earlier call" in error for error in validate_record(record)))

    def test_final_role_and_nonempty_target(self):
        record = example()
        record["messages"] = [{"role": "user", "content": "Read a value."}]
        self.assertTrue(any("final message" in error for error in validate_record(record)))
        record["messages"].append({"role": "assistant", "content": "  "})
        self.assertTrue(any("no text or tool calls" in error for error in validate_record(record)))

    def test_multipart_requires_normalization(self):
        record = example()
        record["messages"][0]["content"] = [{"type": "text", "text": "Read a value."}]
        self.assertTrue(any("normalize multipart" in error for error in validate_record(record)))

    def test_project_leakage_and_exact_duplicate_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            train = Path(directory) / "train.jsonl"
            valid = Path(directory) / "valid.jsonl"
            record = example()
            record["metadata"] = {"project_id": "project_1", "episode_id": "episode_1"}
            train.write_text(json.dumps(record) + "\n")
            record["messages"][0]["content"] = "A different request."
            valid.write_text(json.dumps(record) + "\n")
            report = validate_files({"train": train, "valid": valid})
            self.assertFalse(report["ok"])
            self.assertTrue(any("project_id is shared" in error for error in report["errors"]))
            valid.write_text(train.read_text())
            report = validate_files({"train": train, "valid": valid})
            self.assertTrue(any("exact training example" in error for error in report["errors"]))

    def test_same_project_within_split_is_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            train = Path(directory) / "train.jsonl"
            record = example()
            record["metadata"] = {"project_id": "project_1"}
            train.write_text((json.dumps(record) + "\n") * 2)
            report = validate_files({"train": train})
            self.assertTrue(report["ok"])
            self.assertEqual(report["same_split_exact_duplicates"], 1)

    def test_empty_split_and_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            train = Path(directory) / "train.jsonl"
            train.write_text("")
            self.assertFalse(validate_files({"train": train})["ok"])
            train.write_text('{"messages": NaN}\n')
            report = validate_files({"train": train})
            self.assertTrue(any("invalid JSON" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()

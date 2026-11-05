import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import prepare_mixture as preparation
import run_strict


def row(index):
    return {'messages': [{'role': 'user', 'content': 'Synthetic test query ' + str(index)},
                         {'role': 'assistant', 'content': None, 'tool_calls': [{
                             'id': 'call_' + str(index), 'type': 'function',
                             'function': {'name': 'op', 'arguments': {'x': index}}}]}],
            'tools': [{'type': 'function', 'function': {'name': 'op',
                       'parameters': {'type': 'object', 'properties': {'x': {'type': 'integer'}},
                                      'required': ['x']}}}],
            'metadata': {'episode_id': 'test-' + str(index), 'source_row_id': index,
                         'source_api_families': ['family-' + str(index)],
                         'query_style': ['single', 'parallel_same', 'parallel_multiple'][index % 3],
                         'argument_types': [['integer'], ['array'], ['boolean']][index % 3]}}


class MixtureTests(unittest.TestCase):
    def test_whole_rows_obey_both_limits(self):
        records = [row(i) for i in range(60)]
        selected, report = preparation.diverse_subset(records, [{'target_tokens': 100}] * 60,
                                                       341, 900, 7)
        self.assertEqual(len(selected), 1)
        self.assertLessEqual(report['row_fraction'], 0.1)
        self.assertEqual(report['target_token_fraction'], 0.1)

    def test_diversity_determinism_and_no_input_mutation(self):
        records = [row(i) for i in range(60)]
        before = copy.deepcopy(records)
        args = (records, [{'target_tokens': 10}] * 60, 341, 90000, 7)
        one, report = preparation.diverse_subset(*args)
        two, _ = preparation.diverse_subset(*args)
        self.assertEqual(one, two)
        self.assertEqual(len(one), 37)
        self.assertEqual(len(set(one)), 37)
        self.assertEqual(len(report['query_styles']), 3)
        self.assertEqual(records, before)

    def test_all_three_quality_filters_apply_before_selection(self):
        records = [row(i) for i in range(10)]
        records[0]['metadata']['underspecified_containers_or_values'] = ['untyped object']
        allowed = {preparation.canonical(i) for i in range(10) if i != 1}
        compatible = set(range(10)) - {2}
        selected, report = preparation.diverse_subset(records, [{'target_tokens': 10}] * 10,
                                                       341, 90000, 7, allowed, compatible)
        self.assertEqual(set(selected), set(range(3, 10)))
        self.assertEqual(report['underspecified_pool_rows_excluded'], 1)
        self.assertEqual(report['semantic_review_excluded_pool_rows'], 1)
        self.assertEqual(report['runtime_parser_incompatible_pool_rows'], 1)

    def test_no_eligible_row_fails_closed(self):
        records = [row(0)]
        records[0]['metadata']['underspecified_containers_or_values'] = ['unspecified']
        with self.assertRaisesRegex(ValueError, 'no_xlam_row_fits'):
            preparation.diverse_subset(records, [{'target_tokens': 10}], 341, 90000, 7)

    def test_group_and_exact_record_leaks_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exact_record_cross_split'):
            preparation.check_split_isolation([row(0)], [row(0)])
        training, validation = row(0), row(1)
        validation['metadata']['episode_id'] = training['metadata']['episode_id']
        with self.assertRaisesRegex(ValueError, 'source_group_cross_split'):
            preparation.check_split_isolation([training], [validation])

    def test_allowlist_binds_hashes_and_ids(self):
        manifest = {'source_revision': 'rev', 'source_file_sha256': 'source',
                    'files': {'train.jsonl': {'sha256': 'a'}, 'valid.jsonl': {'sha256': 'b'}}}
        allowed = {'schema': 'kumi-xlam-semantic-allowlist/1',
                   'source_revision': 'rev', 'source_file_sha256': 'source',
                   'pool_files_sha256': {'train.jsonl': 'a', 'valid.jsonl': 'b'},
                   'locally_executed': False, 'review_scope': 'Synthetic unit test',
                   'allowed_source_row_ids': {'train': [0], 'valid': [1]}}
        self.assertEqual(preparation.check_allowlist(allowed, [row(0)], [row(1)], manifest),
                         {'train': {'0'}, 'valid': {'1'}})
        for update in [{'pool_files_sha256': {'train.jsonl': 'wrong', 'valid.jsonl': 'b'}},
                       {'allowed_source_row_ids': {'train': [True], 'valid': [1]}},
                       {'allowed_source_row_ids': {'train': [99], 'valid': [1]}}]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                preparation.check_allowlist(dict(allowed, **update), [row(0)], [row(1)], manifest)

    def test_serialized_arguments_fail_structural_gate(self):
        record = row(0)
        record['messages'][-1]['tool_calls'][0]['function']['arguments'] = '{"x":0}'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train.jsonl'
            path.write_text(json.dumps(record) + '\n')
            with self.assertRaisesRegex(ValueError, 'structural_validation_failed'):
                preparation.jsonl(path)


class ParserTests(unittest.TestCase):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            if kwargs.get('add_generation_prompt'):
                return 'PREFIX'
            return 'PREFIX<tool_call>\n<function=op>\n<parameter=x>0</parameter>\n</function>\n</tool_call>'

    def test_installed_parser_receives_body_and_tools_and_must_preserve_types(self):
        observed = []

        def good(body, tools):
            observed.append((body, tools))
            return {'name': 'op', 'arguments': {'x': 0}}

        record = row(0)
        self.assertTrue(preparation.runtime_parser_roundtrip(record, self.Tokenizer(), good)['ok'])
        self.assertTrue(observed[0][0].startswith('<function=op>'))
        self.assertEqual(observed[0][1], record['tools'])
        for wrong in [True, 0.0, '0', None]:
            result = preparation.runtime_parser_roundtrip(record, self.Tokenizer(),
                lambda body, tools: {'name': 'op', 'arguments': {'x': wrong}})
            self.assertFalse(result['ok'])
            self.assertEqual(result['reason'], 'runtime_parser_typed_arguments_changed')


class RunnerTests(unittest.TestCase):
    def test_loss_is_explicitly_bound_and_cannot_be_overridden(self):
        observed = []
        loss = object()

        def actual(*args, **kwargs):
            observed.append((args, kwargs))
            return 'called'

        lora, trainer = types.SimpleNamespace(train=actual), types.SimpleNamespace(train=actual)
        original = run_strict.bind_loss(lora, trainer, loss)
        self.assertIs(original, actual)
        self.assertEqual(lora.train('model', batch=1), 'called')
        self.assertIs(observed[0][1]['loss'], loss)
        self.assertIs(lora.train.selected_loss, loss)
        with self.assertRaisesRegex(ValueError, 'must_not_override'):
            lora.train(loss=object())
        self.assertIs(trainer.train, actual)

    def test_unreviewed_training_alias_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unreviewed_lora_train_alias'):
            run_strict.bind_loss(types.SimpleNamespace(train=lambda: None),
                                 types.SimpleNamespace(train=lambda: None), object())

    def test_active_v1_only_gets_liveness_signal_zero_and_is_not_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'logs').mkdir()
            (root / 'logs/training-job.json').write_text(json.dumps({'stage': 'running', 'trainer_pid': 123}))
            with patch.object(run_strict.os, 'kill') as signal, self.assertRaisesRegex(ValueError, 'v1 training process is active'):
                run_strict.refuse_active_v1(root)
            signal.assert_called_once_with(123, 0)


if __name__ == '__main__':
    unittest.main()

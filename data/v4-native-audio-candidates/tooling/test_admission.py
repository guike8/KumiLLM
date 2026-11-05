import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import admit_native_audio as admission


HERE = Path(__file__).resolve().parent
BATCH = HERE.parent
TEACHER = Path(os.environ.get('KUMI_NATIVE_TEACHER', str(BATCH / 'raw/teacher'))).resolve()
ASSETS = Path(os.environ.get('KUMI_NATIVE_ASSETS', str(BATCH.parent / 'production-source-seeds/audio-cache/assets'))).resolve()
EVIDENCE = Path(os.environ.get('KUMI_NATIVE_EVIDENCE', str(BATCH / 'raw/native'))).resolve()
REVIEWS = json.loads((BATCH / 'reviewed-public-responses.json').read_text())


class NativeAdmissionTests(unittest.TestCase):
    def test_original_capture_has_three_reads_and_four_reviewed_responses(self):
        rows, manifest = admission.build(TEACHER, ASSETS, REVIEWS, EVIDENCE)
        self.assertEqual(len(rows), 7)
        self.assertEqual([r['target_kind'] for r in manifest['records']].count('tool_only'), 3)
        self.assertEqual([r['target_kind'] for r in manifest['records']].count('checked_public_response'), 4)
        self.assertTrue(all(not row['metadata']['ready_for_training_ingestion'] for row in rows))
        self.assertTrue(all(row['metadata']['mask_all_history_required'] for row in rows))
        self.assertEqual(len(manifest['withheld']), 3)
        self.assertEqual(manifest['episode_count'], 3)
        self.assertEqual(manifest['human_turn_count'], 4)
        self.assertEqual(len({r['metadata']['episode_id'] for r in rows}), 3)
        self.assertEqual([r['metadata']['human_turn_index'] for r in rows].count(1), 1)
        self.assertTrue(all(w['reason'] == 'redundant_native_audition_nudge_excluded' for w in manifest['withheld']))

    def test_missing_native_delivery_evidence_admits_nothing(self):
        rows, manifest = admission.build(TEACHER, ASSETS, REVIEWS, None)
        self.assertEqual(rows, [])
        self.assertEqual(len(manifest['withheld']), 10)

    def test_failed_native_read_withholds_its_call_and_derived_public_response(self):
        events = admission.native_events(EVIDENCE)
        events['call_audio_001'][0][0]['isError'] = True
        with patch.object(admission, 'native_events', return_value=events):
            rows, manifest = admission.build(TEACHER, ASSETS, REVIEWS, EVIDENCE)
        self.assertEqual(len(rows), 5)
        self.assertIn('native_success_event_missing_or_failed', [w['reason'] for w in manifest['withheld']])
        self.assertIn('public_fact_source_native_success_missing_or_failed', [w['reason'] for w in manifest['withheld']])

    def test_wrong_fact_value_withholds_the_public_target(self):
        reviews = copy.deepcopy(REVIEWS)
        first = 'req_6cdc9a1c82d94d19a14e1de2f9421fbc'
        reviews[first]['fact_groundings'][0]['value'] = 999
        rows, manifest = admission.build(TEACHER, ASSETS, reviews, EVIDENCE)
        self.assertEqual(len(rows), 6)
        self.assertIn({'call_id': first, 'reason': 'public_fact_grounding_mismatch'}, manifest['withheld'])

    def test_wrong_text_review_digest_withholds_the_public_target(self):
        reviews = copy.deepcopy(REVIEWS)
        first = 'req_6cdc9a1c82d94d19a14e1de2f9421fbc'
        reviews[first]['content_sha256'] = '0' * 64
        rows, manifest = admission.build(TEACHER, ASSETS, reviews, EVIDENCE)
        self.assertEqual(len(rows), 6)
        self.assertIn({'call_id': first, 'reason': 'public_text_changed_after_review'}, manifest['withheld'])

    def test_duplicate_json_and_nonfinite_rejected(self):
        for text in ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}']:
            with self.assertRaises(ValueError):
                admission.load_json(text)

    def test_schema_unknown_constraint_wrong_type_and_bounds_rejected(self):
        invalid = [(1, {'type': 'number', 'not': {'const': 1}}),
                   (True, {'type': 'integer'}), (-1, {'type': 'number', 'minimum': 0}),
                   ('bad', {'type': 'string', 'enum': ['mix', 'sound']})]
        for value, schema in invalid:
            with self.assertRaises(ValueError):
                admission.validate_schema(value, schema)

    def test_native_argument_objects_are_preserved_and_private_parts_rejected(self):
        message = {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'a', 'type': 'function', 'function': {'name': 'listen', 'arguments': '{"seconds":30,"form":false}'}}]}
        normalized = admission.normalize_messages([message])
        self.assertEqual(normalized[0]['tool_calls'][0]['function']['arguments'], {'seconds': 30, 'form': False})
        self.assertIsInstance(message['tool_calls'][0]['function']['arguments'], str)
        with self.assertRaises(ValueError):
            admission.normalize_messages([{'role': 'assistant', 'content': [{'type': 'thinking', 'text': 'private'}]}])

    def test_source_capture_edit_fails_integrity_check(self):
        path = next(TEACHER.glob('*.request.json'))
        request = json.loads(path.read_text())
        request['request']['messages'][0]['content'] += 'changed'
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / path.name
            p.write_text(json.dumps(request))
            with self.assertRaisesRegex(ValueError, 'request_hash_changed'):
                admission.build(Path(directory), ASSETS, REVIEWS, EVIDENCE)


if __name__ == '__main__':
    unittest.main()

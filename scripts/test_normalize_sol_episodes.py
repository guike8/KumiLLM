import copy
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from normalize_sol_episodes import CORPUS_VERSION, TEACHER_PROVENANCE, Rejected, collect, default_validator_path, load_reference_manifest, load_validator, main, normalize_episode


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
            "teacher": "gpt-6.1-sol-ultra", "teacher_model": "gpt-6.1-sol",
            "teacher_reasoning_effort": "ultra", "teacher_transport": "codex_agent_file_queue", "source_revision": "a" * 40, "goals": {"observe": number},
            "verification": {"passed": True, "checks": {"observed": True, "unchanged": True}, "before": {"value": number}, "after": {"value": number}},
            "outcome": {"passed": True, "error": None},
            "calls": [{"finished": True, "request": {"prompt": [system, user], "tools": [tool]},
                       "response": [{"type": "tool-call", "toolCallId": "c1", "toolName": "inspect_value", "input": '{"level":-3}'}, finish("tool-calls")]},
                      {"finished": True, "request": {"prompt": [system, user, assistant, result], "tools": [tool]},
                       "response": [{"type": "text-delta", "id": "answer", "delta": "Observed " + str(number) + "."}, finish("stop")]}]}


def reference_document():
    return {"format_version": 1, "sources": [
        {"id": "lom_song", "url": "https://docs.cycling74.com/apiref/lom/song/", "source_hash": "a" * 64, "status": "success"},
        {"id": "lom_track", "url": "https://docs.cycling74.com/apiref/lom/track/", "source_hash": "B" * 64, "status": "success"},
    ]}


def referenced_episode(manifest_path, number=1):
    raw = episode(number)
    raw.update(reference_manifest_sha256=hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
               reference_source_ids=["lom_song", "lom_track"])
    return raw


class NormalizationTests(unittest.TestCase):
    def write_reference_manifest(self, directory, document=None):
        path = Path(directory) / "reference-manifest.json"
        path.write_text(json.dumps(reference_document() if document is None else document, indent=2))
        return path

    def test_tool_only_targets_filter_final_prose_without_changing_raw(self):
        raw = episode()
        original = copy.deepcopy(raw)
        records = normalize_episode(raw, VALIDATE_RECORD, tool_actions_only=True)
        self.assertEqual(raw, original)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['metadata']['call_index'], 0)
        self.assertEqual(records[0]['metadata']['target_policy'], 'verified_tool_calls_only')
        self.assertTrue(records[0]['messages'][-1]['tool_calls'])
        self.assertEqual(records[0]['messages'][-1]['content'], '')
        default = normalize_episode(raw, VALIDATE_RECORD)
        self.assertEqual(len(default), 2)
        self.assertNotIn('target_policy', default[0]['metadata'])

    def test_tool_only_collection_reports_policy_and_filtered_record_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = episode()
            source, output, manifest = self.collect_sources(directory, [raw], tool_actions_only=True)
            self.assertEqual(manifest['accepted_episodes'], 1)
            self.assertEqual(manifest['records_by_split'], {'train':1, 'valid':0, 'test':0})
            self.assertEqual(manifest['episodes'][0]['accepted_records'], 1)
            self.assertEqual(manifest['target_policy'], 'verified_tool_calls_only')
            self.assertEqual(json.loads((source/'0.json').read_text()), raw)
            target = json.loads((output/'train.jsonl').read_text())
            self.assertEqual(target['metadata']['target_policy'], manifest['target_policy'])
            self.assertTrue(target['messages'][-1]['tool_calls'])

    def test_tool_only_filter_cannot_hide_failures_in_the_full_episode(self):
        cases = [
            (lambda r:r['calls'][1]['request']['prompt'][-1]['content'][0].update(
                output={'type':'json','value':{'ok':False,'error':'failed'}}), 'tool_result_reports_failure'),
            (lambda r:r['calls'][1]['response'].insert(0, {'type':'error','error':{'message':'failed final'}}),
             'error_or_unsupported_stream_part:error'),
            (lambda r:r['calls'][1].update(finished=False), 'unfinished_call'),
            (lambda r:r['calls'][1]['response'][-1]['finishReason'].update(unified='length'), 'incomplete_or_error_finish'),
            (lambda r:r.update(calls=r['calls'][:1]), 'missing_subsequent_successful_tool_result'),
            (lambda r:r['verification']['checks'].update(unchanged=False), 'verification_checks_not_all_true'),
        ]
        for modify, reason in cases:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                raw = episode()
                modify(raw)
                _, output, manifest = self.collect_sources(directory, [raw], tool_actions_only=True)
                self.assertEqual(manifest['accepted_episodes'], 0)
                self.assertEqual(manifest['quarantined_episodes'], 1)
                self.assertEqual(manifest['episodes'][0]['reasons'], [reason])
                self.assertEqual((output/'train.jsonl').read_text(), '')

    def test_tool_only_mode_quarantines_accompanying_prose_instead_of_rewriting(self):
        for text in ('I have verified this action.', ' '):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                raw = episode()
                raw['calls'][0]['response'].insert(0, {'type':'text-delta','id':'explanation','delta':text})
                original = copy.deepcopy(raw)
                self.assertEqual(normalize_episode(raw, VALIDATE_RECORD)[0]['messages'][-1]['content'], text)
                source, output, manifest = self.collect_sources(directory, [raw], tool_actions_only=True)
                self.assertEqual(manifest['accepted_episodes'], 0)
                self.assertEqual(manifest['episodes'][0]['reasons'], ['tool_action_target_contains_public_text'])
                self.assertEqual((output/'train.jsonl').read_text(), '')
                self.assertEqual(raw, original)
                self.assertEqual(json.loads((source/'0.json').read_text()), original)

    def test_tool_only_mode_quarantines_episodes_without_tool_targets(self):
        raw = episode()
        raw['calls'] = [raw['calls'][0]]
        raw['calls'][0]['response'] = [{'type':'text-delta','id':'answer','delta':'No changes needed.'}, finish('stop')]
        self.assertEqual(len(normalize_episode(raw, VALIDATE_RECORD)), 1)
        with tempfile.TemporaryDirectory() as directory:
            _, output, manifest = self.collect_sources(directory, [raw], tool_actions_only=True)
            self.assertEqual(manifest['accepted_episodes'], 0)
            self.assertEqual(manifest['episodes'][0]['reasons'], ['episode_has_no_tool_action_targets'])
            self.assertEqual((output/'train.jsonl').read_text(), '')

    def test_tool_only_mode_keeps_prior_prose_as_prompt_context_not_target(self):
        raw = episode()
        prose = {'role':'assistant','content':[{'type':'text','text':'Observed 1.'}]}
        followup = {'role':'user','content':[{'type':'text','text':'Inspect the same item again.'}]}
        prefix = copy.deepcopy(raw['calls'][1]['request']['prompt']) + [prose, followup]
        raw['calls'].append({'finished':True,
            'request':{'prompt':prefix,'tools':copy.deepcopy(raw['calls'][0]['request']['tools'])},
            'response':[{'type':'tool-call','toolCallId':'c2','toolName':'inspect_value','input':'{"level":-3}'},finish('tool-calls')]})
        historical = copy.deepcopy(prefix) + [
            {'role':'assistant','content':[{'type':'tool-call','toolCallId':'c2','toolName':'inspect_value','input':{'level':-3}}]},
            {'role':'tool','content':[{'type':'tool-result','toolCallId':'c2','toolName':'inspect_value',
                                    'output':{'type':'json','value':{'value':1}}}]}]
        raw['calls'].append({'finished':True,
            'request':{'prompt':historical,'tools':copy.deepcopy(raw['calls'][0]['request']['tools'])},
            'response':[{'type':'text-delta','id':'answer2','delta':'Observed again.'},finish('stop')]})
        original = copy.deepcopy(raw)
        records = normalize_episode(raw, VALIDATE_RECORD, tool_actions_only=True)
        self.assertEqual(raw, original)
        self.assertEqual([r['metadata']['call_index'] for r in records], [0,2])
        self.assertIn({'role':'assistant','content':'Observed 1.'}, records[1]['messages'][:-1])
        self.assertTrue(all(record['messages'][-1].get('tool_calls') for record in records))
        self.assertTrue(all(record['messages'][-1]['content']=='' for record in records))

    def test_valid_references_preserve_raw_and_publish_only_selected_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            document = reference_document()
            document['local_snapshot'] = '/private/local/snapshot'
            document['sources'][0].update(text='ARTICLE TEXT MUST STAY PRIVATE', snapshot_path='/private/local/page')
            document['sources'].append({'id': 'not_used', 'url': '/private/failed', 'status': 'failed', 'source_hash': None})
            reference = self.write_reference_manifest(directory, document)
            raw = referenced_episode(reference)
            original = copy.deepcopy(raw)
            source, output, manifest = self.collect_sources(directory, [raw], corpus_version='v3-research-guided', reference_manifest=reference)
            self.assertEqual(json.loads((source/'0.json').read_text()), original)
            records = [json.loads(line) for line in (output/'train.jsonl').read_text().splitlines()]
            self.assertEqual(manifest['accepted_episodes'], 1)
            self.assertEqual(manifest['corpus_version'], 'v3-research-guided')
            self.assertEqual(manifest['reference_manifest_sha256'], original['reference_manifest_sha256'])
            self.assertEqual(manifest['reference_sources'], {
                item['id']: {key:item[key] for key in ('url','source_hash','status')} for item in reference_document()['sources']})
            for record in records:
                self.assertEqual(record['metadata']['reference_manifest_sha256'], original['reference_manifest_sha256'])
                self.assertEqual(record['metadata']['reference_source_ids'], original['reference_source_ids'])
                self.assertEqual(record['metadata']['corpus_version'], 'v3-research-guided')
            public = json.dumps({'reference_sources':manifest['reference_sources']})
            self.assertNotIn('ARTICLE TEXT', public)
            self.assertNotIn('/private/', public)
            self.assertNotIn('snapshot', public)

    def test_reference_hash_matches_exact_bytes_not_reserialized_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_reference_manifest(directory)
            raw = referenced_episode(path)
            path.write_text(json.dumps(reference_document(), separators=(',', ':')))
            _, output, manifest = self.collect_sources(directory, [raw], reference_manifest=path)
            self.assertEqual(manifest['accepted_episodes'], 0)
            self.assertEqual(manifest['episodes'][0]['reasons'], ['reference_manifest_sha256_mismatch'])
            self.assertEqual((output/'train.jsonl').read_text(), '')

    def test_unchecked_reference_fields_without_manifest_are_quarantined(self):
        for field, value in [('reference_manifest_sha256', 'a'*64), ('reference_source_ids', []), ('reference_source_ids', None)]:
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                raw = episode()
                raw[field] = value
                _, _, manifest = self.collect_sources(directory, [raw])
                self.assertEqual(manifest['accepted_episodes'], 0)
                self.assertEqual(manifest['episodes'][0]['reasons'], ['reference_manifest_not_provided'])
                self.assertNotIn('reference_sources', manifest)

    def test_provided_manifest_requires_nonempty_unique_known_episode_references(self):
        cases = [(lambda r:r.pop('reference_manifest_sha256'), 'missing_or_invalid_reference_manifest_sha256'),
                 (lambda r:r.update(reference_manifest_sha256='not-a-hash'), 'missing_or_invalid_reference_manifest_sha256'),
                 (lambda r:r.pop('reference_source_ids'), 'reference_source_ids_missing_or_empty'),
                 (lambda r:r.update(reference_source_ids=[]), 'reference_source_ids_missing_or_empty'),
                 (lambda r:r.update(reference_source_ids='lom_song'), 'reference_source_ids_missing_or_empty'),
                 (lambda r:r.update(reference_source_ids=['lom_song','lom_song']), 'duplicate_episode_reference_source_id'),
                 (lambda r:r.update(reference_source_ids=['unknown_id']), 'unknown_reference_source_id'),
                 (lambda r:r.update(reference_source_ids=['../private']), 'invalid_episode_reference_source_id'),
                 (lambda r:r.update(reference_source_ids=[{}]), 'invalid_episode_reference_source_id')]
        for modify, reason in cases:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                path = self.write_reference_manifest(directory)
                raw = referenced_episode(path)
                modify(raw)
                _, _, manifest = self.collect_sources(directory, [raw], reference_manifest=path)
                self.assertEqual(manifest['accepted_episodes'], 0)
                self.assertEqual(manifest['episodes'][0]['reasons'], [reason])
                self.assertEqual(manifest['reference_sources'], {})

    def test_duplicate_or_invalid_manifest_source_ids_quarantine_all_episodes(self):
        for source_id, reason in [('lom_song', 'duplicate_reference_source_id'), ('bad/path', 'invalid_reference_source_id')]:
            with self.subTest(source_id=source_id), tempfile.TemporaryDirectory() as directory:
                document = reference_document()
                document['sources'][1]['id'] = source_id
                path = self.write_reference_manifest(directory, document)
                _, _, manifest = self.collect_sources(directory, [referenced_episode(path,1), referenced_episode(path,2)], reference_manifest=path)
                self.assertEqual(manifest['accepted_episodes'], 0)
                self.assertEqual(manifest['quarantined_episodes'], 2)
                self.assertTrue(all(entry['reasons']==[reason] for entry in manifest['episodes']))
                self.assertEqual(manifest['reference_manifest_error'], reason)
                self.assertEqual(manifest['reference_sources'], {})

    def test_cited_source_must_be_successful_https_and_hash_identified(self):
        cases = [('status','partial','reference_source_not_successful'),
                 ('url','http://docs.cycling74.com/apiref/lom/song/','reference_source_url_must_be_https'),
                 ('url','https:///missing-host','reference_source_url_must_be_https'),
                 ('url','https://user:secret@example.org/','reference_source_url_must_be_https'),
                 ('source_hash','x'*64,'invalid_reference_source_hash'),
                 ('source_hash',None,'invalid_reference_source_hash')]
        for field, value, reason in cases:
            with self.subTest(field=field,value=value), tempfile.TemporaryDirectory() as directory:
                document = reference_document()
                document['sources'][0][field] = value
                path = self.write_reference_manifest(directory, document)
                _, _, manifest = self.collect_sources(directory, [referenced_episode(path)], reference_manifest=path)
                self.assertEqual(manifest['accepted_episodes'], 0)
                self.assertEqual(manifest['episodes'][0]['reasons'], [reason])

    def test_missing_or_malformed_reference_manifest_quarantines_without_path_disclosure(self):
        for contents, reason in [(None,'reference_manifest_missing_or_not_file'),
                                 ('not JSON','invalid_reference_manifest_json'),
                                 ('{"format_version":1,"sources":[],"sources":[]}', 'invalid_reference_manifest_json'),
                                 ('{"format_version":true,"sources":[]}', 'unsupported_reference_manifest_format')]:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'private-reference.json'
                if contents is not None:
                    path.write_text(contents)
                raw = episode()
                raw.update(reference_manifest_sha256='a'*64, reference_source_ids=['lom_song'])
                _, _, manifest = self.collect_sources(directory, [raw], reference_manifest=path)
                self.assertEqual(manifest['accepted_episodes'], 0)
                self.assertEqual(manifest['episodes'][0]['reasons'], [reason])
                self.assertNotIn(directory, json.dumps({key:manifest[key] for key in ('reference_sources','reference_manifest_error')}))

    def test_reference_manifest_symlink_is_not_read_or_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            target = self.write_reference_manifest(directory)
            raw = referenced_episode(target)
            link = Path(directory)/'linked-reference.json'
            link.symlink_to(target)
            original_read = Path.read_bytes
            def guarded_read(path):
                if path in (link,target):
                    raise AssertionError('Reference symlink targets must never be opened')
                return original_read(path)
            with mock.patch.object(Path,'read_bytes',guarded_read):
                _, _, manifest = self.collect_sources(directory, [raw], reference_manifest=link)
            self.assertEqual(manifest['episodes'][0]['reasons'], ['reference_manifest_symlink_not_allowed'])
            self.assertIsNone(manifest['reference_manifest_sha256'])

    def test_reference_manifest_cannot_be_deleted_by_output_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'raw'
            source.mkdir()
            (source/'episode.json').write_text(json.dumps(episode()))
            output=Path(directory)/'output'
            output.mkdir()
            reference=self.write_reference_manifest(output)
            before=reference.read_bytes()
            with self.assertRaisesRegex(Rejected,'reference_manifest_must_not_be_inside_output_directory'):
                collect(source,output,VALIDATOR,overwrite=True,reference_manifest=reference)
            self.assertEqual(reference.read_bytes(),before)

    def test_valid_references_do_not_admit_failed_tool_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            path=self.write_reference_manifest(directory)
            raw=referenced_episode(path)
            raw['calls'][1]['request']['prompt'][-1]['content'][0]['output']={'type':'json','value':{'ok':False,'error':'failed'}}
            _,output,manifest=self.collect_sources(directory,[raw],reference_manifest=path)
            self.assertEqual(manifest['episodes'][0]['reasons'],['tool_result_reports_failure'])
            self.assertEqual(manifest['accepted_episodes'],0)
            self.assertEqual((output/'train.jsonl').read_text(),'')

    def test_corpus_version_accepts_safe_labels_and_rejects_paths(self):
        for label in ('v3-research-guided','v3.1_sources','V3'):
            self.assertTrue(all(r['metadata']['corpus_version']==label for r in normalize_episode(episode(),VALIDATE_RECORD,corpus_version=label)))
        for label in ('','../v3','/private/v3','v3/subdir','v3\\subdir','.','..','v3 with spaces','épisode','x'*97,None):
            with self.subTest(label=label), self.assertRaisesRegex(Rejected,'invalid_corpus_version'):
                normalize_episode(episode(),VALIDATE_RECORD,corpus_version=label)

    def test_cli_passes_reference_manifest_corpus_version_and_target_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path=self.write_reference_manifest(directory)
            source=Path(directory)/'raw'
            source.mkdir()
            (source/'one.json').write_text(json.dumps(referenced_episode(path)))
            output=Path(directory)/'output'
            argv=['normalize_sol_episodes.py','--raw-dir',str(source),'--output-dir',str(output),
                  '--corpus-version','v3-cli','--reference-manifest',str(path),'--tool-actions-only']
            stdout=StringIO()
            with mock.patch('sys.argv',argv),redirect_stdout(stdout):
                self.assertEqual(main(),0)
            self.assertTrue(json.loads(stdout.getvalue())['ok'])
            manifest=json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['corpus_version'],'v3-cli')
            self.assertEqual(manifest['accepted_episodes'],1)
            self.assertEqual(manifest['records_by_split']['train'],1)
            self.assertEqual(manifest['target_policy'],'verified_tool_calls_only')
            self.assertEqual(manifest['reference_manifest_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())

    def test_exact_teacher_provenance_is_preserved_without_relabeling(self):
        raw = episode()
        before = copy.deepcopy(raw)
        records = normalize_episode(raw, VALIDATE_RECORD)
        self.assertEqual(raw, before)
        for record in records:
            self.assertEqual({key: record['metadata'][key] for key in TEACHER_PROVENANCE}, TEACHER_PROVENANCE)
            self.assertEqual(record['metadata']['corpus_version'], CORPUS_VERSION)
            self.assertFalse(record['metadata']['teacher_usage_reported'])
            self.assertFalse(record['metadata']['real_live_verified'])

    def test_missing_or_unknown_teacher_provenance_is_rejected(self):
        for key in TEACHER_PROVENANCE:
            for replacement in (None, 'another-teacher-or-transport'):
                with self.subTest(key=key, replacement=replacement):
                    raw = episode()
                    if replacement is None:
                        raw.pop(key)
                    else:
                        raw[key] = replacement
                    with self.assertRaisesRegex(Rejected, 'wrong_or_missing_' + key):
                        normalize_episode(raw, VALIDATE_RECORD)

    def test_file_queue_does_not_claim_provider_token_usage(self):
        cases = [lambda r: r.update(teacher_usage={'input_tokens': 100}),
                 lambda r: r['calls'][0]['response'][-1]['usage']['inputTokens'].update(total=0),
                 lambda r: r['calls'][0]['response'][-1]['usage']['outputTokens'].update(total=12),
                 lambda r: r['calls'][0]['response'][-1]['usage'].update(raw={'model': 'gpt-6.1-sol'})]
        for change in cases:
            with self.subTest(change=change):
                raw = episode()
                change(raw)
                with self.assertRaisesRegex(Rejected, 'teacher_usage_unavailable_for_file_queue'):
                    normalize_episode(raw, VALIDATE_RECORD)

    def test_private_reasoning_fields_are_not_admitted(self):
        cases = [lambda r: r.update(analysis='private reasoning'),
                 lambda r: r['calls'][0].update(chain_of_thought='private reasoning'),
                 lambda r: r['calls'][0]['request'].update(reasoning='private reasoning'),
                 lambda r: r['calls'][0]['response'].insert(0, {'type':'reasoning-delta','id':'r','delta':'private reasoning'})]
        for change in cases:
            raw = episode()
            change(raw)
            with self.assertRaises(Rejected):
                normalize_episode(raw, VALIDATE_RECORD)

    def test_train_only_pilot_remains_explicitly_incomplete_for_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output, manifest = self.collect_sources(directory, [episode(1), episode(2)])
            self.assertEqual(manifest['accepted_episodes'], 2)
            self.assertEqual(manifest['records_by_split'], {'train':4, 'valid':0, 'test':0})
            self.assertEqual(manifest['format_version'], 2)
            self.assertEqual(manifest['raw_format_version'], 1)
            self.assertEqual(manifest['corpus_version'], CORPUS_VERSION)
            self.assertEqual({key: manifest[key] for key in TEACHER_PROVENANCE}, TEACHER_PROVENANCE)
            self.assertEqual(manifest['dataset_readiness']['empty_splits'], ['valid','test'])
            self.assertTrue(manifest['dataset_readiness']['pilot_only'])
            self.assertFalse(manifest['dataset_readiness']['all_training_validation_test_splits_nonempty'])
            self.assertFalse(manifest['dataset_readiness']['generalization_evaluation_established'])
            self.assertEqual((output/'valid.jsonl').read_text(), '')
            self.assertEqual((output/'test.jsonl').read_text(), '')
            for entry in manifest['episodes']:
                self.assertEqual(entry['source_sha256'], hashlib.sha256((source/entry['source']).read_bytes()).hexdigest())

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

    def collect_sources(self, directory, episodes, **options):
        source = Path(directory) / "raw"
        source.mkdir()
        for index, raw in enumerate(episodes):
            (source / (str(index) + ".json")).write_text(json.dumps(raw))
        output = Path(directory) / "dataset"
        return source, output, collect(source, output, VALIDATOR, **options)

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

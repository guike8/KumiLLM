#!/usr/bin/env python3
"""Fail-closed per-step admission for native Kumi file-only audio observations.

This does not train, run a model, edit source audio, or claim DAW verification.
Public text needs a separately reviewed, hash-pinned fact receipt.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def load_json(text):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'duplicate_json_key')
            out[key] = value
        return out
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite_json')))


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def normalize_messages(messages):
    result = []
    for raw in messages:
        require(set(raw) <= {'role', 'content', 'tool_calls', 'tool_call_id', 'name'}, 'unknown_native_message_field')
        message = copy.deepcopy(raw)
        content = message.get('content')
        if isinstance(content, list):
            require(all(set(x) == {'type', 'text'} and x['type'] == 'text' for x in content),
                    'nontext_or_private_native_content')
            message['content'] = '\n'.join(x['text'] for x in content)
        require(message.get('content') is None or isinstance(message['content'], str), 'unsupported_native_content')
        for call in message.get('tool_calls', []):
            require(call.get('type') == 'function', 'unsupported_native_call')
            arguments = call['function']['arguments']
            if isinstance(arguments, str):
                arguments = load_json(arguments)
            require(isinstance(arguments, dict), 'native_arguments_not_object')
            call['function']['arguments'] = arguments
        result.append(message)
    return result


def validate_schema(value, schema):
    """Validate the complete constraint subset used by this capture's listen args."""
    supported = {'type', 'properties', 'additionalProperties', 'required', 'description',
                 'enum', 'minimum', 'maximum', 'exclusiveMinimum', 'minLength', 'maxLength',
                 'items', 'minItems', 'maxItems'}
    require(set(schema) <= supported, 'unsupported_active_schema_constraint')
    types = {'object': lambda x: isinstance(x, dict), 'array': lambda x: isinstance(x, list),
             'string': lambda x: isinstance(x, str), 'number': lambda x: type(x) in (int, float),
             'integer': lambda x: type(x) is int, 'boolean': lambda x: type(x) is bool,
             'null': lambda x: x is None}
    require(schema.get('type') in types and types[schema['type']](value), 'argument_type_mismatch')
    if 'enum' in schema:
        require(value in schema['enum'], 'argument_enum_mismatch')
    for key, test in [('minimum', lambda x, n: x >= n), ('maximum', lambda x, n: x <= n),
                      ('exclusiveMinimum', lambda x, n: x > n)]:
        if key in schema:
            require(test(value, schema[key]), 'argument_' + key)
    for key, test in [('minLength', lambda x, n: len(x) >= n), ('maxLength', lambda x, n: len(x) <= n),
                      ('minItems', lambda x, n: len(x) >= n), ('maxItems', lambda x, n: len(x) <= n)]:
        if key in schema:
            require(test(value, schema[key]), 'argument_' + key)
    if isinstance(value, dict):
        properties = schema.get('properties', {})
        require(set(schema.get('required', [])) <= value.keys(), 'missing_required_argument')
        require(schema.get('additionalProperties') is not False or set(value) <= properties.keys(), 'unknown_argument')
        for key, child in value.items():
            require(key in properties, 'unvalidated_argument')
            validate_schema(child, properties[key])
    if isinstance(value, list):
        require('items' in schema, 'array_item_schema_missing')
        for child in value:
            validate_schema(child, schema['items'])


def is_runtime_nudge(message):
    return message.get('role') == 'user' and (message.get('content') or '').startswith('[Kumi]')


def at_pointer(value, pointer):
    for part in pointer.strip('/').split('/'):
        if part:
            part = part.replace('~1', '/').replace('~0', '~')
            value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def native_events(root):
    result = {}
    if root:
        for path in root.rglob('*.events.jsonl'):
            for line in path.read_text().splitlines():
                event = load_json(line)
                if event.get('type') == 'tool-end':
                    result.setdefault(event['id'], []).append((event, str(path)))
    return result


def native_text_hashes(root):
    result = set()
    if root:
        for path in root.rglob('*.events.jsonl'):
            for line in path.read_text().splitlines():
                event = load_json(line)
                if event.get('type') == 'text':
                    result.add(hashlib.sha256(event['text'].encode()).hexdigest())
    return result


def case_identity(envelope, evidence):
    require(evidence is not None, 'native_task_evidence_missing')
    human = [m['content'].split('\n\n<current_observation_untrusted>')[0]
             for m in envelope['request']['messages']
             if m['role'] == 'user' and not is_runtime_nudge(m)]
    matches = []
    for path in evidence.rglob('task.json'):
        task = load_json(path.read_text())
        turns = task.get('turns', [])
        if turns and human and human[0] == turns[0]:
            require(human == turns[:len(human)], 'native_human_turn_sequence_mismatch')
            matches.append((task['id'], len(human) - 1, hashlib.sha256(path.read_bytes()).hexdigest()))
    require(len(matches) == 1, 'native_source_case_not_uniquely_grounded')
    return matches[0]


def build(teacher, assets, reviews, evidence=None):
    requests = []
    for path in teacher.glob('req_*.request.json'):
        value = load_json(path.read_text())
        require(value['schema'] == 'kumi-private-model-request/1', 'unsupported_request_capture')
        require(value['collection_kind'] == 'file_only_kumi_dsp', 'not_file_only_capture')
        require(digest(value['request']) == value['request_sha256'], 'request_hash_changed')
        require(digest(value['request']['tools']) == value['tools_sha256'], 'tool_schema_hash_changed')
        require(digest(value['scope']) == value['scope_sha256'], 'scope_hash_changed')
        requests.append((path, value))
    requests.sort(key=lambda x: x[1]['received_at'])
    source_hash = digest([{'call_id': r['call_id'], 'request_sha256': r['request_sha256']} for _, r in requests])
    observations = {}
    for path, request in requests:
        for message in request['request']['messages']:
            if message['role'] == 'tool':
                content = load_json(message['content'])
                key = message['tool_call_id']
                if key in observations:
                    require(observations[key]['content'] == content, 'changing_tool_result')
                observations[key] = {'content': content, 'source_call_id': request['call_id'],
                                     'request_file': str(path), 'content_sha256': digest(content)}
    events = native_events(evidence)
    observed_texts = native_text_hashes(evidence)
    rows, receipts, withheld = [], [], []
    verified_assets = {}
    for _, request in requests:
        for item in request['scope']['assets']:
            local = assets / Path(item['path']).name
            require(local.is_file() and hashlib.sha256(local.read_bytes()).hexdigest() == item['sha256'], 'asset_hash_changed')
            verified_assets[item['path']] = item['sha256']
    for index, (path, envelope) in enumerate(requests):
        answer_path = path.with_name(path.name.replace('.request.json', '.response.json'))
        if not answer_path.exists():
            withheld.append({'call_id': envelope['call_id'], 'reason': 'teacher_response_not_yet_captured'})
            continue
        answer = load_json(answer_path.read_text())
        try:
            require(answer['call_id'] == envelope['call_id'], 'wrong_response_id')
            response_receipt = load_json(path.with_name(path.name.replace('.request.json', '.receipt.json')).read_text())
            require(response_receipt['sha256'] == digest(answer), 'teacher_response_receipt_mismatch')
            tools = envelope['request']['tools']
            schemas = {t['function']['name']: t['function']['parameters'] for t in tools}
            history = normalize_messages(envelope['request']['messages'])
            runtime_turns = [i for i, message in enumerate(history) if is_runtime_nudge(message)]
            require(not runtime_turns, 'redundant_native_audition_nudge_excluded')
            case_id, human_turn_index, task_sha256 = case_identity(envelope, evidence)
            target = {'role': 'assistant', 'content': answer.get('content')}
            closure = []
            if answer.get('tool_calls'):
                require(not answer.get('content'), 'mixed_prose_tool_target_requires_separate_review')
                target['tool_calls'] = []
                for call in answer['tool_calls']:
                    require(call['name'] == 'listen' and call['name'] in schemas, 'only_reviewed_listen_target_supported')
                    validate_schema(call['arguments'], schemas[call['name']])
                    require(call['arguments'].get('file') in verified_assets, 'unpinned_file')
                    require('compare_to' not in call['arguments'] or call['arguments']['compare_to'] in verified_assets, 'unpinned_reference')
                    require(not {'mix','track','tracks','form','transcribe','whole','from_beat','beats'} & call['arguments'].keys(), 'nonfile_listen_mode')
                    obs = observations[call['id']]
                    result = obs['content']
                    require(result.get('kumiAudio') == 1 or result.get('comparison', {}).get('kumiAudioComparison') == 1, 'listen_result_not_success')
                    # The following native request must contain the exact target call before its result.
                    following = next(r for _, r in requests if r['call_id'] == obs['source_call_id'])
                    prior = next(c for m in following['request']['messages'] if m['role'] == 'assistant'
                                 for c in m.get('tool_calls', []) if c['id'] == call['id'])
                    require(prior['function']['name'] == call['name'] and
                            load_json(prior['function']['arguments']) == call['arguments'], 'native_executed_call_mismatch')
                    event_matches = events.get(call['id'], [])
                    require(event_matches and all(e['isError'] is False and e['name'] == 'listen' for e, _ in event_matches), 'native_success_event_missing_or_failed')
                    target['tool_calls'].append({'id': call['id'], 'type': 'function', 'function': {
                        'name': call['name'], 'arguments': call['arguments']}})
                    closure.append({'tool_call_id': call['id'], **obs, 'native_event_files': [p for _,p in event_matches]})
                kind = 'tool_only'
                terminal = {}
            else:
                review = reviews[envelope['call_id']]
                require(review['content_sha256'] == hashlib.sha256(answer['content'].encode()).hexdigest(), 'public_text_changed_after_review')
                require(review['content_sha256'] in observed_texts, 'native_public_text_event_missing')
                require(review['all_claims_reviewed'] is True and review['no_subjective_quality_or_audition_claim'] is True, 'public_text_claim_review_missing')
                require(review['fact_groundings'], 'no_reviewed_fact_groundings')
                context_observations = {m['tool_call_id']: load_json(m['content'])
                                        for m in history if m['role'] == 'tool'}
                for fact in review['fact_groundings']:
                    observed = observations[fact['tool_call_id']]
                    require(context_observations.get(fact['tool_call_id']) == observed['content'],
                            'public_fact_not_observed_in_native_context')
                    source_events = events.get(fact['tool_call_id'], [])
                    require(source_events and all(e['isError'] is False and e['name'] == 'listen'
                                                   for e, _ in source_events),
                            'public_fact_source_native_success_missing_or_failed')
                    require(at_pointer(observed['content'], fact['pointer']) == fact['value'], 'public_fact_grounding_mismatch')
                kind = 'checked_public_response'
                terminal = {'contract_validated': True, 'native_response_exact_match': True,
                            'review': review, 'review_scope': 'hash_pinned_manual_numeric_and_comparative_claim_audit'}
            record = {'messages': history + [target], 'tools': tools, 'metadata': {
                'episode_id': 'native-file-001-' + case_id, 'source_case_id': case_id,
                'source_call_id': envelope['call_id'], 'human_turn_index': human_turn_index,
                'source_goal_task_sha256': task_sha256,
                'source_audio_id': 'open_goldberg_2012_kimiko_ishizaka',
                'source_sha256': source_hash, 'call_index': index, 'teacher': 'gpt-6.1-sol-ultra-codex-agent',
                'transport': 'codex_agent_openai_compatible_file_queue', 'source_request_sha256': envelope['request_sha256'],
                'collection_kind': 'file_only_kumi_dsp', 'real_daw_verified': False, 'human_preference_verified': False,
                'mask_all_history_required': True, 'train_on_final_assistant_only_required': True,
                'loss_contract_required': 'strict_final_target_excluding_padding', 'ready_for_training_ingestion': False,
                'source_group_split': 'train_only_no_validation_claim'}}
            rows.append(record)
            receipts.append({'call_id': envelope['call_id'], 'call_index': index, 'target_kind': kind,
                'record_sha256': digest(record), 'tools_sha256': digest(tools),
                'prefix_sha256': digest(history), 'target_sha256': digest(target),
                'source_sha256': source_hash, 'closure_evidence': closure, 'terminal_validation': terminal})
        except (ValueError, KeyError, OSError, StopIteration) as error:
            withheld.append({'call_id': envelope['call_id'], 'reason': str(error)})
    return rows, {'format_version': 1, 'kind': 'reviewed_native_audio_candidates',
        'record_count': len(rows), 'records': receipts, 'withheld': withheld,
        'episode_count': len({row['metadata']['episode_id'] for row in rows}),
        'human_turn_count': len({(row['metadata']['episode_id'], row['metadata']['human_turn_index']) for row in rows}),
        'source_sha256': source_hash, 'source_requests': len(requests), 'exact_native_schema_capture_missing': 0,
        'mask_prompt_required': True, 'max_seq_length': 32768, 'ready_for_training_ingestion': False,
        'native_event_evidence_required': True, 'source_assets_sha256': verified_assets,
        'split': 'train_only', 'real_daw_verified': False, 'human_preference_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() or not any(args.output.iterdir()), 'fresh_output_required')
    rows, manifest = build(args.teacher, args.assets, load_json(args.reviews.read_text()), args.evidence)
    args.output.mkdir(parents=True, exist_ok=True)
    data = ''.join(canonical(row) + '\n' for row in rows)
    (args.output / 'train.jsonl').write_text(data)
    manifest['candidate_file_sha256'] = hashlib.sha256(data.encode()).hexdigest()
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'records': len(rows), 'withheld': len(manifest['withheld']), 'ready_for_training_ingestion': False}))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Prepare offline, hash-pinned Kumi/xLAM ablation data; never train or load weights."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys

from validate_dataset import GROUP_KEYS, validate_record
from verification_helpers import (canonical, digest, inspect_render, load_json,
                                  probe_loss, require, tokenizer_fingerprint)

HERE = Path(__file__).resolve().parent
CONTRACT = 'strict_final_target_excluding_padding'
DATASET = 'Salesforce/xlam-function-calling-60k'


def named_path(value):
    name, separator, path = value.partition('=')
    if not separator or not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or not path:
        raise argparse.ArgumentTypeError('Use SOURCE_NAME=/absolute/file/path')
    return name, Path(path)


def snapshot_identity(record):
    return digest(canonical({key: record[key] for key in ('messages', 'tools')}).encode())


def jsonl(path):
    require(path.is_file() and not path.is_symlink(), 'regular_source_file_required')
    records = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = load_json(line)
        require(not validate_record(record), f'structural_validation_failed:{path.name}:{i}')
        require(isinstance(record.get('metadata'), dict), 'source_metadata_required')
        records.append(record)
    require(records, 'empty_source_file')
    return records


def verify_source_hash(path, manifest, split):
    expected = manifest.get('output_sha256', {}).get(split + '.jsonl')
    if expected is None and split == 'train':
        expected = manifest.get('candidate_file_sha256')
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected),
            'source_manifest_lacks_split_hash')
    require(digest(path.read_bytes()) == expected, 'source_file_manifest_hash_mismatch')
    return expected


def check_split_isolation(train, valid):
    fingerprints = {snapshot_identity(r) for r in train}
    require(not fingerprints.intersection(snapshot_identity(r) for r in valid),
            'exact_record_cross_split')
    groups = {}
    for split, records in [('train', train), ('valid', valid)]:
        for record in records:
            metadata = record['metadata']
            for field in GROUP_KEYS:
                value = metadata.get(field)
                if value is None:
                    continue
                key = field, str(value)
                require(key not in groups or groups[key] == split,
                        'source_group_cross_split:' + field)
                groups[key] = split


def check_xlam_pool(train, valid, manifest):
    for split, records in [('train', train), ('valid', valid)]:
        for record in records:
            m = record['metadata']
            require(m.get('source_dataset') == DATASET and m.get('license') == 'CC-BY-4.0',
                    'xlam_source_and_license_required')
            require(m.get('source_revision') == manifest['source_revision'] and
                    m.get('source_file_sha256') == manifest['source_file_sha256'],
                    'xlam_pinned_provenance_mismatch')
            require(m.get('source_group_split') == split, 'xlam_split_metadata_mismatch')
            require(m.get('locally_executed') is False, 'xlam_execution_claim_not_allowed')
            require(m.get('source_api_families') and isinstance(m['source_api_families'], list),
                    'xlam_api_families_required')
            require(len(record['messages']) == 3 and not any(x['role'] == 'tool' for x in record['messages']),
                    'unexpected_xlam_tool_result_or_trajectory')
    families = [{f for record in records for f in record['metadata']['source_api_families']}
                for records in [train, valid]]
    require(not families[0] & families[1], 'xlam_api_family_cross_split')
    check_split_isolation(train, valid)


def check_allowlist(allowlist, train, valid, manifest):
    require(allowlist.get('schema') == 'kumi-xlam-semantic-allowlist/1', 'semantic_allowlist_schema')
    require(allowlist.get('source_revision') == manifest['source_revision'] and
            allowlist.get('source_file_sha256') == manifest['source_file_sha256'],
            'semantic_allowlist_source_mismatch')
    expected = {name: manifest['files'][name]['sha256'] for name in ['train.jsonl', 'valid.jsonl']}
    require(allowlist.get('pool_files_sha256') == expected, 'semantic_allowlist_pool_hash_mismatch')
    require(allowlist.get('locally_executed') is False and allowlist.get('review_scope'),
            'semantic_review_must_not_claim_execution')
    allowed = {}
    for split, records in [('train', train), ('valid', valid)]:
        values = allowlist['allowed_source_row_ids'][split]
        require(isinstance(values, list) and all(type(x) in [str, int] for x in values),
                'semantic_allowlist_bad_source_ids')
        ids = {canonical(x) for x in values}
        require(len(ids) == len(values), 'semantic_allowlist_duplicate_ids')
        available = {canonical(r['metadata']['source_row_id']) for r in records}
        require(ids <= available and ids, 'semantic_allowlist_unknown_or_empty_split')
        allowed[split] = ids
    return allowed


def runtime_parser_roundtrip(record, tokenizer, parse_tool_call):
    """Use the installed inference parser; no alternative decoder or coercion."""
    try:
        options = {'tools': record['tools'], 'tokenize': False}
        prompt = tokenizer.apply_chat_template(record['messages'][:-1], add_generation_prompt=True, **options)
        full = tokenizer.apply_chat_template(record['messages'], add_generation_prompt=False, **options)
        require(isinstance(prompt, str) and isinstance(full, str) and full.startswith(prompt),
                'runtime_parser_render_prefix_mismatch')
        suffix = full[len(prompt):]
        bodies = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', suffix, re.S)
        wanted = [{'name': call['function']['name'], 'arguments': call['function']['arguments']}
                  for call in record['messages'][-1]['tool_calls']]
        require(len(bodies) == len(wanted), 'runtime_parser_tool_count_mismatch')
        for body, expected in zip(bodies, wanted):
            actual = parse_tool_call(body.strip(), record['tools'])
            require(canonical(actual) == canonical(expected), 'runtime_parser_typed_arguments_changed')
        return {'ok': True, 'tool_calls': len(wanted)}
    except Exception as error:
        reason = str(error) if str(error).startswith('runtime_parser_') else type(error).__name__
        return {'ok': False, 'reason': reason}


def diverse_subset(records, counts, kumi_rows, kumi_targets, seed, allowed_ids=None,
                   compatible_indices=None):
    """Whole rows, both <=10% equal-row updates and <=10% target-token share."""
    row_limit = kumi_rows // 9
    token_limit = kumi_targets // 9
    remaining = [i for i, record in enumerate(records)
                 if not record['metadata'].get('underspecified_containers_or_values') and
                 (allowed_ids is None or canonical(record['metadata']['source_row_id']) in allowed_ids) and
                 (compatible_indices is None or i in compatible_indices)]
    underspecified = sum(bool(r['metadata'].get('underspecified_containers_or_values')) for r in records)
    semantic_excluded = (sum(canonical(r['metadata']['source_row_id']) not in allowed_ids for r in records)
                         if allowed_ids is not None else 0)
    selected, used_tokens = [], 0
    style_use, type_use = Counter(), Counter()
    covered_families = set()
    stable = {i: digest(canonical([seed, records[i]['metadata'].get('source_row_canonical_sha256',
                                       snapshot_identity(records[i]))]).encode()) for i in remaining}
    while remaining and len(selected) < row_limit:
        fits = [i for i in remaining if used_tokens + counts[i]['target_tokens'] <= token_limit]
        if not fits:
            break

        def priority(i):
            m = records[i]['metadata']
            types = m.get('argument_types', [])
            fresh = len(set(m['source_api_families']) - covered_families)
            return (style_use[m.get('query_style', 'unknown')],
                    min((type_use[t] for t in types), default=0), -fresh, stable[i])

        index = min(fits, key=priority)
        selected.append(index)
        remaining.remove(index)
        used_tokens += counts[index]['target_tokens']
        m = records[index]['metadata']
        style_use[m.get('query_style', 'unknown')] += 1
        type_use.update(m.get('argument_types', []))
        covered_families.update(m['source_api_families'])
    require(selected, 'no_xlam_row_fits_both_mixture_budgets')
    require(len(selected) * 9 <= kumi_rows and used_tokens * 9 <= kumi_targets,
            'mixture_budget_exceeded')
    return selected, {'row_limit': row_limit, 'target_token_limit': token_limit,
                      'pool_rows': len(records), 'fully_specified_eligible_pool_rows': len(records) - underspecified,
                      'underspecified_pool_rows_excluded': underspecified,
                      'semantic_review_excluded_pool_rows': semantic_excluded,
                      'runtime_parser_incompatible_pool_rows': len(records) - len(compatible_indices) if compatible_indices is not None else 0,
                      'fully_specified_and_semantically_allowed_pool_rows': len(remaining) + len(selected),
                      'schema_filter': 'Exclude every nonempty underspecified_containers_or_values flag before mixture sampling',
                      'selected_rows': len(selected), 'selected_target_tokens': used_tokens,
                      'row_fraction': len(selected) / (kumi_rows + len(selected)),
                      'target_token_fraction': used_tokens / (kumi_targets + used_tokens),
                      'query_styles': dict(style_use), 'argument_types': dict(type_use),
                      'api_families': sorted(covered_families)}


def dump_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def dump_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = ''.join(canonical(record) + '\n' for record in records).encode()
    path.write_bytes(raw)
    return {'sha256': digest(raw), 'bytes': len(raw), 'records': len(records)}


def tokenize(records, name, tokenizer, dataset_class, cap):
    report = []
    for index, record in enumerate(records):
        target_kind = 'tool_only' if record['messages'][-1].get('tool_calls') else 'checked_public_response'
        counts = inspect_render(record, {'target_kind': target_kind}, tokenizer, dataset_class, cap)
        report.append({'index': index, 'record_sha256': digest(canonical(record).encode()), **counts})
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            print(json.dumps({'stage': 'tokenizing', 'source': name,
                              'checked': index + 1, 'total': len(records)}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--kumi-data', type=Path, required=True,
                        help='Root-prepared train/valid/source-manifest directory; no tests')
    parser.add_argument('--xlam-data', type=Path, required=True,
                        help='Converted xLAM train/valid/manifest directory; tests ignored')
    parser.add_argument('--xlam-allowlist', type=Path,
                        help='Hash-bound semantic review IDs; required for a trainable stage')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--microbatches', type=int, default=384)
    parser.add_argument('--max-seq-length', type=int, default=32768)
    args = parser.parse_args()
    require(args.max_seq_length == 32768, 'reviewed_sequence_cap_required')
    require(args.microbatches > 0 and args.microbatches % 8 == 0,
            'microbatches_must_complete_accumulation_groups_of_eight')
    require(args.training_root.is_dir() and args.model.is_dir(), 'existing_workspace_and_model_required')
    output = args.output.resolve()
    require(not output.exists(), 'fresh_staging_output_required')
    require(not output.is_relative_to(HERE), 'output_must_not_nest_inside_staging_source')
    require(output.is_relative_to(args.training_root.resolve()), 'staging_must_stay_inside_training_workspace')
    output.mkdir(parents=True, mode=0o700)
    report = {'schema': 'kumi-xlam-next-run-stage/1', 'stage': 'preparing',
              'training_started': False, 'base_weights_loaded': False,
              'running_v1_modified': False, 'provided_test_data': False,
              'loss_contract': CONTRACT, 'stage_validated': False}
    try:
        require(importlib.metadata.version('mlx-lm') == '0.32.2', 'pinned_mlx_lm_required')
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          HF_HUB_DISABLE_IMPLICIT_TOKEN='1', DO_NOT_TRACK='1',
                          TOKENIZERS_PARALLELISM='false')
        from transformers import AutoTokenizer
        from mlx_lm.tuner.datasets import ChatDataset
        from mlx_lm.tuner.trainer import default_loss
        from mlx_lm.tool_parsers import qwen3_coder
        from final_target_loss import loss as strict_loss
        tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()),
                                                 local_files_only=True, trust_remote_code=False)
        strict_probe = probe_loss(strict_loss)
        require(strict_probe['ok'], 'strict_final_target_loss_probe_failed')
        require(not (args.kumi_data / 'test.jsonl').exists(), 'kumi_stage_input_must_not_have_test')
        kmanifest = load_json((args.kumi_data / 'source-manifest.json').read_text())
        require(kmanifest['schema'] == 'kumi-next-run-source-inputs/1' and
                kmanifest['contains_test_file'] is False and
                kmanifest['requires_strict_future_loss'] is True,
                'verified_aggregate_source_manifest_required')
        sources, kumi = [], {'train': [], 'valid': []}
        for split in ['train', 'valid']:
            path = args.kumi_data / (split + '.jsonl')
            raw = path.read_bytes()
            details = kmanifest['sources'][split]
            require(digest(raw) == details['sha256'], 'kumi_aggregate_file_hash_mismatch')
            offset = 0
            for part in details['parts']:
                portion = raw[offset:offset + part['bytes']]
                require(len(portion) == part['bytes'] and digest(portion) == part['sha256'],
                        'kumi_aggregate_part_hash_mismatch')
                require(sum(bool(line.strip()) for line in portion.splitlines()) == part['records'],
                        'kumi_aggregate_part_count_mismatch')
                offset += part['bytes']
            require(offset == len(raw), 'kumi_aggregate_extra_or_missing_bytes')
            kumi[split] = jsonl(path)
            require(len(kumi[split]) == details['records'], 'kumi_aggregate_count_mismatch')
            sources.append({'split': split, 'path': str(path.resolve()),
                            'file_sha256': details['sha256'], 'records': len(kumi[split]),
                            'parts': details['parts'],
                            'manifest_sha256': digest((args.kumi_data / 'source-manifest.json').read_bytes())})
        require(len(kumi['train']) == 341 and len(kumi['valid']) == 86,
                'expected_verified_kumi_341_train_86_valid_required')
        check_split_isolation(kumi['train'], kumi['valid'])
        xmanifest = load_json((args.xlam_data / 'manifest.json').read_text())
        xtrain, xvalid = jsonl(args.xlam_data / 'train.jsonl'), jsonl(args.xlam_data / 'valid.jsonl')
        for split, path in [('train', args.xlam_data / 'train.jsonl'), ('valid', args.xlam_data / 'valid.jsonl')]:
            require(digest(path.read_bytes()) == xmanifest['files'][split + '.jsonl']['sha256'],
                    'xlam_pool_file_hash_mismatch')
        check_xlam_pool(xtrain, xvalid, xmanifest)
        allowlist = load_json(args.xlam_allowlist.read_text()) if args.xlam_allowlist else None
        allowed = check_allowlist(allowlist, xtrain, xvalid, xmanifest) if allowlist else None
        token_reports = {}
        for name, records in [('kumi_train', kumi['train']), ('kumi_valid', kumi['valid']),
                              ('xlam_pool_train', xtrain), ('xlam_aux_valid', xvalid)]:
            token_reports[name] = tokenize(records, name, tokenizer, ChatDataset, args.max_seq_length)
        parser_reports = {}
        for split, records in [('train', xtrain), ('valid', xvalid)]:
            parser_reports[split] = [dict(index=i, source_row_id=r['metadata']['source_row_id'],
                                         **runtime_parser_roundtrip(r, tokenizer, qwen3_coder.parse_tool_call))
                                     for i, r in enumerate(records)]
        compatible = {side: {r['index'] for r in rows if r['ok']}
                      for side, rows in parser_reports.items()}
        kumi_targets = sum(x['target_tokens'] for x in token_reports['kumi_train'])
        indices, budget = diverse_subset(xtrain, token_reports['xlam_pool_train'],
                                        len(kumi['train']), kumi_targets, args.seed,
                                        allowed['train'] if allowed else None, compatible['train'])
        chosen = [xtrain[i] for i in indices]
        mixed = kumi['train'] + chosen
        random.Random(args.seed).shuffle(mixed)
        files = {}
        for variant, training in [('baseline', kumi['train']), ('mixed', mixed)]:
            folder = output / variant / 'data'
            for split, records in [('train', training), ('valid', kumi['valid'])]:
                relative = str(Path(variant) / 'data' / (split + '.jsonl'))
                files[relative] = dump_jsonl(output / relative, records)
            require({p.name for p in folder.iterdir()} == {'train.jsonl', 'valid.jsonl'},
                    'training_directory_must_have_only_train_and_valid')
        for split, records in [('train', xtrain), ('valid', xvalid)]:
            relative = 'auxiliary/xlam-reservoir/' + split + '.jsonl'
            files[relative] = dump_jsonl(output / relative, records)
        aux_valid = [r for i, r in enumerate(xvalid) if i in compatible['valid'] and
                     (allowed is None or canonical(r['metadata']['source_row_id']) in allowed['valid'])]
        require(aux_valid, 'no_reviewed_runtime_compatible_auxiliary_validation_rows')
        relative = 'auxiliary/xlam-validation/valid.jsonl'
        files[relative] = dump_jsonl(output / relative, aux_valid)
        dump_json(output / 'auxiliary/xlam-source-manifest.json', xmanifest)
        dump_json(output / 'auxiliary/kumi-source-manifest.json', kmanifest)
        if allowlist:
            dump_json(output / 'auxiliary/xlam-semantic-allowlist.json', allowlist)
        for relative in ['auxiliary/xlam-source-manifest.json', 'auxiliary/kumi-source-manifest.json'] + (['auxiliary/xlam-semantic-allowlist.json'] if allowlist else []):
            path = output / relative
            files[relative] = {'sha256': digest(path.read_bytes()), 'bytes': path.stat().st_size}
        import yaml
        base_config = {'model': str(args.model.resolve()), 'train': True, 'test': False,
                       'fine_tune_type': 'lora', 'optimizer': 'adamw', 'seed': args.seed,
                       'num_layers': -1, 'batch_size': 1, 'grad_accumulation_steps': 8,
                       'iters': args.microbatches, 'val_batches': -1,
                       'learning_rate': 0.00002, 'steps_per_report': 1,
                       'steps_per_eval': args.microbatches, 'save_every': min(64, args.microbatches),
                       'max_seq_length': args.max_seq_length, 'grad_checkpoint': True,
                       'mask_prompt': True, 'trust_remote_code': False, 'report_to': 'local',
                       'lora_parameters': {'keys': ['self_attn.q_proj', 'self_attn.v_proj'],
                                           'rank': 16, 'scale': 16.0, 'dropout': 0.0},
                       'loss_contract': CONTRACT}
        for variant in ['baseline', 'mixed']:
            config = dict(base_config, data=str(output / variant / 'data'),
                          adapter_path=str(output / variant / 'adapters'))
            path = output / variant / 'config.yaml'
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            files[str(path.relative_to(output))] = {'sha256': digest(path.read_bytes()),
                                                   'bytes': path.stat().st_size}
        shutil.copytree(HERE, output / 'tooling', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        hook_hash = digest((output / 'tooling/final_target_loss.py').read_bytes())
        report.update(stage='prepared_not_started' if allowlist else 'tokenized_pending_semantic_review',
                      stage_validated=allowlist is not None,
                      ready_only_with_explicit_strict_runner=allowlist is not None,
                      model=str(args.model.resolve()), training_root=str(args.training_root.resolve()),
                      sources=sources, xlam_source_file_sha256=xmanifest['source_file_sha256'],
                      source_revision=xmanifest['source_revision'], seed=args.seed,
                      semantic_review_applied=allowlist is not None,
                      semantic_review_sha256=digest(args.xlam_allowlist.read_bytes()) if allowlist else None,
                      auxiliary_validation_allowed_records=len(aux_valid),
                      microbatches=args.microbatches, complete_optimizer_updates=args.microbatches // 8,
                      primary_validation='Kumi-only 86 records; xLAM auxiliary validation is outside MLX data dirs',
                      kumi_training_target_tokens=kumi_targets, mixture=budget,
                      selected_xlam_pool_indices=indices,
                      selected_xlam_source_ids=[r['metadata']['source_row_id'] for r in chosen],
                      tokenization=token_reports,
                      runtime_parser_compatibility=parser_reports,
                      runtime_parser_source_sha256=digest(Path(inspect.getsourcefile(qwen3_coder)).read_bytes()),
                      runtime_parser_unchanged=True,
                      tokenizer_files_sha256=tokenizer_fingerprint(args.model),
                      native_chatdataset_source_sha256=digest(inspect.getsource(ChatDataset).encode()),
                      native_loss_probe=probe_loss(default_loss), strict_loss_probe=strict_probe,
                      strict_loss_file_sha256=hook_hash,
                      tooling_sha256={p.name: digest(p.read_bytes()) for p in sorted((output / 'tooling').glob('*.py'))},
                      versions={n: importlib.metadata.version(n) for n in ['mlx', 'mlx-lm', 'transformers']},
                      files=files, source_metadata_preserved=True,
                      source_readiness_flags_are_provenance=True,
                      stage_eligibility='All source hashes, schema/masks/tokens checked; native audio and external rows require the explicit new-run loss contract.',
                      runtime_sampling='MLX shuffles rows; metadata weights are not used. Materialized-row and one-pass supervised-token shares are bounded. The fixed update budget may replay records; actual gradient contribution is not bounded by these percentages.')
        for name in files:
            (output / name).chmod(0o444)
        dump_json(output / 'manifest.json', report)
        print(json.dumps({'stage': report['stage'], 'training_started': False,
                          'kumi_train': 341, 'kumi_valid': 86, 'xlam_selected': len(chosen),
                          'mixture': budget, 'output': str(output)}), flush=True)
    except Exception as error:
        report.update(stage='failed', error=type(error).__name__ + ': ' + str(error))
        dump_json(output / 'manifest.json', report)
        raise


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Offline exact-tokenizer and final-target loss diagnostic; never trains."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys

from verification_helpers import (canonical, digest, inspect_render, load_json,
                                  probe_loss, require, tokenizer_fingerprint,
                                  RenderError)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path(__file__).parent / 'artifacts')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seq-length', type=int, default=32768)
    args = parser.parse_args()
    require(1 <= args.max_seq_length <= 32768, 'production_cap_is_32768')
    require(args.model.is_dir(), 'existing_local_model_required')
    require(not args.output.exists() or not any(args.output.iterdir()), 'fresh_output_required')
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'kumi-native-audio-tokenization/1', 'ok': False,
              'max_seq_length': args.max_seq_length, 'records': [],
              'ready_for_current_v1_trainer': False, 'ready_for_training_ingestion': False,
              'requires_future_trainer_loss_contract': 'strict_final_target_excluding_padding',
              'no_training_or_model_load_performed': True, 'local_files_only': True}
    try:
        manifest = load_json((args.data / 'manifest.json').read_text())
        data = (args.data / 'train.jsonl').read_bytes()
        require(manifest['kind'] == 'reviewed_native_audio_candidates', 'wrong_candidate_kind')
        require(digest(data) == manifest['candidate_file_sha256'], 'candidate_file_changed')
        records = [load_json(line) for line in data.decode().splitlines() if line.strip()]
        require(len(records) == manifest['record_count'] == len(manifest['records']), 'candidate_count_mismatch')
        require(records, 'no_candidates')
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        from transformers import AutoTokenizer
        from mlx_lm.tuner.datasets import ChatDataset
        from mlx_lm.tuner.trainer import default_loss
        from final_target_loss import loss as final_target_loss
        tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()), local_files_only=True,
                                                 trust_remote_code=False)
        report.update(model=str(args.model.resolve()), candidate_file_sha256=digest(data),
                      tokenizer_files_sha256=tokenizer_fingerprint(args.model),
                      versions={n: importlib.metadata.version(n) for n in ('transformers', 'mlx', 'mlx-lm')},
                      python=sys.version.split()[0],
                      native_chatdataset_source_sha256=digest(inspect.getsource(ChatDataset).encode()),
                      native_loss_source_sha256=digest(inspect.getsource(default_loss).encode()),
                      strict_loss_source_sha256=digest(inspect.getsource(final_target_loss).encode()),
                      native_loss_probe=probe_loss(default_loss), strict_loss_probe=probe_loss(final_target_loss))
        for record, receipt in zip(records, manifest['records']):
            result = {'call_id': receipt['call_id'], 'target_kind': receipt['target_kind'], 'ok': False}
            try:
                require(digest(canonical(record).encode()) == receipt['record_sha256'], 'record_hash_mismatch')
                for field, value in [('tools_sha256', record['tools']), ('prefix_sha256', record['messages'][:-1]),
                                     ('target_sha256', record['messages'][-1])]:
                    require(digest(canonical(value).encode()) == receipt[field], field + '_mismatch')
                require(record['metadata']['mask_all_history_required'] is True
                        and record['metadata']['train_on_final_assistant_only_required'] is True, 'unsafe_mask_contract')
                if receipt['target_kind'] == 'checked_public_response':
                    require(receipt['terminal_validation']['contract_validated'] is True
                            and receipt['terminal_validation']['native_response_exact_match'] is True,
                            'unreviewed_public_response')
                result.update(inspect_render(record, receipt, tokenizer, ChatDataset, args.max_seq_length), ok=True)
            except RenderError as error:
                result.update(error=str(error), **error.diagnostics)
            except Exception as error:
                result['error'] = type(error).__name__ + ': ' + str(error)
            report['records'].append(result)
        report['tokenized_eligible_records'] = sum(r['ok'] for r in report['records'])
        report['ok'] = all(r['ok'] for r in report['records']) and report['strict_loss_probe']['ok']
        report['future_contract_diagnostic_only'] = True
    except Exception as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
    report['verifier_script_sha256'] = digest(Path(__file__).read_bytes())
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['ok'] else 1)


if __name__ == '__main__':
    main()

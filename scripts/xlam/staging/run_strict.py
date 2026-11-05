#!/usr/bin/env python3
"""Validate a prepared stage by default; --start explicitly starts one new variant.

The loss binding is process-local. No installed package or existing v1 file is edited.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys

from verification_helpers import load_json, probe_loss, require, tokenizer_fingerprint

CONTRACT = 'strict_final_target_excluding_padding'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_manifest(stage, variant):
    stage = stage.resolve()
    manifest = load_json((stage / 'manifest.json').read_text())
    require(manifest.get('schema') == 'kumi-xlam-next-run-stage/1' and
            manifest.get('stage_validated') is True and manifest.get('training_started') is False,
            'prepared_not_started_stage_required')
    require(manifest.get('loss_contract') == CONTRACT and
            manifest.get('strict_loss_probe', {}).get('ok') is True,
            'strict_loss_contract_not_verified')
    require(manifest.get('provided_test_data') is False, 'test_data_must_stay_excluded')
    for relative, spec in manifest['files'].items():
        path = stage / relative
        require(path.resolve().is_relative_to(stage) and path.is_file() and not path.is_symlink(),
                'unsafe_stage_file')
        require(sha(path) == spec['sha256'], 'stage_file_hash_changed:' + relative)
    for name, expected in manifest['tooling_sha256'].items():
        require(sha(stage / 'tooling' / name) == expected, 'stage_tooling_hash_changed:' + name)
    require(sha(stage / 'tooling/final_target_loss.py') == manifest['strict_loss_file_sha256'],
            'strict_loss_file_hash_changed')
    import yaml
    config_path = stage / variant / 'config.yaml'
    config = yaml.safe_load(config_path.read_text())
    require(config['loss_contract'] == CONTRACT and config['mask_prompt'] is True and
            config['test'] is False and config['trust_remote_code'] is False,
            'unsafe_training_config')
    data = stage / variant / 'data'
    require({p.name for p in data.iterdir()} == {'train.jsonl', 'valid.jsonl'},
            'only_train_and_primary_valid_allowed')
    require(Path(config['data']).resolve() == data and
            Path(config['adapter_path']).resolve() == stage / variant / 'adapters',
            'config_paths_leave_prepared_variant')
    require(config['model'] == manifest['model'] and config['iters'] == manifest['microbatches'] and
            config['grad_accumulation_steps'] == 8 and config['iters'] % 8 == 0,
            'training_budget_changed')
    return manifest, config_path


def refuse_active_v1(training_root):
    ledger = training_root / 'logs/training-job.json'
    if not ledger.exists():
        return
    record = load_json(ledger.read_text())
    if record.get('stage') not in ['preparing', 'launching', 'running']:
        return
    for key in ['trainer_pid', 'monitor_pid', 'launcher_pid']:
        pid = record.get(key)
        if type(pid) is not int or pid <= 0:
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            pass
        raise ValueError('Existing v1 training process is active; this stage never stops it')


def bind_loss(lora_module, trainer_module, loss):
    """Install the explicit public training hook only in this Python process."""
    require(lora_module.train is trainer_module.train, 'unreviewed_lora_train_alias')
    original = lora_module.train

    def strict_train(*args, **kwargs):
        require('loss' not in kwargs, 'caller_must_not_override_strict_loss')
        return original(*args, loss=loss, **kwargs)

    strict_train.loss_contract = CONTRACT
    strict_train.selected_loss = loss
    lora_module.train = strict_train
    return original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--variant', choices=['baseline', 'mixed'], required=True)
    parser.add_argument('--start', action='store_true', help='Explicitly start this new run; omitted validates only')
    args = parser.parse_args()
    stage = args.stage.resolve()
    manifest, config = check_manifest(stage, args.variant)
    require(Path(__file__).resolve() == stage / 'tooling/run_strict.py',
            'run_the_hash_sealed_staged_runner')
    require(importlib.metadata.version('mlx-lm') == '0.32.2', 'pinned_mlx_lm_required')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_IMPLICIT_TOKEN='1', DO_NOT_TRACK='1',
                      TOKENIZERS_PARALLELISM='false')
    from final_target_loss import loss
    require(tokenizer_fingerprint(Path(manifest['model'])) == manifest['tokenizer_files_sha256'],
            'staged_tokenizer_files_changed')
    from mlx_lm.tool_parsers import qwen3_coder
    require(sha(Path(inspect.getsourcefile(qwen3_coder))) == manifest['runtime_parser_source_sha256'],
            'staged_runtime_parser_changed')
    diagnostic = probe_loss(loss)
    require(diagnostic['ok'], 'strict_loss_gradient_probe_failed')
    from mlx_lm import lora
    from mlx_lm.tuner import trainer
    require('loss' in inspect.signature(trainer.train).parameters and
            'loss' in inspect.signature(trainer.evaluate).parameters,
            'unreviewed_training_loss_api')
    bind_loss(lora, trainer, loss)
    if not args.start:
        print(json.dumps({'stage': 'validated_not_started', 'variant': args.variant,
                          'loss_contract': CONTRACT, 'strict_loss_probe': diagnostic,
                          'process_local_train_alias_bound': lora.train.selected_loss is loss,
                          'training_started': False, 'base_weights_loaded': False}))
        return
    refuse_active_v1(Path(manifest['training_root']))
    variant = stage / args.variant
    require(not (variant / 'adapters').exists() and not (variant / 'run-status.json').exists(),
            'fresh_variant_required_no_accidental_resume')
    import fcntl
    with (stage / '.new-run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from mlx_lm.tuner import callbacks
        from local_metrics import LocalMetrics
        callbacks.SUPPORT_CALLBACK['local'] = LocalMetrics
        status = {'stage': 'running', 'variant': args.variant, 'pid': os.getpid(),
                  'started_at': datetime.now(timezone.utc).isoformat(),
                  'loss_contract': CONTRACT, 'config_sha256': sha(config),
                  'strict_loss_file_sha256': sha(stage / 'tooling/final_target_loss.py'),
                  'installed_trainer_source_sha256': hashlib.sha256(inspect.getsource(trainer.train).encode()).hexdigest(),
                  'running_v1_modified': False}
        status_path = variant / 'run-status.json'
        status_path.write_text(json.dumps(status, indent=2) + '\n')
        sys.argv = [sys.argv[0], '--config', str(config)]
        try:
            lora.main()
            status['stage'] = 'completed'
        except BaseException as error:
            status.update(stage='failed', error=type(error).__name__ + ': ' + str(error))
            raise
        finally:
            status['finished_at'] = datetime.now(timezone.utc).isoformat()
            status_path.write_text(json.dumps(status, indent=2) + '\n')


if __name__ == '__main__':
    main()

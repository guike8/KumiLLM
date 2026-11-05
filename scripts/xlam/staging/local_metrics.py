#!/usr/bin/env python3
"""Run pinned MLX-LM with local JSONL metrics; no package changes or cloud reporting."""
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import time

import mlx.core as mx
from mlx_lm import lora
from mlx_lm.tuner import callbacks


class LocalMetrics:
    def __init__(self, project_name=None, log_dir=None, config=None, wrapped_callback=None):
        if wrapped_callback is not None:
            raise ValueError('This runner permits only local reporting.')
        if importlib.metadata.version('mlx-lm') != '0.32.2':
            raise ValueError('Recheck the callback contract before changing MLX-LM versions.')
        self.config = config or {}
        if self.config.get('report_to') != 'local' or self.config.get('test'):
            raise ValueError('Use local reporting and leave held-out test evaluation disabled.')
        self.started = time.monotonic()
        directory = Path(log_dir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / 'metrics.jsonl'
        self.append('configuration', {'config': self.config, 'mlx_lm_version': '0.32.2',
                                      'reported_token_scope': 'supervised target tokens; not full context throughput'})

    def append(self, event, metrics):
        record = {'event': event, 'at': datetime.now(timezone.utc).isoformat(),
                  'wall_seconds': time.monotonic() - self.started, **metrics}
        for key in ('train_loss', 'val_loss', 'learning_rate', 'peak_memory'):
            if key in record and not math.isfinite(float(record[key])):
                raise RuntimeError('Nonfinite ' + key + '; stopping this training process.')
        record.update(mlx_active_memory_gib=mx.get_active_memory() / 2**30,
                      mlx_peak_memory_gib=mx.get_peak_memory() / 2**30,
                      mlx_cache_memory_gib=mx.get_cache_memory() / 2**30)
        if 'iteration' in record:
            record['microbatches_completed'] = int(record['iteration'])
            record['optimizer_updates_completed'] = int(record['iteration']) // int(self.config['grad_accumulation_steps'])
        with self.path.open('a') as handle:
            handle.write(json.dumps(record, allow_nan=False) + '\n')
            handle.flush()

    def on_train_loss_report(self, metrics):
        self.append('train', metrics)

    def on_val_loss_report(self, metrics):
        self.append('validation', metrics)


if __name__ == '__main__':
    callbacks.SUPPORT_CALLBACK['local'] = LocalMetrics
    lora.main()

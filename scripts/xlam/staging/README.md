# Offline next-run staging

These scripts prepare a controlled Kumi/xLAM mixture and a Kumi-only baseline.
They do not download anything, load base weights, start training, alter an
installed package, or write the running v1 job's configuration/status/data.
Only `run_strict.py --start` starts a new run; preparation never invokes it.

Run `prepare_mixture.py` using the Studio's existing Python environment:

```sh
.venv/bin/python /path/to/staging-tools/prepare_mixture.py \
  --kumi-data /path/to/input/kumi \
  --xlam-data /path/to/official-subset-128-32 \
  --xlam-allowlist /path/to/semantic-review/allowlist.json \
  --model /Users/studio/Projects/kumi-training/models/Qwen3-Coder-30B-A3B-Instruct-4bit \
  --training-root /Users/studio/Projects/kumi-training \
  --output /Users/studio/Projects/kumi-training/staging/kumi-xlam-next-20261105-01
```

The output must be new and inside the training workspace. The Kumi input
directory contains only `train.jsonl`, `valid.jsonl` and `source-manifest.json`.
The source manifest binds the exact 341 training and 86 validation records and
every concatenated source part. Test files and protected evaluation prompts are
never read or copied. xLAM input uses its converted `train.jsonl`, `valid.jsonl`
and `manifest.json`; any source-pool test placeholder is ignored.

Every Kumi and xLAM row is checked with the existing Qwen tokenizer, its exact
template and installed MLX `ChatDataset`. The check verifies flat token IDs,
exact prompt prefix, native final-assistant mask offset, positive target length,
typed XML tool arguments and the full 32,768-token limit without truncation.
The loss diagnostic uses a tiny gradient probe, not the base model.

Each xLAM target also goes through the **actual installed**
`mlx_lm.tool_parsers.qwen3_coder.parse_tool_call` function. Its parsed function
name and typed arguments must exactly equal the source values. Incompatible
rows are excluded from the mixture and auxiliary validation; parser exceptions
and changed values are reported by source ID. The parser is never patched.
Its file hash is pinned in the stage and checked again by the strict runner.

Selection requires all three gates: fully specified active schemas, inclusion in
the hash-bound independent semantic-review allowlist, and installed-parser
compatibility. Whole eligible rows are sampled deterministically for call-style,
argument-type and API-family diversity. The selected auxiliary rows must satisfy
both `n_xlam <= floor(341 / 9)` and
`xlam_target_tokens <= floor(kumi_target_tokens / 9)`. These bound the
**materialized data's** row share and one-pass supervised-token share to 10%.
They are not a bound on actual gradient contribution. Fixed-duration training
may replay rows; the manifest records that limitation and the configured budget.

The primary validation file remains Kumi-only (86 records). The full converted
128/32 xLAM reservoir is preserved outside trainer data directories, together
with source provenance and the semantic allowlist. Only allowed and runtime-
compatible auxiliary validation rows appear under `auxiliary/xlam-validation`.
No locally executed API results, Kumi execution or musical judgments are claimed
for xLAM. Original row metadata and source readiness flags remain preserved;
the new stage's conditional eligibility requires its explicit strict runner.

Each variant's MLX data directory contains exactly `train.jsonl` and
`valid.jsonl`. Baseline and mixed configs use the same pinned base, q/v LoRA
rank 16, batch 1, accumulation 8, learning rate 2e-5, seed 7, prompt masking,
gradient checkpointing and 384 microbatches / 48 complete updates. Full Kumi
validation runs at the beginning and end; checkpoints are saved every 64 steps.
The output contains unique adapter/status locations; v1 is not resumed.

After preparation, validate a variant using its sealed staged copy:

```sh
.venv/bin/python /absolute/stage/tooling/run_strict.py \
  --stage /absolute/stage --variant mixed
```

This checks all data/config/tooling/tokenizer/parser hashes, probes the strict
loss and binds the explicit process-local `lora.train(..., loss=...)` hook, then
exits without loading weights. The installed native default loss is reported
separately; the strict hook excludes all history and padding. MLX forwards that
same explicit hook to validation. Nothing is installed or monkey-patched on
disk. Starting a future variant requires the additional `--start`; that path
refuses an active v1 process, concurrent stage runs and accidental resumes.

Local logic regressions:

```sh
python3 -m unittest discover -s scripts/xlam/staging -p 'test_*.py'
```

The local logic tests do not claim that the Studio tokenizer or parser checks
passed. Their actual results are produced only when preparation runs there.

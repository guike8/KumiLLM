# xLAM next-run preparation

Official Salesforce xLAM access was granted to the project account. The source
is pinned to revision `26d14ebfe18b1f7b524bd39b404b50af5dc97866`, with verified
SHA-256 `4ef5c6f0dc552f2231f93f5853a9ef431e9e806d7aa514d0f6b615606ce576c6`.
The source dataset is CC BY 4.0; retain the Salesforce/APIGen attribution,
[dataset link](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k),
[license link](https://creativecommons.org/licenses/by/4.0/) and
[APIGen citation](https://arxiv.org/abs/2406.18518) with any converted data.

The private Studio stage contains **341 Kumi + 29 xLAM training records**,
86 Kumi primary validation records and 16 separate xLAM validation records.
xLAM supplies 7.84% of materialized rows and 7.96% of supervised target tokens.
These are sampling proportions, not guarantees of gradient contribution. A
Kumi-only baseline uses the same training configuration and update budget.

Both variants passed exact tokenizer/mask checks and validation-only runner
preflight. Training has not started. The existing v1 job is unchanged.
See the [summary](../../provenance/xlam-next-run/summary.json) and
[actual Studio report](../../provenance/xlam-next-run/studio-stage.json).
Raw/converted xLAM rows and the materialized mixtures remain in the private
Studio stage and local review copy. This public directory supplies code,
source IDs, hashes and review evidence.

## Obtain and convert the source

Access must first be granted through the official Hugging Face form. The
downloader uses an existing `HF_TOKEN` or saved Hugging Face credential;
credentials and signed artifact URLs are never stored in its receipt.
Its private output directory is ignored by Git. It refuses existing outputs.

```sh
python3 scripts/xlam/download_official.py
python3 scripts/xlam/prepare_xlam.py \
  --input scripts/xlam/private/xlam_function_calling_60k.json \
  --source-sha256 4ef5c6f0dc552f2231f93f5853a9ef431e9e806d7aa514d0f6b615606ce576c6 \
  --output /path/to/fresh-xlam-reservoir --train-size 128 --valid-size 32
```

The converter keeps source queries and answer arguments unchanged. It emits
one assistant target containing the answer calls, without invented tool results
or reasoning targets. Offered API families are assigned to splits before
sampling; cross-split rows and duplicate normalized queries are removed.

Actual source schemas omit every `required` field and contain unreliable
defaults. The documented conservative projection requires each parameter
unless its type explicitly says optional; defaults do not authorize omission.
Raw defaults are retained in metadata and removed from the active schema.
This projection is not a reconstruction of executable backend signatures.
Unsupported types are rejected; untyped containers are flagged and excluded
from the final mixture.

The 128/32 reservoir is not the final subset. Source-quality filtering, manual
semantic consistency review and the installed Qwen parser select the final
29/16 rows. Six reviewed labels were withheld for changed inputs, cancellation,
ambiguity or incomplete requests. Four further allowed training rows failed
typed inference-parser roundtrip. Source APIs were not re-executed. The small
subset is math/algorithm-heavy and does not represent all upstream domains.

## Prepare the mixture

Build the Kumi aggregate from the train/validation parts listed in
`provenance/xlam-next-run/kumi-source-manifest.json`, checking each part's size
and hash and preserving its bytes. Keep that manifest beside `train.jsonl` and
`valid.jsonl`. Do not place test files in the input or MLX data directories.

Use [staging/README.md](staging/README.md) for the preparation command with
the existing local tokenizer and the hash-bound
[semantic allowlist](../../provenance/xlam-next-run/semantic-allowlist.json).
Preparation checks all 587 source rows, then applies installed-parser and
semantic gates. Final sampling caps both materialized xLAM row share and
supervised-token share at 10%.

The supplied runner explicitly passes the strict final-target loss to training
and validation; history and padding have zero gradient in the checked probe.
It validates only by default and loads no base weights. Only an explicit
`--start` begins training; that path refuses an active v1 job. Both prepared
variants start from the pinned base and use 384 microbatches / 48 updates.

```sh
python3 -m unittest discover -s scripts/xlam -p 'test_*.py'
python3 -m unittest discover -s scripts/xlam/staging -p 'test_*.py'
```

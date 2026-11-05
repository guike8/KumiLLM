# Native Kumi audio observation candidates

This batch contains **7 candidate prefixes from 3 authored goals and 4 authored
user turns**: three verified `listen` calls and four reviewed factual responses.
The prompts and F03 followup are authored collection scenarios, not feedback
from a real producer. Real piano recordings and actual native Kumi file-DSP
execution supply the observations. GPT-6.1 Sol with Ultra reasoning supplied
public actions through a Codex agent file queue.

There are no DAW edits, before/after DAW renders, human preference labels or
musical quality scores here. The student interface is text: structured audio
measurements accompany the conversation. The model did not directly hear audio.
All three CC0 recordings belong to Open Goldberg Variations / Kimiko Ishizaka
and remain in one **train-only source group**. No validation or test split is
claimed for this small batch.

**Current training readiness is false.** All seven records pass the exact local
Qwen tokenizer check: 8,330–9,846 full tokens and 80–313 target tokens, below
32,768. Prompt prefixes, typed tool arguments and native dataset mask offsets
match. The installed MLX loss probe also exposed supervision of the first
padding token. A separately supplied strict final-target loss passes the probe;
it must be selected explicitly in a future trainer. It has not been installed
in or applied to the running v1 job.

## Evidence and admission

- `train.jsonl`: exact normalized native request prefixes ending at one target.
- `manifest.json`: source grouping, per-target schema/context hashes, closure
  evidence, reviewed claim receipts and three excluded outputs.
- `studio-tokenization.json`: matching dataset hash and all seven exact
  tokenizer/mask results; current-trainer readiness remains false.
- `reviewed-public-responses.json`: hash-pinned reviews and observed field
  references for every admitted public response.
- `independent-audio-checks.json`: independent FFmpeg R128 and PCM checks.
- `implementation-report.json`: collection-side implementation hashes and
  checks; `packaging.json` records public tooling and archive hashes.
- `raw-evidence.tar.gz`: exact requests, teacher responses/receipts, task files,
  native events, conversation exports and capture status. No audio, model
  weights, database files, article bodies or SSH helper is included.
- `tooling/`: admission, tests, offline tokenizer diagnostic, strict loss and
  independent audio checker. Kumi's embedded instructions/schemas retain the
  [upstream MIT notice](../../licenses/Kumi-MIT.txt).

Each positive action has a matching successful subsequent native tool result
and `isError:false` event. Public text must occur in an actual native text event
and match a reviewed claim receipt. Its facts must be present in that request's
observed tool results, backed by successful native events. Failed, unsupported
or unreviewed results cannot supply positive labels.

F02 triggered two automatic Kumi audition reminders despite being a read-only
file comparison. Three resulting outputs are excluded. Those runtime-injected
messages remain in raw evidence and are not described as producer feedback.
For compatibility, metadata fields `human_turn_index` and `human_turn_count`
count authored user-role turns; they do not identify real producer judgments.
Episode IDs are grounded in the three saved task files; source call IDs identify
the seven different model requests separately.

The queue supplied no provider token/cost accounting. Native Kumi events contain
default usage zeros; those zeros mean unreported usage, not measured free
inference, zero tokens or a cost estimate. No private chain-of-thought is stored.

## Reproduce checks

Run these commands from this directory. Fetch the three reference audio seeds
using the sibling licensed asset manifest, then extract the raw evidence:

```sh
python3 ../production-source-seeds/fetch_assets.py \
  --asset open-goldberg-01 --asset open-goldberg-02 --asset open-goldberg-03 \
  --output ../production-source-seeds/audio-cache --derive
mkdir -p raw
tar -xzf raw-evidence.tar.gz -C raw
python3 -m unittest discover -s tooling -p 'test_admission.py'
python3 tooling/admit_native_audio.py \
  --teacher raw/teacher \
  --assets ../production-source-seeds/audio-cache/assets \
  --reviews reviewed-public-responses.json --evidence raw/native \
  --output reproduced-candidates
python3 ../../scripts/validate_dataset.py --train train.jsonl
```

The source downloader needs installed FFmpeg for WAV derivation. The reference
decoder was FFmpeg 9.0.2. Admission pins whole-WAV hashes; a different container
hash still requires review even when decoded PCM matches. Do not weaken this
gate to make a different decoder's result pass. Tests also accept explicit
`KUMI_NATIVE_TEACHER`, `KUMI_NATIVE_ASSETS` and `KUMI_NATIVE_EVIDENCE` paths.

The seven rebuilt records must be byte-identical to `train.jsonl`. Rebuilt
manifest evidence paths can differ with the extraction location; the published
manifest also adds global prompt provenance. Preserve the original raw archive.

For the exact Studio tokenizer check, use an existing local model/tokenizer and
MLX environment; no weights are loaded or downloaded:

```sh
python3 tooling/verify_tokenization.py --model /path/to/local/Qwen-tokenizer \
  --data . --output fresh-tokenizer-check
python3 tooling/check_audio.py \
  --assets ../production-source-seeds/audio-cache/assets --output fresh-audio-check.json
```

Nine admission tests and the seven-record structural check pass. The strict
loss hook is provided for a future training invocation, never as an automatic
patch. The tokenizer helper is an unchanged shared diagnostic module; run
`verify_tokenization.py` for this batch.

## Measurement limits

Independent FFmpeg measurements agree with Kumi's integrated loudness / true
peak to displayed precision for the three files: −25.7/−10.1, −17.0/−2.8 and
−17.9/−2.4 LUFS/dBTP. All contain nonzero PCM and no full-scale samples.

Other meter definitions differ: FFmpeg LRA is 8.4/4.6/7.0 LU while Kumi reports
7.9/4.5/6.7 LU. Global stereo sample-peak/RMS crest is approximately
18.59/16.77/18.07 dB while Kumi reports 20.4/18.7/19.8 dB. The factual responses
are checked against Kumi-reported definitions, not certified interchangeability
between meters. Estimated tempo/key fields are not musical ground truth.

Credit: J.S. Bach, Goldberg Variations BWV 988; Kimiko Ishizaka, piano;
produced by Anne-Marie Sylvestre, Teldex Studio, Berlin, 2012. Open Goldberg
recording, CC0 1.0. Recording provenance and exact download hashes are in the
[source seed bundle](../production-source-seeds/README.md).

# KumiLLM

Synthetic tool-use demonstrations and collection tools for training a local language model to control [Kumi](https://github.com/user1303836/kumi).

The first release contains **96 executed synthetic episodes and 432 assistant-target examples**. A scripted teacher calls Kumi's real kernel and tool engine against a corrected synthetic Ableton bridge. Admission requires successful tool results, requested final state, preservation of unrelated state, and fresh readback.

This is a bootstrap corpus. It does not contain real Ableton sessions, audio evaluations, trained model weights, or a claim of production reliability.

## Sol Ultra pilot (v2)

The [first Sol teacher batch](data/v2-sol-ultra-pilot/README.md) adds **4 admitted episodes and 20 train-only assistant targets**. GPT-6.1 Sol at Ultra supplied public actions through a Codex file queue; Kumi executed them on the Studio's synthetic bridge. All records passed strict admission and exact tokenizer checks. Two original traces remain quarantined after exposing verifier defects; corrected fresh attempts provide the admitted transport and preference examples.

The existing v1 LoRA pilot continues on its original snapshot and has saved a real adapter checkpoint. It has not consumed v2 data. No trained-model competitive result is established. See the [training snapshot](provenance/v1-training-pilot.json), [distillation workflow](docs/sol-distillation.md), [collector protocol](docs/sol-collector-protocol.md), and [verification evidence](provenance/sol-verification.json).

## Dataset v1

| Split | Episodes | Assistant-target examples |
| --- | ---: | ---: |
| Train | 64 | 288 |
| Validation | 16 | 72 |
| Test | 16 | 72 |
| Total | 96 | 432 |

The four task templates cover:

- A numeric dB edit and a track rename.
- Two track renames and a tempo change.
- Two new MIDI layers, followed by a grounded gain edit and readback.
- A gain edit and a correctly scoped note queued for an unsaved Set.

All 432 records passed structural validation and the exact Qwen tokenizer checks. Full examples contain **22,335–24,957 tokens**, under the checked 32,768-token limit; supervised targets contain 20–116 tokens. The original 18 benchmark cases were not imported into this corpus.

Read [DATASET_CARD.md](DATASET_CARD.md) for construction, verification, split rules, and limitations.

## Repository layout

```text
data/v1/                 Normalized train, validation, and test JSONL; manifest
data/v1/raw-evidence.tar.gz  All 96 raw episodes with full requests/results/state
data/v1/raw/             Evidence after local extraction (not tracked)
scripts/                 Admission, structure-validation, and tokenizer tools
patches/                 Kumi collector/simulator and MLX parser patches
provenance/              Public build and verification records
docs/                    Additional collection notes
```

The raw evidence archive contains all 96 episodes and a Kumi license notice. Extract it for inspection or admission reproduction:

```sh
tar -xzf data/v1/raw-evidence.tar.gz -C data/v1
```

Each normalized record has `messages`, `tools`, and group metadata. It ends at one supervised assistant action. Tool arguments remain JSON objects. Historical calls have matching tool results; a final supervised tool call intentionally has no result in that record.

## Validate a copy

```sh
python3 scripts/validate_dataset.py \
  --train data/v1/train.jsonl \
  --valid data/v1/valid.jsonl \
  --test data/v1/test.jsonl

python3 scripts/check_tokenization.py \
  --model /path/to/local/Qwen-tokenizer \
  --data data/v1 \
  --max-seq-length 32768
```

Tokenization uses existing local tokenizer files and disables downloads and remote code. The recorded tokenizer is from `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`, revision `6e302ea604ad9ab206367e2c501d1571023e7b6d`.

## Collect another synthetic batch

The [Kumi patch](patches/kumi-verified-collection.patch) targets commit `2ff07a5be515e0a267ea86f7c3d56fa579b607ea` (1.24.68). Apply it to that checkout and run from the checkout root:

```sh
git apply /path/to/KumiLLM/patches/kumi-verified-collection.patch
mkdir -p generated
KUMI_COLLECT_DIR="$PWD/generated/collection-v1" \
EVAL_TIMEOUT_MS=30000 \
cargo run --locked --profile ci-release -p kumi --example eval_changes
```

The destination must be new. This bounded mode uses a scripted teacher and a synthetic bridge; it needs no model server, DAW, audio files, or sign-in. It produces raw evidence pending admission. From this repository, normalize a batch with:

```sh
python3 scripts/normalize_episodes.py \
  --raw-dir /path/to/collection-v1/raw \
  --output-dir generated/episodes \
  --validator scripts/validate_dataset.py
```

The admission tool preserves raw inputs, quarantines entire episodes with failures or incomplete evidence, and refuses to overwrite a nonempty destination by default. Validate and tokenize the result before training. Collection and validation commands do not train a model.

## Fixes included

- **MLX-LM 0.32.2 Qwen parser:** preserves numeric scalar arguments when an `anyOf`/`oneOf` union unambiguously admits them. String sentinels such as `-inf` remain strings. This does not repair model output that omits native tool-call markers.
- **Kumi simulator:** supports the production mixer-display read protocol, including label resolution and dB edit/readback. Its display curve remains synthetic. The simulator never executes incoming Python.

Recorded verification includes 19 parser regressions, 3 simulator regressions, 14 admission regressions, and the complete dataset structure/tokenizer checks.

## Status and limits

No completed task-specific model or model weights are released here. A v1 LoRA pilot is in progress; its dated checkpoint evidence is linked above. An earlier base-model observation passed **5 of 18 existing synthetic predicates** with the 4-bit Qwen checkpoint and greedy decoding. That segmented baseline included a timeout correction and a prompt-cache reset; it is not a post-fix score, a trained-model result, or a production success rate.

The v1 corpus uses four scripted templates and one project layout. Seed groups vary names and values, so its validation/test splits do not establish unseen-template or unseen-layout generalization. Memory examples verify queued notes for an unsaved Set, not persistence after saving. Real Ableton execution and audio-quality verification remain future collection stages.

## Attribution

Full prompts and tool definitions contain upstream Kumi material. Patches and copied components also retain their upstream notices. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the license files included with this repository. Model weights, audio, and scraped research papers are not bundled.

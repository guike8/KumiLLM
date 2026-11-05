# Kumi demonstrations from source research

This expansion contains **12 admitted episodes and 40 verified assistant targets**: 26 training records and 14 validation records. Thirteen episodes were attempted; one original trace is preserved in quarantine and contributes no positive targets.

The installed Evidence AI Scraper collected twelve official [Live Object Model references](https://docs.cycling74.com/apiref/lom/) on the Studio, preserving source hashes, snapshots and field citations. It used direct retrieval with zero inference calls. The [exact source catalog](../../provenance/research-source-manifest.json) is public; article bodies and extraction snapshots remain private reference material and are excluded from prompts, targets and this repository.

Two Codex agents configured as GPT-6.1 Sol with Ultra reasoning supplied public Kumi actions through the file queue. Kumi's real kernel executed them against synthetic project state on the Mac Studio. This is not direct provider API sampling, and the normalizer cannot independently authenticate the declared model. No private reasoning or unavailable provider token usage was collected.

## Coverage

| Family | Train episodes | Validation episodes |
| --- | ---: | ---: |
| Displayed gain and pan | 2 | 0 |
| Isolated native send change | 1 | 1 |
| Append MIDI/audio tracks, then edit only the new audio track | 1 | 1 |
| Exact MIDI pattern and note readback | 2 | 0 |
| Existing device: continuous Transpose and discrete Algorithm | 1 | 1 |
| Read-only track/return/clip/transport inventory | 1 | 1 |

The three layouts have three, five and six regular tracks, different MIDI/audio ordering, different return counts and different existing clip content. Layouts 0 and 1 supply training episodes; layout 2 is reserved entirely for validation. Related layout families and every prefix from an episode stay together. The test file is intentionally empty. This validation set covers four families and does not establish generalization across the whole task space.

## Admission and targets

Admission requires successful native tool-result closure, exact requested state, actual fresh readback, no playback/recording, and preservation of unrelated state. A guard latches forbidden changes after every bridge call, including changes later restored; fast parameter batches validate all targets before mutating any item. Inventory completion uses a publicly requested JSON format with exact track/kind/order and clip/slot associations.

Only assistant targets containing tool calls are supervised in this expansion. They contain no accompanying public prose. Entire episodes validate before this filtering, so a failed action cannot disappear through target selection. Text-only final reports remain in raw evidence and historical prompt context; prompt loss masking is required and those reports are not positive targets.

Original episode S04R09 reached the requested parameter values, but an exploratory Python read returned an unsupported error. Strict admission quarantined the entire trace. A fresh simulator execution, S05R09, reached the same kind of goal through successful native calls under the unchanged collector and checks. The original trace was not relabeled. All thirteen raw files and all three collection manifests are retained.

Typed parameter discovery exposed current displays rather than a full enum-label catalog. The teacher used explicit user-requested labels through native fitting and checked the resulting values. These examples do not claim complete enumeration support. See [verification and known limits](../../provenance/research-verification.json).

## Reproduce

```sh
tar -xzf data/v3-research-guided/raw-evidence.tar.gz -C data/v3-research-guided
python3 scripts/normalize_sol_episodes.py \
  --raw-dir data/v3-research-guided/raw \
  --output-dir generated/research-reproduced \
  --corpus-version v3-research-guided \
  --reference-manifest data/v3-research-guided/reference_manifest.json \
  --tool-actions-only
python3 scripts/validate_dataset.py \
  --train generated/research-reproduced/train.jsonl \
  --valid generated/research-reproduced/valid.jsonl
```

The JSONL bytes reproduce the verified Studio output. The public dataset manifest changes only its private source-directory label and records that export change. All raw episode bytes and the reference-manifest bytes remain identical.

All 40 populated records passed the exact pinned Qwen tokenizer checks: full sequences contain 22,316–31,326 tokens; supervised targets contain 35–248 tokens. Every prompt is an exact token prefix, targets are positive and no example exceeds the checked 32,768-token context. The [tokenizer report](../../provenance/research-tokenization.json) explicitly records the empty test split.

The running v1 LoRA snapshot is unchanged and has not consumed this data. No real Ableton execution, rendered-audio result or student-versus-Sol score is established. The separate frozen 40-task evaluation remains a prospective specification requiring an evaluator implementation. Kumi instructions and schemas retain their MIT notice in LICENSE.txt; model weights, tokenizer assets and compiled binaries are excluded.

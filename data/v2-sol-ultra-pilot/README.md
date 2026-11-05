# Sol Ultra pilot

This first Sol teacher batch contains **4 admitted episodes and 20 assistant-target records**, all assigned to train. Six episodes were attempted; two original traces remain quarantined and contribute no training targets. Validation and test files are intentionally empty.

The configured teacher was **GPT-6.1 Sol with Ultra reasoning**, through a Codex agent file queue. Kumi executed its public tool calls on the Mac Studio against a synthetic Ableton bridge. Full native requests, public responses, actual tool observations and independent verification evidence were captured. This is not a direct provider API sampling log; the declared model provenance is not independently authenticated by the normalizer. Private reasoning and unavailable provider token usage were not collected.

| Admitted episode | Task | Records |
| --- | --- | ---: |
| S02E01 | Exact MIDI melody, note readback, existing clips preserved | 4 |
| S02E02 | Tempo then rename, undo only the rename, verify both turns | 7 |
| S03E03 | Discover transport, seek to bar 9 while stopped, fresh readback | 4 |
| S03E04 | Persist a producer preference before a later turn, add MARIMBA, read back | 5 |

Admission requires successful matching native tool results, the requested state, fresh readback, preservation of unrelated state and no transient playback or recording. The preference case reads the actual producer file before the second turn and again at completion; it does not test saved/reopened projects or a new application session.

## Original outcomes remain visible

The original collector rejected S02E03 because it recognized direct deferred calls but missed the valid native make_changes transport composition. It rejected S02E04 because its affirmative preference check missed the phrasing “names to be UPPERCASE.” Their actual state evidence remains unchanged, including the failed original checks.

The verifier was corrected with regression coverage. Two fresh simulator episodes, S03E03 and S03E04, then passed. The original rejected episodes were not relabeled or used as positive targets. These curation counts are not a teacher benchmark score.

The archived initial patch and corrected patch are both in the repository. Per-batch binary/source hashes, all six raw hashes and reproduction checks are in [sol-verification.json](../../provenance/sol-verification.json).

## Checks and use

All 20 records passed structural admission and the pinned Qwen tokenizer checks. Full examples contain **22,263–24,002 tokens** and final supervised targets contain **15–200 tokens**. The rendered prompt is an exact token prefix, each target is nonempty, and every example fits the checked 32,768-token context. See [the tokenizer report](../../provenance/sol-tokenization.json).

The JSONL files are byte-identical to the verified Studio output. The public manifest replaces only its private source-directory label and documents that export change. The evidence archive retains every raw episode byte, including quarantined cases, plus both collection manifests and progress logs. Extract it with:

```sh
tar -xzf data/v2-sol-ultra-pilot/raw-evidence.tar.gz -C data/v2-sol-ultra-pilot
python3 scripts/normalize_sol_episodes.py \
  --raw-dir data/v2-sol-ultra-pilot/raw \
  --output-dir generated/sol-pilot-reproduced
python3 scripts/validate_dataset.py --train generated/sol-pilot-reproduced/train.jsonl
```

The separate v1 LoRA pilot has not consumed this corpus. This train-only batch is a pipeline pilot, with four task families and one underlying project layout. It does not establish generalization, real Ableton operation, audio quality or competitiveness with Sol. The frozen 40-task evaluation is a prospective specification requiring an evaluator implementation; no teacher/student comparison has been run.

Kumi instructions and schemas retain the upstream MIT notice in LICENSE.txt. Model weights, tokenizer files and compiled binaries are not included.

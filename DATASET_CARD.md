# Kumi synthetic demonstrations v1

## Summary

| Field | Value |
| --- | --- |
| Collection date | 2026-11-05 |
| Scope | `executed_synthetic_kumi` |
| Teacher | `scripted_oracle_v1` |
| Verifier | `synthetic_ableton_final_state` |
| Accepted / quarantined episodes | 96 / 0 |
| Assistant-target records | 432 |
| Real Live verification | No |
| Audio-quality verification | No |
| Task-specific model training | Not performed |

The corpus is intended for initial experiments in native tool-call formatting, grounded references, compound edits, readback, and truthful completion reporting. It is not representative evidence of general musical competence or reliable DAW operation.

## Sources and construction

The collector uses Kumi 1.24.68 at commit `2ff07a5be515e0a267ea86f7c3d56fa579b607ea`, with the local collection/simulator patch supplied in this repository. It creates fresh synthetic projects and temporary memory stores, then executes a finite scripted plan through Kumi's actual kernel and tool engine.

The teacher observes references returned by `live_discover`; it does not assume references for newly added tracks. Full native requests, active schemas, assistant responses, tool results, and before/after state form the raw evidence. Each accepted model call becomes one normalized assistant-target record. Truncated console traces are not used as the source of training labels.

Four templates cover a dB edit plus rename, two renames plus tempo, two new MIDI layers plus gain, and a gain edit plus a queued Set note. Names and initial/target values vary across seed groups. Some fixtures include a hostile track label; that label is project data rather than an instruction to the teacher.

No benchmark episode, model weight, audio file, or scraped research paper was imported into the corpus. Full prompts and schemas do include upstream Kumi text and tool definitions; their attribution and license notices are part of the distribution.

## Acceptance criteria

An episode is admitted only when:

1. The requested final state matches the goal, with numeric comparison for tempo and display-unit comparison for the requested dB value.
2. Unrelated project state is preserved.
3. Applied changes have the expected count and families.
4. Scoped memory matches the request; unsaved-Set notes are explicitly pending.
5. Fresh readback occurs after the changes, with no transport action.
6. Every captured call is complete and each supervised tool call has a matching successful result in a subsequent captured prefix.
7. No rejected step, explicit error, unsupported content, malformed structured result, or unresolved call history is admitted.

Admission quarantines the whole episode when evidence is uncertain. Structure checks validate message/tool closure, object-valued arguments, exact duplicates, and known group identifiers. Raw hashes and split membership are recorded in the manifest.

## Splits

| Split | Seed groups | Episodes | Records |
| --- | --- | ---: | ---: |
| Train | 00–07 | 64 | 288 |
| Validation | 08–09 | 16 | 72 |
| Test | 10–11 | 16 | 72 |

All expanded prefixes from an episode and its original seed group remain together. There were no cross-split group collisions, exact duplicate records, or missing group identifiers in the recorded structure check.

These are seed-group holdouts within four shared templates and one layout. They do not measure generalization to new task templates, new project structures, real sessions, or other DAWs. The earlier 18-case benchmark was not used as training input, but capabilities such as mixer editing naturally overlap.

## Tokenization

All 432 records passed checks with the local tokenizer from `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`, revision `6e302ea604ad9ab206367e2c501d1571023e7b6d`. The recorded environment used Transformers 5.22.1.

- Full record length: 22,335–24,957 tokens.
- Supervised target length: 20–116 tokens.
- Checked sequence limit: 32,768 tokens.
- Prompt-prefix consistency and nonempty supervised target masks: passed.
- Tokenizer downloads and remote code: disabled.

The long system/tool context is intentional. Check complete examples with the exact tokenizer before changing a sequence limit or masking strategy.

## Known limitations

- The teacher is scripted; these records are not successful generations from the base model.
- The simulator's fader display law is synthetic, rather than measured Ableton `str_for_value` calibration.
- The first batch does not exercise every simulator operation. Successful writes to fields without faithful readback are not evidence of real DAW correctness.
- Set memory is queued while the fixture is unsaved. Persistence after saving or reopening a project was not evaluated.
- No real Live project, plug-in sound, rendered audio, human musical preference, or audible quality was evaluated.
- A raw state predicate and clean execution do not establish coverage of all natural-language intents.
- No trained checkpoint is released. The earlier 5/18 base-model predicate observation is not a quality score for this corpus or a post-fix evaluation.

## Reuse

Use this batch as a small bootstrap and transport/format experiment. Broaden task and layout coverage, add independently verified real DAW episodes, and reserve fresh evaluation tasks before drawing conclusions about a trained model.

Preserve the dataset's synthetic scope, teacher and verifier identifiers, provenance, and upstream notices when redistributing it. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the distributed license files for component terms.

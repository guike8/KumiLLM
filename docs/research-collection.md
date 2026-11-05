# Reference-grounded Kumi collection

This bounded mode creates twelve new synthetic Kumi episodes from curated primary-source facts and faithfully supported Kumi operations. The Evidence AI Scraper's documentation snapshots remain private reference evidence. Article bodies are never model prompts, response targets, or corpus records. The teacher receives only actual Kumi requests, public task intent and actual tool results through the existing Sol Ultra file queue.

Apply `patches/kumi-research-collection.patch` directly to upstream `2ff07a5be515e0a267ea86f7c3d56fa579b607ea`. It includes the previous collection changes; apply it instead of the other full Kumi collector patches. The original four-case pilot mode is unchanged unless `KUMI_SOL_RESEARCH_MODE=1` is set.

## Evidence

Set `KUMI_SOL_REFERENCE_MANIFEST` to an absolute JSON manifest path. The collector requires format version 1, a sources array, and exactly one successful source entry for each selected episode's referenced ID. Every referenced source needs an HTTPS URL and a 64-hex `source_hash`. Missing, pending, failed, duplicate or malformed referenced entries prevent collection. The manifest must be a regular file under 4 MB.

The current scraped source catalog is [released source manifest](../provenance/research-source-manifest.json). Its supplied SHA256 is `9a8f6ca34e23186d8b551c924a3ae6c9411dc10cae9c7b7280f8eed06c6428b0`. The collector computes the actual manifest-byte hash independently. Per-episode raw records retain `research_mode=true`, `reference_manifest_sha256`, `reference_source_ids`, and `reference_catalog_source_count`.

| Facts used | Source IDs |
| --- | --- |
| Song/track relationships and native mixer controls | `lom_song`, `lom_track`, `lom_mixerdevice` |
| Session slots, empty-slot creation and beat-relative MIDI notes | `lom_track`, `lom_clip`, `lom_clipslot` |
| Existing devices, continuous values, discrete parameter labels and observed ranges | `lom_track`, `lom_device`, `lom_deviceparameter` |

These references inform novel task compositions; they do not provide answer traces. No scripted teacher replies are generated.

## Selection and launch

Use new collection and queue directories for each run. The Studio command for the first six episodes is:

```sh
KUMI_SOL_RESEARCH_MODE=1 \
KUMI_SOL_COLLECT_DIR=/absolute/new/research-S04-first \
KUMI_SOL_QUEUE_DIR=/absolute/new/research-S04-first/queue \
KUMI_SOL_REFERENCE_MANIFEST=/absolute/path/to/research-source-manifest.json \
KUMI_SOL_EXECUTION_HOST=mac_studio \
KUMI_SOL_START=0 KUMI_SOL_LIMIT=6 KUMI_SOL_SERIES=S04 \
/absolute/path/eval_changes
```

Run the remaining six with a new output/queue directory and `KUMI_SOL_START=6 KUMI_SOL_LIMIT=6 KUMI_SOL_SERIES=S04`. Episode IDs stay unique because the task index is included. `START` accepts 0–11; positive `LIMIT` is a count bounded by the remaining tasks through index 11. Series values contain 1–32 ASCII letters or digits and default to `S04`. Default research selection is all twelve episodes. Without research mode, the original 0–3 selection and `S02` default remain intact.

A six-case job permits at most 60 teacher calls. The whole twelve-case job permits 120, with a hard implementation bound of 128. Response waits remain 300 seconds and each conversation turn has a 600,000 ms hard limit. This process loads no GPU language model on Studio; only the actual synthetic bridge and Kumi engine run there.

## Curriculum and separation

| Indices | Family | Independent evidence |
| --- | --- | --- |
| 0–1 | Gain plus pan in displayed units | Requested mixer labels, fresh actual mixer readback, exact preservation of sends and unrelated state |
| 2–3 | Isolated send in native units | Full original send list, exact targeted native send value, all other sends/fader/pan preserved, fresh send readback |
| 4–5 | Append MIDI and audio tracks, then target only the new audio track | First-turn exact appended state plus observed names/media kinds; second-turn displayed gain/pan and fresh readback; original content and new MIDI track unchanged |
| 6–7 | Exact novel MIDI patterns | Empty slot, exact pitch/start/duration/velocity/unmuted note multiset (including duplicate counts), actual `read_notes` readback, all existing clips preserved |
| 8–9 | Existing Operator Transpose and Algorithm | Observed ranges/current displays, explicit requested labels, native fitting and fresh parameter values, all other parameters preserved |
| 10–11 | Read-only project inventory | Actual track/return/slot/clip/song reads, exact structured inventory JSON with track/kind/order/clip-slot associations, no writes/history/state changes |

There are three different layouts: 3, 5 and 6 regular tracks; different MIDI/audio ordering; 1, 3 and 2 returns; and different occupied Session slots, clip lengths and pre-existing note content. The existing Operator is attached to an actual MIDI track in each layout, with consistent device and parameter references.

Layout family 2 is held out entirely for **validation**. Layout families 0 and 1 are **training**. The full set contains **8 train, 4 valid, 0 test** episodes. `project_family_id` follows layout family, and project IDs include that layout, so related layout variants never cross splits. A six-case half contains four train and two valid episodes. The frozen benchmark/test set is untouched; this collection is not a competition result.

The terminal manifest reports `raw_episode_count`, completed/predicate-passed counts, selected split counts, source-catalog count, validated source IDs, selection, provenance and limitations.

The public inventory request specifies exactly `tempo_bpm`, `transport` (`playing`, `position_beats`), ordered `tracks` (`index`, `name`, `kind`), ordered `returns` (`index`, `name`), and `midi_clips` (`track_index`, `track_name`, `scene_index`, `name`, `length_beats`). Indices are zero-based; positions/lengths are beats; kinds are `midi` or `audio`. The independent check parses the last actual assistant message as JSON and compares all associations and ordered arrays with actual state, rejecting extra/missing fields. Equivalent JSON numeric spellings are accepted.

## Execution and admission

All model calls use the existing strict public file-queue protocol. Teacher provenance remains `gpt-6.1-sol-ultra`, requested model `gpt-6.1-sol`, effort `ultra`, transport `codex_agent_file_queue`. This is a Codex subagent bridge, not direct provider API sampling. Complete actual requests and public streamed responses are saved; usage is unavailable and never invented.

Independent predicates inspect actual bridge results, exact before/after state, history, per-turn checkpoints and fresh readbacks. Every research episode forbids transient playback/recording and unrequested memories/artifacts. Write acknowledgements alone never establish success. Unsupported synthetic mutation paths are refused instead of receiving generic success acknowledgements. The supported research mutation paths are native mixer volume/pan/sends, append-only MIDI/audio tracks, empty-slot MIDI creation, and Kumi's marked fast parameter helpers. Arbitrary Python, track insertion elsewhere, scene creation and unsupported mutation families are refused.

After every actual MCP execution, a research-only invariant guard compares state with the initial fixture while allowing only the current task's declared targets. Separate gain/pan or parameter edits are allowed. Append operations must keep the requested new-track prefix and defaults during the first turn; only the new audio track's gain/pan may change in the follow-up turn. New clips must have the requested musical note multiset. Read-only inventory allows no state changes. The first forbidden intermediate change is latched, even if later restored, and `never_unrequested_mutation` must pass. Raw audit evidence records the bridge call index/name and up to 32 changed paths with expected/actual values; ordinary intermediate snapshots are not retained.

Fast-set batches are also checked **before any item executes**, using the parsed helper inputs and actual parameter references. Only the parameter family may fast-write, and only to the existing requested Operator's Transpose and Algorithm. Any unknown or unrelated target rejects the whole batch, including duplicate unrelated entries that would otherwise change and restore a value inside one MCP call. Other research families refuse all fast-set writes. The original pilot mode is unchanged.

The static project snapshot export/diff fixtures belong to the original benchmark layout and are explicitly refused in research mode. Research inventory uses actual native discovery rows instead. The original pilot's snapshot path remains unchanged.

All attempts, including failures, remain in raw storage. `collection.json` means the bounded collector finished; it still says `raw_collection_complete_pending_strict_admission`. Only independently normalized/admitted examples may enter training or validation. Failed original pilots and earlier predicate outcomes are never rewritten by this mode.

## Focused checks

From the patched Kumi checkout, use the isolated runner and select `sol_collection::research::tests::`, the existing `sol_collection::tests::`, and `simulator_tests::`. New regressions verify layout/reference consistency, genuine unsupported-operation refusal, wrong/extra MIDI and mixer readback rejection, unrelated-state preservation, manifest evidence validation, bounded selection, and the 8/4 layout split with no family leakage.

For this batch, admission uses `--corpus-version v3-research-guided --reference-manifest /path/to/research-source-manifest.json --tool-actions-only`. All native responses and results validate before target filtering. Only tool-call targets with empty public text are supervised. Other public reports remain raw evidence and masked prompt context. Article text is not used as a training target.

## Refresh references

The optional driver was tested with the installed Evidence AI Scraper 0.2.0 Python environment. It reads at most twelve discovered official URLs sequentially, saves snapshots privately, and explicitly constructs the pipeline without an inference adapter or private configuration. Use an existing parent directory and a new output path:

```sh
/path/to/ai-scraper/.venv/bin/python scripts/collect_references.py \
  --output /absolute/new/reference-batch
```

Use the exact released reference manifest when reproducing this corpus. A refreshed catalog has a new provenance hash and must not be substituted into existing episode claims.

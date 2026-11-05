# Sol Ultra distillation for Kumi

This workflow prepares a separate **v2 Sol teacher corpus** for training a smaller local Kumi specialist. The first batch contains four admitted episodes and 20 train-only assistant targets from six attempts. Two rejected original traces remain unchanged; corrected fresh attempts follow verifier fixes. See the dataset card for exact outcomes. No trained-student score or competitive result is reported. The existing v1 scripted corpus and its training snapshot remain separate.

The teacher is **GPT-6.1 Sol Ultra**, requested as model `gpt-6.1-sol` with reasoning effort `ultra`, through a **Codex agent file queue**. A Codex agent supplies public assistant actions; Kumi executes them and captures actual tool results against its synthetic bridge. These are model-authored demonstrations rather than scripted v1 actions. This transport is not a direct OpenAI API sampling log. Its declared model provenance is preserved, but the normalizer cannot independently authenticate a provider.

## Pilot scope

The four admitted task families cover:

- MIDI writing with exact note readback and preservation of existing clips.
- Selective undo that preserves a preceding requested edit.
- Deferred transport discovery and seeking without starting playback.
- A producer preference remembered and applied on a subsequent turn.

Goals and preservation checks are evaluated against the synthetic project state independently of the teacher's final answer. Kumi's real kernel and tool engine execute the actions; the bridge and project state are synthetic. A passing synthetic predicate does not establish real Ableton execution, audible quality or production reliability.

All four admitted pilot episodes belong to `train`. Empty `valid.jsonl` and `test.jsonl` files are intentional for this pilot. The admission manifest marks the missing splits and sets `generalization_evaluation_established` to `false`. This is not a complete evaluation corpus, and a successful teacher demonstration is not a measured teacher success rate.

## Raw contract and admission

Each raw episode uses `format_version: 1`, an episode/project/project-family identity, an assigned split, nonempty goals, before/after state evidence, independent checks, and full native requests and streamed responses. Its exact required provenance is:

```json
{
  "scope": "executed_synthetic_kumi",
  "teacher": "gpt-6.1-sol-ultra",
  "teacher_model": "gpt-6.1-sol",
  "teacher_reasoning_effort": "ultra",
  "teacher_transport": "codex_agent_file_queue"
}
```

The Sol normalizer uses this identity directly. It never relabels a teacher as scripted or upgrades v1 scripted actions to Sol demonstrations. It emits normalized JSONL with the exact provenance, source revision, episode groups and raw-source SHA-256 hashes. The output manifest uses `format_version: 2` and `corpus_version: "v2-sol-ultra-pilot"`; this does not change the raw format version.

Admission requires every verification check to pass, a successful final outcome, complete calls, valid active tool schemas and successful matching tool results for all supervised tool actions. A result must appear in a later captured request prefix before its action can become a positive target. A failed, refused, truncated, unsupported or incomplete action quarantines the entire episode. The manifest records the reason; the raw source remains unchanged. Duplicate episodes/examples and project-family overlap across splits are also quarantined. Source symlinks are rejected before reading.

Native messages, active tool schemas and argument types are retained in the normalized representation. Tool calls and tool results remain distinguishable; each record ends at one supervised assistant target. Raw inputs are not rewritten. Private reasoning is not collected or admitted, and unavailable provider token usage is never invented. The file queue supplies no token-accounting receipt, so usage stays explicitly unreported.

From the repository root, admit a separately collected pilot into a new destination:

```sh
python3 scripts/normalize_sol_episodes.py \
  --raw-dir /path/to/sol-pilot/raw \
  --output-dir generated/v2-sol-ultra-pilot
```

The validator defaults to the adjacent `scripts/validate_dataset.py`. `--validator` can select another compatible validator explicitly. A nonempty output directory is refused unless `--overwrite` is supplied; keep the v1 corpus and active training snapshot outside this destination.

Inspect the manifest's accepted/quarantined counts and reasons. `ok: true` means the admission run completed; it does not imply that every source episode was accepted. Run the existing structure and exact local-tokenizer checks before starting a new training job:

```sh
python3 scripts/validate_dataset.py \
  --train generated/v2-sol-ultra-pilot/train.jsonl \
  --valid generated/v2-sol-ultra-pilot/valid.jsonl \
  --test generated/v2-sol-ultra-pilot/test.jsonl

python3 scripts/check_tokenization.py \
  --model /path/to/local/student-tokenizer \
  --data generated/v2-sol-ultra-pilot \
  --max-seq-length 32768
```

For a train-only pilot, both full-directory commands above return a nonzero status with `valid: no examples found` and `test: no examples found`. Those expected missing-split errors must remain visible. Inspect every populated training-record check separately; passing those checks does not provide validation/test coverage. To run only structural checks on the populated split, use `python3 scripts/validate_dataset.py --train generated/v2-sol-ultra-pilot/train.jsonl`. Verify sequence lengths, target preservation and the actual training loss mask for the chosen trainer before using the data. Admission and validation do not train a model.

Regression checks for the Sol admission gate:

```sh
python3 -m unittest discover -s scripts -p test_normalize_sol_episodes.py -v
```

## Smaller-student comparison

The current provisional student is `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`: a 30B MoE with approximately 3B active parameters. Its existing v1 LoRA pilot uses its existing immutable data snapshot. The Sol v2 pilot does not silently replace that snapshot. A later 7–14B dense comparison is provisional; no model in that range has been selected as best. Compare each chosen student's unmodified base and trained adapter.

The prospective target is Kumi task performance on fresh projects and task families. Freeze independent held-out goals, checks, project layouts, splits and action/time budgets before generating training demonstrations. Keep original benchmark cases, close task copies and held-out teacher answers out of v2 training. Related project variants and all assistant prefixes from an episode stay in one split. Use validation for configuration selection and keep the final test sealed until the student is frozen.

Run the teacher and student on the same task, initial state, Kumi instructions, tool definitions, observations and public-action budgets. The teacher's outer Codex adapter and queue add differences from local inference; report those differences and end-to-end latency instead of presenting the transports as identical. Score every held-out attempt, including failures and timeouts. Never measure competitiveness only on successful teacher demonstrations.

The primary metric is requested-goal success without unrequested mutations. Report invalid-call and unrequested-change rates, grounding/readback, selective undo, memory scope/persistence, recovery where supported, and latency/resource use. Current clean-episode admission excludes failed-action recovery histories; training those needs a separately reviewed per-step policy. Report trial counts and uncertainty. Real Ableton and audio evaluations require separate evidence.

After a verified pilot and coverage/error analysis, the expansion plan targets 5,000 or more accepted demonstrations conditionally on quality and budget. This is a planned scale target, not a collected count or a guarantee of frontier-level performance. See [the machine-readable competition plan](../provenance/sol-competition-plan.json) for the proposed controls and gates. No claim of matching Sol outside Kumi follows from this work.

The prospective held-out specification is frozen as 40 tasks across five layouts and 14 families, SHA-256 `97fae5ba82bef03cc4a8cdaeceaba3f76542f01c9141c1919165270383e68013`. This is a specification only: it still requires an isolated evaluator initializer and predicate adapter, and has no execution results. Only [its freeze summary](../provenance/sol-heldout-freeze.json) is public; the protected task inputs and checks are excluded from training and this release.

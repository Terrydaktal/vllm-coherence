# Current 320-token confirmations

These checks bind the repaired release to finite numerical evidence. They do
not prove an arbitrary-input theorem or an independently correct target model.
The performance tables are separate, uninstrumented or activity-profile runs.

## Whole-model checks

Each arm starts from a fresh 60,000-token prefill and consumes the same 321
forced continuation tokens, producing 320 compared next-token predictions.
The input is a numeric slice recovered from the retained private Pi corpus;
it is not the deleted historical standalone 320-token fixture. No chat text is
decoded or published.

The four comparisons are compiled M1/M8, eager/compiled M1, eager/compiled M8,
and eager M1/compiled M8. On September 25, compiled M1 replays the original PIECEWISE graphs while compiled M8 replays the candidate FULL graph. Eager arms retain the existing RoPE rounding repair and original operators. Every comparison checks the
prefill prediction, full-vocabulary logit hashes, top-1/10/20 sets and ordering,
boundary ties and retained scores.

All four current comparisons pass 320/320, including full-logit hashes and the
prefill prediction. [Execution identities and results](../benchmarks/results/current-320-confirmations-20260925.json).

The comparison head is **full BF16** to expose target-body differences.
Production still uses **Global-512**; its candidate recall is qualified in the
separate [head study](HEAD_CANDIDATE_DEPTH.md), which does not establish complete
candidate recall for arbitrary inputs.

## Isolated stages and independent operators

The native tape checks 22 boundaries across 770 layer instances. Each layer
instance receives the same captured correct inputs in the compared versions.
Local outputs and mutable state are compared exactly. Equal results can reuse
the already validated vocabulary suffix; differences must execute that suffix
again. A position passes a stage only when every layer instance passes.

Fused normalization and FP8 production are checked jointly. Attention decode
and its split merge are also checked jointly. Separate attention-output and
MLP-input quantizers have explicit checks. The drafter and generic bookkeeping
are not silently counted as target-stage arithmetic.

Stage replay disables graphs to expose individual compiled calls. Its complete
320-token result must match the compiled graph control before stage evidence
is accepted. All 40 groups must pass output, hidden-input and early-projection
corruption controls, restore authoritative cache state, and have identical
source identities and complete layer inventories. This is diagnostic replay,
not a serving-time measurement.

The current run passes 320/320 at all 22 boundaries for both M1/M8 and
eager/compiled M8, including local outputs/state, top-1/10/20 sets/order and
full-logit hashes. All 40 corruption/restoration controls pass, and the complete
diagnostic result matches the compiled graph control.
[Per-stage and per-layer results](../benchmarks/results/current-stage-confirmations-20260925.json).

The independent operator rerun covers 2,817 checks, with zero failures. It uses
320 inputs at the audited sites, including 496 projection matrices, all target
normalizations, 48 convolution/recurrent layers, and a 32,768-step recurrence
check. Attention additionally exercises 88 structured-page cases. Projection
checks sample output channels; the earlier broader full-output and varied-page
checks remain separate evidence. FP64 error tolerances and exact encoding/state
checks are distinguished in the [operator results](../benchmarks/results/current-operator-confirmations-20260925.json).

## Repeating the checks

Use a disposable container with the pinned production image, release payload,
models and runtime mounts. Stop inference only after pending session tails have
been flushed, and restore production after the runs. Use a new output directory
and startup receipt for each arm. Do not reuse an installed-source binding from
another release.

Inside that container, `experiments/radiance-public/benchmark_current_confirmations.py`
provides `bootstrap --arm compiled-m1|compiled-m8|eager-m1|eager-m8` with
`--fixture`, `--spec`, `--output` and `--private`. Bootstrap authenticates the
release, starts a fresh interpreter and records actual installed source hashes.
The fixture contains 60,000 prefix tokens and 321 continuation tokens, sealed
with the diagnostic contract. The specification supplies the native model
configuration and source-binding inventory. Fixtures must be owned regular files
with mode 0600; private directories use mode 0700.

For the banked speed candidate, add `--speed-candidate` to every arm. The compiled M8 arm uses `SpeedMatchedStageWorker` and one FULL target graph; the stage arm uses `SpeedTapeWorker`, keeping the original M1/eager projection operators as controls. The worker validates the candidate binary and drafter-source qualification identities. These results bind the measured source and binary hashes in the [qualified speed refresh](SPEED_INVESTIGATION_20260925.md), including its measurement adapters and cache-preparation hook repair. Original capture checkout identities remain unchanged after the history rewrite; the experimental worker is not the normal Pi deployment.

For stage capture, use `--arm stages` and set `QWEN_D7_STAGE_MATRIX=1`,
`QWEN_D7_CURRENT_STAGES=1`, `QWEN_D7_STATE_STAGES=1`,
`QWEN_D7_ATTENTION_STAGES=1` and `QWEN_D7_TAPE_GROUPS=40`. Current-stage mode
rejects historical operator substitutions and unrecognized interfaces.

For the independent audit, use `probe_m1_stage_audit.py --rows 320
--transitions 32768` with the authenticated release and its explicit GPU/idle
controls. The driver must run after the release bootstrap in a fresh Python
process, so it imports the same installed operators as production.

The `report`, `stage-report` and `operator-report` actions of
`benchmark_current_confirmations.py` export aggregate JSON. They reject missing
positions, layer inventories, mode/source mismatches, failed negative controls
and failures concealed by a passing status. Run these CPU report actions with
the same installed diagnostic Python package/import paths as the worker. The
stage reporter also imports `analyze_native_d7_stages.py`: include the repository's
`experiments/radiance-public` directory in `PYTHONPATH` when copying only selected
worker files into a container. A missing reporter dependency can be repaired and
the CPU report rerun against the completed capture without repeating GPU work.
Raw numeric fixtures, per-token rankings and stage captures stay in the private
artifact directory.

Remaining scope includes independent full-model arithmetic verification,
prefill versus serial decode and arbitrary speculative rejection, cancellation and concurrent-session histories. A separate [finite native snapshot test](../benchmarks/results/snapshot-lifecycle-20260925.json) passes A/B/A handover, disk restore and three verified generation replacements; it does not exhaust those histories. Matching tests do not
prove that model-generated loops or other reasoning errors are eliminated.

# Backend conformance: executable contracts and recovery qualification

This extends [the conformance prototype](backend-conformance.md). Implementation
and regression tests run without a GPU. Native inference/extraction still needs
an authorized qualification run. Importing these modules installs no live hooks.
See [equivalence status](backend-equivalence-status.md) for the current precision
profiles and complete list of declared component obligations.
The [native campaign](backend-conformance-gpu-campaign.md) now wires the real
connector, scheduler, graph/head, corruption and parser/provider paths into an
executable finite matrix. Its CPU orchestration checks do not qualify GPU execution.

## Recovery comparison

Capture three states at the **same consumed prefix and pending token**: an
independent reference, the live candidate and the restored candidate. Required
components come from the model contract, never an intersection of available
observations. Compare actual KV, GDN, convolution, position, scale and pending
state bytes. Hashes identify artifacts; they do not replace byte comparison.

| Reference = live | Reference = restored | Live = restored | Classification |
|---|---|---|---|
| yes | yes | yes | All observed state matches |
| no | yes | no | Live differs; restored matches reference |
| yes | no | no | Restore introduced a difference |
| no | no | yes | The same difference survives restoration |
| no | no | no | Both differ; restoration changed the difference |

This distinguishes damaged live state, damage introduced by restore and an
already-bad snapshot. It does not identify a particular faulty kernel without
earlier boundary observations, or explain an incident whose live state was lost.

`FrameTransport` tests the **actual Radiance ChatStore** codec, locking,
publication and garbage collection with a diagnostic frame envelope. It keeps
immutable RAM copies, flushes before eviction, preserves the previous valid head
after failed replacement and restores after process restart. Its envelope is not
a new production snapshot format; native connector mapping remains unqualified.

`RecoveryCapture` archives create-once, quiescent capture points and rejects mixed
chats/generations, duplicate stages and incomplete coverage. Original frames are
retained. State comparison removes only the producer's `execution_mode` label;
other logical metadata remains part of equality.

## Executable mathematical obligations

The reference specifies checkpoint and quantizers, activation/KV formats,
recurrent precision, ordered FP32 arithmetic, BF16 rounding, native dense causal
attention and greedy semantics. Quest96 is not inherited from the old stack.

With physical allocation `P`, logical mapping `alpha`, reference `R` and candidate
`B`, the desired transition property is:

```text
alpha(P) = S
  => alpha(B(P, u).state) = R(S, u).state
     and B(P, u).observables = R(S, u).observables
```

Frames execute finite equality checks; they do not universally prove this
formula. Physical page IDs may differ. Logical values, positions, ownership
validity and state versions must agree.

| Obligation | Actual implementation and scope |
|---|---|
| `n' + pending' = n + pending + emitted` | `remaining_materialized`, also used by the durable checked session. Materialized tokens and emitted-but-unprocessed tokens are distinct. |
| D7: `0 <= k <= 7`, `emitted = k + 1` | `validate_speculative_commit`; every acceptance width is exercised. Every required component version must equal `n'`. |
| State slot `running + accepted_count - 1` | `temporal_column`; convolution starts at `accepted_count - 1`. The native processed count is 1–8, distinct from the 0–7 proposal count. |
| Rejected-suffix noninterference | Actual eight-slot `copy_accepted_prefix` checked at widths 0–8; reference rollback tests compare retained state and future logits. This does not prove native M8 execution. |
| Checked publication | `output_equal & state_equal & identity_equal & coverage & writes_completed & !cancelled & (base == current)`. Z3 calls the actual predicate used by `ControlLedger`. |
| Published-prefix induction step | Successful gate implies equality of symbolic next-state/output payloads. Reference/checker correctness, isolation and faithful delivery remain assumptions. Refusal gives safety, not completion. |
| Immutable sharing | `writable_exclusively`: one owner, no external pins and a mutable block. Ledger tests cover sharing, copy-on-write and release without altering another owner. |
| Snapshot publication | Verified, durable, current generation required. Failed replacement leaves the observed head intact; stale writers cannot publish it. Actual ChatStore filesystem tests check that implementation separately. |
| Snapshot roundtrip | `Load(Save(S)) = S`, followed by identical forced suffix state/logits. Compressed transport, RAM eviction, restart, corruption and compaction are tested; universal codec/native connector equivalence is not proved. |
| Greedy interval certificate | `lower[winner] > upper[every other token]`. All vocabulary entries required. Ties or an excluded possible winner refuse certification. Bound soundness is ASSUMED. |
| Margin theorem | Given `abs(error_i) <= epsilon`, `margin > 2*epsilon` preserves argmax. Z3 checks rational/real inequalities; it does not derive bounds for GPU kernels. |
| Rejection distribution | Exact `Fraction` reference for `min(p_i,q_i) + max(p_i-q_i,0) = p_i`, including zero draft probabilities and zero residual mass. Native sampler/RNG behavior is still unqualified. |
| BF16 rounding | The actual `bf16_round_bits` expression is compared symbolically with an independent retained/discarded-bit RNE specification for finite/Inf FP32 input patterns. NaN handling, NumPy and native GPU casts are separate obligations. |

`prove` runs 55 scoped cases, including expected counterexamples. Each saves its
query, domain query, implementation identities and result; satisfiable queries
retain witnesses. Preconditions must themselves be satisfiable. Contradictory
assumptions, timeouts and unknown results are **UNPROVED**, never a vacuous proof.
Helper-level PROVED results are distinct from TESTED components,
RUNTIME-CHECKED invocations and ASSUMED boundaries.

## Instrumentation

The independent reference emits detailed per-layer projections, normalization,
convolution input/history/output, GDN input/state/output/gating, Q/K normalization
and RoPE, stored KV, attention, residual and MLP boundaries. The common decoder
schedule remains available for cross-backend comparison. Missing or duplicate
required observations cannot complete a schedule.
The default domain now includes early prefill chunks and intermediate accepted
D7 rows. Rejected suffixes and emitted-but-pending tokens are excluded. Explicit
position subsets are recorded as subsets, never promoted to complete coverage.

Native replay also wraps actual target modules and loaded Radiance GDN/MXFP4
glue. `CallRecorder` preserves inputs before mutation, mutated arguments and
outputs afterward, logical positions, parent calls and observation order. Both
source and dispatched Python bytecode are bound; a wrapper cannot claim only its
wrapped function's identity. Hooks are restored after success, validation failure
and partial installation; cleanup does not overwrite a later independent hook.

| Mode | Evidence and limitation |
|---|---|
| `tensor` | Exact tensors and immutable input copies. Native export synchronizes and can be expensive; its timings are unsuitable for performance comparisons. |
| `metadata` | Call order, shapes, types and supplied scalars. Does not export tensors, read device values or synchronize the GPU. CPU recording can still perturb timing; it cannot pass numerical comparison. |

`calls` locates the earliest differing **observed** boundary, including nested
callees before their parent returns. It verifies original call receipts, tensor
identities and complete order. Different call structures need an explicit
logical adapter. Unexported argument objects cause exact comparison to refuse;
guessing their contents would create false assurance. A recorded Python function
does not establish which HSACO instructions executed.

`DispatchRecorder` optionally binds a reviewed native library export inventory
and explicit direct/list/tuple/dictionary aliases. It verifies library identity,
records entry and return without dereferencing device pointers, and preserves
failed-call receipts. Missing required dispatches or changed exports fail closed.
It needs `native_entrypoints` in the reviewed binding; absence remains a reported
gap. It does not observe hidden C++ calls, captured closures or graph replay and
cannot assert device completion merely because the host function returned.

`capture_committed_state` is an armed worker RPC using the same V2 state extractor
for a resident sequence without forcing tokens or disabling its connector. The
caller must hold scheduler admission and wait for restore/handover operations.
It checks expected prefix, consumed count, pending token, checkpoint path and
source binding. Synchronization alone does not prove quiescence. The RPC is
implemented but not GPU-qualified.

`ControlRecorder` supplies a private chained event spool. `control` audits
completion, validation, publication, cancellation, generation, ownership,
snapshot and restore sequences. Truncated traces, unknown events, stale commits
and missing versions fail. Producer claims about GPU fences and tensor equality
need independent qualification; their presence in JSON is not proof of device
behavior. This observer is not automatically installed into production.

## Behavior inventory and negative controls

Stable IDs identify release obligations. Tests below are executable CPU tests.

| ID | Behavior | Tests and credible failures |
|---|---|---|
| CF-01 | Quantized reference | `test_conformance_reference.py`, `test_conformance_replay.py`: operators, prefill/continuation, changed artifacts and reference identity. |
| CF-02 | Canonical equality | `test_conformance_state.py`: changed bytes, nonfinite values, missing components, wrong positions, payload identity and bounded streaming. |
| CF-03 | Speculative commit | `test_conformance_replay.py`, `test_conformance_control.py`, `test_conformance_radiance.py`: all D7 widths, rejected suffixes, wrong GDN/conv slots, physical page normalization. |
| CF-04 | Recovery | `test_conformance_lifecycle.py`, `test_radiance_chat_cache.py`: RAM/disk, eviction, process restart, damaged compression/KV/GDN/conv, failed publication and compaction retirement. |
| CF-05 | Concurrent ownership | `test_conformance_control.py`, `test_radiance_fair_scheduler.py`: cross-chat writes, shared mutable blocks, copy-on-write, stale requests and bank handovers. |
| CF-06 | Publication | `test_conformance_gate.py`, `test_conformance_session.py`: wrong outputs, latent state differences, stale revisions, crash durability, missing protocol coverage and forced diagnostic tokens. |
| CF-07 | Observer integrity | `test_conformance_instrumentation.py`, `test_conformance_boundaries.py`: missing calls/stages, invalid observation domains, input aliasing, nested differences, changed source/bytecode, misleading manifests and cleanup failure. |
| CF-08 | Formal evidence | `test_conformance_proofs.py`: implementation hashes, witnesses, broken gate/prefix/shortlist counterexamples, timeout and contradictory domains. |
| CF-09 | CPU CLI | `test_conformance_cli.py`, `test_conformance_isolation.py`: actual wrapper/symlink outside the repository, exit statuses, private reports, forbidden GPU imports, unarmed requests. |
| CF-10 | Native integration | `test_conformance_isolation.py` tests refusal without authorization/quiescence. Actual native extraction, connector restoration and asynchronous device behavior remain UNPROVED until GPU qualification. |

## Offline commands

Destinations are create-once. Raw states/tokens belong in private ignored
directories; public summaries contain only hashes, counts and scope statements.

```bash
scripts/qwen-conformance inventory
scripts/qwen-conformance proof-obligations --require-proved
scripts/qwen-conformance prove --output /tmp/private-proof
scripts/qwen-conformance recovery --plan /tmp/private/plan.json \
  --reference /tmp/private/reference-state --live /tmp/private/live-state \
  --restored /tmp/private/restored-state --output /tmp/private/recovery-result
scripts/qwen-conformance control --trace /tmp/private/events.json \
  --output /tmp/private/control-result.json
scripts/qwen-conformance calls --reference /tmp/private/reference-calls \
  --candidate /tmp/private/candidate-calls --output /tmp/private/call-result.json
scripts/qwen-conformance certificate --bounds /tmp/private/bounds.json \
  --output /tmp/private/certificate.json
```

The bounds file contains `lower`, `upper`, `winner` and `bound_origin`; rational
numbers can be strings. An empirical maximum error is not a universal bound.

```bash
UV_CACHE_DIR=/data/.cache/uv CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' \
  ROCR_VISIBLE_DEVICES='' QWEN_CONFORMANCE_GPU=0 uv run pytest -q \
  tests/test_conformance_*.py tests/test_diagnostic_contract.py \
  tests/test_radiance_fair_scheduler.py tests/test_radiance_chat_cache.py
```

The [latest CPU evidence report](backend-conformance-extended-evidence-20260915.json)
binds the run to source/test hashes and solver evidence. It does not certify native
kernels, device binaries, parser/RNG behavior or future optimizations. Those
require their own applicable proof or checked execution.

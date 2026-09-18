# Backend conformance prototype

The [verification and recovery guide](backend-conformance-verification.md) covers
the current recovery tests, detailed capture, control-event checks and executable
formulae. The [equivalence status](backend-equivalence-status.md) explains the
current precision profiles, proof coverage and remaining native qualification.
Historical evidence remains available; it does not qualify later source changes.

`scripts/qwen-conformance` provides an independent CPU reference, complete logical
state frames, forced-token replay, first-divergence comparison, small scoped SMT
proofs and a durable experimental publication gate. It reuses the existing
diagnostic contracts and numerical probes. It is not installed in Pi's request
path. CLI GPU work requires `native --allow-gpu`; direct native capture RPCs
require explicit arming and a caller-held quiescent worker boundary.

The native adapter is written for the pinned Radiance V2 runner in
`configs/profiles/radiance-conformance-v320260914.json`. Its layout and rejection
helpers have CPU tests. **Its real GPU execution is not yet qualified.** A working
CPU harness is not evidence that the deployed model is equivalent or cannot loop.

## Reference and admitted domain

The reference reads the original safetensors checkpoint through a read-only
loader and checks the supplied hashes for the configuration, index and shards.
Its supported architecture is the dense Qwen3.5 text decoder used by this stack:
hybrid GDN/convolution layers and full-attention layers, with gated attention and
MLP projections. Unsupported shapes/configurations fail explicitly.

New plans default to `weight-only-bf16`: MXFP4 E2M1 weights with E8M0 scales per
32 values, BF16 activations/KV and FP32 recurrent state. Explicit
`weight-fp8-activations`, `weight-fp8-kv` and `radiance-fp8` profiles isolate the
additional quantizers. Older profile-less plans retain their historical
`radiance-fp8` meaning and must be regenerated after reference source changes.
FP32 reductions use ordered separate multiply/add operations. The CPU reference
records executable library hashes and NumPy CPU dispatch features, in addition to
versions. Its scalar operations, transcendentals, compiler and hardware remain
trusted dependencies. There is no
Torch, vLLM, R4D or BLAS matrix multiplication in the CPU reference.

These activation and KV approximations are declared separately from weight
quantization. Dense native attention is used; Quest is not silently included.
The chosen reduction order defines a strict diagnostic reference. A small
optimized-versus-reference numerical difference is recorded as a mismatch, not
automatically diagnosed as a bug or silently accepted under a tolerance.

Inputs are explicit token IDs. Tokenization and chat-template construction are
outside this first replay's admitted domain. Logits cover the full vocabulary;
greedy ties select the lowest token ID. Sampling distributions, natural DFlash
acceptance, stop/EOS selection and tool parsing require separate qualification.

## Components and reusable boundaries

| File in `src/qwen_r9700_lab/` | Responsibility |
| --- | --- |
| `conformance_reference.py` | Independent finite-precision operators and quantizers |
| `conformance_model.py` | Original checkpoint loader, serial model and logical state restore |
| `conformance_state.py` | Private immutable tensor frames, byte comparison and archival |
| `conformance_replay.py` | Sealed schedules, independent prefill and operator capsules |
| `conformance_boundaries.py` | Ordered token/layer observations and first-divergence reports |
| `conformance_radiance.py` | Explicitly armed native V2 capture and forced M1/D7 adapter |
| `conformance_artifacts.py` | Mapped library hashes, package versions and safe runtime flags |
| `conformance_session.py` | Atomic checked state/output revisions in SQLite |
| `conformance_proofs.py` | Implementation-bound small helper obligations and counterexamples |
| `conformance_cli.py` | Offline commands and isolated native worker launch |

`scripts/qwen-diagnostics inventory` also lists the reusable earlier W4A16/Quest,
decoder, GDN, MXFP4, R4D and sampler instrumentation. Its old native adapters keep
their own capability limits. A new stack needs an explicit adapter; passing old
tests does not automatically qualify its native extraction.

Logical state includes consumed token IDs and position, an emitted-but-unprocessed
pending token, each GDN state and convolution history, logical attention K/V and
their scales. Physical block numbers may differ. The native adapter follows the
actual page tables and accepted recurrent-state column, and copies only committed
logical state. Shared immutable prefixes are legal; duplicate writable pages
within one logical sequence are rejected by this initial adapter.

Both executions start from their own empty state. At subsequent positions both
consume the same forced tokens, keeping early output divergence from changing
later inputs. D7 schedules explicitly choose accepted widths 0 through 7. A
correction/bonus token remains pending until processed. Forced acceptance checks
transactional state; it does not certify the natural speculative sampler.

Every retained step contains full target logits and persistent state. Common
decoder boundaries record input normalization, post-attention normalization and
layer output at every materialized prefill and accepted token by default. An
explicit `observation_positions` subset limits the capture domain visibly. These locate
the first *observed* differing boundary, not necessarily the earliest GPU
instruction. Operator capsules then replay a selected operator using identical
captured inputs. Existing deeper probes remain available for causal localization.

## CLI workflow

Run from the repository, using its uv environment:

```bash
scripts/qwen-conformance inventory
scripts/qwen-conformance --help
scripts/qwen-conformance prove --output /tmp/qwen-private-proof-run
```

The wrapper resolves symlinks and selects the repository's `.venv/bin/python`
without requiring shell activation. Use `uv sync --group dev` if dependencies are
missing. Inside the pinned native container, explicitly set
`QWEN_CONFORMANCE_PYTHON` to that container's Python if the mounted repository also
contains a host virtual environment. The older diagnostic wrapper supports the
equivalent `QWEN_DIAGNOSTICS_PYTHON` override.

Use new output paths for every run. Private input JSON must have mode 0600; evidence
directories have mode 0700. Keep tensor blobs, checkpoints and token IDs out of git.

`make-plan --spec FILE --output FILE` accepts exactly these fields:

| Field | Meaning |
| --- | --- |
| `checkpoint` | Absolute directory containing the original checkpoint |
| `checkpoint_files` | Relative filename to SHA-256 map, including config, index and all shards |
| `kv_scales` | FP8: full-attention layer index to `[key_scale, value_scale]`; BF16: `{}` or explicit unit scales |
| `prefix` | Nonempty list of token IDs for independent initial prefill |
| `forced_tokens` | Nonempty continuation token IDs, used only as diagnostic inputs |
| `accepted_widths` | Optional list of D7 widths; defaults to serial M1 steps |
| `reference_profile` | Optional precision contract; new plans default to `weight-only-bf16` |
| `observation_positions` | Optional nonempty, sorted materialized token positions; omitted means every position |

The schedule requires `len(forced_tokens) = 1 + sum(k + 1 for k in accepted_widths)`.
It writes the plan plus `.semantics.json` and `.execution.json` companions. Source
hashes bind the actual CPU reference, replay and tensor-code implementation.
Changing that implementation requires regenerating the plan. CPU executable
changes also refuse reference replay; different precision choices cannot retain
the same semantic identity.

```bash
scripts/qwen-conformance make-plan --spec /tmp/private/spec.json --output /tmp/private/plan.json
scripts/qwen-conformance reference --plan /tmp/private/plan.json --output /tmp/private/reference
```

The CPU reference is deliberately very slow. The tiny public four-layer fixture
tests the full path quickly; a complete 27B long-context run is not a practical
first qualification step. Begin native qualification with small numerical
capsules and short public prefixes.

After an exclusive GPU window is authorized, `native` runs a separate worker in
the pinned container/runtime. Its private LLM configuration must match the plan's
model, use one eager TP1/DP1 sequence, aligned Mamba state, the declared BF16 or
E4M3FN KV format and no snapshot
connector or asynchronous scheduling. Set `VLLM_USE_V2_MODEL_RUNNER=1` and
`RADIANCE_VERIFY_HEAD=0` for the initial full-vocabulary comparison. This diagnostic
restriction changes no live serving configuration. A D7 run also needs its pinned
drafter configuration; M1 and D7 use the same logical comparison format.

```bash
scripts/qwen-conformance native --plan /tmp/private/plan.json \
  --config /tmp/private/native.json \
  --binding configs/profiles/radiance-conformance-v320260914.json \
  --output /tmp/private/native-run --allow-gpu
scripts/qwen-conformance compare --reference /tmp/private/reference \
  --candidate /tmp/private/native-run/capture --output /tmp/private/comparison
scripts/qwen-conformance boundaries --reference /tmp/private/reference/boundaries \
  --candidate /tmp/private/native-run/capture/boundaries --output /tmp/private/boundaries
```

The native worker retains its logs, reviewed source binding and before/after
mapped-library manifests. Hashing a library file does not attest GPU code objects
dispatched from it; that gap is explicitly UNPROVED. Capture synchronizes the GPU,
copies state and hashes files, so its timings are unsuitable for speed benchmarks.

`operator --capsule DIRECTORY --output DIRECTORY` replays an operator capsule.
`frame` compares one pair of complete frames. Comparisons report exact byte
equality and diagnostic maximum absolute error, RMSE and relative L2 error.
Nonfinite values, missing observations and incompatible layouts cannot pass.

## Publication gate and failure behavior

`gate --authority DIRECTORY --transition FILE [--create]` accepts the contract,
ordered coverage list, reference/candidate frame paths, base revision, both output
token lists and both stop reasons. A trusted reference producer must supply the
reference transition; matching arbitrary files is not proof that either producer
ran the model. **Forced replay frames are marked and cannot be published.**

The authority archives and validates actual state bytes before a SQLite
transaction can publish anything. Files and their parent directory entries are
synced before the database can reference them. It checks position/pending-token
conservation, immutable prefix continuity, output equality and the current
revision under the write lock. A mismatch retains evidence without advancing the
session; it never fabricates EOS. Tool publication requires explicit independent
protocol-event coverage, which the native prototype does not yet provide.

Readers use `CheckedSession.outputs_since(revision)`. A committed revision can be
retrieved after a crash. External exactly-once tool effects still require a
consumer that acknowledges or idempotently applies that revision. This authority
is an experimental component, not a Pi proxy, and provides no sandbox against
malicious kernels or processes with the same UID.

## Evidence and next qualification

The [2026-09-14 CPU evidence report](backend-conformance-cpu-evidence-20260914.json)
records 157 passing checks, the 35 expected solver results, an injected latent GDN
state fault that was detected, and the exact source hashes. Native GPU validation
was not run. Private reproduction artifacts are retained under the ignored
`.qwen-conformance/20260914-cpu-readiness/` directory.

The helper suite has 50 solver obligations, including expected counterexamples.
Their retained SMT queries and helper source hashes establish only the encoded
pure-function properties under the solver/Python assumptions. Timeout, unsupported
operations or unknown results remain UNPROVED. They do not prove vLLM's allocator,
scheduler, compiler or GPU binary.

CPU tests cover real CLI replay, initial prefill, restore/continuation, all eight
D7 widths, physical page permutation, latent-state corruption, omitted/reordered
observations, stale source bindings and nonfinite tensors. Publication tests
kill a process before and after the actual database commit and race two real
connections for one revision. These are finite tests, not power-loss or hardware
proofs.

Next: qualify the native hooks on the actual GPU, capture a short independent
prefill, locate the first numerical difference, then exercise every D7 boundary
with deliberately altered rejected suffixes. Real snapshot restoration,
cancellation, concurrent chats, graphs/asynchronous execution and sampled/tool
semantics remain explicit extensions of the admitted domain. No production
correctness certificate or percentage of loops eliminated follows from this
CPU-only implementation.

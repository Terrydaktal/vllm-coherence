# Verification and evidence

There is no universal correctness certificate for this backend. The numerical
contract, reachable state, trusted compiler/hardware assumptions and measured
domain must be explicit. Weight, activation and KV quantization are distinct
approximations. Global-256 candidate selection is an additional approximation.

Let `R_C(S,u)` be the reference transition under contract C, and `alpha(P)` map
physical candidate state to logical state. The desired refinement obligation is:

```text
alpha(P) = S implies
  alpha(B_C(P,u).state) = R_C(S,u).state
  B_C(P,u).output      = R_C(S,u).output
```

Logical state includes KV order/values, GDN recurrence, convolution history,
processed position, any emitted-but-pending token, sampler and protocol state.
Physical block IDs may differ and immutable prefixes may be shared.

For speculative execution, compare each target row with the serial causal prefix,
commit only the processed accepted prefix, and establish rejected-suffix
noninterference. Do not confuse an emitted correction/bonus token with a token
already processed into persistent state. For snapshots, save/load must preserve
the declared logical state and continuation. For checked execution, no candidate
output or state may publish before both compare equal; rejection produces a
reference fallback or explicit error, never a fabricated EOS.

The implementation distinguishes these evidence levels:

| Label | Meaning |
| --- | --- |
| PROVED | A scoped mechanically checked obligation with explicit model/implementation binding |
| RUNTIME-CHECKED | This invocation was checked before its declared publication boundary |
| TESTED | Finite independent reference/invariant tests, including negative controls |
| ASSUMED | An explicit trusted dependency or hardware/compiler assumption |
| UNPROVED | Missing coverage, unsupported mode, timeout or incomplete correspondence |

The small SMT/control proofs are not end-to-end proofs of the actual GPU binary.
Floating-point reassociation can be unequal while remaining close; a tolerance
must never be silently described as exact agreement. Hash identity binds evidence
to artifacts but is not itself a numerical correctness proof.

## CPU suite

```sh
uv sync --frozen --extra cpu-tests
uv run --frozen --extra cpu-tests pytest -q
python3 tools/check_publication.py
```

The CPU-only Torch extra cannot execute ROCm kernels. Tests cover synthetic
reference arithmetic, negative controls, source/evidence admission, state and
storage transitions, scheduler decisions, Pi runtime patches and extensions,
archive extraction and portable command construction. Packaging checks reject
traversal, symlink substitution, missing artifacts and changed bytes.

`verification/coverage.json` maps the public behavior groups to their implementation
and test families. It documents native obligations that CPU CI cannot discharge.
CPU success is never reported as a new GPU numerical result.

The obsolete pre-Radiance W4 deployment launcher and its deployment-only tests
are outside this release. Reusable operator/reference instrumentation is retained.
Source fixtures under `tests/fixtures` contain upstream code and one pinned GDN
executable used for integrity tests; they contain no chat tokens or tensor captures.
The reflink host check additionally requires `QWEN_REQUIRE_REFLINK=1` on the
intended evidence filesystem and otherwise reports an explicit skip.

## Native sequence

1. Pin checkpoint/quantizers, template, backend/compiler, arithmetic and sampler.
2. Build an isolated candidate and authenticate its source/artifact identities.
3. Compare initial prefill, then common-input operator transitions including state.
4. Exercise boundary shapes, every admitted rejection width, cancellation,
   ownership, cache flush/restore and deliberately injected corruption.
5. Replay identical forced tokens while each path evolves its own state. Capture
   the first differing token/layer/operation. Keep private values owner-only.
6. Verify actual graph execution, then time the release path without instrumentation.
7. Publish aggregate evidence with exact scope and unresolved cells.

The native drivers require explicit fixture/spec/output paths and GPU admission;
consult `coherence-conformance --help`, the drivers' help and
[BUILDING.md](BUILDING.md). Runs must use isolated state, never a production chat's
live mutable cache. The supplied synthetic fixtures are redistributable. The
historical 23-continuation Pi corpus and raw tensor captures remain private.

## Completed historical evidence

The [report](../reports/d7-rdna4-2026-09-17/REPORT.md) records full-head agreement for
10,000 forced decode tokens and 23 prefills after two numerical repairs. The
all-seven-accepted replay is not exhaustive over arbitrary inputs or speculative
rollback histories. It does not independently certify eager M1 against a bare
mathematical model. Later GEMM/norm/prefill changes have separately recorded
operator checks, 320 whole-model rows and natural-completion speed controls.

Global-256 has separate candidate-recall/head-latency and natural-generation
measurements. Its finite shortlist is not a completeness certificate. The
experimental certified-head code falls back when its proof/admission conditions
are not met; it is not the default serving head.

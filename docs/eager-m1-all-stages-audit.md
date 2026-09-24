# Independent eager-M1 stage audit — 24 September 2026

This audit extends the [earlier repairs](eager-m1-followup-audit.md) with independent
arithmetic checks across every target transformer layer. It uses public checkpoint
weights and synthetic inputs; no chat contents are read. It does not establish
universal equivalence or a probability that an operator has no bugs.

The native runner is [probe_m1_stage_audit.py](../experiments/radiance-public/probe_m1_stage_audit.py).
Its [CPU oracles](../experiments/radiance-public/m1_stage_oracles.py) calculate FP64
equations independently, with explicit BF16/FP8 conversions where specified.
Numerical error checks and exact representation/state checks are recorded separately.
An operator passing an error threshold is not reported as bit-identical to FP64.

## Results

[Complete numerical results and source identities](../benchmarks/results/eager-m1-all-stages-20260924.json)
retain each site's checks, actual public weight hashes, native source/binary
bindings and the separate pointwise/sampling evidence.

The complete operator audit recorded **2,705 passing checks out of 2,745**. All
40 failed checks were comparisons of larger-batch prefill normalization with
serial M1: **55 differing FP8 bytes among 211,353,600**, with identical scales.
Every M1-specific check passed its stated criterion. These are synthetic operator
checks, not generated tokens or a percentage estimate of model correctness.

A separate **445/445** pointwise follow-up checked both the one-dimensional text
position path and the three-axis MRoPE path. Launch counters confirmed that all
18 intended Q/K/position cases executed the repaired eager MRoPE kernel. This
follow-up replaces the narrower one-dimensional RoPE coverage when assessing
that stage; it does not require repeating unchanged matrix/state checks.

The CPU regression run passed **355 tests**. Sampling used **600,000 trials**
with the independent proposal stream: the largest observed absolute probability
error was **0.118 percentage points**. Deliberately sharing the noise stream
produced **1.78–1.95 percentage points** of error and was detected. Greedy controls
were exact. The 59 scoped solver cases returned their expected results: 47 scoped
proofs and 12 counterexample/negative-control results. The complete backend's
proof status remains **UNPROVED**. The 25 GPU-only head-selection and fallback
tests skipped by the CPU run were then executed on the installed native modules:
**25/25 passed**, including the clustered-top-20 failure, Global-512 selection,
unchanged drafter behavior and unsupported-input full-head fallbacks.

The complete audit took 214.8 seconds and peaked at 93.2 MiB of PyTorch allocator
memory, in addition to the diagnostic process's HIP context. The pointwise
follow-up took 5.3 seconds. These are audit costs, not inference stage timings.

## Coverage

| Stage | New native coverage | Comparison |
| --- | --- | --- |
| Token embedding | Public vocabulary slabs at four offsets, repeated and boundary indices | Exact stored BF16 values |
| Input, post-attention and final RMS normalization | All 129 hidden-width sites, 320 input rows each | Independent FP64 RMS equations; residual carry checked exactly |
| Attention Q/K normalization | All 32 sites, 320 positions each using their actual 24-query/4-KV-head geometry | Independent FP64 RMS equations, actual corrected M1 kernel |
| Normalization → FP8 | All 128 layer-norm producer sites plus a final-norm-weight control, 320 rows each | Exact FP8 bytes/scales against independent encoding of the separately checked M1 normalization; the serving final norm itself remains BF16 |
| GDN QKV/gate/output, attention Q/K/V/output and MLP projections | All 496 matrices across 64 layers, 320 M1 inputs per matrix | Independent decoding of checkpoint coefficients and FP64 dot products at four output channels per matrix, eight at the original coefficient-failure site; exact one-hot witnesses |
| GDN convolution | All 48 layers, 320 consecutive inputs each | FP64 convolution/SiLU, exact final convolution history and unrelated-slot preservation |
| GDN gates and recurrence | All 48 layers, 320 transitions each; additional 32,768-step decay trajectory | Independent delta-rule state/output equations and closed-form decay; exact unrelated-slot preservation |
| GDN gated normalization and FP8 producer | All 48 sites, 320 inputs each | Independent FP64 normalization/gate equations, then exact fused FP8 bytes/scales against the native BF16 boundary |
| SiLU, sigmoid and MLP FP8 producer | All finite BF16 gate encodings with a bounded multiplier; 320 fused M1 inputs | Independent pointwise equations; exact fused FP8 comparison against the separately evaluated native BF16 operations |
| RoPE | Q/K geometries at nine position ranges, including 60K and 200K, in one-dimensional and three-axis text-position layouts | Corrected eager RNE kernel against FP64 rotation; untouched tail exact; actual repaired-kernel dispatch verified |
| KV writes and representation | BF16/OCP FP8, non-unit scales, negative slot mask and page boundaries; every finite FP8 code | Exact logical write destinations, stored values and FP8 widening |
| Attention | 88 format/scale/context cases, context lengths 1–253,792 | Independent dense FP64 softmax over periodic pages; exact output/scratch/cache isolation guards |
| Target-head scoring | Three public 512-row vocabulary slabs × 320 hidden inputs | Independent FP64 dot products for BF16 head and candidate reranker; exact one-hot witnesses |
| Sampling | 200,000 trials at each of three token positions | Known target distribution, independent proposal stream, greedy controls and deliberately correlated-stream negative controls |
| Snapshot, commit, rollback and ownership | Existing CPU state/corruption suite and scoped solver cases | Logical invariants and injected failures; not new native snapshot-cycle qualification |

The projection checks execute the full native matrix shape but independently
compare selected output channels. They are not exhaustive comparisons of every
coefficient/input combination. Attention uses three immutable physical pages
repeated into the full logical context, allowing the long-context kernels to be
tested beside an idle server without allocating another full KV cache. CPU tests
compare this multiplicity-based oracle with explicitly expanded dense attention.

RoPE uses native GPU frequency construction at the real positions, with a compact
lookup table to bound memory. It tests the arithmetic and lookup operation, not
physical accesses to every row of a full 250K-position table. The corrected eager
RNE intervention is loaded explicitly from the frozen release: importing the
original on-disk vLLM module alone would test the pre-alignment eager kernel.

## Remaining differences

Prefill normalization and M1 normalization have different reduction orders.
The fused kernel uses 512 reduction lanes through eight rows, then 64/32 for
larger batches to preserve the existing large-batch PyTorch arithmetic. The audit
compares prefill with serial M1 explicitly and retains those differences as
failures of exact cross-path equality. It does not silently widen the tolerance
or classify prefill as bit-identical to M1.

Global-512 remains an approximate candidate selector. Checking retained scores
does not prove that every required vocabulary token survived selection. The
[head-depth study](HEAD_CANDIDATE_DEPTH.md) remains the applicable measured recall
evidence. Unconditional full-head scoring, or a sound exclusion certificate with
fallback, is required to remove that particular approximation.

FP64 comparisons expose rounding differences as well as defects. Native GDN gated
normalization also has a different intermediate-cast contract from the Hugging
Face implementation, as recorded in the [first audit](eager-m1-independent-audit.md).
Neither agreement with another fast kernel nor a small FP64 error proves that
the complete model is identical to an independently specified reference.

This audit adds no per-token serving instrumentation, changes no production
arithmetic, and requires no backend restart. It does not supersede historical
full-model evidence or qualify every combination of compilation, asynchronous
execution, snapshot restoration and concurrent requests.

## Reproduction and failure handling

Run the probe against the frozen release in an idle R9700 environment. Its module
directory must include `m1_stage_oracles.py` and `probe_eager_m1_independent.py`;
the remaining native modules and the eager RoPE repair come from the release.

```bash
python probe_m1_stage_audit.py --release /qualification \
  --model /models/Qwen3.8-27B-Uncensored-MXFP4-awq \
  --output /tmp/m1-audit/result.json --allow-gpu \
  --rows 320 --transitions 32768
```

Create the output directory beforehand. The runner refuses an existing result,
checks for an idle backend and available VRAM, and limits its GPU allocator to
192 MiB. Set `QWEN_CONFORMANCE_GPU_LOCK` for the shared qualification lease. The
idle check does not reserve the serving scheduler; a new Pi request causes a
subsequent check to stop the audit. Never treat an interrupted stage as qualified.

Results bind the exact probe/oracle source, release manifests, loaded native
binaries and public weight tensors. An exception, omitted stage, empty inventory
or missing record makes the result incomplete. Expected numerical discrepancies
remain visible in the machine-readable result and produce a nonzero exit code.
The underlying CPU tests also reject injected state corruption, empty proof
domains, solver timeouts and incomplete evidence.

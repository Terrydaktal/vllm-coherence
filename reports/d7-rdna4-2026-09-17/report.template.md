# Restoring M1/M8 and Eager/Compiled Agreement on RDNA4

**Terrydaktal · 17 September 2026 · Technical report, version 6**

## Abstract

A pinned Radiance/vLLM inference stack produced different target logits when the
same saved token history was processed one position at a time (M1) or eight
positions at a time under D7 speculative verification (M8). The investigation
localized numerical differences in GDN prefill and decode, convolution, gated
normalization, attention, residual normalization and the BF16 vocabulary head.
Four performance changes reduced a repaired verification round from approximately
76 ms to 60 ms in the measured 60K-input control.

A later eager/compiled investigation isolated intermediate BF16 rounding and
native RoPE multiplication differences. Aligning both produced identical full
logit vectors at 320 decode positions and the final prefill prediction, with
identical captured inputs and outputs at all 465 matched activation boundaries.
The combined repair is now measured directly as **eager M1 versus compiled M8**
before and after both fixes, using the same 10,000-position Pi corpus.

This is an empirical result on one pinned configuration; it does not establish
arbitrary-input equivalence or improved task accuracy.

The work consists of **two major fixes**:

| Revision | What changed | Comparison it repairs |
| --- | --- | --- |
| Original | Pinned backend before these repairs | Baseline |
| Fix 1 | Align M1/M8 arithmetic and state transitions, then recover performance | Compiled M8 versus compiled M1 |
| Final: Fix 1 + Fix 2 | Preserve BF16 intermediate rounding in compiled execution and use nearest-even RoPE products in eager execution | Compiled M8 versus eager M1 and eager M8 |

## 1. Configuration and reference

- One AMD Radeon AI PRO R9700, gfx1201, TP1.
- Radiance 1.0.16 / vLLM 0.28, PyTorch 2.12.0, ROCm 7.14, Triton 3.7.1.
- Qwen3.8-27B-Uncensored-MXFP4-awq; 64 layers, including 48 GDN and 16 attention
  layers; hidden width 5,120; vocabulary size 248,320.
- The target vocabulary head uses the full original BF16 weights in both arms.
  INT2 candidate selection is excluded from the equality claim. Other model
  weights, activation quantization and cache formats remain part of the pinned
  quantized backend; this is not an unquantized BF16-model comparison.
- The compiled comparisons use Inductor and piecewise GPU graphs, with capture
  sizes `[1, 2, 4, 8]`. Actual target graph replay is audited.
- The numerical reference is the explicitly selected serial implementation.
  Repairs also affect M1/preparation paths, so repaired M1 is measured afresh.

The 10K corpus comprises 23 saved Pi continuations with 57,008–65,527-token
prefixes. Both arms consume identical saved token histories at aligned logical
positions. Each M8 group processes eight positions with all seven supplied
proposals accepted. The denominator is 10,000 processed decode positions,
excluding intermediate-prefill head calls. Final initial-prefill predictions
have a separate denominator of 23. No generated tools execute during replay.

## 2. Agreement definitions

For each position t, let z1(t) and z8(t) be the full vocabulary logit vectors.
T_k sorts logits by decreasing score and then increasing token ID for ties.

```text
set agreement(k)   = count(set(T_k(z1)) == set(T_k(z8))) / N
order agreement(k) = count(T_k(z1) == T_k(z8)) / N
mean shared(k)     = sum(|set(T_k(z1)) intersect set(T_k(z8))|) / N
```

Retained-score equality, inclusive boundary-tie membership and full-vector
SHA-256 equality are separate checks. Matching hashes are strong empirical
evidence, not a collision-free mathematical proof. Agreement measures execution
consistency; it is not a percentage of questions answered correctly.

## 3. Top-1/10/20 and ordering results

### Eager M1 versus compiled M8: before and after both fixes, full 10K corpus

**Before:** original eager M1 versus original compiled M8.
**After both fixes:** final eager M1 versus final compiled M8 (Fix 1 + Fix 2).
Each revision has a freshly measured eager M1 reference. All four runs use the
same 23 continuations, context lengths and full BF16 target head. Compiled M8
uses Inductor and piecewise GPU graphs; eager M1 disables compilation and graphs.

{{CROSSMODE_10K_TABLE}}

**Same set** means the same tokens regardless of order. **Same order** additionally
requires identical ranking. **Mean shared** is the average number of shared
tokens per position.

{{CROSSMODE_10K_SUMMARY}}
[Counts, run identities and completion audit](evidence/crossmode-before-final-10k.json).

### Fresh four-way comparison: 320 positions

Seven fresh native runs used the same saved 60,000-token Pi prefix and 320
aligned decode positions. Each M1/M8 pair uses its own revision's M1. Compiled
arms ran with Inductor and piecewise GPU graphs; eager arms explicitly disabled
compilation and graphs.

{{FOUR_COMPARISON_TABLE}}

These are **whole-model** results. The isolated stage comparisons in section 4
use common correct inputs to prevent upstream errors contaminating the stage
being measured. [Run identities and complete counts](evidence/final-study-four-comparisons.json).

## 4. Compiled stage timings and isolated correctness measurements

Both timing profiles use the same 60,000-input-token Pi fixture and the
optimized compiled configuration: **`enforce_eager=false`, Inductor, PIECEWISE
GPU graphs, capture sizes `[1, 2, 4, 8]`**. The actual traces contain **65 target
GPU graph launches in every captured round**, in both arms. Runtime receipts
are recorded in [final-study-profile-audit.json](evidence/final-study-profile-audit.json).
These are not eager timings. The full BF16 target head is used on both sides.

The fresh profiles contain one incomplete target-kernel inventory in each
arm: original round 8 and final round 7 (one-based). The stage table uses the
**six paired complete rounds, 1–6**, selected by inventory completeness rather
than timing. Every recorded dispatch, including excluded rounds, remains in
the evidence. [Profile audit](evidence/final-study-profile-audit.json).

All times are milliseconds per verification round: summed GPU dispatch
durations for all layer instances, divided by six.
The 256 projection dispatches per round are assigned to their four operations
in each of 64 layers. The approximately 25 ms gate/up figure is the joint MLP
projection across all 64 layers; its individual layer contributions follow.

The four correctness columns report **vocabulary top-20 set/order agreement on
320 positions**, with each layer instance given common correct inputs. Only
one instance is substituted at a time before the native reference remainder
produces logits. A position passes an aggregated stage only when every instance
passes. Exact local outputs and state can reuse the already validated reference
remainder; shared implementations across revisions are identified explicitly.
These are not rankings of intermediate activation coordinates.
The all-layer requirement makes these rates stricter than a single whole-model
comparison.

“Performance” marks overhead removed from the initial repaired implementation;
it does not imply that a stage is faster than the original unrepaired M8.
A Fix 1 label records alignment to the selected serial arithmetic. Some original
compiled M1/M8 stages already agree with each other on this sample; their actual
agreement counts remain shown rather than being labelled as failures.

[Per-layer comparison counts](isolated-stage-layer-comparisons.csv) retain the
individual results behind the aggregate table, including top-1/10/20 and full
logit-vector agreement.

Original comparisons use the original M1 and M8 stage implementations; Fix 1
comparisons use the repaired implementations. Every column uses the same
final-correct captured inputs. Thus these isolated rates diagnose individual
stages; they are not the whole-model rates from section 3.

Compiled diagnostic replay disables graph replay to expose individual calls;
its ordinary outputs must first match the graph-enabled release. **Diagnostic
replay durations are excluded from this timing table.** The original and final
columns below use fresh graph-enabled release profiles.

<div class="wide-table">

{{STAGE_TABLE}}

</div>

† Attention decode and split-KV merge have separate GPU timings but one native
numerical interface, so their correctness cells report the joint comparison.
GDN transport copies are included in the convolution/recurrence state checks;
they do not have an independent vocabulary-prediction result. Drafter and
bookkeeping work are outside the target-stage comparison.

Target-body subtotal: **{{TARGET_TOTAL}}**. Total recorded GPU kernel duration:
**{{ALL_KERNEL_TOTAL}}**. These are subtotals, not extra stages. Kernel durations
can overlap and exclude host/queue gaps; they are not wall-clock round latency.

### Interpreting the timing differences

The FP8 quantizer is unchanged and runs 256 times per round in both versions.
Its fresh totals are **0.806 ms original and 0.810 ms final**. The earlier large
quantization timing difference does not reproduce in this comparison.

Attention decode averages **4.224 ms original and 5.285 ms final**. Final
rounds split into approximately 3.55 ms and 7.02 ms groups. The repaired kernel
uses one or two groups of queries aligned to 16-token tiles to preserve each
query's M1 causal and softmax decisions; a second group repeats context work.
The split-KV merge is timed separately: **0.138 ms original and 0.120 ms final**.

An unchanged stage's timing delta alone does not establish a performance fix.

### Every layer, including the 64 contributions to the gate/up total

Layer numbers are zero-based. “Other three projections” means the input,
attention/GDN output, and MLP down projections; all individual semantic stage
values are also retained in `stage-times.json`. Non-projection work includes
normalization, gating, quantization, attention/GDN operations and copies.

<details>
<summary>Expand all 64 layers</summary>

{{LAYER_TABLE}}

</details>

### Every recorded compiled kernel within each semantic stage

Calls are totals over the six retained rounds; times are per-round means.
Compiler-fused operations remain indivisible. The tables do not invent times
for separate constituents of a single GPU dispatch. Individual dispatch
durations, layer IDs and round IDs are retained in the two
`evidence/final-study-*-dispatches.json` exports; they contain no token IDs, raw
activations, absolute timestamps or process identifiers.

<details>
<summary>Expand the complete kernel breakdown</summary>

{{KERNEL_TABLE}}

</details>

Prefill is outside this steady-decode timing table and has a separate
causal/chunk-partition repair. CPU scheduling has no GPU-kernel duration here.
The last row retains GPU bookkeeping outside the attributed model scopes;
its individual kernels are listed without inventing missing semantic labels.

## 5. Brief unprofiled engine controls — 60K input, not 60K output

**The approximately 80–87 tok/s figures are short controls on one 60,000-input-
token Pi fixture. They are not results from generating 60,000 tokens.**

{{SPEED_TABLE}}

{{SPEED_SUMMARY}}

These direct-engine controls exclude HTTP/Pi, tools, snapshot publication,
cold prefill and warm-up. Rates pool tokens and time after the first output
chunk; round latency is the median of the per-response steady-round medians.
Throughput also depends on speculative acceptance, so it is not a pure kernel
speed ratio. Clean timing passes do not record GPU profiles or copy activations.

### Cost of preserving compiled BF16 rounding

Four fresh compiled runs used **before–after–after–before order**, with three
natural responses per run on the same 60,000-token Pi prefix. Both settings use
the final four performance repairs, full BF16 target head and piecewise GPU
graphs. The only changed compiler setting is
`TORCHINDUCTOR_EMULATE_PRECISION_CASTS=0/1`.
The stage profiles in section 4 now include this rounding alignment. The
following separate ABBA comparison isolates Fix 2 from Fix 1.

{{ROUNDING_SPEED_TABLE}}

{{ROUNDING_SPEED_SUMMARY}}

Round time is effectively unchanged in this sample. The slightly lower token
rate is consistent with fewer committed tokens per round. Every run used the
same sampling settings and three seeds, but changed rounding produced different
natural continuations. The nearest-even native RoPE correction applies to the
**eager reference**; this measures the compiled half of the alignment. Startup,
compilation, warm-up and cold prefill are excluded from the decode timings.
All twelve natural completions are included; per-round timings and source
identities are retained in [rounding-speed-abba.json](evidence/rounding-speed-abba.json).

### Separate, earlier 60K-generated-token verify-head experiment

This earlier compiled experiment used 115 natural responses per method across
11 intact Pi request boundaries, with 57,008–65,527 input tokens. It predates
the final D7 repairs and **does not measure the final corrected M8's long-run
speed**. The head methods also have different numerical guarantees.

| Earlier head method | Natural responses | Generated tokens | Timed post-first seconds | Post-first tok/s |
| --- | ---: | ---: | ---: | ---: |
| Full BF16 reference head | 115 | 60,598 | 946.230 | 63.920 |
| INT2 block-8 / rerank-80 | 115 | 60,075 | 892.548 | 67.178 |
| INT2 global-128 / BF16 rerank | 115 | 60,348 | 892.266 | 67.506 |
| INT2 global-256 / BF16 rerank | 115 | 60,675 | 906.873 | 66.779 |

The first output chunks are excluded from both the rate numerator and timed
denominator. See [earlier-60k-output-speed.json](evidence/earlier-60k-output-speed.json).

## 6. Repairs and upstream ownership

1. **GDN prefill:** raw per-token gates and an ordered transition avoid the
   diagnosed dependence on future gates, cumulative-gate rounding and chunk
   partition. Core implementation belongs with libr4d; adapters belong in Radiance.
2. **GDN decode:** match serial convolution arithmetic, gate precision, recurrent
   reductions, output rounding and retained-prefix state. Rejecting a speculative
   suffix must not make its state persistent. Packed QKV views remove copies.
3. **Gated normalization:** retain the serial row tile without serializing all
   eight rows. The vendored FLA dispatcher belongs in vLLM.
4. **Attention:** retain each query's causal range and serial reduction/split
   policy; share KV reads to recover performance. Core ownership is libr4d.
5. **Residual/QK/final normalization:** preserve the chosen finite-precision
   boundaries; retain FP32 residual sums in registers. Existing vLLM rounding
   PRs must be checked for overlap before proposing another repair.
6. **Full BF16 head:** interleave two arithmetic-preserving M4 groups in one
   HIP launch. This avoids the first repair's duplicated launch/concatenation
   overhead. The upstream skinny-GEMM implementation belongs in vLLM.
7. **Fix 1 compiled integration:** install dispatch before tracing/capture, use opaque
   bindings where necessary, and validate startup-only fallbacks separately from
   real requests. Integration and conformance adapters belong in Radiance.

8. **Fix 2, execution-mode rounding:** enable
   `TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1` for the compiled target and preserve
   nearest-even BF16 products in native eager MRoPE. These changes address a
   separate eager/compiled discrepancy after Fix 1 had aligned M1 and M8.

These are submission scopes, not a claim that the pinned adapters apply
unchanged to current upstream main. The new ports require their own tests.
Upstream links and submission state are maintained in `submissions.md`.

## 7. Execution-mode diagnosis and remaining limits

The eager/compiled mismatch was localized to intermediate BF16 casts and native
RoPE multiplication. Aligning both produced exact eager/compiled **M8** output
agreement on the controlled 320-position Pi replay. This is separate from the
fresh 10K eager-M1/compiled-M8 before/after comparison in section 3, which
measures both fixes together across all 23 continuations.

The [complete boundary comparison](common-rounding-boundaries.md) also matches
captured inputs and outputs at all 465 matched boundaries, across 320 decode
and nine sampled prefill positions. This sampled activation check supplements
the whole-model and isolated-stage results; it does not cover all arbitrary
inputs or every persistent state transition.

### Intermediate BF16 casts

Before alignment, the first difference occurred in layer 0's MLP SiLU/gating,
after identical GDN and gate/up projection outputs. The
[isolated native replay](evidence/isolated-silu-modes-320.json) tested all 64
layers on common inputs: 20,480 decode layer-position evaluations and 576
sampled prefill evaluations. Each kernel reproduced its own captured output;
none of the complete decode output rows agreed between modes.

```text
eager:    BF16(BF16(SiLU(gate)) * up)
compiled: BF16(SiLU_FP32(gate) * up)
```

The BF16-intermediate arithmetic oracle matched eager in all 20,480 decode
evaluations. An independent FP32-intermediate calculation still differed from
the generated compiled kernel at 17 individual values, so that formula alone
is not a bit-exact specification of the compiled operation.

Attention sigmoid gating had the same cast-elision distinction: eager rounded
the sigmoid to BF16 before multiplying; compiled retained it in FP32. A
[common-input native pilot](evidence/rope-gate-native-pilot.json) reproduced
each implementation's own output and found 0/8 equal output rows between modes.

`TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1` preserves these intermediate casts.
The experiment checks the loaded compiler boolean as well as the environment
setting. Cast preservation alone did not restore equality, in the [cast-only control](evidence/precision-casts-vs-eager.json);
it exposed the remaining RoPE difference after matching the first three layers.

### Native RoPE multiplication

With compiler casts preserved, the first remaining discrepancy occurred in
layer 3 after identical QKV projection and Q/K normalization. The
[fine attention comparison](evidence/precision-attention-cut.json) found:

| Layer 3 boundary | Exact decode rows / 320 | Exact sampled prefill rows / 9 | Differing decode values |
| --- | ---: | ---: | ---: |
| QKV projection | 320 | 9 | 0 |
| Query normalization | 320 | 9 | 0 |
| Key normalization | 320 | 9 | 0 |
| Query after RoPE | 0 | 1 | 184,437 |
| Key after RoPE | 0 | 1 | 30,984 |
| Value input | 320 | 9 | 0 |
| Attention output | 0 | 1 | 1,335,594 |
| Gate input | 320 | 9 | 0 |
| Gated attention output | 0 | 1 | 1,282,380 |

The later output differences inherit changed rotary inputs and state; this
table does not establish separate attention-kernel defects. On the same
normalized Q/K and selected BF16 cosine/sine coefficients, the
[explicit arithmetic oracle](evidence/precision-rotary-formulae.json)
reproduced each mode exactly at all 320 decode and nine sampled prefill positions:

```text
native eager:       RNE_BF16(RTZ_BF16(a*c) - RTZ_BF16(b*s))
compiled emulation: RNE_BF16(RNE_BF16(a*c) - RNE_BF16(b*s))
```

The second rotary half uses addition with the same product-rounding rules;
the 192 non-rotary coordinates are unchanged. `RTZ` means round toward zero;
`RNE` means round to nearest, ties to even. This isolates rotation arithmetic,
not the separate selection of cosine/sine coefficients.

The [native replay](evidence/rotary-rne-native-replay.json) then compared the
installed kernel with a private copy that changed only product rounding:

| Native RoPE variant | Q and K exact versus original eager | Q and K exact versus compiled cast preservation |
| --- | ---: | ---: |
| Original native multiplication | 320/320 decode; 9/9 prefill | 0/320 decode; 1/9 prefill |
| Explicit nearest-even multiplication | 0/320 decode; 1/9 prefill | 320/320 decode; 9/9 prefill |

Both Q and K independently meet those counts. Position zero is the identity
rotation. Input/coefficient immutability and injected one-bit errors are
checked. Source, LLVM IR, AMD assembly and GPU-binary hashes bind the result
to the executed implementations; the full-model worker also verifies that the
modified kernel runs during the request.

The installed **Triton 3.7.1 / PyTorch 2.12.0+rocm7.14** build lowers BF16
multiplication to `llvm.amdgcn.fdot2.bf16.bf16`, emitted as
`v_dot2_bf16_bf16` on the R9700. Its observed truncation is the defect already
fixed by [Triton PR #11227](https://github.com/triton-lang/triton/pull/11227),
merged August 14, 2026. That fix replaces the special lowering with an FP32
multiply and explicit nearest-even conversion. The new evidence here is the
defect's occurrence and causal isolation inside the captured Qwen RoPE path;
the upstream repair is credited to its existing authors.

### Remaining coverage limits

- The fresh 10K comparison measures eager M1 versus compiled M8 before and after
  both fixes. The separate eager-M8/compiled-M8 comparison covers 320 positions.
  Neither establishes arbitrary-input equality. The brief compiled speed
  comparison is reported in §5; instrumented replay durations are not serving speed.
- Captured activation equality does not certify uncaptured KV/GDN/convolution
  state, unpaired operations or independent MRoPE coefficient selection.
- The 10K replays cover the all-seven-accepted D7 path. Small
  operator tests exercise rejection boundaries and injected faults, but full
  model coverage of every rejection history, long-context range, concurrency,
  snapshot restore and hardware platform is not established.
- Neither experiment proves arbitrary-input equivalence to an independent
  mathematical reference, eliminates model-generated loops or measures improved
  coding accuracy. Evidence remains tied to its source, binary and configuration
  identities; changed implementations require qualification.

## 8. Evidence and reproduction

The `evidence/` directory contains aggregate comparisons and kernel profiles,
with file SHA-256 values in `evidence-sha256.json`. Recompute the stage table,
kernel accounting and aggregate assertions without GPU use:

```sh
uv run python reports/d7-rdna4-2026-09-17/build_report.py
```

The original Pi transcripts, token IDs, ranked token IDs and raw traces remain
private. The aggregate measurements can be audited from the published receipts;
independent end-to-end reproduction needs a public replacement workload.
Synthetic operator regressions accompany the corresponding source submissions.

The [10K before/after audit](evidence/crossmode-before-final-10k.json) binds all
four runs to the same corpus and records their source, repair, compiler, graph
and per-continuation receipt identities. The final eager M1 reference and final
compiled M8 candidate both include Fix 1 and their respective Fix 2 rounding
repairs.

## References and attribution

- [Radiance](https://github.com/magiccodingman/vllm-radiance), the qualified downstream stack.
- [libr4d](https://codeberg.org/StillDeadcode/libr4d), RDNA4 attention and recurrent kernels.
- [vLLM](https://github.com/vllm-project/vllm), including ROCm skinny GEMMs and vendored FLA.
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention), serial GDN/normalization foundations.
- [vLLM batch invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/), related existing work.
- [PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html), batched/reduction-order limitations.
- Existing normalization work: [vLLM #49639](https://github.com/vllm-project/vllm/pull/49639) and [#52243](https://github.com/vllm-project/vllm/pull/52243).

This contribution builds on those implementations. Batch invariance itself is
not introduced by this report. Earlier DFlash RNG work in Radiance PR #8
backports [vLLM #54282](https://github.com/vllm-project/vllm/pull/54282) and is
not presented as a new upstream discovery here.

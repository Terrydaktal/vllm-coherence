# Eager M1 independent operator audit — 24 September 2026

The [follow-up audit](eager-m1-followup-audit.md) investigates remaining attention
precision and conditional layout/format issues after these repairs.

Native R9700 tests confirmed two numerical defects in operators used by eager
M1, plus a layout defect in a disabled fallback. A fourth finding is a difference
in the normalization rounding contract, not evidence that the current operation
is less accurate. No production kernel, model configuration or server was changed.
No private conversation was read.

The repairs and their subsequent qualification are recorded under
[Implemented repairs](#implemented-repairs) below. The tables in the audit
sections describe the original defects.

These tests compare operators with independent checkpoint decoding or arithmetic
expectations. They do not use M8 as the reference. M1/M8 agreement can therefore
coexist with the defects below: both paths can implement the same approximation.

| Finding | Native evidence | Classification |
| --- | --- | --- |
| Folded MXFP4 scaling changes stored weights | 298 coefficient differences in four layer-15 attention gate rows, reproduced by one-hot M1 inputs; 146 originally nonzero coefficients become zero | Confirmed extra weight approximation in the active GEMM |
| GDN softplus cancels small gates to zero | Native persistent state remains 1.0 after 2,048 isolated transitions; a stable-softplus candidate gives 0.9957275391 and a different BF16 output | Confirmed numerical-stability defect in packed M1 recurrence |
| Per-block fallback reads the wrong layout under WPERM | Checkpoint-order control: 24/24 values match; loader-supplied fragment order: 3/24 match | Confirmed conditional layout defect; this fallback is disabled in the live profile |
| GDN normalization rounds differently from plain HF | 3,255,209/9,437,184 BF16 elements differ; current native relative L2 error against FP64 is 0.1661%, versus 0.2870% for the HF rounding schedule | Different finite-precision contract; do not automatically add HF's rounding losses |

All counts concern synthetic operator inputs with actual public checkpoint
weights. They are not output-token disagreement rates, observed chat frequency,
or evidence that these defects caused the reported thinking loops.

## Folded weight scaling

The initial CPU scan examined all 496 two-dimensional U8 scale tensors in the
checkpoint. Layer 15's `self_attn.q_proj` alone contained reference/block exponent
gaps above eight: 36 blocks with gap nine. Their 1,152 coefficients contain 298
changed values, in rows 9544, 10568, 10711 and 12048. All four are output-gate
rows in the checkpoint's per-head `[query, gate]` layout.

For packed code `c`, stored scale byte `s`, and row reference exponent `r`:

```text
checkpoint value = E2M1(c) × 2^(s − 127)
folded value     = E4M3(kMag[r − s][c]) × 2^(r − 127)
```

The folding table is exact for gaps 0–8, but FP8 cannot represent every required
coefficient at gap nine. Row 9544, column 170 is a direct counterexample:
the checkpoint coefficient is **0.000030517578125**; the native kernel gives
**0** for its one-hot projection. This does not depend on summation order or
choosing a competing model implementation.

The probe calls the actual release GEMM binary with the deployed TP1 fused-QKV
shape `M=1, N=14336, K=5120`, its permuted weights and its FP8 activation format.
Each input is a separate M1 call, not a batched M320 computation.

| Input/control | Comparison | Different / compared |
| --- | --- | ---: |
| 800 one-hot inputs covering affected block columns | Four affected rows, against direct checkpoint decoding | 298 / 3,200 |
| Same inputs | Two unaffected control rows | 0 / 1,600 |
| 320 seeded synthetic normal inputs | Four affected projection rows, against FP64 dot products rounded to BF16 | 496 / 1,280 |
| Same inputs | Two unaffected control rows | 0 / 640 |
| Same inputs after BF16 sigmoid | Four affected gate values | 20 / 1,280 |

The one-hot experiment isolates coefficient loss exactly. The normal-input test
shows that discrepancies can survive the gate activation; it does not assign
every difference between native FP32 and FP64 dot products exclusively to folding.
This lookup predates the recent GEMM dispatch backport.

Repair requirement: preserve exceptional block scales instead of silently
rounding them through the folded FP8 representation. A selected-row/block repair
may avoid penalizing every projection, but needs independent correctness and
performance checks.

The existing per-block route cannot simply be enabled. In
[`radiance_mxfp4.py`](../radiance_mxfp4.py), the loader still permutes weights when
`WPERM` is enabled and `PERBLOCK_NK` selects a layer. The `wref=0` branch in
[`radiance_mxfp4_fp8.hip`](../radiance_mxfp4_fp8.hip) expects checkpoint order.
Using four one-hot input columns and six selected output rows, checkpoint order
matched all 24 reference values; fragment order differed in 21, with maximum
absolute error 0.0234375. This is a separate, disabled-path defect that a repair
must avoid.

## GDN gate cancellation

Packed M1 decode evaluates `log(1 + exp(x))` in FP32. For sufficiently negative
`x`, adding `exp(x)` to one loses the increment, turning a nonzero softplus into
zero. Multiplication by `exp(A_log)` can make that lost increment consequential
for the recurrent decay. The aligned
[`stock_gdn_scan_kernel.py`](../experiments/radiance-public/stock_gdn_scan_kernel.py)
contains the same expression; repairing only M1 would break the shared contract.

The native witness uses real layer-12/head-29 parameters (`A_log=4.9375`,
`dt_bias=-3.078125`) and legal synthetic BF16 `a=-14.9375`, giving
`x=-18.015625`. Zero keys/values isolate state retention from new state writes.

| Measurement | Installed kernel | Isolated `log1p(exp(x))` candidate |
| --- | ---: | ---: |
| Head-29 decay | 1.0 | 0.9999979138374329 |
| State after 2,048 transitions | 1.0 | 0.9957275390625 |
| BF16 output after 2,048 transitions | 0.08837890625 | 0.087890625 |
| Different decay factors versus FP64-derived FP32, across 48 actual parameter pairs at approximately x=−18 | 8 / 48 | 1 / 48 |
| Maximum absolute decay error in that comparison | 0.0000020861625671 | 0.0000000596046448 |

The ideal FP64 state after those transitions is 0.9957278834341519. The experiment
also swept 13 gate values from −24 to +24 across all 48 parameter pairs. Stable
softplus removes the demonstrated cancellation, but does not establish exact
FP64-rounded equivalence: exponentials, products and repeated FP32 updates still
round. Some very small decays remain one FP32 ULP from the oracle, including
values rounded to one near x=−16/−17.

The candidate was compiled into a separate diagnostic process only. A production
repair must apply a consistent stable gate to the admitted M1/M8/prefill paths,
then recheck their state/output agreement and performance. Natural gate frequency
and full-model quality effects remain unmeasured.

## Normalization: reference choice

Plain Hugging Face rounds the normalized activation and weighted result to BF16
before applying the FP32 SiLU gate. Native FLA retains FP32 intermediates until
the final BF16 output. The native test covers 32 complete `[48,128]` M1 inputs
for each of 48 GDN layers, using their actual norm weights.

Against the FP64 real-function approximation on those inputs, the current native
path had lower error. Matching HF's intermediate casts would satisfy a particular
HF arithmetic contract but would not automatically improve mathematical accuracy
or answers. The contract must specify those casts before calling either path the
reference. This audit leaves the current normalization unchanged.

## Reproduction and evidence

The source and identities are in
[`probe_eager_m1_independent.py`](../experiments/radiance-public/probe_eager_m1_independent.py)
and [the public fixture manifest](../benchmarks/fixtures/eager-m1-independent-20260924/).
[Native results](../benchmarks/results/eager-m1-independent-20260924.json)
retain measured values, individual norm-layer results, the complete gate sweep,
and source/binary/input hashes. The original CPU scan and native generated-code
artifacts are retained in the private lab's audit directory; they contain public
weights and synthetic inputs, not conversations.

Run inside the matching pinned ROCm environment, with the public checkpoint and
the release GEMM build available:

```bash
QWEN_CONFORMANCE_GPU_LOCK=/tmp/qwen-conformance-gpu.lock \
  python experiments/radiance-public/probe_eager_m1_independent.py \
  --allow-gpu --model /models/Qwen3.8-27B-Uncensored-MXFP4-awq \
  --gemm-build /qualification/preflight/mxfp4-dispatch-v1/build-v2 \
  --audit benchmarks/fixtures/eager-m1-independent-20260924 \
  --output /tmp/eager-m1-independent-new
```

The output directory must be new. `--stages softplus gemm` selects a subset.
The probe requires an idle API, at least 512 MiB free VRAM and a cooperative
diagnostic lease; its allocator budget is 192 MiB. The idle check does not reserve
the production scheduler. It stops if requests appear at a check boundary.
`torch.cuda` is PyTorch's API name on ROCm; the GPU execution is HIP.

The CPU tests validate the comparator, exact MXFP4 decoding, input identity and
admission guards. Native completion means these specified operator experiments
completed. It is not full-model qualification, a universal proof, a performance
measurement or approval to deploy the experimental softplus change.

## Implemented repairs

The corrected candidate keeps the GGZ14 GEMM dispatch, fused normalization,
FP8 activations, Global-256 head and existing nine-slot state layout. It changes
three things:

- All admitted GDN gate paths use FP32 `log1p(exp(x))`, including ordinary M1,
  the M8 scan and prefill. This prevents the intermediate `1 + exp(x)` from
  cancelling a small, nonzero gate. Normalization is unchanged.
- Folded MXFP4 GEMM uses E4M3's available upper range. A row reference is
  `max(1, min(max_scale, min_scale + 8))`. The native lookup handles exponent
  differences −6 through +8 exactly: its extremes are `6 × 2^6 = 384` and
  `0.5 × 2^-8 = 2^-9`. Rows with scale spans up to 14 fit without an additional
  weight approximation. Only four reference exponents change in this checkpoint;
  no extra GEMM launch or per-token synchronization is introduced.
- Wider spans select the per-block path. That path now handles fragment-ordered
  weights, tiled activations and E8M0 exponent zero. The prequantized wrapper
  passes a null reference for this fallback; the fused prefill consumer also
  checks the reference-array size before taking its fast path.

The finite-format tests exhaust all signed E2M1 codes over the admitted exponent
window against independently decoded E4M3 bytes. This establishes the lookup
property, not universal correctness of the complete GEMM or model.

| Native regression | Corrected result |
| --- | --- |
| One-hot coefficient oracle, 800 inputs × 6 selected output rows | 0/4,800 differences; original kernel: 298/4,800 |
| Independent dot-product oracle, 320 synthetic inputs × 6 output rows | 0/1,920 differences; original kernel: 496/1,920 |
| BF16 sigmoid gates on the four affected rows, same 320 inputs | 0/1,280 differences; original kernel: 20/1,280 |
| GEMM M8, M32, M65 and tiled prefill versus corrected M1 | Each 0/4,587,520 output-value differences |
| Fragment-layout fallback: M1, M8, M65, M128 and tiled input | Each 0/6,144 differences from independent coefficient decoding |
| Native expanded fold window: spans 0, 8, 9 and 14 | Each 0/6,144 differences |
| GDN M1 versus M8, every output and persistent state after each transition | 320/320 matching transitions; 251,658,240 recurrent-state elements compared |
| GDN prefill versus serial corrected M1 | 28/28 length/tile cases exact, through 2,048 inputs |
| Full production prefill shapes and fused consumer | 12/12 cases exact; 18,784 rows and 288,522,240 output elements; canaries and injected-error checks passed |
| Known cancellation witness | Old decay: 1.0; corrected and independently calculated decay: 0.9999979138374329 |

GDN prefill lengths are 1, 8, 64, 320, 1,000, 1,648 and 2,048, with spatial
tiles 8, 16, 32 and the selected tile. The GEMM prefill cases use 1,000, 1,648
and 2,048 inputs across all four admitted production shapes. These are operator
rows, not generated chat tokens. Negative controls reproduce the original
defects and detect deliberately changed output bits.

| Isolated operation | Before | Corrected |
| --- | ---: | ---: |
| Fused attention QKV GEMM, M1 | 61.50–61.57 µs | 61.66 µs |
| Same GEMM, M8 | 62.18–62.19 µs | 62.37 µs |
| GDN recurrence, M1 | 7.50 µs | 7.79 µs |
| GDN scan, M8 | 32.46 µs | 31.40 µs |

Each figure is the median of seven samples, each containing 512 invocations in
HIP graphs. Events surround replay batches; no instrumentation is inserted into
the kernels. GEMM controls bracket the candidate. The M8 GDN samples fluctuate
enough that their difference does not establish a speedup. These measurements
show small operator costs and retention of the optimized paths, not an updated
whole-round latency or tokens-per-second measurement.

The [repair results](../benchmarks/results/eager-m1-repairs-20260924.json) and
[prefill results](../benchmarks/results/eager-m1-prefill-repairs-20260924.json)
record the tested source, binary and probe identities. A separate container,
without GPU devices, also passed the release bootstrap.

The corrected build was deployed on 24 September 2026. The running worker's
startup receipt identifies the corrected GEMM binary. A public synthetic coding
request produced 585 tokens, valid Python syntax and a natural stop; a request
through the Whonix relay produced a correctly parsed tool call. The
[deployment checks](../benchmarks/results/eager-m1-deployment-20260924.json)
record those results and the release identities. The arithmetic change uses a
new snapshot data namespace; previous snapshots remain available for rollback,
and resumed chats rebuild their cache once. Relaunch Pi to obtain the new cache
identity.

This operator audit did not establish full-model or chat-quality results.
The subsequently deployed release passes the fresh
[320-token M1/M8 and eager/compiled confirmations](CURRENT_CONFIRMATIONS.md).
Those checks do not measure chat quality, and historical 10K results must not
be relabelled as qualification of the new arithmetic.

### Build and prepare the candidate

In the pinned ROCm environment:

```bash
python experiments/radiance-public/mxfp4_fold_precision.py \
  --source . --output /tmp/m1-fold-build
QWEN_CONFORMANCE_GPU_LOCK=/tmp/qwen-conformance-gpu.lock \
  python experiments/radiance-public/probe_m1_arithmetic_repairs.py \
  --allow-gpu --model /models/Qwen3.8-27B-Uncensored-MXFP4-awq \
  --build /tmp/m1-fold-build \
  --audit benchmarks/fixtures/eager-m1-independent-20260924 \
  --output /tmp/m1-operator-check
flock -n /tmp/qwen-conformance-gpu.lock \
  python experiments/radiance-public/probe_prefill_activation_tiles.py \
  --build /tmp/m1-fold-build \
  --model /models/Qwen3.8-27B-Uncensored-MXFP4-awq \
  --api http://127.0.0.1:8080 --memory-mib 384 \
  --output /tmp/m1-prefill-check.json
python experiments/radiance-public/prepare_m1_arithmetic_release.py \
  --parent /path/to/frozen-serving-payload \
  --build /tmp/m1-fold-build \
  --qualification /tmp/m1-operator-check/result.json \
  --activation /tmp/m1-prefill-check.json \
  --profile experiments/radiance-public/runtime-radiance-1.0.16.json \
  --output /tmp/m1-candidate-release
```

The preparation command copies only manifest-listed artifacts, binds the new
tests, replaces every frozen scan copy and emits a matching runtime profile.
It does not restart or publish anything. On deployment, use that matching
profile and manifest, then regenerate snapshot bindings and package the runtime
as described in [BUILDING.md](BUILDING.md). The corrected gate and weights change
cached state, so the new arithmetic identity requires rebuilding each resumed
chat's cache once. Old snapshots remain separately identifiable for rollback.
Both the wrapper and binary are installed before weight loading; later dispatch
setup preserves that binary instead of replacing it with the parent fold table.

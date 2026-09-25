# RX5-safe MXFP4 and FP8-KV continuation

## Decision

This continuation starts from merged `main` commit
`6f7814d` and selectively integrates Brian's later RX5 work from
[`ggz14/radiance-vllm-mxfp4`](https://codeberg.org/ggz14/radiance-vllm-mxfp4).
Original authorship is retained on the imported commits. The public launcher
and model-specific defaults were not imported; this fork keeps its portable
Compose, model-neutral benchmark contract, and existing Radiance fallbacks.

The production-safe selection is deliberately narrower than full RX5:

- enable `RADIANCE_MXFP4_WPERM=1` and
  `RADIANCE_MXFP4_DECODE_NT=1` together for Quark MXFP4 W4A8;
- retain the corrected multi-sequence GDN routing;
- keep `RADIANCE_MXFP4_A_TILED_MIN_M=0`,
  `RADIANCE_GDN_NORM_QUANT=0`, `RADIANCE_NORMQUANT_FUSION=0`, and
  `RADIANCE_FP8_STREAM=0`;
- keep FP8-KV calibration sidecars and persisted TunableOp explicitly opt-in.

The disabled code is retained for future isolation. “RX5 available” must not
be interpreted as “full RX5 production-qualified.”

## FP8-KV calibration and TunableOp

The new calibration workflow observes BF16 attention inputs using public vLLM
worker-extension and PyTorch hook APIs, reduces maxima across TP ranks, and
writes immutable safetensors sidecars bound to the exact source checkpoint.
The completed TP2 calibration used eight representative prompts (tool use,
code, long-context retrieval, structured data, reasoning, multilingual, math,
and instruction boundaries), ten observations per full-attention layer, and
produced 64 scalar scale tensors across 16 layers. Artifact integrity and
source-model verification passed.

Calibration remains a fidelity experiment, not a throughput feature. It is
not enabled by default until a disjoint held-out suite demonstrates better or
unchanged quality and healthy DFlash acceptance.

Persisted TunableOp support implements a safe `collect` → offline `tune` →
checksum-verified `serve` lifecycle. Collection cannot tune live traffic, and
serve mode refuses missing or modified result sets. Plumbing and tamper checks
pass; an actual TunableOp result set has not yet shown a qualified serving
gain, so the shipped mode remains `off`.

## Safe-subset performance

The matched quick comparison used the AMD Quark MXFP4 target, matched
tcclaviger DFlash2-FP8 drafter, K7, TP2, FP8 KV, PIECEWISE graphs, an 8K/C8
envelope, and identical deterministic prompts. Only WPERM and decode-NT changed.

| Concurrency | Control output TPS | Safe subset output TPS | Delta |
|---:|---:|---:|---:|
| c1 | 182.3 | 193.4 | +6.1% |
| c2 | 216.8 | 224.7 | +3.6% |
| c4 | 330.5 | 352.6 | +6.7% |
| c8 | 522.0 | 553.1 | +6.0% |

The post-reboot repeat measured 195.2/231.1/355.3/556.4 TPS at
c1/c2/c4/c8 with zero request failures. The original and repeated sampled
tool gates passed 100/100 and 30/30 respectively.

Publication-grade BetterBench v0.2.2 then ran ten passes per category. It
measured 183.1 weighted single-stream TPS, 137.6 TPS ITL 1%-low, 64 ms TTFT
p50, and 163.0/286.1/462.0/523.5 aggregate TPS at c1/c2/c4/c8. Every
concurrency arm completed 24/24 requests and the attached tool gate passed
100/100. Cold 2K/4K/7K prefill measured
4,031.8/4,444.7/4,268.6 prompt TPS.

Relative to the previous RX4-dark publication, the safe candidate improved
weighted single-stream decode by 5.2%, ITL 1%-low by 3.5%, and aggregate
c1/c2/c4/c8 by 5.0%/4.3%/15.7%/6.1%. Prefill was essentially neutral:
+1.3%/-2.3%/-1.0% at 2K/4K/7K. Category results remain mixed—prose and
reasoning medians did not improve—so the raw per-category report remains the
source of truth.

Exact runs:

- isolation matrix: [benchmarks/results/20260908T1645Z_fp8kv-rx5/](https://github.com/Terrydaktal/vllm-coherence/tree/4e604dcccb311ca709858a4bd57a11fc636e7f43/benchmarks/results/20260908T1645Z_fp8kv-rx5);
- post-reboot quick gate:
  [benchmarks/results/20260908T1905Z_safe-rx5-final/safe-wperm-nt-postreboot/](https://github.com/Terrydaktal/vllm-coherence/tree/4e604dcccb311ca709858a4bd57a11fc636e7f43/benchmarks/results/20260908T1905Z_safe-rx5-final/safe-wperm-nt-postreboot);
- publication BetterBench:
  [benchmarks/results/20260908T1905Z_safe-rx5-final/safe-wperm-nt-betterbench-standard/](https://github.com/Terrydaktal/vllm-coherence/tree/4e604dcccb311ca709858a4bd57a11fc636e7f43/benchmarks/results/20260908T1905Z_safe-rx5-final/safe-wperm-nt-betterbench-standard).

## Why full RX5 remains experimental

The full bundle combined WPERM/decode-NT with the alternative norm-quant,
FP8-stream, tiled-prefill, and GDN paths. Its throughput signal was attractive,
but DFlash K7 exposed a reproducible structured-tool stopping failure:

| Configuration | Tool gate |
|---|---:|
| WPERM + decode-NT + DFlash K7 | 100/100 |
| Full numerical profile without DFlash | 100/100 |
| Full numerical profile + DFlash K7 | 93/100 |
| Full profile + stop-after-first experiment | 98/100 |

Failures generally contained repeated schema-valid tool blocks or whitespace
until the token limit, rather than the earlier missing-required-property bug.
Because neither parser constraints nor whitespace controls reached 100/100,
the stop-after-first experiment was removed from the deliverable. K5
qualification was interrupted before model startup by an unrelated host-level
GPU runtime-resume lockup and provides no result. Full RX5 therefore remains
off and will be revisited independently rather than delaying the safe gains.

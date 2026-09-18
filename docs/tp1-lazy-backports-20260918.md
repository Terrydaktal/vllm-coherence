# TP1 FP8 and lazy GDN backports — 18 September 2026

**Selected configuration: keep the existing nine-slot GDN cache layout.**
The prepared bundle retains GEMM dispatch and FP8 norm/quant fusion. Lazy
snapshots are disabled; their experimental implementation and evidence remain
available for separate qualification.

Adapted the TP1 FP8 producer/consumer fusion and lazy GDN snapshots from
[GGZ14 commit 3f542b7](https://github.com/GGZ14/vllm-mxfp4/commit/3f542b7cbfce3fa0d01dc55665af4c77a7093ce8)
to the corrected Radiance arithmetic. The earlier GEMM dispatch backport remains
in the parent runtime. Recurrent state stays FP32; convolution, gate rounding,
reductions, normalization and RoPE retain the qualified numerical contract.

## Measured stage timings

GPU-event medians with 100 operators inside each captured graph, five measured
replays after warm-up. These are isolated M8 operator times, not complete model
rounds. The FP8 comparison uses the same opaque corrected HIP norm as compiled
M8, followed by native vLLM quantization. The SiLU reference is compiled with
precision-cast emulation enabled.

| Operator | Previous | Candidate | Decision |
| --- | ---: | ---: | --- |
| Residual RMSNorm + FP8 quantization | 0.010345 ms | 0.010105 ms | Enable fused producer |
| SiLU/multiply + FP8 quantization | 0.008847 ms | 0.011637 ms | Retain previous compiled producer |
| GDN recurrence, 1 previous accepted position | 0.023624 ms | 0.018054 ms | Lazy state |
| GDN recurrence, 4 previous accepted positions | 0.021654 ms | 0.018957 ms | Lazy state |
| GDN recurrence, all 8 previous positions accepted | 0.021435 ms | 0.022195 ms | Lazy state; slightly slower in this case |

At four previously accepted positions, extrapolating over 48 GDN layers and
128 decoder norms gives **0.160 ms saved per target verification pass**. Adding
the [GEMM dispatch estimate](gemm-dispatch-backport-20260918.md) of 16.802 ms gives
**16.962 ms, approximately 17 ms**, across all three backports. This sum is an
estimate from separate stage measurements; whole-model elapsed time and tokens
per second have not been remeasured.

With lazy snapshots disabled, the selected configuration's stage estimate is
**16.833 ms** (16.802 ms GEMM + 0.031 ms norm/quant). The 0.129 ms lazy recurrence
saving is excluded.

Lazy GDN reduces the active recurrent-state window from nine slots to three.
It stores one FP32 base state and the speculative inputs, reconstructing only
the accepted prefix before advancing. This saves state slots within the cache
pool; it does not automatically shrink a fixed-size VRAM pool.

## Short correctness checks

| Check | Result |
| --- | --- |
| FP8 producer: bytes, scales and BF16 residual | 1,892 exact checks; 19,118,440 bytes; zero differences |
| Lazy GDN: outputs and all retained prefix states | 537 exact checks; 1,508,939,784 bytes; zero differences |
| CPU admission, default-layout preservation, graph warm-up isolation and existing dispatch checks | 51 passed |
| Selected existing-layout compiled runtime | Two public synthetic requests, 64 tokens each; lazy GDN absent; FP8 active; both output hashes match the earlier smoke; clean shutdown |
| Experimental lazy-layout compiled runtime | Two public synthetic requests, 64 generated tokens each; both paths exercised; clean shutdown |

The FP8 probe uses 320 deterministic synthetic rows plus zero, tiny and
adversarial BF16 values, M1/M8 shapes, both residual modes, poisoned graph outputs
and deliberate corruption. The lazy probe includes 320 sequential rows, all 64
current-width/previous-acceptance pairs, 24 checkpoint/migration cases,
allocation guards, canonical state serialization/restore into different slots,
and deliberate invalid-stash and bit-flip controls. No private chats were read.

The frozen corrected GDN scan and norm implementations are independent oracles
for these backports. These checks establish sampled operator agreement, not a
new 10K-token model comparison or mathematical proof over arbitrary inputs.

## Runtime and cache handling

`prepare_tp1_lazy_backports.py` builds a separate runtime bundle from the
qualified GEMM bundle. By default it preserves the existing state layout,
creates no vLLM overlay, and explicitly sets both lazy-state environment flags
to zero. Runtime admission rejects inherited flags that disagree with the
bundle. Source, binary and successful test evidence are checked before installation.

The selected bundle is
`/qualification/preflight/tp1-lazy-backport-v1/bundle-existing-v1/` in the
qualification container. `launch.json` supplies the matching import paths and
environment; `performance.json` binds the checked artifacts.
The selected bundle passed compiled execution with piecewise graphs and both
lazy flags disabled. This is a short integration check, not a full performance
measurement or a new 10K correctness run.

The earlier experimental lazy bundle remains at `bundle-v5/`. Rebuilding that
variant requires explicit `--enable-lazy-gdn`, its `--lazy` evidence and pinned
`--vllm` source. Only that variant creates an overlay for three pinned vLLM
source files. The installed package remains intact.

`stock_fp8_stream.py` changes decoder norms feeding tuple-aware Radiance
linears. Residuals, inter-layer hidden states, DFlash taps and the final model
norm keep their BF16 interface. Native FP8 division and rounding are preserved;
the approximate traced quantizer is not admitted. The slower fused SiLU path
remains disabled.

`stock_gdn_lazy_runtime.py` and `stock_gdn_lazy_kernel.py` implement the FP32
lazy state window and accepted-prefix reconstruction. Alignment hints preserve
the corrected Triton reduction order during pointer-table cache migrations.
Unsupported metadata fails explicitly instead of invoking an old kernel that
would write beyond the smaller state window.

The experimental lazy cache ABI is `stock-fp32-lazy-v1`. The canonical state roundtrip is
checked, but the production Pi disk offloader is **not yet qualified** for this
representation. The experimental worker therefore refuses that combination.
The selected existing-layout bundle makes no such state-layout change. No
production backend has been restarted or upgraded by this backport work.

Aggregate evidence: [tp1-lazy-backports-20260918.json](tp1-lazy-backports-20260918.json).
Full local operator receipts: `artifacts/conformance/20260918-tp1-lazy/`.

# Arithmetic preserved by the repaired M1 path

[The executable profile](../configs/profiles/m1-arithmetic-contract-v2.json)
separates accepted quantization from backend arithmetic. It selects the existing
MXFP4 checkpoint, per-token OCP FP8 activations, and OCP FP8 KV storage with its
recorded descales. Attention remains full causal attention. Consequently this
is a **weight + activation + KV quantization** target; equality to a model with
only quantized weights is not claimed. Global-512 is a separately identified
approximate output policy, not an exact part of the reference head.

Every qualification result must bind the actual checkpoint coefficients,
scales, runtime configuration, source and binary. The contract profile is a
declaration, not a certificate that the complete backend implements it.

## Rounding boundaries

| Operation | Required boundary |
| --- | --- |
| Residual addition | Add in FP32; store a BF16 residual carry, but normalize the unrounded FP32 sum |
| Hidden RMS normalization | Deterministic M1 reduction; FP32 arithmetic and a final BF16 RNE result |
| Dynamic activation FP8 | Quantize that BF16 result; store per-token FP32 scale and OCP E4M3FN codes |
| MXFP4 projection | Reconstruct every stored coefficient; FP32 accumulation; final BF16 RNE |
| GDN gated normalization | FP32 norm, weight and SiLU gate calculation; one final BF16 rounding before FP8 |
| GDN recurrence | FP32 state and normalized Q/K; explicit BF16 beta boundary; stable softplus; BF16 output |
| Attention | Preserve BF16 Q/K range; FP32 scores/softmax and split state; one final BF16 output |
| MLP | BF16 SiLU output, then BF16 gate/up product, then FP8 |
| Final normalization | BF16; the diagnostic final-weight FP8 control is not a serving operation |

Here RNE means round to nearest, ties to even. OCP FP8 and FNUZ are different
formats. Neither an extra FP16 narrowing nor an omitted BF16 boundary can be
justified by saying that the model is already quantized.

## The normalization decision

Hidden normalization uses the same 512-lane arithmetic for every logical row.
Its four component sums, inter-wave reduction and intra-wave reduction are
specified in the profile and independently evaluated by
[`hidden_norm`](../experiments/radiance-public/m1_arithmetic_contract.py).
The old 64/32-lane prefill choices are retained only as diagnostic controls.
Parallel execution of rows remains allowed; changing their numerical reduction
based on the number of rows does not.

GDN gated normalization keeps the repaired native M1 sequence:

\[
u=\operatorname{FP32}(\operatorname{FP32}(x\,r)w),\qquad
g=\operatorname{FP32}(z\,\sigma(z)),\qquad
y=\operatorname{BF16}_{\rm RNE}(\operatorname{FP32}(u\,g)).
\]

It does not insert Hugging Face's extra intermediate BF16 casts. That is a
deliberate reference choice: the previous independent audit found lower error
against FP64 for the native sequence on its sample. It does not prove the native
sequence more accurate for every input. The per-head reduction layout must also
be preserved between M1 and prefill; a fused quantizer cannot hide that decision.

## Exact checks and numerical checks

Byte equality is required for discrete representations, residual/history
preservation and the declared batch-partition invariants. Independent FP64
projection, softmax and recurrence calculations measure arithmetic error.
Their diagnostic tolerances are **not proven forward-error bounds** and do not
turn a differing result into an exact one.

The profile records where compiler intrinsics or a reduction topology still
need an implementation binding. CPU `1/sqrt` versus the pinned GPU reciprocal
square root, compiler contraction, denormals, overflow and arbitrary-input
behavior are not silently assumed equal. Unsupported/nonfinite cases remain
qualification failures, not passes. This does not claim every production
wrapper already rejects them before a GPU launch.

Any adopted arithmetic repair needs a new snapshot identity and fresh relevant
qualification. The earlier 10K full-model table cannot qualify these changes.

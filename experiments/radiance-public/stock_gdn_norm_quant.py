"""Experimental GDN gated norm + native-contract per-token FP8 quantization.

Keep FLA's per-head arithmetic and the BF16 rounding before quantization.
Neither this module nor its installer is admitted to production without evidence.
"""

import functools

from triton.experimental import gluon
from triton.experimental.gluon import language as tl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout


@gluon.jit
def gdn_norm_quant_kernel(
    x_ptr,
    z_ptr,
    w_ptr,
    q_ptr,
    s_ptr,
    sx: tl.constexpr,
    sz: tl.constexpr,
    eps,
    prefill: tl.constexpr,
    warps: tl.constexpr,
):
    row = tl.program_id(0)
    # Native M1 uses four adjacent elements per lane, 32 lanes per head.
    # The alternate historical prefill layout is retained for diagnostic
    # controls. The serving wrapper preserves the M1 layout at every row count.
    if prefill:
        layout: tl.constexpr = BlockedLayout([1, 8], [2, 16], [warps, 1], [1, 0])
    else:
        layout: tl.constexpr = BlockedLayout([1, 4], [1, 32], [warps, 1], [1, 0])
    heads = tl.arange(0, 64, layout=SliceLayout(1, layout))
    cols = tl.arange(0, 128, layout=SliceLayout(0, layout))
    off = heads[:, None] * 128 + cols[None, :]
    x = tl.load(x_ptr + row * sx + off, heads[:, None] < 48, 0).to(tl.float32)
    z = tl.load(z_ptr + row * sz + off, heads[:, None] < 48, 0).to(tl.float32)
    w = tl.load(w_ptr + cols).to(tl.float32)
    var = tl.sum(x * x, 1) / 128
    rstd = tl.rsqrt(var + eps)
    y = (x * rstd[:, None]) * w[None, :]
    y *= z * (1 / (1 + tl.exp(-z)))
    # This store-rounding is part of the reference, including compiled mode.
    y = y.to(tl.bfloat16).to(tl.float32)
    maximum = tl.max(tl.max(tl.where(heads[:, None] < 48, tl.abs(y), 0.0), 1), 0)
    scale = tl.maximum(maximum / 448.0, 1.0 / (448.0 * 512.0))
    value = tl.minimum(tl.maximum(tl.div_rn(y, scale), -448.0), 448.0)
    tl.store(q_ptr + row * 6144 + off, value, heads[:, None] < 48)
    tl.store(s_ptr + row, scale)


def fused(x, z, weight, eps):
    import torch

    if (
        x.ndim != 3
        or x.shape[1:] != (48, 128)
        or z.shape != x.shape
        or x.dtype != torch.bfloat16
        or z.dtype != x.dtype
        or weight.shape != (128,)
        or weight.dtype not in (torch.bfloat16, torch.float32)
        or any(t.device != x.device for t in (z, weight))
        or x.device.type != "cuda"
        or x.stride(1) != 128
        or z.stride(1) != 128
        or any(t.stride(-1) != 1 for t in (x, z, weight))
        or not 1 <= x.shape[0] <= 2048
    ):
        raise ValueError("GDN norm/quant outside qualified TP1 shape/precision")
    q = torch.empty((x.shape[0], 6144), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device)
    warps = 16 if x.shape[0] <= 8 else 8
    gdn_norm_quant_kernel[(x.shape[0],)](
        x,
        z,
        weight,
        q,
        scale,
        x.stride(0),
        z.stride(0),
        eps,
        prefill=False,
        warps=warps,
        num_warps=warps,
    )
    return q, scale


def install(model, hooks):
    import torch

    selected = [m for m in model.modules() if type(m).__name__ == "QwenGatedDeltaNetAttention"]
    if len(selected) != 48:
        raise ValueError("GDN norm/quant requires the pinned 48-site inventory")
    for layer in selected:
        if (
            layer.tp_size != 1
            or layer.norm.activation != "silu"
            or not layer.norm.norm_before_gate
            or layer.norm.bias is not None
            or getattr(layer.out_proj, "radiance_wref", None) is None
        ):
            raise ValueError("GDN norm/quant consumer or arithmetic differs")
    calls = {"sites": len(selected), "decode": 0, "prefill": 0}

    @torch.library.custom_op("qwen_stock_gdn_fp8::norm_quant", mutates_args=())
    def norm_quant(
        x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls["decode" if x.shape[0] <= 8 else "prefill"] += 1
        return fused(x, z, weight, eps)

    @norm_quant.register_fake
    def _(x, z, weight, eps):
        return (
            torch.empty((x.shape[0], 6144), dtype=torch.float8_e4m3fn, device=x.device),
            torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device),
        )

    def wrap(layer):
        @functools.wraps(layer._output_projection)
        def output_projection(x, z):
            q, scale = norm_quant(x, z, layer.norm.weight, layer.norm.eps)
            output, _ = layer.out_proj((q, scale))
            return output

        return output_projection

    for layer in selected:
        hooks.replace(layer, "_output_projection", wrap(layer))
    return calls

# SPDX-License-Identifier: Apache-2.0
# Transition derived from stock_gdn_scan_kernel.py and vLLM/FLA packed GDN.
# Original FLA authors: Songlin Yang and Yu Zhang, MIT, 2023-2025.
"""Prepare invariant GDN inputs once, preserving each chronological transition.

Experimental until full output/state and model qualification. FP32 preparation
never converts normalized Q/K or decay/beta to a narrower storage format.
"""

from vllm.third_party.flash_linear_attention.ops.op import exp
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["ROWS"])
def prepare_inputs(
    mixed,
    a,
    b,
    a_log,
    dt_bias,
    qk,
    gates,
    scale,
    stride_mixed: tl.constexpr,
    stride_a: tl.constexpr,
    stride_b: tl.constexpr,
    ROWS,
    TRANSPOSED: tl.constexpr,
):
    row, hv = tl.program_id(0), tl.program_id(1)
    h = hv // 3
    if hv % 3 == 0:
        k = tl.arange(0, 128)
        q = tl.load(mixed + row * stride_mixed + h * 128 + k).to(tl.float32)
        key = tl.load(mixed + row * stride_mixed + 16 * 128 + h * 128 + k).to(
            tl.float32
        )
        q = q / tl.sqrt(tl.sum(q * q) + 1e-6)
        key = key / tl.sqrt(tl.sum(key * key) + 1e-6)
        q = q * scale
        qoffset = (h * ROWS + row) * 128 if TRANSPOSED else row * 4096 + h * 128
        koffset = (
            ((16 + h) * ROWS + row) * 128 if TRANSPOSED else row * 4096 + (16 + h) * 128
        )
        tl.store(qk + qoffset + k, q)
        tl.store(qk + koffset + k, key)
    av = tl.load(a + row * stride_a + hv).to(tl.float32)
    bv = tl.load(b + row * stride_b + hv).to(tl.float32)
    al = tl.load(a_log + hv).to(tl.float32)
    dt = tl.load(dt_bias + hv).to(tl.float32)
    x = av + dt
    softplus = tl.where(x <= 20.0, tl.extra.libdevice.log1p(tl.exp(x)), x)
    g = -tl.exp(al) * softplus
    beta = tl.sigmoid(bv).to(b.dtype.element_ty).to(tl.float32)
    offset = (hv * ROWS + row) * 2 if TRANSPOSED else (row * 48 + hv) * 2
    tl.store(gates + offset, exp(g))
    tl.store(gates + offset + 1, beta)


@triton.jit(do_not_specialize=["T"])
def prepared_scan(
    initial,
    mixed,
    qk,
    gates,
    output,
    states,
    T,
    stride_mixed: tl.constexpr,
    BV: tl.constexpr,
    PIPELINE: tl.constexpr,
    TRANSPOSED: tl.constexpr,
):
    iv, hv = tl.program_id(0), tl.program_id(1)
    h = hv // 3
    k = tl.arange(0, 128)
    v = iv * BV + tl.arange(0, BV)
    offsets = hv * 128 * 128 + v[:, None] * 128 + k[None, :]
    state = tl.load(initial + offsets).to(tl.float32)
    for row in tl.range(0, T, num_stages=PIPELINE):
        qoffset = (h * T + row) * 128 if TRANSPOSED else row * 4096 + h * 128
        koffset = (
            ((16 + h) * T + row) * 128 if TRANSPOSED else row * 4096 + (16 + h) * 128
        )
        q = tl.load(qk + qoffset + k)
        key = tl.load(qk + koffset + k)
        value = tl.load(mixed + row * stride_mixed + 32 * 128 + hv * 128 + v).to(
            tl.float32
        )
        offset = (hv * T + row) * 2 if TRANSPOSED else (row * 48 + hv) * 2
        decay = tl.load(gates + offset)
        beta = tl.load(gates + offset + 1)
        state *= decay
        value -= tl.sum(state * key[None, :], 1)
        value *= beta
        state += value[:, None] * key[None, :]
        result = tl.sum(state * q[None, :], 1)
        p = output + (row * 48 + hv) * 128 + v
        tl.store(p, result.to(p.dtype.element_ty))
    tl.store(states + offsets, state)


@triton.jit
def load_row(mixed, qk, gates, row, hv, k, v, T, stride_mixed: tl.constexpr):
    h = hv // 3
    q = tl.load(qk + row * 4096 + h * 128 + k, row < T, 0)
    key = tl.load(qk + row * 4096 + (16 + h) * 128 + k, row < T, 0)
    value = tl.load(mixed + row * stride_mixed + 4096 + hv * 128 + v, row < T, 0).to(
        tl.float32
    )
    decay = tl.load(gates + (row * 48 + hv) * 2, row < T, 0)
    beta = tl.load(gates + (row * 48 + hv) * 2 + 1, row < T, 0)
    return q, key, value, decay, beta


@triton.jit(do_not_specialize=["T"])
def lookahead_scan(
    initial,
    mixed,
    qk,
    gates,
    output,
    states,
    T,
    stride_mixed: tl.constexpr,
    BV: tl.constexpr,
    UNUSED: tl.constexpr,
):
    iv, hv = tl.program_id(0), tl.program_id(1)
    k = tl.arange(0, 128)
    v = iv * BV + tl.arange(0, BV)
    offsets = hv * 128 * 128 + v[:, None] * 128 + k[None, :]
    state = tl.load(initial + offsets).to(tl.float32)
    q, key, value, decay, beta = load_row(
        mixed, qk, gates, 0, hv, k, v, T, stride_mixed
    )
    for row in range(T):
        nq, nk, nv, nd, nb = load_row(
            mixed, qk, gates, row + 1, hv, k, v, T, stride_mixed
        )
        state *= decay
        value -= tl.sum(state * key[None, :], 1)
        value *= beta
        state += value[:, None] * key[None, :]
        result = tl.sum(state * q[None, :], 1)
        p = output + (row * 48 + hv) * 128 + v
        tl.store(p, result.to(p.dtype.element_ty))
        q, key, value, decay, beta = nq, nk, nv, nd, nb
    tl.store(states + offsets, state)


@triton.jit(do_not_specialize=["T"])
def buffered_scan(
    initial,
    mixed,
    qk,
    gates,
    output,
    states,
    T,
    stride_mixed: tl.constexpr,
    BV: tl.constexpr,
    BATCH: tl.constexpr,
):
    iv, hv = tl.program_id(0), tl.program_id(1)
    h = hv // 3
    k = tl.arange(0, 128)
    v = iv * BV + tl.arange(0, BV)
    offsets = hv * 128 * 128 + v[:, None] * 128 + k[None, :]
    state = tl.load(initial + offsets).to(tl.float32)
    for start in range(tl.cdiv(T, BATCH)):
        rows = start * BATCH + tl.arange(0, BATCH)
        qs = tl.load(
            qk + rows[:, None] * 4096 + h * 128 + k[None, :], rows[:, None] < T, 0
        )
        ks = tl.load(
            qk + rows[:, None] * 4096 + (16 + h) * 128 + k[None, :],
            rows[:, None] < T,
            0,
        )
        vs = tl.load(
            mixed + rows[:, None] * stride_mixed + 4096 + hv * 128 + v[None, :],
            rows[:, None] < T,
            0,
        ).to(tl.float32)
        ds = tl.load(gates + (rows * 48 + hv) * 2, rows < T, 0)
        bs = tl.load(gates + (rows * 48 + hv) * 2 + 1, rows < T, 0)
        for i in tl.static_range(BATCH):
            if start * BATCH + i < T:
                q = tl.gather(qs, tl.full((1, 128), i, tl.int32), 0).reshape((128,))
                key = tl.gather(ks, tl.full((1, 128), i, tl.int32), 0).reshape((128,))
                value = tl.gather(vs, tl.full((1, BV), i, tl.int32), 0).reshape((BV,))
                decay = tl.gather(ds, tl.full((1,), i, tl.int32), 0).reshape(())
                beta = tl.gather(bs, tl.full((1,), i, tl.int32), 0).reshape(())
                state *= decay
                value -= tl.sum(state * key[None, :], 1)
                value *= beta
                state += value[:, None] * key[None, :]
                result = tl.sum(state * q[None, :], 1)
                p = output + ((start * BATCH + i) * 48 + hv) * 128 + v
                tl.store(p, result.to(p.dtype.element_ty))
    tl.store(states + offsets, state)


def run(
    initial,
    mixed,
    a,
    b,
    a_log,
    dt_bias,
    *,
    tile=4,
    warps=1,
    pipeline=2,
    batch=1,
    transposed=False,
):
    import torch

    rows = mixed.shape[0]
    if (
        tile not in (1, 2, 4, 8, 16, 32)
        or not 1 <= rows <= 4096
        or warps not in (1, 2, 4)
        or pipeline not in (1, 2, 3, 4)
        or batch not in (0, 1, 2, 4, 8)
        or (transposed and batch != 1)
    ):
        raise ValueError("unsupported prepared scan geometry")
    for tensor, shape, dtype in (
        (initial, (48, 128, 128), torch.float32),
        (mixed, (rows, 10240), torch.bfloat16),
        (a, (rows, 48), torch.bfloat16),
        (b, (rows, 48), torch.bfloat16),
        (a_log, (48,), torch.float32),
        (dt_bias, (48,), torch.bfloat16),
    ):
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device != mixed.device
            or tensor.stride(-1) != 1
        ):
            raise ValueError("unsupported prepared scan tensor representation")
    if mixed.device.type != "cuda" or not initial.is_contiguous():
        raise ValueError("prepared scan requires contiguous GPU initial state")
    qk = torch.empty((rows, 32, 128), device=mixed.device, dtype=torch.float32)
    gates = torch.empty((rows, 48, 2), device=mixed.device, dtype=torch.float32)
    out = torch.empty((rows, 48, 128), device=mixed.device, dtype=torch.bfloat16)
    state = torch.empty_like(initial)
    prepare_inputs[(rows, 48)](
        mixed,
        a,
        b,
        a_log,
        dt_bias,
        qk,
        gates,
        128**-0.5,
        mixed.stride(0),
        a.stride(0),
        b.stride(0),
        rows,
        transposed,
        num_warps=1,
        enable_fp_fusion=True,
        allow_flush_denorm=False,
    )
    kernel = (
        prepared_scan
        if batch == 1
        else (lookahead_scan if batch == 0 else buffered_scan)
    )
    kernel[(128 // tile, 48)](
        initial,
        mixed,
        qk,
        gates,
        out,
        state,
        rows,
        mixed.stride(0),
        tile,
        pipeline if batch == 1 else batch,
        *([transposed] if batch == 1 else []),
        num_warps=warps,
        num_stages=3,
        enable_fp_fusion=True,
        allow_flush_denorm=False,
    )
    return out, state


def install(hooks):
    """Optimize only final-state prefill; retained speculative rows stay original."""
    from optimized_prefill_scan import PrefillScan
    from stock_gdn_sequence import SequenceResult

    previous = PrefillScan.run
    calls = {"prepared": 0}

    def prepared(self, initial, mixed, a, b, a_log, dt_bias, *, retain_rows=()):
        keep = tuple(retain_rows)
        if keep or mixed.ndim != 2 or not 127 < mixed.shape[0] <= 4096:
            return previous(
                self, initial, mixed, a, b, a_log, dt_bias, retain_rows=keep
            )
        out, state = run(initial, mixed, a, b, a_log, dt_bias)
        calls["prepared"] += 1
        return SequenceResult(out, state, {}, mixed.shape[0])

    hooks.replace(PrefillScan, "run", prepared)
    return calls

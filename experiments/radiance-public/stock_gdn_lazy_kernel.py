# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
# ruff: noqa: N803
"""Deferred GDN snapshots, retaining the corrected FP32 transition.

Adaptation of GGZ14's base-plus-replay design. Each 32-value state partition has
its own stash region, so it cannot overwrite another workgroup's replay input.
This module is an experimental operator until its integration is qualified.
"""

from vllm.third_party.flash_linear_attention.ops.op import exp
from vllm.triton_utils import tl, triton


@triton.jit
def transition(h, q, k, v, a, b, alog, bias, scale):
    q = q / tl.sqrt(tl.sum(q * q) + 1e-6)
    k = k / tl.sqrt(tl.sum(k * k) + 1e-6)
    q = q * scale
    x = a + bias
    softplus = tl.where(x <= 20.0, tl.extra.libdevice.log1p(tl.exp(x)), x)
    g = -tl.exp(alog) * softplus
    beta = tl.sigmoid(b).to(tl.bfloat16).to(tl.float32)
    h *= exp(g)
    v -= tl.sum(h * k[None, :], 1)
    v *= beta
    h += v[:, None] * k[None, :]
    out = tl.sum(h * q[None, :], 1)
    return h, out


@triton.jit
def replay(h, stash, count, alog, bias, scale):
    ok = tl.arange(0, 128)
    ov = tl.arange(0, 32)
    for row in range(count):
        p = stash + 16 + row * 320
        q = tl.load(p + ok).to(tl.float32)
        k = tl.load(p + 128 + ok).to(tl.float32)
        v = tl.load(p + 256 + ov).to(tl.float32)
        a = tl.load(p + 288).to(tl.float32)
        b = tl.load(p + 289).to(tl.float32)
        h, _ = transition(h, q, k, v, a, b, alog, bias, scale)
    return h


@triton.jit
def lazy_update(
    packed,
    a,
    b,
    A_log,
    dt_bias,
    state,
    output,
    indices,
    accepted,
    cu,
    errors,
    scale,
    stride_x: tl.constexpr,
    stride_a: tl.constexpr,
    stride_b: tl.constexpr,
    stride_state: tl.constexpr,
    stride_indices: tl.constexpr,
):
    part, head, seq = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ok = tl.arange(0, 128)
    ov = part * 32 + tl.arange(0, 32)
    offsets = head * 16384 + ov[:, None] * 128 + ok[None, :]
    base = tl.load(indices + seq * stride_indices).to(tl.int64)
    stash_slot = tl.load(indices + seq * stride_indices + 1).to(tl.int64)
    begin, end = tl.load(cu + seq), tl.load(cu + seq + 1)
    previous = tl.load(accepted + seq)
    if end <= begin:
        return
    if base <= 0 or stash_slot <= 0 or base == stash_slot:
        tl.atomic_or(errors, 1)
        tl.device_assert(False, "lazy GDN base/stash alias or null slot")
        return
    stash = (state + stash_slot * stride_state + head * 16384).to(tl.pointer_type(tl.bfloat16))
    stash += part * 8192
    valid = tl.load(stash.to(tl.pointer_type(tl.int32)))
    owner = tl.load(stash.to(tl.pointer_type(tl.int32)) + 1)
    if (
        previous < 1
        or previous > 8
        or (previous > 1 and (previous > valid or owner != base))
        or end - begin > 8
    ):
        tl.atomic_or(errors, 2)
        tl.device_assert(False, "lazy GDN acceptance or stash identity invalid")
        return
    h = tl.load(state + base * stride_state + offsets)
    alog = tl.load(A_log + head).to(tl.float32)
    bias = tl.load(dt_bias + head).to(tl.float32)
    h = replay(h, stash, previous - 1, alog, bias, scale)
    # This workgroup alone owns the stash region being overwritten below.
    for row in range(begin, end):
        p = packed + row * stride_x
        q = tl.load(p + (head // 3) * 128 + ok).to(tl.float32)
        k = tl.load(p + 2048 + (head // 3) * 128 + ok).to(tl.float32)
        v = tl.load(p + 4096 + head * 128 + ov).to(tl.float32)
        av = tl.load(a + row * stride_a + head).to(tl.float32)
        bv = tl.load(b + row * stride_b + head).to(tl.float32)
        if row > begin:
            target = stash + 16 + (row - begin - 1) * 320
            tl.store(target + ok, q)
            tl.store(target + 128 + ok, k)
            tl.store(target + 256 + tl.arange(0, 32), v)
            tl.store(target + 288, av)
            tl.store(target + 289, bv)
        h, out = transition(h, q, k, v, av, bv, alog, bias, scale)
        tl.store(output + (row * 48 + head) * 128 + ov, out)
        if row == begin:
            tl.store(state + base * stride_state + offsets, h)
    tl.store(stash.to(tl.pointer_type(tl.int32)), end - begin)
    tl.store(stash.to(tl.pointer_type(tl.int32)) + 1, base.to(tl.int32))


@triton.jit
def materialize(
    state,
    indices,
    widths,
    destinations,
    A_log,
    dt_bias,
    errors,
    scale,
    stride_state: tl.constexpr,
    stride_indices: tl.constexpr,
):
    part, head, seq = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ok, ov = tl.arange(0, 128), part * 32 + tl.arange(0, 32)
    offsets = head * 16384 + ov[:, None] * 128 + ok[None, :]
    base = tl.load(indices + seq * stride_indices).to(tl.int64)
    stash_slot = tl.load(indices + seq * stride_indices + 1).to(tl.int64)
    destination = tl.load(destinations + seq).to(tl.int64)
    count = tl.load(widths + seq)
    if destination <= 0 or base <= 0 or stash_slot <= 0:
        tl.atomic_or(errors, 4)
        return
    stash = (state + stash_slot * stride_state + head * 16384).to(tl.pointer_type(tl.bfloat16))
    stash += part * 8192
    valid = tl.load(stash.to(tl.pointer_type(tl.int32)))
    owner = tl.load(stash.to(tl.pointer_type(tl.int32)) + 1)
    if count < 1 or count > valid or owner != base:
        tl.atomic_or(errors, 8)
        return
    h = tl.load(state + base * stride_state + offsets)
    alog, bias = tl.load(A_log + head), tl.load(dt_bias + head).to(tl.float32)
    h = replay(h, stash, count - 1, alog, bias, scale)
    tl.debug_barrier()
    tl.store(state + destination * stride_state + offsets, h)


@triton.jit
def invalidate(state, slots, stride_state: tl.constexpr):
    head, seq = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slots + seq).to(tl.int64)
    if slot > 0:
        for part in tl.static_range(4):
            p = (state + slot * stride_state + head * 16384).to(tl.pointer_type(tl.int32))
            tl.store(p + part * 4096, 0)


@triton.jit
def materialize_align(
    state_ptrs,
    slot_strides,
    group_indices,
    alog_ptrs,
    bias_ptrs,
    bt_ptrs,
    mapping,
    state_indices,
    sources,
    offsets,
    accepted,
    computed,
    errors,
    scale,
    bt_stride: tl.constexpr,
    block_size: tl.constexpr,
    MODE: tl.constexpr,
    MAPPED: tl.constexpr,
):
    """All-layer V2 align hook, with one disjoint stash region per state tile.

    MODE 0 may overwrite the old stash with the new running state. Its tile
    reads only the same byte range it later overwrites; no grid barrier is
    required. MODE 1 materializes the aligned checkpoint before V2 resets the
    acceptance count. Neither hook changes the convolution copy algorithm.
    """
    tile, batch, layer = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head, part = tile // 4, tile % 4
    req = tl.load(mapping + batch) if MAPPED else batch
    if req < 0:
        return
    if MODE == 0:
        base_col = tl.load(sources + req)
        dest_col = tl.load(state_indices + req)
        if base_col < 0 or base_col == dest_col:
            return
        extra = tl.load(offsets + req)
    else:
        nacc = tl.load(accepted + req)
        base_col = tl.load(state_indices + req)
        new_count = tl.load(computed + req)
        running = new_count - nacc + 1
        aligned = (new_count // block_size) * block_size
        if aligned < running:
            return
        extra = aligned - running
        dest_col = aligned // block_size - 1
        if dest_col == base_col and extra == 0:
            return
    group = tl.load(group_indices + layer)
    bt = tl.load(bt_ptrs + group).to(tl.pointer_type(tl.int32)) + batch * bt_stride
    base = tl.load(bt + base_col).to(tl.int64)
    stash_slot = tl.load(bt + base_col + 1).to(tl.int64)
    destination = tl.load(bt + dest_col).to(tl.int64)
    if base <= 0 or stash_slot <= 0 or destination <= 0:
        tl.atomic_or(errors, 16)
        tl.device_assert(False, "lazy GDN align copy has a null slot")
        return
    state = tl.multiple_of(tl.load(state_ptrs + layer).to(tl.pointer_type(tl.float32)), 16)
    stride = tl.multiple_of(tl.load(slot_strides + layer), 128)
    alogs = tl.load(alog_ptrs + layer).to(tl.pointer_type(tl.float32))
    biases = tl.load(bias_ptrs + layer).to(tl.pointer_type(tl.bfloat16))
    alog, bias = tl.load(alogs + head), tl.load(biases + head).to(tl.float32)
    stash = (state + stash_slot * stride + head * 16384).to(tl.pointer_type(tl.bfloat16))
    stash += part * 8192
    valid = tl.load(stash.to(tl.pointer_type(tl.int32)))
    owner = tl.load(stash.to(tl.pointer_type(tl.int32)) + 1)
    if extra < 0 or extra > 7 or (extra > 0 and (extra >= valid or owner != base)):
        tl.atomic_or(errors, 32)
        tl.device_assert(False, "lazy GDN align replay has stale state")
        return
    ok, ov = tl.arange(0, 128), part * 32 + tl.arange(0, 32)
    address = head * 16384 + ov[:, None] * 128 + ok[None, :]
    h = tl.load(state + base * stride + address)
    h = replay(h, stash, extra, alog, bias, scale)
    tl.debug_barrier()
    tl.store(state + destination * stride + address, h)
    if MODE == 0:
        new_stash = tl.load(bt + dest_col + 1).to(tl.int64)
        header = state + new_stash * stride + head * 16384 + part * 4096
        tl.store(header.to(tl.pointer_type(tl.int32)), 0)
    elif destination == base:
        # The accompanying V2 postprocess resets accepted to one. Mark the old
        # stash invalid so it cannot be replayed against the advanced base.
        tl.store(stash.to(tl.pointer_type(tl.int32)), 0)

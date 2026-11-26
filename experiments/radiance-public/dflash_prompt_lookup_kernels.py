"""Look up committed history and replace both proposals and their distribution."""

import triton
import triton.language as tl


@triton.jit
def scan(
    history,
    stride,
    lengths,
    mapping,
    sampled,
    best,
    window,
    steps: tl.constexpr,
    max_match: tl.constexpr,
    position_bits: tl.constexpr,
    block: tl.constexpr,
):
    group = tl.program_id(0)
    slot = tl.load(mapping)
    length = tl.load(lengths + slot)
    if tl.load(sampled) <= 0 or group * block + window > length:
        tl.store(best + group, 0)
        return
    row = history + slot.to(tl.int64) * stride
    source = group * block + tl.arange(0, block)
    live = (source > 0) & (source + window <= length) & (source + steps <= length)
    count = tl.full((block,), 0, tl.int32)
    for offset in tl.static_range(max_match):
        valid = live & (source > offset)
        earlier = tl.load(row + source - 1 - offset, mask=valid, other=-1)
        current = tl.load(row + length - 1 - offset, mask=length > offset, other=-2)
        live = valid & (earlier == current)
        count += live.to(tl.int32)
    key = tl.where(count > 0, (count << position_bits) | source, 0)
    tl.store(best + group, tl.max(key, 0))


@triton.jit
def apply(
    history,
    stride,
    lengths,
    mapping,
    sampled,
    best,
    blocks,
    previous,
    stats,
    same_request,
    proposals,
    logits,
    logits_row_stride,
    logits_step_stride,
    cached,
    top_k,
    steps: tl.constexpr,
    enter: tl.constexpr,
    stay: tl.constexpr,
    hot: tl.constexpr,
    position_bits: tl.constexpr,
    block_count: tl.constexpr,
    block_k: tl.constexpr,
):
    indices = tl.arange(0, block_count)
    key = tl.max(tl.load(best + indices, mask=indices < blocks, other=0), 0)
    slot = tl.load(mapping)
    length = tl.load(lengths + slot)
    new_tokens = tl.load(sampled)
    last_lookup = tl.load(previous)
    warm = (same_request != 0) & (last_lookup >= 0) & (last_lookup + new_tokens == length)
    warm &= new_tokens - 1 >= hot
    use = (key >> position_bits) >= tl.where(warm, stay, enter)
    tl.store(previous, tl.where(use, length, -1))
    tl.store(stats, tl.load(stats) + 1)
    tl.store(stats + 2, tl.load(stats + 2) + warm.to(tl.int32))
    if use:
        tl.store(stats + 1, tl.load(stats + 1) + 1)
        source = key & ((1 << position_bits) - 1)
        offsets = tl.arange(0, block_k)
        for step in tl.static_range(steps):
            token = tl.load(history + slot.to(tl.int64) * stride + source + step)
            cache_row = cached + (slot.to(tl.int64) * steps + step) * top_k
            old = tl.load(cache_row + offsets, offsets < top_k, other=0)
            logit_row = (
                logits
                + slot.to(tl.int64) * logits_row_stride
                + step * logits_step_stride
            )
            tl.store(logit_row + old, -float("inf"), (offsets < top_k) & (old != token))
            tl.store(logit_row + token, 0.0)
            # Keep the point mass in the sparse cache's list of entries to
            # clear on the next draft. Existing entries have already been
            # cleared and can stay in the list.
            present = tl.sum(((offsets < top_k) & (old == token)).to(tl.int32), 0)
            tl.store(cache_row, token.to(tl.int64), present == 0)
            tl.store(proposals + step, token)

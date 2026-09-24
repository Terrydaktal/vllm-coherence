"""CPU arithmetic witnesses for the explicit M1 normalization contract.

No GPU implementation is called. Floating-point operations and reduction edges
are specified here; FP64 equations in m1_stage_oracles remain a separate oracle.
"""

import m1_stage_oracles as ref
import numpy as np


def norm_moments(value, epsilon, *, lanes=512):
    """FP32 four-component reduction; lanes=32/64 are historical controls."""
    x = ref.finite(value).astype(np.float32)
    if x.ndim != 2 or x.shape[1] != 5120 or lanes not in (32, 64, 512):
        raise ValueError("normalization requires [rows,5120] and a declared reduction")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("positive finite epsilon required")
    accum = np.zeros((len(x), lanes, 4), dtype=np.float32)
    vectors = x.reshape(len(x), 1280, 4)
    for begin in range(0, 1280, lanes):
        values = vectors[:, begin : begin + lanes]
        accum[:, : values.shape[1]] += np.multiply(values, values, dtype=np.float32)
    sums = ((accum[:, :, 0] + accum[:, :, 1]) + accum[:, :, 2]) + accum[:, :, 3]
    offset = lanes // 2
    while offset >= 32:
        sums[:, :offset] = sums[:, :offset] + sums[:, offset : 2 * offset]
        offset //= 2
    sums = sums[:, :32].copy()
    for offset in (1, 2, 4, 8, 16):
        sums[:, : 32 - offset] = sums[:, : 32 - offset] + sums[:, offset:]
    variance = sums[:, 0:1] * np.float32(1.0 / 5120)
    argument = (variance + np.float32(epsilon)).astype(np.float64)
    inverse = (1.0 / np.sqrt(argument)).astype(np.float32)
    if not np.isfinite(variance).all() or not np.isfinite(inverse).all():
        raise ValueError(
            "nonfinite normalization intermediate is outside qualification"
        )
    return variance, inverse


def hidden_norm(value, weight, epsilon, residual=None, *, lanes=512):
    x, w = ref.finite(value).astype(np.float32), ref.finite(weight).astype(np.float32)
    if w.shape != (5120,) or x.ndim != 2 or x.shape[1] != 5120:
        raise ValueError("hidden normalization shape changed")
    if residual is not None:
        r = ref.finite(residual).astype(np.float32)
        if r.shape != x.shape:
            raise ValueError("residual shape differs")
        x = np.add(x, r, dtype=np.float32)
    variance, inverse = norm_moments(x, epsilon, lanes=lanes)
    y = np.multiply(
        np.multiply(x, inverse, dtype=np.float32), w + np.float32(1), dtype=np.float32
    )
    return (
        ref.bf16(y),
        (ref.bf16(x) if residual is not None else None),
        variance,
        inverse,
    )


def dense_paged_attention(query, physical, page_ids, length, *, kscale=1, vscale=1):
    """Independent dense FP64 attention, reconstructing logical paged KV order."""
    q, cache = ref.finite(query), ref.finite(physical)
    ids = np.asarray(page_ids)
    if (
        q.ndim != 2
        or cache.ndim != 4
        or cache.shape[3] != 2 * q.shape[1]
        or q.shape[0] % cache.shape[1]
        or ids.ndim != 1
        or ids.dtype.kind not in "iu"
        or not 1 <= length <= len(ids) * cache.shape[2]
        or (ids < 0).any()
        or (ids >= len(cache)).any()
    ):
        raise ValueError("invalid logical attention geometry")
    scales = [
        np.broadcast_to(ref.finite(s), (cache.shape[1],)) for s in (kscale, vscale)
    ]
    if any((s <= 0).any() for s in scales):
        raise ValueError("descale must be positive")
    out = np.empty_like(q)
    group = q.shape[0] // cache.shape[1]
    # Process one KV head at a time to avoid a second full FP64 cache copy.
    for head in range(cache.shape[1]):
        logical = cache[ids, head].reshape(-1, cache.shape[3])[:length]
        keys, values = np.split(logical, 2, axis=1)
        for h in range(head * group, (head + 1) * group):
            scores = (keys @ q[h]) * (scales[0][head] / np.sqrt(q.shape[1]))
            weights = np.exp(scores - scores.max())
            out[h] = (weights @ values) * (scales[1][head] / weights.sum())
    return out


def equal_bytes(left, right):
    a, b = np.asarray(left), np.asarray(right)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def minimize_norm_witness(value, residual, weight, epsilon):
    """Keep one row and greedily remove input coordinates while preserving failure."""
    x, r = ref.finite(value).copy(), ref.finite(residual).copy()
    if x.shape != (1, 5120) or r.shape != x.shape:
        raise ValueError("minimizer accepts exactly one hidden row")

    def differs(a, b):
        old, *_ = hidden_norm(a, weight, epsilon, b, lanes=32)
        new, *_ = hidden_norm(a, weight, epsilon, b, lanes=512)
        oq, os = ref.dynamic_fp8(old)
        nq, ns = ref.dynamic_fp8(new)
        return not (equal_bytes(oq, nq) and equal_bytes(os, ns))

    if not differs(x, r):
        raise ValueError("supplied row does not reproduce the normalization mismatch")
    attempts = 0
    for width in (1024, 256, 64, 16, 4, 1):
        for begin in range(0, 5120, width):
            if not np.any(x[:, begin : begin + width]) and not np.any(
                r[:, begin : begin + width]
            ):
                continue
            a, b = x.copy(), r.copy()
            a[:, begin : begin + width] = 0
            b[:, begin : begin + width] = 0
            attempts += 1
            if differs(a, b):
                x, r = a, b
    return x, r, attempts

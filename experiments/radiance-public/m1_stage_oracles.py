"""Independent CPU arithmetic for the stage audit, never a serving replacement.

Real-function FP64 comparisons diagnose error; they do not silently redefine the
backend's FP32 reduction order. Exact checks are reserved for representation,
indexing, and deliberately constructed arithmetic witnesses.
"""

from __future__ import annotations

import numpy as np


def finite(value):
    value = np.asarray(value, dtype=np.float64)
    if not value.size or not np.isfinite(value).all():
        raise ValueError("nonempty finite input required")
    return value


def bf16(value):
    """Direct binary64 -> BF16 RNE, without an intervening FP32 double rounding."""
    value = finite(value)
    magnitude = np.minimum(np.abs(value), float(2**128))
    _, exponent = np.frexp(magnitude)
    step = np.exp2(np.maximum(exponent - 8, -133).astype(np.float64))
    rounded = np.rint(magnitude / step) * step
    rounded = np.where(rounded >= float(2**128), np.inf, rounded)
    return np.copysign(np.abs(rounded), value)


def fp8_values():
    codes = np.arange(127, dtype=np.int64)
    exponent, fraction = codes >> 3, codes & 7
    return np.where(
        exponent == 0, fraction * 2.0**-9, (1 + fraction / 8) * np.exp2(exponent - 7.0)
    )


def fp8_encode(value):
    """OCP E4M3FN satfinite/RNE, independently enumerating all finite codes."""
    value = finite(value)
    grid = fp8_values()
    magnitude = np.minimum(np.abs(value), 448.0)
    hi = np.minimum(np.searchsorted(grid, magnitude), 126)
    lo = np.maximum(hi - 1, 0)
    left, right = magnitude - grid[lo], grid[hi] - magnitude
    chosen = np.where((left < right) | ((left == right) & ((lo & 1) == 0)), lo, hi)
    return (chosen | (np.signbit(value).astype(np.int64) << 7)).astype(np.uint8)


def fp8_decode(codes):
    codes = np.asarray(codes, dtype=np.uint8)
    if np.any((codes & 127) == 127):
        raise ValueError("NaN FP8 is outside the finite contract")
    return np.copysign(fp8_values()[codes & 127], np.where(codes & 128, -1.0, 1.0))


def dynamic_fp8(value):
    """Declared per-row FP32 scaling and RNE conversion, CPU arithmetic only."""
    value = finite(value).astype(np.float32)
    scale = np.maximum(
        np.max(np.abs(value), axis=-1, keepdims=True) / np.float32(448),
        np.float32(1 / (448 * 512)),
    )
    normalized = np.divide(value, scale, dtype=np.float32)
    return fp8_encode(normalized), scale


def rms(value, weight, epsilon, *, residual=None, offset=0):
    x, w = finite(value), finite(weight)
    if residual is not None:
        # The residual interface specifies FP32 addition before normalization.
        x = (x.astype(np.float32) + finite(residual).astype(np.float32)).astype(
            np.float64
        )
    if not np.isfinite(epsilon) or epsilon <= 0 or w.shape != (x.shape[-1],):
        raise ValueError("invalid normalization contract")
    # Scaling the reference avoids overflow/underflow even for extreme BF16 x.
    magnitude = np.maximum(np.abs(x).max(axis=-1, keepdims=True), np.sqrt(epsilon))
    scaled = x / magnitude
    denominator = np.sqrt(
        np.mean(scaled * scaled, axis=-1, keepdims=True)
        + (epsilon / magnitude) / magnitude
    )
    return scaled / denominator * (w + offset)


def sigmoid(value):
    x = finite(value)
    decay = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + decay), decay / (1 + decay))


def silu(value):
    x = finite(value)
    return x * sigmoid(x)


def periodic_attention(query, logical_period, length, *, kscale=1.0, vscale=1.0):
    """FP64 full attention over a repeated logical KV period, without expansion."""
    query, period = finite(query), finite(logical_period)
    if (
        query.ndim != 2
        or period.ndim != 3
        or period.shape[-1] != query.shape[-1] * 2
        or query.shape[0] % period.shape[1]
        or type(length) is not int
        or length < 1
        or not np.isfinite([kscale, vscale]).all()
    ):
        raise ValueError("invalid repeated attention geometry")
    counts = np.full(len(period), length // len(period), dtype=np.float64)
    counts[: length % len(period)] += 1
    valid = counts > 0
    dimension = query.shape[-1]
    group = query.shape[0] // period.shape[1]
    outputs = []
    for head, q in enumerate(query):
        keys = period[:, head // group, :dimension] * kscale
        values = period[:, head // group, dimension:] * vscale
        scores = keys @ q / np.sqrt(dimension)
        weights = np.zeros(len(period))
        weights[valid] = np.exp(scores[valid] - scores[valid].max()) * counts[valid]
        outputs.append(weights @ values / weights.sum())
    return np.stack(outputs)


def gdn_step(state, mixed, a, b, a_log, bias, *, normalize=True):
    """FP64 delta-rule equations with the explicitly chosen BF16 beta boundary."""
    state, mixed = finite(state), finite(mixed)
    if state.shape != (48, 128, 128) or mixed.shape != (10240,):
        raise ValueError("pinned GDN geometry required")
    q = mixed[:2048].reshape(16, 128).repeat(3, axis=0)
    k = mixed[2048:4096].reshape(16, 128).repeat(3, axis=0)
    v = mixed[4096:].reshape(48, 128)
    if normalize:
        q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6)
        k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    q = q * 128**-0.5
    x = finite(a) + finite(bias)
    softplus = np.logaddexp(0, x)
    decay = np.exp(-np.exp(finite(a_log)) * softplus)
    beta = bf16(sigmoid(b))
    result = state * decay[:, None, None]
    prediction = np.einsum("hvk,hk->hv", result, k)
    result = result + ((v - prediction) * beta[:, None])[:, :, None] * k[:, None, :]
    out = np.einsum("hvk,hk->hv", result, q)
    return result, out


def error(actual, expected):
    expected = finite(expected)
    actual = np.asarray(actual, dtype=np.float64)
    if actual.shape != expected.shape:
        raise ValueError("shape mismatch")
    valid = bool(np.isfinite(actual).all())
    delta = actual - expected
    return {
        "elements": int(actual.size),
        "finite": valid,
        "max_abs": float(np.abs(delta).max()) if valid else None,
        "relative_l2": float(
            np.linalg.norm(delta.ravel())
            / max(np.linalg.norm(expected.ravel()), 1e-300)
        )
        if valid
        else None,
        "bf16_mismatches": int(np.count_nonzero(actual != bf16(expected))),
    }


def acceptable(measured, *, relative=0.004, absolute=None):
    if not measured["finite"]:
        return False
    if absolute is not None:
        return measured["max_abs"] <= absolute
    return measured["relative_l2"] <= relative

"""CPU specification of interval-guided target-head refinement.

The inequalities are conditional on the declared arithmetic contract. This
module neither qualifies a GPU binary nor changes the serving sampler. No GPU
library is imported. Approximate ranks order work; only intervals exclude tokens.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

FP32_DOT_CONTRACT = "finite-bf16-inputs/fp32-dot/bf16-rne/v1"
F64 = np.finfo(np.float64)
F32 = np.finfo(np.float32)


class InvalidCertificate(ValueError):  # noqa: N818 - protocol rejection, not a serving exception
    """An unsupported or inconsistent certificate must select the full head."""


def upward(value):
    return np.nextafter(np.asarray(value, dtype=np.float64), np.inf)


def downward(value):
    return np.nextafter(np.asarray(value, dtype=np.float64), -np.inf)


def _finite(value):
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise InvalidCertificate("nonfinite numerical input or bound")
    return value


def _denominator(operations: int, bits: int):
    if not 0 < operations < 2**bits:
        raise InvalidCertificate("unsupported arithmetic operation count")
    return downward(1.0 - operations * 2.0**-bits)


def gamma(operations: int, bits: int) -> float:
    return float(upward((operations * 2.0**-bits) / _denominator(operations, bits)))


def positive_sum_upper(value, axis=-1):
    value = _finite(value)
    if np.any(value < 0):
        raise InvalidCertificate("positive reduction received a negative operand")
    count = value.shape[axis]
    # Covers any FP64 addition tree with at most count rounding operations,
    # including a conservative allowance for flushed underflow.
    total = np.sum(value, axis=axis, dtype=np.float64)
    result = upward(upward(total + count * F64.tiny) / _denominator(count, 53))
    return _finite(result)


def positive_dot_upper(left, right):
    left, right = _finite(left), _finite(right)
    if left.shape[-1] != right.shape[0] or np.any(left < 0) or np.any(right < 0):
        raise InvalidCertificate("invalid positive matrix product")
    count = 2 * left.shape[-1]
    value = left @ right
    return _finite(upward(upward(value + count * F64.tiny) / _denominator(count, 53)))


def norm_upper(value, axis=-1):
    value = np.abs(_finite(value))
    squared = upward(upward(value * value) + F64.tiny)
    return _finite(upward(np.sqrt(positive_sum_upper(squared, axis))))


def bf16(value):
    """RNE conversion of binary32 values; finite range is checked by callers."""
    single = np.asarray(value, dtype=np.float32)
    bits = single.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def bf16_interval(lower, upper):
    """Outward FP32 conversion followed by monotone BF16 RNE conversion.

    Outward conversion also avoids claiming a singleton through FP64 -> FP32
    -> BF16 double rounding near a BF16 midpoint.
    """
    lower, upper = _finite(lower), _finite(upper)
    if np.any(lower > upper) or np.any(np.abs(lower) > F32.max) or np.any(np.abs(upper) > F32.max):
        raise InvalidCertificate("reversed or overflowing reference interval")
    lo = np.nextafter(lower.astype(np.float32), np.float32(-np.inf))
    hi = np.nextafter(upper.astype(np.float32), np.float32(np.inf))
    return _finite(bf16(lo)), _finite(bf16(hi))


@dataclass(frozen=True)
class SearchWeights:
    """One search representation; the BF16 model weights are never replaced."""

    codes: np.ndarray
    scale: np.ndarray
    bias_scale: np.ndarray
    bits: int
    group: int

    def effective(self):
        bias = 4 if self.bits == 2 else 16
        groups = self.codes.reshape(*self.scale.shape, self.group).astype(np.float64)
        return ((1.0 + groups / bias) * self.scale[..., None] - self.bias_scale[..., None]).reshape(
            self.codes.shape
        )


def quantize_search(weight, bits=2, group=128) -> SearchWeights:
    """Reconstruct the pinned biased INT2 packer's finite FP32/BF16 operations.

    INT4 uses the analogous 16x biased representation as a separate search
    experiment. CPU reconstruction is not a claim about captured GPU packing.
    """
    original = _finite(weight)
    if np.any(np.abs(original) > F32.max):
        raise InvalidCertificate("reference weights overflow binary32")
    w = original.astype(np.float32)
    if bits not in (2, 4) or w.ndim != 2 or group < 1 or not w.size or w.shape[1] % group:
        raise InvalidCertificate("unsupported search shape or quantization")
    if not np.array_equal(bf16(w).astype(np.float64), original):
        raise InvalidCertificate("reference weights must already be BF16")
    g = w.reshape(w.shape[0], -1, group)
    levels = (1 << bits) - 1
    scale = np.maximum((g.max(-1) - g.min(-1)) / np.float32(levels), np.float32(1e-8))
    zero = np.clip(np.rint(-g.min(-1) / scale), 0, levels)
    codes = np.clip(np.rint(g / scale[..., None] + zero[..., None]), 0, levels).astype(np.uint8)
    bias = np.float32(4 if bits == 2 else 16)
    return SearchWeights(
        codes.reshape(w.shape),
        bf16(bias * scale).astype(np.float64),
        bf16(bias * scale + zero * scale).astype(np.float64),
        bits,
        group,
    )


@dataclass(frozen=True)
class GroupBounds:
    residual_l2: np.ndarray
    weight_l2: np.ndarray
    coarse_l2: np.ndarray
    weight_max: np.ndarray
    effective: np.ndarray
    group: int


def prepare_bounds(weight, search: SearchWeights) -> GroupBounds:
    w, effective = _finite(weight), _finite(search.effective())
    if w.shape != effective.shape:
        raise InvalidCertificate("weight and search representation shapes differ")
    shape = (*search.scale.shape, search.group)
    # The declared CPU search vector is effective, a finite binary64 vector.
    # Bound subtraction rounding before computing the residual norm.
    delta = w - effective
    residual_abs = np.maximum(np.abs(downward(delta)), np.abs(upward(delta)))
    return GroupBounds(
        norm_upper(residual_abs.reshape(shape)),
        norm_upper(w.reshape(shape)),
        norm_upper(effective.reshape(shape)),
        np.max(np.abs(w), axis=1),
        effective,
        search.group,
    )


def reference_error(hidden, weight_l2, weight_max, group, *, contract: str):
    """FP32 dot envelope before output conversion; see the accompanying derivation.

    The contract permits FMA or separate products/adds with <=2K roundings,
    FP32 accumulation, and flushed underflow. BF16 partial accumulators,
    omitted/duplicated products and unknown native arithmetic are not admitted.
    """
    if contract != FP32_DOT_CONTRACT:
        raise InvalidCertificate("unbound reference arithmetic")
    x = _finite(hidden)
    if x.ndim != 2 or group < 1 or not x.size or x.shape[1] % group:
        raise InvalidCertificate("invalid hidden shape")
    if not np.array_equal(bf16(x).astype(np.float64), x):
        raise InvalidCertificate("hidden values must already be BF16")
    k = x.shape[1]
    xnorm = norm_upper(x.reshape(x.shape[0], -1, group))
    magnitude = positive_dot_upper(xnorm, weight_l2.T)
    # This sufficient condition excludes intermediate overflow for the admitted
    # accumulation. It is not inferred from the observed final logit alone.
    if np.any(magnitude >= F32.max / 4):
        raise InvalidCertificate("reference accumulation could overflow")
    arithmetic = upward(gamma(2 * k, 24) * magnitude)
    underflow = upward((2 * k * F32.tiny) / _denominator(2 * k, 24))
    flush = upward(
        k * F32.tiny * upward(weight_max[None, :] + np.max(np.abs(x), axis=1)[:, None] + F32.tiny)
    )
    return upward(upward(arithmetic + underflow) + flush), xnorm


def search_intervals(hidden, model: GroupBounds, *, contract: str):
    """Conservative CPU search intervals, including reference arithmetic.

    The center here is FP64 evaluation of the reconstructed search weights.
    It is deliberately not substituted for the native INT2 kernel's arithmetic.
    """
    x = _finite(hidden)
    error, xnorm = reference_error(
        x, model.weight_l2, model.weight_max, model.group, contract=contract
    )
    residual = positive_dot_upper(xnorm, model.residual_l2.T)
    coarse_magnitude = positive_dot_upper(xnorm, model.coarse_l2.T)
    k = x.shape[1]
    coarse_rounding = upward(upward(gamma(2 * k, 53) * coarse_magnitude) + 2 * k * F64.tiny)
    center = _finite(x @ model.effective.T)
    radius = upward(upward(residual + error) + coarse_rounding)
    lo, hi = bf16_interval(downward(center - radius), upward(center + radius))
    return center, lo, hi


def reference_score_intervals(hidden, weight, *, group=128, contract: str):
    """Score fidelity: enclose the reference dot, then require a BF16 singleton."""
    x, w = _finite(hidden), _finite(weight)
    if (
        w.ndim != 2
        or group < 1
        or not w.size
        or w.shape[1] % group
        or not np.array_equal(bf16(w).astype(np.float64), w)
    ):
        raise InvalidCertificate("unsupported reference weights")
    wn = norm_upper(w.reshape(w.shape[0], -1, group))
    error, xn = reference_error(x, wn, np.max(np.abs(w), axis=1), group, contract=contract)
    magnitude = positive_dot_upper(xn, wn.T)
    rounding = upward(upward(gamma(2 * x.shape[1], 53) * magnitude) + 2 * x.shape[1] * F64.tiny)
    center = _finite(x @ w.T)
    radius = upward(error + rounding)
    return bf16_interval(downward(center - radius), upward(center + radius))


def strictly_excluded(upper, cutoff):
    """This actual predicate is also exercised by the symbolic proof runner."""
    return upper < cutoff


@dataclass(frozen=True)
class IntervalRow:
    lower: np.ndarray
    upper: np.ndarray
    identity: str

    def checked(self, size=None):
        lo, hi = _finite(self.lower), _finite(self.upper)
        if (
            not self.identity
            or lo.ndim != 1
            or hi.shape != lo.shape
            or (size is not None and len(lo) != size)
        ):
            raise InvalidCertificate("interval shape or invocation identity differs")
        if not len(lo) or np.any(lo > hi):
            raise InvalidCertificate("empty or reversed interval")
        return lo.copy(), hi.copy()


@dataclass(frozen=True)
class Refinement:
    ids: np.ndarray
    intervals: IntervalRow


@dataclass(frozen=True)
class Selection:
    status: str
    ids: np.ndarray
    scores: np.ndarray
    rounds: int
    rescored: int
    reason: str


def refine_topk(
    intervals: IntervalRow,
    approximate,
    k: int,
    refiners: tuple[Callable[[np.ndarray], Refinement], ...],
    fallback: Callable[[], np.ndarray],
    *,
    seed_size=128,
    max_work=1024,
) -> Selection:
    """Variable support, optional approximation-ranked seed, bounded refinement.

    Each callback must return a valid enclosure under the *same* reference and
    invocation identity. That is an explicit trusted interface, not a claim that
    an arbitrary callback or a matching token proves native correctness.
    """
    rounds = work = 0

    def full(reason):
        scores = np.asarray(fallback())
        # Preserve the complete fallback row, including nonfinite values: their
        # handling belongs to the unchanged reference backend/sampler.
        return Selection("FULL_REFERENCE", np.arange(scores.size), scores, rounds, work, reason)

    try:
        lo, hi = intervals.checked()
        approx = _finite(approximate)
        integers = all(
            isinstance(v, (int, np.integer)) and not isinstance(v, bool)
            for v in (k, seed_size, max_work)
        )
        if (
            not integers
            or approx.shape != lo.shape
            or not 1 <= k <= len(lo)
            or seed_size < 0
            or max_work < 0
        ):
            raise InvalidCertificate("invalid selection parameters")
        attempted = [np.zeros(len(lo), dtype=bool) for _ in refiners]
        seeded = False
        while True:
            cutoff = np.partition(lo, len(lo) - k)[len(lo) - k]
            survivors = np.flatnonzero(~strictly_excluded(hi, cutoff))
            exact = lo.view(np.uint64) == hi.view(np.uint64)
            if np.all(exact[survivors]):
                # Re-evaluate the final cutoff; retain every tie, not exactly k
                # arbitrary token IDs. The normal sampler still selects support.
                final_cutoff = np.partition(lo[survivors], len(survivors) - k)[len(survivors) - k]
                keep = survivors[lo[survivors] >= final_cutoff]
                if np.any(lo[keep] == 0):
                    return full("zero_sign_not_certified")
                if np.any(np.abs(lo[keep]) > F32.max) or not np.array_equal(
                    bf16(lo[keep]).astype(np.float64), lo[keep]
                ):
                    raise InvalidCertificate("singleton is not a finite BF16 reference score")
                return Selection("CERTIFIED_UNDER_CONTRACT", keep, lo[keep], rounds, work, "")
            if work >= max_work:
                return full("refinement_budget")
            unresolved = survivors[~exact[survivors]]
            for stage in range(len(refiners)):
                pending = unresolved[~attempted[stage][unresolved]]
                if len(pending):
                    break
            else:
                return full("score_fidelity_unresolved")
            # Global top-N is merely a first work order. Tokens outside it stay
            # represented by their intervals and remain eligible for later work.
            if not seeded and seed_size:
                pending = pending[np.argsort(-approx[pending], kind="stable")[:seed_size]]
                seeded = True
            elif len(pending) > max_work - work:
                return full("candidate_buffer_budget")
            if len(pending) > max_work - work:
                return full("refinement_budget")
            attempted[stage][pending] = True
            response = refiners[stage](pending.copy())
            rounds += 1
            work += len(pending)
            ids = np.asarray(response.ids)
            if ids.dtype.kind not in "iu" or not np.array_equal(ids, pending):
                raise InvalidCertificate("refiner changed candidate identities")
            if response.intervals.identity != intervals.identity:
                raise InvalidCertificate("stale refinement invocation")
            newer_lo, newer_hi = response.intervals.checked(len(ids))
            lo[ids] = np.maximum(lo[ids], newer_lo)
            hi[ids] = np.minimum(hi[ids], newer_hi)
            if np.any(lo[ids] > hi[ids]):
                raise InvalidCertificate("inconsistent refinement intervals")
    except (InvalidCertificate, FloatingPointError) as error:
        return full(str(error))

from __future__ import annotations

import importlib.util
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from qwen_r9700_lab import certified_head as ch


def rounded_fraction(value: Fraction, mantissa=24, minimum_exponent=-149):
    """Independent rational RNE oracle, including subnormals (finite test domain)."""
    if not value:
        return 0.0
    sign = -1 if value < 0 else 1
    value = abs(value)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < Fraction(2) ** exponent:
        exponent -= 1
    step = Fraction(2) ** max(exponent - mantissa + 1, minimum_exponent)
    quotient = value / step
    integral, remainder = divmod(quotient.numerator, quotient.denominator)
    if 2 * remainder > quotient.denominator or (
        2 * remainder == quotient.denominator and integral % 2
    ):
        integral += 1
    return float(sign * integral * step)


def oracle_dot(x, w, reverse=False):
    total = 0.0
    pairs = list(zip(x, w, strict=True))
    for a, b in reversed(pairs) if reverse else pairs:
        product = rounded_fraction(Fraction(float(a)) * Fraction(float(b)))
        total = rounded_fraction(Fraction(total) + Fraction(product))
    return rounded_fraction(Fraction(total), 8, -133)


def exact_refiner(scores, identity="invocation-1"):
    def callback(ids):
        values = scores[ids]
        return ch.Refinement(ids, ch.IntervalRow(values, values, identity))

    return callback


def select(reference, approximate, lo, hi, *, k=20, **kwargs):
    return ch.refine_topk(
        ch.IntervalRow(lo, hi, "invocation-1"),
        approximate,
        k,
        (exact_refiner(reference),),
        lambda: reference,
        **kwargs,
    )


@pytest.mark.parametrize("seed", [0, 1, 8, 128, 256])
def test_poor_rank_winner_survives_and_no_eight_per_block_limit(seed):
    scores = np.full(1024, -12.0)
    scores[:20] = np.arange(40, 60)  # All twenty in one 64-token block.
    scores[1001] = 80  # Approximate rank 1002, outside either proposed seed.
    approx = -np.arange(1024, dtype=float)
    lo, hi = np.full(1024, -20.0), np.full(1024, 100.0)
    outcome = select(scores, approx, lo, hi, seed_size=seed, max_work=2048)
    expected = np.flatnonzero(scores >= np.sort(scores)[-20])
    assert outcome.status == "CERTIFIED_UNDER_CONTRACT"
    np.testing.assert_array_equal(outcome.ids, expected)
    np.testing.assert_array_equal(outcome.scores, scores[expected])
    assert 1001 in outcome.ids


def test_more_than_256_ties_are_retained():
    scores = np.full(600, 10.0)
    result = select(scores, np.arange(600), scores - 1, scores + 1, max_work=700)
    assert result.status == "CERTIFIED_UNDER_CONTRACT"
    assert len(result.ids) == 600


def test_equal_cutoff_is_not_excluded():
    assert not ch.strictly_excluded(10, 10)
    scores = np.array([10.0, 10.0, 9.0])
    result = select(scores, scores, scores, scores, k=1)
    assert result.ids.tolist() == [0, 1]
    # Negative control: a <= exclusion would silently discard the tied token.
    assert (scores <= 10)[1]


@pytest.mark.parametrize("work,seed", [(0, 128), (1, 128), (256, 128)])
def test_work_budget_falls_back_without_truncating(work, seed):
    scores = np.arange(500) / 2 + 1
    result = select(
        scores, -scores, np.zeros(500), np.full(500, 300.0), seed_size=seed, max_work=work
    )
    assert result.status == "FULL_REFERENCE"
    assert len(result.ids) == len(scores)
    np.testing.assert_array_equal(result.scores, scores)
    assert result.rescored <= work


def test_int4_stage_intersection_and_score_fidelity_are_separate():
    scores = np.array([10.0, 9.0, 8.0, 1.0])
    broad = ch.IntervalRow(scores - 20, scores + 20, "invocation-1")

    def tighten(ids):
        return ch.Refinement(
            ids, ch.IntervalRow(scores[ids] - 0.1, scores[ids] + 0.1, "invocation-1")
        )

    incomplete = ch.refine_topk(broad, scores, 2, (tighten,), lambda: scores, seed_size=0)
    assert incomplete.status == "FULL_REFERENCE"
    assert incomplete.reason == "score_fidelity_unresolved"
    exact = ch.refine_topk(
        broad, scores, 2, (tighten, exact_refiner(scores)), lambda: scores, seed_size=0
    )
    assert exact.status == "CERTIFIED_UNDER_CONTRACT"
    assert exact.ids.tolist() == [0, 1]
    assert exact.rounds == 2
    assert exact.rescored == 6


@pytest.mark.parametrize("fault", ["nan", "infinity", "reversed", "identity", "shape"])
def test_invalid_initial_bounds_fall_back(fault):
    lo, hi, identity = np.ones(4), np.full(4, 10.0), "request"
    if fault == "nan":
        hi[0] = np.nan
    elif fault == "infinity":
        hi[0] = np.inf
    elif fault == "reversed":
        lo[0] = 11
    elif fault == "identity":
        identity = ""
    else:
        hi = hi[:3]
    full = np.array([10.0, 3.0, 2.0, 1.0])
    outcome = ch.refine_topk(ch.IntervalRow(lo, hi, identity), full, 1, (), lambda: full)
    assert outcome.status == "FULL_REFERENCE"
    np.testing.assert_array_equal(outcome.scores, full)


@pytest.mark.parametrize("fault", ["stale", "ids", "outside", "non_bf16", "zero_sign"])
def test_bad_refinement_cannot_publish(fault):
    full = np.array([10.0, 9.0, 8.0])

    def bad(ids):
        identity = "old-request" if fault == "stale" else "request"
        values = full[ids].copy()
        if fault == "outside":
            values[:] = 30
        elif fault == "non_bf16":
            values[:] = 10.001
        elif fault == "zero_sign":
            values[:] = 0.0
        returned = ids[::-1] if fault == "ids" else ids
        return ch.Refinement(returned, ch.IntervalRow(values, values, identity))

    outcome = ch.refine_topk(
        ch.IntervalRow(np.full(3, -1), np.full(3, 20), "request"), full, 1, (bad,), lambda: full
    )
    assert outcome.status == "FULL_REFERENCE"
    np.testing.assert_array_equal(outcome.scores, full)


def test_no_progress_or_invalid_approximation_never_loops():
    row = ch.IntervalRow(np.ones(3), np.full(3, 3), "request")
    calls = []

    def no_progress(ids):
        calls.append(ids)
        return ch.Refinement(ids, ch.IntervalRow(row.lower[ids], row.upper[ids], row.identity))

    result = ch.refine_topk(row, np.ones(3), 1, (no_progress,), lambda: np.ones(3))
    assert len(calls) == 1
    assert result.reason == "score_fidelity_unresolved"
    result = ch.refine_topk(row, np.full(3, np.nan), 1, (no_progress,), lambda: np.ones(3))
    assert result.status == "FULL_REFERENCE"
    assert len(calls) == 1


@pytest.mark.parametrize("k", [0, -1, 4, 1.5, True])
def test_invalid_k_is_a_full_head_fallback(k):
    scores = np.ones(3)
    result = select(scores, scores, scores, scores, k=k)
    assert result.status == "FULL_REFERENCE"


def test_randomized_variable_support_matches_independent_full_sort():
    rng = np.random.default_rng(97531)
    for _ in range(100):
        scores = ch.bf16(rng.normal(10, 4, 400)).astype(np.float64)
        radii = rng.uniform(0, 20, 400)
        k = int(rng.integers(1, 80))
        result = select(
            scores,
            rng.normal(size=400),
            scores - radii,
            scores + radii,
            k=k,
            seed_size=int(rng.integers(0, 300)),
            max_work=1000,
        )
        expected = np.flatnonzero(scores >= sorted(scores, reverse=True)[k - 1])
        assert result.status == "CERTIFIED_UNDER_CONTRACT"
        np.testing.assert_array_equal(result.ids, expected)
        np.testing.assert_array_equal(result.scores, scores[expected])


def test_bf16_conversion_and_midpoints_against_rational_oracle():
    values = [
        0.5,
        1.00390625,
        1.01171875,
        -1.00390625,
        -1.01171875,
        2.0**-133,
        2.0**-134,
        3 * 2.0**-134,
    ]
    expected = [rounded_fraction(Fraction(v), 8, -133) for v in values]
    np.testing.assert_array_equal(ch.bf16(values), expected)
    middle = 1.00390625
    for value in (np.nextafter(middle, -np.inf), middle, np.nextafter(middle, np.inf)):
        lo, hi = ch.bf16_interval(value, value)
        assert lo <= rounded_fraction(Fraction(value), 8, -133) <= hi


def test_positive_reduction_bounds_dominate_exact_rational_arithmetic():
    rng = np.random.default_rng(6789)
    a = np.exp2(rng.integers(-500, 400, size=(4, 31))).astype(np.float64)
    b = np.exp2(rng.integers(-500, 400, size=(31, 5))).astype(np.float64)
    upper = ch.positive_dot_upper(a, b)
    for row in range(4):
        for col in range(5):
            exact = sum(
                Fraction(float(x)) * Fraction(float(y))
                for x, y in zip(a[row], b[:, col], strict=True)
            )
            assert Fraction(float(upper[row, col])) >= exact
    norms = ch.norm_upper(a)
    for row, norm in zip(a, norms, strict=True):
        assert Fraction(float(norm)) ** 2 >= sum(Fraction(float(v)) ** 2 for v in row)


@pytest.mark.parametrize("bits", [2, 4])
def test_group_intervals_enclose_independent_reference_orders(bits):
    rng = np.random.default_rng(42)
    x = ch.bf16(rng.normal(size=(4, 16)))
    w = ch.bf16(rng.normal(size=(17, 16)))
    x[0, :4] = [2**20, 1, -(2**20), 2**-126]
    w[0] = 1
    model = ch.prepare_bounds(w, ch.quantize_search(w, bits, group=8))
    _, lo, hi = ch.search_intervals(x, model, contract=ch.FP32_DOT_CONTRACT)
    ref_lo, ref_hi = ch.reference_score_intervals(x, w, group=8, contract=ch.FP32_DOT_CONTRACT)
    for i, hidden in enumerate(x):
        for j, weight in enumerate(w):
            for reverse in (False, True):
                reference = oracle_dot(hidden, weight, reverse)
                assert lo[i, j] <= reference <= hi[i, j]
                assert ref_lo[i, j] <= reference <= ref_hi[i, j]
    # The arithmetic contract allows different FP32 reductions; output may not
    # be a singleton even with exact weights. This must remain unresolved.
    assert ref_lo[0, 0] != ref_hi[0, 0]


def test_unknown_arithmetic_and_bad_quantizer_inputs_are_rejected():
    weights = np.ones((2, 16))
    for group in (0, -1, 3):
        with pytest.raises(ch.InvalidCertificate):
            ch.quantize_search(weights, group=group)
    for values in (weights + 0.0000000001, weights * 1e200):
        with pytest.raises(ch.InvalidCertificate):
            ch.quantize_search(values, group=8)
    with pytest.raises(ch.InvalidCertificate, match="unbound"):
        ch.reference_score_intervals(weights, weights, group=8, contract="BF16-partial-sums")


def test_cpu_pilot_real_file_path_and_identity_failure(tmp_path):
    path = Path(__file__).parents[1] / "experiments/radiance-public/pilot_certified_head.py"
    spec = importlib.util.spec_from_file_location("head_pilot_test", path)
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    rng = np.random.default_rng(19)
    w = ch.bf16(rng.normal(size=(320, 16)))
    x = ch.bf16(rng.normal(size=(2, 16)))
    scores = np.array([[oracle_dot(h, row) for row in w] for h in x], dtype=np.float32)
    weights = tmp_path / "head.bf16"
    weights.write_bytes((w.view(np.uint32) >> 16).astype("<u2").tobytes())
    metadata = tmp_path / "head.json"
    metadata.write_text(
        json.dumps({"shape": list(w.shape), "dtype": "BF16", "sha256": pilot.digest(weights)})
    )
    rows = tmp_path / "rows.npz"
    np.savez(rows, hidden=x, reference=scores)
    receipt = {"output_sha256": pilot.digest(rows), "scope": "synthetic test"}
    rows.with_suffix(".json").write_text(json.dumps(receipt))
    result = pilot.run(weights, metadata, rows, tmp_path / "result", group=8, chunk=64)
    assert not result["native_certificate"]
    assert not result["gpu_used"]
    for variant in result["variants"].values():
        assert variant["captured_scores_outside_intervals"] == 0
        assert variant["selection"]["20"]["captured_topk_misses"] == 0
    weights.write_bytes(b"broken")
    with pytest.raises(ValueError, match="identity mismatch"):
        pilot.run(weights, metadata, rows, tmp_path / "broken")

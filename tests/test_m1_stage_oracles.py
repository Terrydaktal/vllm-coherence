"""Independent small truths and negative controls for the native stage auditor."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).parents[1] / "experiments/radiance-public/m1_stage_oracles.py"
spec = importlib.util.spec_from_file_location("stage_oracles", PATH)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def test_all_fp8_encodings_and_midpoint_ties():
    codes = np.array([i for i in range(256) if i & 127 != 127], dtype=np.uint8)
    assert np.array_equal(oracle.fp8_encode(oracle.fp8_decode(codes)), codes)
    grid = oracle.fp8_values()
    midpoint = (grid[:-1] + grid[1:]) / 2
    even = np.where(np.arange(126) % 2 == 0, np.arange(126), np.arange(1, 127))
    assert np.array_equal(oracle.fp8_encode(midpoint), even)
    assert np.array_equal(oracle.fp8_encode(-midpoint), even | 128)
    assert oracle.fp8_encode([1000, -1000]).tolist() == [126, 254]


def test_bf16_direct_rounding_and_gradual_underflow():
    assert oracle.bf16([1 + 2**-8, 1 + 3 * 2**-8]).tolist() == [1, 1 + 2**-6]
    # A value just above the BF16 midpoint would lose that distinction in FP32.
    assert oracle.bf16([1 + 2**-8 + 2**-30])[0] == 1 + 2**-7
    assert oracle.bf16([2**-134, 3 * 2**-134]).tolist() == [0, 2**-132]
    assert np.signbit(oracle.bf16([-0.0]))[0]
    maximum = float(2**128 - 2**120)
    midpoint = float(2**128 - 2**119)
    assert oracle.bf16([maximum, np.nextafter(midpoint, 0)]).tolist() == [
        maximum,
        maximum,
    ]
    assert oracle.bf16([midpoint, -midpoint, 1e300]).tolist() == [
        np.inf,
        -np.inf,
        np.inf,
    ]


def test_bf16_all_finite_storage_values_roundtrip():
    bits = np.arange(65536, dtype=np.uint32) << 16
    values = bits.view(np.float32)
    values = values[np.isfinite(values)].astype(np.float64)
    result = oracle.bf16(values)
    assert np.array_equal(result, values)
    assert np.array_equal(np.signbit(result), np.signbit(values))


@pytest.mark.parametrize("scale", [0, 1e-30, 1, 1e20, 1e38])
def test_rms_constant_closed_form(scale):
    x = np.full((1, 256), scale)
    result = oracle.rms(x, np.ones(256), 1e-6)
    assert np.allclose(result, scale / np.hypot(scale, 0.001), rtol=2e-15, atol=0)


def test_pointwise_extremes_and_quantization_floor():
    assert oracle.sigmoid([-1000, 0, 1000]).tolist() == [0, 0.5, 1]
    assert oracle.silu([-1000, 0, 1000]).tolist() == [0, 0, 1000]
    q, s = oracle.dynamic_fp8(np.zeros((2, 5120)))
    assert not q.any()
    assert np.all(s == np.float32(1 / (448 * 512)))


def test_delta_rule_exact_rank_one_witness():
    x = np.zeros(10240)
    x[:2048].reshape(16, 128)[:, 0] = 1
    x[2048:4096].reshape(16, 128)[:, 0] = 1
    x[4096:] = 2
    state, out = oracle.gdn_step(
        np.zeros((48, 128, 128)),
        x,
        np.zeros(48),
        np.zeros(48),
        np.zeros(48),
        np.zeros(48),
        normalize=False,
    )
    assert np.all(state[:, :, 0] == 1)
    assert not state[:, :, 1:].any()
    assert np.all(out == 128**-0.5)


def test_oracle_rejects_injected_wrong_state_and_nan():
    expected = np.ones((3, 7))
    assert oracle.acceptable(oracle.error(expected, expected))
    broken = expected.copy()
    broken[0, 0] = 2
    assert not oracle.acceptable(oracle.error(broken, expected))
    broken[0, 0] = np.nan
    assert not oracle.acceptable(oracle.error(broken, expected))
    with pytest.raises(ValueError):
        oracle.error([], [])


@pytest.mark.parametrize("length", [1, 2, 3, 6, 7, 31, 128])
def test_periodic_attention_matches_independently_expanded_dense_attention(length):
    rng = np.random.default_rng(87)
    q = rng.normal(size=(4, 5))
    period = rng.normal(size=(3, 2, 10))
    expanded = period[np.arange(length) % 3]
    expected = []
    for head in range(4):
        logits = (expanded[:, head // 2, :5] * 0.125) @ q[head] / np.sqrt(5)
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        expected.append(probabilities @ (expanded[:, head // 2, 5:] * 16))
    actual = oracle.periodic_attention(q, period, length, kscale=0.125, vscale=16)
    assert np.allclose(actual, expected, rtol=2e-14, atol=2e-14)
    # Wrong value descale is visible even when every physical page is reused.
    mutant = oracle.periodic_attention(q, period, length, kscale=0.125, vscale=1)
    assert not np.allclose(mutant, expected)


def test_periodic_attention_rejects_invalid_head_mapping():
    with pytest.raises(ValueError):
        oracle.periodic_attention(np.zeros((3, 5)), np.zeros((4, 2, 10)), 7)

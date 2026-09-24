"""Independent truths, boundary cases and negative controls for the contract."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def contract(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[1] / "experiments/radiance-public")
    )
    return importlib.import_module("m1_arithmetic_contract")


@pytest.mark.parametrize("lanes", [32, 64, 512])
def test_constant_norm_exact_moments(contract, lanes):
    x = np.full((3, 5120), 2, dtype=np.float32)
    y, carry, variance, inv = contract.hidden_norm(
        x, np.zeros(5120), 1e-6, x, lanes=lanes
    )
    assert np.all(variance == 16)
    assert np.all(carry == 4)
    assert np.all(y == 1)
    assert np.all(inv == np.float32(1 / np.sqrt(float(np.float32(16 + 1e-6)))))


def test_norm_contract_independent_of_batch_partition(contract):
    rng = np.random.default_rng(29)
    x = rng.normal(size=(17, 5120)).astype(np.float32)
    w = rng.normal(size=5120).astype(np.float32)
    all_rows = contract.hidden_norm(x, w, 1e-6)[0]
    singles = np.concatenate([contract.hidden_norm(row[None], w, 1e-6)[0] for row in x])
    assert np.array_equal(all_rows, singles)
    m1 = contract.norm_moments(x, 1e-6, lanes=512)[0]
    prefill = contract.norm_moments(x, 1e-6, lanes=32)[0]
    assert np.any(m1 != prefill), "negative control must expose reduction reassociation"


def test_residual_is_not_rounded_before_normalization(contract):
    x = np.full((1, 5120), 1, dtype=np.float32)
    r = np.full_like(x, 2**-8)
    _, carry, variance, _ = contract.hidden_norm(x, np.zeros(5120), 1e-6, r)
    assert np.all(carry == 1)
    assert variance[0, 0] > 1  # premature BF16 carry substitution would give 1


def test_attention_logical_layout_and_masking(contract):
    cache = np.zeros((5, 1, 2, 4))
    cache[3, 0, :, 2:] = [[2, 4], [6, 8]]
    cache[1, 0, :, 2:] = [[10, 12], [1000, 1000]]
    q = np.zeros((2, 2))
    expected = np.array([[6, 8], [6, 8]], dtype=np.float64)
    actual = contract.dense_paged_attention(q, cache, [3, 1], 3)
    assert np.array_equal(actual, expected)
    # Neither padding nor unreferenced physical pages can affect the answer.
    cache[[0, 2, 4]] = -100
    assert np.array_equal(contract.dense_paged_attention(q, cache, [3, 1], 3), expected)
    assert not np.array_equal(
        contract.dense_paged_attention(q, cache, [1, 3], 3), expected
    )


def test_attention_matches_explicit_dense_softmax(contract):
    rng = np.random.default_rng(49)
    q = rng.normal(size=(6, 3))
    cache = rng.normal(size=(9, 2, 4, 6))
    ids = np.array([7, 1, 8, 3])
    logical = cache[ids].transpose(0, 2, 1, 3).reshape(-1, 2, 6)[:13]
    expect = []
    for h in range(6):
        score = logical[:, h // 3, :3] @ q[h] / np.sqrt(3)
        prob = np.exp(score - np.max(score))
        expect.append(prob @ logical[:, h // 3, 3:] / prob.sum())
    assert np.allclose(
        contract.dense_paged_attention(q, cache, ids, 13),
        expect,
        rtol=1e-14,
        atol=1e-14,
    )


@pytest.mark.parametrize("ids,length", [([], 1), ([-1], 1), ([5], 1), ([0], 3)])
def test_invalid_attention_fails(contract, ids, length):
    with pytest.raises(ValueError):
        contract.dense_paged_attention(
            np.zeros((2, 2)), np.zeros((5, 1, 2, 4)), ids, length
        )


def test_exact_means_exact_storage(contract):
    assert not contract.equal_bytes(np.array([0.0]), np.array([-0.0]))
    assert not contract.equal_bytes(
        np.array([1], dtype=np.float32), np.array([1], dtype=np.float64)
    )


def test_norm_overflow_is_not_reported_as_a_valid_zero(contract):
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="nonfinite normalization"),
    ):
        contract.hidden_norm(np.full((1, 5120), 1e30), np.ones(5120), 1e-6)


def test_minimizer_rejects_a_non_counterexample(contract):
    with pytest.raises(ValueError, match="does not reproduce"):
        contract.minimize_norm_witness(
            np.ones((1, 5120)), np.zeros((1, 5120)), np.zeros(5120), 1e-6
        )


def test_cancellation_oracle_accounts_for_the_bf16_output_boundary(contract):
    # The correct BF16 result itself can be >0.001 from a real-number mean.
    cache = np.zeros((1, 1, 16, 4))
    cache[0, 0, :, 2:] = np.array([-16, 16] * 8)[:, None]
    expected = contract.dense_paged_attention(
        np.zeros((2, 2)), cache, [0], 15, vscale=16
    )
    rounded = contract.ref.bf16(expected)
    assert np.max(np.abs(rounded - expected)) > 0.001
    assert np.array_equal(rounded, np.full((2, 2), -17.125))
    assert not np.array_equal(rounded, contract.ref.bf16(expected + 1))

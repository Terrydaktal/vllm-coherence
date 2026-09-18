"""Distribution-level regression for independent speculative sampling draws."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "patch_dflash_sampling_rng", ROOT / "experiments/radiance-public/patch_dflash_sampling_rng.py"
)
patch = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(patch)

probe_spec = importlib.util.spec_from_file_location(
    "probe_dflash_sampling_rng", ROOT / "experiments/radiance-public/probe_dflash_sampling_rng.py"
)
probe = importlib.util.module_from_spec(probe_spec)
assert probe_spec.loader is not None
probe_spec.loader.exec_module(probe)


@pytest.mark.parametrize("position", [0, 37, 86142, 132739, 253792])
@pytest.mark.parametrize("internal_offset", [0, 1 << 30])
@pytest.mark.parametrize("independent", [False, True])
def test_probe_reaches_requested_rng_stream_on_old_and_repaired_selectors(
    position, internal_offset, independent
):
    expression = "tl.load(sample_pos_ptr + flat) - 1"
    if internal_offset:
        expression += " + (1 << 30)"
    actual_offset = probe.selector_stream_offset(f"def selector():\n    position = {expression}\n")
    supplied_position = position + 1 + probe.proposal_input_offset(actual_offset, independent)
    actual_counter = supplied_position - 1 + internal_offset
    assert actual_counter == position + ((1 << 30) if independent else 0)


@pytest.mark.parametrize(
    "source",
    [
        "def selector():\n    position = tl.load(sample_pos_ptr + flat) - 1 + (1 << 29)\n",
        "def selector():\n    position = tl.load(other_ptr + flat) - 1\n",
        "def selector():\n    position = tl.load(sample_pos_ptr + flat) - 1\n    position = 3\n",
        "def selector():\n    position = alias = tl.load(sample_pos_ptr + flat) - 1\n",
        "def selector():\n    pass\n",
    ],
)
def test_probe_rejects_unreviewed_rng_counter_implementations(source):
    with pytest.raises(ValueError, match="unreviewed selector position"):
        probe.selector_stream_offset(source)


def test_probe_rejects_unreviewed_rng_stream_offset():
    with pytest.raises(ValueError, match="unreviewed selector stream offset"):
        probe.proposal_input_offset(1 << 29, True)


def test_shared_proposal_and_residual_noise_biases_three_category_target():
    # Two categories cannot expose this: the residual has only one possible
    # outcome. This asymmetric three-category case exposes the conditioning.
    rng = np.random.default_rng(1789366457)
    n = 400000
    p = np.array([0.1, 0.5, 0.4])
    q = np.array([0.5, 0.3, 0.2])
    proposal_noise = -np.log(-np.log(rng.random((n, 3))))
    proposed = np.argmax(np.log(q) + proposal_noise, axis=1)
    accepted = rng.random(n) < np.minimum(1, p[proposed] / q[proposed])
    with np.errstate(divide="ignore"):
        residual = np.log(np.maximum(p - q, 0))
    shared = np.argmax(residual + proposal_noise, axis=1)
    independent_noise = -np.log(-np.log(rng.random((n, 3))))
    independent = np.argmax(residual + independent_noise, axis=1)
    observed_shared = np.bincount(np.where(accepted, proposed, shared), minlength=3) / n
    observed_independent = np.bincount(np.where(accepted, proposed, independent), minlength=3) / n
    assert np.max(np.abs(observed_shared - p)) > 0.01
    assert np.max(np.abs(observed_independent - p)) < 0.003


def test_patch_rejects_unknown_and_modified_postimages():
    with pytest.raises(ValueError, match="preimage"):
        patch.patched_source(patch.NEW + "unverified_other_changes = True\n")


def test_candidate_patch_uses_upstream_stream_salt_and_is_idempotent(monkeypatch):
    source = "def example():\n    if True:\n" + patch.OLD + "    return position\n"
    monkeypatch.setattr(patch, "PREIMAGE", hashlib.sha256(source.encode()).hexdigest())
    result = patch.patched_source(source)
    monkeypatch.setattr(patch, "POSTIMAGE", hashlib.sha256(result.encode()).hexdigest())
    assert patch.patched_source(result) == result
    compile(result, "candidate.py", "exec")
    assert "- 1 + (1 << 30)" in result

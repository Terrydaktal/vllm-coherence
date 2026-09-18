"""Exhaustive value-map and model-integration checks for the CPU decoder."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from qwen_r9700_lab import conformance_reference as reference
from qwen_r9700_lab.diagnostic_contract import DiagnosticError

DIRECTORY = Path(__file__).resolve().parents[1] / "experiments/radiance-public"


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    return importlib.import_module("mxfp4_lookup_reference")


@pytest.fixture
def decoder(module):
    return module.MXFP4Lookup(reference)


def pair_domain(scales):
    packed = np.broadcast_to(
        np.tile(np.arange(256, dtype=np.uint8), len(scales))[:, None], (len(scales) * 256, 16)
    )
    groups = np.repeat(np.asarray(scales, dtype=np.uint8), 256)[:, None]
    return packed, groups


def test_every_table_entry_matches_canonical_fp32_bits(decoder):
    packed, scales = pair_domain(range(255))
    with np.errstate(over="ignore"):
        expected = reference.unpack_mxfp4(packed, scales)
    pairs = expected.reshape(255, 256, 32)[:, :, :2].copy()
    assert pairs.view("<u8").reshape(-1).tobytes() == decoder.table.tobytes()
    assert decoder.table.size == 65280
    assert decoder.table.nbytes == 522240
    # This checks the actual immutable table, not the extreme-scale fallback.
    assert decoder.calls == decoder.fallbacks == 0


def test_every_fast_pair_is_executed_without_canonical_fallback(decoder):
    packed, scales = pair_domain(range(2, 253))
    actual = decoder(packed, scales)
    expected = reference.unpack_mxfp4(packed, scales)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()
    assert decoder.calls == 1 and decoder.fallbacks == 0


@pytest.mark.parametrize("shape", [(1, 16), (3, 32), (257, 48), (256, 2560), (1, 8704)])
def test_group_boundaries_nibble_order_and_input_storage_are_preserved(decoder, shape):
    rng = np.random.default_rng(9081)
    packed = rng.integers(0, 256, (shape[0] * 2, shape[1] * 2), dtype=np.uint8)[::2, ::2]
    scales = rng.integers(2, 253, (shape[0] * 2, shape[1] // 8), dtype=np.uint8)[::2, ::2]
    before = packed.tobytes(), scales.tobytes(), decoder.table.tobytes()
    actual = decoder(packed, scales)
    expected = reference.unpack_mxfp4(packed, scales)
    assert actual.tobytes() == expected.tobytes()
    actual.fill(5)
    assert before == (packed.tobytes(), scales.tobytes(), decoder.table.tobytes())
    assert decoder.fallbacks == 0


def test_table_cannot_be_made_writeable(decoder):
    with pytest.raises(ValueError):
        decoder.table.flags.writeable = True
    with pytest.raises(ValueError):
        decoder.table[0] = 1


def test_signed_zero_is_preserved(decoder):
    packed = np.full((1, 16), 0x80, dtype=np.uint8)
    actual = decoder(packed, np.array([[127]], dtype=np.uint8)).view(np.uint32)
    assert actual[0].tolist() == [0, 0x80000000] * 16


@pytest.mark.parametrize("scale", [0, 1, 253, 254])
def test_extreme_scales_keep_canonical_floating_point_behavior(decoder, scale):
    packed, scales = pair_domain([scale])
    with np.errstate(all="ignore"):
        expected = reference.unpack_mxfp4(packed, scales)
        actual = decoder(packed, scales)
    assert actual.tobytes() == expected.tobytes()
    assert decoder.fallbacks == 1
    if scale >= 253:
        with np.errstate(over="raise"), pytest.raises(FloatingPointError):
            decoder(packed, scales)
        with np.errstate(over="warn"), pytest.warns(RuntimeWarning, match="overflow"):
            decoder(packed, scales)


def test_empty_dimensions_keep_existing_behavior(decoder):
    packed, scales = np.zeros((3, 0), dtype=np.uint8), np.zeros((3, 0), dtype=np.uint8)
    expected = reference.unpack_mxfp4(packed, scales)
    actual = decoder(packed, scales)
    assert (actual.shape, actual.dtype, actual.tobytes()) == (
        expected.shape,
        expected.dtype,
        expected.tobytes(),
    )
    empty_rows = np.empty((0, 16), dtype=np.uint8), np.empty((0, 1), dtype=np.uint8)
    with pytest.raises(ValueError):
        reference.unpack_mxfp4(*empty_rows)
    with pytest.raises(ValueError):
        decoder(*empty_rows)


@pytest.mark.parametrize(
    "kind", ["dtype", "ndim", "width", "scale_shape", "reserved", "scale_dtype"]
)
def test_invalid_storage_is_rejected_like_the_reference(decoder, kind):
    packed, scales = np.zeros((1, 16), dtype=np.uint8), np.full((1, 1), 127, dtype=np.uint8)
    if kind == "dtype":
        packed = packed.astype(np.int8)
    elif kind == "ndim":
        packed = packed.reshape(-1)
    elif kind == "width":
        packed = packed[:, :15]
    elif kind == "scale_shape":
        scales = scales.reshape(-1)
    elif kind == "reserved":
        scales[0, 0] = 255
    elif kind == "scale_dtype":
        scales = scales.astype(np.int32)
    for operation in (reference.unpack_mxfp4, decoder):
        with pytest.raises(DiagnosticError):
            operation(packed, scales)


@pytest.mark.parametrize("profile", ["weight-only-bf16", "radiance-fp8"])
def test_all_hybrid_boundaries_logits_and_state_match(module, decoder, tmp_path, profile):
    from conformance_fixture import tiny_checkpoint

    from qwen_r9700_lab.conformance_model import Checkpoint, QuantizedQwenReference

    root = tmp_path / "checkpoint"
    checkpoint = Checkpoint(root, tiny_checkpoint(root))
    observations = [[], []]
    settings = {
        "kv_scales": {"3": [1.0, 1.0]},
        "contract": "0" * 64,
        "execution": "1" * 64,
        "adapter": "2" * 64,
        "reference_profile": profile,
    }

    def record(which):
        def capture(position, layer, stage, value):
            observations[which].append(
                (position, layer, stage, value.shape, value.dtype.str, value.tobytes())
            )

        return capture

    baseline = QuantizedQwenReference(checkpoint, capture=record(0), **settings)
    candidate = module.LookupQuantizedQwenReference(
        checkpoint, unpacker=decoder, capture=record(1), **settings
    )
    try:
        for token in (1, 4, 8, 7, 9, 13, 3):
            assert baseline.step(token).tobytes() == candidate.step(token).tobytes()
            assert baseline.tokens == candidate.tokens
            assert observations[0] == observations[1]
            for layer, values in baseline.state.items():
                for name, value in values.items():
                    actual = candidate.state[layer][name]
                    assert (actual.shape, actual.dtype, actual.tobytes()) == (
                        value.shape,
                        value.dtype,
                        value.tobytes(),
                    )
        assert decoder.calls > 100 and decoder.fallbacks == 0
        assert len(observations[0]) > 500
    finally:
        checkpoint.close()


def test_reference_projection_revision_cannot_be_silently_substituted(module, decoder, monkeypatch):
    def changed_project(self, name, x):
        raise AssertionError("must reject before executing another reference revision")

    monkeypatch.setattr(module.QuantizedQwenReference, "project", changed_project)
    with pytest.raises(DiagnosticError, match="projection changed"):
        module.LookupQuantizedQwenReference(unpacker=decoder)

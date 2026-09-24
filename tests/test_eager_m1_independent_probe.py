"""CPU validation of the native audit's admission and independent comparator.

These tests do not qualify the GPU kernels or repair their numerical defects.
"""

import hashlib
import importlib
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiments/radiance-public"))
    return importlib.import_module("probe_eager_m1_independent")


@pytest.mark.parametrize(
    "metrics",
    [
        'vllm:num_requests_running{engine="0"} 1\nvllm:num_requests_waiting{engine="0"} 0',
        'vllm:num_requests_running{engine="0"} 0\nvllm:num_requests_waiting{engine="0"} 1',
        'vllm:num_requests_running{engine="0"} 0\nvllm:num_requests_running{engine="1"} 0',
        "vllm:num_requests_running 0\nvllm:num_requests_waiting NaN",
        "vllm:num_requests_running 0\nvllm:num_requests_waiting +Inf",
        "",
    ],
)
def test_busy_missing_or_nonfinite_counters_reject(probe, monkeypatch, metrics):
    monkeypatch.setattr(
        probe.urllib.request, "urlopen", lambda *_a, **_k: io.BytesIO(metrics.encode())
    )
    with pytest.raises(RuntimeError, match="busy or request counters are missing"):
        probe.idle("http://invalid")


def test_idle_requires_both_counter_types_for_all_reported_engines(probe, monkeypatch):
    metrics = "\n".join(
        f'vllm:num_requests_{kind}{{engine="{engine}"}} 0.0'
        for kind in ("running", "waiting")
        for engine in (0, 1)
    )
    monkeypatch.setattr(
        probe.urllib.request, "urlopen", lambda *_a, **_k: io.BytesIO(metrics.encode())
    )
    probe.idle("http://invalid/")


@pytest.mark.parametrize("admit", [False, True])
def test_missing_admission_or_lease_stops_before_creating_output(
    probe, monkeypatch, tmp_path, admit
):
    monkeypatch.delenv("QWEN_CONFORMANCE_GPU_LOCK", raising=False)
    output = tmp_path / "output"
    argv = [
        "probe",
        "--model",
        "/unused",
        "--gemm-build",
        "/unused",
        "--audit",
        "/unused",
        "--output",
        str(output),
    ]
    if admit:
        argv.append("--allow-gpu")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="explicit GPU admission and shared lease"):
        probe.main()
    assert not output.exists()


def test_numeric_comparator_detects_one_bit_state_corruption(probe):
    import torch

    expected = torch.tensor([1.0, 2.0])
    actual = expected.clone()
    actual.view(torch.int32)[0] ^= 1
    row = probe.compare(torch, actual, expected)
    assert row["different"] == 1
    assert row["elements"] == 2
    assert row["max_abs"] == 2**-23
    assert probe.compare(torch, expected, expected)["different"] == 0


def test_comparator_rejects_broadcasting_dtype_mismatch_and_nonfinite(probe):
    import torch

    expected = torch.ones(2)
    for actual in (
        torch.ones(1),
        expected.double(),
        expected[:0],
        torch.tensor([1.0, float("nan")]),
        torch.tensor([1.0, float("inf")]),
    ):
        with pytest.raises(ValueError):
            probe.compare(torch, actual, expected)


def test_independent_decoder_preserves_every_e2m1_code_and_stored_scale(probe):
    import torch

    packed = torch.tensor(
        [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2], dtype=torch.uint8
    )
    expected = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6] * 2,
        dtype=torch.float64,
    ).reshape(1, 32)
    actual = probe.decode_mxfp4_rows(
        torch, packed, torch.tensor([[127]], dtype=torch.uint8)
    )
    assert torch.equal(actual, expected)
    assert torch.signbit(actual[0, 8])
    assert torch.equal(
        probe.decode_mxfp4_rows(
            torch, packed, torch.tensor([[113]], dtype=torch.uint8)
        ),
        expected * 2**-14,
    )
    # The audited row's coefficient must survive despite another block's much
    # larger exponent. The oracle never folds it into that block's FP8 scale.
    both = packed.repeat(1, 2)
    scaled = probe.decode_mxfp4_rows(
        torch, both, torch.tensor([[113, 122]], dtype=torch.uint8)
    )
    assert scaled[0, 1].item() == 0.000030517578125
    assert scaled[0, 33].item() == 0.5 * 2**-5


def test_decoder_rejects_invalid_mxfp4_input(probe):
    import torch

    for packed, scale in (
        (torch.zeros((1, 16)), torch.ones((1, 1), dtype=torch.uint8)),
        (
            torch.zeros((1, 16), dtype=torch.uint8),
            torch.tensor([[255]], dtype=torch.uint8),
        ),
        (
            torch.zeros((1, 17), dtype=torch.uint8),
            torch.tensor([[127]], dtype=torch.uint8),
        ),
    ):
        with pytest.raises(ValueError):
            probe.decode_mxfp4_rows(torch, packed, scale)


def test_checkpoint_reader_rejects_changed_public_tensor(probe, tmp_path):
    import torch
    from safetensors.torch import save_file

    value = torch.tensor([1.0, -3.078125], dtype=torch.bfloat16)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"bias": "test.safetensors"}})
    )
    save_file({"bias": value}, tmp_path / "test.safetensors")
    identity = {
        "bias": {
            "shape": [2],
            "sha256": hashlib.sha256(
                value.view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        }
    }
    read = probe.checkpoint_reader(tmp_path, identity)
    assert torch.equal(read("bias"), value)
    value[0] = 2
    save_file({"bias": value}, tmp_path / "test.safetensors")
    with pytest.raises(RuntimeError, match="differs from audit"):
        read("bias")

"""Experimental head replay must retain admission and reject stale bindings."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_r9700_lab.diagnostic_contract import DiagnosticError


@pytest.fixture
def head_class():
    path = Path(__file__).parents[1] / "experiments/radiance-public/speed_graph_head.py"
    spec = importlib.util.spec_from_file_location("speed_graph_head_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GraphHead


@pytest.mark.parametrize("reason", ["disabled", "bias", "shape", "outer_capture"])
def test_head_replay_keeps_original_fallback(monkeypatch, head_class, reason):
    calls = []
    head = head_class(lambda *args: calls.append(args) or "original")
    head.warmups = 2
    value = SimpleNamespace(
        shape=(1, 5120) if reason == "shape" else (8, 5120),
        dtype="bf16",
        is_cuda=True,
        is_contiguous=lambda: True,
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            bfloat16="bf16",
            cuda=SimpleNamespace(
                is_current_stream_capturing=lambda: reason == "outer_capture"
            ),
        ),
    )
    head.enabled = reason != "disabled"
    bias = object() if reason == "bias" else None
    assert head("weights", value, bias) == "original"
    assert calls == [("weights", value, bias)]
    assert head.graph is None and head.replays == 0


def test_head_replay_rejects_changed_weight_allocation(monkeypatch, head_class):
    head = head_class(lambda *_: pytest.fail("stale graph must not publish"))
    head.warmups = 2
    head.graph = object()
    head.weight_identity = (1, (248320, 5120), (5120, 1), "bf16", "gpu")
    weight = SimpleNamespace(
        data_ptr=lambda: 2,
        shape=(248320, 5120),
        stride=lambda: (5120, 1),
        dtype="bf16",
        device="gpu",
    )
    value = SimpleNamespace(
        shape=(8, 5120), dtype="bf16", is_cuda=True, is_contiguous=lambda: True
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            bfloat16="bf16",
            cuda=SimpleNamespace(is_current_stream_capturing=lambda: False),
        ),
    )
    with pytest.raises(DiagnosticError, match="weights changed"):
        head(SimpleNamespace(weight=weight), value)
    assert head.replays == 0


def test_head_requires_320_checked_rows(head_class):
    with pytest.raises(ValueError, match="320"):
        head_class(None, checks=1)

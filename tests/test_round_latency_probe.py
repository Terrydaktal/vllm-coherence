"""Keep diagnostic profiling from silently clearing the state being measured."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_probe():
    path = Path(__file__).parents[1] / "experiments/radiance-public/round_latency_probe.py"
    spec = importlib.util.spec_from_file_location("round_latency_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RoundLatencyProbe()


def test_raw_profile_does_not_presynchronize(monkeypatch, tmp_path):
    calls = []

    class FakeProfile:
        def start(self):
            calls.append("start")

        def stop(self):
            calls.append("stop")

        def export_chrome_trace(self, path):
            calls.append(("export", path))

    def make_profile(**kwargs):
        assert kwargs["activities"] == ["cpu", "gpu"]
        assert not kwargs["record_shapes"]
        assert not kwargs["profile_memory"]
        assert not kwargs["with_stack"]
        return FakeProfile()

    # No cuda attribute: any explicit CUDA synchronization in the wrapper fails.
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(profiler=SimpleNamespace(
        profile=make_profile,
        ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="gpu"),
    )))
    probe = load_probe()
    root = tmp_path / "trace"
    result = probe.qwen_round_latency_profile("start", str(root))
    assert result["explicit_presynchronization"] is False
    assert calls == ["start"]
    with pytest.raises(RuntimeError, match="already active"):
        probe.qwen_round_latency_profile("start", str(tmp_path / "second"))
    probe.qwen_round_latency_profile("stop", str(root))
    assert calls == ["start", "stop", ("export", str(root / "profile-trace.json"))]
    assert not hasattr(probe, "_round_latency_profile")


def test_invalid_profile_mode_cannot_create_trace(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    root = tmp_path / "trace"
    with pytest.raises(ValueError, match="unsupported profile mode"):
        load_probe().qwen_round_latency_profile("start", str(root), mode="unknown")
    assert not root.exists()

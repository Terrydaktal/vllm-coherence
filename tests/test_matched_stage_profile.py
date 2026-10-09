import importlib.util
import sys
import types
from pathlib import Path

import pytest


def load(monkeypatch, *, failure=None):
    calls = []

    class Base:
        def execute_model(self, scheduled, *args, **kwargs):
            calls.append(("execute", scheduled.total_num_scheduled_tokens))
            return "unchanged"

        def qwen_optimized_metadata(self):
            return {"enforce_eager": False}

    class Observer:
        def __init__(self, runner, root, profile):
            self.root = root
            self.profile_active = False
            self.profiler = None

        def start_profile(self):
            calls.append(("start",))
            self.profile_active = True

        def stop_profile(self):
            calls.append(("stop",))
            if failure == "stop":
                raise RuntimeError("injected profiler stop failure")
            self.profile_active = False
            if failure == "export":
                raise OSError("injected trace export failure")

        def close(self):
            return {"profile_files": [str(self.root / "profile-trace.json")]}

    import contextlib
    monkeypatch.setitem(sys.modules, "optimized_d7_worker", types.SimpleNamespace(
        GraphObservation=Observer, OptimizedWorker=Base))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        profiler=types.SimpleNamespace(record_function=lambda _: contextlib.nullcontext())))
    path = Path(__file__).parents[1] / "experiments/radiance-public/matched_stage_profile_worker.py"
    spec = importlib.util.spec_from_file_location("matched_worker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    worker = module.MatchedStageWorker()
    worker.model_runner = object()
    return worker, calls


def test_control_never_activates_profiler_or_changes_model_return(tmp_path, monkeypatch):
    worker, calls = load(monkeypatch)

    def forbidden_collect():
        pytest.fail("an unprofiled control attempted profiler garbage collection")

    monkeypatch.setitem(
        worker.observation_class.stop_profile.__globals__,
        "gc", types.SimpleNamespace(collect=forbidden_collect),
    )
    worker.qwen_timing_arm(str(tmp_path / "control"), "control", 2, 3, 6)
    for count in (2048, 2048, 8, 8, 8, 8, 8):
        assert worker.execute_model(types.SimpleNamespace(total_num_scheduled_tokens=count)) == "unchanged"
    report = worker.qwen_timing_finish()
    assert calls == [("execute", n) for n in (2048, 2048, 8, 8, 8, 8, 8)]
    assert len(report["rows"]) == 5
    assert not any(r["profiled"] for r in report["rows"])
    assert report["chunks"] == []
    assert not report["timing_contract"]["forced_replay_hooks"]


@pytest.mark.parametrize("failure", ["stop", "export"])
def test_failed_profile_finish_blocks_the_next_control_arm(tmp_path, monkeypatch, failure):
    worker, _calls = load(monkeypatch, failure=failure)
    worker.qwen_timing_arm(str(tmp_path / "profile"), "profile", 0, 3, 3)
    worker.execute_model(types.SimpleNamespace(total_num_scheduled_tokens=8))

    error_type = RuntimeError if failure == "stop" else OSError
    error_message = "profiler stop" if failure == "stop" else "trace export"
    with pytest.raises(error_type, match=f"injected {error_message} failure"):
        worker.qwen_timing_finish()

    assert worker.qwen_timing_status()["arm_active"]
    assert worker._timing_run["observer"] is not None
    control_path = tmp_path / "control"
    with pytest.raises(ValueError, match="unfinished run"):
        worker.qwen_timing_arm(str(control_path), "control", 0, 3, 3)
    assert not control_path.exists()


def test_profile_setup_and_export_are_outside_declared_round_windows(tmp_path, monkeypatch):
    worker, calls = load(monkeypatch)
    worker.qwen_timing_arm(str(tmp_path / "profile"), "profile", 2, 3, 6)
    for _ in range(12):
        worker.execute_model(types.SimpleNamespace(total_num_scheduled_tokens=8))
    report = worker.qwen_timing_finish()
    assert [c["first_decode_index"] for c in report["chunks"]] == [2, 6]
    assert [c["end_exclusive"] for c in report["chunks"]] == [6, 10]
    assert [r["decode_index"] for r in report["rows"] if r["profiled"]] == list(range(2, 10))
    assert [r[0] for r in calls].count("start") == 2
    assert [r[0] for r in calls].count("stop") == 2
    assert not hasattr(worker, "_timing_run")


def test_finish_closes_partial_chunk_without_claiming_missing_rounds(tmp_path, monkeypatch):
    worker, calls = load(monkeypatch)
    worker.qwen_timing_arm(str(tmp_path / "partial"), "profile", 2, 8, 16)
    for _ in range(5):
        worker.execute_model(types.SimpleNamespace(total_num_scheduled_tokens=8))
    report = worker.qwen_timing_finish()
    assert len(report["rows"]) == 5
    assert len(report["chunks"]) == 1
    assert calls[-1] == ("stop",)


def test_capture_identity_is_stable_for_one_worker_and_changes_on_restart(monkeypatch):
    worker, calls = load(monkeypatch)
    worker.vllm_config = types.SimpleNamespace(model_config=types.SimpleNamespace(
        model="model", revision="pinned", tokenizer="tokenizer", dtype="bfloat16"))
    monkeypatch.setenv("RADIANCE_VERIFY_HEAD_GLOBAL_TOPK", "512")
    first = worker.qwen_timing_identity()
    second = worker.qwen_timing_identity()
    assert first == second
    assert first["target_head_environment"]["RADIANCE_VERIFY_HEAD_GLOBAL_TOPK"] == "512"
    replacement, _ = load(monkeypatch)
    assert replacement.qwen_timing_status()["instance"] != first["instance"]
    assert calls == []  # Identity checks neither execute a model nor profile it.

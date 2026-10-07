from __future__ import annotations

import gc
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_r9700_lab import radiance_cache as cache
from qwen_r9700_lab import radiance_cache_telemetry as telemetry

ROOT = Path(__file__).resolve().parents[1]
CHAT = {"id": "a" * 64, "generation": "b" * 64}
REQUEST = hashlib.sha256(b"request").hexdigest()


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    value = telemetry.Recorder(tmp_path / "status", start=False, gc_events=False)
    monkeypatch.setattr(telemetry, "_recorder", value)
    monkeypatch.setattr(telemetry, "_round", {})
    monkeypatch.setattr(telemetry, "_gpu_origins", {})
    monkeypatch.setattr(telemetry, "_completion_baseline", None)
    yield value
    value.close()


def rows(value):
    value.flush()
    while value.pending:
        value.flush()
    return [json.loads(line) for line in value.path.read_text().splitlines()]


def test_main_thread_interval_covers_time_outside_forward(recorder, monkeypatch):
    samples = deque(
        [(1_000_000, 0.01, 0.005, 10, 2, 3, 4), (6_000_000, 0.013, 0.007, 13, 4, 6, 5)]
    )
    monkeypatch.setattr(telemetry, "_thread_snapshot", lambda: samples.popleft())
    context = {
        "chat_id": CHAT["id"],
        "generation": CHAT["generation"],
        "request_id": REQUEST,
    }
    telemetry.complete_round({**context, "round": 1}, 10_000_000, contiguous=False)
    telemetry.complete_round({**context, "round": 2}, 110_000_000, contiguous=True)
    (event,) = rows(recorder)
    assert event["stage"] == "main_thread_round" and event["duration_ms"] == 100
    assert event["thread_cpu_ms"] == 5 and event["off_cpu_ms"] == 95
    assert event["thread_user_ms"] == 3 and event["thread_system_ms"] == 2
    assert event["minor_faults"] == 3 and event["major_faults"] == 2
    assert event["voluntary_switches"] == 3 and event["involuntary_switches"] == 1
    assert event["thread_id"] == threading.get_native_id()


@pytest.mark.parametrize(
    "boundary", ["handover", "request", "generation", "round", "thread"]
)
def test_main_thread_counters_never_bridge_an_unrelated_interval(
    recorder, monkeypatch, boundary
):
    context = {
        "chat_id": CHAT["id"],
        "generation": CHAT["generation"],
        "request_id": REQUEST,
        "round": 1,
    }
    monkeypatch.setattr(telemetry, "_thread_snapshot", lambda: (0, 0, 0, 0, 0, 0, 0))
    telemetry.complete_round(context, 1, contiguous=False)
    changed = {**context, "round": 2}
    if boundary in ("request", "generation"):
        changed["request_id" if boundary == "request" else "generation"] = "f" * 64
    elif boundary == "round":
        changed["round"] = 3
    elif boundary == "thread":
        monkeypatch.setattr(telemetry.threading, "get_native_id", lambda: -1)
    telemetry.complete_round(changed, 1_000_000_000, contiguous=boundary != "handover")
    assert not recorder.pending and recorder.main_thread_samples == 0


def test_thread_counter_failure_does_not_escape_or_fake_a_zero(recorder, monkeypatch):
    monkeypatch.setattr(
        telemetry, "_thread_snapshot", lambda: (_ for _ in ()).throw(OSError())
    )
    telemetry.complete_round({"round": 1}, 1, contiguous=False)
    assert recorder.round_sample_errors == 1 and telemetry._completion_baseline is None
    assert not recorder.pending


def fake_round_cuda():
    calls = []
    state = SimpleNamespace(
        ready=False, marker=0, stream=SimpleNamespace(cuda_stream=7)
    )

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing
            self.marker = None

        def record(self, stream):
            calls.append("record")
            assert stream.cuda_stream == 7
            state.marker += 1
            self.marker = state.marker

        def query(self):
            calls.append("query")
            assert self.marker is not None
            return state.ready

        def elapsed_time(self, other):
            calls.append("elapsed")
            assert state.ready and self.marker is not None and other.marker is not None
            return float(other.marker - self.marker)

        def synchronize(self):
            pytest.fail("diagnostics must never synchronize")

    return (
        SimpleNamespace(Event=Event, current_stream=lambda: state.stream),
        state,
        calls,
    )


def round_context(number=1, **changes):
    return {
        "request_id": REQUEST,
        "chat_id": CHAT["id"],
        "generation": CHAT["generation"],
        "round": number,
        **changes,
    }


def test_gpu_round_queries_are_deferred_and_event_slots_survive_reuse(recorder):
    cuda, state, calls = fake_round_cuda()
    sampler = recorder.round_sampler = telemetry.GpuRoundTimings(
        cuda, recorder, capacity=3
    )
    first = sampler.begin(round_context(1))
    sampler.end(first, success=True, reason="sample_return")
    second = sampler.begin(round_context(2))
    sampler.end(second, success=True, reason="sample_return")
    assert calls == ["record"] * 4
    sampler.collect()
    assert not recorder.pending and len(sampler.pending) == 2
    state.ready = True
    events = rows(recorder)
    assert len(events) == 2 and all(e["gpu_elapsed_ms"] == 1 for e in events)
    assert "gpu_inter_round_gap_ms" not in events[0]
    assert events[1]["gpu_inter_round_gap_ms"] == 1
    # More than twice the production ring capacity: no quiet stop at 64 rounds,
    # and no reading a previous end marker after it has been overwritten.
    for number in range(3, 140):
        sampler.end(
            sampler.begin(round_context(number)), success=True, reason="sample_return"
        )
        sampler.collect()
    events = [event for event in rows(recorder) if "gpu_elapsed_ms" in event]
    assert len(events) == 139 and sampler.completed == 139
    assert all(e.get("gpu_inter_round_gap_ms") == 1 for e in events[1:])
    assert sampler.health() == {
        "capacity": 3,
        "pending": 0,
        "free": 2,
        "completed": 139,
        "dropped": 0,
        "errors": 0,
        "quarantined": 0,
    }


def test_gpu_round_pool_is_bounded_and_skips_instead_of_waiting(recorder):
    cuda, state, _ = fake_round_cuda()
    sampler = telemetry.GpuRoundTimings(cuda, recorder, capacity=2)
    for number in (1, 2):
        sampler.end(
            sampler.begin(round_context(number)), success=True, reason="sample_return"
        )
    assert sampler.begin(round_context(3)) is None and sampler.dropped == 1
    assert not sampler.free and len(sampler.pending) == 2
    state.ready = True
    sampler.collect()
    sampler.end(sampler.begin(round_context(4)), success=True, reason="sample_return")
    sampler.collect()
    assert "gpu_inter_round_gap_ms" not in rows(recorder)[-1]


@pytest.mark.parametrize(
    "boundary", ["request", "generation", "round", "stream", "failed"]
)
def test_gpu_gaps_are_not_reported_across_unrelated_rounds(recorder, boundary):
    cuda, state, _ = fake_round_cuda()
    state.ready = True
    sampler = telemetry.GpuRoundTimings(cuda, recorder, capacity=3)
    sampler.end(
        sampler.begin(round_context(1)),
        success=boundary != "failed",
        reason="sample_return",
    )
    sampler.collect()
    context = round_context(2)
    if boundary in ("request", "generation"):
        context["request_id" if boundary == "request" else boundary] = "f" * 64
    elif boundary == "round":
        context["round"] = 3
    item = sampler.begin(context)
    if boundary == "stream":
        state.stream = SimpleNamespace(cuda_stream=8)
    sampler.end(item, success=True, reason="sample_return")
    sampler.collect()
    event = rows(recorder)[-1]
    assert "gpu_inter_round_gap_ms" not in event
    if boundary == "stream":
        assert event["stream_match"] is False and "gpu_elapsed_ms" not in event


def test_failed_gpu_query_quarantines_markers_without_waiting(recorder):
    cuda, _, _ = fake_round_cuda()
    sampler = telemetry.GpuRoundTimings(cuda, recorder, capacity=2)
    item = sampler.begin(round_context())
    sampler.end(item, success=True, reason="sample_return")
    item["pair"][1].query = lambda: (_ for _ in ()).throw(RuntimeError())
    sampler.collect()
    assert (
        sampler.errors == 1 and len(sampler.quarantined) == 1 and len(sampler.free) == 1
    )
    assert not sampler.pending and not recorder.pending


def test_round_hooks_keep_model_results_arguments_and_exceptions(recorder):
    cuda, state, calls = fake_round_cuda()
    result, failure = object(), RuntimeError("original model error")
    observed = []

    class Runner:
        def execute_model(self, output, *args, **kwargs):
            observed.append((output, args, kwargs))
            self.execute_model_state = object()

        def sample_tokens(self, grammar_output):
            self.execute_model_state = None
            if grammar_output is failure:
                raise failure
            return result

    telemetry.install_round_hooks(Runner, cuda)
    telemetry.install_round_hooks(Runner, cuda)
    runner = Runner()
    output = SimpleNamespace(
        total_num_scheduled_tokens=8,
        num_scheduled_tokens={"private": 8},
        qwen_fair={"cache_timing_context": round_context(text="SECRET")},
    )
    intermediate = object()
    assert runner.execute_model(output, intermediate, dummy_run=False) is None
    assert runner.sample_tokens(None) is result
    assert observed == [(output, (intermediate,), {"dummy_run": False})]
    assert calls == ["record", "record"]
    output.qwen_fair["cache_timing_context"]["round"] = 2
    runner.execute_model(output)
    with pytest.raises(RuntimeError) as caught:
        runner.sample_tokens(failure)
    assert caught.value is failure
    state.ready = True
    events = rows(recorder)
    events = [e for e in events if e["stage"] == "gpu_round"]
    assert len(events) == 2 and events[0]["success"] and not events[1]["success"]
    assert events[1]["reason"] == "sample_failed"
    assert "SECRET" not in recorder.path.read_text()
    assert runner._qwen_cache_round_item is None


def test_approved_kfd_capture_starts_once_on_owned_real_forward(recorder, monkeypatch):
    from qwen_r9700_lab import radiance_kfd_trace

    monkeypatch.setenv("QWEN_KFD_CAPTURE_PATH", "/operator-approved/events.kfd")
    captures = []
    monkeypatch.setattr(radiance_kfd_trace, "start_if_approved", captures.append)
    cuda, _, _ = fake_round_cuda()

    class Runner:
        execute_model_state = None

        def execute_model(self, output, *args, **kwargs):
            return "unchanged"

        def sample_tokens(self, *args):
            return "unchanged"

    telemetry.install_round_hooks(Runner, cuda)
    output = SimpleNamespace(
        total_num_scheduled_tokens=8,
        num_scheduled_tokens={"private": 8},
        qwen_fair={"cache_timing_context": round_context()},
    )
    runner = Runner()
    assert runner.execute_model(output, dummy_run=True) == "unchanged"
    assert captures == []
    assert runner.execute_model(output) == "unchanged"
    assert runner.execute_model(output) == "unchanged"
    assert captures == [recorder.trace_id]


@pytest.mark.parametrize(
    "excluded", ["dummy", "barrier", "prefill", "batch", "unowned"]
)
def test_round_hooks_do_not_instrument_startup_handover_or_prefill(recorder, excluded):
    cuda, _, calls = fake_round_cuda()

    class Runner:
        execute_model_state = None

        def execute_model(self, output, **kwargs):
            return output

        def sample_tokens(self, value):
            return value

    telemetry.install_round_hooks(Runner, cuda)
    output = SimpleNamespace(
        total_num_scheduled_tokens=8,
        num_scheduled_tokens={"a": 8},
        qwen_fair={"cache_timing_context": round_context()},
    )
    if excluded == "barrier":
        output.qwen_fair["barrier"] = True
    elif excluded == "prefill":
        output.total_num_scheduled_tokens = 4096
    elif excluded == "batch":
        output.num_scheduled_tokens["b"] = 8
    elif excluded == "unowned":
        output.qwen_fair = None
    runner = Runner()
    assert runner.execute_model(output, dummy_run=excluded == "dummy") is output
    assert not calls and not recorder.round_sampler.pending


def test_generation_diagnostic_switch_disables_cpu_and_gpu_work(recorder, monkeypatch):
    monkeypatch.setenv("QWEN_GENERATION_ROUND_TELEMETRY", "0")
    monkeypatch.setattr(
        telemetry, "_thread_snapshot", lambda: pytest.fail("disabled CPU sampler")
    )

    class Runner:
        pass

    telemetry.install_round_hooks(Runner, object())
    telemetry.complete_round(round_context(), 1, contiguous=False)
    assert (
        not hasattr(Runner, "_qwen_cache_round_hooks")
        and recorder.round_sampler is None
    )
    assert not recorder.pending


@pytest.mark.parametrize("fault", ["event", "counter", "metadata", "execute"])
def test_round_diagnostic_faults_cannot_replace_the_models_result(
    recorder, monkeypatch, fault
):
    cuda, state, _ = fake_round_cuda()
    output_value, failure = object(), RuntimeError("original execute failure")

    class Runner:
        execute_model_state = None

        def execute_model(self, output):
            if fault == "execute":
                raise failure
            return output_value

        def sample_tokens(self, value):
            return value

    telemetry.install_round_hooks(Runner, cuda)
    output = SimpleNamespace(
        total_num_scheduled_tokens=8,
        num_scheduled_tokens={"a": 8},
        qwen_fair={"cache_timing_context": round_context()},
    )
    if fault == "event":
        cuda.current_stream = lambda: (_ for _ in ()).throw(OSError())
    elif fault == "counter":
        monkeypatch.setattr(
            telemetry, "_thread_snapshot", lambda: (_ for _ in ()).throw(OSError())
        )
    elif fault == "metadata":
        output.qwen_fair = object()
    runner = Runner()
    if fault == "execute":
        with pytest.raises(RuntimeError) as caught:
            runner.execute_model(output)
        assert caught.value is failure
        state.ready = True
        assert rows(recorder)[0]["success"] is False
    else:
        assert runner.execute_model(output) is output_value
        assert recorder.round_sample_errors + recorder.round_sampler.errors == 1


def test_slow_marker_query_is_visible_and_does_not_block_the_producer(
    recorder, monkeypatch
):
    cuda, state, calls = fake_round_cuda()
    state.ready = True
    sampler = recorder.round_sampler = telemetry.GpuRoundTimings(
        cuda, recorder, capacity=2
    )
    item = sampler.begin(round_context())
    sampler.end(item, success=True, reason="sample_return")
    assert calls == ["record", "record"]
    times = deque([100_000_000, 107_000_000])
    with monkeypatch.context() as clocks:
        clocks.setattr(telemetry.time, "monotonic_ns", lambda: times.popleft())
        sampler.collect()
    assert (
        next(e for e in rows(recorder) if e["stage"] == "gpu_observer_poll")[
            "duration_ms"
        ]
        == 7
    )


def test_task_captures_origin_and_keeps_active_chat_separate(recorder):
    telemetry.begin_round({"request_id": REQUEST, "round": 37, **CHAT})
    context = telemetry.request_context(
        "request", CHAT, 19, is_store=True, size=123, block_count=1
    )
    telemetry.begin_round({"request_id": "c" * 64, "round": 4})
    result = telemetry.task(
        lambda: "unchanged result", context, time.monotonic_ns() - 20_000_000
    )
    assert result == "unchanged result"
    job = next(r for r in rows(recorder) if r["stage"] == "cpu_job")
    assert job["request_id"] == REQUEST and job["round"] == 37
    assert job["active"]["request_id"] == "c" * 64
    assert job["thread_cpu_ms"] >= 0
    assert job["minor_faults"] >= 0
    assert (
        next(r for r in rows(recorder) if r["stage"] == "cpu_queue")["duration_ms"]
        >= 20
    )
    assert telemetry._local.context == {}
    other = telemetry.request_context(
        "other", CHAT, 20, is_store=False, size=1, block_count=1
    )
    assert "round" not in other  # unrelated request must not inherit the active round


def test_snapshot_encoding_and_payload_are_unchanged(recorder, tmp_path, monkeypatch):
    data = bytes(range(256)) * 32
    key = "g0-" + "1" * 64 + ".qkv"
    with monkeypatch.context() as inactive:
        inactive.setattr(telemetry, "_recorder", None)
        expected = cache.encode_block(data)
    assert cache.encode_block(data) == expected
    store = cache.ChatStore(tmp_path, CHAT)
    store.activate()
    context = telemetry.request_context(
        "request", CHAT, 1, is_store=True, size=len(data), block_count=1
    )
    assert telemetry.task(
        lambda: store.write(key, memoryview(data)), context, time.monotonic_ns()
    )
    assert store.read(key, len(data)) == data
    names = {r["stage"] for r in rows(recorder)}
    assert {
        "compression",
        "checksum",
        "disk_write",
        "file_fsync",
        "directory_fsync",
        "disk_read",
    } <= names


def test_existing_lock_wait_is_timed_without_changing_locking(recorder):
    mutex = threading.Lock()
    mutex.acquire()
    acquired = threading.Event()

    def take():
        with telemetry.lock(mutex):
            acquired.set()

    thread = threading.Thread(target=take)
    thread.start()
    assert not acquired.wait(0.02)
    mutex.release()
    thread.join(timeout=1)
    assert acquired.is_set() and not mutex.locked()
    assert (
        next(r for r in rows(recorder) if r["stage"] == "tier_lock")["duration_ms"]
        >= 20
    )


def test_faults_drops_and_sink_failures_do_not_change_operation(recorder, monkeypatch):
    recorder.capacity = 1
    telemetry.emit("round_start")
    telemetry.emit("round_start")
    assert len(recorder.pending) == 1 and recorder.dropped == 1
    recorder.lock.acquire()
    try:
        telemetry.emit("python_gc")  # models a callback interrupting a producer
    finally:
        recorder.lock.release()
    assert recorder.dropped == 2
    failure = ValueError("original operation failed")
    monkeypatch.setattr(
        recorder, "emit", lambda *_args, **_kw: (_ for _ in ()).throw(OSError())
    )

    def operation():
        raise failure

    with pytest.raises(ValueError) as caught:
        telemetry.task(operation, {}, time.monotonic_ns())
    assert caught.value is failure
    assert telemetry._local.context == {}
    recorder.path = recorder.path.parent / "missing" / "unwritable.jsonl"
    recorder.flush()
    assert recorder.write_errors >= 1
    assert json.loads(recorder.health_path.read_text())["dropped"] >= 3


def test_rotation_short_write_and_private_records(recorder, monkeypatch):
    recorder.max_bytes = 800
    original = telemetry.os.write
    monkeypatch.setattr(telemetry.os, "write", lambda fd, data: original(fd, data[:40]))
    for i in range(3):
        telemetry.emit(
            "round_start",
            round=i,
            prompt="SECRET CHAT",
            token_ids=[111],
            path="SECRET PATH",
            reason={"arbitrary": "not allowed"},
        )
        recorder.flush()
    assert recorder.path.exists() and Path(str(recorder.path) + ".1").exists()
    assert "SECRET" not in recorder.path.read_text()
    assert recorder.path.stat().st_mode & 0o077 == 0
    assert len(list(recorder.path.parent.glob("*.jsonl*"))) == 2
    assert recorder.write_errors == 0


def test_gc_callback_is_scoped_and_does_not_replace_existing_callbacks(tmp_path):
    prior = list(gc.callbacks)
    recorder = telemetry.Recorder(tmp_path / "gc", start=False)
    try:
        assert all(callback in gc.callbacks for callback in prior)
        recorder.gc_callback("start", {"generation": 2})
        recorder.gc_callback(
            "stop", {"generation": 2, "collected": 17, "uncollectable": 0}
        )
        event = next(r for r in rows(recorder) if r["stage"] == "python_gc")
        assert event["gc_generation"] == 2 and event["collected"] == 17
    finally:
        recorder.close()
    assert gc.callbacks == prior


@pytest.mark.parametrize("is_store", [True, False])
def test_gpu_hooks_reuse_native_timing_and_preserve_results_and_waits(
    recorder, is_store
):
    calls = []

    class Handler:
        gpu_to_cpu = is_store

        def __init__(self):
            self._transfers = deque()
            self.result = SimpleNamespace(job_id=71, success=True, transfer_time=0.0008)

        def transfer_async(self, job_id, src, dst):
            calls.append(("submit", job_id, src, dst))
            self._transfers.append(SimpleNamespace(num_bytes=4096))
            return True

        def get_finished(self):
            calls.append("native query and elapsed_time")
            return [self.result]

        def wait(self, ids):
            calls.append(("original wait", ids))
            return "wait result"

    telemetry.install_transfer_hooks(Handler)
    once = Handler.transfer_async
    telemetry.install_transfer_hooks(Handler)
    assert Handler.transfer_async is once
    telemetry.begin_round({"request_id": REQUEST, "round": 10})
    handler = Handler()
    gpu = SimpleNamespace(block_indices=[5, 9], group_sizes=[1, 2])
    source, destination = (gpu, object()) if is_store else (object(), gpu)
    assert handler.transfer_async(71, source, destination)
    assert handler.get_finished()[0] is handler.result
    ids = {71}
    assert handler.wait(ids) == "wait result"
    assert calls == [
        ("submit", 71, source, destination),
        "native query and elapsed_time",
        ("original wait", ids),
    ]
    event = next(r for r in rows(recorder) if r["stage"] == "gpu_complete")
    assert event["gpu_elapsed_ms"] == 0.8 and event["bytes"] == 4096
    assert event["logical_ranges"] == [[5, 1], [9, 2]]
    assert event["direction"] == ("store" if is_store else "load")
    # Transfer timing still reuses native measurements. The independent new
    # generation sampler adds markers, never a synchronize, for its own scope.
    source_text = inspect.getsource(telemetry.install_transfer_hooks)
    assert (
        ".synchronize(" not in source_text
        and ".record(" not in source_text
        and ".query(" not in source_text
    )
    assert ".synchronize(" not in Path(telemetry.__file__).read_text()


def test_connector_retains_origin_across_deferred_handover(recorder):
    class Connector:
        def prepare_store_kv(self, metadata):
            return metadata

        start_kv_transfers = handle_preemptions = prepare_store_kv

    telemetry.install_connector_hooks(Connector)
    telemetry.begin_round({"request_id": REQUEST, "round": 3})
    entry = SimpleNamespace(req_id="request")
    metadata = SimpleNamespace(store_jobs={9: entry}, load_jobs={})
    connector = Connector()
    assert connector.prepare_store_kv(metadata) is metadata
    telemetry.begin_round({"request_id": "d" * 64, "round": 8})
    assert telemetry._gpu_origins[9]["request_id"] == REQUEST
    assert telemetry._gpu_origins[9]["round"] == 3


def load_analyzer():
    spec = importlib.util.spec_from_file_location(
        "cache_job_analysis", ROOT / "tools/analyze_cache_job_telemetry.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analysis_joins_lifecycle_and_reports_nested_overlaps_and_drops():
    analyzer = load_analyzer()
    assert analyzer.union_ms([(0, 20_000_000), (1, 2), (10_000_000, 25_000_000)]) == 25
    round_row = {
        "pid": 17,
        "cache_trace_id": "new",
        "request_id": REQUEST,
        "round": 30,
        "round_ms": 100,
        "monotonic_ns": 200_000_000,
    }
    event = {
        "pid": 17,
        "trace_id": "new",
        "stage": "compression",
        "sequence": 2,
        "start_ns": 130_000_000,
        "end_ns": 160_000_000,
        "duration_ms": 30,
    }
    old = {**event, "trace_id": "old", "stage": "not this process"}
    gap = {**event, "sequence": 4, "start_ns": 165_000_000, "end_ns": 190_000_000}
    result = analyzer.analyze([round_row], [event, old, gap], {"dropped": 1})
    assert result["outliers"][0]["overlap_by_stage_ms"] == {"compression": 55}
    assert result["coverage"]["sequence_gaps"] == 1
    assert result["coverage"]["recorder_health"]["dropped"] == 1
    assert "does not establish" in result["interpretation"]


def test_real_background_writer_closes_and_disable_switch(tmp_path, monkeypatch):
    monkeypatch.setattr(telemetry, "_recorder", None)
    monkeypatch.setenv("QWEN_CACHE_JOB_TELEMETRY", "0")
    assert telemetry.configure(tmp_path / "disabled") is None
    assert not list(tmp_path.iterdir())
    monkeypatch.setenv("QWEN_CACHE_JOB_TELEMETRY", "1")
    recorder = telemetry.configure(tmp_path / "enabled")
    telemetry.emit("round_start", round=1)
    recorder.close()
    assert recorder.written >= 1
    assert not recorder.thread.is_alive()
    events = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    assert any(
        event["stage"] == "round_start" and event.get("round") == 1 for event in events
    )


@pytest.mark.parametrize("loading", ["script", "stdin", "flush_helper"])
def test_standalone_cache_tools_need_no_installed_lab_package(tmp_path, loading):
    (tmp_path / "radiance_cache.py").write_bytes(Path(cache.__file__).read_bytes())
    (tmp_path / "radiance_cache_telemetry.py").write_bytes(
        Path(telemetry.__file__).read_bytes()
    )
    if loading == "script":
        command, source = [sys.executable, "radiance_cache.py", "--help"], None
    elif loading == "stdin":
        command = [sys.executable, "-I", "-", "--help"]
        source = Path(cache.__file__).read_text()
    else:
        command = [sys.executable, "-I", "-"]
        source = (
            "import importlib.util\n"
            "s = importlib.util.spec_from_file_location('flush', 'radiance_cache.py')\n"
            "m = importlib.util.module_from_spec(s)\n"
            "s.loader.exec_module(m)\n"
            "assert callable(m.request_tail_flush)\n"
            "print(m.FORMAT)\n"
        )
    result = subprocess.run(
        command,
        input=source,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": ""},
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (
        "qwen-chat-cache-v1" if loading == "flush_helper" else "usage:"
    ) in result.stdout
    assert not list(tmp_path.glob("*jsonl"))

"""CPU boundary tests for the first-request work omitted by round sampling."""

import hashlib
import json
import sys
import weakref
from types import SimpleNamespace

import pytest
from test_cache_job_telemetry import fake_round_cuda
from test_radiance_fair_scheduler import load_module

from qwen_r9700_lab import radiance_cache_telemetry as telemetry


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    result = telemetry.Recorder(tmp_path / "timeline", start=False, gc_events=False)
    monkeypatch.setattr(telemetry, "_recorder", result)
    monkeypatch.setattr(telemetry, "_round", {})
    yield result
    result.close()


def test_scheduler_retains_anonymous_prefill_phases(monkeypatch, tmp_path):
    module = load_module(monkeypatch)
    recorder = telemetry.Recorder(tmp_path / "status", start=False, gc_events=False)
    monkeypatch.setattr(telemetry, "_recorder", recorder)
    now = [1.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    request = SimpleNamespace(
        request_id="anonymous-smoke-no-chat-label",
        sampling_params=None,
        kv_transfer_params=None,
        num_prompt_tokens=182600,
        num_computed_tokens=0,
    )
    phases = module.RequestPhases()
    phases.set(request, "admission")
    now[0] = 2
    phases.set(request, "gpu_queue")
    now[0] = 5
    phases.set(request, "cache_restore")
    now[0] = 17
    phases.set(request, "prefill")
    now[0] = 20
    request.num_computed_tokens = request.num_prompt_tokens
    phases.set(request, "generate")
    recorder.flush()
    events = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    assert [(e["stage"], e["duration_ms"]) for e in events if e["duration_ms"]] == [
        ("phase_admission", 1000),
        ("phase_gpu_queue", 3000),
        ("phase_cache_restore", 12000),
        ("phase_prefill", 3000),
    ]
    first = next(e for e in events if e["stage"] == "scheduler_first_output")
    assert first["start_ns"] == 20_000_000_000
    assert (
        first["request_id"] == hashlib.sha256(request.request_id.encode()).hexdigest()
    )
    assert phases.live[request.request_id]["first_token_ms"] == 19000
    assert request.request_id not in recorder.path.read_text()
    recorder.close()


@pytest.mark.parametrize("fail", [False, True])
def test_large_prefill_and_first_sample_have_host_spans_and_async_gpu_markers(
    recorder, fail
):
    cuda, state, calls = fake_round_cuda()
    error = RuntimeError("private exception")
    result = object()

    class Runner:
        def execute_model(self, output):
            self.execute_model_state = object()
            return result

        def sample_tokens(self, value):
            self.execute_model_state = None
            if fail:
                raise error
            return value

    telemetry.install_round_hooks(Runner, cuda)
    runner = Runner()
    output = SimpleNamespace(
        total_num_scheduled_tokens=2048,
        num_scheduled_tokens={"internal": 2048},
        qwen_fair={"cache_timing_context": {"request_id": "a" * 64, "round": 1}},
    )
    for _ in range(2):  # More than one prefill chunk before generation starts.
        assert runner.execute_model(output) is result
        if fail:
            with pytest.raises(RuntimeError) as caught:
                runner.sample_tokens(result)
            assert caught.value is error
        else:
            assert runner.sample_tokens(result) is result
    assert calls == ["record"] * 4  # Only markers; queries run on the writer.
    recorder.flush()
    events = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    assert [e["stage"] for e in events] == ["worker_execute", "worker_sample"] * 2
    assert all(
        e["request_id"] == "a" * 64 and e["scheduled_tokens"] == 2048 for e in events
    )
    assert all(
        e["success"] == (not fail) for e in events if e["stage"] == "worker_sample"
    )
    assert all("thread_cpu_ms" in e and "major_faults" in e for e in events)
    assert "private exception" not in recorder.path.read_text()
    assert runner._qwen_first_work_context == {}
    state.ready = True
    recorder.round_sampler.collect()
    recorder.flush()
    gpu = [
        json.loads(line)
        for line in recorder.path.read_text().splitlines()
        if json.loads(line)["stage"] == "gpu_prefill"
    ]
    assert len(gpu) == 2 and all(row["gpu_elapsed_ms"] == 1 for row in gpu)
    assert all(row["success"] == (not fail) for row in gpu)
    assert all("gpu_inter_prefill_gap_ms" not in row for row in gpu)
    output.qwen_fair["cache_timing_context"]["round"] = 2
    runner.execute_model(output)
    with pytest.raises(RuntimeError) if fail else telemetry._NO_SPAN:
        runner.sample_tokens(result)
    assert not recorder.pending


def test_timeline_sink_keeps_only_hashed_identity_and_numeric_metadata(recorder):
    telemetry.emit_at(
        "internal_id_bridge",
        1,
        2,
        request_id="a" * 64,
        external_request_id="b" * 64,
        http_request_id="c" * 64,
        status_code=200,
        streaming=True,
        body="private",
        error="private",
    )
    telemetry.emit_at("http_end", 2, 3, request_id="unsafe raw id")
    recorder.flush()
    events = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    assert events[0]["external_request_id"] == "b" * 64
    assert events[0]["http_request_id"] == "c" * 64
    assert events[0]["streaming"] and events[0]["status_code"] == 200
    assert "request_id" not in events[1]
    assert "private" not in recorder.path.read_text()


def test_first_work_does_not_attribute_a_batch_to_one_request(recorder):
    cuda, _, calls = fake_round_cuda()

    class Runner:
        execute_model_state = None

        def execute_model(self, output):
            return output

        def sample_tokens(self, value):
            return value

    telemetry.install_round_hooks(Runner, cuda)
    output = SimpleNamespace(
        total_num_scheduled_tokens=2048,
        num_scheduled_tokens={"a": 1024, "b": 1024},
        qwen_fair={"cache_timing_context": {"request_id": "a" * 64, "round": 1}},
    )
    assert Runner().execute_model(output) is output
    assert not recorder.pending and not calls


def test_prefill_cancellation_retains_failure_without_changing_exception(recorder):
    cuda, _, calls = fake_round_cuda()
    cancelled = KeyboardInterrupt("not recorded")

    class Runner:
        def execute_model(self, output):
            raise cancelled

        def sample_tokens(self, value):
            return value

    telemetry.install_round_hooks(Runner, cuda)
    output = SimpleNamespace(
        total_num_scheduled_tokens=2048,
        num_scheduled_tokens={"a": 2048},
        qwen_fair={"cache_timing_context": {"request_id": "a" * 64, "round": 1}},
    )
    runner = Runner()
    with pytest.raises(KeyboardInterrupt) as caught:
        runner.execute_model(output)
    assert caught.value is cancelled
    assert recorder.pending[0]["stage"] == "worker_execute"
    assert recorder.pending[0]["success"] is False
    assert runner._qwen_first_work_context == {} and calls == ["record", "record"]


def test_runtime_hook_retry_is_bounded_and_worker_capability_is_visible(
    recorder, monkeypatch
):
    from types import ModuleType

    cuda, _, _ = fake_round_cuda()

    class Runner:
        execute_model_state = None

        def execute_model(self, output):
            return output

        def sample_tokens(self, value):
            return value

    module = ModuleType("vllm.v1.worker.gpu.model_runner")
    module.GPUModelRunner = Runner
    monkeypatch.setitem(sys.modules, module.__name__, module)
    async_module = ModuleType("vllm.v1.worker.gpu.async_utils")
    async_module.AsyncOutput = type(
        "AsyncOutput", (), {"get_output": lambda self: self}
    )
    monkeypatch.setitem(sys.modules, async_module.__name__, async_module)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(telemetry, "_runtime_hooks_installed", True)
    install = telemetry.install_round_hooks
    attempts = []

    def flaky(owner, device):
        attempts.append(1)
        if len(attempts) == 1:
            raise ImportError("not logged")
        return install(owner, device)

    monkeypatch.setattr(telemetry, "install_round_hooks", flaky)
    now = [1_000_000_000]
    monkeypatch.setattr(telemetry.time, "monotonic_ns", lambda: now[0])
    telemetry.install_runtime_hooks()
    assert recorder.round_sample_errors == 1 and not recorder.worker_first_work_hooks
    telemetry.install_runtime_hooks()
    assert len(attempts) == 1
    now[0] += 1_000_000_000
    telemetry.install_runtime_hooks()
    assert len(attempts) == 2 and recorder.worker_first_work_hooks
    telemetry.install_runtime_hooks()
    assert len(attempts) == 2
    recorder._health()
    assert (
        json.loads(recorder.health_path.read_text())["worker_first_work_hooks"] is True
    )


@pytest.mark.parametrize("from_sampler", [False, True])
def test_async_completion_span_belongs_to_its_result(recorder, from_sampler):
    cuda, _, calls = fake_round_cuda()

    class AsyncOutput:
        def get_output(self):
            self.original_waits += 1
            return self

        def __init__(self):
            self.original_waits = 0

    telemetry.install_async_output_hooks(AsyncOutput)
    wrapped = AsyncOutput.get_output
    telemetry.install_async_output_hooks(AsyncOutput)
    assert AsyncOutput.get_output is wrapped

    class Runner:
        execute_model_state = None

        def execute_model(self, output):
            self.execute_model_state = object() if from_sampler else None
            return None if from_sampler else AsyncOutput()

        def sample_tokens(self):
            self.execute_model_state = None
            return AsyncOutput()

    telemetry.install_round_hooks(Runner, cuda)
    runner = Runner()
    outputs = []
    for identity in ("a", "b"):
        scheduled = SimpleNamespace(
            total_num_scheduled_tokens=3296,
            num_scheduled_tokens={identity: 3296},
            qwen_fair={
                "cache_timing_context": {
                    "request_id": identity * 64,
                    "round": 1,
                    "input_tokens": 60000,
                    "computed_tokens": 3296,
                }
            },
        )
        result = runner.execute_model(scheduled)
        outputs.append(runner.sample_tokens() if from_sampler else result)
    for result in reversed(outputs):
        assert result.get_output() is result and result.original_waits == 1
    assert calls == ["record"] * 4  # No extra event, query, or wait from the hook.
    recorder.flush()
    events = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    completed = [row for row in events if row["stage"] == "worker_get_output"]
    assert [row["request_id"] for row in completed] == ["b" * 64, "a" * 64]
    assert all(row["scheduled_tokens"] == 3296 for row in completed)
    assert all("thread_cpu_ms" in row and "major_faults" in row for row in completed)
    reference = weakref.ref(outputs[0])
    outputs.clear()
    del result
    assert reference() is None  # Instrumentation does not retain GPU-bearing output.


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_async_completion_preserves_exception_and_untagged_outputs(recorder, failure):
    error = failure("private exception")

    class AsyncOutput:
        def get_output(self):
            raise error

    telemetry.install_async_output_hooks(AsyncOutput)
    result = AsyncOutput()
    with pytest.raises(failure) as caught:
        result.get_output()
    assert caught.value is error and not recorder.pending
    telemetry._tag_first_work_output(result, {"request_id": "c" * 64, "round": 1})
    with pytest.raises(failure) as caught:
        result.get_output()
    assert caught.value is error
    recorder.flush()
    row = json.loads(recorder.path.read_text())
    assert row["stage"] == "worker_get_output" and row["success"] is False
    assert "private exception" not in recorder.path.read_text()


@pytest.mark.parametrize("kind", ["later", "dummy", "batch", "barrier"])
def test_async_completion_excludes_unowned_or_later_work(recorder, kind):
    cuda, _, _ = fake_round_cuda()

    class AsyncOutput:
        def get_output(self):
            return self

    class Runner:
        execute_model_state = None

        def execute_model(self, scheduled):
            return AsyncOutput()

        def sample_tokens(self):
            return AsyncOutput()

    telemetry.install_async_output_hooks(AsyncOutput)
    telemetry.install_round_hooks(Runner, cuda)
    context = {"request_id": "d" * 64, "round": 2 if kind == "later" else 1}
    scheduled = SimpleNamespace(
        total_num_scheduled_tokens=3296,
        num_scheduled_tokens={"d": 3296} if kind != "batch" else {"d": 1648, "e": 1648},
        qwen_fair={"cache_timing_context": context},
    )
    if kind == "dummy":
        scheduled.qwen_fair = {}
    if kind == "barrier":
        scheduled.qwen_fair["barrier"] = True
    result = Runner().execute_model(scheduled)
    assert result.get_output() is result
    recorder.flush()
    assert (
        not recorder.path.exists()
        or "worker_get_output" not in recorder.path.read_text()
    )

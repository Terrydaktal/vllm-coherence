from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).parents[1]


def load_hooks(monkeypatch, tmp_path, *, sync=True):
    modules = {}
    names = (
        "vllm",
        "vllm.v1",
        "vllm.v1.worker",
        "vllm.v1.worker.gpu_model_runner",
        "vllm.v1.worker.gpu",
        "vllm.v1.worker.gpu.spec_decode",
        "vllm.v1.worker.gpu.spec_decode.rejection_sampler",
        "vllm.v1.core",
        "vllm.v1.core.sched",
        "vllm.v1.core.sched.scheduler",
        "vllm.model_executor",
        "vllm.model_executor.models",
        "vllm.model_executor.models.qwen3_5",
    )
    for name in names:
        modules[name] = types.ModuleType(name)

    class Event:
        sequence = 0

        def __init__(self, *, enable_timing):
            assert enable_timing is True
            self.order = None

        def record(self, stream):
            Event.sequence += 1
            self.order = Event.sequence

        def synchronize(self):
            return None

        def query(self):
            return True

        def elapsed_time(self, other):
            assert self.order is not None and other.order is not None
            return float(max(0, other.order - self.order))

    class Stream:
        cuda_stream = 17

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        Event=Event,
        current_stream=lambda: Stream(),
        current_device=lambda: 0,
        synchronize=lambda: None,
    )

    class Runner:
        def execute_model(self, scheduler_output):
            return scheduler_output

        def sample(self, hidden_states, input_batch, grammar_output=None):
            return hidden_states, input_batch, grammar_output

        def sample_tokens(self, grammar_output=None):
            return grammar_output

        def propose_draft_token_ids(self):
            return [1, 2]

        def postprocess_sampled(self, *args, **kwargs):
            return args, kwargs

    class CausalLM:
        def compute_logits(self, hidden_states):
            return hidden_states

    class RejectionSampler:
        def __call__(self, logits, input_batch, draft_logits=None):
            return logits, input_batch, draft_logits

    class Scheduler:
        def update_from_output(self, output):
            return output

    modules["vllm.v1.worker.gpu_model_runner"].GPUModelRunner = Runner
    modules["vllm.model_executor.models.qwen3_5"].Qwen3_5ForCausalLMBase = CausalLM
    modules["vllm.v1.worker.gpu.spec_decode.rejection_sampler"].RejectionSampler = RejectionSampler
    modules["vllm.v1.core.sched.scheduler"].Scheduler = Scheduler
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("QWEN_STAGE_TIMING", raising=False)
    monkeypatch.setenv("QWEN_OUTER_STAGE_TIMING", "1")
    monkeypatch.setenv("QWEN_ROUND_EVENT_TELEMETRY", "1")
    monkeypatch.setenv("QWEN_ROUND_EVENT_TELEMETRY_SYNC", "1" if sync else "0")
    monkeypatch.setenv("QWEN_ROUND_EVENT_STATUS_PATH", str(tmp_path / "configured-status"))

    spec = importlib.util.spec_from_file_location(
        "runtime_round_telemetry_test_instance",
        ROOT / "src/qwen_r9700_lab/runtime_stage_hooks.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, Runner, CausalLM, RejectionSampler, Scheduler


def test_round_telemetry_records_shape_event_gaps_and_sync_without_payload(
    tmp_path, monkeypatch
):
    hooks, Runner, CausalLM, RejectionSampler, Scheduler = load_hooks(monkeypatch, tmp_path)
    assert hooks.install_runtime_stage_hooks() is True
    status_path = str(tmp_path / "fair")
    scheduler_output = types.SimpleNamespace(
        total_num_scheduled_tokens=8,
        num_scheduled_tokens={"private-request-id": 8},
        scheduled_spec_decode_tokens={"private-request-id": [11, 12]},
        qwen_fair={"status_path": status_path},
    )
    runner = Runner()
    runner.execute_model(scheduler_output)
    CausalLM().compute_logits(None)
    RejectionSampler()(None, None)
    runner.postprocess_sampled()
    runner.propose_draft_token_ids()
    runner.sample_tokens()
    Scheduler().update_from_output(None)

    runner.execute_model(scheduler_output)
    CausalLM().compute_logits(None)
    RejectionSampler()(None, None)
    runner.postprocess_sampled()
    runner.propose_draft_token_ids()
    runner.sample_tokens()
    Scheduler().update_from_output(None)

    records = [
        json.loads(line)
        for line in Path(status_path + "-gpu-rounds.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    record = records[0]
    second = records[1]
    assert record["schema"] == "urn:qwen-r9700:decode-round-gpu:v2"
    assert record["scheduled_shape"] == {
        "mode": "decode",
        "scheduled_tokens": 8,
        "request_count": 1,
        "tokens_per_request": [8],
        "draft_widths": [2],
    }
    assert record["gpu_events"]["backend"] == "rocm-hip"
    assert record["gpu_events"]["device"] == 0
    assert record["gpu_events"]["stream"] == "17"
    assert record["gpu_events"]["event_status"] == "complete"
    assert record["gpu_events"]["gaps_ms"]
    assert record["dispatch"]["previous_round_end_to_start_ms"] is None
    assert record["dropped_records_before"] == 0
    assert second["dispatch"]["previous_round_end_to_start_ms"] is not None
    assert record["queue_sync"]["count_delta"] is None
    assert hooks._EVENT_RING_CURSOR == 0
    assert "private-request-id" not in json.dumps(record)
    hooks._PENDING_EVENTS.clear()
    hooks._TOTALS.clear()


def test_async_round_telemetry_reclaims_event_ring_for_long_responses(
    tmp_path, monkeypatch
):
    hooks, Runner, CausalLM, RejectionSampler, Scheduler = load_hooks(
        monkeypatch, tmp_path, sync=False
    )
    assert hooks.install_runtime_stage_hooks() is True
    status_path = str(tmp_path / "long")
    scheduler_output = types.SimpleNamespace(
        total_num_scheduled_tokens=8,
        num_scheduled_tokens={"opaque-request": 8},
        scheduled_spec_decode_tokens={"opaque-request": [11, 12]},
        qwen_fair={"status_path": status_path},
    )

    def run_round():
        runner.execute_model(scheduler_output)
        CausalLM().compute_logits(None)
        RejectionSampler()(None, None)
        runner.postprocess_sampled()
        runner.propose_draft_token_ids()
        runner.sample_tokens()
        Scheduler().update_from_output(None)

    runner = Runner()
    for _ in range(80):
        run_round()

    records = Path(status_path + "-gpu-rounds.jsonl").read_text().splitlines()
    assert len(records) == 80
    assert hooks._EVENT_RING_CAPPED_LOGGED is False
    hooks._reclaim_completed_ring_pairs()
    assert not hooks._EVENT_SLOT_BY_START
    assert len(hooks._EVENT_RING_FREE_SLOTS) == hooks._EVENT_RING_SIZE


def test_async_round_telemetry_defers_until_current_round_events_complete(
    tmp_path, monkeypatch
):
    hooks, *_ = load_hooks(monkeypatch, tmp_path, sync=False)
    status_path = str(tmp_path / "deferred")

    class PendingEvent:
        def elapsed_time(self, other):
            del other
            return 4.25

    start = PendingEvent()
    end = PendingEvent()
    ready = [False]
    monkeypatch.setattr(hooks, "_event_complete", lambda event: ready[0])
    context = {
        "status_path": status_path,
        "round": 7,
        "phase": "verification",
        "scheduled_shape": {"mode": "decode", "scheduled_tokens": 8},
        "round_start_event": start,
        "round_end_event": end,
        "previous_round_end_event": None,
        "previous_round_stream": None,
        "stream": "17",
        "device": 0,
    }
    hooks._PENDING_TELEMETRY.append(
        {
            "context": context,
            "prefix": "",
            "events": [("round.gpu", start, end)],
            "event_order": [],
        }
    )
    hooks._reclaim_completed_telemetry()
    assert not Path(status_path + "-gpu-rounds.jsonl").exists()
    assert len(hooks._PENDING_TELEMETRY) == 1

    ready[0] = True
    hooks._reclaim_completed_telemetry()
    record = json.loads(Path(status_path + "-gpu-rounds.jsonl").read_text())
    assert record["gpu_events"]["event_status"] == "complete"
    assert record["gpu_events"]["round_span_ms"] == 4.25
    assert not hooks._PENDING_TELEMETRY


def test_async_reclaimer_holds_stage_pairs_until_round_row_is_written(
    tmp_path, monkeypatch
):
    hooks, *_ = load_hooks(monkeypatch, tmp_path, sync=False)
    status_path = str(tmp_path / "held")

    class Event:
        def __init__(self, ready=False):
            self.ready = ready

        def query(self):
            return self.ready

        def elapsed_time(self, other):
            del other
            return 2.5

    stage_start = Event()
    stage_end = Event(True)
    round_start = Event()
    round_end = Event(False)
    context = {
        "status_path": status_path,
        "round": 9,
        "scheduled_shape": {"mode": "decode", "scheduled_tokens": 8},
        "round_start_event": round_start,
        "round_end_event": round_end,
        "previous_round_end_event": None,
        "previous_round_stream": None,
        "stream": "17",
        "device": 0,
    }
    hooks._PENDING_TELEMETRY.append(
        {
            "context": context,
            "prefix": "",
            "events": [("target.forward M=8", stage_start, stage_end)],
            "event_order": [("target.forward M=8.start", stage_start, "17")],
        }
    )
    hooks._PENDING_EVENTS.append(("target.forward M=8", stage_start, stage_end))
    hooks._PENDING_RING_PAIRS.append((stage_start, stage_end))
    hooks._EVENT_SLOT_BY_START[id(stage_start)] = 3

    hooks._reclaim_completed_ring_pairs()
    assert hooks._PENDING_TELEMETRY
    assert hooks._PENDING_EVENTS
    assert hooks._PENDING_RING_PAIRS
    assert id(stage_start) in hooks._EVENT_SLOT_BY_START

    round_end.ready = True
    hooks._reclaim_completed_ring_pairs()
    assert not hooks._PENDING_TELEMETRY
    assert not hooks._PENDING_EVENTS
    assert not hooks._PENDING_RING_PAIRS
    record = json.loads(Path(status_path + "-gpu-rounds.jsonl").read_text())
    assert record["gpu_events"]["durations_ms"] == [
        {"label": "target.forward M=8", "ms": 2.5}
    ]


def test_runtime_shape_records_graph_padding_and_context_without_payload(
    tmp_path, monkeypatch
):
    hooks, Runner, CausalLM, RejectionSampler, Scheduler = load_hooks(
        monkeypatch, tmp_path
    )

    class Descriptor:
        num_tokens = 8
        num_reqs = 1
        max_query_len = 8
        max_seq_len = 60_001
        cg_mode = types.SimpleNamespace(name="PIECEWISE")

    def determine(self, *args, **kwargs):
        del self, args, kwargs
        return types.SimpleNamespace(name="PIECEWISE"), Descriptor(), False, None, None

    Runner._determine_batch_execution_and_padding = determine

    def execute(self, scheduler_output):
        del scheduler_output
        self._determine_batch_execution_and_padding(
            8,
            1,
            [8],
            8,
            False,
            allow_microbatching=True,
        )

    Runner.execute_model = execute
    hooks.install_runtime_stage_hooks()
    status_path = str(tmp_path / "shape")
    scheduler_output = types.SimpleNamespace(
        total_num_scheduled_tokens=8,
        num_scheduled_tokens={"opaque-request": 8},
        scheduled_spec_decode_tokens={"opaque-request": [11, 12]},
        qwen_fair={"status_path": status_path},
    )
    runner = Runner()
    runner.input_batch = types.SimpleNamespace(
        num_computed_tokens_cpu=[60_000],
        num_prompt_tokens_cpu=[60_001],
        input_ids=[999_999],
    )
    runner.optimistic_seq_lens_cpu = [60_008]
    runner.execute_model(scheduler_output)
    CausalLM().compute_logits(None)
    RejectionSampler()(None, None)
    runner.postprocess_sampled()
    runner.propose_draft_token_ids()
    runner.sample_tokens()
    Scheduler().update_from_output(None)

    record = json.loads(Path(status_path + "-gpu-rounds.jsonl").read_text())
    shape = record["runtime_shape"]
    assert shape["cudagraph_mode"] == "PIECEWISE"
    assert shape["padded_tokens"] == 8
    assert shape["max_sequence_len"] == 60_001
    assert shape["computed_tokens"] == [60_000]
    assert shape["prompt_tokens"] == [60_001]
    assert shape["sequence_lengths"] == [60_008]
    assert "input_ids" not in json.dumps(record)


def test_runtime_shape_wraps_current_prepare_inputs_boundary(tmp_path, monkeypatch):
    hooks, Runner, *_ = load_hooks(monkeypatch, tmp_path)

    def prepare(self, scheduler_output, batch_req_state, batch_desc):
        del self, scheduler_output, batch_req_state, batch_desc
        return types.SimpleNamespace(
            num_tokens=8,
            num_tokens_after_padding=8,
            num_reqs=1,
            num_reqs_after_padding=1,
            num_computed_tokens_np=[60_000],
            prefill_len_np=[60_000],
            seq_lens_cpu_upper_bound=[60_008],
        )

    Runner.prepare_inputs = prepare
    assert hooks._wrap_outer_prepare_inputs(Runner) is True
    runner = Runner()
    runner._qwen_outer_round_context = {}
    hooks._OUTER_ROUND_ACTIVE = True
    try:
        result = runner.prepare_inputs(
            types.SimpleNamespace(),
            types.SimpleNamespace(
                req_ids=["opaque"],
                num_scheduled_tokens=[8],
                has_prefill=False,
            ),
            types.SimpleNamespace(
                cg_mode=types.SimpleNamespace(name="PIECEWISE"),
                num_tokens=8,
                num_reqs=1,
                max_query_len=8,
            ),
        )
    finally:
        hooks._OUTER_ROUND_ACTIVE = False
    assert result.num_tokens == 8
    shape = runner._qwen_outer_round_context["runtime_shape"]
    assert shape["cudagraph_mode"] == "PIECEWISE"
    assert shape["descriptor_tokens"] == 8
    assert shape["input_tokens_after_padding"] == 8
    assert shape["computed_tokens"] == [60_000]
    assert shape["sequence_lengths"] == [60_008]
    assert "opaque" not in json.dumps(shape)


def test_tvm_optional_builder_does_not_bootstrap_worker_hooks(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["/opt/vllm/lib/python3.12/site-packages/tvm_ffi/utils/build_optional_torch_c_dlpack.py"],
    )
    hooks, *_ = load_hooks(monkeypatch, tmp_path)
    assert hooks._IS_TVM_DLPACK_BUILDER is True
    assert hooks._ENABLED is False
    assert hooks.install_runtime_stage_hooks() is False


def test_rocm_agent_enumerator_does_not_bootstrap_worker_hooks(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["/opt/rocm/bin/rocm_agent_enumerator", "-name"])
    hooks, *_ = load_hooks(monkeypatch, tmp_path)
    assert hooks._IS_RUNTIME_HELPER is True
    assert hooks._ENABLED is False
    assert hooks.install_runtime_stage_hooks() is False


def test_round_telemetry_uses_process_status_path_when_scheduler_output_has_none(
    tmp_path, monkeypatch
):
    hooks, Runner, CausalLM, RejectionSampler, Scheduler = load_hooks(
        monkeypatch, tmp_path
    )
    assert hooks.install_runtime_stage_hooks() is True
    status_path = tmp_path / "configured-status"
    scheduler_output = types.SimpleNamespace(
        total_num_scheduled_tokens=8,
        num_scheduled_tokens={"opaque-request": 8},
        scheduled_spec_decode_tokens={"opaque-request": [11, 12]},
        qwen_fair=None,
    )
    runner = Runner()
    runner.execute_model(scheduler_output)
    CausalLM().compute_logits(None)
    RejectionSampler()(None, None)
    runner.postprocess_sampled()
    runner.propose_draft_token_ids()
    runner.sample_tokens()
    Scheduler().update_from_output(None)
    records = Path(str(status_path) + "-gpu-rounds.jsonl")
    assert records.exists()
    record = json.loads(records.read_text(encoding="utf-8"))
    assert record["gpu_events"]["event_status"] == "complete"


def test_async_event_ring_overflow_uses_reclaimable_dynamic_pairs(tmp_path, monkeypatch):
    hooks, *_ = load_hooks(monkeypatch, tmp_path, sync=False)
    hooks.install_runtime_stage_hooks()
    hooks._prepare_event_ring()
    hooks._EVENT_RING_FREE_SLOTS.clear()
    hooks._DYNAMIC_EVENT_PAIR_LIMIT = 1
    pair = hooks._event_pair("overflow")
    assert pair is not None
    assert hooks._EVENT_SLOT_BY_START[id(pair[0])] is None
    hooks._release_ring_pair(pair)
    assert not hooks._DYNAMIC_EVENT_PAIRS

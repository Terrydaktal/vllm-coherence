from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).parents[1]


def load_hooks(monkeypatch, tmp_path):
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
    monkeypatch.setenv("QWEN_ROUND_EVENT_TELEMETRY_SYNC", "1")

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

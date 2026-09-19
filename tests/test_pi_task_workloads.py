"""CPU-only checks for the chained task workload benchmark."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/radiance-public/benchmark_pi_task_workloads.py"
SPEC = importlib.util.spec_from_file_location("pi_task_workloads", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class LinesResponse:
    def __init__(self, frames):
        self.lines = [frame.encode() for frame in frames]

    def readline(self):
        return self.lines.pop(0) if self.lines else b""


def event(choice, usage=None):
    payload = {"choices": [choice]}
    if usage is not None:
        payload["usage"] = usage
    return f"data: {json.dumps(payload)}\n\n"


def test_workload_order_is_eight_single_prompt_categories():
    assert [name for name, _ in MODULE.TASKS] == [
        "Chat",
        "Code",
        "File edit",
        "JSON",
        "Math",
        "Prose",
        "Reasoning",
        "Summarisation",
    ]
    assert len(MODULE.TASKS) == 8


def test_sse_strips_prompt_ids_and_retains_exact_generated_ids():
    prompt = [1, 2, 3]
    response = LinesResponse(
        [
            event(
                {"index": 0, "token_ids": [1, 2, 3, 7], "text": "x"},
                {"completion_tokens": 2},
            ),
            event({"index": 0, "token_ids": [8], "text": "y", "finish_reason": "stop"}),
            "data: [DONE]\n\n",
        ]
    )
    result = MODULE._read_sse(response, prompt)
    assert result["token_ids"] == [7, 8]
    assert result["finish_reason"] == "stop"
    assert result["usage"]["completion_tokens"] == 2


def test_sse_rejects_a_missing_done_boundary():
    response = LinesResponse(
        [event({"index": 0, "token_ids": [7], "finish_reason": "stop"})]
    )
    with pytest.raises(RuntimeError, match="finish_reason and DONE"):
        MODULE._read_sse(response, [])


def test_metric_summary_reports_mean_round_and_acceptance():
    delta = {
        "vllm:spec_decode_num_drafts_total": 4.0,
        "vllm:spec_decode_num_draft_tokens_total": 20.0,
        "vllm:spec_decode_num_accepted_tokens_total": 15.0,
        "vllm:generation_tokens_total": 18.0,
        "vllm:inter_token_latency_seconds_sum": 0.32,
        "vllm:inter_token_latency_seconds_count": 16.0,
    }
    result = MODULE._summarize_metrics(delta)
    assert result["mean_generation_round_ms"] == pytest.approx(20.0)
    assert result["acceptance_rate"] == pytest.approx(0.75)
    assert result["generated_tokens"] == 18


def test_published_result_has_both_chained_arms():
    results = json.loads(
        (ROOT / "benchmarks/results/pi-task-workloads.json").read_text()
    )
    assert results["status"] == "complete"
    assert results["prompt_count_per_arm"] == 8
    assert results["arms"]["60K"]["initial_context_tokens"] == 60000
    assert results["arms"]["0K"]["initial_context_tokens"] == 0
    if results["status"] == "complete":
        for arm in results["arms"].values():
            assert len(arm["results"]) == 8
            for row in arm["results"]:
                assert row["finish_reason"] == "stop"
                assert row["metrics_delta"]["vllm:request_success_total"] == 1
                assert row["metrics_delta"]["vllm:num_preemptions_total"] == 0


def test_missing_metrics_are_not_silently_reported_as_zero():
    with pytest.raises(RuntimeError, match="required backend metric"):
        MODULE._metric_values("vllm:num_requests_running 0\n")


def test_turn_suffix_preserves_all_prior_tokens_and_closes_only_the_boundary():
    class Tokenizer:
        def token_to_id(self, _):
            return 99

        def encode(self, text, **_):
            return type("Encoded", (), {"ids": list(text.encode())})()

    tokenizer = Tokenizer()
    prefix = [1, 2, 3]
    rendered = [10, 11]
    assert MODULE._turn_suffix(tokenizer, [], rendered, first=True) == rendered
    result = MODULE._turn_suffix(tokenizer, prefix, rendered, first=True)
    assert result == list(b"\n</think>\n<|im_end|>\n") + rendered
    assert prefix == [1, 2, 3]
    assert MODULE._turn_suffix(tokenizer, [1, 99], rendered, first=False) == [
        10,
        *rendered,
    ]

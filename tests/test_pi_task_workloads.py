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
    assert [task.name for task in MODULE.TASKS] == [
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
    assert [task.output_kind for task in MODULE.TASKS] == [
        "free_text",
        "python_code",
        "unified_diff",
        "json_object",
        "free_text",
        "free_text",
        "reasoning_prompt",
        "free_text",
    ]
    assert [task.enable_thinking for task in MODULE.TASKS] == [
        True,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
    ]


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
    assert result["content"] == "xy"
    assert result["reasoning"] == ""
    assert result["phase_token_counts_cover_output"] is False


def test_sse_separates_reasoning_and_content_channels():
    response = LinesResponse(
        [
            event(
                {
                    "index": 0,
                    "token_ids": [7],
                    "delta": {"reasoning_content": "check first"},
                }
            ),
            event(
                {
                    "index": 0,
                    "token_ids": [8],
                    "delta": {"content": "answer"},
                    "finish_reason": "stop",
                }
            ),
            "data: [DONE]\n\n",
        ]
    )
    result = MODULE._read_sse(response, [])
    assert result["reasoning"] == "check first"
    assert result["content"] == "answer"
    assert result["unclassified"] == ""


def test_sse_accepts_legacy_reasoning_text_and_message_fields():
    response = LinesResponse(
        [
            event(
                {
                    "index": 0,
                    "token_ids": [7],
                    "delta": {"reasoning_text": "legacy thought"},
                }
            ),
            event(
                {
                    "index": 0,
                    "token_ids": [8],
                    "message": {"reasoning": "final thought"},
                    "finish_reason": "stop",
                }
            ),
            "data: [DONE]\n\n",
        ]
    )
    result = MODULE._read_sse(response, [])
    assert result["reasoning"] == "legacy thoughtfinal thought"


def test_content_only_validators_reject_the_wrong_phase_or_shape():
    code = next(task for task in MODULE.TASKS if task.name == "Code")
    valid_code = MODULE._validate_task_output(
        code,
        content="def first_repeat(items: list[str]) -> str:\n    return items[0]\n",
        reasoning="",
        unclassified="",
    )
    assert valid_code["passed"] is True
    invalid_phase = MODULE._validate_task_output(
        code,
        content="def first_repeat(items):\n    return items[0]\n",
        reasoning="drafting",
        unclassified="",
    )
    assert invalid_phase["passed"] is False

    json_task = next(task for task in MODULE.TASKS if task.name == "JSON")
    valid_json = MODULE._validate_task_output(
        json_task,
        content='{"name":"x","count":3,"items":["a","b","c"]}',
        reasoning="",
        unclassified="",
    )
    assert valid_json["passed"] is True
    invalid_json = MODULE._validate_task_output(
        json_task,
        content='Here is the JSON: {"name":"x","count":3,"items":["a","b","c"]}',
        reasoning="",
        unclassified="",
    )
    assert invalid_json["passed"] is False

    diff_task = next(task for task in MODULE.TASKS if task.name == "File edit")
    valid_diff = MODULE._validate_task_output(
        diff_task,
        content="--- a/main.py\n+++ b/main.py\n@@ -1 +1 @@\n-old\n+new\n",
        reasoning="",
        unclassified="",
    )
    assert valid_diff["passed"] is True

    prose = next(task for task in MODULE.TASKS if task.name == "Prose")
    assert MODULE._validate_task_output(
        prose, content="rain", reasoning="", unclassified="token-id-only-frame"
    )["passed"] is False

    reasoning_task = next(task for task in MODULE.TASKS if task.name == "Reasoning")
    reasoning_result = MODULE._validate_task_output(
        reasoning_task, content="answer", reasoning="", unclassified=""
    )
    assert reasoning_result["passed"] is True
    assert reasoning_result["reasoning_channel_observed"] is False


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


def test_peak_rate_uses_only_complete_three_second_windows():
    samples = [
        (0.0, 0),
        (0.5, 5),
        (1.0, 8),
        (2.0, 12),
        (3.0, 20),
        (4.0, 27),
        (5.0, 40),
    ]
    assert MODULE._peak_rolling_tokens_per_second(samples) == pytest.approx(
        28 / 3
    )
    assert MODULE._peak_rolling_tokens_per_second(samples[:3]) is None


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

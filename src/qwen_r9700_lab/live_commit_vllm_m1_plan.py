"""Derive independent one-token worker inputs from one scheduled D7/M8 decode.

The scheduler may plan eight target rows for one target token plus seven draft
tokens.  The authoritative serial arm must not reuse that batched transition.
This module converts the immutable scheduling receipt into ``c`` distinct M1
worker inputs, each with no speculative tokens and with the pre-step counters
advanced exactly once per preceding serial result.

The function is intentionally limited to an already-running, TP=1, synchronous
decode request.  Prefill/admission and connector publication require separate
transaction boundaries and fail closed here rather than being silently treated
as equivalent decode work.
"""

from __future__ import annotations

from dataclasses import is_dataclass, replace
from typing import Any

from qwen_r9700_lab.live_commit_vllm_root_tables import MAX_COMMIT_WIDTH


class VllmM1PlanError(RuntimeError):
    """The scheduled output cannot be transformed into an independent M1 plan."""


def _single(value: object, label: str) -> Any:
    if not isinstance(value, list) or len(value) != 1:
        raise VllmM1PlanError(f"{label} must contain exactly one request")
    return value[0]


def _empty_or_none(value: object, label: str) -> None:
    if value not in (None, [], {}, set()):
        raise VllmM1PlanError(f"{label} has side effects outside a decode round")


def build_serial_m1_steps(
    candidate_output: object,
    *,
    request_id: str,
    commit_count: int,
) -> tuple[object, ...]:
    """Return ``commit_count`` fresh SchedulerOutputs for serial M1 execution."""

    if not isinstance(request_id, str) or not request_id:
        raise VllmM1PlanError("request ID is invalid")
    if (
        isinstance(commit_count, bool)
        or not isinstance(commit_count, int)
        or not 0 <= commit_count <= MAX_COMMIT_WIDTH
    ):
        raise VllmM1PlanError("commit count is outside 0..8")
    if commit_count == 0:
        return ()
    if not is_dataclass(candidate_output):
        raise VllmM1PlanError("candidate scheduler output is not a dataclass")
    if getattr(candidate_output, "scheduled_new_reqs", None) != []:
        raise VllmM1PlanError("serial M1 decode cannot admit or resume a request")
    cached = getattr(candidate_output, "scheduled_cached_reqs", None)
    if not is_dataclass(cached):
        raise VllmM1PlanError("cached request payload is not a dataclass")
    if getattr(cached, "req_ids", None) != [request_id]:
        raise VllmM1PlanError("candidate output is not the protected C1 request")
    if getattr(cached, "resumed_req_ids", None) != set():
        raise VllmM1PlanError("serial M1 decode cannot resume a worker request")
    if getattr(candidate_output, "num_scheduled_tokens", None) != {
        request_id: MAX_COMMIT_WIDTH
    }:
        raise VllmM1PlanError("candidate output is not one complete D7/M8 round")
    draft = getattr(candidate_output, "scheduled_spec_decode_tokens", None)
    if (
        not isinstance(draft, dict)
        or set(draft) != {request_id}
        or not isinstance(draft[request_id], list)
        or len(draft[request_id]) != MAX_COMMIT_WIDTH - 1
    ):
        raise VllmM1PlanError("candidate output lacks exactly seven draft tokens")
    base_computed = _single(
        getattr(cached, "num_computed_tokens", None), "computed-token counters"
    )
    base_output = _single(
        getattr(cached, "num_output_tokens", None), "output-token counters"
    )
    for value, label in (
        (base_computed, "computed-token counter"),
        (base_output, "output-token counter"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise VllmM1PlanError(f"{label} is invalid")
    if getattr(cached, "new_token_ids", None) != []:
        raise VllmM1PlanError("serial M1 plan requires TP=1 token ownership")
    new_blocks = getattr(cached, "new_block_ids", None)
    if not isinstance(new_blocks, list) or len(new_blocks) != 1:
        raise VllmM1PlanError("candidate block delta is invalid")
    for field in (
        "scheduled_encoder_inputs",
        "finished_req_ids",
        "free_encoder_mm_hashes",
        "preempted_req_ids",
        "kv_cache_block_copies",
        "partial_tail_offloads",
        "draft_tree_metadata",
    ):
        _empty_or_none(getattr(candidate_output, field, None), field)
    if getattr(candidate_output, "total_num_scheduled_tokens", None) != MAX_COMMIT_WIDTH:
        raise VllmM1PlanError("candidate total scheduled-token count differs")

    steps: list[object] = []
    for index in range(commit_count):
        serial_cached = replace(
            cached,
            new_block_ids=[None],
            num_computed_tokens=[base_computed + index],
            num_output_tokens=[base_output + index],
        )
        steps.append(
            replace(
                candidate_output,
                scheduled_cached_reqs=serial_cached,
                num_scheduled_tokens={request_id: 1},
                total_num_scheduled_tokens=1,
                scheduled_spec_decode_tokens={},
                scheduled_encoder_inputs={},
                scheduled_encoder_input_stats=None,
                num_invalid_spec_tokens=None,
                kv_connector_metadata=None,
                ec_connector_metadata=None,
                ec_manager_metadata=None,
                new_block_ids_to_zero=None,
                kv_cache_block_copies=None,
                partial_tail_offloads=None,
                num_spec_tokens_to_schedule=0,
                draft_tree_metadata={},
            )
        )
    return tuple(steps)

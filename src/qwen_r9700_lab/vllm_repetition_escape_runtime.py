# ruff: noqa: N803
"""GPU-side bounded repetition escape for the pinned vLLM decode path.

This module intentionally imports Torch and Triton lazily.  The repository's
CPU-only assurance tests can therefore inspect the exact serving contract
without installing the ROCm runtime.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import os
import queue
import stat
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Bound lazily by ``_load_kernels`` so CPU-only assurance imports do not require
# the ROCm/Triton runtime.  The annotation also makes the nested JIT functions'
# module-global dependency explicit to static analysis.
tl: Any = None

# Keep this deployment module self-contained.  These values are asserted against
# the CPU reference policy by tests before an overlay can be qualified.
DEFAULT_MAX_PERIOD = 64
DEFAULT_MIN_COPIES = 8
DEFAULT_MIN_REPEATED_TOKENS = 256
DEFAULT_MAX_ESCAPES = 4
DEFAULT_MAX_CANDIDATES = 16

LOGGER = logging.getLogger(__name__)

ENABLE_ENV = "QWEN_BOUNDED_REPETITION_ESCAPE"
TELEMETRY_ENV = "QWEN_BOUNDED_REPETITION_ESCAPE_TELEMETRY"
EOS_IDS_ENV = "QWEN_BOUNDED_REPETITION_ESCAPE_EOS_TOKEN_IDS"
MAX_LOGIT_GAP_ENV = "QWEN_BOUNDED_REPETITION_ESCAPE_MAX_LOGIT_GAP"

REPLAY_ENVIRONMENTS = (
    "QWEN_DFLASH_ASSURANCE_CAPTURE",
    "QWEN_DFLASH_ASSURANCE_TARGET_ONLY",
    "QWEN_DFLASH_ASSURANCE_LAYER_OUTPUT",
    "QWEN_FAILED_PROMPT_REPLAY",
    "QWEN_FIXED_SLOT_TARGET_ONLY_ORACLE",
    "QWEN_HAUHAU_BF16_TARGET_ONLY_ORACLE",
)

TELEMETRY_QUEUE_CAPACITY = 128

EVENT_NONE = 0
EVENT_ESCAPE = 1
EVENT_TERMINATE = 2
EVENT_BLOCKED = 3
EVENT_WIDTH = 10


class RuntimeContractError(RuntimeError):
    """Raised before mutation when the serving contract is incomplete."""


@dataclass(frozen=True)
class RuntimePolicy:
    max_period: int = DEFAULT_MAX_PERIOD
    min_copies: int = DEFAULT_MIN_COPIES
    min_repeated_tokens: int = DEFAULT_MIN_REPEATED_TOKENS
    max_escapes: int = DEFAULT_MAX_ESCAPES
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    top_per_vocab_block: int = DEFAULT_MAX_CANDIDATES
    vocab_block_size: int = 4096
    max_logit_gap: float = 5.0

    @property
    def max_repeated_span(self) -> int:
        return max(
            self.min_repeated_tokens + self.max_period - 1,
            self.min_copies * self.max_period,
        )

    def validate(self) -> None:
        if self.max_period < 1 or self.min_copies < 2:
            raise RuntimeContractError("invalid periodicity bounds")
        if self.min_repeated_tokens < 1 or self.max_escapes < 1:
            raise RuntimeContractError("invalid repetition escape limits")
        if self.min_repeated_tokens < 2 * self.max_period:
            raise RuntimeContractError(
                "repetition span must prove that changing the final token breaks every period"
            )
        if self.max_candidates < 2 or self.top_per_vocab_block < 1:
            raise RuntimeContractError("candidate bounds cannot prove an alternative")
        if self.vocab_block_size < 128 or self.vocab_block_size & (
            self.vocab_block_size - 1
        ):
            raise RuntimeContractError("vocab block size must be a power of two")
        if not math.isfinite(self.max_logit_gap) or self.max_logit_gap <= 0:
            raise RuntimeContractError("maximum logit gap must be finite and positive")


def _enabled(environment: dict[str, str] | os._Environ[str] = os.environ) -> bool:
    value = environment.get(ENABLE_ENV, "0")
    if value not in {"0", "1"}:
        raise RuntimeContractError(f"{ENABLE_ENV} must be 0 or 1")
    return value == "1"


def _parse_eos_ids(value: str) -> tuple[int, ...]:
    try:
        token_ids = tuple(int(token.strip()) for token in value.split(",") if token.strip())
    except ValueError as error:
        raise RuntimeContractError(f"{EOS_IDS_ENV} contains a non-integer token ID") from error
    if not token_ids or len(token_ids) > 16:
        raise RuntimeContractError(f"{EOS_IDS_ENV} must contain between 1 and 16 token IDs")
    if len(set(token_ids)) != len(token_ids) or any(token_id < 0 for token_id in token_ids):
        raise RuntimeContractError(f"{EOS_IDS_ENV} must contain unique non-negative token IDs")
    return token_ids


def policy_from_environment(
    environment: dict[str, str] | os._Environ[str] = os.environ,
) -> tuple[RuntimePolicy, tuple[int, ...], Path] | None:
    """Validate activation before the first serving mutation."""

    if not _enabled(environment):
        return None
    conflicts = [name for name in REPLAY_ENVIRONMENTS if environment.get(name, "") not in {"", "0"}]
    if conflicts:
        joined = ", ".join(conflicts)
        raise RuntimeContractError(
            f"loop escape must be disabled during differential replay: {joined}"
        )

    eos_ids = _parse_eos_ids(environment.get(EOS_IDS_ENV, ""))
    try:
        max_gap = float(environment.get(MAX_LOGIT_GAP_ENV, "5.0"))
    except ValueError as error:
        raise RuntimeContractError(f"{MAX_LOGIT_GAP_ENV} must be numeric") from error
    policy = RuntimePolicy(max_logit_gap=max_gap)
    policy.validate()

    telemetry_raw = environment.get(TELEMETRY_ENV, "")
    telemetry = Path(telemetry_raw)
    if not telemetry_raw or not telemetry.is_absolute():
        raise RuntimeContractError(f"{TELEMETRY_ENV} must be an absolute create-or-append path")
    parent = telemetry.parent
    if not parent.is_dir() or parent.is_symlink():
        raise RuntimeContractError("loop-escape telemetry parent must be a real directory")
    parent_mode = stat.S_IMODE(parent.stat().st_mode)
    if parent_mode & 0o077:
        raise RuntimeContractError(
            "loop-escape telemetry parent must not be group/world accessible"
        )
    if parent.stat().st_uid != os.getuid():
        raise RuntimeContractError("loop-escape telemetry parent has the wrong owner")
    if telemetry.exists():
        metadata = telemetry.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) & 0o077
            or metadata.st_uid != os.getuid()
        ):
            raise RuntimeContractError("loop-escape telemetry must be a private regular file")
    return policy, eos_ids, telemetry


def register_bounded_loop_escape_request(
    *, sampler: Any, req_index: int, sampling_params: Any
) -> None:
    """Remember whether request-specific stopping semantics make escape unsafe.

    The GPU sampler does not otherwise retain arbitrary stop strings/token IDs or
    ``ignore_eos`` in a form available at the post-verification hook.  Rather
    than guessing, disable mutation for those requests.  The array starts true
    so an unregistered slot also fails closed.
    """

    configuration = getattr(sampler, "_qwen_bounded_loop_escape_configuration", ...)
    if configuration is ...:
        configuration = policy_from_environment()
        sampler._qwen_bounded_loop_escape_configuration = configuration
    if configuration is None:
        return
    capacity = getattr(sampler.req_states, "max_num_reqs", None)
    if not isinstance(capacity, int) or capacity < 1:
        raise RuntimeContractError("request-state capacity is unavailable")
    unsupported = getattr(sampler, "_qwen_bounded_loop_escape_unsupported_stop", None)
    if unsupported is None:
        import numpy as np

        unsupported = np.ones(capacity, dtype=bool)
        sampler._qwen_bounded_loop_escape_unsupported_stop = unsupported
    if req_index < 0 or req_index >= len(unsupported):
        raise RuntimeContractError("request-state index is outside sampler capacity")
    has_custom_stop = bool(getattr(sampling_params, "stop_token_ids", None)) or bool(
        getattr(sampling_params, "stop", None)
    )
    unsupported[req_index] = has_custom_stop or bool(
        getattr(sampling_params, "ignore_eos", False)
    )


_KERNELS: tuple[Any, Any, Any] | None = None


def _load_kernels() -> tuple[Any, Any, Any]:
    global _KERNELS, _qwen_escape_has_periodic_suffix, _qwen_escape_sequence_token
    if _KERNELS is not None:
        return _KERNELS

    triton = importlib.import_module("triton")
    globals()["tl"] = importlib.import_module("triton.language")

    @triton.jit
    def _qwen_escape_sequence_token(
        sequence_idx,
        valid,
        history_len,
        all_token_ids_ptr,
        all_token_ids_stride,
        prompt_len,
        sampled_ptr,
        sampled_stride,
        history_req_idx,
        sampled_batch_idx,
    ):
        from_history = sequence_idx < history_len
        historical = tl.load(
            all_token_ids_ptr
            + history_req_idx * all_token_ids_stride
            + prompt_len
            + sequence_idx,
            mask=valid & from_history,
            other=-1,
        ).to(tl.int64)
        current = tl.load(
            sampled_ptr
            + sampled_batch_idx * sampled_stride
            + sequence_idx
            - history_len,
            mask=valid & ~from_history,
            other=-1,
        ).to(tl.int64)
        token = tl.where(from_history, historical, current)
        return token

    @triton.jit
    def _qwen_escape_has_periodic_suffix(
        sequence_len,
        history_len,
        all_token_ids_ptr,
        all_token_ids_stride,
        prompt_len,
        sampled_ptr,
        sampled_stride,
        history_req_idx,
        sampled_batch_idx,
        MAX_PERIOD: tl.constexpr,
        MIN_COPIES: tl.constexpr,
        MIN_REPEATED_TOKENS: tl.constexpr,
        MAX_REPEATED_SPAN: tl.constexpr,
    ):
        found = False
        first_period = tl.zeros((), tl.int32)
        first_span = tl.zeros((), tl.int32)
        offsets = tl.arange(0, MAX_REPEATED_SPAN)
        for period in tl.range(1, MAX_PERIOD + 1):
            copies_for_span = (MIN_REPEATED_TOKENS + period - 1) // period
            copies = tl.maximum(copies_for_span, MIN_COPIES)
            repeated_span = copies * period
            enough = sequence_len >= repeated_span
            compare_count = repeated_span - period
            compare = enough & (offsets < compare_count)
            left_idx = sequence_len - 1 - offsets
            right_idx = left_idx - period
            left = _qwen_escape_sequence_token(
                left_idx,
                compare,
                history_len,
                all_token_ids_ptr,
                all_token_ids_stride,
                prompt_len,
                sampled_ptr,
                sampled_stride,
                history_req_idx,
                sampled_batch_idx,
            )
            right = _qwen_escape_sequence_token(
                right_idx,
                compare,
                history_len,
                all_token_ids_ptr,
                all_token_ids_stride,
                prompt_len,
                sampled_ptr,
                sampled_stride,
                history_req_idx,
                sampled_batch_idx,
            )
            mismatches = tl.sum((compare & (left != right)).to(tl.int32), axis=0)
            repeats = enough & (mismatches == 0)
            take = repeats & ~found
            first_period = tl.where(take, period, first_period)
            first_span = tl.where(take, repeated_span, first_span)
            found |= repeats
        return found, first_period, first_span

    @triton.jit
    def detect_kernel(
        detection_ptr,
        detection_stride,
        sampled_ptr,
        sampled_stride,
        num_sampled_ptr,
        cu_num_logits_ptr,
        idx_mapping_ptr,
        all_token_ids_ptr,
        all_token_ids_stride,
        prompt_len_ptr,
        total_len_ptr,
        terminal_ids_ptr,
        MAX_PERIOD: tl.constexpr,
        MIN_COPIES: tl.constexpr,
        MIN_REPEATED_TOKENS: tl.constexpr,
        MAX_REPEATED_SPAN: tl.constexpr,
        MAX_ROUND_TOKENS: tl.constexpr,
        NUM_TERMINALS: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        for column in tl.static_range(0, 5):
            tl.store(detection_ptr + batch_idx * detection_stride + column, -1)
        req_idx = tl.load(idx_mapping_ptr + batch_idx).to(tl.int64)
        num_sampled = tl.load(num_sampled_ptr + batch_idx).to(tl.int32)
        if req_idx < 0:
            return
        if num_sampled <= 0:
            return
        round_has_terminal = False
        for round_idx in tl.static_range(0, MAX_ROUND_TOKENS):
            sampled_token = tl.load(
                sampled_ptr + batch_idx * sampled_stride + round_idx,
                mask=round_idx < num_sampled,
                other=-1,
            ).to(tl.int64)
            for terminal_idx in tl.static_range(0, NUM_TERMINALS):
                terminal_id = tl.load(terminal_ids_ptr + terminal_idx).to(tl.int64)
                round_has_terminal |= (round_idx < num_sampled) & (
                    sampled_token == terminal_id
                )
        # An EOS already selected by the target is a normal completion, even if
        # preceding output happened to be periodic.  Never rewrite that round.
        if round_has_terminal:
            return
        prompt_len = tl.load(prompt_len_ptr + req_idx).to(tl.int64)
        total_len = tl.load(total_len_ptr + req_idx).to(tl.int64)
        history_len = total_len - prompt_len
        sequence_len = history_len + num_sampled
        found, period, repeated_span = _qwen_escape_has_periodic_suffix(
            sequence_len,
            history_len,
            all_token_ids_ptr,
            all_token_ids_stride,
            prompt_len,
            sampled_ptr,
            sampled_stride,
            req_idx,
            batch_idx,
            MAX_PERIOD=MAX_PERIOD,
            MIN_COPIES=MIN_COPIES,
            MIN_REPEATED_TOKENS=MIN_REPEATED_TOKENS,
            MAX_REPEATED_SPAN=MAX_REPEATED_SPAN,
        )
        if not found:
            return
        final_round_idx = num_sampled - 1
        looping_token = tl.load(
            sampled_ptr + batch_idx * sampled_stride + final_round_idx
        ).to(tl.int64)
        logit_row = tl.load(cu_num_logits_ptr + batch_idx).to(tl.int64) + final_round_idx
        tl.store(detection_ptr + batch_idx * detection_stride, logit_row)
        tl.store(detection_ptr + batch_idx * detection_stride + 1, period)
        tl.store(detection_ptr + batch_idx * detection_stride + 2, repeated_span)
        tl.store(detection_ptr + batch_idx * detection_stride + 3, looping_token)
        tl.store(detection_ptr + batch_idx * detection_stride + 4, sequence_len - 1)

    @triton.jit
    def block_candidates_kernel(
        logits_ptr,
        logits_stride,
        vocab_size,
        detection_ptr,
        detection_stride,
        terminal_ids_ptr,
        candidates_value_ptr,
        candidates_id_ptr,
        candidate_stride,
        BLOCK_SIZE: tl.constexpr,
        TOP_PER_BLOCK: tl.constexpr,
        NUM_TERMINALS: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        block_idx = tl.program_id(1)
        candidate_base = batch_idx * candidate_stride + block_idx * TOP_PER_BLOCK
        logit_row = tl.load(detection_ptr + batch_idx * detection_stride).to(tl.int64)
        if logit_row < 0:
            return
        looping_token = tl.load(detection_ptr + batch_idx * detection_stride + 3).to(tl.int64)
        token_ids = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = token_ids < vocab_size
        scores = tl.load(
            logits_ptr + logit_row * logits_stride + token_ids,
            mask=valid,
            other=-float("inf"),
        ).to(tl.float32)
        admissible = valid & (token_ids != looping_token)
        for terminal_idx in tl.static_range(0, NUM_TERMINALS):
            terminal_id = tl.load(terminal_ids_ptr + terminal_idx).to(tl.int64)
            admissible &= token_ids != terminal_id
        scores = tl.where(
            admissible & (scores > -float("inf")) & (scores < float("inf")),
            scores,
            -float("inf"),
        )
        for rank in tl.static_range(0, TOP_PER_BLOCK):
            best_value = tl.max(scores, axis=0)
            best_offset = tl.argmax(scores, axis=0)
            best_token = block_idx * BLOCK_SIZE + best_offset
            finite = best_value > -float("inf")
            tl.store(candidates_value_ptr + candidate_base + rank, best_value)
            tl.store(candidates_id_ptr + candidate_base + rank, tl.where(finite, best_token, -1))
            scores = tl.where(token_ids == best_token, -float("inf"), scores)

    @triton.jit
    def choose_kernel(
        logits_ptr,
        logits_stride,
        detection_ptr,
        detection_stride,
        candidates_value_ptr,
        candidates_id_ptr,
        candidate_stride,
        terminal_ids_ptr,
        sampled_ptr,
        sampled_stride,
        num_sampled_ptr,
        idx_mapping_ptr,
        escape_count_ptr,
        blocked_reported_ptr,
        events_ptr,
        events_stride,
        NUM_CANDIDATE_SLOTS: tl.constexpr,
        NUM_CANDIDATE_SLOTS_PADDED: tl.constexpr,
        MAX_CANDIDATES: tl.constexpr,
        MAX_ESCAPES: tl.constexpr,
        NUM_TERMINALS: tl.constexpr,
        MAX_LOGIT_GAP: tl.constexpr,
        MAX_PERIOD: tl.constexpr,
        MIN_COPIES: tl.constexpr,
        MIN_REPEATED_TOKENS: tl.constexpr,
        MAX_REPEATED_SPAN: tl.constexpr,
        EVENT_WIDTH_CONST: tl.constexpr,
        EVENT_NONE_CODE: tl.constexpr,
        EVENT_ESCAPE_CODE: tl.constexpr,
        EVENT_TERMINATE_CODE: tl.constexpr,
        EVENT_BLOCKED_CODE: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        event_base = batch_idx * events_stride
        for column in tl.static_range(0, EVENT_WIDTH_CONST):
            tl.store(events_ptr + event_base + column, 0)
        logit_row = tl.load(detection_ptr + batch_idx * detection_stride).to(tl.int64)
        if logit_row < 0:
            req_idx = tl.load(idx_mapping_ptr + batch_idx).to(tl.int64)
            if req_idx >= 0:
                tl.store(blocked_reported_ptr + req_idx, 0)
            return

        req_idx = tl.load(idx_mapping_ptr + batch_idx).to(tl.int64)
        num_sampled = tl.load(num_sampled_ptr + batch_idx).to(tl.int32)
        sequence_index = tl.load(detection_ptr + batch_idx * detection_stride + 4)
        looping_token = tl.load(detection_ptr + batch_idx * detection_stride + 3).to(tl.int64)
        loop_value = tl.load(logits_ptr + logit_row * logits_stride + looping_token).to(tl.float32)
        escape_count = tl.load(escape_count_ptr + req_idx).to(tl.int32)

        offsets = tl.arange(0, NUM_CANDIDATE_SLOTS_PADDED)
        slot_valid = offsets < NUM_CANDIDATE_SLOTS
        values = tl.load(
            candidates_value_ptr + batch_idx * candidate_stride + offsets,
            mask=slot_valid,
            other=-float("inf"),
        ).to(tl.float32)
        token_ids = tl.load(
            candidates_id_ptr + batch_idx * candidate_stride + offsets,
            mask=slot_valid,
            other=-1,
        ).to(tl.int64)
        highest_nonterminal_value = tl.max(values, axis=0)
        lower_nonterminal_tie = (
            tl.sum(
                (
                    (token_ids >= 0)
                    & (token_ids < looping_token)
                    & (values == loop_value)
                ).to(tl.int32),
                axis=0,
            )
            > 0
        )

        selected_token = tl.full((), -1, tl.int64)
        selected_value = tl.full((), -float("inf"), tl.float32)
        if escape_count < MAX_ESCAPES:
            for _candidate_rank in tl.static_range(0, MAX_CANDIDATES):
                best_value = tl.max(values, axis=0)
                best_slot = tl.argmax(values, axis=0)
                best_token = tl.sum(tl.where(offsets == best_slot, token_ids, 0), axis=0)
                finite = best_value > -float("inf")
                target_order_valid = best_value <= loop_value
                gap_valid = best_value >= loop_value - MAX_LOGIT_GAP
                # The detector proves at least MIN_REPEATED_TOKENS exact tokens
                # for periods <= MAX_PERIOD, with MIN_REPEATED_TOKENS >=
                # 2*MAX_PERIOD.  By Fine-Wilf, changing the final token to any
                # different token cannot leave another qualifying <=MAX_PERIOD
                # suffix.  Candidate collection already excludes looping_token,
                # EOS, grammar-masked and non-finite logits.
                take = (
                    (selected_token < 0)
                    & finite
                    & target_order_valid
                    & gap_valid
                )
                selected_token = tl.where(take, best_token, selected_token)
                selected_value = tl.where(take, best_value, selected_value)
                values = tl.where(offsets == best_slot, -float("inf"), values)

        terminal_token = tl.full((), -1, tl.int64)
        terminal_value = tl.full((), -float("inf"), tl.float32)
        lower_terminal_tie = False
        for terminal_idx in tl.static_range(0, NUM_TERMINALS):
            token_id = tl.load(terminal_ids_ptr + terminal_idx).to(tl.int64)
            value = tl.load(logits_ptr + logit_row * logits_stride + token_id).to(tl.float32)
            finite = (value > -float("inf")) & (value < float("inf"))
            lower_terminal_tie |= finite & (token_id < looping_token) & (value == loop_value)
            take = finite & (value > terminal_value)
            terminal_token = tl.where(take, token_id, terminal_token)
            terminal_value = tl.where(take, value, terminal_value)

        # The emitted greedy token must itself be the target argmax.  A higher
        # finite raw/grammar-masked target logit means an upstream invariant is
        # broken; do not conceal that divergence with an escape.
        loop_value_finite = (loop_value > -float("inf")) & (loop_value < float("inf"))
        target_argmax_matches = (
            loop_value_finite
            & (highest_nonterminal_value <= loop_value)
            & (terminal_value <= loop_value)
            & ~lower_nonterminal_tie
            & ~lower_terminal_tie
        )
        selected_token = tl.where(target_argmax_matches, selected_token, -1)
        terminal_token = tl.where(target_argmax_matches, terminal_token, -1)

        event_code = EVENT_NONE_CODE
        replacement = tl.full((), -1, tl.int64)
        replacement_value = tl.full((), -float("inf"), tl.float32)
        if selected_token >= 0:
            event_code = EVENT_ESCAPE_CODE
            replacement = selected_token
            replacement_value = selected_value
            tl.store(escape_count_ptr + req_idx, escape_count + 1)
            tl.store(blocked_reported_ptr + req_idx, 0)
        elif terminal_token >= 0:
            event_code = EVENT_TERMINATE_CODE
            replacement = terminal_token
            replacement_value = terminal_value
        else:
            reported = tl.load(blocked_reported_ptr + req_idx).to(tl.int32)
            event_code = tl.where(
                reported == 0, EVENT_BLOCKED_CODE, EVENT_NONE_CODE
            )
            tl.store(blocked_reported_ptr + req_idx, 1)

        if replacement >= 0:
            tl.store(
                sampled_ptr + batch_idx * sampled_stride + num_sampled - 1,
                replacement,
            )
        if event_code != EVENT_NONE_CODE:
            logit_gap_micros = tl.where(
                replacement >= 0,
                ((loop_value - replacement_value) * 1_000_000.0).to(tl.int64),
                -1,
            )
            tl.store(events_ptr + event_base, event_code)
            tl.store(events_ptr + event_base + 1, req_idx)
            tl.store(events_ptr + event_base + 2, sequence_index)
            tl.store(
                events_ptr + event_base + 3,
                tl.load(detection_ptr + batch_idx * detection_stride + 1),
            )
            tl.store(
                events_ptr + event_base + 4,
                tl.load(detection_ptr + batch_idx * detection_stride + 2),
            )
            tl.store(events_ptr + event_base + 5, looping_token)
            tl.store(events_ptr + event_base + 6, replacement)
            tl.store(events_ptr + event_base + 7, escape_count)
            tl.store(events_ptr + event_base + 8, logit_gap_micros)
            tl.store(events_ptr + event_base + 9, num_sampled - 1)

    _KERNELS = detect_kernel, block_candidates_kernel, choose_kernel
    return _KERNELS


class BoundedLoopEscapeRuntime:
    """Per-sampler device state, keyed by vLLM request-state slot."""

    def __init__(
        self,
        *,
        torch: Any,
        max_num_reqs: int,
        device: Any,
        policy: RuntimePolicy,
        eos_ids: tuple[int, ...],
        vocab_size: int,
    ) -> None:
        if vocab_size < 1 or any(token_id >= vocab_size for token_id in eos_ids):
            raise RuntimeContractError("EOS token IDs must be inside the target vocabulary")
        self.torch = torch
        self.policy = policy
        self.vocab_size = vocab_size
        self.eos_ids = torch.tensor(eos_ids, dtype=torch.int64, device=device)
        self.escape_counts = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.blocked_reported = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.slot_request_ids: list[str | None] = [None] * max_num_reqs
        self._workspace: tuple[Any, Any, Any] | None = None
        self._workspace_shape: tuple[int, int] | None = None

    def _reset_reused_slots(self, input_batch: Any) -> None:
        changed: list[int] = []
        for req_id, slot_raw in zip(
            input_batch.req_ids, input_batch.idx_mapping_np.tolist(), strict=True
        ):
            slot = int(slot_raw)
            if slot < 0:
                continue
            if self.slot_request_ids[slot] != req_id:
                self.slot_request_ids[slot] = req_id
                changed.append(slot)
        if changed:
            indices = self.torch.tensor(changed, dtype=self.torch.int64, device=self.eos_ids.device)
            self.escape_counts.index_fill_(0, indices, 0)
            self.blocked_reported.index_fill_(0, indices, 0)

    def apply(self, *, logits: Any, input_batch: Any, sampler: Any, sampler_output: Any) -> None:
        """Mutate only a proved looping final token and attach async telemetry."""

        if not self._eligible(
            input_batch=input_batch,
            sampler=sampler,
            sampler_output=sampler_output,
        ):
            return
        self._reset_reused_slots(input_batch)
        sampled = sampler_output.sampled_token_ids
        num_sampled = sampler_output.num_sampled
        assert num_sampled is not None
        num_reqs = input_batch.num_reqs
        vocab_size = logits.shape[1]
        if vocab_size != self.vocab_size:
            raise RuntimeContractError("target vocabulary size changed after runtime activation")
        num_blocks = (vocab_size + self.policy.vocab_block_size - 1) // self.policy.vocab_block_size
        candidate_slots = num_blocks * self.policy.top_per_vocab_block
        candidate_slots_padded = 1 << (candidate_slots - 1).bit_length()

        workspace_shape = (num_reqs, candidate_slots)
        if self._workspace is None or self._workspace_shape != workspace_shape:
            self._workspace = (
                self.torch.empty((num_reqs, 5), dtype=self.torch.int64, device=logits.device),
                self.torch.empty(
                    workspace_shape, dtype=self.torch.float32, device=logits.device
                ),
                self.torch.empty(
                    workspace_shape, dtype=self.torch.int64, device=logits.device
                ),
            )
            self._workspace_shape = workspace_shape
        detection, candidate_values, candidate_ids = self._workspace
        events = self.torch.zeros(
            (num_reqs, EVENT_WIDTH), dtype=self.torch.int64, device=logits.device
        )
        detect, block_candidates, choose = _load_kernels()
        common = {
            "MAX_PERIOD": self.policy.max_period,
            "MIN_COPIES": self.policy.min_copies,
            "MIN_REPEATED_TOKENS": self.policy.min_repeated_tokens,
            "MAX_REPEATED_SPAN": self.policy.max_repeated_span,
        }
        detect[(num_reqs,)](
            detection,
            detection.stride(0),
            sampled,
            sampled.stride(0),
            num_sampled,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            sampler.req_states.all_token_ids.gpu,
            sampler.req_states.all_token_ids.gpu.stride(0),
            sampler.req_states.prompt_len.gpu,
            sampler.req_states.total_len.gpu,
            self.eos_ids,
            **common,
            MAX_ROUND_TOKENS=sampled.shape[1],
            NUM_TERMINALS=len(self.eos_ids),
            num_warps=1,
        )
        block_candidates[(num_reqs, num_blocks)](
            logits,
            logits.stride(0),
            vocab_size,
            detection,
            detection.stride(0),
            self.eos_ids,
            candidate_values,
            candidate_ids,
            candidate_slots,
            BLOCK_SIZE=self.policy.vocab_block_size,
            TOP_PER_BLOCK=self.policy.top_per_vocab_block,
            NUM_TERMINALS=len(self.eos_ids),
            num_warps=4,
        )
        choose[(num_reqs,)](
            logits,
            logits.stride(0),
            detection,
            detection.stride(0),
            candidate_values,
            candidate_ids,
            candidate_slots,
            self.eos_ids,
            sampled,
            sampled.stride(0),
            num_sampled,
            input_batch.idx_mapping,
            self.escape_counts,
            self.blocked_reported,
            events,
            events.stride(0),
            NUM_CANDIDATE_SLOTS=candidate_slots,
            NUM_CANDIDATE_SLOTS_PADDED=candidate_slots_padded,
            MAX_CANDIDATES=self.policy.max_candidates - 1,
            MAX_ESCAPES=self.policy.max_escapes,
            NUM_TERMINALS=len(self.eos_ids),
            MAX_LOGIT_GAP=self.policy.max_logit_gap,
            EVENT_WIDTH_CONST=EVENT_WIDTH,
            EVENT_NONE_CODE=EVENT_NONE,
            EVENT_ESCAPE_CODE=EVENT_ESCAPE,
            EVENT_TERMINATE_CODE=EVENT_TERMINATE,
            EVENT_BLOCKED_CODE=EVENT_BLOCKED,
            **common,
            num_warps=1,
        )
        # AsyncOutput copies this tiny tensor on its existing copy stream; there
        # is no per-round host synchronization or decode-path file I/O.
        sampler_output.qwen_loop_escape_events = events

    @staticmethod
    def _eligible(*, input_batch: Any, sampler: Any, sampler_output: Any) -> bool:
        import numpy as np

        indices = input_batch.idx_mapping_np
        states = sampler.sampling_states
        if (
            sampler_output.num_sampled is None
            or sampler_output.logprobs_tensors is not None
            or sampler_output.sampling_mask_tensors is not None
        ):
            return False
        unsupported_stop = getattr(
            sampler, "_qwen_bounded_loop_escape_unsupported_stop", None
        )
        if unsupported_stop is None or np.any(unsupported_stop[indices]):
            return False
        if np.any(indices < 0) or np.any(states.temperature.np[indices] != 0.0):
            return False
        if np.any(states.min_p.np[indices] != 0.0) or np.any(states.top_p.np[indices] != 1.0):
            return False
        top_k = states.top_k.np[indices]
        if np.any((top_k != 1) & (top_k != states.vocab_size)):
            return False
        if np.any(sampler.penalties_state.use_penalty[indices]):
            return False
        if np.any(sampler.logit_bias_state.use_logit_bias[indices]):
            return False
        if np.any(sampler.bad_words_state.num_bad_words.np[indices] != 0):
            return False
        thinking = sampler.thinking_budget_state
        return not (
            thinking.enabled and np.any(thinking.use_thinking_budget[indices])
        )


def maybe_apply_bounded_loop_escape(
    *,
    logits: Any,
    input_batch: Any,
    sampler: Any,
    sampler_output: Any,
) -> None:
    """Apply the enabled policy without changing the normal sampling ABI."""

    configuration = getattr(sampler, "_qwen_bounded_loop_escape_configuration", ...)
    if configuration is ...:
        configuration = policy_from_environment()
        sampler._qwen_bounded_loop_escape_configuration = configuration
    if configuration is None:
        return
    policy, eos_ids, telemetry = configuration
    global _ACTIVE_TELEMETRY
    _ACTIVE_TELEMETRY = telemetry
    runtime = getattr(sampler, "_qwen_bounded_loop_escape_runtime", None)
    if runtime is None:
        import torch

        max_num_reqs = getattr(sampler.req_states, "max_num_reqs", None)
        if not isinstance(max_num_reqs, int) or max_num_reqs < 1:
            raise RuntimeContractError("request-state capacity is unavailable")
        runtime = BoundedLoopEscapeRuntime(
            torch=torch,
            max_num_reqs=max_num_reqs,
            device=logits.device,
            policy=policy,
            eos_ids=eos_ids,
            vocab_size=logits.shape[1],
        )
        sampler._qwen_bounded_loop_escape_runtime = runtime
    runtime.apply(
        logits=logits,
        input_batch=input_batch,
        sampler=sampler,
        sampler_output=sampler_output,
    )


_ACTIVE_TELEMETRY: Path | None = None
_TELEMETRY_QUEUE: queue.Queue[tuple[Path, tuple[bytes, ...]]] = queue.Queue(
    maxsize=TELEMETRY_QUEUE_CAPACITY
)
_TELEMETRY_WORKER: threading.Thread | None = None
_TELEMETRY_WORKER_LOCK = threading.Lock()


def record_loop_escape_events(req_ids: list[str], rows: Any) -> None:
    """Queue rare events after D2H without ever failing a sampled response."""

    try:
        _record_loop_escape_events(req_ids, rows)
    except Exception:
        # Telemetry is diagnostic evidence, not part of token correctness.  A
        # malformed row or local logging failure must not discard model output.
        LOGGER.exception("failed to enqueue Qwen bounded-loop telemetry")


def _record_loop_escape_events(req_ids: list[str], rows: Any) -> None:
    """Validate and serialize event rows before the bounded background queue."""

    telemetry = _ACTIVE_TELEMETRY
    if telemetry is None:
        configuration = policy_from_environment()
        if configuration is None:
            return
        _policy, _eos_ids, telemetry = configuration
    actions = {EVENT_ESCAPE: "escape", EVENT_TERMINATE: "terminate", EVENT_BLOCKED: "blocked"}
    records: list[bytes] = []
    for req_id, raw in zip(req_ids, rows.tolist(), strict=True):
        code = int(raw[0])
        if code == EVENT_NONE:
            continue
        action = actions.get(code)
        if action is None:
            raise RuntimeContractError(f"unknown loop-escape event code: {code}")
        payload = {
            "schema": "qwen-bounded-loop-escape-event-v2",
            "timestamp": datetime.now(UTC).isoformat(),
            "request_id": req_id,
            "action": action,
            "request_slot": int(raw[1]),
            "output_token_index": int(raw[2]),
            "period": int(raw[3]),
            "minimum_proven_repeated_suffix_tokens": int(raw[4]),
            "looping_token_id": int(raw[5]),
            "replacement_token_id": None if int(raw[6]) < 0 else int(raw[6]),
            "prior_escape_count": int(raw[7]),
            "logit_gap": None if int(raw[8]) < 0 else int(raw[8]) / 1_000_000.0,
            "round_token_index": int(raw[9]),
        }
        records.append(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        LOGGER.warning("Qwen bounded loop %s: %s", action, payload)
    if not records:
        return
    _enqueue_telemetry(telemetry, tuple(records))


def _enqueue_telemetry(telemetry: Path, records: tuple[bytes, ...]) -> None:
    """Start the single daemon writer lazily and enqueue without blocking output."""

    global _TELEMETRY_WORKER
    try:
        worker = _TELEMETRY_WORKER
        if worker is None or not worker.is_alive():
            with _TELEMETRY_WORKER_LOCK:
                worker = _TELEMETRY_WORKER
                if worker is None or not worker.is_alive():
                    worker = threading.Thread(
                        target=_telemetry_worker_main,
                        name="qwen-bounded-loop-telemetry",
                        daemon=True,
                    )
                    worker.start()
                    _TELEMETRY_WORKER = worker
        _TELEMETRY_QUEUE.put_nowait((telemetry, records))
    except queue.Full:
        LOGGER.error(
            "Qwen bounded-loop telemetry queue is full; event remains in server logs"
        )
    except Exception:
        LOGGER.exception(
            "failed to start or enqueue Qwen bounded-loop telemetry; "
            "event remains in server logs"
        )


def _telemetry_worker_main() -> None:
    """Persist telemetry off the token-delivery path; never terminate on I/O errors."""

    while True:
        telemetry, records = _TELEMETRY_QUEUE.get()
        try:
            _append_telemetry_records(telemetry, records)
        except Exception:
            LOGGER.exception(
                "failed to persist Qwen bounded-loop telemetry to %s", telemetry
            )
        finally:
            _TELEMETRY_QUEUE.task_done()


def _append_telemetry_records(telemetry: Path, records: tuple[bytes, ...]) -> None:
    """Durably append one event batch from the background writer."""

    descriptor = os.open(
        telemetry,
        os.O_APPEND | os.O_CLOEXEC | os.O_CREAT | os.O_NOFOLLOW | os.O_WRONLY,
        0o600,
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) & 0o077
            or metadata.st_uid != os.getuid()
        ):
            raise RuntimeContractError(
                "opened loop-escape telemetry is not a private owned regular file"
            )
        for record in records:
            pending = memoryview(record)
            while pending:
                written = os.write(descriptor, pending)
                if written <= 0:
                    raise RuntimeContractError("short loop-escape telemetry write")
                pending = pending[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

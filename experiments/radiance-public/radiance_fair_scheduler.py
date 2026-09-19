"""Single-GPU response scheduling with separate cache allocators and RAM images.

This module is qualified only for the pinned single-rank, synchronous V2 R4D /
DFlash runtime. The scheduler dispatches exactly one request through completion,
including prefill, thinking, tool arguments, and answer tokens. A waiting chat
gets the GPU at a response boundary, overlapping handover with client-side tools.
Explicit higher priority can reserve an answer or park a response at a safe step.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import weakref
from pathlib import Path
from typing import ClassVar
from uuid import uuid4

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler

STATUS = "/dev/shm/qwen-radiance-fair-public"
HEX = re.compile(r"[0-9a-f]{64}\Z")
HANDOVER_TOKEN = "qwen_tool_handover"
OUTCOME_ACK_SECONDS = 0.25
_outcome_tasks = set()
_phase_scheduler = None
PHASE_SCHEMA = "urn:qwen-r9700:request-phases:v2"
ROUND_LOG_SCHEMA = "urn:qwen-r9700:decode-rounds:v1"
ROUND_LOG_MAX_BYTES = 8 * 1024 * 1024
# The qualified DFlash decode path schedules at most eight model tokens per
# execution. A larger batch is a prompt-prefill execution and must not be
# treated as the first decode boundary.
DECODE_SYNC_MAX_SCHEDULED_TOKENS = 8
# The release diagnosis reproduced the slow queue state after roughly thirty
# decode launches, while a host pause and HIP GPU timing events did not repair it.
# Retire that state before the following launch and re-arm the same bounded
# recovery every interval.  A single fence per request was insufficient: the
# live 60K capture returned to the slow state later in the same answer.
DECODE_SYNC_RECOVERY_AFTER_ROUNDS = 30


class RequestPhases:
    """Bounded numeric timings; never retain prompts, token IDs or tool data."""

    def __init__(self):
        self.live = {}
        self.recent = {}
        # Completed-round events are flushed to a separate bounded JSONL feed
        # by FairScheduler._write_phase_status. Keeping this queue here means
        # a failed telemetry write can be retried without losing a round.
        self._round_events = []
        # The qualified scheduler dispatches one response at a time. Keeping
        # the previous completed generation request here lets us measure the
        # wall-clock duration of consecutive backend rounds without charging a
        # GPU handover or another chat's work to the resumed response.
        self._last_generation_request_id = None
        self._last_generation_at = None

    def _reset_generation_clock(self, request_id=None):
        if request_id is None or request_id == self._last_generation_request_id:
            self._last_generation_request_id = None
            self._last_generation_at = None

    def set(self, request, phase, blocker=None):
        now = time.monotonic()
        rid = request.request_id
        if rid not in self.live:
            self.live[rid] = {
                **bank_identity(bank_key(request)),
                "request_id": hashlib.sha256(rid.encode()).hexdigest(),
                "phase": phase,
                "blocker": blocker,
                "timings_ms": {},
                "started": now,
                "since": now,
                "first_token_ms": None,
                "last_round_ms": None,
                "last_acceptance_rate": None,
                "generation_rounds": 0,
                "draft_tokens": 0,
                "accepted_tokens": 0,
            }
            return True
        row = self.live[rid]
        if row["phase"] == phase and row["blocker"] == blocker:
            return False
        elapsed = max(0, (now - row["since"]) * 1000)
        row["timings_ms"][row["phase"]] = row["timings_ms"].get(row["phase"], 0) + elapsed
        row.update(phase=phase, blocker=blocker, since=now)
        if phase == "generate" and row["first_token_ms"] is None:
            row["first_token_ms"] = max(0, (now - row["started"]) * 1000)
        if phase != "generate":
            self._reset_generation_clock(rid)
        return True

    def observe_generation(
        self,
        request,
        *,
        draft_tokens=0,
        accepted_tokens=0,
        scheduled_shape=None,
        now=None,
    ):
        """Record content-free round and speculative acceptance counters.

        This runs after a completed engine step. It only reads scheduler-side
        integer counts; it never synchronizes the GPU or inspects token data.
        A round duration is published only for consecutive steps of the same
        request, so time spent while another chat owns the GPU is excluded.
        """
        row = self.live.get(request.request_id)
        if row is None:
            return
        now = time.monotonic() if now is None else now
        if not isinstance(draft_tokens, int) or draft_tokens < 0:
            draft_tokens = 0
        if not isinstance(accepted_tokens, int) or accepted_tokens < 0:
            accepted_tokens = 0
        if accepted_tokens > draft_tokens:
            accepted_tokens = draft_tokens
        if (
            self._last_generation_request_id == request.request_id
            and self._last_generation_at is not None
        ):
            row["last_round_ms"] = max(0.0, (now - self._last_generation_at) * 1000)
        self._last_generation_request_id = request.request_id
        self._last_generation_at = now
        # A final target-only/EOS step can legitimately have no draft
        # tokens.  It still counts as a generation round, but it must not
        # erase the most recent meaningful speculative acceptance value from
        # the live feed.  Otherwise Pi loses the percentage exactly when a
        # response finishes and the retained recent row is displayed.
        if draft_tokens:
            row["last_acceptance_rate"] = accepted_tokens / draft_tokens
        row["generation_rounds"] += 1
        row["draft_tokens"] += draft_tokens
        row["accepted_tokens"] += accepted_tokens
        event = {
            "schema": ROUND_LOG_SCHEMA,
            "pid": os.getpid(),
            "observed_at_ms": int(time.time() * 1000),
            "chat_id": row["chat_id"],
            "generation": row["generation"],
            "request_id": row["request_id"],
            "round": row["generation_rounds"],
            "round_ms": (
                round(row["last_round_ms"], 3)
                if row["last_round_ms"] is not None
                else None
            ),
            "draft_tokens": draft_tokens,
            "accepted_tokens": accepted_tokens,
            "acceptance_rate": (
                accepted_tokens / draft_tokens if draft_tokens else None
            ),
        }
        if scheduled_shape is not None:
            event["scheduled_shape"] = scheduled_shape
        self._round_events.append(event)

    def take_round_events(self):
        events = self._round_events
        self._round_events = []
        return events

    def restore_round_events(self, events):
        if events:
            self._round_events[0:0] = events

    def row(self, request):
        value = self.live.get(request.request_id)
        if value is None:
            return None
        now = time.monotonic()
        elapsed = max(0, (now - value["since"]) * 1000)
        timings = {key: round(ms, 3) for key, ms in value["timings_ms"].items()}
        timings[value["phase"]] = round(timings.get(value["phase"], 0) + elapsed, 3)
        return {
            **{
                key: value[key]
                for key in ("chat_id", "generation", "request_id", "phase", "blocker")
            },
            "input_tokens": request.num_prompt_tokens,
            "computed_tokens": request.num_computed_tokens,
            "cached_tokens": value.get("cached_tokens"),
            "elapsed_ms": round(max(0, (now - value["started"]) * 1000), 3),
            "phase_elapsed_ms": round(elapsed, 3),
            "first_token_ms": value["first_token_ms"],
            "last_round_ms": (
                round(value["last_round_ms"], 3)
                if value["last_round_ms"] is not None
                else None
            ),
            "last_acceptance_rate": value["last_acceptance_rate"],
            "generation_rounds": value["generation_rounds"],
            "draft_tokens": value["draft_tokens"],
            "accepted_tokens": value["accepted_tokens"],
            "acceptance_rate": (
                value["accepted_tokens"] / value["draft_tokens"]
                if value["draft_tokens"] > 0
                else None
            ),
            "timings_ms": timings,
        }

    def finish(self, request):
        if request.request_id not in self.live:
            return
        self.set(request, "complete")
        row = self.row(request)
        key = (row["chat_id"], row["generation"])
        # Only the most recent response per generation, at most 16 generations.
        self.recent.pop(key, None)
        self.recent[key] = row
        while len(self.recent) > 16:
            self.recent.pop(next(iter(self.recent)))
        self.live.pop(request.request_id)
        self._reset_generation_clock(request.request_id)


def worker_phase(phase):
    # The qualified TP1 synchronous runner shares the scheduler's process.
    # Other process layouts simply omit this optional diagnostic hook.
    scheduler = _phase_scheduler() if _phase_scheduler is not None else None
    if scheduler is not None and scheduler.response_request is not None:
        scheduler._phase(scheduler.response_request, phase)


def _record_decode_sync(runner, banks, *, key, total, reason):
    """Synchronize the transition and publish only numeric diagnostic state.

    Cache/mamba preparation can enqueue work on more than the worker's current
    stream.  The latency investigation showed that a device-wide fence retires
    the residual HIP queue state, while a current-stream fence is not sufficient
    for every connector/runtime path.  This is called only once at a transition
    boundary (and once for the long-response recovery), so it is not part of the
    steady-state round.
    """
    cuda = banks.torch.cuda
    started = time.perf_counter()
    device_sync = getattr(cuda, "synchronize", None)
    if callable(device_sync):
        device_sync()
        mode = "device"
    else:
        # Older/test runtimes may expose only current_stream(). Keep the
        # compatibility fallback instead of silently skipping the fence.
        cuda.current_stream().synchronize()
        mode = "current_stream"
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    stream = cuda.current_stream()
    stream_value = getattr(stream, "cuda_stream", stream)
    try:
        stream_id = str(int(stream_value))
    except (TypeError, ValueError):
        stream_id = str(stream_value)
    count = int(getattr(runner, "_qwen_decode_sync_count", 0)) + 1
    runner._qwen_decode_sync_count = count
    runner._qwen_decode_last_sync_mode = mode
    runner._qwen_decode_last_sync_reason = reason
    runner._qwen_decode_last_sync_elapsed_ms = round(elapsed_ms, 3)
    runner._qwen_decode_last_sync_at_ms = int(time.time() * 1000)
    status_path = getattr(banks, "status_path", None)
    if status_path:
        write_status(
            status_path + "-sync.json",
            {
                "schema": "urn:qwen-r9700:decode-sync:v1",
                "pid": os.getpid(),
                "updated_at": time.time(),
                "count": count,
                "mode": mode,
                "reason": reason,
                "scheduled_tokens": int(total),
                "elapsed_ms": round(elapsed_ms, 3),
                "stream": stream_id,
                # The normal scheduler key is already a digest. Hash the
                # compatibility fallback as well so chat/generation identity
                # never enters this diagnostic file.
                "transition": hashlib.sha256(str(key).encode()).hexdigest(),
            },
        )
    return True


def _sync_decode_transition(runner, scheduler_output, metadata):
    """Fence the first decode after each prefill/cache-restore transition.

    The slow-round diagnosis found persistent HIP stream/queue state after a
    prefill or cache restore. A device-wide synchronization clears that state,
    but doing it for every decode round would turn the workaround into a
    permanent throughput penalty. The scheduler supplies a content-free
    request/transition key, so a later request in the same chat bank is armed
    again without fencing every decode round.

    This hook runs before the connector's final preparation. It only arms the
    transition and counts the round; the actual fence is deferred to
    ``after_forward_prepare`` so every cache/mamba copy that belongs to the
    round is included in the drain.
    """
    total = getattr(scheduler_output, "total_num_scheduled_tokens", None)
    if total is None:
        return False
    try:
        total = int(total)
    except (TypeError, ValueError):
        return False
    # A prefill, cache-restore frame, or explicit bank barrier starts a new
    # transition. It does not need this decode fence itself, but it must clear
    # the previous decode latch so the first subsequent decode is fenced.
    if metadata.get("barrier") or total <= 0 or total > DECODE_SYNC_MAX_SCHEDULED_TOKENS:
        if metadata.get("barrier") or total > DECODE_SYNC_MAX_SCHEDULED_TOKENS:
            runner._qwen_decode_sync_key = None
            runner._qwen_decode_prepared_key = None
            runner._qwen_decode_round_key = None
            runner._qwen_decode_rounds = 0
            runner._qwen_decode_recovery_pending = None
        return False
    bank = metadata.get("bank")
    if bank is None:
        return False
    key = metadata.get("decode_sync_key") or bank
    banks = getattr(runner, "qwen_banks", None)
    if banks is None:
        return False
    previous_key = getattr(runner, "_qwen_decode_round_key", None)
    if previous_key != key:
        runner._qwen_decode_round_key = key
        runner._qwen_decode_rounds = 0
    runner._qwen_decode_rounds = int(getattr(runner, "_qwen_decode_rounds", 0)) + 1

    # The first fence is performed by ``after_forward_prepare``. That hook is
    # placed after mamba/cache preparation and immediately before the model
    # forward; the old before-forward location was too early to drain those
    # asynchronous copies. The deferred fence below remains the recovery path
    # for residual queue state that accumulates during a long response.
    should_sync = (
        runner._qwen_decode_rounds > DECODE_SYNC_RECOVERY_AFTER_ROUNDS
        and (runner._qwen_decode_rounds - 1) % DECODE_SYNC_RECOVERY_AFTER_ROUNDS == 0
    )
    if not should_sync:
        return False
    runner._qwen_decode_recovery_pending = (key, total)
    return True


def after_forward_prepare(runner, scheduler_output):
    """Fence the first decode after all cache/mamba preparation has run."""
    metadata = scheduler_output.qwen_fair
    if metadata is None or metadata.get("barrier"):
        return False
    total = getattr(scheduler_output, "total_num_scheduled_tokens", None)
    try:
        total = int(total)
    except (TypeError, ValueError):
        return False
    if total <= 0 or total > DECODE_SYNC_MAX_SCHEDULED_TOKENS:
        return False
    bank = metadata.get("bank")
    key = metadata.get("decode_sync_key") or bank
    if key is None or getattr(runner, "_qwen_decode_prepared_key", None) == key:
        prepared = False
    else:
        prepared = True
    banks = getattr(runner, "qwen_banks", None)
    if banks is None:
        return False
    did_sync = False
    if prepared:
        _record_decode_sync(
            runner,
            banks,
            key=key,
            total=total,
            reason="after_cache_prepare",
        )
        runner._qwen_decode_prepared_key = key
        runner._qwen_decode_sync_key = key
        did_sync = True
    pending = getattr(runner, "_qwen_decode_recovery_pending", None)
    if pending is not None:
        pending_key, pending_total = pending
        if pending_key == key:
            _record_decode_sync(
                runner,
                banks,
                key=key,
                total=pending_total,
                reason="long_response_recovery",
            )
            did_sync = True
        runner._qwen_decode_recovery_pending = None
    return did_sync


def prepare_tool_handover(request):
    """Opt in only labelled, single-choice chat requests that can produce tools."""
    params = dict(request.kv_transfer_params or {})
    params.pop(HANDOVER_TOKEN, None)
    if (
        params.get("qwen_chat") is not None
        and request.tools
        and request.tool_choice != "none"
        and (request.n or 1) == 1
        and not request.use_beam_search
    ):
        params[HANDOVER_TOKEN] = uuid4().hex
    request.kv_transfer_params = params or None


def report_tool_handover(engine_client, request, is_tool):
    """Send a parser outcome through existing local IPC without delaying the SSE."""
    token = (request.kv_transfer_params or {}).get(HANDOVER_TOKEN)
    if token is None:
        return

    async def send():
        try:
            await asyncio.wait_for(
                engine_client.engine_core.call_utility_async(
                    "qwen_response_outcome", token, bool(is_tool)
                ),
                timeout=1,
            )
        except Exception:
            # The scheduler's bounded acknowledgement window expires safely.
            # Do not turn an already-complete response into an API failure.
            logging.getLogger(__name__).warning("Tool handover outcome unavailable; grace skipped")

    task = asyncio.create_task(send())
    _outcome_tasks.add(task)
    task.add_done_callback(_outcome_tasks.discard)


class ToolHandover:
    """One response's outcome; no generated text, tool arguments, or tool results."""

    def __init__(self, grace_seconds=2.0):
        if not 0 <= grace_seconds <= 5:
            raise ValueError("invalid tool handover grace")
        self.grace_seconds = grace_seconds
        self.request = None
        self.token = None
        self.finished_at = None
        self.is_tool = None

    def select(self, request):
        if request is self.request:
            return
        self.request = request
        params = getattr(request, "kv_transfer_params", None) or {}
        token = params.get(HANDOVER_TOKEN)
        self.token = (
            token if isinstance(token, str) and re.fullmatch(r"[0-9a-f]{32}", token) else None
        )
        self.finished_at = None
        self.is_tool = None

    def finish(self):
        if self.request is not None and self.request.is_finished() and self.finished_at is None:
            self.finished_at = time.monotonic()

    def acknowledge(self, token, is_tool):
        if self.token is None or token != self.token or type(is_tool) is not bool:
            return False
        self.finish()
        # Late or duplicate acknowledgements cannot renew the two-second window.
        if self.is_tool is None:
            self.is_tool = is_tool
        return True

    def remaining(self):
        self.finish()
        if (
            self.token is None
            or self.finished_at is None
            or self.request.status.name != "FINISHED_STOPPED"
            or self.is_tool is False
            or self.grace_seconds == 0
        ):
            return 0.0
        # Anchor grace to the completed response, never the acknowledgement's
        # processing time: pending connector work can delay local IPC handling.
        deadline = self.finished_at + (self.grace_seconds if self.is_tool else OUTCOME_ACK_SECONDS)
        return max(0.0, deadline - time.monotonic())


def bank_key(request):
    chat = (request.kv_transfer_params or {}).get("qwen_chat")
    if chat is not None:
        if not all(HEX.fullmatch(str(chat.get(k, ""))) for k in ("id", "generation")):
            raise ValueError("invalid chat identity for RAM cache")
        return chat["id"] + ":" + chat["generation"]
    return hashlib.sha256(request.request_id.encode()).hexdigest() + ":" + "0" * 64


def bank_identity(key):
    chat_id, generation = key.split(":", 1)
    if not HEX.fullmatch(chat_id) or not HEX.fullmatch(generation):
        raise ValueError("invalid RAM-cache bank identity")
    return {"chat_id": chat_id, "generation": generation}


def supersedes(candidate, existing):
    """Return whether candidate is another generation of the same labelled chat."""
    if candidate is None or existing is None or candidate == existing:
        return False
    return candidate.split(":", 1)[0] == existing.split(":", 1)[0]


def spans(blocks):
    """Coalesce physical pages so transfers use large contiguous DMA copies."""
    result = []
    for block in sorted(set(blocks)):
        if result and result[-1][1] == block:
            result[-1][1] += 1
        else:
            result.append([block, block + 1])
    return result


def write_status(path, data):
    # Diagnostic metadata lives in tmpfs. Never put a disk fsync in a decode step.
    path = Path(path)
    temporary = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(data, stream, separators=(",", ":"))
    temporary.replace(path)


def append_round_log(path, events):
    """Append numeric decode-round events without syncing the filesystem."""
    if not events:
        return
    path = Path(path)
    payload = b"".join(
        (json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n").encode()
        for event in events
    )
    if path.exists() and path.stat().st_size + len(payload) > ROUND_LOG_MAX_BYTES:
        rotated = path.with_name(path.name + ".1")
        try:
            rotated.unlink()
        except FileNotFoundError:
            pass
        path.replace(rotated)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short decode-round telemetry write")
            view = view[written:]
    finally:
        os.close(fd)


class CacheBanks:
    """Each RAM image has its own allocator and prefix-cache metadata."""

    REQUEST_METHODS: ClassVar = {
        "free",
        "pop_blocks_for_free",
        "get_blocks",
        "get_block_ids",
        "get_block_ids_for_computed_tokens",
        "cache_blocks",
        "remove_skipped_blocks",
        "get_zeroing_block_ids_in_range",
        "record_prefix_cache_stats",
        "get_computed_blocks",
        "get_computed_blocks_for_connector",
        "get_num_common_prefix_blocks",
        "allocate_slots",
        "estimate_cached_tokens",
        "record_blocks_for_zeroing",
        "truncate_computed_blocks",
    }

    def __init__(self, initial, make_manager):
        self.initial = initial
        self.make_manager = make_manager
        self.managers = {}
        self.owners = {}
        self.active = None

    def activate(self, key):
        if key not in self.managers:
            self.managers[key] = (
                self.initial if not self.managers and self.active is None else self.make_manager()
            )
        self.active = key

    def manager(self, request):
        rid = request if isinstance(request, str) else request.request_id
        key = self.owners.get(rid, self.active)
        return self.managers[key] if key is not None else self.initial

    def live_blocks(self):
        pool = self.block_pool
        aliases = getattr(pool, "cached_block_hashes_by_block", {})
        return [
            b.block_id
            for b in pool.blocks
            if b.ref_cnt or b.block_hash is not None or b.block_id in aliases
        ]

    def reset_prefix_cache(self):
        return all(manager.reset_prefix_cache() for manager in self.managers.values())

    def take_events(self):
        return [event for manager in self.managers.values() for event in manager.take_events()]

    def __getattr__(self, name):
        if name in self.REQUEST_METHODS:

            def routed(*args, **kwargs):
                owner = (
                    args[0]
                    if args
                    else next(
                        (
                            kwargs[key]
                            for key in ("request", "request_id", "running_request_id")
                            if key in kwargs
                        ),
                        None,
                    )
                )
                if owner is None:
                    raise TypeError(f"{name} requires a request identity")
                return getattr(self.manager(owner), name)(*args, **kwargs)

            return routed
        manager = self.managers[self.active] if self.active is not None else self.initial
        return getattr(manager, name)


class AnswerPriorities:
    """CPU-only, leased answer ownership. Priority never changes model/cache state."""

    lease_seconds = 60.0

    def __init__(self):
        self.chats = {}

    @staticmethod
    def validate(value):
        if not isinstance(value, dict) or set(value) != {
            "chat_id",
            "client",
            "answer",
            "sequence",
            "priority",
            "active",
        }:
            raise ValueError("invalid priority control fields")
        if not isinstance(value["chat_id"], str) or not HEX.fullmatch(value["chat_id"]):
            raise ValueError("invalid priority chat identity")
        for field in ("client", "answer"):
            if not isinstance(value[field], str) or not re.fullmatch(r"[0-9a-f]{32}", value[field]):
                raise ValueError("invalid priority owner identity")
        if type(value["sequence"]) is not int or not 0 < value["sequence"] < 2**53:
            raise ValueError("invalid priority sequence")
        if type(value["priority"]) is not int or value["priority"] not in (0, 1, 2):
            raise ValueError("priority must be 0, 1 or 2")
        if type(value["active"]) is not bool:
            raise ValueError("invalid answer activity")

    def update(self, value):
        self.validate(value)
        now = time.monotonic()
        old = self.chats.get(value["chat_id"])
        if old is not None:
            if old["client"] == value["client"]:
                if value["sequence"] <= old["sequence"]:
                    if all(old[k] == v for k, v in value.items()):
                        return {"applied": True, "priority": old["priority"]}
                    raise ValueError("stale priority update")
            elif old["active"] and old["expires"] > now:
                raise ValueError("this chat has an active answer in another Pi window")
        # Retain recent sequence tombstones so delayed start/heartbeat messages
        # cannot resurrect a released answer. Bound idle bookkeeping.
        self.chats = {k: v for k, v in self.chats.items() if v["expires"] + 3600 > now}
        if value["chat_id"] not in self.chats and len(self.chats) >= 4096:
            raise ValueError("priority registry is full")
        self.chats[value["chat_id"]] = {**value, "expires": now + self.lease_seconds}
        return {"applied": True, "priority": value["priority"]}

    def level(self, key):
        row = self.chats.get(key.split(":", 1)[0]) if key else None
        return row["priority"] if row and row["active"] and row["expires"] > time.monotonic() else 0


class FairScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = (self.vllm_config.additional_config or {}).get("qwen_fair", {})
        if (
            self.max_num_running_reqs != 2
            or self.scheduler_config.async_scheduling
            or not self.use_v2_model_runner
            or self.defer_block_free
            or self.parallel_config.world_size != 1
        ):
            raise ValueError(
                "RAM time sharing requires the qualified synchronous V2 two-state runtime"
            )
        # V2 needs two persistent request-state slots so a parked request can
        # remain resident. The custom scheduler still dispatches one request.
        self.runner_state_slots = self.max_num_running_reqs
        self.max_num_running_reqs = 1
        if config.get("policy", "response_boundary") != "response_boundary":
            raise ValueError("only response-boundary scheduling is supported")
        # Older launchers may still send quantum_seconds. It cannot enable
        # preemption; zero in telemetry means there is no time-slice deadline.
        self.quantum = 0.0
        self.max_banks = int(config.get("max_cached_chats", 2))
        if not 2 <= self.max_banks <= 4:
            raise ValueError("invalid RAM-cache bank limit")
        self.status_path = str(config.get("status_path", STATUS))

        def make_manager():
            prefill_options = (
                {"num_prefill_lookahead": self.num_prefill_lookahead}
                if hasattr(self, "num_prefill_lookahead")
                else {}
            )
            return KVCacheManager(
                kv_cache_config=self.kv_cache_config,
                max_model_len=self.max_model_len,
                max_in_flight_tokens=self.vllm_config.max_in_flight_tokens,
                enable_caching=self.cache_config.enable_prefix_caching,
                use_eagle=self.use_eagle,
                log_stats=self.log_stats,
                enable_kv_cache_events=self.enable_kv_cache_events,
                dcp_world_size=self.dcp_world_size,
                pcp_world_size=1,
                scheduler_block_size=self.block_size,
                hash_block_size=self.hash_block_size,
                metrics_collector=self.kv_metrics_collector,
                watermark=self.scheduler_config.watermark,
                **prefill_options,
            )

        self.banks = CacheBanks(self.kv_cache_manager, make_manager)
        self.kv_cache_manager = self.banks
        self.parked = {}
        self.last_served = {}
        self.response_request = None
        self.tool_handover = ToolHandover(float(config.get("tool_grace_seconds", 2)))
        self.max_tool_deferral_seconds = float(config.get("max_tool_deferral_seconds", 30))
        if not 1 <= self.max_tool_deferral_seconds <= 300:
            raise ValueError("invalid tool handover fairness limit")
        self.queued_at = {}
        self.grace_status = None
        self.priorities = AnswerPriorities()
        self.priority_hold = None
        self.pending_switch = None
        self.pending_discard = False
        self.drop_banks = []
        self.last_status = 0.0
        self.switch_count = 0
        self.last_handover_seconds = 0.0
        self._step_worker_metadata = None
        self._decode_sync_epochs = {}
        self.request_phases = RequestPhases()
        # The full scheduler feed remains rate-limited, but generation phase
        # counters are published whenever a completed engine round changes
        # them.  Pi watches the phase file separately, so round latency and
        # acceptance can follow streamed token-rate updates without making the
        # larger scheduler snapshot a per-round write.
        self._last_phase_signature = None
        self._install_phase_hooks()
        global _phase_scheduler
        _phase_scheduler = weakref.ref(self)
        self._publish_status(force=True)

    def add_request(self, request):
        key = bank_key(request)
        self.banks.owners[request.request_id] = key
        if hasattr(self, "request_phases"):
            self.request_phases.set(request, "admission")
        try:
            super().add_request(request)
        except Exception:
            # Connector admission can reject a retired generation. Do not leave
            # an owner behind that could later be mistaken for a live bank.
            if self.banks.owners.get(request.request_id) == key:
                self.banks.owners.pop(request.request_id)
            if hasattr(self, "request_phases"):
                self.request_phases.live.pop(request.request_id, None)
            raise
        self.queued_at[request.request_id] = time.monotonic()
        self._publish_status(force=True)

    def _phase(self, request, phase, blocker=None):
        phases = getattr(self, "request_phases", None)
        if phases is None or request.request_id not in phases.live:
            return
        previous = phases.live[request.request_id]["phase"]
        changed = phases.set(request, phase, blocker)
        if not changed:
            return
        if phase in {"cache_restore", "prefill"} and previous not in {
            "cache_restore",
            "prefill",
        }:
            epochs = getattr(self, "_decode_sync_epochs", None)
            if epochs is None:
                epochs = self._decode_sync_epochs = {}
            epochs[request.request_id] = epochs.get(request.request_id, 0) + 1
        self._publish_status(force=True)

    def _cache_wait(self, request):
        connector = getattr(getattr(self, "connector", None), "connector_scheduler", None)
        states = getattr(connector, "_req_status", {})
        if getattr(connector, "_snapshot_settled_tail_only", False):
            for rid, state in states.items():
                if rid != request.request_id and state.req.is_finished():
                    return "cache_update"
        if getattr(states.get(request.request_id), "transfer_jobs", None):
            return "cache_update"
        return "cache_lookup"

    def _install_phase_hooks(self):
        connector = getattr(self, "connector", None)
        if connector is None:
            return
        lookup = connector.get_num_new_matched_tokens

        def measured_lookup(request, computed_tokens):
            self._phase(request, self._cache_wait(request))
            result = lookup(request, computed_tokens)
            if result[0] is not None and request.request_id in self.request_phases.live:
                self.request_phases.live[request.request_id]["cached_tokens"] = (
                    computed_tokens + result[0]
                )
            if result[0]:
                self._phase(request, "cache_restore")
            return result

        connector.get_num_new_matched_tokens = measured_lookup

    def _get_local_prefix_cache_hit(self, request):
        self._phase(request, self._cache_wait(request))
        result = super()._get_local_prefix_cache_hit(request)
        phases = getattr(self, "request_phases", None)
        if phases is not None and request.request_id in phases.live:
            phases.live[request.request_id]["cached_tokens"] = result[1]
        return result

    def _refresh_request_phases(self):
        if not hasattr(self, "request_phases"):
            return
        selected = self.response_request
        if selected is not None and selected.is_finished():
            selected = None
        for request in self.requests.values():
            if request.is_finished():
                continue
            key = self._key(request)
            if getattr(self, "priority_hold", None) is not None and key != self.banks.active:
                self._phase(request, "priority_wait", bank_identity(self.banks.active))
            elif self.grace_status is not None and key != self.banks.active:
                self._phase(request, "tool_grace", bank_identity(self.banks.active))
            elif (
                key != self.banks.active
                and self.pending_switch is None
                and self.priorities.level(key) == 2 > self.priorities.level(self.banks.active)
            ):
                self._phase(request, "priority_preempt", bank_identity(self.banks.active))
            elif selected is not None and request is not selected:
                self._phase(request, "gpu_queue", bank_identity(self._key(selected)))
            elif self.pending_switch is not None and key == self.pending_switch:
                self._phase(request, "handover")
            elif request in self.running:
                self._phase(request, "generate" if request.num_output_tokens else "prefill")
            elif request.status.name == "WAITING_FOR_REMOTE_KVS":
                self._phase(request, "cache_restore")

    def _key(self, request):
        rid = request if isinstance(request, str) else request.request_id
        return self.banks.owners.get(rid)

    def _busy_banks(self):
        # A finished request can remain in self.requests while its connector
        # finishes delayed cleanup. Its allocator must stay alive until removal.
        result = {self._key(r) for r in self.requests.values()} | {
            self._key(rid) for rid in self.finished_req_ids
        }
        result.discard(None)
        return result

    def _retire_bank(self, key):
        self.banks.managers.pop(key, None)
        self.parked.pop(key, None)
        self.last_served.pop(key, None)
        if key not in self.drop_banks:
            self.drop_banks.append(key)
        live_request_ids = set(self.requests) | set(self.finished_req_ids)
        self.banks.owners = {
            request_id: owner
            for request_id, owner in self.banks.owners.items()
            if owner != key or request_id in live_request_ids
        }

    def _make_room(self, target):
        busy = self._busy_banks()
        replaced = [key for key in self.banks.managers if supersedes(target, key)]
        # Never let a successor overwrite a generation whose request or delayed
        # connector cleanup is still live. The active bank will be reconsidered
        # after the ordinary scheduler has drained that work.
        if any(key in busy for key in replaced):
            return False
        # Inactive images of older generations have no future consumer. Drop
        # them now; an active predecessor is retired after the flush barrier.
        for key in replaced:
            if key != self.banks.active:
                self._retire_bank(key)
        if target in self.banks.managers or len(self.banks.managers) < self.max_banks:
            return True
        if self.banks.active in replaced:
            return len(self.banks.managers) - 1 < self.max_banks
        idle = set(self.banks.managers) - busy
        if not idle:
            return False
        victim = min(idle, key=lambda k: self.last_served.get(k, 0))
        self._retire_bank(victim)
        return True

    def _choose(self):
        active = self.banks.active
        self.grace_status = None
        self.priority_hold = None
        self.parked = {k: r for k, r in self.parked.items() if not r.is_finished()}
        waiting = [*self.waiting, *self.skipped_waiting]
        contenders = {self._key(r) for r in waiting} | set(self.parked)
        contenders.discard(None)
        others = {k for k in contenders if k != active and not supersedes(k, active)}
        level = self.priorities.level(active)
        # Priority 2 alone can interrupt a lower-priority response. Synchronous
        # scheduling has consumed the previous output before this decision. Do
        # not swap a bank with a still-pending connector receive or GPU step.
        urgent = {k for k in others if self.priorities.level(k) == 2 > level}
        safe = bool(urgent) and not any(
            getattr(r, "num_in_flight_tokens", 0) or r.status.name == "WAITING_FOR_REMOTE_KVS"
            for r in (*self.running, *waiting)
            if self._key(r) == active
        )
        for key in sorted(urgent, key=lambda k: (self.last_served.get(k, 0), k)):
            slots = len(self.running) + len(self.parked)
            if (
                safe
                and (key in self.parked or slots < self.runner_state_slots)
                and self._make_room(key)
            ):
                self.response_request = next(
                    (r for r in waiting if self._key(r) == key), self.parked.get(key)
                )
                self.tool_handover.select(self.response_request)
                return key
        if level and all(self.priorities.level(k) < level for k in others):
            self.priority_hold = {**bank_identity(active), "priority": level}
        # Silence, a decode-step boundary, and elapsed wall time are not tool
        # execution boundaries. Keep the whole response on its current GPU bank.
        if self.running:
            return active
        # Ownership starts when a response is selected, before the connector
        # can admit it. In particular, asynchronous prefix lookup returns None
        # while vLLM still labels the request WAITING. Reconsidering ownership
        # there lets a fast tool continuation reverse an already-paid RAM/GPU
        # swap without the selected chat ever running.
        if self.response_request is not None and not self.response_request.is_finished():
            return active
        now = time.monotonic()
        self.queued_at = {r.request_id: self.queued_at.get(r.request_id, now) for r in waiting}
        oldest = {}
        for request in waiting:
            key = self._key(request)
            oldest[key] = min(oldest.get(key, now), self.queued_at[request.request_id])
        overdue = {
            key for key, since in oldest.items() if now - since >= self.max_tool_deferral_seconds
        }
        if self.priority_hold is not None:
            # A committed compaction keeps the answer's priority but replaces
            # its bank generation. Holding the predecessor here would deadlock
            # the first request after compaction.
            for key in sorted(k for k in contenders if supersedes(k, active)):
                if self._make_room(key):
                    self.response_request = next(r for r in waiting if self._key(r) == key)
                    self.tool_handover.select(self.response_request)
                    self.priority_hold = None
                    return key
            # The client is executing tools (possibly with no backend request).
            # Keep the bank and drain cleanup; admit its next tool continuation.
            self.response_request = next((r for r in waiting if self._key(r) == active), None)
            self.tool_handover.select(self.response_request)
            return active
        self.tool_handover.select(self.response_request)
        remaining = self.tool_handover.remaining()
        higher = any(self.priorities.level(k) > level for k in others)
        if remaining > 0 and not higher and not overdue - {active}:
            continuation = next((r for r in waiting if self._key(r) == active), None)
            if self.tool_handover.is_tool and continuation is not None:
                # Tool result returned during grace. No barrier or RAM copy has
                # started, so continue on the already-resident GPU bank.
                self.response_request = continuation
                self.tool_handover.select(continuation)
            else:
                self.grace_status = {
                    **bank_identity(active),
                    "phase": "tool_grace" if self.tool_handover.is_tool else "response_outcome",
                    "remaining_seconds": remaining,
                }
            return active
        self.response_request = None
        self.tool_handover.select(None)
        # Restore and allocator retries also belong to the current response.
        # A pending restore may still write GPU pages; a memory preemption is
        # not permission to interrupt this response by switching chat banks.
        if any(
            self._key(r) == active and r.status.name in {"WAITING_FOR_REMOTE_KVS", "PREEMPTED"}
            for r in waiting
        ):
            return active
        candidates = {self._key(r) for r in waiting} | set(self.parked)
        candidates.update(self._key(rid) for rid in self.finished_req_ids)
        candidates.discard(None)
        if not candidates:
            return active
        for key in sorted(
            candidates,
            key=lambda k: (
                -self.priorities.level(k),
                k not in overdue,
                oldest[k] if k in overdue else self.last_served.get(k, 0),
                k == active,
                k,
            ),
        ):
            if self._make_room(key):
                self.response_request = next(
                    (r for r in waiting if self._key(r) == key), self.parked.get(key)
                )
                self.tool_handover.select(self.response_request)
                return key
        return active

    def _parent_step(self, throttle_prefills=False, *, barrier=False):
        active = self.banks.active
        held_waiting = [r for r in self.waiting if self._key(r) != active]
        held_skipped = [r for r in self.skipped_waiting if self._key(r) != active]
        held_finished = {rid for rid in self.finished_req_ids if self._key(rid) != active}
        self.waiting.remove_requests(held_waiting)
        self.skipped_waiting.remove_requests(held_skipped)
        self.finished_req_ids.difference_update(held_finished)
        budget = self.max_num_scheduled_tokens
        if barrier or (
            self.grace_status is not None and self.grace_status["phase"] == "response_outcome"
        ):
            self.max_num_scheduled_tokens = 0
        try:
            output = super().schedule(throttle_prefills)
        finally:
            self.max_num_scheduled_tokens = budget
            for request in held_waiting:
                self.waiting.add_request(request)
            for request in held_skipped:
                self.skipped_waiting.add_request(request)
            self.finished_req_ids.update(held_finished)
        return output

    def schedule(self, throttle_prefills: bool = False):
        if self.pending_switch is not None:
            if self.running:
                raise RuntimeError("cannot hand over the GPU before the current response finishes")
            target = self.pending_switch
            self.pending_switch = None
            previous = self.banks.active
            discard_previous = self.pending_discard or supersedes(target, previous)
            self.pending_discard = False
            if discard_previous:
                if self.running or previous in self._busy_banks():
                    raise RuntimeError("cannot replace a RAM-cache generation while it is live")
                # A new generation is the complete successor for this chat. Its
                # predecessor must not consume an inactive RAM bank.
                self._retire_bank(previous)
                save_blocks = []
            else:
                save_blocks = self.banks.live_blocks() if previous in self.banks.managers else []
            self.banks.activate(target)
            parked = self.parked.pop(target, None)
            if parked is not None and not parked.is_finished():
                self.running.append(parked)
            self.switch_count += 1
            metadata = self._worker_metadata(
                save_blocks,
                discard_active=discard_previous,
            )
            self._step_worker_metadata = metadata
            output = self._parent_step(throttle_prefills)
            output.qwen_fair = metadata
        else:
            target = self._choose()
            if target is not None and self.banks.active is None:
                self.banks.activate(target)
            if target != self.banks.active:
                if self.running:
                    if self.priorities.level(target) != 2 or self.priorities.level(
                        target
                    ) <= self.priorities.level(self.banks.active):
                        raise RuntimeError("only higher priority 2 may interrupt a response")
                    if len(self.running) != 1 or self.banks.active in self.parked:
                        raise RuntimeError("invalid parked GPU request ownership")
                    # Keep RUNNING status, allocator references, speculative
                    # tokens and V2 request-state slot. Normal vLLM preemption
                    # frees these and would force a rebuild on resume.
                    self.parked[self.banks.active] = self.running.pop()
                # The zero-token frame drains the old bank's final store jobs
                # before any GPU bytes are overwritten by the next bank.
                self.pending_switch = target
                self.pending_discard = self.banks.active in self.drop_banks or supersedes(
                    target, self.banks.active
                )
                metadata = self._worker_metadata([], barrier=True)
                self._step_worker_metadata = metadata
                output = self._parent_step(throttle_prefills, barrier=True)
                output.qwen_fair = metadata
            else:
                metadata = self._worker_metadata([])
                self._step_worker_metadata = metadata
                output = self._parent_step(throttle_prefills)
                output.qwen_fair = metadata
        if self.running:
            # Use the request actually admitted by vLLM (it may skip a blocked
            # request in the same chat). A cache swap or cleanup-only frame is
            # not service; only an admitted response updates fairness order.
            self.response_request = self.running[0]
            self.tool_handover.select(self.response_request)
            self.last_served[self.banks.active] = time.monotonic()
        self._refresh_request_phases()
        self._attach_decode_sync_key(output, metadata)
        self._publish_status()
        return output

    def _build_kv_connector_meta(self, connector, scheduler_output):
        # The connector builds its flush set inside super().schedule(), before
        # schedule() returns. Attach the barrier early enough for that pass.
        scheduler_output.qwen_fair = self._step_worker_metadata
        return super()._build_kv_connector_meta(connector, scheduler_output)

    @staticmethod
    def _scheduled_request_ids(scheduler_output):
        scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
        if not hasattr(scheduled, "items"):
            return ()
        request_ids = []
        for request_id, count in scheduled.items():
            try:
                count = int(count)
            except (TypeError, ValueError):
                continue
            if count > 0:
                request_ids.append(str(request_id))
        return tuple(sorted(set(request_ids)))

    @staticmethod
    def _scheduled_shape(scheduler_output):
        """Return bounded per-round shape metadata without request payloads."""
        total = getattr(scheduler_output, "total_num_scheduled_tokens", None)
        try:
            total = int(total) if total is not None else None
        except (TypeError, ValueError):
            total = None
        scheduled = getattr(scheduler_output, "num_scheduled_tokens", None)
        counts = []
        if hasattr(scheduled, "items"):
            for value in scheduled.values():
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    counts.append(value)
        elif total is not None and total > 0:
            counts = [total]
        counts.sort()
        drafts = getattr(scheduler_output, "scheduled_spec_decode_tokens", None)
        draft_widths = []
        if hasattr(drafts, "items"):
            for value in drafts.values():
                try:
                    width = len(value)
                except TypeError:
                    continue
                if width > 0:
                    draft_widths.append(width)
        draft_widths.sort()
        return {
            "mode": (
                "decode"
                if total is not None and 1 <= total <= 16
                else "prefill_or_other"
                if total is not None
                else "unknown"
            ),
            "scheduled_tokens": total,
            "request_count": len(counts),
            "tokens_per_request": counts,
            "draft_widths": draft_widths,
        }

    def _attach_decode_sync_key(self, scheduler_output, metadata):
        """Bind the fence to this request, not to its reusable chat bank.

        A bank survives multiple Pi turns. ``request_id`` does not, so it is
        the right content-free identity for a fresh prefill/cache-restore to
        decode transition. The phase epoch also changes if one request enters
        another restore/prefill cycle. The qualified scheduler dispatches one
        request at a time; retaining all scheduled IDs keeps this safe if that
        contract is widened later. The digest avoids carrying a raw client
        identifier into worker metadata.
        """
        request_ids = self._scheduled_request_ids(scheduler_output)
        if not request_ids:
            request_ids = tuple(
                str(request.request_id)
                for request in self.running
                if getattr(request, "request_id", None) is not None
            )
        response_request = getattr(self, "response_request", None)
        if not request_ids and response_request is not None:
            request_id = getattr(response_request, "request_id", None)
            if request_id is not None:
                request_ids = (str(request_id),)
        bank = metadata.get("bank")
        if request_ids and bank is not None:
            epochs = getattr(self, "_decode_sync_epochs", {})
            transition = "|".join(
                f"{request_id}:{epochs.get(request_id, 0)}" for request_id in request_ids
            )
            seed = bank + "|" + transition
            metadata["decode_sync_key"] = hashlib.sha256(seed.encode()).hexdigest()
        else:
            # Preserve the old bounded fallback for runtimes that do not
            # expose num_scheduled_tokens to this overlay.
            metadata["decode_sync_key"] = bank

    def _worker_metadata(self, save_blocks, *, barrier=False, discard_active=False):
        value = {
            "bank": self.banks.active,
            "save_blocks": save_blocks,
            "drop_banks": self.drop_banks,
            "barrier": barrier,
            "discard_active": discard_active,
            "decode_sync_key": None,
            "max_banks": self.max_banks,
            "status_path": self.status_path,
        }
        self.drop_banks = []
        return value

    def get_num_unfinished_requests(self):
        value = super().get_num_unfinished_requests()
        pause_state = getattr(self, "_pause_state", None)
        if pause_state is not None and pause_state.name == "PAUSED_ALL":
            return value
        return value + len(getattr(self, "parked", {}))

    def update_from_output(self, scheduler_output, model_runner_output):
        before = len(self.requests)
        tracked = list(self.requests.values()) if hasattr(self, "request_phases") else []
        result = super().update_from_output(scheduler_output, model_runner_output)
        # vLLM has already resolved speculative acceptance by this point. Read
        # the same scheduler-side counts used by its commit path so the public
        # status feed can show exact cumulative acceptance without touching the
        # GPU or scraping the global Prometheus endpoint. The qualified runtime
        # dispatches one response at a time, but keep the lookup per request so
        # this remains content-free and safe if that contract is widened later.
        scheduled_drafts = getattr(scheduler_output, "scheduled_spec_decode_tokens", {}) or {}
        scheduled_shape = self._scheduled_shape(scheduler_output)
        req_to_index = getattr(model_runner_output, "req_id_to_index", {}) or {}
        sampled = getattr(model_runner_output, "sampled_token_ids", None) or []
        sampled_per_step = max(1, int(getattr(self, "num_sampled_tokens_per_step", 1)))
        observed_at = time.monotonic()
        for request in tracked:
            if request.num_output_tokens:
                self._phase(request, "generate")
                draft = scheduled_drafts.get(request.request_id) or ()
                index = req_to_index.get(request.request_id)
                generated = sampled[index] if index is not None and index < len(sampled) else ()
                accepted = max(len(generated) - sampled_per_step, 0) if draft else 0
                self.request_phases.observe_generation(
                    request,
                    draft_tokens=len(draft),
                    accepted_tokens=accepted,
                    scheduled_shape=scheduled_shape,
                    now=observed_at,
                )
            if request.is_finished():
                self.request_phases.finish(request)
        self.tool_handover.finish()
        # Normal progress is rate-limited to two tmpfs writes per second. A
        # completion forces one final empty status so the UI cannot pin a chat
        # as running after its HTTP response has finished.
        self._publish_status(force=len(self.requests) != before)
        return result

    def response_outcome(self, token, is_tool):
        return self.tool_handover.acknowledge(token, is_tool)

    def answer_priority(self, value):
        result = self.priorities.update(value)
        self._publish_status(force=True)
        return result

    def finish_requests(self, *args, **kwargs):
        result = super().finish_requests(*args, **kwargs)
        self.parked = {k: r for k, r in self.parked.items() if not r.is_finished()}
        if hasattr(self, "request_phases"):
            for request in result or ():
                self.request_phases.finish(request)
                getattr(self, "_decode_sync_epochs", {}).pop(request.request_id, None)
            self._publish_status(force=True)
        return result

    def _publish_status(self, *, force=False):
        now = time.monotonic()
        rows = []
        running = {r.request_id for r in self.running}
        for request in self.requests.values():
            if request.is_finished():
                continue
            key = self._key(request)
            if key is None:
                continue
            state = (
                "running"
                if request.request_id in running
                else "paused"
                if key in self.parked
                else "queued"
            )
            rows.append(
                {
                    "chat_id": key.split(":")[0],
                    "generation": key.split(":")[1],
                    "state": state,
                    "computed_tokens": request.num_computed_tokens,
                    "input_tokens": request.num_prompt_tokens,
                }
            )
        signature = (
            tuple(
                (r.request_id, r.request_id in running)
                for r in self.requests.values()
                if not r.is_finished()
            ),
            self.grace_status and (self.grace_status["chat_id"], self.grace_status["phase"]),
            self.switch_count,
        )
        phase_rows = self._phase_rows()
        phase_signature = self._phase_signature(phase_rows)
        if (
            not force
            and signature == getattr(self, "_last_status_signature", None)
            and now - self.last_status < 0.5
        ):
            # Keep the scheduler/worker status cadence at two writes per
            # second, while letting the small phase feed follow each completed
            # generation round.  This is content-free numeric telemetry and
            # remains on tmpfs; it does not synchronize or inspect the GPU.
            if phase_signature != getattr(self, "_last_phase_signature", None):
                self._write_phase_status(phase_rows, time.time(), phase_signature)
            return
        self.last_status = now
        self._last_status_signature = signature
        published_at = time.time()
        write_status(
            self.status_path + "-scheduler.json",
            {
                "pid": os.getpid(),
                "updated_at": published_at,
                "quantum_seconds": self.quantum,
                "switches": self.switch_count,
                "cached_chats": len(self.banks.managers),
                "max_cached_chats": self.max_banks,
                **({"tool_grace": self.grace_status} if self.grace_status is not None else {}),
                **({"priority_hold": self.priority_hold} if self.priority_hold is not None else {}),
                "requests": rows,
            },
        )
        self._write_phase_status(phase_rows, published_at, phase_signature)

    def _phase_rows(self):
        phases = getattr(self, "request_phases", None)
        if phases is None:
            return []
        return [
            row
            for request in self.requests.values()
            if not request.is_finished()
            and (row := phases.row(request)) is not None
        ]

    @staticmethod
    def _phase_signature(rows):
        """Return only fields whose change requires a fresh Pi redraw."""
        return tuple(
            (
                row["request_id"],
                row["phase"],
                row["blocker"],
                row["last_round_ms"],
                row["last_acceptance_rate"],
                row["generation_rounds"],
                row["draft_tokens"],
                row["accepted_tokens"],
                row["acceptance_rate"],
            )
            for row in rows
        )

    def _write_phase_status(self, rows, published_at, signature):
        if not hasattr(self, "request_phases"):
            return
        events = self.request_phases.take_round_events()
        if events:
            try:
                append_round_log(self.status_path + "-rounds.jsonl", events)
            except OSError as exc:
                # Telemetry must never abort inference. Keep the events queued
                # so a later status publication can retry the same rounds.
                self.request_phases.restore_round_events(events)
                logging.getLogger(__name__).warning(
                    "decode-round telemetry write failed: %s", type(exc).__name__
                )
        write_status(
            self.status_path + "-phases.json",
            {
                "schema": PHASE_SCHEMA,
                "pid": os.getpid(),
                "updated_at": published_at,
                "requests": rows,
                "recent": list(self.request_phases.recent.values()),
            },
        )
        self._last_phase_signature = signature


class WorkerBanks:
    def __init__(self, runner, metadata):
        import torch

        self.torch = torch
        self.runner = runner
        self.active = None
        self.images = {}
        self.free_buffers = []
        self.stage = None
        self.transferred_bytes = 0
        self.transfer_seconds = 0.0
        self.allocation_events = 0
        self.allocation_seconds = 0.0
        self.generation_replacements = 0
        self.switches = 0
        self.status_path = metadata["status_path"]
        tensors = []
        for entry in runner.kv_caches:
            tensors.extend(entry if isinstance(entry, list) else [entry])
        storages = {
            value.untyped_storage().data_ptr(): value.untyped_storage()
            for value in tensors
            if isinstance(value, torch.Tensor)
        }
        num_blocks = runner.kv_cache_config.num_blocks
        if not storages or num_blocks <= 0:
            raise ValueError("RAM time sharing requires block-addressable KV storage")
        self.regions = []
        offset = 0
        for storage in storages.values():
            gpu = torch.empty(0, dtype=torch.uint8, device=runner.device).set_(storage)
            if gpu.numel() % num_blocks:
                raise ValueError("KV storage is not an integer number of physical blocks")
            stride = gpu.numel() // num_blocks
            if stride <= 0:
                raise ValueError("KV storage has an empty physical block")
            self.regions.append((gpu, offset, stride))
            offset += gpu.numel()
        self.capacity = offset
        self.max_banks = metadata["max_banks"]
        self.stage_capacity = min(
            self.capacity,
            max(256 * 1024**2, max(stride for _, _, stride in self.regions)),
        )
        required = self.capacity * (self.max_banks - 1) + self.stage_capacity
        self.allocated_bytes = 0
        self.reserved_capacity_bytes = required

    def _ensure_buffers(self):
        if self.stage is not None:
            return 0, 0.0
        # Defer the large pinned allocation until a second chat actually needs
        # the GPU. A single-chat workload pays no allocation or page-registration
        # cost merely because fair scheduling is enabled.
        # Check current headroom here too: unused RAM capacity must not reject
        # the first chat, and memory availability can change since initialization.
        memory = {
            line.split(":", 1)[0]: int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        }
        available = memory.get("MemAvailable", 0)
        if available < self.reserved_capacity_bytes + 8 * 1024**3:
            raise MemoryError(
                "insufficient RAM for pinned chat caches and 8 GiB system headroom "
                f"(available={available} bytes, allocation={self.reserved_capacity_bytes} bytes)"
            )
        started = time.monotonic()
        buffers = [
            self.torch.empty(self.capacity, dtype=self.torch.uint8, device="cpu", pin_memory=True)
            for _ in range(self.max_banks - 1)
        ]
        stage = self.torch.empty(
            self.stage_capacity,
            dtype=self.torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        # Publish the arena only after every allocation succeeds. A failed stage
        # allocation must not leave partial buffers that a retry duplicates.
        self.free_buffers = buffers
        self.stage = stage
        self.allocated_bytes = self.reserved_capacity_bytes
        elapsed = time.monotonic() - started
        self.allocation_events += 1
        self.allocation_seconds += elapsed
        return self.reserved_capacity_bytes, elapsed

    def _copy_to_buffer(self, destination, ranges):
        amount = 0
        for gpu, offset, stride in self.regions:
            for begin, end in ranges:
                start, stop = begin * stride, end * stride
                if not 0 <= start <= stop <= gpu.numel():
                    raise ValueError("RAM cache transfer exceeds a KV storage region")
                destination[offset + start : offset + stop].copy_(
                    gpu[start:stop], non_blocking=True
                )
                amount += stop - start
        return amount

    def _copy_from_buffer(self, source, ranges):
        amount = 0
        for gpu, offset, stride in self.regions:
            for begin, end in ranges:
                start, stop = begin * stride, end * stride
                if not 0 <= start <= stop <= gpu.numel():
                    raise ValueError("RAM cache transfer exceeds a KV storage region")
                gpu[start:stop].copy_(source[offset + start : offset + stop], non_blocking=True)
                amount += stop - start
        return amount

    def _range_bytes(self, ranges):
        return sum((end - begin) * stride for _, _, stride in self.regions for begin, end in ranges)

    def _publish_status(
        self,
        *,
        last_transfer_bytes=0,
        last_transfer_seconds=0.0,
        last_allocation_bytes=0,
        last_allocation_seconds=0.0,
        handover="status",
    ):
        images = [
            {
                **bank_identity(key),
                "data_bytes": self._range_bytes(image["ranges"]),
                "allocated_bytes": self.capacity,
            }
            for key, image in sorted(self.images.items())
        ]
        write_status(
            self.status_path + "-worker.json",
            {
                "pid": os.getpid(),
                "updated_at": time.time(),
                "allocated_bytes": self.allocated_bytes,
                "reserved_capacity_bytes": self.reserved_capacity_bytes,
                "cached_chats": len(self.images) + int(self.active is not None),
                "switches": self.switches,
                "last_transfer_bytes": last_transfer_bytes,
                "last_transfer_seconds": last_transfer_seconds,
                "transferred_bytes": self.transferred_bytes,
                "transfer_seconds": self.transfer_seconds,
                "last_allocation_bytes": last_allocation_bytes,
                "last_allocation_seconds": last_allocation_seconds,
                "allocation_events": self.allocation_events,
                "allocation_seconds": self.allocation_seconds,
                "generation_replacements": self.generation_replacements,
                "last_handover": handover,
                "residency": {
                    "active": bank_identity(self.active) if self.active is not None else None,
                    "images": images,
                    "free_buffer_bytes": len(self.free_buffers) * self.capacity,
                    "staging_buffer_bytes": self.stage_capacity if self.stage is not None else 0,
                },
            },
        )

    @staticmethod
    def _intersections(ranges, begin, end):
        return [
            (max(start, begin), min(stop, end))
            for start, stop in ranges
            if start < end and stop > begin
        ]

    def _swap(self, buffer, outgoing, incoming):
        """Swap the active image through one inactive buffer and a small stage."""
        amount = 0
        union = spans(block for begin, end in (*outgoing, *incoming) for block in range(begin, end))
        for gpu, offset, stride in self.regions:
            stage_blocks = max(1, self.stage_capacity // stride)
            for first, last in union:
                for begin in range(first, last, stage_blocks):
                    end = min(last, begin + stage_blocks)
                    start_byte, stop_byte = begin * stride, end * stride
                    length = stop_byte - start_byte
                    # Preserve outgoing GPU bytes before the incoming H2D copy
                    # can overwrite them. Each region uses its own block stride.
                    self.stage[:length].copy_(gpu[start_byte:stop_byte], non_blocking=True)
                    self.torch.cuda.synchronize()
                    for part_begin, part_end in self._intersections(incoming, begin, end):
                        part_start, part_stop = part_begin * stride, part_end * stride
                        gpu[part_start:part_stop].copy_(
                            buffer[offset + part_start : offset + part_stop],
                            non_blocking=True,
                        )
                        amount += part_stop - part_start
                    self.torch.cuda.synchronize()
                    buffer[offset + start_byte : offset + stop_byte].copy_(self.stage[:length])
                    amount += length
        return amount

    def before(self, metadata):
        target = metadata["bank"]
        if target is None:
            return
        changed = target != self.active
        if changed or metadata["barrier"]:
            # Includes asynchronous recurrent-state/count copies and offloader
            # DMA from the preceding step. No GPU write crosses the handover.
            self.torch.cuda.synchronize()
        dropped = False
        for key in metadata["drop_banks"]:
            old = self.images.pop(key, None)
            if old is not None:
                self.free_buffers.append(old["buffer"])
                dropped = True
        if not changed:
            if dropped:
                self._publish_status(handover="drop")
            return
        if self.active is None:
            # Fresh blocks are zeroed by the normal scheduler output. Keep the
            # first-chat path allocation-free and leave the GPU cache in place.
            self.active = target
            self._publish_status(handover="activate")
            return
        if metadata.get("discard_active", False):
            if self.active not in metadata["drop_banks"]:
                raise ValueError("discarded GPU bank is missing from the retirement set")
            if target in self.images:
                raise ValueError("successor generation unexpectedly has a RAM image")
            self.active = target
            self.switches += 1
            self.generation_replacements += 1
            self._publish_status(handover="generation-replace")
            return
        if self.stage is None:
            worker_phase("ram_allocation")
        allocation_bytes, allocation_seconds = self._ensure_buffers()
        worker_phase("handover")
        started = time.monotonic()
        amount = 0
        outgoing = spans(metadata["save_blocks"])
        incoming = self.images.pop(target, None)
        if incoming is None:
            if not self.free_buffers:
                raise MemoryError("RAM chat-cache capacity exhausted")
            buffer = self.free_buffers.pop()
            amount += self._copy_to_buffer(buffer, outgoing)
            self.torch.cuda.synchronize()
        else:
            buffer = incoming["buffer"]
            amount += self._swap(buffer, outgoing, incoming["ranges"])
        if self.active not in metadata["drop_banks"]:
            self.images[self.active] = {"buffer": buffer, "ranges": outgoing}
        else:
            self.free_buffers.append(buffer)
        self.active = target
        self.switches += 1
        elapsed = time.monotonic() - started
        self.transferred_bytes += amount
        self.transfer_seconds += elapsed
        self._publish_status(
            last_transfer_bytes=amount,
            last_transfer_seconds=elapsed,
            last_allocation_bytes=allocation_bytes,
            last_allocation_seconds=allocation_seconds,
            handover="swap",
        )


def before_forward(runner, scheduler_output):
    metadata = scheduler_output.qwen_fair
    if metadata is None:
        return
    if not hasattr(runner, "qwen_banks"):
        runner.qwen_banks = WorkerBanks(runner, metadata)
    changed = metadata["bank"] != runner.qwen_banks.active
    if changed or metadata["barrier"]:
        worker_phase("handover")
    runner.qwen_banks.before(metadata)
    _sync_decode_transition(runner, scheduler_output, metadata)
    scheduler = _phase_scheduler() if _phase_scheduler is not None else None
    if scheduler is not None and not metadata["barrier"]:
        scheduler._refresh_request_phases()

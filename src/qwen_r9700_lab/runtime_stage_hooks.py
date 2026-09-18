# QWEN_ASSURANCE_ONLY_BEGIN: runtime-stage-hooks
"""Opt-in GPU-event accounting for the pinned vLLM runtime.

The kernel adapters already report their own intervals.  This module adds the
missing outer boundaries around a vLLM worker forward, draft proposal, and
sampler call.  It is deliberately opt-in: importing it does nothing unless
``QWEN_STAGE_TIMING=1`` or ``QWEN_OUTER_STAGE_TIMING=1`` is set, and the
wrappers are installed lazily after the vLLM worker classes have been imported.

``QWEN_STAGE_TIMING=1`` preserves the original component-heavy diagnostic.
``QWEN_OUTER_STAGE_TIMING=1`` installs only the six low-observer-overhead M8
boundaries used by the current decode decomposition.  The latter reuses a
fixed ring of HIP events instead of constructing event objects in every call.

The hook is diagnostic only.  It never changes tensors or scheduling decisions
and it is safe to leave installed for a short warm profiling run.  The process
exit line is consumed by ``scripts/summarize_stage_timing.py``.
"""

from __future__ import annotations

import atexit
import functools
import os
import signal
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

_LEGACY_ENABLED = os.environ.get("QWEN_STAGE_TIMING", "0") == "1"
_OUTER_ONLY = os.environ.get("QWEN_OUTER_STAGE_TIMING", "0") == "1"
_ENABLED = _LEGACY_ENABLED or _OUTER_ONLY
_MODE_CONFLICT = _LEGACY_ENABLED and _OUTER_ONLY
_MODEL_HOOKS_INSTALLED = False
_COMPONENT_HOOKS_INSTALLED = False
_DRAFT_HOOK_INSTALLED = False
_SCHEDULER_HOOK_INSTALLED = False
_SAMPLE_CORE_HOOK_INSTALLED = False
_TARGET_LOGITS_HOOK_INSTALLED = False
_REJECTION_HOOK_INSTALLED = False
_ROUND_HOOK_INSTALLED = False
_STATE_COMMIT_HOOK_INSTALLED = False
_PENDING_LOGGED = False
_READY_LOGGED = False
_LAST_CRITICAL_MISSING: tuple[str, ...] | None = None
_LOCK = threading.Lock()
_TOTALS: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])


_PENDING_EVENTS: list[tuple[str, Any, Any]] = []
_EVENT_RING_SIZE = 1_024
_EVENT_PAIRS_PER_ROUND_RESERVE = 16
_EVENT_RING: list[tuple[Any, Any]] | None = None
_EVENT_RING_CURSOR = 0
_EVENT_RING_CAPPED_LOGGED = False
_OUTER_ROUND_ACTIVE = False
_OUTER_COMMIT_PENDING = False
_OUTER_COMMIT_LABEL = "scheduler.commit"
_OUTER_REJECTION_SEEN = False
_OUTER_CURRENT_EVENTS: list[tuple[str, Any, Any]] = []


def _prepare_event_ring() -> None:
    global _EVENT_RING
    if not _ENABLED or not _OUTER_ONLY or _EVENT_RING is not None:
        return
    import torch

    _EVENT_RING = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(_EVENT_RING_SIZE)
    ]


def _event_pair():
    if not _ENABLED:
        return None
    import torch

    stream = torch.cuda.current_stream()
    if _OUTER_ONLY:
        global _EVENT_RING_CURSOR
        _prepare_event_ring()
        assert _EVENT_RING is not None
        if len(_EVENT_RING) <= _EVENT_RING_CURSOR:
            raise RuntimeError(
                "outer stage-timing event ring exhausted; refuse to reuse pending events"
            )
        start, end = _EVENT_RING[_EVENT_RING_CURSOR]
        _EVENT_RING_CURSOR += 1
    else:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    return start, end


def _finish(label: str, interval) -> None:
    if interval is None:
        return
    import torch

    start, end = interval
    stream = torch.cuda.current_stream()
    end.record(stream)
    with _LOCK:
        if _OUTER_ONLY and _OUTER_ROUND_ACTIVE:
            _OUTER_CURRENT_EVENTS.append((label, start, end))
        else:
            _PENDING_EVENTS.append((label, start, end))


def _publish_outer_round(*, prefix: str = "") -> None:
    """Publish one closed outer round, classifying non-verification calls."""
    with _LOCK:
        _PENDING_EVENTS.extend(
            (prefix + label, start, end) for label, start, end in _OUTER_CURRENT_EVENTS
        )
        _OUTER_CURRENT_EVENTS.clear()


def _shape_label(scheduler_output: Any) -> str:
    """Classify a worker call without assuming a particular vLLM dataclass."""
    value = _scheduled_tokens(scheduler_output)
    return f"M={value}" if value is not None else "M=unknown"


def _scheduled_tokens(scheduler_output: Any) -> int | None:
    """Return the flattened scheduled-row count when the ABI exposes it."""
    for name in ("total_num_scheduled_tokens", "num_scheduled_tokens"):
        value = getattr(scheduler_output, name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return None


def _wrap_method(owner: type, name: str, label: Callable[..., str]) -> bool:
    original = getattr(owner, name, None)
    if original is None:
        return False
    if getattr(original, "_qwen_stage_hook", False):
        return True

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        global _OUTER_REJECTION_SEEN
        if _OUTER_ONLY and not _OUTER_ROUND_ACTIVE:
            return original(self, *args, **kwargs)
        interval = _event_pair()
        try:
            return original(self, *args, **kwargs)
        finally:
            try:
                stage = label(self, args, kwargs)
            except Exception:
                stage = name
            _finish(stage, interval)
            if _OUTER_ONLY and stage == "rejection.verify":
                _OUTER_REJECTION_SEEN = True

    wrapped._qwen_stage_hook = True  # type: ignore[attr-defined]
    setattr(owner, name, wrapped)
    return True


def _wrap_host_method(owner: type, name: str, label: str) -> bool:
    """Wrap a CPU scheduler boundary for commit/queue accounting.

    Scheduler state mutation is host work rather than a CUDA interval.  Keep it
    separate from ``_wrap_method`` so GPU-event totals are never mislabeled as a
    device measurement, while still reporting the commit cost in the same
    process-exit summary.
    """
    original = getattr(owner, name, None)
    if original is None:
        return False
    if getattr(original, "_qwen_stage_hook", False):
        return True

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        global _OUTER_COMMIT_LABEL, _OUTER_COMMIT_PENDING
        if _OUTER_ONLY and label == "scheduler.commit" and not _OUTER_COMMIT_PENDING:
            return original(self, *args, **kwargs)
        observed_label = _OUTER_COMMIT_LABEL if _OUTER_ONLY else label
        started = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            elapsed = (time.perf_counter() - started) * 1_000.0
            total, count = _TOTALS[observed_label]
            _TOTALS[observed_label] = [total + elapsed, count + 1.0]
            if not _OUTER_ONLY:
                print(
                    f"[qwen-runtime] stage timing: {label} ms={elapsed:.3f}",
                    flush=True,
                )
            elif label == "scheduler.commit":
                _OUTER_COMMIT_PENDING = False

    wrapped._qwen_stage_hook = True  # type: ignore[attr-defined]
    setattr(owner, name, wrapped)
    return True


def _record_host_total(label: str, elapsed_ms: float) -> None:
    with _LOCK:
        total, count = _TOTALS[label]
        _TOTALS[label] = [total + elapsed_ms, count + 1.0]


def _wrap_outer_target_and_round(owner: type) -> bool:
    """Open one whole-round span and time the outer target forward."""

    original = getattr(owner, "execute_model", None)
    if original is None:
        return False
    if getattr(original, "_qwen_outer_stage_hook", False):
        return True

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        global _EVENT_RING_CAPPED_LOGGED, _OUTER_REJECTION_SEEN, _OUTER_ROUND_ACTIVE
        scheduler_output = args[0] if args else kwargs.get("scheduler_output")
        scheduled_tokens = _scheduled_tokens(scheduler_output)
        # vLLM startup and prompt-prefill forwards do not call sample_tokens.
        # Only a decode-sized target batch constitutes the round measured here.
        if scheduled_tokens is None or not 1 <= scheduled_tokens <= 16:
            return original(self, *args, **kwargs)
        # Keep observer state strictly bounded during long qualification
        # responses. Stop measuring only at a round boundary; serving itself
        # must continue after the fixed event pool has enough complete rounds.
        if _EVENT_RING_CURSOR + _EVENT_PAIRS_PER_ROUND_RESERVE > _EVENT_RING_SIZE:
            if not _EVENT_RING_CAPPED_LOGGED:
                print(
                    "[qwen-runtime] outer GPU-event measurement capped at complete rounds "
                    f"ring_pairs={_EVENT_RING_SIZE}",
                    flush=True,
                )
                _EVENT_RING_CAPPED_LOGGED = True
            return original(self, *args, **kwargs)
        previous = getattr(self, "_qwen_outer_round_interval", None)
        if previous is not None or _OUTER_ROUND_ACTIVE:
            raise RuntimeError("outer stage-timing round was not closed by sample_tokens")
        if _OUTER_CURRENT_EVENTS:
            raise RuntimeError("outer stage-timing found unpublished events before a new round")
        round_interval = _event_pair()
        target_interval = _event_pair()
        self._qwen_outer_round_interval = round_interval
        self._qwen_outer_round_started = time.perf_counter()
        _OUTER_ROUND_ACTIVE = True
        _OUTER_REJECTION_SEEN = False
        failed = True
        try:
            result = original(self, *args, **kwargs)
            failed = False
            return result
        finally:
            _finish("target.forward " + _shape_label(scheduler_output), target_interval)
            if failed:
                _finish("round.gpu", round_interval)
                _publish_outer_round(prefix="failed.")
                self._qwen_outer_round_interval = None
                self._qwen_outer_round_started = None
                _OUTER_ROUND_ACTIVE = False

    wrapped._qwen_stage_hook = True  # type: ignore[attr-defined]
    wrapped._qwen_outer_stage_hook = True  # type: ignore[attr-defined]
    owner.execute_model = wrapped
    return True


def _wrap_outer_round_close(owner: type) -> bool:
    """Close the whole-round span after sample/commit/draft proposal."""

    original = getattr(owner, "sample_tokens", None)
    if original is None:
        return False
    if getattr(original, "_qwen_outer_stage_hook", False):
        return True

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        global _OUTER_COMMIT_LABEL, _OUTER_COMMIT_PENDING
        global _OUTER_REJECTION_SEEN, _OUTER_ROUND_ACTIVE
        try:
            return original(self, *args, **kwargs)
        finally:
            interval = getattr(self, "_qwen_outer_round_interval", None)
            started = getattr(self, "_qwen_outer_round_started", None)
            self._qwen_outer_round_interval = None
            self._qwen_outer_round_started = None
            if interval is not None:
                _finish("round.gpu", interval)
                verified = _OUTER_REJECTION_SEEN
                prefix = "" if verified else "nonverification."
                _publish_outer_round(prefix=prefix)
                _OUTER_COMMIT_PENDING = True
                _OUTER_COMMIT_LABEL = prefix + "scheduler.commit"
                wall_label = prefix + "round.dispatch-wall"
            else:
                wall_label = "nonverification.round.dispatch-wall"
            _OUTER_ROUND_ACTIVE = False
            _OUTER_REJECTION_SEEN = False
            if started is not None:
                _record_host_total(
                    wall_label,
                    (time.perf_counter() - float(started)) * 1_000.0,
                )

    wrapped._qwen_stage_hook = True  # type: ignore[attr-defined]
    wrapped._qwen_outer_stage_hook = True  # type: ignore[attr-defined]
    owner.sample_tokens = wrapped
    return True


def install_runtime_stage_hooks() -> bool:
    """Install worker-level event hooks when vLLM classes are available.

    vLLM imports the worker classes lazily and some DFlash startup paths load
    the Quest/GDN adapters before ``GPUModelRunner`` is importable.  A failed
    early attempt must therefore remain retryable; otherwise the process can
    run an entire qualification request with only the scheduler hook (or no
    outer GPU totals at all).
    """
    global _MODEL_HOOKS_INSTALLED, _COMPONENT_HOOKS_INSTALLED
    global _DRAFT_HOOK_INSTALLED, _SCHEDULER_HOOK_INSTALLED
    global _SAMPLE_CORE_HOOK_INSTALLED, _TARGET_LOGITS_HOOK_INSTALLED
    global _REJECTION_HOOK_INSTALLED, _ROUND_HOOK_INSTALLED
    global _STATE_COMMIT_HOOK_INSTALLED
    global _PENDING_LOGGED, _READY_LOGGED, _LAST_CRITICAL_MISSING
    if not _ENABLED:
        return False
    if _MODE_CONFLICT:
        raise RuntimeError("QWEN_STAGE_TIMING and QWEN_OUTER_STAGE_TIMING are mutually exclusive")
    with _LOCK:
        runner_classes: list[type] = []
        import_errors: list[str] = []
        for module_name, class_name in (
            ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner"),
            # DFlash and the current gfx1201 qualification lane use V2.
            ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner"),
        ):
            try:
                module = __import__(module_name, fromlist=[class_name])
                owner = getattr(module, class_name)
            except Exception as exc:  # pragma: no cover - runtime-specific
                import_errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
            else:
                if owner not in runner_classes:
                    runner_classes.append(owner)

        target_ready = False
        sampler_ready = False
        sample_core_ready = False
        round_ready = False
        state_commit_ready = False
        draft_ready = False
        for owner in runner_classes:
            if _OUTER_ONLY:
                target_ready |= _wrap_outer_target_and_round(owner)
                round_ready |= _wrap_outer_round_close(owner)
                state_commit_ready |= _wrap_method(
                    owner, "postprocess_sampled", lambda *_: "state.commit"
                )
                sampler_ready = round_ready
                sample_core_ready = True
            else:
                target_ready |= _wrap_method(
                    owner,
                    "execute_model",
                    lambda self, args, kwargs: (
                        "target.execute_model "
                        + _shape_label(args[0] if args else kwargs.get("scheduler_output"))
                    ),
                )
                sampler_ready |= _wrap_method(owner, "sample_tokens", lambda *_: "sampler")
                # ``sample_tokens`` also performs post-processing and the next draft
                # proposal. Keep the core target-logits + verification interval
                # separate so the outer sampler total is not misattributed to the
                # rejection sampler.
                sample_core_ready |= _wrap_method(owner, "sample", lambda *_: "sample.core")
            draft_ready |= _wrap_method(
                owner, "propose_draft_token_ids", lambda *_: "draft.propose"
            )

        # Account for the model components which are not represented by the
        # kernel-adapter totals.  In particular, Qwen3-Next routes a number of
        # layers through MoE expert GEMMs; those calls are the most likely
        # source of the remaining M=16 target latency after Native-B has taken
        # every dense W4A16 projection.  These wrappers are diagnostic only and
        # are installed on the concrete module classes after lazy model import.
        component_ready = False
        target_logits_ready = False
        for mod_name in (
            "vllm.model_executor.models.qwen3_5",
            "vllm.model_executor.models.qwen3_next",
        ):
            try:
                module = __import__(
                    mod_name,
                    fromlist=[
                        "Qwen3NextSparseMoeBlock",
                        "Qwen3NextMLP",
                        "Qwen3NextAttention",
                        "Qwen3_5DecoderLayer",
                        "Qwen3_5Model",
                    ],
                )
                if not _OUTER_ONLY and hasattr(module, "Qwen3NextSparseMoeBlock"):
                    component_ready |= _wrap_method(
                        module.Qwen3NextSparseMoeBlock,
                        "forward",
                        lambda *_: "target.moe",
                    )
                if not _OUTER_ONLY and hasattr(module, "Qwen3NextMLP"):
                    component_ready |= _wrap_method(
                        module.Qwen3NextMLP,
                        "forward",
                        lambda *_: "target.dense-mlp",
                    )
                if not _OUTER_ONLY and hasattr(module, "Qwen3NextAttention"):
                    component_ready |= _wrap_method(
                        module.Qwen3NextAttention,
                        "forward",
                        lambda *_: "target.full-attention-layer",
                    )
                if not _OUTER_ONLY and hasattr(module, "Qwen3_5DecoderLayer"):
                    component_ready |= _wrap_method(
                        module.Qwen3_5DecoderLayer,
                        "forward",
                        lambda *_: "target.decoder-layer",
                    )
                for causal_lm_name in (
                    "Qwen3_5ForCausalLMBase",
                    "Qwen3NextForCausalLM",
                ):
                    if hasattr(module, causal_lm_name):
                        target_logits_ready |= _wrap_method(
                            getattr(module, causal_lm_name),
                            "compute_logits",
                            lambda *_: "target.logits",
                        )
            except Exception as exc:  # pragma: no cover - runtime-specific
                import_errors.append(f"{mod_name} components: {type(exc).__name__}: {exc}")

        # V2 has a separate rejection sampler from the legacy
        # ``vllm.v1.sample`` implementation. Time the class actually called by
        # ``vllm.v1.worker.gpu.model_runner`` so a legacy microbenchmark cannot
        # be mistaken for evidence about the live verifier.
        rejection_ready = False
        try:
            module = __import__(
                "vllm.v1.worker.gpu.spec_decode.rejection_sampler",
                fromlist=["RejectionSampler"],
            )
            rejection_ready |= _wrap_method(
                module.RejectionSampler,
                "__call__",
                lambda *_: "rejection.verify",
            )
        except Exception as exc:  # pragma: no cover - runtime-specific
            import_errors.append(f"V2 rejection sampler: {type(exc).__name__}: {exc}")

        # The GDN class lives in the pinned overlay rather than in the model
        # module.  Its fused kernel reports its own narrower interval; this
        # outer interval captures projections, state updates, and epilogue too.
        if not _OUTER_ONLY:
            try:
                module = __import__(
                    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
                    fromlist=["QwenGatedDeltaNetAttention"],
                )
                component_ready |= _wrap_method(
                    module.QwenGatedDeltaNetAttention,
                    "forward",
                    lambda *_: "target.gdn-layer",
                )
            except Exception as exc:  # pragma: no cover - runtime-specific
                import_errors.append(f"qwen_gdn component: {type(exc).__name__}: {exc}")

        _COMPONENT_HOOKS_INSTALLED = component_ready
        _SAMPLE_CORE_HOOK_INSTALLED = sample_core_ready
        _TARGET_LOGITS_HOOK_INSTALLED = target_logits_ready
        _REJECTION_HOOK_INSTALLED = rejection_ready
        _ROUND_HOOK_INSTALLED = round_ready
        _STATE_COMMIT_HOOK_INSTALLED = state_commit_ready

        # V2 performs drafting inside a speculator rather than exposing the V1
        # propose_draft_token_ids method.  Wrap the concrete classes when they
        # are present; failed imports remain retryable during lazy startup.
        for module_name, class_name in (
            (
                "vllm.v1.worker.gpu.spec_decode.dflash.speculator",
                "DFlashSpeculator",
            ),
            (
                "vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator",
                "MultiModuleMTPSpeculator",
            ),
        ):
            try:
                module = __import__(module_name, fromlist=[class_name])
                owner = getattr(module, class_name)
            except Exception as exc:  # pragma: no cover - runtime-specific
                import_errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
            else:
                draft_ready |= _wrap_method(owner, "propose", lambda *_: "draft.propose")

        _MODEL_HOOKS_INSTALLED = target_ready and sampler_ready
        _DRAFT_HOOK_INSTALLED = draft_ready
        if not runner_classes and not _PENDING_LOGGED:
            detail = "; ".join(import_errors[:2]) or "runner classes unavailable"
            print(f"[qwen-runtime] stage hooks pending: {detail}", flush=True)
            _PENDING_LOGGED = True

        try:
            from vllm.v1.core.sched.scheduler import Scheduler

            _SCHEDULER_HOOK_INSTALLED = _wrap_host_method(
                Scheduler, "update_from_output", "scheduler.commit"
            )
        except Exception as exc:  # pragma: no cover - runtime-specific
            if not _PENDING_LOGGED:
                print(
                    f"[qwen-runtime] scheduler hook pending: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                _PENDING_LOGGED = True

        if _OUTER_ONLY:
            critical = {
                "round.gpu": _ROUND_HOOK_INSTALLED,
                "draft.propose": _DRAFT_HOOK_INSTALLED,
                "target.logits": _TARGET_LOGITS_HOOK_INSTALLED,
                "rejection.verify": _REJECTION_HOOK_INSTALLED,
                "state.commit": _STATE_COMMIT_HOOK_INSTALLED,
            }
        else:
            critical = {
                "sample.core": _SAMPLE_CORE_HOOK_INSTALLED,
                "target.logits": _TARGET_LOGITS_HOOK_INSTALLED,
                "rejection.verify": _REJECTION_HOOK_INSTALLED,
            }
        critical_missing = tuple(label for label, installed in critical.items() if not installed)
        if critical_missing != _LAST_CRITICAL_MISSING:
            if critical_missing:
                print(
                    "[qwen-runtime] critical stage hooks pending: " + ", ".join(critical_missing),
                    flush=True,
                )
            _LAST_CRITICAL_MISSING = critical_missing
        ready = _MODEL_HOOKS_INSTALLED and _SCHEDULER_HOOK_INSTALLED and not critical_missing
        if ready and not _READY_LOGGED:
            if _OUTER_ONLY:
                _prepare_event_ring()
                print(
                    "[qwen-runtime] outer GPU-event stage hooks installed "
                    f"ring_pairs={_EVENT_RING_SIZE}",
                    flush=True,
                )
            else:
                print("[qwen-runtime] GPU-event stage hooks installed", flush=True)
            _READY_LOGGED = True
        elif _MODEL_HOOKS_INSTALLED and not _SCHEDULER_HOOK_INSTALLED and not _PENDING_LOGGED:
            print(
                "[qwen-runtime] worker GPU-event hooks installed; scheduler hook pending",
                flush=True,
            )
            _PENDING_LOGGED = True
        return ready


def _flush_events() -> None:
    global _EVENT_RING_CURSOR
    with _LOCK:
        if not _PENDING_EVENTS:
            return
        pending = list(_PENDING_EVENTS)
        _PENDING_EVENTS.clear()

    import torch

    torch.cuda.synchronize()
    for label, start, end in pending:
        try:
            elapsed = float(start.elapsed_time(end))
            total, count = _TOTALS[label]
            _TOTALS[label] = [total + elapsed, count + 1.0]
        except Exception:
            pass
    if _OUTER_ONLY:
        _EVENT_RING_CURSOR = 0


def _dump_totals() -> None:
    _flush_events()
    if not _TOTALS:
        return
    fields = []
    for label, (total, count) in sorted(_TOTALS.items()):
        fields.append(f"{label}:count={int(count)} total_ms={total:.3f}")
    print("[qwen-runtime] stage totals: " + " | ".join(fields), flush=True)


atexit.register(_dump_totals)


def _signal_handler(signum, frame):
    _dump_totals()
    try:
        from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import (
            _dump_stage_totals as d1,
        )

        d1()
    except Exception:
        pass
    try:
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            _dump_gdn_stage_totals as d2,
        )

        d2()
    except Exception:
        pass
    try:
        from vllm.v1.attention.backends.quest_vllm_attention import _dump_quest_stage_totals as d3

        d3()
    except Exception:
        pass
    sys.exit(0)


try:
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
except Exception:
    pass


__all__ = ["install_runtime_stage_hooks"]
# QWEN_ASSURANCE_ONLY_END: runtime-stage-hooks

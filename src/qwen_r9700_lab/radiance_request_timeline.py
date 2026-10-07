"""Content-free request boundaries, including the work before engine admission.

The ASGI observer does not decode a body or inspect a streamed response. API
hooks retain only opaque identifier hashes and the presence of output fields.
All timings are host monotonic clocks; no GPU wait is added.
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import os
import re
import stat
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

try:
    import qwen_radiance_cache_telemetry as telemetry
except ModuleNotFoundError as error:
    if error.name != "qwen_radiance_cache_telemetry":
        raise
    from qwen_r9700_lab import radiance_cache_telemetry as telemetry


_request = contextvars.ContextVar("qwen_request_timeline", default=None)
_api_recorder_pid = None
_worker_recorder_pid = None


def _failed_diagnostic():
    recorder = getattr(telemetry, "_recorder", None)
    if recorder is not None:
        recorder.dropped += 1


def _hash(value):
    try:
        return (
            hashlib.sha256(value.encode()).hexdigest()
            if isinstance(value, str)
            else None
        )
    except UnicodeError:
        return None


def _context():
    state = _request.get()
    return dict(state["identities"]) if state is not None else {}


def _emit(stage, started=None, *, point_ns=None, **values):
    """Diagnostics must not affect cancellation, admission or output delivery."""
    try:
        now = time.monotonic_ns() if point_ns is None else point_ns
        telemetry.emit_at(
            stage, now if started is None else started, now, **_context(), **values
        )
    except Exception:  # noqa: BLE001 - optional diagnostics are not serving logic
        _failed_diagnostic()


def _api_recorder():
    global _api_recorder_pid
    if _api_recorder_pid == os.getpid():
        return
    # The API and EngineCore are separate processes. Never rotate or replace
    # the scheduler's recorder or health files from an API writer.
    base = os.environ.get("QWEN_REQUEST_TIMELINE_STATUS", telemetry.DEFAULT_STATUS)
    try:
        _retire_api_recorders(base)
    except OSError:
        pass
    recorder = telemetry.configure(f"{base}-api-{os.getpid()}")
    _api_recorder_pid = os.getpid()
    if recorder is not None:
        recorder.source_hashes["request_timeline"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        _emit("api_recorder_started")


def _retire_api_recorders(base):
    """Keep live API writers and two previous dead-PID diagnostic sets."""
    prefix = Path(base)
    pattern = re.compile(
        re.escape(prefix.name)
        + r"-api-([0-9]+)-cache-jobs(?:\.jsonl(?:\.1)?|-health\.json)\Z"
    )
    dead = {}
    for path in prefix.parent.glob(prefix.name + "-api-*-cache-jobs*"):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            continue
        pid = int(match[1])
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            rows = dead.setdefault(pid, [])
            rows.append((info.st_mtime_ns, path))
        except PermissionError:
            continue
    ordered = sorted(
        dead.values(), key=lambda rows: max(row[0] for row in rows), reverse=True
    )
    for rows in ordered[2:]:
        for _, path in rows:
            # Recheck each entry; never unlink a symlink substituted meanwhile.
            if stat.S_ISREG(path.lstat().st_mode):
                path.unlink()


@contextmanager
def span(stage):
    started = time.monotonic_ns()
    success = False
    try:
        yield
        success = True
    finally:
        _emit(stage, started, success=success)


def api_render():
    return span("api_render")


def bind_external(request_id):
    state = _request.get()
    value = _hash(request_id)
    if state is not None and value is not None:
        state["identities"]["external_request_id"] = value
        _emit("external_id_bridge")


def internal_id_bridge(request):
    state = _request.get()
    if state is None:
        return
    external = _hash(getattr(request, "external_req_id", None))
    internal = _hash(getattr(request, "request_id", None))
    if external is not None:
        state["identities"]["external_request_id"] = external
    if internal is not None:
        state["identities"]["request_id"] = internal
    _emit("internal_id_bridge")


def _once(stage):
    state = _request.get()
    if state is None or stage in state["observed"]:
        return False
    state["observed"].add(stage)
    return True


def first_engine_output(output):
    state = _request.get()
    if state is None or "first_engine_output" in state["observed"]:
        return
    # CPU token-list lengths are the boundary, never the token values/text.
    if any(
        len(getattr(row, "token_ids", ())) for row in getattr(output, "outputs", ())
    ) and _once("first_engine_output"):
        _emit("first_engine_output")


def first_api_content(delta):
    state = _request.get()
    if state is None or "first_api_content" in state["observed"]:
        return
    # Empty role/usage chunks are protocol activity, not generated content.
    if (
        delta is not None
        and any(
            bool(getattr(delta, field, None))
            for field in ("content", "reasoning", "tool_calls")
        )
        and _once("first_api_content")
    ):
        _emit("first_api_content")


def model_warmup(function):
    """Capture base GPUWorker warmup before readiness, including deferred work."""

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        global _worker_recorder_pid
        if _worker_recorder_pid != os.getpid():
            try:
                recorder = telemetry.configure()
                _worker_recorder_pid = os.getpid()
                if recorder is not None:
                    recorder.source_hashes["request_timeline"] = hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest()
                    _emit("request_recorder_start")
            except Exception:  # noqa: BLE001 - diagnostics cannot abort startup
                _failed_diagnostic()
        with span("model_warmup"):
            return function(*args, **kwargs)

    return wrapped


class RequestTimelineMiddleware:
    """Observe ASGI transport milestones without buffering or parsing bodies."""

    def __init__(self, app):
        self.app = app
        # ASGI builds middleware before its lifespan/readiness handshake. Start
        # the observer here so the first HTTP arrival belongs to an established
        # recorder lifecycle, rather than creating that lifecycle mid-request.
        try:
            _api_recorder()
        except Exception:  # noqa: BLE001 - diagnostics cannot prevent startup
            _failed_diagnostic()

    async def __call__(self, scope, receive, send):
        entered = time.monotonic_ns()
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in ("/v1/chat/completions", "/v1/completions")
        ):
            return await self.app(scope, receive, send)
        try:
            _api_recorder()
        except Exception:  # noqa: BLE001 - logging failure cannot reject inference
            _failed_diagnostic()
        state = {
            "identities": {"http_request_id": _hash(uuid.uuid4().hex)},
            "observed": set(),
        }
        token = _request.set(state)
        _emit("http_request", point_ns=entered)
        body_started = None
        body_complete = False
        stream_complete = False
        normal_return = False

        async def observed_receive():
            nonlocal body_started, body_complete
            if body_started is None:
                body_started = time.monotonic_ns()
            message = await receive()
            if (
                message.get("type") == "http.request"
                and not message.get("more_body", False)
                and not body_complete
            ):
                body_complete = True
                _emit("http_body_receive", body_started, success=True)
            return message

        async def observed_send(message):
            nonlocal stream_complete
            await send(message)
            kind = message.get("type")
            if kind == "http.response.start":
                status = message.get("status")
                _emit(
                    "http_headers",
                    **({"status_code": status} if type(status) is int else {}),
                )
            elif kind == "http.response.body":
                if message.get("body") and _once("http_first_body"):
                    _emit("http_first_body")
                if not message.get("more_body", False):
                    stream_complete = True

        try:
            await self.app(scope, observed_receive, observed_send)
            normal_return = True
        finally:
            if body_started is not None and not body_complete:
                _emit("http_body_receive", body_started, success=False)
            _emit("http_end", success=stream_complete and normal_return)
            _request.reset(token)

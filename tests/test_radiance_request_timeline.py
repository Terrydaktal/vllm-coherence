from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def observer(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "request_timeline_test",
        ROOT / "src/qwen_r9700_lab/radiance_request_timeline.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events, configurations = [], []
    recorder = SimpleNamespace(source_hashes={})
    monkeypatch.setattr(
        module.telemetry,
        "emit_at",
        lambda stage, a, b, **values: events.append(
            {"stage": stage, "start_ns": a, "end_ns": b, **values}
        ),
    )
    monkeypatch.setattr(
        module.telemetry,
        "configure",
        lambda *args: configurations.append(args) or recorder,
    )
    monkeypatch.setattr(module, "_retire_api_recorders", lambda _base: None)
    return module, events, configurations


def scope(path="/v1/chat/completions"):
    return {"type": "http", "method": "POST", "path": path}


def test_transport_and_id_bridge_observe_without_parsing_or_changing_messages(observer):
    module, events, configs = observer
    original_body = {
        "type": "http.request",
        "body": b"private request body",
        "more_body": False,
    }
    messages = [
        {"type": "http.response.start", "status": 200, "headers": []},
        {
            "type": "http.response.body",
            "body": b"role-only private SSE",
            "more_body": True,
        },
        {
            "type": "http.response.body",
            "body": b"private generated text",
            "more_body": False,
        },
    ]
    seen = []

    async def app(_scope, receive, send):
        assert await receive() is original_body
        with module.api_render():
            pass
        module.bind_external("external-secret")
        request = SimpleNamespace(
            external_req_id="external-secret", request_id="external-secret-random"
        )
        module.internal_id_bridge(request)
        module.first_engine_output(
            SimpleNamespace(outputs=[SimpleNamespace(token_ids=[])])
        )
        module.first_engine_output(
            SimpleNamespace(outputs=[SimpleNamespace(token_ids=[42])])
        )
        module.first_engine_output(
            SimpleNamespace(outputs=[SimpleNamespace(token_ids=[43])])
        )
        module.first_api_content(
            SimpleNamespace(content="", reasoning=None, tool_calls=None)
        )
        await send(messages[0])
        await send(messages[1])
        module.first_api_content(
            SimpleNamespace(
                content=None, reasoning="private reasoning", tool_calls=None
            )
        )
        await send(messages[2])

    async def receive():
        return original_body

    async def send(message):
        seen.append(message)

    asyncio.run(module.RequestTimelineMiddleware(app)(scope(), receive, send))
    assert all(a is b for a, b in zip(messages, seen, strict=True))
    assert configs == [(f"{module.telemetry.DEFAULT_STATUS}-api-{os.getpid()}",)]
    stages = [row["stage"] for row in events]
    assert stages.count("first_engine_output") == stages.count("first_api_content") == 1
    assert stages.index("http_first_body") < stages.index("first_api_content")
    bridge = next(row for row in events if row["stage"] == "internal_id_bridge")
    assert bridge["request_id"] == module._hash("external-secret-random")
    assert bridge["external_request_id"] == module._hash("external-secret")
    assert len(bridge["http_request_id"]) == 64
    assert events[-1]["stage"] == "http_end" and events[-1]["success"] is True
    serialized = json.dumps(events)
    assert "private" not in serialized and "external-secret" not in serialized
    assert module._request.get() is None


def test_http_entry_includes_initial_recorder_work(observer, monkeypatch):
    module, events, _ = observer
    clock = [100]
    monkeypatch.setattr(module.time, "monotonic_ns", lambda: clock[0])

    async def app(_scope, _receive, _send):
        pass

    middleware = module.RequestTimelineMiddleware(app)
    monkeypatch.setattr(module, "_api_recorder", lambda: clock.__setitem__(0, 1000))
    asyncio.run(middleware(scope(), None, None))
    start = next(row for row in events if row["stage"] == "http_request")
    assert start["start_ns"] == start["end_ns"] == 100
    assert events[-1]["end_ns"] == 1000 and not events[-1]["success"]


def test_first_http_request_is_inside_its_real_recorder_lifecycle(
    observer, monkeypatch, tmp_path
):
    module, _, _ = observer
    records = []
    monkeypatch.setenv("QWEN_REQUEST_TIMELINE_STATUS", str(tmp_path / "status"))

    def configure(status):
        recorder = module.telemetry.Recorder(status, start=False, gc_events=False)
        records.append(recorder)
        return recorder

    monkeypatch.setattr(module.telemetry, "configure", configure)
    monkeypatch.setattr(
        module.telemetry,
        "emit_at",
        lambda stage, start, end, **values: records[-1].emit(stage, start, end, values),
    )

    async def app(_scope, _receive, send):
        await send({"type": "http.response.body", "body": b"x", "more_body": False})

    async def send(_message):
        pass

    middleware = module.RequestTimelineMiddleware(app)
    asyncio.run(middleware(scope(), None, send))
    recorder = records[0]
    start = next(row for row in recorder.pending if row["stage"] == "http_request")
    assert start["start_ns"] >= recorder.started_ns
    stages = [row["stage"] for row in recorder.pending]
    assert stages.index("api_recorder_started") < stages.index("http_request")
    recorder.close()


@pytest.mark.parametrize(
    "error", [RuntimeError("private exception"), asyncio.CancelledError()]
)
def test_failure_and_cancellation_preserve_original_exception(observer, error):
    module, events, _ = observer

    async def app(_scope, _receive, send):
        await send({"type": "http.response.body", "body": b"done", "more_body": False})
        with module.span("engine_submit"):
            raise error

    async def send(_message):
        pass

    with pytest.raises(type(error)) as caught:
        asyncio.run(module.RequestTimelineMiddleware(app)(scope(), None, send))
    assert caught.value is error
    assert not events[-1]["success"]
    assert (
        next(row for row in events if row["stage"] == "engine_submit")["success"]
        is False
    )
    assert "private exception" not in json.dumps(events)


def test_concurrent_api_contexts_remain_separate(observer):
    module, events, _ = observer

    async def app(value, _receive, send):
        module.bind_external(value["external"])
        await asyncio.sleep(0)
        module.internal_id_bridge(
            SimpleNamespace(
                external_req_id=value["external"],
                request_id=value["external"] + "-rand",
            )
        )
        await send({"type": "http.response.body", "body": b"x", "more_body": False})

    async def send(_message):
        pass

    async def run():
        middleware = module.RequestTimelineMiddleware(app)
        await asyncio.gather(
            *(
                middleware({**scope(), "external": value}, None, send)
                for value in ("a", "b")
            )
        )

    asyncio.run(run())
    bridges = [row for row in events if row["stage"] == "internal_id_bridge"]
    assert len({row["http_request_id"] for row in bridges}) == 2
    assert {row["request_id"] for row in bridges} == {
        module._hash("a-rand"),
        module._hash("b-rand"),
    }
    assert module._request.get() is None


def test_recorder_failure_never_prevents_request(observer, monkeypatch):
    module, _, _ = observer

    def broken(*_args, **_kwargs):
        raise OSError("private failure")

    monkeypatch.setattr(module, "_api_recorder", broken)
    monkeypatch.setattr(module.telemetry, "emit_at", broken)
    returned = []

    async def app(_scope, _receive, _send):
        returned.append(True)

    asyncio.run(module.RequestTimelineMiddleware(app)(scope(), None, None))
    assert returned == [True]


def test_other_routes_do_not_start_or_record(observer):
    module, events, configs = observer

    async def app(_scope, _receive, _send):
        return "normal"

    middleware = module.RequestTimelineMiddleware(app)
    events.clear()
    configs.clear()
    assert asyncio.run(middleware(scope("/health"), None, None)) == "normal"
    assert events == configs == []


def test_base_warmup_observes_success_and_original_failure(observer):
    module, events, configs = observer
    value = object()

    @module.model_warmup
    def warmup():
        return value

    assert warmup() is value
    assert configs == [()]
    assert events[-1]["stage"] == "model_warmup" and events[-1]["success"]

    @module.model_warmup
    def failed():
        raise RuntimeError("private failure")

    with pytest.raises(RuntimeError):
        failed()
    assert events[-1]["stage"] == "model_warmup" and not events[-1]["success"]


def test_inactive_api_file_retention_preserves_live_and_latest_two(
    tmp_path, monkeypatch
):
    spec = importlib.util.spec_from_file_location(
        "retention_test", ROOT / "src/qwen_r9700_lab/radiance_request_timeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prefix = tmp_path / "status"
    for pid in (1, 2, 3, 4):
        path = tmp_path / f"status-api-{pid}-cache-jobs.jsonl"
        path.write_text("evidence")
        os.utime(path, ns=(pid, pid))
    unrelated = tmp_path / "status-cache-jobs.jsonl"
    unrelated.write_text("worker")
    link = tmp_path / "status-api-5-cache-jobs.jsonl"
    link.symlink_to(unrelated)

    def liveness(pid, _signal):
        if pid != 1:
            raise ProcessLookupError

    monkeypatch.setattr(module.os, "kill", liveness)
    module._retire_api_recorders(prefix)
    assert (tmp_path / "status-api-1-cache-jobs.jsonl").exists()
    assert not (tmp_path / "status-api-2-cache-jobs.jsonl").exists()
    assert (tmp_path / "status-api-3-cache-jobs.jsonl").exists()
    assert (tmp_path / "status-api-4-cache-jobs.jsonl").exists()
    assert link.is_symlink() and unrelated.read_text() == "worker"

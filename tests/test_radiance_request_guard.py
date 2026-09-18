from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("route", ["/v1/completions", "/v1/chat/completions"])
@pytest.mark.parametrize("client_abi", [None, "old", "current"])
def test_stale_pi_request_is_rejected_before_engine_admission(monkeypatch, route, client_abi):
    responses = ModuleType("starlette.responses")
    responses.JSONResponse = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "starlette.responses", responses)
    monkeypatch.setenv("QWEN_RADIANCE_CACHE_ABI", "current")
    source = (
        Path(__file__).resolve().parents[1]
        / "experiments/radiance-public/radiance_request_guard.py"
    )
    spec = importlib.util.spec_from_file_location("request_guard_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    admitted = []
    payload = {"kv_transfer_params": {"qwen_chat": {"id": "synthetic"}}}
    if client_abi is not None:
        payload["kv_transfer_params"]["qwen_snapshot_abi"] = client_abi

    async def body():
        return payload

    request = SimpleNamespace(method="POST", url=SimpleNamespace(path=route), json=body)

    async def proceed(value):
        admitted.append(value)
        return "accepted"

    response = asyncio.run(module.require_snapshot_abi(request, proceed))
    if client_abi == "current":
        assert response == "accepted" and admitted == [request]
    else:
        assert not admitted
        assert response.status_code == 409
        assert response.content["error"]["code"] == "snapshot_abi_mismatch"


@pytest.mark.parametrize(
    "case,status",
    [("valid", 200), ("stale", 409), ("malformed", 400), ("huge", 400), ("unavailable", 503)],
)
def test_priority_route_reaches_cpu_utility_not_inference(monkeypatch, case, status):
    from test_radiance_fair_scheduler import load_module
    from test_radiance_priority import control

    scheduler = load_module(monkeypatch)
    levels = scheduler.AnswerPriorities()
    monkeypatch.setitem(sys.modules, "qwen_radiance_fair_scheduler", scheduler)
    responses = ModuleType("starlette.responses")
    responses.JSONResponse = lambda content, status_code=200: SimpleNamespace(
        content=content, status_code=status_code
    )
    monkeypatch.setitem(sys.modules, "starlette.responses", responses)
    monkeypatch.setenv("QWEN_RADIANCE_CACHE_ABI", "a" * 64)
    source = (
        Path(__file__).resolve().parents[1]
        / "experiments/radiance-public/radiance_request_guard.py"
    )
    spec = importlib.util.spec_from_file_location("priority_guard_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    called = []

    async def utility(name, value):
        called.append(name)
        if case == "unavailable":
            raise RuntimeError("not running")
        return levels.update(value)

    payload = {**control(), "abi": "old" if case == "stale" else "a" * 64}
    if case == "malformed":
        payload["prompt"] = "synthetic forbidden field"
    raw = b" " * 4097 if case == "huge" else json.dumps(payload).encode()

    async def stream():
        for i in range(0, len(raw), 13):
            yield raw[i : i + 13]

    async def inference(_):
        raise AssertionError("priority must not create an inference request")

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/qwen-radiance/priority"),
        stream=stream,
        app=SimpleNamespace(
            state=SimpleNamespace(
                engine_client=SimpleNamespace(
                    engine_core=SimpleNamespace(call_utility_async=utility)
                )
            )
        ),
    )
    result = asyncio.run(module.require_snapshot_abi(request, inference))
    assert result.status_code == status
    assert called == (["qwen_answer_priority"] if case in ("valid", "unavailable") else [])
    if case == "valid":
        assert levels.level(control()["chat_id"] + ":" + "0" * 64) == 1

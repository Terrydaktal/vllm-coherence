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
def test_stale_pi_request_is_rejected_before_engine_admission(
    monkeypatch, route, client_abi
):
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
    [
        ("valid", 200),
        ("stale", 409),
        ("malformed", 400),
        ("huge", 400),
        ("unavailable", 503),
    ],
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
    assert called == (
        ["qwen_answer_priority"] if case in ("valid", "unavailable") else []
    )
    if case == "valid":
        assert levels.level(control()["chat_id"] + ":" + "0" * 64) == 1


@pytest.mark.parametrize(
    "case,status",
    [
        ("valid", 200),
        ("unauthorized", 401),
        ("stale", 409),
        ("salt", 400),
        ("unknown_field", 400),
        ("huge", 400),
        ("unsafe_prefix", 409),
        ("unavailable", 503),
    ],
)
def test_compaction_preparation_authenticates_before_cpu_journal_access(
    monkeypatch, case, status
):
    responses = ModuleType("starlette.responses")
    responses.JSONResponse = lambda content, status_code=200: SimpleNamespace(
        content=content, status_code=status_code
    )
    monkeypatch.setitem(sys.modules, "starlette.responses", responses)
    protocol = ModuleType("vllm.entrypoints.openai.chat_completion.protocol")
    protocol.ChatCompletionRequest = SimpleNamespace(model_validate=lambda value: value)
    monkeypatch.setitem(sys.modules, protocol.__name__, protocol)
    runtime = ModuleType("qwen_radiance_token_continuation_runtime")
    called = []

    async def prepare(serving, chat_request, instruction, capacity):
        called.append((serving, chat_request, instruction, capacity))
        if case == "unsafe_prefix":
            raise ValueError("private synthetic content must not escape")
        if case == "unavailable":
            raise RuntimeError("private synthetic content must not escape")
        return {"contract": "radiance-compaction-prefix-v1", "prompt": [1, 2, 3]}

    runtime.prepare_compaction = prepare
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    monkeypatch.setenv("QWEN_RADIANCE_CACHE_ABI", "a" * 64)
    source = (
        Path(__file__).resolve().parents[1]
        / "experiments/radiance-public/radiance_request_guard.py"
    )
    spec = importlib.util.spec_from_file_location("compaction_guard_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class AuthenticationMiddleware:
        def __init__(self, app, tokens):
            assert app is None
            self.tokens = tokens

        def verify_token(self, headers):
            return headers.get("authorization") in [
                f"Bearer {token}" for token in self.tokens
            ]

    chat = {"id": "b" * 64, "generation": "c" * 64}
    payload = {
        "model": "synthetic",
        "messages": [{"role": "user", "content": "public fixture"}],
        "cache_salt": f"qwen-chat-cache-v1:{chat['id']}:{chat['generation']}",
        "kv_transfer_params": {"qwen_chat": chat, "qwen_snapshot_abi": "a" * 64},
        "instruction": "checkpoint",
        "context_window": 4096,
    }
    if case == "stale":
        payload["kv_transfer_params"]["qwen_snapshot_abi"] = "old"
    elif case == "salt":
        payload["cache_salt"] = "wrong"
    elif case == "unknown_field":
        payload["prompt"] = [1, 2, 3]
    raw = (
        b" " * (8 * 1024 * 1024 + 1) if case == "huge" else json.dumps(payload).encode()
    )
    reads = []

    async def stream():
        reads.append(True)
        yield raw

    async def inference(_request):
        raise AssertionError("checkpoint preparation must not create inference work")

    serving = object()
    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/qwen/prepare-compaction"),
        stream=stream,
        headers={
            "authorization": "Bearer wrong"
            if case == "unauthorized"
            else "Bearer fixture"
        },
        app=SimpleNamespace(
            user_middleware=[
                SimpleNamespace(
                    cls=AuthenticationMiddleware, kwargs={"tokens": ["fixture"]}
                )
            ],
            state=SimpleNamespace(openai_serving_chat=serving),
        ),
    )
    result = asyncio.run(module.require_snapshot_abi(request, inference))
    assert result.status_code == status
    assert bool(called) == (case in {"valid", "unsafe_prefix", "unavailable"})
    if case == "unauthorized":
        assert not reads
    assert "private synthetic" not in json.dumps(result.content)

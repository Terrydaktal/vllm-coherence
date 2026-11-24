"""Reject stale Pi cache identities at HTTP admission, before engine submission."""

from __future__ import annotations

import asyncio
import os

from starlette.responses import JSONResponse


async def require_snapshot_abi(request, call_next):
    if request.method == "POST" and request.url.path == "/qwen/prepare-compaction":
        return await prepare_compaction_control(request)
    if request.method == "POST" and request.url.path == "/qwen-radiance/priority":
        return await priority_control(request)
    if request.method == "POST" and request.url.path in (
        "/v1/chat/completions",
        "/v1/completions",
    ):
        try:
            payload = await request.json()
        except (ValueError, UnicodeDecodeError):
            return await call_next(request)
        if isinstance(payload, dict):
            params = payload.get("kv_transfer_params")
            if isinstance(params, dict) and params.get("qwen_chat") is not None:
                expected = os.environ.get("QWEN_RADIANCE_CACHE_ABI")
                if not expected or params.get("qwen_snapshot_abi") != expected:
                    return JSONResponse(
                        status_code=409,
                        content={
                            "error": {
                                "message": (
                                    "Radiance was upgraded. Exit and resume this Pi chat with "
                                    "pi-remote-qwen-radiance so its cache controls use the new "
                                    "snapshot version. Your transcript is unchanged."
                                ),
                                "type": "stale_snapshot_client",
                                "code": "snapshot_abi_mismatch",
                            }
                        },
                    )
    return await call_next(request)


async def prepare_compaction_control(request):
    """CPU-only checkpoint preparation with the configured API authentication.

    User middleware runs outside vLLM's authentication middleware, and this
    control path is outside /v1. Therefore authentication must happen here too,
    before accepting a body or opening a generated-token journal.
    """
    try:
        for middleware in request.app.user_middleware:
            if middleware.cls.__name__ == "AuthenticationMiddleware":
                verifier = middleware.cls(app=None, **middleware.kwargs)
                if not verifier.verify_token(request.headers):
                    return JSONResponse({"error": "Unauthorized"}, status_code=401)
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 8 * 1024 * 1024:
                raise ValueError("compaction preparation request too large")
        import json

        value = json.loads(raw)
        allowed = {
            "model",
            "messages",
            "tools",
            "chat_template_kwargs",
            "cache_salt",
            "kv_transfer_params",
            "instruction",
            "context_window",
        }
        if not isinstance(value, dict) or set(value) - allowed:
            raise ValueError("invalid compaction preparation request")
        params = value.get("kv_transfer_params")
        if not isinstance(params, dict) or not isinstance(
            params.get("qwen_chat"), dict
        ):
            raise TypeError("compaction cache identity is missing")
        expected = os.environ.get("QWEN_RADIANCE_CACHE_ABI")
        if not expected or params.get("qwen_snapshot_abi") != expected:
            return JSONResponse(
                {"error": "compaction snapshot ABI differs"}, status_code=409
            )
        try:
            from qwen_radiance_cache import cache_salt
        except ModuleNotFoundError as error:
            if error.name != "qwen_radiance_cache":
                raise
            from qwen_r9700_lab.radiance_cache import cache_salt
        if value.get("cache_salt") != cache_salt(params["qwen_chat"]):
            raise ValueError("compaction cache salt differs")
        instruction = value.pop("instruction")
        capacity = value.pop("context_window")
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )

        chat_request = ChatCompletionRequest.model_validate(
            {
                **value,
                "stream": True,
                "add_special_tokens": False,
            }
        )
    except (ValueError, TypeError, KeyError, UnicodeDecodeError):
        return JSONResponse(
            {"error": "invalid compaction preparation request"}, status_code=400
        )
    try:
        try:
            from qwen_radiance_token_continuation_runtime import prepare_compaction
        except ModuleNotFoundError as error:
            if error.name != "qwen_radiance_token_continuation_runtime":
                raise
            from qwen_r9700_lab.radiance_token_continuation_runtime import (
                prepare_compaction,
            )
        result = await prepare_compaction(
            request.app.state.openai_serving_chat, chat_request, instruction, capacity
        )
    except ValueError:
        return JSONResponse(
            {"error": "checkpoint prefix could not be prepared safely"}, status_code=409
        )
    except Exception:  # noqa: BLE001 - keep private rendering errors out of HTTP responses.
        return JSONResponse(
            {"error": "compaction preparation is unavailable"}, status_code=503
        )
    return JSONResponse(result)


async def priority_control(request):
    """Small control-plane IPC, independently serviceable during an SSE response."""
    from qwen_radiance_fair_scheduler import AnswerPriorities

    try:
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 4096:
                raise ValueError("priority request too large")
        import json

        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("invalid priority request")
        value = dict(value)
        abi = value.pop("abi", None)
        if (
            not os.environ.get("QWEN_RADIANCE_CACHE_ABI")
            or abi != os.environ["QWEN_RADIANCE_CACHE_ABI"]
        ):
            return JSONResponse(
                {
                    "error": "Priority control requires the current Radiance connection; exit and resume Pi."
                },
                status_code=409,
            )
        AnswerPriorities.validate(value)
    except (ValueError, TypeError, UnicodeDecodeError):
        return JSONResponse({"error": "invalid priority request"}, status_code=400)
    try:
        result = await asyncio.wait_for(
            request.app.state.engine_client.engine_core.call_utility_async(
                "qwen_answer_priority", value
            ),
            timeout=5,
        )
    except ValueError as error:
        return JSONResponse({"error": str(error)}, status_code=409)
    except Exception:
        return JSONResponse(
            {
                "error": "Priority was not confirmed; the scheduler is unavailable or needs the priority update."
            },
            status_code=503,
        )
    return JSONResponse(result)

"""Reject stale Pi cache identities at HTTP admission, before engine submission."""

from __future__ import annotations

import asyncio
import os

from starlette.responses import JSONResponse


async def require_snapshot_abi(request, call_next):
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

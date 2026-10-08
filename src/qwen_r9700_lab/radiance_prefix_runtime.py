"""Pinned serving integration for bounded, content-free prefix provenance.

Raw CPU token sequences stay in the observer's bounded RAM. No model arithmetic,
cache decisions, tokenizer outputs, or request/response fields are changed.
The template trace is inserted only into the authenticated template and produces
no rendered bytes; unknown templates report an unsupported diagnostic boundary.
"""

from __future__ import annotations

import hashlib
import os
import time

try:
    import qwen_radiance_cache_telemetry as telemetry
    import qwen_radiance_prefix_lineage as lineage
except ModuleNotFoundError as error:
    if error.name not in {
        "qwen_radiance_prefix_lineage",
        "qwen_radiance_cache_telemetry",
    }:
        raise
    from qwen_r9700_lab import radiance_cache_telemetry as telemetry
    from qwen_r9700_lab import radiance_prefix_lineage as lineage

TEMPLATE_SHA256 = "6e1439c913ad7df4a966493ad70de7e7fc5a548d41bbe417c1571f766603629b"
_observer = None
_observer_pid = None


def _get_observer():
    global _observer, _observer_pid
    if _observer_pid != os.getpid():
        _observer = lineage.PrefixLineageObserver()
        _observer_pid = os.getpid()
    return _observer


def _emit(row):
    values = dict(row)
    stage = values.pop("stage", "prefix_lineage")
    now = time.monotonic_ns()
    telemetry.emit_at(stage, now, now, **values)


def begin(state, request, tokenizer, config):
    if state is None or getattr(telemetry, "_recorder", None) is None:
        return
    observer = _get_observer()
    recorder = telemetry._recorder
    recorder.prefix_lineage_hooks = {
        "api_request": True,
        "engine_tokens": True,
        "serialized_delta": True,
        "input_processor": True,
        "template_trace": "per_request",
        "max_chats": observer.max_chats,
        "max_bytes": observer.max_bytes,
        "max_tokens": observer.max_tokens,
    }
    for name, module in (("prefix_lineage", lineage), ("prefix_runtime", __file__)):
        if name not in recorder.source_hashes:
            from pathlib import Path

            source = module if isinstance(module, str) else module.__file__
            recorder.source_hashes[name] = hashlib.sha256(
                Path(source).read_bytes()
            ).hexdigest()
    state["prefix_observer"] = observer
    state["prefix_handle"] = observer.begin(
        request,
        tokenizer,
        config,
        request_id=state["identities"].get("http_request_id"),
        emit=_emit,
    )


def call(state, method, *args, **kwargs):
    if state is None or "prefix_handle" not in state:
        return None
    return getattr(state["prefix_observer"], method)(
        state["prefix_handle"], *args, **kwargs
    )


def render_params(state, conversation, kwargs):
    if state is None or "prefix_handle" not in state:
        return

    def operation(before, after, name, index):
        try:
            call(state, "template_operation", before, after, name, index)
        except Exception:  # noqa: BLE001 - tracing cannot affect Jinja output
            state["prefix_trace_failed"] = True
        return ""

    kwargs["_coherence_prefix_trace"] = operation

    def supported(value):
        state["prefix_template_supported"] = bool(value)

    kwargs["_coherence_prefix_support"] = supported


def instrument_template(template, kwargs):
    operation = kwargs.pop("_coherence_prefix_trace", None)
    supported = kwargs.pop("_coherence_prefix_support", None)
    if operation is None:
        return template
    if (
        not isinstance(template, str)
        or hashlib.sha256(template.encode()).hexdigest() != TEMPLATE_SHA256
    ):
        if supported is not None:
            supported(False)
        return template
    replacements = [
        (
            "{%- elif message.role == 'assistant' %}",
            "{%- elif message.role == 'assistant' %}"
            "{%- set _lineage_assistant_rendered %}",
        ),
        (
            "{{- '<|im_end|>\\n' }}\n    {%- elif message.role == 'tool' %}",
            "{{- '<|im_end|>\\n' }}"
            "{%- endset %}"
            "{{- _lineage_assistant_rendered }}"
            "{%- set _lineage_observed = _coherence_prefix_observe('', _lineage_assistant_rendered, 'assistant_delimiters', head.count + loop.index0) %}\n"
            "    {%- elif message.role == 'tool' %}",
        ),
        (
            "{%- set content = render_content(message.content, true, is_system) | trim %}",
            "{%- set _lineage_content = render_content(message.content, true, is_system) %}"
            "{%- set content = _lineage_content | trim %}"
            "{%- set _lineage_observed = _coherence_prefix_observe(_lineage_content, content, 'content_trim', head.count + loop.index0) %}",
        ),
        (
            "{%- set content = content.split(_lead_end)[-1].lstrip('\\n') %}",
            "{%- set _lineage_before = content %}"
            "{%- set content = content.split(_lead_end)[-1].lstrip('\\n') %}"
            "{%- set _lineage_observed = _coherence_prefix_observe(_lineage_before, content, 'leading_delimiter', head.count + loop.index0) %}",
        ),
        (
            "{%- set reasoning_content = content.split(_think_end)[0].rstrip('\\n') %}",
            "{%- set _lineage_before = content %}"
            "{%- set reasoning_content = content.split(_think_end)[0].rstrip('\\n') %}",
        ),
        (
            "{%- set content = content.split(_think_end)[-1].lstrip('\\n') %}",
            "{%- set content = content.split(_think_end)[-1].lstrip('\\n') %}"
            "{%- set _lineage_observed = _coherence_prefix_observe(_lineage_before, '<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content, 'inline_reasoning_split', head.count + loop.index0) %}",
        ),
        (
            "{%- set reasoning_content = reasoning_content | trim %}",
            "{%- set _lineage_reasoning = reasoning_content %}"
            "{%- set reasoning_content = reasoning_content | trim %}"
            "{%- set _lineage_observed = _coherence_prefix_observe(_lineage_reasoning, reasoning_content, 'reasoning_trim', head.count + loop.index0) %}",
        ),
    ]
    traced = template
    for old, new in replacements:
        if traced.count(old) != 1:
            if supported is not None:
                supported(False)
            return template
        traced = traced.replace(old, new)
    kwargs["_coherence_prefix_observe"] = operation
    if supported is not None:
        supported(True)
    return traced


def rendered(state, serving, engine_inputs):
    if state is None or "prefix_handle" not in state:
        return
    if len(engine_inputs) != 1:
        call(state, "rendered", None)
        return
    tokens = serving._extract_prompt_components(engine_inputs[0]).token_ids
    with telemetry.span("prefix_lineage_observer", **state["identities"]):
        call(state, "rendered", tokens)
        call(
            state,
            "template_complete",
            state.get("prefix_template_supported", False)
            and not state.get("prefix_trace_failed", False),
        )


def output(state, result):
    if state is None or "prefix_handle" not in state:
        return
    outputs = getattr(result, "outputs", ())
    if len(outputs) == 1 and getattr(outputs[0], "index", 0) == 0:
        call(state, "engine_output", outputs[0].token_ids)
        if getattr(outputs[0], "finish_reason", None) is not None:
            state["prefix_terminal_observed"] = True
    elif outputs:
        call(state, "engine_output", None)


def finish(state, complete):
    if state is None:
        return
    call(
        state,
        "finish",
        complete=complete and bool(state.get("prefix_terminal_observed")),
    )


def full_choices(state, choices):
    if state is None or "prefix_handle" not in state:
        return
    if len(choices) != 1:
        call(state, "delivered_delta", {"content": []})
        return
    choice = choices[0]
    message = choice.message
    tools = []
    for index, item in enumerate(getattr(message, "tool_calls", None) or ()):
        value = item.model_dump(exclude_none=True)
        tools.append({**value, "index": index})
    call(
        state,
        "delivered_delta",
        {
            "content": getattr(message, "content", None),
            "reasoning": getattr(message, "reasoning", None),
            "tool_calls": tools,
        },
    )
    if getattr(choice, "finish_reason", None) is not None:
        state["prefix_terminal_observed"] = True

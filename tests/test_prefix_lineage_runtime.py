"""CPU-only serving/template tests using synthetic content and pinned source."""

from __future__ import annotations

import ast
import gzip
import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from jinja2 import Environment, meta

from qwen_r9700_lab import radiance_prefix_runtime as runtime
from qwen_r9700_lab import radiance_request_timeline as timeline

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (ROOT / "experiments/radiance-public/qwen-fixed-v22.3.jinja").read_text()


class Tokenizer:
    def __init__(self):
        self.decode_calls = 0

    def decode(self, tokens, **kwargs):
        assert kwargs == {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        self.decode_calls += 1
        return "".join("r" if value == 999 else chr(value) for value in tokens)

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return list(map(ord, text))

    def apply_chat_template(
        self, conversation, *, chat_template, tokenize, tools=None, **kwargs
    ):
        text = (
            Environment()
            .from_string(chat_template)
            .render(messages=conversation, tools=tools or [], **kwargs)
        )
        return self.encode(text) if tokenize else text


@pytest.fixture
def observed(monkeypatch):
    rows = []
    recorder = SimpleNamespace(source_hashes={}, dropped=0)
    monkeypatch.setattr(runtime, "_observer", None)
    monkeypatch.setattr(runtime, "_observer_pid", None)
    monkeypatch.setattr(runtime.telemetry, "_recorder", recorder)
    monkeypatch.setattr(
        runtime.telemetry,
        "emit_at",
        lambda stage, _start, _end, **values: rows.append({"stage": stage, **values}),
    )
    monkeypatch.setattr(runtime.telemetry, "span", lambda *_a, **_kw: nullcontext())
    return rows


def state(letter="a"):
    return {"identities": {"http_request_id": letter * 64}, "observed": set()}


def request(messages):
    return SimpleNamespace(messages=messages, cache_salt="synthetic", model="pinned")


def render(messages, *, st=None, tokenizer=None, original=False, template=TEMPLATE):
    tokenizer = tokenizer or Tokenizer()
    kwargs = {
        "enable_thinking": True,
        "preserve_thinking": True,
        "add_generation_prompt": True,
    }
    if not original:
        runtime.render_params(st, messages, kwargs)
        template = runtime.instrument_template(template, kwargs)
    return tokenizer.apply_chat_template(
        messages, chat_template=template, tokenize=False, **kwargs
    )


def record_render(st, tokenizer, text):
    ids = tokenizer.encode(text)
    serving = SimpleNamespace(
        _extract_prompt_components=lambda item: SimpleNamespace(token_ids=item)
    )
    runtime.rendered(st, serving, [ids])
    runtime.call(st, "input_processor", ids)


def first_response(
    tokenizer, *, reasoning, content, noncanonical=False, missing_terminal=False
):
    messages = [
        {"role": "system", "content": "synthetic system"},
        {"role": "user", "content": "synthetic question"},
    ]
    st = state()
    runtime.begin(st, request(messages), tokenizer, {})
    prompt = render(messages, st=st, tokenizer=tokenizer)
    record_render(st, tokenizer, prompt)
    raw_text = reasoning + "\n</think>\n\n" + content + "<|im_end|>"
    tokens = tokenizer.encode(raw_text)
    if noncanonical:
        tokens[raw_text.index("r")] = 999
    runtime.output(
        st,
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    token_ids=tokens,
                    index=0,
                    finish_reason=None if missing_terminal else "stop",
                )
            ]
        ),
    )
    runtime.call(st, "delivered_delta", {"reasoning": reasoning, "content": content})
    runtime.finish(st, True)
    following = [
        *messages,
        {"role": "assistant", "content": content, "reasoning_content": reasoning},
        {"role": "user", "content": "next synthetic question"},
    ]
    return following


def latest(rows, comparison):
    return next(row for row in reversed(rows) if row.get("comparison") == comparison)


@pytest.mark.parametrize(
    "assistant",
    [
        {"role": "assistant", "content": "answer", "reasoning_content": "reason"},
        {"role": "assistant", "content": " answer ", "reasoning_content": " reason "},
        {"role": "assistant", "content": "<think>\nreason\n</think>\nanswer"},
        {
            "role": "assistant",
            "content": "</think>\nanswer",
            "reasoning_content": "reason",
        },
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "reason",
            "tool_calls": [
                {
                    "id": "call",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path":"synthetic"}'},
                }
            ],
        },
    ],
)
def test_real_jinja_instrumentation_preserves_every_output_byte(observed, assistant):
    messages = [
        {"role": "system", "content": "synthetic system"},
        {"role": "user", "content": "question"},
        assistant,
        {"role": "user", "content": "next"},
    ]
    tokenizer, st = Tokenizer(), state()
    runtime.begin(st, request(messages), tokenizer, {})
    assert render(messages, st=st, tokenizer=tokenizer) == render(
        messages, tokenizer=tokenizer, original=True
    )
    assert st["prefix_template_supported"] is True


@pytest.mark.parametrize(
    "noncanonical,trim", [(True, False), (False, True), (True, True)]
)
def test_real_runtime_distinguishes_roundtrip_trim_and_both(
    observed, noncanonical, trim
):
    tokenizer = Tokenizer()
    messages = first_response(
        tokenizer,
        reasoning=" reason " if trim else "reason",
        content="answer",
        noncanonical=noncanonical,
    )
    st = state("b")
    runtime.begin(st, request(messages), tokenizer, {})
    reconstructed = render(messages, st=st, tokenizer=tokenizer)
    assert reconstructed == render(messages, tokenizer=tokenizer, original=True)
    record_render(st, tokenizer, reconstructed)
    assert latest(observed, "prompt_prefix")["equal"] is False
    assert latest(observed, "raw_to_reencoded")["equal"] is (not noncanonical)
    aggregate = latest(observed, "template_normalization")
    assert aggregate["diagnostic_status"] == "complete"
    assert aggregate["trim_changed"] is trim
    assert aggregate["delimiter_changed"] is trim
    assert latest(observed, "output_to_message")["equal"] is True
    serialized = json.dumps(observed)
    assert (
        "synthetic question" not in serialized
        and "reason " not in serialized
        and "999" not in serialized
    )


def test_same_serialization_is_a_hit_without_extra_tokenizer_work(observed):
    tokenizer = Tokenizer()
    messages = first_response(tokenizer, reasoning="reason", content="answer")
    st = state("b")
    runtime.begin(st, request(messages), tokenizer, {})
    record_render(st, tokenizer, render(messages, st=st, tokenizer=tokenizer))
    assert latest(observed, "prompt_prefix")["equal"] is True
    assert tokenizer.decode_calls == 0


def test_actual_tool_serialization_difference_is_not_hidden_by_parser_equality(
    observed,
):
    tokenizer, st = Tokenizer(), state()
    messages = [{"role": "user", "content": "synthetic question"}]
    config = {"tool_call_format": "json"}
    runtime.begin(st, request(messages), tokenizer, config)
    record_render(st, tokenizer, render(messages, st=st, tokenizer=tokenizer))
    raw = 'reason\n</think>\n\n<tool_call>\n{"name":"read","arguments":{"path":"synthetic"}}\n</tool_call><|im_end|>'
    runtime.output(
        st,
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    token_ids=tokenizer.encode(raw), index=0, finish_reason="stop"
                )
            ]
        ),
    )
    runtime.call(
        st,
        "delivered_delta",
        {
            "reasoning": "reason",
            "content": "",
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path":"synthetic"}'},
                }
            ],
        },
    )
    runtime.finish(st, True)
    following = [
        *messages,
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "reason",
            "tool_calls": [
                {
                    "id": "call",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path":"synthetic"}'},
                }
            ],
        },
        {"role": "tool", "content": "synthetic result", "tool_call_id": "call"},
    ]
    st = state("b")
    runtime.begin(st, request(following), tokenizer, config)
    kwargs = {
        "enable_thinking": True,
        "preserve_thinking": True,
        "add_generation_prompt": True,
        **config,
    }
    runtime.render_params(st, following, kwargs)
    traced = runtime.instrument_template(TEMPLATE, kwargs)
    text = tokenizer.apply_chat_template(
        following, chat_template=traced, tokenize=False, **kwargs
    )
    assert text == tokenizer.apply_chat_template(
        following,
        chat_template=TEMPLATE,
        tokenize=False,
        enable_thinking=True,
        preserve_thinking=True,
        add_generation_prompt=True,
        **config,
    )
    record_render(st, tokenizer, text)
    assert latest(observed, "output_to_message")["equal"] is True
    assert latest(observed, "raw_to_reencoded")["equal"] is True
    assert latest(observed, "prompt_prefix")["equal"] is False
    aggregate = latest(observed, "template_normalization")
    assert aggregate["diagnostic_status"] == "complete"
    assert aggregate["delimiter_changed"] is True and aggregate["trim_changed"] is False


def test_timeline_wrapper_isolates_observer_exceptions_from_stream_result(
    observed, monkeypatch
):
    tokenizer, st = Tokenizer(), state()
    runtime.begin(st, request([{"role": "user", "content": "question"}]), tokenizer, {})
    monkeypatch.setattr(
        st["prefix_observer"],
        "engine_output",
        lambda *_a: (_ for _ in ()).throw(RuntimeError("private observer failure")),
    )
    result = SimpleNamespace(
        outputs=[SimpleNamespace(token_ids=[97], index=0, finish_reason="stop")]
    )
    original = result.outputs[0].token_ids
    token = timeline._request.set(st)
    try:
        timeline.first_engine_output(result)
    finally:
        timeline._request.reset(token)
    assert result.outputs[0].token_ids is original and original == [97]
    assert runtime.telemetry._recorder.dropped == 1
    assert "private observer failure" not in json.dumps(observed)


def test_unknown_template_does_not_claim_normalization_coverage(observed):
    tokenizer = Tokenizer()
    messages = first_response(tokenizer, reasoning=" reason ", content="answer")
    st = state("b")
    runtime.begin(st, request(messages), tokenizer, {})
    altered = TEMPLATE + "{# unsupported source #}"
    text = render(messages, st=st, tokenizer=tokenizer, template=altered)
    record_render(st, tokenizer, text)
    assert (
        latest(observed, "template_normalization")["diagnostic_status"] == "unavailable"
    )
    assert latest(observed, "template_normalization")["reason"] == "unsupported"


def test_missing_model_terminal_is_incomplete_even_when_http_finishes(observed):
    tokenizer = Tokenizer()
    messages = first_response(
        tokenizer, reasoning="reason", content="answer", missing_terminal=True
    )
    st = state("b")
    runtime.begin(st, request(messages), tokenizer, {})
    record_render(st, tokenizer, render(messages, st=st, tokenizer=tokenizer))
    assert latest(observed, "prompt_prefix")["diagnostic_status"] == "incomplete"
    assert latest(observed, "prompt_prefix")["reason"] == "partial_output"


def test_disabled_recorder_does_not_capture_or_retokenize(monkeypatch):
    monkeypatch.setattr(runtime.telemetry, "_recorder", None)
    tokenizer, st = Tokenizer(), state()
    messages = [{"role": "user", "content": "question"}]
    runtime.begin(st, request(messages), tokenizer, {})
    assert "prefix_handle" not in st
    assert render(messages, st=st, tokenizer=tokenizer) == render(
        messages, tokenizer=tokenizer, original=True
    )
    runtime.output(
        st, SimpleNamespace(outputs=[SimpleNamespace(token_ids=[999], index=0)])
    )
    runtime.finish(st, True)
    assert tokenizer.decode_calls == 0


def test_template_callback_exception_keeps_identical_rendered_output(
    observed, monkeypatch
):
    tokenizer, st = Tokenizer(), state()
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": " answer ", "reasoning_content": " reason "},
    ]
    runtime.begin(st, request(messages), tokenizer, {})
    monkeypatch.setattr(
        st["prefix_observer"],
        "template_operation",
        lambda *_a: (_ for _ in ()).throw(RuntimeError("private failure")),
    )
    assert render(messages, st=st, tokenizer=tokenizer) == render(
        messages, tokenizer=tokenizer, original=True
    )
    assert st["prefix_trace_failed"] is True


@pytest.fixture
def pinned_hf_helpers():
    """Execute authenticated HF helpers with CPU-only tokenizer stubs."""
    spec = importlib.util.spec_from_file_location(
        "prefix_patch_test", ROOT / "experiments/radiance-public/patch_chat_snapshot.py"
    )
    patcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patcher)
    source = gzip.decompress(
        (ROOT / "tests/fixtures/vllm_028_prefix_lineage_hf.py.gz").read_bytes()
    ).decode()
    patched = patcher.prefix_renderer(source)
    tree = ast.parse(patched)
    wanted = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"safe_apply_chat_template", "resolve_chat_template_kwargs"}
        and node.body
        and not isinstance(node.body[0], ast.Expr)
    ]
    # Select the implementation, excluding overloaded ellipsis signatures.
    wanted = [
        node
        for node in wanted
        if not any(
            isinstance(item, ast.Name) and item.id == "overload"
            for item in node.decorator_list
        )
    ]
    for node in wanted:
        node.decorator_list = []
    env = Environment()
    namespace = {
        "request_timeline": SimpleNamespace(
            prefix_template=runtime.instrument_template,
            prefix_tokenize=timeline.prefix_tokenize,
        ),
        "resolve_chat_template": lambda _tokenizer, **values: values["chat_template"],
        "supports_kw": lambda *_a, **_kw: False,
        "_cached_resolve_chat_template_kwargs": lambda text: (
            meta.find_undeclared_variables(env.parse(text))
        ),
        "_get_hf_base_chat_template_params": lambda: {
            "add_generation_prompt",
            "enable_thinking",
            "preserve_thinking",
            "return_dict",
        },
        "ChatTemplateResolutionError": ValueError,
    }
    exec(  # noqa: S102 - execute only the pinned local synthetic fixture
        compile(
            "from __future__ import annotations\n"
            + "\n".join(ast.unparse(node) for node in wanted),
            "pinned_synthetic_renderer",
            "exec",
        ),
        namespace,
    )
    return namespace


def test_actual_renderer_kwarg_filter_keeps_the_injected_observer(
    observed, pinned_hf_helpers
):
    """Callbacks inserted after filtering would be lost before Jinja sees them."""
    namespace = pinned_hf_helpers
    tokenizer, st = Tokenizer(), state()
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": " answer ", "reasoning_content": " reason "},
    ]
    runtime.begin(st, request(messages), tokenizer, {})
    kwargs = {
        "add_generation_prompt": True,
        "enable_thinking": True,
        "preserve_thinking": True,
    }
    runtime.render_params(st, messages, kwargs)
    text = namespace["safe_apply_chat_template"](
        None, tokenizer, messages, chat_template=TEMPLATE, tokenize=False, **kwargs
    )
    assert text == render(messages, tokenizer=tokenizer, original=True)
    assert st["prefix_template_supported"] is True


@pytest.mark.parametrize("tagged", [True, False])
def test_actual_hf_text_mode_only_accepts_request_bound_token_callback(
    pinned_hf_helpers, tagged
):
    messages = [{"role": "user", "content": "synthetic question"}]
    tokenizer = Tokenizer()
    expected_ids = [80, 999, 1000, 78]
    calls = []

    def continuation(*args):
        calls.append(args)
        assert args[0] is tokenizer
        assert args[1] == messages
        assert "_coherence_token_encode" not in args[4]
        return expected_ids

    if tagged:
        continuation._coherence_text_to_tokens = True
    actual = pinned_hf_helpers["safe_apply_chat_template"](
        None,
        tokenizer,
        messages,
        chat_template=TEMPLATE,
        tokenize=False,
        _coherence_token_encode=continuation,
        enable_thinking=True,
        preserve_thinking=True,
        add_generation_prompt=True,
    )
    if tagged:
        assert actual is expected_ids
        assert len(calls) == 1
    else:
        assert actual == render(messages, tokenizer=tokenizer, original=True)
        assert calls == []


def test_token_mode_only_hf_patch_upgrades_to_text_mode_continuation():
    spec = importlib.util.spec_from_file_location(
        "prefix_v2_upgrade_test",
        ROOT / "experiments/radiance-public/patch_chat_snapshot.py",
    )
    patcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patcher)
    source = gzip.decompress(
        (ROOT / "tests/fixtures/vllm_028_prefix_lineage_hf.py.gz").read_bytes()
    ).decode()
    previous = source
    for old, new in patcher.HF_PREFIX_V2_HOOKS:
        assert old in previous  # Both sync and async renderer entry points exist.
        previous = previous.replace(old, new)
    latest = patcher.prefix_renderer(source)
    assert patcher.prefix_renderer(previous) == latest
    assert patcher.prefix_renderer(latest) == latest
    compile(latest, "upgraded_hf_renderer", "exec")
    with pytest.raises(ValueError, match="source differs"):
        patcher.prefix_renderer(previous + "\n# unqualified source change\n")

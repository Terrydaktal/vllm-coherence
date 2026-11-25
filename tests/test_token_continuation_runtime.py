"""CPU checks for admission of unchanged, exactly generated token histories.

The synthetic tokenizer deliberately has two encodings of the same text.  The
tests check admitted token IDs, never infer cache validity from decoded text.
"""

from __future__ import annotations

import ast
import asyncio
import copy
import gzip
import hashlib
import importlib.util
import json
import unicodedata
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from qwen_r9700_lab import radiance_token_continuation_runtime as runtime
from qwen_r9700_lab.radiance_token_continuation import TokenContinuationLedger

TEMPLATE = "synthetic pinned assistant template"
END = 1000
ALTERNATIVE_XY = 999
PROMPT = [ord("P")]
GENERATED = [ALTERNATIVE_XY, END]
CANONICAL_HISTORY = [ord("P"), ord("x"), ord("y"), END]
NEXT = CANONICAL_HISTORY + [ord("N")]


class AlternateSegmentationTokenizer:
    name_or_path = "synthetic-continuation-tokenizer"
    all_special_ids: ClassVar = [END]
    special_tokens_map: ClassVar = {"eos_token": "<|im_end|>"}

    def get_vocab(self):
        return {
            **{chr(value): value for value in range(128)},
            "xy": ALTERNATIVE_XY,
            "<|im_end|>": END,
        }

    def convert_tokens_to_ids(self, value):
        return self.get_vocab().get(value)

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        result = []
        while text:
            if text.startswith("<|im_end|>"):
                result.append(END)
                text = text[len("<|im_end|>") :]
            else:
                result.append(ord(text[0]))
                text = text[1:]
        return result

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(
            "xy"
            if value == ALTERNATIVE_XY
            else "<|im_end|>"
            if value == END
            else chr(value)
            for value in ids
        )


class Serving:
    @staticmethod
    def _extract_prompt_components(value):
        return SimpleNamespace(token_ids=value["prompt_token_ids"])


class CanonicalMergedTokenizer(AlternateSegmentationTokenizer):
    """The observed incident shortened old history when encoded from text."""

    def encode(self, text, *, add_special_tokens):
        result = super().encode(text, add_special_tokens=add_special_tokens)
        merged = []
        while result:
            if result[:2] == [ord("x"), ord("y")]:
                merged.append(ALTERNATIVE_XY)
                del result[:2]
            else:
                merged.append(result.pop(0))
        return merged


class SuffixOnlyTokenizer(AlternateSegmentationTokenizer):
    """Fail if incremental continuation ever re-encodes retained history."""

    def __init__(self):
        self.rendered_text = ""
        self.encoded_texts = []
        self.render_calls = []
        self.reject_history_encoding = False

    def apply_chat_template(
        self, *, conversation, tools, chat_template, tokenize, **kwargs
    ):
        assert tokenize is False
        assert chat_template == TEMPLATE
        assert not any(key.startswith("_coherence_") for key in kwargs)
        self.render_calls.append((conversation, tools, kwargs))
        return self.rendered_text

    def encode(self, text, *, add_special_tokens):
        self.encoded_texts.append(text)
        if self.reject_history_encoding:
            assert "P" not in text, (
                "retained history was passed back to tokenizer.encode"
            )
        return super().encode(text, add_special_tokens=add_special_tokens)


def request(messages=None, **overrides):
    values = {
        "model": "synthetic-model",
        "messages": messages or [{"role": "user", "content": "first"}],
        "cache_salt": "synthetic-chat-generation",
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        "stream": True,
        "n": 1,
        "chat_template": None,
        "tools": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def next_request(**overrides):
    return request(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "xy"},
            {"role": "user", "content": "next"},
        ],
        **overrides,
    )


def config():
    return {
        "template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
        "request_template": None,
        "model": "synthetic-model",
        "tools": None,
        "tokenizer_class": "AlternateSegmentationTokenizer",
        "tokenizer_name": AlternateSegmentationTokenizer.name_or_path,
    }


def state():
    return {"identities": {"http_request_id": "1" * 64}, "observed": set()}


def inputs(tokens, **extra):
    return [
        {
            "type": "token",
            "prompt_token_ids": list(tokens),
            "cache_salt": "synthetic-chat-generation",
            **extra,
        }
    ]


@pytest.fixture
def setup_runtime(tmp_path, monkeypatch):
    ledger = TokenContinuationLedger(tmp_path / "journals")
    monkeypatch.setattr(runtime, "_get_ledger", lambda *args, **kwargs: ledger)
    monkeypatch.setattr(
        runtime, "TEMPLATE_SHA256", hashlib.sha256(TEMPLATE.encode()).hexdigest()
    )
    monkeypatch.setenv("QWEN_RADIANCE_CACHE_ABI", "a" * 64)
    supported = runtime._incremental_boundary_supported
    monkeypatch.setattr(
        runtime,
        "_incremental_boundary_supported",
        lambda tokenizer: (
            isinstance(tokenizer, SuffixOnlyTokenizer) or supported(tokenizer)
        ),
    )
    return ledger, AlternateSegmentationTokenizer()


def begin(tokenizer, *, req=None, configuration=None, template=TEMPLATE):
    value = state()
    runtime.begin(value, req or request(), tokenizer, configuration or config())
    runtime.template(value, template)
    return value


def complete_first(
    tokenizer,
    *,
    finish_reason="stop",
    complete=True,
    admitted=True,
    output=None,
    delivered=None,
    configuration=None,
):
    value = begin(tokenizer, configuration=configuration)
    original = inputs(PROMPT)
    actual = runtime.rendered(value, Serving(), original)
    assert actual[0]["prompt_token_ids"] == PROMPT
    if admitted:
        runtime.input_processor(value, PROMPT)
    runtime.output(
        value,
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    index=0,
                    token_ids=GENERATED if output is None else output,
                    finish_reason=finish_reason,
                )
            ]
        ),
    )
    runtime.delivered(value, {"content": "xy"} if delivered is None else delivered)
    runtime.finish(value, complete)


def next_inputs(
    tokenizer, *, req=None, configuration=None, original=None, template=TEMPLATE
):
    value = begin(
        tokenizer,
        req=req or next_request(),
        configuration=configuration,
        template=template,
    )
    original = inputs(NEXT) if original is None else original
    before = copy.deepcopy(original)
    actual = runtime.rendered(value, Serving(), original)
    assert original == before, "admission must not mutate caller-owned engine inputs"
    return value, original, actual


def test_unchanged_history_preserves_actual_generated_ids_without_telemetry(
    setup_runtime,
):
    _, tokenizer = setup_runtime
    complete_first(tokenizer)
    _, original, actual = next_inputs(tokenizer)
    assert actual is not original
    assert actual[0]["prompt_token_ids"] == PROMPT + GENERATED + [ord("N")]
    assert actual[0]["cache_salt"] == original[0]["cache_salt"]
    assert tokenizer.decode(
        actual[0]["prompt_token_ids"],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ) == tokenizer.decode(
        original[0]["prompt_token_ids"],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def test_arrival_time_survives_generated_prefix_substitution(setup_runtime):
    _, tokenizer = setup_runtime
    complete_first(tokenizer)
    _, original, actual = next_inputs(
        tokenizer, original=inputs(NEXT, arrival_time=1234.5678)
    )
    assert actual[0]["prompt_token_ids"] == PROMPT + GENERATED + [ord("N")]
    assert actual[0]["arrival_time"] == original[0]["arrival_time"]


@pytest.mark.parametrize(
    "changed",
    ["old_user", "old_assistant", "model", "salt", "configuration", "template"],
)
def test_changes_require_canonical_fallback(setup_runtime, changed):
    _, tokenizer = setup_runtime
    complete_first(tokenizer)
    req, configuration, template = next_request(), config(), TEMPLATE
    if changed == "old_user":
        req.messages[0]["content"] = "edited"
    elif changed == "old_assistant":
        req.messages[1]["content"] = "edited"
    elif changed == "model":
        req.model = "another-model"
    elif changed == "salt":
        req.cache_salt = "another-chat-generation"
    elif changed == "configuration":
        configuration["template_kwargs"]["preserve_thinking"] = False
    elif changed == "template":
        template = "unrecognized template"
    _, original, actual = next_inputs(
        tokenizer, req=req, configuration=configuration, template=template
    )
    assert actual is original


@pytest.mark.parametrize("kind", ["cancel", "unterminated", "not_admitted", "not_stop"])
def test_partial_or_unadmitted_response_does_not_authorize_reuse(setup_runtime, kind):
    _, tokenizer = setup_runtime
    complete_first(
        tokenizer,
        complete=kind != "cancel",
        admitted=kind != "not_admitted",
        finish_reason=None
        if kind == "unterminated"
        else "length"
        if kind == "not_stop"
        else "stop",
    )
    _, original, actual = next_inputs(tokenizer)
    assert actual is original


@pytest.mark.parametrize(
    "extra",
    [
        {"token_type_ids": [0] * len(NEXT)},
        {"multi_modal_data": {"image": "placeholder"}},
        {"multi_modal_placeholders": {"image": []}},
        {"prompt_embeds": "placeholder"},
    ],
)
def test_per_token_or_multimodal_metadata_requires_fallback(setup_runtime, extra):
    _, tokenizer = setup_runtime
    complete_first(tokenizer)
    _, original, actual = next_inputs(tokenizer, original=inputs(NEXT, **extra))
    assert actual is original


def test_multiple_engine_inputs_require_fallback(setup_runtime):
    _, tokenizer = setup_runtime
    complete_first(tokenizer)
    _, original, actual = next_inputs(tokenizer, original=inputs(NEXT) + inputs(NEXT))
    assert actual is original


def test_same_text_with_changed_canonical_prefix_is_not_enough(setup_runtime):
    _, tokenizer = setup_runtime
    complete_first(tokenizer)
    # This is another valid encoding of the same rendered text, but not the
    # canonical-prefix extension certified against the retained history.
    _, original, actual = next_inputs(
        tokenizer, original=inputs(PROMPT + GENERATED + [ord("N")])
    )
    assert actual is original


def test_no_atomic_end_token_requires_fallback(setup_runtime):
    _, tokenizer = setup_runtime
    complete_first(tokenizer, output=[ALTERNATIVE_XY])
    _, original, actual = next_inputs(
        tokenizer, original=inputs([ord("P"), ord("x"), ord("y"), ord("N")])
    )
    assert actual is original


def test_input_processor_must_confirm_exact_admitted_sequence(setup_runtime):
    _, tokenizer = setup_runtime
    value = begin(tokenizer)
    runtime.rendered(value, Serving(), inputs(PROMPT))
    runtime.input_processor(value, [ord("Q")])
    runtime.output(
        value,
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    index=0,
                    token_ids=GENERATED,
                    finish_reason="stop",
                )
            ]
        ),
    )
    runtime.delivered(value, {"content": "xy"})
    runtime.finish(value, True)
    _, original, actual = next_inputs(tokenizer)
    assert actual is original


def test_ledger_survives_api_process_restart(setup_runtime, tmp_path, monkeypatch):
    _, tokenizer = setup_runtime
    complete_first(tokenizer)
    reloaded = TokenContinuationLedger(tmp_path / "journals")
    monkeypatch.setattr(runtime, "_get_ledger", lambda *args, **kwargs: reloaded)
    _, _, actual = next_inputs(tokenizer)
    assert actual[0]["prompt_token_ids"] == PROMPT + GENERATED + [ord("N")]


def test_retokenization_exception_returns_original_without_partial_changes(
    setup_runtime, monkeypatch
):
    _, tokenizer = setup_runtime
    complete_first(tokenizer)
    monkeypatch.setattr(
        tokenizer,
        "encode",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("synthetic")),
    )
    _, original, actual = next_inputs(tokenizer)
    assert actual is original


def test_nonstream_choices_do_not_authorize_delta_id_reuse(setup_runtime):
    _, tokenizer = setup_runtime
    value = begin(tokenizer, req=request(stream=False))
    runtime.rendered(value, Serving(), inputs(PROMPT))
    runtime.input_processor(value, PROMPT)
    runtime.output(
        value,
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    index=0,
                    token_ids=GENERATED,
                    finish_reason="stop",
                )
            ]
        ),
    )
    runtime.full_choices(
        value,
        [
            SimpleNamespace(
                index=0,
                message=SimpleNamespace(
                    content="xy",
                    reasoning=None,
                    reasoning_content=None,
                    tool_calls=None,
                ),
            )
        ],
    )
    runtime.finish(value, True)
    _, original, actual = next_inputs(tokenizer)
    assert actual is original


def test_rewrite_cannot_exceed_model_context_limit(setup_runtime):
    _, tokenizer = setup_runtime
    complete_first(tokenizer)
    value = begin(tokenizer, req=next_request())
    original = inputs(NEXT)
    serving = Serving()
    serving.model_config = SimpleNamespace(max_model_len=4)
    assert runtime.rendered(value, serving, original) is original


def test_preserved_generated_history_may_have_more_tokens_than_canonical_text(
    setup_runtime,
):
    tokenizer = CanonicalMergedTokenizer()
    generated = [ord("x"), ord("y"), END]
    complete_first(tokenizer, output=generated)
    canonical = [ord("P"), ALTERNATIVE_XY, END, ord("N")]
    _, _, actual = next_inputs(tokenizer, original=inputs(canonical))
    assert actual[0]["prompt_token_ids"] == PROMPT + generated + [ord("N")]
    assert len(actual[0]["prompt_token_ids"]) == len(canonical) + 1


def test_longer_generated_prefix_must_fit_even_when_canonical_prompt_fits(
    setup_runtime,
):
    tokenizer = CanonicalMergedTokenizer()
    complete_first(tokenizer, output=[ord("x"), ord("y"), END])
    value = begin(tokenizer, req=next_request())
    original = inputs([ord("P"), ALTERNATIVE_XY, END, ord("N")])
    serving = Serving()
    serving.model_config = SimpleNamespace(max_model_len=5)
    assert len(original[0]["prompt_token_ids"]) < serving.model_config.max_model_len
    assert runtime.rendered(value, serving, original) is original


def test_pending_emitted_token_is_not_misrepresented_as_processed_cache_state(
    setup_runtime,
):
    ledger, tokenizer = setup_runtime
    complete_first(tokenizer)
    value, _, actual = next_inputs(tokenizer)
    turn = value["token_continuation"]
    record = ledger.get_record(turn["identity"])
    assert list(record.tokens) == PROMPT + GENERATED
    assert record.processed_tokens is None
    assert actual[0]["prompt_token_ids"] == PROMPT + GENERATED + [ord("N")]


def test_overlapping_requests_cannot_reuse_old_journal(setup_runtime):
    _, tokenizer = setup_runtime
    complete_first(tokenizer)
    first = begin(tokenizer, req=next_request())
    second = begin(tokenizer, req=next_request())
    original = inputs(NEXT)
    assert runtime.rendered(first, Serving(), original) is original
    assert runtime.rendered(second, Serving(), original) is original
    runtime.finish(first, False)
    runtime.finish(second, False)


def test_generated_prefix_can_be_extended_across_multiple_completed_turns(
    setup_runtime,
):
    _, tokenizer = setup_runtime
    complete_first(tokenizer)
    second, _, actual = next_inputs(tokenizer)
    runtime.input_processor(second, actual[0]["prompt_token_ids"])
    runtime.output(
        second,
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    index=0,
                    token_ids=GENERATED,
                    finish_reason="stop",
                )
            ]
        ),
    )
    runtime.delivered(second, {"content": "xy"})
    runtime.finish(second, True)
    req = next_request()
    req.messages.extend(
        [
            {"role": "assistant", "content": "xy"},
            {"role": "user", "content": "third"},
        ]
    )
    canonical = NEXT + [ord("x"), ord("y"), END, ord("T")]
    _, _, actual = next_inputs(tokenizer, req=req, original=inputs(canonical))
    assert actual[0]["prompt_token_ids"] == PROMPT + GENERATED + [
        ord("N")
    ] + GENERATED + [ord("T")]


def test_delivered_reasoning_and_split_tool_arguments_must_match_next_request(
    setup_runtime,
):
    _, tokenizer = setup_runtime
    value = begin(tokenizer)
    runtime.rendered(value, Serving(), inputs(PROMPT))
    runtime.input_processor(value, PROMPT)
    runtime.output(
        value,
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    index=0,
                    token_ids=GENERATED,
                    finish_reason="tool_calls",
                )
            ]
        ),
    )
    runtime.delivered(
        value,
        {
            "reasoning": "consider ",
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "execute", "arguments": '{"x":'},
                }
            ],
        },
    )
    runtime.delivered(
        value,
        {
            "reasoning_content": "this",
            "tool_calls": [
                {
                    "index": 0,
                    "function": {"arguments": "1}"},
                }
            ],
        },
    )
    runtime.finish(value, True)
    req = next_request()
    req.messages[1] = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "consider this",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "execute",
                    "arguments": '{"x":1}',
                },
            }
        ],
    }
    req.messages[2] = {"role": "tool", "tool_call_id": "call-1", "content": "result"}
    _, _, actual = next_inputs(tokenizer, req=req)
    assert actual[0]["prompt_token_ids"] == PROMPT + GENERATED + [ord("N")]


@pytest.mark.parametrize(
    "fault", ["multiple_rows", "second_terminal", "invalid_id", "invalid_delta"]
)
def test_unreliable_output_accounting_prevents_journal_commit(setup_runtime, fault):
    _, tokenizer = setup_runtime
    value = begin(tokenizer)
    runtime.rendered(value, Serving(), inputs(PROMPT))
    runtime.input_processor(value, PROMPT)
    row = SimpleNamespace(index=0, token_ids=GENERATED, finish_reason="stop")
    rows = [row, row] if fault == "multiple_rows" else [row]
    if fault == "invalid_id":
        row.token_ids = [True, END]
    runtime.output(value, SimpleNamespace(outputs=rows))
    if fault == "second_terminal":
        runtime.output(value, SimpleNamespace(outputs=[row]))
    runtime.delivered(
        value,
        {"content": ["unsupported"]} if fault == "invalid_delta" else {"content": "xy"},
    )
    runtime.finish(value, True)
    _, original, actual = next_inputs(tokenizer)
    assert actual is original


def test_nonlossless_canonical_roundtrip_cannot_authorize_rewrite(
    setup_runtime, monkeypatch
):
    _, tokenizer = setup_runtime
    complete_first(tokenizer)
    monkeypatch.setattr(
        tokenizer, "encode", lambda *args, **kwargs: [ord("Q"), ord("x"), ord("y"), END]
    )
    _, original, actual = next_inputs(
        tokenizer, original=inputs([ord("Q"), ord("x"), ord("y"), END, ord("N")])
    )
    assert actual is original


def test_full_decoded_text_must_still_match_after_token_substitution(
    setup_runtime, monkeypatch
):
    _, tokenizer = setup_runtime
    complete_first(tokenizer)
    decode = tokenizer.decode

    def context_sensitive_decode(ids, **kwargs):
        text = decode(ids, **kwargs)
        # Simulate tokenizer normalization dependent on tokens beyond the old
        # boundary. Prefix equality alone must not let this rewrite through.
        if list(ids) == PROMPT + GENERATED + [ord("N")]:
            return text + " changed"
        return text

    monkeypatch.setattr(tokenizer, "decode", context_sensitive_decode)
    _, original, actual = next_inputs(tokenizer)
    assert actual is original


@pytest.mark.parametrize(
    "mode",
    [
        {"stream": False},
        {"n": 2},
        {"use_beam_search": True},
        {"truncate_prompt_tokens": 100},
        {"pad_prompt_tokens": 100},
        {"add_special_tokens": True},
        {"continue_final_message": True},
        {"return_assistant_tokens_mask": True},
    ],
)
def test_unsupported_serving_modes_require_original_prompt(setup_runtime, mode):
    _, tokenizer = setup_runtime
    complete_first(tokenizer)
    _, original, actual = next_inputs(tokenizer, req=next_request(**mode))
    assert actual is original


@pytest.mark.parametrize(
    "content",
    [
        [
            {
                "type": "image_url",
                "image_url": {"url": "https://synthetic.invalid/image"},
            }
        ],
        [
            {
                "type": "input_audio",
                "input_audio": {"data": "synthetic", "format": "wav"},
            }
        ],
        [
            {
                "type": "video_url",
                "video_url": {"url": "https://synthetic.invalid/video"},
            }
        ],
        {"unexpected": "content"},
    ],
)
def test_multimodal_or_unknown_content_refuses_continuation_before_rendering(
    setup_runtime, content
):
    _, tokenizer = setup_runtime
    req = request([{"role": "user", "content": content}])
    value = begin(tokenizer, req=req)
    assert "token_continuation" not in value
    original = inputs(PROMPT)
    assert runtime.rendered(value, Serving(), original) is original


def test_reports_never_include_raw_content_or_token_values(setup_runtime, monkeypatch):
    _, tokenizer = setup_runtime
    from qwen_r9700_lab import radiance_cache_telemetry

    reports = []
    monkeypatch.setattr(
        radiance_cache_telemetry,
        "emit_at",
        lambda *args, **kwargs: reports.append((args, kwargs)),
    )
    complete_first(tokenizer)
    next_inputs(tokenizer)
    assert reports
    for args, fields in reports:
        assert args[0] == "token_continuation"
        assert (
            not {
                "content",
                "reasoning",
                "tokens",
                "prompt_token_ids",
                "messages",
                "text",
            }
            & fields.keys()
        )
        assert all(
            type(value) in (str, int, bool, float, type(None))
            for value in fields.values()
        )


class CompactionRequest(SimpleNamespace):
    def model_copy(self, *, update):
        return CompactionRequest(**{**vars(self), **update})


class CompactionTokenizer(SuffixOnlyTokenizer):
    def apply_chat_template(
        self, *, conversation, tools, chat_template, tokenize, **kwargs
    ):
        assert tokenize is False
        assert chat_template == TEMPLATE
        text = ""
        for index, message in enumerate(conversation):
            if index == 0:
                text += "P" if message["content"] == "first" else "EDITED"
            elif message["role"] == "assistant":
                text += message["content"] + "<|im_end|>\n"
            else:
                text += message["content"]
        if kwargs.get("add_generation_prompt"):
            text += "<|im_start|>assistant\n<think>\n"
        return text


class CompactionServing(Serving):
    def __init__(self, tokenizer):
        self.renderer = SimpleNamespace(tokenizer=tokenizer)
        self.model_config = SimpleNamespace(max_model_len=4096)

    def _effective_chat_template_kwargs(self, _request):
        return config()["template_kwargs"]

    async def render_chat_request(self, req):
        from qwen_r9700_lab import radiance_request_timeline as timeline

        tokenizer = self.renderer.tokenizer
        kwargs = {
            **self._effective_chat_template_kwargs(req),
            "add_generation_prompt": req.add_generation_prompt,
        }
        timeline.prefix_render_params(req.messages, kwargs)
        continuation = kwargs.pop("_coherence_token_encode", None)
        template = timeline.prefix_template(TEMPLATE, kwargs)
        ids = timeline.prefix_tokenize(
            tokenizer,
            req.messages,
            req.tools,
            template,
            kwargs,
            False,
            False,
            continuation=continuation,
        )
        if ids is None:
            text = tokenizer.apply_chat_template(
                conversation=req.messages,
                tools=req.tools,
                chat_template=template,
                tokenize=False,
                **kwargs,
            )
            ids = tokenizer.encode(text, add_special_tokens=False)
        return req.messages, inputs(ids)


def compaction_setup():
    tokenizer = CompactionTokenizer()
    configuration = config()
    configuration["tokenizer_class"] = type(tokenizer).__qualname__
    complete_first(tokenizer, configuration=configuration)
    tokenizer.encoded_texts.clear()
    tokenizer.reject_history_encoding = True
    req = CompactionRequest(
        **vars(
            request(
                [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "xy"},
                ]
            )
        )
    )
    return tokenizer, req


def test_compaction_preserves_generated_history_instead_of_reencoding(setup_runtime):
    ledger, _ = setup_runtime
    tokenizer, req = compaction_setup()
    result = asyncio.run(
        runtime.prepare_compaction(
            CompactionServing(tokenizer), req, "checkpoint", 4096
        )
    )
    assert result["contract"] == "radiance-compaction-prefix-v1"
    assert result["prompt"][:3] == PROMPT + GENERATED
    assert result["historical_tokens"] == 4  # raw P + xy + boundary + newline
    assert result["continuation"] == {
        "reason": "generated_prefix_preserved",
        "preserved_tokens": 3,
        "canonical_tokens": None,
    }
    assert all("P" not in text for text in tokenizer.encoded_texts)
    assert not ledger._active


@pytest.mark.parametrize(
    ("model_limit", "expectation"),
    [
        (319_000, nullcontext()),
        (253_792, pytest.raises(ValueError, match="invalid compaction prompt tokens")),
    ],
)
def test_compaction_supports_319k_but_enforces_the_engine_context(
    setup_runtime, model_limit, expectation
):
    ledger, _ = setup_runtime
    tokenizer, req = compaction_setup()
    req.messages[0]["content"] = "changed to require canonical encoding"
    tokenizer.reject_history_encoding = False
    serving = CompactionServing(tokenizer)
    serving.model_config.max_model_len = model_limit
    render_chat_request = serving.render_chat_request

    async def long_synthetic_prompt(request):
        messages, engine_inputs = await render_chat_request(request)
        prompt = engine_inputs[0]["prompt_token_ids"]
        return messages, inputs([42] * 318_900 + prompt)

    serving.render_chat_request = long_synthetic_prompt
    with expectation:
        result = asyncio.run(
            runtime.prepare_compaction(serving, req, "checkpoint", 319_000)
        )
        assert 318_900 < len(result["prompt"]) <= 319_000
    assert not ledger._active


@pytest.mark.parametrize("changed", ["user", "assistant"])
def test_compaction_changed_messages_fall_back_without_forcing_old_tokens(
    setup_runtime, changed
):
    ledger, _ = setup_runtime
    tokenizer, req = compaction_setup()
    tokenizer.reject_history_encoding = False
    req.messages[0 if changed == "user" else 1]["content"] = "edited"
    result = asyncio.run(
        runtime.prepare_compaction(
            CompactionServing(tokenizer), req, "checkpoint", 4096
        )
    )
    assert result["continuation"]["preserved_tokens"] == 0
    assert result["continuation"]["canonical_tokens"] == result["historical_tokens"]
    assert result["prompt"][:3] != PROMPT + GENERATED
    assert not ledger._active


def test_compaction_keeps_prior_completed_prefix_when_latest_tail_was_aborted(
    setup_runtime,
):
    ledger, _ = setup_runtime
    tokenizer, req = compaction_setup()
    # A cancelled later request does not replace the completed journal. Its
    # partial text may be retained, but is encoded only after that exact prefix.
    req.messages += [
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "partial answer"},
    ]
    result = asyncio.run(
        runtime.prepare_compaction(
            CompactionServing(tokenizer), req, "checkpoint", 4096
        )
    )
    assert result["prompt"][:3] == PROMPT + GENERATED
    assert result["continuation"]["preserved_tokens"] == 3
    assert result["historical_tokens"] > 4
    assert not ledger._active


def test_compaction_refuses_instruction_that_rewrites_historical_template(
    setup_runtime,
):
    ledger, _ = setup_runtime
    tokenizer, req = compaction_setup()
    tokenizer.reject_history_encoding = False
    render = tokenizer.apply_chat_template

    def changed_render(**kwargs):
        text = render(**kwargs)
        return (
            text.replace("P", "EDITED", 1)
            if kwargs.get("add_generation_prompt")
            else text
        )

    tokenizer.apply_chat_template = changed_render
    with pytest.raises(ValueError, match="changed the historical prompt"):
        asyncio.run(
            runtime.prepare_compaction(
                CompactionServing(tokenizer), req, "checkpoint", 4096
            )
        )
    assert not ledger._active


@pytest.mark.parametrize("second_template", [TEMPLATE, "unrecognized template"])
def test_actual_timeline_pipeline_preserves_history_across_executor_without_recorder(
    setup_runtime, monkeypatch, second_template
):
    """The pinned HF renderer runs in an executor without the ASGI context."""
    _, tokenizer = setup_runtime
    from qwen_r9700_lab import radiance_cache_telemetry
    from qwen_r9700_lab import radiance_request_timeline as timeline

    monkeypatch.setattr(radiance_cache_telemetry, "_recorder", None)
    monkeypatch.setattr(
        radiance_cache_telemetry, "emit_at", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(timeline, "_api_recorder", lambda: None)
    admitted = []
    requests = iter([request(), next_request()])

    async def app(_scope, receive, send):
        await receive()
        req = next(requests)
        timeline.prefix_begin(req, tokenizer, config()["template_kwargs"])
        kwargs = config()["template_kwargs"]
        timeline.prefix_render_params(req.messages, kwargs)
        assert callable(kwargs["_coherence_token_template"])
        selected_template = second_template if admitted else TEMPLATE

        def render_in_executor():
            # This deliberately uses run_in_executor rather than to_thread:
            # vLLM's make_async path does not copy the request ContextVar.
            assert timeline._request.get() is None
            kwargs.pop("_coherence_token_encode", None)
            result = timeline.prefix_template(selected_template, kwargs)
            assert "_coherence_token_template" not in kwargs
            assert kwargs == config()["template_kwargs"]
            return result

        assert (
            await asyncio.get_running_loop().run_in_executor(None, render_in_executor)
            == selected_template
        )
        original = inputs(PROMPT if not admitted else NEXT, arrival_time=1234.5678)
        actual = timeline.prefix_rendered(Serving(), original)
        admitted.append(actual[0]["prompt_token_ids"])
        timeline.internal_id_bridge(
            SimpleNamespace(
                prompt_token_ids=actual[0]["prompt_token_ids"],
                external_req_id="external",
                request_id="internal",
            )
        )
        timeline.first_engine_output(
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        index=0,
                        token_ids=GENERATED,
                        finish_reason="stop",
                    )
                ]
            )
        )
        timeline.first_api_content(
            SimpleNamespace(content="xy", reasoning=None, tool_calls=None)
        )
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send(
            {"type": "http.response.body", "body": b"synthetic", "more_body": False}
        )

    async def receive():
        return {"type": "http.request", "body": b"synthetic", "more_body": False}

    async def send(_message):
        pass

    async def run():
        middleware = timeline.RequestTimelineMiddleware(app)
        for _ in range(2):
            await middleware(
                {"type": "http", "method": "POST", "path": "/v1/chat/completions"},
                receive,
                send,
            )

    asyncio.run(run())
    expected = PROMPT + GENERATED + [ord("N")] if second_template == TEMPLATE else NEXT
    assert admitted == [PROMPT, expected]
    assert timeline._request.get() is None
    assert radiance_cache_telemetry._recorder is None


def test_pinned_serving_uses_rewritten_length_for_completion_budget(setup_runtime):
    """Execute the actual patched assignment and budget nodes with CPU stubs."""
    _, _tokenizer = setup_runtime
    tokenizer = CanonicalMergedTokenizer()
    complete_first(tokenizer, output=[ord("x"), ord("y"), END])
    value = begin(tokenizer, req=next_request())
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "continuation_pinned_serving_patch",
        root / "experiments/radiance-public/patch_chat_snapshot.py",
    )
    patcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patcher)
    original = gzip.decompress(
        (root / "tests/fixtures/vllm_028_request_timeline_serving.py.gz").read_bytes()
    ).decode()
    tree = ast.parse(patcher.stream_buffered_tool_usage(original))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "_create_chat_completion"
    )
    rewrite = next(
        node
        for node in method.body
        if isinstance(node, ast.Assign) and "prefix_rendered" in ast.unparse(node)
    )
    loop = next(node for node in method.body if isinstance(node, ast.For))
    prompt = next(
        node
        for node in loop.body
        if isinstance(node, ast.Assign)
        and "prompt_token_ids" in ast.unparse(node.targets[0])
    )
    budget = next(
        node
        for node in loop.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "get_max_tokens"
    )
    assert rewrite.lineno < loop.lineno <= budget.lineno
    collect = ast.parse("captured.append((prompt_token_ids, max_tokens))").body[0]
    block = ast.Module(
        body=[
            rewrite,
            ast.For(
                target=loop.target,
                iter=loop.iter,
                body=[prompt, budget, collect],
                orelse=[],
            ),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(block)
    serving = Serving()
    serving._extract_prompt_len = lambda item: len(item["prompt_token_ids"])
    serving.default_sampling_params = {}
    serving.override_max_tokens = None
    namespace = {
        "request_timeline": SimpleNamespace(
            prefix_rendered=lambda self, items: runtime.rendered(value, self, items)
        ),
        "self": serving,
        "engine_inputs": inputs(
            [ord("P"), ALTERNATIVE_XY, END, ord("N")], arrival_time=1234.5678
        ),
        "request": SimpleNamespace(
            max_completion_tokens=None, max_tokens=None, truncate_prompt_tokens=None
        ),
        "max_model_len": 100,
        "get_max_tokens": lambda model_max, requested, prompt_len, *args, **kwargs: (
            model_max - prompt_len
        ),
        "captured": [],
    }
    exec(  # noqa: S102 - authenticated production AST with CPU stubs
        compile(block, "<actual pinned serving admission/budget nodes>", "exec"),
        namespace,
    )
    assert namespace["captured"] == [(PROMPT + [ord("x"), ord("y"), END, ord("N")], 95)]


def prepare_suffix_only(
    tokenizer, *, output=None, delivered=None, complete=True, configuration=None
):
    complete_first(
        tokenizer,
        output=output,
        delivered=delivered,
        complete=complete,
        configuration=configuration,
    )
    tokenizer.encoded_texts.clear()
    tokenizer.reject_history_encoding = True


def encode_incremental(
    value, tokenizer, req, *, text="Pxy<|im_end|>N", template=TEMPLATE
):
    tokenizer.rendered_text = text
    return runtime.encode_prompt(
        value, tokenizer, req.messages, req.tools, template, config()["template_kwargs"]
    )


def finish_incremental(value, admitted):
    runtime.input_processor(value, admitted)
    runtime.output(
        value,
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    index=0,
                    token_ids=GENERATED,
                    finish_reason="stop",
                )
            ]
        ),
    )
    runtime.delivered(value, {"content": "xy"})
    runtime.finish(value, True)


def test_incremental_encoding_never_tokenizes_previous_history(setup_runtime):
    tokenizer = SuffixOnlyTokenizer()
    prepare_suffix_only(tokenizer)
    req = next_request()
    value = begin(tokenizer, req=req)
    admitted = encode_incremental(value, tokenizer, req)
    assert admitted == PROMPT + GENERATED + [ord("N")]
    assert tokenizer.encoded_texts == ["N"]
    assert len(tokenizer.render_calls) == 1
    original = inputs(admitted, arrival_time=1234.5)
    actual = runtime.rendered(value, Serving(), original)
    assert actual[0]["prompt_token_ids"] == admitted
    assert tokenizer.encoded_texts == ["N"], (
        "post-render admission must not re-encode old IDs"
    )


def test_incremental_encoding_retains_exact_raw_ids_across_three_turns(setup_runtime):
    tokenizer = SuffixOnlyTokenizer()
    prepare_suffix_only(tokenizer)
    req = next_request()
    value = begin(tokenizer, req=req)
    admitted = encode_incremental(value, tokenizer, req)
    runtime.rendered(value, Serving(), inputs(admitted))
    finish_incremental(value, admitted)
    req.messages.extend(
        [
            {"role": "assistant", "content": "xy"},
            {"role": "user", "content": "third"},
        ]
    )
    third = begin(tokenizer, req=req)
    actual = encode_incremental(
        third, tokenizer, req, text="Pxy<|im_end|>Nxy<|im_end|>T"
    )
    assert actual == PROMPT + GENERATED + [ord("N")] + GENERATED + [ord("T")]
    assert tokenizer.encoded_texts == ["N", "T"]


@pytest.mark.parametrize(
    "changed",
    [
        "old_user",
        "old_assistant",
        "reasoning_purge",
        "configuration",
        "template",
        "cancel",
    ],
)
def test_incremental_history_changes_use_canonical_fallback(setup_runtime, changed):
    tokenizer = SuffixOnlyTokenizer()
    prepare_suffix_only(
        tokenizer,
        delivered={"content": "xy", "reasoning": "old thought"},
        complete=changed != "cancel",
    )
    req, configuration, template = next_request(), config(), TEMPLATE
    req.messages[1]["reasoning_content"] = "old thought"
    if changed == "old_user":
        req.messages[0]["content"] = "edited"
    elif changed == "old_assistant":
        req.messages[1]["content"] = "edited"
    elif changed == "reasoning_purge":
        del req.messages[1]["reasoning_content"]
    elif changed == "configuration":
        configuration["template_kwargs"]["preserve_thinking"] = False
    elif changed == "template":
        template = "unknown template"
    value = begin(tokenizer, req=req, configuration=configuration, template=template)
    assert encode_incremental(value, tokenizer, req, template=template) is None
    assert tokenizer.encoded_texts == []


def test_incremental_encoding_does_not_split_at_literal_old_end_marker(setup_runtime):
    tokenizer = SuffixOnlyTokenizer()
    literal = list(map(ord, "<|im_end|>"))
    generated = [ALTERNATIVE_XY] + literal + [ord("x"), END]
    prepare_suffix_only(
        tokenizer, output=generated, delivered={"content": "xy<|im_end|>x"}
    )
    req = next_request()
    req.messages[1]["content"] = "xy<|im_end|>x"
    value = begin(tokenizer, req=req)
    actual = encode_incremental(value, tokenizer, req, text="Pxy<|im_end|>x<|im_end|>N")
    assert actual == PROMPT + generated + [ord("N")]
    assert tokenizer.encoded_texts == ["N"]


def test_literal_end_marker_cannot_substitute_for_actual_terminal_token(setup_runtime):
    tokenizer = SuffixOnlyTokenizer()
    generated = [ALTERNATIVE_XY] + list(map(ord, "<|im_end|>"))
    prepare_suffix_only(tokenizer, output=generated)
    req = next_request()
    value = begin(tokenizer, req=req)
    assert encode_incremental(value, tokenizer, req) is None
    assert tokenizer.encoded_texts == []


@pytest.mark.parametrize(
    "text", ["Qxy<|im_end|>N", "Pxy<|im_end|>", "Pxy<|im_end|>N trailing"]
)
def test_incremental_uses_whole_retained_prefix_and_actual_rendered_suffix(
    setup_runtime, text
):
    tokenizer = SuffixOnlyTokenizer()
    prepare_suffix_only(tokenizer)
    req = next_request()
    value = begin(tokenizer, req=req)
    actual = encode_incremental(value, tokenizer, req, text=text)
    if not text.startswith("Pxy<|im_end|>") or text == "Pxy<|im_end|>":
        assert actual is None
    else:
        assert actual == PROMPT + GENERATED + list(map(ord, "N trailing"))
        assert tokenizer.encoded_texts == ["N trailing"]


def test_incremental_suffix_normalization_cannot_silently_change_rendered_text(
    setup_runtime, monkeypatch
):
    tokenizer = SuffixOnlyTokenizer()
    prepare_suffix_only(tokenizer)
    req = next_request()
    value = begin(tokenizer, req=req)
    encode = tokenizer.encode
    monkeypatch.setattr(
        tokenizer, "encode", lambda text, **kwargs: encode(text.lstrip(), **kwargs)
    )
    assert encode_incremental(value, tokenizer, req, text="Pxy<|im_end|> N") is None


def test_incremental_nfc_change_requires_canonical_fallback(setup_runtime, monkeypatch):
    tokenizer = SuffixOnlyTokenizer()
    prepare_suffix_only(tokenizer)
    req = next_request()
    value = begin(tokenizer, req=req)
    encode = tokenizer.encode
    monkeypatch.setattr(
        tokenizer,
        "encode",
        lambda text, **kwargs: encode(unicodedata.normalize("NFC", text), **kwargs),
    )
    assert (
        encode_incremental(value, tokenizer, req, text="Pxy<|im_end|>e\u0301") is None
    )


def test_incremental_cross_boundary_decode_change_is_rejected(
    setup_runtime, monkeypatch
):
    tokenizer = SuffixOnlyTokenizer()
    prepare_suffix_only(tokenizer)
    req = next_request()
    value = begin(tokenizer, req=req)
    decode = tokenizer.decode

    def changed_decode(ids, **kwargs):
        text = decode(ids, **kwargs)
        return (
            text + "changed" if list(ids) == PROMPT + GENERATED + [ord("N")] else text
        )

    monkeypatch.setattr(tokenizer, "decode", changed_decode)
    assert encode_incremental(value, tokenizer, req) is None


def test_incremental_length_limit_falls_back_without_truncation(
    setup_runtime, monkeypatch, tmp_path
):
    ledger = TokenContinuationLedger(tmp_path / "small-journals", max_tokens=4)
    monkeypatch.setattr(runtime, "_get_ledger", lambda: ledger)
    tokenizer = SuffixOnlyTokenizer()
    prepare_suffix_only(tokenizer)
    req = next_request()
    value = begin(tokenizer, req=req)
    assert encode_incremental(value, tokenizer, req, text="Pxy<|im_end|>NN") is None


def test_incremental_tool_result_preserves_actual_tool_call_history(setup_runtime):
    tokenizer = SuffixOnlyTokenizer()
    tool = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "execute", "arguments": '{"x":1}'},
    }
    prepare_suffix_only(
        tokenizer,
        delivered={"reasoning": "consider", "tool_calls": [{"index": 0, **tool}]},
    )
    req = next_request()
    req.messages[1] = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "consider",
        "tool_calls": [tool],
    }
    req.messages[2] = {"role": "tool", "tool_call_id": "call-1", "content": "result"}
    value = begin(tokenizer, req=req)
    actual = encode_incremental(value, tokenizer, req)
    assert actual == PROMPT + GENERATED + [ord("N")]
    assert tokenizer.encoded_texts == ["N"]


def test_incremental_overlap_after_render_begins_cannot_authorize_admission(
    setup_runtime, monkeypatch
):
    tokenizer = SuffixOnlyTokenizer()
    prepare_suffix_only(tokenizer)
    req = next_request()
    value = begin(tokenizer, req=req)
    render = tokenizer.apply_chat_template
    competing = []

    def start_competing(**kwargs):
        competing.append(begin(tokenizer, req=next_request()))
        return render(**kwargs)

    monkeypatch.setattr(tokenizer, "apply_chat_template", start_competing)
    assert encode_incremental(value, tokenizer, req) is None
    for other in competing:
        runtime.finish(other, False)


@pytest.mark.parametrize("tokenize", [True, False])
def test_incremental_encode_callback_survives_executor_without_request_context(
    setup_runtime, monkeypatch, tokenize
):
    from qwen_r9700_lab import radiance_cache_telemetry
    from qwen_r9700_lab import radiance_request_timeline as timeline

    tokenizer = SuffixOnlyTokenizer()
    configuration = config()
    configuration["tokenizer_class"] = "SuffixOnlyTokenizer"
    prepare_suffix_only(tokenizer, configuration=configuration)
    req = next_request()
    tokenizer.rendered_text = "Pxy<|im_end|>N"
    monkeypatch.setattr(radiance_cache_telemetry, "_recorder", None)
    monkeypatch.setattr(
        radiance_cache_telemetry, "emit_at", lambda *args, **kwargs: None
    )

    async def run():
        value = state()
        context = timeline._request.set(value)
        try:
            timeline.prefix_begin(req, tokenizer, config()["template_kwargs"])
            kwargs = config()["template_kwargs"]
            timeline.prefix_render_params(req.messages, kwargs)

            def worker():
                assert timeline._request.get() is None
                continuation = kwargs.pop("_coherence_token_encode")
                assert continuation._coherence_text_to_tokens is True
                selected_template = timeline.prefix_template(TEMPLATE, kwargs)
                assert not any(key.startswith("_coherence_") for key in kwargs)
                return timeline.prefix_tokenize(
                    tokenizer,
                    req.messages,
                    req.tools,
                    selected_template,
                    kwargs,
                    tokenize,
                    False,
                    continuation=continuation,
                )

            actual = await asyncio.get_running_loop().run_in_executor(None, worker)
            assert actual == PROMPT + GENERATED + [ord("N")]
            assert tokenizer.encoded_texts == ["N"]
        finally:
            timeline._continuation("finish", False)
            timeline._request.reset(context)

    asyncio.run(run())


@pytest.mark.parametrize("tokenize,mask", [(False, False), (True, True)])
def test_incremental_tokenize_hook_refuses_unsupported_output_shapes(
    setup_runtime, tokenize, mask
):
    from qwen_r9700_lab import radiance_request_timeline as timeline

    calls = []
    assert (
        timeline.prefix_tokenize(
            None,
            [],
            None,
            TEMPLATE,
            {},
            tokenize,
            mask,
            continuation=lambda *args: calls.append(args),
        )
        is None
    )
    assert calls == []


@pytest.mark.parametrize(
    "fault",
    [
        "stale_lease",
        "metadata",
        "model_limit",
        "changed_ids",
        "different_salt",
        "multiple_inputs",
    ],
)
def test_incremental_admission_changes_are_rejected_after_ids_returned(
    setup_runtime, fault
):
    tokenizer = SuffixOnlyTokenizer()
    prepare_suffix_only(tokenizer)
    req = next_request()
    value = begin(tokenizer, req=req)
    admitted = encode_incremental(value, tokenizer, req)
    original = inputs(admitted)
    serving = Serving()
    if fault == "stale_lease":
        begin(tokenizer, req=next_request())
    elif fault == "metadata":
        original[0]["token_type_ids"] = [0] * len(admitted)
    elif fault == "model_limit":
        serving.model_config = SimpleNamespace(max_model_len=len(admitted))
    elif fault == "changed_ids":
        original[0]["prompt_token_ids"][-1] = ord("Q")
    elif fault == "different_salt":
        original[0]["cache_salt"] = "other-chat"
    else:
        original.append(copy.deepcopy(original[0]))
    before = copy.deepcopy(original)
    with pytest.raises(
        ValueError, match="incremental prompt admission is no longer valid"
    ):
        runtime.rendered(value, serving, original)
    assert original == before


def test_timeline_cannot_hide_rejected_incremental_admission(setup_runtime):
    from qwen_r9700_lab import radiance_request_timeline as timeline

    tokenizer = SuffixOnlyTokenizer()
    prepare_suffix_only(tokenizer)
    req = next_request()
    value = begin(tokenizer, req=req)
    admitted = encode_incremental(value, tokenizer, req)
    begin(tokenizer, req=next_request())
    context = timeline._request.set(value)
    try:
        with pytest.raises(
            ValueError, match="incremental prompt admission is no longer valid"
        ):
            timeline.prefix_rendered(Serving(), inputs(admitted))
    finally:
        timeline._request.reset(context)


class ToolRoundtripTokenizer(SuffixOnlyTokenizer):
    """Use the real pinned Jinja while retaining a transparent CPU token oracle."""

    def __init__(self):
        super().__init__()
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        self.environment = ImmutableSandboxedEnvironment()
        self.environment.globals["raise_exception"] = self.fail_render
        self.template_text = (
            Path(__file__).resolve().parents[1]
            / "experiments/radiance-public/qwen-fixed-v22.3.jinja"
        ).read_text()

    @staticmethod
    def fail_render(message):
        raise ValueError(message)

    def apply_chat_template(
        self, *, conversation, tools, chat_template, tokenize, **kwargs
    ):
        assert tokenize is False and chat_template == self.template_text
        messages = copy.deepcopy(conversation)
        for message in messages:
            for call in message.get("tool_calls", []):
                arguments = call["function"]["arguments"]
                if isinstance(arguments, str):
                    call["function"]["arguments"] = json.loads(arguments)
        values = {"add_generation_prompt": True, **kwargs}
        return self.environment.from_string(chat_template).render(
            messages=messages, tools=tools, **values
        )

    def encode(self, text, *, add_special_tokens):
        self.encoded_texts.append(text)
        if self.reject_history_encoding:
            assert "UNIQUE_PRIOR_HISTORY" not in text
        return AlternateSegmentationTokenizer.encode(
            self, text, add_special_tokens=add_special_tokens
        )


def completed_tool_roundtrip(setup_runtime, monkeypatch, *, preserve_thinking=True):
    ledger, _ = setup_runtime
    tokenizer = ToolRoundtripTokenizer()
    monkeypatch.setattr(
        runtime,
        "TEMPLATE_SHA256",
        hashlib.sha256(tokenizer.template_text.encode()).hexdigest(),
    )
    cfg = config()
    cfg["template_kwargs"]["preserve_thinking"] = preserve_thinking
    cfg["template_kwargs"]["reasoning_effort"] = "xhigh"
    tools = [
        {
            "type": "function",
            "function": {
                "name": "inspect_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "flags": {"type": "array"},
                    },
                },
            },
        }
    ]
    messages = [
        {
            "role": "user",
            "content": "UNIQUE_PRIOR_HISTORY inspect the two public fixture files",
        }
    ]
    req = request(messages, tools=tools)
    initial = begin(
        tokenizer, req=req, configuration=cfg, template=tokenizer.template_text
    )
    assert (
        runtime.encode_prompt(
            initial,
            tokenizer,
            messages,
            tools,
            tokenizer.template_text,
            cfg["template_kwargs"],
        )
        is None
    )
    prompt = tokenizer.encode(
        tokenizer.apply_chat_template(
            conversation=messages,
            tools=tools,
            chat_template=tokenizer.template_text,
            tokenize=False,
            **cfg["template_kwargs"],
        ),
        add_special_tokens=False,
    )
    runtime.rendered(initial, Serving(), inputs(prompt))
    runtime.input_processor(initial, prompt)
    fields = {
        "reasoning": " Check both files carefully. \n",
        "content": "Inspecting both files. \n\n",
        "tool_calls": [],
    }
    xml = []
    for index, filename in enumerate(("one.txt", "two.txt")):
        arguments = {"path": filename, "flags": ["café", "beta"]}
        fields["tool_calls"].append(
            {
                "id": f"call-{index}",
                "type": "function",
                "function": {
                    "name": "inspect_file",
                    "arguments": json.dumps(arguments, ensure_ascii=True),
                },
            }
        )
        xml.append(
            f'<tool_call>\n<function=inspect_file>\n<parameter=path>\n{filename}\n</parameter>\n<parameter=flags>\n["café", "beta"]\n</parameter>\n</function>\n</tool_call>'
        )
    output_text = (
        fields["reasoning"]
        + "</think>\n\n"
        + fields["content"]
        + "\n".join(xml)
        + "<|im_end|>"
    )
    generated = tokenizer.encode(output_text, add_special_tokens=False)
    runtime.output(
        initial,
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    index=0, token_ids=generated, finish_reason="tool_calls"
                )
            ]
        ),
    )
    runtime.delivered(
        initial,
        {
            "reasoning": fields["reasoning"],
            "content": fields["content"],
            "tool_calls": [
                {"index": index, **call}
                for index, call in enumerate(fields["tool_calls"])
            ],
        },
    )
    runtime.finish(initial, True)
    assert initial["token_continuation_reason"] == "journal_saved"
    assistant = {
        "role": "assistant",
        "content": fields["content"],
        "reasoning_content": fields["reasoning"],
        "tool_calls": copy.deepcopy(fields["tool_calls"]),
    }
    for call in assistant["tool_calls"]:
        call["function"]["arguments"] = json.dumps(
            json.loads(call["function"]["arguments"]),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    messages.extend(
        [
            assistant,
            {
                "role": "tool",
                "tool_call_id": "call-0",
                "content": "first fixture result",
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "second fixture result",
            },
        ]
    )
    tokenizer.encoded_texts.clear()
    tokenizer.reject_history_encoding = True
    return ledger, tokenizer, cfg, req, prompt + generated


def encode_tool_roundtrip(tokenizer, cfg, req):
    value = begin(
        tokenizer, req=req, configuration=cfg, template=tokenizer.template_text
    )
    admitted = runtime.encode_prompt(
        value,
        tokenizer,
        req.messages,
        req.tools,
        tokenizer.template_text,
        cfg["template_kwargs"],
    )
    return value, admitted


def test_two_tool_calls_keep_raw_tokens_across_json_and_jinja_normalization(
    setup_runtime, monkeypatch
):
    ledger, tokenizer, cfg, req, raw = completed_tool_roundtrip(
        setup_runtime, monkeypatch
    )
    record = next(iter(ledger._records.values()))
    assert record.canonical_prefix_sha256 is not None
    assert record.output_identity_format == "json_semantics_v1"
    assert record.prompt_tokens < len(raw)
    value, admitted = encode_tool_roundtrip(tokenizer, cfg, req)
    assert admitted[: len(raw)] == raw
    assert value["token_continuation_reason"] == "rendered_assistant_prefix_preserved"
    assert len(tokenizer.encoded_texts) == 1
    assert "first fixture result" in tokenizer.encoded_texts[0]
    assert "UNIQUE_PRIOR_HISTORY" not in tokenizer.encoded_texts[0]
    canonical = tokenizer.apply_chat_template(
        conversation=req.messages,
        tools=req.tools,
        chat_template=tokenizer.template_text,
        tokenize=False,
        **cfg["template_kwargs"],
    )
    assert (
        tokenizer.decode(
            admitted, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        != canonical
    )


def test_tool_normalization_witness_survives_journal_restart_and_third_turn(
    setup_runtime, monkeypatch
):
    ledger, tokenizer, cfg, req, raw = completed_tool_roundtrip(
        setup_runtime, monkeypatch
    )
    root = ledger.root
    ledger = TokenContinuationLedger(root)
    monkeypatch.setattr(runtime, "_get_ledger", lambda *args, **kwargs: ledger)
    value, admitted = encode_tool_roundtrip(tokenizer, cfg, req)
    runtime.rendered(value, Serving(), inputs(admitted))
    runtime.input_processor(value, admitted)
    generated = tokenizer.encode(
        "I checked both.\n</think>\n\nBoth fixtures pass.<|im_end|>",
        add_special_tokens=False,
    )
    runtime.output(
        value,
        SimpleNamespace(
            outputs=[
                SimpleNamespace(index=0, token_ids=generated, finish_reason="stop")
            ]
        ),
    )
    runtime.delivered(
        value, {"reasoning": "I checked both.\n", "content": "Both fixtures pass."}
    )
    runtime.finish(value, True)
    req.messages.extend(
        [
            {
                "role": "assistant",
                "content": "Both fixtures pass.",
                "reasoning_content": "I checked both.\n",
            },
            {"role": "user", "content": "Continue with the next fixture"},
        ]
    )
    tokenizer.encoded_texts.clear()
    _, third = encode_tool_roundtrip(tokenizer, cfg, req)
    assert third[: len(admitted) + len(generated)] == admitted + generated
    assert third[: len(raw)] == raw
    assert (
        len(tokenizer.encoded_texts) == 1
        and "UNIQUE_PRIOR_HISTORY" not in tokenizer.encoded_texts[0]
    )


@pytest.mark.parametrize(
    "fault",
    [
        "missing_function_close",
        "missing_call_close",
        "changed_value",
        "duplicate_parameter",
        "extra_parameter",
        "trailing_text",
        "injected_value_members",
    ],
)
def test_raw_tool_correspondence_rejects_incomplete_or_changed_xml(
    setup_runtime, monkeypatch, fault
):
    ledger, tokenizer, _, req, raw = completed_tool_roundtrip(
        setup_runtime, monkeypatch
    )
    record = next(iter(ledger._records.values()))
    prompt = raw[: record.prompt_tokens]
    output = tokenizer.decode(
        raw[record.prompt_tokens :],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    fields = runtime._assistant_fields(req.messages[1])
    turn = {
        "tokenizer": tokenizer,
        "prompt": prompt,
        "output": raw[record.prompt_tokens :],
    }
    assert runtime._raw_output_matches(turn, fields) is True
    if fault == "missing_function_close":
        output = output.replace("</function>", "", 1)
    elif fault == "missing_call_close":
        output = output.replace("</tool_call>", "", 1)
    elif fault == "changed_value":
        output = output.replace("one.txt", "changed.txt", 1)
    elif fault == "injected_value_members":
        output = output.replace(
            '["café", "beta"]', '["café", "beta"],"ignored":true', 1
        )
    elif fault == "duplicate_parameter":
        output = output.replace(
            "</function>", "<parameter=path>\none.txt\n</parameter>\n</function>", 1
        )
    elif fault == "extra_parameter":
        output = output.replace(
            "</function>", "<parameter=extra>\nunexpected\n</parameter>\n</function>", 1
        )
    else:
        output = output.replace("<|im_end|>", "unexpected suffix<|im_end|>")
    turn["output"] = tokenizer.encode(output, add_special_tokens=False)
    assert runtime._raw_output_matches(turn, fields) is False
    assert runtime._completed_canonical_prefix(turn, fields) is None


def test_unverified_xml_keeps_the_strict_journal_without_a_render_witness(
    setup_runtime, monkeypatch
):
    ledger, tokenizer, cfg, req, _ = completed_tool_roundtrip(
        setup_runtime, monkeypatch
    )
    value, admitted = encode_tool_roundtrip(tokenizer, cfg, req)
    runtime.rendered(value, Serving(), inputs(admitted))
    runtime.input_processor(value, admitted)
    generated = tokenizer.encode(
        "Work.\n</think>\n\n<tool_call>\n<function=inspect_file>\n<|im_end|>",
        add_special_tokens=False,
    )
    runtime.output(
        value,
        SimpleNamespace(
            outputs=[
                SimpleNamespace(index=0, token_ids=generated, finish_reason="stop")
            ]
        ),
    )
    runtime.delivered(value, {"reasoning": "Work.\n", "content": ""})
    runtime.finish(value, True)
    record = next(iter(ledger._records.values()))
    assert list(record.tokens) == admitted + generated
    assert record.canonical_prefix_sha256 is None
    assert record.canonical_prefix_characters is None
    assert value["token_continuation_reason"] == "journal_saved"


@pytest.mark.parametrize(
    "change",
    [
        "string",
        "array_order",
        "boolean",
        "call_id",
        "call_name",
        "call_type",
        "old_user",
        "reasoning",
        "content",
        "new_thinking_control",
        "configuration",
        "duplicate_keys",
        "nonfinite",
    ],
)
def test_tool_normalization_rejects_changed_payload_or_render_context(
    setup_runtime, monkeypatch, change
):
    _, tokenizer, cfg, req, _ = completed_tool_roundtrip(setup_runtime, monkeypatch)
    call = req.messages[1]["tool_calls"][0]
    args = json.loads(call["function"]["arguments"])
    if change == "string":
        args["path"] = " one.txt"
    elif change == "array_order":
        args["flags"].reverse()
    elif change == "boolean":
        args["flags"] = [True, "beta"]
    elif change == "call_id":
        call["id"] = "changed-call"
    elif change == "call_name":
        call["function"]["name"] = "another_tool"
    elif change == "call_type":
        call["type"] = "custom"
    elif change == "old_user":
        req.messages[0]["content"] += " changed"
    elif change == "reasoning":
        req.messages[1]["reasoning_content"] += " changed"
    elif change == "content":
        req.messages[1]["content"] += " changed"
    elif change == "new_thinking_control":
        req.messages.append({"role": "user", "content": "<|think_off|> next"})
    elif change == "configuration":
        cfg["template_kwargs"]["preserve_thinking"] = False
    if change == "duplicate_keys":
        call["function"]["arguments"] = '{"path":"one.txt","path":"changed.txt"}'
    elif change == "nonfinite":
        call["function"]["arguments"] = '{"path":NaN}'
    else:
        call["function"]["arguments"] = json.dumps(
            args, ensure_ascii=False, separators=(",", ":")
        )
    _, admitted = encode_tool_roundtrip(tokenizer, cfg, req)
    assert admitted is None
    assert tokenizer.encoded_texts == []


def test_preserve_thinking_false_cannot_replay_when_new_user_changes_old_history(
    setup_runtime, monkeypatch
):
    _, tokenizer, cfg, req, _ = completed_tool_roundtrip(
        setup_runtime, monkeypatch, preserve_thinking=False
    )
    # Adding a real user query changes which old reasoning blocks Jinja keeps,
    # despite unchanged previous messages and unchanged template configuration.
    req.messages.append({"role": "user", "content": "Now inspect the third fixture"})
    _, admitted = encode_tool_roundtrip(tokenizer, cfg, req)
    assert admitted is None
    assert tokenizer.encoded_texts == []


@pytest.mark.parametrize(
    "left,right,equal",
    [
        ('{"x": "café", "a": [1, false]}', '{"x":"caf\\u00e9","a":[1,false]}', True),
        ('{"x":" a "}', '{"x":"a"}', False),
        ('{"x":[1,2]}', '{"x":[2,1]}', False),
        ('{"x":true}', '{"x":1}', False),
        ('{"x":1}', '{"x":1.0}', False),
        ('{"x":{"a":1,"b":2}}', '{"x":{"b":2,"a":1}}', False),
    ],
)
def test_tool_argument_identity_only_normalizes_unambiguous_json_format(
    left, right, equal
):
    make = lambda args: {
        "tool_calls": [
            {
                "id": "id",
                "type": "function",
                "function": {"name": "tool", "arguments": args},
            }
        ]
    }
    assert (
        runtime._output_digest(make(left)) == runtime._output_digest(make(right))
    ) is equal


@pytest.mark.parametrize(
    "arguments",
    [
        '{"x":1,"x":2}',
        '{"x":NaN}',
        '{"x":Infinity}',
        '{"x":1e999}',
        '{"x":',
        "[]",
        "null",
    ],
)
def test_tool_argument_identity_rejects_ambiguous_or_incomplete_json(arguments):
    with pytest.raises((TypeError, ValueError)):
        runtime._json_arguments(arguments)


class BackendMetadataTokenizer(AlternateSegmentationTokenizer):
    """Synthetic metadata for testing the production hash gate, never its pin."""

    def __init__(self, backend):
        self.backend_tokenizer = SimpleNamespace(to_str=lambda: json.dumps(backend))
        self.split_special_tokens = False
        self.clean_up_tokenization_spaces = False


def synthetic_backend_metadata():
    return {
        "model": {"type": "BPE", "merges": []},
        "normalizer": {"type": "NFC"},
        "decoder": {"type": "ByteLevel"},
        "added_tokens": [
            {
                "id": END,
                "content": "<|im_end|>",
                "special": True,
                "normalized": False,
                "lstrip": False,
                "rstrip": False,
            }
        ],
    }


@pytest.mark.parametrize(
    "fault",
    [
        "unknown_backend",
        "split_special",
        "cleanup",
        "normalizer",
        "decoder",
        "merges",
        "lstrip",
        "rstrip",
        "normalized",
    ],
)
def test_incremental_backend_gate_rejects_unqualified_semantics(monkeypatch, fault):
    backend = synthetic_backend_metadata()
    # Bind this isolated unit test to its synthetic fixture. This does not
    # replace the production model's authenticated tokenizer digest on disk.
    pin = hashlib.sha256(json.dumps(backend).encode()).hexdigest()
    monkeypatch.setattr(runtime, "INCREMENTAL_BACKEND_SHA256", pin)
    reference = BackendMetadataTokenizer(copy.deepcopy(backend))
    assert runtime._incremental_boundary_supported(reference) is True
    if fault == "unknown_backend":
        backend["unexpected"] = True
    elif fault == "normalizer":
        backend["normalizer"] = {"type": "NFKC"}
    elif fault == "decoder":
        backend["decoder"] = {"type": "WordPiece"}
    elif fault == "merges":
        backend["model"]["merges"] = [["x", "y"]]
    elif fault in {"lstrip", "rstrip", "normalized"}:
        backend["added_tokens"][0][fault] = True
    candidate = BackendMetadataTokenizer(backend)
    if fault == "split_special":
        candidate.split_special_tokens = True
    elif fault == "cleanup":
        candidate.clean_up_tokenization_spaces = True
    assert candidate.get_vocab() == reference.get_vocab()
    assert runtime._incremental_boundary_supported(candidate) is False


def test_tokenizer_identity_binds_backend_even_when_vocabulary_is_identical():
    first_backend = synthetic_backend_metadata()
    second_backend = copy.deepcopy(first_backend)
    second_backend["normalizer"] = {"type": "NFKC"}
    first = BackendMetadataTokenizer(first_backend)
    second = BackendMetadataTokenizer(second_backend)
    assert first.get_vocab() == second.get_vocab()
    assert (
        runtime._tokenizer_identity(first)[0] != runtime._tokenizer_identity(second)[0]
    )


def test_synthetic_backend_is_never_accepted_as_the_production_pin():
    tokenizer = BackendMetadataTokenizer(synthetic_backend_metadata())
    assert runtime._incremental_boundary_supported(tokenizer) is False
    assert (
        runtime._incremental_boundary_supported(AlternateSegmentationTokenizer())
        is False
    )


def test_mutating_cached_tokenizer_backend_invalidates_identity_and_fast_path(
    monkeypatch,
):
    backend = synthetic_backend_metadata()
    pin = hashlib.sha256(json.dumps(backend).encode()).hexdigest()
    monkeypatch.setattr(runtime, "INCREMENTAL_BACKEND_SHA256", pin)
    tokenizer = BackendMetadataTokenizer(backend)
    original_vocabulary = tokenizer.get_vocab()
    original_identity = runtime._tokenizer_identity(tokenizer)[0]
    assert runtime._incremental_boundary_supported(tokenizer) is True
    backend["normalizer"] = {"type": "NFKC"}
    assert tokenizer.get_vocab() == original_vocabulary
    assert runtime._incremental_boundary_supported(tokenizer) is False
    assert runtime._tokenizer_identity(tokenizer)[0] != original_identity
    assert runtime._incremental_boundary_supported(tokenizer) is False


@pytest.mark.parametrize("path", ["direct", "timeline"])
def test_incremental_ids_changed_during_input_processing_cannot_be_submitted(
    setup_runtime, path
):
    from qwen_r9700_lab import radiance_request_timeline as timeline

    tokenizer = SuffixOnlyTokenizer()
    prepare_suffix_only(tokenizer)
    req = next_request()
    value = begin(tokenizer, req=req)
    admitted = encode_incremental(value, tokenizer, req)
    runtime.rendered(value, Serving(), inputs(admitted))
    changed = [*admitted[:-1], ord("Q")]
    context = timeline._request.set(value)
    try:
        with pytest.raises(ValueError, match="incremental prompt"):
            if path == "direct":
                runtime.input_processor(value, changed)
            else:
                timeline.internal_id_bridge(
                    SimpleNamespace(
                        prompt_token_ids=changed,
                        external_req_id="external",
                        request_id="internal",
                    )
                )
        assert value["token_continuation"]["admitted"] is False
        assert value["token_continuation"]["failed"] is True
    finally:
        runtime.finish(value, False)
        timeline._request.reset(context)

"""Run the authenticated renderer hook without importing vLLM or a GPU backend."""

from __future__ import annotations

import ast
import gzip
import importlib.util
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def patcher():
    spec = importlib.util.spec_from_file_location(
        "token_continuation_renderer_patch",
        ROOT / "experiments/radiance-public/patch_chat_snapshot.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source():
    return gzip.decompress(
        (ROOT / "tests/fixtures/vllm_028_prefix_lineage_hf.py.gz").read_bytes()
    ).decode()


def renderer(source, timeline):
    candidates = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "safe_apply_chat_template"
        and not any(
            isinstance(decorator, ast.Name) and decorator.id == "overload"
            for decorator in node.decorator_list
        )
    ]
    assert len(candidates) == 1
    function = candidates[0]
    function.decorator_list = []

    def resolve_kwargs(**values):
        kwargs = values["chat_template_kwargs"]
        assert "_coherence_token_encode" not in kwargs
        return dict(kwargs)

    namespace = {
        "request_timeline": timeline,
        "resolve_chat_template": lambda _tokenizer, **values: values["chat_template"],
        "resolve_chat_template_kwargs": resolve_kwargs,
        "ChatTemplateResolutionError": ValueError,
        "Mapping": Mapping,
        "logger": SimpleNamespace(exception=lambda *_args: None),
    }
    exec(  # noqa: S102 - execute only the authenticated local CPU fixture
        compile(
            "from __future__ import annotations\n" + ast.unparse(function),
            "pinned_token_continuation_renderer",
            "exec",
        ),
        namespace,
    )
    return namespace["safe_apply_chat_template"]


class Tokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, **kwargs):
        assert "_coherence_token_encode" not in kwargs
        self.calls.append(kwargs)
        if kwargs.get("return_assistant_tokens_mask"):
            return {"input_ids": [90, 91], "assistant_masks": [0, 1]}
        return [90, 91] if kwargs["tokenize"] else "public rendered text"


class Timeline:
    def __init__(self):
        self.calls = []

    @staticmethod
    def prefix_template(template, kwargs):
        assert "_coherence_token_encode" not in kwargs
        return template

    def prefix_tokenize(
        self,
        tokenizer,
        conversation,
        tools,
        template,
        kwargs,
        tokenize,
        return_mask,
        *,
        continuation=None,
    ):
        self.calls.append((tokenize, return_mask, dict(kwargs)))
        return (
            continuation(tokenizer, conversation, tools, template, kwargs)
            if continuation is not None
            else None
        )


def call_renderer(patcher, source, timeline, tokenizer, **kwargs):
    function = renderer(patcher.prefix_renderer(source), timeline)
    return function(
        None,
        tokenizer,
        [{"role": "user", "content": "public synthetic question"}],
        chat_template=kwargs.pop("chat_template", "public synthetic template"),
        **kwargs,
    )


def test_renderer_patch_authenticates_and_is_idempotent(patcher, source):
    patched = patcher.prefix_renderer(source)
    ast.parse(patched)
    assert patcher.prefix_renderer(patched) == patched
    assert patched.count("request_timeline.prefix_tokenize(") == 1
    assert patched.index('kwargs.pop("_coherence_token_encode", None)') < patched.index(
        "resolved_kwargs = resolve_chat_template_kwargs("
    )


def test_previous_renderer_patch_upgrades_to_new_hook(patcher, source):
    previous = source
    for old, new in patcher.HF_PREFIX_V1_HOOKS:
        previous = previous.replace(old, new)
    assert patcher.prefix_renderer(previous) == patcher.prefix_renderer(source)


def test_unrecognized_renderer_source_is_rejected(patcher, source):
    with pytest.raises(ValueError, match="source differs"):
        patcher.prefix_renderer(source + "\n# unreviewed source change\n")


def test_accepted_continuation_skips_full_tokenization(patcher, source):
    timeline, tokenizer = Timeline(), Tokenizer()
    received = []

    def continuation(*args):
        received.append(args)
        return [11, 12, 13]

    result = call_renderer(
        patcher, source, timeline, tokenizer, _coherence_token_encode=continuation
    )
    assert result == [11, 12, 13]
    assert tokenizer.calls == []
    assert len(received) == 1
    assert received[0][-1] == {"return_dict": False}
    assert timeline.calls == [(True, False, {"return_dict": False})]


@pytest.mark.parametrize("has_callback", [True, False])
def test_declined_or_missing_continuation_runs_original_tokenizer(
    patcher, source, has_callback
):
    timeline, tokenizer = Timeline(), Tokenizer()
    kwargs = {"_coherence_token_encode": lambda *_args: None} if has_callback else {}
    assert call_renderer(patcher, source, timeline, tokenizer, **kwargs) == [90, 91]
    assert len(tokenizer.calls) == 1
    assert tokenizer.calls[0]["return_dict"] is False


def test_text_rendering_never_invokes_token_continuation(patcher, source):
    timeline, tokenizer = Timeline(), Tokenizer()
    result = call_renderer(
        patcher,
        source,
        timeline,
        tokenizer,
        tokenize=False,
        _coherence_token_encode=lambda *_args: pytest.fail("unexpected token hook"),
    )
    assert result == "public rendered text"
    assert timeline.calls == []
    assert len(tokenizer.calls) == 1


@pytest.mark.parametrize("template", ["plain template", "{% generation %}template"])
def test_assistant_masks_keep_original_renderer_path(patcher, source, template):
    timeline, tokenizer = Timeline(), Tokenizer()
    result = call_renderer(
        patcher,
        source,
        timeline,
        tokenizer,
        chat_template=template,
        return_assistant_tokens_mask=True,
        _coherence_token_encode=lambda *_args: pytest.fail("unexpected token hook"),
    )
    assert result == ([90, 91], [0, 1] if "generation" in template else None)
    assert timeline.calls == []
    assert len(tokenizer.calls) == 1

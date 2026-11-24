"""CPU-only qualification of the pinned generated-token continuation adapter.

Uses a local tokenizer and public synthetic messages; never loads model weights
or initializes a GPU runtime. The report contains hashes, counts and checks only.
"""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import importlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

SCHEMA = "urn:coherence:token-continuation-cpu-qualification:v1"
ANSWER = "Hello world"
PUBLIC_CONTEXT = "Public synthetic context: the reader is testing a greeting.\n"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_runtime(module_root):
    if module_root is not None:
        sys.path.insert(0, str(module_root.resolve()))
        # A frozen package may retain source filenames until its installer
        # assigns the qwen_radiance_* deployment names. Load those exact files.
        source = module_root / "radiance_token_continuation_runtime.py"
        if (
            source.is_file()
            and not (module_root / "qwen_radiance_token_continuation_runtime.py").exists()
        ):
            for name, filename in (
                ("qwen_radiance_token_continuation", "radiance_token_continuation.py"),
                ("qwen_radiance_token_continuation_runtime", source.name),
            ):
                spec = importlib.util.spec_from_file_location(name, module_root / filename)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
            return sys.modules["qwen_radiance_token_continuation_runtime"]
    try:
        return importlib.import_module("qwen_radiance_token_continuation_runtime")
    except ModuleNotFoundError as error:
        if error.name != "qwen_radiance_token_continuation_runtime":
            raise
        return importlib.import_module("qwen_r9700_lab.radiance_token_continuation_runtime")


def _load_timeline(module_root):
    if module_root is not None:
        source = module_root / "radiance_request_timeline.py"
        if source.is_file():
            name = "qwen_radiance_request_timeline"
            spec = importlib.util.spec_from_file_location(name, source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module
    try:
        return importlib.import_module("qwen_radiance_request_timeline")
    except ModuleNotFoundError as error:
        if error.name != "qwen_radiance_request_timeline":
            raise
        return importlib.import_module("qwen_r9700_lab.radiance_request_timeline")


def _decode(tokenizer, ids):
    return tokenizer.decode(
        list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


def _render(tokenizer, template, messages):
    return list(
        tokenizer.apply_chat_template(
            messages,
            chat_template=template,
            tokenize=True,
            return_dict=False,
            enable_thinking=False,
            preserve_thinking=True,
            add_generation_prompt=True,
        )
    )


def _render_in_executor(
    timeline,
    state,
    tokenizer,
    template,
    messages,
    expected_ids,
    checks,
    *,
    expected_suffix=None,
    label="renderer",
    native_pipeline=None,
    conversation=None,
    render_kwargs=None,
):
    """Match pinned make_async: executor threads receive no copied context."""
    kwargs = {
        "return_dict": False,
        "enable_thinking": False,
        "preserve_thinking": True,
        "add_generation_prompt": True,
    }
    kwargs.update(render_kwargs or {})
    conversation = messages if conversation is None else conversation
    encode_calls, tokenizing_calls = [], []
    original_encode = tokenizer.encode
    original_apply = tokenizer.apply_chat_template
    full_prompt_text = _decode(tokenizer, expected_ids)

    def encode_spy(text, *args, **values):
        if text == full_prompt_text:
            tokenizing_calls.append(True)
        encode_calls.append(
            {
                "characters": len(text) if isinstance(text, str) else 0,
                "exact_suffix": expected_suffix is not None and text == expected_suffix,
            }
        )
        return original_encode(text, *args, **values)

    @functools.wraps(original_apply)
    def apply_spy(*args, **values):
        if values.get("tokenize", True):
            tokenizing_calls.append(True)
        return original_apply(*args, **values)

    tokenizer.encode = encode_spy
    tokenizer.apply_chat_template = apply_spy
    token = timeline._request.set(state)
    try:
        timeline.prefix_render_params(conversation, kwargs)
        callback_present = callable(kwargs.get("_coherence_token_template"))
        encode_callback_present = callable(kwargs.get("_coherence_token_encode"))

        def worker():
            no_request_context = timeline._request.get() is None
            if native_pipeline is not None:
                callback_removed = []
                original_prefix_template = timeline.prefix_template

                def observe_template(text, values):
                    result = original_prefix_template(text, values)
                    callback_removed.append(
                        not any(
                            name in values
                            for name in ("_coherence_token_template", "_coherence_token_encode")
                        )
                    )
                    return result

                timeline.prefix_template = observe_template
                try:
                    rendered = native_pipeline["apply"](
                        Serving.model_config,
                        tokenizer,
                        conversation,
                        chat_template=template,
                        tokenize=False,
                        return_assistant_tokens_mask=False,
                        **kwargs,
                    )
                finally:
                    timeline.prefix_template = original_prefix_template
                prompt = native_pipeline["parse"](rendered)
                params = native_pipeline["params"]
                if "prompt_token_ids" not in prompt:
                    prompt = params.apply_pre_tokenization(tokenizer, prompt)
                    prompt["prompt_token_ids"] = tokenizer.encode(
                        prompt["prompt"], **params.get_encode_kwargs()
                    )
                prompt = params.apply_post_tokenization(tokenizer, prompt)
                return (
                    no_request_context,
                    bool(callback_removed) and all(callback_removed),
                    list(prompt["prompt_token_ids"]),
                )
            continuation = kwargs.pop("_coherence_token_encode", None)
            resolved = timeline.prefix_template(template, kwargs)
            callback_removed = not any(
                name in kwargs for name in ("_coherence_token_template", "_coherence_token_encode")
            )
            rendered = timeline.prefix_tokenize(
                tokenizer,
                conversation,
                None,
                resolved,
                kwargs,
                True,
                False,
                continuation=continuation,
            )
            if rendered is None:
                rendered = tokenizer.apply_chat_template(
                    conversation,
                    chat_template=resolved,
                    tokenize=True,
                    **kwargs,
                )
            return no_request_context, callback_removed, list(rendered)

        with ThreadPoolExecutor(max_workers=1) as executor:
            no_context, removed, rendered_ids = executor.submit(worker).result(timeout=60)
        values = {
            "renderer_thread_has_no_request_context": no_context,
            "request_bound_template_callback_passed": callback_present,
            "request_bound_tokenize_callback_passed": encode_callback_present,
            "template_callback_removed_before_rendering": removed,
            "worker_render_matches_expected_admitted_tokens": rendered_ids == expected_ids,
            "template_authenticated_across_executor_boundary": bool(
                state.get("token_continuation", {}).get("template_supported")
            ),
        }
        for name, passed in values.items():
            checks[name] = checks.get(name, True) and passed
        if expected_suffix is not None:
            checks[f"{label}_skips_full_prompt_tokenization"] = not tokenizing_calls
            checks[f"{label}_encodes_only_the_verified_suffix"] = (
                len(encode_calls) == 1 and encode_calls[0]["exact_suffix"]
            )
        return rendered_ids, {
            "encode_calls": len(encode_calls),
            "encode_input_characters": sum(row["characters"] for row in encode_calls),
            "full_prompt_tokenization_calls": len(tokenizing_calls),
            "continuation_reason": state.get("token_continuation_reason", "unknown"),
        }
    finally:
        timeline._request.reset(token)
        tokenizer.encode = original_encode
        tokenizer.apply_chat_template = original_apply


def _request(messages):
    return SimpleNamespace(
        model="public-token-continuation-cpu-probe",
        messages=messages,
        cache_salt="qwen-chat-cache-v1:public-token-continuation-cpu-probe:1",
        stream=True,
        n=1,
        use_beam_search=False,
        continue_final_message=False,
        add_generation_prompt=True,
        add_special_tokens=False,
        echo=False,
        tools=None,
        chat_template=None,
    )


def _state():
    return {"identities": {"http_request_id": "1" * 64}, "observed": set()}


def _inputs(request, ids):
    return [
        {
            "type": "token",
            "prompt_token_ids": list(ids),
            "cache_salt": request.cache_salt,
            "arrival_time": time.time(),
        }
    ]


class Serving:
    model_config = SimpleNamespace(max_model_len=253_792)

    @staticmethod
    def _extract_prompt_components(value):
        return SimpleNamespace(token_ids=value["prompt_token_ids"])


def qualify(
    tokenizer,
    template,
    runtime,
    *,
    prefix_tokens=0,
    timeline=None,
    native_pipeline=None,
):
    """Execute adapter boundaries against real tokenizer/template output."""
    if not 0 <= prefix_tokens <= 250_000:
        raise ValueError("unsupported prefix token budget")
    if timeline is None:
        timeline = _load_timeline(None)
    started = time.perf_counter()
    messages = [{"role": "user", "content": "Reply with a greeting"}]
    initial_ids = _render(tokenizer, template, messages)
    if prefix_tokens > len(initial_ids):
        per_repeat = max(1, len(tokenizer.encode(PUBLIC_CONTEXT, add_special_tokens=False)))
        repetitions = math.ceil((prefix_tokens - len(initial_ids)) / per_repeat)
        for _ in range(8):
            messages[0]["content"] = PUBLIC_CONTEXT * repetitions + "Reply with a greeting"
            initial_ids = _render(tokenizer, template, messages)
            if len(initial_ids) >= prefix_tokens:
                break
            repetitions += math.ceil((prefix_tokens - len(initial_ids)) / per_repeat)
        else:
            raise ValueError("synthetic prefix budget was not reached")
    if len(initial_ids) >= Serving.model_config.max_model_len - 100:
        raise ValueError("synthetic prompt exceeds admitted context")

    boundary = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if type(boundary) is not int or boundary not in tokenizer.all_special_ids:
        raise ValueError("chat boundary is not an authenticated special token")
    generated = [
        token
        for character in ANSWER
        for token in tokenizer.encode(character, add_special_tokens=False)
    ] + [boundary]
    canonical_answer = [*tokenizer.encode(ANSWER, add_special_tokens=False), boundary]
    checks = {
        "generated_segmentation_is_noncanonical": generated != canonical_answer,
        "generated_answer_decodes_identically": _decode(tokenizer, generated)
        == ANSWER + "<|im_end|>",
    }
    if not all(checks.values()):
        raise ValueError("tokenizer did not construct the required witness")
    raw_history = initial_ids + generated
    canonical_history = tokenizer.encode(_decode(tokenizer, raw_history), add_special_tokens=False)
    next_messages = [
        *copy.deepcopy(messages),
        {"role": "assistant", "content": ANSWER},
        {"role": "user", "content": "Reply with another greeting"},
    ]
    next_ids = _render(tokenizer, template, next_messages)
    checks["canonical_history_is_actual_next_prompt_prefix"] = (
        next_ids[: len(canonical_history)] == canonical_history
    )
    if not checks["canonical_history_is_actual_next_prompt_prefix"]:
        raise ValueError("actual template reconstruction did not extend the witness")
    expected = raw_history + next_ids[len(canonical_history) :]
    full_next_text = _decode(tokenizer, next_ids)
    raw_history_text = _decode(tokenizer, raw_history)
    if not full_next_text.startswith(raw_history_text):
        raise ValueError("rendering does not extend the generated-history text")
    suffix_text = full_next_text[len(raw_history_text) :]
    config = {
        "template_kwargs": {
            "enable_thinking": False,
            "preserve_thinking": True,
            "add_generation_prompt": True,
        },
        "request_template": None,
        "model": _request(messages).model,
        "tools": None,
        "tokenizer_class": type(tokenizer).__qualname__,
        "tokenizer_name": getattr(tokenizer, "name_or_path", None),
    }
    timings = {"fixture_prepare_ms": (time.perf_counter() - started) * 1000}
    tokenizer_work = {}
    old_get_ledger = runtime._get_ledger
    try:
        with tempfile.TemporaryDirectory(prefix="coherence-token-continuation-") as root:
            ledger_root = Path(root) / "journals"
            ledger = runtime.journal.TokenContinuationLedger(ledger_root)
            runtime._get_ledger = lambda: ledger
            first = _state()
            first_request = _request(messages)
            first_input = _inputs(first_request, initial_ids)
            before = copy.deepcopy(first_input)
            tick = time.perf_counter()
            runtime.begin(first, first_request, tokenizer, config)
            _, tokenizer_work["initial"] = _render_in_executor(
                timeline,
                first,
                tokenizer,
                template,
                messages,
                initial_ids,
                checks,
                native_pipeline=native_pipeline,
            )
            admitted = runtime.rendered(first, Serving(), first_input)
            checks["first_request_remains_canonical"] = admitted is first_input
            checks["first_original_input_unchanged"] = first_input == before
            runtime.input_processor(first, admitted[0]["prompt_token_ids"])
            runtime.output(
                first,
                SimpleNamespace(
                    outputs=[SimpleNamespace(index=0, token_ids=generated, finish_reason="stop")]
                ),
            )
            runtime.delivered(first, {"content": ANSWER})
            runtime.finish(first, True)
            checks["completed_answer_saved"] = (
                first.get("token_continuation_reason") == "journal_saved"
            )
            timings["first_answer_record_ms"] = (time.perf_counter() - tick) * 1000

            second_request = _request(next_messages)
            second = _state()
            tick = time.perf_counter()
            runtime.begin(second, second_request, tokenizer, config)
            second_ids, tokenizer_work["unchanged_continuation"] = _render_in_executor(
                timeline,
                second,
                tokenizer,
                template,
                next_messages,
                expected,
                checks,
                expected_suffix=suffix_text,
                label="unchanged_continuation",
                native_pipeline=native_pipeline,
            )
            second_input = _inputs(second_request, second_ids)
            before = copy.deepcopy(second_input)
            rewritten = runtime.rendered(second, Serving(), second_input)
            rewritten_ids = rewritten[0]["prompt_token_ids"]
            checks.update(
                {
                    "proposal_preserves_generated_prefix": rewritten_ids[: len(raw_history)]
                    == raw_history,
                    "proposal_appends_only_new_canonical_suffix": rewritten_ids == expected,
                    "proposal_preserves_full_decoded_prompt": _decode(tokenizer, rewritten_ids)
                    == _decode(tokenizer, next_ids),
                    "original_input_unchanged": second_input == before,
                    "proposal_was_applied_before_full_tokenization": second_ids == expected
                    and second_ids != next_ids,
                    "arrival_time_preserved": rewritten[0]["arrival_time"]
                    == before[0]["arrival_time"],
                }
            )
            runtime.input_processor(second, rewritten_ids)
            checks["rewritten_ids_confirmed_at_engine_admission"] = bool(
                second.get("token_continuation", {}).get("admitted")
            )
            timings["continuation_admission_ms"] = (time.perf_counter() - tick) * 1000
            runtime.finish(second, False)

            # A new ledger instance proves this is a persisted restart check,
            # rather than a second lookup into the first instance's RAM.
            restarted_ledger = runtime.journal.TokenContinuationLedger(ledger_root)
            runtime._get_ledger = lambda: restarted_ledger
            restarted = _state()
            tick = time.perf_counter()
            runtime.begin(restarted, second_request, tokenizer, config)
            restored_ids, tokenizer_work["persisted_restart"] = _render_in_executor(
                timeline,
                restarted,
                tokenizer,
                template,
                next_messages,
                expected,
                checks,
                expected_suffix=suffix_text,
                label="persisted_restart",
                native_pipeline=native_pipeline,
            )
            restarted_input = _inputs(second_request, restored_ids)
            restored = runtime.rendered(restarted, Serving(), restarted_input)
            checks["persisted_restart_preserves_same_token_history"] = (
                restored_ids == expected and restored[0]["prompt_token_ids"] == expected
            )
            timings["persisted_restart_admission_ms"] = (time.perf_counter() - tick) * 1000
            runtime.finish(restarted, False)

            edited_messages = copy.deepcopy(next_messages)
            edited_messages[0]["content"] += " Edited history."
            edited_request = _request(edited_messages)
            edited_input = _inputs(edited_request, _render(tokenizer, template, edited_messages))
            before = copy.deepcopy(edited_input)
            edited = _state()
            runtime.begin(edited, edited_request, tokenizer, config)
            _, tokenizer_work["edited_history"] = _render_in_executor(
                timeline,
                edited,
                tokenizer,
                template,
                edited_messages,
                edited_input[0]["prompt_token_ids"],
                checks,
                native_pipeline=native_pipeline,
            )
            fallback = runtime.rendered(edited, Serving(), edited_input)
            checks["edited_history_retains_canonical_fallback"] = (
                fallback is edited_input
                and fallback == before
                and tokenizer_work["edited_history"]["full_prompt_tokenization_calls"] == 1
            )
            runtime.finish(edited, False)

            # The raw history already fills this deliberately smaller journal.
            # A new suffix must fail closed, never be truncated to fit.
            limited_ledger = runtime.journal.TokenContinuationLedger(
                ledger_root, max_tokens=len(raw_history)
            )
            runtime._get_ledger = lambda: limited_ledger
            limited = _state()
            runtime.begin(limited, second_request, tokenizer, config)
            limited_ids, tokenizer_work["capacity_fallback"] = _render_in_executor(
                timeline,
                limited,
                tokenizer,
                template,
                next_messages,
                next_ids,
                checks,
                label="capacity_fallback",
                native_pipeline=native_pipeline,
            )
            limited_input = _inputs(second_request, limited_ids)
            before = copy.deepcopy(limited_input)
            fallback = runtime.rendered(limited, Serving(), limited_input)
            checks["capacity_overflow_keeps_full_canonical_prompt"] = (
                limited_ids == next_ids
                and fallback is limited_input
                and fallback == before
                and tokenizer_work["capacity_fallback"]["full_prompt_tokenization_calls"] == 1
            )
            runtime.finish(limited, False)
    finally:
        runtime._get_ledger = old_get_ledger
    return {
        "schema": SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "CPU tokenizer/template/adapter/journal; no inference or GPU state",
        "renderer_pipeline": (
            "vllm_safe_apply_text_then_parse_and_tokenize_params"
            if native_pipeline is not None
            else "direct_tokenized_template"
        ),
        "requested_prefix_tokens": prefix_tokens,
        "counts": {
            "initial_prompt_tokens": len(initial_ids),
            "generated_output_tokens": len(generated),
            "canonical_answer_tokens": len(canonical_answer),
            "raw_history_tokens": len(raw_history),
            "canonical_history_tokens": len(canonical_history),
            "next_canonical_prompt_tokens": len(next_ids),
            "next_admitted_prompt_tokens": len(expected),
            "new_suffix_tokens": len(next_ids) - len(canonical_history),
            "new_suffix_characters": len(suffix_text),
        },
        "checks": checks,
        "cpu_timings_ms": timings,
        "tokenizer_work": tokenizer_work,
        "tensor_backend_imported": any(
            name in sys.modules for name in ("torch", "tensorflow", "jax")
        ),
    }


def qualify_tool_roundtrip(
    tokenizer,
    template,
    runtime,
    *,
    prefix_tokens=0,
    timeline=None,
    native_pipeline=None,
):
    """Replay complete public XML calls through the real request renderer.

    Request messages retain OpenAI JSON argument strings. The renderer receives
    a separate, postprocessed conversation, as it does in the serving path.
    Neither body is included in the numeric qualification report.
    """
    if not 0 <= prefix_tokens <= 250_000:
        raise ValueError("unsupported prefix token budget")
    timeline = timeline or _load_timeline(None)
    checks, work, timings = {}, {}, {}
    options = {
        "return_dict": False,
        "enable_thinking": False,
        "preserve_thinking": True,
        "add_generation_prompt": True,
    }

    def conversation(messages):
        result = copy.deepcopy(messages)
        postprocess = (native_pipeline or {}).get("postprocess")
        if postprocess is not None:
            postprocess(result)
        else:
            # CPU direct-tokenizer mode has no vLLM preprocessing module. The
            # native qualification always supplies the actual installed helper.
            for message in result:
                for call in message.get("tool_calls", ()):
                    function = call["function"]
                    if isinstance(function["arguments"], str):
                        function["arguments"] = json.loads(function["arguments"])
        return result

    def render_text(messages, **overrides):
        return tokenizer.apply_chat_template(
            conversation(messages),
            chat_template=template,
            tokenize=False,
            **{**options, **overrides},
        )

    def render_ids(messages, **overrides):
        return tokenizer.encode(render_text(messages, **overrides), add_special_tokens=False)

    def generated_ids(text):
        # Valid alternative BPE segmentation is intentional. No generated token
        # is inferred from a later canonical rendering of the history.
        return [
            token
            for character in text
            for token in tokenizer.encode(character, add_special_tokens=False)
        ] + [boundary]

    prepared = time.perf_counter()
    messages = [
        {
            "role": "user",
            "content": "Run the two public synthetic tools, then explain their results.",
        }
    ]
    initial_ids = render_ids(messages)
    repetitions = 0
    per_repeat = max(1, len(tokenizer.encode(PUBLIC_CONTEXT, add_special_tokens=False)))
    for _ in range(8):
        if len(initial_ids) >= prefix_tokens:
            break
        repetitions += math.ceil((prefix_tokens - len(initial_ids)) / per_repeat)
        messages[0]["content"] = (
            PUBLIC_CONTEXT * repetitions
            + "Run the two public synthetic tools, then explain their results."
        )
        initial_ids = render_ids(messages)
    else:
        raise ValueError("synthetic tool prefix budget was not reached")
    if len(initial_ids) >= Serving.model_config.max_model_len - 4096:
        raise ValueError("synthetic tool prompt exceeds admitted context")
    boundary = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if type(boundary) is not int or boundary not in tokenizer.all_special_ids:
        raise ValueError("chat boundary is not an authenticated special token")
    argument_objects = [
        {"query": "alpha café 😀", "limit": 2},
        {"path": "public/result.json", "enabled": True},
    ]
    names = ["public_lookup", "public_write"]
    emitted_calls = [
        {
            "id": f"public-call-{index}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }
        for index, (name, arguments) in enumerate(zip(names, argument_objects, strict=True))
    ]
    restored_calls = copy.deepcopy(emitted_calls)
    for call in restored_calls:
        call["function"]["arguments"] = json.dumps(
            json.loads(call["function"]["arguments"]),
            ensure_ascii=True,
            separators=(",", ":"),
        )
    preamble = "Public synthetic tool plan."
    blocks = []
    for name, arguments in zip(names, argument_objects, strict=True):
        parameters = "".join(
            "<parameter="
            + key
            + ">\n"
            + (value if isinstance(value, str) else json.dumps(value))
            + "\n</parameter>\n"
            for key, value in arguments.items()
        )
        blocks.append(
            "<tool_call>\n<function=" + name + ">\n" + parameters + "</function>\n</tool_call>"
        )
    raw_answer = preamble + "\n\n" + "\n".join(blocks)
    generated = generated_ids(raw_answer)
    assistant = {"role": "assistant", "content": preamble, "tool_calls": restored_calls}
    first_history = [*messages, assistant]
    next_messages = [
        *first_history,
        {"role": "tool", "tool_call_id": "public-call-0", "content": "Synthetic lookup completed."},
        {"role": "tool", "tool_call_id": "public-call-1", "content": "Synthetic write completed."},
    ]
    canonical_prefix = render_text(first_history, add_generation_prompt=False).removesuffix("\n")
    next_text = render_text(next_messages)
    if not next_text.startswith(canonical_prefix):
        raise ValueError("canonical tool history did not extend its completed assistant")
    checks.update(
        {
            "canonical_template_retains_two_newlines_between_xml_calls": (
                "</tool_call>\n\n<tool_call>"
            )
            in canonical_prefix,
            "generated_xml_uses_one_newline_between_calls": "</tool_call>\n<tool_call>"
            in raw_answer,
            "generated_and_canonical_tool_layouts_differ": not _decode(
                tokenizer, initial_ids + generated
            ).startswith(canonical_prefix),
            "pi_json_roundtrip_changes_wire_arguments_only": all(
                before["function"]["arguments"] != after["function"]["arguments"]
                and json.loads(before["function"]["arguments"])
                == json.loads(after["function"]["arguments"])
                for before, after in zip(emitted_calls, restored_calls, strict=True)
            ),
        }
    )
    suffix = next_text[len(canonical_prefix) :]
    raw_history = initial_ids + generated
    expected_second = raw_history + tokenizer.encode(suffix, add_special_tokens=False)
    config = {
        "template_kwargs": {key: value for key, value in options.items() if key != "return_dict"},
        "request_template": None,
        "model": _request(messages).model,
        "tools": None,
        "tokenizer_class": type(tokenizer).__qualname__,
        "tokenizer_name": getattr(tokenizer, "name_or_path", None),
    }
    timings["fixture_prepare_ms"] = (time.perf_counter() - prepared) * 1000
    old_get_ledger = runtime._get_ledger

    def admit(
        request_messages,
        expected,
        label,
        *,
        expected_suffix=None,
        configuration=None,
        render_options=None,
    ):
        state, request = _state(), _request(request_messages)
        runtime.begin(state, request, tokenizer, configuration or config)
        ids, work[label] = _render_in_executor(
            timeline,
            state,
            tokenizer,
            template,
            request_messages,
            expected,
            checks,
            expected_suffix=expected_suffix,
            label=label,
            native_pipeline=native_pipeline,
            conversation=conversation(request_messages),
            render_kwargs=render_options,
        )
        inputs = _inputs(request, ids)
        before = copy.deepcopy(inputs)
        admitted = runtime.rendered(state, Serving(), inputs)
        actual = admitted[0]["prompt_token_ids"]
        runtime.input_processor(state, actual)
        checks[f"{label}_preserves_original_input_metadata"] = (
            inputs == before and admitted[0]["arrival_time"] == before[0]["arrival_time"]
        )
        return state, actual

    def complete(state, ids, *, content="", tool_calls=()):
        runtime.output(
            state,
            SimpleNamespace(
                outputs=[SimpleNamespace(index=0, token_ids=ids, finish_reason="stop")]
            ),
        )
        runtime.delivered(
            state,
            {
                "content": content,
                "tool_calls": [{"index": index, **call} for index, call in enumerate(tool_calls)],
            },
        )
        runtime.finish(state, True)
        return state.get("token_continuation_reason") == "journal_saved"

    try:
        with tempfile.TemporaryDirectory(prefix="coherence-public-tool-continuation-") as directory:
            root = Path(directory) / "journals"
            ledger = runtime.journal.TokenContinuationLedger(root)
            runtime._get_ledger = lambda: ledger
            tick = time.perf_counter()
            first, first_ids = admit(messages, initial_ids, "tool_initial")
            checks["tool_initial_canonical_input_unchanged"] = first_ids == initial_ids
            checks["completed_two_tool_answer_saved"] = complete(
                first, generated, content=preamble, tool_calls=emitted_calls
            )
            timings["first_tool_answer_record_ms"] = (time.perf_counter() - tick) * 1000
            tick = time.perf_counter()
            second, second_ids = admit(
                next_messages, expected_second, "tool_roundtrip", expected_suffix=suffix
            )
            checks["tool_roundtrip_preserves_exact_generated_prefix"] = (
                second_ids[: len(raw_history)] == raw_history
            )
            checks["tool_roundtrip_appends_only_new_tool_responses"] = second_ids == expected_second
            checks["tool_roundtrip_preserves_noncanonical_tool_layout"] = _decode(
                tokenizer, second_ids[: len(raw_history)]
            ) == _decode(tokenizer, raw_history)
            timings["tool_roundtrip_admission_ms"] = (time.perf_counter() - tick) * 1000
            second_answer = "The public synthetic tools completed successfully."
            second_generated = generated_ids(second_answer)
            tick = time.perf_counter()
            checks["second_completed_answer_saved"] = complete(
                second, second_generated, content=second_answer
            )
            timings["second_answer_record_ms"] = (time.perf_counter() - tick) * 1000
            raw_second_history = second_ids + second_generated
            third_history = [*next_messages, {"role": "assistant", "content": second_answer}]
            third_messages = [
                *third_history,
                {"role": "user", "content": "Continue the public synthetic task."},
            ]
            third_canonical_prefix = render_text(
                third_history, add_generation_prompt=False
            ).removesuffix("\n")
            third_text = render_text(third_messages)
            if not third_text.startswith(third_canonical_prefix):
                raise ValueError("third canonical prompt did not extend its assistant boundary")
            third_suffix = third_text[len(third_canonical_prefix) :]
            expected_third = raw_second_history + tokenizer.encode(
                third_suffix, add_special_tokens=False
            )
            tick = time.perf_counter()
            third, third_ids = admit(
                third_messages, expected_third, "three_turn_chain", expected_suffix=third_suffix
            )
            checks["three_turn_chain_keeps_earliest_tool_answer_ids"] = (
                third_ids[: len(raw_history)] == raw_history
            )
            checks["three_turn_chain_keeps_both_completed_answers"] = third_ids == expected_third
            runtime.finish(third, False)
            timings["three_turn_chain_admission_ms"] = (time.perf_counter() - tick) * 1000
            restarted_ledger = runtime.journal.TokenContinuationLedger(root)
            runtime._get_ledger = lambda: restarted_ledger
            restarted, restart_ids = admit(
                third_messages,
                expected_third,
                "tool_persisted_restart",
                expected_suffix=third_suffix,
            )
            checks["persisted_tool_restart_keeps_identical_token_history"] = (
                restart_ids == expected_third
            )
            runtime.finish(restarted, False)
            for changed in ("tool_arguments", "earlier_history", "latest_output", "configuration"):
                edited, changed_config, overrides = (
                    copy.deepcopy(third_messages),
                    copy.deepcopy(config),
                    {},
                )
                if changed == "tool_arguments":
                    edited[len(messages)]["tool_calls"][0]["function"]["arguments"] = json.dumps(
                        {"query": "different", "limit": 2}
                    )
                elif changed == "earlier_history":
                    edited[0]["content"] += " Altered historical instruction."
                elif changed == "latest_output":
                    edited[-2]["content"] += " Altered answer."
                else:
                    changed_config["template_kwargs"]["preserve_thinking"] = False
                    overrides["preserve_thinking"] = False
                canonical = render_ids(edited, **overrides)
                failed, actual = admit(
                    edited,
                    canonical,
                    f"reject_changed_{changed}",
                    configuration=changed_config,
                    render_options=overrides,
                )
                checks[f"changed_{changed}_keeps_canonical_fallback"] = (
                    actual == canonical
                    and work[f"reject_changed_{changed}"]["full_prompt_tokenization_calls"] > 0
                )
                runtime.finish(failed, False)
    finally:
        runtime._get_ledger = old_get_ledger
    return {
        "schema": SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": (
            "CPU native renderer/tokenizer/XML tool continuation; "
            "public synthetic data only; no inference or GPU state"
        ),
        "requested_prefix_tokens": prefix_tokens,
        "counts": {
            "initial_prompt_tokens": len(initial_ids),
            "first_generated_tokens": len(generated),
            "completed_tool_calls": len(emitted_calls),
            "raw_first_history_tokens": len(raw_history),
            "second_admitted_tokens": len(second_ids),
            "third_admitted_tokens": len(third_ids),
        },
        "checks": checks,
        "cpu_timings_ms": timings,
        "tokenizer_work": work,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--module-root", type=Path)
    parser.add_argument("--prefix-tokens", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tool-roundtrip",
        action="store_true",
        help="Also qualify public XML tool calls, Pi JSON round trips and chained histories",
    )
    args = parser.parse_args()
    # Load the actual serving tokenizer/renderer, including its copied tokenizer
    # pool. CPU Torch imports are allowed; GPU lazy initialization is checked.
    os.environ.update(
        {
            "USE_TORCH": "1",
            "USE_TF": "0",
            "USE_FLAX": "0",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    runtime = _load_runtime(args.module_root)
    timeline = _load_timeline(args.module_root)
    from vllm.entrypoints import chat_utils as native_chat_utils
    from vllm.renderers import hf as native_renderer
    from vllm.renderers.inputs.preprocess import parse_dec_only_prompt
    from vllm.renderers.params import TokenizeParams
    from vllm.tokenizers.hf import maybe_make_thread_pool
    from vllm.tokenizers.registry import get_tokenizer

    installed_renderer_source = Path(native_renderer.__file__).read_text()
    executed_renderer_source = installed_renderer_source
    patcher_source = None
    if args.module_root is not None:
        patcher_source = args.module_root / "patch_chat_snapshot.py"
        spec = importlib.util.spec_from_file_location(
            "coherence_cpu_renderer_patcher", patcher_source
        )
        patcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(patcher)
        executed_renderer_source = patcher.prefix_renderer(installed_renderer_source)
        # Change only this isolated CPU process's module, never installed files
        # or the model server. This qualifies the candidate on the pinned source.
        exec(  # Pinned-source patcher authenticates the complete source.
            compile(executed_renderer_source, native_renderer.__file__, "exec"),
            native_renderer.__dict__,
        )
    native_renderer.request_timeline = timeline
    tokenizer = maybe_make_thread_pool(
        copy.copy(
            get_tokenizer(
                str(args.tokenizer),
                local_files_only=True,
                trust_remote_code=False,
                runner_type="generate",
                tokenizer_mode="auto",
            )
        ),
        copies=2,
    )
    native_pipeline = {
        "apply": native_renderer.safe_apply_chat_template,
        "parse": parse_dec_only_prompt,
        "postprocess": native_chat_utils._postprocess_messages,
        "params": TokenizeParams(
            max_total_tokens=Serving.model_config.max_model_len,
            max_output_tokens=64,
            add_special_tokens=False,
        ),
    }
    template = args.template.read_text()
    report = qualify(
        tokenizer,
        template,
        runtime,
        prefix_tokens=args.prefix_tokens,
        timeline=timeline,
        native_pipeline=native_pipeline,
    )
    if args.tool_roundtrip:
        report["tool_roundtrip"] = qualify_tool_roundtrip(
            tokenizer,
            template,
            runtime,
            prefix_tokens=args.prefix_tokens,
            timeline=timeline,
            native_pipeline=native_pipeline,
        )
        report["checks"]["tool_roundtrip_qualification_passed"] = (
            report["tool_roundtrip"]["status"] == "PASS"
        )
        if not report["checks"]["tool_roundtrip_qualification_passed"]:
            report["status"] = "FAIL"
    report["source_sha256"] = {
        "driver": _sha(__file__),
        "runtime": _sha(runtime.__file__),
        "ledger": _sha(runtime.journal.__file__),
        "timeline": _sha(timeline.__file__),
        "native_renderer": _sha(native_renderer.__file__),
        "native_chat_utils": _sha(native_chat_utils.__file__),
        "executed_native_renderer": hashlib.sha256(executed_renderer_source.encode()).hexdigest(),
    }
    if patcher_source is not None:
        report["source_sha256"]["renderer_patcher"] = _sha(patcher_source)
    report["template_sha256"] = _sha(args.template)
    report["tokenizer_identity_sha256"] = runtime._tokenizer_identity(tokenizer)[0]
    report["tokenizer_file_sha256"] = {
        name: _sha(args.tokenizer / name)
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        )
        if (args.tokenizer / name).is_file()
    }
    # Some pinned Transformers builds import CPU Torch despite USE_TORCH=0.
    # Importing it is not GPU execution. Retain that fact and check the actual
    # lazy-initialization flag rather than reporting a false qualification loss.
    torch_module = sys.modules.get("torch")
    cuda_module = getattr(torch_module, "cuda", None)
    report["gpu_runtime_initialized"] = bool(
        cuda_module is not None and cuda_module.is_initialized()
    )
    report["checks"]["gpu_runtime_not_initialized"] = not report["gpu_runtime_initialized"]
    if report["gpu_runtime_initialized"]:
        report["status"] = "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "checks": len(report["checks"])}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

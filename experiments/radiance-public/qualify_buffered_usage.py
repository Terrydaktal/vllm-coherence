"""Exercise the installed vLLM parser/stream generator with synthetic engine output.

Run in the pinned backend's Python environment. No HTTP/model requests, GPU work,
chat transcripts, or file edits; output contains only scalar qualification data.
"""

import ast
import asyncio
import inspect
import json
import os
from itertools import product
from types import SimpleNamespace

from patch_chat_snapshot import (
    BUFFERED_USAGE_NEW,
    BUFFERED_USAGE_OLD,
    stream_buffered_tool_usage,
)
from transformers import AutoTokenizer
from vllm.entrypoints.openai.chat_completion import serving
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.qwen3 import Qwen3Parser


def generator_from_source(source):
    tree = ast.parse(source)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "OpenAIServingChat"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "chat_completion_stream_generator"
    )
    namespace = dict(vars(serving))
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), "<qualified-chat-stream>", "exec"),
        namespace,
    )
    return namespace[method.name]


async def exercise(generator, tokenizer, continuous, raw_ids, reasoning, choices):
    args = {"path": "synthetic.txt", "edits": [{"oldText": "a" * 5000, "newText": "b" * 5000}]}
    tools = [
        {
            "type": "function",
            "function": {
                "name": "edit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "edits": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "oldText": {"type": "string"},
                                    "newText": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
        }
    ]
    request = ChatCompletionRequest(
        model="synthetic",
        messages=[{"role": "user", "content": "synthetic"}],
        tools=tools,
        tool_choice="auto",
        stream=True,
        n=choices,
        return_token_ids=raw_ids,
        include_reasoning=reasoning,
        stream_options={"include_usage": True, "continuous_usage_stats": continuous},
    )
    server = object.__new__(serving.OpenAIServingChat)
    for key, value in {
        "parser_cls": Qwen3Parser,
        "model_config": None,
        "response_role": "assistant",
        "enable_force_include_usage": False,
        "enable_log_outputs": False,
        "enable_log_deltas": False,
        "request_logger": None,
        "enable_prompt_tokens_details": True,
        "enable_per_request_metrics": False,
        "system_fingerprint": "synthetic",
    }.items():
        setattr(server, key, value)
    header = (
        "synthetic thinking</think><tool_call>\n<function=edit>\n"
        "<parameter=path>synthetic.txt</parameter>\n<parameter=edits>"
    )
    body = json.dumps(args["edits"])
    parts = [("header", header)] + [("body", body[i : i + 128]) for i in range(0, len(body), 128)]
    parts += [("close", "</parameter>\n</function>\n</tool_call>")]
    stage = ""
    expected = 0

    async def engine_output():
        nonlocal stage, expected
        for stage, part in parts:
            # Use real token IDs for synthetic text, including the parser's special
            # delimiters. Only the CPU tokenizer is used; there is no inference.
            token_ids = tokenizer.encode(part, add_special_tokens=False)
            expected += len(token_ids)
            yield SimpleNamespace(
                prompt_token_ids=tokenizer.encode(
                    "<|im_start|>assistant\n<think>\n", add_special_tokens=False
                ),
                encoder_prompt_token_ids=None,
                num_cached_tokens=0,
                num_cache_creation_tokens=0,
                metrics=None,
                outputs=[
                    SimpleNamespace(
                        index=i,
                        text=part,
                        token_ids=token_ids,
                        logprobs=None,
                        finish_reason="stop" if stage == "close" else None,
                        stop_reason=None,
                    )
                    for i in range(choices)
                ],
            )

    metadata = SimpleNamespace(final_usage_info=None)
    arguments = [""] * choices
    finish_reasons = []
    body_usage = body_ids = body_argument_chars = done = 0
    final_usage = None
    async for frame in generator(
        server,
        request,
        engine_output(),
        "synthetic-request",
        "synthetic",
        [],
        tokenizer,
        metadata,
    ):
        payload = frame.removeprefix("data: ").strip()
        if payload == "[DONE]":
            done += 1
            continue
        chunk = json.loads(payload)
        assert "error" not in chunk, chunk
        if not chunk["choices"]:
            final_usage = chunk["usage"]["completion_tokens"]
        for choice in chunk["choices"]:
            index = choice["index"]
            assert 0 <= index < choices
            delta = choice["delta"]
            if not reasoning:
                assert not delta.get("reasoning")
                assert not choice.get("token_ids")
            if not raw_ids:
                assert not choice.get("token_ids")
            if choice["finish_reason"]:
                finish_reasons.append(choice["finish_reason"])
            for call in delta.get("tool_calls") or []:
                text = call.get("function", {}).get("arguments") or ""
                arguments[index] += text
                if stage == "body":
                    body_argument_chars += len(text)
            if stage == "body":
                body_ids += len(choice.get("token_ids") or [])
                if chunk.get("usage"):
                    assert chunk["usage"]["completion_tokens"] == expected
                    body_usage += 1
    assert done == 1 and finish_reasons == ["tool_calls"] * choices, {
        "done": done,
        "finish_reasons": finish_reasons,
        "argument_lengths": [len(value) for value in arguments],
        "case": [continuous, raw_ids, reasoning, choices],
    }
    assert final_usage == metadata.final_usage_info.completion_tokens == expected * choices
    assert all(json.loads(value) == args for value in arguments)
    assert body_argument_chars == 0, "the real parser should still buffer the synthetic edit array"
    return {
        "body_usage_frames": body_usage,
        "body_token_ids": body_ids,
        "total_tokens": final_usage,
    }


async def main():
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ.get("QWEN_USAGE_TEST_TOKENIZER", "/models/Qwen3.8-27B-Uncensored-MXFP4-awq"),
        local_files_only=True,
    )
    source = inspect.getsource(serving).replace(BUFFERED_USAGE_NEW, BUFFERED_USAGE_OLD)
    before = generator_from_source(source)
    after = generator_from_source(stream_buffered_tool_usage(source))
    results = []
    for continuous, raw_ids, reasoning, choices in product(
        (False, True), (False, True), (False, True), (1, 2)
    ):
        original = await exercise(before, tokenizer, continuous, raw_ids, reasoning, choices)
        fixed = await exercise(after, tokenizer, continuous, raw_ids, reasoning, choices)
        assert fixed["total_tokens"] == original["total_tokens"]
        assert fixed["body_token_ids"] == original["body_token_ids"]
        if continuous:
            assert fixed["body_usage_frames"] == 79 * choices
        else:
            assert fixed == original
        results.append(
            {
                "continuous": continuous,
                "raw_ids": raw_ids,
                "reasoning": reasoning,
                "choices": choices,
                "before": original,
                "after": fixed,
            }
        )
    print(
        json.dumps({"synthetic": True, "passed_cases": len(results), "results": results}, indent=2)
    )


if __name__ == "__main__":
    asyncio.run(main())

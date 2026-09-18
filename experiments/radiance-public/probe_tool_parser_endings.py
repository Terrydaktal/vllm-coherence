"""CPU-only diagnosis against the installed server's parser; no tool execution.

Run with the serving environment's Python, passing the local tokenizer directory.
Uses only synthetic number-lookup text. Does not modify the server or tokenizer.
"""

import hashlib
import inspect
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.qwen3 import Qwen3Parser


tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
tool = {
    "type": "function",
    "function": {
        "name": "lookup_number",
        "description": "Harmless fixture, never executed.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
}
request = ChatCompletionRequest(
    model="fixture",
    messages=[{"role": "user", "content": "Look up beta."}],
    tools=[tool],
    tool_choice="auto",
    stream=True,
)
xml = (
    "<tool_call>\n<function=lookup_number>\n<parameter=key>\n"
    "beta\n</parameter>\n</function>\n</tool_call>"
)
fixtures = {
    "valid_xml": "I need beta.\n</think>\n\n" + xml,
    "json_in_xml_wrapper": (
        "I need beta.\n</think>\n\n<tool_call>\n"
        '{"name":"lookup_number","arguments":{"key":"beta"}}\n</tool_call>'
    ),
    "announcement_then_json": (
        "I need beta.\n</think>\n\nI will look up beta now.\n<tool_call>\n"
        '{"name":"lookup_number","arguments":{"key":"beta"}}\n</tool_call>'
    ),
    "unfinished_tool_name": "I need beta.\n</think>\n\n<tool_call>\n<function=",
    "announcement_without_call": "I need beta.\n</think>\n\nI will look up beta now.",
    "quoted_marker_then_answer": (
        'The token "<tool_call>" is syntax, not an action.\n'
        "There is no lookup needed.\n</think>\n\nThe answer is 12."
    ),
    "quoted_marker_then_xml": (
        'The token "<tool_call>" is syntax, not an action.\n'
        "I should now look up beta.\n</think>\n\n" + xml
    ),
}
parser_path = Path(inspect.getfile(Qwen3Parser))
print(json.dumps({
    "parser_path": str(parser_path),
    "parser_sha256": hashlib.sha256(parser_path.read_bytes()).hexdigest(),
    "gpu_inference": False,
    "tool_execution": False,
}))
for name, text in fixtures.items():
    ids = tokenizer.encode(text, add_special_tokens=False)
    for width in (1, 8, len(ids)):
        parser = Qwen3Parser(tokenizer, tools=request.tools)
        deltas = []
        previous = ""
        for offset in range(0, len(ids), width):
            end = min(offset + width, len(ids))
            current = tokenizer.decode(ids[:end], skip_special_tokens=False)
            assert current.startswith(previous), "fixture tokenizer prefix changed"
            delta = parser.parse_delta(
                current[len(previous):], ids[offset:end], request,
                finished=end == len(ids),
            )
            if delta is not None:
                deltas.append(delta.model_dump(exclude_none=True))
            previous = current
        calls = [call for d in deltas for call in d.get("tool_calls", [])]
        print(json.dumps({
            "fixture": name,
            "chunk_tokens": width,
            "input_chars": len(text),
            "reasoning": "".join(d.get("reasoning", "") for d in deltas),
            "content": "".join(d.get("content", "") for d in deltas),
            "named_calls": [c for c in calls if c.get("function", {}).get("name")],
            "tool_delta_count": len(calls),
        }))

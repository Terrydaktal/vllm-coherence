"""CPU-only characterization of installed strict auto-tool grammar.

Pass the checkpoint tokenizer directory. Does not enable flags, change model
configuration, contact any service, generate tokens or execute tools.
"""

import hashlib
import inspect
import json
import sys
from pathlib import Path

import xgrammar as xgr
from transformers import AutoTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag


tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=1)
function = {"name": "lookup_number", "parameters": {
    "type": "object", "properties": {"key": {"type": "string"}},
    "required": ["key"], "additionalProperties": False,
}}
non_strict = ChatCompletionToolsParam(type="function", function=function)
strict = ChatCompletionToolsParam(type="function", function={**function, "strict": True})
assert get_model_structural_tag("qwen_3_coder", [non_strict], "auto", False) is None
registry = Path(inspect.getfile(get_model_structural_tag))
print(json.dumps({"registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
                  "gpu_inference": False, "tool_execution": False,
                  "auto_without_strict_has_no_grammar": True}))
xml = "<tool_call>\n<function=lookup_number>\n<parameter=key>beta</parameter>\n</function>\n</tool_call>"
fixtures = {
    "valid_xml": xml,
    "json_wrong_contract": '<tool_call>{"name":"lookup_number","arguments":{"key":"beta"}}</tool_call>',
    "only_opening_marker": "<tool_call>",
    "unfinished_function": "<tool_call>\n<function=",
    "unregistered_function": xml.replace("lookup_number", "unknown"),
    "answer_only": "The answer is 12.",
    "announcement_only": "I will look up beta now:",
    "literal_marker_in_reasoning": 'The token "<tool_call>" is syntax.\n</think>\n\nThe answer is 12.',
    "fenced_example_in_answer": "```xml\n" + xml.replace("lookup_number", "example") + "\n```\n12",
}
for reasoning in (False, True):
    tag = get_model_structural_tag("qwen_3_coder", [strict], "auto", reasoning)
    grammar = compiler.compile_structural_tag(tag)
    for name, text in fixtures.items():
        if reasoning and name != "literal_marker_in_reasoning":
            text = "Number lookup reasoning.\n</think>\n\n" + text
        matcher = xgr.GrammarMatcher(grammar)
        accepted = matcher.accept_string(text)
        stop_accepted = matcher.accept_token(tokenizer.eos_token_id) if accepted else False
        print(json.dumps({"fixture": name, "reasoning": reasoning,
                          "text_accepted": accepted, "eos_accepted": stop_accepted}))

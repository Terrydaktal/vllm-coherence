"""CPU-only qualification of constraints for an already selected tool phase.

This is NOT a global tool_choice=required deployment. Run in the installed
serving Python with its tokenizer path. Synthetic logits only, no model or
GPU inference, no requests, no tool execution, no runtime modifications.
"""

import hashlib
import inspect
import json
import sys
from pathlib import Path

import torch
import xgrammar as xgr
from transformers import AutoTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag


tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
info = xgr.TokenizerInfo.from_huggingface(tokenizer)
compiler = xgr.GrammarCompiler(info, max_threads=1)
tool = ChatCompletionToolsParam(type="function", function={
    "name": "lookup_number", "strict": True,
    "parameters": {"type": "object", "properties": {"key": {"type": "string"}},
                   "required": ["key"], "additionalProperties": False},
})
tag = get_model_structural_tag("qwen_3_coder", [tool], "required", False)
grammar = compiler.compile_structural_tag(tag)
xml = "<tool_call>\n<function=lookup_number>\n<parameter=key>beta</parameter>\n</function>\n</tool_call>"
fixtures = [
    ("opening", "<tool_call>", True, False),
    ("function_prefix", "<tool_call>\n<function=", True, False),
    ("json_in_xml", '<tool_call>{"name":"lookup_number"}', False, False),
    ("unknown_name", xml.replace("lookup_number", "unknown"), False, False),
    ("missing_argument", xml.replace("<parameter=key>beta</parameter>", ""), False, False),
    ("complete", xml, True, True),
]
outcomes = []
for name, text, expected_text, expected_eos in fixtures:
    matcher = xgr.GrammarMatcher(grammar)
    accepted = matcher.accept_string(text)
    stopped = matcher.accept_token(tokenizer.eos_token_id) if accepted else False
    assert (accepted, stopped) == (expected_text, expected_eos), (name, accepted, stopped)
    outcomes.append({"fixture": name, "text_accepted": accepted, "eos_accepted": stopped})

# Give EOS the highest synthetic logit at EVERY incomplete token boundary.
# The real installed mask must remove it so decoding can continue normally.
ids = tokenizer.encode(xml, add_special_tokens=False)
matcher = xgr.GrammarMatcher(grammar)
mask = xgr.allocate_token_bitmask(1, info.vocab_size)
for index, token_id in enumerate(ids):
    matcher.fill_next_token_bitmask(mask)
    logits = torch.full((1, info.vocab_size), -1000.0, device="cpu")
    logits[0, token_id] = 0.0
    logits[0, tokenizer.eos_token_id] = 1000.0
    xgr.apply_token_bitmask_inplace(logits, mask, backend="cpu")
    assert not torch.isfinite(logits[0, tokenizer.eos_token_id]), index
    assert int(logits.argmax()) == token_id, index
    assert matcher.accept_token(token_id), index
matcher.fill_next_token_bitmask(mask)
word = int(mask[0, tokenizer.eos_token_id // 32])
assert (word >> (tokenizer.eos_token_id % 32)) & 1
assert matcher.accept_token(tokenizer.eos_token_id)

# Bounded grammar-state rollback checks, NOT GPU KV/GDN/allocator validation.
rollback_checks = 0
for start in range(0, len(ids), 8):
    provisional = ids[start:start + 8]
    for committed in range(len(provisional) + 1):
        candidate = xgr.GrammarMatcher(grammar)
        reference = xgr.GrammarMatcher(grammar)
        for token_id in ids[:start + len(provisional)]:
            assert candidate.accept_token(token_id)
        suffix = len(provisional) - committed
        if suffix:
            candidate.rollback(suffix)
        for token_id in ids[:start + committed]:
            assert reference.accept_token(token_id)
        candidate_mask = xgr.allocate_token_bitmask(1, info.vocab_size)
        reference_mask = xgr.allocate_token_bitmask(1, info.vocab_size)
        candidate.fill_next_token_bitmask(candidate_mask)
        reference.fill_next_token_bitmask(reference_mask)
        assert torch.equal(candidate_mask, reference_mask), (start, committed)
        rollback_checks += 1

source = Path(inspect.getfile(get_model_structural_tag))
print(json.dumps({
    "registry_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "grammar_sha256": hashlib.sha256(tag.model_dump_json().encode()).hexdigest(),
    "gpu_inference": False, "tool_execution": False, "synthetic_logits_only": True,
    "outcomes": outcomes, "premature_eos_masked_at_token_boundaries": len(ids),
    "grammar_rollback_checks": rollback_checks,
    "scope": "already selected tool phase; not enabled for reasoning or final answers",
}))

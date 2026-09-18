"""Create opaque diagnostic inputs, without modifying any live conversation.

All payloads and token IDs remain in the supplied protected RAM directory.
Changing historical thinking is an experiment, never a serving-time filter.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import random
import urllib.request
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def without_thinking(payload):
    result = copy.deepcopy(payload)
    removed = 0
    messages = 0
    for before, message in zip(payload["messages"], result["messages"], strict=True):
        if message.get("role") != "assistant":
            assert before == message
            continue
        changed = False
        for key in ("reasoning", "reasoning_content", "thinking"):
            value = message.pop(key, None)
            if isinstance(value, str):
                removed += len(value)
                changed |= bool(value)
        messages += changed
        # Preserve user-visible answers and all tool calls/arguments exactly.
        assert message == {k: v for k, v in before.items()
                           if k not in ("reasoning", "reasoning_content", "thinking")}
    return result, {"removed_thinking_characters": removed,
                    "changed_assistant_messages": messages}


def without_looping_thinking(payload, tokenizer):
    """Offline causal control: remove whole failed scratchpads, not just copies.

    This diagnostic is never imported by the provider or the serving runtime.
    It leaves ordinary reasoning, visible answers and the complete tool ledger
    intact. It does not authorize rewriting the original conversation.
    """
    from diagnose_loop_replay import repetition

    result = copy.deepcopy(payload)
    removed = messages = retained = 0
    for before, message in zip(payload["messages"], result["messages"], strict=True):
        if message.get("role") != "assistant":
            assert before == message
            continue
        changed = False
        for key in ("reasoning", "reasoning_content", "thinking"):
            value = message.get(key)
            if not isinstance(value, str) or not value:
                continue
            ids = tokenizer(value, add_special_tokens=False)["input_ids"]
            if repetition(value, ids)["loop_candidate"]:
                message.pop(key)
                removed += len(value)
                changed = True
            else:
                retained += len(value)
        messages += changed
        assert {k: v for k, v in message.items()
                if k not in ("reasoning", "reasoning_content", "thinking")} == {
                    k: v for k, v in before.items()
                    if k not in ("reasoning", "reasoning_content", "thinking")}
    return result, {"removed_thinking_characters": removed,
                    "retained_thinking_characters": retained,
                    "changed_assistant_messages": messages}


def normalized_messages(payload):
    messages = copy.deepcopy(payload["messages"])
    for message in messages:
        if message.get("role") != "assistant":
            continue
        if "reasoning" in message:
            message["reasoning_content"] = message["reasoning"]
        for call in message.get("tool_calls", []):
            function = call.get("function", call)
            if isinstance(function.get("arguments"), str):
                function["arguments"] = json.loads(function["arguments"])
    return messages


def scrambled_thinking(payload, tokenizer, template, reference):
    """Length-matched diagnostic control; leave every non-thinking token intact.

    Marking is used only to locate the historical scratchpad spans. The markers
    never enter the model input. Shuffling their token IDs tests ordered-history
    conditioning separately from a shorter prompt. No live request uses this.
    """
    marked = copy.deepcopy(payload)
    markers = []
    for index, message in enumerate(marked["messages"]):
        if message.get("role") != "assistant":
            continue
        key = next((key for key in ("reasoning", "reasoning_content", "thinking")
                    if isinstance(message.get(key), str) and message[key].strip()), None)
        if key is None:
            continue
        begin, end = f"QWEN_DIAGNOSTIC_SPAN_{index}_BEGIN", f"QWEN_DIAGNOSTIC_SPAN_{index}_END"
        markers.append((begin, end))
        message[key] = begin + message[key].strip() + end

    def render(value):
        return tokenizer.apply_chat_template(normalized_messages(value), tools=value.get("tools"),
            chat_template=template.read_text(), add_generation_prompt=True, tokenize=False,
            **value.get("chat_template_kwargs", {}))

    original_text = render(payload)
    marked_text = render(marked)
    pieces, spans, cursor, size = [], [], 0, 0
    for begin, end in markers:
        assert begin not in original_text and end not in original_text
        assert marked_text.count(begin) == marked_text.count(end) == 1
        left = marked_text.index(begin, cursor)
        right = marked_text.index(end, left)
        prefix = marked_text[cursor:left]
        body = marked_text[left + len(begin):right]
        pieces += [prefix, body]
        size += len(prefix)
        spans.append((size, size + len(body)))
        size += len(body)
        cursor = right + len(end)
    pieces.append(marked_text[cursor:])
    assert "".join(pieces) == original_text
    encoded = tokenizer(original_text, add_special_tokens=False, return_offsets_mapping=True)
    assert encoded["input_ids"] == reference
    ids = list(reference)
    changed_positions = set()
    rng = random.Random(1789373261)
    for low, high in spans:
        positions = [i for i, (start, end) in enumerate(encoded["offset_mapping"])
                     if low <= start < end <= high]
        values = [ids[i] for i in positions]
        rng.shuffle(values)
        for i, value in zip(positions, values, strict=True):
            ids[i] = value
        changed_positions.update(positions)
    assert len(ids) == len(reference)
    assert all(value == reference[i] for i, value in enumerate(ids) if i not in changed_positions)
    return ids, {"historical_thinking_spans": len(spans),
                 "shuffled_token_positions": len(changed_positions),
                 "changed_token_positions": sum(a != b for a, b in zip(ids, reference, strict=True)),
                 "nonthinking_tokens_preserved": True, "length_matched": True}


def prepare(private, endpoint, labels, official=None, tokenizer_path=None, base_template=None,
            scramble=False, failed_history=False):
    assert str(private) == "/private-fixtures" or re.fullmatch(r"/dev/shm/qwen-private-replay-[a-z0-9_]+", str(private))
    assert private.stat().st_mode & 0o077 == 0
    tokenizer = None
    if tokenizer_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    def offline_tokens(payload, template):
        return tokenizer.apply_chat_template(
            normalized_messages(payload), tools=payload.get("tools"),
            chat_template=template.read_text(), add_generation_prompt=True,
            return_dict=False, **payload.get("chat_template_kwargs", {}))

    report = []
    for label in labels:
        assert re.fullmatch(r"[a-z_]+", label)
        original = json.loads((private / f"payload-{label}.json").read_text())
        if tokenizer:
            reference = json.loads((private / f"fixture-{label}.json").read_text())["tokens"]
            # Establish byte-for-byte agreement with vLLM's tokenization path
            # before using offline normalization for any counterfactual.
            assert offline_tokens(original, base_template) == reference
        clean, metadata = without_thinking(original)
        variants = [(f"{label}_without_history_thinking", clean, None, metadata)]
        if official:
            variants += [(f"{label}_official_template", original, official, {}),
                         (f"{label}_official_without_thinking", clean, official, metadata)]
        if scramble:
            # Existing inputs may be in use by a replay worker. Do not rewrite
            # them while producing a separate length-matched control.
            variants = []
        if failed_history:
            assert tokenizer is not None and not scramble
            cleaned, stats = without_looping_thinking(original, tokenizer)
            variants = [(f"{label}_without_looping_thinking", cleaned, None, stats)]
        for name, payload, template, extra in variants:
            request = {key: payload[key] for key in ("model", "messages", "tools", "chat_template_kwargs")
                       if key in payload}
            request.update(add_generation_prompt=True, return_token_strs=False)
            if template:
                request["chat_template"] = template.read_text()
            if tokenizer:
                tokens = offline_tokens(payload, template or base_template)
            else:
                data = json.dumps(request).encode()
                req = urllib.request.Request(endpoint + "/tokenize", data=data,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=60) as response:
                    tokenized = json.load(response)
                tokens = tokenized["tokens"]
            value = {"label": name, "tokens": tokens, "sha256": digest(tokens)}
            (private / f"fixture-{name}.json").write_text(json.dumps(value))
            saved_payload = dict(payload)
            if template:
                saved_payload["chat_template"] = template.read_text()
            (private / f"payload-{name}.json").write_text(json.dumps(saved_payload))
            report.append({"label": name, "tokens": len(tokens), "sha256": value["sha256"],
                           "official_template_sha256": hashlib.sha256(template.read_bytes()).hexdigest() if template else None,
                           **extra})
        if scramble:
            assert tokenizer is not None
            ids, metadata = scrambled_thinking(original, tokenizer, base_template, reference)
            name = f"{label}_scrambled_history"
            sha = digest(ids)
            (private / f"fixture-{name}.json").write_text(json.dumps({"label": name, "tokens": ids, "sha256": sha}))
            # This companion supplies only the original tool schemas to the
            # validator. The actual diagnostic input is the token-array fixture.
            (private / f"payload-{name}.json").write_text(json.dumps(dict(original, token_array_counterfactual=True)))
            report.append({"label": name, "tokens": len(ids), "sha256": sha, **metadata})
    report_name = ("scrambled-thinking-report.json" if scramble else
                   "failed-thinking-report.json" if failed_history else
                   "thinking-template-counterfactual-report.json")
    (private / report_name).write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18080")
    parser.add_argument("--official-template", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--base-template", type=Path)
    parser.add_argument("--scrambled-thinking", action="store_true")
    parser.add_argument("--failed-history-only", action="store_true")
    parser.add_argument("labels", nargs="+")
    args = parser.parse_args()
    try:
        prepare(args.private, args.endpoint, args.labels, args.official_template,
                args.tokenizer, args.base_template, args.scrambled_thinking, args.failed_history_only)
    except Exception as error:
        print(json.dumps({"error_type": type(error).__name__}))
        raise SystemExit(1) from None

"""Validate private replay emissions through the installed tool parser.

Only counts and validity flags leave the protected fixture directory. This
does not execute tools, display arguments, or modify the source conversation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path


def authenticated_output(private, experiment, row):
    """Reject stale private output rather than scoring another trial's tokens."""
    label = row["label"]
    candidates = [private / f"output-{experiment}-{label}.json",
                  private / f"output-{label}.json"]
    output = next(path for path in candidates if path.exists())
    ids = json.loads(output.read_text())["token_ids"]
    digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    if digest != row["output_sha256"]:
        raise ValueError("private output differs from the recorded trial hash")
    return ids


def score(root: Path, private: Path, tokenizer_path: Path):
    # A malformed call can make third-party parsers log its private arguments.
    # Validation below records only exception types and validity counts.
    logging.disable(logging.CRITICAL)
    import jsonschema
    from transformers import AutoTokenizer
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser
    from diagnose_loop_replay import thinking_repetition

    assert private.stat().st_mode & 0o077 == 0
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    experiment = json.loads((root / "manifest.json").read_text()).get("experiment_id")
    rows = []
    for row in json.loads((root / "results.json").read_text()):
        label = row["label"]
        ids = authenticated_output(private, experiment, row)
        payload = json.loads((private / f"payload-{row['fixture']}.json").read_text())
        request = ChatCompletionRequest(model=payload["model"],
                    messages=[{"role": "user", "content": "Opaque replay validation."}],
                    tools=payload.get("tools"))
        parser = Qwen3EngineToolParser(tokenizer, tools=request.tools)
        text = tokenizer.decode(ids, skip_special_tokens=False)
        thinking = thinking_repetition(text, ids, tokenizer.convert_tokens_to_ids("</think>"))
        # The completion fixture starts inside thinking. Tools are interpreted
        # only after its close, just as the production reasoning parser does.
        closed = "</think>" in text
        content = text.split("</think>", 1)[1] if closed else ""
        parsed = parser.extract_tool_calls(content, request)
        schemas = {tool["function"]["name"]: tool["function"]["parameters"]
                   for tool in payload.get("tools", [])}
        valid = 0
        failures = []
        for call in parsed.tool_calls:
            try:
                schema = schemas[call.function.name]
                jsonschema.validate(json.loads(call.function.arguments), schema)
                valid += 1
            except Exception as error:
                failures.append(type(error).__name__)
        rows.append({"label": label, "thinking_closed": closed,
                     "parsed_tool_calls": len(parsed.tool_calls),
                     "valid_tool_calls": valid, "validation_error_types": failures,
                     "complete_valid_tool_emission": row["finish_reason"] == "stop"
                     and bool(parsed.tool_calls) and valid == len(parsed.tool_calls),
                     "complete_final_text": row["finish_reason"] == "stop" and closed
                     and not parsed.tool_calls and bool(content.strip())
                     and "<tool_call>" not in content and "</tool_call>" not in content,
                     "loop_candidate": row["repetition"]["loop_candidate"],
                     "thinking_repetition": thinking})
    (root / "tool-validation.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows))


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--private", type=Path, default=Path("/private-fixtures"))
    parser.add_argument("--tokenizer", type=Path, default=Path("/models/Qwen3.8-27B-Uncensored-MXFP4-awq"))
    args = parser.parse_args()
    try:
        score(args.root, args.private, args.tokenizer)
    except Exception as error:
        print(json.dumps({"error_type": type(error).__name__}))
        raise SystemExit(1) from None

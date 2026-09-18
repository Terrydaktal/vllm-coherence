"""Strict, non-executing framing for a Qwen reasoning/text/tool stream.

Candidate only: not installed by the production launcher. Protocol delimiters
must occur at a line boundary outside a Markdown fence. Tool-looking text inside
reasoning is inert; an explicit </think> boundary precedes executable output.
Only complete schema-valid tool records are returned, never partial arguments.
Complete JSON name/arguments records inside the explicit tool frame normalize
to the same structured call as XML; no missing fields or values are invented.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator


class ProtocolViolation(ValueError):
    """Terminal protocol failure, carrying preserved non-executable output."""

    def __init__(self, message: str, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolViolation("duplicate JSON key")
        result[key] = value
    return result


def strict_json(text: str) -> Any:
    def reject_constant(_value: str) -> None:
        raise ProtocolViolation("non-finite JSON value")

    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ProtocolViolation("non-finite JSON value")
        return number

    try:
        return json.loads(text, object_pairs_hook=_unique_object, parse_constant=reject_constant,
                          parse_float=finite_float)
    except (json.JSONDecodeError, RecursionError) as error:
        raise ProtocolViolation("incomplete or invalid JSON") from error


@dataclass(frozen=True)
class ToolRecord:
    name: str
    arguments: dict[str, Any]
    wire_format: str = "xml"


@dataclass
class Delta:
    reasoning: str = ""
    content: str = ""
    calls: list[ToolRecord] = field(default_factory=list)


class FramedToolProtocol:
    """Incremental framing. feed() does not execute or dispatch anything.

    Calls are withheld until the entire response passes finish(). This makes a
    mixed batch (valid call followed by a malformed one) failure-atomic at the
    parser boundary. Reasoning and ordinary content may be displayed as they
    arrive. The caller must preserve them when finish raises ProtocolViolation.
    """

    def __init__(self, tools: dict[str, dict[str, Any]], *, thinking: bool = True,
                 max_chars: int = 1_000_000, allow_json_calls: bool = True) -> None:
        self.validators = {}
        if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < 1:
            raise ValueError("max_chars must be a positive integer")
        for name, schema in tools.items():
            Draft202012Validator.check_schema(schema)
            # Remote schema retrieval is never appropriate during tool dispatch.
            def check_refs(value: Any) -> None:
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key in {"$ref", "$dynamicRef"} and not item.startswith("#"):
                            raise ProtocolViolation("external schema reference is prohibited")
                        check_refs(item)
                elif isinstance(value, list):
                    for item in value:
                        check_refs(item)

            check_refs(schema)
            self.validators[name] = Draft202012Validator(schema)
        self.mode = "reasoning" if thinking else "content"
        self.allow_json_calls = allow_json_calls
        self.max_chars = max_chars
        self.pending = ""
        self.raw_parts: list[str] = []
        self.raw_chars = 0
        self.line = ""
        self.fence: str | None = None
        self.line_overflow = False
        self.tool_parts: list[str] = []
        self.calls: list[ToolRecord] = []
        self.visible_nonwhite = False
        self.finished = False
        self.fault: str | None = None

    def _track(self, text: str) -> None:
        for char in text:
            if char == "\n":
                marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", self.line)
                if marker:
                    if self.fence is None:
                        self.fence = marker[1]
                    elif (marker[1][0] == self.fence[0]
                          and len(marker[1]) >= len(self.fence)
                          and not marker[2].strip() and not self.line_overflow):
                        self.fence = None
                self.line = ""
                self.line_overflow = False
            elif len(self.line) < 256:
                self.line += char
            else:
                self.line_overflow = True

    def _emit(self, text: str, delta: Delta) -> None:
        if self.mode == "reasoning":
            delta.reasoning += text
        else:
            delta.content += text
            self.visible_nonwhite |= bool(text.strip())
        self._track(text)

    def _decode_call(self, body: str) -> ToolRecord:
        wire_format = "xml"
        if body.lstrip().startswith("{"):
            if not self.allow_json_calls:
                raise ProtocolViolation("JSON tool call emitted under the XML contract")
            record = strict_json(body)
            if not isinstance(record, dict) or set(record) != {"name", "arguments"}:
                raise ProtocolViolation("invalid JSON tool record")
            name, arguments = record["name"], record["arguments"]
            wire_format = "json"
        else:
            match = re.fullmatch(r"\s*<function=([A-Za-z_][A-Za-z0-9_.-]*)>(.*)</function>\s*",
                                 body, re.DOTALL)
            if not match:
                raise ProtocolViolation("incomplete or invalid XML tool record")
            name, remaining = match[1], match[2]
            if name not in self.validators:
                raise ProtocolViolation("unregistered tool name")
            arguments = {}
            schema = self.validators[name].schema
            while remaining.strip():
                parameter = re.match(r"\s*<parameter=([^<>\s=]+)>(.*?)</parameter>",
                                     remaining, re.DOTALL)
                if not parameter:
                    raise ProtocolViolation("incomplete XML parameter")
                key, raw = parameter[1], parameter[2]
                if key in arguments:
                    raise ProtocolViolation("duplicate XML parameter")
                # Match Qwen's one-newline wrapping, not arbitrary .strip().
                raw = raw.removeprefix("\n").removesuffix("\n")
                property_schema = schema.get("properties", {}).get(key, {})
                if self.validators[name].evolve(schema=property_schema).is_valid(raw):
                    arguments[key] = raw
                else:
                    arguments[key] = strict_json(raw)
                remaining = remaining[parameter.end():]
        if not isinstance(name, str) or name not in self.validators:
            raise ProtocolViolation("unregistered tool name")
        if not isinstance(arguments, dict):
            raise ProtocolViolation("tool arguments must be an object")
        if not self.validators[name].is_valid(arguments):
            raise ProtocolViolation("tool arguments do not satisfy the registered schema")
        return ToolRecord(name=name, arguments=arguments, wire_format=wire_format)

    @staticmethod
    def _json_string_open(text: str) -> bool:
        """An XML closing-marker spelling inside a JSON string is argument data."""
        quoted = escaped = False
        for char in text:
            if escaped:
                escaped = False
            elif quoted and char == "\\":
                escaped = True
            elif char == '"':
                quoted = not quoted
        return quoted

    def feed(self, text: str) -> Delta:
        if self.finished:
            raise ProtocolViolation("data after response completion")
        if not isinstance(text, str):
            raise TypeError("stream delta must be text")
        if self.raw_chars + len(text) > self.max_chars:
            available = max(0, self.max_chars - self.raw_chars)
            self.raw_parts.append(text[:available])
            self.raw_chars += available
            self.fault = "protocol capture limit exceeded"
            raise ProtocolViolation(self.fault, "".join(self.raw_parts))
        self.raw_chars += len(text)
        self.raw_parts.append(text)
        self.pending += text
        delta = Delta()
        while self.pending:
            if self.mode == "tool":
                end = self.pending.find("</tool_call>")
                if end < 0:
                    # Keep only enough lookbehind for a split closing marker.
                    safe = max(0, len(self.pending) - len("</tool_call>"))
                    self.tool_parts.append(self.pending[:safe])
                    self.pending = self.pending[safe:]
                    break
                self.tool_parts.append(self.pending[:end])
                body = "".join(self.tool_parts)
                if body.lstrip().startswith("{") and self._json_string_open(body):
                    # Preserve literal control-marker text in a JSON argument.
                    # This does not synthesize or close an unterminated string.
                    self.tool_parts.append("</tool_call>")
                    self.pending = self.pending[end + len("</tool_call>"):]
                    continue
                try:
                    self.calls.append(self._decode_call(body))
                except ProtocolViolation as error:
                    self.fault = str(error)
                self.tool_parts.clear()
                self.pending = self.pending[end + len("</tool_call>"):]
                self.mode = "content"
                self.line = ""
                continue
            boundary = not self.line.strip() and not self.line_overflow and self.fence is None
            markers = ("<think>", "</think>") if self.mode == "reasoning" else ("<tool_call>",)
            if boundary:
                marker = next((m for m in markers if self.pending.startswith(m)), None)
                if marker:
                    self.pending = self.pending[len(marker):]
                    if marker == "</think>":
                        self.mode = "content"
                    elif marker == "<tool_call>":
                        self.mode = "tool"
                    self.line = ""
                    continue
                if any(m.startswith(self.pending) for m in markers):
                    break
            self._emit(self.pending[0], delta)
            self.pending = self.pending[1:]
        return delta

    def finish(self, finish_reason: str) -> Delta:
        if self.finished:
            raise ProtocolViolation("duplicate response completion")
        self.finished = True
        raw = "".join(self.raw_parts)
        if finish_reason not in {"stop", "tool_calls"}:
            raise ProtocolViolation(f"response not complete ({finish_reason})", raw)
        if self.fault:
            raise ProtocolViolation(self.fault, raw)
        if self.mode == "tool":
            raise ProtocolViolation("response ended inside a tool call", raw)
        if self.mode == "reasoning":
            raise ProtocolViolation("response ended before reasoning closed", raw)
        # A suffix that could begin a tool marker is incomplete framing, not
        # proof of an actionable call. Never guess its name or arguments.
        if self.pending and "<tool_call>".startswith(self.pending) and self.pending.startswith("<"):
            raise ProtocolViolation("response ended inside a tool opening marker", raw)
        delta = Delta()
        self._emit(self.pending, delta)
        self.pending = ""
        if not self.calls and not self.visible_nonwhite:
            raise ProtocolViolation("response contains neither a tool call nor a final answer", raw)
        if finish_reason == "tool_calls" and not self.calls:
            raise ProtocolViolation("provider claimed tool calls but none passed validation", raw)
        delta.calls = list(self.calls)
        return delta


class OpenAIFramedStream:
    """Non-networking, single-choice SSE adapter for the isolated candidate.

    The caller supplies raw deltas, not already-parsed tool deltas. The caller
    must supply the real provider finish reason. No retries, inference requests,
    tool execution or transcript writes take place here. Integration into the
    actual server remains a separate, gated change.
    """

    def __init__(self, parser: FramedToolProtocol, request_id: str, model: str) -> None:
        self.parser = parser
        self.request_id = request_id
        self.model = model
        self.closed = False
        self.failure: ProtocolViolation | None = None

    @staticmethod
    def _sse(record: dict[str, Any]) -> str:
        return "data: " + json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n\n"

    def _chunk(self, delta: Delta, finish: str | None = None) -> str:
        body: dict[str, Any] = {}
        if delta.reasoning:
            body["reasoning"] = delta.reasoning
        if delta.content:
            body["content"] = delta.content
        if delta.calls:
            body["tool_calls"] = [
                {"index": index, "id": f"{self.request_id}-call-{index}", "type": "function",
                 "function": {"name": call.name,
                              "arguments": json.dumps(call.arguments, allow_nan=False)}}
                for index, call in enumerate(delta.calls)
            ]
        if not body and finish is None:
            return ""
        return self._sse({"id": self.request_id, "object": "chat.completion.chunk",
                          "created": 0, "model": self.model,
                          "choices": [{"index": 0, "delta": body, "finish_reason": finish}]})

    def _failure(self, error: ProtocolViolation) -> str:
        self.closed = True
        self.failure = error
        # Use a stable, non-retryable message. Do not interpolate model output
        # or arbitrary exception text: Pi's retry classifier matches substrings
        # such as "500" and "ended without". Raw evidence stays on failure.raw,
        # never in a tool delta or a model-visible corrective prompt.
        return self._sse({"error": {
            "type": "invalid_model_output", "code": "qwen_invalid_tool_frame",
            "message": "Qwen tool protocol rejected incomplete or invalid output. "
                       "Earlier text retained; no tools dispatched. Automatic retry disabled.",
        }}) + "data: [DONE]\n\n"

    def feed(self, text: str) -> str:
        if self.closed:
            raise ProtocolViolation("data after SSE completion")
        try:
            return self._chunk(self.parser.feed(text))
        except ProtocolViolation as error:
            return self._failure(error)

    def finish(self, finish_reason: str) -> str:
        if self.closed:
            raise ProtocolViolation("duplicate SSE completion")
        try:
            delta = self.parser.finish(finish_reason)
        except ProtocolViolation as error:
            return self._failure(error)
        self.closed = True
        return self._chunk(delta, "tool_calls" if delta.calls else "stop") + "data: [DONE]\n\n"

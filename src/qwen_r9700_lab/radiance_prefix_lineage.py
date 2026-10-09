"""Bounded CPU-only comparisons across prompt reconstruction boundaries.

Inputs are observations from the serving path, never a reread of a transcript.
Token IDs, strings, and fingerprints live only in bounded process memory. The
emitter receives enums, booleans, and counts; it never receives their values or
hashes. A matching token prefix is evidence about serialization, not a proof of
model arithmetic or a statement about which emitted token was processed into KV.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
from array import array
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

COMPARISONS = frozenset(
    {
        "raw_to_reencoded",
        "output_to_message",
        "template_normalization",
        "prompt_prefix",
        "input_processor",
    }
)
OPERATIONS = frozenset(
    {
        "trim",
        "delimiter",
        "content_trim",
        "reasoning_trim",
        "leading_delimiter",
        "inline_reasoning_split",
        "assistant_delimiters",
    }
)
STATUSES = frozenset({"complete", "incomplete", "unavailable", "not_applicable"})
_FINGERPRINT_KEY = secrets.token_bytes(32)


def compare_sequences(previous, current, *, prefix=False):
    """Compare observed CPU sequences without including their contents."""
    limit = min(len(previous), len(current))
    first = next((i for i in range(limit) if previous[i] != current[i]), None)
    if (
        first is None
        and len(previous) != len(current)
        and not (prefix and len(current) >= len(previous))
    ):
        first = limit
    result = {
        "equal": first is None,
        "compared_tokens": limit,
        "previous_tokens": len(previous),
        "current_tokens": len(current),
    }
    if first is not None:
        result["first_difference"] = first
    return result


def _get(value, name, default=None):
    return (
        value.get(name, default)
        if isinstance(value, dict)
        else getattr(value, name, default)
    )


def _fingerprint(value):
    digest, size = hashlib.blake2b(key=_FINGERPRINT_KEY, digest_size=32), 0
    encoder = json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    for part in encoder.iterencode(value):
        if len(part) > 8 * 1024 * 1024:
            raise ValueError("diagnostic fingerprint budget exceeded")
        encoded = part.encode()
        size += len(encoded)
        if size > 8 * 1024 * 1024:
            raise ValueError("diagnostic fingerprint budget exceeded")
        digest.update(encoded)
    return digest.digest()


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(exclude_none=True))
    raise TypeError("unsupported diagnostic input")


def _assistant_fields(message):
    content = _get(message, "content")
    reasoning = _get(message, "reasoning_content")
    if reasoning is None:
        reasoning = _get(message, "reasoning")
    if reasoning is None:
        reasoning = _get(message, "thinking")
    if not (content is None or isinstance(content, str)) or not (
        reasoning is None or isinstance(reasoning, str)
    ):
        raise TypeError("unsupported assistant representation")
    calls = _get(message, "tool_calls") or []
    normalized = []
    for call in calls:
        function = _get(call, "function") or {}
        name, arguments = (
            _get(function, "name") or "",
            _get(function, "arguments") or "",
        )
        if not isinstance(name, str) or not isinstance(arguments, str):
            raise TypeError("unsupported tool arguments")
        normalized.append(
            {
                "id": _get(call, "id") or "",
                "type": _get(call, "type") or "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return {
        "content": content or "",
        "reasoning": reasoning or "",
        "tool_calls": normalized,
    }


@dataclass
class _Turn:
    handle: int
    chat_key: str | None
    tokenizer: Any
    config: bytes | None
    history: bytes | None
    message_count: int
    assistant_index: int | None = None
    previous: Any = None
    prompt: array | None = None
    processed_prompt: array | None = None
    output: array = field(default_factory=lambda: array("i"))
    delivered_content: list = field(default_factory=list)
    delivered_reasoning: list = field(default_factory=list)
    delivered_tools: dict = field(default_factory=dict)
    delivered_seen: bool = False
    delivered_digest: bytes | None = None
    delivered_bytes: int = 0
    complete: bool = False
    capture_gap: str | None = None
    template_seen: bool = False
    template_changed: bool = False
    trim_changed: bool = False
    delimiter_changed: bool = False
    template_previous_characters: int = 0
    template_current_characters: int = 0
    template_compared_characters: int = 0
    canonical_body: str | None = None
    canonical_body_bytes: int = 0
    delimiter_checked: bool = False
    delimiter_gap: str | None = None
    prefix_matches: bool | None = None
    delivered_tool_count: int = 0
    config_changed: bool = False
    history_changed: bool = False
    incremental_prompt: bool = False

    @property
    def bytes(self):
        return (
            (len(self.prompt) if self.prompt is not None else 0) * 4
            + (
                len(self.processed_prompt)
                if self.processed_prompt is not None
                and self.processed_prompt is not self.prompt
                else 0
            )
            * 4
            + len(self.output) * 4
            + self.delivered_bytes
            + self.canonical_body_bytes
        )


class PrefixLineageObserver:
    """Keep a bounded last-completed-response journal for each exact cache salt.

    The adapter must call ``engine_output`` with DELTA token IDs, not a tensor,
    and pass the actual normalized template operations through
    ``template_operation``. Unobserved paths are explicitly unavailable.
    """

    def __init__(
        self,
        emit: Callable[[dict], None] | None = None,
        *,
        max_chats=8,
        max_pending=8,
        max_tokens=300_000,
        max_bytes=32 * 1024 * 1024,
        max_text_bytes=8 * 1024 * 1024,
    ):
        if min(max_chats, max_pending, max_tokens, max_bytes, max_text_bytes) <= 0:
            raise ValueError("diagnostic bounds must be positive")
        self.emit = emit or (lambda _record: None)
        self.max_chats, self.max_pending, self.max_tokens = (
            max_chats,
            max_pending,
            max_tokens,
        )
        self.max_bytes, self.max_text_bytes = max_bytes, max_text_bytes
        self._pending = OrderedDict()
        self._completed = OrderedDict()
        self._evicted = OrderedDict()
        self._next = 0
        self._callbacks = {}
        self._lock = threading.RLock()
        self.dropped = 0

    def _record_for(self, handle, comparison, status="complete", **values):
        assert comparison in COMPARISONS and status in STATUSES
        row = {
            "stage": "prefix_lineage",
            "comparison": comparison,
            "diagnostic_status": status,
            **values,
        }
        callback, request_id = self._callbacks.get(handle, (self.emit, None))
        if request_id is not None:
            row["http_request_id"] = request_id
        if self.dropped:
            row["dropped_records"] = self.dropped
        try:
            callback(row)
        except Exception:  # noqa: BLE001 - optional diagnostics never affect serving
            self.dropped += 1
        return row

    def _turn(self, handle):
        return self._pending.get(handle)

    def _ids(self, ids):
        # Reject tensors and array protocols rather than copying/synchronizing GPU.
        if (
            not isinstance(ids, (list, tuple, array, range))
            or len(ids) > self.max_tokens
        ):
            raise TypeError("unsupported or over-budget token sequence")
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            or value > 2**31 - 1
            for value in ids
        ):
            raise TypeError("invalid token sequence")
        return array("i", ids)

    def _remember_eviction(self, key):
        if key is not None:
            self._evicted[key] = True
            self._evicted.move_to_end(key)
            while len(self._evicted) > self.max_chats * 4:
                self._evicted.popitem(last=False)

    def _bound(self, current=None):
        while self._completed and (
            len(self._completed) > self.max_chats
            or sum(t.bytes for t in self._completed.values())
            + sum(t.bytes for t in self._pending.values())
            > self.max_bytes
        ):
            key, old = self._completed.popitem(last=False)
            self._remember_eviction(key)
            # Pending comparisons may have retained this object. Drop both refs.
            for turn in self._pending.values():
                if turn.previous is old:
                    turn.previous = None
                    turn.capture_gap = "evicted"
        if (
            current is not None
            and sum(t.bytes for t in self._pending.values()) > self.max_bytes
        ):
            current.prompt = None
            current.processed_prompt = None
            current.output = array("i")
            current.delivered_content.clear()
            current.delivered_reasoning.clear()
            current.delivered_tools.clear()
            current.delivered_bytes = 0
            current.canonical_body = None
            current.canonical_body_bytes = 0
            current.capture_gap = "dropped"

    def begin(
        self, request, tokenizer, config, *, chat_key=None, request_id=None, emit=None
    ):
        with self._lock:
            self._next += 1
            handle = self._next
            safe_id = (
                request_id
                if isinstance(request_id, str)
                and len(request_id) == 64
                and all(c in "0123456789abcdef" for c in request_id)
                else None
            )
            self._callbacks[handle] = (emit or self.emit, safe_id)
            key = chat_key if chat_key is not None else _get(request, "cache_salt")
            key = key if isinstance(key, str) and key else None
            messages = _get(request, "messages") or []
            previous = self._completed.get(key) if key is not None else None
            overlapping = [
                active
                for active in self._pending.values()
                if key is not None and active.chat_key == key
            ]
            if overlapping:
                for active in overlapping:
                    active.capture_gap = "overlapping_requests"
                self._completed.pop(key, None)
                previous = None
            turn = _Turn(
                handle, key, tokenizer, None, None, len(messages), previous=previous
            )
            if overlapping:
                turn.capture_gap = "overlapping_requests"
            self._pending[handle] = turn
            while len(self._pending) > self.max_pending:
                victim_handle, victim = self._pending.popitem(last=False)
                self._remember_eviction(victim.chat_key)
                self._completed.pop(victim.chat_key, None)
                self._record_for(
                    victim_handle, "prompt_prefix", "incomplete", reason="evicted"
                )
                self._callbacks.pop(victim_handle, None)
            try:
                turn.config = _fingerprint(
                    {"model": _get(request, "model"), "template": _plain(config)}
                )
                turn.history = _fingerprint(_plain(messages))
                if previous is not None:
                    turn.config_changed = (
                        turn.config != previous.config
                        or tokenizer is not previous.tokenizer
                    )
                    turn.history_changed = (
                        len(messages) < previous.message_count
                        or _fingerprint(_plain(messages[: previous.message_count]))
                        != previous.history
                    )
                    turn.assistant_index = next(
                        (
                            index
                            for index in range(previous.message_count, len(messages))
                            if _get(messages[index], "role") == "assistant"
                        ),
                        None,
                    )
                    if turn.assistant_index is None:
                        self._record_for(
                            handle,
                            "output_to_message",
                            "unavailable",
                            reason="missing",
                            history_changed=turn.history_changed,
                            config_changed=turn.config_changed,
                        )
                    elif not previous.complete or previous.capture_gap:
                        self._record_for(
                            handle,
                            "output_to_message",
                            "incomplete",
                            reason=previous.capture_gap or "partial_output",
                        )
                    elif previous.delivered_digest is None:
                        self._record_for(
                            handle, "output_to_message", "unavailable", reason="missing"
                        )
                    else:
                        incoming = _fingerprint(
                            _assistant_fields(messages[turn.assistant_index])
                        )
                        self._record_for(
                            handle,
                            "output_to_message",
                            equal=incoming == previous.delivered_digest,
                            unit="message_fields",
                            previous_fields=3,
                            current_fields=3,
                            compared_fields=3,
                            difference_position_available=False,
                            history_changed=turn.history_changed,
                            config_changed=turn.config_changed,
                        )
                else:
                    self._record_for(
                        handle,
                        "output_to_message",
                        "unavailable" if key else "not_applicable",
                        reason=turn.capture_gap or "evicted"
                        if key in self._evicted
                        else turn.capture_gap or "restart"
                        if key
                        else "missing_cache_salt",
                    )
            except Exception:  # noqa: BLE001 - failures become explicit diagnostic gaps
                turn.capture_gap = "unsupported"
                self._record_for(
                    handle, "output_to_message", "unavailable", reason="unsupported"
                )
            self._bound(turn)
            return handle

    def assistant_index(self, handle):
        with self._lock:
            turn = self._turn(handle)
            return turn.assistant_index if turn else None

    def rendered(self, handle, prompt_token_ids):
        with self._lock:
            turn = self._turn(handle)
            if turn is None:
                return self._record_for(
                    handle, "prompt_prefix", "unavailable", reason="evicted"
                )
            try:
                turn.prompt = self._ids(prompt_token_ids)
            except Exception:  # noqa: BLE001 - failures become explicit diagnostic gaps
                turn.capture_gap = "unsupported"
                return self._record_for(
                    handle, "prompt_prefix", "unavailable", reason="unsupported"
                )
            previous = turn.previous
            if previous is None:
                return self._record_for(
                    handle,
                    "prompt_prefix",
                    "unavailable",
                    reason=turn.capture_gap
                    or ("evicted" if turn.chat_key in self._evicted else "restart"),
                )
            if not previous.complete or previous.prompt is None or previous.capture_gap:
                return self._record_for(
                    handle,
                    "prompt_prefix",
                    "incomplete",
                    reason=previous.capture_gap or "partial_output",
                )
            raw = (
                previous.processed_prompt
                if previous.processed_prompt is not None
                else previous.prompt
            ) + previous.output
            if len(raw) > self.max_tokens:
                return self._record_for(
                    handle, "prompt_prefix", "incomplete", reason="dropped"
                )
            result = compare_sequences(raw, turn.prompt, prefix=True)
            turn.prefix_matches = result["equal"]
            row = self._record_for(
                handle,
                "prompt_prefix",
                **result,
                config_changed=turn.config_changed,
                history_changed=turn.history_changed,
                processed_endpoint_known=False,
            )
            if turn.incremental_prompt:
                self._record_for(
                    handle,
                    "raw_to_reencoded",
                    "not_applicable",
                    reason="incremental_generated_tokens",
                )
            elif result["equal"]:
                self._record_for(
                    handle,
                    "raw_to_reencoded",
                    "not_applicable",
                    reason="matching_prefix",
                )
            elif turn.config_changed:
                self._record_for(
                    handle,
                    "raw_to_reencoded",
                    "unavailable",
                    reason="configuration_changed",
                )
            else:
                self._roundtrip(turn, raw)
                self._check_delimiters(turn)
            self._bound(turn)
            return row

    def _roundtrip(self, turn, raw):
        try:
            text = turn.tokenizer.decode(
                list(raw), skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            if (
                not isinstance(text, str)
                or len(text) > self.max_text_bytes
                or len(text.encode()) > self.max_text_bytes
            ):
                return self._record_for(
                    turn.handle, "raw_to_reencoded", "incomplete", reason="dropped"
                )
            encoded = self._ids(turn.tokenizer.encode(text, add_special_tokens=False))
            return self._record_for(
                turn.handle, "raw_to_reencoded", **compare_sequences(raw, encoded)
            )
        except Exception:  # noqa: BLE001 - failures become explicit diagnostic gaps
            return self._record_for(
                turn.handle, "raw_to_reencoded", "unavailable", reason="unsupported"
            )

    def _check_delimiters(self, turn):
        """Compare observed canonical assistant text with its previous raw body.

        Decode a bounded prompt tail and its output jointly, retaining tokenizer
        context at the boundary. Only the pinned generation suffix and recognized
        trailing stop marker are admitted. The observed assistant representation
        includes tool serialization when the template adapter captures it.
        """
        previous = turn.previous
        if turn.canonical_body is None or previous is None:
            turn.delimiter_gap = "missing"
            return
        prompt = (
            previous.processed_prompt
            if previous.processed_prompt is not None
            else previous.prompt
        )
        marker = "<|im_start|>assistant\n"
        try:
            tail = prompt[-128:]
            prefix = turn.tokenizer.decode(
                list(tail),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            position = prefix.rfind(marker)
            if position < 0 or prefix[position:] not in {
                marker,
                marker + "<think>\n",
                marker + "<think>\n\n</think>\n\n",
            }:
                turn.delimiter_gap = "unsupported"
                return
            text = turn.tokenizer.decode(
                list(tail + previous.output),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if (
                not isinstance(text, str)
                or len(text.encode()) > self.max_text_bytes
                or text[: position + len(marker)] != prefix[: position + len(marker)]
            ):
                turn.delimiter_gap = "unsupported"
                return
            body = text[position + len(marker) :]
            stop_removed = body.endswith("<|im_end|>")
            if stop_removed:
                body = body[: -len("<|im_end|>")]
            canonical = turn.canonical_body
            canonical = canonical.removeprefix(marker)
            if canonical.endswith("<|im_end|>\n"):
                canonical = canonical[: -len("<|im_end|>\n")]
            elif canonical.endswith("<|im_end|>"):
                canonical = canonical[: -len("<|im_end|>")]
            result = compare_sequences(body, canonical)
            turn.delimiter_checked = True
            turn.template_seen = True
            turn.template_changed |= not result["equal"]
            turn.delimiter_changed |= not result["equal"]
            turn.template_previous_characters += len(body)
            turn.template_current_characters += len(canonical)
            turn.template_compared_characters += min(len(body), len(canonical))
            values = {
                "equal": result["equal"],
                "unit": "characters",
                "previous_characters": len(body),
                "current_characters": len(canonical),
                "compared_characters": min(len(body), len(canonical)),
                "operation": "assistant_delimiters",
                "scope": "assistant_message_without_terminator",
                "recognized_stop_marker_removed": stop_removed,
            }
            if "first_difference" in result:
                values["difference_position"] = result["first_difference"]
            self._record_for(turn.handle, "template_normalization", **values)
        except Exception:  # noqa: BLE001 - failures become explicit diagnostic gaps
            turn.delimiter_gap = "unsupported"

    def prompt_construction(self, handle, construction):
        with self._lock:
            turn = self._turn(handle)
            if turn is not None:
                turn.incremental_prompt = construction == "incremental_generated_tokens"

    def admitted_prompt(self, handle, token_ids):
        """Track an explicitly admitted generated-token continuation.

        rendered() retains the canonical reconstruction comparison. The next
        boundary must compare the processor against the actual admitted IDs.
        """
        with self._lock:
            turn = self._turn(handle)
            if turn is not None:
                turn.prompt = self._ids(token_ids)
                self._bound(turn)

    def input_processor(self, handle, token_ids):
        with self._lock:
            turn = self._turn(handle)
            if turn is None or turn.prompt is None:
                return self._record_for(
                    handle,
                    "input_processor",
                    "unavailable",
                    reason="missing" if turn else "evicted",
                )
            try:
                actual = self._ids(token_ids)
                result = compare_sequences(turn.prompt, actual)
                turn.processed_prompt = turn.prompt if result["equal"] else actual
                row = self._record_for(handle, "input_processor", **result)
                self._bound(turn)
                return row
            except Exception:  # noqa: BLE001 - failures become explicit diagnostic gaps
                return self._record_for(
                    handle, "input_processor", "unavailable", reason="unsupported"
                )

    def engine_output(self, handle, delta_token_ids):
        with self._lock:
            turn = self._turn(handle)
            if turn is None or turn.capture_gap:
                return
            try:
                ids = self._ids(delta_token_ids)
                if (
                    len(turn.output)
                    + len(ids)
                    + (len(turn.prompt) if turn.prompt is not None else 0)
                    > self.max_tokens
                ):
                    turn.capture_gap = "dropped"
                    return
                turn.output.extend(ids)
                self._bound(turn)
            except Exception:  # noqa: BLE001 - failures become explicit diagnostic gaps
                turn.capture_gap = "unsupported"

    def delivered_delta(self, handle, delta):
        with self._lock:
            turn = self._turn(handle)
            if turn is None or turn.capture_gap:
                return
            try:
                content, reasoning = _get(delta, "content"), _get(delta, "reasoning")
                if reasoning is None:
                    reasoning = _get(delta, "reasoning_content")
                for value, target in (
                    (content, turn.delivered_content),
                    (reasoning, turn.delivered_reasoning),
                ):
                    if value is not None:
                        if not isinstance(value, str):
                            raise TypeError("unsupported delta")
                        target.append(value)
                        turn.delivered_bytes += len(value.encode())
                        turn.delivered_seen = True
                for call in _get(delta, "tool_calls") or []:
                    index = _get(call, "index")
                    if not isinstance(index, int) or index < 0 or index > 64:
                        raise TypeError("unsupported tool delta index")
                    item = turn.delivered_tools.setdefault(
                        index,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    for name in ("id", "type"):
                        value = _get(call, name)
                        if value is not None:
                            if not isinstance(value, str):
                                raise TypeError("unsupported tool delta")
                            item[name] = value
                            turn.delivered_bytes += len(value.encode())
                    function = _get(call, "function") or {}
                    for name in ("name", "arguments"):
                        value = _get(function, name)
                        if value is not None:
                            if not isinstance(value, str):
                                raise TypeError("unsupported tool delta")
                            item["function"][name] += value
                            turn.delivered_bytes += len(value.encode())
                    turn.delivered_seen = True
                if turn.delivered_bytes > self.max_text_bytes:
                    turn.capture_gap = "dropped"
                self._bound(turn)
            except Exception:  # noqa: BLE001 - failures become explicit diagnostic gaps
                turn.capture_gap = "unsupported"

    def template_operation(self, handle, before, after, operation, message_index):
        with self._lock:
            turn = self._turn(handle)
            if turn is None or message_index != turn.assistant_index:
                return None
            if (
                operation not in OPERATIONS
                or not isinstance(before, str)
                or not isinstance(after, str)
            ):
                turn.capture_gap = "unsupported"
                return self._record_for(
                    handle,
                    "template_normalization",
                    "unavailable",
                    reason="unsupported",
                )
            if max(len(before), len(after)) > self.max_text_bytes:
                return self._record_for(
                    handle, "template_normalization", "incomplete", reason="dropped"
                )
            if operation == "assistant_delimiters" and before == "":
                size = len(after.encode())
                if size > self.max_text_bytes:
                    turn.delimiter_gap = "dropped"
                    return self._record_for(
                        handle, "template_normalization", "incomplete", reason="dropped"
                    )
                turn.canonical_body, turn.canonical_body_bytes = after, size
                self._bound(turn)
                return None
            result = compare_sequences(before, after)
            turn.template_seen = True
            turn.template_previous_characters += len(before)
            turn.template_current_characters += len(after)
            turn.template_compared_characters += min(len(before), len(after))
            changed = not result["equal"]
            turn.template_changed |= changed
            turn.trim_changed |= changed and operation in {
                "trim",
                "content_trim",
                "reasoning_trim",
            }
            turn.delimiter_changed |= changed and operation in {
                "delimiter",
                "leading_delimiter",
                "inline_reasoning_split",
                "assistant_delimiters",
            }
            values = {
                "equal": result["equal"],
                "previous_characters": len(before),
                "current_characters": len(after),
                "compared_characters": min(len(before), len(after)),
                "unit": "characters",
                "operation": operation,
            }
            if "first_difference" in result:
                values["difference_position"] = result["first_difference"]
            return self._record_for(handle, "template_normalization", **values)

    def template_complete(self, handle, supported):
        with self._lock:
            turn = self._turn(handle)
            if turn is None:
                return self._record_for(
                    handle, "template_normalization", "unavailable", reason="evicted"
                )
            if turn.previous is None:
                return self._record_for(
                    handle,
                    "template_normalization",
                    "not_applicable",
                    reason="missing_previous_output",
                )
            if not supported or not turn.template_seen:
                return self._record_for(
                    handle,
                    "template_normalization",
                    "unavailable",
                    reason="unsupported" if not supported else "missing",
                )
            if turn.prefix_matches is False and not turn.delimiter_checked:
                return self._record_for(
                    handle,
                    "template_normalization",
                    "incomplete",
                    reason=turn.delimiter_gap
                    or ("configuration_changed" if turn.config_changed else "missing"),
                    trim_changed=turn.trim_changed,
                    delimiter_changed=turn.delimiter_changed,
                )
            return self._record_for(
                handle,
                "template_normalization",
                equal=not turn.template_changed,
                trim_changed=turn.trim_changed,
                delimiter_changed=turn.delimiter_changed,
                comparison_complete=True,
                previous_characters=turn.template_previous_characters,
                current_characters=turn.template_current_characters,
                compared_characters=turn.template_compared_characters,
                difference_position_available=False,
                unit="characters",
            )

    def finish(self, handle, *, complete=True):
        with self._lock:
            turn = self._pending.pop(handle, None)
            if turn is None:
                return
            turn.complete = bool(complete)
            turn.previous = None
            if turn.delivered_seen and not turn.capture_gap:
                try:
                    turn.delivered_digest = _fingerprint(
                        {
                            "content": "".join(turn.delivered_content),
                            "reasoning": "".join(turn.delivered_reasoning),
                            "tool_calls": [
                                turn.delivered_tools[index]
                                for index in sorted(turn.delivered_tools)
                            ],
                        }
                    )
                except Exception:  # noqa: BLE001 - finalization must retire unsupported metadata
                    # JSON escaping can exceed the fingerprint budget even
                    # when the unescaped stream fit. Retire the callback and
                    # all transient text instead of leaking completion state.
                    turn.capture_gap = "unsupported"
            turn.delivered_content.clear()
            turn.delivered_reasoning.clear()
            turn.delivered_tool_count = len(turn.delivered_tools)
            turn.delivered_tools.clear()
            turn.delivered_bytes = 0
            turn.canonical_body = None
            turn.canonical_body_bytes = 0
            if turn.chat_key is not None:
                self._completed[turn.chat_key] = turn
                self._completed.move_to_end(turn.chat_key)
            if not complete or turn.capture_gap:
                self._record_for(
                    handle,
                    "prompt_prefix",
                    "incomplete",
                    reason=turn.capture_gap or "partial_output",
                )
            self._bound()
            self._callbacks.pop(handle, None)

    @property
    def retained_bytes(self):
        with self._lock:
            return sum(turn.bytes for turn in self._pending.values()) + sum(
                turn.bytes for turn in self._completed.values()
            )

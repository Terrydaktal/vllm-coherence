"""Preserve generated token history for authenticated, unchanged chat turns.

This is an explicit generated-token session contract, not a promise of equality
to re-tokenizing all historical text. Cache/state validation remains unchanged.
All work here uses existing CPU IDs; no GPU imports or synchronization occur.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from array import array

try:
    import qwen_radiance_token_continuation as journal
except ModuleNotFoundError as error:
    if error.name != "qwen_radiance_token_continuation":
        raise
    from qwen_r9700_lab import radiance_token_continuation as journal

TEMPLATE_SHA256 = "6e1439c913ad7df4a966493ad70de7e7fc5a548d41bbe417c1571f766603629b"
# Bind normalization, pretokenization, merges, AddedToken flags and decoding,
# rather than assuming an unchanged vocabulary specifies the tokenizer.
INCREMENTAL_BACKEND_SHA256 = (
    "ffb7a28b27dabcc333662fd3e0b0005d9e79a1c22e31453ab5a3017fbd5f25c0"
)
MAX_TEXT_BYTES = 8 * 1024 * 1024
_ledger = None
_ledger_pid = None
_tokenizers = {}


def _get(value, name, default=None):
    return (
        value.get(name, default)
        if isinstance(value, dict)
        else getattr(value, name, default)
    )


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(exclude_none=True))
    raise TypeError("unsupported continuation metadata")


def _digest(value):
    digest, size = hashlib.sha256(), 0
    encoder = json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    for part in encoder.iterencode(_plain(value)):
        encoded = part.encode()
        size += len(encoded)
        if size > MAX_TEXT_BYTES:
            raise ValueError("continuation metadata budget exceeded")
        digest.update(encoded)
    return digest.hexdigest()


def _json_arguments(text):
    """Parse a complete argument object without losing ambiguous JSON data."""

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate tool argument key")
            result[key] = value
        return result

    def invalid_constant(_value):
        raise ValueError("nonfinite tool argument")

    if not isinstance(text, str) or len(text.encode()) > MAX_TEXT_BYTES:
        raise ValueError("unsupported tool argument representation")
    value = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise TypeError("tool arguments must be an object")
    _json_identity(value)
    return value


def _json_identity(value):
    # Object insertion order is intentionally retained: unlike whitespace or
    # Unicode escaping, reordering a tool's nested mapping can be observable.
    if isinstance(value, dict):
        return ["object", [[key, _json_identity(item)] for key, item in value.items()]]
    if isinstance(value, list):
        return ["array", [_json_identity(item) for item in value]]
    if value is None:
        return ["null"]
    if type(value) is bool:
        return ["boolean", value]
    if type(value) is int:
        return ["integer", str(value)]
    if type(value) is float and math.isfinite(value):
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["string", value]
    raise ValueError("unsupported tool argument value")


def _assistant_fields(message, *, semantic_arguments=False):
    content = _get(message, "content") or ""
    reasoning = _get(message, "reasoning_content")
    if reasoning is None:
        reasoning = _get(message, "reasoning")
    if reasoning is None:
        reasoning = _get(message, "thinking")
    reasoning = reasoning or ""
    if not isinstance(content, str) or not isinstance(reasoning, str):
        raise TypeError("unsupported assistant fields")
    tools = []
    for call in _get(message, "tool_calls") or ():
        function = _get(call, "function") or {}
        tools.append(
            {
                "id": _get(call, "id") or "",
                "type": _get(call, "type") or "function",
                "function": {
                    "name": _get(function, "name") or "",
                    "arguments": (
                        json.dumps(
                            _json_identity(
                                _json_arguments(_get(function, "arguments"))
                            ),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        if semantic_arguments
                        else _get(function, "arguments") or ""
                    ),
                },
            }
        )
    if len(tools) > 64 or any(
        not isinstance(item[key], str) for item in tools for key in ("id", "type")
    ):
        raise TypeError("unsupported tool fields")
    if any(
        not isinstance(value, str)
        for item in tools
        for value in item["function"].values()
    ):
        raise TypeError("unsupported tool function")
    return {"content": content, "reasoning": reasoning, "tool_calls": tools}


def _output_digest(message, previous=None):
    return _digest(
        _assistant_fields(
            message,
            semantic_arguments=previous is None
            or previous.output_identity_format == "json_semantics_v1",
        )
    )


def _canonical_prefix(tokenizer, conversation, tools, template_text, kwargs):
    values = {
        key: value for key, value in kwargs.items() if not key.startswith("_coherence_")
    }
    values["add_generation_prompt"] = False
    text = tokenizer.apply_chat_template(
        conversation=conversation,
        tools=tools,
        chat_template=template_text,
        tokenize=False,
        **values,
    )
    if (
        not isinstance(text, str)
        or len(text.encode()) > MAX_TEXT_BYTES
        or not text.endswith("<|im_end|>\n")
    ):
        raise ValueError("unsupported canonical history boundary")
    return text[:-1]


def _raw_output_matches(turn, fields):
    """Conservatively validate exact decoded fields and complete XML calls.

    This is not a replacement parser. Unsupported/ambiguous syntax disables
    normalization replay; ordinary canonical admission remains available.
    """
    tokenizer = turn["tokenizer"]
    # The authenticated generation header is short ASCII. Decode its bounded
    # token tail jointly with output, preserving boundary decoding semantics
    # without rereading a 140K--250K prompt twice after every response.
    tail = list(turn["prompt"][-128:])
    prompt = _decode(tokenizer, tail)
    raw = _decode(tokenizer, [*tail, *turn["output"]])
    if not raw.startswith(prompt) or not raw.endswith("<|im_end|>"):
        return False
    marker = "<|im_start|>assistant\n"
    generation = next(
        (
            suffix
            for suffix in ("<think>\n\n</think>\n\n", "<think>\n", "")
            if prompt.endswith(marker + suffix)
        ),
        None,
    )
    if generation is None:
        return False
    body = generation + raw[len(prompt) : -len("<|im_end|>")]
    if not body.startswith("<think>\n") or body.count("</think>") != 1:
        return False
    reasoning, body = body[len("<think>\n") :].split("</think>", 1)
    if reasoning.strip("\n") != fields["reasoning"].strip("\n"):
        return False
    calls = fields["tool_calls"]
    if not calls:
        return "<tool_call>" not in body and body.strip("\n") == fields[
            "content"
        ].strip("\n")
    first = body.find("<tool_call>")
    if first < 0 or body[:first].strip("\n") != fields["content"].strip("\n"):
        return False
    body = body[first:]
    for call in calls:
        body = body.lstrip("\n")
        match = re.match(r"<tool_call>\n<function=([A-Za-z_][A-Za-z0-9_.-]*)>\n", body)
        if (
            match is None
            or call["type"] != "function"
            or match[1] != call["function"]["name"]
        ):
            return False
        arguments = _json_arguments(call["function"]["arguments"])
        body = body[match.end() :]
        for key, expected in arguments.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", key):
                return False
            start = f"<parameter={key}>\n"
            if not body.startswith(start):
                return False
            value, delimiter, rest = body[len(start) :].partition("\n</parameter>")
            if not delimiter:
                return False
            if isinstance(expected, str):
                if value != expected:
                    return False
            else:
                # Reuse the duplicate-key/nonfinite/type checks for every
                # nested scalar/array/object without interpreting XML as data.
                wrapped = _json_arguments('{"value":' + value + "}")
                if list(wrapped) != ["value"] or _json_identity(
                    wrapped["value"]
                ) != _json_identity(expected):
                    return False
            body = rest.removeprefix("\n")
        end = "</function>\n</tool_call>"
        if not body.startswith(end):
            return False
        body = body[len(end) :]
    return not body.strip("\n")


def _completed_canonical_prefix(turn, fields):
    context = turn.get("render_context")
    if context is None or not _raw_output_matches(turn, fields):
        return None
    conversation, tools, template_text, kwargs = context
    assistant = {
        "role": "assistant",
        "content": fields["content"],
        "reasoning_content": fields["reasoning"],
        "tool_calls": [
            {
                **call,
                "function": {
                    **call["function"],
                    "arguments": _json_arguments(call["function"]["arguments"]),
                },
            }
            for call in fields["tool_calls"]
        ],
    }
    return _canonical_prefix(
        turn["tokenizer"], [*conversation, assistant], tools, template_text, kwargs
    )


def _get_ledger():
    global _ledger, _ledger_pid, _tokenizers
    if _ledger_pid != os.getpid():
        _ledger = journal.TokenContinuationLedger()
        _ledger_pid = os.getpid()
        _tokenizers = {}
    return _ledger


def _backend_identity(tokenizer):
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return None
    serialized = backend.to_str()
    if not isinstance(serialized, str) or len(serialized) > 64 * 1024 * 1024:
        raise ValueError("unsupported tokenizer backend")
    # Bind the exact artifact serialization. A benign serializer change can
    # conservatively fall back; sorting a 12 MiB vocabulary on every turn would
    # itself recreate first-token latency without strengthening this pin.
    return hashlib.sha256(serialized.encode()).hexdigest()


def _incremental_boundary_supported(tokenizer):
    cached = _tokenizers.get(id(tokenizer))
    backend = _backend_identity(tokenizer)
    # The pinned backend isolates BPE/NFC at the non-normalized im_end special
    # token. Unknown tokenizers keep the ordinary full-render path.
    return (
        backend == INCREMENTAL_BACKEND_SHA256
        and (cached is None or cached[0] is not tokenizer or cached[3] == backend)
        and getattr(tokenizer, "split_special_tokens", True) is False
        and getattr(tokenizer, "clean_up_tokenization_spaces", True) is False
    )


def _tokenizer_identity(tokenizer):
    key = id(tokenizer)
    cached = _tokenizers.get(key)
    backend = _backend_identity(tokenizer)
    if cached is not None and cached[0] is tokenizer and cached[3] == backend:
        return cached[1:3]
    vocabulary = tokenizer.get_vocab()
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if (
        type(end) is not int
        or end not in tokenizer.all_special_ids
        or vocabulary.get("<|im_end|>") != end
    ):
        raise ValueError("chat boundary is not an authenticated special token")
    identity = _digest(
        {
            "class": type(tokenizer).__qualname__,
            "name": getattr(tokenizer, "name_or_path", None),
            "vocabulary": vocabulary,
            "special_tokens": tokenizer.special_tokens_map,
            "backend": backend,
            "split_special_tokens": getattr(tokenizer, "split_special_tokens", None),
            "cleanup": getattr(tokenizer, "clean_up_tokenization_spaces", None),
        }
    )
    if len(_tokenizers) >= 4:
        _tokenizers.clear()
    _tokenizers[key] = (tokenizer, identity, end, backend)
    return identity, end


def _report(state, reason, **counts):
    # Fixed reason codes/counts only. Journals and exceptions never enter logs.
    if state is not None:
        state["token_continuation_reason"] = reason
    try:
        try:
            import qwen_radiance_cache_telemetry as telemetry
        except ModuleNotFoundError:
            from qwen_r9700_lab import radiance_cache_telemetry as telemetry
        now = time.monotonic_ns()
        telemetry.emit_at(
            "token_continuation",
            now,
            now,
            reason=reason,
            **(state or {}).get("identities", {}),
            **counts,
        )
    except Exception:  # noqa: BLE001, S110 - optional telemetry never authorizes a proposal
        pass  # Optional logging does not authorize or invalidate a proposal.


def begin(state, request, tokenizer, config):
    if state is None:
        return
    try:
        # The pinned streaming serving path supplies DELTA output IDs. Other
        # output modes need their own accounting, rather than guessing here.
        if (
            not _get(request, "stream", False)
            or _get(request, "n", 1) != 1
            or _get(request, "use_beam_search", False)
        ):
            return _report(state, "unsupported_output_mode")
        if (
            any(
                _get(request, field)
                for field in (
                    "truncate_prompt_tokens",
                    "pad_prompt_tokens",
                    "continue_final_message",
                    "add_special_tokens",
                    "return_assistant_tokens_mask",
                    "echo",
                )
            )
            or _get(request, "add_generation_prompt", True) is False
        ):
            return _report(state, "unsupported_prompt_mode")
        salt = _get(request, "cache_salt")
        abi = os.environ.get("QWEN_RADIANCE_CACHE_ABI", "")
        if (
            not isinstance(salt, str)
            or not salt
            or len(abi) != 64
            or any(c not in "0123456789abcdef" for c in abi)
        ):
            return _report(state, "missing_identity")
        ledger = _get_ledger()
        tokenizer_id, boundary = _tokenizer_identity(tokenizer)
        identity = journal.ContinuationIdentity(
            cache_salt=salt,
            model=_digest({"model": _get(request, "model"), "data_abi": abi}),
            tokenizer=tokenizer_id,
            template=TEMPLATE_SHA256,
            configuration=_digest(config),
        )
        messages = _plain(_get(request, "messages"))
        if not isinstance(messages, list) or not messages:
            return _report(state, "unsupported_messages")
        for message in messages:
            content = _get(message, "content")
            if (
                content is not None
                and not isinstance(content, str)
                and (
                    not isinstance(content, list)
                    or any(
                        not isinstance(part, dict) or part.get("type") != "text"
                        for part in content
                    )
                )
            ):
                return _report(state, "unsupported_messages")
        history = _digest(messages)
        lease = ledger.begin(identity)
        if lease is None:
            return _report(state, "overlapping_or_unavailable")
        state["token_continuation"] = {
            "ledger": ledger,
            "lease": lease,
            "identity": identity,
            "tokenizer": tokenizer,
            "boundary": boundary,
            "messages": messages,
            "history": history,
            "message_count": len(messages),
            "output": array("i"),
            "content": [],
            "reasoning": [],
            "tools": {},
            "bytes": 0,
            "terminal": False,
            "admitted": False,
            "template_supported": False,
        }
    except Exception:  # noqa: BLE001 - retain canonical input on unsupported journal data
        _report(state, "setup_unavailable")


def template(state, text):
    turn = (state or {}).get("token_continuation")
    if turn is not None:
        turn["template_supported"] = (
            isinstance(text, str)
            and hashlib.sha256(text.encode()).hexdigest() == TEMPLATE_SHA256
        )
        if turn["template_supported"]:
            turn["template_text"] = text


def render_params(state, conversation, kwargs):
    if (state or {}).get("token_continuation") is not None:
        # The pinned HF renderer uses run_in_executor without copying ContextVars.
        # Carry only this request's authentication callback across that boundary.
        # prefix_template removes it before resolving or executing Jinja kwargs.
        kwargs["_coherence_token_template"] = lambda text: template(state, text)

        def encode(*args):
            return encode_prompt(state, *args)

        # OnlineRenderer normally renders text (tokenize=False) before a later
        # tokenization stage. Its prompt parser also accepts validated raw IDs.
        # Only this request-bound session adapter may choose that representation.
        encode._coherence_text_to_tokens = True
        kwargs["_coherence_token_encode"] = encode


def _decode(tokenizer, ids):
    text = tokenizer.decode(
        list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    if not isinstance(text, str) or len(text.encode()) > MAX_TEXT_BYTES:
        raise ValueError("continuation text budget exceeded")
    return text


def _previous(turn, state):
    previous = turn["ledger"].get_record(turn["identity"])
    if previous is None:
        _report(state, "no_previous_journal")
        return None
    count, messages = previous.message_count, turn["messages"]
    if (
        type(count) is not int
        or count >= len(messages)
        or _digest(messages[:count]) != previous.history_identity
    ):
        _report(state, "history_changed")
        return None
    if (
        _get(messages[count], "role") != "assistant"
        or _output_digest(messages[count], previous) != previous.output_identity
    ):
        _report(state, "output_changed")
        return None
    return previous


def _append(turn, suffix, previous):
    return turn["ledger"].append_verified_suffix(
        turn["identity"],
        suffix,
        history_identity=previous.history_identity,
        output_identity=previous.output_identity,
        expected_version=previous.version,
        boundary_token_ids={turn["boundary"]},
        lease=turn["lease"],
    )


def encode_prompt(
    state, tokenizer, conversation, tools, text_template, resolved_kwargs
):
    """Render text for validation, then tokenize only the authenticated suffix.

    Runs inside the actual HF executor, before its full-history tokenization.
    Rendering all text is intentional: a template can retroactively alter old
    thinking or tool framing. Fingerprints alone cannot establish text equality.
    """
    turn = (state or {}).get("token_continuation")
    if turn is None:
        return None
    try:
        if (
            not turn["template_supported"]
            or tokenizer is not turn["tokenizer"]
            or not _incremental_boundary_supported(tokenizer)
        ):
            _report(state, "incremental_unsupported_tokenizer_or_template")
            return None
        kwargs = dict(resolved_kwargs)
        if any(
            key in kwargs
            for key in ("_coherence_token_template", "_coherence_token_encode")
        ):
            raise ValueError("private renderer callback not consumed")
        if kwargs.get("return_dict") not in (None, False):
            return None
        turn["render_context"] = (
            _plain(conversation),
            _plain(tools),
            turn["template_text"],
            {
                key: _plain(value)
                for key, value in kwargs.items()
                if not key.startswith("_coherence_")
            },
        )
        rendered_text = tokenizer.apply_chat_template(
            conversation=conversation,
            tools=tools,
            chat_template=text_template,
            tokenize=False,
            **kwargs,
        )
        if (
            not isinstance(rendered_text, str)
            or len(rendered_text.encode()) > MAX_TEXT_BYTES
        ):
            raise ValueError("unsupported rendered text")
        previous = _previous(turn, state)
        if previous is None:
            return None
        if previous.tokens[-1] != turn["boundary"]:
            _report(state, "unsupported_boundary")
            return None
        old_text = _decode(tokenizer, previous.tokens)
        reconstructed = False
        if rendered_text.startswith(old_text) and len(rendered_text) > len(old_text):
            suffix_text = rendered_text[len(old_text) :]
        else:
            count = previous.message_count
            if (
                previous.canonical_prefix_sha256 is None
                or previous.prompt_tokens is None
                or type(count) is not int
                or not isinstance(conversation, (list, tuple))
                or count >= len(conversation)
                or _get(conversation[count], "role") != "assistant"
            ):
                _report(state, "incremental_rendered_prefix_changed")
                return None
            canonical = _canonical_prefix(
                tokenizer,
                conversation[: count + 1],
                tools,
                turn["template_text"],
                kwargs,
            )
            if (
                len(canonical) != previous.canonical_prefix_characters
                or hashlib.sha256(canonical.encode()).hexdigest()
                != previous.canonical_prefix_sha256
                or not rendered_text.startswith(canonical)
                or len(rendered_text) == len(canonical)
            ):
                _report(state, "canonical_history_changed")
                return None
            suffix_text = rendered_text[len(canonical) :]
            reconstructed = True
        suffix = tokenizer.encode(suffix_text, add_special_tokens=False)
        if _decode(tokenizer, suffix) != suffix_text:
            _report(state, "nonlossless_text_roundtrip")
            return None
        proposal = _append(turn, suffix, previous)
        if proposal.tokens is None:
            _report(state, proposal.reason)
            return None
        ids = list(proposal.tokens)
        if _decode(tokenizer, ids) != old_text + suffix_text:
            _report(state, "incremental_decoded_text_changed")
            return None
        turn["incremental_prompt"] = array("i", ids)
        turn["incremental_suffix"] = array("i", suffix)
        turn["incremental_previous"] = previous
        _report(
            state,
            "rendered_assistant_prefix_preserved"
            if reconstructed
            else "incremental_suffix_encoded",
            previous_tokens=len(previous.tokens),
            suffix_tokens=len(suffix),
            admitted_tokens=len(ids),
        )
        return ids
    except Exception:  # noqa: BLE001 - no IDs were returned; use the original tokenizer
        _report(state, "incremental_unavailable")
        return None


def rendered(state, serving, engine_inputs):
    turn = (state or {}).get("token_continuation")
    if turn is None:
        return engine_inputs
    try:
        if not turn["template_supported"]:
            raise ValueError("unsupported template")
        if len(engine_inputs) != 1 or not isinstance(engine_inputs[0], dict):
            raise ValueError("unsupported input")
        item = engine_inputs[0]
        # Masks/offsets/embeddings cannot be copied across a new segmentation.
        if item.get("type") != "token" or set(item) - {
            "type",
            "prompt_token_ids",
            "prompt",
            "cache_salt",
            "arrival_time",
        }:
            raise ValueError("unsupported input metadata")
        current = serving._extract_prompt_components(item).token_ids
        if (
            not isinstance(current, list)
            or not current
            or len(current) > turn["ledger"].max_tokens
        ):
            raise ValueError("unsupported input length")
        if (
            item.get("cache_salt", turn["identity"].cache_salt)
            != turn["identity"].cache_salt
        ):
            raise ValueError("cache identity mismatch")
        turn["prompt"] = array("i", current)
        incremental = turn.get("incremental_prompt")
        if incremental is not None:
            previous = turn["incremental_previous"]
            proposal = _append(turn, turn["incremental_suffix"], previous)
            max_len = _get(
                _get(serving, "model_config"),
                "max_model_len",
                turn["ledger"].max_tokens,
            )
            if (
                proposal.tokens is None
                or list(incremental) != current
                or list(proposal.tokens) != current
                or len(current) >= max_len
            ):
                raise ValueError("incremental admission changed")
            return engine_inputs
        previous = turn["ledger"].get_record(turn["identity"])
        if previous is None:
            _report(state, "no_previous_journal")
            return engine_inputs
        count = previous.message_count
        messages = turn["messages"]
        if (
            type(count) is not int
            or count >= len(messages)
            or _digest(messages[:count]) != previous.history_identity
        ):
            _report(state, "history_changed")
            return engine_inputs
        # The immediately following message must be the answer actually sent.
        if (
            _get(messages[count], "role") != "assistant"
            or _output_digest(messages[count], previous) != previous.output_identity
        ):
            _report(state, "output_changed")
            return engine_inputs
        old = previous.tokens
        if len(current) >= len(old) and all(a == b for a, b in zip(old, current)):
            _report(
                state,
                "already_exact",
                previous_tokens=len(old),
                current_tokens=len(current),
            )
            return engine_inputs
        tokenizer = turn["tokenizer"]
        raw_text = _decode(tokenizer, old)
        canonical = tokenizer.encode(raw_text, add_special_tokens=False)
        if _decode(tokenizer, canonical) != raw_text:
            _report(state, "nonlossless_text_roundtrip")
            return engine_inputs
        proposal = turn["ledger"].propose(
            turn["identity"],
            current,
            canonical_previous_ids=canonical,
            history_identity=previous.history_identity,
            output_identity=previous.output_identity,
            boundary_token_ids={turn["boundary"]},
            history_unchanged=True,
            output_unchanged=True,
            configuration_unchanged=True,
            lease=turn["lease"],
        )
        if proposal.tokens is None:
            _report(
                state,
                proposal.reason,
                previous_tokens=len(old),
                canonical_tokens=len(canonical),
                current_tokens=len(current),
            )
            return engine_inputs
        replacement = list(proposal.tokens)
        max_len = _get(
            _get(serving, "model_config"), "max_model_len", turn["ledger"].max_tokens
        )
        if len(replacement) >= max_len or _decode(tokenizer, replacement) != _decode(
            tokenizer, current
        ):
            _report(state, "changed_text_or_context_limit")
            return engine_inputs
        result = [{**item, "prompt_token_ids": replacement}]
        turn["prompt"] = array("i", replacement)
        _report(
            state,
            "preserved_generated_tokens",
            previous_tokens=len(old),
            canonical_tokens=len(canonical),
            current_tokens=len(current),
            admitted_tokens=len(replacement),
        )
        return result
    except Exception:  # noqa: BLE001 - retain canonical input on unsupported journal data
        # All proposals were constructed privately. The original input has not
        # been touched, so fallback cannot submit a partly rewritten prompt.
        turn["failed"] = True
        _report(state, "render_unavailable")
        if turn.get("incremental_prompt") is not None:
            # The HF renderer already returned these IDs. Reject instead of
            # silently submitting a proposal whose lease or metadata changed.
            raise ValueError(
                "incremental prompt admission is no longer valid"
            ) from None
        return engine_inputs


def input_processor(state, ids):
    turn = (state or {}).get("token_continuation")
    if turn is not None:
        prompt = turn.get("prompt")
        turn["admitted"] = (
            prompt is not None
            and ids is not None
            and len(prompt) == len(ids)
            and all(a == b for a, b in zip(prompt, ids))
        )
        if not turn["admitted"]:
            turn["failed"] = True
            _report(state, "input_processor_changed")
            if turn.get("incremental_prompt") is not None:
                raise ValueError("incremental prompt changed during input processing")


def output(state, result):
    turn = (state or {}).get("token_continuation")
    if turn is None or turn.get("failed"):
        return
    try:
        outputs = result.outputs
        if len(outputs) != 1 or outputs[0].index != 0 or turn["terminal"]:
            raise ValueError("unsupported output layout")
        row = outputs[0]
        ids = row.token_ids
        if any(type(value) is not int or not 0 <= value <= 2147483647 for value in ids):
            raise ValueError("unsupported output IDs")
        if (
            len(turn.get("prompt", ())) + len(turn["output"]) + len(ids)
            > turn["ledger"].max_tokens
        ):
            raise ValueError("output budget exceeded")
        turn["output"].extend(ids)
        if row.finish_reason is not None:
            turn["terminal"] = row.finish_reason in {"stop", "tool_calls"}
            if not turn["terminal"]:
                turn["failed"] = True
    except Exception:  # noqa: BLE001 - retain canonical input on unsupported journal data
        turn["failed"] = True


def delivered(state, delta):
    turn = (state or {}).get("token_continuation")
    if turn is None or turn.get("failed"):
        return
    try:
        for field in ("content", "reasoning"):
            value = _get(delta, field)
            if field == "reasoning" and value is None:
                value = _get(delta, "reasoning_content")
            if value is not None:
                if not isinstance(value, str):
                    raise TypeError("unsupported delta")
                turn[field].append(value)
                turn["bytes"] += len(value.encode())
        for item in _get(delta, "tool_calls") or ():
            index = _get(item, "index")
            if type(index) is not int or not 0 <= index < 64:
                raise TypeError("unsupported tool index")
            call = turn["tools"].setdefault(
                index,
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            for field in ("id", "type"):
                value = _get(item, field)
                if value is not None:
                    if not isinstance(value, str):
                        raise TypeError("unsupported tool field")
                    call[field] = value
                    turn["bytes"] += len(value.encode())
            for field in ("name", "arguments"):
                value = _get(_get(item, "function") or {}, field)
                if value is not None:
                    if not isinstance(value, str):
                        raise TypeError("unsupported tool function")
                    call["function"][field] += value
                    turn["bytes"] += len(value.encode())
        if turn["bytes"] > MAX_TEXT_BYTES:
            raise ValueError("output metadata budget exceeded")
    except Exception:  # noqa: BLE001 - retain canonical input on unsupported journal data
        turn["failed"] = True


def full_choices(state, choices):
    # Non-streamed output has a different ID accounting contract and is not
    # admitted by begin(). Keep the hook explicit for future support.
    return None


async def prepare_compaction(serving, request, instruction, context_window):
    """Prepare a checkpoint using the same authenticated token history as chat.

    This is CPU-only preparation. It neither submits a model request nor commits
    or replaces the journal. In particular, cancellation of the preceding turn
    leaves its earlier completed journal available for an unchanged prefix.
    """
    try:
        import qwen_radiance_request_timeline as timeline
    except ModuleNotFoundError as error:
        if error.name != "qwen_radiance_request_timeline":
            raise
        from qwen_r9700_lab import radiance_request_timeline as timeline

    if (
        not isinstance(instruction, str)
        or not instruction
        or len(instruction.encode()) > 262_144
        or any(control in instruction for control in ("<|think", "<|im_"))
        or type(context_window) is not int
        or not 1 <= context_window <= 262_144
    ):
        raise ValueError("invalid compaction preparation parameters")
    tokenizer = serving.renderer.tokenizer
    if tokenizer is None:
        raise ValueError("compaction tokenizer is unavailable")
    capacity = min(context_window, serving.model_config.max_model_len)
    history_messages = request.messages
    if not isinstance(history_messages, list) or not history_messages:
        raise ValueError("compaction history is empty")
    continuation_request = request.model_copy(
        update={
            "messages": [*history_messages, {"role": "user", "content": instruction}],
            "stream": True,
            "add_generation_prompt": True,
            "add_special_tokens": False,
        }
    )
    history_request = continuation_request.model_copy(
        update={
            "messages": history_messages,
            "add_generation_prompt": False,
        }
    )
    chat = _get(_get(request, "kv_transfer_params") or {}, "qwen_chat") or {}
    identities = {
        name: value
        for name, value in (
            ("chat_id", _get(chat, "id")),
            ("generation", _get(chat, "generation")),
        )
        if isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    }
    state = {"identities": identities, "observed": set()}
    context_token = timeline._request.set(state)
    turn = None
    try:
        timeline.prefix_begin(
            continuation_request,
            tokenizer,
            serving._effective_chat_template_kwargs(continuation_request),
        )
        turn = state.get("token_continuation")
        if turn is None:
            raise ValueError("compaction continuation contract is unavailable")
        previous = turn["ledger"].get_record(turn["identity"])

        async def render(render_request):
            # The two renderings share a lease and its immutable base version,
            # but each has its own verified suffix and admission checks.
            for name in (
                "incremental_prompt",
                "incremental_suffix",
                "incremental_previous",
            ):
                turn.pop(name, None)
            result = await serving.render_chat_request(render_request)
            if not isinstance(result, tuple) or len(result) != 2:
                raise ValueError("compaction prompt rendering failed")
            if not turn.get("template_supported"):
                raise ValueError("compaction template is not authenticated")
            admitted = timeline.prefix_rendered(serving, result[1])
            if not isinstance(admitted, list) or len(admitted) != 1:
                raise ValueError("compaction needs one text prompt")
            ids = serving._extract_prompt_components(admitted[0]).token_ids
            if (
                not isinstance(ids, list)
                or not ids
                or len(ids) > capacity
                or any(
                    type(value) is not int or not 0 <= value < 2**31 for value in ids
                )
            ):
                raise ValueError("invalid compaction prompt tokens")
            return ids

        history = await render(history_request)
        continuation = await render(continuation_request)
        if len(history) > len(continuation) or history != continuation[: len(history)]:
            raise ValueError("checkpoint instructions changed the historical prompt")

        open_header = tokenizer.encode(
            "<|im_start|>assistant\n<think>\n", add_special_tokens=False
        )
        closed_header = tokenizer.encode(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            add_special_tokens=False,
        )
        if not open_header or not closed_header:
            raise ValueError("compaction generation boundary is unavailable")
        if continuation[-len(closed_header) :] == closed_header:
            prompt = continuation
        elif continuation[-len(open_header) :] == open_header:
            prompt = continuation[: -len(open_header)] + closed_header
        else:
            raise ValueError("unrecognized compaction generation boundary")
        if len(prompt) > capacity:
            raise ValueError("compaction prompt exceeds the model context")
        reason = state.get("token_continuation_reason", "canonical_fallback")
        preserved = 0
        if (
            previous is not None
            and reason
            in {
                "incremental_suffix_encoded",
                "rendered_assistant_prefix_preserved",
                "already_exact",
                "preserved_generated_tokens",
            }
            and list(previous.tokens) == prompt[: len(previous.tokens)]
        ):
            preserved = len(previous.tokens)
            reason = "generated_prefix_preserved"
        _report(
            state,
            "compaction_prefix_prepared",
            previous_tokens=preserved,
            historical_tokens=len(history),
            admitted_tokens=len(prompt),
            preparation_reason=reason,
        )
        return {
            "contract": "radiance-compaction-prefix-v1",
            "prompt": prompt,
            "historical_tokens": len(history),
            "continuation": {
                "reason": reason,
                "preserved_tokens": preserved,
                # Computing this counter must not re-encode retained history.
                "canonical_tokens": None if preserved else len(history),
            },
        }
    finally:
        if turn is not None:
            turn["ledger"].abort(turn["lease"])
        timeline._request.reset(context_token)


def finish(state, complete):
    turn = (state or {}).pop("token_continuation", None)
    if turn is None:
        return
    ledger, lease = turn["ledger"], turn["lease"]
    try:
        ids = turn["output"]
        if not (
            complete
            and turn["terminal"]
            and turn["admitted"]
            and not turn.get("failed")
            and ids
            and ids[-1] == turn["boundary"]
        ):
            ledger.abort(lease)
            return _report(state, "incomplete_answer")
        fields = {
            "content": "".join(turn["content"]),
            "reasoning": "".join(turn["reasoning"]),
            "tool_calls": [turn["tools"][i] for i in sorted(turn["tools"])],
        }
        try:
            canonical = _completed_canonical_prefix(turn, fields)
            if canonical is None and turn.get("render_context") is not None:
                _report(state, "raw_assistant_correspondence_unverified")
        except Exception:  # noqa: BLE001 - a witness is optional, never infer it.
            canonical = None
            _report(state, "canonical_render_unavailable")
        saved = ledger.record_completed(
            lease,
            turn["prompt"],
            ids,
            history_identity=turn["history"],
            output_identity=_output_digest(fields),
            message_count=turn["message_count"],
            processed_tokens=None,
            canonical_prefix_sha256=hashlib.sha256(canonical.encode()).hexdigest()
            if canonical is not None
            else None,
            canonical_prefix_characters=len(canonical)
            if canonical is not None
            else None,
            output_identity_format="json_semantics_v1",
        )
        _report(state, "journal_saved" if saved else "journal_unavailable")
    except Exception:  # noqa: BLE001 - retain canonical input on unsupported journal data
        ledger.abort(lease)
        _report(state, "journal_unavailable")

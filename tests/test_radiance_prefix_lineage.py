from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from qwen_r9700_lab.radiance_prefix_lineage import (
    PrefixLineageObserver,
    compare_sequences,
)


class Tokenizer:
    """A deliberately noncanonical alternative token proves roundtrip detection."""

    def __init__(self):
        self.decode_calls = 0

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False and clean_up_tokenization_spaces is False
        self.decode_calls += 1
        return "".join("a" if value == 999 else chr(value) for value in ids)

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(map(ord, text))


def request(messages=None, *, chat="chat", model="model"):
    return SimpleNamespace(
        cache_salt=chat,
        model=model,
        messages=messages or [{"role": "user", "content": "question"}],
    )


def completed(
    observer,
    tokenizer,
    *,
    output=None,
    content="a",
    reasoning=None,
    complete=True,
    chat="chat",
    config=None,
):
    handle = observer.begin(request(chat=chat), tokenizer, config or {})
    observer.rendered(handle, [80])
    observer.input_processor(handle, [80])
    observer.engine_output(handle, output if output is not None else [97])
    observer.delivered_delta(handle, {"content": content, "reasoning": reasoning})
    observer.finish(handle, complete=complete)


def next_request(content="a", reasoning=None, *, chat="chat"):
    return request(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": content, "reasoning_content": reasoning},
            {"role": "user", "content": "next"},
        ],
        chat=chat,
    )


def latest(rows, comparison):
    return next(row for row in reversed(rows) if row["comparison"] == comparison)


@pytest.mark.parametrize(
    "previous,current,prefix,equal,first",
    [
        ([1, 2], [1, 2], False, True, None),
        ([1, 2], [1, 3], False, False, 1),
        ([1, 2], [1], False, False, 1),
        ([1], [1, 2], True, True, None),
        ([1, 2], [1], True, False, 1),
        ([], [], False, True, None),
    ],
)
def test_exact_first_difference_and_prefix_scope(
    previous, current, prefix, equal, first
):
    result = compare_sequences(previous, current, prefix=prefix)
    assert result["equal"] is equal
    assert result.get("first_difference") == first


def test_matching_prefix_avoids_retokenization_and_matches_delivered_fields():
    rows, tokenizer = [], Tokenizer()
    observer = PrefixLineageObserver(rows.append)
    completed(observer, tokenizer)
    handle = observer.begin(next_request(), tokenizer, {})
    assert latest(rows, "output_to_message")["equal"] is True
    observer.rendered(handle, [80, 97, 78])
    assert latest(rows, "prompt_prefix")["equal"] is True
    assert latest(rows, "raw_to_reencoded")["diagnostic_status"] == "not_applicable"
    assert tokenizer.decode_calls == 0


@pytest.mark.parametrize(
    "noncanonical,trim", [(True, False), (False, True), (True, True)]
)
def test_roundtrip_trim_and_both_causes_are_independently_observed(noncanonical, trim):
    rows, tokenizer = [], Tokenizer()
    observer = PrefixLineageObserver(rows.append)
    output = [999 if noncanonical else 97] + ([32] if trim else [])
    completed(observer, tokenizer, output=output, content="a " if trim else "a")
    handle = observer.begin(next_request(content="a " if trim else "a"), tokenizer, {})
    target = observer.assistant_index(handle)
    observer.template_operation(
        handle, "a " if trim else "a", "a", "content_trim", target
    )
    observer.template_complete(handle, True)
    observer.rendered(handle, [80, 97, 78])
    assert latest(rows, "prompt_prefix")["equal"] is False
    assert latest(rows, "raw_to_reencoded")["equal"] is (not noncanonical)
    aggregate = latest(rows, "template_normalization")
    assert aggregate["trim_changed"] is trim
    assert aggregate["delimiter_changed"] is False
    assert aggregate["comparison_complete"] is True


def test_delimiter_and_trim_flags_are_not_mutually_exclusive():
    rows, tokenizer = [], Tokenizer()
    observer = PrefixLineageObserver(rows.append)
    completed(observer, tokenizer)
    handle = observer.begin(next_request(), tokenizer, {})
    index = observer.assistant_index(handle)
    observer.template_operation(handle, "\na\n", "a", "reasoning_trim", index)
    observer.template_operation(
        handle, "</think>a", "\n</think>\n\na", "assistant_delimiters", index
    )
    result = observer.template_complete(handle, True)
    assert result["trim_changed"] and result["delimiter_changed"]
    assert result["unit"] == "characters"
    assert "first_difference" not in result


def test_other_message_normalization_cannot_be_attributed_to_latest_response():
    rows, tokenizer = [], Tokenizer()
    observer = PrefixLineageObserver(rows.append)
    completed(observer, tokenizer)
    handle = observer.begin(next_request(), tokenizer, {})
    assert (
        observer.template_operation(handle, " old ", "old", "content_trim", 0) is None
    )
    result = observer.template_complete(handle, True)
    assert (
        result["diagnostic_status"] == "unavailable" and result["reason"] == "missing"
    )


def test_missing_and_unsupported_template_observation_are_explicit():
    rows, tokenizer = [], Tokenizer()
    observer = PrefixLineageObserver(rows.append)
    completed(observer, tokenizer)
    handle = observer.begin(next_request(), tokenizer, {})
    assert observer.template_complete(handle, False)["reason"] == "unsupported"
    assert observer.template_complete(handle, True)["reason"] == "missing"


def test_history_and_configuration_changes_do_not_masquerade_as_roundtrip():
    rows, tokenizer = [], Tokenizer()
    observer = PrefixLineageObserver(rows.append)
    completed(observer, tokenizer, config={"preserve_thinking": True})
    changed = next_request()
    changed.messages[0]["content"] = "edited history"
    handle = observer.begin(changed, tokenizer, {"preserve_thinking": False})
    result = latest(rows, "output_to_message")
    assert result["history_changed"] and result["config_changed"]
    observer.rendered(handle, [81, 97, 78])
    assert latest(rows, "raw_to_reencoded")["reason"] == "configuration_changed"
    assert tokenizer.decode_calls == 0


def test_parser_or_client_modification_is_distinct_from_template_change():
    rows, tokenizer = [], Tokenizer()
    observer = PrefixLineageObserver(rows.append)
    completed(observer, tokenizer)
    handle = observer.begin(next_request(content="edited"), tokenizer, {})
    assert latest(rows, "output_to_message")["equal"] is False
    observer.template_operation(handle, "edited", "edited", "content_trim", 1)
    assert observer.template_complete(handle, True)["equal"] is True


@pytest.mark.parametrize("mutated", [False, True])
def test_streamed_tool_deltas_reassemble_and_tool_mutation_is_detected(mutated):
    rows, tokenizer = [], Tokenizer()
    observer = PrefixLineageObserver(rows.append)
    handle = observer.begin(request(), tokenizer, {})
    observer.rendered(handle, [80])
    observer.engine_output(handle, [97])
    observer.delivered_delta(
        handle,
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call",
                    "function": {"name": "read", "arguments": '{"x":'},
                }
            ]
        },
    )
    observer.delivered_delta(
        handle, {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}
    )
    observer.finish(handle)
    following = next_request(content=None)
    following.messages[1]["tool_calls"] = [
        {
            "id": "call",
            "type": "function",
            "function": {"name": "read", "arguments": '{"x":1}'},
        }
    ]
    if mutated:
        following.messages[1]["tool_calls"][0]["function"]["arguments"] = '{"x":2}'
    observer.begin(following, tokenizer, {})
    assert latest(rows, "output_to_message")["equal"] is (not mutated)


def test_input_processor_mutation_is_detected_and_used_as_next_raw_baseline():
    rows, tokenizer = [], Tokenizer()
    observer = PrefixLineageObserver(rows.append)
    handle = observer.begin(request(), tokenizer, {})
    observer.rendered(handle, [80])
    assert observer.input_processor(handle, [81])["equal"] is False
    observer.engine_output(handle, [97])
    observer.delivered_delta(handle, {"content": "a"})
    observer.finish(handle)
    following = observer.begin(next_request(), tokenizer, {})
    assert observer.rendered(following, [81, 97, 78])["equal"] is True


def test_partial_output_is_incomplete_not_a_complete_mismatch():
    rows, tokenizer = [], Tokenizer()
    observer = PrefixLineageObserver(rows.append)
    completed(observer, tokenizer, complete=False)
    handle = observer.begin(next_request(), tokenizer, {})
    assert latest(rows, "output_to_message")["diagnostic_status"] == "incomplete"
    assert observer.rendered(handle, [80, 97, 78])["reason"] == "partial_output"


def test_restart_eviction_and_missing_cache_salt_are_explicit():
    rows, tokenizer = [], Tokenizer()
    observer = PrefixLineageObserver(rows.append, max_chats=1)
    observer.begin(request(chat=None), tokenizer, {})
    assert latest(rows, "output_to_message")["reason"] == "missing_cache_salt"
    completed(observer, tokenizer, chat="first")
    completed(observer, tokenizer, chat="second")
    observer.begin(next_request(chat="first"), tokenizer, {})
    assert latest(rows, "output_to_message")["reason"] == "evicted"
    fresh = PrefixLineageObserver(rows.append)
    fresh.begin(next_request(), tokenizer, {})
    assert latest(rows, "output_to_message")["reason"] == "restart"


def test_cpu_token_list_only_and_capacity_failure_remain_visible():
    rows, tokenizer = [], Tokenizer()
    observer = PrefixLineageObserver(rows.append, max_tokens=2)
    handle = observer.begin(request(), tokenizer, {})
    assert observer.rendered(handle, object())["reason"] == "unsupported"
    observer.finish(handle)
    completed(observer, tokenizer, output=[97, 98])
    handle = observer.begin(next_request(), tokenizer, {})
    assert observer.rendered(handle, [80, 97])["reason"] == "dropped"


def test_payload_memory_and_concurrent_request_count_are_bounded():
    rows, tokenizer = [], Tokenizer()
    observer = PrefixLineageObserver(rows.append, max_bytes=12, max_pending=1)
    first = observer.begin(request(chat="first"), tokenizer, {})
    observer.rendered(first, [80, 81, 82])
    observer.engine_output(first, [97])
    assert observer.retained_bytes <= 12
    second = observer.begin(request(chat="second"), tokenizer, {})
    assert latest(rows, "prompt_prefix")["reason"] == "evicted"
    assert first not in observer._pending and second in observer._pending


def test_request_callbacks_are_correct_across_threads_and_interleaved_chats():
    rows_a, rows_b, tokenizer = [], [], Tokenizer()
    observer = PrefixLineageObserver()
    a = observer.begin(
        request(chat="a"), tokenizer, {}, emit=rows_a.append, request_id="a" * 64
    )
    b = observer.begin(
        request(chat="b"), tokenizer, {}, emit=rows_b.append, request_id="b" * 64
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda handle: observer.rendered(handle, [80]), [a, b]))
    assert rows_a and rows_b
    assert all(row["http_request_id"] == "a" * 64 for row in rows_a)
    assert all(row["http_request_id"] == "b" * 64 for row in rows_b)


def test_same_chat_overlap_is_an_explicit_gap_not_a_stale_completed_comparison():
    rows, tokenizer = [], Tokenizer()
    observer = PrefixLineageObserver(rows.append)
    completed(observer, tokenizer)
    first = observer.begin(next_request(), tokenizer, {})
    second = observer.begin(next_request(), tokenizer, {})
    assert latest(rows, "output_to_message")["reason"] == "overlapping_requests"
    assert observer.rendered(second, [80, 97, 78])["reason"] == "overlapping_requests"
    observer.finish(first)
    observer.finish(second)
    assert observer._completed["chat"].capture_gap == "overlapping_requests"


def test_only_enums_counts_and_booleans_leave_the_observer():
    rows, tokenizer = [], Tokenizer()
    observer = PrefixLineageObserver(rows.append)
    completed(observer, tokenizer, output=[999], content="private marker")
    handle = observer.begin(next_request(content="different secret"), tokenizer, {})
    observer.template_operation(handle, " secret ", "secret", "content_trim", 1)
    observer.template_complete(handle, True)
    observer.rendered(handle, [80, 97, 78])
    serialized = json.dumps(rows)
    assert (
        "private marker" not in serialized
        and "different secret" not in serialized
        and "secret" not in serialized
    )
    assert (
        "999" not in serialized
        and "digest" not in serialized
        and "hash" not in serialized
    )


def test_emitter_and_tokenizer_failures_do_not_raise_or_report_a_false_pass():
    rows, tokenizer = [], Tokenizer()
    observer = PrefixLineageObserver(
        lambda _row: (_ for _ in ()).throw(RuntimeError("secret"))
    )
    completed(observer, tokenizer, output=[999])
    assert observer.dropped > 0
    observer.emit = rows.append
    handle = observer.begin(next_request(), tokenizer, {}, emit=rows.append)
    tokenizer.decode = lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("secret"))
    observer.rendered(handle, [80, 97, 78])
    assert latest(rows, "raw_to_reencoded")["diagnostic_status"] == "unavailable"
    assert latest(rows, "raw_to_reencoded")["dropped_records"] > 0
    assert "secret" not in json.dumps(rows)

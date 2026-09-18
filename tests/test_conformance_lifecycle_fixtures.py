"""Exercise the intended storage boundaries without overflowing the context."""

import pytest

from qwen_r9700_lab import conformance_scenarios as scenarios
from qwen_r9700_lab.diagnostic_contract import DiagnosticError


@pytest.mark.parametrize("blocks", [2, 25, 77, 323])
def test_eviction_capacity_admits_one_head_but_not_two(blocks):
    block_size = 27_000_832
    original = 18 * 1024**3
    capacity = scenarios.eviction_capacity(
        {"head": list(range(blocks)), "verified_block_size": block_size}, original
    )
    assert blocks * block_size < capacity < 2 * blocks * block_size
    assert capacity % block_size == 0
    assert capacity < original


@pytest.mark.parametrize(
    "head,original",
    [
        ({}, 100),
        ({"head": [1], "verified_block_size": 10}, 100),
        ({"head": [1, 2], "verified_block_size": 10}, 30),
    ],
)
def test_eviction_trial_refuses_unmeasured_or_insufficient_capacity(head, original):
    with pytest.raises(DiagnosticError):
        scenarios.eviction_capacity(head, original)


@pytest.mark.parametrize("context", [8192, 60000, 200000, 253535])
def test_interrupted_write_changes_a_full_tail_within_existing_context(monkeypatch, context):
    original = list(range(context))
    seen = []

    def tokens(_spec, count, seed):
        seen.append((count, seed))
        return [context + index for index in range(count)]

    monkeypatch.setattr(scenarios, "tokens", tokens)
    changed = scenarios.interrupted_write_prefix(None, {"seed": 17}, original)
    assert len(changed) == context
    assert changed[:-8192] == original[:-8192]
    assert all(a != b for a, b in zip(changed[-8192:], original[-8192:], strict=True))
    assert seen == [(8192, 17 + 9419)]
    assert original == list(range(context))


def test_interrupted_write_refuses_unchanged_fixture(monkeypatch):
    monkeypatch.setattr(scenarios, "tokens", lambda *_args: [1, 2, 3])
    with pytest.raises(DiagnosticError, match="did not change"):
        scenarios.interrupted_write_prefix(None, {"seed": 0}, [1, 2, 3])


@pytest.fixture
def tail_status():
    return {
        "schema": "urn:qwen-r9700:radiance-tail-residency:v1",
        "chats": [
            {
                "chat_id": "a",
                "generation": "g",
                "tokens": 8192,
                "durable_tokens": 0,
                "blocks": 5,
                "bytes": 12345,
            }
        ],
    }


def test_pending_tail_requires_the_actual_chat_generation(tail_status):
    assert (
        scenarios.pending_tail(tail_status, {"id": "a", "generation": "g"})
        == tail_status["chats"][0]
    )
    assert scenarios.pending_tail(tail_status, {"id": "a", "generation": "old"}) is None
    assert scenarios.pending_tail(tail_status, {"id": "other", "generation": "g"}) is None


@pytest.mark.parametrize("change", [{"durable_tokens": 8192}, {"blocks": 0}, {"bytes": 0}])
def test_already_durable_or_empty_tail_cannot_qualify_shutdown(tail_status, change):
    tail_status["chats"][0].update(change)
    assert scenarios.pending_tail(tail_status, {"id": "a", "generation": "g"}) is None


@pytest.mark.parametrize("bad", [True, -1, None, 1.5, "5"])
def test_malformed_tail_counts_fail_closed(tail_status, bad):
    tail_status["chats"][0]["blocks"] = bad
    with pytest.raises(DiagnosticError, match="pending-tail counts"):
        scenarios.pending_tail(tail_status, {"id": "a", "generation": "g"})


def test_duplicate_tail_cannot_qualify_shutdown(tail_status):
    tail_status["chats"].append(dict(tail_status["chats"][0]))
    with pytest.raises(DiagnosticError, match="duplicate"):
        scenarios.pending_tail(tail_status, {"id": "a", "generation": "g"})


@pytest.mark.parametrize("context", [8192, 60000, 200000, 253535])
def test_shutdown_fixture_advances_a_durable_predecessor_without_crossing_flush_threshold(context):
    prefix = list(range(context))
    seed = scenarios.shutdown_seed_prefix(prefix)
    assert prefix[: len(seed)] == seed
    assert len(prefix) - len(seed) == 2048
    assert prefix == list(range(context))


def test_shutdown_fixture_rejects_an_empty_predecessor():
    with pytest.raises(DiagnosticError, match="durable predecessor"):
        scenarios.shutdown_seed_prefix([1] * 2048)


def shutdown_events():
    return [
        {"execution": "x", "pid": 1, "event": name, "monotonic_ns": when}
        for name, when in [
            ("snapshot.shutdown.enter", 50),
            ("snapshot.publish.return", 55),
            ("snapshot.flush.return", 60),
            ("snapshot.shutdown.return", 70),
        ]
    ]


def test_publication_must_occur_within_this_process_shutdown():
    scenarios.require_shutdown_flush(shutdown_events(), "x")


@pytest.mark.parametrize(
    "change", [{"monotonic_ns": 20}, {"monotonic_ns": 80}, {"pid": 2}, {"execution": "old"}]
)
def test_seed_or_unrelated_publication_cannot_qualify_shutdown(change):
    events = shutdown_events()
    events[1].update(change)
    with pytest.raises(DiagnosticError, match="inside shutdown"):
        scenarios.require_shutdown_flush(events, "x")


@pytest.mark.parametrize("which", [0, 3])
def test_shutdown_requires_both_enter_and_return(which):
    events = shutdown_events()
    del events[which]
    with pytest.raises(DiagnosticError, match="missing or duplicated"):
        scenarios.require_shutdown_flush(events, "x")

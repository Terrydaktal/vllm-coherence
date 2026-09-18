from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from qwen_r9700_lab.conformance_gate import (
    CheckedAuthority,
    ConformanceMismatchError,
    TentativeTransition,
    copy_accepted_prefix,
)
from qwen_r9700_lab.diagnostic_contract import DiagnosticError


def transition(**changes):
    result = TentativeTransition(
        0,
        "a" * 64,
        "b" * 64,
        129,
        b"pending-token",
        {"kv": b"kv-bytes", "gdn": b"gdn-bytes", "conv": b"history"},
        (b"tool-event",),
        "tool_call",
    )
    return replace(result, **changes)


def authority():
    return CheckedAuthority("a" * 64, {"kv", "gdn", "conv"})


@pytest.mark.parametrize(
    "fault", ["kv", "gdn", "conv", "tool", "stop", "pending", "position", "identity"]
)
def test_matching_output_never_hides_wrong_state_or_dropped_tool_event(fault):
    gate = authority()
    reference = transition()
    if fault in {"kv", "gdn", "conv"}:
        candidate = transition(state={**reference.state, fault: b"wrong-version-or-bit"})
    elif fault == "tool":
        candidate = transition(output_events=())
    elif fault == "stop":
        candidate = transition(stop_reason="eos")
    elif fault == "pending":
        candidate = transition(pending_token=None)
    elif fault == "position":
        candidate = transition(consumed_tokens=130)
    else:
        candidate = transition(input_sha256="c" * 64)
    with pytest.raises(ConformanceMismatchError) as caught:
        gate.compare_and_commit(reference, candidate)
    assert gate.revision == 0
    assert not caught.value.report["published"]
    assert "wrong-version-or-bit" not in str(caught.value.report)
    assert gate.compare_and_commit(reference, reference) == (b"tool-event",)
    assert gate.revision == 1


def test_missing_state_is_rejected_even_if_both_adapters_omit_it():
    gate = authority()
    partial = transition(state={"kv": b"kv-bytes"})
    with pytest.raises(ConformanceMismatchError) as caught:
        gate.compare_and_commit(partial, partial)
    assert not caught.value.report["complete_state_coverage"]
    assert gate.revision == 0


def test_tentative_buffers_cannot_mutate_after_check_and_hashes_are_not_the_gate(monkeypatch):
    import qwen_r9700_lab.conformance_gate as module

    monkeypatch.setattr(module, "digest", lambda value: "0" * 64)
    reference = transition()
    with pytest.raises(ConformanceMismatchError):
        authority().compare_and_commit(
            reference, transition(state={**reference.state, "kv": b"bad"})
        )
    with pytest.raises(TypeError):
        reference.state["kv"] = b"changed"
    with pytest.raises(DiagnosticError, match="immutable"):
        transition(state={"kv": bytearray(b"mutable")})


def test_two_same_revision_publications_cannot_both_commit():
    gate = authority()
    reference = transition()

    def publish():
        try:
            gate.compare_and_commit(reference, reference)
            return True
        except ConformanceMismatchError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: publish(), range(2)))
    assert sorted(outcomes) == [False, True]
    assert gate.revision == 1


@pytest.mark.parametrize("count", range(9))
def test_accepted_prefix_retains_no_rejected_suffix(count):
    current, proposed = tuple(range(8)), tuple(range(20, 28))
    result = copy_accepted_prefix(current, proposed, count)
    assert all(result[i] == (20 + i if i < count else i) for i in range(8))
    perturbed = tuple(value if i < count else 999 for i, value in enumerate(proposed))
    assert copy_accepted_prefix(current, perturbed, count) == result


def test_unsupported_sampled_mode_and_failure_stop_are_not_silent_eos():
    with pytest.raises(DiagnosticError, match="distribution"):
        CheckedAuthority("a" * 64, {"kv"}, sampler="sampled")
    with pytest.raises(DiagnosticError, match="not successful"):
        transition(stop_reason="engine_error")
    assert authority().evidence()["formal_backend_equivalence"] == "UNPROVED"

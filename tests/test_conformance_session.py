import numpy as np
import pytest

from qwen_r9700_lab.conformance_session import CheckedSession, SessionMismatchError
from qwen_r9700_lab.conformance_state import FrameWriter
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, authenticate, digest

REQUIRED = ["sequence.tokens", "sequence.position", "gdn", "conv", "kv"]


def frame(root, tokens, pending, *, bad=None):
    writer = FrameWriter(
        root,
        contract=digest("contract"),
        execution=digest("build"),
        adapter=digest("adapter"),
        input_digest=digest(tokens),
        phase="prefill" if len(tokens) == 2 else "step",
        consumed=len(tokens),
        pending=pending,
        expected=REQUIRED,
    )
    writer.array("sequence.tokens", np.asarray(tokens, dtype="<i4"))
    writer.array("sequence.position", np.asarray([len(tokens)], dtype="<i8"))
    for name in REQUIRED[2:]:
        writer.array(name, np.asarray([1 if bad == name else 0], dtype=np.float32))
    return writer.finish()


def commit(session, a, b, revision=0, tokens=(3,), stop=None):
    return session.commit(
        a,
        b,
        base_revision=revision,
        reference_tokens=tokens,
        candidate_tokens=tokens,
        reference_stop=stop,
        candidate_stop=stop,
    )


@pytest.mark.parametrize("corruption", ["gdn", "conv", "kv"])
def test_latent_state_fault_cannot_publish_even_if_token_matches(tmp_path, corruption):
    frame(tmp_path / "a", [1, 2], 3)
    frame(tmp_path / "b", [1, 2], 3, bad=corruption)
    session = CheckedSession(
        tmp_path / "authority",
        contract=digest("contract"),
        required_components=REQUIRED,
        create=True,
    )
    try:
        with pytest.raises(SessionMismatchError) as err:
            commit(session, tmp_path / "a", tmp_path / "b")
        assert err.value.receipt["first_difference"]["boundary"] == corruption
        assert session.outputs_since(0) == [] and session.revision == 0
        assert session.summary()["rejected_transitions"] == 1
    finally:
        session.close()


def test_durable_gate_preserves_pending_and_rejects_stale_writer(tmp_path):
    root = tmp_path / "authority"
    s = CheckedSession(root, contract=digest("contract"), required_components=REQUIRED, create=True)
    for name in ("a", "b"):
        frame(tmp_path / name, [1, 2], 3)
    assert commit(s, tmp_path / "a", tmp_path / "b")["tokens"] == (3,)
    s.close()
    s = CheckedSession(root, contract=digest("contract"), required_components=REQUIRED)
    try:
        for name in ("c", "d"):
            frame(tmp_path / name, [1, 2, 3], 4)
        with pytest.raises(DiagnosticError, match="stale"):
            commit(s, tmp_path / "c", tmp_path / "d", tokens=(4,))
        assert (
            commit(s, tmp_path / "c", tmp_path / "d", revision=1, tokens=(4,), stop="eos")[
                "revision"
            ]
            == 2
        )
        assert s.outputs_since(1) == [{"revision": 2, "tokens": (4,), "stop_reason": "eos"}]
        import json

        for row in s.db.execute("SELECT receipt FROM commits"):
            authenticate(json.loads(row[0]))
    finally:
        s.close()


def test_tool_events_and_errors_cannot_be_forged_as_eos(tmp_path):
    for name in ("a", "b"):
        frame(tmp_path / name, [1, 2], 3)
    s = CheckedSession(
        tmp_path / "authority",
        contract=digest("contract"),
        required_components=REQUIRED,
        create=True,
    )
    try:
        for stop in ("length", "error", "tool_call"):
            with pytest.raises(DiagnosticError):
                commit(s, tmp_path / "a", tmp_path / "b", stop=stop)
        assert s.outputs_since(0) == []
    finally:
        s.close()


def test_invalid_position_cannot_commit_even_when_both_workers_agree(tmp_path):
    s = CheckedSession(
        tmp_path / "authority",
        contract=digest("contract"),
        required_components=REQUIRED,
        create=True,
    )
    try:
        for name in ("a", "b"):
            frame(tmp_path / name, [1, 2], 3)
        commit(s, tmp_path / "a", tmp_path / "b")
        for name in ("c", "d"):
            frame(tmp_path / name, [1, 2, 3, 4], 5)
        with pytest.raises(DiagnosticError, match="conservation"):
            commit(s, tmp_path / "c", tmp_path / "d", revision=1, tokens=(5,))
        assert s.revision == 1
    finally:
        s.close()


@pytest.mark.parametrize("boundary", ["before_commit", "after_commit"])
def test_process_crash_cannot_publish_half_a_transition(tmp_path, boundary):
    import multiprocessing
    import os

    root = tmp_path / "authority"
    s = CheckedSession(root, contract=digest("contract"), required_components=REQUIRED, create=True)
    s.close()
    for name in ("a", "b"):
        frame(tmp_path / name, [1, 2], 3)

    def crash():
        session = CheckedSession(root, contract=digest("contract"), required_components=REQUIRED)
        if boundary == "before_commit":
            session.db.set_trace_callback(lambda sql: os._exit(61) if sql == "COMMIT" else None)
        commit(session, tmp_path / "a", tmp_path / "b")
        os._exit(62)  # commit is durable, but nothing has been delivered externally

    process = multiprocessing.get_context("fork").Process(target=crash)
    process.start()
    process.join(5)
    try:
        assert not process.is_alive()
        assert process.exitcode == (61 if boundary == "before_commit" else 62)
        s = CheckedSession(root, contract=digest("contract"), required_components=REQUIRED)
        try:
            assert s.revision == (0 if boundary == "before_commit" else 1)
            assert len(s.outputs_since(0)) == s.revision
            assert s.db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        finally:
            s.close()
    finally:
        if process.is_alive():
            process.kill()
            process.join(5)
        process.close()


def test_two_real_authority_connections_cannot_commit_the_same_revision(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    root = tmp_path / "authority"
    s = CheckedSession(root, contract=digest("contract"), required_components=REQUIRED, create=True)
    s.close()
    for name in ("a", "b"):
        frame(tmp_path / name, [1, 2], 3)
    barrier = Barrier(2)

    def writer():
        session = CheckedSession(root, contract=digest("contract"), required_components=REQUIRED)
        try:
            barrier.wait(5)
            commit(session, tmp_path / "a", tmp_path / "b")
            return "committed"
        except DiagnosticError as exc:
            assert "stale transition" in str(exc)
            return "rejected"
        finally:
            session.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = [pool.submit(writer) for _ in range(2)]
        assert sorted(task.result(timeout=10) for task in tasks) == ["committed", "rejected"]
    s = CheckedSession(root, contract=digest("contract"), required_components=REQUIRED)
    try:
        assert s.outputs_since(0) == [{"revision": 1, "tokens": (3,), "stop_reason": None}]
    finally:
        s.close()

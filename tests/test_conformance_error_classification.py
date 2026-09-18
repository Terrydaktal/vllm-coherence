"""A transport/harness failure must never qualify corrupt-cache rejection."""

import json
import os
import socket
import sys
from types import SimpleNamespace

import pytest

from qwen_r9700_lab import conformance_scenarios as scenarios
from qwen_r9700_lab.conformance_transport import (
    BackendResponseError,
    OwnedClient,
    OwnedProcess,
    ProtocolError,
)
from qwen_r9700_lab.diagnostic_contract import DiagnosticError

SERVER = r"""
import contextlib, json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_GET(self):
        self.send_response(200); self.end_headers()
        self.wfile.write(b'{"nonce":"fixture","execution":"test"}')
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        mode = body['mode']
        self.send_response(int(mode[4:]) if mode.startswith('http') else 200)
        self.end_headers()
        event = b'{"error":{"message":"synthetic storage failure"}}'
        if mode == 'timeout':
            time.sleep(1)
        elif mode == 'eof':
            self.wfile.write(b': unfinished stream\n\n')
        elif mode == 'malformed':
            self.wfile.write(b'data: {"error":null}\n\n')
        else:
            wire = b'data: '+event+b'\n\n' if body['stream'] else event
            self.wfile.write(wire)
ThreadingHTTPServer(('127.0.0.1',int(sys.argv[1])),Handler).serve_forever()
"""


@pytest.fixture
def client(tmp_path):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    with OwnedProcess(
        [sys.executable, "-c", SERVER, str(port)],
        tmp_path / "process",
        env=dict(os.environ),
        timeout=15,
    ) as child:
        owned = OwnedClient(child, port, "fixture", "test")
        owned.connect(timeout=5)
        yield owned


@pytest.mark.parametrize(
    "mode,stream,expected,status",
    [
        ("http500", False, BackendResponseError, 500),
        ("http400", False, BackendResponseError, 400),
        ("json", False, BackendResponseError, None),
        ("sse", True, BackendResponseError, None),
        ("timeout", False, TimeoutError, None),
        ("eof", True, ProtocolError, None),
        ("malformed", True, ProtocolError, None),
    ],
)
def test_actual_http_classifies_and_preserves_failure(
    client, tmp_path, mode, stream, expected, status
):
    with pytest.raises(expected) as caught:
        client.completion(
            "/v1/completions",
            {"mode": mode, "stream": stream},
            evidence=tmp_path / "reply",
            timeout=0.2,
        )
    assert type(caught.value) is expected
    receipt = json.loads((tmp_path / "reply.failure.json").read_text())
    assert receipt["error_type"] == expected.__name__
    assert receipt["http_status"] == status
    assert receipt["observed_events"] == 0
    assert receipt["response_bytes"] == (tmp_path / "reply.response").stat().st_size
    if mode.startswith("http"):
        assert b"synthetic storage failure" in (tmp_path / "reply.response").read_bytes()
    for name in ["reply.response", "reply.failure.json"]:
        assert (tmp_path / name).stat().st_mode & 0o077 == 0
    assert not (tmp_path / "reply.result.json").exists()


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError("read timed out"),
        ConnectionResetError("connection closed"),
        ProtocolError("stream ended without DONE"),
        ValueError("bad JSON"),
        TypeError("broken fixture"),
        DiagnosticError("owned process failed"),
    ],
)
def test_corruption_check_propagates_client_and_protocol_failures(monkeypatch, failure):
    calls = []

    def generate(*args, **kwargs):
        assert kwargs["timeout"] == 30
        raise failure

    monkeypatch.setattr(scenarios, "captured_generation", generate)
    monkeypatch.setattr(scenarios, "observed", lambda *_: calls.append("observed"))
    with pytest.raises(type(failure)) as caught:
        scenarios.corrupt_restore_recovery(*([None] * 9))
    assert caught.value is failure
    assert calls == []


@pytest.mark.parametrize("status", [None, 500, 400, 401, 404])
@pytest.mark.parametrize("load_observed", [False, True])
def test_corruption_check_requires_server_rejection_and_native_load_error(
    monkeypatch, status, load_observed
):
    def generate(*_args, **_kwargs):
        raise BackendResponseError("fixture", http_status=status)

    def observed(_server, *names):
        assert names == ("snapshot.load.error",)
        if not load_observed:
            raise DiagnosticError("native load failure not observed")

    monkeypatch.setattr(scenarios, "captured_generation", generate)
    monkeypatch.setattr(scenarios, "observed", observed)
    if load_observed and status in [None, 500]:
        result = scenarios.corrupt_restore_recovery(*([None] * 9))
        assert "native_restore_rejected_corruption" in result
    else:
        with pytest.raises(DiagnosticError):
            scenarios.corrupt_restore_recovery(*([None] * 9))


def test_corruption_check_does_not_pass_an_unverified_successful_response(monkeypatch):
    monkeypatch.setattr(scenarios, "captured_generation", lambda *_a, **_k: {"token_ids": [1]})

    def unobserved(*_args):
        raise DiagnosticError("native corruption detection not observed")

    monkeypatch.setattr(scenarios, "observed", unobserved)
    with pytest.raises(DiagnosticError, match="not observed"):
        scenarios.corrupt_restore_recovery(*([None] * 9))


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "output",
        "state",
        "head",
        "publication",
        "bytes",
        "write_event",
        "reload_state",
        "reload_path",
        "repeated_failure",
    ],
)
def test_recomputed_success_requires_state_disk_repair_and_durable_reload(
    tmp_path, monkeypatch, fault
):
    baseline = {"token_ids": [17, 29], "finish_reason": "stop"}
    server = SimpleNamespace(root=tmp_path, executions=["recompute"])
    server.restart = lambda **kwargs: server.executions.append("reload")
    calls = []

    def captured(*args, **kwargs):
        assert kwargs["timeout"] == 30
        calls.append(args[5])
        return {**baseline, "token_ids": [19]} if fault == "output" else baseline

    def compared(left, right):
        wrong = (fault == "state" and right.name == "state-corrupt-restore") or (
            fault == "reload_state" and right.name == "state-repair-reload"
        )
        return {"equal": not wrong}

    def events(*_args):
        rows = [
            {"execution": "recompute", "event": "snapshot.load.error"},
            {"execution": "recompute", "event": "state.captured"},
        ]
        if fault != "write_event":
            rows.append({"execution": "recompute", "event": "snapshot.publish.return"})
        if "reload" in server.executions:
            if fault != "reload_path":
                rows.append({"execution": "reload", "event": "snapshot.load.return"})
            if fault == "repeated_failure":
                rows.append({"execution": "reload", "event": "snapshot.load.error"})
        return rows

    def observed(_server, *names):
        rows = events()
        for name in names:
            scenarios.require(any(row["event"] == name for row in rows), "missing native event")
        return rows

    def read(key, size):
        assert key == "damaged" and size == 4
        if fault == "bytes":
            raise ValueError("bad repaired block")
        return b"good"

    store = SimpleNamespace(
        metadata=lambda: {
            "head": [] if fault == "head" else ["damaged"],
            "publication": {"result": "failed" if fault == "publication" else "committed"},
            "verified_block_size": 4,
        },
        read=read,
    )
    monkeypatch.setattr(scenarios, "captured_generation", captured)
    monkeypatch.setattr(scenarios, "compare_frames", compared)
    monkeypatch.setattr(scenarios, "observed", observed)
    monkeypatch.setattr(scenarios, "read_events", events)
    monkeypatch.setattr(scenarios, "flush", lambda *_args: None)
    args = (server, None, None, None, None, baseline, store, tmp_path, "damaged")
    if fault is None:
        checks = scenarios.corrupt_restore_recovery(*args)
        assert "identical_durable_reload_state_and_output" in checks
        assert calls == ["corrupt-restore", "repair-reload"]
    else:
        with pytest.raises((DiagnosticError, ValueError)):
            scenarios.corrupt_restore_recovery(*args)

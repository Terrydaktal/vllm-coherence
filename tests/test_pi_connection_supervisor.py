"""The offline connection supervisor must fail closed on bad local metadata."""

import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader(
    "pi_connection_supervisor", str(ROOT / "scripts/pi-radiance-connection")
)
spec = importlib.util.spec_from_loader(loader.name, loader)
supervisor = importlib.util.module_from_spec(spec)
loader.exec_module(supervisor)


@pytest.mark.parametrize(
    "models, expected",
    [
        ({"data": [{"id": supervisor.MODEL}]}, True),
        ({"data": [{"id": "wrong-model"}]}, False),
        ({"data": [{"id": supervisor.MODEL}, {"id": "extra"}]}, False),
        ({"data": None}, False),
        ({"data": "malformed"}, False),
        ([], False),
    ],
)
def test_ready_requires_exact_model_contract(models, expected):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(
                json.dumps(models if self.path == "/v1/models" else {}).encode()
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert supervisor.endpoint_ready(server.server_port) is expected
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_private_atomic_publication_rejects_symlink(tmp_path):
    tmp_path.chmod(0o700)
    path = tmp_path / "status.json"
    supervisor.publish(path, {"state": "offline"})
    assert json.loads(path.read_text()) == {"state": "offline"}
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.stat().st_uid == os.getuid()
    path.unlink()
    target = tmp_path / "keep"
    target.write_text("unchanged")
    path.symlink_to(target)
    with pytest.raises(ValueError, match="unsafe connection state file"):
        supervisor.publish(path, {"state": "ready"})
    assert target.read_text() == "unchanged"


def test_diagnostics_do_not_repeat_arbitrary_worker_output():
    text = b"arbitrary secret sentinel 91340 hash mismatch"
    message = supervisor.failure_text(text)
    assert "authentication failed" in message
    assert "91340" not in message
    assert "arbitrary" not in message


@pytest.mark.parametrize(
    "output, kind",
    [
        (
            b"Radiance preflight file hash mismatch: private-sentinel",
            "release_mismatch",
        ),
        (
            b"existing Radiance container differs from the authenticated lane",
            "lane_mismatch",
        ),
        (b"the authenticated Radiance backend is not running", "backend_stopped"),
        (b"ssh: connect to host ai port 22: Connection refused", "host_unreachable"),
        (
            b"ssh: Could not resolve hostname ai: Name or service not known",
            "host_unreachable",
        ),
        (b"Permission denied (publickey)", "authentication_failed"),
        (b"SSH tunnel exited during startup", "tunnel_failed"),
        (b"local Radiance endpoint is unavailable", "tunnel_failed"),
        (b"Radiance startup warmup failed", "warmup_failed"),
        (b"curl: Connection refused", "connection_unavailable"),
        (b"unclassified private-sentinel", "connection_unavailable"),
    ],
)
def test_connection_failure_layer_comes_only_from_observed_worker_evidence(
    output, kind
):
    assert supervisor.failure_kind(output) == kind
    assert "private-sentinel" not in supervisor.failure_text(output)


def test_process_identity_is_bound_to_start_ticks():
    assert supervisor.process_start(os.getpid())
    assert supervisor.process_start(999999999) is None


def test_unready_api_does_not_cut_existing_tunnel_and_recovers(tmp_path):
    """Health probes may block new requests, never kill an answer's transport."""
    tmp_path.chmod(0o700)
    healthy = threading.Event()
    healthy.set()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.send_response(200 if healthy.is_set() else 503)
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": supervisor.MODEL}]}).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    worker = tmp_path / "worker"
    workers = tmp_path / "workers"
    worker.write_text(
        f"#!{sys.executable}\n"
        "import os, time\n"
        f"with open({str(workers)!r}, 'a') as stream: stream.write(str(os.getpid()) + '\\n')\n"
        "print('COHERENCE_PI_CONNECTION_READY', flush=True)\n"
        "time.sleep(60)\n"
    )
    worker.chmod(0o700)
    state = tmp_path / "status.json"
    process = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "scripts/pi-radiance-connection"),
            "--state",
            str(state),
            "--parent-pid",
            str(os.getpid()),
            "--launcher",
            str(worker),
            "--remote",
            "synthetic-test",
            "--local-port",
            str(server.server_port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    def wait_state(expected):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail(f"supervisor exited: {process.stderr.read().decode()}")
            try:
                if json.loads(state.read_text())["state"] == expected:
                    return
            except FileNotFoundError:
                pass
            time.sleep(0.05)
        pytest.fail(f"supervisor did not publish {expected}")

    try:
        wait_state("ready")
        initial = json.loads(state.read_text())
        assert initial["connection_attempt"] == 1
        assert initial["attempt_started_at_unix_ms"] <= initial["last_ready_at_unix_ms"]
        assert "next_retry_at_unix_ms" not in initial
        worker_pid = int(workers.read_text().strip())
        worker_start = supervisor.process_start(worker_pid)
        assert worker_start
        healthy.clear()
        wait_state("offline")
        unavailable = json.loads(state.read_text())
        assert unavailable["failure_kind"] == "health_failed"
        assert (
            unavailable["last_failure_at_unix_ms"] >= initial["last_ready_at_unix_ms"]
        )
        assert unavailable["last_ready_at_unix_ms"] >= initial["last_ready_at_unix_ms"]
        assert "next_retry_at_unix_ms" not in unavailable
        assert supervisor.process_start(worker_pid) == worker_start
        assert workers.read_text().splitlines() == [str(worker_pid)]
        healthy.set()
        wait_state("ready")
        recovered = json.loads(state.read_text())
        assert (
            recovered["last_ready_at_unix_ms"] >= unavailable["last_failure_at_unix_ms"]
        )
        assert recovered["connection_attempt"] == 1
        assert supervisor.process_start(worker_pid) == worker_start
        assert workers.read_text().splitlines() == [str(worker_pid)]
    finally:
        process.terminate()
        process.wait(timeout=8)
        process.stderr.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert supervisor.process_start(worker_pid) != worker_start

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest
from test_radiance_port_allocation import (
    RADIANCE_LAUNCHER,
    _fake_environment,
    _write_executable,
)


def _offline_environment(tmp_path: Path, mode: str) -> tuple[dict[str, str], Path, Path]:
    environment, home, project = _fake_environment(tmp_path)
    environment.update(
        {
            "FAKE_BACKEND_MODE": mode,
            "FAKE_SSH_CALLS": str(tmp_path / "ssh-calls.jsonl"),
            "FAKE_PI_INVOCATION": str(tmp_path / "pi-invocation.json"),
        }
    )
    fake_bin = tmp_path / "bin"
    _write_executable(
        fake_bin / "ssh",
        """\
#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
import time

arguments = sys.argv[1:]
source = "" if "-N" in arguments else sys.stdin.read()
if "-L" in arguments:
    kind = "tunnel"
elif "policy = \\\"pi-startup-warmup-v1\\\"" in source:
    kind = "warmup"
elif "validate_backend()" in source:
    kind = "authenticate-backend"
else:
    kind = "preflight"
with Path(os.environ["FAKE_SSH_CALLS"]).open("a") as output:
    output.write(json.dumps({
        "kind": kind, "arguments": arguments, "pid": os.getpid(),
        "start_ticks": Path("/proc/self/stat").read_text().rpartition(")")[2].split()[19],
    }) + "\\n")
mode = os.environ["FAKE_BACKEND_MODE"]
if mode == "transport-stalled":
    time.sleep(60)
    sys.exit(255)
if mode == "transport-offline":
    print("ssh: connect to host synthetic.invalid port 22: Connection refused", file=sys.stderr)
    sys.exit(255)
if mode == "preflight-mismatch" and kind == "preflight":
    print("Radiance preflight file hash mismatch: synthetic-runtime", file=sys.stderr)
    sys.exit(2)
if kind == "tunnel":
    forward = arguments[arguments.index("-L") + 1]
    port = forward.split(":")[1]
    os.execvp("python3", ["python3", os.environ["FAKE_RADIANCE_SERVER"], port])
if kind == "warmup":
    print(f"{1:064d} {2:064d} hit")
    sys.exit(0)
if kind == "authenticate-backend":
    if mode == "backend-offline":
        print("the authenticated Radiance backend is not running", file=sys.stderr)
        sys.exit(2)
    print("existing")
""",
    )
    _write_executable(
        fake_bin / "pi",
        """\
#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
import time

if sys.argv[1:] == ["--version"]:
    print(os.environ.get("FAKE_PI_VERSION", "0.84.2"))
    sys.exit(0)
connection_path = Path(os.environ["QWEN_RADIANCE_CONNECTION_STATE"])
invocation = {
    "arguments": sys.argv[1:],
    "environment": {
        key: value for key, value in os.environ.items()
        if key.startswith("QWEN_RADIANCE_") or key in ("PI_CODING_AGENT_DIR", "PI_OFFLINE")
    },
    "connection": json.loads(connection_path.read_text()),
    "connection_mode": connection_path.stat().st_mode & 0o777,
}
invocation_path = Path(os.environ["FAKE_PI_INVOCATION"])
invocation_path.write_text(json.dumps(invocation))
if os.environ.get("FAKE_PI_WAIT_FOR_SSH") == "1":
    deadline = time.monotonic() + 5
    calls = Path(os.environ["FAKE_SSH_CALLS"])
    while not (calls.exists() and calls.read_text().endswith("\\n")):
        if time.monotonic() >= deadline:
            sys.exit(97)
        time.sleep(0.02)
if os.environ.get("FAKE_PI_WAIT_FOR_READY") == "1":
    deadline = time.monotonic() + 10
    while True:
        state = json.loads(connection_path.read_text())
        if state["state"] == "ready":
            invocation["connection_after_wait"] = state
            invocation_path.write_text(json.dumps(invocation))
            break
        if time.monotonic() >= deadline:
            sys.exit(98)
        time.sleep(0.02)
""",
    )
    return environment, home, project


def _run_launcher(environment: dict[str, str], project: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(RADIANCE_LAUNCHER), "--reuse-existing", "--no-install", *arguments],
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )


def _same_process_alive(pid: int, start_ticks: str) -> bool:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rpartition(")")[2].split()
    except FileNotFoundError:
        return False
    return fields[0] not in {"Z", "X"} and fields[19] == start_ticks


@pytest.mark.parametrize("mode", ["transport-offline", "backend-offline", "preflight-mismatch"])
@pytest.mark.parametrize("selector", ["absolute", "last"])
def test_reuse_opens_exact_saved_session_when_backend_is_offline(
    tmp_path: Path, mode: str, selector: str
) -> None:
    environment, _, project = _offline_environment(tmp_path, mode)
    sessions = project / ".pi" / "sessions"
    sessions.mkdir(parents=True)
    session = sessions / "2026-11-14T00-00-00-000Z_aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
    original = b'{"type":"session","id":"synthetic-offline-session"}\n'
    session.write_bytes(original)
    before = session.stat()

    result = _run_launcher(
        environment, project, "--session", str(session) if selector == "absolute" else "last"
    )

    assert result.returncode == 0, result.stderr
    invocation = json.loads(Path(environment["FAKE_PI_INVOCATION"]).read_text(encoding="utf-8"))
    arguments = invocation["arguments"]
    assert arguments[arguments.index("--session") + 1] == str(session)
    assert arguments[arguments.index("--session-dir") + 1] == str(sessions)
    assert "--no-context-files" in arguments
    assert invocation["environment"]["PI_OFFLINE"] == "1"
    assert invocation["connection"]["schema"] == "coherence.pi.backend-connection.v1"
    assert invocation["connection"]["state"] in {"offline", "checking"}
    assert invocation["connection"]["local_port"] == int(
        invocation["environment"]["QWEN_RADIANCE_LOCAL_PORT"]
    )
    assert invocation["connection_mode"] == 0o600
    supervisor_pid = invocation["connection"]["supervisor_pid"]
    supervisor_start = invocation["connection"]["supervisor_start_ticks"]
    assert supervisor_pid >= 2
    assert isinstance(supervisor_start, str) and supervisor_start.isdecimal()
    assert not _same_process_alive(supervisor_pid, supervisor_start)
    assert session.read_bytes() == original
    assert (session.stat().st_dev, session.stat().st_ino) == (before.st_dev, before.st_ino)
    calls_file = Path(environment["FAKE_SSH_CALLS"])
    calls = [json.loads(line) for line in calls_file.read_text(encoding="utf-8").splitlines()] if calls_file.exists() else []
    assert not any(call["kind"] == "warmup" for call in calls)


def test_offline_preflight_still_fails_without_starting_pi(tmp_path: Path) -> None:
    environment, _, project = _offline_environment(tmp_path, "transport-offline")

    result = _run_launcher(environment, project, "--preflight-only")

    assert result.returncode != 0
    assert not Path(environment["FAKE_PI_INVOCATION"]).exists()


def test_offline_reuse_does_not_bypass_the_existing_pinned_pi(tmp_path: Path) -> None:
    environment, _, project = _offline_environment(tmp_path, "transport-offline")
    environment["FAKE_PI_VERSION"] = "0.84.1"

    result = _run_launcher(environment, project)

    assert result.returncode != 0
    assert not Path(environment["FAKE_PI_INVOCATION"]).exists()


def test_reuse_opens_pi_during_stalled_ssh_and_cleans_up_its_worker(tmp_path: Path) -> None:
    environment, _, project = _offline_environment(tmp_path, "transport-stalled")
    environment["FAKE_PI_WAIT_FOR_SSH"] = "1"
    calls_file = Path(environment["FAKE_SSH_CALLS"])
    try:
        started = time.monotonic()
        result = _run_launcher(environment, project)
        assert result.returncode == 0, result.stderr
        assert time.monotonic() - started < 10
        assert Path(environment["FAKE_PI_INVOCATION"]).exists()
        calls = [json.loads(line) for line in calls_file.read_text(encoding="utf-8").splitlines()]
        assert calls
        for call in calls:
            assert not _same_process_alive(call["pid"], call["start_ticks"])
    finally:
        if calls_file.exists():
            for line in calls_file.read_text(encoding="utf-8").splitlines():
                call = json.loads(line)
                if _same_process_alive(call["pid"], call["start_ticks"]):
                    try:
                        os.kill(call["pid"], signal.SIGKILL)
                    except ProcessLookupError:
                        pass


def test_reuse_publishes_ready_only_after_the_authenticated_worker_is_ready(tmp_path: Path) -> None:
    environment, _, project = _offline_environment(tmp_path, "online")
    environment["FAKE_PI_WAIT_FOR_READY"] = "1"

    result = _run_launcher(environment, project)

    assert result.returncode == 0, result.stderr
    invocation = json.loads(Path(environment["FAKE_PI_INVOCATION"]).read_text(encoding="utf-8"))
    assert invocation["connection"]["state"] in {"offline", "checking"}
    ready = invocation["connection_after_wait"]
    assert ready["state"] == "ready"
    assert ready["local_port"] == int(invocation["environment"]["QWEN_RADIANCE_LOCAL_PORT"])
    kinds = [
        json.loads(line)["kind"]
        for line in Path(environment["FAKE_SSH_CALLS"]).read_text(encoding="utf-8").splitlines()
    ]
    assert kinds == ["preflight", "authenticate-backend", "tunnel", "warmup"]
    assert not _same_process_alive(ready["supervisor_pid"], ready["supervisor_start_ticks"])

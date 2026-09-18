from __future__ import annotations

import fcntl
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RADIANCE_LAUNCHER = ROOT / "scripts" / "pi-remote-qwen-radiance"
MAIN_LAUNCHER = ROOT / "scripts" / "pi-remote-qwen"


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o700)


def _wait_for_ports(path: Path, count: int, processes: list[subprocess.Popen[str]]) -> list[int]:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if path.exists():
            ports = [int(line) for line in path.read_text(encoding="utf-8").splitlines()]
            if len(ports) >= count:
                return ports
        failed = [process.returncode for process in processes if process.poll() is not None]
        if failed:
            raise AssertionError(f"launcher exited before Pi started: {failed}")
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {count} Pi launchers")


def _fake_environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    home = tmp_path / "home"
    fake_bin = tmp_path / "bin"
    project = tmp_path / "project"
    home.mkdir()
    fake_bin.mkdir()
    project.mkdir()
    runtime_directory = tmp_path / "runtime"
    runtime_directory.mkdir(mode=0o700)

    server = tmp_path / "radiance_server.py"
    server.write_text(
        """\
import json
import socket
import sys

model = "qwen3.8-27b-uncensored-mxfp4-public-snapshot-candidate"
listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("127.0.0.1", int(sys.argv[1])))
listener.listen()
while True:
    connection, _ = listener.accept()
    with connection:
        request = b""
        while b"\\r\\n\\r\\n" not in request:
            chunk = connection.recv(4096)
            if not chunk:
                break
            request += chunk
        if b" /v1/models " in request:
            body = json.dumps({"data": [{"id": model}]}).encode()
        else:
            body = b"{}"
        response = (
            b"HTTP/1.1 200 OK\\r\\nContent-Type: application/json\\r\\n"
            + f"Content-Length: {len(body)}\\r\\nConnection: close\\r\\n\\r\\n".encode()
            + body
        )
        connection.sendall(response)
""",
        encoding="utf-8",
    )

    _write_executable(
        fake_bin / "ssh",
        """\
#!/usr/bin/env bash
set -euo pipefail
arguments=" $* "
forward=''
while (($#)); do
    case $1 in
    -L)
        forward=$2
        shift 2
        ;;
    *)
        shift
        ;;
    esac
done
if [[ -n $forward ]]; then
    port=${forward#127.0.0.1:}
    port=${port%%:*}
    exec python3 "$FAKE_RADIANCE_SERVER" "$port"
fi
cat >/dev/null
if [[ $arguments == *" qwen38-27b-uncensored-mxfp4-public-snapshot-candidate "* ]]; then
    printf 'existing\\n'
fi
""",
    )
    fake_pi = fake_bin / "pi"
    _write_executable(
        fake_pi,
        """\
#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --version ]]; then
    printf '0.84.2\\n'
    exit 0
fi
printf '%s\\n' "$QWEN_RADIANCE_LOCAL_PORT" >>"$HOME/selected-ports"
printf '%s\\n' "$QWEN_RADIANCE_GPU_TEMPERATURE_STATE" >>"$HOME/temperature-states"
while [[ ! -e $HOME/release-pi ]]; do
    sleep 0.05
done
""",
    )
    local_bin = home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    (local_bin / "pi").symlink_to(fake_pi)

    searchtool = tmp_path / "searchtool.mjs"
    searchtool.write_text("export default function () {}\n", encoding="utf-8")

    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(home),
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "FAKE_RADIANCE_SERVER": str(server),
            "XDG_RUNTIME_DIR": str(runtime_directory),
            "QWEN_PI_SEARCHTOOL_EXTENSION": str(searchtool),
        }
    )
    return environment, home, project


def test_omitted_radiance_port_stays_automatic_through_main_launcher(tmp_path: Path) -> None:
    environment, _, project = _fake_environment(tmp_path)
    direct = subprocess.run(
        [str(RADIANCE_LAUNCHER), "--dry-run"],
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    delegated = subprocess.run(
        [str(MAIN_LAUNCHER), "--model", "radiance-uncensored", "--dry-run"],
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert direct.returncode == 0, direct.stderr
    assert delegated.returncode == 0, delegated.stderr
    expected = "local API:    automatic; first free port in 8012..8099"
    assert expected in direct.stdout
    assert expected in delegated.stdout


def test_concurrent_radiance_launchers_reserve_distinct_ports(tmp_path: Path) -> None:
    environment, home, project = _fake_environment(tmp_path)
    selected = home / "selected-ports"
    command = [str(RADIANCE_LAUNCHER), "--reuse-existing", "--no-install"]
    lock_root = (
        home / ".local/state/qwen-r9700/pi-remote/radiance-public-clean-candidate/port-locks"
    )
    lock_root.mkdir(mode=0o700, parents=True)
    blocked_port = next(
        port
        for port in range(8012, 8100)
        if not subprocess.run(
            ["ss", "-H", "-ltn", f"sport = :{port}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    blocked_path = lock_root / f"{blocked_port}.lock"
    blocked_path.touch(mode=0o600)
    blocked_lock = blocked_path.open("a", encoding="utf-8")
    fcntl.flock(blocked_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    processes: list[subprocess.Popen[str]] = []
    outputs: list[tuple[str, str]] = []
    try:
        first = subprocess.Popen(
            command,
            cwd=project,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(first)
        _wait_for_ports(selected, 1, processes)

        second = subprocess.Popen(
            command,
            cwd=project,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(second)
        ports = _wait_for_ports(selected, 2, processes)
    finally:
        (home / "release-pi").touch()
        for process in processes:
            try:
                outputs.append(process.communicate(timeout=15))
            except subprocess.TimeoutExpired:
                process.kill()
                outputs.append(process.communicate(timeout=5))
        blocked_lock.close()

    assert all(process.returncode == 0 for process in processes), outputs
    assert len(set(ports)) == 2
    assert blocked_port not in ports
    assert all(8012 <= port <= 8099 for port in ports)
    temperature_states = (home / "temperature-states").read_text(encoding="utf-8").splitlines()
    assert len(temperature_states) == 2
    assert len(set(temperature_states)) == 1
    assert temperature_states[0].startswith(str(tmp_path / "runtime"))

    for port, (_, stderr) in zip(ports, outputs, strict=True):
        assert f"Radiance local tunnel reserved: 127.0.0.1:{port} (automatic)" in stderr
        assert f"local API 127.0.0.1:{port}" in stderr
        lock_path = lock_root / f"{port}.lock"
        assert lock_path.stat().st_mode & 0o777 == 0o600
        with lock_path.open("a", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)

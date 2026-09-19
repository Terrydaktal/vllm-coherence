from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "qwen-radiance-gpu-temperature"


def _process_start_ticks() -> str:
    stat = Path("/proc/self/stat").read_text(encoding="utf-8")
    return stat[stat.rfind(")") + 2 :].split()[19]


def test_concurrent_clients_share_one_persistent_remote_probe(tmp_path: Path) -> None:
    state = tmp_path / "state"
    clients = state / "clients"
    clients.mkdir(parents=True, mode=0o700)
    state.chmod(0o700)
    marker = clients / f"{os.getpid()}-{_process_start_ticks()}-0123456789abcdef.heartbeat"
    marker.touch(mode=0o600)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    probe_log = tmp_path / "probe.log"
    interval_log = tmp_path / "interval.log"
    ssh = fake_bin / "ssh"
    ssh.write_text(
        """#!/usr/bin/env bash
	set -euo pipefail
	cat >/dev/null
	printf 'probe\\n' >>"$PROBE_LOG"
	printf '%s\\n' "${!#}" >"$INTERVAL_LOG"
	printf '34000\\t36000\\t25\\n'
sleep 0.5
printf '35000\\t37000\\t31\\n'
""",
        encoding="utf-8",
    )
    ssh.chmod(0o700)
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["PROBE_LOG"] = str(probe_log)
    environment["INTERVAL_LOG"] = str(interval_log)

    for _ in range(2):
        subprocess.run(
            [str(HELPER), "ensure", str(state), "lewis@ai"],
            check=True,
            env=environment,
        )

    deadline = time.monotonic() + 3
    sample_path = state / "sample.json"
    while not sample_path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert sample_path.exists()
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    assert sample["edge_millicelsius"] == 34_000
    assert sample["junction_millicelsius"] == 36_000
    assert sample["fan_percent"] == 25
    assert sample["schema"] == "urn:qwen-r9700:gpu-temperature:v2"
    assert probe_log.read_text(encoding="utf-8").splitlines() == ["probe"]
    assert interval_log.read_text(encoding="utf-8").strip() == "1"

    marker.unlink()
    lock_path = state / "monitor.lock"
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        with lock_path.open("a", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(0.02)
                continue
            break
    else:
        raise AssertionError("shared GPU temperature monitor did not stop after its final client")

    subprocess.run(
        [str(HELPER), "ensure", str(state), "lewis@ai"],
        check=True,
        env=environment,
    )
    time.sleep(0.1)
    assert probe_log.read_text(encoding="utf-8").splitlines() == ["probe"]

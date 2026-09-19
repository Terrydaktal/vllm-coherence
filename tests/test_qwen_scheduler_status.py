from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "qwen-radiance-scheduler-status"


def _process_start_ticks() -> str:
    stat = Path("/proc/self/stat").read_text(encoding="utf-8")
    return stat[stat.rfind(")") + 2 :].split()[19]


@pytest.mark.parametrize("container", ["vllm-coherence", "legacy-radiance-fixture", None])
def test_concurrent_clients_share_one_content_free_scheduler_probe(tmp_path: Path, container) -> None:
    state = tmp_path / "state"
    clients = state / "clients"
    clients.mkdir(parents=True, mode=0o700)
    state.chmod(0o700)
    marker = clients / f"{os.getpid()}-{_process_start_ticks()}-0123456789abcdef.heartbeat"
    marker.touch(mode=0o600)
    (state / "sample.json").write_text(
        json.dumps({
            "schema": "urn:qwen-r9700:gpu-temperature:v2",
            "observed_at_ms": int(time.time() * 1000),
            "edge_millicelsius": 35_000,
            "junction_millicelsius": 42_000,
            "fan_percent": 31,
        }),
        encoding="utf-8",
    )
    (state / "sample.json").chmod(0o600)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    probe_log = tmp_path / "probe.log"
    interval_log = tmp_path / "interval.log"
    ssh = fake_bin / "ssh"
    ssh.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
import time
from pathlib import Path

sys.stdin.read()
with Path(os.environ['PROBE_LOG']).open('a') as out:
    print('probe', file=out)
start = sys.argv.index('-u') + 2
Path(os.environ['INTERVAL_LOG']).write_text(sys.argv[start])
Path(os.environ['NAMESPACE_LOG']).write_text(sys.argv[start + 2])
Path(os.environ['CONTAINER_LOG']).write_text(sys.argv[sys.argv.index('--container') + 1])
scheduler = {
    'pid': 1234, 'updated_at': time.time(), 'quantum_seconds': 30.0,
    'switches': 7, 'cached_chats': 2, 'max_cached_chats': 5,
    'requests': [{'chat_id': '0' * 64, 'generation': '0' * 64, 'state': 'paused',
                  'computed_tokens': 52635, 'input_tokens': 60000}],
}
print(json.dumps(scheduler), 'null',
      json.dumps({'schema': 'urn:qwen-r9700:cache-residency:v1', 'chats': []}),
      json.dumps({'schema': 'urn:qwen-r9700:cache-residency:v2', 'chats': []}),
      sep='\t', flush=True)
time.sleep(0.5)
scheduler.update(updated_at=time.time(), switches=8, requests=[])
print(json.dumps(scheduler), 'null', sep='\t', flush=True)
""",
        encoding="utf-8",
    )
    ssh.chmod(0o700)
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["PROBE_LOG"] = str(probe_log)
    environment["INTERVAL_LOG"] = str(interval_log)
    namespace_log = tmp_path / "namespace.log"
    environment["NAMESPACE_LOG"] = str(namespace_log)
    container_log = tmp_path / "container.log"
    environment["CONTAINER_LOG"] = str(container_log)
    environment.pop("QWEN_RADIANCE_CONTAINER", None)
    if container is not None:
        environment["QWEN_RADIANCE_CONTAINER"] = container

    for abi in ("7" * 64, "6" * 64):
        environment["QWEN_RADIANCE_CACHE_ABI"] = abi
        subprocess.run(
            [str(HELPER), "ensure", str(state), "lewis@ai"],
            check=True,
            env=environment,
        )

    deadline = time.monotonic() + 3
    sample_path = state / "scheduler-v2.json"
    while not sample_path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert sample_path.exists()
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    assert sample["schema"] == "urn:qwen-r9700:scheduler-telemetry:v2"
    assert set(sample) == {"schema", "observed_at_ms", "backend"}
    assert sample["backend"]["scheduler"]["requests"][0]["state"] == "paused"
    assert sample["backend"]["worker"] is None
    coverage = json.loads((state / "cache-residency.json").read_text(encoding="utf-8"))
    assert coverage == {"schema": "urn:qwen-r9700:cache-residency:v1", "chats": []}
    coverage_v2 = json.loads((state / "cache-residency-v2.json").read_text(encoding="utf-8"))
    assert coverage_v2 == {"schema": "urn:qwen-r9700:cache-residency:v2", "chats": []}
    combined = json.loads((state / "telemetry-v1.json").read_text(encoding="utf-8"))
    assert combined["schema"] == "urn:qwen-r9700:telemetry:v1"
    assert set(combined) == {
        "schema", "observed_at_ms", "scheduler", "worker", "phases", "cache", "temperature",
    }
    assert combined["scheduler"]["requests"][0]["state"] == "paused"
    assert combined["worker"] is None
    assert combined["temperature"]["junction_millicelsius"] == 42_000
    assert set(sample["backend"]["scheduler"]["requests"][0]) == {
        "chat_id",
        "generation",
        "state",
        "computed_tokens",
        "input_tokens",
    }
    legacy = json.loads((state / "scheduler.json").read_text(encoding="utf-8"))
    assert legacy["schema"] == "urn:qwen-r9700:scheduler-telemetry:v1"
    assert set(legacy) == {"schema", "observed_at_ms", "scheduler"}
    assert probe_log.read_text(encoding="utf-8").splitlines() == ["probe"]
    assert interval_log.read_text(encoding="utf-8").strip() == "0.5"
    assert namespace_log.read_text(encoding="utf-8") == "auto"
    expected = container or "qwen38-27b-uncensored-mxfp4-public-snapshot-candidate"
    assert container_log.read_text(encoding="utf-8") == expected

    marker.unlink()
    lock_path = state / "scheduler-monitor.lock"
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
        raise AssertionError("shared scheduler monitor did not stop after its final client")

    subprocess.run(
        [str(HELPER), "ensure", str(state), "lewis@ai"],
        check=True,
        env=environment,
    )
    time.sleep(0.1)
    assert probe_log.read_text(encoding="utf-8").splitlines() == ["probe"]

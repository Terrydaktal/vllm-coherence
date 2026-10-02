"""Synthetic shared telemetry and real terminal coverage; no inference requests."""

import fcntl
import hashlib
import importlib.util
import json
import os
import pty
import select
import signal
import struct
import subprocess
import sys
import termios
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_r9700_lab import radiance_cache_cli as cli
from qwen_r9700_lab import radiance_cache_live as live

ROOT = Path(__file__).resolve().parents[1]
CHAT, GENERATION, ABI = "a" * 64, "b" * 64, "c" * 64


def arguments(tmp_path, **overrides):
    return SimpleNamespace(**{
        "host": "local", "cache_root": str(tmp_path), "abi": None, "chat": None,
        "telemetry_state": str(tmp_path / "telemetry"), "interval": 0.1,
        "inventory_interval": 30.0, "json": False, "count": 0, **overrides,
    })


def sample(now_ms=None):
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    return {
        "schema": live.SCHEMA, "observed_at_ms": now_ms,
        "scheduler": {"pid": 123, "requests": [], "updated_at": now_ms / 1000},
        "worker": {"pid": 123, "allocated_bytes": 2**30, "cached_chats": 1},
        "cache": {
            "schema": "urn:qwen-r9700:cache-residency:v2", "observed_at_ms": now_ms,
            "abi": ABI, "live": True, "complete": True,
            "chats": [{"chat_id": CHAT, "generation": GENERATION, "gpu_tokens": 10_000,
                       "ram_tokens": 0, "disk_tokens": 9_000, "disk_saved_tokens": 9_000,
                       "input_tokens": 10_010}],
        },
        "phases": {
            "pid": 123, "updated_at": now_ms / 1000, "recent": [],
            "requests": [{"chat_id": CHAT, "generation": GENERATION, "phase": "generate",
                          "input_tokens": 10_010, "last_round_ms": 43.5, "acceptance_rate_3s": 0.6}],
        },
        "temperature": {"observed_at_ms": now_ms, "junction_millicelsius": 90_000,
                        "edge_millicelsius": 50_000, "fan_percent": 40},
    }


def report():
    return {
        "chats": [{"id": CHAT, "abi": ABI, "consistent": True, "expected_blocks": 2,
                   "issues": [], "metadata": {"generation": GENERATION, "tokens": 9_000,
                                               "title": "Synthetic code work", "cwd": "/fixture"},
                   "session": {"generation": GENERATION, "last_turn_tokens": 10_010},
                   "active_processes": [], "totals": {"file_bytes": 1_000_000},
                   "io": {"available": True, "written_file_bytes": 2_000_000}}],
        "unsnapshotted_chats": [], "issues": [], "totals": {"file_bytes": 1_000_000},
        "io": {"written_file_bytes": 2_000_000}, "filesystem": {"available_bytes": 3_000_000},
    }


def write_sample(path, value=None):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(json.dumps(value or sample()))
    path.chmod(0o600)
    return path


def test_snapshot_refresh_uses_latest_file_and_same_three_second_acceptance(tmp_path):
    path = write_sample(tmp_path / "telemetry-v1.json")
    first = live.build_rows(report(), live.read_sample(path))[0]
    assert (first["gpu"], first["ram"], first["disk_saved"], first["cold"]) == (10_000, 0, 9_000, 10)
    assert first["state"] == "Generating"
    assert first["round_ms"] == 43.5
    assert first["acceptance"] == 0.6
    changed = sample()
    changed["cache"]["chats"][0]["gpu_tokens"] = 10_005
    changed["phases"]["requests"][0].update(last_round_ms=45.7, acceptance_rate_3s=0.72)
    # Production writes are atomic renames, not modifications to an open file.
    replacement = write_sample(tmp_path / "new.json", changed)
    replacement.replace(path)
    second = live.build_rows(report(), live.read_sample(path))[0]
    assert (second["gpu"], second["cold"], second["round_ms"], second["acceptance"]) == (10_005, 5, 45.7, 0.72)


@pytest.mark.parametrize("fault", ["old", "future", "schema", "cache_count", "duplicate", "phase_array", "scheduler_array"])
def test_bad_or_stale_snapshot_is_rejected(tmp_path, fault):
    value = sample()
    if fault == "old":
        value["observed_at_ms"] -= 6_000
    elif fault == "future":
        value["observed_at_ms"] += 6_000
    elif fault == "schema":
        value["schema"] = "unknown"
    elif fault == "cache_count":
        value["cache"]["chats"][0]["gpu_tokens"] = -1
    elif fault == "duplicate":
        value["cache"]["chats"] *= 2
    elif fault == "phase_array":
        value["phases"]["requests"] = None
    else:
        value["scheduler"]["requests"] = "not an array"
    with pytest.raises(ValueError):
        live.read_sample(write_sample(tmp_path / "telemetry.json", value))


@pytest.mark.parametrize("fault", ["symlink", "hardlink", "permissions", "oversized", "fifo"])
def test_unsafe_sample_is_not_read(tmp_path, fault):
    path = write_sample(tmp_path / "sample.json")
    if fault == "symlink":
        alias = tmp_path / "alias.json"
        alias.symlink_to(path)
        path = alias
    elif fault == "hardlink":
        os.link(path, tmp_path / "alias.json")
    elif fault == "permissions":
        path.chmod(0o666)
    elif fault == "oversized":
        path.write_bytes(b" " * (256 * 1024 + 1))
    else:
        path.unlink()
        os.mkfifo(path)
    with pytest.raises((OSError, ValueError)):
        live.read_sample(path)


def test_compaction_does_not_borrow_previous_generation_context_or_statistics():
    value = sample()
    new_generation = "d" * 64
    value["cache"]["chats"][0].update(generation=new_generation, gpu_tokens=1_000,
                                       disk_tokens=0, disk_saved_tokens=0, input_tokens=1_010)
    value["phases"]["requests"][0].update(generation=new_generation, input_tokens=1_010,
                                           last_round_ms=None, acceptance_rate_3s=None)
    row = live.build_rows(report(), value)[0]
    assert row["generation"] == new_generation
    assert (row["gpu"], row["cold"], row["disk_saved"]) == (1_000, 10, 0)
    assert row["round_ms"] is None
    assert row["acceptance"] is None
    # Also cover the gap after completion before the next metadata scan.
    value["phases"]["requests"] = []
    assert live.build_rows(report(), value)[0]["cold"] == 10


def test_stale_nested_cache_or_other_abi_does_not_look_live():
    value = sample()
    value["cache"]["observed_at_ms"] -= 6_000
    row = live.build_rows(report(), value)[0]
    assert row["gpu"] is None
    value = sample()
    value["cache"]["abi"] = "e" * 64
    row = live.build_rows(report(), value)[0]
    assert row["gpu"] is None and row["round_ms"] is None


@pytest.mark.parametrize("fault", ["pid", "age"])
def test_rounds_from_stale_or_different_backend_are_not_displayed(fault):
    value = sample()
    if fault == "pid":
        value["phases"]["pid"] = 456
    else:
        value["phases"]["updated_at"] -= 31
    row = live.build_rows(report(), value)[0]
    assert row["round_ms"] is None
    assert row["acceptance"] is None


def test_unknown_generation_and_genuinely_absent_cache_are_distinguished():
    value = sample()
    value["cache"]["chats"] = []
    row = live.build_rows(report(), value)[0]
    assert (row["gpu"], row["ram"], row["cold"]) == (0, 0, 10_010)
    assert live.build_rows(report(), None)[0]["gpu"] is None
    row = live.build_rows(None, sample())[0]
    assert row["gpu"] == 10_000 and row["disk_bytes"] is None


def test_queue_names_blocker_and_does_not_relabel_admission_as_cold_fill():
    value = sample()
    value["phases"]["requests"][0].update(phase="gpu_queue", blocker={
        "chat_id": "f" * 64, "generation": "e" * 64,
    })
    row = live.build_rows(report(), value)[0]
    assert row["state"] == "Queued for GPU · ffffffffffff"
    assert row["gpu"] == 10_000


def test_coverage_partition_matches_pi_for_varied_memory_and_disk_states():
    cases = []
    for is_live in (False, True):
        for complete in (False, True):
            for gpu, ram, disk in ((None, None, None), (0, 0, 0), (700, 0, 500), (0, 900, 500), (300, 600, 900)):
                for context in (None, 1_000):
                    cache = {"schema": "urn:qwen-r9700:cache-residency:v2", "complete": complete,
                             "live": is_live, "chats": [{"chat_id": CHAT, "generation": GENERATION,
                             "gpu_tokens": gpu, "ram_tokens": ram, "disk_tokens": disk,
                             "disk_saved_tokens": disk, "input_tokens": context}]}
                    cases.append([cache, context])
    result = subprocess.run([
        "node", "--input-type=module", "-e",
        ("import {cacheBreakdown} from './integrations/pi/qwen-cache-residency.mjs';"
         "let data=''; for await (const part of process.stdin) data+=part;"
         "console.log(JSON.stringify(JSON.parse(data).map(([s,n])=>"
         f"cacheBreakdown(s,{{id:'{CHAT}',generation:'{GENERATION}'}},n))));"),
    ], cwd=ROOT, input=json.dumps(cases), capture_output=True, text=True, check=True)
    reference = json.loads(result.stdout)
    for (cache, context), expected in zip(cases, reference, strict=True):
        actual = live.cache_breakdown(cache, cache["chats"][0], context)
        assert actual == {"gpu": expected["gpu"], "ram": expected["ram"],
                          "cold": expected["cold"], "disk_saved": expected["diskSaved"]}


def test_client_joins_pi_state_and_preserves_existing_window_marker(tmp_path, monkeypatch):
    state = tmp_path / "telemetry"
    state.mkdir(mode=0o700)
    clients = state / "clients"
    clients.mkdir(mode=0o700)
    existing = clients / "pi-window.heartbeat"
    existing.write_bytes(b"")
    existing.chmod(0o600)
    launches = []

    def spawn(command, **kwargs):
        launches.append(command)
        return SimpleNamespace(poll=lambda: 0, wait=lambda **_: 0)

    monkeypatch.setattr(live.subprocess, "Popen", spawn)
    with live.SharedTelemetry(arguments(tmp_path)) as reader:
        marker = reader.marker
        assert marker.exists() and marker.stat().st_mode & 0o777 == 0o600
        assert marker.name.startswith(f"{os.getpid()}-")
        for _ in range(50):
            reader.read()
        assert existing.exists()
        assert len(launches) == 2  # Not a pair of probes per display tick.
        assert all(command[1:] == ["ensure", str(state), "local"] for command in launches)
    assert existing.exists() and not marker.exists()


def test_failed_monitor_start_keeps_valid_feed_without_fast_retry_storm(tmp_path, monkeypatch):
    state = tmp_path / "telemetry"
    write_sample(state / "telemetry-v1.json")
    (state / "clients").mkdir(mode=0o700)
    attempts = []

    def unavailable(command, **_kwargs):
        attempts.append(command)
        raise OSError("helper unavailable")

    monkeypatch.setattr(live.subprocess, "Popen", unavailable)
    with live.SharedTelemetry(arguments(tmp_path)) as reader:
        for _ in range(50):
            assert reader.read()[0]["schema"] == live.SCHEMA
        assert len(attempts) == 2


def test_default_state_path_is_the_same_host_hash_as_legacy_pi(tmp_path, monkeypatch):
    monkeypatch.delenv("QWEN_RADIANCE_GPU_TEMPERATURE_STATE", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    args = arguments(tmp_path, host="ai", telemetry_state=None)
    reader = live.SharedTelemetry(args)
    assert reader.directory == tmp_path / "qwen-radiance-gpu-temperature" / hashlib.sha256(b"ai").hexdigest()


def test_portable_cache_uses_same_state_and_deployment_as_portable_pi(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        spec = importlib.util.spec_from_file_location("portable_cache_fixture", ROOT / "tools/coherence_cli.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(ROOT / "tools"))
    connection = {"cache_root": str(tmp_path / "cache"), "abi": ABI}
    (tmp_path / "connection.json").write_text(json.dumps(connection))
    calls = []
    monkeypatch.setattr(module.subprocess, "call", lambda command, **kwargs: calls.append((command, kwargs)) or 0)
    assert module.main(["cache", "--state", str(tmp_path)]) == 0
    command, options = calls[0]
    expected = tmp_path / "telemetry" / hashlib.sha256(("local" + connection["cache_root"]).encode()).hexdigest()[:16]
    assert command[command.index("--telemetry-state") + 1] == str(expected)
    assert expected.stat().st_mode & 0o777 == 0o700
    assert (expected / "clients").is_dir()
    assert options["env"]["QWEN_RADIANCE_CONTAINER"] == "vllm-coherence"


def test_unsafe_state_is_refused_without_starting_probe(tmp_path, monkeypatch):
    path = tmp_path / "telemetry"
    path.mkdir(mode=0o755)
    monkeypatch.setattr(live.subprocess, "Popen", lambda *_a, **_k: pytest.fail("unsafe probe launch"))
    with live.SharedTelemetry(arguments(tmp_path)) as reader:
        assert reader.read()[0] is None
        assert reader.error and reader.marker is None


def test_background_scan_is_single_and_cannot_block_fast_json_frames(tmp_path, monkeypatch, capsys):
    unblock = threading.Event()
    calls = []

    def collect(_args):
        calls.append("scan")
        unblock.wait(5)
        return report()

    class Reader:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            return sample(), None

    monkeypatch.setattr(live, "SharedTelemetry", lambda _: Reader())
    started = time.monotonic()
    try:
        assert live.watch(arguments(tmp_path, json=True, count=3), collect) == 0
        assert time.monotonic() - started < 1
        frames = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        assert len(frames) == 3 and calls == ["scan"]
        assert all(frame["inventory"] is None and frame["chats"][0]["gpu"] == 10_000 for frame in frames)
    finally:
        unblock.set()


def test_keyboard_interrupt_cleans_up_only_dashboard_client(tmp_path, monkeypatch):
    state = tmp_path / "telemetry"
    state.mkdir(mode=0o700)
    (state / "clients").mkdir(mode=0o700)
    monkeypatch.setattr(live.subprocess, "Popen", lambda *_a, **_k: SimpleNamespace(poll=lambda: 0, wait=lambda **_: 0))
    monkeypatch.setattr(live.time, "sleep", lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        live.watch(arguments(tmp_path, json=True), lambda _: report())
    assert not list((state / "clients").iterdir())


def test_dashboard_counters_labels_and_terminal_content_are_unambiguous(tmp_path):
    value = report()
    value["chats"][0]["metadata"]["title"] = "fixture\x1b[2J"
    inventory = SimpleNamespace(completed_at=time.monotonic(), error=None, thread=None)
    lines, _, footer = live.dashboard_lines(value, sample(), None, inventory, arguments(tmp_path))
    output = "\n".join([*lines, footer])
    assert "100 ms refresh" in output
    assert "43.5" in output and "60.0%" in output
    assert "90°C · 50°C · 40%" in output
    assert "Disk tok = saved backup" in output
    assert "every 30s" in output and "\x1b" not in output


@pytest.mark.parametrize("arguments_extra,live_expected", [([], True), (["status"], True), (["--once"], False), (["--json"], False)])
def test_terminal_default_is_live_but_machine_output_stays_one_shot(monkeypatch, arguments_extra, live_expected):
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    invoked = []
    monkeypatch.setattr(live, "watch", lambda *_: invoked.append(True) or 0)
    monkeypatch.setattr(cli, "collect", lambda *_: (_ for _ in ()).throw(ValueError("single scan")))
    assert cli.main(arguments_extra) == (0 if live_expected else 2)
    assert bool(invoked) == live_expected


def test_fullscreen_real_pty_can_resize_scroll_and_exit_without_leaking_terminal_state(tmp_path):
    state = tmp_path / "telemetry"
    write_sample(state / "telemetry-v1.json")
    (state / "clients").mkdir(mode=0o700)
    master, slave = pty.openpty()
    original = termios.tcgetattr(slave)
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 180, 0, 0))
    env = {**os.environ, "TERM": "xterm-256color", "QWEN_RADIANCE_SCHEDULER_HELPER": "/usr/bin/true",
           "QWEN_RADIANCE_GPU_TEMPERATURE_HELPER": "/usr/bin/true"}
    child = subprocess.Popen([
        sys.executable, str(ROOT / "scripts/qwen-radiance-cache"), "--host", "local",
        "--cache-root", str(tmp_path), "--no-sessions", "--telemetry-state", str(state),
    ], stdin=slave, stdout=slave, stderr=slave, env=env)
    try:
        output = b""
        deadline = time.monotonic() + 5
        while b"100 ms refresh" not in output and time.monotonic() < deadline:
            if select.select([master], [], [], 0.2)[0]:
                output += os.read(master, 65536)
        assert b"100 ms refresh" in output
        # Exercise a small resize and all navigation paths before quitting.
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 8, 50, 0, 0))
        child.send_signal(signal.SIGWINCH)
        os.write(master, b"\x1bOB\x1bOC\x1b[6~ir")
        time.sleep(0.15)
        os.write(master, b"q")
        assert child.wait(timeout=5) == 0
        assert termios.tcgetattr(slave) == original
        assert not list((state / "clients").glob("*.heartbeat"))
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)
        os.close(master)
        os.close(slave)

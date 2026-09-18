from __future__ import annotations

import json
import shlex
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_r9700_lab import radiance_cache_cli as cli
from qwen_r9700_lab import radiance_memory as memory


class Storage:
    def __init__(self, address, size):
        self.address, self.size = address, size

    def nbytes(self):
        return self.size

    def data_ptr(self):
        return self.address


class Tensor:
    def __init__(self, storage, device="cuda"):
        self.storage = storage
        self.device = SimpleNamespace(type=device, index=0)

    def untyped_storage(self):
        return self.storage

    def __repr__(self):
        raise AssertionError("tensor values must never be formatted")

    def __getattr__(self, name):
        raise AssertionError(f"unexpected tensor access: {name}")


class Module:
    def __init__(self, parameters=(), children=()):
        self._parameters = dict(enumerate(parameters))
        self._modules = dict(enumerate(children))


class Runner:
    def __init__(self):
        shared = Storage(0xDEADBEEF1234, 200)
        self.device = "already_initialized_cuda_0"
        self.kv_caches = [Tensor(Storage(100, 1000))]
        self.model = Module([Tensor(shared)], [Module([Tensor(Storage(200, 300))])])
        self.speculator = SimpleNamespace(model=Module([Tensor(shared), Tensor(Storage(300, 400))]))
        self.input_buffers = SimpleNamespace(
            gpu=Tensor(Storage(400, 50)),
            cpu=Tensor(Storage(500, 100000), "cpu"),
        )
        self.req_states = SimpleNamespace(private_session="PRIVATE_CHAT_MUST_NOT_APPEAR")


def fake_torch(read=None):
    def stats(device):
        assert device == "already_initialized_cuda_0"
        return {
            "allocated_bytes.all.current": 2200,
            "reserved_bytes.all.current": 3000,
            "active_bytes.all.current": 2300,
            "inactive_split_bytes.all.current": 150,
            "allocated_bytes.all.peak": 2500,
        }

    return SimpleNamespace(
        Tensor=Tensor,
        nn=SimpleNamespace(Module=Module),
        cuda=SimpleNamespace(memory_stats=read or stats),
    )


def test_inventory_counts_shared_storage_once_and_only_reads_metadata():
    runner = Runner()
    inventory = memory.buffer_inventory(runner, fake_torch())
    assert inventory["known_storage_bytes"] == 1950
    assert inventory["groups"]["model"]["bytes"] == 500
    assert inventory["groups"]["draft_model"]["bytes"] == 400
    assert inventory["groups"]["input_buffers"]["bytes"] == 50
    encoded = json.dumps(inventory)
    assert "PRIVATE_CHAT" not in encoded
    assert str(0xDEADBEEF1234) not in encoded
    assert not inventory["truncated"]
    bounded = memory.buffer_inventory(runner, fake_torch(), max_nodes=2)
    assert bounded["truncated"]
    assert bounded["known_storage_bytes"] <= inventory["known_storage_bytes"]


def test_allocator_reports_pending_frees_and_missing_counters_honestly():
    counts = memory.allocator_counts(fake_torch().cuda.memory_stats("already_initialized_cuda_0"))
    assert counts["unused_reserved_bytes"] == 800
    assert counts["pending_free_bytes"] == 100
    assert counts["inactive_split_bytes"] == 150
    assert counts["allocation_retries"] is None
    assert "reclaimable_bytes" not in counts
    assert all(value is None for value in memory.allocator_counts({}).values())


def test_single_reporter_background_refresh_and_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "INTERVAL", 0.03)
    runner = Runner()
    reporter = memory.MemoryReporter(runner, fake_torch(), tmp_path)
    reporter.start()
    try:
        first = memory.read_report(tmp_path)
        assert first["available"]
        assert first["sequence"] == 1
        another = memory.MemoryReporter(runner, fake_torch(), tmp_path)
        with pytest.raises(BlockingIOError):
            another.start()
        requested = memory.refresh_report(tmp_path, timeout=1)
        assert requested["available"]
        assert requested["inventory"]["requested_at_ns"] > 0
        assert requested["inventory"]["collected_at"] >= first["inventory"]["collected_at"]
        assert {p.name for p in tmp_path.iterdir()} == {"report.json", "refresh.json", "owner.lock"}
        assert (tmp_path / "report.json").stat().st_mode & 0o777 == 0o600
        assert (tmp_path / "report.json").stat().st_size < 5000
    finally:
        reporter.stop()
    assert not reporter.thread.is_alive()
    assert reporter.lock_fd is None


def test_normal_samples_do_not_rescan_buffers(tmp_path, monkeypatch):
    runner = Runner()
    reporter = memory.MemoryReporter(runner, fake_torch(), tmp_path)
    reporter.sample()
    initial = reporter.inventory
    monkeypatch.setattr(memory, "buffer_inventory", lambda *_: pytest.fail("unexpected rescan"))
    reporter.sample()
    assert reporter.inventory is initial
    assert memory.read_report(tmp_path)["sequence"] == 2


def test_refresh_requests_are_coalesced_and_previous_instance_is_ignored(tmp_path):
    runner = Runner()
    reporter = memory.MemoryReporter(runner, fake_torch(), tmp_path)
    reporter.sample()
    memory.atomic_json(tmp_path / "refresh.json", {"instance_id": "a" * 32, "requested_at_ns": 999})
    reporter.sample()
    assert reporter.last_refresh == 0
    for requested in (100, 200, 300):
        memory.atomic_json(
            tmp_path / "refresh.json",
            {"instance_id": reporter.instance_id, "requested_at_ns": requested},
        )
    reporter.sample()
    assert reporter.last_refresh == 300


def test_report_reader_strips_unknown_fields_and_detects_stale_metadata(tmp_path):
    runner = Runner()
    reporter = memory.MemoryReporter(runner, fake_torch(), tmp_path)
    reporter.sample()
    payload = memory.read_json(tmp_path / "report.json")
    payload["private"] = "PRIVATE_CHAT"
    payload["allocator"]["private"] = "PRIVATE_CHAT"
    payload["inventory"]["groups"]["model"]["private"] = "PRIVATE_CHAT"
    memory.atomic_json(tmp_path / "report.json", payload)
    assert "PRIVATE_CHAT" not in json.dumps(memory.read_report(tmp_path))
    payload["updated_at"] = time.time() - 20
    memory.atomic_json(tmp_path / "report.json", payload)
    assert memory.read_report(tmp_path)["state"] == "stale"


def test_reader_refuses_symlinks_and_unsafe_or_missing_files(tmp_path):
    assert memory.read_report(tmp_path)["state"] == "not_enabled"
    private = tmp_path / "private"
    private.write_text("PRIVATE_CHAT")
    (tmp_path / "report.json").symlink_to(private)
    assert memory.read_report(tmp_path)["state"] == "unavailable"
    (tmp_path / "report.json").unlink()
    tmp_path.chmod(0o755)
    assert memory.read_report(tmp_path)["state"] == "unavailable"


def test_collector_failure_does_not_escape_background_thread(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "INTERVAL", 0.02)
    runner = Runner()
    reporter = memory.MemoryReporter(runner, fake_torch(), tmp_path)
    reporter.start()
    called = threading.Event()

    def fail():
        called.set()
        raise RuntimeError("PRIVATE_CHAT")

    monkeypatch.setattr(reporter, "sample", fail)
    try:
        assert called.wait(1)
        assert reporter.thread.is_alive()
        assert "PRIVATE_CHAT" not in (tmp_path / "report.json").read_text()
    finally:
        reporter.stop()


def test_memory_command_never_discovers_sessions_or_scans_snapshots(monkeypatch, capsys):
    monkeypatch.setattr(cli, "collect", lambda *_: pytest.fail("snapshot scan"))
    monkeypatch.setattr(cli, "discover_sessions", lambda *_: pytest.fail("chat discovery"))
    monkeypatch.setattr(memory, "read_report", lambda: {"available": False, "state": "not_enabled"})
    assert cli.main(["memory", "--host", "local", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["state"] == "not_enabled"


def test_memory_command_renders_inventory_and_missing_stats(tmp_path, monkeypatch, capsys):
    runner = Runner()
    reporter = memory.MemoryReporter(runner, fake_torch(), tmp_path)
    reporter.sample()
    monkeypatch.setattr(cli, "collect_memory", lambda _: memory.read_report(tmp_path))
    assert cli.main(["memory"]) == 0
    output = capsys.readouterr().out
    assert "Unused reservation" in output
    assert "not guaranteed reclaimable" in output
    assert "Draft model weights and buffers" in output
    assert "PRIVATE_CHAT" not in output


def test_runtime_source_has_no_gpu_work_or_heap_profiling_calls():
    import ast

    tree = ast.parse(Path(memory.__file__).read_text())
    prohibited = {
        "synchronize",
        "empty_cache",
        "reset_peak_memory_stats",
        "_snapshot",
        "_record_memory_history",
        "_dump_snapshot",
        "collect",
        "get_objects",
        "tolist",
        "numpy",
        "item",
        "cpu",
        "cuda",
        "copy_",
        "clone",
        "zeros",
        "empty",
        "get_memory_info",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in prohibited
    snapshots = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "memory_snapshot"
    ]
    assert len(snapshots) == 1
    assert any(
        item.arg == "include_traces" and item.value.value is False for item in snapshots[0].keywords
    )


def map_fixture():
    mib = 1024**2
    base = 0xDEADBEEF12340000
    runner = Runner()
    runner.device = SimpleNamespace(type="cuda", index=0)
    runner.kv_caches = []
    runner.model = Module([Tensor(Storage(base, 128 * mib))])
    runner.speculator = SimpleNamespace(model=Module([Tensor(Storage(base + 448 * mib, 64 * mib))]))
    runner.input_buffers = None

    def segment(offset, pool, blocks, device=0):
        result = {
            "device": device,
            "address": base + offset * mib,
            "total_size": sum(size for size, _ in blocks) * mib,
            "blocks": [
                {
                    "size": size * mib,
                    "state": state,
                    "frames": [{"name": "PRIVATE_CHAT_MUST_NOT_APPEAR"}],
                }
                for size, state in blocks
            ],
        }
        if pool is not None:
            result["segment_pool_id"] = pool
        return result

    segments = [
        segment(
            0, (0, 0), [(128, "active_allocated"), (320, "inactive"), (64, "active_allocated")]
        ),
        segment(1024, (0, 0), [(384, "inactive")]),
        segment(2048, (0, 1), [(256, "inactive")]),
        segment(3072, None, [(64, "active_awaiting_free"), (64, "inactive")]),
        segment(4096, (0, 0), [(1000, "inactive")], device=1),
    ]
    torch = fake_torch(read=lambda _: {"allocated_bytes.all.current": 192 * mib})
    calls = []

    def snapshot(*, include_traces):
        assert include_traces is False
        calls.append(True)
        return segments

    torch.cuda.memory_snapshot = snapshot
    return runner, torch, segments, calls


def test_map_separates_fragmentation_private_pools_and_pending_frees_without_contents():
    runner, torch, _, calls = map_fixture()
    report = memory.allocation_map(runner, torch)
    mib = 1024**2
    assert calls == [True]
    assert report["totals"]["segments"] == 4
    assert report["totals"]["inactive_bytes"] == 1024 * mib
    assert report["totals"]["fragmented_bytes"] == 384 * mib
    assert report["totals"]["pending_bytes"] == 64 * mib
    assert report["pools"]["default"]["fully_inactive_bytes"] == 384 * mib
    assert report["pools"]["private"]["fully_inactive_bytes"] == 256 * mib
    assert report["pools"]["unknown"]["largest_free_block_bytes"] == 64 * mib
    first = report["segments"][0]
    assert first["owners"]["model"] == 128 * mib
    assert first["owners"]["draft_model"] == 64 * mib
    assert first["unattributed_allocated_bytes"] == 0
    assert sum(row["bytes"] for row in report["free_block_histogram"]) == 1024 * mib
    encoded = json.dumps(report)
    for forbidden in (
        "PRIVATE_CHAT",
        "frames",
        "address",
        str(0xDEADBEEF12340000),
        "reclaimable_bytes",
    ):
        assert forbidden not in encoded


def test_map_reports_partial_scan_and_unmatched_owners():
    _, _, segments, _ = map_fixture()
    report = memory.summarize_allocation_map(segments, {}, 0, max_segments=1)
    assert report["truncated"]
    assert report["totals"]["segments"] == 1
    assert report["segments"][0]["unattributed_allocated_bytes"] == 192 * 1024**2
    limited = memory.summarize_allocation_map(segments, {}, 0, max_blocks=1)
    assert limited["truncated"] and limited["totals"]["segments"] == 0
    segments[0]["blocks"][0]["state"] = "unrecognised"
    with pytest.raises(ValueError):
        memory.summarize_allocation_map(segments, {}, 0)


def test_map_is_on_demand_and_shared_across_clients(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "INTERVAL", 0.02)
    runner, torch, _, calls = map_fixture()
    reporter = memory.MemoryReporter(runner, torch, tmp_path)
    reporter.start()
    try:
        assert calls == []
        first = memory.request_allocation_map(tmp_path, timeout=1)
        assert first["available"] and calls == [True]
        again = memory.request_allocation_map(tmp_path, timeout=1)
        assert again["reused"] and calls == [True]
        assert again["collected_at"] == first["collected_at"]
        memory.atomic_json(
            tmp_path / "map-request.json",
            {
                "instance_id": reporter.instance_id,
                "requested_at_ns": time.time_ns(),
            },
        )
        sequence = memory.read_report(tmp_path)["sequence"]
        deadline = time.monotonic() + 1
        while memory.read_report(tmp_path)["sequence"] <= sequence and time.monotonic() < deadline:
            time.sleep(0.01)
        assert memory.read_report(tmp_path)["sequence"] > sequence
        assert calls == [True]
        assert (tmp_path / "map.json").stat().st_mode & 0o777 == 0o600
        assert (tmp_path / "map.json").stat().st_size < memory.MAX_REPORT_BYTES
    finally:
        reporter.stop()


def test_map_failure_is_isolated_and_exception_contents_are_not_published(tmp_path):
    runner, torch, _, _ = map_fixture()
    reporter = memory.MemoryReporter(runner, torch, tmp_path)
    reporter.sample()

    def fail(**_):
        raise ValueError("PRIVATE_CHAT")

    torch.cuda.memory_snapshot = fail
    memory.atomic_json(
        tmp_path / "map-request.json",
        {
            "instance_id": reporter.instance_id,
            "requested_at_ns": time.time_ns(),
        },
    )
    reporter.sample()
    assert memory.read_report(tmp_path)["available"]
    assert (
        memory.read_allocation_map(tmp_path, reporter.instance_id)["state"] == "map_capture_failed"
    )
    assert "PRIVATE_CHAT" not in (tmp_path / "map.json").read_text()
    assert memory.request_allocation_map(tmp_path)["state"] == "map_capture_failed"
    reporter.sample()
    assert memory.read_report(tmp_path)["sequence"] == 3


def test_map_reader_strips_unknown_data_and_rejects_other_instances(tmp_path):
    runner, torch, _, _ = map_fixture()
    reporter = memory.MemoryReporter(runner, torch, tmp_path)
    reporter.sample()
    memory.atomic_json(
        tmp_path / "map-request.json",
        {
            "instance_id": reporter.instance_id,
            "requested_at_ns": time.time_ns(),
        },
    )
    reporter.sample()
    payload = memory.read_json(tmp_path / "map.json")
    payload["frames"] = "PRIVATE_CHAT"
    payload["segments"][0]["address"] = "PRIVATE_CHAT"
    payload["segments"][0]["owners"]["PRIVATE_CHAT"] = 100
    memory.atomic_json(tmp_path / "map.json", payload)
    assert "PRIVATE_CHAT" not in json.dumps(
        memory.read_allocation_map(tmp_path, reporter.instance_id)
    )
    assert not memory.read_allocation_map(tmp_path, "a" * 32)["available"]
    payload["segments"][0]["pool"] = "PRIVATE_CHAT"
    memory.atomic_json(tmp_path / "map.json", payload)
    assert not memory.read_allocation_map(tmp_path, reporter.instance_id)["available"]


def test_old_backend_map_request_is_read_only_and_cli_avoids_chats(tmp_path, monkeypatch, capsys):
    runner = Runner()
    reporter = memory.MemoryReporter(runner, fake_torch(), tmp_path)
    reporter.sample()
    payload = memory.read_json(tmp_path / "report.json")
    payload.pop("allocation_map_version")
    memory.atomic_json(tmp_path / "report.json", payload)
    assert memory.request_allocation_map(tmp_path)["state"] == "map_not_enabled"
    assert not (tmp_path / "map-request.json").exists()
    monkeypatch.setattr(cli, "collect", lambda *_: pytest.fail("snapshot scan"))
    monkeypatch.setattr(cli, "discover_sessions", lambda *_: pytest.fail("chat discovery"))
    monkeypatch.setattr(
        memory, "request_allocation_map", lambda: memory.read_allocation_map(tmp_path, "a" * 32)
    )
    assert cli.main(["memory", "--map", "--host", "local", "--json"]) == 2
    assert "PRIVATE_CHAT" not in capsys.readouterr().out


def test_map_cli_renders_owners_and_pool_limits(tmp_path, monkeypatch, capsys):
    runner, torch, _, _ = map_fixture()
    reporter = memory.MemoryReporter(runner, torch, tmp_path)
    reporter.sample()
    memory.atomic_json(
        tmp_path / "map-request.json",
        {
            "instance_id": reporter.instance_id,
            "requested_at_ns": time.time_ns(),
        },
    )
    reporter.sample()
    monkeypatch.setattr(
        memory,
        "request_allocation_map",
        lambda: memory.read_allocation_map(tmp_path, reporter.instance_id),
    )
    assert cli.main(["memory", "--map", "--host", "local"]) == 0
    output = capsys.readouterr().out
    assert "Main model weights and buffers" in output
    assert "Largest free block" in output
    assert "private" in output
    assert "not a reclaimable-byte promise" in output
    assert "PRIVATE_CHAT" not in output


def test_stale_map_is_refreshed_and_old_request_instance_is_ignored(tmp_path):
    runner, torch, _, calls = map_fixture()
    reporter = memory.MemoryReporter(runner, torch, tmp_path)
    reporter.sample()
    memory.atomic_json(
        tmp_path / "map-request.json",
        {
            "instance_id": "a" * 32,
            "requested_at_ns": time.time_ns(),
        },
    )
    reporter.sample()
    assert calls == []
    for _ in range(2):
        reporter.last_map_attempt -= memory.MAP_INTERVAL
        memory.atomic_json(
            tmp_path / "map-request.json",
            {
                "instance_id": reporter.instance_id,
                "requested_at_ns": time.time_ns(),
            },
        )
        reporter.sample()
    assert len(calls) == 2


@pytest.mark.parametrize(
    "runtime,valid",
    [("current", True), ("previous", True), ("pre_report", True), ("unknown", False)],
)
def test_launcher_accepts_only_explicit_telemetry_compatible_runtime(tmp_path, runtime, valid):
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "scripts/pi-remote-qwen-radiance").read_text()
    remote = launcher.split("<<'REMOTE_BACKEND'\n", 1)[1].split("\nREMOTE_BACKEND\n", 1)[0]
    function = remote[
        remote.index("validate_backend() {") : remote.index("\nif podman container exists")
    ]
    current, previous, unknown, pre_report = "a" * 64, "b" * 64, "c" * 64, "d" * 64
    selected = {
        "current": current,
        "previous": previous,
        "unknown": unknown,
        "pre_report": pre_report,
    }[runtime]
    inspection = [
        {
            "Name": "test-container",
            "Image": "test-image",
            "State": {"Running": True},
            "Args": [
                "--served-model-name",
                "test-model",
                "--max-model-len",
                "253792",
                "/cache/snapshots/test-data-abi/data",
                "qwen_chat_fs",
                f"qwen-radiance-public-clean-{selected[:16]}",
                "--kv-cache-dtype",
                "fp8",
                "--speculative-config",
            ],
        }
    ]
    path = tmp_path / "inspect.json"
    path.write_text(json.dumps(inspection))
    setup = "\n".join(
        f"{key}={shlex.quote(value)}"
        for key, value in {
            "container": "test-container",
            "image_id": "test-image",
            "model": "test-model",
            "abi": "test-data-abi",
            "runtime_abi": current,
            "compatible_runtime_abi": f"{previous} {pre_report}",
            "port": "8080",
            "inspect_file": str(path),
        }.items()
    )
    script = (
        setup
        + """
podman() { cat "$inspect_file"; }
curl() { printf '%s' '{"data":[{"id":"test-model"}]}'; }
"""
        + function
        + "\nvalidate_backend\n"
    )
    result = subprocess.run(["bash", "-s"], input=script, text=True, capture_output=True)
    assert (result.returncode == 0) is valid
    # A compatibility match never bypasses the pinned image/model checks.
    inspection[0]["Image"] = "unqualified-image"
    path.write_text(json.dumps(inspection))
    result = subprocess.run(["bash", "-s"], input=script, text=True, capture_output=True)
    assert result.returncode != 0

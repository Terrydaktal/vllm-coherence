from __future__ import annotations

import json

import pytest

from qwen_r9700_lab import radiance_cache_residency as residency_module
from qwen_r9700_lab.radiance_cache_residency import ResidencyProbe, legacy_sample

ABI, CHAT, GEN, OTHER = "a" * 64, "b" * 64, "c" * 64, "d" * 64


def backend_metadata(root, abi=ABI):
    config = {
        "kv_connector": "OffloadingConnector",
        "kv_connector_extra_config": {
            "secondary_tiers": [
                {"type": "qwen_chat_fs", "root_dir": f"/cache/snapshots/{abi}/data"}
            ]
        },
    }
    return {
        "running": True,
        "mounts": [{"Destination": "/cache", "Source": str(root)}],
        "args": ["--kv-transfer-config", json.dumps(config)],
    }


def test_live_namespace_uses_the_backend_cache_mount_and_not_client_environment(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("QWEN_RADIANCE_CACHE_ABI", "7" * 64)
    metadata = backend_metadata(tmp_path)
    assert residency_module.backend_namespace(metadata, tmp_path) == ABI
    for invalid in (
        {**metadata, "running": False},
        {**metadata, "mounts": [{"Destination": "/cache", "Source": "/wrong-cache"}]},
        {**metadata, "args": metadata["args"] * 2},
        backend_metadata(tmp_path, "../old"),
    ):
        with pytest.raises(ValueError):
            residency_module.backend_namespace(invalid, tmp_path)


def test_namespace_discovery_is_shared_cached_and_invalidated_on_backend_replacement(
    tmp_path, monkeypatch
):
    clock = [0]
    report = {"instance_id": "1" * 32}
    inspected = []

    def inspect(root):
        assert root == tmp_path
        inspected.append(report["instance_id"])
        if report["instance_id"] == "3" * 32:
            raise ValueError("backend metadata unavailable")
        return ABI if report["instance_id"] == "1" * 32 else OTHER

    monkeypatch.setattr(residency_module, "optional_json", lambda _: report)
    monkeypatch.setattr(residency_module, "inspect_backend_namespace", inspect)
    monkeypatch.setattr(residency_module.time, "monotonic", lambda: clock[0])
    namespace = residency_module.LiveCacheNamespace(tmp_path)
    for _ in range(100):
        assert namespace.read() == ABI
    assert inspected == ["1" * 32]
    report["instance_id"] = "2" * 32
    assert namespace.read() == OTHER
    assert inspected == ["1" * 32, "2" * 32]
    report["instance_id"] = "3" * 32
    assert namespace.read() is None  # Never carry a predecessor's ABI into the new instance.
    assert namespace.read() is None
    assert len(inspected) == 3
    clock[0] = 10
    assert namespace.read() is None
    assert len(inspected) == 4


def fixture(tmp_path):
    probe = ResidencyProbe(tmp_path, ABI)
    directory = probe.managed / CHAT
    generation = directory / "generations" / GEN
    generation.mkdir(parents=True)
    block = generation / ("g0-" + "1" * 64 + ".qkv")
    block.write_bytes(b"synthetic opaque KV payload")
    details = block.stat()
    metadata = {
        "id": CHAT,
        "generation": GEN,
        "tokens": 40_000,
        "head": [block.name],
        "title": "private fixture title must not leave the probe",
        "verified_head": {
            block.name: [details.st_ino, details.st_size, details.st_mtime_ns, details.st_ctime_ns]
        },
    }
    (directory / "chat.json").write_text(json.dumps(metadata))
    chat = {"chat_id": CHAT, "generation": GEN}
    scheduler = {"pid": 123, "updated_at": 100, "switches": 3, "requests": []}
    worker = {
        "pid": 123,
        "switches": 3,
        "last_handover": "swap",
        "residency": {"active": chat, "images": []},
    }
    tail = {"pid": 123, "updated_at": 100, "chats": [{**chat, "tokens": 47_000}]}
    return probe, scheduler, worker, tail, block


def test_shared_feed_refreshes_memory_every_half_second_and_verifies_disk_once_a_second(
    tmp_path, monkeypatch, capsys
):
    probe, scheduler, worker, tail, _ = fixture(tmp_path)
    request = {
        "chat_id": CHAT,
        "generation": GEN,
        "state": "running",
        "computed_tokens": 47_000,
        "input_tokens": 50_000,
    }
    scheduler["requests"] = [request]
    clock = [0.0]
    disk_checks = []

    def disk_heads():
        disk_checks.append(clock[0])
        return {(CHAT, GEN): 40_000 if clock[0] < 1 else 45_000}, True

    class FinishedError(Exception):
        pass

    def sleep(seconds):
        assert seconds == 0.5
        clock[0] += seconds
        request["computed_tokens"] += 1000
        if clock[0] > 1:
            raise FinishedError

    sources = {
        "/dev/shm/qwen-radiance-fair-public-phases.json": None,
        "/dev/shm/qwen-radiance-fair-public-scheduler.json": scheduler,
        "/dev/shm/qwen-radiance-fair-public-worker.json": worker,
        "/dev/shm/qwen-radiance-snapshot-tail.json": tail,
    }
    monkeypatch.setattr(probe, "disk_heads", disk_heads)
    monkeypatch.setattr(residency_module, "ResidencyProbe", lambda *_: probe)
    monkeypatch.setattr(residency_module, "optional_json", sources.__getitem__)
    monkeypatch.setattr(residency_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(residency_module.time, "time", lambda: 100 + clock[0])
    monkeypatch.setattr(residency_module.time, "sleep", sleep)
    monkeypatch.setattr(
        residency_module,
        "StatusChanges",
        lambda: type("Changes", (), {"wait": staticmethod(sleep)})(),
    )
    monkeypatch.setattr(residency_module.sys, "argv", ["probe", "0.5", str(tmp_path), ABI])
    with pytest.raises(FinishedError):
        residency_module.main()
    samples = [json.loads(line.split("\t")[3]) for line in capsys.readouterr().out.splitlines()]
    assert [value["observed_at_ms"] for value in samples] == [100000, 100500, 101000]
    assert [value["chats"][0]["gpu_tokens"] for value in samples] == [47000, 48000, 49000]
    assert [value["chats"][0]["disk_saved_tokens"] for value in samples] == [40000, 40000, 45000]
    assert disk_checks == [0, 1]


def test_gpu_ram_disk_and_live_progress_are_generation_scoped(tmp_path):
    probe, scheduler, worker, tail, _ = fixture(tmp_path)
    first = probe.sample(scheduler, worker, tail, now=100)
    assert first["live"] and first["complete"]
    assert first["chats"][0] == {
        "chat_id": CHAT,
        "generation": GEN,
        "gpu_tokens": 47_000,
        "ram_tokens": 0,
        "disk_tokens": 47_000,
        "disk_saved_tokens": 40_000,
        "input_tokens": None,
    }
    assert "private fixture" not in json.dumps(first)
    scheduler["requests"] = [
        {
            "chat_id": CHAT,
            "generation": GEN,
            "state": "running",
            "computed_tokens": 48_532,
            "input_tokens": 48_000,
        }
    ]
    assert probe.sample(scheduler, worker, tail, now=101)["chats"][0]["gpu_tokens"] == 48_532
    scheduler["requests"] = []
    assert probe.sample(scheduler, worker, tail, now=102)["chats"][0]["gpu_tokens"] == 48_532
    worker["residency"] = {
        "active": {"chat_id": OTHER, "generation": GEN},
        "images": [{"chat_id": CHAT, "generation": GEN}],
    }
    row = probe.sample(scheduler, worker, tail, now=102)["chats"][0]
    assert row["gpu_tokens"] == 0 and row["ram_tokens"] == 48_532
    worker["residency"]["images"] = []
    row = probe.sample(scheduler, worker, tail, now=102)["chats"][0]
    assert row["gpu_tokens"] == row["ram_tokens"] == 0
    assert row["disk_tokens"] == 47_000  # RAM tail still assists the disk restore.


def test_idle_backend_is_kept_live_by_tail_heartbeat_and_stale_memory_is_unknown(tmp_path):
    probe, scheduler, worker, tail, _ = fixture(tmp_path)
    tail["updated_at"] = 200
    assert probe.sample(scheduler, worker, tail, now=201)["chats"][0]["gpu_tokens"] == 47_000
    stale = probe.sample(scheduler, worker, tail, now=300)
    assert not stale["live"]
    assert stale["chats"][0]["gpu_tokens"] is None
    assert stale["chats"][0]["disk_tokens"] == 40_000
    assert not probe.sample(scheduler, {**worker, "switches": 4}, tail, now=201)["live"]


def test_idle_scheduler_observation_requires_same_live_worker_and_no_requests():
    scheduler = {"pid": 123, "updated_at": 10, "requests": []}
    tail = {"pid": 123, "updated_at": 200}
    observed = residency_module.observe_idle_scheduler(scheduler, tail, now=201)
    assert observed == {**scheduler, "updated_at": 200}
    assert scheduler["updated_at"] == 10
    for invalid in (
        None,
        {**tail, "pid": 124},
        {**tail, "updated_at": 100},
        {**tail, "updated_at": 999},
        {**tail, "updated_at": float("nan")},
    ):
        assert residency_module.observe_idle_scheduler(scheduler, invalid, now=201) is scheduler
    for state in ("running", "queued", "parked"):
        active = {**scheduler, "requests": [{"state": state}]}
        assert residency_module.observe_idle_scheduler(active, tail, now=201) is active


def test_missing_or_changed_blocks_are_unknown_without_reading_payloads(tmp_path):
    probe, scheduler, worker, tail, block = fixture(tmp_path)
    block.write_bytes(b"changed")
    row = probe.sample(scheduler, worker, tail, now=100)["chats"][0]
    assert row["disk_tokens"] is None
    block.unlink()
    assert probe.sample(scheduler, worker, tail, now=100)["chats"][0]["disk_tokens"] is None


def test_compaction_does_not_count_the_previous_generation_as_current(tmp_path):
    probe, scheduler, worker, tail, _ = fixture(tmp_path)
    new_generation = "e" * 64
    worker["residency"]["active"]["generation"] = new_generation
    worker["last_handover"] = "generation-replace"
    scheduler["requests"] = [
        {
            "chat_id": CHAT,
            "generation": new_generation,
            "state": "running",
            "computed_tokens": 0,
            "input_tokens": 4000,
        }
    ]
    rows = probe.sample(scheduler, worker, tail, now=100)["chats"]
    new = next(row for row in rows if row["generation"] == new_generation)
    assert new["gpu_tokens"] == new["ram_tokens"] == new["disk_tokens"] == 0


def test_unreadable_inventory_is_not_reported_as_an_empty_cache(tmp_path):
    probe = ResidencyProbe(tmp_path, ABI)
    sample = probe.sample(None, None, None, now=100)
    assert not sample["live"] and not sample["complete"] and not sample["chats"]


@pytest.mark.parametrize("state", ["running", "paused"])
def test_prefill_cannot_claim_the_old_disk_head_as_remaining_reusable_tokens(
    tmp_path, monkeypatch, state
):
    probe, scheduler, worker, tail, _ = fixture(tmp_path)
    monkeypatch.setattr(probe, "disk_heads", lambda: ({(CHAT, GEN): 234_774}, True))
    tail["chats"][0]["tokens"] = 237_918
    chat = {"chat_id": CHAT, "generation": GEN}
    scheduler["requests"] = [
        {**chat, "state": state, "computed_tokens": 16_384, "input_tokens": 238_880}
    ]
    if state == "paused":
        worker["residency"] = {"active": {"chat_id": OTHER, "generation": GEN}, "images": [chat]}
    row = next(
        row
        for row in probe.sample(scheduler, worker, tail, now=100)["chats"]
        if row["chat_id"] == CHAT
    )
    assert row["gpu_tokens"] + row["ram_tokens"] == 16_384
    assert row["disk_tokens"] == 16_384
    assert (
        row["input_tokens"] - max(row["gpu_tokens"], row["ram_tokens"], row["disk_tokens"])
        == 222_496
    )


def test_queued_restore_reservation_is_not_loaded_gpu_data_or_evidence_of_cold_prefill(tmp_path):
    probe, scheduler, worker, tail, _ = fixture(tmp_path)
    scheduler["requests"] = [
        {
            "chat_id": CHAT,
            "generation": GEN,
            "state": "queued",
            "computed_tokens": 47_000,
            "input_tokens": 48_000,
        }
    ]
    row = probe.sample(scheduler, worker, tail, now=100)["chats"][0]
    assert row["gpu_tokens"] is None
    assert row["disk_tokens"] == 47_000
    assert (CHAT, GEN) not in probe.progress
    scheduler["requests"][0]["state"] = "running"
    row = probe.sample(scheduler, worker, tail, now=100)["chats"][0]
    assert row["gpu_tokens"] == 47_000


@pytest.mark.parametrize("handover", ["activate", "generation-replace", "swap"])
def test_tool_continuation_does_not_turn_a_resident_gpu_bank_into_disk(tmp_path, handover):
    probe, scheduler, worker, tail, _ = fixture(tmp_path)
    worker["last_handover"] = handover
    request = {
        "chat_id": CHAT,
        "generation": GEN,
        "state": "running",
        "computed_tokens": 48_858,
        "input_tokens": 48_000,
    }
    scheduler["requests"] = [request]
    assert probe.sample(scheduler, worker, tail, now=100)["chats"][0]["gpu_tokens"] == 48_858
    scheduler["requests"] = []
    assert probe.sample(scheduler, worker, tail, now=101)["chats"][0]["gpu_tokens"] == 48_858
    scheduler["requests"] = [{**request, "state": "queued", "computed_tokens": 0}]
    row = probe.sample(scheduler, worker, tail, now=102)["chats"][0]
    assert row["gpu_tokens"] == 48_858
    assert row["disk_saved_tokens"] == 40_000
    # Additional reserved restore tokens are not yet in the retained GPU bank.
    scheduler["requests"][0]["computed_tokens"] = 60_000
    assert probe.sample(scheduler, worker, tail, now=103)["chats"][0]["gpu_tokens"] == 48_858


def test_confirmed_zero_token_cache_miss_reports_cold_without_erasing_the_backup(tmp_path):
    probe, scheduler, worker, tail, _ = fixture(tmp_path)
    scheduler["requests"] = [
        {
            "chat_id": CHAT,
            "generation": GEN,
            "state": "running",
            "computed_tokens": 48_858,
            "input_tokens": 49_000,
        }
    ]
    probe.sample(scheduler, worker, tail, now=100)
    scheduler["requests"][0]["computed_tokens"] = 0
    row = probe.sample(scheduler, worker, tail, now=101)["chats"][0]
    assert row["gpu_tokens"] == row["ram_tokens"] == row["disk_tokens"] == 0
    assert row["disk_saved_tokens"] == 40_000


def test_disk_saved_counts_only_verified_publications_and_resets_on_compaction(tmp_path):
    probe, scheduler, worker, tail, _ = fixture(tmp_path)
    path = probe.managed / CHAT / "chat.json"
    first = probe.sample(scheduler, worker, tail, now=100)["chats"][0]
    assert first["disk_tokens"] == 47_000  # Includes the volatile tail.
    assert first["disk_saved_tokens"] == 40_000
    metadata = json.loads(path.read_text())
    metadata["publication"] = {"tokens": 47_000, "result": "rolled_back"}
    path.write_text(json.dumps(metadata))
    assert probe.sample(scheduler, worker, tail, now=101)["chats"][0]["disk_saved_tokens"] == 40_000
    metadata["tokens"] = 47_000
    metadata["publication"]["result"] = "committed"
    path.write_text(json.dumps(metadata))
    assert probe.sample(scheduler, worker, tail, now=102)["chats"][0]["disk_saved_tokens"] == 47_000

    new_generation = "e" * 64
    metadata.update(generation=new_generation, head=[], tokens=0, verified_head={})
    path.write_text(json.dumps(metadata))
    worker["residency"]["active"]["generation"] = new_generation
    new = next(
        row
        for row in probe.sample(scheduler, worker, tail, now=103)["chats"]
        if row["generation"] == new_generation
    )
    # Retained old files/tail never count for the new chat state.
    assert new["disk_saved_tokens"] == 0


def test_disk_backup_survives_gpu_telemetry_gaps_but_requires_verified_files(tmp_path):
    probe, scheduler, worker, tail, block = fixture(tmp_path)
    scheduler["switches"] += 1
    during = probe.sample(scheduler, worker, tail, now=100)["chats"][0]
    assert during["gpu_tokens"] is None
    assert during["disk_saved_tokens"] == 40_000
    block.write_bytes(b"changed synthetic block")
    assert probe.sample(scheduler, worker, tail, now=101)["chats"][0]["disk_saved_tokens"] is None


def test_legacy_monitor_sample_cannot_mistake_the_ram_tail_for_durable_coverage(tmp_path):
    probe, scheduler, worker, tail, _ = fixture(tmp_path)
    modern = probe.sample(scheduler, worker, tail, now=100)
    legacy = legacy_sample(modern)
    assert modern["schema"] == "urn:qwen-r9700:cache-residency:v2"
    assert legacy["schema"] == "urn:qwen-r9700:cache-residency:v1"
    assert legacy["chats"][0]["disk_tokens"] == 47_000
    assert "disk_saved_tokens" not in legacy["chats"][0]
    assert modern["chats"][0]["disk_saved_tokens"] == 40_000


def test_incoherent_handover_does_not_cap_saved_disk_coverage(tmp_path):
    probe, scheduler, worker, tail, _ = fixture(tmp_path)
    worker["switches"] += 1
    scheduler["requests"] = [
        {
            "chat_id": CHAT,
            "generation": GEN,
            "state": "running",
            "computed_tokens": 1000,
            "input_tokens": 48_000,
        }
    ]
    row = probe.sample(scheduler, worker, tail, now=100)["chats"][0]
    assert row["gpu_tokens"] is None
    assert row["disk_tokens"] == 47_000


@pytest.mark.parametrize("disk_tokens", [None, 0])
def test_other_chat_compaction_does_not_forget_idle_ram_coverage(
    tmp_path, monkeypatch, disk_tokens
):
    probe, scheduler, worker, tail, _ = fixture(tmp_path)
    # An unpublished or unverified disk head cannot reconstruct a forgotten
    # memory count. These are synthetic counts, with no transcript or KV reads.
    monkeypatch.setattr(probe, "disk_heads", lambda: ({(CHAT, GEN): disk_tokens}, True))
    tail["chats"] = []
    scheduler["requests"] = [
        {
            "chat_id": CHAT,
            "generation": GEN,
            "state": "running",
            "computed_tokens": 35_207,
            "input_tokens": 35_000,
        }
    ]
    assert probe.sample(scheduler, worker, tail, now=100)["chats"][0]["gpu_tokens"] == 35_207

    scheduler["requests"] = []
    scheduler["switches"] = worker["switches"] = 4
    worker["residency"] = {
        "active": {"chat_id": OTHER, "generation": GEN},
        "images": [{"chat_id": CHAT, "generation": GEN}],
    }
    assert probe.sample(scheduler, worker, tail, now=101)["chats"][0]["ram_tokens"] == 35_207

    # The other chat compacts. Scheduler admission precedes the worker's
    # generation replacement, so the two atomic status files disagree briefly.
    scheduler["switches"] = 5
    during = probe.sample(scheduler, worker, tail, now=102)
    assert not during["live"]
    assert during["chats"][0]["ram_tokens"] is None
    worker["switches"] = 5
    worker["last_handover"] = "generation-replace"
    worker["residency"]["active"]["generation"] = "e" * 64
    after = probe.sample(scheduler, worker, tail, now=103)
    assert after["live"]
    assert after["chats"][0]["gpu_tokens"] == 0
    assert after["chats"][0]["ram_tokens"] == 35_207
    assert after["chats"][0]["disk_tokens"] == disk_tokens


@pytest.mark.parametrize(
    "invalidate", ["eviction", "generation", "restart", "stale", "missed_handover", "reset"]
)
def test_remembered_memory_coverage_requires_the_same_live_bank(tmp_path, invalidate):
    probe, scheduler, worker, tail, _ = fixture(tmp_path)
    probe.sample(scheduler, worker, tail, now=100)
    probe.progress[(CHAT, GEN)] = 48_532
    scheduler["switches"] += 1
    probe.sample(scheduler, worker, tail, now=101)
    worker["switches"] = scheduler["switches"]
    now = 102
    if invalidate == "eviction":
        worker["residency"]["active"] = None
    elif invalidate == "generation":
        worker["residency"]["active"]["generation"] = "e" * 64
    elif invalidate == "restart":
        scheduler["pid"] = worker["pid"] = tail["pid"] = 456
    elif invalidate == "missed_handover":
        scheduler["switches"] = worker["switches"] = 6
    elif invalidate == "reset":
        scheduler["switches"] = worker["switches"] = 0
    else:
        now = 300
    probe.sample(scheduler, worker, tail, now=now)
    assert (CHAT, GEN) not in probe.progress


def test_phase_file_notifications_wake_the_shared_probe_without_faster_polling(tmp_path):
    import threading
    import time

    from qwen_r9700_lab.radiance_cache_residency import StatusChanges

    changes = StatusChanges(str(tmp_path))
    assert changes.fd >= 0

    def publish():
        time.sleep(0.02)
        stage = tmp_path / "phase.tmp"
        stage.write_text("{}")
        stage.replace(tmp_path / "qwen-radiance-fair-public-phases.json")

    thread = threading.Thread(target=publish)
    thread.start()
    try:
        start = time.monotonic()
        changes.wait(5)
        assert time.monotonic() - start < 1
    finally:
        thread.join()
        changes.close()
    assert changes.fd == -1

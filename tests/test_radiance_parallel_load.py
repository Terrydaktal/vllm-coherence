from __future__ import annotations

import contextlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import pytest
from test_radiance_chat_cache import BLOCK, chat, load_tier, new_tier, store

from qwen_r9700_lab import radiance_cache as cache
from qwen_r9700_lab import radiance_cache_telemetry as telemetry


def objects(current, count=6):
    names = [f"g0-{index:064x}.qkv" for index in range(count)]
    payloads = [bytes([index]) * len(BLOCK) for index in range(count)]
    assert current.write_many([(name, memoryview(data)) for name, data in zip(names, payloads)])
    return names, payloads


def test_parallel_reads_overlap_and_keep_requested_order(tmp_path, monkeypatch):
    current = store(tmp_path)
    names, expected = objects(current)
    entered, release = Event(), Event()
    original = cache.decode_block
    mutex = Lock()
    active = maximum = 0

    def decode(data, size):
        nonlocal active, maximum
        with mutex:
            active += 1
            maximum = max(maximum, active)
            if active == 3:
                entered.set()
        try:
            assert release.wait(5)
            return original(data, size)
        finally:
            with mutex:
                active -= 1

    # Avoid unrelated hash collisions obscuring the actual parallelism.
    class IndependentLocks:
        def __init__(self):
            self.items = {}

        def __len__(self):
            return 2**62

        def __getitem__(self, index):
            return self.items.setdefault(index, Lock())

    monkeypatch.setattr(cache, "_WRITE_LOCKS", IndependentLocks())
    monkeypatch.setattr(cache, "decode_block", decode)
    with ThreadPoolExecutor(max_workers=1) as runner:
        future = runner.submit(lambda: list(current.read_many(names, len(BLOCK), workers=3)))
        try:
            assert entered.wait(5), "independent blocks were still decoded serially"
        finally:
            release.set()
        assert future.result(timeout=5) == expected
    assert maximum == 3
    assert current.io_totals()["read_blocks"] == len(names)


def test_early_close_joins_readers_before_generation_retirement(tmp_path, monkeypatch):
    current = store(tmp_path)
    names, expected = objects(current, 3)
    blocked, release = Event(), Event()
    original = cache.decode_block

    def decode(data, size):
        if original(data, size) == expected[1]:
            blocked.set()
            assert release.wait(5)
        return original(data, size)

    monkeypatch.setattr(cache, "decode_block", decode)
    blocks = current.read_many(names, len(BLOCK), workers=2)
    assert next(blocks) == expected[0]
    assert blocked.wait(5)
    successor = cache.ChatStore(tmp_path, chat(generation="after-load"))
    with ThreadPoolExecutor(max_workers=2) as runner:
        closed = runner.submit(blocks.close)
        retired = runner.submit(successor.activate)
        try:
            assert not closed.done()
            assert not retired.done(), "retirement escaped the outstanding worker lease"
        finally:
            release.set()
        closed.result(timeout=5)
        retired.result(timeout=5)
    assert not current.generation.exists()


@pytest.mark.parametrize("workers", [1, 4])
def test_corruption_and_size_errors_keep_the_existing_invalidation_contract(tmp_path, workers):
    current = store(tmp_path)
    names, expected = objects(current, 4)
    damaged = bytearray(current.path(names[2]).read_bytes())
    damaged[20] ^= 1
    current.path(names[2]).write_bytes(damaged)
    with pytest.raises(cache.SnapshotIntegrityError):
        list(current.read_many(names, len(BLOCK), workers=workers))
    assert not current.path(names[2]).exists()
    assert current.read(names[0], len(BLOCK)) == expected[0]
    assert current.io_totals()["invalidated_blocks"] == 1
    assert current.io_totals()["read_failures"] == 1
    for size in (len(BLOCK) - 1, len(BLOCK) + 1):
        with pytest.raises(ValueError, match="size"):
            list(current.read_many([names[1]], size, workers=workers))
        assert current.path(names[1]).exists(), "ABI mismatch must not delete a valid object"


def test_byte_budget_limits_prefetched_blocks(tmp_path, monkeypatch):
    current = store(tmp_path)
    names, expected = objects(current, 4)
    original = cache.decode_block
    entered, release = Event(), Event()
    active = maximum = 0
    mutex = Lock()

    def decode(data, size):
        nonlocal active, maximum
        with mutex:
            active += 1
            maximum = max(maximum, active)
            entered.set()
        try:
            assert release.wait(5)
            return original(data, size)
        finally:
            with mutex:
                active -= 1

    monkeypatch.setattr(cache, "decode_block", decode)
    with ThreadPoolExecutor(max_workers=1) as runner:
        pending = runner.submit(lambda: list(current.read_many(
            names, len(BLOCK), workers=4, max_inflight_bytes=3 * len(BLOCK))))
        try:
            assert entered.wait(5)
        finally:
            release.set()
        assert pending.result(timeout=5) == expected
    assert maximum == 1


def test_tier_closes_disk_iterator_if_destination_copy_fails(tmp_path, monkeypatch):
    module = load_tier(monkeypatch)
    tier = new_tier(module)
    current = store(tmp_path)
    key = bytes.fromhex("ab" * 32) + (0).to_bytes(4, "big")
    name = module.object_key(key)
    assert current.write(name, memoryview(BLOCK))
    tier._block_size = len(BLOCK)
    tier._primary_kv_view = memoryview(bytes(len(BLOCK)))  # read-only destination
    tier._ram_blocks = lambda *_: {}
    closed = Event()
    original = current.read_many

    def tracked(*args, **kwargs):
        with contextlib.closing(original(*args, **kwargs)) as values:
            try:
                yield from values
            finally:
                closed.set()

    monkeypatch.setattr(current, "read_many", tracked)
    with pytest.raises(TypeError, match="read-only"):
        tier._load({"store": current}, [key], [0])
    assert closed.is_set()


def test_failure_joins_other_workers_before_releasing_generation(tmp_path, monkeypatch):
    current = store(tmp_path)
    names, expected = objects(current, 8)
    first = names[0]
    second = next(name for name in names[1:] if
                  hash(str(current.path(name))) % len(cache._WRITE_LOCKS) !=
                  hash(str(current.path(first))) % len(cache._WRITE_LOCKS))
    damaged = bytearray(current.path(first).read_bytes())
    damaged[20] ^= 1
    current.path(first).write_bytes(damaged)
    blocked, release, failed = Event(), Event(), Event()
    original = cache.decode_block
    second_data = expected[names.index(second)]

    def decode(data, size):
        if data == damaged:
            assert blocked.wait(5)
            failed.set()
            return original(data, size)
        result = original(data, size)
        if result == second_data:
            blocked.set()
            assert release.wait(5)
        return result

    monkeypatch.setattr(cache, "decode_block", decode)
    successor = cache.ChatStore(tmp_path, chat(generation="failed-load-retirement"))
    with ThreadPoolExecutor(max_workers=2) as runner:
        loading = runner.submit(lambda: list(current.read_many([first, second], len(BLOCK), workers=2)))
        try:
            assert failed.wait(5)
            retirement = runner.submit(successor.activate)
            assert not loading.done(), "failed batch returned with a live reader"
            assert not retirement.done()
        finally:
            release.set()
        with pytest.raises(cache.SnapshotIntegrityError):
            loading.result(timeout=5)
        retirement.result(timeout=5)
    assert current.io_totals()["invalidated_blocks"] == 1


def test_repeated_keys_and_ram_blocks_map_to_original_offsets(tmp_path, monkeypatch):
    module = load_tier(monkeypatch)
    tier = new_tier(module)
    current = store(tmp_path)
    disk_key = bytes.fromhex("ab" * 32) + (0).to_bytes(4, "big")
    ram_key = bytes.fromhex("cd" * 32) + (0).to_bytes(4, "big")
    disk_name, ram_name = module.object_key(disk_key), module.object_key(ram_key)
    assert current.write(disk_name, memoryview(BLOCK))
    ram_data = bytes(reversed(BLOCK))
    tier._block_size = len(BLOCK)
    destination = bytearray(len(BLOCK) * 4)
    tier._primary_kv_view = memoryview(destination)
    tier._ram_blocks = lambda *_: {ram_name: ram_data}
    tier._load({"store": current}, [disk_key, ram_key, disk_key, ram_key],
               [2 * len(BLOCK), 3 * len(BLOCK), 0, len(BLOCK)])
    assert destination == BLOCK + ram_data + BLOCK + ram_data
    assert current.io_totals()["read_blocks"] == 2


def test_oversize_corrupt_object_is_bounded_and_wrong_raw_size_is_preserved(tmp_path):
    current = store(tmp_path)
    names, _ = objects(current, 2)
    encoded = cache.HEADER.pack(cache.RAW, len(BLOCK), b"\0" * 32) + bytes(3 * len(BLOCK))
    current.path(names[0]).write_bytes(encoded)
    with pytest.raises(cache.SnapshotIntegrityError, match="exceeds"):
        current.read(names[0], len(BLOCK))
    assert not current.path(names[0]).exists()
    assert current.io_totals()["invalidated_file_bytes"] == len(encoded)
    valid = cache.HEADER.pack(cache.RAW, len(BLOCK), cache.hashlib.sha256(BLOCK).digest()) + BLOCK
    current.path(names[1]).write_bytes(valid)
    with pytest.raises(ValueError, match="size"):
        current.read(names[1], len(BLOCK) - 1)
    assert current.path(names[1]).read_bytes() == valid


@pytest.mark.parametrize("workers,budget", [(0, 1), (5, 1), (True, 1), (1, 0)])
def test_invalid_parallel_limits_do_not_start_or_invalidate_reads(tmp_path, workers, budget):
    current = store(tmp_path)
    names, _ = objects(current, 1)
    with pytest.raises(ValueError):
        list(current.read_many(names, len(BLOCK), workers=workers, max_inflight_bytes=budget))
    assert current.path(names[0]).exists()
    assert "read_blocks" not in current.io_totals()


def test_concurrent_batches_bound_active_readers_without_ordered_result_deadlock(tmp_path, monkeypatch):
    left, right = store(tmp_path, "left"), store(tmp_path, "right")
    left_names, left_data = objects(left, 12)
    right_names, right_data = objects(right, 12)
    original = cache.decode_block
    mutex = Lock()
    ready, release = Event(), Event()
    active = maximum = 0

    def decode(data, size):
        nonlocal active, maximum
        with mutex:
            active += 1
            maximum = max(maximum, active)
            ready.set()
        try:
            assert release.wait(5)
            return original(data, size)
        finally:
            with mutex:
                active -= 1

    monkeypatch.setattr(cache, "decode_block", decode)
    with ThreadPoolExecutor(max_workers=2) as runner:
        pending = [runner.submit(lambda current=current, names=names:
                   list(current.read_many(names, len(BLOCK))))
                   for current, names in [(left, left_names), (right, right_names)]]
        try:
            assert ready.wait(5)
        finally:
            release.set()
        assert [future.result(timeout=5) for future in pending] == [left_data, right_data]
    assert 1 <= maximum <= cache.READ_WORKERS
    assert left.io_totals()["read_blocks"] == right.io_totals()["read_blocks"] == 12


def test_reported_parallelism_respects_global_byte_cap():
    assert cache.read_parallelism(27 * 1024**2, 159) == 3
    assert cache.read_parallelism(27 * 1024**2, 159, max_inflight_bytes=1024**3) == 3
    assert cache.read_parallelism(len(BLOCK), 0) == 0
    assert cache.read_parallelism(len(BLOCK), 1) == 1


@pytest.mark.parametrize("damage", [False, True])
def test_worker_telemetry_keeps_job_identity_and_finishes_after_join(tmp_path, monkeypatch, damage):
    module = load_tier(monkeypatch)
    tier = new_tier(module)
    current = store(tmp_path)
    names, expected = objects(current, 6)
    if damage:
        data = bytearray(current.path(names[3]).read_bytes())
        data[20] ^= 1
        current.path(names[3]).write_bytes(data)
    tier._block_size = len(BLOCK)
    destination = bytearray(len(BLOCK) * len(names))
    tier._primary_kv_view = memoryview(destination)
    tier._ram_blocks = lambda *_: {}
    keys = [index.to_bytes(32, "big") + b"\0" * 4 for index in range(len(names))]
    context = {"request_id": "1" * 64, "chat_id": current.chat["id"],
               "generation": current.chat["generation"], "job_id": 731,
               "job_kind": "filesystem", "direction": "load"}
    recorder = telemetry.Recorder(tmp_path / "telemetry", start=False, gc_events=False)
    monkeypatch.setattr(telemetry, "_recorder", recorder)
    monkeypatch.setattr(telemetry, "_cache_loads", {})
    monkeypatch.setattr(telemetry, "_round", {"request_id": "f" * 64, "chat_id": "e" * 64})
    monkeypatch.setattr(telemetry._local, "context", {}, raising=False)
    monkeypatch.setattr(telemetry._local, "cache_load", None, raising=False)
    try:
        def load():
            tier._load({"store": current}, keys, [index * len(BLOCK) for index in range(len(keys))])

        if damage:
            with pytest.raises(cache.SnapshotIntegrityError):
                telemetry.task(load, context, time.monotonic_ns())
        else:
            telemetry.task(load, context, time.monotonic_ns())
            assert destination == b"".join(expected)
        progress = telemetry.cache_load_progress(context["request_id"])
        assert progress["status"] == ("failed" if damage else "complete")
        assert progress["completed_blocks"] == (3 if damage else 6)
        assert not any(progress["active_stages"].values())
        assert telemetry.capture_context() == ({}, None)
        while recorder.pending:
            recorder.flush()
        rows = [json.loads(line) for line in recorder.path.read_text().splitlines()]
        workers = [row for row in rows if row["stage"] in {"disk_read", "decompression", "checksum"}]
        assert workers and any(row["thread_id"] != threading.get_native_id() for row in workers)
        for row in workers:
            assert all(row[name] == context[name] for name in ("request_id", "chat_id", "generation", "job_id"))
        assert sum(row["stage"] == "cpu_job" for row in rows) == 1
    finally:
        recorder.close()

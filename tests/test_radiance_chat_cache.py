from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, RLock
from types import ModuleType, SimpleNamespace

import pytest
import zstandard

from qwen_r9700_lab import radiance_cache as cache

ROOT = Path(__file__).resolve().parents[1]
BLOCK = bytes(range(256)) * 256
FIRST = "g6-" + "a" * 64 + ".qkv"
SECOND = "g0-" + "b" * 64 + ".qkv"
THIRD = "g0-" + "c" * 64 + ".qkv"


def chat(name="one", generation="initial"):
    return {
        "id": hashlib.sha256(name.encode()).hexdigest(),
        "generation": hashlib.sha256(generation.encode()).hexdigest(),
        "title": name,
        "session_file": f"/sessions/{name}.jsonl",
        "cwd": "/work",
    }


def store(tmp_path, name="one", generation="initial"):
    result = cache.ChatStore(tmp_path, chat(name, generation))
    result.activate()
    return result


@pytest.mark.parametrize("data", [BLOCK, os.urandom(65536), b"\0" * 1024])
def test_codec_is_lossless_and_detects_damage(data):
    encoded = cache.encode_block(data)
    assert cache.decode_block(encoded, len(data)) == data
    assert len(encoded) <= len(data) + cache.HEADER.size
    with pytest.raises((ValueError, zstandard.ZstdError)):
        cache.decode_block(encoded[:-1], len(data))
    with pytest.raises(ValueError, match="size"):
        cache.decode_block(encoded, len(data) + 1)
    corrupt = bytearray(encoded)
    corrupt[20] ^= 1  # authenticated content digest
    with pytest.raises(ValueError, match="checksum"):
        cache.decode_block(bytes(corrupt), len(data))


@pytest.mark.parametrize("damage", ["header", "compressed", "checksum"])
def test_corrupt_read_becomes_a_miss_and_can_be_repaired(tmp_path, damage):
    current = store(tmp_path)
    current.write(FIRST, memoryview(BLOCK))
    current.write(SECOND, memoryview(BLOCK))
    current.publish([FIRST, SECOND], 1648, len(BLOCK))
    original = current.path(FIRST).read_bytes()
    damaged = bytearray(original)
    if damage == "header":
        damaged = damaged[:20]
    elif damage == "compressed":
        damaged[-1] ^= 1
    else:
        damaged[20] ^= 1
    current.path(FIRST).write_bytes(damaged)
    with pytest.raises(cache.SnapshotIntegrityError):
        current.read(FIRST, len(BLOCK))
    assert current.exists_many([FIRST, SECOND]) == [False, True]
    assert current.read(SECOND, len(BLOCK)) == BLOCK
    assert current.io_totals()["invalidated_blocks"] == 1
    assert current.io_totals()["invalidated_file_bytes"] == len(damaged)
    current.write(FIRST, memoryview(BLOCK))
    assert current.publish([FIRST, SECOND], 1648, len(BLOCK))
    assert current.read(FIRST, len(BLOCK)) == BLOCK
    assert current.metadata()["publication"]["result"] == "committed"
    assert {p.name for p in current.generation.iterdir()} == {FIRST, SECOND}


@pytest.mark.parametrize("failure", ["runtime_size", "io"])
def test_read_failure_does_not_remove_valid_or_unreadable_objects(tmp_path, monkeypatch, failure):
    current = store(tmp_path)
    current.write(FIRST, memoryview(BLOCK))
    path = current.path(FIRST)
    original = path.read_bytes()
    if failure == "io":
        read = Path.read_bytes

        def denied(p):
            if p == path:
                raise PermissionError("synthetic I/O denial")
            return read(p)

        monkeypatch.setattr(Path, "read_bytes", denied)
        with pytest.raises(PermissionError):
            current.read(FIRST, len(BLOCK))
    else:
        with pytest.raises(ValueError) as caught:
            current.read(FIRST, len(BLOCK) + 1)
        assert not isinstance(caught.value, cache.SnapshotIntegrityError)
    assert path.exists()
    with path.open("rb") as stream:
        assert stream.read() == original
    assert current.io_totals()["invalidated_blocks"] == 0


def test_failed_reader_cannot_delete_a_concurrent_repair(tmp_path, monkeypatch):
    current = store(tmp_path)
    current.write(FIRST, memoryview(BLOCK))
    damaged = bytearray(current.path(FIRST).read_bytes())
    damaged[20] ^= 1
    current.path(FIRST).write_bytes(damaged)
    entered, release, writing = Event(), Event(), Event()
    decode = cache.decode_block

    def held_decode(data, size):
        if data == damaged:
            entered.set()
            assert release.wait(5)
        return decode(data, size)

    def repair():
        writing.set()
        return current.write(FIRST, memoryview(BLOCK))

    monkeypatch.setattr(cache, "decode_block", held_decode)
    with ThreadPoolExecutor(max_workers=2) as pool:
        reader = pool.submit(current.read, FIRST, len(BLOCK))
        try:
            assert entered.wait(5)
            writer = pool.submit(repair)
            assert writing.wait(5)
        finally:
            release.set()
        with pytest.raises(cache.SnapshotIntegrityError):
            reader.result(timeout=5)
        assert writer.result(timeout=5)
    assert current.read(FIRST, len(BLOCK)) == BLOCK


def test_compaction_retires_only_its_chat_and_never_resurrects(tmp_path):
    legacy = tmp_path / "old.bin"
    legacy.write_bytes(b"legacy")
    one, two = store(tmp_path), store(tmp_path, "two")
    for current in (one, two):
        current.write(FIRST, memoryview(BLOCK))
        assert current.publish([FIRST], 1648, len(BLOCK))
    successor = cache.ChatStore(tmp_path, chat(generation="compaction1"))
    result = successor.activate()
    assert result["retained_file_bytes"] > 0
    assert one.generation.exists()
    assert not one.write(SECOND, memoryview(BLOCK))
    assert not one.publish([], 0, len(BLOCK))
    with pytest.raises(cache.RetiredGenerationError, match="retired"):
        one.activate()
    assert two.read(FIRST, len(BLOCK)) == BLOCK
    assert legacy.read_bytes() == b"legacy"
    successor.write(SECOND, memoryview(BLOCK))
    assert successor.publish([SECOND], 200, len(BLOCK))
    assert not one.generation.exists()


def test_compactions_and_tail_advances_do_not_accumulate(tmp_path):
    current = store(tmp_path)
    for generation in range(5):
        if generation:
            current = store(tmp_path, generation=str(generation))
        current.write(FIRST, memoryview(BLOCK))
        current.write(SECOND, memoryview(BLOCK))
        assert current.publish([FIRST, SECOND], 1648, len(BLOCK))
        current.write(THIRD, memoryview(BLOCK))
        assert current.publish([FIRST, THIRD], 3296, len(BLOCK))
        assert {p.name for p in current.generation.iterdir()} == {FIRST, THIRD}
        assert len(list(current.generations.iterdir())) == 1


def test_incomplete_successor_preserves_previous_complete_head(tmp_path):
    current = store(tmp_path)
    current.write(FIRST, memoryview(BLOCK))
    assert current.publish([FIRST], 1648, len(BLOCK))
    current.write(THIRD, memoryview(BLOCK))
    assert not current.publish([SECOND, THIRD], 3296, len(BLOCK))
    assert current.metadata()["head"] == [FIRST]
    assert current.metadata()["status"] == "incomplete"
    assert current.read(FIRST, len(BLOCK)) == BLOCK
    assert not current.path(THIRD).exists()
    assert current.metadata()["publication"]["missing_keys"] == [SECOND]
    assert current.metadata()["gc"]["status"] == "complete"


def test_only_one_complete_fallback_survives_repeated_unpublished_compactions(tmp_path):
    initial = store(tmp_path)
    initial.write(FIRST, memoryview(BLOCK))
    initial.publish([FIRST], 200_000, len(BLOCK))
    for generation in ("compact-one", "compact-two", "compact-three"):
        latest = store(tmp_path, generation=generation)
        assert latest.metadata()["fallback"]["generation"] == initial.chat["generation"]
        assert len(list(latest.generations.iterdir())) == 2
        latest.write(SECOND, memoryview(BLOCK))
        assert not latest.publish([SECOND, THIRD], 4_000, len(BLOCK))
        assert (initial.generation / FIRST).exists()
    latest.write(THIRD, memoryview(BLOCK))
    assert latest.publish([THIRD], 4_000, len(BLOCK))
    assert len(list(latest.generations.iterdir())) == 1
    assert latest.metadata()["fallback"] is None


def test_publication_detects_payload_damage_before_retiring_previous_head(tmp_path):
    current = store(tmp_path)
    current.write(FIRST, memoryview(BLOCK))
    current.publish([FIRST], 100, len(BLOCK))
    current.write(SECOND, memoryview(BLOCK))
    encoded = bytearray(current.path(SECOND).read_bytes())
    encoded[-1] ^= 1
    current.path(SECOND).write_bytes(encoded)
    assert not current.publish([SECOND], 200, len(BLOCK))
    assert current.metadata()["head"] == [FIRST]
    assert current.metadata()["publication"]["invalid_keys"] == [SECOND]
    assert current.read(FIRST, len(BLOCK)) == BLOCK


def test_incremental_publication_does_not_reverify_unchanged_prefix(tmp_path, monkeypatch):
    current = store(tmp_path)
    current.write(FIRST, memoryview(BLOCK))
    current.publish([FIRST], 100, len(BLOCK))
    current = cache.ChatStore(tmp_path, chat())  # Includes process restart.
    current.write(SECOND, memoryview(BLOCK))
    decoded = []
    decode = cache.decode_block
    monkeypatch.setattr(
        cache, "decode_block", lambda data, size: (decoded.append(data), decode(data, size))[1]
    )
    assert current.publish([FIRST, SECOND], 200, len(BLOCK))
    assert len(decoded) == 1
    prepared = current.prepare_publication([FIRST, SECOND], len(BLOCK))
    current.path(SECOND).write_bytes(b"bad")
    assert not current.publish([FIRST, SECOND], 200, len(BLOCK), prepared=prepared)


def test_write_counters_survive_gc_compaction_and_duplicate_jobs(tmp_path):
    current = store(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(lambda _: current.write(FIRST, memoryview(BLOCK)), range(16)))
    size = current.path(FIRST).stat().st_size
    assert current.io_totals()["written_file_bytes"] == size
    assert current.io_totals()["written_blocks"] == 1
    assert current.io_totals()["reused_blocks"] == 15
    current.publish([FIRST], 100, len(BLOCK))
    current.write(SECOND, memoryview(BLOCK))
    current.publish([SECOND], 200, len(BLOCK))
    assert current.io_totals()["written_file_bytes"] == 2 * size
    successor = store(tmp_path, generation="compacted")
    successor.write(THIRD, memoryview(BLOCK))
    successor.publish([THIRD], 50, len(BLOCK))
    assert successor.io_totals()["written_file_bytes"] == 3 * size
    assert successor.io_totals()["written_blocks"] == 3
    assert len(list(successor.generations.rglob("*.qkv"))) == 1


def test_batch_lookup_opens_metadata_once(tmp_path, monkeypatch):
    current = store(tmp_path)
    current.write(FIRST, memoryview(BLOCK))
    count = []
    metadata = current.metadata
    monkeypatch.setattr(current, "metadata", lambda: (count.append(1), metadata())[1])
    assert current.exists_many([FIRST, SECOND, THIRD]) == [True, False, False]
    assert len(count) == 1


def test_interrupted_generation_activation_can_recreate_its_empty_directory(tmp_path):
    current = store(tmp_path)
    current.generation.rmdir()
    current.activate()
    assert current.generation.is_dir()
    assert current.metadata()["status"] == "empty"
    current.write(FIRST, memoryview(BLOCK))
    assert current.publish([FIRST], 1648, len(BLOCK))


def test_failed_metadata_commit_does_not_collect_previous_head(tmp_path, monkeypatch):
    current = store(tmp_path)
    current.write(FIRST, memoryview(BLOCK))
    current.publish([FIRST], 1648, len(BLOCK))
    current.write(SECOND, memoryview(BLOCK))
    original = cache.atomic_write

    def fail(path, content):
        if path.name == "chat.json":
            raise OSError("disk full")
        return original(path, content)

    monkeypatch.setattr(cache, "atomic_write", fail)
    with pytest.raises(OSError, match="disk full"):
        current.publish([SECOND], 3296, len(BLOCK))
    assert current.metadata()["head"] == [FIRST]
    assert current.read(FIRST, len(BLOCK)) == BLOCK


def test_retirement_waits_for_inflight_write_then_rejects_late_writer(tmp_path, monkeypatch):
    current = store(tmp_path)
    entered, release, retiring = Event(), Event(), Event()
    original = cache.atomic_write

    def slow_write(path, content):
        if path.suffix == ".qkv":
            entered.set()
            assert release.wait(5)
        return original(path, content)

    monkeypatch.setattr(cache, "atomic_write", slow_write)
    successor = cache.ChatStore(tmp_path, chat(generation="compacted"))

    def retire():
        retiring.set()
        return successor.activate()

    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(current.write, FIRST, memoryview(BLOCK))
        assert entered.wait(5)
        cleanup = pool.submit(retire)
        assert retiring.wait(5)
        assert not cleanup.done()
        release.set()
        assert writer.result(timeout=5)
        assert cleanup.result(timeout=5)["removed_file_bytes"] > 0
    assert not current.write(SECOND, memoryview(BLOCK))
    assert not current.generation.exists()


def test_report_lists_chat_and_compression_without_guessing_legacy_ownership(tmp_path):
    current = store(tmp_path)
    current.write(FIRST, memoryview(BLOCK))
    current.publish([FIRST], 1648, len(BLOCK))
    (tmp_path / "legacy.bin").write_bytes(BLOCK)
    result = cache.report(tmp_path)
    row = result["chats"][0]
    assert row["raw_bytes"] == len(BLOCK)
    assert row["file_bytes"] < row["raw_bytes"]
    assert row["tokens"] == 1648
    assert row["session_file"] == "/sessions/one.jsonl"
    assert result["legacy_unassigned_bytes"] == len(BLOCK)


def test_object_paths_and_generation_symlinks_are_rejected(tmp_path):
    current = store(tmp_path)
    with pytest.raises(ValueError, match="key"):
        current.write("../victim", memoryview(BLOCK))
    victim = tmp_path / "victim"
    victim.write_bytes(BLOCK)
    (current.generation / FIRST).symlink_to(victim)
    with pytest.raises(ValueError, match="symlink"):
        current.write(FIRST, memoryview(BLOCK))
    assert victim.read_bytes() == BLOCK


def load_tier(monkeypatch):
    fake_base = ModuleType("vllm.v1.kv_offload.base")
    fake_base.OffloadPolicy = SimpleNamespace(
        BLOCK_LEVEL="block_level", REQUEST_LEVEL="request_level"
    )
    fake_base.get_offload_block_hash = lambda key: key[:-4]
    fake_base.get_offload_group_idx = lambda key: int.from_bytes(key[-4:], "big")
    fake_fs = ModuleType("vllm.v1.kv_offload.tiering.fs.manager")
    fake_fs.FileSystemTierManager = type(
        "FS",
        (),
        {
            "drain_jobs": lambda self: None,
            "get_finished_jobs": lambda self: list(self.completions),
            "on_request_finished": lambda *_: None,
            "on_new_request": lambda *_: SimpleNamespace(policy="block_level"),
            "shutdown": lambda self: None,
        },
    )
    fake_fs.FsAsyncLookupManager = type("Lookup", (), {})
    fake_tiering_base = ModuleType("vllm.v1.kv_offload.tiering.base")
    fake_tiering_base.JobResult = lambda *, job_id, success: SimpleNamespace(
        job_id=job_id, success=success
    )
    monkeypatch.setitem(sys.modules, "vllm.v1.kv_offload.base", fake_base)
    monkeypatch.setitem(sys.modules, "vllm.v1.kv_offload.tiering.base", fake_tiering_base)
    monkeypatch.setitem(sys.modules, "vllm.v1.kv_offload.tiering.fs.manager", fake_fs)
    monkeypatch.setitem(sys.modules, "qwen_radiance_cache", cache)
    spec = importlib.util.spec_from_file_location(
        "chat_tier", ROOT / "experiments/radiance-public/radiance_chat_tier.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def new_tier(module):
    tier = module.ChatFileSystemTierManager.__new__(module.ChatFileSystemTierManager)
    tier._chat_mutex = RLock()
    tier._publish_wake = Event()
    tier._chat_stores = {}
    tier._tail_heads = {}
    tier._ignored_jobs = []
    tier._tail_flush_tokens = 8192
    tier._tail_ram_max_bytes = 6 * 1024**3
    tier._tail_ram_max_chats = 5
    tier._tail_block_limit = 15
    tier._last_tail_flush = None
    tier._publish_tail_status = lambda **_kwargs: None
    return tier


@pytest.mark.parametrize("damage", [False, True])
def test_load_completion_invalidates_a_failed_cached_hit(monkeypatch, tmp_path, damage):
    module = load_tier(monkeypatch)
    tier = new_tier(module)
    current = store(tmp_path)
    key = bytes.fromhex("ab" * 32) + (0).to_bytes(4, "big")
    name = module.object_key(key)
    current.write(name, memoryview(BLOCK))
    if damage:
        path = current.path(name)
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
    state = {
        "store": current,
        "jobs": set(),
        "failed": False,
        "finished": False,
        "head": None,
        "tail_keys": set(),
        "tail_blocks": {},
        "force_flush": True,
        "sequence": 1,
    }
    tier._chat_requests = {"read": state}
    tier._chat_jobs = {}
    tier._load_job_keys = {}
    tier._store_job_keys = {}
    tier.events = None
    tier._block_size = len(BLOCK)
    destination = bytearray(b"x" * len(BLOCK))
    tier._primary_kv_view = memoryview(destination)
    tier._ram_blocks = lambda *_: {}
    cached_hits = {key: True, b"unrelated": True}
    completions = []

    def enqueue(job_id, count, tasks):
        assert count == len(tasks) == 1
        try:
            tasks[0]()
        except cache.SnapshotIntegrityError:
            success = False
        else:
            success = True
        completions.append(SimpleNamespace(job_id=job_id, success=success))

    def parent_completions(self):
        # v0.28's parent consumes this registration to invalidate cached HITs.
        # Its native implementation is also exercised in the pinned CPU probe.
        for result in completions:
            keys = self._load_job_keys.pop(result.job_id, None)
            if not result.success and keys is not None:
                for failed in keys:
                    cached_hits[failed] = False
        return list(completions)

    monkeypatch.setattr(module.FileSystemTierManager, "get_finished_jobs", parent_completions)
    tier._pool = SimpleNamespace(enqueue_load=enqueue, enqueue_store=enqueue)
    tier.submit_load(
        SimpleNamespace(
            job_id=71, req_context=SimpleNamespace(req_id="read"), keys=[key], block_ids=[0]
        )
    )
    result = tier.get_finished_jobs()
    assert result[0].success is not damage
    assert cached_hits[key] is not damage
    assert cached_hits[b"unrelated"] is True
    assert state["failed"] is False
    assert current.path(name).exists() is not damage
    assert not state["jobs"] and not tier._chat_jobs and not tier._load_job_keys
    assert destination == (b"x" * len(BLOCK) if damage else BLOCK)
    tier._publish_ready()
    assert tier._chat_requests == {"read": state}
    assert current.metadata()["tokens"] == 0

    # Successful recomputation supplies new primary bytes. A failed read alone
    # does not complete a request or certify those bytes for publication.
    destination[:] = BLOCK
    completions.clear()
    tier.submit_store(
        SimpleNamespace(
            job_id=72, req_context=SimpleNamespace(req_id="read"), keys=[key], block_ids=[0]
        )
    )
    tier.get_finished_jobs()
    state["head"] = ([name], 1648)
    tier.on_request_finished(SimpleNamespace(req_id="read"))
    tier._publish_ready()
    assert current.read(name, len(BLOCK)) == BLOCK
    assert current.metadata()["publication"]["result"] == "committed"
    assert current.metadata()["tokens"] == 1648


def test_runtime_head_uses_declared_windows_and_eagle_tail(monkeypatch):
    tier = load_tier(monkeypatch)
    configs = [
        SimpleNamespace(sliding_window_size_in_chunks=window, is_eagle_group=eagle)
        for window, eagle in ((1, False), (None, False), (2, True))
    ]
    states = [
        SimpleNamespace(offload_keys=[f"g{g}-{i}" for i in range(6)], next_stored_chunk_idx=6)
        for g in range(3)
    ]
    status = SimpleNamespace(
        config=SimpleNamespace(kv_group_configs=configs),
        group_states=states,
        storable_chunks=lambda *_: 6,
    )
    assert tier.head_keys(status, 100) == [
        "g0-5",
        *states[1].offload_keys,
        *states[2].offload_keys[-3:],
    ]
    status.storable_chunks = lambda *_: 5
    # Final EAGLE exclusion must not remove the stable full-attention page that
    # was already stored at the end of prefill. Sliding tails use the settled
    # range, so they still exclude the volatile final chunk.
    assert tier.head_keys(status, 100) == [
        "g0-4",
        *states[1].offload_keys,
        *states[2].offload_keys[2:5],
    ]
    # In v0.28 the target Mamba group is no longer marked EAGLE. Keep its prior
    # aligned state so lookup can converge with the draft group's rollback.
    configs[0].requires_cow_source = True
    assert tier.tail_keys(status, 100) == ["g0-3", "g0-4", *states[2].offload_keys[2:5]]
    assert tier.head_keys(status, 100) == [
        "g0-3",
        "g0-4",
        *states[1].offload_keys,
        *states[2].offload_keys[2:5],
    ]


def test_changing_tail_stays_in_ram_until_explicit_flush(tmp_path, monkeypatch):
    module = load_tier(monkeypatch)
    tier = new_tier(module)
    current = store(tmp_path)
    key = bytes.fromhex("a" * 64) + (6).to_bytes(4, "big")
    state = {
        "store": current,
        "jobs": set(),
        "finished": True,
        "failed": False,
        "head": ([FIRST], 4000),
        "tail_keys": {FIRST},
        "tail_blocks": {},
        "force_flush": False,
        "sequence": 1,
    }
    tier._block_size = len(BLOCK)
    tier._primary_kv_view = memoryview(BLOCK)
    tier._chat_requests = {"request": state}
    tier._chat_stores[current.chat["id"]] = current

    tier._store(state, [key], [0])
    assert not current.path(FIRST).exists()
    tier._publish_ready()
    assert current.metadata()["tokens"] == 0
    assert len(tier._tail_heads) == 1

    context = SimpleNamespace(req_id="lookup")
    tier._chat_requests["lookup"] = {**state, "tail_blocks": {}, "finished": False}
    lookup = module.ChatLookup.__new__(module.ChatLookup)
    lookup._tier = tier
    assert lookup.batch_lookup([key], context) == [True]
    destination = bytearray(len(BLOCK))
    tier._primary_kv_view = memoryview(destination)
    tier._load(tier._chat_requests["lookup"], [key], [0])
    assert destination == BLOCK
    del tier._chat_requests["lookup"]

    result = tier._force_identity(current.chat)
    assert result["status"] == "flushed"
    assert result["reason"] == "explicit"
    assert current.read(FIRST, len(BLOCK)) == BLOCK
    assert current.metadata()["tokens"] == 4000
    assert not tier._tail_heads


def test_tail_interval_keeps_old_head_until_replacement_verifies(tmp_path, monkeypatch):
    module = load_tier(monkeypatch)
    tier = new_tier(module)
    current = store(tmp_path)
    current.write(FIRST, memoryview(BLOCK))
    current.publish([FIRST], 100, len(BLOCK))
    tier._block_size = len(BLOCK)
    tier._chat_stores[current.chat["id"]] = current
    tier._chat_requests = {
        "request": {
            "store": current,
            "jobs": set(),
            "finished": True,
            "failed": False,
            "head": ([FIRST, SECOND], 100 + 8192),
            "tail_keys": {SECOND},
            "tail_blocks": {SECOND: BLOCK},
            "force_flush": False,
            "sequence": 1,
        }
    }
    prepare = current.prepare_publication

    def observe_old_head(keys, block_size):
        assert current.metadata()["head"] == [FIRST]
        assert current.path(SECOND).exists()
        return prepare(keys, block_size)

    monkeypatch.setattr(current, "prepare_publication", observe_old_head)
    tier._publish_ready()
    assert current.metadata()["head"] == sorted([FIRST, SECOND])
    assert current.metadata()["tokens"] == 8292
    assert not tier._tail_heads


def test_tmpfs_control_request_waits_for_backend_tail_publication(tmp_path, monkeypatch):
    module = load_tier(monkeypatch)
    tier = new_tier(module)
    current = store(tmp_path)
    tier._block_size = len(BLOCK)
    tier._chat_stores[current.chat["id"]] = current
    tier._chat_requests = {
        "request": {
            "store": current,
            "jobs": set(),
            "finished": True,
            "failed": False,
            "head": ([FIRST], 4000),
            "tail_keys": {FIRST},
            "tail_blocks": {FIRST: BLOCK},
            "force_flush": False,
            "sequence": 1,
        }
    }
    tier._publish_ready()
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    tier._control_directory = control

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(
            cache.request_tail_flush,
            current.chat,
            timeout=2,
            control_directory=control,
        )
        deadline = time.monotonic() + 1
        while not list(control.glob("*.request.json")) and time.monotonic() < deadline:
            time.sleep(0.01)
        tier._process_control_requests()
        result = pending.result(timeout=2)

    assert result["status"] == "flushed"
    assert current.metadata()["tokens"] == 4000


def test_ram_budget_flushes_before_evicting_oldest_tail(tmp_path, monkeypatch):
    module = load_tier(monkeypatch)
    tier = new_tier(module)
    tier._block_size = len(BLOCK)
    tier._tail_ram_max_chats = 1
    stores = [store(tmp_path, name) for name in ("one", "two")]
    tier._chat_stores = {current.chat["id"]: current for current in stores}
    tier._chat_requests = {
        f"request-{index}": {
            "store": current,
            "jobs": set(),
            "finished": True,
            "failed": False,
            "head": ([FIRST], 4000),
            "tail_keys": {FIRST},
            "tail_blocks": {FIRST: BLOCK},
            "force_flush": False,
            "sequence": index,
        }
        for index, current in enumerate(stores, start=1)
    }

    tier._publish_ready()

    assert len(tier._tail_heads) == 1
    assert sorted(current.metadata()["tokens"] for current in stores) == [0, 4000]
    durable = next(current for current in stores if current.metadata()["tokens"] == 4000)
    assert durable.read(FIRST, len(BLOCK)) == BLOCK


@pytest.mark.parametrize("phase", ["request", "settled", "retry"])
def test_clean_shutdown_flushes_complete_ram_tail(tmp_path, monkeypatch, phase):
    module = load_tier(monkeypatch)
    tier = new_tier(module)
    current = store(tmp_path)
    tier._block_size = len(BLOCK)
    tier._chat_stores = {current.chat["id"]: current}
    tier._chat_jobs = {}
    tier._chat_requests = {
        "complete": {
            "store": current,
            "jobs": set(),
            "finished": True,
            "failed": False,
            "head": ([FIRST], 4000),
            "tail_keys": {FIRST},
            "tail_blocks": {FIRST: BLOCK},
            "force_flush": False,
            "sequence": 1,
        },
        "interrupted": {
            "store": current,
            "jobs": set(),
            "finished": False,
            "failed": False,
            "head": None,
            "tail_keys": set(),
            "tail_blocks": {},
            "force_flush": False,
            "sequence": 2,
        },
    }
    tier.completions = []
    tier._retry_stop = Event()
    tier._retry_thread = SimpleNamespace(join=lambda: None)
    tier._control_directory = tmp_path / "control"
    tier._control_directory.mkdir(mode=0o700)
    tier._tail_status_path = tmp_path / "tail-status.json"
    tier._engine_lock = os.open(tmp_path / "engine.lock", os.O_CREAT | os.O_RDWR, 0o600)

    if phase != "request":
        del tier._chat_requests["interrupted"]
        tier._publish_ready()
        assert not tier._chat_requests
        assert current.metadata().get("tokens", 0) == 0
        assert not current.path(FIRST).exists()
        assert len(tier._tail_heads) == 1
        if phase == "retry":
            import time

            next(iter(tier._tail_heads.values()))["retry_after"] = time.monotonic() + 3600

    tier.shutdown()

    assert current.metadata()["tokens"] == 4000
    assert current.read(FIRST, len(BLOCK)) == BLOCK
    assert not tier._tail_heads


def test_ram_hit_requests_repair_disk_and_retired_writers_bypass_durable_tier(
    tmp_path, monkeypatch
):
    module = load_tier(monkeypatch)
    tier = new_tier(module)
    tier._root = tmp_path
    tier._sequence = 0
    tier._chat_requests = {}
    current = store(tmp_path)
    request = SimpleNamespace(req_id="first", kv_transfer_params={"qwen_chat": current.chat})
    result = tier.on_new_request(request)
    assert result.policy == "request_level"

    successor = cache.ChatStore(tmp_path, chat(generation="compacted"))
    successor.activate()
    request.req_id = "late-old-generation"
    result = tier.on_new_request(request)
    assert result.policy == "block_level"
    assert request.req_id not in tier._chat_requests
    assert successor.current()

    # Exercise the other path after the manager has learned the new generation.
    tier._chat_stores[current.chat["id"]] = successor
    request.req_id = "late-old-generation-after-successor"
    result = tier.on_new_request(request)
    assert result.policy == "block_level"
    assert request.req_id not in tier._chat_requests
    assert successor.current()


def test_unlabelled_requests_never_read_or_write_the_snapshot_tier(monkeypatch):
    module = load_tier(monkeypatch)
    tier = new_tier(module)
    tier._chat_requests = {}
    tier._chat_jobs = {}
    tier.completions = []
    context = SimpleNamespace(req_id="unlabelled")
    job = SimpleNamespace(job_id=7, req_context=context)

    lookup = module.ChatLookup.__new__(module.ChatLookup)
    lookup._tier = tier
    assert lookup.batch_lookup([b"one", b"two"], context) == [False, False]
    assert tier.submit_store(job) is None
    results = tier.get_finished_jobs()
    assert [(result.job_id, result.success) for result in results] == [(7, True)]

    job.job_id = 8
    assert tier.submit_load(job) is None
    results = tier.get_finished_jobs()
    assert [(result.job_id, result.success) for result in results] == [(8, False)]


def test_tier_waits_for_all_async_jobs_and_successor_before_collecting(tmp_path, monkeypatch):
    module = load_tier(monkeypatch)
    tier = new_tier(module)
    current = store(tmp_path)
    for key in (FIRST, SECOND):
        current.write(key, memoryview(BLOCK))
    tier._block_size = len(BLOCK)
    tier._chat_jobs = {1: ("old", True), 2: ("new", True)}
    tier._chat_requests = {
        "old": {
            "store": current,
            "jobs": {1},
            "finished": True,
            "failed": False,
            "head": ([FIRST], 1648),
            "tail_keys": set(),
            "tail_blocks": {},
            "force_flush": False,
            "sequence": 1,
        },
        "new": {
            "store": current,
            "jobs": {2},
            "finished": False,
            "failed": False,
            "head": ([SECOND], 3296),
            "tail_keys": set(),
            "tail_blocks": {},
            "force_flush": True,
            "sequence": 2,
        },
    }
    tier.completions = [SimpleNamespace(job_id=1, success=True)]
    tier.get_finished_jobs()
    assert current.path(FIRST).exists() and current.path(SECOND).exists()
    tier.on_request_finished(SimpleNamespace(req_id="new"))
    assert current.path(FIRST).exists()
    tier.completions = [SimpleNamespace(job_id=2, success=True)]
    tier.get_finished_jobs()
    assert current.path(FIRST).exists()  # Scheduler polling does no publication I/O.
    tier._publish_ready()
    assert not current.path(FIRST).exists()
    assert current.read(SECOND, len(BLOCK)) == BLOCK
    assert current.metadata()["tokens"] == 3296


def test_tier_store_failure_rolls_back_partial_writes(tmp_path, monkeypatch):
    module = load_tier(monkeypatch)
    tier = new_tier(module)
    current = store(tmp_path)
    current.write(FIRST, memoryview(BLOCK))
    current.publish([FIRST], 1648, len(BLOCK))
    current.write(THIRD, memoryview(BLOCK))
    tier._block_size = len(BLOCK)
    tier._chat_jobs = {1: ("new", True)}
    tier._chat_requests = {
        "new": {
            "store": current,
            "jobs": {1},
            "finished": True,
            "failed": False,
            "head": ([SECOND], 3296),
            "tail_keys": set(),
            "tail_blocks": {},
            "force_flush": True,
            "sequence": 1,
        }
    }
    tier.completions = [SimpleNamespace(job_id=1, success=False)]
    tier.get_finished_jobs()
    tier._publish_ready()
    assert current.metadata()["head"] == [FIRST]
    assert current.read(FIRST, len(BLOCK)) == BLOCK
    assert not current.path(THIRD).exists()
    assert current.metadata()["status"] == "incomplete"


@pytest.mark.parametrize("missing", [True, False])
def test_finished_prefill_collects_replaced_or_abandoned_snapshot(tmp_path, monkeypatch, missing):
    module = load_tier(monkeypatch)
    tier = new_tier(module)
    current, other = store(tmp_path), store(tmp_path, "other")
    for item in (current, other):
        item.write(FIRST, memoryview(BLOCK))
        item.publish([FIRST], 207186, len(BLOCK))
    current.write(SECOND, memoryview(BLOCK))
    tier._block_size = len(BLOCK)
    tier._chat_requests = {
        "prefill": {
            "store": current,
            "jobs": set(),
            "finished": True,
            "failed": False,
            "head": ([SECOND, THIRD] if missing else [SECOND], 211152),
            "tail_keys": set(),
            "tail_blocks": {},
            "force_flush": True,
            "sequence": 1,
        }
    }
    tier._publish_ready()
    assert not tier._chat_requests
    assert {p.name for p in current.generation.iterdir()} == ({FIRST} if missing else {SECOND})
    assert other.read(FIRST, len(BLOCK)) == BLOCK


def test_failed_gc_retries_during_idle_and_preserves_committed_head(tmp_path, monkeypatch):
    import errno

    module = load_tier(monkeypatch)
    tier = new_tier(module)
    current = store(tmp_path)
    current.write(FIRST, memoryview(BLOCK))
    current.publish([FIRST], 100, len(BLOCK))
    current.write(SECOND, memoryview(BLOCK))
    tier._block_size = len(BLOCK)
    tier._chat_requests = {
        "prefill": {
            "store": current,
            "jobs": set(),
            "finished": True,
            "failed": False,
            "head": ([SECOND], 200),
            "tail_keys": set(),
            "tail_blocks": {},
            "force_flush": True,
            "sequence": 1,
        }
    }
    unlink = Path.unlink

    def fail_once(path, *args, **kwargs):
        if path == current.path(FIRST):
            raise OSError(errno.EIO, "synthetic deletion failure")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_once)
    tier._publish_ready()
    assert current.metadata()["head"] == [SECOND]
    assert current.metadata()["gc"]["status"] == "failed"
    assert tier._tail_heads
    monkeypatch.setattr(Path, "unlink", unlink)
    # Exercise the idle timer's callback with a virtual clock, without a chat turn.
    stop = iter([False, False, True])
    tier._retry_stop = SimpleNamespace(is_set=lambda: next(stop))
    tier._publish_wake = SimpleNamespace(wait=lambda _: None, clear=lambda: None)
    tier._process_control_requests = lambda: None
    monkeypatch.setattr(module.time, "monotonic", lambda: float("inf"))
    tier._retry_gc()
    assert not tier._tail_heads
    assert not current.path(FIRST).exists()
    assert current.read(SECOND, len(BLOCK)) == BLOCK
    assert current.metadata()["gc"]["status"] == "complete"


def test_recovery_uses_durable_head_after_process_loss(tmp_path):
    current = store(tmp_path)
    current.write(FIRST, memoryview(BLOCK))
    current.publish([FIRST], 100, len(BLOCK))
    current.write(SECOND, memoryview(BLOCK))
    current.save_metadata({**current.metadata(), "gc": {"status": "pending"}})
    reopened = cache.ChatStore(tmp_path, chat())
    reopened.collect()
    assert reopened.read(FIRST, len(BLOCK)) == BLOCK
    assert not reopened.path(SECOND).exists()
    assert reopened.metadata()["gc"]["status"] == "complete"


def test_old_prune_refuses_unsafe_orphan_heuristic(tmp_path):
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/qwen-radiance-cache-prune"),
            "--host",
            "local",
            "--cache-root",
            str(tmp_path),
            "--prune-orphans",
            "--apply",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "deletion refused" in result.stderr

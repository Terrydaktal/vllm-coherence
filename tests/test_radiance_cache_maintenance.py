"""Cache maintenance against real ChatStore writes; no inference or chat transcripts."""

import fcntl
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_r9700_lab import radiance_cache as cache
from qwen_r9700_lab import radiance_cache_audit as audit
from qwen_r9700_lab import radiance_cache_cli as cli

ABI = "a" * 64
OTHER_ABI = "b" * 64
KEY = "g0-" + "c" * 64 + ".qkv"
BLOCK = b"numeric cache fixture" * 1024


def fixture(root, *, abi=ABI, title="Synthetic release smoke", cwd="/qualification", name="test", generation="first"):
    data = root / "snapshots" / abi / "data"
    data.mkdir(parents=True, exist_ok=True)
    identity = {"id": hashlib.sha256(name.encode()).hexdigest(),
                "generation": hashlib.sha256(generation.encode()).hexdigest(),
                "title": title, "cwd": cwd, "session_file": ""}
    store = cache.ChatStore(data, identity)
    store.activate()
    (store.managed / ".engine.lock").touch()
    assert store.write(KEY, memoryview(BLOCK))
    assert store.publish([KEY], 100, len(BLOCK))
    return store


def file_inventory(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.fixture(autouse=True)
def metadata_only_audit(monkeypatch):
    monkeypatch.setattr(audit, "drive_health", lambda _: {"available": False})
    monkeypatch.setattr(audit, "fair_scheduler_status", lambda: {"available": False})
    monkeypatch.setattr(audit, "snapshot_tail_status", lambda: {"available": False})


@pytest.mark.parametrize("title,cwd", [
    ("Synthetic release smoke", "/qualification"),
    ("Synthetic relay probe", "/workspace/qualification"),
])
def test_purge_preserves_real_chat_and_lifetime_history(tmp_path, title, cwd):
    test = fixture(tmp_path, title=title, cwd=cwd)
    real = fixture(tmp_path, title="work", cwd="/home/lewis/tasks/work", name="real")
    real_bytes = file_inventory(real.directory)
    before = audit.scan(tmp_path)["io"]["lifetime"]
    preview_inventory = file_inventory(tmp_path)
    preview = cache.purge_test_chats(tmp_path, dry_run=True)
    assert len(preview["chats"]) == 1 and preview["removed_file_bytes"] == 0
    assert file_inventory(tmp_path) == preview_inventory
    result = cache.purge_test_chats(tmp_path)
    assert len(result["chats"]) == 1 and result["removed_file_bytes"] > 0
    assert not result["skipped"]
    assert not list(test.directory.rglob("*.qkv"))
    assert file_inventory(real.directory) == real_bytes
    after = audit.scan(tmp_path)
    assert [row["id"] for row in after["chats"]] == [real.chat["id"]]
    assert after["io"]["lifetime"]["written_file_bytes"] == before["written_file_bytes"]
    assert after["io"]["lifetime"]["written_blocks"] == before["written_blocks"]
    assert after["io"]["lifetime"]["deleted_chats"] == 1
    assert after["io"]["lifetime"]["deleted_written_file_bytes"] == test.io_totals()["written_file_bytes"]
    assert not test.write(KEY, memoryview(BLOCK))
    assert not test.publish([KEY], 100, len(BLOCK))
    with pytest.raises(cache.RetiredGenerationError):
        test.activate()
    assert cache.purge_test_chats(tmp_path)["chats"] == []


@pytest.mark.parametrize("title,cwd", [
    ("Synthetic release smoke", "/home/lewis/tasks/work"),
    ("Synthetic relay probe copy", "/workspace/qualification"),
    ("Real release smoke", "/qualification"),
])
def test_test_selector_does_not_delete_similarly_named_real_chats(tmp_path, title, cwd):
    store = fixture(tmp_path, title=title, cwd=cwd)
    before = file_inventory(store.directory)
    assert cache.purge_test_chats(tmp_path)["chats"] == []
    assert file_inventory(store.directory) == before


def test_reused_test_identity_keeps_cumulative_counter_and_retires_cleanly(tmp_path):
    initial = fixture(tmp_path)
    first_io = initial.io_totals()
    cache.purge_test_chats(tmp_path)
    revived = fixture(tmp_path, generation="second")
    assert revived.io_totals()["written_file_bytes"] == 2 * first_io["written_file_bytes"]
    assert audit.scan(tmp_path)["io"]["lifetime"]["written_blocks"] == 2
    cache.purge_test_chats(tmp_path)
    assert audit.scan(tmp_path)["io"]["lifetime"]["written_blocks"] == 2
    current = fixture(tmp_path, abi=OTHER_ABI, title="real", cwd="/work", name="current")
    cache.retire_incompatible_snapshots(current.root, apply=True)
    assert not initial.directory.exists()
    total = audit.scan(tmp_path)["io"]["lifetime"]
    assert total["written_blocks"] == 3 and total["deleted_chats"] == 1


def test_abi_retirement_is_in_lifetime_total_without_double_counting_aliases(tmp_path):
    current = fixture(tmp_path, title="real", cwd="/work")
    old = fixture(tmp_path, abi=OTHER_ABI, title="real", cwd="/work")
    alias = tmp_path / "snapshots" / ("d" * 64)
    alias.mkdir()
    (alias / "data").symlink_to(current.root, target_is_directory=True)
    (alias / "abi.json").write_text(json.dumps({"storage": {"data_abi": ABI}}))
    before = audit.scan(tmp_path)["io"]["lifetime"]
    cache.retire_incompatible_snapshots(current.root, apply=True)
    assert not old.directory.exists()
    for filter_abi in (None, ABI, "d" * 64):
        after = audit.scan(tmp_path, abi=filter_abi)["io"]["lifetime"]
        assert after["written_file_bytes"] == before["written_file_bytes"]
        assert after["written_blocks"] == 2 and after["deleted_chats"] == 1


def test_pending_deletion_is_not_double_counted_and_keeps_latest_high_water(tmp_path):
    store = fixture(tmp_path)
    counters = store.io_totals()
    entry = {"abi": ABI, "chat_id": store.chat["id"], "status": "pending", "io": counters}
    (tmp_path / "snapshot-retirements.json").write_text(json.dumps({
        "schema": cache.RETIREMENT_SCHEMA, "entries": {f"{ABI}/{store.chat['id']}": entry},
    }))
    total = audit.scan(tmp_path)["io"]["lifetime"]
    assert total["written_file_bytes"] == counters["written_file_bytes"]
    assert total["deleted_chats"] == 0
    other_key = "g0-" + "e" * 64 + ".qkv"
    store.write(other_key, memoryview(BLOCK))
    assert audit.scan(tmp_path)["io"]["lifetime"]["written_blocks"] == 2
    cache.purge_test_chats(tmp_path)
    assert audit.scan(tmp_path)["io"]["lifetime"]["written_blocks"] == 2


def test_recreated_retired_abi_adds_new_counter_epoch_without_forgetting_old_writes(tmp_path):
    old = fixture(tmp_path, abi=OTHER_ABI, title="work", cwd="/work")
    current = fixture(tmp_path, title="work", cwd="/work")
    cache.retire_incompatible_snapshots(current.root, apply=True)
    assert not old.directory.exists()
    fixture(tmp_path, abi=OTHER_ABI, title="work", cwd="/work")
    assert audit.scan(tmp_path)["io"]["lifetime"]["written_blocks"] == 3
    cache.retire_incompatible_snapshots(current.root, apply=True)
    assert audit.scan(tmp_path)["io"]["lifetime"]["written_blocks"] == 3
    fixture(tmp_path, abi=OTHER_ABI, title="work", cwd="/work")
    assert audit.scan(tmp_path)["io"]["lifetime"]["written_blocks"] == 4
    cache.retire_incompatible_snapshots(current.root, apply=True)
    assert audit.scan(tmp_path)["io"]["lifetime"]["written_blocks"] == 4


def test_purge_retries_after_crash_without_losing_traffic_or_allowing_late_writers(tmp_path, monkeypatch):
    store = fixture(tmp_path)
    counters = store.io_totals()
    remove = cache.shutil.rmtree

    def interrupted(path):
        remove(path)
        raise OSError("injected crash after payload removal")

    with monkeypatch.context() as patch:
        patch.setattr(cache.shutil, "rmtree", interrupted)
        result = cache.purge_test_chats(tmp_path)
        assert not result["chats"] and result["skipped"][0]["reason"] == "OSError"
    assert not store.current()
    assert not store.write(KEY, memoryview(BLOCK))
    assert audit.scan(tmp_path)["io"]["lifetime"]["written_file_bytes"] == counters["written_file_bytes"]
    assert len(cache.purge_test_chats(tmp_path)["chats"]) == 1
    assert cache.purge_test_chats(tmp_path)["chats"] == []
    assert audit.scan(tmp_path)["chats"] == []


def test_purge_never_removes_payload_if_archiving_history_fails(tmp_path, monkeypatch):
    store = fixture(tmp_path)
    before = file_inventory(store.directory)
    write = cache.atomic_write

    def no_space(path, data):
        if path.name == "snapshot-retirements.json":
            raise OSError("injected ENOSPC")
        return write(path, data)

    monkeypatch.setattr(cache, "atomic_write", no_space)
    assert cache.purge_test_chats(tmp_path)["skipped"][0]["reason"] == "OSError"
    assert file_inventory(store.directory) == before


@pytest.mark.parametrize("unsafe", ["payload_symlink", "unknown_file", "io_symlink", "wrong_id", "fifo_lock", "missing_lease"])
def test_purge_refuses_unknown_or_unsafe_layouts(tmp_path, unsafe):
    store = fixture(tmp_path)
    if unsafe == "payload_symlink":
        (store.generation / ("g0-" + "f" * 64 + ".qkv")).symlink_to(store.path(KEY))
    elif unsafe == "unknown_file":
        (store.directory / "keep.txt").write_text("unowned")
    elif unsafe == "io_symlink":
        (store.directory / "io.json").unlink()
        (store.directory / "io.json").symlink_to(store.directory / "chat.json")
    elif unsafe == "wrong_id":
        store.save_metadata({**store.metadata(), "id": "f" * 64})
    elif unsafe == "fifo_lock":
        import os

        (store.directory / ".lock").unlink()
        os.mkfifo(store.directory / ".lock")
    else:
        (store.managed / ".engine.lock").unlink()
    result = cache.purge_test_chats(tmp_path)
    assert not result["chats"] and result["skipped"]
    assert store.path(KEY).exists()


@pytest.mark.parametrize("protection", ["running", "paused", "queued", "gpu", "ram", "tail", "stale", "missing", "pid_mismatch"])
def test_live_test_requests_and_banks_are_skipped(tmp_path, protection):
    store = fixture(tmp_path)
    prefix, tail = tmp_path / "fair", tmp_path / "tail.json"
    row = {"chat_id": store.chat["id"], "generation": store.chat["generation"]}
    scheduler = {"updated_at": time.time(), "pid": 123, "requests": []}
    worker = {"updated_at": time.time(), "pid": 123, "residency": {"active": None, "images": []}}
    journal = {"updated_at": time.time(), "pid": 123, "chats": []}
    if protection in {"running", "paused", "queued"}:
        scheduler["requests"].append({**row, "state": protection})
    elif protection == "gpu":
        worker["residency"]["active"] = row
    elif protection == "ram":
        worker["residency"]["images"].append(row)
    elif protection == "tail":
        journal["chats"].append(row)
    elif protection == "stale":
        journal["updated_at"] -= 30
    elif protection == "pid_mismatch":
        worker["pid"] = 456
    for suffix, value in (("-scheduler.json", scheduler), ("-worker.json", worker)):
        Path(str(prefix) + suffix).write_text(json.dumps(value))
    if protection != "missing":
        tail.write_text(json.dumps(journal))
    with (store.managed / ".engine.lock").open("r+") as engine:
        fcntl.flock(engine, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = cache.purge_test_chats(tmp_path, status_prefix=prefix, tail_path=tail)
    assert not result["chats"] and result["skipped"]
    assert store.path(KEY).exists()


def test_idle_test_can_be_purged_while_other_chat_owns_engine(tmp_path):
    store = fixture(tmp_path)
    prefix, tail = tmp_path / "fair", tmp_path / "tail.json"
    now = time.time()
    Path(str(prefix) + "-scheduler.json").write_text(json.dumps({"updated_at": now - 600, "pid": 123, "requests": []}))
    Path(str(prefix) + "-worker.json").write_text(json.dumps({"updated_at": now - 600, "pid": 123, "residency": {
        "active": {"chat_id": "f" * 64}, "images": [],
    }}))
    tail.write_text(json.dumps({"updated_at": now, "pid": 123, "chats": []}))
    with (store.managed / ".engine.lock").open("r+") as engine:
        fcntl.flock(engine, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = cache.purge_test_chats(tmp_path, status_prefix=prefix, tail_path=tail)
    assert len(result["chats"]) == 1 and not result["skipped"]
    assert not store.current()


def test_busy_chat_and_global_maintenance_locks_skip_without_waiting(tmp_path):
    store = fixture(tmp_path)
    with store.lock():
        assert cache.purge_test_chats(tmp_path)["skipped"][0]["reason"] == "BlockingIOError"
    with (tmp_path / ".snapshot-retirement.lock").open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert cache.purge_test_chats(tmp_path)["skipped"] == [{"reason": "retirement_busy"}]
    assert store.path(KEY).exists()


@pytest.mark.parametrize("broken", ["malformed", "symlink", "negative_counter"])
def test_invalid_ledger_is_reported_and_never_overwritten(tmp_path, broken):
    store = fixture(tmp_path)
    ledger = tmp_path / "snapshot-retirements.json"
    if broken == "malformed":
        ledger.write_text("not-json")
    elif broken == "symlink":
        ledger.symlink_to(store.directory / "chat.json")
    else:
        ledger.write_text(json.dumps({"schema": cache.RETIREMENT_SCHEMA, "entries": {
            f"{OTHER_ABI}/{'f' * 64}": {"abi": OTHER_ABI, "chat_id": "f" * 64,
                                       "status": "complete", "io": {"available": True, "written_file_bytes": -1}},
        }}))
    total = audit.scan(tmp_path)
    assert not total["io"]["lifetime"]["complete"]
    assert total["io"]["lifetime"]["written_file_bytes"] == store.io_totals()["written_file_bytes"]
    with pytest.raises((OSError, ValueError)):
        cache.purge_test_chats(tmp_path)
    assert store.path(KEY).exists()


def test_cli_preview_and_purge_json_are_installed_commands(tmp_path, capsys):
    fixture(tmp_path)
    common = ["--host", "local", "--cache-root", str(tmp_path), "--no-sessions"]
    assert cli.main([*common, "purge-tests", "--dry-run", "--json"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["dry_run"] and len(preview["chats"]) == 1
    assert cli.main([*common, "purge-tests", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert not result["dry_run"] and result["removed_file_bytes"] > 0
    assert cli.main([*common, "status"]) == 0
    assert "Disk traffic (lifetime):" in capsys.readouterr().out


def test_remote_purge_sends_tested_standalone_source_and_only_requested_args(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cache.subprocess, "run", run)
    assert cli.main(["--host", "test-host", "--cache-root", "/cache with spaces", "purge-tests", "--dry-run", "--json"]) == 0
    command, kwargs = calls[0]
    assert command[-2] == "test-host" and "'/cache with spaces'" in command[-1]
    assert "purge-tests --dry-run --json" in command[-1]
    assert "def purge_test_chats(" in kwargs["input"]

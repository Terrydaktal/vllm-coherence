from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from qwen_r9700_lab import radiance_cache as cache
from qwen_r9700_lab import radiance_cache_audit as audit
from qwen_r9700_lab import radiance_cache_cli as cli

ROOT = Path(__file__).resolve().parents[1]
ABI = "a" * 64
OTHER_ABI = "b" * 64
FIRST = "g6-" + "1" * 64 + ".qkv"
SECOND = "g0-" + "2" * 64 + ".qkv"
EXTRA = "g6-" + "3" * 64 + ".qkv"
BLOCK = b"snapshot fixture\n" * 4096


def make_store(root, *, abi=ABI, name="chat", generation="initial"):
    data = root / "snapshots" / abi / "data"
    data.mkdir(parents=True, exist_ok=True)
    identity = {
        "id": cli.js_digest(name),
        "generation": cli.js_digest(generation),
        "title": name,
        "cwd": "/work",
        "session_file": "/local/chat.jsonl",
    }
    store = cache.ChatStore(data, identity)
    store.activate()
    for key in (FIRST, SECOND):
        store.write(key, memoryview(BLOCK))
    assert store.publish([FIRST, SECOND], 10_000, len(BLOCK))
    return store


def codes(row):
    return {p["code"] for p in row["issues"]}


def inventory(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def run_cli(root, *args):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/qwen-radiance-cache"),
            "--host",
            "local",
            "--cache-root",
            str(root),
            "--no-sessions",
            *args,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("quantum", [0, 15])
def test_fair_scheduler_status_contains_only_identity_state_and_transfer_metrics(tmp_path, quantum):
    prefix = tmp_path / "fair"
    now = time.time()
    (tmp_path / "fair-scheduler.json").write_text(
        json.dumps(
            {
                "updated_at": now,
                "quantum_seconds": quantum,
                "switches": 4,
                "cached_chats": 2,
                "max_cached_chats": 2,
                "requests": [
                    {
                        "chat_id": "1" * 64,
                        "generation": "2" * 64,
                        "state": "paused",
                        "computed_tokens": 120_000,
                        "input_tokens": 119_000,
                    }
                ],
            }
        )
    )
    (tmp_path / "fair-worker.json").write_text(
        json.dumps(
            {
                "updated_at": now,
                "allocated_bytes": 10_000,
                "reserved_capacity_bytes": 10_000,
                "cached_chats": 2,
                "switches": 4,
                "last_transfer_bytes": 8_000,
                "last_transfer_seconds": 0.4,
                "transferred_bytes": 32_000,
                "transfer_seconds": 1.6,
                "last_allocation_bytes": 10_000,
                "last_allocation_seconds": 3.2,
                "allocation_events": 1,
                "allocation_seconds": 3.2,
                "generation_replacements": 2,
                "last_handover": "generation-replace",
                "residency": {
                    "active": {"chat_id": "1" * 64, "generation": "2" * 64},
                    "images": [
                        {
                            "chat_id": "3" * 64,
                            "generation": "4" * 64,
                            "data_bytes": 8_000,
                            "allocated_bytes": 9_000,
                        }
                    ],
                    "free_buffer_bytes": 0,
                    "staging_buffer_bytes": 1_000,
                },
            }
        )
    )
    result = audit.fair_scheduler_status(prefix)
    assert result["available"] and result["active"]
    assert result["policy"] == ("response_boundary" if quantum == 0 else "time_slice")
    assert result["requests"][0]["state"] == "paused"
    assert result["worker"]["last_transfer_seconds"] == 0.4
    assert result["worker"]["last_allocation_seconds"] == 3.2
    assert result["worker"]["generation_replacements"] == 2
    assert result["worker"]["last_handover"] == "generation-replace"
    assert result["worker"]["residency"]["images"][0]["data_bytes"] == 8_000
    assert cli.handover_cell({"scheduler": result}, "1" * 64) == "GPU"
    assert cli.handover_cell({"scheduler": result}, "3" * 64) == "7.8 KiB"
    assert cli.handover_cell({"scheduler": result}, "5" * 64) == "0.0 B"


def test_legacy_worker_telemetry_marks_per_chat_ram_unknown(tmp_path):
    prefix = tmp_path / "fair"
    now = time.time()
    (tmp_path / "fair-scheduler.json").write_text(
        json.dumps(
            {
                "updated_at": now,
                "quantum_seconds": 15,
                "switches": 0,
                "cached_chats": 1,
                "max_cached_chats": 2,
                "requests": [],
            }
        )
    )
    (tmp_path / "fair-worker.json").write_text(
        json.dumps(
            {
                "updated_at": now,
                "allocated_bytes": 0,
                "reserved_capacity_bytes": 10_000,
                "cached_chats": 1,
                "switches": 0,
                "last_transfer_bytes": 0,
                "last_transfer_seconds": 0,
                "transferred_bytes": 0,
                "transfer_seconds": 0,
            }
        )
    )

    result = audit.fair_scheduler_status(prefix)

    assert result["available"]
    assert "residency" not in result["worker"]
    assert cli.handover_cell({"scheduler": result}, "1" * 64) == "?"


def test_snapshot_tail_status_reports_only_bounded_content_free_residency(tmp_path):
    path = tmp_path / "tail.json"
    path.write_text(
        json.dumps(
            {
                "schema": "urn:qwen-r9700:radiance-tail-residency:v1",
                "pid": 123,
                "updated_at": time.time(),
                "flush_tokens": 8192,
                "max_bytes": 6 * 1024**3,
                "max_chats": 5,
                "chats": [
                    {
                        "chat_id": "1" * 64,
                        "generation": "2" * 64,
                        "tokens": 18_000,
                        "durable_tokens": 12_000,
                        "blocks": 15,
                        "bytes": 8000,
                    }
                ],
                "last_flush": {
                    "chat_id": "3" * 64,
                    "generation": "4" * 64,
                    "tokens": 20_000,
                    "reason": "token_interval",
                },
            }
        )
    )

    result = audit.snapshot_tail_status(path)

    assert result["available"] and result["active"]
    assert result["last_flush"]["reason"] == "token_interval"
    report = {"tail_journal": result}
    assert cli.tail_cell(report, "1" * 64) == "7.8 KiB"
    assert cli.tail_residency(report, "1" * 64)["dirty_tokens"] == 6000
    assert cli.tail_cell(report, "5" * 64) == "0.0 B"


def test_complete_manifest_and_file_breakdown_are_read_only(tmp_path):
    make_store(tmp_path)
    (tmp_path / "compiler-cache").write_bytes(b"compiled" * 300)
    before = inventory(tmp_path)
    result = audit.scan(tmp_path)
    row = result["chats"][0]
    assert row["coverage_percent"] == 100
    assert row["present_blocks"] == row["valid_blocks"] == row["expected_blocks"] == 2
    assert row["verified_blocks"] == 0
    assert row["totals"]["raw_bytes"] == 2 * len(BLOCK)
    assert result["totals"]["file_bytes"] == sum(map(len, before.values()))
    assert result["storage"]["other_cache_files"]["file_bytes"] == 2400
    assert [g["id"] for g in row["groups"]] == [0, 6]
    assert not row["issues"]
    assert inventory(tmp_path) == before


@pytest.mark.parametrize("status,code", [("pending", "GC_PENDING"), ("failed", "GC_FAILED")])
def test_persisted_gc_failure_is_visible_even_with_a_complete_manifest(tmp_path, status, code):
    store = make_store(tmp_path)
    store.save_metadata({**store.metadata(), "gc": {"status": status, "errno": 5}})
    row = audit.scan(tmp_path)["chats"][0]
    assert row["coverage_percent"] == 100
    assert code in codes(row)
    result = run_cli(tmp_path, "show", store.chat["id"])
    assert result.returncode == 0
    assert f"Garbage collection: {status}" in result.stdout


def test_failed_gc_and_abandoned_temporary_files_are_visible(tmp_path):
    store = make_store(tmp_path)
    old = store.generations / cli.js_digest("retired")
    old.mkdir()
    shutil.copyfile(store.path(FIRST), old / FIRST)
    (store.generation / ".pending-abandoned").write_bytes(b"unfinished")
    before = inventory(tmp_path)
    result = audit.scan(tmp_path)
    row = result["chats"][0]
    assert {"GC_LEFTOVERS", "TEMP_LEFTOVERS", "DUPLICATE_CONTENT"} <= codes(row)
    assert row["storage"]["old_generations"]["files"] == 1
    assert row["storage"]["temporary"]["file_bytes"] == 10
    assert row["coverage_percent"] == 100
    assert len(result["duplicates"]) == 1
    assert result["duplicates"][0]["duplicate_file_bytes"] == store.path(FIRST).stat().st_size
    assert not result["duplicates"][0]["verified"]
    assert inventory(tmp_path) == before


def test_empty_retired_directory_is_also_gc_residue(tmp_path):
    store = make_store(tmp_path)
    (store.generations / cli.js_digest("retired")).mkdir()
    row = audit.scan(tmp_path)["chats"][0]
    assert "GC_LEFTOVERS" in codes(row)
    assert row["storage"]["old_generations"]["file_bytes"] == 0


def test_missing_blocks_are_not_reported_as_fully_backed_up(tmp_path):
    store = make_store(tmp_path)
    store.path(SECOND).unlink()
    row = audit.scan(tmp_path)["chats"][0]
    assert row["coverage_percent"] == 50
    assert row["missing_blocks"] == [SECOND]
    assert "MISSING_BLOCKS" in codes(row)
    assert cli.health(row) == "ERROR"


@pytest.mark.parametrize("corruption", [b"short", cache.HEADER.pack(b"UNKNOWN!", 20, b"x" * 32)])
def test_invalid_header_does_not_inflate_raw_usage(tmp_path, corruption):
    store = make_store(tmp_path)
    store.path(SECOND).write_bytes(corruption)
    row = audit.scan(tmp_path)["chats"][0]
    assert row["coverage_percent"] == 50
    assert row["totals"]["raw_bytes"] == len(BLOCK)
    assert "BAD_BLOCK_HEADER" in codes(row)


@pytest.mark.skipif(not shutil.which("zstd"), reason="full verification needs zstd")
def test_full_verification_finds_corruption_beyond_a_valid_header(tmp_path):
    store = make_store(tmp_path)
    encoded = store.path(SECOND).read_bytes()
    store.path(SECOND).write_bytes(encoded[:-2])
    assert audit.scan(tmp_path)["chats"][0]["coverage_percent"] == 100
    row = audit.scan(tmp_path, verify=True)["chats"][0]
    assert row["verified_blocks"] == 1
    assert row["coverage_percent"] == 50
    assert "CORRUPT_PAYLOAD" in codes(row)


@pytest.mark.skipif(not shutil.which("zstd"), reason="full verification needs zstd")
def test_raw_and_compressed_payload_verification_are_lossless(tmp_path):
    store = make_store(tmp_path)
    digest = hashlib.sha256(BLOCK).digest()
    store.path(SECOND).write_bytes(cache.HEADER.pack(cache.RAW, len(BLOCK), digest) + BLOCK)
    row = audit.scan(tmp_path, verify=True)["chats"][0]
    assert row["verified_blocks"] == 2
    assert row["coverage_percent"] == 100
    assert not row["issues"]


def test_duplicate_current_copies_across_abis_are_flagged(tmp_path):
    make_store(tmp_path)
    make_store(tmp_path, abi=OTHER_ABI)
    result = audit.scan(tmp_path)
    assert len(result["duplicates"]) == 2
    assert all("MULTIPLE_ABIS" in codes(row) for row in result["chats"])
    assert len(result["duplicate_snapshots"]) == 1
    assert all("DUPLICATE_SNAPSHOTS" in codes(row) for row in result["chats"])


def test_runtime_abi_aliases_share_one_data_inventory_without_false_errors(tmp_path):
    make_store(tmp_path)
    runtime_abi = "c" * 64
    runtime = tmp_path / "snapshots" / runtime_abi
    runtime.mkdir()
    (runtime / "abi.json").write_text(json.dumps({"storage": {"data_abi": ABI}}))
    result = audit.scan(tmp_path)
    assert len(result["chats"]) == 1
    assert not any(problem["code"] == "BAD_ABI" for problem in result["issues"])
    description = next(item for item in result["abis"] if item["id"] == runtime_abi)
    assert description["data_alias"] and description["data_abi"] == ABI

    selected = audit.scan(tmp_path, abi=runtime_abi)
    assert len(selected["chats"]) == 1
    assert selected["chats"][0]["abi"] == runtime_abi


def test_unrecognized_snapshot_directories_are_counted_and_flagged(tmp_path):
    make_store(tmp_path)
    unknown = tmp_path / "snapshots" / "manual-copy"
    unknown.mkdir()
    (unknown / "copy.qkv").write_bytes(b"unassigned duplicate")
    result = audit.scan(tmp_path)
    assert "UNKNOWN_ABI" in codes(result)
    assert result["storage"]["other_snapshot_files"]["file_bytes"] == 20


def test_equal_required_states_and_isolated_chats_are_not_gc_duplicates(tmp_path):
    store = make_store(tmp_path)
    store.write(EXTRA, memoryview(BLOCK))
    store.publish([FIRST, SECOND, EXTRA], 10_000, len(BLOCK))
    make_store(tmp_path, name="another chat")
    assert not audit.scan(tmp_path)["duplicates"]


def test_busy_writer_is_not_misdiagnosed_as_failed_gc(tmp_path):
    store = make_store(tmp_path)
    with store.lock():
        (store.generations / cli.js_digest("retiring")).mkdir()
        (store.generation / ".pending-live-writer").write_bytes(b"not abandoned")
        row = audit.scan(tmp_path)["chats"][0]
    assert row["coverage_percent"] is None
    assert row["consistent"] is False
    assert cli.health(row) == "BUSY/CHANGED"
    assert "BUSY" in codes(row)
    assert not {"GC_LEFTOVERS", "TEMP_LEFTOVERS"} & codes(row)


def test_full_verification_releases_the_lock_and_detects_new_publication(tmp_path, monkeypatch):
    store = make_store(tmp_path)

    def change_metadata(_record):
        with store.lock(exclusive=True):
            metadata = store.metadata()
            store.save_metadata({**metadata, "tokens": 12_000})
        return "verified", None

    monkeypatch.setattr(audit, "verify_block", change_metadata)
    row = audit.scan(tmp_path, verify=True)["chats"][0]
    assert row["consistent"] is False
    assert row["coverage_percent"] is None
    assert "SCAN_CHANGED" in codes(row)


def test_bad_chat_metadata_does_not_hide_its_files_or_other_chats(tmp_path):
    broken = make_store(tmp_path)
    make_store(tmp_path, name="working")
    (broken.directory / "chat.json").write_text("not json")
    result = audit.scan(tmp_path)
    assert len(result["chats"]) == 2
    row = next(r for r in result["chats"] if r["id"] == broken.chat["id"])
    assert row["totals"]["files"] == 6  # Includes durable I/O counters and their lock.
    assert row["coverage_percent"] is None
    assert "BAD_METADATA" in codes(row)


def test_missing_lock_is_not_created_by_inspection(tmp_path):
    store = make_store(tmp_path)
    (store.directory / ".lock").unlink()
    before = inventory(tmp_path)
    row = audit.scan(tmp_path)["chats"][0]
    assert "MISSING_LOCK" in codes(row)
    assert inventory(tmp_path) == before


def test_symlink_and_fifo_metadata_are_skipped_without_following_or_blocking(tmp_path):
    first = make_store(tmp_path, name="symlink")
    second = make_store(tmp_path, name="fifo")
    secret = tmp_path / "outside"
    secret.write_text("never read as metadata")
    (first.directory / "chat.json").unlink()
    (first.directory / "chat.json").symlink_to(secret)
    (second.directory / "chat.json").unlink()
    os.mkfifo(second.directory / "chat.json")
    result = audit.scan(tmp_path)
    assert len(result["chats"]) == 2
    assert all("BAD_METADATA" in codes(r) for r in result["chats"])
    assert secret.read_text() == "never read as metadata"


def test_unpublished_new_files_are_distinguished_from_old_residue(tmp_path):
    store = make_store(tmp_path)
    store.write(EXTRA, memoryview(b"different\n" * 4096))
    assert "UNPUBLISHED" in codes(audit.scan(tmp_path)["chats"][0])
    os.utime(store.path(EXTRA), (1, 1))
    assert "UNREFERENCED_OLD" in codes(audit.scan(tmp_path)["chats"][0])


def write_session(path, *, compaction=False, name="Work", session_id="fixture-session"):
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        {"type": "session", "id": session_id, "cwd": "/work"},
        {"type": "model_change", "modelId": cli.MODEL},
        {"type": "session_info", "name": name},
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "stopReason": "stop",
                "usage": {"input": 20, "output": 5, "cacheRead": 75},
            },
        },
    ]
    if compaction:
        entries.append({"type": "compaction", "id": "new-compaction"})
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return entries


def test_sessions_include_uncached_chats_but_exclude_qualification_artifacts(tmp_path):
    root = tmp_path / "pi"
    actual = root / "agent-8012/sessions/project/real.jsonl"
    write_session(actual)
    write_session(root / "qualification-run/copy.jsonl")
    sessions, problems = cli.discover_sessions([root])
    assert len(sessions) == 1
    assert not problems
    assert sessions[0]["id"] == cli.js_digest([str(actual), "fixture-session"])
    assert sessions[0]["last_turn_tokens"] == 100
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    report = audit.scan(cache_root)
    cli.correlate(report, sessions)
    assert report["unsnapshotted_chats"] == sessions


def test_pi_remote_root_discovers_radiance_sessions_across_profiles(tmp_path):
    root = tmp_path / "pi-remote"
    current = root / "radiance/agent-8012/sessions/project/current.jsonl"
    resumed = root / "legacy/agent-8013/sessions/project/resumed.jsonl"
    write_session(current, session_id="current-session")
    write_session(resumed, session_id="resumed-session")
    write_session(root / "qualification-run/copy.jsonl", session_id="qualification")

    sessions, problems = cli.discover_sessions([root])

    assert not problems
    assert {session["session_id"] for session in sessions} == {
        "current-session",
        "resumed-session",
    }


def test_project_history_deduplicates_hardlinks_and_preserves_cache_identity(tmp_path):
    projects = tmp_path / "tasks"
    legacy_root = tmp_path / "pi-remote"
    legacy = legacy_root / "old/agent-8012/sessions/project/chat.jsonl"
    write_session(legacy)
    local = projects / "money/.pi/sessions/chat.jsonl"
    local.parent.mkdir(parents=True)
    local.hardlink_to(legacy)
    info = local.stat()
    (local.parent / ".identity.json").write_text(
        json.dumps(
            {
                "schema": cli.HISTORY_SCHEMA,
                "sessions": {
                    local.name: {
                        "identity_path": str(legacy),
                        "device": info.st_dev,
                        "inode": info.st_ino,
                    }
                },
            }
        )
    )

    sessions, problems = cli.discover_sessions([projects, legacy_root])

    assert not problems
    assert len(sessions) == 1
    assert sessions[0]["session_file"] == str(local)
    assert sessions[0]["id"] == cli.js_digest([str(legacy), "fixture-session"])


def test_active_pi_marker_is_matched_to_chat_and_rejects_stale_pid(tmp_path):
    root = tmp_path / "pi-remote"
    session_path = root / "radiance/agent-8012/sessions/project/chat.jsonl"
    write_session(session_path)
    sessions, problems = cli.discover_sessions([root])
    assert not problems
    chat_id = sessions[0]["id"]

    process = tmp_path / "proc/123"
    process.mkdir(parents=True)
    (process / "comm").write_text("pi\n")
    fields_after_comm = ["S", *(["0"] * 18), "456"]
    (process / "stat").write_text(f"123 (pi) {' '.join(fields_after_comm)}\n")
    activity = root / "radiance/agent-8012/radiance-active"
    activity.mkdir()
    marker = {
        "schema": cli.ACTIVITY_SCHEMA,
        "chat_id": chat_id,
        "pid": 123,
        "port": 8012,
        "process_start_ticks": "456",
    }
    (activity / "123.json").write_text(json.dumps(marker))

    processes, problems = cli.discover_active_processes([root], tmp_path / "proc")
    assert not problems
    assert processes == [{"chat_id": chat_id, "pid": 123, "port": 8012}]
    report = {"chats": [{"id": chat_id, "metadata": {}, "issues": []}]}
    cli.correlate(report, sessions, processes)
    assert report["chats"][0]["active_processes"] == processes

    marker["process_start_ticks"] = "455"
    (activity / "123.json").write_text(json.dumps(marker))
    processes, problems = cli.discover_active_processes([root], tmp_path / "proc")
    assert not processes
    assert not problems


def test_compaction_generation_mismatch_and_counter_reset_are_explicit(tmp_path):
    path = tmp_path / "session.jsonl"
    write_session(path, compaction=True)
    session = cli.session_info(path)
    assert session["generation"] == cli.js_digest("new-compaction")
    assert session["last_turn_tokens"] is None
    row = {"id": session["id"], "metadata": {"generation": cli.js_digest("initial")}, "issues": []}
    report = {"chats": [row]}
    cli.correlate(report, [session])
    assert "GENERATION_BEHIND" in codes(row)
    assert not report["unsnapshotted_chats"]


def test_cli_exit_status_json_help_and_legacy_compatibility(tmp_path):
    store = make_store(tmp_path)
    healthy = run_cli(tmp_path, "audit", "--json")
    assert healthy.returncode == 0, healthy.stderr
    assert json.loads(healthy.stdout)["chats"][0]["coverage_percent"] == 100
    store.path(FIRST).unlink()
    broken = run_cli(tmp_path, "audit", "--json")
    assert broken.returncode == 1
    assert "MISSING_BLOCKS" in codes(json.loads(broken.stdout)["chats"][0])
    status = run_cli(tmp_path, "status")
    assert status.returncode == 0
    assert "HANDOVER RAM" in status.stdout
    assert "DISK TRAFFIC" in status.stdout
    assert "WRITE TRAFFIC" not in status.stdout
    legacy = run_cli(tmp_path, "list", "--json")
    assert legacy.returncode == 0
    assert isinstance(json.loads(legacy.stdout), list)
    help_text = run_cli(tmp_path, "--help").stdout
    assert all(
        heading in help_text
        for heading in (
            "NAME",
            "SYNOPSIS",
            "DESCRIPTION",
            "OPTIONS",
            "OPERATION",
            "EXAMPLES",
            "FILES",
            "PATHS",
            "SECURITY NOTES",
            "EXIT STATUS",
            "AUTHORS",
        )
    )


def test_watch_is_bounded_and_json_is_one_record_per_refresh(tmp_path):
    result = run_cli(tmp_path, "watch", "--interval", "1", "--count", "2", "--json")
    assert result.returncode == 0, result.stderr
    assert len([json.loads(line) for line in result.stdout.splitlines()]) == 2


def test_single_chat_selection_handles_ambiguity_and_terminal_control_characters(tmp_path):
    make_store(tmp_path, name="Work one")
    make_store(tmp_path, name="Work two\x1b[2J")
    ambiguous = run_cli(tmp_path, "show", "Work")
    assert ambiguous.returncode == 2
    assert "ambiguous" in ambiguous.stderr
    single = run_cli(tmp_path, "show", cli.js_digest("Work two\x1b[2J")[:12])
    assert single.returncode == 0
    assert "\x1b" not in single.stdout


def test_legacy_compaction_hook_still_retires_generations(tmp_path):
    store = make_store(tmp_path)
    successor = {**store.chat, "generation": cli.js_digest("next")}
    result = run_cli(tmp_path, "--abi", ABI, "compact", "--identity-json", json.dumps(successor))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["retained_file_bytes"] > 0
    assert store.generation.exists()
    report = audit.scan(tmp_path)
    assert report["storage"]["fallback"]["file_bytes"] > 0
    assert "GC_LEFTOVERS" not in {item["code"] for row in report["chats"] for item in row["issues"]}

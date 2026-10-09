"""Reusable dashboard inventory tests using synthetic metadata only."""

import copy
import json
import os
from types import SimpleNamespace

import pytest

from qwen_r9700_lab import radiance_cache_inventory as inventory

CHAT, GENERATION, ABI = "a" * 64, "b" * 64, "c" * 64


def arguments(**changes):
    return SimpleNamespace(**{
        "host": "fixture", "cache_root": "/fixture/cache", "abi": None,
        "verify": False, "stale_after": 300, "sessions_root": ["/fixture/sessions"],
        "no_sessions": False, **changes,
    })


def report():
    return {
        "chats": [{
            "id": CHAT, "abi": ABI, "consistent": True, "expected_blocks": 2,
            "metadata": {"generation": GENERATION, "tokens": 900, "title": "Synthetic work"},
            "session": {"generation": GENERATION, "last_turn_tokens": 910},
            "issues": [], "totals": {"file_bytes": 1000},
        }],
        "unsnapshotted_chats": [], "issues": [], "totals": {"file_bytes": 1000},
        "io": {"written_file_bytes": 2000}, "filesystem": {"available_bytes": 3000},
    }


@pytest.fixture
def directory(tmp_path):
    path = tmp_path / "telemetry"
    path.mkdir(mode=0o700)
    return path


def cache_path(directory):
    paths = list(directory.glob("inventory-*.json"))
    assert len(paths) == 1
    return paths[0]


def rewrite(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def test_roundtrip_preserves_metadata_and_reports_real_age(directory, monkeypatch):
    monkeypatch.setattr(inventory.time, "time", lambda: 1000)
    inventory.save(arguments(), directory, report())
    path = cache_path(directory)
    assert path.stat().st_mode & 0o777 == 0o600
    monkeypatch.setattr(inventory.time, "time", lambda: 1012.5)
    assert inventory.load(arguments(), directory) == (report(), 12.5)


def test_synthetic_arguments_use_stable_defaults(directory):
    args = SimpleNamespace(host="fixture", cache_root="/fixture/cache")
    inventory.save(args, directory, report())
    restored, age = inventory.load(args, directory)
    assert restored == report() and 0 <= age < 1


@pytest.mark.parametrize("changes", [
    {"host": "another"}, {"cache_root": "/another"}, {"abi": ABI},
    {"verify": True}, {"stale_after": 301}, {"sessions_root": ["/another"]},
    {"no_sessions": True},
])
def test_scope_never_borrows_another_inventory(directory, changes):
    inventory.save(arguments(), directory, report())
    assert inventory.load(arguments(**changes), directory) == (None, None)
    rewrite(cache_path(directory), lambda value: value["scope"].update(changes))
    assert inventory.load(arguments(), directory) == (None, None)


@pytest.mark.parametrize("age,accepted", [
    (0, True), (600, True), (600.01, False), (-1, True), (-1.01, False),
])
def test_expired_and_future_inventory_is_rejected(directory, monkeypatch, age, accepted):
    monkeypatch.setattr(inventory.time, "time", lambda: 1000)
    inventory.save(arguments(), directory, report())
    monkeypatch.setattr(inventory.time, "time", lambda: 1000 + age)
    restored, actual_age = inventory.load(arguments(), directory)
    assert (restored is not None) is accepted
    assert actual_age == (max(0, age) if accepted else None)


@pytest.mark.parametrize("change", [
    lambda value: value.update(schema="unknown"),
    lambda value: value["scope"].update(verify=0),
    lambda value: value.update(saved_at="yesterday"),
    lambda value: value.update(saved_at=True),
    lambda value: value.update(saved_at=float("nan")),
    lambda value: value.update(report={"error": "source unavailable"}),
    lambda value: value["report"].update(chats=[{}]),
    lambda value: value["report"].update(unsnapshotted_chats=[{}]),
    lambda value: value["report"].update(issues=[{}]),
    lambda value: value["report"]["io"].update(lifetime={}),
    lambda value: value["report"]["io"].update(deleted_written_file_bytes="unknown"),
    lambda value: value["report"]["totals"].update(file_bytes=-1),
    lambda value: value["report"]["filesystem"].update(available_bytes=True),
    lambda value: value["report"]["chats"][0]["metadata"].update(generation=[]),
    lambda value: value["report"]["chats"][0]["session"].update(title=[]),
    lambda value: value["report"]["chats"][0]["metadata"].update(tokens=[1, 2, 3]),
    lambda value: value["report"]["chats"][0].update(io={"written_file_bytes": "unknown"}),
])
def test_invalid_envelopes_and_report_shapes_are_misses(directory, change):
    inventory.save(arguments(), directory, report())
    rewrite(cache_path(directory), change)
    assert inventory.load(arguments(), directory) == (None, None)


@pytest.mark.parametrize("contents", [b"{", b"[]", b"\xff", b"null"])
def test_corrupt_cache_is_a_miss(directory, contents):
    inventory.save(arguments(), directory, report())
    cache_path(directory).write_bytes(contents)
    assert inventory.load(arguments(), directory) == (None, None)


@pytest.mark.parametrize("mode", [0o644, 0o660, 0o400, 0o000])
def test_cache_must_be_owner_private(directory, mode):
    inventory.save(arguments(), directory, report())
    cache_path(directory).chmod(mode)
    assert inventory.load(arguments(), directory) == (None, None)


def test_file_symlinks_and_hardlinks_are_not_read(directory, tmp_path):
    inventory.save(arguments(), directory, report())
    path = cache_path(directory)
    target = tmp_path / "outside.json"
    path.rename(target)
    path.symlink_to(target)
    assert inventory.load(arguments(), directory) == (None, None)
    path.unlink()
    os.link(target, path)
    assert inventory.load(arguments(), directory) == (None, None)


def test_fifo_is_rejected_without_waiting_for_a_writer(directory):
    inventory.save(arguments(), directory, report())
    path = cache_path(directory)
    path.unlink()
    os.mkfifo(path, mode=0o600)
    assert inventory.load(arguments(), directory) == (None, None)


def test_directory_symlinks_and_public_directories_are_rejected(directory, tmp_path):
    inventory.save(arguments(), directory, report())
    link = tmp_path / "linked"
    link.symlink_to(directory, target_is_directory=True)
    assert inventory.load(arguments(), link) == (None, None)
    with pytest.raises(OSError):
        inventory.save(arguments(), link, report())
    directory.chmod(0o755)
    assert inventory.load(arguments(), directory) == (None, None)
    with pytest.raises(ValueError, match="unsafe inventory directory"):
        inventory.save(arguments(), directory, report())


def test_wrong_owner_is_rejected(directory, monkeypatch):
    inventory.save(arguments(), directory, report())
    real_uid = os.getuid()
    monkeypatch.setattr(inventory.os, "getuid", lambda: real_uid + 1)
    assert inventory.load(arguments(), directory) == (None, None)
    with pytest.raises(ValueError, match="unsafe inventory directory"):
        inventory.save(arguments(), directory, report())


def test_oversized_cache_is_not_loaded_and_oversized_reports_preserve_old_cache(directory, monkeypatch):
    inventory.save(arguments(), directory, report())
    path = cache_path(directory)
    original = path.read_bytes()
    monkeypatch.setattr(inventory, "MAX_BYTES", len(original))
    large = report()
    large["chats"][0]["metadata"]["title"] = "x" * len(original)
    with pytest.raises(ValueError, match="size limit"):
        inventory.save(arguments(), directory, large)
    assert path.read_bytes() == original
    path.write_bytes(original + b" ")
    assert inventory.load(arguments(), directory) == (None, None)


@pytest.mark.parametrize("bad", [{"error": "SSH unavailable"}, None, {"chats": []}])
def test_source_errors_cannot_replace_a_completed_report(directory, bad):
    inventory.save(arguments(), directory, report())
    original = cache_path(directory).read_bytes()
    with pytest.raises(ValueError, match="invalid completed"):
        inventory.save(arguments(), directory, bad)
    assert cache_path(directory).read_bytes() == original


def test_failed_publication_preserves_old_report_and_removes_temporary(directory, monkeypatch):
    inventory.save(arguments(), directory, report())
    original = cache_path(directory).read_bytes()

    def fail(*_args, **_kwargs):
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(inventory.os, "replace", fail)
    changed = copy.deepcopy(report())
    changed["totals"]["file_bytes"] += 1
    with pytest.raises(OSError, match="publication failure"):
        inventory.save(arguments(), directory, changed)
    assert cache_path(directory).read_bytes() == original
    assert len(list(directory.iterdir())) == 1

"""CPU-only lifecycle tests. No real container, model request or GPU is used."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from qwen_r9700_lab import radiance_backend as backend


@pytest.fixture
def controller(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "SCHEDULER", tmp_path / "scheduler.json")
    monkeypatch.setattr(backend, "TAIL", tmp_path / "tail.json")
    config = {
        "container": "test-backend",
        "model": backend.MODEL,
        "port": 8080,
        "image": "a" * 64,
        "runtime_abi": "b" * 64,
        "data_abi": "c" * 64,
        "compatible_runtime_abis": [],
        "cache_root": str(tmp_path),
        "launcher": str(tmp_path / "launcher"),
        "files": {},
        "cache_module": str(tmp_path / "cache.py"),
    }
    result = backend.Controller(config)
    transfer = {
        "engine_id": "qwen-radiance-public-clean-" + "b" * 16,
        "kv_connector_extra_config": {
            "secondary_tiers": [
                {
                    "type": "qwen_chat_fs",
                    "root_dir": "/cache/snapshots/" + "c" * 64 + "/data",
                }
            ]
        },
    }
    result.row = {
        "Id": "container-identity",
        "Name": "test-backend",
        "Image": "a" * 64,
        "State": {"Running": True, "StartedAt": "2026-01-01T00:00:00Z"},
        "Args": [
            "--served-model-name",
            backend.MODEL,
            "--max-model-len",
            "253792",
            "--kv-cache-dtype",
            "fp8",
            "--port",
            "8080",
            "--speculative-config",
            "{}",
            "--kv-transfer-config",
            json.dumps(transfer),
        ],
    }
    monkeypatch.setattr(result, "inspect", lambda: result.row)
    monkeypatch.setattr(result, "ready", lambda: True)
    backend.SCHEDULER.write_text(json.dumps({"updated_at": 1800000000, "requests": []}))
    return result


def test_same_contract_as_launcher():
    root = Path(__file__).resolve().parents[1]
    config = backend.legacy_contract(root)
    assert (
        config["data_abi"]
        == "d1d796bfb20355f97eaa4d31911647bd707543dc11d79224bd38fc59a1256d2d"
    )
    assert config["files"][config["launcher"]]
    with pytest.raises(ValueError, match="not configured"):
        backend.legacy_contract(root, container="another-backend")
    with pytest.raises(ValueError, match="not configured"):
        backend.legacy_contract(root, abi="f" * 64)


def test_status_counts_without_exposing_request_metadata(controller):
    backend.SCHEDULER.write_text(
        json.dumps(
            {
                "updated_at": 1800000000,
                "requests": [
                    {"state": "running", "request_id": "private", "tokens": [123]},
                    {"state": "queued"},
                ],
            }
        )
    )
    value = controller.status()
    assert value["state"] == "generating"
    assert value["active_requests"] == 2
    assert value["queued_requests"] == 1
    assert "private" not in json.dumps(value)
    backend.SCHEDULER.write_text(json.dumps({"updated_at": 1, "requests": []}))
    assert controller.status()["state"] == "running", (
        "stale metadata must not claim idle"
    )


def test_stopped_status_ignores_stale_telemetry(controller, monkeypatch):
    monkeypatch.setattr(controller, "inspect", lambda: None)
    assert controller.status()["state"] == "stopped"


def test_existing_ready_start_and_absent_stop_are_idempotent(controller, monkeypatch):
    with patch.object(controller, "detach") as detach:
        value = controller.execute("start")
        assert value["state"] == "idle" and not value["busy"]
        monkeypatch.setattr(controller, "inspect", lambda: None)
        assert controller.execute("stop")["state"] == "stopped"
        detach.assert_not_called()


def test_shared_host_lock_refuses_opposite_operation_and_joins_same(controller):
    descriptor = controller.lock()
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        controller.record(
            id="d" * 32, action="start", status="pending", stage="Loading model"
        )
        assert controller.execute("start")["busy"]
        with pytest.raises(RuntimeError, match="another backend operation"):
            controller.execute("stop")
    finally:
        os.close(descriptor)
    assert controller.status()["operation"]["status"] == "failed"


@pytest.mark.parametrize("change", ["Image", "Name", "Args"])
def test_foreign_container_never_stopped(controller, change):
    controller.row[change] = [] if change == "Args" else "foreign"
    with patch.object(backend.subprocess, "run") as run:
        assert not controller.status()["pinned"]
        with pytest.raises(RuntimeError, match="differs"):
            controller.execute("stop")
        run.assert_not_called()


def test_release_check_rejects_modified_source_and_symlink(controller, tmp_path):
    file = tmp_path / "release.py"
    file.write_bytes(b"qualified")
    controller.config["files"] = {str(file): hashlib.sha256(b"qualified").hexdigest()}
    controller.verify_release()
    file.write_bytes(b"modified")
    with pytest.raises(ValueError, match="mismatch"):
        controller.verify_release()
    link = tmp_path / "link"
    link.symlink_to(file)
    controller.config["files"] = {str(link): hashlib.sha256(b"modified").hexdigest()}
    with pytest.raises(OSError):
        controller.verify_release()


def test_failed_flush_keeps_backend_running(controller):
    with (
        patch.object(controller, "flush_tails", side_effect=RuntimeError("disk full")),
        patch.object(backend.subprocess, "run") as run,
    ):
        descriptor = controller.lock()
        controller.record(id="d" * 32, action="stop", status="pending", stage="Pending")
        controller.worker("stop", descriptor)
        assert controller.status()["running"]
        assert controller.status()["operation"]["error"] == "disk full"
        run.assert_not_called()


def test_all_buffered_tails_flush_before_term(controller, tmp_path):
    helper = Path(controller.config["cache_module"])
    evidence = tmp_path / "flushed.txt"
    helper.write_text(
        "def request_tail_flush(chat, timeout):\n"
        f"    with open({str(evidence)!r}, 'a') as out: out.write(chat['id'] + '\\n')\n"
        "    return {'status': 'flushed', 'tokens': 123}\n"
    )
    controller.config["files"][str(helper)] = hashlib.sha256(
        helper.read_bytes()
    ).hexdigest()
    backend.TAIL.write_text(
        json.dumps(
            {
                "schema": "urn:qwen-r9700:radiance-tail-residency:v1",
                "updated_at": 1800000000,
                "chats": [
                    {"chat_id": "e" * 64, "generation": "f" * 64, "tokens": 123},
                    {"chat_id": "1" * 64, "generation": "2" * 64, "tokens": 123},
                ],
            }
        )
    )

    def signal(command, **_):
        assert evidence.read_text().splitlines() == ["e" * 64, "1" * 64]
        assert command == ["podman", "kill", "--signal", "TERM", "container-identity"]
        controller.row = None
        return SimpleNamespace(returncode=0)

    with patch.object(backend.subprocess, "run", side_effect=signal):
        controller.stop(backend.time.monotonic() + 1)
    assert controller.status()["state"] == "stopped"
    assert backend.bounded_json(controller.record_path)["flushed_chats"] == 2


def test_replacement_container_not_signalled(controller):
    def flush(_):
        controller.row = {**controller.row, "Id": "replacement"}
        return 0

    with (
        patch.object(controller, "flush_tails", side_effect=flush),
        patch.object(backend.subprocess, "run") as run,
    ):
        with pytest.raises(RuntimeError, match="replacement left running"):
            controller.stop(backend.time.monotonic() + 1)
        run.assert_not_called()


def test_missing_or_old_tail_status_prevents_stop(controller):
    with pytest.raises(RuntimeError, match="left running"):
        controller.flush_tails(controller.row)
    backend.TAIL.write_text(
        json.dumps(
            {
                "schema": "urn:qwen-r9700:radiance-tail-residency:v1",
                "updated_at": 1,
                "chats": [],
            }
        )
    )
    with pytest.raises(RuntimeError, match="older backend"):
        controller.flush_tails(controller.row)


def test_unowned_port_does_not_launch(controller, monkeypatch):
    monkeypatch.setattr(controller, "inspect", lambda: None)
    with (
        patch.object(controller, "verify_release"),
        patch.object(controller, "port_occupied", return_value=True),
        patch.object(backend.subprocess, "Popen") as launch,
    ):
        with pytest.raises(RuntimeError, match="occupied"):
            controller.start(backend.time.monotonic() + 1)
        launch.assert_not_called()


def test_start_uses_private_logs_with_existing_legacy_0755_directory(
    controller, tmp_path
):
    legacy = tmp_path / "logs"
    legacy.mkdir(mode=0o755)
    legacy.chmod(0o755)
    prior_log = legacy / "existing.log"
    prior_log.write_text("untouched legacy log")
    row = controller.row
    controller.row = None

    def spawn(command, **options):
        assert command == [controller.config["launcher"]]
        assert options["start_new_session"]
        assert options["stdout"] is options["stderr"]
        assert os.readlink(f"/proc/self/fd/{options['stdout'].fileno()}").startswith(
            str(controller.directory / "logs") + "/"
        )
        assert os.fstat(options["stdout"].fileno()).st_mode & 0o077 == 0
        controller.row = row
        return SimpleNamespace(poll=lambda: None)

    with (
        patch.object(controller, "verify_release"),
        patch.object(controller, "port_occupied", return_value=False),
        patch.object(backend.subprocess, "Popen", side_effect=spawn) as launch,
    ):
        controller.start(backend.time.monotonic() + 1)
    assert launch.call_count == 1
    assert (controller.directory / "logs").stat().st_mode & 0o077 == 0
    assert legacy.stat().st_mode & 0o777 == 0o755
    assert prior_log.read_text() == "untouched legacy log"
    assert backend.bounded_json(controller.record_path)["log_path"].startswith(
        str(controller.directory / "logs") + "/"
    )


def test_private_directory_error_identifies_rejected_path(tmp_path):
    directory = tmp_path / "unsafe"
    directory.mkdir(mode=0o755)
    directory.chmod(0o755)
    with pytest.raises(ValueError, match=str(directory)):
        backend.private_directory(directory)


def test_podman_failure_is_not_reported_as_stopped(controller):
    with (
        patch.object(
            backend.subprocess, "run", return_value=SimpleNamespace(returncode=125)
        ),
        pytest.raises(RuntimeError, match="cannot inspect"),
    ):
        backend.Controller.inspect(controller)


def test_graceful_stop_timeout_never_force_kills(controller):
    with (
        patch.object(controller, "flush_tails", return_value=0),
        patch.object(
            backend.subprocess, "run", return_value=SimpleNamespace(returncode=0)
        ) as signal,
    ):
        with pytest.raises(TimeoutError, match="no force-kill"):
            controller.stop(backend.time.monotonic() - 1)
        assert signal.call_count == 1
        assert signal.call_args.args[0][3] == "TERM"


def test_detached_worker_returns_promptly_keeps_host_lock_and_finishes(
    controller, tmp_path, monkeypatch
):
    started, release = tmp_path / "started", tmp_path / "release"
    row = controller.row
    monkeypatch.setattr(
        controller, "inspect", lambda: row if release.exists() else None
    )

    def start(_):
        started.touch()
        deadline = backend.time.monotonic() + 3
        while not release.exists():
            if backend.time.monotonic() > deadline:
                raise TimeoutError("synthetic operation timeout")
            backend.time.sleep(0.01)

    monkeypatch.setattr(controller, "start", start)
    try:
        before = backend.time.monotonic()
        result = controller.execute("start")
        assert backend.time.monotonic() - before < 1
        assert result["state"] == "starting" and result["busy"]
        assert controller.busy(), "the detached worker retains the shared lock"
        with pytest.raises(RuntimeError, match="another backend operation"):
            controller.execute("stop")
    finally:
        release.touch()
    deadline = backend.time.monotonic() + 3
    while controller.busy() and backend.time.monotonic() < deadline:
        backend.time.sleep(0.01)
    assert not controller.busy()
    assert controller.status()["ready"]
    assert controller.status()["operation"]["status"] == "complete"

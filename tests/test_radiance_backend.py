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
    monkeypatch.setattr(backend.time, "time", lambda: 1800000000.0)
    monkeypatch.setattr(backend, "SCHEDULER", tmp_path / "scheduler.json")
    monkeypatch.setattr(backend, "TAIL", tmp_path / "tail.json")
    monkeypatch.setattr(backend, "DRM", tmp_path / "drm")
    monkeypatch.setattr(backend, "MEMINFO", tmp_path / "meminfo")
    config = {
        "container": "test-backend",
        "model": backend.MODEL,
        "max_model_len": 253792,
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
    assert config["files"][config["launcher"]]
    manifest_path = root / "experiments/radiance-public/snapshot-abi-chat-cache-v1.json"
    manifest = json.loads(manifest_path.read_text())
    assert config["data_abi"] == manifest["storage"]["data_abi"]
    assert config["max_model_len"] == manifest["serving"]["max_model_len"]
    with pytest.raises(ValueError, match="not configured"):
        backend.legacy_contract(root, container="another-backend")
    with pytest.raises(ValueError, match="not configured"):
        backend.legacy_contract(root, abi="f" * 64)


def test_context_boundary_must_match_pinned_command_without_override(controller):
    assert controller.matches(controller.row)
    controller.config["max_model_len"] = 319000
    assert not controller.matches(controller.row)
    args = controller.row["Args"]
    args[args.index("--max-model-len") + 1] = "319000"
    assert controller.matches(controller.row)
    args.extend(["--max-model-len", "253792"])
    assert not controller.matches(controller.row)


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


def host_resource_fixture(tmp_path):
    device = tmp_path / "drm/card1/device"
    hwmon = device / "hwmon/hwmon3"
    hwmon.mkdir(parents=True)
    attributes = {
        device / "vendor": "0x1002",
        device / "gpu_busy_percent": "73",
        device / "mem_info_vram_used": str(30 * 1024**3),
        device / "mem_info_vram_total": str(32 * 1024**3),
        hwmon / "temp1_label": "edge",
        hwmon / "temp1_input": "55000",
        hwmon / "temp2_label": "junction",
        hwmon / "temp2_input": "88000",
        hwmon / "pwm1": "128",
        hwmon / "pwm1_max": "255",
        tmp_path
        / "meminfo": "MemTotal:       67108864 kB\nMemAvailable:   33554432 kB\n",
    }
    for path, text in attributes.items():
        path.write_text(text + "\n")
    return device


def test_stopped_backend_still_reports_current_host_resources(
    controller, tmp_path, monkeypatch
):
    host_resource_fixture(tmp_path)
    monkeypatch.setattr(controller, "inspect", lambda: None)
    value = controller.status()
    assert value["state"] == "stopped" and value["running"] is False
    assert value["host_control"] == {"reachable": True, "observed_at": 1800000000.0}
    assert value["release"]["observed"] is None
    assert value["release"]["expected"]["model"] == backend.MODEL
    assert value["uptime_seconds"] is None
    assert value["telemetry"]["scheduler"]["status"] == "backend_stopped"
    assert "active_requests" not in value
    assert value["resources"]["source"] == "host_sysfs_proc"
    assert value["resources"]["gpu"] == {
        "edge_millicelsius": 55000,
        "junction_millicelsius": 88000,
        "fan_percent": 50,
        "utilization_percent": 73,
        "vram_used_bytes": 30 * 1024**3,
        "vram_total_bytes": 32 * 1024**3,
    }
    assert value["resources"]["ram"] == {
        "total_bytes": 64 * 1024**3,
        "available_bytes": 32 * 1024**3,
        "used_bytes": 32 * 1024**3,
    }


def test_inspection_fault_preserves_reachability_and_does_not_claim_stopped(
    controller, tmp_path, monkeypatch
):
    host_resource_fixture(tmp_path)

    def failed():
        raise RuntimeError("do not expose arbitrary runtime exception text")

    monkeypatch.setattr(controller, "inspect", failed)
    value = controller.status()
    assert value["state"] == "unavailable"
    assert value["running"] is value["ready"] is value["pinned"] is None
    assert value["host_control"]["reachable"] is True
    assert value["resources"]["gpu"]["utilization_percent"] == 73
    assert value["issues"] == ["container_inspection_failed"]
    assert "arbitrary" not in json.dumps(value)
    assert "active_requests" not in value
    with pytest.raises(RuntimeError, match="differs"):
        controller.execute("stop")


@pytest.mark.parametrize("offset", [-11, 6])
def test_old_or_future_scheduler_is_not_current(controller, offset):
    backend.SCHEDULER.write_text(
        json.dumps(
            {
                "updated_at": backend.time.time() + offset,
                "requests": [{"state": "running"}],
            }
        )
    )
    value = controller.status()
    assert value["state"] == "running"
    assert value["telemetry"]["scheduler"]["status"] == "stale"
    assert "active_requests" not in value


def test_previous_engine_scheduler_and_missing_metadata_are_unavailable(controller):
    for value in (
        {
            "updated_at": controller.started_at(controller.row) - 1,
            "requests": [{"state": "running"}],
        },
        {"updated_at": float("nan"), "requests": []},
        {"updated_at": backend.time.time(), "requests": [None]},
    ):
        backend.SCHEDULER.write_text(json.dumps(value))
        report = controller.status()
        assert report["telemetry"]["scheduler"]["status"] == "unavailable"
        assert "active_requests" not in report
    backend.SCHEDULER.write_text("not json")
    assert controller.status()["telemetry"]["scheduler"]["status"] == "unavailable"


def test_fresh_same_engine_tail_can_confirm_idle_scheduler(controller):
    backend.SCHEDULER.write_text(
        json.dumps(
            {"updated_at": backend.time.time() - 60, "pid": 1234, "requests": []}
        )
    )
    backend.TAIL.write_text(
        json.dumps({"updated_at": backend.time.time() - 1, "pid": 1234})
    )
    value = controller.status()
    assert value["state"] == "idle"
    assert value["telemetry"]["scheduler"]["age_seconds"] == 1
    backend.TAIL.write_text(
        json.dumps({"updated_at": backend.time.time() - 1, "pid": 4321})
    )
    assert controller.status()["state"] == "running"


def test_scheduler_observation_uses_one_atomic_snapshot_and_exact_states(
    controller, monkeypatch
):
    backend.SCHEDULER.write_text(
        json.dumps(
            {
                "updated_at": backend.time.time() - 1,
                "requests": [
                    {"state": "running"},
                    {"state": "paused"},
                    {"state": "queued"},
                ],
            }
        )
    )
    original = backend.bounded_json
    scheduler_reads = []

    def read(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        if path == backend.SCHEDULER:
            scheduler_reads.append(path)
            backend.SCHEDULER.write_text(
                json.dumps({"updated_at": backend.time.time(), "requests": []})
            )
        return value

    monkeypatch.setattr(backend, "bounded_json", read)
    report = controller.status()
    sample = report["telemetry"]["scheduler"]
    assert len(scheduler_reads) == 1
    assert sample["age_seconds"] == 1
    assert sample["active_requests"] == 3
    assert (
        sample["queued_requests"]
        == sample["generating_requests"]
        == sample["paused_requests"]
        == 1
    )


def test_paused_requests_are_not_labelled_generating(controller):
    backend.SCHEDULER.write_text(
        json.dumps(
            {"updated_at": backend.time.time(), "requests": [{"state": "paused"}]}
        )
    )
    report = controller.status()
    assert report["state"] == "running"
    assert report["telemetry"]["scheduler"]["generating_requests"] == 0


def test_readiness_probe_fault_keeps_confirmed_container_state(controller, monkeypatch):
    def failed():
        raise OSError("not a container failure")

    monkeypatch.setattr(controller, "ready", failed)
    report = controller.status()
    assert report["state"] == "unavailable"
    assert report["running"] is report["pinned"] is True
    assert report["ready"] is None
    assert report["issues"] == ["backend_readiness_probe_failed"]


def test_unknown_start_time_does_not_invent_uptime(controller):
    controller.row["State"]["StartedAt"] = "missing"
    report = controller.status()
    assert report["uptime_seconds"] is None
    assert "backend_start_time_unavailable" in report["issues"]


def test_release_distinguishes_container_configuration_from_verified_files(controller):
    controller.row["Config"] = {
        "Env": [
            "TRITON_CACHE_DIR=/cache/runtime/" + "b" * 64 + "/triton",
            "PRIVATE=never expose arbitrary container environment",
        ]
    }
    value = controller.status()
    assert value["release"]["expected"]["runtime_abi"] == "b" * 64
    assert value["release"]["observed"]["runtime_abi"] == "b" * 64
    assert value["release"]["observed"]["runtime_abi_prefix"] == "b" * 16
    assert value["release"]["verification"] == "container_configuration"
    assert value["release"]["matches"] is True
    assert "PRIVATE" not in json.dumps(value)
    controller.row["Image"] = "d" * 64
    report = controller.status()
    assert report["release"]["matches"] is False
    assert report["release"]["observed"]["image"] == "d" * 64
    assert report["state"] == "unavailable" and not report["ready"]


def test_release_does_not_expand_a_runtime_prefix_into_an_invented_hash(controller):
    value = controller.status()
    observed = value["release"]["observed"]
    assert observed["runtime_abi"] is None
    assert observed["runtime_abi_prefix"] == "b" * 16
    assert observed["data_abi"] == "c" * 64
    assert observed["model"] == backend.MODEL


def test_host_uptime_and_operation_ages_are_source_relative(controller, monkeypatch):
    started = controller.started_at(controller.row)
    controller.record(id="1" * 32, action="start", status="pending")
    monkeypatch.setattr(backend.time, "time", lambda: 1800000020.0)
    controller.record(status="complete")
    monkeypatch.setattr(backend.time, "time", lambda: 1800000030.0)
    value = controller.status()
    assert value["started_at"] == started
    assert value["uptime_seconds"] == 1800000030 - started
    assert value["operation_age_seconds"] == 10
    assert value["operation_elapsed_seconds"] == 20
    assert value["operation"]["finished_at"] == 1800000020


def test_missing_or_bad_resource_attributes_are_optional(controller, tmp_path):
    value = controller.status()
    assert value["resources"]["gpu"] is None
    assert value["resources"]["ram"] is None
    device = host_resource_fixture(tmp_path)
    (device / "gpu_busy_percent").write_text("101")
    (device / "mem_info_vram_used").write_text(str(33 * 1024**3))
    (device / "hwmon/hwmon3/temp2_input").write_text("9" * 100)
    backend.MEMINFO.write_text("MemTotal: 100 kB\nMemAvailable: 101 kB\n")
    report = controller.status()
    assert report["state"] == "idle"
    assert report["resources"]["gpu"]["edge_millicelsius"] == 55000
    for field in (
        "utilization_percent",
        "vram_used_bytes",
        "vram_total_bytes",
        "junction_millicelsius",
    ):
        assert report["resources"]["gpu"][field] is None
    assert report["resources"]["ram"] is None


@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_remote_control_failure_is_not_backend_stopped(
    controller, monkeypatch, capsys, failure
):
    monkeypatch.setattr(
        backend, "legacy_contract", lambda *args, **kwargs: controller.config
    )
    with patch.object(
        backend.subprocess,
        "run",
        return_value=SimpleNamespace(
            returncode=255, stdout="", stderr="private remote failure"
        ),
        side_effect=backend.subprocess.TimeoutExpired("private command", 15)
        if failure == "timeout"
        else None,
    ):
        assert backend.main(["status", "--host", "test-host"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["state"] == "unavailable"
    assert value["host_control"]["reachable"] is False
    assert "private" not in json.dumps(value)


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

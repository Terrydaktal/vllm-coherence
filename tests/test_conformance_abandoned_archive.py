"""Interrupted evidence may be relocated without pretending its work completed."""

import fcntl
import hashlib
import importlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from qwen_r9700_lab.diagnostic_contract import seal


@pytest.fixture
def modules(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "experiments/radiance-public")
    )
    archive, abandoned, retire = (
        importlib.import_module(name)
        for name in (
            "archive_conformance_run",
            "archive_abandoned_run",
            "retire_conformance_replica",
        )
    )
    from archive_inactive_benchmark_cache import live_references

    proc = tmp_path / "proc"
    proc.mkdir()
    monkeypatch.setattr(abandoned, "live_references", lambda name: live_references(name, proc))
    return archive, abandoned, retire, proc


@pytest.fixture
def abandoned_root(tmp_path, modules):
    _, helper, _, _ = modules
    root = tmp_path / "conformance/20260915/run-777-cpu"
    root.mkdir(parents=True)
    (root / "queue.lock").touch()
    campaign = seal({"schema": "fixture", "cases": ["failed", "unfinished"]})
    checkpoint = seal(
        {
            "campaign": campaign["sha256"],
            "status": "running_case",
            "current": "unfinished",
            "attempt": "case-00002",
            "counts": {"FAILED": 1, "NOT_RUN": 1},
        }
    )
    (root / "campaign.json").write_text(json.dumps(campaign))
    (root / "checkpoint.json").write_text(json.dumps(checkpoint))
    blobs = []
    for n, directory in ((1, "capture"), (2, "reference")):
        case = root / f"case-{n:05d}"
        frame = case / directory / "frame-000000"
        frame.mkdir(parents=True)
        (case / "input.json").write_text("{}")
        (frame / "frame.json").write_text('{"retain": true}')
        (frame / "unrelated.bin").write_bytes(b"retain arbitrary binary")
        data = bytes([n]) * 128
        path = frame / (hashlib.sha256(data).hexdigest() + ".bin")
        path.write_bytes(data)
        blobs.append(path)
    (root / "case-00001/result.json").write_text(json.dumps(seal({"status": "FAILED"})))
    original = helper.inventory(root)
    info = root.stat()
    review = seal(
        {
            "schema": helper.SCHEMA,
            "root": str(root),
            "root_signature": {"device": info.st_dev, "inode": info.st_ino, "owner": info.st_uid},
            "original": original,
            "known_live_references": [],
            "preserve_incomplete_status": True,
        }
    )
    (root / "archive-abandoned-admission.json").write_text(json.dumps(review))
    return root, blobs


def bundle(root, archive, tmp_path):
    raw = io.BytesIO()
    receipt, manifest, report, completion = (
        tmp_path / name for name in ("stream.json", "members.gz", "readback.json", "completed.json")
    )
    archive.stream(root, receipt, raw, abandoned=True)
    archive.verify(io.BytesIO(raw.getvalue()), receipt, manifest, report)
    completion.write_text(
        json.dumps(
            {
                "snapshot_id": "9" * 64,
                "readback_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            }
        )
    )
    return (root, report, manifest, completion, "workstation:/archive"), raw.getvalue()


def test_real_restore_preserves_failed_and_incomplete_status(modules, abandoned_root, tmp_path):
    archive, helper, retire, _ = modules
    root, blobs = abandoned_root
    before = helper.inventory(root)
    args, raw = bundle(root, archive, tmp_path)
    retired = retire.retire(*args)
    assert retired["removed_files"] == 2 and retired["removed_bytes"] == 256
    assert helper.inventory(root) == before
    assert json.loads((root / "checkpoint.json").read_text())["status"] == "running_case"
    assert not (root / "case-00002/result.json").exists()
    assert (root / "case-00001/capture/frame-000000/frame.json").exists()
    assert (root / "case-00002/reference/frame-000000/unrelated.bin").exists()
    assert retire.retire(*args) == retired
    with tarfile.open(fileobj=io.BytesIO(raw)) as saved:
        for n, blob in enumerate(blobs, 1):
            assert not blob.exists()
            restored = saved.extractfile(root.name + "/" + str(blob.relative_to(root))).read()
            assert restored == bytes([n]) * 128


def test_an_actual_controller_lock_blocks_archive(modules, abandoned_root, tmp_path):
    archive, _, _, _ = modules
    root, _ = abandoned_root
    with (root / "queue.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            bundle(root, archive, tmp_path)


@pytest.mark.parametrize(
    "fault",
    [
        "checkpoint",
        "new_result",
        "gpu_lease",
        "payload",
        "parent_symlink",
        "reactivated",
        "cwd_reactivated",
    ],
)
def test_source_changes_or_reactivation_block_retirement(modules, abandoned_root, tmp_path, fault):
    archive, _, retire, proc = modules
    root, blobs = abandoned_root
    args, _ = bundle(root, archive, tmp_path)
    child = None
    try:
        if fault == "checkpoint":
            path = root / "checkpoint.json"
            value = json.loads(path.read_text())
            value.pop("sha256")
            value["current"] = "resumed"
            path.write_text(json.dumps(seal(value)))
        elif fault == "new_result":
            (root / "case-00002/result.json").write_text(json.dumps(seal({"status": "FAILED"})))
        elif fault == "gpu_lease":
            (root / "case-00002/gpu-lease").mkdir()
        elif fault == "payload":
            blobs[0].write_bytes(b"changed payload")
        elif fault == "parent_symlink":
            parent = blobs[0].parent
            other = parent.with_name("moved")
            parent.rename(other)
            parent.symlink_to(other, target_is_directory=True)
        else:
            command = [sys.executable, "-c", "import time; time.sleep(30)"]
            if fault == "reactivated":
                command.append(root.name)
            child = subprocess.Popen(command, cwd=root if fault == "cwd_reactivated" else None)
            (proc / str(child.pid)).symlink_to(Path("/proc") / str(child.pid))
        with pytest.raises((ValueError, FileNotFoundError)):
            retire.retire(*args)
        assert blobs[0].exists()
    finally:
        if child:
            child.terminate()
            child.wait(timeout=5)


def test_completed_sources_do_not_use_abandoned_admission(modules, abandoned_root, tmp_path):
    archive, _, _, _ = modules
    root, _ = abandoned_root
    path = root / "checkpoint.json"
    value = json.loads(path.read_text())
    value.pop("sha256")
    value.update(status="stage_finished", current=None)
    path.write_text(json.dumps(seal(value)))
    with pytest.raises(ValueError, match="abandoned incomplete"):
        bundle(root, archive, tmp_path)


@pytest.mark.parametrize("mode", ["case", "paused", "diagnostic"])
def test_admission_modes_cannot_be_combined(modules, abandoned_root, tmp_path, mode):
    archive, _, _, _ = modules
    root, _ = abandoned_root
    with pytest.raises(ValueError, match="mutually exclusive"):
        archive.stream(root, tmp_path / "stream.json", io.BytesIO(), abandoned=True, **{mode: True})


@pytest.mark.parametrize(
    "path",
    [
        "run-777-cpu/case-00001/reference/frame-000000/unknown.bin",
        "run-777-cpu/case-00001/runtime/" + "a" * 64 + ".bin",
        "run-777-cpu/other/reference/" + "a" * 64 + ".bin",
        "run-777-cpu/case-00001/input.json",
    ],
)
def test_retirement_keeps_compilers_metadata_and_unrelated_files(modules, path):
    assert modules[1].abandoned_payload_path(path, "run-777-cpu") is None

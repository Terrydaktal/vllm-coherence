"""Completed diagnostic tensors remain recoverable before replica retirement."""

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
def diagnostic_proc(tmp_path):
    path = tmp_path / "owned-proc"
    path.mkdir()
    return path


@pytest.fixture
def diagnostic_modules(monkeypatch, diagnostic_proc):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "experiments/radiance-public")
    )
    modules = tuple(
        importlib.import_module(name)
        for name in (
            "archive_conformance_run",
            "archive_completed_diagnostic",
            "retire_conformance_replica",
        )
    )
    from archive_inactive_benchmark_cache import live_references

    # Real child /proc entries are added by the reactivation test. Unrelated
    # nondumpable desktop processes must not make this isolated fixture unusable.
    monkeypatch.setattr(
        modules[1], "live_references", lambda name: live_references(name, proc=diagnostic_proc)
    )
    return modules


@pytest.fixture(params=["diagnostic", "admission"])
def diagnostic_root(tmp_path, diagnostic_modules, request):
    _, helper, _ = diagnostic_modules
    if request.param == "diagnostic":
        root = tmp_path / "diagnostics/test-isolated-001"
        anchor = root / "result.json"
    else:
        root = tmp_path / "preflight/reference-reuse-native-admission-777/run"
        anchor = root.parent / "driver-result.json"
    root.mkdir(parents=True)
    anchor.write_text(json.dumps(seal({"returncode": 1, "status": "DISCREPANCY"})))
    payload = bytes(range(128)) * 10
    frame = root / "serial/run/capture/frame-000001"
    frame.mkdir(parents=True)
    (frame / (hashlib.sha256(payload).hexdigest() + ".bin")).write_bytes(payload)
    (frame / "frame.json").write_text('{"evidence": "keep"}')
    (frame / "candidate.so").write_bytes(b"keep compiled diagnostic")
    (root / "unrelated.bin").write_bytes(b"keep non-capture payload")
    info = root.stat()
    review = seal(
        {
            "schema": helper.SCHEMA,
            "root": str(root),
            "root_signature": {"device": info.st_dev, "inode": info.st_ino, "owner": info.st_uid},
            "known_live_references": [],
            "gpu_releases": {},
            "receipts": {anchor.name: hashlib.sha256(anchor.read_bytes()).hexdigest()},
        }
    )
    (root / "archive-admission.json").write_text(json.dumps(review))
    return root, anchor, frame, payload


def archive_bundle(root, archive, tmp_path):
    stream = io.BytesIO()
    receipt, manifest, report, completion = (
        tmp_path / n for n in ("stream.json", "members.gz", "readback.json", "complete.json")
    )
    archive.stream(root, receipt, stream, diagnostic=True)
    archive.verify(io.BytesIO(stream.getvalue()), receipt, manifest, report)
    completion.write_text(
        json.dumps(
            {
                "snapshot_id": "9" * 64,
                "readback_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            }
        )
    )
    return (root, report, manifest, completion, "workstation:/verified-repo"), stream.getvalue()


def test_diagnostic_archive_restore_and_restricted_retirement(
    diagnostic_modules, diagnostic_root, tmp_path
):
    archive, _, retire = diagnostic_modules
    root, anchor, frame, payload = diagnostic_root
    args, raw = archive_bundle(root, archive, tmp_path)
    result = retire.retire(*args)
    assert result["removed_files"] == 1 and result["removed_bytes"] == len(payload)
    assert not list(frame.glob("*.bin"))
    assert anchor.is_file() and (frame / "frame.json").is_file()
    assert (frame / "candidate.so").is_file() and (root / "unrelated.bin").is_file()
    assert retire.retire(*args) == result
    with tarfile.open(fileobj=io.BytesIO(raw)) as bundle:
        name = (
            f"{root.name}/serial/run/capture/frame-000001/{hashlib.sha256(payload).hexdigest()}.bin"
        )
        restored = bundle.extractfile(name).read()
        assert restored == payload
        target = tmp_path / "restored.bin"
        target.write_bytes(restored)
        assert target.read_bytes() == payload


@pytest.mark.parametrize(
    "fault",
    [
        "receipt",
        "payload",
        "symlink",
        "new_lease",
        "new_live_worker",
        "bad_review",
        "bad_completion",
    ],
)
def test_diagnostic_retirement_refuses_reactivation_or_changed_evidence(
    diagnostic_modules, diagnostic_root, diagnostic_proc, tmp_path, fault
):
    archive, helper, retire = diagnostic_modules
    root, anchor, frame, _payload = diagnostic_root
    args, _ = archive_bundle(root, archive, tmp_path)
    blob = next(frame.glob("*.bin"))
    child = None
    try:
        if fault == "receipt":
            anchor.write_text(json.dumps(seal({"returncode": 0})))
        elif fault == "payload":
            blob.write_bytes(b"changed")
        elif fault == "symlink":
            moved = frame.with_name("replaced")
            frame.rename(moved)
            frame.symlink_to(moved)
        elif fault == "new_lease":
            (root / "gpu-lease").mkdir()
        elif fault == "new_live_worker":
            _, label, _ = helper.domain(root)
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", label])
            (diagnostic_proc / str(child.pid)).symlink_to(Path("/proc") / str(child.pid))
        elif fault == "bad_review":
            (root / "archive-admission.json").write_text("{}")
        else:
            args[3].write_text("{}")
        with pytest.raises((ValueError, FileNotFoundError)):
            retire.retire(*args)
        assert blob.exists()
        assert not (root / "archive-retirement.json").exists()
    finally:
        if child is not None:
            child.terminate()
            child.wait(timeout=5)


@pytest.mark.parametrize(
    "fault", ["unreviewed", "not_terminal", "foreign_inode", "wrong_namespace"]
)
def test_diagnostic_stream_rejects_unadmitted_source(
    diagnostic_modules, diagnostic_root, tmp_path, fault
):
    archive, _, _ = diagnostic_modules
    root, anchor, _, _ = diagnostic_root
    p = root / "archive-admission.json"
    review = json.loads(p.read_text())
    review.pop("sha256")
    if fault == "unreviewed":
        p.unlink()
    elif fault == "not_terminal":
        anchor.write_text(json.dumps(seal({"status": "RUNNING"})))
        review["receipts"][anchor.name] = hashlib.sha256(anchor.read_bytes()).hexdigest()
        p.write_text(json.dumps(seal(review)))
    elif fault == "foreign_inode":
        review["root_signature"]["inode"] += 1
        p.write_text(json.dumps(seal(review)))
    else:
        other = tmp_path / "production"
        root.rename(other)
        root = other
    with pytest.raises((ValueError, FileNotFoundError)):
        archive.stream(root, tmp_path / "absent.json", io.BytesIO(), diagnostic=True)
    assert not (tmp_path / "absent.json").exists()


def test_diagnostic_payload_selector_excludes_non_tensors_and_path_escapes(diagnostic_modules):
    _, helper, _ = diagnostic_modules
    name = "0" * 64 + ".bin"
    assert helper.diagnostic_payload_path(f"diag/reference/frame-000001/{name}", "diag")
    assert helper.diagnostic_payload_path(f"diag/frame-000001/{name}", "diag")
    assert helper.diagnostic_payload_path(f"diag/semantic/p000000001-l000-input/{name}", "diag")
    for path in (
        f"diag/{name}",
        "diag/capture/frame.json",
        "diag/capture/weights.pt",
        "diag/capture/not-a-digest.bin",
    ):
        assert helper.diagnostic_payload_path(path, "diag") is None
    for path in (f"diag/../capture/{name}", f"/diag/capture/{name}", f"other/capture/{name}"):
        with pytest.raises(ValueError):
            helper.diagnostic_payload_path(path, "diag")


def test_diagnostic_retirement_receipt_corruption_is_rejected(
    diagnostic_modules, diagnostic_root, tmp_path
):
    archive, _, retire = diagnostic_modules
    root, _, _, _ = diagnostic_root
    args, _ = archive_bundle(root, archive, tmp_path)
    retire.retire(*args)
    path = root / "archive-retirement.json"
    value = json.loads(path.read_text())
    value["removed_bytes"] += 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        retire.retire(*args)


@pytest.mark.parametrize(
    "arms,valid",
    [
        ([{"arm": "serial", "returncode": 0}, {"arm": "d7", "returncode": 1}], True),
        ([], False),
        ([{"arm": "serial", "returncode": None}], False),
        ([{"arm": "serial", "returncode": True}], False),
    ],
)
def test_completed_arm_receipts_require_all_process_results(
    diagnostic_modules, diagnostic_root, arms, valid
):
    _, helper, _ = diagnostic_modules
    root, anchor, _, _ = diagnostic_root
    anchor.write_text(json.dumps(seal({"complete": True, "arms": arms})))
    path = root / "archive-admission.json"
    review = json.loads(path.read_text())
    review.pop("sha256")
    review["receipts"][anchor.name] = hashlib.sha256(anchor.read_bytes()).hexdigest()
    path.write_text(json.dumps(seal(review)))
    if valid:
        assert helper.finished_diagnostic_identity(root)["scope"] == helper.SCHEMA
    else:
        with pytest.raises(ValueError, match="terminal"):
            helper.finished_diagnostic_identity(root)

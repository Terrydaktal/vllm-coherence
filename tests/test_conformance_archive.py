"""Evidence must survive verified archival before a replica can be retired."""

import fcntl
import gzip
import hashlib
import importlib.util
import io
import json
import shutil
from pathlib import Path

import pytest

from qwen_r9700_lab.diagnostic_contract import seal


@pytest.fixture
def archive():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/radiance-public/archive_conformance_run.py"
    )
    spec = importlib.util.spec_from_file_location("archive_conformance_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def completed(tmp_path):
    root = tmp_path / "run-test"
    root.mkdir()
    (root / "queue.lock").touch()
    campaign = seal({"cases": ["one"]})
    (root / "campaign.json").write_text(json.dumps(campaign))
    (root / "checkpoint.json").write_text(
        json.dumps(
            seal(
                {
                    "campaign": campaign["sha256"],
                    "status": "stage_finished",
                    "current": None,
                }
            )
        )
    )
    case = root / "case-00001"
    case.mkdir()
    (case / "input.json").write_text("{}")
    (case / "result.json").write_text(
        json.dumps(
            seal(
                {
                    "campaign": campaign["sha256"],
                    "status": "FAILED",
                }
            )
        )
    )
    (case / "state.bin").write_bytes(bytes(range(256)) * 100)
    (case / "state-link").symlink_to("state.bin")
    return root


def test_real_tar_roundtrip_preserves_failed_evidence_and_links(archive, completed, tmp_path):
    stream = io.BytesIO()
    receipt = tmp_path / "source.json"
    manifest = tmp_path / "members.gz"
    report = tmp_path / "verified.json"
    archive.stream(completed, receipt, stream)
    archive.verify(io.BytesIO(stream.getvalue()), receipt, manifest, report)
    result = json.loads(report.read_text())
    with gzip.open(manifest, "rt") as file:
        records = {row["path"]: row for row in map(json.loads, file)}
    binary = records["run-test/case-00001/state.bin"]
    assert binary["sha256"] == hashlib.sha256(bytes(range(256)) * 100).hexdigest()
    assert records["run-test/case-00001/state-link"]["linkname"] == "state.bin"
    assert result["verified"] is True
    assert (completed / "case-00001/state.bin").is_file()
    assert result["stream_receipt"]["tar_bytes"] == len(stream.getvalue())


@pytest.mark.parametrize("fault", ["incomplete", "active_status", "bad_seal", "wrong_campaign"])
def test_incomplete_or_untrusted_run_cannot_be_archived(archive, completed, tmp_path, fault):
    if fault == "incomplete":
        (completed / "case-00001/result.json").unlink()
    else:
        path = completed / (
            "case-00001/result.json" if fault == "wrong_campaign" else "checkpoint.json"
        )
        doc = json.loads(path.read_text())
        doc.pop("sha256")
        if fault == "active_status":
            doc["status"] = "running_case"
        elif fault == "wrong_campaign":
            doc["campaign"] = "0" * 64
        doc = seal(doc)
        if fault == "bad_seal":
            doc["sha256"] = "0" * 64
        path.write_text(json.dumps(doc))
    receipt = tmp_path / "source.json"
    with pytest.raises((ValueError, FileNotFoundError)):
        archive.stream(completed, receipt, io.BytesIO())
    assert not receipt.exists()


def test_real_controller_lock_prevents_archive(archive, completed, tmp_path):
    with (completed / "queue.lock").open("rb") as controller:
        fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            archive.stream(completed, tmp_path / "source.json", io.BytesIO())


@pytest.mark.parametrize("fault", ["changed_payload", "truncated_padding", "wrong_expected_root"])
def test_corrupt_restore_never_publishes_success(archive, completed, tmp_path, fault):
    output = io.BytesIO()
    receipt = tmp_path / "source.json"
    archive.stream(completed, receipt, output)
    data = output.getvalue()
    if fault == "changed_payload":
        needle = bytes(range(256))
        index = data.index(needle)
        data = data[:index] + b"x" + data[index + 1 :]
    elif fault == "truncated_padding":
        data = data[:-1]
    else:
        value = json.loads(receipt.read_text())
        value["root_name"] = "another-run"
        receipt.write_text(json.dumps(value))
    report = tmp_path / "verified.json"
    with pytest.raises(ValueError):
        archive.verify(io.BytesIO(data), receipt, tmp_path / "members.gz", report)
    assert not report.exists()


def test_broken_destination_preserves_source_and_no_receipt(archive, completed, tmp_path):
    class BrokenDestination:
        def write(self, _data):
            raise BrokenPipeError

    receipt = tmp_path / "source.json"
    with pytest.raises(BrokenPipeError):
        archive.stream(completed, receipt, BrokenDestination())
    assert not receipt.exists()
    assert (completed / "case-00001/state.bin").stat().st_size == 25600


@pytest.fixture
def retirement(archive, completed, tmp_path, monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "experiments/radiance-public"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "retire_conformance_replica", directory / "retire_conformance_replica.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    capture = completed / "case-00001/capture"
    capture.mkdir()
    (capture / "a.bin").write_bytes(b"first original payload")
    (capture / "b.bin").write_bytes(b"second original payload")
    output = io.BytesIO()
    receipt, manifest, report, completion = (
        tmp_path / n
        for n in (
            "stream.json",
            "members.gz",
            "verified.json",
            "completed.json",
        )
    )
    archive.stream(completed, receipt, output)
    archive.verify(io.BytesIO(output.getvalue()), receipt, manifest, report)
    completion.write_text(
        json.dumps(
            {
                "snapshot_id": "a" * 64,
                "readback_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            }
        )
    )
    return module, (completed, report, manifest, completion, "workstation:/private/archive")


def test_retirement_keeps_results_and_unarchived_high_level_files(retirement):
    module, args = retirement
    result = module.retire(*args)
    assert result["removed_files"] == 2
    root = args[0]
    assert (root / "case-00001/result.json").is_file()
    assert (root / "case-00001/state.bin").is_file()
    assert not (root / "case-00001/capture/a.bin").exists()
    assert (root / "archive-locator.json").is_file()
    assert module.retire(*args) == result


@pytest.mark.parametrize(
    "fault", ["manifest", "report", "completion", "source_status", "symlink_parent", "payload"]
)
def test_retirement_refuses_changed_evidence(retirement, fault):
    module, args = retirement
    root, report, manifest, completion, _ = args
    if fault in {"manifest", "report", "completion"}:
        path = {"manifest": manifest, "report": report, "completion": completion}[fault]
        path.write_bytes(path.read_bytes() + b"changed")
    elif fault == "source_status":
        path = root / "checkpoint.json"
        doc = json.loads(path.read_text())
        doc.pop("sha256")
        doc["status"] = "running_case"
        path.write_text(json.dumps(seal(doc)))
    elif fault == "symlink_parent":
        path = root / "case-00001/capture"
        path.rename(path.with_name("original-capture"))
        path.symlink_to("original-capture")
    else:
        (root / "case-00001/capture/a.bin").write_bytes(b"modified original payload")
    with pytest.raises(ValueError):
        module.retire(*args)
    assert not (root / "archive-retirement.json").exists()
    assert (root / "case-00001/capture/a.bin").exists()
    assert (root / "case-00001/capture/b.bin").exists()


def test_retirement_resumes_partial_cleanup_without_losing_changed_file(retirement):
    module, args = retirement
    root = args[0]
    changed = root / "case-00001/capture/b.bin"
    original = changed.read_bytes()
    changed.write_bytes(b"unarchived change")
    with pytest.raises(ValueError):
        module.retire(*args)
    assert not (root / "case-00001/capture/a.bin").exists()
    assert changed.read_bytes() == b"unarchived change"
    changed.write_bytes(original)
    result = module.retire(*args)
    assert result["previously_absent_files"] == 1
    assert result["removed_files"] == 1


def test_retirement_never_runs_under_an_active_controller(retirement):
    module, args = retirement
    with (args[0] / "queue.lock").open("rb") as controller:
        fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            module.retire(*args)
    assert (args[0] / "case-00001/capture/a.bin").is_file()


@pytest.fixture
def finished_case(tmp_path):
    run = tmp_path / "live-run"
    case = run / "case-00000"
    (case / "process").mkdir(parents=True)
    (case / "capture").mkdir()
    (run / "case-00001").mkdir()
    (run / "case-00001/live.bin").write_bytes(b"active worker output")
    (run / "queue.lock").touch()
    spec = {"id": "finished-case", "family": "forced_d7"}
    campaign = seal({"cases": [spec, {"id": "next-case"}]})
    (run / "campaign.json").write_text(json.dumps(campaign))
    (run / "checkpoint.json").write_text(
        json.dumps(
            seal(
                {
                    "campaign": campaign["sha256"],
                    "status": "running_case",
                    "current": "next-case",
                    "attempt": "case-00001",
                }
            )
        )
    )
    (run / "checkpoint-case-00000.json").write_text(
        json.dumps(seal({"campaign": campaign["sha256"], "counts": {"FAILED": 1}}))
    )
    (case / "input.json").write_text(json.dumps({"campaign": campaign, "case": spec}))
    result = seal({"campaign": campaign["sha256"], "case": spec, "status": "FAILED"})
    for name in ("result.json", "worker-result.json"):
        (case / name).write_text(json.dumps(result))
    (case / "process/shutdown.json").write_text(
        json.dumps({"cleanup_error": None, "returncode": 1})
    )
    (case / "capture/tensor.bin").write_bytes(b"failed comparison evidence")
    return case


def test_case_archives_and_retires_while_different_case_owns_queue(
    archive, retirement, finished_case, tmp_path
):
    retire, _ = retirement
    root = finished_case
    paths = [
        tmp_path / ("case-" + n)
        for n in ("stream.json", "members.gz", "readback.json", "complete.json")
    ]
    receipt, manifest, report, completion = paths
    output = io.BytesIO()
    with (root.parent / "queue.lock").open("rb") as controller:
        fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
        archive.stream(root, receipt, output, case=True)
        archive.verify(io.BytesIO(output.getvalue()), receipt, manifest, report)
        with gzip.open(manifest, "rt") as records:
            names = [json.loads(line)["path"] for line in records]
        assert all(name.startswith("case-00000") for name in names)
        completion.write_text(
            json.dumps(
                {
                    "snapshot_id": "b" * 64,
                    "readback_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                }
            )
        )
        result = retire.retire(root, report, manifest, completion, "workstation:/private/archive")
    assert result["removed_files"] == 1
    assert (root / "result.json").exists()
    assert not (root / "capture/tensor.bin").exists()
    assert (root.parent / "case-00001/live.bin").read_bytes() == b"active worker output"


@pytest.mark.parametrize(
    "fault",
    [
        "current",
        "checkpoint_missing",
        "cleanup_error",
        "cleanup_missing",
        "worker_mismatch",
        "input_mismatch",
        "result_missing",
        "lease_not_released",
    ],
)
def test_case_archive_refuses_uncertain_worker_lifetime(archive, finished_case, tmp_path, fault):
    root = finished_case
    if fault == "current":
        p = root.parent / "checkpoint.json"
        d = json.loads(p.read_text())
        d.update(current="finished-case", attempt=root.name)
        d.pop("sha256")
        p.write_text(json.dumps(seal(d)))
    elif fault == "checkpoint_missing":
        (root.parent / f"checkpoint-{root.name}.json").unlink()
    elif fault == "cleanup_error":
        (root / "process/shutdown.json").write_text(
            json.dumps({"cleanup_error": "workers remain", "returncode": 1})
        )
    elif fault == "cleanup_missing":
        (root / "process/shutdown.json").unlink()
    elif fault == "worker_mismatch":
        (root / "worker-result.json").write_text(json.dumps(seal({"status": "TESTED"})))
    elif fault == "input_mismatch":
        p = root / "input.json"
        d = json.loads(p.read_text())
        d["case"]["id"] = "wrong-case"
        p.write_text(json.dumps(d))
    elif fault == "result_missing":
        (root / "result.json").unlink()
    else:
        (root / "gpu-lease").mkdir()
    receipt = tmp_path / "refused.json"
    with pytest.raises((ValueError, FileNotFoundError)):
        archive.stream(root, receipt, io.BytesIO(), case=True)
    assert not receipt.exists()
    assert (root / "capture/tensor.bin").exists()


def test_case_archive_identity_survives_parent_progress(archive, finished_case, tmp_path):
    root = finished_case
    before = archive.finished_case_identity(root)

    class QueueAdvances(io.BytesIO):
        def write(self, data):
            p = root.parent / "checkpoint.json"
            d = json.loads(p.read_text())
            d.pop("sha256")
            d.update(current="later-case", attempt="case-00002")
            p.write_text(json.dumps(seal(d)))
            return super().write(data)

    archive.stream(root, tmp_path / "advancing.json", QueueAdvances(), case=True)
    assert archive.finished_case_identity(root) == before


@pytest.mark.parametrize(
    "state",
    [
        "active",
        "different",
        "evicted",
        "pending",
        "partial",
        "corrupt",
        "wrong_campaign",
        "unknown",
    ],
)
def test_archive_defers_live_serial_reference_origin(archive, finished_case, tmp_path, state):
    root = finished_case
    campaign = json.loads((root.parent / "campaign.json").read_text())
    store = root.parent / "serial-reference-store"
    store.mkdir()
    (store / "store.json").write_text(
        json.dumps(seal({"schema": "urn:qwen:serial-reference-store:v1"}))
    )
    current = store / "current"
    if state != "evicted":
        current.mkdir()
        entry = {
            "schema": "urn:qwen:serial-reference-store:v1/entry",
            "campaign": campaign["sha256"],
            "case": campaign["cases"][1 if state == "different" else 0],
            "origin": "/qualification/live-run/case-00000",
        }
        if state == "wrong_campaign":
            entry["campaign"] = "a" * 64
        if state == "unknown":
            entry["case"] = {"id": "undeclared"}
        if state != "partial":
            document = seal(entry)
            if state == "corrupt":
                document["sha256"] = "f" * 64
            (current / "entry.json").write_text(json.dumps(document))
    if state == "pending":
        (store / "pending").mkdir()
    # The archiver must never contend with or alter the store writer's lock.
    with (store / "lock").open("wb") as writer:
        fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state in {"different", "evicted"}:
            archive.stream(root, tmp_path / "reference-stream.json", io.BytesIO(), case=True)
        else:
            with pytest.raises((ValueError, FileNotFoundError)):
                archive.stream(root, tmp_path / "reference-stream.json", io.BytesIO(), case=True)
    assert (root / "capture/tensor.bin").read_bytes() == b"failed comparison evidence"


def test_case_archiver_lock_excludes_other_archivers(archive, finished_case, tmp_path):
    with archive.source_guard(finished_case, case=True), pytest.raises(BlockingIOError):
        archive.stream(finished_case, tmp_path / "duplicate.json", io.BytesIO(), case=True)


@pytest.fixture
def paused(completed):
    p = completed / "checkpoint.json"
    d = json.loads(p.read_text())
    d.pop("sha256")
    d.update(status="paused", current="next-unstarted-case")
    p.write_text(json.dumps(seal(d)))
    return completed


def test_paused_queue_needs_explicit_admission_and_keeps_failed_results(archive, paused, tmp_path):
    receipt = tmp_path / "paused.json"
    with pytest.raises(ValueError):
        archive.stream(paused, receipt, io.BytesIO())
    output = io.BytesIO()
    archive.stream(paused, receipt, output, paused=True)
    archive.verify(
        io.BytesIO(output.getvalue()), receipt, tmp_path / "paused.gz", tmp_path / "readback.json"
    )
    assert json.loads(receipt.read_text())["scope"] == "paused-run"
    assert json.loads((paused / "case-00001/result.json").read_text())["status"] == "FAILED"


@pytest.mark.parametrize(
    "fault", ["lock_held", "unrecorded_attempt", "running", "changed_completion"]
)
def test_paused_admission_does_not_bypass_live_or_incomplete_run(archive, paused, tmp_path, fault):
    stream = io.BytesIO()
    if fault == "unrecorded_attempt":
        case = paused / "case-00002"
        case.mkdir()
        (case / "input.json").write_text("{}")
    elif fault == "running":
        p = paused / "checkpoint.json"
        d = json.loads(p.read_text())
        d.pop("sha256")
        d["status"] = "running_case"
        p.write_text(json.dumps(seal(d)))
    elif fault == "changed_completion":

        class ChangeResult(io.BytesIO):
            def write(self, data):
                p = paused / "case-00001/result.json"
                d = json.loads(p.read_text())
                d.pop("sha256")
                d["status"] = "TESTED"
                p.write_text(json.dumps(seal(d)))
                return super().write(data)

        stream = ChangeResult()
    with (paused / "queue.lock").open("rb") as controller:
        if fault == "lock_held":
            fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises((ValueError, FileNotFoundError, BlockingIOError)):
            archive.stream(paused, tmp_path / "refused.json", stream, paused=True)
    assert not (tmp_path / "refused.json").exists()


def test_verified_paused_retirement_preserves_resume_metadata(archive, retirement, tmp_path):
    module, args = retirement
    root = tmp_path / "paused-copy"
    shutil.copytree(args[0], root, symlinks=True)
    checkpoint = root / "checkpoint.json"
    value = json.loads(checkpoint.read_text())
    value.pop("sha256")
    value.update(status="paused", current="next-case")
    checkpoint.write_text(json.dumps(seal(value)))
    original = checkpoint.read_bytes()
    receipt, manifest, report, completion = [
        tmp_path / ("paused-" + name)
        for name in ("source.json", "members.gz", "readback.json", "complete.json")
    ]
    output = io.BytesIO()
    archive.stream(root, receipt, output, paused=True)
    archive.verify(io.BytesIO(output.getvalue()), receipt, manifest, report)
    completion.write_text(
        json.dumps(
            {
                "snapshot_id": "c" * 64,
                "readback_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            }
        )
    )
    result = module.retire(root, report, manifest, completion, "workstation:/private/archive")
    assert result["removed_files"] == 2
    assert checkpoint.read_bytes() == original
    assert (root / "case-00001/input.json").is_file()
    assert (root / "case-00001/result.json").is_file()
    assert (root / "case-00001/state.bin").is_file()

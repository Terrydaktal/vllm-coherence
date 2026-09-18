"""Already archived CPU reference tensors can be retired without losing metadata."""

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest
from test_conformance_archive import archive as archive
from test_conformance_archive import finished_case as finished_case

from qwen_r9700_lab.diagnostic_contract import seal


@pytest.fixture
def reference_archive(archive, finished_case, tmp_path, monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "experiments/radiance-public"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "reference_retirement", directory / "retire_conformance_replica.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = finished_case
    values = {}
    for index, frame in enumerate(
        ["boundaries/p000000063-l000-input_norm", "frame-000000", "semantic/p000000000-l000-input"]
    ):
        relative = "reference/" + frame + "/" + hashlib.sha256(frame.encode()).hexdigest() + ".bin"
        value = bytes([index + 1]) * 4096
        path = root / relative
        path.parent.mkdir(parents=True)
        path.write_bytes(value)
        (path.parent / "frame.json").write_text('{"fixture":"retain descriptor"}\n')
        values[relative] = value
    (root / "reference/schedule.json").write_text('{"fixture":"retain schedule"}\n')
    (root / "reference/unrelated.bin").write_bytes(b"not a canonical tensor")
    stream = io.BytesIO()
    receipt, manifest, report, complete = [
        tmp_path / name for name in ["stream.json", "members.gz", "readback.json", "complete.json"]
    ]
    archive.stream(root, receipt, stream, case=True)
    archive.verify(io.BytesIO(stream.getvalue()), receipt, manifest, report)
    complete.write_text(
        json.dumps(
            {
                "snapshot_id": hashlib.sha256(stream.getvalue()).hexdigest(),
                "readback_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            }
        )
    )
    args = root, report, manifest, complete, "test:retained-repository"
    return module, args, values, stream.getvalue()


def test_retire_reference_payload_after_primary_preserves_archive_and_metadata(reference_archive):
    module, args, values, saved = reference_archive
    root = args[0]
    module.retire(*args)
    original_receipt = (root / "archive-retirement.json").read_bytes()
    assert all((root / name).read_bytes() == value for name, value in values.items())
    result = module.retire(*args, reference_captures_only=True)
    assert result["removed_files"] == len(values)
    assert result["removed_bytes"] == sum(map(len, values.values()))
    assert all(not (root / name).exists() for name in values)
    assert (root / "archive-retirement.json").read_bytes() == original_receipt
    assert (root / "reference/unrelated.bin").read_bytes() == b"not a canonical tensor"
    assert (root / "reference/schedule.json").is_file()
    assert all((root / name).with_name("frame.json").is_file() for name in values)
    assert (root / "archive-reference-locator.json").is_file()
    with tarfile.open(fileobj=io.BytesIO(saved)) as stored:
        for name, value in values.items():
            assert stored.extractfile(root.name + "/" + name).read() == value
    assert module.retire(*args, reference_captures_only=True) == result


@pytest.mark.parametrize(
    "fault", ["missing_primary", "changed_primary", "changed_payload", "symlink"]
)
def test_reference_retirement_requires_the_verified_primary_and_unchanged_bytes(
    reference_archive, fault
):
    module, args, values, _ = reference_archive
    root = args[0]
    if fault != "missing_primary":
        module.retire(*args)
    if fault == "changed_primary":
        path = root / "archive-retirement.json"
        value = json.loads(path.read_text())
        value["snapshot_id"] = "0" * 64
        path.write_text(json.dumps(value))
    elif fault == "changed_payload":
        (root / next(iter(values))).write_bytes(b"changed archived bytes")
    elif fault == "symlink":
        parent = root / "reference/boundaries"
        parent.rename(root / "retained-boundaries")
        parent.symlink_to(root / "retained-boundaries")
    with pytest.raises(ValueError):
        module.retire(*args, reference_captures_only=True)
    assert all((root / name).exists() for name in values)
    assert not (root / "archive-reference-retirement.json").exists()


def test_partial_reference_retirement_resumes_only_the_original_archive(reference_archive):
    module, args, values, _ = reference_archive
    root = args[0]
    module.retire(*args)
    last = sorted(values)[-1]
    (root / last).write_bytes(b"changed later file")
    with pytest.raises(ValueError):
        module.retire(*args, reference_captures_only=True)
    assert not (root / sorted(values)[0]).exists()
    assert (root / last).read_bytes() == b"changed later file"
    (root / last).write_bytes(values[last])
    result = module.retire(*args, reference_captures_only=True)
    assert result["previously_absent_files"] == 2
    assert result["removed_files"] == 1
    assert not any((root / name).exists() for name in values)


def test_changed_supplemental_receipt_cannot_claim_completion(reference_archive):
    module, args, _, _ = reference_archive
    module.retire(*args)
    module.retire(*args, reference_captures_only=True)
    receipt = args[0] / "archive-reference-retirement.json"
    value = json.loads(receipt.read_text())
    value["snapshot_id"] = "0" * 64
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        module.retire(*args, reference_captures_only=True)


def test_reference_retirement_refuses_a_reactivated_case(reference_archive):
    module, args, values, _ = reference_archive
    root = args[0]
    module.retire(*args)
    path = root.parent / "checkpoint.json"
    checkpoint = json.loads(path.read_text())
    checkpoint.pop("sha256")
    checkpoint.update(current="finished-case", attempt=root.name)
    path.write_text(json.dumps(seal(checkpoint)))
    with pytest.raises(ValueError):
        module.retire(*args, reference_captures_only=True)
    assert all((root / name).read_bytes() == value for name, value in values.items())


@pytest.mark.parametrize(
    "path,allowed",
    [
        ("reference/frame-000000/" + "a" * 64 + ".bin", True),
        ("reference/boundaries/p000000001-l000-input_norm/" + "a" * 64 + ".bin", True),
        ("reference/semantic/p000000001-l000-gdn_qkv/" + "a" * 64 + ".bin", True),
        ("reference/frame-000000/frame.json", False),
        ("reference/semantic/report.json", False),
        ("reference/frame-000000/arbitrary.bin", False),
        ("reference/frame-000000/nested/" + "a" * 64 + ".bin", False),
        ("other/reference/frame-000000/" + "a" * 64 + ".bin", False),
        ("reference/other/" + "a" * 64 + ".bin", False),
    ],
)
def test_reference_selector_is_limited_to_canonical_tensor_blobs(reference_archive, path, allowed):
    module, args, _, _ = reference_archive
    root = args[0]
    actual = module.reference_payload_path(root.name + "/" + path, root.name, case=True)
    assert (actual is not None) is allowed
    run = module.reference_payload_path("run/" + root.name + "/" + path, "run", case=False)
    assert (run is not None) is allowed


def test_reference_selector_rejects_path_escape(reference_archive):
    module, args, _, _ = reference_archive
    with pytest.raises(ValueError):
        module.reference_payload_path(
            args[0].name + "/reference/../secret.bin", args[0].name, case=True
        )

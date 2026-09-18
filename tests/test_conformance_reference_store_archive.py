"""Exercise reference eviction through the actual archive/readback/retirement code."""

import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

from test_conformance_archive import archive as archive
from test_conformance_archive import finished_case as finished_case
from test_conformance_reference_store import REFLINK, first_blob
from test_conformance_reference_store import baseline as baseline

from qwen_r9700_lab.conformance_reference_store import SCHEMA, SerialReferenceStore
from qwen_r9700_lab.diagnostic_contract import private_json, seal


def test_verified_archive_retains_tensor_after_original_and_cache_eviction(
    archive, finished_case, baseline, tmp_path, monkeypatch
):
    directory = Path(__file__).resolve().parents[1] / "experiments/radiance-public"
    monkeypatch.syspath_prepend(str(directory))
    import retire_conformance_replica

    # The archive fixture models a completed failed case beside an active case.
    # Give its owned metadata the same private permissions as real case workers.
    origin = finished_case
    origin.chmod(0o700)
    for name in ["input.json", "result.json", "worker-result.json"]:
        (origin / name).chmod(0o600)
    plan, baseline_capture = baseline
    source = origin / "serial/run/capture"
    source.parent.mkdir(parents=True, mode=0o700)
    shutil.copytree(baseline_capture, source)
    original_blob = first_blob(source)
    original_bytes = original_blob.read_bytes()
    archived_name = str(Path(origin.name) / original_blob.relative_to(origin))
    payload = private_json(origin / "input.json")
    identity = seal(
        {
            "schema": SCHEMA + "/identity",
            "campaign": payload["campaign"]["sha256"],
            "configuration": {"implementation": "CPU reference fixture"},
            "environment": {"purpose": "store/archive integration test"},
        }
    )
    receipt, manifest, report, completion = (
        tmp_path / name for name in ("stream.json", "members.gz", "verified.json", "complete.json")
    )
    output = io.BytesIO()
    with SerialReferenceStore(tmp_path / "store", reflink=REFLINK) as store:
        store.publish(identity, plan, source, origin)
        archive.stream(origin, receipt, output, case=True)
        archive.verify(io.BytesIO(output.getvalue()), receipt, manifest, report)
        completion.write_text(
            json.dumps(
                {
                    "snapshot_id": hashlib.sha256(output.getvalue()).hexdigest(),
                    "readback_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                }
            )
        )
        retired = retire_conformance_replica.retire(
            origin, report, manifest, completion, "test:retained-archive"
        )
        assert retired["removed_files"] > 1
        assert not original_blob.exists()
        assert first_blob(tmp_path / "store/current/capture").read_bytes() == original_bytes
        result = store.retire()
        assert result["retained"]["kind"] == "verified-case-archive"
        assert not (tmp_path / "store/current").exists()
    # The retained archive, rather than the two deleted replicas, still has the
    # exact real tensor. Restic transport is qualified separately.
    with tarfile.open(fileobj=io.BytesIO(output.getvalue())) as stored:
        assert stored.extractfile(archived_name).read() == original_bytes
    assert (origin / "result.json").exists()
    assert (origin.parent / "case-00001/live.bin").read_bytes() == b"active worker output"

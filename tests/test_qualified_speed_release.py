"""Deployment must select authenticated, fully qualified speed artifacts."""

import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/radiance-public"
spec = importlib.util.spec_from_file_location(
    "qualified_speed_release", BASE / "qualified_speed_release.py"
)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
CONTINUATION_DEPLOYMENT_SCHEMA = "urn:coherence:token-continuation-deployment:v1"


def _assert_continuation_smoke(receipt):
    """A live continuation receipt needs real turns and committed journals."""
    smoke = receipt["smoke"]
    assert len(smoke) >= 2
    for row in smoke:
        assert row["status"] == 200 and row["streamed"] is True
        assert row["finish_reason"] == "stop"
        for field in ("prompt_tokens", "completion_tokens"):
            assert type(row[field]) is int and row[field] > 0
        assert type(row["cached_tokens"]) is int
        assert 0 <= row["cached_tokens"] <= row["prompt_tokens"]
        assert math.isfinite(row["first_content_ms"])
        assert math.isfinite(row["elapsed_ms"])
        assert 0 <= row["first_content_ms"] <= row["elapsed_ms"]
        assert len(row["output_sha256"]) == 64
        assert all(value in "0123456789abcdef" for value in row["output_sha256"])
    assert smoke[1]["cached_tokens"] > 0
    events = receipt["token_continuation_events"]
    assert events and all(
        type(count) is int and count >= 0 for count in events.values()
    )
    assert events.get("journal_saved", 0) >= 2
    assert (
        events.get("already_exact", 0)
        + events.get("preserved_generated_tokens", 0)
        + events.get("incremental_suffix_encoded", 0)
        >= 1
    )
    if (
        receipt.get("continuation_contract")
        == "suffix-only-before-history-tokenization-v1"
    ):
        assert events.get("incremental_suffix_encoded", 0) >= 1
        records = receipt.get("incremental_tokenization")
        assert isinstance(records, list) and records
        assert len(records) <= events["incremental_suffix_encoded"]
        for row in records:
            assert isinstance(row, dict)
            assert row.get("reason") == "incremental_suffix_encoded"
            for field in ("previous_tokens", "suffix_tokens", "admitted_tokens"):
                assert type(row.get(field)) is int and row[field] > 0
            assert row["previous_tokens"] >= smoke[0]["prompt_tokens"]
            assert (
                row["admitted_tokens"] == row["previous_tokens"] + row["suffix_tokens"]
            )
        assert records[-1]["admitted_tokens"] == smoke[1]["prompt_tokens"]
    assert type(receipt["journal_files"]) is int and 1 <= receipt["journal_files"] <= 8


@pytest.fixture
def deployment(tmp_path):
    patches, payload, package = [
        tmp_path / p for p in ("patches", "payload", "package")
    ]
    for p in (patches, payload, package):
        p.mkdir()
    (patches / "optimized-release.json").write_text("parent")
    (patches / "speed_candidate_worker.py").write_text("qualified worker")
    (payload / "kernel.so").write_bytes(b"qualified kernel")
    manifest = copy.deepcopy(
        json.loads((BASE / "qualified-speed-release.json").read_text())
    )
    manifest["parent_manifest_sha256"] = release.digest(
        patches / "optimized-release.json"
    )
    manifest["worker_sha256"] = release.digest(patches / "speed_candidate_worker.py")
    manifest["files"] = {"kernel.so": release.digest(payload / "kernel.so")}
    evidence = {
        "status": "PASS_FOR_DECLARED_SCOPE",
        "sources": {"speed_candidate_worker.py": manifest["worker_sha256"]},
        "runs": {
            name: {
                "status": "PASS",
                "observer_enabled": observed,
                "case_group": "all",
                "cases": [{"status": "PASS"} for _ in range(11)],
            }
            for name, observed in (("observed", True), ("plain", False))
        },
    }
    (payload / "lifecycle-qualification.json").write_text(json.dumps(evidence))
    manifest["qualification_sha256"] = release.digest(
        payload / "lifecycle-qualification.json"
    )
    (patches / "qualified-speed-release.json").write_text(json.dumps(manifest))
    args = [
        "--worker-cls",
        "speed_candidate_worker.SpeedCandidateWorker",
        "--compilation-config",
        '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8]}',
    ]
    return patches, payload, package, args


def test_installed_worker_and_all_three_speed_paths(deployment):
    patches, payload, package, args = deployment
    env = {}
    release.install(patches, package, args, root=payload, environ=env)
    assert (package / "speed_candidate_worker.py").read_bytes() == (
        patches / "speed_candidate_worker.py"
    ).read_bytes()
    assert env["QWEN_SPEED_FULL_GRAPH_CAPTURE"] == "1"
    assert env["QWEN_SPEED_TARGET_GEMM_BUILD"] == "/work/gemm-tuning-v9"
    assert env["QWEN_SPEED_DRAFT_ATTN_SOURCE"].endswith("unit_w4_s1_occ2.py")


@pytest.mark.parametrize(
    "fault",
    [
        "kernel",
        "worker",
        "parent",
        "old-launcher",
        "incomplete-qualification",
        "experiment",
    ],
)
def test_bad_deployment_is_rejected_before_installing_or_setting_environment(
    deployment, fault
):
    patches, payload, package, args = deployment
    env = {}
    if fault == "kernel":
        (payload / "kernel.so").write_bytes(b"unqualified")
    elif fault == "worker":
        (patches / "speed_candidate_worker.py").write_text("different worker")
    elif fault == "parent":
        (patches / "optimized-release.json").write_text("different parent")
    elif fault == "old-launcher":
        args[1] = "optimized_d7_worker.OptimizedWorker"
    elif fault == "experiment":
        env["QWEN_SPEED_SAMPLER_GRAPH"] = "1"
    else:
        p = payload / "lifecycle-qualification.json"
        evidence = json.loads(p.read_text())
        evidence["runs"]["plain"]["cases"].pop()
        p.write_text(json.dumps(evidence))
        m = patches / "qualified-speed-release.json"
        manifest = json.loads(m.read_text())
        manifest["qualification_sha256"] = release.digest(p)
        m.write_text(json.dumps(manifest))
    before = dict(env)
    with pytest.raises(ValueError):
        release.install(patches, package, args, root=payload, environ=env)
    assert env == before and not list(package.iterdir())


def test_runtime_packaging_includes_speed_installer_and_bound_worker(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    from package_runtime import PATCHES

    assert {
        "qualified_speed_release.py",
        "qualified-speed-release.json",
        "speed_candidate_worker.py",
    } <= set(PATCHES)
    launcher = (BASE / "launch_public_clean_snapshot_server.sh").read_text()
    assert "-e QWEN_QUALIFIED_SPEED=1" in launcher
    assert "--worker-cls speed_candidate_worker.SpeedCandidateWorker" in launcher
    assert "FULL_AND_PIECEWISE" in launcher


def test_deployment_receipt_matches_current_sources_and_qualification():
    results = ROOT / "benchmarks/results"
    current = json.loads((results / "coherence-current.json").read_text())
    receipt = json.loads((ROOT / current["current_deployment"]).read_text())
    manifest = json.loads((BASE / "qualified-speed-release.json").read_text())
    qualification = results / "speed-lifecycle-20260926.json"
    assert receipt["smoke_status"] == "PASS" and receipt["health_status"] == 200
    assert receipt["worker"] == manifest["worker"] == release.WORKER
    assert receipt["qualification_manifest_sha256"] == release.digest(
        BASE / "qualified-speed-release.json"
    )
    assert manifest["qualification_sha256"] == release.digest(qualification)
    for installed, source in (
        ("speed_candidate_worker.py", "speed_candidate_worker.py"),
        ("qwen_radiance_fair_scheduler.py", "radiance_fair_scheduler.py"),
        ("qwen_radiance_response_end.py", "radiance_response_end.py"),
        ("qwen_radiance_response_offload.py", "radiance_response_offload.py"),
        ("qwen_radiance_chat_tier.py", "radiance_chat_tier.py"),
    ):
        assert receipt["installed_source_hashes"][installed] == release.digest(
            BASE / source
        )
    snapshot_manifest = BASE / "snapshot-abi-chat-cache-v1.json"
    snapshot = json.loads(snapshot_manifest.read_text())
    authenticated_abi = receipt.get(
        "authenticated_manifest_abi", receipt["runtime_abi"]
    )
    assert authenticated_abi == release.digest(snapshot_manifest)
    # A metadata-only refresh can retain the already-running, explicitly
    # reviewed predecessor. Installed serving sources must still match above.
    assert receipt["runtime_abi"] in {
        authenticated_abi,
        *snapshot["runtime"]["memory_report"]["compatible_runtime_abis"],
    }
    if "worker_processes" in receipt:
        assert receipt["worker_processes"]
        assert all(
            worker["THP_enabled"] == "0" for worker in receipt["worker_processes"]
        )
        assert (
            receipt["installed_source_hashes"]["qwen_radiance_memory.py"]
            == (snapshot["runtime"]["chat_storage"]["modules"]["radiance_memory.py"])
        )
    assert receipt["data_abi"] == snapshot["storage"]["data_abi"]
    # Different repairs need different qualification workloads. The historical
    # prefill repair used five calls; that count is not a serving invariant.
    assert receipt["smoke"]
    assert all(row["status"] == 200 for row in receipt["smoke"])
    if receipt["schema"] == CONTINUATION_DEPLOYMENT_SCHEMA:
        _assert_continuation_smoke(receipt)
        assert receipt["model_arithmetic_changed"] is False
        assert receipt["snapshot_data_abi_changed"] is False
        for name in (
            "radiance_token_continuation.py",
            "radiance_token_continuation_runtime.py",
        ):
            expected = release.digest(ROOT / "src/qwen_r9700_lab" / name)
            assert receipt["installed_source_hashes"]["qwen_" + name] == expected
            assert snapshot["runtime"]["chat_storage"]["modules"][name] == expected


@pytest.mark.parametrize(
    "fault",
    [
        "missing-turn",
        "not-streamed",
        "not-natural-stop",
        "empty-output",
        "uncached-followup",
        "missing-journal",
        "missing-continuation",
        "missing-file",
        "non-finite-latency",
    ],
)
def test_continuation_receipt_rejects_missing_or_failed_evidence(fault):
    smoke = [
        {
            "status": 200,
            "streamed": True,
            "finish_reason": "stop",
            "prompt_tokens": 4096 + index * 10,
            "completion_tokens": 10,
            "cached_tokens": 4096 if index else 0,
            "first_content_ms": 100.0,
            "elapsed_ms": 200.0,
            "output_sha256": "a" * 64,
        }
        for index in range(2)
    ]
    receipt = {
        "smoke": smoke,
        "journal_files": 1,
        "token_continuation_events": {"journal_saved": 2, "already_exact": 1},
    }
    if fault == "missing-turn":
        smoke.pop()
    elif fault == "not-streamed":
        smoke[1]["streamed"] = False
    elif fault == "not-natural-stop":
        smoke[1]["finish_reason"] = "length"
    elif fault == "empty-output":
        smoke[1]["completion_tokens"] = 0
    elif fault == "uncached-followup":
        smoke[1]["cached_tokens"] = 0
    elif fault == "missing-journal":
        receipt["token_continuation_events"]["journal_saved"] = 1
    elif fault == "missing-continuation":
        receipt["token_continuation_events"]["already_exact"] = 0
    elif fault == "missing-file":
        receipt["journal_files"] = 0
    else:
        smoke[1]["first_content_ms"] = float("nan")
    with pytest.raises(AssertionError):
        _assert_continuation_smoke(receipt)


@pytest.mark.parametrize(
    "fault",
    [
        "missing-event",
        "zero-event",
        "boolean-event",
        "missing-records",
        "empty-records",
        "mapping-records",
        "nonmapping-row",
        "missing-reason",
        "wrong-reason",
        "missing-suffix",
        "boolean-suffix",
        "zero-suffix",
        "negative-suffix",
        "string-previous",
        "negative-previous",
        "boolean-admitted",
        "wrong-sum",
        "wrong-prompt-count",
        "too-many-records",
    ],
)
def test_suffix_only_receipt_rejects_missing_or_malformed_incremental_evidence(fault):
    receipt = {
        "continuation_contract": "suffix-only-before-history-tokenization-v1",
        "smoke": [
            {
                "status": 200,
                "streamed": True,
                "finish_reason": "stop",
                "prompt_tokens": 4096 + index * 10,
                "completion_tokens": 10,
                "cached_tokens": 4096 if index else 0,
                "first_content_ms": 100.0,
                "elapsed_ms": 200.0,
                "output_sha256": "a" * 64,
            }
            for index in range(2)
        ],
        "journal_files": 1,
        "token_continuation_events": {
            "journal_saved": 2,
            "already_exact": 1,
            "incremental_suffix_encoded": 1,
        },
        "incremental_tokenization": [
            {
                "reason": "incremental_suffix_encoded",
                "previous_tokens": 4100,
                "suffix_tokens": 6,
                "admitted_tokens": 4106,
            }
        ],
    }
    _assert_continuation_smoke(receipt)
    events, records = (
        receipt["token_continuation_events"],
        receipt["incremental_tokenization"],
    )
    row = records[0]
    if fault == "missing-event":
        del events["incremental_suffix_encoded"]
    elif fault == "zero-event":
        events["incremental_suffix_encoded"] = 0
    elif fault == "boolean-event":
        events["incremental_suffix_encoded"] = True
    elif fault == "missing-records":
        del receipt["incremental_tokenization"]
    elif fault == "empty-records":
        records.clear()
    elif fault == "mapping-records":
        receipt["incremental_tokenization"] = row
    elif fault == "nonmapping-row":
        records[0] = None
    elif fault == "missing-reason":
        del row["reason"]
    elif fault == "wrong-reason":
        row["reason"] = "already_exact"
    elif fault == "missing-suffix":
        del row["suffix_tokens"]
    elif fault == "boolean-suffix":
        row["suffix_tokens"] = True
    elif fault == "zero-suffix":
        row["suffix_tokens"] = 0
    elif fault == "negative-suffix":
        row["suffix_tokens"] = -1
    elif fault == "string-previous":
        row["previous_tokens"] = "4100"
    elif fault == "negative-previous":
        row["previous_tokens"] = -1
    elif fault == "boolean-admitted":
        row["admitted_tokens"] = True
    elif fault == "wrong-sum":
        row["suffix_tokens"] = 5
    elif fault == "wrong-prompt-count":
        row["previous_tokens"] += 1
        row["admitted_tokens"] += 1
    else:
        records.append(dict(row))
    with pytest.raises(AssertionError):
        _assert_continuation_smoke(receipt)

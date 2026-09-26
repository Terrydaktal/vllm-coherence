"""Deployment must select authenticated, fully qualified speed artifacts."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/radiance-public"
spec = importlib.util.spec_from_file_location(
    "qualified_speed_release", BASE / "qualified_speed_release.py"
)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


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
    ):
        assert receipt["installed_source_hashes"][installed] == release.digest(
            BASE / source
        )
    snapshot_manifest = BASE / "snapshot-abi-chat-cache-v1.json"
    assert receipt["runtime_abi"] == release.digest(snapshot_manifest)
    assert (
        receipt["data_abi"]
        == json.loads(snapshot_manifest.read_text())["storage"]["data_abi"]
    )
    assert len(receipt["smoke"]) == 5

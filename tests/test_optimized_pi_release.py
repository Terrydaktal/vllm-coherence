import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "optimized_pi_release", ROOT / "experiments/radiance-public/optimized_pi_release.py"
)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def payload(tmp_path):
    root = tmp_path / "payload"
    root.mkdir()
    artifact = root / "kernel.py"
    artifact.write_text("qualified source\n")
    manifest = {
        "schema": "urn:qwen:optimized-pi-release:v1",
        "state_layout": "existing-nine-slot",
        "container_root": "/qualification",
        "files": {"kernel.py": hashlib.sha256(artifact.read_bytes()).hexdigest()},
        "environment": {
            "QWEN_STOCK_GDN_LAZY": "0",
            "RADIANCE_GDN_LAZY": "0",
            "TORCHINDUCTOR_EMULATE_PRECISION_CASTS": "1",
            "RADIANCE_VERIFY_HEAD": "0",
        },
        "pythonpath": ["/qualification/runtime"],
    }
    return root, artifact, manifest


def publish(tmp_path, manifest):
    path = tmp_path / "optimized-release.json"
    path.write_text(json.dumps(manifest))
    return {"optimized_d7": {"manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}


def test_installs_existing_layout_and_required_precision(tmp_path):
    root, _, manifest = payload(tmp_path)
    env = {"RADIANCE_GDN_LAZY": "1"}
    release.configure(publish(tmp_path, manifest), tmp_path, root=root, environ=env)
    assert env["RADIANCE_GDN_LAZY"] == env["QWEN_STOCK_GDN_LAZY"] == "0"
    assert env["TORCHINDUCTOR_EMULATE_PRECISION_CASTS"] == "1"
    assert env["PYTHONPATH"] == "/opt/vllm/lib/python3.12/site-packages:/qualification/runtime"


def test_modified_payload_fails_before_environment_publication(tmp_path):
    root, artifact, manifest = payload(tmp_path)
    profile = publish(tmp_path, manifest)
    artifact.write_text("changed source\n")
    env = {}
    with pytest.raises(ValueError, match="payload changed"):
        release.configure(profile, tmp_path, root=root, environ=env)
    assert env == {}


@pytest.mark.parametrize(
    "key,value",
    [("RADIANCE_GDN_LAZY", "1"), ("TORCHINDUCTOR_EMULATE_PRECISION_CASTS", "0")],
)
def test_manifest_cannot_silently_change_layout_or_arithmetic(tmp_path, key, value):
    root, _, manifest = payload(tmp_path)
    manifest["environment"][key] = value
    with pytest.raises(ValueError, match="contract changed"):
        release.configure(publish(tmp_path, manifest), tmp_path, root=root, environ={})


@pytest.mark.parametrize("depth", [256, 512])
def test_global_head_is_explicit_and_authenticated(tmp_path, depth):
    root, _, manifest = payload(tmp_path)
    manifest["target_head"] = f"global{depth}"
    manifest["environment"].update(
        RADIANCE_VERIFY_HEAD="1", RADIANCE_VERIFY_HEAD_GLOBAL_TOPK=str(depth)
    )
    profile = publish(tmp_path, manifest)
    with pytest.raises(ValueError, match="target head differs"):
        release.configure(profile, tmp_path, root=root, environ={})
    profile["optimized_d7"]["target_head"] = {"mode": f"global{depth}"}
    env = {}
    release.configure(profile, tmp_path, root=root, environ=env)
    assert env["RADIANCE_VERIFY_HEAD_GLOBAL_TOPK"] == str(depth)
    assert env["RADIANCE_VERIFY_HEAD"] == "1"


def test_global512_cannot_publish_the_old_candidate_depth(tmp_path):
    root, _, manifest = payload(tmp_path)
    manifest["target_head"] = "global512"
    manifest["environment"].update(
        RADIANCE_VERIFY_HEAD="1", RADIANCE_VERIFY_HEAD_GLOBAL_TOPK="256"
    )
    profile = publish(tmp_path, manifest)
    profile["optimized_d7"]["target_head"] = {"mode": "global512"}
    env = {}
    with pytest.raises(ValueError, match="contract changed"):
        release.configure(profile, tmp_path, root=root, environ=env)
    assert env == {}


def test_global512_retains_global256_snapshot_data_identity():
    from copy import deepcopy

    old = {
        "kernel_environment": {"RADIANCE_VERIFY_HEAD_GLOBAL_TOPK": "256"},
        "optimized_arithmetic": {"m1": "fixed", "layout": "nine-slot"},
    }
    new = deepcopy(old)
    new["kernel_environment"]["RADIANCE_VERIFY_HEAD_GLOBAL_TOPK"] = "512"
    assert release.compatible_output_head_contract(old, new) is old


def test_output_head_change_reuses_only_identical_backbone_state_contract():
    from copy import deepcopy

    old = {
        "kernel_environment": {"RADIANCE_VERIFY_HEAD": "0", "PRECISION": "fixed"},
        "serving": {"target_verify_head": False},
        "layout": "nine-slot",
        "model": "same",
    }
    new = deepcopy(old)
    new["kernel_environment"].update(
        RADIANCE_VERIFY_HEAD="1", RADIANCE_VERIFY_HEAD_GLOBAL_TOPK="256"
    )
    new["serving"]["target_verify_head"] = True
    assert release.compatible_output_head_contract(old, new) is old
    for field, value in (("layout", "three-slot"), ("model", "different")):
        changed = deepcopy(new)
        changed[field] = value
        assert release.compatible_output_head_contract(old, changed) is changed
    new["kernel_environment"]["PRECISION"] = "different"
    assert release.compatible_output_head_contract(old, new) is new
    assert old["kernel_environment"]["RADIANCE_VERIFY_HEAD"] == "0"

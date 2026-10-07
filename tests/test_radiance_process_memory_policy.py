"""Isolated release-binding refresh checks; no GPU imports or inference."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "experiments/radiance-public/bootstrap_radiance_release.py"


@pytest.mark.parametrize("change", ["unchanged", "changed-data", "invalid-predecessor"])
def test_binding_refresh_preserves_only_reviewed_compatible_predecessors(
    tmp_path, change
):
    """Exercise the actual generator on an isolated copy, without GPU imports."""
    base = BOOTSTRAP.parent
    manifest = json.loads((base / "snapshot-abi-chat-cache-v1.json").read_text())
    sources = {
        *(
            Path("experiments/radiance-public") / name
            for name in manifest["runtime"]["release_files"]
        ),
        *(
            (
                Path("src/qwen_r9700_lab")
                if (ROOT / "src/qwen_r9700_lab" / name).is_file()
                else Path("experiments/radiance-public")
            )
            / name
            for name in manifest["runtime"]["chat_storage"]["modules"]
        ),
        *(
            Path("experiments/radiance-public") / name
            for name in (
                "update_chat_snapshot_bindings.py",
                "snapshot-abi.json",
                "snapshot-abi-chat-cache-v1.json",
                "patch_streaming_snapshot.py",
                "launch_public_clean_snapshot_server.sh",
                "optimized_pi_release.py",
            )
        ),
        Path("scripts/pi-remote-qwen-radiance"),
    }
    for relative in sources:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    isolated_base = tmp_path / "experiments/radiance-public"
    if change == "changed-data":
        profile_path = isolated_base / "runtime-radiance-1.0.16.json"
        profile = json.loads(profile_path.read_text())
        profile["image_digest"] = "sha256:" + "f" * 64
        profile_path.write_text(json.dumps(profile))
    elif change == "invalid-predecessor":
        manifest["runtime"]["memory_report"]["compatible_runtime_abis"].append(
            "invalid"
        )
        (isolated_base / "snapshot-abi-chat-cache-v1.json").write_text(
            json.dumps(manifest)
        )
    completed = subprocess.run(
        [sys.executable, str(isolated_base / "update_chat_snapshot_bindings.py")],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if change == "invalid-predecessor":
        assert completed.returncode != 0
        assert "invalid reviewed predecessor runtime ABI" in completed.stderr
        return
    assert completed.returncode == 0, completed.stderr
    result_path = isolated_base / "snapshot-abi-chat-cache-v1.json"
    result = json.loads(result_path.read_text())
    reviewed = manifest["runtime"]["memory_report"]["compatible_runtime_abis"]
    compatible = result["runtime"]["memory_report"]["compatible_runtime_abis"]
    if change == "unchanged":
        assert result["storage"]["data_abi"] == manifest["storage"]["data_abi"]
        assert compatible == reviewed
    else:
        assert result["storage"]["data_abi"] != manifest["storage"]["data_abi"]
        assert not set(reviewed[3:]) & set(compatible)
    for name in (
        "radiance_memory.py",
        "radiance_cache_telemetry.py",
        "radiance_kfd_trace.py",
    ):
        expected = hashlib.sha256(
            (tmp_path / "src/qwen_r9700_lab" / name).read_bytes()
        ).hexdigest()
        assert result["runtime"]["chat_storage"]["modules"][name] == expected
    if change == "unchanged":
        # A second refresh retains the exact binding, rather than silently
        # shrinking the compatibility list on every release refresh.
        before = result_path.read_bytes()
        subprocess.run(
            [sys.executable, str(isolated_base / "update_chat_snapshot_bindings.py")],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        assert result_path.read_bytes() == before

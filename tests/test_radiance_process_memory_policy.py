"""Production startup checks use CPU-only subprocesses and synthetic installers."""

from __future__ import annotations

import ctypes
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_r9700_lab import radiance_memory as memory

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "experiments/radiance-public/bootstrap_radiance_release.py"


def thp_state():
    zero = ctypes.c_ulong(0)
    return ctypes.CDLL(None).prctl(42, zero, zero, zero, zero)


def test_policy_checks_full_disable_with_full_width_arguments():
    calls = []

    def prctl(operation, *args):
        assert all(isinstance(value, ctypes.c_ulong) for value in args)
        calls.append((operation, [value.value for value in args]))
        return 0 if operation == 41 else 1

    receipt = memory.disable_transparent_hugepages(SimpleNamespace(prctl=prctl))
    assert calls == [(41, [1, 0, 0, 0]), (42, [0, 0, 0, 0])]
    assert receipt["schema"] == "urn:qwen-r9700:process-memory-policy:v1"
    assert receipt["host_global_policy_changed"] is False
    assert receipt["pr_get_thp_disable"] == 1


def test_production_entrypoint_sets_policy_before_installer_import_and_worker_exec(
    tmp_path,
):
    parent_before = thp_state()
    global_policy = Path("/sys/kernel/mm/transparent_hugepage/enabled")
    global_before = global_policy.read_bytes()
    (tmp_path / BOOTSTRAP.name).write_bytes(BOOTSTRAP.read_bytes())
    (tmp_path / "radiance_memory.py").write_bytes(Path(memory.__file__).read_bytes())
    query = (
        "import ctypes,json; from pathlib import Path; z=ctypes.c_ulong(0); "
        "print(json.dumps({'prctl':ctypes.CDLL(None).prctl(42,z,z,z,z),"
        "'status':[r for r in Path('/proc/self/status').read_text().splitlines() "
        "if r.startswith('THP_enabled:')]}))"
    )
    (tmp_path / "patch_chat_snapshot.py").write_text(
        "import ctypes,json,subprocess,sys; from pathlib import Path\n"
        "z=ctypes.c_ulong(0)\n"
        "assert ctypes.CDLL(None).prctl(42,z,z,z,z)==1\n"
        f"child=json.loads(subprocess.check_output([sys.executable,'-c',{query!r}]))\n"
        "Path('observed.json').write_text(json.dumps(child))\n"
        "raise SystemExit(0)  # Stop before any real release/GPU installer.\n"
    )
    completed = subprocess.run(
        [sys.executable, str(tmp_path / BOOTSTRAP.name), "--check-only"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    receipt = json.loads(completed.stdout.split(": ", 1)[1])
    observed = json.loads((tmp_path / "observed.json").read_text())
    assert receipt["pr_get_thp_disable"] == observed["prctl"] == 1
    assert observed["status"] == ["THP_enabled:\t0"]
    assert thp_state() == parent_before
    assert global_policy.read_bytes() == global_before


@pytest.mark.parametrize("arguments", [[], ["--check-only"]])
def test_production_policy_failure_prevents_all_release_installer_imports(
    tmp_path, arguments
):
    (tmp_path / BOOTSTRAP.name).write_bytes(BOOTSTRAP.read_bytes())
    (tmp_path / "radiance_memory.py").write_text(
        "def disable_transparent_hugepages():\n"
        "    raise OSError(1, 'could not disable transparent huge pages')\n"
    )
    (tmp_path / "patch_chat_snapshot.py").write_text(
        "from pathlib import Path\nPath('unsafe-installer-imported').touch()\n"
    )
    completed = subprocess.run(
        [sys.executable, str(tmp_path / BOOTSTRAP.name), *arguments],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode != 0
    assert "could not disable transparent huge pages" in completed.stderr
    assert not completed.stdout
    assert not (tmp_path / "unsafe-installer-imported").exists()


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
        "radiance_pinned_memory.py",
        "radiance_kfd_trace.py",
    ):
        expected = hashlib.sha256(
            (tmp_path / "src/qwen_r9700_lab" / name).read_bytes()
        ).hexdigest()
        assert result["runtime"]["chat_storage"]["modules"][name] == expected
    assert result["runtime"]["process_memory_policy"]["per_round_calls"] == 0
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

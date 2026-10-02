"""Reuse authenticates the deployed backend independently of local development."""

import hashlib
import json
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/pi-remote-qwen-radiance"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight_fixture(tmp_path):
    files = {
        name: tmp_path / name
        for name in (
            "launcher.sh", "patch.py", "radiance_cache.py", "radiance_memory.py",
            "radiance_chat_tier.py", "radiance_fair_scheduler.py", "patch_chat_snapshot.py",
            "release_integration.py",
        )
    }
    for path in files.values():
        path.write_text("pinned release source\n")
    abi = tmp_path / "abi.json"
    abi.write_text(json.dumps({
        "storage": {"data_abi": "a" * 64},
        "runtime": {
            "chat_storage": {"modules": {
                name: digest(path) for name, path in files.items()
                if name not in ("launcher.sh", "patch.py", "release_integration.py")
            }},
            "release_files": {"release_integration.py": digest(files["release_integration.py"])},
        },
    }))
    variables = {
        "abi_source": str(abi), "SNAPSHOT_ABI": digest(abi), "SNAPSHOT_DATA_ABI": "a" * 64,
        "REMOTE_LAUNCHER_SHA256": digest(files["launcher.sh"]),
        "SNAPSHOT_PATCH_SHA256": digest(files["patch.py"]),
        "launcher_source": str(files["launcher.sh"]), "patch_source": str(files["patch.py"]),
        "cache_module": str(files["radiance_cache.py"]),
        "memory_module": str(files["radiance_memory.py"]),
        "cache_tier_module": str(files["radiance_chat_tier.py"]),
        "fair_scheduler_module": str(files["radiance_fair_scheduler.py"]),
        "cache_patch_module": str(files["patch_chat_snapshot.py"]),
        "REMOTE_ROOT": "/deployed", "remote_abi": "/deployed/abi.json",
        "remote_launcher": "/deployed/launcher.sh", "remote_patch": "/deployed/patch.py",
        "remote_host": "test-host", "preflight_only": "1",
    }
    return files, abi, variables


def run_local_preflight(variables, reuse):
    source = LAUNCHER.read_text()
    start = source.index('[[ $(sha256sum -- "$abi_source"')
    end = source.index("\njq -e --arg model", start)
    setup = "\n".join(f"{key}={shlex.quote(value)}" for key, value in variables.items())
    script = (
        "set -euo pipefail\n" + setup + f"\nreuse_existing={int(reuse)}\n"
        + 'die() { printf "%s\\n" "$*" >&2; exit 2; }\n'
        + source[start:end]
        + '\nprintf "%s\\n" "${release_preflight[@]}"\n'
    )
    return subprocess.run(["bash", "-s"], input=script, capture_output=True, text=True, timeout=5, check=False)


@pytest.mark.parametrize("changed", [
    "launcher.sh", "patch.py", "radiance_cache.py", "radiance_fair_scheduler.py",
    "release_integration.py",
])
def test_reuse_ignores_unused_local_backend_edits_but_starting_rejects_them(tmp_path, changed):
    files, _abi, variables = preflight_fixture(tmp_path)
    assert run_local_preflight(variables, False).returncode == 0
    files[changed].write_text("unpublished development source\n")
    reused = run_local_preflight(variables, True)
    assert reused.returncode == 0, reused.stderr
    assert "/deployed/radiance-vllm-mxfp4/release_integration.py" in reused.stdout
    assert "hash mismatch" in run_local_preflight(variables, False).stderr


@pytest.mark.parametrize("reuse", [False, True])
def test_local_contract_stays_pinned_in_both_modes(tmp_path, reuse):
    _files, abi, variables = preflight_fixture(tmp_path)
    abi.write_text(abi.read_text() + "\n")
    result = run_local_preflight(variables, reuse)
    assert result.returncode == 2
    assert "local snapshot ABI hash mismatch" in result.stderr


@pytest.mark.parametrize("tamper", [False, True])
def test_reuse_always_checks_remote_hashes_before_preflight_exit(tmp_path, tamper):
    files, abi, variables = preflight_fixture(tmp_path)
    # Exercise the actual SSH argument list and remote verification body. The
    # mock supplies deployment paths and user identity, without contacting a GPU.
    variables.update(
        remote_launcher=str(files["launcher.sh"]), remote_patch=str(files["patch.py"]),
        remote_abi=str(abi), REMOTE_ROOT=str(tmp_path),
    )
    modules = json.loads(abi.read_text())["runtime"]["chat_storage"]["modules"]
    deployed = tmp_path / "radiance-vllm-mxfp4"
    deployed.mkdir()
    for name in [*modules, "release_integration.py"]:
        (deployed / name).write_bytes(files[name].read_bytes())
    if tamper:
        (deployed / "radiance_cache.py").write_text("wrong deployed source\n")
    source = LAUNCHER.read_text()
    start = source.index('ssh -T "$remote_host" /usr/bin/bash -s -- \\\n\t"$remote_launcher"')
    end = source.index("\n((preflight_only == 0)) || exit 0", start)
    setup = "\n".join(f"{key}={shlex.quote(value)}" for key, value in variables.items())
    release_hash = digest(files["release_integration.py"])
    script = "set -euo pipefail\n" + setup + "\n" + f"""
release_preflight=({shlex.quote(str(deployed / 'release_integration.py'))} {release_hash})
ssh() {{
    shift 2
    command bash -c 'stat() {{
      if [[ $1 == -c && $2 == "%U:%h" ]]; then
        printf "lewis:%s\\n" "$(command stat -c %h -- "${{@:3}}")"
      else command stat "$@"; fi
    }}
    source /dev/stdin' bash "${{@:4}}"
}}
""" + source[start:end]
    result = subprocess.run(["bash", "-s"], input=script, capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode == (2 if tamper else 0), result.stderr

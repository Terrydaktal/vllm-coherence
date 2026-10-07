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
            "launcher.sh",
            "patch.py",
            "radiance_cache.py",
            "radiance_memory.py",
            "radiance_cache_telemetry.py",
            "radiance_request_timeline.py",
            "radiance_prefix_lineage.py",
            "radiance_prefix_runtime.py",
            "radiance_future_diagnostic.py",
            "radiance_pinned_memory.py",
            "radiance_kfd_trace.py",
            "radiance_chat_tier.py",
            "radiance_fair_scheduler.py",
            "patch_chat_snapshot.py",
            "release_integration.py",
        )
    }
    for path in files.values():
        path.write_text("pinned release source\n")
    abi = tmp_path / "abi.json"
    abi.write_text(
        json.dumps(
            {
                "storage": {"data_abi": "a" * 64},
                "runtime": {
                    "chat_storage": {
                        "modules": {
                            name: digest(path)
                            for name, path in files.items()
                            if name
                            not in ("launcher.sh", "patch.py", "release_integration.py")
                        }
                    },
                    "release_files": {
                        "release_integration.py": digest(
                            files["release_integration.py"]
                        )
                    },
                },
            }
        )
    )
    variables = {
        "abi_source": str(abi),
        "project_root": str(tmp_path),
        "SNAPSHOT_ABI": digest(abi),
        "SNAPSHOT_DATA_ABI": "a" * 64,
        "REMOTE_LAUNCHER_SHA256": digest(files["launcher.sh"]),
        "SNAPSHOT_PATCH_SHA256": digest(files["patch.py"]),
        "launcher_source": str(files["launcher.sh"]),
        "patch_source": str(files["patch.py"]),
        "cache_module": str(files["radiance_cache.py"]),
        "memory_module": str(files["radiance_memory.py"]),
        "cache_telemetry_module": str(files["radiance_cache_telemetry.py"]),
        "pinned_memory_module": str(files["radiance_pinned_memory.py"]),
        "kfd_trace_module": str(files["radiance_kfd_trace.py"]),
        "cache_tier_module": str(files["radiance_chat_tier.py"]),
        "fair_scheduler_module": str(files["radiance_fair_scheduler.py"]),
        "cache_patch_module": str(files["patch_chat_snapshot.py"]),
        "REMOTE_ROOT": "/deployed",
        "remote_abi": "/deployed/abi.json",
        "remote_launcher": "/deployed/launcher.sh",
        "remote_patch": "/deployed/patch.py",
        "remote_host": "test-host",
        "preflight_only": "1",
    }
    return files, abi, variables


def run_local_preflight(variables, reuse):
    source = LAUNCHER.read_text()
    start = source.index('[[ $(sha256sum -- "$abi_source"')
    end = source.index("\njq -e --arg model", start)
    setup = "\n".join(f"{key}={shlex.quote(value)}" for key, value in variables.items())
    script = (
        "set -euo pipefail\n"
        + setup
        + f"\nreuse_existing={int(reuse)}\n"
        + 'die() { printf "%s\\n" "$*" >&2; exit 2; }\n'
        + source[start:end]
        + '\nprintf "%s\\n" "${chat_storage_preflight[@]}" "${release_preflight[@]}"\n'
    )
    return subprocess.run(
        ["bash", "-s"],
        input=script,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )


@pytest.mark.parametrize(
    "changed",
    [
        "launcher.sh",
        "patch.py",
        "radiance_cache.py",
        "radiance_fair_scheduler.py",
        "radiance_cache_telemetry.py",
        "radiance_request_timeline.py",
        "radiance_prefix_lineage.py",
        "radiance_prefix_runtime.py",
        "radiance_future_diagnostic.py",
        "radiance_pinned_memory.py",
        "radiance_kfd_trace.py",
        "release_integration.py",
    ],
)
def test_reuse_ignores_unused_local_backend_edits_but_starting_rejects_them(
    tmp_path, changed
):
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


@pytest.mark.parametrize(
    "tamper",
    [None, "radiance_cache.py", "radiance_request_timeline.py", "radiance_prefix_lineage.py",
     "radiance_prefix_runtime.py", "radiance_future_diagnostic.py"],
)
def test_reuse_always_checks_remote_hashes_before_preflight_exit(tmp_path, tamper):
    files, abi, variables = preflight_fixture(tmp_path)
    # Exercise the actual SSH argument list and remote verification body. The
    # mock supplies deployment paths and user identity, without contacting a GPU.
    variables.update(
        remote_launcher=str(files["launcher.sh"]),
        remote_patch=str(files["patch.py"]),
        remote_abi=str(abi),
        REMOTE_ROOT=str(tmp_path),
    )
    modules = json.loads(abi.read_text())["runtime"]["chat_storage"]["modules"]
    deployed = tmp_path / "radiance-vllm-mxfp4"
    deployed.mkdir()
    for name in [*modules, "release_integration.py"]:
        (deployed / name).write_bytes(files[name].read_bytes())
    if tamper:
        (deployed / tamper).write_text("wrong deployed source\n")
    source = LAUNCHER.read_text()
    start = source.index(
        'ssh -T "$remote_host" /usr/bin/bash -s -- \\\n\t"$remote_launcher"'
    )
    end = source.index("\n((preflight_only == 0)) || exit 0", start)
    setup = "\n".join(f"{key}={shlex.quote(value)}" for key, value in variables.items())
    release_hash = digest(files["release_integration.py"])
    local_start = source.index('[[ $(sha256sum -- "$abi_source"')
    local_end = source.index("\njq -e --arg model", local_start)
    script = (
        "set -euo pipefail\n"
        + setup
        + "\n"
        + "\nreuse_existing=1\n"
        + 'die() { printf "%s\\n" "$*" >&2; exit 2; }\n'
        + source[local_start:local_end]
        + f"""
release_preflight=({shlex.quote(str(deployed / "release_integration.py"))} {release_hash})
ssh() {{
    shift 2
    command bash -c 'stat() {{
      if [[ $1 == -c && $2 == "%U:%h" ]]; then
        printf "lewis:%s\\n" "$(command stat -c %h -- "${{@:3}}")"
      else command stat "$@"; fi
    }}
    source /dev/stdin' bash "${{@:4}}"
}}
"""
        + source[start:end]
    )
    result = subprocess.run(
        ["bash", "-s"],
        input=script,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == (2 if tamper else 0), result.stderr


@pytest.mark.parametrize("reuse", [False, True])
def test_every_manifest_module_is_in_remote_preflight_without_name_allowlist(tmp_path, reuse):
    _files, abi, variables = preflight_fixture(tmp_path)
    result = run_local_preflight(variables, reuse)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    pairs = dict(zip(lines[::2], lines[1::2], strict=True))
    modules = json.loads(abi.read_text())["runtime"]["chat_storage"]["modules"]
    for name, expected in modules.items():
        assert pairs["/deployed/radiance-vllm-mxfp4/" + name] == expected
    assert len(pairs) == len(modules) + 1  # Separate release integration.


def test_new_module_can_come_from_authoritative_python_source_tree(tmp_path):
    files, _abi, variables = preflight_fixture(tmp_path)
    canonical = tmp_path / "src/qwen_r9700_lab/radiance_future_diagnostic.py"
    canonical.parent.mkdir(parents=True)
    files[canonical.name].rename(canonical)
    assert run_local_preflight(variables, False).returncode == 0
    canonical.write_text("modified module\n")
    assert "chat storage module hash mismatch" in run_local_preflight(variables, False).stderr
    assert run_local_preflight(variables, True).returncode == 0


@pytest.mark.parametrize("invalid", [{}, {"../escape.py": "a" * 64}, {"module.py": "not-a-hash"}])
def test_invalid_module_inventory_fails_before_remote_call(tmp_path, invalid):
    _files, abi, variables = preflight_fixture(tmp_path)
    manifest = json.loads(abi.read_text())
    manifest["runtime"]["chat_storage"]["modules"] = invalid
    abi.write_text(json.dumps(manifest))
    variables["SNAPSHOT_ABI"] = digest(abi)
    result = run_local_preflight(variables, True)
    assert result.returncode == 2
    assert "invalid chat storage module inventory" in result.stderr


@pytest.mark.parametrize(
    "case",
    [
        "current",
        "first_compatible",
        "last_compatible",
        "empty_compatible",
        "unknown_runtime",
        "wrong_image",
        "wrong_data_abi",
        "wrong_model",
        "stopped",
    ],
)
def test_actual_remote_authentication_keeps_complete_compatibility_list_over_ssh(
    tmp_path, case
):
    current, first, last, data_abi = (c * 64 for c in ("a", "b", "c", "d"))
    abi = tmp_path / "abi.json"
    compatible = [] if case == "empty_compatible" else [first, last]
    abi.write_text(
        json.dumps(
            {"runtime": {"memory_report": {"compatible_runtime_abis": compatible}}}
        )
    )
    runtime = (
        first
        if case == "first_compatible"
        else last
        if case == "last_compatible"
        else current
    )
    if case == "unknown_runtime":
        runtime = "f" * 64
    lane_abi = "e" * 64 if case == "wrong_data_abi" else data_abi
    model = "wrong-model" if case == "wrong_model" else "synthetic-model"
    inspection = tmp_path / "inspect.json"
    inspection.write_text(
        json.dumps(
            [
                {
                    "Image": "wrong-image" if case == "wrong_image" else "test-image",
                    "Name": "test-container",
                    "State": {"Running": case != "stopped"},
                    "Args": [
                        "--served-model-name",
                        model,
                        "--max-model-len",
                        "253792",
                        f"/cache/snapshots/{lane_abi}/data",
                        "qwen_chat_fs",
                        "--kv-cache-dtype",
                        "fp8",
                        "--speculative-config",
                        f"qwen-radiance-public-clean-{runtime[:16]}",
                    ],
                }
            ]
        )
    )
    models = tmp_path / "models.json"
    models.write_text(json.dumps({"data": [{"id": "synthetic-model"}]}))
    variables = {
        "CONTAINER": "test-container",
        "MODEL_ID": "synthetic-model",
        "IMAGE_ID": "test-image",
        "REMOTE_PORT": "8080",
        "remote_launcher": "/unused-launcher",
        "REMOTE_CACHE": "/unused-cache",
        "SNAPSHOT_DATA_ABI": data_abi,
        "reuse_existing": "1",
        "SNAPSHOT_ABI": current,
        "abi_source": str(abi),
        "remote_host": "fixture-host",
    }
    source = LAUNCHER.read_text()
    start = source.index("backend_log=$(\n")
    end = source.index("\nprepare_local_port_state", start)
    setup = "\n".join(
        f"{name}={shlex.quote(value)}" for name, value in variables.items()
    )
    script = (
        "set -euo pipefail\n"
        + setup
        + "\n"
        + f"""
podman() {{
    if [[ $1 == inspect ]]; then cat {shlex.quote(str(inspection))};
    elif [[ $1 == container && $2 == exists ]]; then return 0;
    else printf 'Unexpected mutation: %s\\n' "$*" >&2; return 99; fi
}}
curl() {{ cat {shlex.quote(str(models))}; }}
export -f podman curl
ssh() {{
    shift 2
    # Actual SSH joins remote argv into shell text; spaces inside a local quoted
    # argument do not survive as a single positional parameter on the server.
    command bash -c "$*"
}}
die() {{ printf '%s\\n' "$*" >&2; exit 2; }}
"""
        + source[start:end]
        + '\nprintf "%s\\n" "$backend_log"\n'
    )
    result = subprocess.run(
        ["bash", "-s"],
        input=script,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    accepted = case in {
        "current",
        "first_compatible",
        "last_compatible",
        "empty_compatible",
    }
    assert result.returncode == (0 if accepted else 2), result.stderr
    if accepted:
        assert result.stdout == "existing\n"
    else:
        assert (
            "existing Radiance container differs from the authenticated lane"
            in result.stderr
        )

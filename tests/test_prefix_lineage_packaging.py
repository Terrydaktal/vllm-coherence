from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKEND_MODULES = {
    "radiance_prefix_lineage.py", "radiance_prefix_runtime.py",
    "radiance_token_continuation.py", "radiance_token_continuation_runtime.py",
}


def test_runtime_export_binds_both_backend_prefix_modules(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    import package_runtime as package

    source = tmp_path / "source"
    source.mkdir()
    (source / "sample.bin").write_bytes(b"synthetic serving payload")
    (source / "optimized-release.json").write_text("{}")
    support = source / "support/libhsa-runtime64.so.1.21.0"
    support.parent.mkdir()
    support.write_bytes(b"synthetic support library")
    backoff = json.loads(
        (
            ROOT / "experiments/radiance-public/rocr-poll-backoff/runtime.json"
        ).read_text()
    )
    original_digest = package.digest
    monkeypatch.setattr(
        package,
        "verify_payload",
        lambda root: {"files": {"sample.bin": original_digest(root / "sample.bin")}},
    )
    monkeypatch.setattr(
        package,
        "digest",
        lambda path: (
            backoff["library_sha256"] if path == support else original_digest(path)
        ),
    )
    output = tmp_path / "runtime.tar.xz"
    result = package.package(source, output)
    for name in BACKEND_MODULES:
        relative = "src/qwen_r9700_lab/" + name
        assert relative in result["patch_files"]
        assert (
            result["integration_sha256"][relative]
            == hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        )
    with tarfile.open(output) as archive:
        assert set(archive.getnames()) == {
            "sample.bin",
            "optimized-release.json",
            "support/libhsa-runtime64.so.1.21.0",
        }
    # Pi is installed separately on the client, not placed in the GPU payload.
    assert not any(
        name.endswith("qwen-prefix-lineage.mjs") for name in result["patch_files"]
    )


def test_backend_install_and_source_inventory_include_prefix_modules():
    patcher = (ROOT / "experiments/radiance-public/patch_chat_snapshot.py").read_text()
    updater = (
        ROOT / "experiments/radiance-public/update_chat_snapshot_bindings.py"
    ).read_text()
    manifest = json.loads(
        (
            ROOT / "experiments/radiance-public/snapshot-abi-chat-cache-v1.json"
        ).read_text()
    )
    for name in BACKEND_MODULES:
        assert name in updater
        assert name in manifest["runtime"]["chat_storage"]["modules"]
        assert "qwen_" + name in patcher
    assert 'packages = ["src/qwen_r9700_lab"]' in (ROOT / "pyproject.toml").read_text()


@pytest.mark.parametrize("already_installed", [False, True])
def test_portable_pi_reaches_the_existing_lineage_installer_and_checks(
    tmp_path, monkeypatch, already_installed
):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    import coherence_pi as pi

    monkeypatch.chdir(tmp_path)
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    connection = {
        "abi": "a" * 64,
        "model": pi.MODEL,
        "port": 8080,
        "cache_root": "/synthetic-cache",
        "head": "global512",
    }
    (state / "connection.json").write_text(json.dumps(connection))
    if already_installed:
        binary = state / "bin/pi"
        binary.parent.mkdir()
        binary.write_text("synthetic binary")
    calls = []
    launches = []
    monkeypatch.setattr(
        pi.subprocess, "run", lambda command, **kwargs: calls.append(command)
    )
    monkeypatch.setattr(pi.subprocess, "call", lambda command, **kwargs: launches.append(command) or 0)
    monkeypatch.setattr(
        pi.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.StringIO(json.dumps({"data": [{"id": pi.MODEL}]})),
    )
    args = SimpleNamespace(state=state, ssh=None, search_extension=None)
    assert pi.launch(args, ["--continue"]) == 0
    extension = str(ROOT / "integrations/pi/qwen-session-search.mjs")
    assert len(launches) == 1
    assert launches[0].count(extension) == 1
    assert launches[0][launches[0].index(extension) - 1] == "--extension"
    settings = json.loads((state / "agents/8080/settings.json").read_text())
    assert "pi_session_search" in settings["defaultTools"]
    assert "session_search" not in settings["defaultTools"]
    names = str(ROOT / "integrations/pi/qwen-tool-names.ts")
    assert launches[0].count(names) == 1
    assert launches[0][launches[0].index(names) - 1] == "--extension"
    assert launches[0].index(names) > launches[0].index(extension)
    assert launches[0][launches[0].index("--exclude-tools") + 1] == "read,bash,edit,write,grep,find,ls"
    installers = [
        call for call in calls if Path(call[0]).name == "install-pi-coding-agent"
    ]
    assert bool(installers) is not already_installed
    assert [
        str(ROOT / "scripts/patch-pi-tool-call-integrity"),
        "--check",
        str(state / "pi/0.84.2"),
    ] in calls
    installer_source = (ROOT / "scripts/install-pi-coding-agent").read_text()
    assert '"$tool_integrity_patch_script" --apply "$stage_path"' in installer_source
    assert '"$tool_integrity_patch_script" --apply "$runtime"' in installer_source

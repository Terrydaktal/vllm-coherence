import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import coherence_cli as cli


def test_help_and_doctor_are_cpu_only(tmp_path):
    for args in (("--help",), ("doctor",)):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/coherence"),
                *args,
                "--state",
                str(tmp_path / "new"),
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert not (tmp_path / "new").exists()
    report = json.loads(result.stdout)
    assert report["gpu_opened"] is False


@pytest.mark.parametrize("head", [None, "global256", "global512", "full-bf16"])
def test_dry_run_keeps_compiled_profile_and_correct_mounts(tmp_path, head):
    state = tmp_path / "untouched"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/coherence"),
            "serve",
            "--dry-run",
            "--state",
            str(state),
            "--model",
            "/models/a space/target",
            "--draft",
            "/models/draft",
            *(["--head", head] if head else []),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    import shlex

    command = shlex.split(result.stdout)
    assert "--enforce-eager" not in command
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert "TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1" in command
    assert f"RADIANCE_VERIFY_HEAD={0 if head == 'full-bf16' else 1}" in command
    assert (
        f"RADIANCE_VERIFY_HEAD_GLOBAL_TOPK={256 if head == 'global256' else 512}"
        in command
    )
    assert "/models/a space/target:/models/target:ro" in command
    assert "--privileged" not in command
    assert not state.exists()
    assert json.loads(command[command.index("--compilation-config") + 1])[
        "cudagraph_capture_sizes"
    ] == [1, 2, 4, 8]
    assert (
        json.loads(command[command.index("--additional-config") + 1])["qwen_fair"][
            "tool_grace_seconds"
        ]
        == 2
    )


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../escape", "file"),
        ("/tmp/escape", "file"),
        ("link", "symlink"),
        ("hard", "hardlink"),
    ],
)
def test_release_archive_rejects_path_and_link_attacks(tmp_path, name, kind):
    archive = tmp_path / "bad.tar.xz"
    with tarfile.open(archive, "w:xz") as out:
        member = tarfile.TarInfo(name)
        if kind == "symlink":
            member.type = tarfile.SYMTYPE
        if kind == "hardlink":
            member.type = tarfile.LNKTYPE
        member.linkname = "/etc/passwd"
        out.addfile(member, io.BytesIO())
    with pytest.raises(ValueError):
        cli.extract_archive(archive, tmp_path / "out")
    assert not (tmp_path / "escape").exists()


def test_payload_tampering_and_empty_evidence_are_rejected(tmp_path):
    file = tmp_path / "kernel.py"
    file.write_text("correct = True\n")
    manifest = {
        "schema": "urn:qwen:optimized-pi-release:v1",
        "files": {"kernel.py": cli.digest(file)},
    }
    path = tmp_path / "optimized-release.json"
    path.write_text(json.dumps(manifest))
    assert cli.verify_payload(tmp_path) == manifest
    file.write_text("correct = False\n")
    with pytest.raises(ValueError, match="changed"):
        cli.verify_payload(tmp_path)
    manifest["files"] = {}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="empty"):
        cli.verify_payload(tmp_path)


def test_symlink_ancestor_cannot_substitute_runtime(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "kernel.py").write_text("ok\n")
    (tmp_path / "link").symlink_to(real, target_is_directory=True)
    manifest = {
        "schema": "urn:qwen:optimized-pi-release:v1",
        "files": {"link/kernel.py": cli.digest(real / "kernel.py")},
    }
    (tmp_path / "optimized-release.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="indirect"):
        cli.verify_payload(tmp_path)


def test_remote_command_quotes_state_and_rejects_option_injection(monkeypatch):
    import coherence_pi as pi

    with pytest.raises(ValueError):
        pi.remote_connection("-ProxyCommand=anything", "state")
    seen = []

    def run(command, **kwargs):
        seen.append(command)
        return SimpleNamespace(stdout='{"port":8080}')

    monkeypatch.setattr(pi.subprocess, "run", run)
    value = pi.remote_connection("gpu-host", "~/a folder/$(echo bad)")
    assert value["port"] == 8080
    import shlex

    assert shlex.split(seen[0][-1])[-1] == "~/a folder/$(echo bad)"


def test_port_locks_allocate_distinct_ports_and_release(tmp_path):
    import coherence_pi as pi

    p1, l1 = pi.reserve_port(tmp_path)
    try:
        p2, l2 = pi.reserve_port(tmp_path)
        try:
            assert p1 != p2
        finally:
            l2.close()
    finally:
        l1.close()
    p3, l3 = pi.reserve_port(tmp_path)
    try:
        assert p3 == p1
    finally:
        l3.close()


def test_preparation_cannot_replace_a_live_state(tmp_path):
    state = tmp_path / "deployment"
    with cli.deployment_lock(state):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/coherence"),
                "prepare",
                "--state",
                str(state),
                "--archive",
                str(tmp_path / "absent.tar.xz"),
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        assert result.returncode == 2
        assert "serving or being prepared" in result.stderr
        assert not (state / "patches").exists()
        assert not (state / "runtime").exists()
    with cli.deployment_lock(state):
        pass


def test_full_head_dry_run_disables_approximate_target_selection(tmp_path):
    import shlex

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/coherence"),
            "serve",
            "--dry-run",
            "--head",
            "full-bf16",
            "--state",
            str(tmp_path / "unused"),
            "--model",
            "/m",
            "--draft",
            "/d",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout)
    assert "RADIANCE_VERIFY_HEAD=0" in command
    assert "RADIANCE_VERIFY_HEAD=1" not in command


def test_release_integration_manifest_covers_current_source():
    spec = cli.release()
    assert set(spec["patch_files"]) == set(spec["integration_sha256"])
    assert all(
        cli.digest(ROOT / name) == expected
        for name, expected in spec["integration_sha256"].items()
    )


def test_support_library_integrity_is_checked_on_reuse(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    (runtime / "support").mkdir(parents=True)
    library = runtime / "support/cpu.so"
    library.write_bytes(b"expected")
    expected = cli.digest(library)
    monkeypatch.setattr(cli, "verify_payload", lambda *_: {"files": {"code.py": "x"}})
    monkeypatch.setattr(
        cli,
        "release",
        lambda: {
            "manifest_sha256": "manifest",
            "support_files": {"support/cpu.so": expected},
        },
    )
    assert cli.verify_runtime(tmp_path)["files"]
    library.write_bytes(b"changed")
    with pytest.raises(ValueError, match="support library changed"):
        cli.verify_runtime(tmp_path)

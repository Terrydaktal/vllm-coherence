from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/pi-build-manifest"
loader = importlib.machinery.SourceFileLoader("pi_build_manifest", str(SCRIPT))
spec = importlib.util.spec_from_loader(loader.name, loader)
manifest = importlib.util.module_from_spec(spec)
loader.exec_module(manifest)


@pytest.fixture
def installation(tmp_path):
    runtime = tmp_path / "runtime"
    patch_root = tmp_path / "source"
    package = runtime / "node_modules/@earendil-works/pi-coding-agent"
    (package / "dist").mkdir(parents=True)
    (package / "package.json").write_text(json.dumps({"name": manifest.PACKAGE, "version": "0.84.2"}))
    (package / "dist/cli.js").write_text("export const fixture = true;\n")
    (patch_root / "scripts").mkdir(parents=True)
    for name in manifest.PATCHERS:
        (patch_root / "scripts" / ("patch-pi-" + name)).write_text(
            "import pathlib, sys\n"
            "assert sys.argv[1] == '--check'\n"
            "runtime = pathlib.Path(sys.argv[2])\n"
            "with (runtime / 'checks.log').open('a') as log:\n"
            f"    log.write({name!r} + '\\n')\n")
    return runtime, patch_root, package


def run(installation, *arguments):
    runtime, patch_root, _ = installation
    return subprocess.run([sys.executable, str(SCRIPT), *arguments,
                           "--runtime", str(runtime), "--patch-root", str(patch_root)],
                          capture_output=True, text=True, timeout=10)


def test_apply_records_exact_fingerprint_and_six_verified_patch_sources(installation):
    runtime, patch_root, package = installation
    assert run(installation, "--apply").returncode == 0
    path = package / "coherence-build.json"
    value = json.loads(path.read_bytes())
    assert value["schema"] == "coherence.pi.installed-build.v1"
    assert value["package"] == manifest.PACKAGE
    assert value["entry_point"] == "dist/cli.js"
    assert value["version"] == "0.84.2"
    assert value["patches_verified"] is True
    assert path.stat().st_mode & 0o777 == 0o600
    assert (runtime / "checks.log").read_text().splitlines() == list(manifest.PATCHERS)
    assert value["sdk_files"] == [{"path": "pi-coding-agent/dist/cli.js",
                                   "sha256": hashlib.sha256((package / "dist/cli.js").read_bytes()).hexdigest()}]
    entry = value["sdk_files"][0]
    assert value["sdk_fingerprint"] == hashlib.sha256(f'{entry["path"]}\0{entry["sha256"]}\n'.encode()).hexdigest()
    assert len(value["patches"]) == 6
    for entry, name in zip(value["patches"], manifest.PATCHERS):
        actual = patch_root / "scripts" / ("patch-pi-" + name)
        assert (package / entry["path"]).resolve() == actual
        assert entry["sha256"] == hashlib.sha256(actual.read_bytes()).hexdigest()
    before = path.read_bytes()
    assert run(installation, "--apply").returncode == 0
    assert path.read_bytes() == before


def test_check_is_read_only_and_detects_sdk_or_patcher_changes(installation):
    runtime, patch_root, package = installation
    assert run(installation, "--apply").returncode == 0
    path = package / "coherence-build.json"
    before = path.read_bytes()
    checks = (runtime / "checks.log").read_bytes()
    assert run(installation, "--check").returncode == 0
    assert (runtime / "checks.log").read_bytes() == checks
    target = package / "dist/cli.js"
    original = target.read_bytes()
    target.write_bytes(original + b"// updated\n")
    assert run(installation, "--check").returncode == 2
    assert path.read_bytes() == before
    target.write_bytes(original)
    assert run(installation, "--check").returncode == 0
    target = patch_root / "scripts/patch-pi-footer"
    target.write_text(target.read_text() + "# updated patch source\n")
    assert run(installation, "--check").returncode == 2
    assert path.read_bytes() == before


def test_failed_patch_check_never_publishes_or_overwrites_manifest(installation):
    _, patch_root, package = installation
    assert run(installation, "--apply").returncode == 0
    before = (package / "coherence-build.json").read_bytes()
    (patch_root / "scripts/patch-pi-footer").write_text("raise SystemExit(1)\n")
    assert run(installation, "--apply").returncode == 2
    assert (package / "coherence-build.json").read_bytes() == before


def test_inventory_allows_internal_npm_alias_but_ignores_generated_or_private_files(installation):
    runtime, _, package = installation
    nested = package / "node_modules/@earendil-works/pi-tui/dist"
    nested.mkdir(parents=True)
    (nested / "terminal.js").write_text("export const terminal = 1;\n")
    alias = runtime / "node_modules/@earendil-works/pi-tui"
    alias.symlink_to("pi-coding-agent/node_modules/@earendil-works/pi-tui", target_is_directory=True)
    for name in manifest.SKIP:
        ignored = package / "dist" / name
        ignored.mkdir()
        (ignored / "private.js").write_text("must not be read")
    (package / "dist/not-code.map").write_text("must not be read")
    files, _ = manifest.sdk_inventory(runtime)
    assert [entry["path"] for entry in files] == ["pi-coding-agent/dist/cli.js", "pi-tui/dist/terminal.js"]
    alias.unlink()
    alias.symlink_to(package.parent.parent.parent.parent.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the runtime"):
        manifest.sdk_inventory(runtime)


def test_inventory_refuses_escape_links_limits_and_incomplete_cli(installation, monkeypatch):
    runtime, _, package = installation
    target = package / "dist/extra.js"
    target.symlink_to("cli.js")
    with pytest.raises(ValueError, match="symlink inside SDK"):
        manifest.sdk_inventory(runtime)
    target.unlink()
    with monkeypatch.context() as context:
        context.setattr(manifest, "MAX_FILES", 0)
        with pytest.raises(ValueError, match="budget exceeded"):
            manifest.sdk_inventory(runtime)
    with monkeypatch.context() as context:
        context.setattr(manifest, "MAX_DIRECTORIES", 0)
        with pytest.raises(ValueError, match="directory budget exceeded"):
            manifest.sdk_inventory(runtime)
    with monkeypatch.context() as context:
        context.setattr(manifest, "MAX_BYTES", 1)
        with pytest.raises(ValueError, match="budget exceeded"):
            manifest.sdk_inventory(runtime)
    (package / "dist/cli.js").unlink()
    with pytest.raises(ValueError, match="CLI is missing"):
        manifest.sdk_inventory(runtime)


def test_publication_refuses_symlink_and_shared_writable_package(installation):
    _, _, package = installation
    victim = package / "unrelated.txt"
    victim.write_text("unchanged")
    (package / "coherence-build.json").symlink_to(victim)
    assert run(installation, "--apply").returncode == 2
    assert victim.read_text() == "unchanged"
    (package / "coherence-build.json").unlink()
    package.chmod(0o777)
    assert run(installation, "--apply").returncode == 2
    assert not (package / "coherence-build.json").exists()


def test_patch_change_during_check_does_not_gain_verified_manifest(installation):
    _, patch_root, package = installation
    patch = patch_root / "scripts/patch-pi-project-history"
    patch.write_text("from pathlib import Path\np = Path(__file__)\np.write_text(p.read_text() + '# changed\\n')\n")
    result = run(installation, "--apply")
    assert result.returncode == 2
    assert "changed during verification" in result.stderr
    assert not (package / "coherence-build.json").exists()


def test_utf8_inventory_sort_and_digest_are_cross_language_deterministic(installation):
    runtime, _, package = installation
    for name in ["é.js", "Z.js", "a.js"]:
        (package / "dist" / name).write_text("// code\n")
    files, identity = manifest.sdk_inventory(runtime)
    assert [entry["path"] for entry in files] == sorted([entry["path"] for entry in files], key=lambda value: value.encode())
    assert identity == hashlib.sha256(b"".join(entry["path"].encode() + b"\0" + entry["sha256"].encode() + b"\n" for entry in files)).hexdigest()

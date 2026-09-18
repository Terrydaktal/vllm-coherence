from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "experiments/radiance-public/patch_gdn_extreme_decay.py"
spec = importlib.util.spec_from_file_location("gdn_numerical_repair", SOURCE)
repair = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(repair)


def fixture_files(tmp_path, monkeypatch):
    source = tmp_path / "radiance_gdn.py"
    source.write_text("_CHUNK_SCAN = None\n")
    monkeypatch.setattr(repair, "PREIMAGE", hashlib.sha256(source.read_bytes()).hexdigest())
    library = tmp_path / "gdn_extreme_decay_reference.so"
    library.write_bytes(b"test-only library identity")
    return source, library, hashlib.sha256(library.read_bytes()).hexdigest()


def test_reinstall_does_not_double_wrap_the_scan(tmp_path, monkeypatch):
    source, library, digest = fixture_files(tmp_path, monkeypatch)
    first = repair.install(tmp_path, library, digest)
    installed = source.read_bytes()
    assert repair.install(tmp_path, library, digest) == first
    assert source.read_bytes() == installed
    assert source.read_text().count("def _qwen_gdn_corrected_scan(") == 1


def test_changed_backend_source_is_preserved_and_rejected(tmp_path, monkeypatch):
    source, library, digest = fixture_files(tmp_path, monkeypatch)
    source.write_text(source.read_text() + "# a different upstream revision\n")
    original = source.read_bytes()
    with pytest.raises(ValueError, match="preimage"):
        repair.install(tmp_path, library, digest)
    assert source.read_bytes() == original


def test_changed_library_is_rejected_before_source_modification(tmp_path, monkeypatch):
    source, library, digest = fixture_files(tmp_path, monkeypatch)
    original = source.read_bytes()
    library.write_bytes(b"unexpected binary")
    with pytest.raises(ValueError, match="library hash"):
        repair.install(tmp_path, library, digest)
    assert source.read_bytes() == original


def test_failed_build_keeps_the_previous_artifact(tmp_path, monkeypatch):
    source = tmp_path / "gdn_extreme_decay_reference.hip"
    source.write_text("// fixture input for the build publication contract\n")
    monkeypatch.setattr(repair, "SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    output = tmp_path / "previous.so"
    output.write_bytes(b"previous artifact")

    def mismatched_compiler(command, **kwargs):
        Path(command[-1]).write_bytes(b"unqualified output")

    monkeypatch.setattr(repair.subprocess, "run", mismatched_compiler)
    with pytest.raises(ValueError, match="qualified binary"):
        repair.build(source, output)
    assert output.read_bytes() == b"previous artifact"
    assert not list(tmp_path.glob("qwen-gdn-build-*"))

"""CPU tests for cleanup of explicitly owned test-only offload mappings."""

import mmap
import os

import pytest

from qwen_r9700_lab.conformance_shm import OwnedOffloadRegion
from qwen_r9700_lab.diagnostic_contract import DiagnosticError

ENGINE = "conformance-" + "a" * 64


def test_unlink_only_claimed_name_and_preserve_existing_mappings(tmp_path):
    unrelated = tmp_path / "vllm_offload_production.mmap"
    unrelated.write_bytes(b"keep")
    claim = OwnedOffloadRegion(ENGINE, directory=tmp_path)
    with claim.path.open("w+b") as file:
        file.write(b"x" * 4096)
        file.flush()
        with mmap.mmap(file.fileno(), 4096) as mapped:
            report = claim.release()
            assert report["status"] == "unlinked"
            assert report["bytes"] == 4096
            assert mapped[:4] == b"xxxx"
    assert not claim.path.exists()
    assert unrelated.read_bytes() == b"keep"


def test_absent_after_native_cleanup_is_success(tmp_path):
    claim = OwnedOffloadRegion(ENGINE, directory=tmp_path)
    assert claim.release()["status"] == "absent"
    with pytest.raises(DiagnosticError, match="already released"):
        claim.release()


@pytest.mark.parametrize(
    "engine", ["production", "conformance-../other", "conformance-a", "a" * 64, None]
)
def test_only_exact_conformance_namespace_is_admitted(tmp_path, engine):
    with pytest.raises(DiagnosticError, match="isolated conformance"):
        OwnedOffloadRegion(engine, directory=tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["regular", "dangling_symlink", "directory"])
def test_preexisting_name_is_never_claimed_or_deleted(tmp_path, kind):
    path = tmp_path / f"vllm_offload_{ENGINE}.mmap"
    if kind == "regular":
        path.write_bytes(b"keep")
    elif kind == "dangling_symlink":
        path.symlink_to(tmp_path / "missing")
    else:
        path.mkdir()
    with pytest.raises(DiagnosticError, match="already exists"):
        OwnedOffloadRegion(ENGINE, directory=tmp_path)
    assert os.path.lexists(path)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory", "foreign_owner"])
def test_changed_type_or_owner_is_preserved(tmp_path, monkeypatch, kind):
    claim = OwnedOffloadRegion(ENGINE, directory=tmp_path)
    other = tmp_path / "keep"
    other.write_bytes(b"keep")
    if kind == "symlink":
        claim.path.symlink_to(other)
    elif kind == "hardlink":
        claim.path.hardlink_to(other)
    elif kind == "directory":
        claim.path.mkdir()
    else:
        claim.path.write_bytes(b"keep")
        uid = os.getuid()
        monkeypatch.setattr(os, "getuid", lambda: uid + 1)
    with pytest.raises(DiagnosticError, match="type or ownership changed"):
        claim.release()
    assert claim.path.exists()
    assert other.read_bytes() == b"keep"


def test_parent_directory_symlink_is_refused(tmp_path):
    directory = tmp_path / "real"
    directory.mkdir()
    link = tmp_path / "link"
    link.symlink_to(directory, target_is_directory=True)
    with pytest.raises(OSError):
        OwnedOffloadRegion(ENGINE, directory=link)

"""Independent evidence clones must authenticate data and fail without fallback."""

import errno
import os

import numpy as np
import pytest
from test_conformance_state import frame

from qwen_r9700_lab.conformance_state import archive_frame, compare_frames
from qwen_r9700_lab.diagnostic_contract import DiagnosticError


def simulated_clone(destination_fd, request, source_fd):
    """Model only the copy operation, never its integrity or publication checks."""
    assert request == 0x40049409
    offset = 0
    while chunk := os.pread(source_fd, 1024 * 1024, offset):
        assert os.pwrite(destination_fd, chunk, offset) == len(chunk)
        offset += len(chunk)


def make_source(root):
    return frame(root, {"a": np.arange(1024, dtype=np.int64), "b": np.zeros(17, np.float32)})


def check_isolation(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "clone"
    original = make_source(source)
    result = archive_frame(source, destination, reflink=True)
    assert result == original
    assert compare_frames(source, destination)["equal"]
    for component in original["components"].values():
        a, b = source / component["file"], destination / component["file"]
        payload = a.read_bytes()
        assert a.stat().st_ino != b.stat().st_ino
        assert a.stat().st_nlink == b.stat().st_nlink == 1
        assert a.stat().st_mode & 0o077 == b.stat().st_mode & 0o077 == 0
        with b.open("r+b") as handle:
            handle.write(b"x")
        assert a.read_bytes() == payload
        clone_payload = b.read_bytes()
        with a.open("r+b") as handle:
            handle.write(b"y")
        assert b.read_bytes() == clone_payload
        a.unlink()
        assert b.read_bytes() == clone_payload


def test_clone_bytes_metadata_and_independent_lifetime(tmp_path, monkeypatch):
    monkeypatch.setattr("qwen_r9700_lab.conformance_state.fcntl.ioctl", simulated_clone)
    check_isolation(tmp_path)


@pytest.mark.skipif(
    os.environ.get("QWEN_REQUIRE_REFLINK") != "1",
    reason="Run with QWEN_REQUIRE_REFLINK=1 on the GPU host's evidence filesystem (CPU only)",
)
def test_real_filesystem_clone_is_independent(tmp_path):
    # No feature-detection skip: unsupported FICLONE is a failure when requested.
    check_isolation(tmp_path)


@pytest.mark.parametrize("error", [errno.EIO, errno.ENOSPC, errno.EXDEV, errno.EOPNOTSUPP])
def test_failed_clone_never_falls_back_or_publishes(tmp_path, monkeypatch, error):
    make_source(tmp_path / "source")

    def fail(*_):
        raise OSError(error, "injected clone failure")

    monkeypatch.setattr("qwen_r9700_lab.conformance_state.fcntl.ioctl", fail)
    with pytest.raises(OSError) as caught:
        archive_frame(tmp_path / "source", tmp_path / "clone", reflink=True)
    assert caught.value.errno == error
    assert not (tmp_path / "clone/frame.json").exists()
    assert all(p.stat().st_size == 0 for p in (tmp_path / "clone").iterdir())


@pytest.mark.parametrize("fault", ["empty", "wrong_bytes", "truncated", "overlong"])
def test_ioctl_success_cannot_hide_a_bad_destination(tmp_path, monkeypatch, fault):
    make_source(tmp_path / "source")

    def bad_clone(destination_fd, request, source_fd):
        if fault != "empty":
            simulated_clone(destination_fd, request, source_fd)
        if fault == "wrong_bytes":
            os.pwrite(destination_fd, b"x", 0)
        elif fault == "truncated":
            os.ftruncate(destination_fd, 1)
        elif fault == "overlong":
            os.pwrite(destination_fd, b"x", os.fstat(source_fd).st_size)

    monkeypatch.setattr("qwen_r9700_lab.conformance_state.fcntl.ioctl", bad_clone)
    with pytest.raises(DiagnosticError, match="payload changed"):
        archive_frame(tmp_path / "source", tmp_path / "clone", reflink=True)
    assert not (tmp_path / "clone/frame.json").exists()


def test_changed_source_during_clone_is_refused(tmp_path, monkeypatch):
    doc = make_source(tmp_path / "source")
    source = tmp_path / "source" / doc["components"]["a"]["file"]

    def change_source(*args):
        simulated_clone(*args)
        with source.open("r+b") as handle:
            handle.write(b"changed")

    monkeypatch.setattr("qwen_r9700_lab.conformance_state.fcntl.ioctl", change_source)
    with pytest.raises(DiagnosticError, match="frame changed during archival"):
        archive_frame(tmp_path / "source", tmp_path / "clone", reflink=True)
    assert not (tmp_path / "clone/frame.json").exists()


def test_corrupt_source_never_publishes_clone(tmp_path, monkeypatch):
    doc = make_source(tmp_path / "source")
    source = tmp_path / "source" / doc["components"]["a"]["file"]
    source.write_bytes(b"corruption")
    monkeypatch.setattr("qwen_r9700_lab.conformance_state.fcntl.ioctl", simulated_clone)
    with pytest.raises(DiagnosticError, match="payload changed"):
        archive_frame(tmp_path / "source", tmp_path / "clone", reflink=True)
    assert not (tmp_path / "clone/frame.json").exists()


def test_failed_sync_does_not_publish(tmp_path, monkeypatch):
    make_source(tmp_path / "source")
    monkeypatch.setattr("qwen_r9700_lab.conformance_state.fcntl.ioctl", simulated_clone)

    def fail(_):
        raise OSError(errno.EIO, "injected fsync failure")

    monkeypatch.setattr("qwen_r9700_lab.conformance_state.os.fsync", fail)
    with pytest.raises(OSError, match="fsync failure"):
        archive_frame(tmp_path / "source", tmp_path / "clone", reflink=True)
    assert not (tmp_path / "clone/frame.json").exists()


def test_existing_destination_is_not_replaced(tmp_path, monkeypatch):
    make_source(tmp_path / "source")
    destination = tmp_path / "clone"
    destination.mkdir(mode=0o700)
    sentinel = destination / "existing"
    sentinel.write_bytes(b"keep")
    monkeypatch.setattr("qwen_r9700_lab.conformance_state.fcntl.ioctl", simulated_clone)
    with pytest.raises(FileExistsError):
        archive_frame(tmp_path / "source", destination, reflink=True)
    assert sentinel.read_bytes() == b"keep"
    assert list(destination.iterdir()) == [sentinel]


def test_default_archive_does_not_request_a_reflink(tmp_path, monkeypatch):
    make_source(tmp_path / "source")

    def unexpected(*_):
        pytest.fail("default archive must remain an ordinary independent copy")

    monkeypatch.setattr("qwen_r9700_lab.conformance_state.fcntl.ioctl", unexpected)
    archive_frame(tmp_path / "source", tmp_path / "copy")
    assert compare_frames(tmp_path / "source", tmp_path / "copy")["equal"]

"""KFD capture never exceeds its scope, byte budget or lifecycle bounds."""

import json
import os
import struct
import time

import pytest

from qwen_r9700_lab import radiance_kfd_trace as trace


@pytest.mark.parametrize(
    "seconds,size", [(0, 65536), (301, 65536), (1, 65535), (1, trace.MAX_BYTES + 1)]
)
def test_rejects_unapproved_bounds(tmp_path, seconds, size):
    with pytest.raises(ValueError):
        trace.Capture(tmp_path / "events", "id", seconds=seconds, max_bytes=size)


def test_no_trace_without_explicit_path(monkeypatch):
    monkeypatch.delenv("QWEN_KFD_CAPTURE_PATH", raising=False)
    assert trace.start_if_approved("id") is None


def test_event_subscription_is_process_specific():
    assert trace.EVENT_MASK == 0x7F0
    assert trace.EVENT_MASK & (1 << 63) == 0
    assert trace.SMI_IOCTL == 0xC0084B1F


def capture_pipe(tmp_path, monkeypatch, *, seconds=1, max_bytes=trace.MAX_BYTES):
    tmp_path.chmod(0o700)
    read, write = os.pipe()
    os.set_blocking(read, False)
    monkeypatch.setattr(trace, "gpu_ids", lambda: [51639])
    monkeypatch.setattr(trace, "open_events", lambda node: read)
    item = trace.Capture(
        tmp_path / "events.kfd", "test-id", seconds=seconds, max_bytes=max_bytes
    ).start()
    return item, read, write


def test_retains_raw_events_and_seals_identity_manifest(tmp_path, monkeypatch):
    item, read, write = capture_pipe(tmp_path, monkeypatch)
    event = b"9 100000 -42 51639 1\na 120000 -42 51639 R\n"
    os.write(write, event)
    deadline = time.monotonic() + 1
    while item.bytes < len(event) and time.monotonic() < deadline:
        time.sleep(0.005)
    item.close()
    os.close(write)
    assert item.state == "stopped"
    assert item.path.read_bytes() == event
    assert item.path.stat().st_mode & 0o777 == 0o400
    info = json.loads(item.path.with_name(item.path.name + ".json").read_text())
    assert info["trace_id"] == "test-id"
    assert info["pid"] == os.getpid()
    assert info["all_process_events"] is False
    assert info["end_clock"]["monotonic_ns"] >= info["start_clock"]["monotonic_ns"]
    with pytest.raises(OSError):
        os.fstat(read)


def test_size_limit_discards_whole_over_budget_read(tmp_path, monkeypatch):
    item, _, write = capture_pipe(tmp_path, monkeypatch, max_bytes=65540)
    os.write(write, b"over budget\n")
    item.thread.join(timeout=2)
    os.close(write)
    assert item.state == "size_limit"
    assert item.bytes == 0
    assert sum(p.stat().st_size for p in tmp_path.iterdir()) <= item.max_bytes


def test_idle_capture_expires_without_external_stop(tmp_path, monkeypatch):
    item, _, write = capture_pipe(tmp_path, monkeypatch, seconds=0.03)
    item.thread.join(timeout=1)
    os.close(write)
    assert item.state == "duration_limit"
    assert not item.thread.is_alive()


def test_existing_file_is_preserved(tmp_path, monkeypatch):
    tmp_path.chmod(0o700)
    path = tmp_path / "events.kfd"
    path.write_bytes(b"preserved")
    monkeypatch.setattr(trace, "open_events", lambda node: pytest.fail("opened KFD"))
    item = trace.Capture(path, "test-id").start()
    item.thread.join(timeout=1)
    assert item.state == "failed"
    assert path.read_bytes() == b"preserved"


def test_unowned_public_directory_is_rejected(tmp_path, monkeypatch):
    tmp_path.chmod(0o755)
    monkeypatch.setattr(trace, "open_events", lambda node: pytest.fail("opened KFD"))
    item = trace.Capture(tmp_path / "events.kfd", "test-id").start()
    item.thread.join(timeout=1)
    assert item.state == "failed"
    assert not item.path.exists()


def test_ioctl_subscribes_and_cleans_device_fd(monkeypatch):
    read, write = os.pipe()
    monkeypatch.setattr(
        trace, "open", lambda *a, **kw: os.fdopen(os.dup(write), "wb"), raising=False
    )
    seen = []

    def ioctl(fd, command, args, mutate):
        seen.append((command, struct.unpack("=II", args)[0], mutate))
        args[:] = struct.pack("=II", 51639, os.dup(write))

    monkeypatch.setattr(trace.fcntl, "ioctl", ioctl)
    event_fd = trace.open_events(51639)
    assert os.read(read, 8) == struct.pack("=Q", trace.EVENT_MASK)
    assert seen == [(0xC0084B1F, 51639, True)]
    assert not os.get_inheritable(event_fd)
    os.close(event_fd)
    os.close(read)
    os.close(write)

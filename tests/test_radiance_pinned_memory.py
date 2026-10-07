from __future__ import annotations

import ctypes
import errno
import gc
import mmap
from itertools import pairwise
from types import SimpleNamespace

import pytest

from qwen_r9700_lab import radiance_pinned_memory as module


@pytest.mark.parametrize("size,expected", [
    (1, mmap.PAGESIZE), (mmap.PAGESIZE, mmap.PAGESIZE),
    (mmap.PAGESIZE + 1, 2 * mmap.PAGESIZE),
    (9_990_832_128, 9_990_832_128),
])
def test_backing_size_is_page_aligned_without_power_of_two_rounding(size, expected):
    assert module.backing_bytes(size) == expected


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_size_rejected_before_allocating(size):
    with pytest.raises(ValueError):
        module.backing_bytes(size)


class Tensor:
    def __init__(self, buffer, count, pinned=True):
        self.buffer = memoryview(buffer).cast("B")[:count]
        self.pinned = pinned

    def is_pinned(self):
        return self.pinned

    def view(self):
        result = object.__new__(Tensor)
        result.buffer = self.buffer[1:]
        result.pinned = self.pinned
        return result


@pytest.fixture
def native_mapping(monkeypatch):
    calls = []
    runtime_cdll = ctypes.CDLL
    failures = {"register": 0, "unregister": 0, "policy": False}

    def register(address, size, flags):
        calls.append(("register", address, size, flags))
        assert ctypes.string_at(address, size) == bytes(size)
        return failures["register"]

    def unregister(address):
        calls.append(("unregister", address))
        return failures["unregister"]

    def cdll(name, **kwargs):
        if name == "libamdhip64.so":
            return SimpleNamespace(hipHostRegister=register, hipHostUnregister=unregister)
        libc = runtime_cdll(name, **kwargs)
        real_advise, real_unmap, real_set = libc.madvise, libc.munmap, libc.memset
        real_advise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        real_unmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        real_set.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t)
        def advise(address, size, flag):
            calls.append(("policy", address, size, flag))
            if failures["policy"]:
                ctypes.set_errno(errno.ENOMEM)
                return -1
            return real_advise(address, size, flag)
        def unmap(address, size):
            calls.append(("unmap", address, size))
            return real_unmap(address, size)
        def touch(address, value, size):
            calls.append(("touch", address, size))
            return real_set(address, value, size)
        libc.madvise, libc.munmap, libc.memset = advise, unmap, touch
        return libc

    monkeypatch.setattr(module.ctypes, "CDLL", cdll)
    torch = SimpleNamespace(uint8=object(), frombuffer=lambda buf, **kw: Tensor(buf, kw["count"]))
    return SimpleNamespace(torch=torch, calls=calls, failures=failures)


def test_no_hugepage_policy_precedes_touch_and_registration(native_mapping):
    f = native_mapping
    tensor, (begin, end) = module.allocate_pinned_bytes(f.torch, mmap.PAGESIZE + 1)
    assert end - begin == 2 * mmap.PAGESIZE
    assert len(tensor.buffer) == mmap.PAGESIZE + 1
    assert [row[0] for row in f.calls] == ["policy", "touch", "register"]
    assert f.calls[0][3] == mmap.MADV_NOHUGEPAGE
    tensor.buffer[0] = 17
    tensor.buffer[-1] = 231
    assert ctypes.string_at(begin, 1) == b"\x11"
    assert ctypes.string_at(begin + mmap.PAGESIZE, 1) == b"\xe7"
    del tensor
    gc.collect()
    assert [row[0] for row in f.calls][-2:] == ["unregister", "unmap"]


def test_storage_view_retains_registration_after_original_tensor_is_deleted(native_mapping):
    f = native_mapping
    tensor, _ = module.allocate_pinned_bytes(f.torch, 4096)
    view = tensor.view()
    del tensor
    gc.collect()
    assert not any(row[0] == "unregister" for row in f.calls)
    view.buffer[0] = 99
    assert view.buffer[0] == 99
    del view
    gc.collect()
    assert [row[0] for row in f.calls][-2:] == ["unregister", "unmap"]


@pytest.mark.parametrize("failure", ["policy", "register", "tensor", "not_pinned"])
def test_failures_release_mapping_without_publishing_buffer(native_mapping, failure):
    f = native_mapping
    if failure == "policy":
        f.failures["policy"] = True
    elif failure == "register":
        f.failures["register"] = 2
    elif failure == "tensor":
        def fail(*args, **kwargs):
            raise ValueError("tensor construction failed")
        f.torch.frombuffer = fail
    else:
        f.torch.frombuffer = lambda buf, **kw: Tensor(buf, kw["count"], pinned=False)
    with pytest.raises((OSError, RuntimeError, ValueError)):
        module.allocate_pinned_bytes(f.torch, 8192)
    gc.collect()
    kinds = [row[0] for row in f.calls]
    assert kinds.count("unmap") == 1
    assert kinds.count("unregister") == (1 if failure in {"tensor", "not_pinned"} else 0)


def test_parallel_prefault_partitions_cover_every_byte_once(monkeypatch):
    calls = []
    libc = SimpleNamespace(memset=lambda address, value, size: calls.append((address, size)))
    mapping = SimpleNamespace(bytes=16 * 1024**2 + mmap.PAGESIZE, address=4096, libc=libc)
    module._prefault(mapping, 8)
    ranges = sorted((begin, begin + count) for begin, count in calls)
    assert len(ranges) == 8
    assert ranges[0][0] == mapping.address
    assert ranges[-1][1] == mapping.address + mapping.bytes
    assert all(end == next_begin for (_, end), (next_begin, _) in pairwise(ranges))
    assert all(begin % mmap.PAGESIZE == end % mmap.PAGESIZE == 0 for begin, end in ranges)


def test_unregister_failure_never_unmaps_registered_memory(native_mapping, caplog):
    f = native_mapping
    tensor, _ = module.allocate_pinned_bytes(f.torch, 4096)
    owner = tensor.buffer.obj._qwen_registration
    f.failures["unregister"] = 999
    owner.close()
    assert owner.registered and owner.address
    assert not any(row[0] == "unmap" for row in f.calls)
    assert "mapping retained" in caplog.text
    f.failures["unregister"] = 0
    owner.close()

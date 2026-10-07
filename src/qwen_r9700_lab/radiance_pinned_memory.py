"""Exact-size ROCm host buffers, with ownership attached to their tensor storage.

The ordinary PyTorch pinned allocator rounds large buffers to a power of two.
These long-lived chat banks instead register one page-aligned anonymous mapping.
The ctypes buffer retained by torch.frombuffer owns the registration, including
when only a tensor view remains. Callers must retain buffers until their existing
DMA completion fence; this allocator deliberately introduces no device fence.
"""

from __future__ import annotations

import ctypes
import logging
import mmap
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path


def backing_bytes(size: int) -> int:
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError("pinned buffer size must be a positive integer")
    page = mmap.PAGESIZE
    return (size + page - 1) // page * page


class _Mapping:
    def __init__(self, size, libc, hip):
        self.bytes = backing_bytes(size)
        self.libc = libc
        self.hip = hip
        self.address = None
        self.registered = False

    def close(self):
        if self.address is None:
            return
        if self.registered:
            code = self.hip.hipHostUnregister(self.address)
            if code:
                # A registration still held by the driver must never be unmapped.
                # Keep it until process exit rather than turn cleanup into UAF.
                logging.getLogger(__name__).error(
                    "Cannot unregister chat RAM (HIP code %s); mapping retained", code
                )
                return
            self.registered = False
        if self.libc.munmap(self.address, self.bytes):
            logging.getLogger(__name__).error("Cannot release chat RAM mapping")
            return
        self.address = None

    def __del__(self):
        self.close()


def _runtime():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.argtypes = (
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_long,
    )
    libc.mmap.restype = ctypes.c_void_p
    libc.munmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    libc.munmap.restype = ctypes.c_int
    libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    libc.madvise.restype = ctypes.c_int
    libc.memset.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t)
    libc.memset.restype = ctypes.c_void_p
    try:
        hip = ctypes.CDLL("libamdhip64.so")
    except OSError as error:
        # Runtime-only images can ship the versioned library without its SDK
        # symlink. Reuse the unique HIP runtime already loaded by PyTorch;
        # never load a different ROCm installation into the worker.
        paths = {
            fields[5]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if len(fields := line.split(maxsplit=5)) == 6
            and Path(fields[5]).name.startswith("libamdhip64.so.")
            and not fields[5].endswith(" (deleted)")
        }
        if len(paths) != 1:
            raise RuntimeError("cannot identify the unique loaded HIP runtime") from error
        hip = ctypes.CDLL(paths.pop())
    hip.hipHostRegister.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint)
    hip.hipHostRegister.restype = ctypes.c_int
    hip.hipHostUnregister.argtypes = (ctypes.c_void_p,)
    hip.hipHostUnregister.restype = ctypes.c_int
    return libc, hip


def _prefault(mapping, threads):
    # ctypes releases the GIL around memset. Contiguous, page-aligned partitions
    # touch every page once without a millions-of-iterations Python loop.
    pages = mapping.bytes // mmap.PAGESIZE
    count = min(threads, pages)
    if mapping.bytes < 16 * 1024**2:
        count = 1

    def touch(index):
        begin = pages * index // count * mmap.PAGESIZE
        end = pages * (index + 1) // count * mmap.PAGESIZE
        mapping.libc.memset(mapping.address + begin, 0, end - begin)

    if count == 1:
        touch(0)
    else:
        with ThreadPoolExecutor(max_workers=count, thread_name_prefix="chat-ram") as pool:
            list(pool.map(touch, range(count)))


def allocate_pinned_bytes(torch, size, *, threads=8, span=None):
    """Return a uint8 tensor and its exact (begin, end) registered mapping.

    Huge-page promotion is disabled *before* prefaulting and registration. All
    failure paths release completed allocations. No rounded allocator fallback
    is allowed: otherwise the headroom check would become inaccurate again.
    """
    if not isinstance(threads, int) or isinstance(threads, bool) or not 1 <= threads <= 32:
        raise ValueError("pinned prefault threads must be in 1..32")
    size_bytes = backing_bytes(size)
    stage = span or (lambda *args, **kwargs: nullcontext())
    libc, hip = _runtime()
    owner = _Mapping(size, libc, hip)
    try:
        address = libc.mmap(
            None, size_bytes, mmap.PROT_READ | mmap.PROT_WRITE,
            mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1, 0,
        )
        if address in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), "cannot map pinned chat RAM")
        owner.address = address
        with stage("pinned_page_policy", bytes=size_bytes, resources=True):
            if libc.madvise(address, size_bytes, mmap.MADV_NOHUGEPAGE):
                error = ctypes.get_errno()
                raise OSError(error, "cannot disable huge-page promotion for pinned chat RAM: " + os.strerror(error))
        with stage("pinned_prefault", bytes=size_bytes, resources=True):
            _prefault(owner, threads)
        with stage("pinned_registration", bytes=size_bytes, resources=True):
            code = hip.hipHostRegister(address, size_bytes, 0)
            if code:
                raise RuntimeError(f"cannot register pinned chat RAM (HIP code {code})")
        owner.registered = True
        buffer = (ctypes.c_uint8 * size).from_address(address)
        buffer._qwen_registration = owner
        tensor = torch.frombuffer(buffer, dtype=torch.uint8, count=size)
        if not tensor.is_pinned():
            raise RuntimeError("registered chat RAM is not recognized as pinned by PyTorch")
        return tensor, (address, address + size_bytes)
    except BaseException:
        owner.close()
        raise

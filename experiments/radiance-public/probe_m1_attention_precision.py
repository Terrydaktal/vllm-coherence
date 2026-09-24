"""Small independent attention witnesses; no model or private transcript input.

The CPU oracle uses exact means or a two-score softmax on representable inputs.
The optional native arm calls the pinned R4D binary in a separate process, with
private HIP allocations and a private stream. It never modifies the server.
These are operator counterexamples, not estimates of ordinary-chat error rates.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import time
import urllib.request
from fractions import Fraction
from pathlib import Path
from urllib.error import URLError

import numpy as np

R4D_SHA256 = "daa7a3bf79d2a1e0a7909a6ed9ddac2f0f3ac74b878569f4eabc2ccae839aecc"


def bf16_bits(values):
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("finite inputs required")
    bits = values.view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def from_bf16(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def fp8_bits(values):
    """Encode only exact, finite E4M3 values; never approximate the witness."""
    table = {}
    for byte in range(256):
        sign, exponent, mantissa = byte >> 7, (byte >> 3) & 15, byte & 7
        if exponent == 15 and mantissa == 7:
            continue
        value = (
            mantissa * 2.0**-9
            if exponent == 0
            else (1 + mantissa / 8) * 2.0 ** (exponent - 7)
        )
        table[(-1 if sign else 1) * value] = byte
    source = np.asarray(values, dtype=np.float32)
    out = np.empty(source.shape, dtype=np.uint8)
    for value in np.unique(source):
        if float(value) not in table:
            raise ValueError(f"not an exact E4M3 value: {value}")
        out[source == value] = table[float(value)]
    return out


def fixtures():
    for length in (16, 31, 32, 511, 512, 1023, 4095):
        q = np.zeros((1, 24, 256), dtype=np.float32)
        k = np.zeros((length, 256), dtype=np.float32)
        v = np.ones_like(k)
        # All but the shortened final split have an exactly representable mean.
        # The last split's mean is just above the next BF16 rounding boundary.
        v[30::32] = 1.125
        exact_mean = float(
            Fraction(1) + Fraction(len(range(30, length, 32)), 8 * length)
        )
        yield f"uniform_mean_{length}", "fp8", q, k, v, exact_mean
    q = np.full((1, 24, 256), 2.0**-22, dtype=np.float32)
    k = np.stack((np.zeros(256), np.full(256, 448))).astype(np.float32)
    v = np.stack((-np.ones(256), np.ones(256))).astype(np.float32)
    score_gap = 256 * (2.0**-22) * 448 / 16
    yield "folded_query_underflow", "fp8", q, k, v, math.tanh(score_gap / 2)
    # This is admitted by R4D's BF16 API but not the live FP8-cache profile.
    for value in (1.0, 2.0**-30, 2.0**20):
        q = np.zeros((1, 24, 256), dtype=np.float32)
        k = np.zeros((1, 256), dtype=np.float32)
        v = np.full_like(k, value)
        yield f"bf16_single_value_{value:g}", "bf16", q, k, v, value


def idle(api):
    with urllib.request.urlopen(api.rstrip("/") + "/metrics", timeout=3) as response:
        body = response.read().decode()
    counts = {"running": [], "waiting": []}
    for line in body.splitlines():
        match = re.fullmatch(
            r"vllm:num_requests_(running|waiting)(?:\{[^}]*\})?\s+(\S+)(?:\s+\d+)?",
            line,
        )
        if match:
            counts[match[1]].append(float(match[2]))
    if any(
        not values or any(value != 0 for value in values) for values in counts.values()
    ):
        raise RuntimeError("server is busy or idle counters are unavailable")
    return counts


def backend_stopped(api):
    """Accept only a refused local API connection, not a timeout or HTTP error."""
    try:
        with urllib.request.urlopen(api.rstrip("/") + "/health", timeout=3):
            pass
    except URLError as error:
        if (
            isinstance(error.reason, OSError)
            and error.reason.errno == errno.ECONNREFUSED
        ):
            return {"status": "connection_refused"}
        raise RuntimeError("could not establish that the backend is stopped") from error
    raise RuntimeError("backend still responds; standalone probe refused")


class Hip:
    """Minimal private allocation helper; no torch import or caching allocator."""

    def __init__(self):
        self.lib = ctypes.CDLL("libamdhip64.so")
        signatures = {
            "hipMalloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
            "hipFree": [ctypes.c_void_p],
            "hipMemcpy": [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ],
            "hipMemGetInfo": [
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_size_t),
            ],
            "hipStreamCreateWithFlags": [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_uint,
            ],
            "hipStreamSynchronize": [ctypes.c_void_p],
            "hipStreamDestroy": [ctypes.c_void_p],
        }
        for name, args in signatures.items():
            getattr(self.lib, name).argtypes = args
            getattr(self.lib, name).restype = ctypes.c_int
        self.pointers = []
        self.stream = ctypes.c_void_p()
        self.check(self.lib.hipStreamCreateWithFlags(ctypes.byref(self.stream), 1))
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        self.check(self.lib.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total)))
        # The largest case below allocates under 9 MiB. Keep at least 128 MiB
        # free after creating this process's HIP context before launching it.
        if free.value < 128 * 1024**2:
            self.close()
            raise RuntimeError(
                f"only {free.value} bytes free VRAM; no operator launched"
            )
        self.initial_free = free.value

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"HIP API failed with code {code}")

    def put(self, array):
        array = np.ascontiguousarray(array)
        pointer = ctypes.c_void_p()
        self.check(self.lib.hipMalloc(ctypes.byref(pointer), array.nbytes))
        self.pointers.append(pointer)
        self.check(
            self.lib.hipMemcpy(
                pointer, ctypes.c_void_p(array.ctypes.data), array.nbytes, 1
            )
        )
        return pointer.value

    def read(self, pointer, array):
        self.check(self.lib.hipStreamSynchronize(self.stream))
        self.check(
            self.lib.hipMemcpy(
                ctypes.c_void_p(array.ctypes.data),
                ctypes.c_void_p(pointer),
                array.nbytes,
                2,
            )
        )
        return array

    def clear(self):
        self.check(self.lib.hipStreamSynchronize(self.stream))
        for pointer in self.pointers:
            self.check(self.lib.hipFree(pointer))
        self.pointers.clear()

    def close(self):
        self.clear()
        self.check(self.lib.hipStreamDestroy(self.stream))


def native_case(hip, r4d, fixture):
    name, dtype, query, keys, values, expected = fixture
    length, dim = keys.shape
    blocks = (length + 15) // 16
    cache = np.zeros((blocks, 4, 16, 512), dtype=np.float32)
    for pos in range(length):
        cache[pos // 16, :, pos % 16, :dim] = keys[pos]
        cache[pos // 16, :, pos % 16, dim:] = values[pos]
    encoded = fp8_bits(cache) if dtype == "fp8" else bf16_bits(cache)
    query = bf16_bits(query)
    # Guard both output and scratch; use only the interior for device writes.
    guard = 512
    output = np.full(6144 + guard * 2, 0x4242, dtype=np.uint16)
    scratch_bytes = r4d.attn_decode_h256_gqa6_scratch_bytes(1, 1, 24, 4, 256, length, 0)
    scratch = np.full(scratch_bytes + 2 * guard, 0xA5, dtype=np.uint8)
    try:
        q, kv = hip.put(query), hip.put(encoded)
        table = hip.put(np.arange(blocks, dtype=np.int32)[None, :])
        lens = hip.put(np.array([length], dtype=np.int32))
        optr, sptr = hip.put(output), hip.put(scratch)
        fn = getattr(r4d, f"attn_decode_h256_gqa6_{dtype}kv")
        fn(
            q,
            kv,
            table,
            lens,
            optr + 2 * guard,
            0,
            0,
            sptr + guard,
            1,
            1,
            24,
            4,
            256,
            16,
            blocks,
            4 * 16 * 512,
            16 * 512,
            1 / 16,
            0,
            length,
            hip.stream.value,
        )
        hip.read(optr, output)
        hip.read(sptr, scratch)
        actual = from_bf16(output[guard:-guard])
        oracle = float(from_bf16(bf16_bits(expected)))
        intact = bool(
            (output[:guard] == 0x4242).all()
            and (output[-guard:] == 0x4242).all()
            and (scratch[:guard] == 0xA5).all()
            and (scratch[-guard:] == 0xA5).all()
        )
        if not intact or not np.isfinite(actual).all():
            raise RuntimeError("non-finite output or damaged guard region")
        return {
            "case": name,
            "cache_dtype": dtype,
            "context_tokens": length,
            "mathematical_expectation": expected,
            "expected_bf16": oracle,
            "actual_unique": np.unique(actual).tolist(),
            "different_elements": int(np.count_nonzero(actual != oracle)),
            "elements": int(actual.size),
            "guard_regions_intact": intact,
            "maximum_explicit_device_bytes": query.nbytes
            + encoded.nbytes
            + blocks * 4
            + 4
            + output.nbytes
            + scratch.nbytes,
        }
    finally:
        hip.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--api", default="http://127.0.0.1:8080")
    parser.add_argument("--library", type=Path)
    parser.add_argument(
        "--backend-stopped",
        action="store_true",
        help="Run in a reserved window with the model server stopped; require connection refusal",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "m1-attention-precision-v1",
        "complete": False,
        "scope": "synthetic independent operator witnesses; no model-quality or universal-equivalence claim",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "backend_mode": "stopped" if args.backend_stopped else "idle",
        "cpu_oracles": [
            {
                "case": f[0],
                "cache_dtype": f[1],
                "context_tokens": len(f[3]),
                "expected_bf16": float(from_bf16(bf16_bits(f[-1]))),
            }
            for f in fixtures()
        ],
    }
    result = args.output / "result.json"
    started = time.monotonic()
    try:
        if args.allow_gpu:
            if (
                args.library is None
                or hashlib.sha256(args.library.read_bytes()).hexdigest() != R4D_SHA256
            ):
                raise RuntimeError("the pinned R4D binary is required")
            report["library_sha256"] = R4D_SHA256
            with open(
                os.environ.get(
                    "QWEN_CONFORMANCE_GPU_LOCK", "/tmp/qwen-conformance-gpu.lock"
                ),
                "a",
            ) as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                check_backend = backend_stopped if args.backend_stopped else idle
                report["backend_before"] = check_backend(args.api)
                spec = importlib.util.spec_from_file_location("r4d", args.library)
                r4d = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(r4d)
                hip = Hip()
                try:
                    report["free_vram_after_context_bytes"] = hip.initial_free
                    report["native"] = []
                    for fixture in fixtures():
                        check_backend(args.api)
                        report["native"].append(native_case(hip, r4d, fixture))
                        result.write_text(json.dumps(report, indent=2) + "\n")
                finally:
                    hip.close()
                report["backend_after"] = check_backend(args.api)
        report["complete"] = True
    except Exception as error:
        report["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        result.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

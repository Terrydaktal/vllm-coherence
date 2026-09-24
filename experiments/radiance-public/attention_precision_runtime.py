"""Bind the qualified attention repair before metadata allocation or graph capture."""

import ctypes as c
import functools
import json
from pathlib import Path

from attention_precision import digest
from m1_followup_guards import guard_selector, validate_cache_dtype
from stock_m1_attention_shared import R4DArgs

from qwen_r9700_lab.diagnostic_contract import authenticate


class PrecisionAttention:
    def __init__(self, build):
        root = Path(build)
        self.manifest = json.loads((root / "build.json").read_text())
        authenticate(self.manifest)
        if self.manifest.get("kernel_abi") != "coherence-attention-precision-v1":
            raise ValueError("unknown attention precision ABI")
        for name, expected in self.manifest["files"].items():
            if digest(root / name) != expected:
                raise ValueError("attention artifact changed: " + name)
        self.library = c.CDLL(str(root / "native.so"))
        if c.sizeof(R4DArgs) != 136:
            raise ValueError("R4D host argument layout changed")
        for phase in ("decode", "prefill"):
            for dtype in ("fp8", "bf16"):
                name = f"attn_{phase}_h256_gqa6_{dtype}kv"
                function = getattr(self.library, "r4d_" + name)
                function.argtypes = [c.POINTER(R4DArgs), c.c_void_p]
                function.restype = c.c_int

                def launch(*values, _function=function):
                    if len(values) != 21:
                        raise ValueError("R4D attention call signature changed")
                    args = R4DArgs(*values[:7], 0, values[7], *values[8:-1])
                    code = _function(c.byref(args), values[-1])
                    if code:
                        raise RuntimeError(f"corrected R4D attention failed: {code}")

                launch.__name__ = name + "_precision"
                setattr(self, name, launch)
        self.scratch = self.library.r4d_attn_decode_h256_gqa6_scratch_bytes
        self.scratch.argtypes = [c.POINTER(R4DArgs)]
        self.scratch.restype = c.c_long

    def attn_decode_h256_gqa6_scratch_bytes(
        self, sequences, width, heads, kv_heads, dim, max_ctx, splits
    ):
        args = R4DArgs(
            *([0] * 9),
            sequences,
            width,
            heads,
            kv_heads,
            dim,
            16,
            0,
            0,
            0,
            1.0,
            splits,
            max_ctx,
        )
        return self.scratch(c.byref(args))


def install_bindings(r4d, build):
    current = getattr(r4d, "_coherence_attention_precision", None)
    if current is not None:
        if current != digest(Path(build) / "build.json"):
            raise ValueError("another attention precision build is already installed")
        return
    candidate = PrecisionAttention(build)
    guard_selector(r4d)
    for name in (
        "attn_decode_h256_gqa6_scratch_bytes",
        *(
            f"attn_{phase}_h256_gqa6_{dtype}kv"
            for phase in ("decode", "prefill")
            for dtype in ("fp8", "bf16")
        ),
    ):
        setattr(r4d, name, getattr(candidate, name))
    r4d._coherence_attention_precision = digest(Path(build) / "build.json")


def guard_geometry(implementation):
    original = implementation._geometry
    if getattr(original, "_coherence_format_guard", False):
        return

    @functools.wraps(original)
    def geometry(self, kv_cache, query, out):
        # Check on every invocation, even after geometry has been cached.
        validate_cache_dtype(kv_cache.dtype, query.dtype, out.dtype)
        signature = (
            kv_cache.dtype,
            tuple(kv_cache.shape),
            kv_cache.stride(),
            query.stride(),
            out.stride(),
        )
        if getattr(self, "_coherence_kv_signature", None) != signature:
            self._kv_geometry = None
            self._coherence_kv_signature = signature
        return original(self, kv_cache, query, out)

    geometry._coherence_format_guard = True
    implementation._geometry = geometry

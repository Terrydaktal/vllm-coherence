"""Experimental CPU MXFP4 unpacking with an immutable packed-byte lookup table.

The table changes data movement, not the ordered reference dot products. The
running campaign is unchanged. Extreme scales retain the canonical decoder so
its overflow/subnormal and floating-point error behavior is preserved too.
"""

from __future__ import annotations

import hashlib
import inspect
import sys
from pathlib import Path

import numpy as np

from qwen_r9700_lab import conformance_reference as ref
from qwen_r9700_lab.conformance_model import QuantizedQwenReference
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, seal

MANTISSAS = (0, 1, 2, 3, 4, 6, 8, 12)


def decoded_bits(scale: int, code: int) -> int:
    """Exact RNE/gradual-underflow FP32 bits of a finite E2M1/E8M0 product.

    The real magnitude is m * 2**(scale-128), with m requiring at most four
    bits. Every finite result is therefore exactly representable in FP32.
    """
    if (
        type(scale) is not int
        or type(code) is not int
        or not 0 <= scale < 255
        or not 0 <= code < 16
    ):
        raise DiagnosticError("invalid finite MXFP4 table entry")
    sign = (code >> 3) << 31
    mantissa = MANTISSAS[code & 7]
    if mantissa == 0:
        return sign
    leading = mantissa.bit_length() - 1
    exponent = scale - 128 + leading
    if exponent > 127:
        return sign | 0x7F800000
    if exponent < -126:
        return sign | (mantissa << (scale + 21))
    fraction = (mantissa << (23 - leading)) & 0x7FFFFF
    return sign | ((exponent + 127) << 23) | fraction


def make_table():
    bits = np.array(
        [[decoded_bits(scale, code) for code in range(16)] for scale in range(255)], dtype="<u4"
    )
    packed = np.arange(256, dtype=np.uint16)
    pairs = bits[:, packed & 15].astype("<u8") | (bits[:, packed >> 4].astype("<u8") << 32)
    # An immutable bytes owner prevents callers from re-enabling write access.
    return np.frombuffer(pairs.tobytes(), dtype="<u8")


class MXFP4Lookup:
    def __init__(self, reference):
        if sys.byteorder != "little":
            raise DiagnosticError("MXFP4 lookup qualification currently admits little-endian CPUs")
        expected = np.array([decoded_bits(127, code) for code in range(16)], dtype="<u4")
        if reference.LEVELS.dtype != np.float32 or reference.LEVELS.tobytes() != expected.tobytes():
            raise DiagnosticError("reference MXFP4 levels differ from the lookup contract")
        self.reference = reference
        self.table = make_table()
        self.calls = 0
        self.fallbacks = 0
        self.execution_binding = seal(
            {
                "schema": "urn:qwen:mxfp4-lookup-execution:v1",
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "reference_sha256": hashlib.sha256(
                    Path(reference.__file__).read_bytes()
                ).hexdigest(),
                "numpy_version": np.__version__,
                "table_sha256": hashlib.sha256(self.table.tobytes()).hexdigest(),
                "table_bytes": self.table.nbytes,
                "pair_entries": 255 * 256,
                "scale_group_weights": 32,
                "fast_scales": [2, 252],
                "scope": "Same decoded FP32 bits; no changes to projection arithmetic.",
                "formal_equivalence": "UNPROVED",
            }
        )

    def __call__(self, packed, scales):
        packed, scales = np.asarray(packed), np.asarray(scales)
        if packed.dtype != np.uint8 or scales.dtype != np.uint8 or packed.ndim != 2:
            raise DiagnosticError("MXFP4 storage must be a packed matrix with E8M0 scales")
        rows, half_width = packed.shape
        groups = half_width // 16
        if half_width % 16 or scales.shape != (rows, groups) or np.any(scales == 255):
            raise DiagnosticError("unsupported MXFP4 shape or nonfinite E8M0 scale")
        self.calls += 1
        if rows == 0 or np.any((scales < 2) | (scales >= 253)):
            # Keep the original decoder's empty-row exception, FP environment,
            # warnings and error policy. Normal checkpoint tiles avoid this path.
            self.fallbacks += 1
            return self.reference.unpack_mxfp4(packed, scales)
        # A 32-weight scale group covers 16 packed bytes. Index arithmetic is
        # uint16: the largest admitted index (252*256+255) cannot overflow it.
        base = np.left_shift(scales.astype(np.uint16), 8)[..., None]
        indices = np.add(base, packed.reshape(rows, groups, 16), dtype=np.uint16)
        return self.table[indices].view("<f4").reshape(rows, half_width * 2)


class LookupQuantizedQwenReference(QuantizedQwenReference):
    """Keep the model unchanged except for the decoded-weight tile provider."""

    def __init__(self, *args, unpacker, **kwargs):
        # This method copies the current projection's tiling/quantization order.
        # Refuse a future base implementation rather than silently using stale
        # projection logic after a reference-code update.
        source = inspect.getsource(QuantizedQwenReference.project).encode()
        expected = "c31b1ca2d50d74b5df7631ef8da4ba9a14acfcfae17d45765dbf0196b13fd6b3"
        if hashlib.sha256(source).hexdigest() != expected:
            raise DiagnosticError("reference projection changed; requalify the lookup adapter")
        super().__init__(*args, **kwargs)
        self.unpacker = unpacker

    def project(self, name, x):
        weight_name = name + ".weight"
        scale_name = name + ".weight_scale"
        if scale_name not in self.weights:
            return self._linear(x, self.weights.tensor(weight_name))
        packed = self.weights.tensor(weight_name)
        scales = self.weights.tensor(scale_name)
        outputs = []
        if self.precision["activation_fp8"]:
            code, xs = ref.activation_quantize(x)
            linear_input = ref.fp8_decode(code) * xs
        else:
            linear_input = ref.bf16(x)
        for start in range(0, packed.shape[0], 256):
            weights = self.unpacker(packed[start : start + 256], scales[start : start + 256])
            outputs.append(self._linear(linear_input, weights))
        return np.concatenate(outputs, axis=-1)

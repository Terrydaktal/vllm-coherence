"""Exhaustive finite-domain check of the actual generated integer fold helpers."""

import ctypes
import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest


def test_generated_fold_tables_match_independent_fp4_to_fp8_values(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for generated-code validation")
    path = (
        Path(__file__).parents[1]
        / "experiments/radiance-public/build_mxfp4_register_decode.py"
    )
    spec = importlib.util.spec_from_file_location("fold_builder_under_test", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    candidate = builder.arithmetic_fold_tables(
        "template <int DWN, int DBK, int DKS, int DTM> void decode() {\n"
        "  auto t0 = kMag[d][0], t1 = kMag[d][1];\n}\n"
        "// Split-K partial slab.\n"
    )
    helper = candidate[: candidate.index("template <int DWN")]
    source = tmp_path / "fold.cpp"
    binary = tmp_path / "fold.so"
    source.write_text(
        "#define __device__\n#define __forceinline__ inline\n"
        + helper
        + '\nextern "C" unsigned int fold(int i, int half) {\n'
        "  return half ? coherence_fold_hi(i) : coherence_fold_lo(i);\n}\n"
    )
    subprocess.run(
        [compiler, "-O3", "-shared", "-fPIC", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
    )
    native = ctypes.CDLL(str(binary))
    native.fold.argtypes = [ctypes.c_int, ctypes.c_int]
    native.fold.restype = ctypes.c_uint

    # Independent E4M3FN decoding. Every admitted folded FP4 magnitude is
    # exactly representable, so no tolerance or nearest-value heuristic enters.
    decode = {}
    for bits in range(127):
        exponent, mantissa = bits >> 3, bits & 7
        value = (
            mantissa * 2.0**-9
            if exponent == 0
            else (1 + mantissa / 8) * 2.0 ** (exponent - 7)
        )
        decode[value] = bits
    fp4_magnitudes = (0, 0.5, 1, 1.5, 2, 3, 4, 6)
    tables = {}
    for delta in range(-6, 9):
        words = [native.fold(delta + 6, half) for half in (0, 1)]
        for nibble in range(16):
            magnitude = nibble & 7
            actual = (words[magnitude // 4] >> (8 * (magnitude % 4))) & 255
            actual |= (nibble & 8) << 4
            expected = decode[fp4_magnitudes[magnitude] * 2.0**-delta]
            expected |= (nibble & 8) << 4
            assert actual == expected
        tables[delta] = words
    # Cover every pair of byte-valued exponents admitted by the caller's clamp.
    for reference_exponent in range(256):
        for group_exponent in range(256):
            delta = max(-6, min(8, reference_exponent - group_exponent))
            assert [native.fold(delta + 6, half) for half in (0, 1)] == tables[delta]

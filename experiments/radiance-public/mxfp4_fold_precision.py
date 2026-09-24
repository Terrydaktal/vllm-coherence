"""Preserve stored MXFP4 coefficients within the FP8 folded GEMM.

The exponent window is [-6, 8]: 6*2**6=384 and .5*2**-8=2**-9
are both exactly representable in E4M3. Ordinary rows retain their reference
exponent. Wider rows use the existing per-block-scaled kernel, with both
weight and activation layouts handled explicitly.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_row_ref(weight_scale):
    import torch

    if (
        weight_scale.dtype != torch.uint8
        or weight_scale.ndim != 2
        or not weight_scale.numel()
    ):
        raise ValueError("expected nonempty [K/32,N] E8M0 scale bytes")
    hi = weight_scale.amax(0).to(torch.int16)
    lo = weight_scale.amin(0).to(torch.int16)
    if bool((hi == 255).any()):
        raise ValueError("non-finite E8M0 scale is not admitted")
    if bool((hi - lo > 14).any()):
        # The existing ABI uses a two-element sentinel for per-block scaling.
        return torch.zeros(2, dtype=torch.uint8, device=weight_scale.device)
    # Keep ordinary rows unchanged. Reserve exponent zero for the block scale:
    # the folded epilogue's float exponent encoding requires a normal scale.
    return torch.minimum(hi, lo + 8).clamp_min(1).to(torch.uint8).contiguous()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"precision repair anchor missing or ambiguous: {old[:80]}")
    return text.replace(old, new)


def patched_sources(hip, wrapper):
    # Retain the qualified TP1 width dispatch. This input guard also prevents
    # accidentally applying the transformation to a different upstream kernel.
    from build_mxfp4_dispatch import patched_sources as dispatch_sources

    hip, wrapper = dispatch_sources(hip, wrapper)
    begin = hip.index("__device__ __constant__ unsigned int kMag[16][2]")
    end = hip.index("};", begin) + 2
    # Rows are d=-6..8, followed by one unused row. All entries are exact;
    # sign bits are applied separately by the unchanged vector lookup.
    pairs = [
        (0x6C686000, 0x7C787470),
        (0x64605800, 0x74706C68),
        (0x5C585000, 0x6C686460),
        (0x54504800, 0x64605C58),
        (0x4C484000, 0x5C585450),
        (0x44403800, 0x54504C48),
        (0x3C383000, 0x4C484440),
        (0x34302800, 0x44403C38),
        (0x2C282000, 0x3C383430),
        (0x24201800, 0x34302C28),
        (0x1C181000, 0x2C282420),
        (0x14100800, 0x24201C18),
        (0x0C080400, 0x1C181410),
        (0x06040200, 0x14100C08),
        (0x03020100, 0x0C080604),
        (0, 0),
    ]
    table = "// Coherence: exact folded exponent window d=-6..8; lookup index is d+6.\n"
    table += "__device__ __constant__ unsigned int kMag[16][2] = {\n"
    table += "".join(f"  {{0x{a:08x}u, 0x{b:08x}u}},\n" for a, b in pairs) + "};"
    hip = hip[:begin] + table + hip[end:]
    for variable, count in (("d", 3), ("dsh", 3)):
        old = f"{variable} = {variable} < 0 ? 0 : ({variable} > 15 ? 15 : {variable});"
        if hip.count(old) != count:
            raise ValueError("unexpected folded exponent lookup count")
        hip = hip.replace(
            old,
            f"{variable} = ({variable} < -6 ? -6 : ({variable} > 8 ? 8 : {variable})) + 6;",
        )
    # The original per-block route ignores WPERM. Give it both layout contracts
    # so forced and automatically selected fallbacks also work with hoisted FP8.
    hip = replace_once(
        hip,
        "__global__ __launch_bounds__(NWAVE * 32) void radiance_mxfp4_fp8_gemm(",
        "template <bool WPERM, bool ATILED = false>\n__global__ __launch_bounds__(NWAVE * 32) void radiance_mxfp4_fp8_gemm(",
    )
    hip = replace_once(
        hip,
        "      const unsigned char *src = A + (size_t)(m0 + rc) * K + k0 + off;\n"
        "      const uint4_t v0 = *(const uint4_t *)(src);\n"
        "      const uint4_t v1 = *(const uint4_t *)(src + 16);",
        "      const int gm = m0 + rc;\n"
        "      uint4_t v0, v1;\n"
        "      if constexpr (ATILED) {\n"
        "        const size_t ai = (size_t)(gm / 16) * K * 16 +\n"
        "            (size_t)((k0 + off) / 16) * 256 + (gm % 16) * 8;\n"
        "        const uint2_t a0 = *(const uint2_t *)(A + ai), a1 = *(const uint2_t *)(A + ai + 128);\n"
        "        const uint2_t a2 = *(const uint2_t *)(A + ai + 256), a3 = *(const uint2_t *)(A + ai + 384);\n"
        "        v0 = (uint4_t){a0[0], a0[1], a1[0], a1[1]};\n"
        "        v1 = (uint4_t){a2[0], a2[1], a3[0], a3[1]};\n"
        "      } else {\n"
        "        const unsigned char *src = A + (size_t)gm * K + k0 + off;\n"
        "        v0 = *(const uint4_t *)src; v1 = *(const uint4_t *)(src + 16);\n"
        "      }",
    )
    hip = replace_once(
        hip,
        "      const unsigned long long packed =\n"
        "          *(const unsigned long long *)(W + ((size_t)(n0 + rc) * K + k0) / 2 + off);",
        "      const int gn = n0 + rc, kk = k0 + off * 2;\n"
        "      unsigned long long packed;\n"
        "      if constexpr (WPERM) {\n"
        "        const size_t slot = ((size_t)(gn / 16) * (K / 16) + kk / 16) * 32 + (gn % 16);\n"
        "        const unsigned int *wp = (const unsigned int *)W;\n"
        "        packed = (unsigned long long)wp[slot] | ((unsigned long long)wp[slot + 16] << 32);\n"
        "      } else packed = *(const unsigned long long *)(W + ((size_t)gn * K + kk) / 2);",
    )
    hip = replace_once(
        hip,
        "        const float s = __int_as_float(\n"
        "            (int)sS[(wn * 32 + j * 16 + col) * (BK / 32) + blk] << 23);",
        "        const int exponent = sS[(wn * 32 + j * 16 + col) * (BK / 32) + blk];\n"
        "        const float s = __int_as_float(exponent ? exponent << 23 : 0x00400000);",
    )
    old = "  } else\n    hipLaunchKernelGGL(radiance_mxfp4_fp8_gemm, grid, block, 0, (hipStream_t)stream,"
    # Read the layout at the host dispatch boundary, never from device data.
    hip = replace_once(
        hip,
        old,
        '  } else {\n    static const bool wp = [] { const char *v = getenv("RADIANCE_MXFP4_WPERM"); return v && atoi(v); }();\n'
        "    if (wp) hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm<true>), grid, block, 0, (hipStream_t)stream,\n"
        "      (const unsigned char *)a, (const unsigned char *)w, (const unsigned char *)ws,\n"
        "      (const float *)as, (__bf16 *)c, M, N, K);\n"
        "    else hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm<false>), grid, block, 0, (hipStream_t)stream,",
    )
    hip = replace_once(
        hip,
        "                       (const unsigned char *)ws, (const float *)as, (__bf16 *)c, M, N, K);\n}",
        "                       (const unsigned char *)ws, (const float *)as, (__bf16 *)c, M, N, K);\n  }\n}",
    )
    hip = replace_once(
        hip,
        '  if (!wref) throw std::runtime_error("radiance_mxfp4_fp8: tiled A needs the folded path (wref)");',
        "  if (!wref) {\n"
        '    static const bool wp = [] { const char *v = getenv("RADIANCE_MXFP4_WPERM"); return v && atoi(v); }();\n'
        "    dim3 block(NWAVE * 32), grid((N + BN - 1) / BN, (M + BM - 1) / BM);\n"
        "    if (wp) hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm<true, true>), grid, block, 0, (hipStream_t)stream,\n"
        "      (const unsigned char *)a, (const unsigned char *)w, (const unsigned char *)ws,\n"
        "      (const float *)as, (__bf16 *)c, M, N, K);\n"
        "    else hipLaunchKernelGGL((radiance_mxfp4_fp8_gemm<false, true>), grid, block, 0, (hipStream_t)stream,\n"
        "      (const unsigned char *)a, (const unsigned char *)w, (const unsigned char *)ws,\n"
        "      (const float *)as, (__bf16 *)c, M, N, K);\n"
        "    return;\n"
        "  }",
    )
    wrapper = replace_once(
        wrapper,
        "    return weight_scale.max(dim=0).values.contiguous()",
        "    from mxfp4_fold_precision import make_row_ref as exact_fold_ref\n"
        "    return exact_fold_ref(weight_scale)",
    )
    wrapper = replace_once(
        wrapper,
        "                ref = make_row_ref(layer.weight_scale.data)          # folded",
        "                ref = make_row_ref(layer.weight_scale.data)\n"
        "                if ref.numel() == 2:\n"
        '                    STATS["fast"] -= 1\n'
        '                    STATS["perblock_precision"] = STATS.get("perblock_precision", 0) + 1',
    )
    wrapper = replace_once(
        wrapper,
        "           weight_ref.data_ptr(), x_scale.data_ptr(), out.data_ptr(),",
        "           weight_ref.data_ptr() if weight_ref.numel() == N else 0, x_scale.data_ptr(), out.data_ptr(),",
    )
    wrapper = replace_once(
        wrapper,
        "if R4D_DECODE_MAX_M > 0:\n",
        "if R4D_DECODE_MAX_M > 0:\n"
        '    raise RuntimeError("exact folding requires the Coherence GEMM, not the old libr4d fold table")\n',
    )
    return hip, wrapper


def build(source, output):
    from build_mxfp4_dispatch import patched_sources as dispatch_sources

    hip = (source / "radiance_mxfp4_fp8.hip").read_text()
    wrapper = (source / "radiance_mxfp4.py").read_text()
    control = dispatch_sources(hip, wrapper)
    candidate = patched_sources(hip, wrapper)
    output.mkdir(mode=0o700)
    includes = shlex.split(
        subprocess.check_output(
            [sys.executable, "-m", "pybind11", "--includes"], text=True
        )
    )
    report = {
        "schema": "coherence-mxfp4-fold-precision-v1",
        "status": "BUILT_UNTESTED",
        "patch_sha256": digest(__file__),
        "variants": {},
    }
    for name, (native, python) in (("control", control), ("candidate", candidate)):
        directory = output / name
        directory.mkdir()
        (directory / "radiance_mxfp4_fp8.hip").write_text(native)
        (directory / "radiance_mxfp4.py").write_text(python)
        binary = directory / "radiance_mxfp4_fp8.so"
        command = [
            "/opt/rocm/bin/hipcc",
            "-O3",
            "-std=c++17",
            "-fPIC",
            "-shared",
            "--offload-arch=gfx1201",
            "-Wno-unused-result",
            *includes,
            str(directory / "radiance_mxfp4_fp8.hip"),
            "-o",
            str(binary),
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=600, check=False
        )
        (directory / "compiler.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
        report["variants"][name] = {
            "command": command,
            "source_sha256": digest(directory / "radiance_mxfp4_fp8.hip"),
            "python_sha256": digest(directory / "radiance_mxfp4.py"),
            "binary_sha256": digest(binary),
        }
        print(json.dumps({"built": name, **report["variants"][name]}), flush=True)
    (output / "build.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)

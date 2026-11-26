"""Isolated TP1 input-projection experiments on Coherence's corrected tiled GEMM.

The candidate keeps the original model parameters for decode and fallback.
The cached encoding is the mode-2 layout from SlyBase/vllm-sly-radiance
ffeeff45, using Coherence's signed exponent difference and table.
"""

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

SHAPES = {(14336, 5120), (16480, 5120), (34816, 5120)}
SOURCE_NAME = "mxfp4-prefill.hip"
LIBRARY_NAME = "mxfp4-prefill.so"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pack_words(weight, *, chunk_bytes=1 << 20):
    """Permute the eight nibbles in each existing fragment-order lane word."""
    import torch

    if weight.dtype != torch.uint8 or not weight.is_contiguous() or weight.numel() % 4:
        raise ValueError("expected packed contiguous uint8 lane words")
    if chunk_bytes < 4:
        raise ValueError("a packing chunk must contain at least one lane word")
    out = torch.empty_like(weight)
    dst = out.view(-1, 4)
    words = weight.view(-1, 4)
    step = chunk_bytes // 4
    for i in range(0, words.shape[0], step):
        band = words[i : i + step]
        codes = torch.stack((band & 15, band >> 4), dim=-1).reshape(-1, 8)
        lo, hi = codes[:, :4], codes[:, 4:]
        dst[i : i + step].copy_(
            (lo & 7) | (hi.roll(1, dims=1) & 8) | ((hi & 7) << 4) | ((lo & 8) << 4)
        )
    return out


def fold_indices(scales, ref):
    """Return signed-window lookup indices. Sentinels retain the ordinary path."""
    import torch

    if (
        scales.ndim != 2
        or ref.shape != (scales.shape[1],)
        or scales.dtype != torch.uint8
        or ref.dtype != torch.uint8
        or scales.device != ref.device
        or not scales.is_contiguous()
        or not ref.is_contiguous()
        or ref.numel() <= 2
        or not scales.shape[0]
    ):
        return None
    references = ref.to(torch.int16)
    maximum, minimum = scales.amax(dim=0), scales.amin(dim=0)
    if (
        (
            (references < 1)
            | (references == 255)
            | (maximum == 255)
            | (references - minimum > 8)
            | (references - maximum < -6)
        )
        .any()
        .item()
    ):
        return None
    return (references.unsqueeze(0) - scales + 6).to(torch.uint8).contiguous()


def source_for_candidate(root):
    """Derive both arms from the same checksum-pinned, repaired kernel."""
    from mxfp4_fold_precision import patched_sources, replace_once

    root = Path(root)
    hip, _ = patched_sources(
        (root / "radiance_mxfp4_fp8.hip").read_text(),
        (root / "radiance_mxfp4.py").read_text(),
    )
    hip = hip[
        : hip.index(
            "// ---------------------------------------------------------------- decode path"
        )
    ]
    hip = hip.replace("#include <pybind11/pybind11.h>\n", "").replace(
        "namespace py = pybind11;\n", ""
    )
    start = hip.index("template <int TN, bool WPERM, int LBK = BK>")
    prefix, kernel = hip[:start], hip[start:]
    control = kernel.replace(
        "radiance_mxfp4_fp8_gemm_atiled", "coherence_control_atiled"
    )
    kernel = replace_once(kernel, "int LBK = BK>", "int LBK, bool PSW>")
    decoding = kernel.index("          int d = (int)((wref[it]")
    end = kernel.index("\n        }", decoding)
    original = kernel[decoding:end]
    if "(d < -6 ? -6 : (d > 8 ? 8 : d)) + 6" not in original:
        raise ValueError("expected the corrected signed fold window")
    kernel = (
        kernel[:decoding]
        + """          if constexpr (PSW) {
            const unsigned int idx = (wsv[it] >> (8 * q)) & 255;
            const uint2_t t = *(const uint2_t *)&sMag[idx * 2];
            const unsigned word = pk[it][q];
            const unsigned hi = (word >> 4) | (word << 28);
            const unsigned lo4 = __builtin_amdgcn_perm(t[1], t[0], word & 0x07070707u) | (word & 0x80808080u);
            const unsigned hi4 = __builtin_amdgcn_perm(t[1], t[0], hi & 0x07070707u) | (hi & 0x80808080u);
            *(uint2_t *)&sW[(wrow[it] + q) * LWSTR + wcol[it]] = uint2_t{lo4, hi4};
          } else {
"""
        + original
        + "\n          }"
        + kernel[end:]
    )
    return (
        prefix
        + control
        + kernel
        + """
template <bool PSW, int TN> static void launch_input(const unsigned char* a,
    const unsigned char* w, const unsigned char* ws, const unsigned char* ref, const float* scale,
    __bf16* out, int m, int n, int k, hipStream_t stream) {
  constexpr int width = BNF_OF(TN);
  dim3 grid((n + width - 1) / width, (m + BMF - 1) / BMF);
  if constexpr (!PSW) {
    coherence_control_atiled<TN, true, TN == 2 ? 128 : 64>
        <<<grid, NTHREADS, 0, stream>>>(a, w, ws, ref, scale, out, m, n, k);
  } else {
    radiance_mxfp4_fp8_gemm_atiled<TN, true, TN == 2 ? 128 : 64, PSW>
        <<<grid, NTHREADS, 0, stream>>>(a, w, ws, ref, scale, out, m, n, k);
  }
}

extern "C" int coherence_input_prefill(const unsigned char* a, const unsigned char* w,
    const unsigned char* ws, const unsigned char* ref, const float* scale, __bf16* out,
    int m, int n, int k, hipStream_t stream, int tn, int mode) {
  if (!a || !w || !ws || !ref || !scale || !out || m <= 64 || m > 4096 || k != 5120
      || (n != 14336 && n != 16480 && n != 34816) || (tn != 2 && tn != 4)) return -1;
#define RUN(P) if (tn == 4) launch_input<P,4>(a,w,ws,ref,scale,out,m,n,k,stream); \\
                  else launch_input<P,2>(a,w,ws,ref,scale,out,m,n,k,stream)
  switch (mode) {
    case 0: { RUN(false); break; }
    case 1: { RUN(true); break; }
    default: return -1;
  }
#undef RUN
  return int(hipGetLastError());
}
"""
    )


def build(root, output, hipcc="/opt/rocm/bin/hipcc"):
    """Build a private library. Publish the hash receipt after compilation succeeds."""
    from qwen_r9700_lab.diagnostic_contract import seal

    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        tmp = Path(tmp)
        src, binary = tmp / SOURCE_NAME, tmp / LIBRARY_NAME
        src.write_text(source_for_candidate(root))
        subprocess.run(
            [
                hipcc,
                "-O3",
                "-std=c++17",
                "--offload-arch=gfx1201",
                "-shared",
                "-fPIC",
                "-cuid=coherence_input_" + sha256(src),
                str(src),
                "-o",
                str(binary),
            ],
            check=True,
        )
        receipt = seal(
            {
                "kernel_abi": "coherence-mxfp4-input-prefill-v1",
                "module": sha256(__file__),
                "files": {SOURCE_NAME: sha256(src), LIBRARY_NAME: sha256(binary)},
                "parent": sha256(Path(root) / "radiance_mxfp4_fp8.hip"),
            }
        )
        (tmp / "build.json").write_text(json.dumps(receipt, sort_keys=True))
        for name in (SOURCE_NAME, LIBRARY_NAME, "build.json"):
            (tmp / name).replace(output / name)
    return output / LIBRARY_NAME


def checked_launch(library):
    from qwen_r9700_lab.diagnostic_contract import authenticate

    library = Path(library)
    if library.is_dir():
        library = library / LIBRARY_NAME
    receipt = json.loads(library.with_name("build.json").read_text())
    authenticate(receipt)
    if (
        receipt["kernel_abi"] != "coherence-mxfp4-input-prefill-v1"
        or receipt["files"]
        != {
            SOURCE_NAME: sha256(library.with_name(SOURCE_NAME)),
            LIBRARY_NAME: sha256(library),
        }
        or sha256(__file__) != receipt["module"]
    ):
        raise ValueError("MXFP4 prefill build does not match its receipt")
    launch = ctypes.CDLL(str(library.resolve())).coherence_input_prefill
    launch.argtypes = (
        [ctypes.c_void_p] * 6
        + [ctypes.c_int] * 3
        + [ctypes.c_void_p]
        + [ctypes.c_int] * 2
    )
    launch.restype = ctypes.c_int
    return launch


def install(
    worker,
    *,
    library,
    max_preprocessed_bytes=96 << 20,
):
    """Attach after the qualified prefill hooks and before graph capture."""
    import torch

    import radiance_mxfp4 as gemm

    if max_preprocessed_bytes <= 0:
        raise ValueError("MXFP4 weight preprocessing requires a positive byte budget")
    alignment = worker._qwen_performance_repairs.manifest["prefill_alignment"]
    device = torch.device(worker.device)
    if (
        worker.vllm_config.parallel_config.tensor_parallel_size != 1
        or not gemm.WPERM
        or torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        != "gfx1201"
        or not alignment.get("input_tiles")
        or sha256(gemm._ext.__file__) != alignment["input_binary_sha256"]
    ):
        raise ValueError(
            "expected the qualified single-R9700 tiled input projection runtime"
        )
    native = checked_launch(library)
    original = gemm._ext.launch_at
    if getattr(original, "_coherence_input_candidate", False):
        raise RuntimeError("MXFP4 prefill candidate already installed")
    tn4_min = int(os.getenv("RADIANCE_MXFP4_TN4_MIN_M", "2048"))
    entries = {}
    usage = {
        "eligible_layers": 0,
        "preprocessed_layers": 0,
        "preprocessed_bytes": 0,
        "preprocessed_calls": 0,
    }
    for module in worker.model_runner.model.modules():
        if not getattr(module, "radiance_w4a8_ok", False):
            continue
        weight, scales, ref = module.weight, module.weight_scale, module.radiance_wref
        n, half_k = weight.shape
        k = half_k * 2
        if (
            (n, k) not in SHAPES
            or ref.shape != (n,)
            or scales.shape != (k // 32, n)
            or not all(
                t.dtype == torch.uint8
                and t.device == weight.device
                and t.is_contiguous()
                for t in (weight, scales, ref)
            )
            or weight.device.type != "cuda"
            or weight.device.index != torch.cuda.current_device()
        ):
            continue
        key = weight.data_ptr(), scales.data_ptr(), ref.data_ptr(), n, k
        if key in entries:
            continue
        packed = None
        size = weight.nbytes + scales.nbytes
        if usage["preprocessed_bytes"] + size <= max_preprocessed_bytes:
            free, _ = torch.cuda.mem_get_info(weight.device)
            reserve = worker.vllm_config.cache_config.kv_cache_memory_bytes
            if reserve is None:
                raise ValueError(
                    "weight preprocessing requires an explicit KV byte budget"
                )
            # Keep room for the configured KV arena and the bounded preparation scratch.
            if free >= reserve + size + (32 << 20):
                indices = fold_indices(scales, ref)
                if indices is not None:
                    packed = pack_words(weight), indices
                    usage["preprocessed_bytes"] += size
                    usage["preprocessed_layers"] += 1
        entries[key] = packed
        usage["eligible_layers"] += 1
    if not usage["preprocessed_layers"]:
        raise ValueError(
            "no eligible MXFP4 input projections fit the selected candidate budget"
        )

    launch = _make_dispatch(original, native, entries, usage, tn4_min)
    torch.cuda.current_stream(device).synchronize()
    gemm._ext.launch_at = launch
    return usage


def _make_dispatch(original, native, entries, usage, tn4_min):

    def launch(a, w, ws, ref, scale, out, m, n, k, stream):
        packed = entries.get((w, ws, ref, n, k))
        if packed is None or not 64 < m <= 4096:
            return original(a, w, ws, ref, scale, out, m, n, k, stream)
        w, ws = packed[0].data_ptr(), packed[1].data_ptr()
        code = native(
            a, w, ws, ref, scale, out, m, n, k, stream, 4 if m >= tn4_min else 2, 1
        )
        if code:
            raise RuntimeError(f"MXFP4 input projection failed with HIP status {code}")
        usage["preprocessed_calls"] += 1

    launch._coherence_input_candidate = True
    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "source"))
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hipcc", default="/opt/rocm/bin/hipcc")
    args = parser.parse_args()
    if args.action == "build":
        build(args.repo, args.output, args.hipcc)
    else:
        args.output.write_text(source_for_candidate(args.repo))

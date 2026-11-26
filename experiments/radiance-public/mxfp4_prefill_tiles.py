"""Experimental gfx1201 thin-tail dispatch for TP1 input GEMM.

The source retains the existing full-tile kernel as a separate specialization.
Thin blocks skip only the WMMAs of padding rows. Each live accumulator keeps its
original inputs, reduction order, epilogue and barriers.
"""

import argparse
import ctypes
import subprocess
from pathlib import Path

from mxfp4_fold_precision import replace_once
from mxfp4_prefill_candidate import sha256, source_for_candidate


def eligible_tail(rows):
    return 64 < rows < 1024 and 0 < rows % 256 <= 160


def source(root):
    hip = source_for_candidate(root)
    begin = hip.index("template <int TN, bool WPERM, int LBK, bool PSW, bool AHEAD>")
    end = hip.index("template <bool PSW, bool AHEAD, int TN> static void launch_input")
    kernel = hip[begin:end]
    header, body = kernel.split("int M, int N, int K) {", 1)
    header = (
        header.replace("bool AHEAD>", "bool AHEAD, bool PART>")
        .replace("__global__ __launch_bounds__(NTHREADS)", "__device__ __forceinline__")
        .replace("radiance_mxfp4_fp8_gemm_atiled", "coherence_prefill_tile_body")
    )
    body = replace_once(body, "  __shared__ unsigned char sW[BNF_T * LWSTR];", "")
    body = replace_once(body, "  __shared__ unsigned int sMag[32];", "")
    body = replace_once(
        body,
        "  const int Mt = (M + 15) >> 4;",
        """  const int Mt = (M + 15) >> 4;
  const int live = PART ? __builtin_amdgcn_readfirstlane(min(TM, max(0, Mt - (m0 / 16 + wm * TM)))) : TM;""",
    )
    anchor = "      for (int i = 0; i < TM; ++i)\n#pragma unroll\n        for (int j = 0; j < TN; ++j)\n          acc[i][j] = "
    body = replace_once(
        body, anchor, anchor.replace("++i)\n", "++i) {\n        if (PART && i >= live) break;\n")
    )
    body = replace_once(body, "wf[j], acc[i][j]);\n", "wf[j], acc[i][j]);\n      }\n")
    variant = header + "int M, int N, int K, unsigned char* sW, unsigned int* sMag) {" + body
    wrapper = r"""
template <int TN, bool PSW, bool AHEAD> __global__ __launch_bounds__(NTHREADS) void coherence_input_thin(
    const unsigned char* AT, const unsigned char* W, const unsigned char* Ws, const unsigned char* Wref,
    const float* As, __bf16* C, int M, int N, int K) {
  constexpr int LBK = TN == 2 ? 128 : 64;
  __shared__ unsigned char sW[BNF_OF(TN) * (LBK + PAD)];
  __shared__ unsigned int sMag[32];
  if ((int)(blockIdx.y + 1) * BMF > M)
    coherence_prefill_tile_body<TN,true,LBK,PSW,AHEAD,true>(AT,W,Ws,Wref,As,C,M,N,K,sW,sMag);
  else
    coherence_prefill_tile_body<TN,true,LBK,PSW,AHEAD,false>(AT,W,Ws,Wref,As,C,M,N,K,sW,sMag);
}
extern "C" int coherence_prefill_tiles(const unsigned char* a,const unsigned char* w,
    const unsigned char* ws,const unsigned char* ref,const float* scale,__bf16* out,
    int m,int n,int k,hipStream_t stream,int tn,int mode,int thin) {
  if (!thin || m >= 1024 || m % BMF == 0 || m % BMF > 160)
    return coherence_input_prefill(a,w,ws,ref,scale,out,m,n,k,stream,tn,mode);
  if (!a || !w || !ws || !ref || !scale || !out || m <= 64 || k != 5120
      || (n != 14336 && n != 16480 && n != 34816) || (tn != 2 && tn != 4) || (mode != 0 && mode != 1 && mode != 3)) return -1;
  dim3 grid((n + BNF_OF(tn)-1)/BNF_OF(tn), (m+BMF-1)/BMF);
#define LAUNCH(T,P,A) coherence_input_thin<T,P,A><<<grid,NTHREADS,0,stream>>>(a,w,ws,ref,scale,out,m,n,k)
  if (tn == 4) { if (mode == 3) LAUNCH(4,true,true); else if (mode) LAUNCH(4,true,false); else LAUNCH(4,false,false); }
  else { if (mode == 3) LAUNCH(2,true,true); else if (mode) LAUNCH(2,true,false); else LAUNCH(2,false,false); }
#undef LAUNCH
  return int(hipGetLastError());
}
"""
    return hip + variant + wrapper


def build(root, output):
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    src = output / "prefill-tiles.hip"
    src.write_text(source(root))
    subprocess.run(
        [
            "/opt/rocm/bin/hipcc",
            "-std=c++17",
            "-O3",
            "--offload-arch=gfx1201",
            "-fPIC",
            "-shared",
            "-cuid=coherence_tiles_" + sha256(src),
            str(src),
            "-o",
            str(output / "prefill-tiles.so"),
        ],
        check=True,
    )

    import json

    from qwen_r9700_lab.diagnostic_contract import seal

    receipt = seal(
        {
            "library": sha256(output / "prefill-tiles.so"),
            "source": sha256(src),
            "policy": sha256(__file__),
        }
    )
    (output / "tiles-build.json").write_text(json.dumps(receipt, indent=2) + "\n")


def load(path):
    native = ctypes.CDLL(str(path.resolve())).coherence_prefill_tiles
    native.argtypes = (
        [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_void_p] + [ctypes.c_int] * 3
    )
    native.restype = ctypes.c_int
    return native


_MEASURED_TAIL_ROWS = {129, 160, 320, 384, 513}


def install(worker, library, *, max_preprocessed_bytes=0, slab_ahead=False):
    """Enable the measured routes before graph capture on a disposable TP1 worker.

    When requested, the same fixed byte budget covers preprocessed weights and
    scale indices together. This hook replaces (and must precede graph capture
    in place of) the previous experimental preprocessing hook.
    """
    import json
    import os

    import radiance_mxfp4 as gemm
    import torch
    from mxfp4_prefill_candidate import SHAPES, fold_indices, pack_words

    from qwen_r9700_lab.diagnostic_contract import authenticate

    if max_preprocessed_bytes < 0:
        raise ValueError("negative preprocessing budget")
    library = Path(library)
    receipt = json.loads(library.with_name("tiles-build.json").read_text())
    authenticate(receipt)
    if (
        receipt["policy"] != sha256(__file__)
        or receipt["library"] != sha256(library)
        or receipt["source"] != sha256(library.with_suffix(".hip"))
    ):
        raise ValueError("prefill tile binary/source does not match its build receipt")
    alignment = worker._qwen_performance_repairs.manifest["prefill_alignment"]
    if (
        worker.vllm_config.parallel_config.tensor_parallel_size != 1
        or torch.cuda.get_device_properties(worker.device).gcnArchName.split(":")[0] != "gfx1201"
        or not gemm.WPERM
        or not alignment.get("input_tiles")
        or sha256(gemm._ext.__file__) != alignment["input_binary_sha256"]
    ):
        raise ValueError("expected the qualified TP1 gfx1201 input GEMM")
    original = gemm._ext.launch_at
    if getattr(original, "_coherence_input_candidate", False):
        raise RuntimeError("prefill projection dispatch already installed")
    threshold = int(os.getenv("RADIANCE_MXFP4_TN4_MIN_M", "2048"))
    if threshold != 2048:
        raise ValueError("tile policy requires the measured 2048-row baseline")
    reserve = worker.vllm_config.cache_config.kv_cache_memory_bytes
    if max_preprocessed_bytes and reserve is None:
        raise ValueError("preprocessing needs an explicit KV reserve")
    native = load(library)
    entries = {}
    usage = {
        "eligible_layers": 0,
        "preprocessed_bytes": 0,
        "preprocessed_layers": 0,
        "thin_calls": 0,
        "encoded_calls": 0,
        "library": receipt["library"],
    }
    for mod in worker.model_runner.model.modules():
        if not getattr(mod, "radiance_w4a8_ok", False):
            continue
        w, s, ref = mod.weight, mod.weight_scale, mod.radiance_wref
        # Fused modules can leave an empty placeholder on the old projection.
        # The fused projection that owns the live two-dimensional weight is
        # discovered separately through model.modules().
        if w.ndim != 2:
            continue
        n, k = w.shape[0], w.shape[1] * 2
        if (
            (n, k) not in SHAPES
            or s.shape != (k // 32, n)
            or ref.shape != (n,)
            or w.device.type != "cuda"
            or w.device.index != torch.cuda.current_device()
            or not all(
                t.dtype == torch.uint8 and t.is_contiguous() and t.device == w.device
                for t in (s, ref, w)
            )
        ):
            continue
        key = (w.data_ptr(), s.data_ptr(), ref.data_ptr(), n, k)
        if key in entries:
            continue
        packed = None
        size = w.nbytes + s.nbytes
        if usage["preprocessed_bytes"] + size <= max_preprocessed_bytes and torch.cuda.mem_get_info(
            w.device
        )[0] >= reserve + size + (32 << 20):
            indices = fold_indices(s, ref)
            if indices is not None:
                packed = pack_words(w), indices
                usage["preprocessed_bytes"] += size
                usage["preprocessed_layers"] += 1
        entries[key] = packed
        usage["eligible_layers"] += 1
    if not entries or (max_preprocessed_bytes and not usage["preprocessed_layers"]):
        raise RuntimeError("no projection fits the requested configuration")

    def dispatch(a, w, s, ref, scale, out, m, n, k, stream):
        key = (w, s, ref, n, k)
        if not 64 < m <= 4096 or key not in entries:
            return original(a, w, s, ref, scale, out, m, n, k, stream)
        packed = entries[key]
        default = 4 if m >= threshold else 2
        mode = (3 if slab_ahead else 1) if packed is not None else 0
        thin = m in _MEASURED_TAIL_ROWS
        if packed is None and not thin:
            return original(a, w, s, ref, scale, out, m, n, k, stream)
        if packed is not None:
            w, s = packed[0].data_ptr(), packed[1].data_ptr()
        status = native(
            a,
            w,
            s,
            ref,
            scale,
            out,
            m,
            n,
            k,
            stream,
            default,
            mode,
            int(thin),
        )
        if status:
            raise RuntimeError(f"prefill tiles HIP status {status}")
        usage["thin_calls"] += int(thin)
        usage["encoded_calls"] += int(packed is not None)

    dispatch._coherence_input_candidate = True
    torch.cuda.current_stream(worker.device).synchronize()
    gemm._ext.launch_at = dispatch
    return usage


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    build(args.repo, args.out)

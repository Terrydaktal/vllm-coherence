"""Parallel prefill output projections with the M1 split-K reduction order.

The input is the release's precision-correct folded MXFP4 kernel. This retains
its prefill tiles and weight/activation layouts, but computes the same four K
intervals as decode and combines their FP32 partials in the same order.
"""

import ctypes as c
import hashlib
import json
import subprocess
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import authenticate, seal, write_private


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_for_prefill(source, window=256, tn=2, slab=64, register_partials=False):
    if (
        window not in (128, 256, 512, 1024, 2048)
        or tn not in (2, 4)
        or slab not in (64, 128)
    ):
        raise ValueError("unsupported prefill projection geometry")
    source = source[
        : source.index(
            "// ---------------------------------------------------------------- decode path"
        )
    ]
    source = source.replace("#include <pybind11/pybind11.h>\n", "").replace(
        "namespace py = pybind11;\n", ""
    )
    # Edit each complete function independently; the original unsplit kernels
    # are not accidentally admitted through the new C interface.
    for name, loop_step, epilogue in (
        ("radiance_mxfp4_fp8_gemm_folded", "BK", "  float rf[TN];"),
        ("radiance_mxfp4_fp8_gemm_atiled", "LBK", "  // Epilogue: identical"),
    ):
        begin = source.index("void " + name + "(")
        end = source.index("\n}\n", begin) + 2
        body = source[begin:end]
        if register_partials:
            # Keep all four independently rounded accumulators and their
            # ordered FP32 merge, but eliminate the global partial matrices.
            acc_start = body.index("  floatx8 acc[TM][TN];")
            tail = body.index(epilogue)
            loop = f"  for (int k0 = 0; k0 < K; k0 += {loop_step}) {{"
            compute = body[acc_start:tail]
            if compute.count(loop) != 1:
                raise ValueError("prefill GEMM K-loop changed")
            compute = compute.replace(
                loop,
                f"  const int first_k = split * span, end_k = min(K, first_k + span);\n"
                f"  for (int k0 = first_k; k0 < end_k; k0 += {loop_step}) {{",
            )
            body = (
                body[:acc_start]
                + """
  floatx8 merged[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j)
      #pragma unroll
      for (int e = 0; e < 8; ++e) merged[i][j][e] = 0.0f;
  const int span = ((K / 128 + 3) / 4) * 128;
  #pragma unroll 1
  for (int split = 0; split < 4; ++split) {
"""
                + compute
                + """
    #pragma unroll
    for (int i = 0; i < TM; ++i)
      #pragma unroll
      for (int j = 0; j < TN; ++j)
        #pragma unroll
        for (int e = 0; e < 8; ++e) merged[i][j][e] += acc[i][j][e];
  }
"""
                + body[tail:].replace("acc[", "merged[")
            )
            source = source[:begin] + body + source[end:]
            continue
        body = body.replace("__bf16 *__restrict__ C", "float *__restrict__ C")
        loop = f"  for (int k0 = 0; k0 < K; k0 += {loop_step}) {{"
        if body.count(loop) != 1:
            raise ValueError("prefill GEMM K-loop changed")
        body = body.replace(
            loop,
            f"""  // Decode uses four contiguous ranges of 128-coefficient slabs.
  const int split = blockIdx.z;
  const int span = ((K / 128 + 3) / 4) * 128;
  const int first_k = split * span, end_k = min(K, first_k + span);
  for (int k0 = first_k; k0 < end_k; k0 += {loop_step}) {{""",
        )
        tail = body.index(epilogue)
        body = (
            body[:tail]
            + """  #pragma unroll
  for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j)
      #pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + wm * TM * 16 + i * 16 + kb8 + e;
        const int n = n0 + wn * TN * 16 + j * 16 + col;
        if (m < M && n < N) C[((size_t)split * M + m) * N + n] = acc[i][j][e];
      }
}"""
        )
        source = source[:begin] + body + source[end:]
    source += r"""
__global__ void coherence_prefill_reduce(const float* P, const unsigned char* ref,
    const float* scale, __bf16* out, int M, int N) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t size = size_t(M) * N;
  if (i >= size) return;
  float sum = 0.0f;
  #pragma unroll
  for (int split = 0; split < 4; ++split) sum += P[size_t(split) * size + i];
  const float rf = __int_as_float(int(ref[i % N]) << 23);
  out[i] = (__bf16)(sum * rf * scale[i / N]);
}
template<bool WP> void coherence_prefill_project_window(const unsigned char* a, const unsigned char* w,
    const unsigned char* ws, const unsigned char* ref, const float* scale, __bf16* out,
    float* partials, int M, int N, int K, bool tiled, hipStream_t stream) {
  dim3 block(NTHREADS), grid((N + BNF_OF(2) - 1) / BNF_OF(2), (M + BMF - 1) / BMF, 4);
  if (tiled)
    radiance_mxfp4_fp8_gemm_atiled<2,WP><<<grid,block,0,stream>>>(a,w,ws,ref,scale,partials,M,N,K);
  else
    radiance_mxfp4_fp8_gemm_folded<2,WP,false><<<grid,block,0,stream>>>(a,w,ws,ref,scale,partials,M,N,K);
  coherence_prefill_reduce<<<dim3((size_t(M)*N+255)/256),dim3(256),0,stream>>>(partials,ref,scale,out,M,N);
}
template<bool WP> void coherence_prefill_project(const unsigned char* a, const unsigned char* w,
    const unsigned char* ws, const unsigned char* ref, const float* scale, __bf16* out,
    float* partials, int M, int N, int K, bool tiled, hipStream_t stream) {
  // Bound the workspace by a whole number of 16-row activation tiles.
  // Register-partial builds write final BF16 results directly, without scratch.
  for (int first = 0; first < M; first += 256)
    coherence_prefill_project_window<WP>(a + size_t(first)*K, w, ws, ref,
        scale + first, out + size_t(first)*N, partials,
        min(256, M-first), N, K, tiled, stream);
}
extern "C" int coherence_prefill_gemm(const void* a, const void* w, const void* ws,
    const void* ref, const void* scale, void* out, void* partials, int M, int N, int K,
    int tiled, int wp, void* stream) {
  if (!a || !w || !ws || !ref || !scale || !out || !partials || M < 1 || M > 4096 || N != 5120 || K < 128 || K % 128 || (wp != 0 && wp != 1) || (tiled != 0 && tiled != 1)) return -1;
  if (wp)
    coherence_prefill_project<true>((const unsigned char*)a,(const unsigned char*)w,(const unsigned char*)ws,
      (const unsigned char*)ref,(const float*)scale,(__bf16*)out,(float*)partials,M,N,K,tiled,(hipStream_t)stream);
  else
    coherence_prefill_project<false>((const unsigned char*)a,(const unsigned char*)w,(const unsigned char*)ws,
      (const unsigned char*)ref,(const float*)scale,(__bf16*)out,(float*)partials,M,N,K,tiled,(hipStream_t)stream);
  return int(hipGetLastError());
}
"""
    source = source.replace("BNF_OF(2)", f"BNF_OF({tn})")
    source = source.replace(
        "radiance_mxfp4_fp8_gemm_atiled<2,WP>",
        f"radiance_mxfp4_fp8_gemm_atiled<{tn},WP,{slab}>",
    )
    source = source.replace(
        "radiance_mxfp4_fp8_gemm_folded<2,WP,false>",
        f"radiance_mxfp4_fp8_gemm_folded<{tn},WP,false>",
    )
    source = source.replace("first += 256", f"first += {window}")
    source = source.replace("min(256, M-first)", f"min({window}, M-first)")
    if register_partials:
        source = source.replace("/ BMF, 4);", "/ BMF, 1);")
        source = source.replace(
            ">(a,w,ws,ref,scale,partials,M,N,K);",
            ">(a,w,ws,ref,scale,out,M,N,K);",
        )
        source = source.replace(
            "  coherence_prefill_reduce<<<dim3((size_t(M)*N+255)/256),dim3(256),0,stream>>>(partials,ref,scale,out,M,N);\n",
            "",
        )
    return source


def build(
    parent, output, expected_sha256, window=256, tn=2, slab=64, register_partials=False
):
    parent, output = Path(parent), Path(output)
    if digest(parent) != expected_sha256:
        raise ValueError("parent GEMM source differs from the declared release")
    output.mkdir(mode=0o700)
    source = output / "aligned-prefill-gemm.hip"
    source.write_text(
        source_for_prefill(parent.read_text(), window, tn, slab, register_partials)
    )
    command = [
        "/opt/rocm/bin/hipcc",
        "-O3",
        "-std=c++17",
        "--offload-arch=gfx1201",
        "-shared",
        "-fPIC",
        "-cuid=coherence_prefill_gemm_" + digest(source),
        str(source),
        "-o",
        str(output / "candidate.so"),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=300, check=False
    )
    (output / "compiler.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    report = seal(
        {
            "status": "BUILT_UNTESTED",
            "kernel_abi": "coherence-prefill-m1-gemm-v2",
            "window": window,
            "tn": tn,
            "slab": slab,
            "register_partials": register_partials,
            "scratch_bytes": 4 if register_partials else 4 * window * 5120 * 4,
            "parent_sha256": expected_sha256,
            "generator_sha256": digest(__file__),
            "command": command,
            "files": {
                p.name: digest(p)
                for p in output.iterdir()
                if p.suffix in (".hip", ".so")
            },
        }
    )
    write_private(output / "build.json", report)
    return report


class AlignedPrefillGemm:
    def __init__(self, build):
        root = Path(build)
        self.manifest = json.loads((root / "build.json").read_text())
        authenticate(self.manifest)
        if (
            self.manifest["kernel_abi"] != "coherence-prefill-m1-gemm-v2"
            or self.manifest.get("window") not in (128, 256, 512, 1024, 2048)
            or self.manifest.get("tn", 2) not in (2, 4)
            or self.manifest.get("slab", 64) not in (64, 128)
            or type(self.manifest.get("register_partials", False)) is not bool
            or self.manifest.get("scratch_bytes")
            != (
                4
                if self.manifest.get("register_partials")
                else 4 * self.manifest["window"] * 5120 * 4
            )
        ):
            raise ValueError("unknown prefill projection ABI")
        for name, expected in self.manifest["files"].items():
            if Path(name).name != name or digest(root / name) != expected:
                raise ValueError("prefill projection build changed")
        self.library = c.CDLL(str(root / "candidate.so"))
        self.launch = self.library.coherence_prefill_gemm
        self.launch.argtypes = [c.c_void_p] * 7 + [c.c_int] * 5 + [c.c_void_p]
        self.launch.restype = c.c_int

    def __call__(self, q, scale, weight, weight_scale, ref, *, tiled=False, wperm=True):
        import torch
        from prefill_activation_tiles import pack

        if q.ndim != 2 or weight.ndim != 2:
            raise ValueError("projection requires two-dimensional input and weights")
        m, k = q.shape
        n = weight.shape[0]
        if not (
            q.device.type == "cuda"
            and q.dtype == torch.float8_e4m3fn
            and q.is_contiguous()
            and 1 <= m <= 4096
            and n == 5120
            and k >= 128
            and k % 128 == 0
            and weight.dtype == weight_scale.dtype == ref.dtype == torch.uint8
            and weight.shape == (n, k // 2)
            and weight_scale.shape == (k // 32, n)
            and all(t.is_contiguous() for t in (weight, weight_scale, ref))
            and scale.numel() == m
            and scale.dtype == torch.float32
            and scale.is_contiguous()
            and ref.numel() == n
            and all(t.device == q.device for t in (scale, weight, weight_scale, ref))
        ):
            raise ValueError(
                "projection outside admitted prefill contract: "
                + repr(
                    [
                        (tuple(t.shape), str(t.dtype), t.stride())
                        for t in (q, scale, weight, weight_scale, ref)
                    ]
                )
            )
        a = pack(q) if tiled else q
        out = torch.empty((m, n), device=q.device, dtype=torch.bfloat16)
        partials = torch.empty(
            (1,)
            if self.manifest.get("register_partials")
            else (4, min(m, self.manifest["window"]), n),
            device=q.device,
            dtype=torch.float32,
        )
        code = self.launch(
            a.data_ptr(),
            weight.data_ptr(),
            weight_scale.data_ptr(),
            ref.data_ptr(),
            scale.data_ptr(),
            out.data_ptr(),
            partials.data_ptr(),
            m,
            n,
            k,
            int(tiled),
            int(wperm),
            torch.cuda.current_stream().cuda_stream,
        )
        if code:
            raise RuntimeError(f"prefill projection launch failed: {code}")
        return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument(
        "--window", type=int, default=256, choices=(128, 256, 512, 1024, 2048)
    )
    parser.add_argument("--tn", type=int, default=2, choices=(2, 4))
    parser.add_argument("--slab", type=int, default=64, choices=(64, 128))
    parser.add_argument("--register-partials", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.parent,
                args.output,
                args.expected_sha256,
                args.window,
                args.tn,
                args.slab,
                args.register_partials,
            )
        )
    )

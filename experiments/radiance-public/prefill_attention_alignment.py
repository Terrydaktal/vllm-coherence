"""Prefill attention with the qualified per-query M1 reduction contract.

Queries share KV loads, but never softmax decisions or split boundaries. The
scratch window is bounded independently of prompt length. This is a buildable
candidate; deployment additionally requires operator and model qualification.
"""

import ctypes as c
import hashlib
import json
import subprocess
from pathlib import Path

from stock_m1_attention_shared import R4DArgs

from qwen_r9700_lab.diagnostic_contract import authenticate, seal, write_private


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError("prefill attention source anchor changed: " + old[:90])
    return source.replace(old, new)


def source_from_shared(source, warps=4, window=64):
    if warps not in (4, 8, 16, 24) or not 2 * warps <= window <= 128:
        raise ValueError("unsupported prefill tile")
    source = source[: source.index("template<int KVP> void launch_shared")]
    source = source.replace("qwen_stock_m1_shared", "coherence_prefill_m1")
    source = replace(
        source,
        "void coherence_prefill_m1_decode(const R4DArgs a, int splits)",
        "void coherence_prefill_m1_decode(const R4DArgs a, int splits, int tail)",
    )
    source = replace(
        source,
        "void coherence_prefill_m1_merge(const R4DArgs a, int splits, int tile)",
        "void coherence_prefill_m1_merge(const R4DArgs a, int splits, int tile, int tail)",
    )
    helper = """
// Exactly the single-sequence, four-KV-head M1 law, per logical query.
__device__ __forceinline__ int coherence_query_splits(int ctx) {
    const int tiles = (ctx + 15) / 16;
    return tiles >= 64 ? 32 : min(16, max(1, tiles));
}
"""
    source = replace(
        source,
        "template<int NWARPS, int TILE, int HEAD_DIM, int GQA, int BS, int KVP, int OPT>",
        helper
        + "\ntemplate<int NWARPS, int TILE, int HEAD_DIM, int GQA, int BS, int KVP, int OPT>",
    )
    source = replace(
        source,
        "const int total_ctx = a.seqused_k[0];",
        "const int total_ctx = a.seqused_k[0] - tail;",
    )
    source = replace(
        source,
        "const int first_ctx = total_ctx - a.q_len + 1;\n    const int group_start = 0;\n    const int group_rows = a.q_len;\n    const int ctx = total_ctx;",
        """const int group_start = blockIdx.z * (2 * NWARPS);
    const int group_rows = min(2 * NWARPS, a.q_len - group_start);
    const int first_ctx = total_ctx - a.q_len + group_start + 1;
    const int ctx = first_ctx + group_rows - 1;""",
    )
    begin = source.index("    const int ntl  = r4d_attn_tiles(ctx, TILE);")
    end = source.index("\n    const int pair =", begin)
    source = (
        source[:begin]
        + """    int t_lo = 0x7fffffff, t_hi = 0;
    // A tile may cross the 16-to-32 split transition. Take the actual union,
    // not an endpoint interpolation that assumes a monotone split count.
    for (int row = 0; row < group_rows; ++row) {
        const int n = r4d_attn_tiles(first_ctx + row, TILE);
        const int ns = coherence_query_splits(first_ctx + row);
        const int tps = (n + ns - 1) / ns;
        const int lo = sp * tps, hi = min(lo + tps, n);
        if (sp < ns && lo < hi) { t_lo = min(t_lo, lo); t_hi = max(t_hi, hi); }
    }
    if (t_lo >= t_hi) return;
"""
        + source[end:]
    )
    source = replace(
        source,
        "const int query_tps = r4d_attn_tps(query_ctx, TILE, splits);",
        "const int query_tps = r4d_attn_tps(query_ctx, TILE, coherence_query_splits(query_ctx));",
    )
    source = replace(
        source,
        "if (last_t_lo != t_lo && ti == last_t_lo && query_t_lo == last_t_lo)",
        "if (ti == query_t_lo)",
    )
    source = replace(
        source,
        """        if (first_t_hi != t_hi && ti + 1 == first_t_hi) {
            if (live && query_t_hi == first_t_hi && query_t_lo < query_t_hi)
                write_partial();
        }""",
        """        if (live && sp < coherence_query_splits(query_ctx) &&
            ti + 1 == query_t_hi && query_t_lo < query_t_hi)
            write_partial();""",
    )
    source = replace(
        source,
        "    if (live && query_t_hi == t_hi && query_t_lo < query_t_hi)\n        write_partial();",
        "",
    )
    source = replace(
        source,
        "const int ctx = a.seqused_k[seq] - a.q_len + (tok % a.q_len) + 1;",
        "const int ctx = a.seqused_k[seq] - tail - a.q_len + (tok % a.q_len) + 1;",
    )
    source = replace(
        source,
        "const int u = r4d_attn_used(ctx, tile, splits);",
        "const int u = r4d_attn_used(ctx, tile, coherence_query_splits(ctx));",
    )
    source += f"""
template<int KVP> void coherence_prefill_launch(const R4DArgs& all, hipStream_t stream) {{
  for (int first = 0; first < all.q_len; first += {window}) {{
    R4DArgs a = all;
    a.q_len = min({window}, all.q_len - first);
    a.q = static_cast<const uint16_t*>(all.q) + size_t(first) * 24 * 256;
    a.out = static_cast<uint16_t*>(all.out) + size_t(first) * 24 * 256;
    const int tail = all.q_len - first - a.q_len;
    coherence_prefill_m1_decode<{warps},16,256,6,16,KVP,(3430971 & ~32 & ~8)>
      <<<dim3(32,4,(a.q_len+{2 * warps - 1})/{2 * warps}),dim3({warps * 32}),0,stream>>>(a,32,tail);
    coherence_prefill_m1_merge<256,4,0>
      <<<dim3(a.q_len*24),dim3(256),32*sizeof(float),stream>>>(a,32,16,tail);
  }}
}}
extern "C" int coherence_prefill_attention(const R4DArgs* a, int kvp, void* stream) {{
  if (!a || !a->q || !a->kv || !a->out || !a->scratch || !a->block_table ||
      !a->seqused_k || a->num_seqs != 1 || a->q_len < 1 || a->q_len > 2048 || a->q_heads != 24 ||
      a->kv_heads != 4 || a->head_dim != 256 || a->block_size != 16 ||
      a->splits != 32 || kvp < 0 || kvp > 1) return -1;
  if (kvp) coherence_prefill_launch<1>(*a, reinterpret_cast<hipStream_t>(stream));
  else coherence_prefill_launch<0>(*a, reinterpret_cast<hipStream_t>(stream));
  return int(hipGetLastError());
}}
"""
    return source


def build(parent, output, warps=4, window=64):
    parent, output = Path(parent), Path(output)
    manifest = json.loads((parent / "build.json").read_text())
    authenticate(manifest)
    if manifest["kernel_abi"] != "coherence-attention-precision-v1":
        raise ValueError("requires qualified precision-repair source")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or digest(parent / name) != expected:
            raise ValueError("parent attention build changed")
    output.mkdir(mode=0o700)
    for name in ("r4d.h", "r4d_common.h", "r4d_dt16.h"):
        (output / name).write_bytes((parent / name).read_bytes())
    source = output / "aligned-prefill.hip"
    source.write_text(
        source_from_shared((parent / "shared.hip").read_text(), warps, window)
    )
    command = [
        "/opt/rocm/bin/hipcc",
        "-O3",
        "-std=c++17",
        "--offload-arch=gfx1201",
        "-shared",
        "-fPIC",
        "-ffp-contract=off",
        "-cuid=coherence_prefill_" + digest(source),
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
            "kernel_abi": "coherence-prefill-m1-attention-v1",
            "parent_sha256": digest(parent / "build.json"),
            "generator_sha256": digest(__file__),
            "warps": warps,
            "window": window,
            "scratch_bytes": window * 24 * 32 * 1032,
            "command": command,
            "files": {
                p.name: digest(p)
                for p in output.iterdir()
                if p.suffix in (".hip", ".h", ".so")
            },
        }
    )
    write_private(output / "build.json", report)
    return report


class AlignedPrefillAttention:
    def __init__(self, build):
        root = Path(build)
        self.manifest = json.loads((root / "build.json").read_text())
        authenticate(self.manifest)
        if self.manifest["kernel_abi"] != "coherence-prefill-m1-attention-v1":
            raise ValueError("unknown prefill attention ABI")
        if (
            self.manifest.get("warps") not in (4, 8, 16, 24)
            or not 2 * self.manifest["warps"] <= self.manifest.get("window", 0) <= 128
            or self.manifest.get("scratch_bytes")
            != self.manifest["window"] * 24 * 32 * 1032
        ):
            raise ValueError("prefill attention scratch contract changed")
        for name, expected in self.manifest["files"].items():
            if Path(name).name != name or digest(root / name) != expected:
                raise ValueError("prefill attention build changed")
        self.library = c.CDLL(str(root / "candidate.so"))
        self.launch = self.library.coherence_prefill_attention
        self.launch.argtypes = [c.POINTER(R4DArgs), c.c_int, c.c_void_p]
        self.launch.restype = c.c_int

    def __call__(self, q, kv, table, lengths, scratch, out, *, ks=None, vs=None):
        import torch

        if not (
            q.device.type == "cuda"
            and q.dtype == out.dtype == torch.bfloat16
            and q.ndim == 3
            and 1 <= q.shape[0] <= 2048
            and q.shape[1:] == (24, 256)
            and q.shape == out.shape
            and q.is_contiguous()
            and out.is_contiguous()
            and kv.ndim == 4
            and kv.shape[0] > 0
            and kv.shape[1:] == (4, 16, 512)
            and kv.dtype in (torch.uint8, torch.float8_e4m3fn, torch.bfloat16)
            and kv.stride(3) == 1
            and kv.stride(2) == 512
            and kv.stride(1) >= 16 * 512
            and kv.stride(0) >= 4 * kv.stride(1)
            and table.ndim == 2
            and table.shape[0] == 1
            and table.shape[1] > 0
            and table.is_contiguous()
            and lengths.shape == (1,)
            and lengths.is_contiguous()
            and table.dtype == lengths.dtype == torch.int32
            and scratch.dtype == torch.uint8
            and scratch.is_contiguous()
            and scratch.numel() >= self.manifest["scratch_bytes"]
            and all(t.device == q.device for t in (kv, table, lengths, scratch, out))
        ):
            raise ValueError("prefill attention input outside admitted contract")
        for scale in (ks, vs):
            if scale is not None and not (
                scale.shape == (4,)
                and scale.dtype == torch.float32
                and scale.device == q.device
                and scale.is_contiguous()
            ):
                raise ValueError("prefill attention scale representation changed")
        for other in (q, kv, table, lengths, scratch):
            if torch._C._overlaps(out, other):
                raise ValueError("prefill attention output aliases an input/workspace")
        args = R4DArgs(
            q.data_ptr(),
            kv.data_ptr(),
            table.data_ptr(),
            lengths.data_ptr(),
            out.data_ptr(),
            0 if ks is None else ks.data_ptr(),
            0 if vs is None else vs.data_ptr(),
            0,
            scratch.data_ptr(),
            1,
            q.shape[0],
            24,
            4,
            256,
            16,
            table.shape[1],
            kv.stride(0),
            kv.stride(1),
            1 / 16,
            32,
            253792,
        )
        status = self.launch(
            c.byref(args),
            int(kv.dtype == torch.bfloat16),
            torch.cuda.current_stream().cuda_stream,
        )
        if status:
            raise RuntimeError(f"prefill attention launch failed: {status}")
        return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warps", type=int, default=4)
    parser.add_argument("--window", type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(build(args.parent, args.output, args.warps, args.window)))

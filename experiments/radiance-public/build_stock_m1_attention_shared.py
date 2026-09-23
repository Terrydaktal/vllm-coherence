"""Build query-isolated, shared-KV attention from authenticated libr4d source."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from qwen_r9700_lab.conformance_topk import require
from qwen_r9700_lab.diagnostic_contract import seal, write_private

SOURCES = {
    "r4d_attn_decode_h256_gqa6.hip": (
        "c548b0d9d6d68fabc93fef2245fa2f7a48665ec9ee2e771f82f3ec7587da958c"
    ),
    "r4d.h": "c71bb6946f912d85afeab7fdd1410a368a87aa0e6479ac9641d68905d810bb1c",
    "r4d_common.h": "16db14fa42e7377c39641a3abea6d2ca524bb3a33e2a8afc55e9ccd3f5c0bacd",
    "r4d_dt16.h": "b61525f1dbb6b272642e0e1410a3d753d1136ed87eb07cefca512843096c4c24",
}


def transform(source):
    helper = """
// Unchanged r4d_qcvt from the pinned attention prefill translation unit.
template<class D, int FOLDQ, int F16>
__device__ __forceinline__ uint32_t r4d_qcvt(uint32_t w, float s) {
    if (!FOLDQ) return F16 ? D::from_bf16w(w) : w;
    return D::pk(__builtin_bit_cast(float, w << 16) * s,
                 __builtin_bit_cast(float, w & 0xffff0000u) * s);
}
"""
    require(
        source.count("#define O_F16") == 1, "attention helper insertion anchor changed"
    )
    source = source.replace("#define O_F16", helper + "\n#define O_F16")
    replacements = [
        (
            "const int sp = blockIdx.x, kvh = blockIdx.y, seq = blockIdx.z;\n"
            "    const int ctx = a.seqused_k[seq];\n    if (ctx <= 0) return;",
            """const int sp = blockIdx.x, kvh = blockIdx.y, seq = 0;
    const int total_ctx = a.seqused_k[0];
    if (total_ctx < a.q_len) return;
    const int first_ctx = total_ctx - a.q_len + 1;
    const int group_start = 0;
    const int group_rows = a.q_len;
    const int ctx = total_ctx;""",
        ),
        (
            "const int tps  = r4d_attn_tps(ctx, TILE, splits);\n"
            "    const int t_lo = sp * tps;\n"
            "    if (t_lo >= ntl) return;\n"
            "    const int t_hi = min(t_lo + tps, ntl);",
            """const int first_tps = r4d_attn_tps(first_ctx, TILE, splits);
    const int last_tps = r4d_attn_tps(ctx, TILE, splits);
    // Traverse the union once. Each query retains its own serial-M1 split
    // boundaries below, including when tiles-per-split increases. Splitting
    // the eight queries into two CTAs at every page boundary reread the full
    // context and created a second, context-dependent round-latency mode.
    const int t_lo = sp * first_tps;
    if (t_lo >= ntl) return;
    const int t_hi = min((sp + 1) * last_tps, ntl);
    const int first_t_hi = min((sp + 1) * first_tps, r4d_attn_tiles(first_ctx, TILE));
    const int last_t_lo = sp * last_tps;""",
        ),
        (
            "const int row = warp * 16 + c;\n"
            "    const int qi = row / GQA, hi = row - qi * GQA;\n"
            "    const bool live = (qi < a.q_len);\n"
            "    const int qrow = live ? qi : (a.q_len - 1);",
            """const int pair = (c / GQA) % 2;
    const int qi = warp * 2 + pair, hi = c % GQA;
    const bool live = (c < 2 * GQA && qi < group_rows);
    const int qrow = min(qi, group_rows - 1);
    const int query_ctx = first_ctx + qrow;
    const int query_ntl = r4d_attn_tiles(query_ctx, TILE);
    const int query_tps = r4d_attn_tps(query_ctx, TILE, splits);
    const int query_t_lo = sp * query_tps;
    const int query_t_hi = min(query_t_lo + query_tps, query_ntl);
    // Both halves of the fragment, all six heads, exactly one query.
    auto query_any = [&](bool predicate) {
        const unsigned mask = 0x003f003fu << (pair * GQA);
        return (static_cast<unsigned>(__ballot(predicate)) & mask) != 0;
    };""",
        ),
        (
            "((size_t)(seq * a.q_len + qrow) * a.q_heads + qhead)",
            "((size_t)(group_start + qrow) * a.q_heads + qhead)",
        ),
        (
            "const int klimit = ctx - a.q_len + qrow;",
            "const int klimit = ctx - group_rows + qrow;",
        ),
        (
            "const int k0 = ti * TILE;",
            """const int k0 = ti * TILE;
        // A split-size boundary can give queries different start tiles.
        // Discard any earlier union work before this query's first M1 tile.
        if (last_t_lo != t_lo && ti == last_t_lo && query_t_lo == last_t_lo) {
            m_ref = -3.0e38f;
            off = PSHIFT + 3.0e38f;
            l_i = 0.0f;
            #pragma unroll
            for (int dt = 0; dt < NKS; ++dt)
                acc[dt] = (v8f){0,0,0,0,0,0,0,0};
        }""",
        ),
        ("wave_any(kbase + 7 > klimit)", "query_any(kbase + 7 > klimit)"),
        ("wave_any(smax > m_ref + PGROW)", "query_any(smax > m_ref + PGROW)"),
        (
            "const int tok = seq * a.q_len + qrow;",
            "const int tok = group_start + qrow;",
        ),
        (
            "const int ctx = a.seqused_k[seq];",
            "const int ctx = a.seqused_k[seq] - a.q_len + (tok % a.q_len) + 1;",
        ),
    ]
    for before, after in replacements:
        require(source.count(before) == 1, "attention source patch anchor changed")
        source = source.replace(before, after)
    # Publish each query immediately after its last M1 tile. Later union tiles
    # cannot change this partial. This keeps the original WMMA accumulation
    # instructions, avoiding per-element selects on every tile. The two halves
    # of each query always enter this branch together; no CTA exits before the
    # remaining shared-memory barriers.
    begin = source.index("    if (!live) return;")
    end = source.index("\n}\n\n// Merge.", begin)
    partial = source[begin:end].replace("    if (!live) return;\n", "", 1)
    source = (
        source[:begin]
        + "    if (live && query_t_hi == t_hi && query_t_lo < query_t_hi)\n        write_partial();\n"
        + source[end:]
    )
    require(
        source.count("    int blk[NB];") == 1,
        "attention partial insertion anchor changed",
    )
    source = source.replace(
        "    int blk[NB];",
        "    auto write_partial = [&]() {\n" + partial + "\n    };\n\n    int blk[NB];",
    )
    tail = """        if (BTS) {
            #pragma unroll
            for (int i = 0; i < NB; ++i) blk[i] = blkn[i];
        }
    }"""
    require(source.count(tail) == 1, "attention tile completion anchor changed")
    source = source.replace(
        tail,
        tail[:-5]
        + """        if (first_t_hi != t_hi && ti + 1 == first_t_hi) {
            if (live && query_t_hi == first_t_hi && query_t_lo < query_t_hi)
                write_partial();
        }
    }""",
    )
    source = source.replace("r4d_attn_decode_kernel", "qwen_stock_m1_shared_decode")
    source = source.replace(
        "r4d_attn_splitkv_combine_kernel", "qwen_stock_m1_shared_merge"
    )
    return source


EXPORT = r"""
template<int KVP> void launch_shared(const R4DArgs& a, hipStream_t stream) {
  qwen_stock_m1_shared_decode<4,16,256,6,16,KVP,3430971>
      <<<dim3(a.splits, 4, 1), dim3(128), 0, stream>>>(a, a.splits);
  qwen_stock_m1_shared_merge<256,4,1>
      <<<dim3(8 * 24), dim3(256), a.splits * sizeof(float), stream>>>(a, a.splits, 16);
}
extern "C" int qwen_stock_m1_attention_shared(const R4DArgs* a, int kvp, void* stream) {
  if (!a || !a->q || !a->kv || !a->out || !a->scratch || !a->block_table ||
      !a->seqused_k || a->num_seqs != 1 || a->q_len != 8 || a->q_heads != 24 ||
      a->kv_heads != 4 || a->head_dim != 256 || a->block_size != 16 ||
      a->splits != 32 || a->max_ctx < 1024 || kvp < 0 || kvp > 1) return -1;
  if (kvp) launch_shared<1>(*a, reinterpret_cast<hipStream_t>(stream));
  else launch_shared<0>(*a, reinterpret_cast<hipStream_t>(stream));
  return int(hipGetLastError());
}
"""


def build(source_dir, output):
    output.mkdir(mode=0o700)
    for name, expected in SOURCES.items():
        raw = (source_dir / name).read_bytes()
        require(
            hashlib.sha256(raw).hexdigest() == expected,
            "upstream source binding mismatch",
        )
        if name.endswith(".hip"):
            raw = (transform(raw.decode()) + EXPORT).encode()
        (output / name).write_bytes(raw)
    source = output / "r4d_attn_decode_h256_gqa6.hip"
    command = [
        "/opt/rocm/bin/hipcc",
        "-O3",
        "-std=c++17",
        "--offload-arch=gfx1201",
        "-shared",
        "-fPIC",
        "-ffp-contract=off",
        "-Wall",
        "-Wextra",
        "-cuid=qwen_shared_attn_" + hashlib.sha256(source.read_bytes()).hexdigest(),
        str(source),
        "-o",
        str(output / "candidate.so"),
    ]
    done = subprocess.run(command, capture_output=True, text=True, timeout=240)
    (output / "compiler.log").write_text(done.stdout + done.stderr)
    report = seal(
        {
            "status": "BUILT_UNTESTED" if done.returncode == 0 else "BUILD_FAILED",
            "gpu_used": False,
            "command": command,
            "returncode": done.returncode,
            "upstream_sources": SOURCES,
            "kernel_abi": "qwen-stock-m1-shared-attention-v1",
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "binary_sha256": hashlib.sha256(
                (output / "candidate.so").read_bytes()
            ).hexdigest()
            if done.returncode == 0
            else None,
        }
    )
    write_private(output / "build.json", report)
    print(json.dumps(report), flush=True)
    return done.returncode


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    raise SystemExit(build(args.source_dir, args.output))

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


def source_from_shared(
    source,
    warps=4,
    window=64,
    kv_tile=16,
    shared_k=False,
    prefetch=4,
    query_first=False,
    stream_q=False,
    value_splits=1,
    memory_options_clear=0,
    adaptive_context=0,
    range_fastpath=False,
    prefetch_k=False,
    double_buffer=False,
):
    if (
        warps not in (1, 2, 4, 8, 16, 24)
        or not 2 * warps <= window <= 512
        or kv_tile not in (16, 32)
        or prefetch not in (1, 2, 4, 8)
        or value_splits not in (1, 2, 4)
        or memory_options_clear & ~(2 | 512 | 4096 | 262144 | 1048576 | 2097152)
        or adaptive_context not in (0, 8192, 16384, 32768)
    ):
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
    if kv_tile == 32:
        # Load two adjacent 16-key tiles together, but process their softmax and
        # probability/value updates separately and in exactly the M1 order.
        # Logical split boundaries remain in units of 16, including a query
        # starting/ending at the middle of a physical load tile.
        for old, new in (
            (
                "r4d_attn_tiles(first_ctx + row, TILE)",
                "r4d_attn_tiles(first_ctx + row, 16)",
            ),
            ("r4d_attn_tiles(query_ctx, TILE)", "r4d_attn_tiles(query_ctx, 16)"),
            (
                "r4d_attn_tps(query_ctx, TILE, coherence_query_splits(query_ctx))",
                "r4d_attn_tps(query_ctx, 16, coherence_query_splits(query_ctx))",
            ),
            (
                "for (int ti = t_lo; ti < t_hi; ++ti)",
                "for (int ti = t_lo; ti < t_hi; ti += MT)",
            ),
            ("const int k0 = ti * TILE;", "const int k0 = ti * 16;"),
            ("fetchV(t_lo * TILE, blk)", "fetchV(t_lo * 16, blk)"),
            (
                "blk[i] = bt[t_lo * NB + i]",
                "blk[i] = bt[min(t_lo + i, (ctx - 1) / BS)]",
            ),
            ("const int tb = (ti + 1) * NB;", "const int tb = ti + MT;"),
            ("blkn[i] = bt[tb + i]", "blkn[i] = bt[min(tb + i, (ctx - 1) / BS)]"),
        ):
            source = replace(source, old, new)
        source = source.replace("ti + 1 < t_hi", "ti + MT < t_hi")
        reset_begin = source.index("        if (ti == query_t_lo) {")
        reset_end = source.index("\n        int blkn[NB];", reset_begin)
        reset = source[reset_begin:reset_end].replace(
            "ti == query_t_lo", "ti + mt == query_t_lo"
        )
        source = source[:reset_begin] + source[reset_end:]
        source = replace(
            source,
            "for (int mt = 0; mt < MT; ++mt) {",
            "for (int mt = 0; mt < MT; ++mt) {\n" + reset,
        )
        publish = """        if (live && sp < coherence_query_splits(query_ctx) &&
            ti + 1 == query_t_hi && query_t_lo < query_t_hi)
            write_partial();"""
        source = replace(source, publish, "")
        source = replace(
            source,
            "        }\n        if (BTS) {",
            publish.replace("ti + 1", "ti + mt + 1")
            + "\n        }\n        if (BTS) {",
        )
    if stream_q:
        begin = source.index("    qfrag16 qf[NKS];")
        end = source.index("\n    const int klimit", begin)
        source = (
            source[:begin]
            + """    const uint16_t* qp = (const uint16_t*)a.q
        + ((size_t)(group_start + qrow) * a.q_heads + qhead) * HEAD_DIM;
    typedef unsigned qwords __attribute__((ext_vector_type(2)));
"""
            + source[end:]
        )
        source = replace(
            source,
            "for (int j = 0; j < PF; ++j) s = QD::wmma(kb[j], qf[g + j], s);",
            """for (int j = 0; j < PF; ++j) {
                    // Immutable query bytes stay hot in cache. Volatile vector
                    // loads prevent LICM from retaining all 64 query registers
                    // across the entire KV loop; the widening is unchanged.
                    const int t = g + j;
                    const qwords l0 = *(const volatile qwords*)(qp + 16*t + (CTG ? 8*h : 4*h));
                    const qwords h0 = *(const volatile qwords*)(qp + 16*t + (CTG ? 8*h+4 : 8+4*h));
                    const qfrag16 qj = QD::mk(
                        make_uint2(r4d_qcvt<QD,0,0>(l0[0],mul), r4d_qcvt<QD,0,0>(l0[1],mul)),
                        make_uint2(r4d_qcvt<QD,0,0>(h0[0],mul), r4d_qcvt<QD,0,0>(h0[1],mul)));
                    s = QD::wmma(kb[j], qj, s);
                }""",
        )
    if value_splits != 1:
        # Output channels are independent: repeat the identical QK/softmax for
        # each channel slice, retain its ordered PV accumulation, then publish
        # disjoint partials. Only slice zero owns the shared softmax statistics.
        decode_end = source.index("__global__", source.index("auto write_partial"))
        body, suffix = source[:decode_end], source[decode_end:]
        body = body.replace(
            "constexpr int NKS   = HEAD_DIM / 16;",
            f"constexpr int NKS = HEAD_DIM / 16;\n    constexpr int VDIM = HEAD_DIM / {value_splits}, NVS = VDIM / 16;",
        )
        body = body.replace(
            "kvh = blockIdx.y, seq = 0",
            f"kvh = blockIdx.y / {value_splits}, seq = 0;\n    const int v_first = (blockIdx.y % {value_splits}) * VDIM",
        )
        body = body.replace("(HEAD_DIM / (KVP ? 32 : 64))", "(VDIM / (KVP ? 32 : 64))")
        body = body.replace(
            "DG = HEAD_DIM / (KVP ? 32 : 64)", "DG = VDIM / (KVP ? 32 : 64)"
        )
        body = body.replace("sV16[HEAD_DIM * VSTR]", "sV16[VDIM * VSTR]")
        body = body.replace("+ HEAD_DIM + d0;", "+ HEAD_DIM + v_first + d0;")
        body = body.replace("v8f acc[NKS]", "v8f acc[NVS]")
        body = body.replace("t < NKS; ++t) acc[t]", "t < NVS; ++t) acc[t]")
        body = body.replace("dt < NKS", "dt < NVS")
        body = body.replace(
            "op = so + idx * HEAD_DIM;", "op = so + idx * HEAD_DIM + v_first;"
        )
        body = body.replace(
            "if (h == 0) { sm[idx]", "if (h == 0 && v_first == 0) { sm[idx]"
        )
        pv = body.index("const uint16_t* vbase")
        body = body[:pv] + body[pv:].replace("g < NKS", "g < NVS")
        source = body + suffix
    options = (3430971 & ~32 & ~8 & ~3072) | ({1: 0, 2: 1, 4: 2, 8: 3}[prefetch] << 10)
    if shared_k:
        options &= ~16
    options &= ~memory_options_clear
    grid = f"dim3(32,{4 * value_splits},(a.q_len+{2 * warps - 1})/{2 * warps})"
    if query_first:
        # Only the workgroup enumeration changes. Neighboring query groups read
        # the same KV partition close together, improving reuse in the GPU cache.
        source = replace(
            source, "const int sp = blockIdx.x,", "const int sp = blockIdx.z,"
        )
        source = replace(
            source,
            "const int group_start = blockIdx.z * (2 * NWARPS);",
            "const int group_start = blockIdx.x * (2 * NWARPS);",
        )
        grid = f"dim3((a.q_len+{2 * warps - 1})/{2 * warps},{4 * value_splits},32)"
    if prefetch_k:
        if not (shared_k and kv_tile == 16 and not stream_q):
            raise ValueError("K prefetch requires the shared TILE16 layout")
        begin = source.index("        if (KLDS) {\n            #pragma unroll 1")
        end = source.index("\n        if (!GPREV) fetchV", begin)
        source = source[:begin] + "        if (KLDS) storeK();" + source[end:]
        declarations = """
    constexpr int KREGS = (TILE * (HEAD_DIM / 8) + NTHR - 1) / NTHR;
    uint4 kreg[KREGS];
    auto fetchK = [&](int k0, const int* blks) {
        #pragma unroll
        for (int r = 0; r < KREGS; ++r) {
            const int i = tid + r * NTHR;
            if (i < TILE * (HEAD_DIM / 8)) {
                const int key = i / (HEAD_DIM / 8), ch = i % (HEAD_DIM / 8);
                const int kk = min(k0 + key, ctx - 1);
                const int kb = BTS ? blks[(kk-k0)/BS] : bt[kk/BS];
                const size_t o = size_t(kb)*a.kv_block_stride + size_t(kvh)*a.kv_head_stride
                               + size_t(kk%BS)*(2*HEAD_DIM) + ch*8;
                if (KVP) kreg[r] = *(const uint4*)((const uint16_t*)a.kv + o);
                else {
                    const uint2 raw = *(const uint2*)((const uint8_t*)a.kv + o);
                    kreg[r] = make_uint4(raw.x,raw.y,0,0);
                }
            }
        }
    };
    auto storeK = [&]() {
        #pragma unroll
        for (int r = 0; r < KREGS; ++r) {
            const int i = tid + r*NTHR;
            if (i < TILE*(HEAD_DIM/8)) {
                const int key=i/(HEAD_DIM/8), ch=i%(HEAD_DIM/8);
                uint32_t w4[4];
                const uint4 raw=kreg[r];
                if (KVP) {
                    w4[0]=QD::from_bf16w(raw.x); w4[1]=QD::from_bf16w(raw.y);
                    w4[2]=QD::from_bf16w(raw.z); w4[3]=QD::from_bf16w(raw.w);
                } else fp8x8_to_16x4w<QD>(make_uint2(raw.x,raw.y), w4);
                uint16_t* kd=&sK[key*KSTR];
                *(uint2*)(kd+ch*8)=make_uint2(w4[0],w4[1]);
                *(uint2*)(kd+ch*8+4)=make_uint2(w4[2],w4[3]);
            }
        }
    };
"""
        source = replace(
            source, "    qfrag16 qf[NKS];", declarations + "\n    qfrag16 qf[NKS];"
        )
        source = replace(
            source,
            "    if (GPREV) fetchV(t_lo * TILE, blk);",
            "    if (KLDS) fetchK(t_lo * TILE, blk);\n    if (GPREV) fetchV(t_lo * TILE, blk);",
        )
        source = replace(
            source,
            "        if (GPREV && ti + 1 < t_hi) fetchV(k0 + TILE, blkn);",
            "        if (KLDS && ti + 1 < t_hi) fetchK(k0 + TILE, blkn);\n        if (GPREV && ti + 1 < t_hi) fetchV(k0 + TILE, blkn);",
        )
        # The small-context adaptive path uses register K loads and never reads sK.
    if double_buffer:
        if not prefetch_k or warps != 8:
            raise ValueError("double buffering requires the eight-wave K-prefetch path")
        source = replace(
            source,
            "    __shared__ uint16_t sK[KLDS ? TILE * KSTR : 1];\n    __shared__ uint16_t sV16[HEAD_DIM * VSTR];",
            """    constexpr int DB = KLDS && NWARPS == 8;
    __shared__ uint16_t sK[KLDS ? (1 + DB) * TILE * KSTR : 1];
    __shared__ uint16_t sV16[(1 + DB) * HEAD_DIM * VSTR];
    int read_bank = 0, write_bank = 0;""",
        )
        source = replace(
            source, "&sK[key*KSTR]", "&sK[write_bank*TILE*KSTR + key*KSTR]"
        )
        source = replace(
            source,
            "&sK[(mt * 16 + c) * KSTR + 16 * t]",
            "&sK[read_bank*TILE*KSTR + (mt * 16 + c) * KSTR + 16 * t]",
        )
        for original in (
            "&sV16[(d0 + vj) * VSTR + kg * 8]",
            "&sV16[(d0 + 2 * vj)     * VSTR + kg * 8]",
            "&sV16[(d0 + 2 * vj + 1) * VSTR + kg * 8]",
        ):
            source = replace(
                source,
                original,
                original.replace("&sV16[", "&sV16[write_bank*HEAD_DIM*VSTR + "),
            )
        source = replace(
            source,
            "&sV16[c * VSTR + mt * 16]",
            "&sV16[read_bank*HEAD_DIM*VSTR + c * VSTR + mt * 16]",
        )
        source = replace(
            source,
            "    if (GPREV) fetchV(t_lo * TILE, blk);",
            """    if (GPREV || DB) fetchV(t_lo * TILE, blk);
    if (DB) {
        storeK(); storeV();
        if (LDSB) lds_barrier(); else __syncthreads();
    }""",
        )
        source = replace(
            source,
            """        if (LDSB) lds_barrier(); else __syncthreads();

        if (KLDS) storeK();
        if (!GPREV) fetchV(k0, blk);
        storeV();
        if (LDSB) lds_barrier(); else __syncthreads();""",
            """        if (!DB) {
            if (LDSB) lds_barrier(); else __syncthreads();
            if (KLDS) storeK();
            if (!GPREV) fetchV(k0, blk);
            storeV();
            if (LDSB) lds_barrier(); else __syncthreads();
        }""",
        )
        source = replace(
            source,
            "if (GPREV && ti + 1 < t_hi) fetchV(k0 + TILE, blkn);",
            "if ((GPREV || DB) && ti + 1 < t_hi) fetchV(k0 + TILE, blkn);",
        )
        source = replace(
            source,
            """        if (live && sp < coherence_query_splits(query_ctx) &&
            ti + 1 == query_t_hi && query_t_lo < query_t_hi)
            write_partial();""",
            """        if (live && sp < coherence_query_splits(query_ctx) &&
            ti + 1 == query_t_hi && query_t_lo < query_t_hi)
            write_partial();
        if (DB && ti + 1 < t_hi) {
            // The previous barrier retired every reader of the alternate bank.
            // This barrier publishes its replacement and retires this tile.
            write_bank = read_bank ^ 1;
            storeK(); storeV();
            if (LDSB) lds_barrier(); else __syncthreads();
            read_bank = write_bank;
        }""",
        )
    short_launch = ""
    if adaptive_context:
        if not (
            query_first
            and shared_k
            and warps == 8
            and kv_tile == 16
            and prefetch == 2
            and not stream_q
            and value_splits == 1
        ):
            raise ValueError(
                "adaptive attention requires the qualified long-context geometry"
            )
        source = replace(
            source,
            "template<int NWARPS, int TILE, int HEAD_DIM, int GQA, int BS, int KVP, int OPT>",
            "template<int NWARPS, int TILE, int HEAD_DIM, int GQA, int BS, int KVP, int OPT, bool QUERY_FIRST=true>",
        )
        source = replace(
            source,
            "const int sp = blockIdx.z,",
            "const int sp = QUERY_FIRST ? blockIdx.z : blockIdx.x,",
        )
        source = replace(
            source,
            "const int group_start = blockIdx.x * (2 * NWARPS);",
            "const int group_start = (QUERY_FIRST ? blockIdx.x : blockIdx.z) * (2 * NWARPS);",
        )
        # Same arithmetic; smaller groups/windows and register K loads avoid
        # underutilized waves and large scratch traffic on short contexts.
        short_launch = f"""
  if (all.max_ctx <= {adaptive_context}) {{
    for (int first = 0; first < all.q_len; first += 64) {{
      R4DArgs a = all;
      a.q_len = min(64, all.q_len - first);
      a.q = static_cast<const uint16_t*>(all.q) + size_t(first) * 24 * 256;
      a.out = static_cast<uint16_t*>(all.out) + size_t(first) * 24 * 256;
      const int tail = all.q_len - first - a.q_len;
      coherence_prefill_m1_decode<4,16,256,6,16,KVP,{options | 16},false>
        <<<dim3(32,4,(a.q_len+7)/8),dim3(128),0,stream>>>(a,32,tail);
      coherence_prefill_m1_merge<256,4,0>
        <<<dim3(a.q_len*24),dim3(256),32*sizeof(float),stream>>>(a,32,16,tail);
    }}
    return;
  }}
"""
    if range_fastpath:
        if not (warps == 8 and kv_tile == 16 and value_splits == 1):
            raise ValueError("range fast path requires at most 16 consecutive queries")
        source = replace(
            source,
            helper,
            helper
            + """
__device__ __forceinline__ int coherence_query_tps(int ctx) {
    const int n = (ctx + 15) / 16;
    return n >= 64 ? (n + 31) / 32 : (n >= 16 ? (n + 15) / 16 : 1);
}
""",
        )
        start = source.index("    int t_lo = 0x7fffffff, t_hi = 0;")
        end = source.index("\n    if (t_lo >= t_hi) return;", start)
        source = (
            source[:start]
            + """    int t_lo = 0x7fffffff, t_hi = 0;
    // At most 16 consecutive queries span at most two 16-key tile counts.
    // Both endpoints are checked independently, including the split-count jump.
    static_assert(NWARPS <= 8);
    const bool common_range = (first_ctx + 15) / 16 == (ctx + 15) / 16;
    for (int endpoint = 0; endpoint < 2; ++endpoint) {
        const int qc = endpoint ? ctx : first_ctx;
        const int n = (qc + 15) / 16;
        const int ns = coherence_query_splits(qc);
        const int tps = coherence_query_tps(qc);
        const int lo = sp * tps, hi = min(lo + tps, n);
        if (sp < ns && lo < hi) { t_lo = min(t_lo, lo); t_hi = max(t_hi, hi); }
    }"""
            + source[end:]
        )
        source = replace(
            source,
            "r4d_attn_tps(query_ctx, TILE, coherence_query_splits(query_ctx))",
            "coherence_query_tps(query_ctx)",
        )
        source = replace(
            source,
            "if (ti == query_t_lo) {",
            "if (!common_range && ti == query_t_lo) {",
        )
        marker = """        if (live && sp < coherence_query_splits(query_ctx) &&
            ti + 1 == query_t_hi && query_t_lo < query_t_hi)
            write_partial();
    }"""
        source = replace(
            source,
            marker,
            marker.replace("if (live", "if (!common_range && live")
            + """
    if (common_range && live && sp < coherence_query_splits(query_ctx) && query_t_lo < query_t_hi)
        write_partial();""",
        )
    source += f"""
template<int KVP> void coherence_prefill_launch(const R4DArgs& all, hipStream_t stream) {{
{short_launch}
  for (int first = 0; first < all.q_len; first += {window}) {{
    R4DArgs a = all;
    a.q_len = min({window}, all.q_len - first);
    a.q = static_cast<const uint16_t*>(all.q) + size_t(first) * 24 * 256;
    a.out = static_cast<uint16_t*>(all.out) + size_t(first) * 24 * 256;
    const int tail = all.q_len - first - a.q_len;
    coherence_prefill_m1_decode<{warps},{kv_tile},256,6,16,KVP,{options}>
      <<<{grid},dim3({warps * 32}),0,stream>>>(a,32,tail);
    coherence_prefill_m1_merge<256,4,0>
      <<<dim3(a.q_len*24),dim3(256),32*sizeof(float),stream>>>(a,32,16,tail);
  }}
}}
extern "C" int coherence_prefill_attention(const R4DArgs* a, int kvp, void* stream) {{
  if (!a || !a->q || !a->kv || !a->out || !a->scratch || !a->block_table ||
      !a->seqused_k || a->num_seqs != 1 || a->q_len < 1 || a->q_len > 4096 || a->q_heads != 24 ||
      a->kv_heads != 4 || a->head_dim != 256 || a->block_size != 16 ||
      a->splits != 32 || kvp < 0 || kvp > 1) return -1;
  if (kvp) coherence_prefill_launch<1>(*a, reinterpret_cast<hipStream_t>(stream));
  else coherence_prefill_launch<0>(*a, reinterpret_cast<hipStream_t>(stream));
  return int(hipGetLastError());
}}
"""
    return source


def build(
    parent,
    output,
    warps=4,
    window=64,
    kv_tile=16,
    shared_k=False,
    prefetch=4,
    query_first=False,
    stream_q=False,
    value_splits=1,
    memory_options_clear=0,
    adaptive_context=0,
    range_fastpath=False,
    prefetch_k=False,
    double_buffer=False,
):
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
        source_from_shared(
            (parent / "shared.hip").read_text(),
            warps,
            window,
            kv_tile,
            shared_k,
            prefetch,
            query_first,
            stream_q,
            value_splits,
            memory_options_clear,
            adaptive_context,
            range_fastpath,
            prefetch_k,
            double_buffer,
        )
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
            "kv_load_tile": kv_tile,
            "shared_k": shared_k,
            "prefetch": prefetch,
            "query_first": query_first,
            "stream_q": stream_q,
            "value_splits": value_splits,
            "memory_options_clear": memory_options_clear,
            "adaptive_context": adaptive_context,
            "range_fastpath": range_fastpath,
            "prefetch_k": prefetch_k,
            "double_buffer": double_buffer,
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
            self.manifest.get("warps") not in (1, 2, 4, 8, 16, 24)
            or self.manifest.get("kv_load_tile", 16) not in (16, 32)
            or self.manifest.get("prefetch", 4) not in (1, 2, 4, 8)
            or self.manifest.get("value_splits", 1) not in (1, 2, 4)
            or self.manifest.get("memory_options_clear", 0)
            & ~(2 | 512 | 4096 | 262144 | 1048576 | 2097152)
            or self.manifest.get("adaptive_context", 0) not in (0, 8192, 16384, 32768)
            or type(self.manifest.get("range_fastpath", False)) is not bool
            or type(self.manifest.get("prefetch_k", False)) is not bool
            or type(self.manifest.get("double_buffer", False)) is not bool
            or type(self.manifest.get("shared_k", False)) is not bool
            or type(self.manifest.get("query_first", False)) is not bool
            or type(self.manifest.get("stream_q", False)) is not bool
            or not 2 * self.manifest["warps"] <= self.manifest.get("window", 0) <= 512
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

    def __call__(
        self,
        q,
        kv,
        table,
        lengths,
        scratch,
        out,
        *,
        ks=None,
        vs=None,
        context_limit=253792,
    ):
        import torch

        if not (
            q.device.type == "cuda"
            and q.dtype == out.dtype == torch.bfloat16
            and q.ndim == 3
            and 1 <= q.shape[0] <= 4096
            and type(context_limit) is int
            and q.shape[0] <= context_limit <= 253792
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
            context_limit,
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
    parser.add_argument("--kv-tile", type=int, default=16, choices=(16, 32))
    parser.add_argument("--shared-k", action="store_true")
    parser.add_argument("--prefetch", type=int, default=4, choices=(1, 2, 4, 8))
    parser.add_argument("--query-first", action="store_true")
    parser.add_argument("--stream-q", action="store_true")
    parser.add_argument("--value-splits", type=int, default=1, choices=(1, 2, 4))
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.parent,
                args.output,
                args.warps,
                args.window,
                args.kv_tile,
                args.shared_k,
                args.prefetch,
                args.query_first,
                args.stream_q,
                args.value_splits,
            )
        )
    )

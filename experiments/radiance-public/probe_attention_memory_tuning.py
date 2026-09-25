"""Qualify memory-only attention tuning against the exact current release.

The K/V representation, causal mask, split boundaries, softmax update choices,
WMMA ordering and full-precision partials are unchanged. Build without a GPU;
then compare outputs and partial state before recording graph replay timings.
"""

import argparse
import ctypes
import hashlib
import json
import re
import shutil
import statistics
import subprocess
from pathlib import Path

BASE_OPT = 3430971 & ~32 & ~8
OPTIONS = {
    "control": BASE_OPT,
    "pf1": BASE_OPT & ~3072,
    "pf2": (BASE_OPT & ~3072) | 1024,
    "pf8": BASE_OPT | 3072,
    "klds_pf2": ((BASE_OPT & ~3072) | 1024) & ~16,
    "klds_pf4": BASE_OPT & ~16,
    "pf2_no_prefetch": ((BASE_OPT & ~3072) | 1024) & ~1048576,
    "no_sched_barrier": BASE_OPT & ~4096,
}
SECOND_SWEEP = {
    "control": BASE_OPT,
    "pf2": (BASE_OPT & ~3072) | 1024,
    "pf2_no_sched": ((BASE_OPT & ~3072) | 1024) & ~4096,
    "pf1_no_sched": BASE_OPT & ~(3072 | 4096),
    "reload_q_pf1": BASE_OPT & ~3072,
    "reload_q_pf2": (BASE_OPT & ~3072) | 1024,
    "reload_q_pf4": BASE_OPT,
}
THIRD_SWEEP = {
    "control": BASE_OPT,
    "pf2": (BASE_OPT & ~3072) | 1024,
    "load32_pf2": (BASE_OPT & ~3072) | 1024,
    "load32_pf4": BASE_OPT,
    "load64_pf2": (BASE_OPT & ~3072) | 1024,
}
FOURTH_SWEEP = {
    "control": BASE_OPT,
    "pf2": (BASE_OPT & ~3072) | 1024,
    "packed3_pf2": (BASE_OPT & ~3072) | 1024,
    "packed3_pf4": BASE_OPT,
    "packed3_prefetch_pf2": (BASE_OPT & ~3072) | 1024,
    "packed3_reload_q_pf2": (BASE_OPT & ~3072) | 1024,
}
FIFTH_SWEEP = {
    "control": BASE_OPT,
    "pf2": (BASE_OPT & ~3072) | 1024,
    "packed3_pf2": (BASE_OPT & ~3072) | 1024,
    "packed3_late_v_pf1": BASE_OPT & ~3072,
    "packed3_late_v_pf2": (BASE_OPT & ~3072) | 1024,
    "packed3_resident8_pf2": (BASE_OPT & ~3072) | 1024,
    "packed3_resident12_pf2": (BASE_OPT & ~3072) | 1024,
    "packed3_late_v_resident8_pf2": (BASE_OPT & ~3072) | 1024,
}
SIXTH_SWEEP = {
    "control": BASE_OPT,
    "pf2": (BASE_OPT & ~3072) | 1024,
    "pvsplit2_pf1": BASE_OPT & ~3072,
    "pvsplit2_pf2": (BASE_OPT & ~3072) | 1024,
    "pvsplit2_pf4": BASE_OPT,
    "pvsplit4_pf2": (BASE_OPT & ~3072) | 1024,
}
SEVENTH_SWEEP = {
    "control": BASE_OPT,
    "pf2": (BASE_OPT & ~3072) | 1024,
    "rawk_pf2": ((BASE_OPT & ~3072) | 1024) & ~16,
    "rawk_pf4": BASE_OPT & ~16,
    "rawk_pad8_pf2": ((BASE_OPT & ~3072) | 1024) & ~16,
    "packed3_rawk_pf2": ((BASE_OPT & ~3072) | 1024) & ~16,
}
EIGHTH_SWEEP = {
    "control": BASE_OPT,
    "packed3_pf2": (BASE_OPT & ~3072) | 1024,
    "sharedq_pf2": (BASE_OPT & ~3072) | 1024,
    "packed3_sharedq_pf1": BASE_OPT & ~3072,
    "packed3_sharedq_pf2": (BASE_OPT & ~3072) | 1024,
    "packed3_sharedq_pf4": BASE_OPT,
}
NINTH_SWEEP = {
    "control": BASE_OPT,
    "packed3_pf2": (BASE_OPT & ~3072) | 1024,
    "qk8_pf2": (BASE_OPT & ~3072) | 1024,
    "qk32_pf2": (BASE_OPT & ~3072) | 1024,
    "packed3_qk8_pf2": (BASE_OPT & ~3072) | 1024,
    "packed3_qk32_pf2": (BASE_OPT & ~3072) | 1024,
}
TENTH_SWEEP = {
    "control": BASE_OPT,
    "pf2": (BASE_OPT & ~3072) | 1024,
    "exactq_pf2": (BASE_OPT & ~3072) | 1024,
    "exactq_pf4": BASE_OPT,
    "packed3_exactq_pf2": (BASE_OPT & ~3072) | 1024,
}
ELEVENTH_SWEEP = {
    "control": BASE_OPT,
    "pf2": (BASE_OPT & ~3072) | 1024,
    "exactq_whole_pf2": (BASE_OPT & ~3072) | 1024,
    "exactq_whole_pf4": BASE_OPT,
    "packed3_exactq_whole_pf2": (BASE_OPT & ~3072) | 1024,
}
TWELFTH_SWEEP = {
    "control": BASE_OPT,
    "packed3_pf2": (BASE_OPT & ~3072) | 1024,
    "warpvalues_pf1": BASE_OPT & ~3072,
    "warpvalues_pf2": (BASE_OPT & ~3072) | 1024,
    "warpvalues_shareqk_pf2": (BASE_OPT & ~3072) | 1024,
}
THIRTEENTH_SWEEP = {
    "control": BASE_OPT,
    "packed3_pf2": (BASE_OPT & ~3072) | 1024,
    "rawpref4_pf2": (BASE_OPT & ~3072) | 1024,
    "rawpref8_pf2": (BASE_OPT & ~3072) | 1024,
    "rawpref16_pf2": (BASE_OPT & ~3072) | 1024,
    "packed3_rawpref4_pf2": (BASE_OPT & ~3072) | 1024,
    "packed3_rawpref8_pf2": (BASE_OPT & ~3072) | 1024,
    "packed3_headmajor_pf2": (BASE_OPT & ~3072) | 1024,
}
FOURTEENTH_SWEEP = {
    "control": BASE_OPT,
    "stage32_pf2": (BASE_OPT & ~3072) | 1024,
    "stage32_pf4": BASE_OPT,
    "stage64_pf2": (BASE_OPT & ~3072) | 1024,
    "packed3_stage32_pf2": (BASE_OPT & ~3072) | 1024,
}
FIFTEENTH_SWEEP = {
    "control": BASE_OPT,
    "probpass_v64_pf2": (BASE_OPT & ~3072) | 1024,
    "probpass_v128_pf2": (BASE_OPT & ~3072) | 1024,
}


def separate_probability_pass(source, value_width):
    """Split the identical serial attention recurrence across two kernels.

    The first pass retains the Q/K WMMA chain, softmax, rescale decisions and
    both rounded probability terms. The second pass consumes those exact bits
    and executes the original ordered PV updates on independent value columns.
    It retains all original query-specific split boundaries and FP32 partials.
    """
    if value_width not in (64, 128):
        raise ValueError("unsupported independent value width")
    start = source.index(
        "template<int NWARPS, int TILE, int HEAD_DIM, int GQA, int BS, int KVP, int OPT>"
    )
    end = source.index("// Merge.", start)
    original = source[start:end]
    nrow = "(size_t)a.num_seqs * a.q_len * a.q_heads * splits"
    common = f"""    const size_t original_partial_bytes = ({nrow}) * (HEAD_DIM * sizeof(float) + 2 * sizeof(float));
    const size_t probability_rows = (size_t)((a.max_ctx + 15) / 16) * 4 * 8 * 6;
    uint16_t* probability = (uint16_t*)((uint8_t*)a.scratch + original_partial_bytes);
    float2* rescale = (float2*)(probability + probability_rows * 32);
"""

    def remove_once(text, begin, finish):
        lo = text.index(begin)
        hi = text.index(finish, lo)
        return text[:lo] + text[hi:]

    # Preserve the arithmetic statements verbatim and move only their results.
    prepare = original.replace(
        "void qwen_stock_m1_shared_decode", "void qwen_prepare_probability"
    )
    prepare = prepare.replace(
        "    const int klimit", common + "    const int klimit", 1
    )
    prepare = remove_once(prepare, "    const int vg =", "    auto write_partial =")
    prepare = remove_once(prepare, "    v8f acc[NKS];", "    auto write_partial =")
    lo = prepare.index("    auto write_partial =")
    hi = prepare.index("\n    int blk[NB];", lo)
    prepare = (
        prepare[:lo]
        + f"""    auto write_partial = [&]() {{
        const size_t idx = (size_t)(qrow * a.q_heads + qhead) * splits + sp;
        const size_t rows = {nrow};
        float* sm = (float*)a.scratch + rows * HEAD_DIM;
        float* sl = sm + rows;
        const float lrow = l_i + swap16(l_i);
        if (h == 0) {{ sm[idx] = m_ref; sl[idx] = lrow; }}
    }};
"""
        + prepare[hi:]
    )
    reset = """            #pragma unroll
            for (int dt = 0; dt < NKS; ++dt)
                acc[dt] = (v8f){0,0,0,0,0,0,0,0};
"""
    if prepare.count(reset) != 1:
        raise ValueError("attention accumulator reset changed")
    prepare = prepare.replace(reset, "")
    for line in (
        "    if (GPREV) fetchV(t_lo * TILE, blk);\n",
        "        if (!GPREV) fetchV(k0, blk);\n",
        "        storeV();\n",
        "        if (GPREV && ti + 1 < t_hi) fetchV(k0 + TILE, blkn);\n",
    ):
        if prepare.count(line) != 1:
            raise ValueError("attention value staging changed")
        prepare = prepare.replace(line, "")
    prepare = prepare.replace(
        "        if (LDSB) lds_barrier(); else __syncthreads();",
        "        if constexpr (KLDS) { if (LDSB) lds_barrier(); else __syncthreads(); }",
    )
    branch = "            if (query_any(smax > m_ref + PGROW)) {"
    prepare = prepare.replace(
        branch,
        """            const size_t probability_row = ((size_t)ti * 4 + kvh) * 48 + qi * 6 + hi;
            const bool did_rescale = query_any(smax > m_ref + PGROW);
            float saved_alpha = 1.0f;
            if (did_rescale) {""",
    )
    scale = """                #pragma unroll
                for (int dt = 0; dt < NKS; ++dt)
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) acc[dt][e] *= alpha;
"""
    if prepare.count(scale) != 1:
        raise ValueError("attention accumulator rescale changed")
    prepare = prepare.replace(scale, "                saved_alpha = alpha;\n")
    lo = prepare.index("                if (DOT2) {")
    hi = prepare.index("\n            }\n        }", lo)
    prepare = (
        prepare[:lo]
        + """                if (live) {
                    uint16_t* p = probability + probability_row * 32 + 8 * h;
                    *(uint4*)p = make_uint4(pw[0], pw[1], pw[2], pw[3]);
                    *(uint4*)(p + 16) = make_uint4(pl[0], pl[1], pl[2], pl[3]);
                    if (h == 0) rescale[probability_row] = make_float2(saved_alpha, did_rescale ? 1.0f : 0.0f);
                }"""
        + prepare[hi:]
    )

    consume = original.replace(
        "void qwen_stock_m1_shared_decode", "void qwen_consume_probability"
    )
    consume = consume.replace(
        "    constexpr int NKS   = HEAD_DIM / 16;",
        f"    constexpr int VW = {value_width};\n    constexpr int NKS = VW / 16;",
    )
    consume = consume.replace("(HEAD_DIM / (KVP ? 32 : 64))", "(VW / (KVP ? 32 : 64))")
    consume = consume.replace("sV16[HEAD_DIM * VSTR]", "sV16[VW * VSTR]")
    consume = remove_once(consume, "    qfrag16 qf[NKS];", "    const int klimit")
    consume = consume.replace(
        "    const int klimit",
        common + "    const int value_begin = blockIdx.z * VW;\n    const int klimit",
        1,
    )
    consume = consume.replace(
        "constexpr int DG = HEAD_DIM /", "constexpr int DG = VW /"
    )
    consume = consume.replace("+ HEAD_DIM + d0;", "+ HEAD_DIM + value_begin + d0;")
    lo = consume.index("    auto write_partial =")
    hi = consume.index("\n    int blk[NB];", lo)
    consume = (
        consume[:lo]
        + """    auto write_partial = [&]() {
        const size_t idx = (size_t)(qrow * a.q_heads + qhead) * splits + sp;
        float* op = (float*)a.scratch + idx * HEAD_DIM + value_begin;
        #pragma unroll
        for (int dt = 0; dt < NKS; ++dt) *(v8f*)(op + dt * 16 + 8 * h) = acc[dt];
    };
"""
        + consume[hi:]
    )
    consume = remove_once(
        consume, "        if (KLDS) {\n", "        if (!GPREV) fetchV"
    )
    lo = consume.index("            const uint8_t* grow = nullptr;")
    hi = consume.index("                const uint16_t* vbase =", lo)
    consume = (
        consume[:lo]
        + """            const size_t probability_row = ((size_t)ti * 4 + kvh) * 48 + qrow * 6 + hi;
            const float2 adjust = rescale[probability_row];
            if (adjust.y != 0.0f) {
                #pragma unroll
                for (int dt = 0; dt < NKS; ++dt)
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) acc[dt][e] *= adjust.x;
            }
            {
                const uint16_t* p = probability + probability_row * 32 + 8 * h;
                const uint4 high = *(const uint4*)p;
                const uint4 low = *(const uint4*)(p + 16);
                const frag16 p16 = D::mk(make_uint2(high.x, high.y), make_uint2(high.z, high.w));
                const frag16 p16lo = D::mk(make_uint2(low.x, low.y), make_uint2(low.z, low.w));
"""
        + consume[hi:]
    )
    launch_start = source.index("  qwen_stock_m1_shared_decode<4,16,256")
    launch_end = source.index("  qwen_stock_m1_shared_merge", launch_start)
    launch = source[launch_start:launch_end]
    prepare_launch = launch.replace(
        "qwen_stock_m1_shared_decode", "qwen_prepare_probability"
    )
    consume_launch = launch.replace(
        "qwen_stock_m1_shared_decode", "qwen_consume_probability"
    ).replace("dim3(a.splits, 4, 1)", f"dim3(a.splits, 4, {256 // value_width})")
    source = (
        source[:launch_start] + prepare_launch + consume_launch + source[launch_end:]
    )
    return source[:start] + prepare + consume + source[start:]


def group_attention_staging(source, tile):
    """Stage several original 16-key tiles without changing their arithmetic.

    Split boundaries, rescaling decisions, partial publication and the merge
    retain the original 16-key units. Only K/V staging and its barriers use a
    wider unit. In particular, a query may start or end inside that unit.
    """
    if tile not in (32, 64):
        raise ValueError("unsupported staging width")
    start = source.index("void qwen_stock_m1_shared_decode")
    end = source.index("// Merge.", start)
    body = source[start:end]
    # All split calculations retain exactly the reference tile width.
    body = body.replace("r4d_attn_tiles(ctx, TILE)", "r4d_attn_tiles(ctx, 16)")
    for context in ("first_ctx", "ctx", "query_ctx"):
        body = body.replace(
            f"r4d_attn_tps({context}, TILE, splits)",
            f"r4d_attn_tps({context}, 16, splits)",
        ).replace(f"r4d_attn_tiles({context}, TILE)", f"r4d_attn_tiles({context}, 16)")
    reset_start = body.index("        // A split-size boundary")
    reset_end = body.index("        int blkn[NB];", reset_start)
    reset = body[reset_start:reset_end].replace(
        "ti == last_t_lo", "ti + mt == last_t_lo"
    )
    body = body[:reset_start] + body[reset_end:]
    publish_start = body.index("        if (first_t_hi != t_hi &&")
    publish_end = body.index("\n    }", publish_start)
    publish = body[publish_start:publish_end].replace(
        "ti + 1 == first_t_hi", "ti + mt + 1 == first_t_hi"
    )
    body = body[:publish_start] + body[publish_end:]
    anchor = "        }\n        if (BTS) {"
    if body.count(anchor) != 1:
        raise ValueError("inner attention loop boundary changed")
    body = body.replace(anchor, publish + "\n" + anchor)
    inner = "        for (int mt = 0; mt < MT; ++mt) {"
    if body.count(inner) != 1:
        raise ValueError("inner attention loop changed")
    body = body.replace(
        inner,
        inner + "\n            if (ti + mt >= t_hi) break;\n" + reset,
    )
    body = body.replace(
        "for (int ti = t_lo; ti < t_hi; ++ti)",
        "for (int ti = t_lo; ti < t_hi; ti += MT)",
    ).replace("const int k0 = ti * TILE;", "const int k0 = ti * 16;")
    body = body.replace("fetchV(t_lo * TILE, blk)", "fetchV(t_lo * 16, blk)")
    body = body.replace("ti + 1 < t_hi", "ti + MT < t_hi")
    body = body.replace("const int tb = (ti + 1) * NB;", "const int tb = ti + MT;")
    body = body.replace("bt[t_lo * NB + i]", "bt[min(t_lo + i, (ctx - 1) / BS)]")
    body = body.replace("bt[tb + i]", "bt[min(tb + i, (ctx - 1) / BS)]")
    source = source[:start] + body + source[end:]
    marker = (
        "shared_decode<3,16,256"
        if "shared_decode<3,16,256" in source
        else "shared_decode<4,16,256"
    )
    if source.count(marker) != 1:
        raise ValueError("attention launch geometry changed")
    return source.replace(marker, marker.replace(",16,256", f",{tile},256"))


def prefetch_raw_key_fragments(source, depth):
    """Prefetch compact FP8 bytes, widening immediately before each WMMA.

    Keep every WMMA and its serial order. The unchanged BF16-KV path remains
    available; only the exact FP8-to-BF16 conversion changes its schedule.
    """
    if depth not in (4, 8, 16):
        raise ValueError("unsupported raw key depth")
    start = source.index(
        "            #pragma unroll\n            for (int g = 0; g < NKS; g += PF)"
    )
    end = source.index("            float sc[8];", start)
    original = source[start:end]
    replacement = f"""            if constexpr (!KVP && !KLDS && CTG) {{
                #pragma unroll
                for (int g = 0; g < NKS; g += {depth}) {{
                    uint2 raw[{depth}];
                    #pragma unroll
                    for (int j = 0; j < {depth}; ++j)
                        raw[j] = *(const uint2*)(grow + (g + j) * 16 + 8 * h);
                    if (SGB) sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < {depth}; ++j) {{
                        uint32_t w4[4];
                        fp8x8_to_16x4w<QD>(raw[j], w4);
                        const qfrag16 kj = QD::mk(make_uint2(w4[0], w4[1]), make_uint2(w4[2], w4[3]));
                        s = QD::wmma(kj, qf[g + j], s);
                    }}
                }}
            }} else {{
{original}
            }}
"""
    return source[:start] + replacement + source[end:]


def reorder_attention_grid(source):
    old = "    const int sp = blockIdx.x, kvh = blockIdx.y, seq = 0;"
    if source.count(old) != 1 or source.count("dim3(a.splits, 4, 1)") != 1:
        raise ValueError("attention dispatch grid changed")
    return source.replace(
        old, "    const int sp = blockIdx.y, kvh = blockIdx.x, seq = 0;"
    ).replace("dim3(a.splits, 4, 1)", "dim3(4, a.splits, 1)")


def split_value_waves(source, *, share_scores=False):
    """Split independent PV columns between two sets of four waves.

    All waves retain the original split boundaries and serial FP32 accumulator
    operations. They cooperatively stage the full V tile only once. Optionally
    only the first four waves evaluate QK; the second four receive the exact
    eight FP32 scores through LDS before the unchanged online softmax update.
    """
    body, tail = source.split("// Merge. PF16=1", 1)

    def change(old, new, count=1):
        nonlocal body
        if body.count(old) != count:
            raise ValueError(f"value-wave anchor changed: {old[:70]}")
        body = body.replace(old, new)

    change(
        "    constexpr int NKS   = HEAD_DIM / 16;",
        "    constexpr int NVS = HEAD_DIM / 32;\n"
        "    constexpr int NKS   = HEAD_DIM / 16;",
    )
    change(
        "    const int c = lane & 15, h = lane >> 4;",
        "    const int c = lane & 15, h = lane >> 4;\n"
        "    const int vstart = (warp / 4) * (HEAD_DIM / 2);",
    )
    change("qi = warp * 2 + pair", "qi = (warp % 4) * 2 + pair")
    change("v8f acc[NKS];", "v8f acc[NVS];")
    change("t < NKS; ++t) acc[t]", "t < NVS; ++t) acc[t]")
    change("dt < NKS; ++dt", "dt < NVS; ++dt", 4)
    change("op = so + idx * HEAD_DIM;", "op = so + idx * HEAD_DIM + vstart;", 2)
    change("if (h == 0) { sm[idx]", "if (h == 0 && warp < 4) { sm[idx]", 2)
    change("vbase = &sV16[c * VSTR", "vbase = &sV16[(vstart + c) * VSTR")
    start = body.index("                const uint16_t* vbase = &sV16")
    body = body[:start] + body[start:].replace("g < NKS; g += PF", "g < NVS; g += PF")
    if share_scores:
        change(
            "    __shared__ uint16_t sV16[HEAD_DIM * VSTR];",
            "    __shared__ uint16_t sV16[HEAD_DIM * VSTR];\n"
            "    __shared__ float shared_scores[8][128];",
        )
        start = body.index("    qfrag16 qf[NKS];")
        end = body.index("\n    const int klimit", start)
        query = body[start:end].replace("    {", "    if (warp < 4) {", 1)
        body = body[:start] + query + body[end:]
        start = body.index("            const uint8_t* grow = nullptr;")
        end = body.index("            float smax = sc[0];", start)
        original = body[start:end].replace("            float sc[8];\n", "")
        transfer = (
            "            float sc[8];\n"
            "            if (warp < 4) {\n"
            + original
            + "                #pragma unroll\n"
            "                for (int e = 0; e < 8; ++e) shared_scores[e][tid] = sc[e];\n"
            "            }\n"
            "            lds_barrier();\n"
            "            #pragma unroll\n"
            "            for (int e = 0; e < 8; ++e) sc[e] = shared_scores[e][tid % 128];\n"
        )
        body = body[:start] + transfer + body[end:]
    if tail.count("qwen_stock_m1_shared_decode<4,") != 1:
        raise ValueError("attention wave dispatch changed")
    tail = tail.replace(
        "qwen_stock_m1_shared_decode<4,", "qwen_stock_m1_shared_decode<8,"
    )
    tail = tail.replace("dim3(128), 0, stream", "dim3(256), 0, stream")
    return body + "// Merge. PF16=1" + tail


def guarded_half_queries(source, *, whole_dot_branch=False):
    """Use F16 operands only when the input BF16 values fit F16 exactly.

    A whole wave must contain only signed zero or normal finite F16 values.
    Every BF16 value in [2**-14, 65280] fits F16 without rounding; FP8 finite
    K values also fit. Query scale remains after the original dot product.
    Other query waves and BF16 KV use the original BF16 instruction path.
    Native equality tests are still required for the two WMMA instructions.
    """
    marker = "\n    const int klimit = ctx - group_rows + qrow;"
    if source.count(marker) != 1:
        raise ValueError("query initialization boundary changed")
    code = r"""
    static_assert(!KLDS, "exact-query experiment requires direct K loads");
    bool exact_query = true;
    #pragma unroll
    for (int t = 0; t < NKS; ++t) {
        const uint4 raw = __builtin_bit_cast(uint4, qf[t]);
        const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned low = words[j] & 0x7fffu;
            const unsigned high = (words[j] >> 16) & 0x7fffu;
            exact_query &= (low == 0 || (low >= 0x3880u && low <= 0x477fu));
            exact_query &= (high == 0 || (high >= 0x3880u && high <= 0x477fu));
        }
    }
    const bool use_f16_qk = !KVP && __all(exact_query);
    if (use_f16_qk) {
        #pragma unroll
        for (int t = 0; t < NKS; ++t) {
            const uint4 raw = __builtin_bit_cast(uint4, qf[t]);
            const uint4 converted = make_uint4(
                DT16<1>::from_bf16w(raw.x), DT16<1>::from_bf16w(raw.y),
                DT16<1>::from_bf16w(raw.z), DT16<1>::from_bf16w(raw.w));
            qf[t] = __builtin_bit_cast(qfrag16, converted);
        }
    }
"""
    source = source.replace(marker, code + marker)
    if whole_dot_branch:
        start = source.index("            const uint8_t* grow = nullptr;")
        end = source.index("            float sc[8];", start)
        original = source[start:end].replace(
            "v8f s = (v8f){0,0,0,0,0,0,0,0};",
            "s = (v8f){0,0,0,0,0,0,0,0};",
        )
        # No workgroup barrier exists in this direct-K block. Dispatch once
        # per dot product, retaining the original 16-wide WMMA order within
        # either specialization. Each wave still rejoins before PV barriers.
        fast = original.replace("qfrag16", "DT16<1>::frag").replace("QD::", "DT16<1>::")
        fast = fast.replace("fp8x8_to_16x4w<QD>", "fp8x8_to_16x4w<DT16<1>>")
        fast = fast.replace("qf[g + j]", "__builtin_bit_cast(DT16<1>::frag, qf[g + j])")
        block = (
            "            v8f s;\n            if (use_f16_qk) {\n"
            + fast
            + "            } else {\n"
            + original
            + "            }\n"
        )
        return source[:start] + block + source[end:]
    old = "                        fp8x8_to_16x4w<QD>(raw, w4);"
    if source.count(old) != 1:
        raise ValueError("direct K conversion changed")
    source = source.replace(
        old,
        "                        if (use_f16_qk) fp8x8_to_16x4w<DT16<1>>(raw, w4);\n"
        "                        else fp8x8_to_16x4w<QD>(raw, w4);",
    )
    old = "for (int j = 0; j < PF; ++j) s = QD::wmma(kb[j], qf[g + j], s);"
    if source.count(old) != 1:
        raise ValueError("QK WMMA sequence changed")
    return source.replace(
        old,
        """for (int j = 0; j < PF; ++j) {
                    if (use_f16_qk) s = DT16<1>::wmma(
                        __builtin_bit_cast(DT16<1>::frag, kb[j]),
                        __builtin_bit_cast(DT16<1>::frag, qf[g+j]), s);
                    else s = QD::wmma(kb[j], qf[g+j], s);
                }""",
    )


def precompute_scores(source, tiles_per_cta):
    """Move the unchanged QK WMMA sequence into a separate, ordered launch.

    Scores are stored as FP32 before the original scale/mask/softmax. The
    consumer keeps its original online transitions and FP32 split partials.
    This experiment requires extra scratch after the original partial array.
    """
    if tiles_per_cta not in (8, 32):
        raise ValueError("unsupported QK launch geometry")
    qstart = source.index("    qfrag16 qf[NKS];")
    qend = source.index("\n    const int klimit", qstart)
    query = source[qstart:qend]
    cstart = source.index("            const uint8_t* grow = nullptr;")
    cend = source.index("            float sc[8];", cstart)
    computation = source[cstart:cend]
    helper = r"""
template<int KVP, int NTILE>
__global__ __launch_bounds__(128)
void qwen_precompute_qk(const R4DArgs a) {
    constexpr int NKS=16, HEAD_DIM=256, CTG=1, PF=2, SGB=1;
    constexpr int KLDS=0, BTS=1, BS=16, KSTR=264;
    typedef DT16<0> QD;
    typedef typename QD::frag qfrag16;
    uint16_t* sK = nullptr;
    const int lane=threadIdx.x & 31, warp=threadIdx.x >> 5;
    const int c=lane & 15, h=lane >> 4;
    const int qi=warp*2+(c/6)%2, hi=c%6;
    const bool live=c<12;
    const int ctx=a.seqused_k[0], kvh=blockIdx.y;
    const int first=blockIdx.x*NTILE;
    if (first*16>=ctx || ctx<a.q_len) return;
    const int qrow=qi, group_start=0, qhead=kvh*6+hi;
    const float kdesc=a.k_descale ? a.k_descale[kvh] : 1.0f;
    const float mul=a.scale*kdesc*1.44269504089f;
    const int* bt=a.block_table;
    float* scores=(float*)((uint8_t*)a.scratch + 8*24*32*(256*4+8));
QUERY_FRAGMENT
    for (int ti=first; ti<min(first+NTILE,(ctx+15)/16); ++ti) {
        const int k0=ti*16, mt=0;
        const int blk[1]={bt[k0/16]};
SCORE_FRAGMENT
        if (live) {
            #pragma unroll
            for (int e=0;e<8;++e)
                scores[(((size_t)ti*4+kvh)*8*6+qi*6+hi)*16+8*h+e]=s[e];
        }
    }
}
""".replace("QUERY_FRAGMENT", query).replace("SCORE_FRAGMENT", computation)
    replacement = """            v8f s;
            const float* scores = (const float*)((const uint8_t*)a.scratch
                + 8*24*32*(256*4+8));
            #pragma unroll
            for (int e=0; e<8; ++e)
                s[e] = scores[(((size_t)ti*4+kvh)*8*6+qi*6+hi)*16+8*h+e];
"""
    source = source[:cstart] + replacement + source[cend:]
    source = source[:qstart] + source[qend:]
    marker = "template<int NWARPS, int TILE, int HEAD_DIM, int GQA, int BS, int KVP, int OPT>"
    source = source.replace(marker, helper + "\n" + marker, 1)
    marker = (
        "template<int KVP> void launch_shared(const R4DArgs& a, hipStream_t stream) {"
    )
    if source.count(marker) != 1:
        raise ValueError("attention launch entry changed")
    return source.replace(
        marker,
        marker
        + f"""
  qwen_precompute_qk<KVP,{tiles_per_cta}>
      <<<dim3((a.max_ctx+16*{tiles_per_cta}-1)/(16*{tiles_per_cta}),4),
         dim3(128),0,stream>>>(a);""",
    )


def share_query_fragments(source):
    """Keep immutable BF16 queries in LDS instead of long-lived VGPRs."""
    start = source.index("    qfrag16 qf[NKS];")
    end = source.index("\n    const int klimit", start)
    load = """    constexpr int QSTR = HEAD_DIM + 8;
    __shared__ uint16_t sQuery[8 * GQA * QSTR];
    for (int i = tid; i < 8 * GQA * (HEAD_DIM / 8); i += NTHR) {
        const int qr = i / (HEAD_DIM / 8), col8 = i % (HEAD_DIM / 8);
        const int token = min(qr / GQA, group_rows - 1);
        const int head = kvh * GQA + qr % GQA;
        const uint16_t* qp = (const uint16_t*)a.q
            + ((size_t)(group_start + token) * a.q_heads + head) * HEAD_DIM;
        *(uint4*)(&sQuery[qr * QSTR + col8 * 8]) = *(const uint4*)(qp + col8 * 8);
    }
    __syncthreads();
    const uint16_t* sq = &sQuery[(qrow * GQA + hi) * QSTR];
"""
    source = source[:start] + load + source[end:]
    anchor = "for (int j = 0; j < PF; ++j) s = QD::wmma(kb[j], qf[g + j], s);"
    if source.count(anchor) != 1:
        raise ValueError("shared-query arithmetic anchor changed")
    return source.replace(
        anchor,
        """for (int j = 0; j < PF; ++j) {
                    const int t = g + j;
                    const uint2 l0 = *(const uint2*)(sq + 16 * t + (CTG ? 8 * h : 4 * h));
                    const uint2 h0 = *(const uint2*)(sq + 16 * t + (CTG ? 8 * h + 4 : 8 + 4 * h));
                    const qfrag16 qj = QD::mk(
                        make_uint2(r4d_qcvt<QD, 0, 0>(l0.x, mul), r4d_qcvt<QD, 0, 0>(l0.y, mul)),
                        make_uint2(r4d_qcvt<QD, 0, 0>(h0.x, mul), r4d_qcvt<QD, 0, 0>(h0.y, mul)));
                    s = QD::wmma(kb[j], qj, s);
                }""",
    )


def share_raw_keys(source, pad=16):
    """Stage raw FP8 key bytes once, retaining the exact per-lane widening."""
    if pad not in (8, 16):
        raise ValueError("raw key pad must preserve vector alignment")
    width = 16 if pad == 16 else 8
    vec = "uint4" if width == 16 else "uint2"
    source = source.replace(
        "    __shared__ uint16_t sK[KLDS ? TILE * KSTR : 1];",
        f"    constexpr int RKSTR = HEAD_DIM + {pad};\n"
        "    __shared__ uint16_t sK[KLDS && KVP ? TILE * KSTR : 1];\n"
        "    __shared__ uint8_t sKRaw[KLDS && !KVP ? TILE * RKSTR : 1];",
    )
    start = source.index("        if (KLDS) {\n            #pragma unroll 1")
    end = source.index("        if (!GPREV) fetchV", start)
    original = source[start:end].replace("if (KLDS)", "if (KLDS && KVP)", 1)
    raw = f"""        if (KLDS && !KVP) {{
            #pragma unroll 1
            for (int i = tid; i < TILE * (HEAD_DIM / {width}); i += NTHR) {{
                const int key = i / (HEAD_DIM / {width});
                const int ch = i % (HEAD_DIM / {width});
                const int kk = min(k0 + key, ctx - 1);
                const int kb = BTS ? blk[(kk - k0) / BS] : bt[kk / BS];
                const size_t o = (size_t)kb * a.kv_block_stride
                    + (size_t)kvh * a.kv_head_stride + (size_t)(kk % BS) * (2 * HEAD_DIM);
                *({vec}*)(&sKRaw[key * RKSTR + ch * {width}]) =
                    *(const {vec}*)((const uint8_t*)a.kv + o + ch * {width});
            }}
        }}
"""
    source = source[:start] + original + raw + source[end:]
    anchor = """                    if (KLDS) {
                        const uint16_t* kp"""
    if source.count(anchor) != 1:
        raise ValueError("shared key operand anchor changed")
    source = source.replace(
        anchor,
        """                    if (KLDS && !KVP) {
                        const uint8_t* kp = &sKRaw[(mt * 16 + c) * RKSTR + 16 * t];
                        const uint2 raw = CTG ? *(const uint2*)(kp + 8 * h)
                            : make_uint2(*(const uint32_t*)(kp + 4 * h),
                                         *(const uint32_t*)(kp + 8 + 4 * h));
                        uint32_t w4[4];
                        fp8x8_to_16x4w<QD>(raw, w4);
                        kb[j] = QD::mk(make_uint2(w4[0], w4[1]), make_uint2(w4[2], w4[3]));
                    } else if (KLDS) {
                        const uint16_t* kp""",
    )
    return source


def split_value_columns(source, parts):
    """Parallelize independent output channels without splitting any reduction.

    Each CTA recomputes the same QK/softmax sequence but owns disjoint PV
    channels. This lowers accumulator register pressure and raises the CTA
    count. Only part zero publishes scalar max/sum, so even identical-value
    data races are excluded. Logical KV split boundaries do not change.
    """
    if parts not in (2, 4):
        raise ValueError("unsupported independent-channel partition")
    body, tail = source.split("// Merge. PF16=1", 1)
    body = body.replace(
        "    constexpr int NKS   = HEAD_DIM / 16;",
        f"    constexpr int VCH = HEAD_DIM / {parts};\n"
        "    constexpr int NVS = VCH / 16;\n"
        "    constexpr int NKS   = HEAD_DIM / 16;",
    )
    if "constexpr int VCH" not in body:
        raise ValueError("channel geometry anchor changed")
    body = body.replace(
        "(TILE / 8) * (HEAD_DIM / (KVP ? 32 : 64))",
        "(TILE / 8) * (VCH / (KVP ? 32 : 64))",
    )
    body = body.replace("sV16[HEAD_DIM * VSTR]", "sV16[VCH * VSTR]")
    body = body.replace(
        "    const int sp = blockIdx.x, kvh = blockIdx.y, seq = 0;",
        "    const int sp = blockIdx.x, kvh = blockIdx.y, seq = 0;\n"
        "    const int vstart = blockIdx.z * VCH;",
    )
    body = body.replace("constexpr int DG = HEAD_DIM /", "constexpr int DG = VCH /")
    body = body.replace("+ HEAD_DIM + d0;", "+ HEAD_DIM + vstart + d0;")
    body = body.replace("v8f acc[NKS];", "v8f acc[NVS];")
    body = body.replace("t < NKS; ++t) acc[t]", "t < NVS; ++t) acc[t]")
    body = body.replace("dt < NKS; ++dt", "dt < NVS; ++dt")
    body = body.replace(
        "op = so + idx * HEAD_DIM;", "op = so + idx * HEAD_DIM + vstart;"
    )
    body = body.replace(
        "if (h == 0) { sm[idx]", "if (h == 0 && blockIdx.z == 0) { sm[idx]"
    )
    anchor = "                const uint16_t* vbase = &sV16"
    start = body.index(anchor)
    pv = body[start:]
    if pv.count("g < NKS; g += PF") != 1:
        raise ValueError("PV reduction anchor changed")
    body = body[:start] + pv.replace("g < NKS; g += PF", "g < NVS; g += PF")
    if tail.count("dim3(a.splits, 4, 1)") != 1:
        raise ValueError("attention CTA grid changed")
    tail = tail.replace("dim3(a.splits, 4, 1)", f"dim3(a.splits, 4, {parts})")
    return body + "// Merge. PF16=1" + tail


def partial_query_residency(source, count):
    """Keep some query fragments, reload the rest without changing their bits."""
    if count not in (8, 12):
        raise ValueError("unsupported query residency")
    start = source.index("    qfrag16 qf[NKS];")
    end = source.index("\n    const int klimit", start)
    block = source[start:end]
    block = block.replace("qf[NKS]", f"qf[{count}]").replace(
        "t < NKS; ++t", f"t < {count}; ++t"
    )
    source = source[:start] + block + source[end:]
    anchor = "for (int j = 0; j < PF; ++j) s = QD::wmma(kb[j], qf[g + j], s);"
    if source.count(anchor) != 1:
        raise ValueError("query arithmetic anchor changed")
    return source.replace(
        anchor,
        f"""for (int j = 0; j < PF; ++j) {{
                    const int t = g + j;
                    qfrag16 qj;
                    if (t < {count}) qj = qf[t];
                    else {{
                        const uint16_t* qp = (const uint16_t*)a.q
                            + ((size_t)(group_start + qrow) * a.q_heads + qhead) * HEAD_DIM;
                        const uint2 l0 = *(const uint2*)(qp + 16 * t + (CTG ? 8 * h : 4 * h));
                        const uint2 h0 = *(const uint2*)(qp + 16 * t + (CTG ? 8 * h + 4 : 8 + 4 * h));
                        qj = QD::mk(
                            make_uint2(r4d_qcvt<QD, 0, 0>(l0.x, mul), r4d_qcvt<QD, 0, 0>(l0.y, mul)),
                            make_uint2(r4d_qcvt<QD, 0, 0>(h0.x, mul), r4d_qcvt<QD, 0, 0>(h0.y, mul)));
                    }}
                    s = QD::wmma(kb[j], qj, s);
                }}""",
    )


def combine_packed_value_barrier(source):
    """Publish V and the cross-wave rescale votes at the same block barrier."""
    start = source.index("    auto query_any = [&](bool predicate) {")
    end = source.index("    const int qhead", start)
    predicate = source[start:end]
    source = source[:start] + source[end:]
    stage = """        if (!GPREV) fetchV(k0, blk);
        storeV();
        if (LDSB) lds_barrier(); else __syncthreads();
        if (GPREV && ti + 1 < t_hi) fetchV(k0 + TILE, blkn);
"""
    if source.count(stage) != 1:
        raise ValueError("V staging/barrier anchor changed")
    source = source.replace(stage, "")
    # This transformation only admits the original TILE=16, one inner tile.
    predicate = predicate.replace(
        "__ballot(predicate)", "__ballot(smax > m_ref + PGROW)"
    )
    predicate = predicate.replace(
        "        lds_barrier();",
        "        if (!GPREV) fetchV(k0, blk);\n"
        "        storeV();\n"
        "        if (LDSB) lds_barrier(); else __syncthreads();\n"
        "        if (GPREV && ti + 1 < t_hi) fetchV(k0 + TILE, blkn);",
    )
    predicate = predicate.replace("    auto query_any = [&](bool predicate) {\n", "")
    predicate = predicate.replace("        return ", "        const bool rescale = ")
    predicate = predicate.replace("    };\n", "")
    anchor = "            if (query_any(smax > m_ref + PGROW)) {"
    if source.count(anchor) != 1:
        raise ValueError("online rescale anchor changed")
    return source.replace(anchor, predicate + "            if (rescale) {")


def pack_query_columns(source, *, prefetch=False, reload_query=False):
    """Pack 48 independent query/head columns into three full WMMA waves.

    The original arrangement leaves four columns of each wave unused. Two
    query groups now cross a wave boundary, so their shared softmax-rescale
    predicate must be collected across waves. Head-local predicates would
    silently change the arithmetic; this version preserves all six heads.
    """
    old = """    const int pair = (c / GQA) % 2;
    const int qi = warp * 2 + pair, hi = c % GQA;
    const bool live = (c < 2 * GQA && qi < group_rows);"""
    new = """    static_assert(NWARPS == 3 && GQA == 6, "packed query geometry");
    const int column = warp * 16 + c;
    const int qi = column / GQA, hi = column % GQA;
    const bool live = qi < group_rows;"""
    if source.count(old) != 1:
        raise ValueError("query column layout changed")
    source = source.replace(old, new)
    old = """    auto query_any = [&](bool predicate) {
        const unsigned mask = 0x003f003fu << (pair * GQA);
        return (static_cast<unsigned>(__ballot(predicate)) & mask) != 0;
    };"""
    new = """    __shared__ unsigned sQueryFlags[3];
    auto query_any = [&](bool predicate) {
        const unsigned votes = static_cast<unsigned>(__ballot(predicate));
        if (lane == 0) sQueryFlags[warp] = votes | (votes >> 16);
        lds_barrier();
        const unsigned long long columns =
            (static_cast<unsigned long long>(sQueryFlags[0] & 0xffffu)) |
            (static_cast<unsigned long long>(sQueryFlags[1] & 0xffffu) << 16) |
            (static_cast<unsigned long long>(sQueryFlags[2] & 0xffffu) << 32);
        return (columns & (0x3full << (qi * GQA))) != 0;
    };"""
    if source.count(old) != 1:
        raise ValueError("query rescale predicate changed")
    source = source.replace(old, new)
    # This predicate depends only on the tile and query positions, not on
    # head values. Its old ballot ORs the lower and upper eight keys.
    old = "query_any(kbase + 7 > klimit)"
    if source.count(old) != 1:
        raise ValueError("query causal predicate changed")
    source = source.replace(old, "(k0 + mt * 16 + 15 > klimit)")
    old = "qwen_stock_m1_shared_decode<4,16,256,6,16,KVP,"
    if source.count(old) != 1 or source.count("dim3(128), 0, stream") != 1:
        raise ValueError("attention dispatch changed")
    source = source.replace(old, "qwen_stock_m1_shared_decode<3,16,256,6,16,KVP,")
    source = source.replace("dim3(128), 0, stream", "dim3(96), 0, stream")
    if prefetch:
        source = source.replace("VREGS <= 2", "VREGS <= 3")
    if reload_query:
        source = reload_query_fragments(source)
    return source


def coarsen_value_loads(source, tile):
    """Stage 32/64 KV positions while retaining serial 16-position updates.

    Split boundaries, online-softmax decisions, WMMA order and publication
    stay at the original 16-position granularity. Only the memory staging
    tile and block barriers grow. Partial stores and resets must remain
    inside that inner loop, including boundaries in the middle of a load.
    """
    if tile not in (32, 64):
        raise ValueError("unsupported staging tile")
    body, tail = source.split("// Merge. PF16=1", 1)

    def change(old, new):
        nonlocal body
        if body.count(old) != 1:
            raise ValueError(f"attention staging anchor changed: {old[:65]}")
        body = body.replace(old, new)

    body = re.sub(r"(r4d_attn_(?:tiles|tps)\([^,\n]+), TILE", r"\1, 16", body)
    change("bt[t_lo * NB + i]", "bt[min(t_lo * 16 / BS + i, (ctx - 1) / BS)]")
    change("fetchV(t_lo * TILE, blk)", "fetchV(t_lo * 16, blk)")
    change("ti < t_hi; ++ti", "ti < t_hi; ti += MT")
    change("const int k0 = ti * TILE;", "const int k0 = ti * 16;")
    start = body.index("        // A split-size boundary can give queries")
    end = body.index("        int blkn[NB];", start)
    reset = body[start:end].replace("ti == last_t_lo", "ti + mt == last_t_lo")
    body = body[:start] + body[end:]
    change("BTS && ti + 1 < t_hi", "BTS && ti + MT < t_hi")
    change("const int tb = (ti + 1) * NB;", "const int tb = (ti + MT) * 16 / BS;")
    change("blkn[i] = bt[tb + i]", "blkn[i] = bt[min(tb + i, (ctx - 1) / BS)]")
    change("GPREV && ti + 1 < t_hi", "GPREV && ti + MT < t_hi")
    change(
        "for (int mt = 0; mt < MT; ++mt) {",
        "for (int mt = 0; mt < MT; ++mt) {\n"
        "            if (ti + mt >= t_hi) break;\n" + reset,
    )
    partial = """        if (first_t_hi != t_hi && ti + 1 == first_t_hi) {
            if (live && query_t_hi == first_t_hi && query_t_lo < query_t_hi)
                write_partial();
        }
"""
    change(partial, "")
    change(
        "            }\n        }\n        if (BTS) {",
        "            }\n"
        + partial.replace("ti + 1", "ti + mt + 1")
        + "        }\n        if (BTS) {",
    )
    if tail.count("qwen_stock_m1_shared_decode<4,16,") != 1:
        raise ValueError("attention dispatch anchor changed")
    tail = tail.replace(
        "qwen_stock_m1_shared_decode<4,16,",
        f"qwen_stock_m1_shared_decode<4,{tile},",
    )
    return body + "// Merge. PF16=1" + tail


def reload_query_fragments(source):
    """Trade retained query VGPRs for identical reads of read-only query bytes.

    Keep the WMMA sequence, all conversions and the accumulator order intact.
    Repeated query reads should hit cache; the experiment tests whether the
    lower register pressure pays for those reads. No numerical shortcut.
    """
    start = source.index("    qfrag16 qf[NKS];")
    end = source.index("\n    const int klimit", start)
    source = (
        source[:start]
        + """    const uint16_t* qp = (const uint16_t*)a.q
        + ((size_t)(group_start + qrow) * a.q_heads + qhead) * HEAD_DIM;
"""
        + source[end:]
    )
    anchor = "for (int j = 0; j < PF; ++j) s = QD::wmma(kb[j], qf[g + j], s);"
    if source.count(anchor) != 1:
        raise ValueError("query/WMMA arithmetic anchor changed")
    return source.replace(
        anchor,
        """for (int j = 0; j < PF; ++j) {
                    const int t = g + j;
                    const uint2 l0 = *(const uint2*)(qp + 16 * t + (CTG ? 8 * h : 4 * h));
                    const uint2 h0 = *(const uint2*)(qp + 16 * t + (CTG ? 8 * h + 4 : 8 + 4 * h));
                    const qfrag16 qj = QD::mk(
                        make_uint2(r4d_qcvt<QD, 0, 0>(l0.x, mul), r4d_qcvt<QD, 0, 0>(l0.y, mul)),
                        make_uint2(r4d_qcvt<QD, 0, 0>(h0.x, mul), r4d_qcvt<QD, 0, 0>(h0.y, mul)));
                    s = QD::wmma(kb[j], qj, s);
                }""",
    )


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(args):
    args.build.mkdir(parents=True, mode=0o700, exist_ok=False)
    source = args.source / "shared.hip"
    original = source.read_text()
    anchor = "(3430971 & ~32 & ~8)"
    if original.count(anchor) != 1:
        raise ValueError("shared attention dispatch changed")
    report = {
        "schema": "coherence-attention-memory-tuning-v1",
        "status": "BUILT_UNTESTED",
        "generator_sha256": digest(__file__),
        "parent_binary_sha256": digest(args.source / "candidate.so"),
        "parent_source_sha256": digest(source),
        "variants": {},
    }
    for header in args.source.glob("*.h"):
        shutil.copyfile(header, args.build / header.name)
    variants = {
        1: OPTIONS,
        2: SECOND_SWEEP,
        3: THIRD_SWEEP,
        4: FOURTH_SWEEP,
        5: FIFTH_SWEEP,
        6: SIXTH_SWEEP,
        7: SEVENTH_SWEEP,
        8: EIGHTH_SWEEP,
        9: NINTH_SWEEP,
        10: TENTH_SWEEP,
        11: ELEVENTH_SWEEP,
        12: TWELFTH_SWEEP,
        13: THIRTEENTH_SWEEP,
        14: FOURTEENTH_SWEEP,
        15: FIFTEENTH_SWEEP,
    }[args.sweep]
    for name, options in variants.items():
        # Reject accidental changes to numerical feature bits in this sweep.
        if (options ^ BASE_OPT) & ~(3072 | 16 | 1048576 | 4096):
            raise ValueError("non-memory option in sweep")
        path = args.build / (name + ".hip")
        candidate = original.replace(anchor, str(options))
        if name.startswith("reload_q_"):
            candidate = reload_query_fragments(candidate)
        if name.startswith("load"):
            candidate = coarsen_value_loads(candidate, int(name.split("_")[0][4:]))
        if name.startswith("packed3_"):
            candidate = pack_query_columns(
                candidate, prefetch="prefetch" in name, reload_query="reload_q" in name
            )
        if "resident" in name:
            count = int(re.search(r"resident(\d+)", name).group(1))
            candidate = partial_query_residency(candidate, count)
        if "late_v" in name:
            candidate = combine_packed_value_barrier(candidate)
        if name.startswith("pvsplit"):
            candidate = split_value_columns(
                candidate, int(re.search(r"pvsplit(\d+)", name).group(1))
            )
        if "rawk" in name:
            candidate = share_raw_keys(candidate, pad=8 if "pad8" in name else 16)
        if "sharedq" in name:
            candidate = share_query_fragments(candidate)
        if "exactq" in name:
            candidate = guarded_half_queries(
                candidate, whole_dot_branch="whole" in name
            )
        if re.search(r"qk\d+", name):
            candidate = precompute_scores(
                candidate, int(re.search(r"qk(\d+)", name).group(1))
            )
        if name.startswith("warpvalues"):
            candidate = split_value_waves(candidate, share_scores="shareqk" in name)
        if "rawpref" in name:
            candidate = prefetch_raw_key_fragments(
                candidate, int(re.search(r"rawpref(\d+)", name).group(1))
            )
        if "headmajor" in name:
            candidate = reorder_attention_grid(candidate)
        if "stage" in name:
            candidate = group_attention_staging(
                candidate, int(re.search(r"stage(\d+)", name).group(1))
            )
        if name.startswith("probpass"):
            candidate = separate_probability_pass(
                candidate, int(re.search(r"_v(\d+)", name).group(1))
            )
        path.write_text(candidate)
        binary = args.build / (name + ".so")
        command = [
            "/opt/rocm/bin/hipcc",
            "-O3",
            "-std=c++17",
            "--offload-arch=gfx1201",
            "-shared",
            "-fPIC",
            "-ffp-contract=off",
            "-Rpass-analysis=kernel-resource-usage",
            f"-cuid=coherence_attn_tune_{digest(path)}",
            str(path),
            "-o",
            str(binary),
        ]
        process = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=600
        )
        (args.build / (name + ".compiler.log")).write_text(
            process.stdout + process.stderr
        )
        process.check_returncode()
        report["variants"][name] = {
            "options": options,
            "source_sha256": digest(path),
            "binary_sha256": digest(binary),
            "command": command,
        }
        (args.build / "build.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"built": name}), flush=True)


def probe(args):
    import torch
    from stock_m1_attention_shared import R4DArgs

    torch.set_num_threads(4)
    torch.manual_seed(20260925)
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    metadata = json.loads((args.build / "build.json").read_text())
    if metadata["parent_binary_sha256"] != digest(args.source / "candidate.so"):
        raise ValueError("qualified attention reference changed")
    libraries, functions = {}, {}
    paths = {"release": args.source / "candidate.so"}
    for name, entry in metadata["variants"].items():
        path = args.build / (name + ".so")
        if digest(path) != entry["binary_sha256"]:
            raise ValueError("attention candidate changed")
        paths[name] = path
    for name, path in paths.items():
        lib = ctypes.CDLL(str(path))
        function = lib.qwen_stock_m1_attention_shared
        function.argtypes = [ctypes.POINTER(R4DArgs), ctypes.c_int, ctypes.c_void_p]
        function.restype = ctypes.c_int
        libraries[name], functions[name] = lib, function
    report = {
        "status": "RUNNING",
        "build_sha256": digest(args.build / "build.json"),
        "probe_sha256": digest(__file__),
        "contexts": {},
        "private_chat_read": False,
        "negative_controls": {},
        "comparison": "exact output and FP32 split partials against qualified release",
    }

    def save():
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        for context in (1024, 60000, 200000):
            admitted_context = context + 8 + 512 + args.samples
            blocks = (admitted_context + 15) // 16
            # Random finite FP8 values; allocate directly as bytes, avoiding a
            # multi-gigabyte float32 temporary. Query inputs are ordinary BF16.
            kv = torch.randint(
                32, 85, (blocks, 4, 16, 512), device="cuda", dtype=torch.uint8
            )
            signs = torch.randint(0, 2, kv.shape, device="cuda", dtype=torch.uint8)
            kv.bitwise_or_(signs * 128)
            del signs
            table = torch.randperm(blocks, device="cuda", dtype=torch.int32).reshape(
                1, -1
            )
            lengths = torch.tensor([context + 8], dtype=torch.int32, device="cuda")
            query = torch.randn((8, 24, 256), device="cuda", dtype=torch.bfloat16)
            scales = [torch.full((4,), s, device="cuda") for s in (0.5, 1.5)]
            partial_size = 8 * 24 * 32 * (256 * 4 + 8)
            score_size = (
                ((admitted_context + 15) // 16) * 4 * 8 * 6 * 16 * 4
                if args.sweep == 9
                else 0
            )
            if args.sweep == 15:
                score_size = ((admitted_context + 15) // 16) * 4 * 8 * 6 * (16 * 4 + 8)
            scratch = {
                name: torch.full(
                    (partial_size + score_size + 1024,),
                    0xA5,
                    device="cuda",
                    dtype=torch.uint8,
                )
                for name in functions
            }
            outputs = {name: torch.empty_like(query) for name in functions}
            structures = {
                name: R4DArgs(
                    query.data_ptr(),
                    kv.data_ptr(),
                    table.data_ptr(),
                    lengths.data_ptr(),
                    outputs[name].data_ptr(),
                    scales[0].data_ptr(),
                    scales[1].data_ptr(),
                    0,
                    scratch[name][512:].data_ptr(),
                    1,
                    8,
                    24,
                    4,
                    256,
                    16,
                    blocks,
                    kv.stride(0),
                    kv.stride(1),
                    256**-0.5,
                    32,
                    admitted_context,
                )
                for name in functions
            }

            def launch(name, structures=structures):
                status = functions[name](
                    ctypes.byref(structures[name]),
                    0,
                    torch.cuda.current_stream().cuda_stream,
                )
                if status:
                    raise RuntimeError(f"native attention rejected launch: {status}")

            comparisons = {
                name: {"output_different": 0, "partial_different": 0}
                for name in functions
            }
            for sample in range(args.samples):
                # Visit every page offset and cross a 512-token split boundary.
                lengths.fill_(
                    context + 8 + (sample if sample < 16 else 512 + sample - 16)
                )
                query.normal_(std=(0.25, 1.0, 4.0, 12.0)[sample % 4])
                if args.sweep in (10, 11) and sample % 3 == 0:
                    # Force mixed fallback/fast waves, including underflow and
                    # overflow risks, without changing the independent reference.
                    query.flatten()[sample % query.numel()] = (
                        2.0**-20 if sample % 2 else 2.0**17
                    )
                for name in functions:
                    scratch[name].fill_(0xA5)
                    launch(name)
                torch.cuda.synchronize()
                for name in functions:
                    comparisons[name]["output_different"] += int(
                        (
                            outputs[name].view(torch.int16)
                            != outputs["release"].view(torch.int16)
                        ).sum()
                    )
                    comparisons[name]["partial_different"] += int(
                        (
                            scratch[name][512 : 512 + partial_size]
                            != scratch["release"][512 : 512 + partial_size]
                        ).sum()
                    )
                    if not bool(
                        (scratch[name][:512] == 0xA5).all()
                        and (scratch[name][-512:] == 0xA5).all()
                    ):
                        raise AssertionError("attention scratch guard damaged")
            # Exercise the same exact comparisons with deliberate corruptions.
            corrupted = outputs["release"].clone().view(torch.int16)
            corrupted.flatten()[0].bitwise_xor_(1)
            output_detected = (
                int((corrupted != outputs["release"].view(torch.int16)).sum()) == 1
            )
            corrupted_partial = scratch["release"][512 : 512 + partial_size].clone()
            corrupted_partial[0].bitwise_xor_(1)
            partial_detected = (
                int(
                    (
                        corrupted_partial
                        != scratch["release"][512 : 512 + partial_size]
                    ).sum()
                )
                == 1
            )
            if not output_detected or not partial_detected:
                raise AssertionError("injected attention corruption was not detected")
            report["negative_controls"][str(context)] = {
                "output_bit_flip_detected": output_detected,
                "partial_bit_flip_detected": partial_detected,
            }
            del corrupted, corrupted_partial
            eligible = [
                name for name, check in comparisons.items() if not any(check.values())
            ]
            if "control" not in eligible:
                raise AssertionError("rebuild does not match qualified attention")
            lengths.fill_(context + 8)
            query.normal_()
            graphs = {}
            for name in eligible:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(16):
                        launch(name)
                graphs[name] = graph
            for _ in range(8):
                for graph in graphs.values():
                    graph.replay()
            samples = {name: [] for name in eligible}
            for trial in range(7):
                order = eligible if trial % 2 == 0 else list(reversed(eligible))
                for name in order:
                    start, end = (
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    )
                    start.record()
                    for _ in range(16):
                        graphs[name].replay()
                    end.record()
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end) / 256)
            report["contexts"][str(context)] = {
                "positions_per_variant": args.samples * 8,
                "comparisons": comparisons,
                "median_ms_per_layer": {
                    name: statistics.median(values) for name, values in samples.items()
                },
                "samples_ms": samples,
            }
            save()
            print(
                json.dumps(
                    {
                        "context": context,
                        "eligible": eligible,
                        "median_ms_per_layer": report["contexts"][str(context)][
                            "median_ms_per_layer"
                        ],
                    }
                ),
                flush=True,
            )
            del kv, table, lengths, query, scales, scratch, outputs, graphs, structures
            torch.cuda.empty_cache()
        report["status"] = "SAMPLE_CHECKED_VARIANTS_RECORDED"
        save()
    except Exception:
        report["status"] = "FAILED"
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "probe"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--sweep", type=int, choices=tuple(range(1, 16)), default=1)
    args = parser.parse_args()
    (build if args.action == "build" else probe)(args)

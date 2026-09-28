"""Fill every attention matrix row while preserving per-query renormalization.

Six query heads do not divide a 16-row WMMA tile. The original mapping leaves
four rows idle per wave. This experiment packs three waves with eight complete
queries, and shares each wave's ballot so a split query still makes precisely
the original six-head renormalization decision. No score/reduction is reordered.
"""

import json
import subprocess
from pathlib import Path

from prefill_attention_alignment import build as original_build
from prefill_attention_alignment import digest, replace

from qwen_r9700_lab.diagnostic_contract import seal


def transform(source, old_warps, warps):
    rows = warps * 16 // 6
    source = replace(
        source,
        "const int group_start = blockIdx.x * (2 * NWARPS);",
        "const int group_start = blockIdx.x * (NWARPS * 16 / GQA);",
    )
    source = replace(
        source,
        "const int group_rows = min(2 * NWARPS, a.q_len - group_start);",
        "const int group_rows = min(NWARPS * 16 / GQA, a.q_len - group_start);",
    )
    source = replace(
        source,
        "    const int pair = (c / GQA) % 2;\n    const int qi = warp * 2 + pair, hi = c % GQA;\n    const bool live = (c < 2 * GQA && qi < group_rows);",
        "    const int logical_head = warp * 16 + c;\n    const int qi = logical_head / GQA, hi = logical_head % GQA;\n    const bool live = qi < group_rows;",
    )
    begin = source.index("    // Both halves of the fragment, all six heads, exactly one query.")
    end = source.index("    const int qhead", begin)
    source = source[:begin] + """
    __shared__ unsigned query_votes[NWARPS];
    auto query_any = [&](bool predicate) {
        const unsigned votes = static_cast<unsigned>(__ballot(predicate));
        if (lane == 0) query_votes[warp] = votes;
        lds_barrier();
        const int first_head = qi * GQA, last_head = first_head + GQA;
        const int first_wave = first_head / 16, last_wave = (last_head - 1) / 16;
        const int lo = first_head % 16;
        const int width = min(16 - lo, GQA);
        const unsigned mask = ((1u << width) - 1) << lo;
        bool yes = (query_votes[first_wave] & (mask | (mask << 16))) != 0;
        if (last_wave != first_wave) {
            const unsigned rest = (1u << (GQA - width)) - 1;
            yes |= (query_votes[last_wave] & (rest | (rest << 16))) != 0;
        }
        return yes;
    };
""" + source[end:]
    # Causal masking depends only on this query/key position. It does not need
    # the shared-head early-exit ballot and must not call a barrier divergently.
    source = replace(
        source,
        "if (!MSKIP || query_any(kbase + 7 > klimit))",
        "if (!MSKIP || kbase + 7 > klimit)",
    )
    source = replace(
        source,
        f"dim3((a.q_len+{2 * old_warps - 1})/{2 * old_warps},4,32),dim3({old_warps * 32})",
        f"dim3((a.q_len+{rows - 1})/{rows},4,32),dim3({warps * 32})",
    )
    source = replace(
        source,
        f"coherence_prefill_m1_decode<{old_warps},16,256,6,16,KVP,",
        f"coherence_prefill_m1_decode<{warps},16,256,6,16,KVP,",
    )
    return source


def build(parent, output, warps=6, window=128):
    if warps not in (3, 6, 12):
        raise ValueError("packed query groups require complete multiples of 48 heads")
    old_warps = warps * 4 // 3
    report = original_build(
        parent, output, warps=old_warps, window=window,
        shared_k=True, prefetch=2, query_first=True,
    )
    source = output / "aligned-prefill.hip"
    source.write_text(transform(source.read_text(), old_warps, warps))
    command = [
        "-cuid=coherence_prefill_" + digest(source) if s.startswith("-cuid=") else s
        for s in report["command"]
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    (output / "compiler.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    report.pop("sha256")
    report.update(command=command, packed_query_heads=True, generator_sha256=digest(__file__))
    report["files"] = {
        p.name: digest(p) for p in output.iterdir() if p.suffix in (".hip", ".h", ".so")
    }
    (output / "build.json").write_text(json.dumps(seal(report), indent=2) + "\n")
    return {"status": "BUILT_UNTESTED", "output": str(output)}

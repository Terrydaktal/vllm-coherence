"""Experimental cooperative PV slices sharing the exact QK results in LDS.

Only output channels move between waves. Query splits, all score arithmetic,
ordered high/low probability accumulation and the FP32 merge remain unchanged.
The build stays unqualified until native and full-model comparisons pass.
"""

import argparse
import json
import subprocess
from pathlib import Path

from prefill_attention_alignment import build as original_build
from prefill_attention_alignment import digest, replace

from qwen_r9700_lab.diagnostic_contract import seal


def transform(source, warps):
    split = source.index("__global__", source.index("auto write_partial"))
    body, tail = source[:split], source[split:]
    body = replace(
        body,
        "constexpr int NKS   = HEAD_DIM / 16;",
        "constexpr int NKS = HEAD_DIM / 16, NVS = NKS / 2;",
    )
    body = replace(
        body,
        "const int c = lane & 15, h = lane >> 4;",
        "const int c = lane & 15, h = lane >> 4;\n    const int v_first = (warp / (NWARPS / 2)) * (HEAD_DIM / 2);\n    __shared__ v8f shared_scores[NWARPS / 2][32];",
    )
    body = body.replace("(2 * NWARPS)", "NWARPS").replace(
        "min(2 * NWARPS,", "min(NWARPS,"
    )
    body = replace(
        body, "qi = warp * 2 + pair", "qi = (warp % (NWARPS / 2)) * 2 + pair"
    )
    body = replace(
        body, "qfrag16 qf[NKS];\n    {", "qfrag16 qf[NKS];\n    if (v_first == 0) {"
    )
    body = body.replace("v8f acc[NKS]", "v8f acc[NVS]")
    body = body.replace("t < NKS; ++t) acc[t]", "t < NVS; ++t) acc[t]")
    body = body.replace("dt < NKS", "dt < NVS")
    body = body.replace(
        "op = so + idx * HEAD_DIM;", "op = so + idx * HEAD_DIM + v_first;"
    )
    body = body.replace(
        "if (h == 0) { sm[idx]", "if (h == 0 && v_first == 0) { sm[idx]"
    )
    begin = body.index("            v8f s = (v8f)")
    end = body.index("            float sc[8];", begin)
    qk = body[begin:end]
    declaration, calculation = qk.split("\n", 1)
    body = (
        body[:begin]
        + declaration
        + "\n            if (v_first == 0) {\n"
        + calculation
        + "\n                shared_scores[warp][lane] = s;\n            }\n            lds_barrier();\n            if (v_first != 0) s = shared_scores[warp % (NWARPS / 2)][lane];\n"
        + body[end:]
    )
    pv = body.index("const uint16_t* vbase")
    body = body[:pv] + body[pv:].replace("g < NKS", "g < NVS").replace(
        "sV16[c * VSTR", "sV16[(v_first + c) * VSTR"
    )
    source = body + tail
    source = replace(
        source,
        f"dim3((a.q_len+{2 * warps - 1})/{2 * warps},4,32)",
        f"dim3((a.q_len+{warps - 1})/{warps},4,32)",
    )
    return source


def share_queries(source):
    source = replace(
        source,
        "qfrag16 qf[NKS];\n    if (v_first == 0) {",
        "__shared__ qfrag16 shared_q[NWARPS / 2][NKS][24];\n    if (v_first == 0 && c < 12) {",
    )
    source = replace(source, "qf[t] = QD::mk", "shared_q[warp][t][h * 12 + c] = QD::mk")
    source = source.replace("qf[g + j]", "shared_q[warp][g + j][h * 12 + (c % 12)]")
    if "qf[" in source:
        raise ValueError("query register use not replaced")
    return source


def build(parent, output, warps, window, shared_q=False, shared_k=True):
    if warps not in (4, 8, 16):
        raise ValueError("cooperative PV needs paired waves")
    report = original_build(
        parent,
        output,
        warps=warps,
        window=window,
        shared_k=shared_k,
        prefetch=2,
        query_first=True,
    )
    source = output / "aligned-prefill.hip"
    transformed = transform(source.read_text(), warps)
    source.write_text(share_queries(transformed) if shared_q else transformed)
    command = [
        "-cuid=coherence_prefill_" + digest(source) if s.startswith("-cuid=") else s
        for s in report["command"]
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    (output / "compiler.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    report.pop("sha256")
    report.update(
        command=command, cooperative_values=2, shared_q=shared_q, generator_sha256=digest(__file__)
    )
    report["files"] = {
        p.name: digest(p) for p in output.iterdir() if p.suffix in (".hip", ".h", ".so")
    }
    (output / "build.json").write_text(json.dumps(seal(report), indent=2) + "\n")
    return {"status": "BUILT_UNTESTED", "output": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warps", type=int, default=8)
    parser.add_argument("--window", type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(build(args.parent, args.output, args.warps, args.window)))

"""Build a graph-safe M8 target attention kernel from the released precision kernel."""

import argparse
import ctypes
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import authenticate, seal

PARENT_SOURCE = "2a6f2ada82a6b476be57702ad5c2fccd85af3cb3f937e2075e7478665d57a904"
PARENT_BINARY = "4ec21250a614e9c33d1bb26d57e2f4015457f57d35a712a55b43ebbd9bf92f4d"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_for_candidate(raw):
    if hashlib.sha256(raw).hexdigest() != PARENT_SOURCE:
        raise ValueError("unexpected shared target attention source")
    text = raw.decode()
    begin = text.index("    const int ntl  = r4d_attn_tiles(ctx, TILE);")
    end = text.index("    const int pair =", begin)
    text = (
        text[:begin]
        + """    const int ntl = r4d_attn_tiles(ctx, TILE);
    const int first_tiles = r4d_attn_tiles(first_ctx, TILE);
    int t_lo, t_hi, first_t_hi, last_t_lo;
    if (first_tiles >= 64) {
        const int first_tps = r4d_attn_tps(first_ctx, TILE, splits);
        const int last_tps = r4d_attn_tps(ctx, TILE, splits);
        t_lo = sp * first_tps;
        if (t_lo >= ntl) return;
        t_hi = min((sp + 1) * last_tps, ntl);
        first_t_hi = min((sp + 1) * first_tps, first_tiles);
        last_t_lo = sp * last_tps;
    } else {
        // Eight successive queries span at most two 16-key tile counts.
        // At the transition to 32 splits the later range can begin earlier.
        const int first_tps = r4d_attn_tps(first_ctx, TILE, coherence_query_splits(first_ctx));
        const int last_tps = r4d_attn_tps(ctx, TILE, coherence_query_splits(ctx));
        const int a0 = sp * first_tps, a1 = min(a0 + first_tps, first_tiles);
        const int b0 = sp * last_tps, b1 = min(b0 + last_tps, ntl);
        if (a0 >= a1 && b0 >= b1) return;
        t_lo = min(a0 < a1 ? a0 : b0, b0 < b1 ? b0 : a0);
        t_hi = max(a0 < a1 ? a1 : b1, b0 < b1 ? b1 : a1);
        first_t_hi = min(a0 < a1 ? a1 : b1, b0 < b1 ? b1 : a1);
        last_t_lo = max(a0 < a1 ? a0 : b0, b0 < b1 ? b0 : a0);
    }

"""
        + text[end:]
    )

    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError(f"non-unique target attention anchor: {old}")
        text = text.replace(old, new)

    template = "template<int NWARPS, int TILE, int HEAD_DIM, int GQA, int BS, int KVP, int OPT>"
    replace(
        template,
        """__device__ __forceinline__ int coherence_query_splits(int ctx) {
    const int tiles = (ctx + 15) / 16;
    return tiles >= 64 ? 32 : min(16, max(1, tiles));
}

"""
        + template,
    )
    replace(
        "r4d_attn_tps(query_ctx, TILE, splits)",
        "r4d_attn_tps(query_ctx, TILE, first_tiles >= 64 ? splits : coherence_query_splits(query_ctx))",
    )
    replace(
        "const int u = r4d_attn_used(ctx, tile, splits);",
        "const int u = r4d_attn_used(ctx, tile, coherence_query_splits(ctx));",
    )
    return text


def optimized_source(raw, variant):
    source = source_for_candidate(raw)
    if variant == "combined":
        return source
    if variant not in {"shifts", "separate"}:
        raise ValueError("unrecognized runtime attention specialization")
    helper = source.index("__device__ __forceinline__ int coherence_query_splits")
    template = "template<int NWARPS, int TILE, int HEAD_DIM, int GQA, int BS, int KVP, int OPT>"
    begin = source.index(template, helper)
    source = (
        source[:helper]
        + """__device__ __forceinline__ int coherence_tiles_per_split(unsigned tiles) {
    return tiles >= 64u ? (tiles + 31u) >> 5 : (tiles + 15u) >> 4;
}

"""
        + source[begin:]
    )
    source = source.replace(
        "    if (first_tiles >= 64) {\n        const int first_tps = r4d_attn_tps(first_ctx, TILE, splits);\n        const int last_tps = r4d_attn_tps(ctx, TILE, splits);",
        "    const int first_tps = coherence_tiles_per_split(first_tiles);\n    const int last_tps = coherence_tiles_per_split(ntl);\n    if (first_tps <= last_tps) {",
    )
    source = source.replace(
        "        const int first_tps = r4d_attn_tps(first_ctx, TILE, coherence_query_splits(first_ctx));\n        const int last_tps = r4d_attn_tps(ctx, TILE, coherence_query_splits(ctx));\n",
        "",
    )
    source = source.replace(
        "r4d_attn_tps(query_ctx, TILE, first_tiles >= 64 ? splits : coherence_query_splits(query_ctx))",
        "coherence_tiles_per_split(query_ntl)",
    )
    source = source.replace(
        "const int u = r4d_attn_used(ctx, tile, coherence_query_splits(ctx));",
        """const unsigned tiles = r4d_attn_tiles(ctx, tile);
    const unsigned tps = coherence_tiles_per_split(tiles);
    const int u = (tiles + tps - 1u) / tps;""",
    )
    if "coherence_query_splits" in source:
        raise ValueError("incomplete arithmetic substitution")
    if variant == "shifts":
        return source

    parent = raw.decode()
    guard = "if (total_ctx < a.q_len) return;"
    merge = "template<int HEAD_DIM, int CHAINS, int PF16>"
    long_kernel = parent[parent.index(template) : parent.index(merge)]
    long_kernel = long_kernel.replace(
        "qwen_stock_m1_shared_decode", "coherence_long_decode"
    )
    long_kernel = long_kernel.replace(guard, "if (total_ctx < 1008 + a.q_len) return;")
    if source.count(guard) != 1:
        raise ValueError("unexpected shared attention guard")
    source = source.replace(
        guard, "if (total_ctx < a.q_len || total_ctx >= 1008 + a.q_len) return;"
    )
    source = source.replace(merge, long_kernel + merge)
    first = source.index("  qwen_stock_m1_shared_decode<")
    end = source.index("  qwen_stock_m1_shared_merge<", first)
    launch = source[first:end].replace(
        "qwen_stock_m1_shared_decode", "coherence_long_decode"
    )
    return source[:first] + launch + source[first:]


def build(stock, out, variant="combined"):
    parent = json.loads((stock / "build.json").read_bytes())
    authenticate(parent)
    if (
        parent["files"]["shared.hip"] != PARENT_SOURCE
        or parent["files"]["candidate.so"] != PARENT_BINARY
    ):
        raise ValueError("unqualified shared target precision build")
    out.mkdir(mode=0o700)
    for name in ("r4d.h", "r4d_common.h", "r4d_dt16.h"):
        if sha(stock / name) != parent["files"][name]:
            raise ValueError("parent header changed")
        shutil.copyfile(stock / name, out / name)
    source, binary = out / "runtime-splits.hip", out / "runtime-splits.so"
    source.write_text(optimized_source((stock / "shared.hip").read_bytes(), variant))
    subprocess.run(
        [
            "/opt/rocm/bin/hipcc",
            "-O3",
            "-std=c++17",
            "--offload-arch=gfx1201",
            "-shared",
            "-fPIC",
            "-ffp-contract=off",
            "-cuid=runtime_splits_" + sha(source),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
    )
    manifest = seal(
        {
            "parent_source": PARENT_SOURCE,
            "parent_binary": PARENT_BINARY,
            "variant": variant,
            "generator": sha(Path(__file__)),
            "source": sha(source),
            "binary": sha(binary),
        }
    )
    (out / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load(build_dir):
    from stock_m1_attention_shared import R4DArgs

    manifest = json.loads((build_dir / "build.json").read_bytes())
    authenticate(manifest)
    binary = build_dir / "runtime-splits.so"
    if (
        manifest["parent_binary"] != PARENT_BINARY
        or manifest["generator"] != sha(Path(__file__))
        or manifest["binary"] != sha(binary)
    ):
        raise ValueError("runtime-split binary or adapter differs from its build")
    lib = ctypes.CDLL(str(binary))
    function = lib.qwen_stock_m1_attention_shared
    function.argtypes = [ctypes.POINTER(R4DArgs), ctypes.c_int, ctypes.c_void_p]
    function.restype = ctypes.c_int
    return function


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=("combined", "shifts", "separate"), default="combined"
    )
    options = parser.parse_args()
    print(json.dumps(build(options.stock, options.output, options.variant)), flush=True)

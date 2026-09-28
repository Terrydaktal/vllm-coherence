"""Experimental packed attention with votes and V sharing one LDS barrier.

The arithmetic and per-query six-head decisions stay unchanged. Each wave
publishes its rescale vote alongside the V tile before the existing V-read
barrier. K is either register-loaded or staged for the next tile after the
current QK readers finish. This avoids the earlier packed experiment's extra
per-key-tile barrier. Not a serving default.
"""

import argparse
import json
import subprocess
from pathlib import Path

from prefill_attention_alignment import build as original_build
from prefill_attention_alignment import digest, replace
from prefill_packed_attention import transform as pack_queries

from qwen_r9700_lab.diagnostic_contract import authenticate, seal, write_private


def transform(source, old_warps, warps, shared_keys=False, local_votes=False):
    source = pack_queries(source, old_warps, warps)
    if shared_keys:
        begin = source.index("        if (KLDS) {\n            #pragma unroll 1")
        end = source.index("        if (!GPREV) fetchV", begin)
        load = source[begin:end]
        if not load.endswith("        }\n"):
            raise ValueError("shared key staging boundary changed")
        load = load.replace(
            "        if (KLDS) {", "    auto stageK = [&](int k0, const int* blk) {", 1
        )
        load = load[:-10] + "    };\n"
        source = source[:begin] + source[end:]
        source = replace(
            source,
            "    auto write_partial = [&]() {",
            load + "\n    auto write_partial = [&]() {",
        )
        source = replace(
            source,
            "    if (GPREV) fetchV(t_lo * TILE, blk);",
            "    stageK(t_lo * TILE, blk);\n    if (GPREV) fetchV(t_lo * TILE, blk);",
        )
    source = replace(
        source,
        """        const unsigned votes = static_cast<unsigned>(__ballot(predicate));
        if (lane == 0) query_votes[warp] = votes;
        lds_barrier();""",
        """        // Published with this tile's V values, before its shared barrier.
        (void)predicate;""",
    )
    if local_votes:
        source = replace(
            source,
            "    __shared__ unsigned query_votes[NWARPS];",
            "    __shared__ unsigned query_votes[NWARPS];\n    unsigned current_votes = 0;",
        )
        source = source.replace(
            "query_votes[first_wave] &",
            "(first_wave == warp ? current_votes : query_votes[first_wave]) &",
        ).replace(
            "query_votes[last_wave] &",
            "(last_wave == warp ? current_votes : query_votes[last_wave]) &",
        )
    staging = """        if (!GPREV) fetchV(k0, blk);
        storeV();
        if (LDSB) lds_barrier(); else __syncthreads();
        if (GPREV && ti + 1 < t_hi) fetchV(k0 + TILE, blkn);"""
    source = replace(source, staging, "")
    source = replace(
        source,
        "            if (query_any(smax > m_ref + PGROW)) {",
        """            // The loop-entry barrier protects old V reads and votes,
            // and admits any K tile staged during the preceding PV calculation.
            const unsigned votes = static_cast<unsigned>(__ballot(smax > m_ref + PGROW));
            if (lane == 0) query_votes[warp] = votes;
"""
        + ("            current_votes = votes;\n" if local_votes else "")
        + staging
        + (
            "\n        if (ti + 1 < t_hi) stageK(k0 + TILE, blkn);"
            if shared_keys
            else ""
        )
        + "\n            if (query_any(smax > m_ref + PGROW)) {",
    )
    source = replace(
        source,
        "    constexpr int KLDS  = (OPT & O_KLDS0) ? 0 : 1;",
        "    constexpr int KLDS  = (OPT & O_KLDS0) ? 0 : 1;\n"
        + f'    static_assert(KLDS == {int(shared_keys)} && TILE == 16, "vote staging layout changed");',
    )
    return source


def build(
    parent,
    output,
    warps=6,
    window=128,
    prefetch=2,
    shared_keys=False,
    local_votes=False,
):
    if warps not in (3, 6, 12):
        raise ValueError("packed query groups require complete multiples of 48 heads")
    old_warps = warps * 4 // 3
    report = original_build(
        parent,
        output,
        warps=old_warps,
        window=window,
        shared_k=shared_keys,
        prefetch=prefetch,
        query_first=True,
    )
    source = output / "aligned-prefill.hip"
    source.write_text(
        transform(source.read_text(), old_warps, warps, shared_keys, local_votes)
    )
    command = [
        "-cuid=coherence_prefill_" + digest(source) if s.startswith("-cuid=") else s
        for s in report["command"]
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=300, check=False
    )
    (output / "compiler.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    report.pop("sha256")
    report.update(
        command=command,
        packed_query_heads=True,
        votes_with_value_staging=True,
        key_staged_during_values=shared_keys,
        local_votes=local_votes,
        physical_warps=warps,
        generator_sha256=digest(__file__),
    )
    report["files"] = {
        p.name: digest(p) for p in output.iterdir() if p.suffix in (".hip", ".h", ".so")
    }
    (output / "build.json").write_text(json.dumps(seal(report), indent=2) + "\n")
    return {"status": "BUILT_UNTESTED", "output": str(output)}


def build_hybrid(short_build, packed_build, output):
    """Retain the qualified short-context kernel; use packed work above 16K."""
    short_build, packed_build, output = map(Path, (short_build, packed_build, output))
    parents = []
    for root in (short_build, packed_build):
        data = json.loads((root / "build.json").read_text())
        authenticate(data)
        if data["kernel_abi"] != "coherence-prefill-m1-attention-v1":
            raise ValueError("hybrid parent has a different attention ABI")
        for name, expected in data["files"].items():
            if Path(name).name != name or digest(root / name) != expected:
                raise ValueError("hybrid parent contents changed")
        parents.append(data)
    short, packed = parents
    if not (
        short["adaptive_context"] == 16384
        and short["window"] == packed["window"] == 128
        and short["scratch_bytes"] == packed["scratch_bytes"]
        and packed["packed_query_heads"]
        and packed["votes_with_value_staging"]
        and packed["key_staged_during_values"]
        and packed.get("physical_warps", packed["warps"] * 3 // 4) == 12
        and packed["kv_load_tile"] == 16
    ):
        raise ValueError("hybrid geometry differs from the tested candidates")
    output.mkdir(mode=0o700)
    for name in ("r4d.h", "r4d_common.h", "r4d_dt16.h"):
        if (short_build / name).read_bytes() != (packed_build / name).read_bytes():
            raise ValueError("hybrid parents use different numerical headers")
        (output / name).write_bytes((short_build / name).read_bytes())
    original = replace(
        (short_build / "aligned-prefill.hip").read_text(),
        'extern "C" int coherence_prefill_attention(',
        'extern "C" int coherence_original_prefill_attention(',
    )
    packed_source = replace(
        (packed_build / "aligned-prefill.hip").read_text(),
        'extern "C" int coherence_prefill_attention(',
        'extern "C" int coherence_packed_prefill_attention(',
    )
    # R4DArgs lives in the global namespace; distinct function names also avoid
    # argument-dependent lookup admitting the original launch overload.
    packed_source = packed_source.replace("coherence_prefill_", "coherence_packed_")
    source = output / "aligned-prefill.hip"
    source.write_text(
        original
        + "\nnamespace coherence_packed {\n"
        + packed_source
        + "\n}\n"
        + """
extern "C" int coherence_prefill_attention(const R4DArgs* a, int kvp, void* stream) {
    if (a && a->max_ctx > 16384)
        return coherence_packed::coherence_packed_prefill_attention(a, kvp, stream);
    return coherence_original_prefill_attention(a, kvp, stream);
}
"""
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
    (output / "generator.py").write_bytes(Path(__file__).read_bytes())
    report = {k: v for k, v in short.items() if k not in ("sha256", "files", "command")}
    report.update(
        status="BUILT_UNTESTED",
        command=command,
        generator_sha256=digest(__file__),
        short_parent_sha256=short["sha256"],
        packed_parent_sha256=packed["sha256"],
        packed_query_heads=True,
        votes_with_value_staging=True,
        key_staged_during_values=True,
        long_context_geometry={"queries_per_group": 32, "warps": 12, "cutover": 16384},
    )
    report["files"] = {
        p.name: digest(p)
        for p in output.iterdir()
        if p.suffix in (".hip", ".h", ".so", ".py")
    }
    write_private(output / "build.json", seal(report))
    return {"status": "BUILT_UNTESTED", "output": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warps", type=int, choices=(3, 6, 12), default=6)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--prefetch", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--shared-keys", action="store_true")
    parser.add_argument("--local-votes", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.parent,
                args.output,
                args.warps,
                args.window,
                args.prefetch,
                args.shared_keys,
                args.local_votes,
            )
        )
    )

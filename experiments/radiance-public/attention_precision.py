"""Pinned repairs for R4D attention's intermediate precision losses.

Q/K retain their BF16 exponent range and scaling follows the FP32 dot product.
PV uses two 16-bit probability terms (FP16 for FP8 KV, BF16 for BF16 KV).
Split partials remain FP32 until the final BF16 output. This is finite-precision
attention, not a proof of equivalence to real arithmetic or every other backend.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from build_stock_m1_attention_shared import EXPORT, SOURCES, transform

from qwen_r9700_lab.diagnostic_contract import seal, write_private

PINS = {
    **SOURCES,
    "r4d_attn_paged_h256_gqa6.hip": "b84686cb6f5b0371f3219aa795472120369b7c035a64af55579d1366c694d691",
    "r4d_attn_prefill_h256_gqa6.hip": "6ed3af15476969bd460330f65a05e85765d10f4fb7a61c4c419ff3a21cd4cbe5",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("attention precision patch anchor changed: " + old[:90])
    return source.replace(old, new, 1)


def repair_kernel(source):
    """Apply identical arithmetic changes to native prefill/decode and shared M8."""
    source = replace_once(
        source,
        "typedef DT16<F16> D;",
        "// Q/K widening is exact; probability/value operands are independent.\n"
        "    typedef DT16<0> QD;\n"
        "    typedef typename QD::frag qfrag16;\n"
        "    typedef DT16<KVP ? 0 : 1> D;",
    )
    source = replace_once(source, "frag16 qf[NKS];", "qfrag16 qf[NKS];")
    start = source.index("    qfrag16 qf[NKS];")
    end = source.index("\n    }", start) + len("\n    }")
    q = source[start:end].replace("D::", "QD::")
    q = q.replace("r4d_qcvt<D, FOLDQ, F16>", "r4d_qcvt<QD, 0, 0>")
    source = source[:start] + q + source[end:]
    # K staging and QK, excluding the separately staged V operands.
    source = replace_once(source, "frag16 kb[PF];", "qfrag16 kb[PF];")
    start = (
        source.index("                    qfrag16 kb[PF];")
        if "                    qfrag16 kb[PF];" in source
        else source.index("                qfrag16 kb[PF];")
    )
    end = source.index("            float sc[8];", start)
    k = source[start:end].replace("D::", "QD::").replace("<D>", "<QD>")
    source = source[:start] + k + source[end:]
    # Both kernels have a K staging region before V staging. Direct-K decode
    # compiles it away, while prefill requires the exact same BF16 widening.
    if "    auto storeK =" in source:
        start = source.index("    auto storeK =")
        end = source.index("    auto storeV =", start)
    else:
        start = source.index("        if (KLDS) {")
        end = source.index("        if (!GPREV)", start)
    staging = source[start:end].replace("D::", "QD::").replace("<D>", "<QD>")
    source = source[:start] + staging + source[end:]
    source = replace_once(
        source, "sc[e] = FOLDQ ? s[e] : s[e] * mul;", "sc[e] = s[e] * mul;"
    )
    # Sum FP32 probabilities. The rounded high/low terms approximate that same
    # probability with more precision than the old single FP16/RTZ term.
    source = replace_once(
        source,
        "constexpr int DOT2  = ((OPT & O_DOT2) && (OPT & O_F16)) ? 1 : 0;",
        "constexpr int DOT2 = 0;",
    )
    source = replace_once(source, "uint32_t pw[4];", "uint32_t pw[4], pl[4];")
    source = replace_once(
        source,
        "for (int e = 0; e < 8; e += 2) pw[e >> 1] = D::pk(pf[e], pf[e + 1]);",
        """for (int e = 0; e < 8; e += 2) {
                    const uint32_t packed = D::pk(pf[e], pf[e + 1]);
                    pw[e >> 1] = packed;
                    const float x = KVP ? __builtin_bit_cast(float, packed << 16)
                                        : float(__builtin_bit_cast(__fp16, uint16_t(packed)));
                    const float y = KVP ? __builtin_bit_cast(float, packed & 0xffff0000u)
                                        : float(__builtin_bit_cast(__fp16, uint16_t(packed >> 16)));
                    pl[e >> 1] = D::pk(pf[e] - x, pf[e + 1] - y);
                }""",
    )
    source = replace_once(source, "frag16 p16;", "frag16 p16, p16lo;")
    anchor = "                const uint16_t* vbase ="
    if anchor not in source:
        raise ValueError("attention PV staging anchor changed")
    source = replace_once(
        source,
        anchor,
        """                const uint2 low0 = make_uint2(pl[0], pl[1]);
                const uint2 low1 = make_uint2(pl[2], pl[3]);
                if (CTG) p16lo = D::mk(low0, low1);
                else {
                    const uint2 y = swap16_u2p((h == 0) ? low1 : low0);
                    p16lo = (h == 0) ? D::mk(low0, y) : D::mk(y, low1);
                }
"""
        + anchor,
    )
    source = replace_once(
        source,
        "for (int j = 0; j < PF; ++j) acc[g + j] = D::wmma(vb[j], p16, acc[g + j]);",
        """for (int j = 0; j < PF; ++j) {
                        acc[g + j] = D::wmma(vb[j], p16, acc[g + j]);
                        acc[g + j] = D::wmma(vb[j], p16lo, acc[g + j]);
                    }""",
    )
    return source


def build(source_dir, output):
    output.mkdir(mode=0o700)
    pins = dict(PINS)
    for name, expected in pins.items():
        if digest(source_dir / name) != expected:
            raise ValueError("attention upstream source changed: " + name)
        (output / name).write_bytes((source_dir / name).read_bytes())
    for name in ("r4d_attn_decode_h256_gqa6.hip", "r4d_attn_prefill_h256_gqa6.hip"):
        (output / name).write_text(repair_kernel((source_dir / name).read_text()))
    dispatch = (source_dir / "r4d_attn_paged_h256_gqa6.hip").read_text()
    dispatch = replace_once(
        dispatch, "#define D_OPT  3430971", "#define D_OPT  (3430971 & ~32 & ~8)"
    )
    dispatch = replace_once(
        dispatch, "#define P_OPT    3955211", "#define P_OPT    (3955211 & ~8)"
    )
    (output / "r4d_attn_paged_h256_gqa6.hip").write_text(dispatch)
    shared = repair_kernel(
        transform((source_dir / "r4d_attn_decode_h256_gqa6.hip").read_text())
    )
    shared += EXPORT.replace("3430971>", "(3430971 & ~32 & ~8)>").replace(
        "shared_merge<256,4,1>", "shared_merge<256,4,0>"
    )
    (output / "shared.hip").write_text(shared)
    commands = []
    for unit, library in (
        ("r4d_attn_paged_h256_gqa6.hip", "native.so"),
        ("shared.hip", "candidate.so"),
    ):
        command = [
            "/opt/rocm/bin/hipcc",
            "-O3",
            "-std=c++17",
            "--offload-arch=gfx1201",
            "-shared",
            "-fPIC",
            "-ffp-contract=off",
            "-cuid=coherence_attention_" + digest(output / unit),
            str(output / unit),
            "-o",
            str(output / library),
        ]
        done = subprocess.run(
            command, capture_output=True, text=True, timeout=300, check=False
        )
        (output / (library + ".log")).write_text(done.stdout + done.stderr)
        if done.returncode:
            raise RuntimeError(
                "attention compile failed: " + str(output / (library + ".log"))
            )
        commands.append(command)
    report = seal(
        {
            "status": "BUILT_UNTESTED",
            "kernel_abi": "coherence-attention-precision-v1",
            "gpu_used": False,
            "upstream_sources": pins,
            "generator_sha256": digest(__file__),
            "shared_generator_sha256": digest(
                Path(__file__).with_name("build_stock_m1_attention_shared.py")
            ),
            "commands": commands,
            "files": {
                p.name: digest(p)
                for p in output.iterdir()
                if p.suffix in (".hip", ".h", ".so")
            },
            "binary_sha256": digest(output / "candidate.so"),
        }
    )
    write_private(output / "build.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output)))

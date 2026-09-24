"""Native negative controls and independent oracles for the attention repair."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

from attention_precision import digest
from attention_precision_runtime import PrecisionAttention
from probe_m1_attention_precision import (
    R4D_SHA256,
    Hip,
    backend_stopped,
    fixtures,
    native_case,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    backend_stopped("http://127.0.0.1:8080")
    import r4d

    if digest(r4d.__file__) != R4D_SHA256:
        raise ValueError("native negative-control binary changed")
    candidate = PrecisionAttention(args.build)
    hip = Hip()
    results = []
    started = time.monotonic()
    try:
        for phase in ("decode", "prefill"):
            variants = []
            for engine in (r4d, candidate):
                variants.append(
                    SimpleNamespace(
                        attn_decode_h256_gqa6_scratch_bytes=engine.attn_decode_h256_gqa6_scratch_bytes,
                        **{
                            f"attn_decode_h256_gqa6_{dtype}kv": getattr(
                                engine, f"attn_{phase}_h256_gqa6_{dtype}kv"
                            )
                            for dtype in ("fp8", "bf16")
                        },
                    )
                )
            for fixture in fixtures():
                backend_stopped("http://127.0.0.1:8080")
                old, new = (native_case(hip, engine, fixture) for engine in variants)
                results.append(
                    {"phase": phase, "case": fixture[0], "old": old, "fixed": new}
                )
                print(
                    json.dumps(
                        {
                            "phase": phase,
                            "case": fixture[0],
                            "old_different": old["different_elements"],
                            "fixed_different": new["different_elements"],
                            "fixed_values": new["actual_unique"],
                        }
                    ),
                    flush=True,
                )
    finally:
        hip.close()
    report = {
        "status": "SAMPLE_CHECKED"
        if all(row["fixed"]["different_elements"] == 0 for row in results)
        else "MISMATCH",
        "cases": results,
        "seconds": time.monotonic() - started,
        "build_sha256": digest(args.build / "build.json"),
        "probe_sha256": digest(__file__),
        "negative_control_detected": sum(
            row["old"]["different_elements"] > 0
            for row in results
            if row["phase"] == "decode"
        )
        == 7,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "SAMPLE_CHECKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

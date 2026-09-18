"""Compare sealed private replay rows and publish aggregate evidence only."""

import argparse
import json
from pathlib import Path

from qwen_r9700_lab.conformance_topk import aggregate, compare_saved_rows
from qwen_r9700_lab.diagnostic_contract import authenticate, seal, write_private


def compare(args):
    before, after, run, performance = (
        json.loads(p.read_text())
        for p in (args.reference, args.candidate, args.run, args.performance)
    )
    authenticate(run)
    authenticate(performance)
    observed = run["observation"]["performance"]
    if observed["manifest"] != performance["sha256"]:
        raise ValueError("replay ran a different performance implementation")
    if performance.get("activation_tiles") and observed["activation_tiles"]["tiled"] <= 0:
        raise ValueError("tiled activation path did not execute")
    rows, prefill = compare_saved_rows(before, after, target_rows=8)
    result = seal(
        {
            "reference_sha256": before["sha256"],
            "candidate_sha256": after["sha256"],
            "candidate_performance_sha256": performance["sha256"],
            "candidate_run_sha256": run["sha256"],
            "fixture": run["fixture"],
            "decode": aggregate(rows),
            "prefill": prefill,
            "first_different_position": next(
                (i for i, r in enumerate(rows) if not r["full_logits_exact"]), None
            ),
            "activation_tiles": observed.get("activation_tiles"),
            "scope": (
                "same private 60K prefix; compiled M8; "
                "exact sampled full logits, not a universal proof"
            ),
        }
    )
    write_private(args.output, result)
    print(json.dumps({k: result[k] for k in ("decode", "prefill", "activation_tiles", "sha256")}))
    if (
        len(rows) < 320
        or not prefill["full_logits_exact"]
        or not all(r["full_logits_exact"] for r in rows)
    ):
        raise ValueError("full model output differs or coverage is incomplete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("reference", "candidate", "run", "performance", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    compare(parser.parse_args())

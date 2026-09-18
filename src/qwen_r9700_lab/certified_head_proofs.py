"""Scoped SMT obligations for interval exclusion; not a native GPU certificate.

The implementation's strict comparison is executed on symbolic values. Bounds
and reference fidelity are explicit assumptions; their numerical construction,
NumPy partitioning, refinement orchestration and compiled kernels are not proved
by these queries. Domain satisfiability and deliberate defects are checked too.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

from qwen_r9700_lab import certified_head
from qwen_r9700_lab.conformance_proofs import check_obligation


def run(output: Path, *, timeout_ms=10_000):
    import z3

    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    cases = []

    def check(name, bad, assumptions=(), *, expected=z3.unsat):
        cases.append(
            check_obligation(
                output, name, bad, assumptions=assumptions, expected=expected, timeout_ms=timeout_ms
            )
        )

    threshold, lower, upper, retained, excluded = z3.Reals(
        "threshold lower upper retained excluded"
    )
    check(
        "excluded_token_below_each_cutoff_witness",
        excluded >= retained,
        (
            excluded <= upper,
            lower <= retained,
            lower >= threshold,
            certified_head.strictly_excluded(upper, threshold),
        ),
    )
    # Finite cardinality check, arbitrary real scores and intervals. This tests
    # that k lower-bound witnesses imply k strictly better reference scores.
    # It is deliberately labelled finite; the pairwise lemma above generalizes
    # to each witness independently, with counting as a mathematical argument.
    size = 8
    lo, hi, scores = [[z3.Real(f"{kind}_{i}") for i in range(size)] for kind in ("lo", "hi", "z")]
    enclosures = [item for i in range(size) for item in (lo[i] <= scores[i], scores[i] <= hi[i])]
    witnesses = z3.Sum([z3.If(v >= threshold, 1, 0) for v in lo])
    better = z3.Sum([z3.If(v > scores[0], 1, 0) for v in scores])
    for k in range(1, size):
        check(
            f"eight_token_top_{k}_exclusion",
            better < k,
            (*enclosures, witnesses >= k, certified_head.strictly_excluded(hi[0], threshold)),
        )

    check(
        "mutant_nonstrict_drops_boundary_tie",
        better < 1,
        (*enclosures, witnesses >= 1, hi[0] <= threshold),
        expected=z3.sat,
    )
    check(
        "mutant_uncovered_token",
        better < 1,
        (*enclosures[2:], witnesses >= 1, certified_head.strictly_excluded(hi[0], threshold)),
        expected=z3.sat,
    )
    approximate_winner, approximate_other = z3.Reals("approximate_winner approximate_other")
    check(
        "mutant_approximate_rank_as_certificate",
        scores[0] > scores[1],
        (*enclosures, approximate_winner < approximate_other),
        expected=z3.sat,
    )

    l1, u1, l2, u2, score = z3.Reals("l1 u1 l2 u2 score")
    intersection_lower = z3.If(l1 > l2, l1, l2)
    intersection_upper = z3.If(u1 < u2, u1, u2)
    check(
        "sound_interval_intersection",
        z3.Or(score < intersection_lower, score > intersection_upper),
        (l1 <= score, score <= u1, l2 <= score, score <= u2),
    )
    check("singleton_score_fidelity", score != l1, (l1 <= score, score <= u1, l1 == u1))
    proposed = z3.Real("proposed")
    check(
        "mutant_completeness_without_score_fidelity",
        proposed != score,
        (l1 <= score, score <= u1, l1 <= proposed, proposed <= u1),
        expected=z3.sat,
    )

    report = {
        "schema": "urn:qwen:certified-head-predicate-obligations:v1",
        "cases": cases,
        "all_expected_results": all(c["status"] != "UNPROVED" for c in cases),
        "solver": z3.get_full_version(),
        "timeout_ms": timeout_ms,
        "predicate_sha256": hashlib.sha256(
            inspect.getsource(certified_head.strictly_excluded).encode()
        ).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            Path(certified_head.__file__).read_bytes()
        ).hexdigest(),
        "proof_runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "proved_scope": (
            "Strict exclusion inequality under sound enclosures; finite cardinality cases; "
            "abstract intersection and singleton lemmas."
        ),
        "assumed": ["sound reference-score enclosures", "solver", "Python/Z3 translation"],
        "unproved": [
            "native bound construction",
            "native reference arithmetic binding",
            "GPU kernels",
            "refinement orchestration",
            "sampler equivalence",
        ],
        "native_status": "UNPROVED",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-ms", type=int, default=10_000)
    args = parser.parse_args()
    report = run(args.output, timeout_ms=args.timeout_ms)
    print(
        json.dumps(
            {
                "all_expected_results": report["all_expected_results"],
                "cases": len(report["cases"]),
                "native_status": report["native_status"],
            }
        )
    )
    return 0 if report["all_expected_results"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

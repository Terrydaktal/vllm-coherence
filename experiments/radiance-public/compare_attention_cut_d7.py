"""Localize captured attention sub-boundaries on CPU without reading chat text."""

import argparse
import json
from pathlib import Path

from compare_execution_modes_d7 import load as load_run
from compare_mode_boundaries_d7 import load

from qwen_r9700_lab.conformance_attention_cut import (
    attention_cut,
    rotary_formula,
    selected_rotary_coefficients,
)
from qwen_r9700_lab.conformance_execution_modes import admit_pair
from qwen_r9700_lab.conformance_mode_boundaries import admit_bridge, compare_arrays
from qwen_r9700_lab.conformance_precision_intervention import admit_precision_intervention
from qwen_r9700_lab.conformance_topk import require
from qwen_r9700_lab.diagnostic_contract import authenticate, private_json, seal, write_private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("left", "right", "left-run", "right-run", "left-bridge", "right-bridge", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--precision-casts-intervention", action="store_true")
    parser.add_argument("--layer", type=int, default=3)
    args = parser.parse_args()
    require(args.layer in range(3, 64, 4), "not an attention layer in the pinned model")
    sides = [load_run(r) for r in (args.left_run, args.right_run)]
    admission = (admit_precision_intervention if args.precision_casts_intervention else admit_pair)(
        *sides
    )
    roots = [args.left, args.right]
    bridges = [private_json(p) for p in (args.left_bridge, args.right_bridge)]
    results = {}
    formula_results = {}
    sources = {}
    for phase in ("prefill", "decode"):
        manifests = [
            private_json(r / ("prefill-manifest.json" if phase == "prefill" else "manifest.json"))
            for r in roots
        ]
        for root, manifest, side, bridge in zip(roots, manifests, sides, bridges, strict=True):
            authenticate(manifest)
            admit_bridge(
                bridge,
                side["pass"],
                private_json(root / "manifest.json"),
                manifest if phase == "prefill" else None,
            )
        require(len(manifests[0]["batches"]) == len(manifests[1]["batches"]), "group counts differ")
        totals = {}
        formula_totals = {}
        observed = []
        for a, b in zip(manifests[0]["batches"], manifests[1]["batches"], strict=True):
            ma, ta = load(roots[0], a)
            mb, tb = load(roots[1], b)
            require(ma["positions"] == mb["positions"], "positions differ")
            observed += ma["positions"]
            ca, cb = attention_cut(ma, ta, args.layer), attention_cut(mb, tb, args.layer)
            for name in ca:
                value = compare_arrays([ca[name]], [cb[name]], ma["positions"])
                require(value is not None, "incomplete attention comparison")
                total = totals.setdefault(name, {})
                for k, n in value.items():
                    if k == "different_positions":
                        continue
                    total[k] = total.get(k, 0) + n
            cosine, sine = selected_rotary_coefficients(mb, tb, args.layer)
            for kind in ("query", "key"):
                require(
                    (ca[kind + "_after_normalization"] == cb[kind + "_after_normalization"]).all(),
                    "rotary formula isolation requires identical normalized inputs",
                )
                for rounding in ("rne", "rtz"):
                    predicted = rotary_formula(
                        cb[kind + "_after_normalization"], cosine, sine, rounding
                    )
                    for label, observed_cut in (("left", ca), ("right", cb)):
                        name = f"{kind}/{rounding}_products/{label}"
                        value = compare_arrays(
                            [predicted], [observed_cut[kind + "_after_rotation"]], ma["positions"]
                        )
                        require(value is not None, "incomplete rotary formula comparison")
                        total = formula_totals.setdefault(name, {})
                        for k, n in value.items():
                            if k != "different_positions":
                                total[k] = total.get(k, 0) + n
        expected = [p for b in manifests[0]["batches"] for p in b["positions"]]
        require(
            observed == expected and len(set(observed)) == len(observed),
            "incomplete or duplicate coverage",
        )
        require(len(observed) == (320 if phase == "decode" else 9), "unexpected sampled domain")
        results[phase] = totals
        formula_results[phase] = formula_totals
        sources[phase] = [m["sha256"] for m in manifests]
    report = seal(
        {
            "schema": "qwen.attention-cut-comparison.v1",
            "status": "COMPARED_CAPTURED_SUB_BOUNDARIES",
            "layer": args.layer,
            "admission": admission,
            "bridges": [b["sha256"] for b in bridges],
            "captures": sources,
            "results": results,
            "rotary_formulae": formula_results,
            "formula_basis": {
                "inputs": "identical normalized Q/K from both captures",
                "coefficients": "right compiled selected BF16 cosine/sine, with wiring checked",
                "products": ["round to nearest, ties to even", "round toward zero"],
                "addition_subtraction": "round to nearest, ties to even",
                "tail": "192 non-rotary coordinates copied unchanged",
                "coefficient_selection_equivalence": "not established by this formula test",
            },
            "scope": (
                "Captured Q/K, rotary outputs, values, attention and gating. Attention state "
                "is not captured; inherited differences do not establish an attention kernel "
                "defect. No isolated vocabulary or universal claim."
            ),
        }
    )
    write_private(args.output, report)
    print(json.dumps({"sha256": report["sha256"], "results": results}))


if __name__ == "__main__":
    main()

"""Rebuild public numeric tables from aggregate, transcript-free receipts."""

import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / "evidence" / name).read_text())


def crossmode_corpus_tables():
    from build_detailed_tables import authenticated_read

    data = authenticated_read(ROOT / "evidence/crossmode-before-final-10k.json")
    assert data["schema"] == "qwen.crossmode-corpus-before-final.v1"
    assert data["status"] == "AUDITED"
    assert data["common"]["positions"] == 10000
    for revision in ("before", "final"):
        result = data["comparisons"][revision]
        assert result["decode"]["positions"] == 10000
        assert result["prefill"]["positions"] == 23
        for arm, mode, width in (("m1", "eager", 1), ("m8", "compiled", 8)):
            run = data["runs"][f"{revision}-{arm}"]
            assert (run["mode"], run["target_rows"]) == (mode, width)
            assert run["positions"] == 10000 and run["responses"] == 23
            assert (run["target_graph_replays"] > 0) == (mode == "compiled")
            assert run["precision_casts"] is (revision == "final")
            assert bool(run["repair"]) == (revision == "final")
    table = [
        "| Prediction | Before: same set | Before: same order | Before: mean shared | "
        "After both fixes: same set | After both fixes: same order | "
        "After both fixes: mean shared |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def count(value, total):
        percent = f"{100 * value / total:.2f}".rstrip("0").rstrip(".")
        return f"{value:,} / {total:,} ({percent}%)"

    for k in (1, 10, 20):
        cells = []
        for revision in ("before", "final"):
            row = data["comparisons"][revision]["decode"][str(k)]
            cells.extend(
                [
                    count(row["set_exact"], 10000),
                    count(row["ranked_exact"], 10000),
                    f"{row['mean_overlap_tokens']:.4f}".rstrip("0").rstrip(".") + f" / {k}",
                ]
            )
        table.append(f"| Top {k} | {' | '.join(cells)} |")
    final = data["comparisons"]["final"]
    all_topk = all(
        final[domain][str(k)][field] == final[domain]["positions"]
        for domain in ("decode", "prefill")
        for k in (1, 10, 20)
        for field in (
            "set_exact",
            "ranked_exact",
            "retained_scores_exact",
            "inclusive_tie_set_exact",
        )
    )
    summary = (
        "After both fixes, full-vocabulary hashes match at "
        f"**{final['decode']['full_logits_exact']:,}/10,000 decode positions** and "
        f"**{final['prefill']['full_logits_exact']}/23 initial-prefill predictions**. "
    )
    summary += (
        "Top-1/10/20 retained scores and inclusive boundary-tie sets also match throughout."
        if all_topk
        else "The complete retained-score, tie and prefill counts are in the linked evidence."
    )
    (ROOT / "crossmode-10k-table.md").write_text("\n".join(table) + "\n")
    return {
        "{{CROSSMODE_10K_TABLE}}": "\n".join(table),
        "{{CROSSMODE_10K_SUMMARY}}": summary,
    }


def rounding_speed_tables():
    from build_detailed_tables import authenticated_read

    data = authenticated_read(ROOT / "evidence/rounding-speed-abba.json")
    assert data["status"] == "MEASURED" and data["prefix_tokens"] == 60000
    assert data["order"] == [0, 1, 1, 0]
    assert [c["precision_casts"] for c in data["controls"]] == data["order"]
    assert data["execution"]["compiled"] and data["execution"]["graph_mode"] == "PIECEWISE"
    assert all(data["checks"].values())
    table = [
        "| Compiled setting | Natural responses | Output tokens | Median round | "
        "Committed tokens/round | Pooled post-first rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for casts, name, label in (
        (0, "before", "Before rounding alignment"),
        (1, "aligned", "BF16 intermediate casts preserved"),
    ):
        passes = [p for c in data["controls"] if c["precision_casts"] == casts for p in c["passes"]]
        row = data["arms"][name]
        assert len(passes) == row["responses"] == 6
        assert all(p["finish_reason"] == "stop" for p in passes)
        assert row["output_tokens"] == sum(p["output_tokens"] for p in passes)
        for p in passes:
            steps = p["steps_after_first"]
            assert sum(s["tokens"] for s in steps) == p["after_first_tokens"]
            assert math.isclose(
                p["steady_median_step_ms"],
                1000 * statistics.median(s["seconds"] for s in steps[8:]),
            )
        for key, expected in {
            "tokens_per_second": sum(p["after_first_tokens"] for p in passes)
            / sum(p["after_first_seconds"] for p in passes),
            "median_round_ms": statistics.median(p["steady_median_step_ms"] for p in passes),
            "mean_tokens_per_round": statistics.mean(p["steady_tokens_per_step"] for p in passes),
        }.items():
            assert math.isclose(row[key], expected)
        table.append(
            f"| {label} | 6 | {row['output_tokens']:,} | {row['median_round_ms']:.3f} ms | "
            f"{row['mean_tokens_per_round']:.3f} | {row['tokens_per_second']:.3f} tok/s |"
        )
    before, after = data["arms"]["before"], data["arms"]["aligned"]
    changes = data["percent_change"]
    for key, change in changes.items():
        assert math.isclose(change, 100 * (after[key] / before[key] - 1))
    summary = (
        "Preserving casts changed the median round by "
        f"**{after['median_round_ms'] - before['median_round_ms']:+.3f} ms "
        f"({changes['median_round_ms']:+.2f}%)** and the pooled token rate by "
        f"**{changes['tokens_per_second']:+.2f}%**. Mean committed tokens per round "
        f"changed by {changes['mean_tokens_per_round']:+.2f}%. The per-response median "
        f"rounds ranged from {before['round_ms_range'][0]:.3f} to "
        f"{before['round_ms_range'][1]:.3f} ms before and "
        f"{after['round_ms_range'][0]:.3f} to {after['round_ms_range'][1]:.3f} ms after."
    )
    (ROOT / "rounding-speed-table.md").write_text("\n".join(table) + "\n")
    return {"{{ROUNDING_SPEED_TABLE}}": "\n".join(table), "{{ROUNDING_SPEED_SUMMARY}}": summary}


def main():
    from build_detailed_tables import authenticated_read
    from build_execution_mode_tables import build as build_mode_tables

    modes = authenticated_read(ROOT / "evidence/current-compiled-eager-320.json")
    assert modes["decode"]["positions"] == 320 and modes["decode"]["full_logits_exact"] == 0
    assert [
        (modes["decode"][k]["set_exact"], modes["decode"][k]["ranked_exact"])
        for k in ("1", "10", "20")
    ] == [(319, 319), (146, 14), (66, 0)]
    silu = authenticated_read(ROOT / "evidence/isolated-silu-modes-320.json")
    assert silu["status"] == "SAMPLE_CHECKED" and silu["negative_control_detected"]
    assert silu["inputs_unchanged"]
    decoded = list(silu["results"]["decode"].values())
    assert len(decoded) == 64 and all(row["own_capture_reproduced"] == 320 for row in decoded)
    assert (
        sum(row["compiled_vs_eager_on_common_input"]["different_elements"] for row in decoded)
        == 96380748
    )
    assert sum(row["torch_native_vs_eager"]["exact_positions"] for row in decoded) == 20480
    assert sum(row["torch_fp32_vs_compiled"]["exact_positions"] for row in decoded) == 20463
    pilot = authenticated_read(ROOT / "evidence/rope-gate-native-pilot.json")
    assert pilot["positions"] == 8 and pilot["gate"]["gate_input_exact"]
    assert pilot["rope"]["True"]["eager_q"]["exact_positions"] == 8
    assert pilot["rope"]["True"]["eager_k"]["exact_positions"] == 8
    assert pilot["rope"]["True"]["compiled_q"]["different_elements"] == 1496
    assert pilot["gate"]["common_input_compiled_vs_eager"]["different_elements"] == 13524
    assert pilot["gate"]["compiled_reproduces_own_output"]["exact_positions"] == 8
    assert pilot["gate"]["eager_reproduces_own_output"]["exact_positions"] == 8
    precision = authenticated_read(ROOT / "evidence/precision-casts-vs-eager.json")
    assert precision["decode"]["positions"] == 320
    assert precision["decode"]["full_logits_exact"] == 0
    assert [
        (precision["decode"][k]["set_exact"], precision["decode"][k]["ranked_exact"])
        for k in ("1", "10", "20")
    ] == [(316, 316), (151, 22), (76, 0)]
    cut = authenticated_read(ROOT / "evidence/precision-attention-cut.json")
    formulas = authenticated_read(ROOT / "evidence/precision-rotary-formulae.json")
    rotary = authenticated_read(ROOT / "evidence/rotary-rne-native-replay.json")
    assert rotary["status"] == "SAMPLE_CHECKED"
    assert rotary["negative_controls_detected"] and rotary["coefficients_unchanged"]
    assert rotary["versions"] == {
        "gpu": "AMD Radeon AI PRO R9700",
        "torch": "2.12.0+rocm7.14",
        "triton": "3.7.1",
    }
    for phase, count in (("decode", 320), ("prefill", 9)):
        for name in (
            "qkv_projection",
            "query_after_normalization",
            "key_after_normalization",
            "value",
            "gate_input",
        ):
            assert cut["results"][phase][name]["exact_positions"] == count
        for kind in ("query", "key"):
            for rounding, side in (("rtz", "left"), ("rne", "right")):
                entry = formulas["rotary_formulae"][phase][f"{kind}/{rounding}_products/{side}"]
                assert entry["positions"] == entry["exact_positions"] == count
                assert entry["different_elements"] == 0
            for variant, side in (
                ("native", "eager"),
                ("rne_products", "compiled_casts"),
            ):
                entry = rotary["results"][phase][f"{variant}/{kind}/{side}"]
                assert entry["positions"] == entry["exact_positions"] == count
                assert entry["different_elements"] == 0
            for variant, side in (
                ("native", "compiled_casts"),
                ("rne_products", "eager"),
            ):
                entry = rotary["results"][phase][f"{variant}/{kind}/{side}"]
                assert entry["positions"] == count
                assert entry["exact_positions"] == (0 if phase == "decode" else 1)
    for name, different in {
        "query_after_rotation": 184437,
        "key_after_rotation": 30984,
        "attention_output": 1335594,
        "gated_attention_output": 1282380,
    }.items():
        assert cut["results"]["decode"][name]["different_elements"] == different
    common = authenticated_read(ROOT / "evidence/rotary-common-rounding-320.json")
    eager_change = authenticated_read(ROOT / "evidence/rotary-whole-model-eager-change.json")
    assert common["status"] == "COMPARED_TWO_DECLARED_INTERVENTIONS"
    assert common["observed_rotary"]["calls"] == 1344
    assert common["binding"]["isolated_native_evidence"] == rotary["sha256"]
    assert eager_change["decode"]["1"]["set_exact"] == 316
    assert eager_change["decode"]["full_logits_exact"] == 0
    for phase, count in (("decode", 320), ("prefill", 1)):
        assert common[phase]["positions"] == common[phase]["full_logits_exact"] == count
        for k in ("1", "10", "20"):
            for field in (
                "set_exact",
                "ranked_exact",
                "retained_scores_exact",
                "inclusive_tie_set_exact",
            ):
                assert common[phase][k][field] == count
    build_mode_tables(ROOT)
    from build_detailed_tables import build

    replacements = build(ROOT)
    speed = read("final-study-profile-audit.json")
    speed_table = [
        "| Compiled configuration | Natural responses | Output tokens | "
        "Timed post-first seconds | Median round | Pooled post-first rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, label in (
        ("original", "Original compiled M8"),
        ("final", "Final compiled M8: Fix 1 + Fix 2"),
    ):
        row = speed["arms"][arm]
        controls = row["natural_controls"]
        assert len(controls) == 3 and all(p["finish_reason"] == "stop" for p in controls)
        total = sum(p["output_tokens"] for p in controls)
        seconds = sum(p["after_first_seconds"] for p in controls)
        speed_table.append(
            f"| {label} | 3 | {total:,} | {seconds:.3f} | {row['median_round_ms']:.3f} ms | "
            f"{row['pooled_tokens_per_second']:.3f} tok/s |"
        )
    replacements["{{SPEED_TABLE}}"] = "\n".join(speed_table)
    replacements["{{SPEED_SUMMARY}}"] = (
        "These fresh controls use the same builds as the stage profiles, "
        "in separate unprofiled passes. "
        "All three natural responses per arm are retained in the "
        "[profile and control audit](evidence/final-study-profile-audit.json)."
    )
    comparisons = authenticated_read(ROOT / "evidence/final-study-four-comparisons.json")[
        "comparisons"
    ]
    table = [
        "**Reading paired values:** **first = same token set (any order); "
        "second = same ranked order**. `320/320; 320/320` means both checks "
        "matched at all 320 tested positions.",
        "",
        "| Compared implementations | Top-1 set/order | Top-10 set/order | "
        "Top-20 set/order | Full vectors exact |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    names = {
        "original_compiled_m8_m1": "Original compiled M8 vs original compiled M1",
        "fix1_compiled_m8_m1": "Fix 1 compiled M8 vs Fix 1 compiled M1",
        "fix1_compiled_eager_m8": "Fix 1 compiled M8 vs Fix 1 eager M8",
        "final_compiled_eager_m8": "Final compiled M8 vs final eager M8 (Fix 1 + Fix 2)",
    }
    for key, label in names.items():
        result = comparisons[key]["decode"]
        assert result["positions"] == 320
        cells = [
            f"{result[str(k)]['set_exact']}/320; {result[str(k)]['ranked_exact']}/320"
            for k in (1, 10, 20)
        ]
        table.append(f"| {label} | {' | '.join(cells)} | {result['full_logits_exact']}/320 |")
    replacements["{{FOUR_COMPARISON_TABLE}}"] = "\n".join(table)
    replacements.update(rounding_speed_tables())
    replacements.update(crossmode_corpus_tables())
    template = ROOT / "report.template.md"
    document = template.read_text()
    for marker, replacement in replacements.items():
        document = document.replace(marker, replacement)
    assert "{{" not in document, "unexpanded report field"
    (ROOT / "REPORT.md").write_text(document)
    summary = read("fixed-compiled-10k-summary.json")
    assert summary["decode"]["positions"] == 10000
    assert summary["prefill"]["positions"] == 23
    for domain in ("decode", "prefill"):
        count = summary[domain]["positions"]
        assert summary[domain]["full_logits_exact"] == count
        for k in ("1", "10", "20"):
            assert all(
                summary[domain][k][field] == count
                for field in (
                    "set_exact",
                    "ranked_exact",
                    "retained_scores_exact",
                    "inclusive_tie_set_exact",
                )
            )
    manifests = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "evidence").glob("*.json"))
    }
    (ROOT / "evidence-sha256.json").write_text(json.dumps(manifests, indent=2) + "\n")
    detailed = json.loads((ROOT / "stage-times.json").read_text())
    print(
        json.dumps(
            {
                "accounting_verified": True,
                "comparison_verified": True,
                "target_body_ms": detailed["target_body_ms"],
                "all_kernel_ms": detailed["all_kernel_ms"],
            }
        )
    )


if __name__ == "__main__":
    main()

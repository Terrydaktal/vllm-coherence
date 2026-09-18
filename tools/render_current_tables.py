#!/usr/bin/env python3
"""Render current-only README evidence from the committed aggregate measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- COHERENCE_CURRENT_RESULTS -->"
END = "<!-- /COHERENCE_CURRENT_RESULTS -->"


def render(data):
    lines = [
        START,
        "## Current numerical results",
        "",
        data["correctness_scope"],
        "",
        "| Prediction | Same token set | Same ordering | Mean shared tokens |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in data["predictions"]:
        n = row["rows"]
        k = row["k"]
        lines.append(
            f"| Top {k} | {row['set']:,} / {n:,} ({100 * row['set'] / n:.2f}%) | {row['order']:,} / {n:,} ({100 * row['order'] / n:.2f}%) | {row['overlap']:.4f} / {k} |"
        )
    lines += [
        "",
        data["correctness_limits"],
        "",
        "## Compiled backend stages",
        "",
        data["timing_scope"],
        "",
        (
            "**Set/order** means the same top-20 token set, followed by the same ranking. "
            "For example, `320/320; 320/320` means both checks passed at every tested token. "
            "Operator-byte checks and whole-model checks are labelled separately; each result retains its stated test scope."
        ),
        "",
        "| Stage | Current GPU ms per round | Current correctness evidence | Implementation / measurement boundary |",
        "| --- | ---: | --- | --- |",
    ]
    for row in data["stages"]:
        value = f"{row['ms']:.3f}" if row["ms"] is not None else row["included_in"]
        lines.append(
            f"| {row['stage']} | {value} | {row['evidence']} | {row['note']} |"
        )
    lines += [
        "",
        (
            f"**Sum of measured GPU dispatch durations: {data['gpu_ms']:.3f} ms per profiled round.** "
            "This sum excludes host gaps and queue time and is not the uninstrumented round timer."
        ),
        "",
        "<details>",
        "<summary>Every decoder layer: current projection and remaining-work timings</summary>",
        "",
        "Each layer has four projections. Gate and up are one joint GEMM; there is no separately measured gate/up split.",
        "",
        "| Layer | Type | All layer work ms | Input projection ms | Output projection ms | Gate/up projection ms | Down projection ms | Other work ms |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in data["layers"]:
        lines.append(
            f"| {row['layer']} | {row['type']} | {row['total']:.4f} | {row['input']:.4f} | {row['output']:.4f} | {row['gate_up']:.4f} | {row['down']:.4f} | {row['other']:.4f} |"
        )
    lines += [
        "",
        "</details>",
        "",
        "<details>",
        "<summary>Every recorded kernel, grouped by stage</summary>",
        "",
        "| Stage / compiled kernel | Calls in retained rounds | Current GPU ms per round |",
        "| --- | ---: | ---: |",
    ]
    for row in data["kernels"]:
        name = row["kernel"].replace("|", "&#124;").replace("`", "")
        lines.append(
            f"| {row['stage']} / `{name}` | {row['calls']} | {row['ms']:.6f} |"
        )
    lines += [
        "",
        "</details>",
        "",
        "## Uninstrumented performance",
        "",
        data["performance_scope"],
        "",
        "| Measurement | Current result |",
        "| --- | ---: |",
    ]
    for key, value in data["performance"].items():
        lines.append(f"| {key} | {value} |")
    lines += [
        "",
        data["performance_limits"],
        "",
        (
            "Aggregate data: [current measurements](benchmarks/results/coherence-current.json). "
            "Historical methodology and detailed numerical evidence: [technical report](reports/d7-rdna4-2026-09-17/REPORT.md)."
        ),
        "",
        END,
    ]
    return "\n".join(lines)


def update(text, data):
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError("README measurement markers are missing or duplicated")
    a = text.index(START)
    b = text.index(END) + len(END)
    return text[:a] + render(data) + text[b:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = ROOT / "README.md"
    actual = path.read_text()
    expected = update(
        actual,
        json.loads((ROOT / "benchmarks/results/coherence-current.json").read_text()),
    )
    if args.check:
        if actual != expected:
            raise SystemExit("README does not match current measured evidence")
    else:
        path.write_text(expected)


if __name__ == "__main__":
    main()

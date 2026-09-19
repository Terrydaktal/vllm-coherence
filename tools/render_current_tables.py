#!/usr/bin/env python3
"""Render current-only README evidence from the committed aggregate measurements."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- COHERENCE_CURRENT_RESULTS -->"
END = "<!-- /COHERENCE_CURRENT_RESULTS -->"
REPOSITORY = "https://github.com/Terrydaktal/vllm-coherence"
GLOBAL_BENCHMARK_LINES = (
    "## Global-256 target-head benchmark",
    "",
    "This is the table from [the top-256 PR](https://github.com/magiccodingman/vllm-radiance/pull/9).",
    "It is a **separate, earlier 60K-generated-token benchmark per method**, predating the",
    "M1/M8 and eager/compiled repairs and subsequent performance backports. Its old",
    "end-to-end throughput figures are intentionally omitted because they do not",
    "describe the current complete backend.",
    "",
    "| Target path | Median M8 head time | Top-1 match | Complete reference top-20 retained |",
    "|---|---:|---:|---:|",
    "| Full BF16 fallback | 4.122 ms | 119,988/119,988 (100.0000%) | 119,988/119,988 (100.0000%) |",
    "| Original block-8/64 + rerank-80 | 1.085 ms | 119,956/119,988 (99.9733%) | 98,452/119,988 (82.0515%) |",
    "| Global INT2 top-128 + BF16 rerank | 1.114 ms | 119,986/119,988 (99.9983%) | 118,254/119,988 (98.5549%) |",
    "| Global INT2 top-256 + BF16 rerank (default) | 1.128 ms | 119,986/119,988 (99.9983%) | 119,786/119,988 (99.8316%) |",
    "",
    "There were 115 natural completions per method on 11 private Pi request boundaries",
    "with 57,008–65,527 input tokens. Output totals were 60,598 / 60,075 / 60,348 /",
    "60,675 tokens for full / block / global-128 / global-256 respectively. Tools were",
    "not executed. Head timings are median eight-row GPU-event measurements on the",
    "same captured hidden vectors; 119,988 prediction rows were compared.",
    "",
    "Global-256 removes the eight-per-tile capacity limit, but remains approximate.",
    "The two changed final argmax IDs and 202 incomplete top-20 sets are observed",
    "misses. Complete top-20 retention does not certify score equality, ordering or",
    "sampling probabilities. The full BF16 path is the reference in this head study,",
    "not an independent proof of the model. [Methodology](docs/VERIFY_HEAD_GLOBAL_TOPK.md)",
    "· [Aggregate evidence](benchmarks/results/20260916-verify-head-global-topk-long/summary.json).",
)


def measurement_marker(data):
    commit = data.get("measurement_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("current measurements must identify a full commit hash")
    return f"[`{commit[:7]}`]({REPOSITORY}/commit/{commit})"


def provenance_marker(data, stage):
    provenance = data.get("stage_provenance")
    if not isinstance(provenance, dict):
        raise TypeError("compiled stages must identify their code provenance")
    entry = provenance.get(stage)
    if not isinstance(entry, dict):
        raise TypeError(f"missing code provenance for compiled stage: {stage}")
    commit = entry.get("commit")
    change = entry.get("change")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or not isinstance(change, str)
        or not change.strip()
    ):
        raise ValueError(f"invalid code provenance for compiled stage: {stage}")
    return f"[`{commit[:7]}`]({REPOSITORY}/commit/{commit}): {change}"


def commit_marker(commit):
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("evidence must identify a full commit hash")
    return f"[`{commit[:7]}`]({REPOSITORY}/commit/{commit})"


def evidence_with_commits(data, stage, evidence):
    configured = data.get("evidence_commits", {}).get(stage)
    commits = configured or [data["measurement_commit"]]
    markers = [commit_marker(commit) for commit in commits]
    parts = evidence.split("; ")
    if len(commits) > 1:
        if len(parts) != len(commits):
            raise ValueError(
                f"evidence commit count does not match semicolon-separated evidence: {stage}"
            )
        return "; ".join(f"{part} · {marker}" for part, marker in zip(parts, markers))
    return f"{evidence} · {markers[0]}"


def render_live_chat_diagnosis_section():
    lines = [
        "## 60K live-chat cache-state diagnosis",
        "",
        (
            "This separate, content-free diagnostic used the same retained 60,000-token Pi "
            "prefix with the compiled PIECEWISE backend, corrected arithmetic and Global-256 "
            "head enabled. Chat framing made the request 60,077 tokens. One natural completion "
            "produced 530 output tokens over 196 generation rounds at temperature 1, top-p 0.95, "
            "top-k 20. No prompt or output text was read or saved."
        ),
        "",
        "| Operation | Mean generation round |",
        "| --- | ---: |",
        "| Three warm repeats | 43.59 / 43.67 / 43.75 ms |",
        "| Same prompt with a fresh cache identity | 53.51 ms |",
        "| Immediate repeat on that cache | 53.71 ms |",
        "| Switch to another RAM-backed chat | 43.62 / 43.70 ms |",
        "| Return to the previously slow chat through RAM | 43.64 ms |",
        "| Another fresh cache | 53.86 ms |",
        "",
        (
            "The slow state could also be reproduced by starting a new generation of the same "
            "chat, without allocating a second chat's RAM bank. This separates the effect from "
            "ordinary two-chat RAM admission. The recorded responses had matching output hashes, "
            "output counts and acceptance patterns."
        ),
        "",
        (
            "The controlled interventions below show what changed the state. Before and after "
            "values are per-round arrival intervals; they are neither GPU-only timings nor "
            "whole-request timings."
        ),
        "",
        "| Intervention | Before | After | Intervention duration |",
        "| --- | ---: | ---: | ---: |",
        "| No-op | 53.95 ms | 52.89 ms | <0.01 ms |",
        "| Pause the host for 50 ms | 53.10 ms | 53.03 ms | 50.06 ms |",
        "| Record and synchronize timing events | 52.23 ms | 53.15 ms | 6.04 ms |",
        "| Synchronize the current stream | 52.52 ms | 43.88 ms | 6.38 ms |",
        "| Repeat stream synchronization after a fresh generation | 52.33 ms | 44.11 ms | 6.39 ms |",
        "| Synchronize the device | 52.19 ms | 43.82 ms | 6.41 ms |",
        "| Stream synchronization with original ROCr | 53.18 ms | 43.64 ms | 6.38 ms |",
        "",
        (
            "The optimized GEMM remained active in both fast and slow captures. The evidence "
            "narrowed the extra time to HIP stream/queue cleanup or dependency state; it did not "
            "establish the decisive internal fence and did not deploy a permanent fix. This is "
            "separate from the controlled old-GEMM result above: the old GEMM benchmark measured "
            "56.76 ms versus 42.85 ms after the backport, while this live-chat reproduction could "
            "still reach about 53–54 ms with the backported GEMM already enabled. See the full "
            "[round-latency evidence](docs/round-latency-20260919.md) and "
            "[machine-readable measurements](docs/round-latency-20260919.json)."
        ),
        "",
        (
            "**Status: pending fix.** The intermittent or persistent approximately 10 ms "
            "generation step is not resolved. Current evidence points to residual HIP "
            "stream/queue state after a fresh generation, cache restore or handover; the "
            "specific fence or dependency has not been identified. The context-dependent "
            "widening of the fast/slow gap also remains under investigation. A permanent "
            "repair must be qualified across warm generation, cold prefill, disk restore, "
            "RAM handover and tail flush, with unchanged-output checks. The measured stream "
            "synchronization is a diagnostic recovery control, not a production fix."
        ),
    ]
    return lines


def render(data):
    timing = data["round_timing"]
    mean = timing["mean"]
    retained_cycles = len(timing["rounds"])
    commit = measurement_marker(data)
    serving_overhead = data["serving_overhead"]
    if serving_overhead["status"] != "estimated_from_separate_runs":
        raise ValueError(
            "Serving overhead must be labelled as an estimate from separate runs"
        )
    round_ms = serving_overhead["unprofiled_round_ms"]
    overhead_ms = round_ms - data["gpu_ms"]
    if (
        not math.isfinite(round_ms)
        or round_ms <= 0
        or overhead_ms < 0
        or serving_overhead["method"] != "unprofiled_round_ms - gpu_ms"
        or not math.isclose(
            serving_overhead["ms"], overhead_ms, rel_tol=0, abs_tol=1e-8
        )
    ):
        raise ValueError("Serving overhead estimate does not match its source timings")
    if not math.isclose(data["gpu_ms"], mean["kernel_sum_ms"], rel_tol=0, abs_tol=1e-8):
        raise ValueError("Stage and overhead timing must cover the same rounds")
    if not math.isclose(
        data["gpu_ms"] - mean["gpu_overlap_ms"] + mean["overhead_ms"],
        mean["elapsed_ms"],
        rel_tol=0,
        abs_tol=1e-8,
    ):
        raise ValueError("GPU work and overhead do not reconcile to elapsed time")
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
            "Read the decode cycle from drafting to target verification. Within the target, "
            "run layers **0–63 in order**: input normalization → **GDN or attention** → "
            "post-normalization and MLP, then advance to the next layer. "
            "Each four-layer group is **GDN + MLP → GDN + MLP → GDN + MLP → attention + MLP**, "
            "repeated 16 times. Layer 0's input normalization is counted with embedding. "
            "After layer 63, run final normalization and the target head, then sample/accept "
            "tokens before the next draft."
        ),
        "",
        (
            "The table follows that sequence with alternative layer branches and the repeat "
            "boundary marked. Timings remain totals across the applicable layers per retained "
            "profile cycle. "
            "Copies and bookkeeping span parts of the cycle rather than one instant. "
            "**↳ rows are fused parts of the numbered parent stage above them. "
            "Their time is already counted in that stage's total.**"
        ),
        "",
        (
            "**Set/order** means the same top-20 token set, followed by the same ranking. "
            "For example, `320/320; 320/320` means both checks passed at every tested token. "
            "A single link after a semicolon-separated pair applies to both values; separate links identify separate runs. "
            "Operator-byte checks and whole-model checks are labelled separately; each result retains its stated test scope. "
            "The timing-table header and each correctness result link to the commit that produced that result. "
            "The provenance column links the last relevant implementation commit for the stage and states the performance or correctness change; "
            "fused rows inherit their parent stage's provenance."
        ),
        "",
        f"| Stage | Current ms per retained profile cycle (run {commit}) | Current correctness evidence | Last relevant code commit / change | What this stage does |",
        "| --- | ---: | --- | --- | --- |",
    ]
    stage_numbers = {
        row["stage"]: number
        for number, row in enumerate(
            (row for row in data["stages"] if row["ms"] is not None), start=1
        )
    }
    parent = None
    for row in data["stages"]:
        if row["ms"] is not None:
            parent = row["stage"]
            label = f"**{stage_numbers[parent]}. {parent}**"
            value = f"{row['ms']:.3f}"
        else:
            if row["included_in"] != parent:
                raise ValueError(
                    "Fused operation must follow its measured parent stage"
                )
            label = f"↳ {row['stage']}"
            value = f"Included in **stage {stage_numbers[parent]}**"
        provenance_stage = row["stage"] if row["ms"] is not None else parent
        evidence = evidence_with_commits(data, provenance_stage, row["evidence"])
        provenance = provenance_marker(data, provenance_stage)
        lines.append(
            f"| {label} | {value} | {evidence} | {provenance} | {row['note']} |"
        )
    lines += [
        f"| **26. Estimated runtime overhead** | **≈{overhead_ms:.1f}** | — · {commit} | — (derived measurement; no model-stage implementation) | {round_ms:.3f} ms unprofiled round − {data['gpu_ms']:.3f} ms kernel subtotal; approximately {100 * overhead_ms / round_ms:.1f}% of the round. |",
        "",
        (
            "Numbered stages contain GPU kernel durations from a diagnostic trace. "
            "**Runtime overhead is estimated from separate runs**, assuming the traced kernel "
            "durations are representative of normal execution. It does not use the profiler's "
            "gap total and is not a separately measured overhead figure. The complete round "
            "time with profiling disabled is reported in the 60K live-chat diagnosis below."
        ),
        "",
        (
            "Fusion still permits correctness instrumentation: a diagnostic kernel can expose "
            "intermediate values, and fused outputs can be compared with an unfused reference. "
            "The normal GPU profile measures the combined kernel. Internal probes or splitting "
            "the kernel can change its performance, so those measurements are not an additive "
            "breakdown of the production kernel's time."
        ),
        "",
        (
            f"The stage and layer values are means over {retained_cycles} retained complete "
            "profile cycles; the source trace contains eight observed cycles in total. "
            f"The kernel-detail table's call count is the total across those {retained_cycles} "
            "cycles, while its GPU-ms column is the per-cycle mean. The profile uses a "
            "60,000-input-token Pi prefix and has no natural-completion output-token count; "
            "the cycle count must not be read as an output-token count."
        ),
        "",
        "<details>",
        "<summary>All 64 decoder layers in execution order: projection and remaining-work timings</summary>",
        "",
        (
            "Finish each row's layer, including its MLP, before moving to the next row. "
            "Each layer has four projections. Gate and up are one joint GEMM; there is no "
            "separately measured gate/up split. The remaining-work column combines normalization, "
            "mixing and other operations between those projections."
        ),
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
        f"| Stage / compiled kernel | Calls in {retained_cycles} retained cycles | Current GPU ms per profile cycle |",
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
        *GLOBAL_BENCHMARK_LINES,
        "",
        *render_live_chat_diagnosis_section(),
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

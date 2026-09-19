#!/usr/bin/env python3
"""Measure stages and cycle overhead in a saved R9700 trace without GPU work."""

from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reports/d7-rdna4-2026-09-17"))
from analyze_compiled_trace import launch_groups, projection, semantic_phase


def phase(layer, region, name):
    # A fused launch has one measured duration; its constituents cannot each be
    # assigned that duration without double counting.
    if "norm_quant<" in name:
        if region == "input":
            return (
                "Embedding + first input normalization"
                if layer == 0
                else "Layer input residual/normalization"
            )
        if region == "post":
            return "Post-attention/GDN residual/normalization"
        raise ValueError("norm/quant kernel outside admitted normalization boundary")
    if "gdn_norm_quant_kernel" in name:
        if region != "mix" or layer % 4 == 3:
            raise ValueError("GDN norm/quant in another layer or region")
        return "GDN output gated normalization"
    if "silu" in name:
        if region != "activation":
            raise ValueError("SiLU outside MLP activation boundary")
        return "MLP SiLU and gating"
    if "sigmoid" in name:
        if region != "mix" or layer % 4 != 3:
            raise ValueError("attention gate outside attention boundary")
        return "Attention output gating"
    return semantic_phase(layer, region, name)


def measure_round_windows(events, starts, selected):
    """Measure complete GPU start-to-next-start cycles on one device.

    Kernel durations are summed for stage accounting; interval union measures
    occupied time without double counting concurrent streams. CPU annotations
    are not added to GPU time because their execution can overlap.
    """
    if (
        len(starts) < 2
        or not all(math.isfinite(value) for value in starts)
        or any(b <= a for a, b in pairwise(starts))
        or not selected
        or selected != sorted(set(selected))
        or any(i < 0 or i + 1 >= len(starts) for i in selected)
    ):
        raise ValueError("round timing requires ordered, bounded round windows")
    work = []
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in {
            "kernel",
            "gpu_memcpy",
            "gpu_memset",
        }:
            continue
        begin, duration = float(event["ts"]), float(event["dur"])
        if not math.isfinite(begin) or not math.isfinite(duration) or duration < 0:
            raise ValueError("invalid GPU activity interval")
        work.append((begin, duration, event))
    rows = []
    for index in selected:
        start, stop = starts[index : index + 2]
        elapsed = stop - start
        intervals, devices, streams, durations = [], set(), set(), []
        for begin, duration, event in work:
            relative = begin - start
            end = relative + duration
            if relative >= elapsed or end <= 0:
                continue
            if relative < 0 or end > elapsed:
                raise ValueError("GPU activity crosses a round boundary")
            if event["cat"] != "kernel":
                raise ValueError("GPU copy activity needs explicit stage accounting")
            devices.add(event["args"]["device"])
            streams.add(event["args"]["stream"])
            intervals.append((relative, end))
            durations.append(duration)
        if len(devices) != 1 or not intervals:
            raise ValueError("round timing requires work on exactly one GPU")
        occupied, end = [], 0.0
        for begin, finish in sorted(intervals):
            occupied.append(max(0.0, finish - max(begin, end)))
            end = max(end, finish)
        busy = math.fsum(occupied)
        kernel_sum = math.fsum(durations)
        rows.append(
            {
                "round": index,
                "start_offset_ms": (start - starts[0]) / 1000,
                "elapsed_ms": elapsed / 1000,
                "kernel_sum_ms": kernel_sum / 1000,
                "gpu_busy_ms": busy / 1000,
                "gpu_overlap_ms": max(0.0, kernel_sum - busy) / 1000,
                "overhead_ms": max(0.0, elapsed - busy) / 1000,
                "kernel_count": len(intervals),
                "streams": sorted(streams),
            }
        )
    fields = (
        "elapsed_ms",
        "kernel_sum_ms",
        "gpu_busy_ms",
        "gpu_overlap_ms",
        "overhead_ms",
    )
    return {
        "boundary": "First target GPU kernel to the first target GPU kernel of the next round.",
        "scope": "Profiled GPU cycle; gaps include launch/host/wait and possible profiler overhead. Individual causes are not attributed. Excludes request admission, prefill and Pi/transport latency.",
        "equation": "elapsed_ms = kernel_sum_ms - gpu_overlap_ms + overhead_ms",
        "mean": {
            key: math.fsum(row[key] for row in rows) / len(rows) for key in fields
        },
        "rounds": rows,
    }


def analyze(raw: bytes, head: str):
    trace = json.loads(raw)
    groups, graph_launches = launch_groups(trace["traceEvents"])
    targets = sorted((k, v) for k, v in groups.items() if k[0] == "target_body")
    if len(targets) != 8 or any(graph_launches[k] != 65 for k, _ in targets):
        raise ValueError("expected eight compiled rounds, 65 target graphs each")
    inventories = [
        tuple(sorted(Counter(e["name"] for e in v).items())) for _, v in targets
    ]
    inventory = Counter(inventories).most_common(1)[0][0]
    complete = [i for i, value in enumerate(inventories) if value == inventory]
    # The last target has no following start. Keep it out of *both* the kernel
    # averages and the elapsed averages; never invent its closing boundary.
    selected = [i for i in complete if i + 1 < len(targets)]
    if len(selected) < 6:
        raise ValueError("insufficient complete target inventories")
    starts = [min(e["ts"] for e in events) for _, events in targets]
    timing = measure_round_windows(trace["traceEvents"], starts, selected)
    records, assigned = [], set()

    def append(e, scope, index, layer, stage):
        relative = e["ts"] - starts[index]
        if relative < 0 or relative + e["dur"] > starts[index + 1] - starts[index]:
            raise ValueError("stage kernel is outside its measured round")
        if id(e) in assigned:
            raise ValueError("duplicate kernel accounting")
        assigned.add(id(e))
        records.append((scope, index, layer, stage, e["name"], float(e["dur"])))

    for index in selected:
        _, events = targets[index]
        if len({(e["args"]["device"], e["args"]["stream"]) for e in events}) != 1:
            raise ValueError("non-serial target stream")
        points = [
            (i, projection(e["name"]))
            for i, e in enumerate(events)
            if projection(e["name"])
        ]
        if [p for _, p in points] != ["split1", "split4", "split1", "split4"] * 64:
            raise ValueError("backported four-projection layer inventory differs")
        for layer in range(64):
            a, b, c, d = [points[layer * 4 + j][0] for j in range(4)]
            kind = "Attention" if layer % 4 == 3 else "GDN"
            middle = [e["name"] for e in events[a + 1 : b]]
            if kind == "GDN":
                if (
                    sum("conv" in n for n in middle) != 1
                    or sum("recurrent" in n or "stock_gdn_scan" in n for n in middle)
                    != 1
                ):
                    raise ValueError("incomplete GDN layer inventory")
            elif (
                sum("attn_decode" in n or "shared_decode" in n for n in middle) != 1
                or sum("splitkv_combine" in n or "shared_merge" in n for n in middle)
                != 1
            ):
                raise ValueError("incomplete attention layer inventory")
            if sum("silu" in e["name"] for e in events[c + 1 : d]) != 1:
                raise ValueError("MLP activation inventory differs")
            start = 0 if layer == 0 else points[layer * 4 - 1][0] + 1
            for begin, end, region in (
                (start, a, "input"),
                (a + 1, b, "mix"),
                (b + 1, c, "post"),
                (c + 1, d, "activation"),
            ):
                for event in events[begin:end]:
                    append(
                        event,
                        "target_body",
                        index,
                        layer,
                        phase(layer, region, event["name"]),
                    )
            for point, stage in (
                (a, f"{kind} input projection"),
                (b, f"{kind} output projection"),
                (c, "MLP gate/up projection"),
                (d, "MLP down projection"),
            ):
                append(events[point], "target_body", index, layer, stage)
        for event in events[points[-1][0] + 1 :]:
            append(event, "target_body", index, None, "Final normalization/layout")
    ordinal = {
        key: i
        for scope in ("drafter", "target_vocabulary_head")
        for i, key in enumerate(sorted(k for k in groups if k[0] == scope))
    }
    for scope in ("drafter", "target_vocabulary_head"):
        if sum(k[0] == scope for k in groups) != 8:
            raise ValueError("incomplete head/drafter scopes")
    for key, events in groups.items():
        scope, _ = key
        if scope == "target_body":
            continue
        for event in events:
            index = bisect.bisect_right(starts, event["ts"]) - 1
            if key in ordinal and index != ordinal[key]:
                raise ValueError("non-serial head/drafter ordering")
            if index in selected:
                stage = {
                    "drafter": "Drafter",
                    "target_vocabulary_head": f"Target head ({head})",
                }.get(scope, "Other GPU bookkeeping")
                append(event, scope, index, None, stage)
    stages, layers, kernels, per_round = (
        defaultdict(float),
        defaultdict(lambda: defaultdict(float)),
        defaultdict(lambda: [0, 0.0]),
        defaultdict(float),
    )
    divisor = 1000 * len(selected)
    for scope, index, layer, stage, name, duration in records:
        stages[stage] += duration / divisor
        per_round[index] += duration / 1000
        if layer is not None:
            layers[layer][stage] += duration / divisor
        row = kernels[(scope, stage, name)]
        row[0] += 1
        row[1] += duration / divisor
    if abs(sum(stages.values()) - sum(per_round.values()) / len(selected)) > 1e-8:
        raise ValueError("duration accounting failure")
    counts = Counter(record[1] for record in records)
    for row in timing["rounds"]:
        if counts[row["round"]] != row["kernel_count"] or not math.isclose(
            per_round[row["round"]], row["kernel_sum_ms"], rel_tol=0, abs_tol=1e-8
        ):
            raise ValueError("stage and elapsed timing cover different GPU work")
    return {
        "schema": "urn:coherence:compiled-stage-timings:v2",
        "trace_sha256": hashlib.sha256(raw).hexdigest(),
        "scope": "Saved compiled 60K-input Pi decode trace. GPU dispatch sums and elapsed GPU cycles from the same bounded rounds; profiled, not uninstrumented serving latency. No new GPU run.",
        "profile_rounds": 8,
        "included_rounds": selected,
        "selection": "Complete modal kernel inventory and an observed next-target start; no duration-based exclusions. Work before the first target is excluded.",
        "graph_launches_per_round": [graph_launches[k] for k, _ in targets],
        "target_head": head,
        "stages_ms": dict(stages),
        "layers_ms": dict(layers),
        "all_gpu_ms": sum(stages.values()),
        "per_round_gpu_ms": dict(per_round),
        "round_timing": timing,
        "kernel_groups": [
            {
                "scope": scope,
                "stage": stage,
                "kernel": name,
                "calls": values[0],
                "ms_per_round": values[1],
            }
            for (scope, stage, name), values in sorted(kernels.items())
        ],
        "omitted_inventory_rounds": [i for i in range(8) if i not in complete],
        "omitted_unbounded_rounds": [i for i in complete if i + 1 == len(targets)],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", required=True, type=Path)
    p.add_argument("--head", required=True, choices=["global256", "full-bf16"])
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    raw = a.trace.read_bytes()
    if a.trace.suffix == ".gz":
        raw = gzip.decompress(raw)
    value = analyze(raw, a.head)
    value["trace_file_sha256"] = hashlib.sha256(a.trace.read_bytes()).hexdigest()
    value["analyzer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(value, indent=2) + "\n")
    print(
        json.dumps(
            {
                **{
                    k: value[k]
                    for k in ("included_rounds", "all_gpu_ms", "per_round_gpu_ms")
                },
                "round_timing_ms": value["round_timing"]["mean"],
            }
        )
    )


if __name__ == "__main__":
    main()

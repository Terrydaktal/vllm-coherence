#!/usr/bin/env python3
"""Reattribute a saved compiled R9700 trace; never launch inference or read tokens."""

from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
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
    selected = [i for i, value in enumerate(inventories) if value == inventory]
    if len(selected) < 6:
        raise ValueError("insufficient complete target inventories")
    records = []

    def append(e, scope, index, layer, stage):
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
    starts = [min(e["ts"] for e in events) for _, events in targets]
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
            index = max(0, bisect.bisect_right(starts, event["ts"]) - 1)
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
    return {
        "schema": "urn:coherence:compiled-stage-timings:v1",
        "trace_sha256": hashlib.sha256(raw).hexdigest(),
        "scope": "Saved compiled 60K-input Pi decode trace. GPU dispatch sums, not uninstrumented wall latency; no new GPU run.",
        "profile_rounds": 8,
        "included_rounds": selected,
        "selection": "Complete modal kernel inventory only; no duration-based exclusions.",
        "graph_launches_per_round": [graph_launches[k] for k, _ in targets],
        "target_head": head,
        "stages_ms": dict(stages),
        "layers_ms": dict(layers),
        "all_gpu_ms": sum(stages.values()),
        "per_round_gpu_ms": dict(per_round),
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
        "omitted_inventory_rounds": [i for i in range(8) if i not in selected],
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
            {k: value[k] for k in ("included_rounds", "all_gpu_ms", "per_round_gpu_ms")}
        )
    )


if __name__ == "__main__":
    main()

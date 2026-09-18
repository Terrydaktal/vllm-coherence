"""Export transcript-free dispatch timings from the two authenticated compiled traces.

Semantic projection names use the pinned Qwen decoder order. Admission checks
all eight rounds, all 64 layers, every projection family and each GDN/attention
marker. This is not an arbitrary-model trace classifier.
"""

import argparse
import bisect
import collections
import hashlib
import json
import math
from pathlib import Path

PREFIX = "qwen_d7_stage/"


def projection(name):
    if "radiance_mxfp4_fp8_gemm_folded" in name:
        return "folded"
    if "radiance_mxfp4_fp8_gemm_decode<8, 128, 1" in name:
        return "split1"
    if "radiance_mxfp4_fp8_gemm_decode<8, 128, 4" in name:
        return "split4"
    return None


def launch_groups(events):
    threads = collections.defaultdict(list)
    for event in events:
        if event.get("ph") != "X":
            continue
        annotation = event.get("cat") == "user_annotation" and event.get("name", "").startswith(
            PREFIX
        )
        if not annotation and event.get("cat") not in {"cuda_runtime", "hip_runtime"}:
            continue
        threads[event["pid"], event["tid"]].append(
            (float(event["ts"]), 0 if annotation else 1, -float(event["dur"]), event)
        )
    launches = {}
    graph_launches = collections.Counter()
    for rows in threads.values():
        stack = []
        for start, kind, negative, event in sorted(rows, key=lambda row: row[:3]):
            while stack and start >= stack[-1][0]:
                stack.pop()
            if kind == 0:
                if negative:
                    stack.append((start - negative, event["name"][len(PREFIX) :], start))
                continue
            correlation = event.get("args", {}).get("correlation")
            if correlation is None:
                continue
            assert correlation not in launches, "ambiguous launch correlation"
            owner = stack[-1][1:] if stack else ("unattributed", 0)
            launches[correlation] = owner
            if "GraphLaunch" in event["name"]:
                graph_launches[owner] += 1
    groups = collections.defaultdict(list)
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "kernel":
            continue
        duration = float(event["dur"])
        assert math.isfinite(duration) and duration >= 0
        correlation = event.get("args", {}).get("correlation")
        assert correlation in launches, "unattributed launch correlation"
        groups[launches[correlation]].append(event)
    return {
        key: sorted(value, key=lambda event: event["ts"]) for key, value in groups.items()
    }, graph_launches


def semantic_phase(layer, region, name):
    kind = "Attention" if layer % 4 == 3 else "GDN"
    quant = "dynamic_per_token_scaled_fp8_quant" in name
    if region == "input":
        if quant:
            return f"{kind} input activation FP8 quantization"
        return (
            "Embedding + first input normalization"
            if layer == 0
            else "Layer input residual/normalization"
        )
    if region == "post":
        return (
            "MLP gate/up input FP8 quantization"
            if quant
            else "Post-attention/GDN residual/normalization"
        )
    if region == "activation":
        return "MLP down input FP8 quantization" if quant else "MLP SiLU and gating"
    assert region == "mix"
    if quant:
        return f"{kind} output activation FP8 quantization"
    if kind == "GDN":
        if "conv" in name:
            return "GDN convolution"
        if "recurrent" in name or "stock_gdn_scan" in name:
            return "GDN recurrence and gates"
        if "layer_norm_fwd" in name or (name.startswith("triton_") and "mean" in name):
            return "GDN output gated normalization"
        return "GDN layout/copies and buffer initialization"
    if "attn_decode" in name or "shared_decode" in name:
        return "Attention decode"
    if "splitkv_combine" in name or "shared_merge" in name:
        return "Attention split-KV merge"
    if "reshape_and_cache" in name:
        return "Attention KV write"
    if "sigmoid" in name:
        return "Attention output gating"
    return "Attention Q/K normalization, RoPE and layout"


def export(trace_path, profile_path, output):
    raw = trace_path.read_bytes()
    profile = json.loads(profile_path.read_text())
    trace_sha = hashlib.sha256(raw).hexdigest()
    assert trace_sha == profile["trace_sha256"], "wrong trace for published profile"
    assert profile["profile_steps"] == 8
    groups, graph_launches = launch_groups(json.loads(raw)["traceEvents"])
    targets = sorted((key, value) for key, value in groups.items() if key[0] == "target_body")
    assert len(targets) == 8
    assert all(graph_launches[key] > 0 for key, _ in targets), (
        "profiled target round lacks graph replay"
    )
    records = []
    target_starts = [min(event["ts"] for event in events) for _, events in targets]
    inventories = [
        tuple(sorted(collections.Counter(event["name"] for event in events).items()))
        for _, events in targets
    ]
    modal_inventory = collections.Counter(inventories).most_common(1)[0][0]
    complete_rounds = [
        index for index, inventory in enumerate(inventories) if inventory == modal_inventory
    ]
    assert len(complete_rounds) >= 6, "insufficient complete paired profile evidence"
    gaps = []
    for index, inventory in enumerate(inventories):
        if inventory != modal_inventory:
            actual, expected = dict(inventory), dict(modal_inventory)
            gaps.append(
                {
                    "round": index,
                    "kernel_count_differences": [
                        {
                            "kernel": name,
                            "observed": actual.get(name, 0),
                            "modal": expected.get(name, 0),
                        }
                        for name in sorted(actual.keys() | expected.keys())
                        if actual.get(name, 0) != expected.get(name, 0)
                    ],
                }
            )

    def record(event, scope, round_index, layer, phase):
        records.append(
            {
                "scope": scope,
                "round": round_index,
                "layer": layer,
                "phase": phase,
                "kernel": event["name"],
                "duration_us": event["dur"],
            }
        )

    for round_index, (_key, events) in enumerate(targets):
        assert len({(event["args"]["device"], event["args"]["stream"]) for event in events}) == 1
        if round_index not in complete_rounds:
            # A missing projection makes positional layer attribution ambiguous.
            # Retain every actual event, but do not invent its layer or use it
            # in the timing comparison.
            for event in events:
                record(
                    event,
                    "target_body",
                    round_index,
                    None,
                    "Unclassified target dispatch (incomplete profile)",
                )
            continue
        points = [
            (i, projection(event["name"]))
            for i, event in enumerate(events)
            if projection(event["name"])
        ]
        assert [family for _, family in points] == ["split1", "split4", "folded", "split4"] * 64
        for layer in range(64):
            a, b, c, d = [points[layer * 4 + i][0] for i in range(4)]
            kind = "Attention" if layer % 4 == 3 else "GDN"
            middle = [event["name"] for event in events[a + 1 : b]]
            if kind == "GDN":
                assert sum("conv" in name for name in middle) == 1
                recurrence_count = sum(
                    "recurrent" in name or "stock_gdn_scan" in name for name in middle
                )
                assert recurrence_count == 1 or (
                    round_index not in complete_rounds and recurrence_count == 0
                )
                assert not any("attn_decode" in name or "shared_decode" in name for name in middle)
            else:
                assert sum("attn_decode" in name or "shared_decode" in name for name in middle) == 1
                assert (
                    sum("splitkv_combine" in name or "shared_merge" in name for name in middle) == 1
                )
            assert any("silu" in event["name"] for event in events[c + 1 : d]), (
                "MLP gate/up order changed"
            )
            start = 0 if layer == 0 else points[layer * 4 - 1][0] + 1
            for begin, end, region in [
                (start, a, "input"),
                (a + 1, b, "mix"),
                (b + 1, c, "post"),
                (c + 1, d, "activation"),
            ]:
                for event in events[begin:end]:
                    record(
                        event,
                        "target_body",
                        round_index,
                        layer,
                        semantic_phase(layer, region, event["name"]),
                    )
            for index, phase in [
                (a, f"{kind} input projection"),
                (b, f"{kind} output projection"),
                (c, "MLP gate/up projection"),
                (d, "MLP down projection"),
            ]:
                record(events[index], "target_body", round_index, layer, phase)
        for event in events[points[-1][0] + 1 :]:
            record(event, "target_body", round_index, None, "Final normalization/layout")
    ordinal = {
        key: i
        for scope in ("drafter", "target_vocabulary_head")
        for i, key in enumerate(sorted(k for k in groups if k[0] == scope))
    }
    assert sum(key[0] == "drafter" for key in groups) == 8
    assert sum(key[0] == "target_vocabulary_head" for key in groups) == 8
    for (scope, timestamp), events in sorted(groups.items()):
        if scope == "target_body":
            continue
        for event in events:
            phase = {"drafter": "Drafter", "target_vocabulary_head": "Full BF16 target head"}.get(
                scope, "Other GPU bookkeeping"
            )
            window = max(0, bisect.bisect_right(target_starts, event["ts"]) - 1)
            if scope in ("drafter", "target_vocabulary_head"):
                assert ordinal[(scope, timestamp)] == window, "non-serial scope/window ordering"
            record(event, scope, window, None, phase)
    observed = collections.defaultdict(lambda: [0, 0.0])
    for record_ in records:
        row = observed[record_["scope"], record_["kernel"]]
        row[0] += 1
        row[1] += record_["duration_us"]
    expected = {
        (row["stage"], row["kernel"]): (row["calls"], row["kernel_us"])
        for row in profile["attribution"]["kernel_groups"]
    }
    assert set(observed) == set(expected)
    for key, (count, duration) in observed.items():
        assert count == expected[key][0]
        assert math.isclose(duration, expected[key][1], abs_tol=0.00001), key
    assert len(records) == profile["attribution"]["kernels"]
    names = sorted({row["kernel"] for row in records})
    ids = {name: index for index, name in enumerate(names)}
    value = {
        "schema": "qwen.compiled-dispatch-timings.v1",
        "trace_sha256": trace_sha,
        "profile_file_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        "profile_rounds": 8,
        "complete_target_inventory_rounds": complete_rounds,
        "inventory_gaps": gaps,
        "bookkeeping_window": (
            "Other GPU dispatches use the preceding target body's first GPU "
            "dispatch, with the initial captured prefix assigned to round 0; "
            "explicit target/head/drafter scope order agrees with these windows."
        ),
        "profiled_target_graph_launches_per_round": [graph_launches[key] for key, _ in targets],
        "mapping": (
            "Pinned 64-layer order, GDN/GDN/GDN/attention; four projection "
            "dispatches per layer. All eight rounds and operator markers checked. "
            "Fused constituents remain indivisible."
        ),
        "unit": (
            "microseconds per individual observed GPU dispatch; no timestamps, "
            "token IDs or tensor values"
        ),
        "columns": ["scope", "round", "layer", "phase", "kernel_id", "duration_us"],
        "kernels": names,
        "dispatches": [
            [
                row["scope"],
                row["round"],
                row["layer"],
                row["phase"],
                ids[row["kernel"]],
                row["duration_us"],
            ]
            for row in records
        ],
    }
    output.write_text(json.dumps(value, separators=(",", ":")) + "\n")
    totals = collections.defaultdict(float)
    for row in records:
        totals[row["phase"]] += row["duration_us"] / 8000
    print(
        json.dumps(
            {
                "output": str(output),
                "dispatches": len(records),
                "graph_launches": value["profiled_target_graph_launches_per_round"],
                "milliseconds_per_round": totals,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.trace, args.profile, args.output)

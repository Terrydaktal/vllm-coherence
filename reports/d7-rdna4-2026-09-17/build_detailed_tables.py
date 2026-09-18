"""Audit every retained dispatch and build tables at measured kernel/layer boundaries."""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict

STAGES = {
    "Embedding + first input normalization": (
        "Fix 1: normalization; embedding unchanged",
        (
            "Preserve the reference normalization rounding; the original "
            "embedding/norm fusion is indivisible."
        ),
    ),
    "Layer input residual/normalization": (
        "Fix 1 + performance",
        "Preserve reduction and rounding; retain FP32 residual sums in registers.",
    ),
    "GDN input activation FP8 quantization": (
        "No correctness repair",
        ("Same quantization kernel and call count; the small timing delta is unassigned."),
    ),
    "GDN input projection": (
        "No correctness repair",
        (
            "Unchanged MXFP4 projection arithmetic; timing differences are "
            "observations, not a projection optimization."
        ),
    ),
    "GDN layout/copies and buffer initialization": (
        "Performance only",
        (
            "Preserve packed QKV views; remove split materializations and "
            "repacking. Remaining copies are included."
        ),
    ),
    "GDN convolution": (
        "Fix 1 + performance",
        (
            "Match serial product/accumulation order and rolling history; packed "
            "transport removes surrounding copies."
        ),
    ),
    "GDN recurrence and gates": (
        "Fix 1",
        "Match gate precision, reduction order, recurrent-state transition and output rounding.",
    ),
    "GDN output gated normalization": (
        "Fix 1 + performance",
        (
            "Keep the serial row tile while processing independent rows "
            "concurrently; original fused constituents remain grouped."
        ),
    ),
    "GDN output activation FP8 quantization": (
        "No correctness repair",
        ("Same quantization kernel and call count; the small timing delta is unassigned."),
    ),
    "GDN output projection": (
        "No correctness repair",
        "Unchanged MXFP4 arithmetic; no causal speedup claimed.",
    ),
    "Attention input activation FP8 quantization": (
        "No correctness repair",
        "Same quantization kernel and call count; timing cause is not isolated.",
    ),
    "Attention input projection": (
        "No correctness repair",
        "Unchanged QKV MXFP4 projection; no causal speedup claimed.",
    ),
    "Attention Q/K normalization, RoPE and layout": (
        "Fix 1: normalization; Fix 2: RoPE rounding",
        (
            "Preserve serial normalization and BF16 RoPE product rounding. "
            "Fused original constituents share one timing."
        ),
    ),
    "Attention KV write": (
        "No correctness repair",
        "Same cache-write kernel and KV format; timing cause is not isolated.",
    ),
    "Attention decode": (
        "Fix 1 + performance",
        (
            "Preserve each query's causal tile/softmax decisions. Share KV reads "
            "within one or two tile-aligned query groups; two groups repeat context"
            " work."
        ),
    ),
    "Attention split-KV merge": (
        "Fix 1",
        (
            "Use each query's serial split/merge arithmetic; the observed reduction"
            " has not been isolated from the decode change."
        ),
    ),
    "Attention output gating": (
        "Fix 2: intermediate rounding",
        "Preserve the BF16 sigmoid result before multiplying the output gate.",
    ),
    "Attention output activation FP8 quantization": (
        "No correctness repair",
        "Same quantization kernel and call count; timing cause is not isolated.",
    ),
    "Attention output projection": (
        "No correctness repair",
        "Unchanged output MXFP4 projection; no causal speedup claimed.",
    ),
    "Post-attention/GDN residual/normalization": (
        "Fix 1 + performance",
        "Preserve serial reduction/rounding and keep residual values in registers.",
    ),
    "MLP gate/up input FP8 quantization": (
        "No correctness repair",
        "Same quantization kernel and call count; timing cause is not isolated.",
    ),
    "MLP gate/up projection": (
        "No correctness repair",
        (
            "One joint gate/up GEMM per layer, 64 per round. The per-layer table "
            "subdivides this total; gate and up have no separate measured "
            "durations."
        ),
    ),
    "MLP SiLU and gating": (
        "Fix 2: intermediate rounding",
        "Preserve the BF16 SiLU result before multiplication; retain one fused pointwise launch.",
    ),
    "MLP down input FP8 quantization": (
        "No correctness repair",
        "Same quantization kernel and call count; timing cause is not isolated.",
    ),
    "MLP down projection": (
        "No correctness repair",
        "Unchanged MXFP4 down projection; no causal speedup claimed.",
    ),
    "Final normalization/layout": (
        "Fix 1",
        (
            "Preserve final-normalization reduction and the BF16 rounding of its "
            "retained residual sum; isolated check includes that fused addition."
        ),
    ),
    "Full BF16 target head": (
        "Fix 1 + performance",
        (
            "Interleave two arithmetic-preserving M4 groups in one HIP launch, "
            "removing duplicated launch and concatenation overhead."
        ),
    ),
    "Drafter": (
        "No target correctness repair",
        (
            "Unchanged proposal model; its timings and predictions are not "
            "target-M1 equivalence measurements."
        ),
    ),
    "Other GPU bookkeeping": (
        "Not an isolated numerical stage",
        (
            "Sampling/state bookkeeping outside the model scopes. Exact semantic "
            "attribution is unavailable; each kernel remains listed below."
        ),
    ),
}


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-5), (a, b)


def authenticated_read(path):
    data = json.loads(path.read_text())
    unsigned = {k: v for k, v in data.items() if k != "sha256"}
    encoded = json.dumps(unsigned, allow_nan=False, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(encoded.encode()).hexdigest() == data["sha256"], path.name
    return data


def isolated_measurements(root):
    from combine_native_stage_evidence import combine

    evidence = root / "evidence"
    data = authenticated_read(evidence / "final-study-isolated-stage-matrix.json")
    reconstructed = combine([evidence / name for name in data["receipts"]])
    assert reconstructed == data, "isolated aggregate differs from native receipts"
    return data


def build(root):
    audit = authenticated_read(root / "evidence/final-study-profile-audit.json")
    assert audit["status"] == "MEASURED"
    exports = {}
    for arm in ("old", "fixed"):
        label = "original" if arm == "old" else "fixed"
        audit_arm = "original" if arm == "old" else "final"
        profile_path = root / "evidence" / f"final-study-{label}-profile.json"
        profile = json.loads(profile_path.read_text())
        export_path = root / "evidence" / f"final-study-{label}-dispatches.json"
        export = json.loads(export_path.read_text())
        assert (
            hashlib.sha256(export_path.read_bytes()).hexdigest()
            == audit["arms"][audit_arm]["dispatch_file_sha256"]
        )
        assert (
            export["profile_file_sha256"] == hashlib.sha256(profile_path.read_bytes()).hexdigest()
        )
        assert export["trace_sha256"] == profile["trace_sha256"]
        assert export["profile_rounds"] == profile["profile_steps"] == 8
        assert export["profiled_target_graph_launches_per_round"] == [65] * 8
        counts, durations = Counter(), defaultdict(float)
        for scope, index, layer, phase, kernel_id, us in export["dispatches"]:
            assert index in range(8)
            assert phase in STAGES or (
                phase == "Unclassified target dispatch (incomplete profile)"
                and index not in export["complete_target_inventory_rounds"]
            )
            assert layer is None or layer in range(64)
            assert math.isfinite(us) and us >= 0
            key = scope, export["kernels"][kernel_id]
            counts[key] += 1
            durations[key] += us
        expected = {(r["stage"], r["kernel"]): r for r in profile["attribution"]["kernel_groups"]}
        assert counts.keys() == expected.keys()
        for key, count in counts.items():
            assert count == expected[key]["calls"]
            close(durations[key], expected[key]["kernel_us"])
        exports[arm] = export
    rounds = sorted(
        set.intersection(*(set(e["complete_target_inventory_rounds"]) for e in exports.values()))
    )
    assert rounds == audit["paired_profile_rounds"]
    divisor = 1000 * len(rounds)
    phases, layers, kernel_rows, selected = {}, {}, {}, {}
    for arm, export in exports.items():
        selected[arm] = [r for r in export["dispatches"] if r[1] in rounds]
        phases[arm], layers[arm], kernel_rows[arm] = (
            defaultdict(float),
            defaultdict(float),
            defaultdict(lambda: [0, 0.0]),
        )
        for scope, _index, layer, phase, kernel_id, us in selected[arm]:
            phases[arm][phase] += us / divisor
            if layer is not None:
                layers[arm][layer, phase] += us / divisor
            row = kernel_rows[arm][scope, phase, export["kernels"][kernel_id]]
            row[0] += 1
            row[1] += us / divisor
        close(sum(phases[arm].values()), sum(r[5] for r in selected[arm]) / divisor)
    correctness = isolated_measurements(root)
    with (root / "isolated-stage-layer-comparisons.csv").open("w", newline="") as handle:
        columns = ["stage", "layer", "comparison", "positions", "full_logits_exact"] + [
            f"top{k}_{kind}_exact" for k in (1, 10, 20) for kind in ("set", "order")
        ]
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for stage, instances in sorted(correctness["per_instance"].items()):
            for instance, comparisons in sorted(instances.items()):
                for comparison, values in sorted(comparisons.items()):
                    writer.writerow(
                        {
                            "stage": stage,
                            "layer": instance,
                            "comparison": {
                                "old": "Old compiled M8 vs old compiled M1",
                                "fixed": "Fix 1 compiled M8 vs Fix 1 compiled M1",
                                "fix1_modes": "Fix 1 compiled M8 vs Fix 1 eager M8",
                                "final_modes": "Final compiled M8 vs final eager M8",
                            }[comparison],
                            **values,
                        }
                    )
    rows = []
    for name, (repair, explanation) in STAGES.items():
        shared_stage = (
            "Attention decode and split-KV merge"
            if name in ("Attention decode", "Attention split-KV merge")
            else name
        )
        measurements = correctness["stages"].get(shared_stage, {})
        if name not in {
            "Drafter",
            "Other GPU bookkeeping",
            "GDN layout/copies and buffer initialization",
        }:
            assert set(measurements) == {"old", "fixed", "fix1_modes", "final_modes"}, (
                "incomplete isolated stage columns",
                name,
            )
        row = {
            "stage": name,
            "correctness_change": repair,
            "old_ms": phases["old"][name],
            "fixed_ms": phases["fixed"][name],
            "explanation": explanation,
            "isolated_320": measurements,
        }
        row["delta_ms"] = row["fixed_ms"] - row["old_ms"]
        rows.append(row)
    table = [
        "**Reading each top-20 pair:** **first = the same 20 tokens, in any order; "
        "second = those 20 tokens in exactly the same ranked order**. "
        "`320/320; 320/320` means both checks passed at all 320 tested positions. "
        "A position passes a stage only when every tested layer instance agrees.",
        "",
        (
            "| Compiled stage | Correctness fix? | Old compiled M8 | "
            "Final fixed compiled M8 ms | Change ms | "
            "Old compiled M8 vs old compiled M1<br>Top-20 set/order | "
            "Fix 1 compiled M8 vs Fix 1 compiled M1<br>Top-20 set/order | "
            "Fix 1 compiled M8 vs Fix 1 eager M8<br>Top-20 set/order | "
            "Final compiled M8 vs final eager M8<br>Top-20 set/order | Timing explanation |"
        ),
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        scores = []
        for arm in ("old", "fixed", "fix1_modes", "final_modes"):
            result = row["isolated_320"].get(arm)
            if result is None:
                scores.append(
                    "N/A: not a target prediction stage"
                    if row["stage"] in ("Drafter", "Other GPU bookkeeping")
                    else "Within convolution/recurrence cuts"
                )
            else:
                assert result["positions"] == 320 and result["isolated_inputs_verified"]
                assert result["reference_remainder_verified"]
                suffix = (
                    " †" if row["stage"] in ("Attention decode", "Attention split-KV merge") else ""
                )
                scores.append(
                    f"{result['top20_set_exact']}/320; {result['top20_order_exact']}/320{suffix}"
                )
        delta = f"{row['delta_ms']:+.3f}" if abs(row["delta_ms"]) >= 0.0005 else "0.000"
        table.append(
            f"| {row['stage']} | {row['correctness_change']} | {row['old_ms']:.3f} | "
            f"{row['fixed_ms']:.3f} | {delta} | {scores[0]} | {scores[1]} | "
            f"{scores[2]} | {scores[3]} | "
            f"{row['explanation']} |"
        )
    layer_table = [
        (
            "| Layer (zero-based) | Type | All layer work old ms | Final fixed ms | "
            "Gate/up old ms | Final fixed ms | Other three projections old ms | Final fixed ms "
            "| Non-projection old ms | Final fixed ms |"
        ),
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    layer_data = []
    for layer in range(64):
        data = {"layer": layer, "type": "Attention" if layer % 4 == 3 else "GDN"}
        for arm in exports:
            group = {p: us for (i, p), us in layers[arm].items() if i == layer}
            total = sum(group.values())
            gate = group["MLP gate/up projection"]
            other = sum(
                us
                for p, us in group.items()
                if p.endswith(" projection") and p != "MLP gate/up projection"
            )
            data[arm] = {
                "total_ms": total,
                "gate_up_ms": gate,
                "other_projections_ms": other,
                "non_projection_ms": total - gate - other,
                "phases_ms": group,
            }
        layer_data.append(data)
        values = [
            data[a][key]
            for key in ("total_ms", "gate_up_ms", "other_projections_ms", "non_projection_ms")
            for a in exports
        ]
        layer_table.append(
            f"| {layer} | {data['type']} | " + " | ".join(f"{x:.4f}" for x in values) + " |"
        )
    kernel_table = [
        (
            "| Scope / stage / compiled kernel | Calls old / fixed | Old ms per "
            "round | Final fixed ms per round |"
        ),
        "| --- | ---: | ---: | ---: |",
    ]
    dispatches = []
    for key in sorted(set().union(*(set(v) for v in kernel_rows.values()))):
        a, b = [kernel_rows[arm].get(key, [0, 0.0]) for arm in exports]
        scope, phase, name = key
        escaped_name = name.replace("<", "&lt;").replace(">", "&gt;")
        kernel_table.append(
            f"| {scope} / {phase}<br><code>{escaped_name}</code> | "
            f"{a[0]} / {b[0]} | {a[1]:.6f} | {b[1]:.6f} |"
        )
        for arm, (count, ms) in zip(exports, (a, b), strict=True):
            if count:
                dispatches.append(
                    {
                        "arm": arm,
                        "scope": scope,
                        "stage": phase,
                        "kernel": name,
                        "calls": count,
                        "milliseconds_per_round": ms,
                    }
                )
    target = {a: sum(r[5] for r in selected[a] if r[0] == "target_body") / divisor for a in exports}
    totals = {a: sum(phases[a].values()) for a in exports}
    result = {
        "unit": "GPU dispatch duration summed per round, milliseconds",
        "versions": {"old": "original compiled M8", "fixed": "final compiled M8: Fix 1 + Fix 2"},
        "profile_audit": audit["sha256"],
        "paired_profile_rounds_zero_based": rounds,
        "selection": (
            "Intersection of complete modal target kernel inventories; no timing-based exclusions."
        ),
        "rows": rows,
        "layers": layer_data,
        "target_body_ms": target,
        "all_kernel_ms": totals,
        "per_round": {
            a: {
                str(i): {
                    p: sum(r[5] for r in selected[a] if r[1] == i and r[3] == p) / 1000
                    for p in STAGES
                }
                for i in rounds
            }
            for a in exports
        },
    }
    for name, content in [
        ("stage-table.md", table),
        ("layer-table.md", layer_table),
        ("kernel-table.md", kernel_table),
    ]:
        (root / name).write_text("\n".join(content) + "\n")
    (root / "stage-times.json").write_text(json.dumps(result, indent=2) + "\n")
    with (root / "kernel-dispatches.csv").open("w") as out:
        writer = csv.DictWriter(out, fieldnames=list(dispatches[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(dispatches)
    return {
        "{{STAGE_TABLE}}": "\n".join(table),
        "{{LAYER_TABLE}}": "\n".join(layer_table),
        "{{KERNEL_TABLE}}": "\n".join(kernel_table),
        "{{TARGET_TOTAL}}": f"{target['old']:.3f} ms old; {target['fixed']:.3f} ms fixed",
        "{{ALL_KERNEL_TOTAL}}": f"{totals['old']:.3f} ms old; {totals['fixed']:.3f} ms fixed",
    }

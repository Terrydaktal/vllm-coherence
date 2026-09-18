"""Disjoint GPU-kernel attribution for a pinned, eager D7 profile.

Native HIP calls can inherit stale PyTorch external IDs. Chrome-trace runtime
correlation plus the launch's host scope is authoritative for those calls.
ROCm device annotation spans are not kernel timings and are never summed.
"""

import math
import re
from collections import defaultdict

PREFIX = "qwen_d7_stage/"


def module_stage(name):
    if name.endswith("embed_tokens"):
        return "embedding"
    if name.endswith("model.norm"):
        return "final_norm"
    if re.search(r"\.layers\.\d+$", name):
        return "layer_other"
    suffixes = {
        ".input_layernorm": "input_norm_residual",
        ".post_attention_layernorm": "post_attention_norm_residual",
        ".linear_attn.in_proj_qkvz": "gdn_projections",
        ".linear_attn.in_proj_ba": "gdn_projections",
        ".linear_attn.norm": "gdn_output_norm",
        ".linear_attn.out_proj": "gdn_output_projection",
        ".linear_attn": "gdn_other",
        ".self_attn.qkv_proj": "attention_qkv_projection",
        ".self_attn.q_norm": "attention_qk_norm",
        ".self_attn.k_norm": "attention_qk_norm",
        ".self_attn.rotary_emb": "rope",
        ".self_attn.attn": "kv_cache_and_attention",
        ".self_attn.o_proj": "attention_output_projection",
        ".self_attn": "attention_gating_and_other",
        ".mlp.gate_up_proj": "mlp_gate_up_projection",
        ".mlp.act_fn": "mlp_activation",
        ".mlp.down_proj": "mlp_down_projection",
    }
    for suffix, stage in suffixes.items():
        if name.endswith(suffix):
            return stage
    return None


def attribute_events(events):
    stages = defaultdict(lambda: {"kernel_us": 0.0, "kernels": 0, "scope_calls": 0})
    kernel_totals = defaultdict(lambda: {"kernel_us": 0.0, "calls": 0})
    linked_us = 0.0
    linked_count = 0
    for event in events:
        if str(event.device_type) != "DeviceType.CPU":
            continue
        if event.name.startswith(PREFIX):
            stages[event.name[len(PREFIX) :]]["scope_calls"] += 1
        parent = event
        seen = set()
        while parent is not None and not parent.name.startswith(PREFIX):
            if id(parent) in seen:
                raise ValueError("cycle in profiler parent links")
            seen.add(id(parent))
            parent = parent.cpu_parent
        stage = parent.name[len(PREFIX) :] if parent is not None else "unattributed"
        for kernel in event.kernels:
            duration = float(kernel.duration)
            if not 0 <= duration < float("inf"):
                raise ValueError("invalid GPU kernel duration")
            stages[stage]["kernel_us"] += duration
            stages[stage]["kernels"] += 1
            kernel_totals[(stage, kernel.name)]["kernel_us"] += duration
            kernel_totals[(stage, kernel.name)]["calls"] += 1
            linked_us += duration
            linked_count += 1
    if linked_count == 0:
        raise ValueError("profiler did not link any GPU kernels")
    return {
        "stages": dict(stages),
        "linked_kernel_us": linked_us,
        "linked_kernels": linked_count,
        "top_kernels": [
            {"stage": stage, "kernel": kernel, **value}
            for (stage, kernel), value in sorted(
                kernel_totals.items(), key=lambda item: item[1]["kernel_us"], reverse=True
            )[:40]
        ],
        "accounting": (
            "Each CPU-linked GPU kernel counted once under its innermost stage; "
            "GPU annotation spans excluded."
        ),
    }


def attribute_host_scopes(trace_events):
    """Exclusive host elapsed time from nested CPU annotation intervals.

    This includes waiting inside a scope, not just CPU execution, and must not
    be added to overlapping GPU durations. Device annotations are excluded.
    """
    threads = defaultdict(list)
    for event in trace_events:
        if (
            event.get("ph") == "X"
            and event.get("cat") == "user_annotation"
            and event.get("name", "").startswith(PREFIX)
        ):
            start, duration = float(event["ts"]), float(event["dur"])
            if not 0 <= duration < float("inf"):
                raise ValueError("invalid host duration")
            threads[(event["pid"], event["tid"])].append(
                {
                    "stage": event["name"][len(PREFIX) :],
                    "start": start,
                    "end": start + duration,
                    "duration": duration,
                    "children": 0.0,
                }
            )
    stages = defaultdict(lambda: {"host_exclusive_us": 0.0, "scope_calls": 0})
    root_us = 0.0
    for intervals in threads.values():
        stack = []
        for row in sorted(intervals, key=lambda r: (r["start"], -r["duration"])):
            while stack and row["start"] >= stack[-1]["end"]:
                stack.pop()
            if stack:
                if row["end"] > stack[-1]["end"] + 0.001:
                    raise ValueError("host annotations overlap without nesting")
                stack[-1]["children"] += row["duration"]
            else:
                root_us += row["duration"]
            stack.append(row)
        for row in intervals:
            exclusive = row["duration"] - row["children"]
            if exclusive < -0.01:
                raise ValueError("nested host intervals exceed their parent")
            stages[row["stage"]]["host_exclusive_us"] += max(0.0, exclusive)
            stages[row["stage"]]["scope_calls"] += 1
    if not stages:
        raise ValueError("no CPU stage annotations in trace")
    return {"stages": dict(stages), "root_host_us": root_us}


def attribute_trace(trace_events):
    """Attribute actual GPU dispatches through unique HIP launch correlations.

    A native ctypes HIP launch need not have an ATen parent. Its external ID
    can point at an earlier operation, so neither that ID nor the GPU execution
    timestamp determines its semantic scope. Correlation identifies the host
    launch; the innermost annotation on that thread identifies its stage.
    One runtime call may launch several kernels (for example hipMemsetAsync).
    """
    host = attribute_host_scopes(trace_events)  # Also validate scope nesting.
    stages = defaultdict(lambda: {"kernel_us": 0.0, "kernels": 0, "scope_calls": 0})
    threads = defaultdict(list)
    for event in trace_events:
        if event.get("ph") != "X":
            continue
        category = event.get("cat")
        annotation = category == "user_annotation" and event.get("name", "").startswith(PREFIX)
        runtime = category in {"cuda_runtime", "hip_runtime"}
        if not (annotation or runtime):
            continue
        start, duration = float(event["ts"]), float(event["dur"])
        if not math.isfinite(start) or not 0 <= duration < float("inf"):
            raise ValueError("invalid runtime or annotation interval")
        if annotation:
            stages[event["name"][len(PREFIX) :]]["scope_calls"] += 1
        threads[(event["pid"], event["tid"])].append(
            (start, 0 if annotation else 1, -duration, event)
        )
    launches = {}
    for entries in threads.values():
        stack = []
        for start, kind, negative_duration, event in sorted(entries, key=lambda row: row[:3]):
            while stack and start >= stack[-1][0]:
                stack.pop()
            if kind == 0:
                if negative_duration:
                    stack.append((start - negative_duration, event["name"][len(PREFIX) :]))
                continue
            correlation = event.get("args", {}).get("correlation")
            if correlation is None:
                continue
            if correlation in launches:
                raise ValueError("ambiguous runtime correlation")
            launches[correlation] = stack[-1][1] if stack else "unattributed"
    kernel_totals = defaultdict(lambda: {"kernel_us": 0.0, "calls": 0})
    total_us = 0.0
    count = 0
    unlinked = 0
    for event in trace_events:
        if event.get("ph") != "X" or event.get("cat") != "kernel":
            continue
        duration = float(event["dur"])
        if not 0 <= duration < float("inf"):
            raise ValueError("invalid GPU kernel duration")
        correlation = event.get("args", {}).get("correlation")
        stage = launches.get(correlation, "unattributed")
        unlinked += correlation not in launches
        stages[stage]["kernel_us"] += duration
        stages[stage]["kernels"] += 1
        kernel_totals[(stage, event["name"])]["kernel_us"] += duration
        kernel_totals[(stage, event["name"])]["calls"] += 1
        total_us += duration
        count += 1
    if not count:
        raise ValueError("trace contains no GPU kernels")
    return {
        "stages": dict(stages),
        "kernel_us": total_us,
        "kernels": count,
        "unlinked_kernels": unlinked,
        "host": host,
        "kernel_groups": [
            {"stage": stage, "kernel": kernel, **value}
            for (stage, kernel), value in sorted(
                kernel_totals.items(), key=lambda item: item[1]["kernel_us"], reverse=True
            )
        ],
        "accounting": (
            "Each Chrome kernel dispatch counted once, linked by unique runtime correlation "
            "to the innermost CPU stage at launch. ATen external IDs and GPU annotation "
            "spans are not used for attribution. Kernel durations can overlap in time."
        ),
    }

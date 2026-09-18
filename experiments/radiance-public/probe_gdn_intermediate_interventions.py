"""Localize residual GDN differences by progressively equalizing intermediate inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from analyze_gdn_decode_transition import array
from gdn_intermediate_interventions import STAGES, intervention_source
from gdn_intermediate_trace import ELEMENTS, OFFSETS, SHAPES
from probe_gdn_intermediates import (
    KERNEL,
    exact_comparison,
    load_module,
    replay,
    require_native_admission,
    validate_capture,
)

from qwen_r9700_lab.conformance_gpu_lease import gpu_lease
from qwen_r9700_lab.diagnostic_contract import seal, write_private


def injected_names(stage):
    result = ["query", "key"]
    result.extend(
        "output_fp32" if boundary == "output" else boundary
        for boundary in STAGES[1 : STAGES.index(stage) + 1]
    )
    return result


def run(args, control):
    import torch

    stock_call, _, _ = validate_capture(args.capture)
    inputs = {}
    for name in ("mixed_qkv", "a", "b", "A_log", "dt_bias"):
        value = array(stock_call / "before", "kwargs." + name)
        inputs[name] = torch.from_numpy(value.copy()).to(
            device="cuda", dtype=torch.float32 if name == "A_log" else torch.bfloat16
        )
    initial = array(stock_call / "before", "kwargs.initial_state.selected_values", 0).astype(
        np.float32
    )
    reference = {}
    for name in [*SHAPES, "state", "output"]:
        t = control["rows"]["r4d_trace"]["tensors"][name]
        path = args.output / "control" / t["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != t["sha256"]:
            raise ValueError("native control tensor changed")
        reference[name] = np.load(path, allow_pickle=False)
    ref_trace = torch.from_numpy(np.concatenate([reference[n].ravel() for n in SHAPES])).to("cuda")
    ref_state = torch.from_numpy(reference["state"].copy()).to("cuda")
    rows = {}
    for stage in STAGES:
        module = load_module(args.output / (stage + ".py"), "_qwen_crossfeed_" + stage)
        state = torch.full((3, 48, 128, 128), 17.0, device="cuda", dtype=torch.float32)
        state[1].copy_(torch.from_numpy(initial).to("cuda"))
        output = torch.full((1, 1, 48, 128), float("nan"), device="cuda", dtype=torch.bfloat16)
        indices = torch.ones(1, device="cuda", dtype=torch.int32)
        storage = torch.full((ELEMENTS + 64,), 17.0, device="cuda", dtype=torch.float32)
        trace = storage[32:-32]
        trace.fill_(float("nan"))
        protected = {**inputs, "reference_trace": ref_trace, "reference_state": ref_state}
        before = {k: v.clone() for k, v in protected.items()}
        getattr(module, KERNEL)[(4, 48)](
            **protected,
            trace=trace,
            o=output,
            h0=state,
            ht=state,
            ssm_state_indices=indices,
            scale=128**-0.5,
            stride_mixed_qkv_tok=inputs["mixed_qkv"].stride(0),
            stride_a_tok=inputs["a"].stride(0),
            stride_b_tok=inputs["b"].stride(0),
            stride_init_state_token=state.stride(0),
            stride_final_state_token=state.stride(0),
            stride_indices_seq=1,
            H=16,
            HV=48,
            K=128,
            V=128,
            BK=128,
            BV=32,
            SOFTPLUS_THRESHOLD=20.0,
            USE_QK_L2NORM_IN_KERNEL=True,
            SPLIT_BATCH_HEAD_GRID=False,
            num_warps=1,
            num_stages=3,
        )
        torch.cuda.synchronize()
        flat = trace.cpu().numpy()
        values = {
            n: flat[OFFSETS[n] : OFFSETS[n] + int(np.prod(shape))].reshape(shape).copy()
            for n, shape in SHAPES.items()
        }
        values.update(state=state[1].cpu().numpy(), output=output[0, 0].float().cpu().numpy())
        comparisons = {n: exact_comparison(v, reference[n]) for n, v in values.items()}
        row = {
            "equalized_through": stage,
            "guards_unchanged": bool((state[[0, 2]] == 17).all())
            and bool((indices == 1).all())
            and bool((storage[:32] == 17).all())
            and bool((storage[-32:] == 17).all()),
            "inputs_unchanged": all(
                torch.equal(before[k].view(torch.uint8), v.view(torch.uint8))
                for k, v in protected.items()
            ),
            "trace_complete": bool(np.isfinite(flat).all()),
            "injected_boundaries_equal": all(
                comparisons[n]["bit_equal"] for n in injected_names(stage)
            ),
            "comparisons": comparisons,
            "tensors": {},
        }
        folder = args.output / stage
        folder.mkdir(mode=0o700)
        for name, value in values.items():
            path = folder / (name + ".npy")
            np.save(path, value, allow_pickle=False)
            row["tensors"][name] = {
                "file": str(path.relative_to(args.output)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        rows[stage] = row
    valid = all(
        row[n]
        for row in rows.values()
        for n in (
            "guards_unchanged",
            "inputs_unchanged",
            "trace_complete",
            "injected_boundaries_equal",
        )
    )
    return seal(
        {
            "schema": "urn:qwen:gdn-intermediate-interventions:v1",
            "status": "DIAGNOSTIC_MEASURED" if valid else "INVALID_CONTROL",
            "control": control["sha256"],
            "rows": rows,
            "sources": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob("*.py")
            },
            "driver": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "generator": hashlib.sha256(
                Path(__file__).with_name("gdn_intermediate_interventions.py").read_bytes()
            ).hexdigest(),
            "scope": (
                "One captured GDN transition. Cross-fed intermediates are diagnostic "
                "interventions, not a replacement model contract or production repair."
            ),
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "build", "stock-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    require_native_admission(args)
    os.umask(0o077)
    args.output.mkdir(mode=0o700)
    original = args.stock_source.read_bytes()
    for stage in STAGES:
        (args.output / (stage + ".py")).write_bytes(intervention_source(original, stage))
    with gpu_lease(args.output / "gpu-lease"):
        control_dir = args.output / "control"
        control_dir.mkdir(mode=0o700)
        control = replay(
            SimpleNamespace(build=args.build, capture=args.capture, output=control_dir)
        )
        write_private(control_dir / "probe-result.json", control)
        if control["status"] != "DIAGNOSTIC_MEASURED":
            raise ValueError("native control replay did not reproduce the preserved transition")
        report = run(args, control)
        write_private(args.output / "probe-result.json", report)
    print(json.dumps({k: report[k] for k in ("status", "sha256")}), flush=True)
    return 0 if report["status"] == "DIAGNOSTIC_MEASURED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

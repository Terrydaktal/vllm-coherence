"""Localize a preserved GDN discrepancy with trace-neutrality controls."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from analyze_gdn_decode_transition import array
from build_gdn_causal_prefill import HEADERS
from gdn_intermediate_trace import ELEMENTS, OFFSETS, SHAPES, r4d_source, stock_source
from probe_gdn_beta_precision import (
    KERNEL,
    MODEL_REPORT,
    MODULE,
    SOURCE_SHA256,
    WRAPPER,
    ablation_source,
    exact_comparison,
    validate_capture,
)

from qwen_r9700_lab.diagnostic_contract import authenticate, private_json, seal, write_private


def prepare(args):
    """CPU-only source generation and offline gfx1201 compilation."""
    validate_capture(args.capture)
    args.output.mkdir(mode=0o700)
    stock = args.stock_source.read_bytes()
    files = {
        "stock_trace.py": stock_source(stock),
        "beta_fp32.py": ablation_source(stock),
        "beta_fp32_trace.py": stock_source(stock, beta_fp32=True),
    }
    for name, expected in HEADERS.items():
        files[name] = (args.headers / name).read_bytes()
        if hashlib.sha256(files[name]).hexdigest() != expected:
            raise ValueError("unreviewed R4D trace header")
    for traced, label in ((False, "r4d_control"), (True, "r4d_trace")):
        body = r4d_source(args.r4d_source.read_bytes(), traced=traced)
        wrapper = """
extern "C" int qwen_gdn_trace_simple(
 const void* q, const void* k, const void* v, const void* a, const void* b,
 const void* alog, const void* dt, void* state, void* out,
 const void* cu, const void* indices, const void* accepted, void* trace, void* stream) {
 return qwen_gdn_recurrent_diagnostic(q, k, v, a, b, 48, 1, alog, dt, state,
   48*128*128, 128*128, out, cu, indices, 1, accepted,
   nullptr, nullptr, 0.0f, 0, 1, 48, 16, 128, 128,
   0.08838834764831845f, 20.0f, TRACE_ARG stream);
}
""".replace("TRACE_ARG", "trace," if traced else "")
        files[label + ".hip"] = body + wrapper.encode()
    for name, value in files.items():
        (args.output / name).write_bytes(value)
    builds = []
    for label in ("r4d_control", "r4d_trace"):
        command = [
            "/opt/rocm/bin/hipcc",
            "-O3",
            "-std=c++17",
            "--offload-arch=gfx1201",
            "-shared",
            "-fPIC",
            "-ffp-contract=off",
            "-mcumode",
            str(args.output / (label + ".hip")),
            "-o",
            str(args.output / (label + ".so")),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        (args.output / (label + ".log")).write_text(result.stdout + result.stderr)
        builds.append({"label": label, "command": command, "returncode": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{label} compilation failed")
    report = seal(
        {
            "schema": "urn:qwen:gdn-intermediate-build:v1",
            "status": "BUILT_UNTESTED",
            "files": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in args.output.iterdir()
                if p.is_file()
            },
            "builds": builds,
            "capture": MODEL_REPORT,
            "gpu_used": False,
            "generator": hashlib.sha256(
                Path(__file__).with_name("gdn_intermediate_trace.py").read_bytes()
            ).hexdigest(),
            "driver": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    write_private(args.output / "build.json", report)
    return report


def load_module(path, suffix):
    spec = importlib.util.spec_from_file_location(MODULE + suffix, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def classify(rows):
    required = {
        "stock",
        "stock_repeat",
        "stock_trace",
        "beta_fp32",
        "beta_fp32_trace",
        "r4d_control",
        "r4d_trace",
    }
    if set(rows) != required:
        return "INCOMPLETE"
    if not all(r["guards_unchanged"] and r["inputs_unchanged"] for r in rows.values()):
        return "INVALID_CONTROL"
    for name in ("stock", "stock_repeat", "stock_trace", "r4d_control", "r4d_trace"):
        if not all(
            rows[name][key]["bit_equal"] for key in ("state_to_captured", "output_to_captured")
        ):
            return "INVALID_CONTROL"
    for name in ("stock_trace", "beta_fp32_trace", "r4d_trace"):
        if not all(
            rows[name][key]["bit_equal"] for key in ("state_to_untraced", "output_to_untraced")
        ):
            return "INVALID_CONTROL"
        if rows[name].get("trace_complete") is not True:
            return "INVALID_CONTROL"
    return "DIAGNOSTIC_MEASURED"


def replay(args):
    import torch

    build = private_json(args.build / "build.json")
    authenticate(build)
    for name, digest in build["files"].items():
        if hashlib.sha256((args.build / name).read_bytes()).hexdigest() != digest:
            raise ValueError("trace build changed")
    if build["driver"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("trace driver changed after build")
    stock_call, native_call, diagnosis = validate_capture(args.capture)
    stock = importlib.import_module(MODULE)
    if hashlib.sha256(Path(stock.__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("installed stock source changed")
    modules = {"stock": stock, "stock_repeat": stock}
    for name in ("stock_trace", "beta_fp32", "beta_fp32_trace"):
        modules[name] = load_module(args.build / (name + ".py"), "_qwen_" + name)
    inputs = {}
    for name in ("mixed_qkv", "a", "b", "A_log", "dt_bias"):
        value = array(stock_call / "before", "kwargs." + name)
        inputs[name] = torch.from_numpy(value.copy()).to(
            device="cuda", dtype=torch.float32 if name == "A_log" else torch.bfloat16
        )
    initial = array(stock_call / "before", "kwargs.initial_state.selected_values", 0).astype(
        np.float32
    )
    stock_state = array(stock_call / "after", "kwargs.initial_state.selected_values", 0).astype(
        np.float32
    )
    stock_output = array(stock_call / "after", "kwargs.out")[0, 0].astype(np.float32)
    mapping = diagnosis["selected_storage_metadata"]
    slot = mapping["d7_indices"].index(mapping["d7_mapping"][0][0])
    native_state = array(native_call / "after", "args.7.selected_values", slot).astype(np.float32)
    native_output = array(native_call / "after", "args.8")[0].astype(np.float32)
    packed = inputs["mixed_qkv"]
    other = {
        "q": packed[:, :2048].reshape(1, 16, 128).contiguous(),
        "k": packed[:, 2048:4096].reshape(1, 16, 128).contiguous(),
        "v": packed[:, 4096:].reshape(1, 48, 128).contiguous(),
        "dt": inputs["dt_bias"].float(),
        "cu": torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        "accepted": torch.ones(1, device="cuda", dtype=torch.int32),
    }
    rows, values = {}, {}
    for name in (
        "stock",
        "stock_repeat",
        "stock_trace",
        "beta_fp32",
        "beta_fp32_trace",
        "r4d_control",
        "r4d_trace",
    ):
        state = torch.full((3, 48, 128, 128), 17.0, device="cuda", dtype=torch.float32)
        state[1].copy_(torch.from_numpy(initial).to("cuda"))
        output = torch.full((1, 1, 48, 128), float("nan"), device="cuda", dtype=torch.bfloat16)
        indices = torch.ones(1, device="cuda", dtype=torch.int32)
        trace_storage = torch.full((ELEMENTS + 64,), 17.0, device="cuda", dtype=torch.float32)
        trace = trace_storage[32:-32]
        trace.fill_(float("nan"))
        before = {k: v.clone() for k, v in {**inputs, **other}.items()}
        if name.startswith("r4d"):
            library = ctypes.CDLL(str(args.build / (name + ".so")))
            function = library.qwen_gdn_trace_simple
            function.argtypes = [ctypes.c_void_p] * 14
            function.restype = ctypes.c_int
            tensors = [other[k] for k in ("q", "k", "v")] + [inputs[k] for k in ("a", "b", "A_log")]
            tensors += [other["dt"], state, output, other["cu"], indices, other["accepted"], trace]
            result = function(
                *(ctypes.c_void_p(t.data_ptr()) for t in tensors),
                ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
            )
            if result:
                raise RuntimeError(f"R4D trace launch failed: {result}")
        elif name.endswith("_trace"):
            kernel = getattr(modules[name], KERNEL)
            kernel[(4, 48)](
                trace=trace,
                **inputs,
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
        else:
            getattr(modules[name], WRAPPER)(
                **inputs,
                scale=128**-0.5,
                initial_state=state,
                out=output,
                ssm_state_indices=indices,
                use_qk_l2norm_in_kernel=True,
            )
        torch.cuda.synchronize()
        s, o = state[1].cpu().numpy(), output[0, 0].float().cpu().numpy()
        target_s, target_o = (
            (native_state, native_output) if name.startswith("r4d") else (stock_state, stock_output)
        )
        row = {
            "guards_unchanged": bool((state[[0, 2]] == 17).all())
            and bool((indices == 1).all())
            and bool((trace_storage[:32] == 17).all())
            and bool((trace_storage[-32:] == 17).all()),
            "inputs_unchanged": all(
                torch.equal(before[k].view(torch.uint8), v.view(torch.uint8))
                for k, v in {**inputs, **other}.items()
            ),
            "state_to_captured": exact_comparison(s, target_s),
            "output_to_captured": exact_comparison(o, target_o),
        }
        value = {"state": s, "output": o}
        if name.endswith("_trace"):
            data = trace.cpu().numpy()
            row["trace_complete"] = bool(np.isfinite(data).all())
            value.update(
                {
                    k: data[OFFSETS[k] : OFFSETS[k] + int(np.prod(shape))].reshape(shape).copy()
                    for k, shape in SHAPES.items()
                }
            )
            untraced = "r4d_control" if name == "r4d_trace" else name.removesuffix("_trace")
            row["state_to_untraced"] = exact_comparison(s, values[untraced]["state"])
            row["output_to_untraced"] = exact_comparison(o, values[untraced]["output"])
        folder = args.output / name
        folder.mkdir(mode=0o700)
        row["tensors"] = {}
        for key, data in value.items():
            path = folder / (key + ".npy")
            np.save(path, data, allow_pickle=False)
            row["tensors"][key] = {
                "file": str(path.relative_to(args.output)),
                "shape": list(data.shape),
                "dtype": data.dtype.str,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        rows[name], values[name] = row, value
    status = classify(rows)
    comparisons = {}
    if status == "DIAGNOSTIC_MEASURED":
        comparisons = {
            name: {
                key: exact_comparison(values[name][key], values["r4d_trace"][key])
                for key in [*SHAPES, "state", "output"]
            }
            for name in ("stock_trace", "beta_fp32_trace")
        }
    return seal(
        {
            "schema": "urn:qwen:gdn-intermediate-replay:v1",
            "status": status,
            "build": build["sha256"],
            "capture": MODEL_REPORT,
            "rows": rows,
            "comparisons": comparisons,
            "installed_sources_changed": False,
            "scope": (
                "One captured first GDN transition, with exact trace-neutrality controls; "
                "not full-model qualification."
            ),
        }
    )


def require_native_admission(args):
    if not args.allow_gpu or not args.build:
        raise ValueError("native trace replay requires explicit admission and a frozen build")
    if not os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"):
        raise ValueError("native trace replay requires the campaign GPU lease")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "replay"))
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--stock-source", type=Path)
    parser.add_argument("--r4d-source", type=Path)
    parser.add_argument("--headers", type=Path)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    if args.mode == "prepare":
        result = prepare(args)
    else:
        require_native_admission(args)
        from qwen_r9700_lab.conformance_gpu_lease import gpu_lease

        args.output.mkdir(mode=0o700)
        with gpu_lease(args.output / "gpu-lease"):
            result = replay(args)
        write_private(args.output / "probe-result.json", result)
    print(json.dumps({k: result[k] for k in ("status", "sha256")}), flush=True)
    return 0 if result["status"] in {"BUILT_UNTESTED", "DIAGNOSTIC_MEASURED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

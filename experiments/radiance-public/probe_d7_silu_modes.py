"""Replay actual compiled/eager SiLU kernels on identical private captured inputs.

This diagnoses operator arithmetic. It does not claim isolated vocabulary top-k,
recurrent-state equivalence, release speed, or complete eager/compiled equality.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

from compare_mode_boundaries_d7 import load

from qwen_r9700_lab.conformance_execution_modes import admit_pair
from qwen_r9700_lab.conformance_gpu_lease import gpu_lease
from qwen_r9700_lab.conformance_mode_boundaries import admit_bridge, silu_cut
from qwen_r9700_lab.conformance_queue import replace_private
from qwen_r9700_lab.diagnostic_contract import (
    DiagnosticError,
    authenticate,
    private_json,
    seal,
    write_private,
)


def require(condition, message):
    if not condition:
        raise DiagnosticError(message)


def records(root):
    return {
        "measurement": private_json(root / "measurement.json"),
        "config": private_json(root / "fixed-bf16/requested-config.json"),
        "runtime": private_json(root / "fixed-bf16/actual-runtime.json"),
        "pass": private_json(root / "fixed-bf16/pass-00.json"),
    }


def compare(left, right):
    import torch

    a, b = left.detach().cpu(), right.detach().cpu()
    require(
        a.shape == b.shape and a.dtype == b.dtype == torch.bfloat16,
        "SiLU output shape/dtype changed",
    )
    same = a.view(torch.int16) == b.view(torch.int16)
    return {
        "positions": len(a),
        "exact_positions": int(same.all(dim=1).sum()),
        "elements": same.numel(),
        "different_elements": int((~same).sum()),
        "max_abs": float((a.float() - b.float()).abs().max()),
    }


def add(total, value):
    for key, number in value.items():
        total[key] = (
            max(total.get(key, 0), number) if key == "max_abs" else total.get(key, 0) + number
        )


def run(args):
    import torch
    importlib.import_module("vllm._custom_ops")

    torch.set_num_threads(1)
    runs = [records(root) for root in (args.compiled_run, args.eager_run)]
    admission = admit_pair(*runs)
    require(
        admission["captures"] == [True, True]
        and admission["modes"] == ["compiled-no-graphs", "eager"],
        "wrong capture modes",
    )
    roots = [args.compiled_capture, args.eager_capture]
    bridges = [private_json(p) for p in (args.compiled_bridge, args.eager_bridge)]
    manifests = [private_json(root / "manifest.json") for root in roots]
    prefills = [private_json(root / "prefill-manifest.json") for root in roots]
    for run_record, bridge, manifest, prefill in zip(
        runs, bridges, manifests, prefills, strict=True
    ):
        admit_bridge(bridge, run_record["pass"], manifest, prefill)

    index = private_json(args.source_index)
    authenticate(index)
    kernels = {}
    for operation, entry in index["kernels"].items():
        path = Path(entry["path"])
        require(
            hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"],
            "compiled SiLU source changed",
        )
        name = operation.removeprefix("inductor/")
        spec = importlib.util.spec_from_file_location("isolated_" + name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        kernels[operation] = getattr(module, name)

    totals = {}
    sources = {
        "admission": admission["sha256"],
        "bridges": [b["sha256"] for b in bridges],
        "generated_sources": index["sha256"],
        "probe": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    for phase, documents in (("prefill", prefills), ("decode", manifests)):
        require(
            len(documents[0]["batches"]) == len(documents[1]["batches"]),
            "capture group count differs",
        )
        observed = []
        for ia, ib in zip(documents[0]["batches"], documents[1]["batches"], strict=True):
            ma, ta = load(roots[0], ia)
            mb, tb = load(roots[1], ib)
            positions = ma["positions"]
            require(positions == mb["positions"], "capture positions differ")
            observed += positions
            for layer in range(64):
                ca, cb = silu_cut(ma, ta, layer), silu_cut(mb, tb, layer)
                require(ca["operation"] in kernels, "unbound compiled SiLU implementation")

                def tensor(values, key):
                    return torch.from_numpy(values[key]).view(torch.bfloat16).clone()

                a, b = tensor(ta, ca["input"]), tensor(tb, cb["input"])
                expected_a, expected_b = tensor(ta, ca["output"]), tensor(tb, cb["output"])
                x, own_eager_input = a.cuda(), b.cuda()
                output = torch.empty((len(a), 17408), dtype=torch.bfloat16, device="cuda")
                kernels[ca["operation"]].run(
                    x, output, output.numel(), stream=torch.cuda.current_stream().cuda_stream
                )
                eager_own, eager_common = torch.empty_like(output), torch.empty_like(output)
                torch.ops._C.silu_and_mul(eager_own, own_eager_input)
                torch.ops._C.silu_and_mul(eager_common, x)
                torch_native = torch.nn.functional.silu(x[:, :17408]) * x[:, 17408:]
                torch_fp32 = (
                    torch.nn.functional.silu(x[:, :17408].float()) * x[:, 17408:].float()
                ).bfloat16()
                require(
                    compare(output, expected_a)["exact_positions"] == len(a),
                    "compiled replay differs from its captured output",
                )
                require(
                    compare(eager_own, expected_b)["exact_positions"] == len(b),
                    "eager replay differs from its captured output",
                )
                require(
                    compare(x, a)["exact_positions"] == len(a)
                    and compare(own_eager_input, b)["exact_positions"] == len(b),
                    "SiLU changed an input",
                )
                entry = totals.setdefault(phase, {}).setdefault(str(layer), {})
                for name, result in {
                    "captured_input_agreement": compare(a, b),
                    "compiled_vs_eager_on_common_input": compare(output, eager_common),
                    "torch_native_vs_eager": compare(torch_native, eager_common),
                    "torch_fp32_vs_compiled": compare(torch_fp32, output),
                }.items():
                    add(entry.setdefault(name, {}), result)
                entry["own_capture_reproduced"] = entry.get("own_capture_reproduced", 0) + len(a)
            replace_private(
                args.output,
                "progress.json",
                seal(
                    {
                        "status": "INCOMPLETE",
                        "phase": phase,
                        "positions": len(observed),
                        "sources": sources,
                    }
                ),
            )
        expected_positions = (
            list(range(60000, 60320)) if phase == "decode" else documents[0]["positions"]
        )
        require(observed == expected_positions, "incomplete or duplicate position coverage")
    fault = output.clone()
    fault.view(torch.int16)[0, 0] ^= 1
    require(compare(output, fault)["different_elements"] == 1, "one-bit negative control missed")
    return seal(
        {
            "schema": "qwen.silu-mode-isolation.v1",
            "status": "SAMPLE_CHECKED",
            "sources": sources,
            "results": totals,
            "negative_control_detected": True,
            "inputs_unchanged": True,
            "scope": (
                "SiLU only, every layer on 320 decode and 9 sampled prefill positions; "
                "actual kernels on common inputs. No vocabulary top-k or full-model equality claim."
            ),
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "compiled-run",
        "eager-run",
        "compiled-capture",
        "eager-capture",
        "compiled-bridge",
        "eager-bridge",
        "source-index",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    require(args.allow_gpu and os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"), "GPU lease required")
    os.umask(0o077)
    args.output.mkdir(mode=0o700)
    with gpu_lease(args.output / "gpu-lease"):
        result = run(args)
    write_private(args.output / "result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "sha256": result["sha256"],
                "first_layer": {phase: values["0"] for phase, values in result["results"].items()},
            }
        )
    )


if __name__ == "__main__":
    main()

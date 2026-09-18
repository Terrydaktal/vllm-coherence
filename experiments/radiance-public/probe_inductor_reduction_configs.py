"""Replay one captured Inductor reduction with its two observed launch configurations.

Synthetic inputs only. No GPU imports occur before the explicit execution gate.
This checks configuration-dependent output, not which arithmetic is correct.
Run through the qualification process supervisor and GPU lease.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import os
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import seal, write_private

KERNEL = "triton_red_fused__to_copy_add_fused_add_rms_norm_mxfp4_linear_2"


def extract_kernel(source, expected_sha256):
    if hashlib.sha256(source.encode()).hexdigest() != expected_sha256:
        raise ValueError("captured kernel source identity changed")
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != KERNEL:
        raise ValueError("unexpected captured kernel")
    function = functions[0]
    expected_args = [
        "in_ptr0",
        "in_ptr1",
        "in_ptr2",
        "out_ptr1",
        "xnumel",
        "r0_numel",
        "XBLOCK",
        "R0_BLOCK",
    ]
    if [arg.arg for arg in function.args.args] != expected_args:
        raise ValueError("captured kernel signature changed")
    original_body = ast.dump(ast.Module(body=function.body, type_ignores=[]))
    # Drop only the autotuner decorator. Preserve the actual function body and
    # pass each saved launch configuration explicitly to Triton's JIT.
    function.decorator_list = [ast.parse("triton.jit").body[0].value]
    text = "import triton\nimport triton.language as tl\n\n" + ast.unparse(function) + "\n"
    replay = next(node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef))
    if ast.dump(ast.Module(body=replay.body, type_ignores=[])) != original_body:
        raise ValueError("kernel extraction changed the operation")
    return text, hashlib.sha256(original_body.encode()).hexdigest()


def probe(root, kernel_path):
    import torch
    from safetensors.torch import save_file

    torch.set_num_threads(1)
    write_private(
        root / "runtime.json",
        seal(
            {
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "device": torch.cuda.get_device_name(),
                "dtype": "bfloat16",
                "configs": [
                    {"R0_BLOCK": 1024, "num_warps": 8},
                    {"R0_BLOCK": 8192, "num_warps": 16},
                ],
                "common": {
                    "XBLOCK": 1,
                    "num_stages": 1,
                    "enable_fp_fusion": True,
                    "allow_flush_denorm": False,
                    "sanitize_overflow": False,
                },
            }
        ),
    )
    module_spec = importlib.util.spec_from_file_location("captured_norm_kernel", kernel_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    kernel = getattr(module, KERNEL)
    results = []
    for seed in range(3):
        generator = torch.Generator(device="cpu").manual_seed(19000 + seed)
        for rows in (1, 8, 64, 256):
            host = (
                torch.randn(rows, 5120, generator=generator).to(torch.bfloat16),
                (torch.randn(rows, 5120, generator=generator) * 0.3).to(torch.bfloat16),
                (torch.randn(5120, generator=generator) * 0.1).to(torch.bfloat16),
            )
            inputs = tuple(value.to("cuda") for value in host)
            outputs, guards = [], []
            for block, warps in ((1024, 8), (8192, 16), (1024, 8)):
                storage = torch.full((rows * 5120 + 128,), 123, dtype=torch.bfloat16, device="cuda")
                kernel[(rows,)](
                    *inputs,
                    storage,
                    rows,
                    5120,
                    XBLOCK=1,
                    R0_BLOCK=block,
                    num_warps=warps,
                    num_stages=1,
                    enable_fp_fusion=True,
                    allow_flush_denorm=False,
                    sanitize_overflow=False,
                )
                torch.cuda.synchronize()
                guards.append(bool((storage[-128:] == 123).all()))
                outputs.append(storage[: rows * 5120].reshape(rows, 5120).cpu())
            row = {
                "seed": seed,
                "rows": rows,
                "elements": rows * 5120,
                "guards_intact": all(guards),
                "inputs_unchanged": all(
                    torch.equal(gpu.cpu(), cpu) for gpu, cpu in zip(inputs, host, strict=True)
                ),
                "finite": all(bool(torch.isfinite(value).all()) for value in outputs),
                "repeat_exact": torch.equal(outputs[0], outputs[2]),
                "configurations_exact": torch.equal(outputs[0], outputs[1]),
                "different_elements": int((outputs[0] != outputs[1]).sum()),
                "max_abs_difference": float((outputs[0].float() - outputs[1].float()).abs().max()),
            }
            if row["different_elements"]:
                filename = f"counterexample-seed{seed}-rows{rows}.safetensors"
                save_file(
                    {
                        **{f"input{i}": value for i, value in enumerate(host)},
                        **{f"output{i}": value for i, value in enumerate(outputs)},
                    },
                    str(root / filename),
                )
                row["counterexample"] = filename
            results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    if not args.allow_gpu or os.environ.get("QWEN_CONFORMANCE_GPU") != "1":
        parser.error("requires explicit isolated GPU qualification authorization")
    os.umask(0o077)
    source, body_sha = extract_kernel(args.source.read_text(), args.source_sha256)
    args.output.mkdir(mode=0o700)
    path = args.output / "replay_kernel.py"
    path.write_text(source)
    write_private(
        args.output / "source-binding.json",
        seal(
            {
                "captured_source_sha256": args.source_sha256,
                "function_body_ast_sha256": body_sha,
                "replay_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "scope": "Same function body, explicit saved launch configurations; no autotuning.",
            }
        ),
    )
    rows = probe(args.output, path)
    valid = all(
        all(row[key] for key in ("guards_intact", "inputs_unchanged", "finite", "repeat_exact"))
        for row in rows
    )
    different = any(row["different_elements"] for row in rows)
    report = seal(
        {
            "schema": "urn:qwen:inductor-reduction-config-replay:v1",
            "status": "INVALID_CONTROL"
            if not valid
            else "CONFIGURATION_DIFFERENCE"
            if different
            else "TESTED",
            "gpu_executed": True,
            "proof": "UNPROVED",
            "scope": "One reduction on synthetic inputs; model causality unproved.",
            "rows": rows,
        }
    )
    write_private(args.output / "result.json", report)
    return 2 if not valid else 1 if different else 0


if __name__ == "__main__":
    raise SystemExit(main())

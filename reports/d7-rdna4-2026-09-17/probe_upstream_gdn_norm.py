"""Replay current vLLM's actual norm kernel and tile policy without a model.

This checks the extracted kernel/dispatch policy, not a full current-vLLM build.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_text()
    tree = ast.parse(source)
    nodes = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    lines = source.splitlines(keepends=True)
    selected = []
    for name in ("layer_norm_fwd_kernel", "calc_rows_per_block"):
        node = nodes[name]
        begin = min([node.lineno, *[x.lineno for x in node.decorator_list]])
        selected.append("".join(lines[begin - 1 : node.end_lineno]))
    import torch

    header = (
        "from vllm import envs\nimport torch\n"
        "from vllm.triton_utils import triton, tl\n"
        "from vllm.utils.math_utils import cdiv, next_power_of_2\n"
        "from vllm.utils.platform_utils import num_compute_units\n"
    )
    counts = {
        "cases": 0,
        "original_output_differences": 0,
        "original_rstd_differences": 0,
        "fixed_output_differences": 0,
        "fixed_rstd_differences": 0,
    }
    with tempfile.TemporaryDirectory(prefix="gdn-norm-source-") as tmp:
        path = Path(tmp) / "extracted.py"
        path.write_text(header + "\n\n".join(selected))
        spec = importlib.util.spec_from_file_location("gdn_norm_extracted", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        def run(x, z, w, invariant):
            os.environ["VLLM_BATCH_INVARIANT"] = "1" if invariant else "0"
            rows = module.calc_rows_per_block(x.shape[0], x.device)
            y = torch.empty_like(x)
            rstd = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
            module.layer_norm_fwd_kernel[(module.cdiv(x.shape[0], rows), 1)](
                x,
                y,
                w,
                None,
                z,
                None,
                rstd,
                x.stride(0),
                y.stride(0),
                z.stride(0),
                x.shape[0],
                x.shape[1],
                1e-6,
                BLOCK_N=128,
                ROWS_PER_BLOCK=rows,
                NORM_BEFORE_GATE=True,
                IS_RMS_NORM=True,
                ACTIVATION="silu",
                num_warps=1,
            )
            return y, rstd

        for dtype in (torch.bfloat16, torch.float32):
            for seed in range(16):
                torch.manual_seed(seed)
                x = torch.randn(384, 128, device="cuda", dtype=dtype)
                z, w = torch.randn_like(x), torch.randn(128, device="cuda", dtype=dtype)
                original, old_rstd = run(x, z, w, False)
                fixed, fixed_rstd = run(x, z, w, True)
                for row in range(0, 384, 48):
                    serial, serial_rstd = run(x[row : row + 48], z[row : row + 48], w, True)
                    counts["original_output_differences"] += int(
                        (original[row : row + 48] != serial).sum()
                    )
                    counts["original_rstd_differences"] += int(
                        (old_rstd[row : row + 48] != serial_rstd).sum()
                    )
                    counts["fixed_output_differences"] += int(
                        (fixed[row : row + 48] != serial).sum()
                    )
                    counts["fixed_rstd_differences"] += int(
                        (fixed_rstd[row : row + 48] != serial_rstd).sum()
                    )
                counts["cases"] += 1
        torch.cuda.synchronize()
    result = {
        "scope": "current upstream kernel and row policy; full upstream integration untested",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "gpu": torch.cuda.get_device_name(),
        "counts": counts,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    assert counts["fixed_output_differences"] == counts["fixed_rstd_differences"] == 0


if __name__ == "__main__":
    main()

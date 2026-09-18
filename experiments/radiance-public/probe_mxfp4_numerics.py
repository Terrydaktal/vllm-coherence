"""Compare installed MXFP4 kernels with FP32 math on actual checkpoint weights.

No conversation is loaded. Random BF16 activations exercise decode, its boundary,
and prefill shapes. Guard regions check the native split-K scratch allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def run(root: Path):
    profile = json.loads((root / "production-profile.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    os.environ.update(profile["kernel_environment"])
    linear_reference = manifest.get("reference_linear", False)
    if linear_reference:
        if manifest.get("reference_linear_fast"):
            from patch_reference_linear_fast_experiment import install
        else:
            from patch_reference_linear_experiment import install
        install(Path("/opt/vllm/lib/python3.12/site-packages"))
        os.environ["RADIANCE_MXFP4_REFLINEAR"] = "1"
    import torch
    from safetensors import safe_open
    import radiance_mxfp4 as kernel
    from vllm import _custom_ops as ops

    assert kernel.WPERM and kernel.ENABLED and kernel.R4D_DECODE_MAX_M == 0
    extension_hash = hashlib.sha256(Path(kernel._ext.__file__).read_bytes()).hexdigest()
    assert extension_hash == profile["kernel_hashes"]["radiance_mxfp4_fp8.so"]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(1789372063)
    model = Path(manifest.get("checkpoint", "/models/Qwen3.8-27B-Uncensored-MXFP4-awq"))
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.language_model.layers."

    def tensor(name):
        with safe_open(model / index[name], framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    def weights(parts):
        packed = torch.cat([tensor(prefix + part + ".weight") for part in parts])
        scales = torch.cat([tensor(prefix + part + ".weight_scale") for part in parts])
        assert packed.dtype == torch.uint8 and scales.dtype == torch.uint8
        sha = hashlib.sha256(packed.numpy().tobytes() + scales.numpy().tobytes()).hexdigest()
        n, half_k = packed.shape
        raw = packed.cuda()
        packed = kernel.permute_w(raw, n, half_k * 2)
        scales = scales.T.contiguous().cuda()
        return raw, packed, scales, kernel.make_row_ref(scales), sha

    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6,
                           0, -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")

    def reference(xq, xs, raw, scales):
        # Read the checkpoint order directly. Do not depend on the native
        # dispatcher's inverse permutation to define what correct weights mean.
        x = xq.float() * xs[:, None]
        parts = []
        for start in range(0, raw.shape[0], 2048):
            w = raw[start:start + 2048]
            codes = torch.stack((w & 15, w >> 4), dim=-1).reshape(w.shape[0], -1)
            scale = torch.exp2(scales[:, start:start + 2048].T.float() - 127).repeat_interleave(32, dim=1)
            parts.append(x @ (levels[codes.long()] * scale).T)
        return torch.cat(parts, dim=1)

    groups = {
        "merged_gdn": ["0.linear_attn.in_proj_" + part for part in ("qkv", "z", "b", "a")],
        "gdn_gates": ["0.linear_attn.in_proj_" + part for part in ("b", "a")],
        "attention_qkv": ["3.self_attn." + part + "_proj" for part in ("q", "k", "v")],
        "attention_output": ["3.self_attn.o_proj"],
        "mlp_gate_up": ["0.mlp.gate_proj", "0.mlp.up_proj"],
        "mlp_down": ["0.mlp.down_proj"],
    }
    # Match production's capacity; canaries surround both native workspaces.
    guard = 512
    sentinel = 12345
    scratch_count = 4 * 64 * 32768
    partial = torch.full((scratch_count + 2 * guard,), sentinel, dtype=torch.float32, device="cuda")
    counters = torch.full((32768 // 128 + 8 + 2 * guard,), sentinel, dtype=torch.int32, device="cuda")
    counters[guard:-guard].zero_()
    kernel._ext.set_decode_scratch(partial[guard:].data_ptr(), scratch_count * 4,
                                   counters[guard:].data_ptr())
    rows = []
    started = time.monotonic()
    for group, parts in groups.items():
        raw, packed, scales, ref, sha = weights(parts)
        n, half_k = packed.shape
        k = half_k * 2
        for m in (1, 2, 7, 8, 9, 63, 64, 65, 127, 128, 129, 2047, 2048, 2049):
            x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
            xq, xs = ops.scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=True)
            xs = xs.view(-1).float().contiguous()
            slab = torch.full((m * n + 2 * guard,), 42, device="cuda", dtype=torch.bfloat16)
            out = slab[guard:-guard].view(m, n)
            if linear_reference:
                out.copy_(kernel.mxfp4_linear(x, packed, scales, ref))
                expected = reference(x, torch.ones(m, device="cuda"), raw, scales)
            else:
                kernel._ext.launch(xq.data_ptr(), packed.data_ptr(), scales.data_ptr(), ref.data_ptr(),
                                   xs.data_ptr(), out.data_ptr(), m, n, k,
                                   torch.cuda.current_stream().cuda_stream)
                expected = reference(xq, xs, raw, scales)
            relative = float((out.float() - expected).norm() / expected.norm())
            intact = all(bool(torch.all(t[:guard] == value) & torch.all(t[-guard:] == value))
                         for t, value in ((partial, sentinel), (counters, sentinel), (slab, 42)))
            row = {"group": group, "M": m, "N": n, "K": k, "weight_sha256": sha,
                   "relative_error_vs_fp32": relative, "finite": bool(torch.isfinite(out).all()),
                   "guard_regions_intact": intact, "split_counters_settled": bool((counters[guard:-guard] == 0).all())}
            rows.append(row)
            (root / "mxfp4-probe.json").write_text(json.dumps({"rows": rows}, indent=2))
            print(json.dumps(row), flush=True)
            assert row["finite"] and relative < 0.02 and intact and row["split_counters_settled"]
            del expected, out, slab, x, xq, xs
        del raw, packed, scales, ref
    report = {"rows": rows, "elapsed_seconds": time.monotonic() - started,
              "activation_reference": "BF16" if linear_reference else "FP8",
              "source_sha256": hashlib.sha256(Path(kernel.__file__).read_bytes()).hexdigest(),
              "extension_sha256": extension_hash,
              "all_passed": True}
    (root / "mxfp4-probe.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    run(args.root)

"""Source-pinned CPU counterexamples for remaining eager-M1 contracts.

Executes the actual packed-decode admission function with a launch recorder,
checks its state address calculation, and reproduces attention's explicit
FP16 conversions on exactly representable synthetic inputs. No GPU, model,
session, or private prompt is read. These checks do not execute GPU kernels.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PINNED = {
    "r4d_attn_decode_h256_gqa6.hip": "c548b0d9d6d68fabc93fef2245fa2f7a48665ec9ee2e771f82f3ec7587da958c",
    "r4d_attn_paged_h256_gqa6.hip": "b84686cb6f5b0371f3219aa795472120369b7c035a64af55579d1366c694d691",
    "r4d_dt16.h": "b61525f1dbb6b272642e0e1410a3d753d1136ed87eb07cefca512843096c4c24",
    "fixed_fused_recurrent.py": "cf367195e14880a17d3a06e0b34cfb9f67a96c46a08438b9e91e1459029473b4",
}


def load_attention_probe():
    path = Path(__file__).with_name("probe_m1_attention_precision.py")
    spec = importlib.util.spec_from_file_location("m1_attention_precision", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def half_rtz(values):
    """IEEE binary16 round toward zero, including gradual underflow/saturation."""
    source = np.asarray(values, dtype=np.float32)
    if not np.isfinite(source).all():
        raise ValueError("finite values required")
    with np.errstate(over="ignore", under="ignore"):
        nearest = source.astype(np.float16)
        lower = np.nextafter(nearest, np.zeros_like(nearest))
        result = np.where(
            np.abs(nearest.astype(np.float32)) > np.abs(source), lower, nearest
        )
    return result.astype(np.float32)


def decode_splits(length):
    tiles = (length + 15) // 16
    splits = 128
    # Production geometry: four KV heads, one sequence.
    while splits > 16 and splits > 192 // 4:
        splits >>= 1
    while splits > 16 and splits > tiles // 2:
        splits >>= 1
    return max(1, min(splits, tiles))


def attention_witnesses():
    probe = load_attention_probe()
    rows = []
    for name, dtype, query, _keys, values, expected in probe.fixtures():
        length = len(values)
        expected_bf16 = float(probe.from_bf16(probe.bf16_bits(expected)))
        if name.startswith("uniform_mean_"):
            splits = decode_splits(length)
            tiles_per_split = ((length + 15) // 16 + splits - 1) // splits
            totals = []
            parts = []
            for start in range(0, length, tiles_per_split * 16):
                stop = min(length, start + tiles_per_split * 16)
                # Q=K=0: every exponential is exactly one. No approximation in
                # softmax or the accumulation is needed to expose this loss.
                total = np.float32(values[start:stop, 0].sum(dtype=np.float32))
                inv = np.float32(1.0) / np.float32(stop - start)
                partial = total * inv
                totals.append(stop - start)
                parts.append(float(half_rtz(partial)))
            weighted = sum(n * part for n, part in zip(totals, parts, strict=True))
            candidate = weighted / length
            boundary = "normalized split output -> FP16 RTZ -> BF16 final output"
            detail = {
                "splits": splits,
                "split_lengths": sorted(set(totals)),
                "stored_partial_values": sorted(set(parts)),
            }
        elif name == "folded_query_underflow":
            mul = np.float32(np.float32(1 / 16) * np.float32(1.44269504089))
            narrowed = half_rtz(query * mul)
            if np.count_nonzero(narrowed):
                raise AssertionError(
                    "query witness no longer disappears at FP16 conversion"
                )
            candidate = 0.0
            boundary = "scaled BF16 query -> FP16 RTZ before QK dot product"
            detail = {"scaled_query_before_fp16": float((query * mul).flat[0])}
        else:
            candidate = float(half_rtz(values[0, 0]))
            boundary = "BF16 value -> FP16 before attention accumulation"
            detail = {"live_profile_uses_this_cache_dtype": False}
        actual_bf16 = float(probe.from_bf16(probe.bf16_bits(candidate)))
        rows.append(
            {
                "case": name,
                "cache_dtype": dtype,
                "context_tokens": length,
                "boundary": boundary,
                "expected_bf16": expected_bf16,
                "source_arithmetic_bf16": actual_bf16,
                "equal": actual_bf16 == expected_bf16,
                "evidence": "CPU source-arithmetic reproduction; GPU execution pending",
                **detail,
            }
        )
    return rows


class LaunchRecorder:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(**kwargs):
            self.calls.append((grid, kwargs))

        return launch


def packed_admission(source):
    import torch

    name = "fused_recurrent_gated_delta_rule_packed_decode"
    functions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(functions) != 1:
        raise ValueError("packed-decode function missing or ambiguous")
    recorder = LaunchRecorder()
    namespace = {
        "torch": torch,
        "triton": SimpleNamespace(
            cdiv=lambda a, b: (a + b - 1) // b,
            next_power_of_2=lambda x: 1 << (x - 1).bit_length(),
        ),
        "fused_recurrent_gated_delta_rule_packed_decode_kernel": recorder,
    }
    # This is the explicitly supplied, source-pinned installed wrapper, with
    # its device launch replaced by the recorder above. Do not import vLLM.
    exec(  # noqa: S102 - execute only the explicitly selected installed wrapper
        compile(
            ast.Module(body=functions, type_ignores=[]),
            "<pinned-packed-decode>",
            "exec",
        ),
        namespace,
    )
    return namespace[name], recorder


def state_layout_witnesses(source):
    import torch

    call, recorder = packed_admission(source)
    rows = []
    for padded in (False, True):
        # Rows excluded by the view contain a distinct canary. Only slot 1 is
        # passed to the kernel; slot 0 is the null/padding slot.
        storage = torch.full((2, 48, 256 if padded else 128, 128), 19.0)
        state = storage[:, :, ::2, :] if padded else storage
        state.fill_(7.0)
        try:
            call(
                torch.zeros((1, 10240), dtype=torch.bfloat16),
                torch.zeros((1, 48), dtype=torch.bfloat16),
                torch.zeros((1, 48), dtype=torch.bfloat16),
                torch.zeros(48),
                torch.zeros(48, dtype=torch.bfloat16),
                128**-0.5,
                state,
                torch.zeros((1, 1, 48, 128), dtype=torch.bfloat16),
                torch.tensor([1], dtype=torch.int32),
                True,
            )
            accepted = True
        except ValueError:
            accepted = False
        row = {
            "layout": "padded_inner_rows" if padded else "packed_control",
            "shape": list(state.shape),
            "strides": list(state.stride()),
            "accepted_by_actual_wrapper": accepted,
            "native_launch_executed": False,
        }
        if accepted:
            _, kw = recorder.calls[-1]
            start = kw["stride_init_state_token"]
            count = kw["HV"] * kw["V"] * kw["K"]
            # Kernel addresses: state_idx * stride_token + hv*V*K + v*K + k.
            addressed = storage.flatten()[start : start + count]
            logical = state[1].flatten()
            row.update(
                {
                    "elements": count,
                    "wrongly_addressed_values": int((addressed != logical).sum()),
                    "loaded_values": sorted(torch.unique(addressed).tolist()),
                    "expected_values": sorted(torch.unique(logical).tolist()),
                }
            )
        rows.append(row)
    return rows


def format_witness():
    import torch

    raw = torch.tensor([0x38, 0x40, 0x80], dtype=torch.uint8)
    return {
        "bytes": raw.tolist(),
        "ocp_e4m3": [
            float(x) if math.isfinite(float(x)) else "NaN"
            for x in raw.view(torch.float8_e4m3fn).float()
        ],
        "fnuz_e4m3": [
            float(x) if math.isfinite(float(x)) else "NaN"
            for x in raw.view(torch.float8_e4m3fnuz).float()
        ],
        "scope": "format distinction only; native selector admission recorded separately",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-source", type=Path, required=True)
    parser.add_argument("--packed-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        name: args.attention_source / name
        for name in PINNED
        if name.endswith((".hip", ".h"))
    }
    paths["fixed_fused_recurrent.py"] = args.packed_source
    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }
    if hashes != PINNED:
        raise ValueError("source identity changed; audit encoding must be reviewed")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "eager-m1-followup-cpu-v1",
        "complete": True,
        "source_sha256": hashes,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "attention_probe_sha256": hashlib.sha256(
            Path(__file__).with_name("probe_m1_attention_precision.py").read_bytes()
        ).hexdigest(),
        "attention": attention_witnesses(),
        "packed_m1_state_layout": state_layout_witnesses(
            args.packed_source.read_text()
        ),
        "fp8_formats": format_witness(),
        "limits": [
            "No GPU kernels executed",
            "No model or chat quality assessed",
            "Synthetic admissible inputs do not establish frequency in model-generated activations",
            "FP16 attention is a deliberate approximation; these witnesses refute exactness, not a declared tolerance bound",
            "Padded state layout and FNUZ cache are not established as live configurations",
        ],
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

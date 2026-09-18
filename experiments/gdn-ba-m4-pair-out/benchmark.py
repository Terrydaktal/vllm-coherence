#!/usr/bin/env python3
"""Gate an exact, allocation-free GDN B/A M4+M4 projection on a stopped R9700."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROWS = 8
GROUP_ROWS = 4
HIDDEN_SIZE = 5120
BA_HALF_WIDTH = 48
BA_WIDTH = BA_HALF_WIDTH * 2
EXPECTED_GDN_LAYERS = 48
REQUIRED_WHOLE_ROUND_SAVING_MS = 2.0
QUALIFIED_CU_COUNT = 32
UPSTREAM_WVSPLITK_COMMIT = "d626108b1841888ec90aced33367149a6bbc7e4b"
UPSTREAM_WVSPLITK_SOURCE_SHA256 = "013f14b570cd8f25e254bf47643ba2802ab7d5fdd2069adb111bc6ff560f6682"
UPSTREAM_WVSPLITK_URL = (
    "https://github.com/vllm-project/vllm/blob/"
    f"{UPSTREAM_WVSPLITK_COMMIT}/csrc/rocm/skinny_gemms.cu"
)
DEFAULT_MODEL = Path("/home/lewis/models/Qwen3.8-27B-int4-AutoRound")
WEIGHT_PREFIX = "model.language_model.layers"
EDGE_SCENARIOS = (
    "random",
    "zeros",
    "signed_zeros",
    "alternating",
    "scale_boundaries",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_live_vllm_processes(proc_root: Path = Path("/proc")) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    own_pid = os.getpid()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            comm = (entry / "comm").read_text().strip()
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "VLLM::EngineCore" in f"{comm} {cmdline}" or (
            "vllm" in cmdline and " serve " in cmdline
        ):
            matches.append({"pid": int(entry.name), "comm": comm, "cmdline": cmdline.strip()})
    return sorted(matches, key=lambda item: item["pid"])


def load_extension(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "gdn_ba_m4_pair_wvsplitk_out"):
        raise RuntimeError("extension lacks gdn_ba_m4_pair_wvsplitk_out")
    return module


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "minimum_ms": min(values),
        "p10_ms": percentile(values, 0.10),
        "median_ms": statistics.median(values),
        "p90_ms": percentile(values, 0.90),
        "maximum_ms": max(values),
    }


def bit_mismatches(torch: Any, left: Any, right: Any) -> int:
    if left.dtype != torch.bfloat16 or right.dtype != torch.bfloat16:
        raise RuntimeError("bit comparison requires BF16 tensors")
    if tuple(left.shape) != tuple(right.shape):
        raise RuntimeError("bit comparison requires equal shapes")
    return int(torch.count_nonzero(left.view(torch.int16) != right.view(torch.int16)).item())


def _tensor_sha256(torch: Any, tensor: Any) -> str:
    # Avoid NumPy BF16 conversion: hash the physical 16-bit representation.
    return hashlib.sha256(tensor.contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()


def load_qualified_wvsplitk() -> tuple[Any, dict[str, str]]:
    """Load and source-bind the exact operator used by the release M4 path."""

    version = importlib.metadata.version("vllm")
    if "d626108b1" not in version:
        raise RuntimeError(
            f"installed vLLM is not source-bound to the qualified wvSplitK commit: {version}"
        )
    from vllm import _custom_ops as ops

    if not hasattr(ops, "wvSplitK"):
        raise RuntimeError("installed vLLM lacks the qualified ROCm wvSplitK operator")
    return ops, {
        "vllm_version": version,
        "upstream_commit": UPSTREAM_WVSPLITK_COMMIT,
        "upstream_source": UPSTREAM_WVSPLITK_URL,
        "upstream_source_sha256": UPSTREAM_WVSPLITK_SOURCE_SHA256,
    }


def qualified_baseline(torch: Any, ops: Any, hidden: Any, weight: Any, cu_count: int) -> Any:
    """The authenticated target's ordered M4 then M4 arithmetic."""

    first = ops.wvSplitK(weight, hidden[:GROUP_ROWS], cu_count, None)
    second = ops.wvSplitK(weight, hidden[GROUP_ROWS:], cu_count, None)
    return torch.cat((first, second), dim=0)


def candidate_out(extension: Any, hidden: Any, weight: Any, output: Any) -> Any:
    result = extension.gdn_ba_m4_pair_wvsplitk_out(hidden, weight, output)
    if int(result.data_ptr()) != int(output.data_ptr()):
        raise RuntimeError("candidate returned a different output allocation")
    return result


def resolve_model_contract(model: Path) -> tuple[list[int], dict[str, str], dict[str, Any]]:
    config_path = model / "config.json"
    index_path = model / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("model config or safetensors index is missing")
    config = json.loads(config_path.read_text())
    text_config = config.get("text_config", config)
    if int(text_config.get("hidden_size", -1)) != HIDDEN_SIZE:
        raise RuntimeError("model hidden size is outside the qualified geometry")
    if int(text_config.get("linear_num_value_heads", -1)) != BA_HALF_WIDTH:
        raise RuntimeError("model value-head count is outside the qualified geometry")
    layer_types = text_config.get("layer_types")
    if not isinstance(layer_types, list):
        raise RuntimeError("model does not publish layer_types")
    layers = [index for index, kind in enumerate(layer_types) if kind == "linear_attention"]
    if len(layers) != EXPECTED_GDN_LAYERS:
        raise RuntimeError(f"expected {EXPECTED_GDN_LAYERS} GDN layers, observed {len(layers)}")
    weight_map = json.loads(index_path.read_text()).get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError("safetensors index has no weight_map")
    provenance = {
        "model": str(model),
        "config_sha256": sha256_file(config_path),
        "index_sha256": sha256_file(index_path),
        "gdn_layers": layers,
    }
    return layers, weight_map, provenance


def load_runtime_ordered_weights(
    torch: Any,
    model: Path,
    layers: Sequence[int],
    weight_map: dict[str, str],
) -> tuple[list[Any], list[dict[str, Any]]]:
    from safetensors import safe_open

    weights: list[Any] = []
    provenance: list[dict[str, Any]] = []
    for layer in layers:
        keys = {
            "b": f"{WEIGHT_PREFIX}.{layer}.linear_attn.in_proj_b.weight",
            "a": f"{WEIGHT_PREFIX}.{layer}.linear_attn.in_proj_a.weight",
        }
        tensors: dict[str, Any] = {}
        shards: dict[str, str] = {}
        for label, key in keys.items():
            shard_name = weight_map.get(key)
            if not isinstance(shard_name, str):
                raise RuntimeError(f"model index lacks {key}")
            shard = model / shard_name
            if not shard.is_file():
                raise FileNotFoundError(shard)
            with safe_open(shard, framework="pt", device="cpu") as handle:
                tensors[label] = handle.get_tensor(key)
            shards[label] = shard_name
            if tensors[label].dtype != torch.bfloat16 or tuple(tensors[label].shape) != (
                BA_HALF_WIDTH,
                HIDDEN_SIZE,
            ):
                raise RuntimeError(
                    f"{key} has unsupported {tensors[label].dtype}/{tuple(tensors[label].shape)}"
                )
        # MergedColumnParallelLinear(output_sizes=[B,A]) loads B before A.
        weight = torch.cat((tensors["b"], tensors["a"]), dim=0).contiguous().cuda()
        if tuple(weight.shape) != (BA_WIDTH, HIDDEN_SIZE):
            raise RuntimeError("merged B/A weight has an invalid shape")
        weights.append(weight)
        provenance.append(
            {
                "layer": layer,
                "b_key": keys["b"],
                "a_key": keys["a"],
                "b_shard": shards["b"],
                "a_shard": shards["a"],
                "merged_weight_sha256": _tensor_sha256(torch, weight),
            }
        )
    return weights, provenance


def make_hidden(torch: Any, scenario: str, seed: int) -> Any:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    shape = (ROWS, HIDDEN_SIZE)
    if scenario == "random":
        hidden = torch.randn(shape, generator=generator, device="cuda", dtype=torch.float32).mul_(
            0.25
        )
    elif scenario == "zeros":
        hidden = torch.zeros(shape, device="cuda", dtype=torch.float32)
    elif scenario == "signed_zeros":
        hidden = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
        hidden.view(torch.int16).view(-1)[1::2] = -32768
        return hidden
    elif scenario == "alternating":
        indices = torch.arange(ROWS * HIDDEN_SIZE, device="cuda").reshape(shape)
        hidden = torch.where(indices % 2 == 0, 2.0, -2.0)
    elif scenario == "scale_boundaries":
        values = torch.tensor(
            [2.0**-7, -(2.0**-7), 0.5, -0.5, 8.0, -8.0, 31.5, -31.5],
            device="cuda",
            dtype=torch.float32,
        )
        indices = torch.arange(ROWS * HIDDEN_SIZE, device="cuda").reshape(shape)
        hidden = values[indices % values.numel()]
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    return hidden.to(torch.bfloat16).contiguous()


def mutate_hidden(torch: Any, hidden: Any, seed: int) -> None:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    replacement = torch.randn(
        tuple(hidden.shape), generator=generator, device="cuda", dtype=torch.float32
    ).mul_(0.375)
    row_offsets = torch.arange(ROWS, device="cuda", dtype=torch.float32).view(ROWS, 1)
    replacement.add_(row_offsets * 0.03125)
    hidden.copy_(replacement.to(torch.bfloat16))


def parity_gate(
    torch: Any,
    ops: Any,
    extension: Any,
    weights: Sequence[Any],
    layers: Sequence[int],
    cu_count: int,
) -> dict:
    cases: list[dict[str, Any]] = []
    total_mismatches = 0
    all_outputs_changed = True
    all_repeatable = True
    all_canaries_intact = True
    for scenario_index, scenario in enumerate(EDGE_SCENARIOS):
        for weight, layer in zip(weights, layers, strict=True):
            seed = 20_260_828 + scenario_index * 1000 + layer
            hidden = make_hidden(torch, scenario, seed)
            reference = qualified_baseline(torch, ops, hidden, weight, cu_count)
            backing = torch.full(
                (ROWS + 2, BA_WIDTH),
                123.0,
                device="cuda",
                dtype=torch.bfloat16,
            )
            output = backing[1:-1]
            candidate_out(extension, hidden, weight, output)
            mismatches = bit_mismatches(torch, output, reference)
            first_digest = _tensor_sha256(torch, output)
            prefix_intact = bool(torch.all(backing[0] == 123.0).item())
            suffix_intact = bool(torch.all(backing[-1] == 123.0).item())
            complete_write = not bool(torch.any(torch.isnan(output)).item())

            repeat = torch.empty_like(output)
            candidate_out(extension, hidden, weight, repeat)
            repeatable = bit_mismatches(torch, output, repeat) == 0

            mutate_hidden(torch, hidden, seed + 50_000)
            mutated_reference = qualified_baseline(torch, ops, hidden, weight, cu_count)
            candidate_out(extension, hidden, weight, output)
            mutated_mismatches = bit_mismatches(torch, output, mutated_reference)
            mutated_digest = _tensor_sha256(torch, output)
            changed = mutated_digest != first_digest

            # This crosses the M4 boundary and proves rows are not accidentally
            # tied to their old physical group or a stale output allocation.
            permutation = torch.tensor([7, 0, 6, 1, 5, 2, 4, 3], device="cuda")
            permuted = hidden.index_select(0, permutation).contiguous()
            permuted_reference = qualified_baseline(torch, ops, permuted, weight, cu_count)
            candidate_out(extension, permuted, weight, output)
            permutation_mismatches = bit_mismatches(torch, output, permuted_reference)

            case_mismatches = mismatches + mutated_mismatches + permutation_mismatches
            total_mismatches += case_mismatches
            all_outputs_changed &= changed
            all_repeatable &= repeatable
            all_canaries_intact &= prefix_intact and suffix_intact and complete_write
            cases.append(
                {
                    "layer": layer,
                    "scenario": scenario,
                    "initial_bf16_bit_mismatches": mismatches,
                    "mutated_bf16_bit_mismatches": mutated_mismatches,
                    "permuted_bf16_bit_mismatches": permutation_mismatches,
                    "mutated_output_changed": changed,
                    "bitwise_repeatable": repeatable,
                    "prefix_canary_intact": prefix_intact,
                    "suffix_canary_intact": suffix_intact,
                    "complete_finite_write": complete_write,
                }
            )

    alias_rejected = False
    hidden = make_hidden(torch, "random", 91_337)
    alias = hidden.view(-1)[: ROWS * BA_WIDTH].view(ROWS, BA_WIDTH)
    try:
        candidate_out(extension, hidden, weights[0], alias)
    except RuntimeError:
        alias_rejected = True

    return {
        "cases": cases,
        "case_count": len(cases),
        "full_bf16_bit_mismatches": total_mismatches,
        "all_mutated_outputs_changed": all_outputs_changed,
        "all_bitwise_repeatable": all_repeatable,
        "all_output_canaries_intact": all_canaries_intact,
        "overlapping_output_rejected": alias_rejected,
        "passed": (
            total_mismatches == 0
            and all_outputs_changed
            and all_repeatable
            and all_canaries_intact
            and alias_rejected
            and len(cases) == EXPECTED_GDN_LAYERS * len(EDGE_SCENARIOS)
        ),
    }


def measure_once(torch: Any, operation: Callable[[], Any]) -> tuple[float, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    operation()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)), (time.perf_counter() - wall_start) * 1000.0


def timing_gate(
    torch: Any,
    ops: Any,
    extension: Any,
    weights: Sequence[Any],
    layers: Sequence[int],
    *,
    warmups: int,
    repeats: int,
    required_saving_ms: float,
    cu_count: int,
) -> dict[str, Any]:
    hidden = [make_hidden(torch, "random", 7_000_000 + layer) for layer in layers]
    outputs = [torch.empty((ROWS, BA_WIDTH), device="cuda", dtype=torch.bfloat16) for _ in layers]

    def baseline_round() -> Any:
        result = None
        for activation, weight in zip(hidden, weights, strict=True):
            result = qualified_baseline(torch, ops, activation, weight, cu_count)
        return result

    def candidate_round() -> Any:
        result = None
        for activation, weight, output in zip(hidden, weights, outputs, strict=True):
            result = candidate_out(extension, activation, weight, output)
        return result

    for iteration in range(warmups):
        ordered = (
            (baseline_round, candidate_round)
            if iteration % 2 == 0
            else (
                candidate_round,
                baseline_round,
            )
        )
        for operation in ordered:
            operation()
    torch.cuda.synchronize()

    # A steady candidate call is permitted to construct host-side Tensor views,
    # but it must not retain any new device allocation.
    allocated_before = int(torch.cuda.memory_allocated())
    for _ in range(10):
        candidate_round()
    torch.cuda.synchronize()
    allocated_after = int(torch.cuda.memory_allocated())

    baseline_gpu: list[float] = []
    baseline_wall: list[float] = []
    candidate_gpu: list[float] = []
    candidate_wall: list[float] = []
    for iteration in range(repeats):
        if iteration % 2 == 0:
            baseline_values = measure_once(torch, baseline_round)
            candidate_values = measure_once(torch, candidate_round)
        else:
            candidate_values = measure_once(torch, candidate_round)
            baseline_values = measure_once(torch, baseline_round)
        baseline_gpu.append(baseline_values[0])
        baseline_wall.append(baseline_values[1])
        candidate_gpu.append(candidate_values[0])
        candidate_wall.append(candidate_values[1])

    gpu_savings = [
        baseline - candidate
        for baseline, candidate in zip(baseline_gpu, candidate_gpu, strict=True)
    ]
    wall_savings = [
        baseline - candidate
        for baseline, candidate in zip(baseline_wall, candidate_wall, strict=True)
    ]
    median_gpu_saving = statistics.median(gpu_savings)
    median_wall_saving = statistics.median(wall_savings)
    no_retained_device_allocation = allocated_before == allocated_after
    return {
        "baseline_gpu": summary(baseline_gpu),
        "candidate_gpu": summary(candidate_gpu),
        "paired_gpu_saving": summary(gpu_savings),
        "baseline_dispatch_wall": summary(baseline_wall),
        "candidate_dispatch_wall": summary(candidate_wall),
        "paired_dispatch_wall_saving": summary(wall_savings),
        "required_whole_round_saving_ms": required_saving_ms,
        "steady_device_allocated_before": allocated_before,
        "steady_device_allocated_after": allocated_after,
        "zero_retained_device_allocation": no_retained_device_allocation,
        "passed": (
            median_gpu_saving >= required_saving_ms
            and median_wall_saving >= required_saving_ms
            and no_retained_device_allocation
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--warmups", type=int, default=9)
    parser.add_argument("--repeats", type=int, default=41)
    parser.add_argument("--required-saving-ms", type=float, default=REQUIRED_WHOLE_ROUND_SAVING_MS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.warmups < 3 or args.repeats < 9:
        parser.error("require at least three warmups and nine repeats")
    if args.required_saving_ms < REQUIRED_WHOLE_ROUND_SAVING_MS:
        parser.error(f"required-saving-ms may not be below {REQUIRED_WHOLE_ROUND_SAVING_MS}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    live = find_live_vllm_processes()
    if live:
        details = "; ".join(f"pid={item['pid']} {item['comm']}" for item in live)
        raise RuntimeError(f"refusing component timing while vLLM is live: {details}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm torch.cuda is unavailable")
    if torch.cuda.get_device_name(0) != "AMD Radeon AI PRO R9700":
        raise RuntimeError("this screen is qualified only on AMD Radeon AI PRO R9700")
    cu_count = int(torch.cuda.get_device_properties(0).multi_processor_count)
    if cu_count != QUALIFIED_CU_COUNT:
        raise RuntimeError(
            f"expected the qualified {QUALIFIED_CU_COUNT}-CU geometry, observed {cu_count}"
        )
    if os.environ.get("VLLM_ROCM_USE_AITER_LINEAR", "0") != "0":
        raise RuntimeError("VLLM_ROCM_USE_AITER_LINEAR must be 0 to match release arithmetic")

    extension_path = args.extension.expanduser().resolve(strict=True)
    model = args.model.expanduser().resolve(strict=True)
    extension = load_extension(extension_path)
    ops, wvsplitk_provenance = load_qualified_wvsplitk()
    layers, weight_map, model_provenance = resolve_model_contract(model)
    weights, weight_provenance = load_runtime_ordered_weights(torch, model, layers, weight_map)
    parity = parity_gate(torch, ops, extension, weights, layers, cu_count)
    timing = timing_gate(
        torch,
        ops,
        extension,
        weights,
        layers,
        warmups=args.warmups,
        repeats=args.repeats,
        required_saving_ms=args.required_saving_ms,
        cu_count=cu_count,
    )
    passed = bool(parity["passed"] and timing["passed"])
    result = {
        "schema": "urn:qwen-r9700:gdn-ba-m4-pair-out-component:v1",
        "candidate": "two-source-bound-wvsplitk-m4-launches-into-m8-out",
        "default_off": True,
        "qualified_geometry": {
            "rows": ROWS,
            "group_rows": GROUP_ROWS,
            "hidden_size": HIDDEN_SIZE,
            "output_size": BA_WIDTH,
            "dtype": "bfloat16",
            "gdn_layers": EXPECTED_GDN_LAYERS,
            "compute_units": QUALIFIED_CU_COUNT,
            "checkpoint_order": ["B", "A"],
        },
        "source": {
            "extension": str(extension_path),
            "extension_sha256": sha256_file(extension_path),
            "qualified_wvsplitk": wvsplitk_provenance,
        },
        "model": model_provenance,
        "weights": weight_provenance,
        "parity": parity,
        "timing": timing,
        "promotion_eligible": passed,
        "passed": passed,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

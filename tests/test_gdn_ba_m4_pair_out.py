from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "gdn-ba-m4-pair-out"


def _load_benchmark():
    path = EXPERIMENT / "benchmark.py"
    spec = importlib.util.spec_from_file_location("gdn_ba_m4_pair_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_is_two_source_bound_m4_launches_without_generic_m8() -> None:
    binding = (EXPERIMENT / "gdn_ba_m4_pair_ext.cpp").read_text()
    kernel = (EXPERIMENT / "gdn_ba_m4_pair_kernel.cu").read_text()
    benchmark = (EXPERIMENT / "benchmark.py").read_text()
    assert "constexpr int64_t kRows = 8;" in binding
    assert "constexpr int64_t kGroupRows = 4;" in binding
    assert "constexpr int kBatchRows = 4;" in kernel
    assert "d626108b1841888ec90aced33367149a6bbc7e4b" in kernel
    assert benchmark.count("013f14b570cd8f25e254bf47643ba2802ab7d5fdd2069adb111bc6ff560f6682") == 1
    assert kernel.count("launch_one_m4(") == 3  # definition plus two ordered calls
    assert "hidden + kBatchRows * kHiddenSize" in kernel
    assert "result + kBatchRows * kOutputSize" in kernel
    assert "torch::empty" not in binding + kernel
    assert "torch::cat" not in binding + kernel
    assert "gemm_strided_batched" not in (binding + kernel).lower()
    assert benchmark.count("ops.wvSplitK(") == 2
    assert "return torch.cat((first, second), dim=0)" in benchmark


def test_candidate_has_fixed_bf16_nonoverlap_abi() -> None:
    source = (EXPERIMENT / "gdn_ba_m4_pair_ext.cpp").read_text()
    for marker in (
        "hidden_states must have shape [8, 5120]",
        "weight must have shape [96, 5120]",
        "output must have shape [8, 96]",
        "hidden_states.scalar_type() == torch::kBFloat16",
        "weight.scalar_type() == torch::kBFloat16",
        "output.scalar_type() == torch::kBFloat16",
        "!byte_ranges_overlap(output, hidden_states)",
        "!byte_ranges_overlap(output, weight)",
        'architecture.find("gfx1201")',
        "properties->multiProcessorCount == 32",
    ):
        assert marker in source


def test_benchmark_gate_covers_all_rows_layers_edges_and_mutation() -> None:
    benchmark = _load_benchmark()
    assert benchmark.ROWS == 8
    assert benchmark.GROUP_ROWS == 4
    assert benchmark.EXPECTED_GDN_LAYERS == 48
    assert benchmark.REQUIRED_WHOLE_ROUND_SAVING_MS == 2.0
    assert set(benchmark.EDGE_SCENARIOS) == {
        "random",
        "zeros",
        "signed_zeros",
        "alternating",
        "scale_boundaries",
    }
    source = (EXPERIMENT / "benchmark.py").read_text()
    for marker in (
        '"full_bf16_bit_mismatches"',
        '"mutated_bf16_bit_mismatches"',
        '"permuted_bf16_bit_mismatches"',
        '"all_mutated_outputs_changed"',
        '"all_bitwise_repeatable"',
        '"all_output_canaries_intact"',
        '"overlapping_output_rejected"',
        '"zero_retained_device_allocation"',
        "median_gpu_saving >= required_saving_ms",
        "median_wall_saving >= required_saving_ms",
    ):
        assert marker in source


def test_model_contract_requires_exact_48_gdn_layer_geometry(tmp_path: Path) -> None:
    benchmark = _load_benchmark()
    model = tmp_path / "model"
    model.mkdir()
    layer_types = [
        "full_attention" if index % 4 == 3 else "linear_attention" for index in range(64)
    ]
    (model / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "hidden_size": 5120,
                    "linear_num_value_heads": 48,
                    "layer_types": layer_types,
                }
            }
        )
    )
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    layers, weight_map, provenance = benchmark.resolve_model_contract(model)
    assert layers == [index for index in range(64) if index % 4 != 3]
    assert len(layers) == 48
    assert weight_map == {}
    assert provenance["gdn_layers"] == layers


def test_cli_refuses_to_lower_two_ms_gate(tmp_path: Path) -> None:
    benchmark = _load_benchmark()
    extension = tmp_path / "candidate.so"
    output = tmp_path / "result.json"
    extension.touch()
    try:
        benchmark.parse_args(
            [
                "--extension",
                str(extension),
                "--output",
                str(output),
                "--required-saving-ms",
                "1.999",
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover - argparse must own this failure
        raise AssertionError("benchmark accepted a sub-2ms promotion threshold")


def test_call_graph_audit_records_correct_stride_and_live_transfer() -> None:
    readme = (EXPERIMENT / "README.md").read_text()
    assert "stride [1687552,1,10240]" in readme
    assert "stride [0,1,10240]" in readme
    assert "4.214565 ms" in readme
    assert "4.305174 ms" in readme
    assert "about 4.55x" in readme

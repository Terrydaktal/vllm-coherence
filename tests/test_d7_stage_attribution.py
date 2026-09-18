import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).parents[1] / "experiments/radiance-public/d7_stage_attribution.py"
SPEC = importlib.util.spec_from_file_location("d7_stage_attribution", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def event(name, parent=None, durations=(), device="DeviceType.CPU"):
    return SimpleNamespace(
        name=name,
        cpu_parent=parent,
        device_type=device,
        kernels=[SimpleNamespace(name="kernel", duration=x) for x in durations],
    )


def test_nested_stages_count_each_kernel_once_and_ignore_device_annotations():
    outer = event(MODULE.PREFIX + "target_other")
    inner = event(MODULE.PREFIX + "gdn_output_norm", outer)
    leaf = event("aten::operation", inner, [3.0, 4.0])
    remainder = event("native::operation", outer, [2.0])
    annotation = event(MODULE.PREFIX + "target_other", durations=[1000], device="DeviceType.CUDA")
    out = MODULE.attribute_events([outer, inner, leaf, remainder, annotation])
    assert out["linked_kernel_us"] == 9.0
    assert out["linked_kernels"] == 3
    assert out["stages"]["gdn_output_norm"]["kernel_us"] == 7.0
    assert out["stages"]["target_other"]["kernel_us"] == 2.0
    assert sum(x["kernel_us"] for x in out["stages"].values()) == 9.0


def test_unattributed_work_remains_visible():
    out = MODULE.attribute_events([event("memcpy", durations=[2])])
    assert out["stages"]["unattributed"]["kernel_us"] == 2


def test_missing_gpu_collection_is_not_reported_as_zero_cost():
    with pytest.raises(ValueError, match="did not link"):
        MODULE.attribute_events([event(MODULE.PREFIX + "embedding")])


@pytest.mark.parametrize("duration", [-1, float("nan"), float("inf")])
def test_invalid_timing_rejected(duration):
    with pytest.raises(ValueError, match="invalid"):
        MODULE.attribute_events([event("op", durations=[duration])])


def test_specific_module_stages_override_parent_groups():
    base = "language_model.model.layers.31"
    assert MODULE.module_stage(base) == "layer_other"
    assert MODULE.module_stage(base + ".linear_attn") == "gdn_other"
    assert MODULE.module_stage(base + ".linear_attn.norm") == "gdn_output_norm"
    assert MODULE.module_stage(base + ".self_attn.q_norm") == "attention_qk_norm"
    assert MODULE.module_stage(base + ".mlp.act_fn") == "mlp_activation"
    assert MODULE.module_stage("language_model.model.norm") == "final_norm"
    assert MODULE.module_stage("unrecognized") is None


def annotation(stage, start, duration, category="user_annotation"):
    return {
        "name": MODULE.PREFIX + stage,
        "ts": start,
        "dur": duration,
        "ph": "X",
        "cat": category,
        "pid": 1,
        "tid": 1,
    }


def test_host_scopes_remove_nested_time_but_keep_waits():
    result = MODULE.attribute_host_scopes(
        [
            annotation("target", 0, 10),
            annotation("norm", 1, 2),
            annotation("attention", 4, 5),
            annotation("norm", 5, 1),
            annotation("target", 0, 1000, "gpu_user_annotation"),
        ]
    )
    assert result["root_host_us"] == 10
    assert result["stages"]["target"]["host_exclusive_us"] == 3
    assert result["stages"]["attention"]["host_exclusive_us"] == 4
    assert result["stages"]["norm"]["host_exclusive_us"] == 3


def test_partial_host_overlap_is_not_silently_double_counted():
    with pytest.raises(ValueError, match="overlap"):
        MODULE.attribute_host_scopes([annotation("a", 0, 10), annotation("b", 9, 2)])


def test_missing_host_annotations_not_reported_as_zero():
    with pytest.raises(ValueError, match="no CPU"):
        MODULE.attribute_host_scopes([])


def runtime(start, correlation, *, thread=1):
    return {
        "ph": "X",
        "cat": "cuda_runtime",
        "name": "hipLaunchKernel",
        "ts": start,
        "dur": 0.1,
        "pid": 1,
        "tid": thread,
        "args": {"correlation": correlation, "External id": 999},
    }


def kernel(duration, correlation):
    return {
        "ph": "X",
        "cat": "kernel",
        "name": "native_kernel",
        "ts": 1000,
        "dur": duration,
        "pid": 0,
        "tid": 1,
        "args": {"correlation": correlation, "External id": 999},
    }


def test_native_hip_uses_launch_scope_not_stale_external_id_or_gpu_timestamp():
    result = MODULE.attribute_trace(
        [
            annotation("target", 0, 10),
            annotation("conv", 1, 2),
            annotation("recurrence", 4, 2),
            runtime(2, 17),
            runtime(5, 18),
            kernel(3, 17),
            kernel(4, 18),
            annotation("target", 0, 10000, "gpu_user_annotation"),
        ]
    )
    assert result["kernel_us"] == 7
    assert result["stages"]["conv"]["kernel_us"] == 3
    assert result["stages"]["recurrence"]["kernel_us"] == 4
    assert result["unlinked_kernels"] == 0


def test_multiple_kernels_per_runtime_and_unknown_correlations_are_preserved():
    result = MODULE.attribute_trace(
        [
            annotation("norm", 0, 5),
            runtime(1, 10),
            kernel(2, 10),
            kernel(3, 10),
            kernel(4, 99),
        ]
    )
    assert result["kernels"] == 3
    assert result["stages"]["norm"]["kernel_us"] == 5
    assert result["stages"]["unattributed"]["kernel_us"] == 4
    assert result["unlinked_kernels"] == 1


def test_launch_thread_and_annotation_end_boundaries_are_respected():
    result = MODULE.attribute_trace(
        [
            annotation("target", 0, 10),
            annotation("norm", 1, 2),
            runtime(3, 1),
            runtime(2, 2, thread=2),
            kernel(4, 1),
            kernel(5, 2),
        ]
    )
    assert result["stages"]["target"]["kernel_us"] == 4
    assert result["stages"]["unattributed"]["kernel_us"] == 5


def test_ambiguous_launch_correlation_fails_instead_of_guessing():
    with pytest.raises(ValueError, match="ambiguous runtime"):
        MODULE.attribute_trace(
            [
                annotation("target", 0, 10),
                runtime(1, 1),
                runtime(2, 1),
                kernel(3, 1),
            ]
        )


def test_trace_without_actual_kernels_fails():
    with pytest.raises(ValueError, match="no GPU"):
        MODULE.attribute_trace([annotation("target", 0, 10), runtime(1, 1)])

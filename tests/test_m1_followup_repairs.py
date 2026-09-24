"""Run real admission wrappers with CPU tensors, including the original failures."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiments/radiance-public"))
    return importlib.import_module("m1_followup_guards"), importlib.import_module(
        "probe_m1_followup_cpu"
    )


def test_packed_admission_rejects_the_observed_wrong_stride(modules):
    guards, probe = modules
    source = (
        ROOT / "benchmarks/fixtures/eager-m1-followup-20260924/packed_decode.py.txt"
    ).read_text()
    assert source.count(guards.PACKED_OLD) == 1
    old = probe.state_layout_witnesses(source)
    fixed = probe.state_layout_witnesses(
        source.replace(guards.PACKED_OLD, guards.PACKED_NEW)
    )
    assert old[1]["accepted_by_actual_wrapper"]
    assert old[1]["wrongly_addressed_values"] == 393216
    assert fixed[0]["accepted_by_actual_wrapper"]
    assert fixed[0]["wrongly_addressed_values"] == 0
    assert not fixed[1]["accepted_by_actual_wrapper"]


@pytest.mark.parametrize(
    "stride_kind",
    [
        "packed",
        "outer_padding",
        "outer_overlap",
        "head_padding",
        "row_padding",
        "column_stride",
    ],
)
def test_packed_layout_neighbors(modules, stride_kind):
    import torch

    guards, probe = modules
    source = (
        (ROOT / "benchmarks/fixtures/eager-m1-followup-20260924/packed_decode.py.txt")
        .read_text()
        .replace(guards.PACKED_OLD, guards.PACKED_NEW)
    )
    call, recorder = probe.packed_admission(source)
    h, v, k = 4, 8, 8
    strides = {
        "packed": (h * v * k, v * k, k, 1),
        "outer_padding": (2 * h * v * k, v * k, k, 1),
        "outer_overlap": (0, v * k, k, 1),
        "head_padding": (2 * h * v * k, 2 * v * k, k, 1),
        "row_padding": (2 * h * v * k, 2 * v * k, 2 * k, 1),
        "column_stride": (2 * h * v * k, 2 * v * k, 2 * k, 2),
    }[stride_kind]
    state = torch.empty_strided((2, h, v, k), strides)
    args = (
        torch.zeros(1, 2 * 2 * k + h * v),
        torch.zeros(1, h),
        torch.zeros(1, h),
        torch.zeros(h),
        torch.zeros(h),
        1.0,
        state,
        torch.zeros(1, 1, h, v),
        torch.tensor([1], dtype=torch.int32),
    )
    if stride_kind in ("packed", "outer_padding"):
        call(*args)
        assert len(recorder.calls) == 1
    else:
        with pytest.raises(ValueError, match="packed non-overlapping"):
            call(*args)
        assert not recorder.calls


def test_fnuz_does_not_reach_ocp_selector(modules):
    guards, _ = modules

    def original(op, **geometry):
        dtype = str(geometry["kv_dtype"])
        return (
            "fp8_kernel"
            if dtype in ("fp8_e4m3", "torch.float8_e4m3fn", "torch.float8_e4m3fnuz")
            else None
        )

    native = SimpleNamespace(select=original, explain=original)
    assert native.select("attention", kv_dtype="torch.float8_e4m3fnuz") is not None
    guards.guard_selector(native)
    assert native.select("attention", kv_dtype="torch.float8_e4m3fnuz") is None
    assert native.select("attention", kv_dtype="torch.float8_e4m3fn") == "fp8_kernel"
    assert native.explain("attention", kv_dtype="torch.float8_e4m3fnuz") is None
    guards.guard_selector(native)
    assert native.select("attention", kv_dtype="float8_e4m3fnuz") is None


@pytest.mark.parametrize(
    "dtype", ["torch.float8_e4m3fnuz", "torch.float16", "torch.int8", "torch.float32"]
)
def test_cache_rejects_same_size_wrong_format(modules, dtype):
    with pytest.raises(RuntimeError, match="OCP E4M3"):
        modules[0].validate_cache_dtype(dtype, "torch.bfloat16", "torch.bfloat16")


def test_cache_accepts_declared_storage(modules):
    for dtype in ("torch.uint8", "torch.float8_e4m3fn", "torch.bfloat16"):
        modules[0].validate_cache_dtype(dtype, "torch.bfloat16", "torch.bfloat16")


def test_format_is_checked_after_cached_geometry(modules):
    import torch

    runtime = importlib.import_module("attention_precision_runtime")

    class Impl:
        def __init__(self):
            self._kv_geometry = None

        def _geometry(self, kv, query, out):
            if self._kv_geometry is None:
                self._kv_geometry = (kv.element_size(), kv.stride())
            return self._kv_geometry

    runtime.guard_geometry(Impl)
    impl = Impl()
    query = torch.empty(1, 24, 256, dtype=torch.bfloat16)
    packed = torch.empty(1, 4, 16, 512, dtype=torch.uint8)
    assert impl._geometry(packed, query, query)[0] == 1
    bf16 = torch.empty(1, 4, 16, 512, dtype=torch.bfloat16)
    assert impl._geometry(bf16, query, query)[0] == 2
    fnuz = torch.empty(1, 4, 16, 512, dtype=torch.float8_e4m3fnuz)
    with pytest.raises(RuntimeError, match="OCP E4M3"):
        impl._geometry(fnuz, query, query)


def test_gdn_patch_refuses_an_unknown_complete_source(modules):
    with pytest.raises(ValueError, match="pinned stable-softplus"):
        modules[0].patch_packed(modules[0].PACKED_OLD)

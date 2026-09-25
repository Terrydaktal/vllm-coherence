"""Experimental graph capture must never freeze excluded startup operators."""

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from qwen_r9700_lab.conformance_instrumentation import HookSet
from qwen_r9700_lab.diagnostic_contract import DiagnosticError


@pytest.mark.parametrize("capture_failure", [False, True])
def test_full_capture_rebinds_repairs_and_restores_on_exit(
    monkeypatch, capture_failure
):
    def repaired(*args):
        return "repaired"

    def excluded_startup(*args):
        return "startup only"

    native = types.SimpleNamespace(
        conv_update=repaired, recurrent_update=repaired, forward_core_fused=repaired
    )
    attention = type("Attention", (), {"forward": staticmethod(repaired)})
    target_attention = types.SimpleNamespace(launch=repaired)

    def packed_attention(*args):
        return "sample-qualified packed attention"

    owners = [(native, name) for name in vars(native)] + [(attention, "forward")]
    observed = []

    class Manager:
        def capture(self, factory):
            for mode in ("PIECEWISE", "FULL"):
                # vLLM's outer FULL capture calls the inner forward with NONE.
                # It is the descriptor, not that argument, which determines
                # whether repaired bindings must be active.
                factory(types.SimpleNamespace(cg_mode=mode, num_tokens=8))("NONE")
            return "captured"

    original_capture = Manager.capture

    def factory(desc):
        def forward(_inner_mode):
            expected = repaired if desc.cg_mode == "FULL" else excluded_startup
            assert all(getattr(owner, name) is expected for owner, name in owners)
            assert target_attention.launch is (
                packed_attention if desc.cg_mode == "FULL" else repaired
            )
            observed.append("repaired capture" if desc.cg_mode == "FULL" else "startup")
            if capture_failure and desc.cg_mode == "FULL":
                raise RuntimeError("injected capture failure")

        return forward

    class Parent:
        def compile_or_warm_up_model(self):
            startup = HookSet()
            try:
                for owner, name in owners:
                    startup.replace(owner, name, excluded_startup)
                try:
                    return self.model_runner.cudagraph_manager.capture(factory)
                finally:
                    assert all(
                        getattr(owner, name) is excluded_startup
                        for owner, name in owners
                    )
            finally:
                startup.close()

    monkeypatch.setitem(
        sys.modules,
        "optimized_d7_worker",
        types.SimpleNamespace(OptimizedWorker=Parent),
    )
    monkeypatch.setitem(
        sys.modules,
        "radiance_r4d_attn",
        types.SimpleNamespace(R4DAttentionImpl=attention),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.config",
        types.SimpleNamespace(CUDAGraphMode=types.SimpleNamespace(FULL="FULL")),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.gpu.cudagraph_utils",
        types.SimpleNamespace(CudaGraphManager=Manager),
    )
    path = (
        Path(__file__).parents[1]
        / "experiments/radiance-public/speed_candidate_worker.py"
    )
    spec = importlib.util.spec_from_file_location("speed_candidate_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    worker = module.SpeedCandidateWorker()
    worker.vllm_config = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(cudagraph_capture_sizes=[8])
    )
    worker.model_runner = types.SimpleNamespace(cudagraph_manager=Manager())
    worker._qwen_persistent_repairs = types.SimpleNamespace(
        prefill=types.SimpleNamespace(native=native)
    )
    worker._qwen_performance_repairs = types.SimpleNamespace(attention=target_attention)
    worker._load_qualified_target_attention = lambda: packed_attention
    monkeypatch.setenv("QWEN_SPEED_FULL_GRAPH_CAPTURE", "1")
    if capture_failure:
        with pytest.raises(RuntimeError, match="injected capture failure"):
            worker.compile_or_warm_up_model()
    else:
        assert worker.compile_or_warm_up_model() == "captured"
        assert worker._qwen_speed_capture_bindings
    assert observed == ["startup", "repaired capture"]
    assert Manager.capture is original_capture
    assert all(getattr(owner, name) is repaired for owner, name in owners)
    assert target_attention.launch is repaired
    worker.vllm_config.compilation_config.cudagraph_capture_sizes = [1, 8]
    with pytest.raises(DiagnosticError, match="only one D7 batch"):
        worker.compile_or_warm_up_model()
    assert observed == ["startup", "repaired capture"]


def test_matched_graph_route_restores_original_priority_and_rejects_live_probe(
    monkeypatch,
):
    monkeypatch.setitem(
        sys.modules,
        "optimized_d7_worker",
        types.SimpleNamespace(OptimizedWorker=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.config",
        types.SimpleNamespace(CUDAGraphMode=types.SimpleNamespace(FULL="FULL")),
    )
    path = (
        Path(__file__).parents[1]
        / "experiments/radiance-public/speed_candidate_worker.py"
    )
    spec = importlib.util.spec_from_file_location("speed_candidate_route_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    @dataclass(frozen=True)
    class Descriptor:
        cg_mode: str

    full, piecewise = Descriptor("FULL"), Descriptor("PIECEWISE")
    original = {(8, 0): [full, piecewise], (1, 0): [piecewise]}
    manager = types.SimpleNamespace(
        _graphs_captured=True, graphs={full: object()}, _candidates=original
    )
    worker = module.SpeedCandidateWorker()
    worker.model_runner = types.SimpleNamespace(cudagraph_manager=manager)
    worker.qwen_speed_set_full_graph(False)
    assert manager._candidates == {(8, 0): [piecewise], (1, 0): [piecewise]}
    worker.qwen_speed_set_full_graph(True)
    assert manager._candidates == original
    worker._qwen_observation = object()
    with pytest.raises(DiagnosticError, match="during a qualification capture"):
        worker.qwen_speed_set_full_graph(False)
    assert manager._candidates == original


def test_paired_drafter_retains_both_captures_and_rejects_live_switch(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "optimized_d7_worker",
        types.SimpleNamespace(OptimizedWorker=object),
    )
    path = (
        Path(__file__).parents[1]
        / "experiments/radiance-public/speed_candidate_worker.py"
    )
    spec = importlib.util.spec_from_file_location("speed_candidate_pair_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    worker = module.SpeedCandidateWorker()
    control, candidate = object(), object()
    manager = types.SimpleNamespace(_graphs_captured=True, graphs={8: candidate})

    def capture():
        assert manager.graphs == {}
        manager.graphs[8] = control

    worker.model_runner = types.SimpleNamespace(
        speculator=types.SimpleNamespace(
            query_cudagraph_manager=manager, capture=capture
        )
    )
    worker._qwen_speed_draft_attention_candidate = {"qualified": True}
    monkeypatch.setenv("QWEN_SPEED_DRAFT_ATTN_AB", "1")
    worker._capture_draft_attention_control()
    assert manager.graphs[8] is candidate
    worker.qwen_speed_set_draft_attention(False)
    assert manager.graphs[8] is control
    worker.qwen_speed_set_draft_attention(True)
    assert manager.graphs[8] is candidate
    worker._qwen_observation = object()
    with pytest.raises(DiagnosticError, match="during a qualification capture"):
        worker.qwen_speed_set_draft_attention(False)
    assert manager.graphs[8] is candidate


@pytest.mark.parametrize(
    "target_claim,projection_count", [(False, 20), (True, 20), (False, 19)]
)
def test_tp1_draft_experiment_rejects_target_claims_or_missing_layers(
    monkeypatch, target_claim, projection_count
):
    shapes = [(6144, 5120), (5120, 4096), (34816, 5120), (5120, 17408)]
    originals = [(3072, 5120), (5120, 2048), (17408, 5120), (5120, 8704)]
    draft_config = types.SimpleNamespace(
        ENABLED=True,
        _CFG={shape: [(16, (1, 4, 1, 1, 1))] for shape in originals},
        _CFG_A8={shape: [(64, (1, 4, 1, 1, 1))] for shape in originals},
    )

    class Model:
        def __init__(self, modules):
            self.modules = modules

        def named_modules(self):
            return self.modules

    class Parent:
        def load_model(self, **kwargs):
            assert all(shape in draft_config._CFG for shape in shapes)
            target = (
                [("forbidden_target", types.SimpleNamespace(_radiance_w4=shapes[0]))]
                if target_claim
                else []
            )
            draft = [
                (
                    f"draft_projection_{i}",
                    types.SimpleNamespace(_radiance_w4=shapes[i % 4]),
                )
                for i in range(projection_count)
            ]
            self.model_runner = types.SimpleNamespace(
                model=Model(target),
                speculator=types.SimpleNamespace(model=Model(draft)),
            )
            return "loaded"

    monkeypatch.setitem(
        sys.modules,
        "optimized_d7_worker",
        types.SimpleNamespace(OptimizedWorker=Parent),
    )
    monkeypatch.setitem(sys.modules, "radiance_w4", draft_config)
    monkeypatch.setenv("QWEN_SPEED_TP1_DRAFT_W4", "1")
    path = (
        Path(__file__).parents[1]
        / "experiments/radiance-public/speed_candidate_worker.py"
    )
    spec = importlib.util.spec_from_file_location("speed_candidate_draft_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    worker = module.SpeedCandidateWorker()
    worker.vllm_config = types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(tensor_parallel_size=1)
    )
    if target_claim or projection_count != 20:
        with pytest.raises(DiagnosticError):
            worker.load_model()
    else:
        assert worker.load_model() == "loaded"
        assert worker._qwen_speed_draft_w4["target_layers_changed"] == []
        assert len(worker._qwen_speed_draft_w4["converted_projections"]) == 20


@pytest.mark.parametrize("enabled", [False, True])
def test_combined_selection_switches_every_installed_candidate(monkeypatch, enabled):
    monkeypatch.setitem(
        sys.modules,
        "optimized_d7_worker",
        types.SimpleNamespace(OptimizedWorker=object),
    )
    path = (
        Path(__file__).parents[1]
        / "experiments/radiance-public/speed_candidate_worker.py"
    )
    spec = importlib.util.spec_from_file_location("speed_combined_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    worker = module.SpeedCandidateWorker()
    worker._qwen_speed_gemm_graphs = {}
    worker._qwen_speed_sampler_graph = object()
    calls = []

    def selector(name):
        def select(value):
            calls.append((name, value))
            return {name: value}

        return select

    for name in ("full_graph", "draft_attention", "gemm", "sampler_graph"):
        setattr(worker, "qwen_speed_set_" + name, selector(name))
    result = worker.qwen_speed_set_combined(enabled)
    assert calls == [
        (name, enabled)
        for name in ("full_graph", "draft_attention", "gemm", "sampler_graph")
    ]
    assert result["gemm"] is enabled
    del worker._qwen_speed_gemm_graphs
    del worker._qwen_speed_sampler_graph
    calls.clear()
    worker.qwen_speed_set_combined(enabled)
    assert calls == [("full_graph", enabled), ("draft_attention", enabled)]

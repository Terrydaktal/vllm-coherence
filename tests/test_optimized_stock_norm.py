"""Real Dynamo/custom-op binding checks using CPU operators as dispatch sentinels."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from qwen_r9700_lab.conformance_instrumentation import HookSet

torch = pytest.importorskip("torch")


def test_compiled_binding_retains_opaque_small_rows_and_original_prefill(monkeypatch):
    calls = []

    class NativeNorm:
        def __init__(self, build):
            self.manifest = {"sha256": "test-binary"}
            self.fast = build == "fast-residual-build"

        def __call__(self, x, residual, weight, eps):
            calls.append(("fast-residual" if self.fast else "norm", x.shape[0]))
            if self.fast:
                assert residual is not None
                return x + 20, residual + 30
            return (x + 2, residual + 3) if residual is not None else x + 2

    class NativeGdn:
        def __call__(self, x, z, weight, eps):
            calls.append(("gdn", x.shape[0]))
            return x + z + 4

    monkeypatch.setitem(sys.modules, "stock_m1_norm", types.SimpleNamespace(StockM1Norm=NativeNorm))
    monkeypatch.setitem(
        sys.modules, "stock_m1_gdn_norm", types.SimpleNamespace(StockM1GdnNorm=NativeGdn)
    )
    path = Path(__file__).parents[1] / "experiments/radiance-public/optimized_stock_norm.py"
    spec = importlib.util.spec_from_file_location("optimized_norm_binding_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inventory = []
    hooks = HookSet()

    def norm_original(x, residual=None):
        return (x + 5, residual + 6) if residual is not None else x + 5

    def gdn_original(x, z=None):
        return x + z + 7

    def add(name, gated=False):
        def repaired(*args, **kwargs):
            raise AssertionError("the non-compiler repair wrapper must be replaced")

        repaired.__wrapped__ = gdn_original if gated else norm_original
        item = types.SimpleNamespace(
            forward=gdn_original if gated else norm_original,
            weight=torch.ones(128 if gated else 5120),
            variance_epsilon=1e-6,
            eps=1e-6,
        )
        hooks.replace(item, "forward", repaired)
        inventory.append((name, item))

    for layer in range(64):
        for suffix in ("input_layernorm", "post_attention_layernorm"):
            add(f"model.layers.{layer}.{suffix}")
        if layer % 4 == 3:
            for suffix in ("q_norm", "k_norm"):
                add(f"model.layers.{layer}.self_attn.{suffix}")
        else:
            add(f"model.layers.{layer}.linear_attn.norm", True)
    add("model.norm")
    model = types.SimpleNamespace(named_modules=lambda: inventory)
    receipt = module.install_compiled_norms(
        model,
        types.SimpleNamespace(manifest={"norm_build": "test-build"}, hooks=hooks),
        residual_build="fast-residual-build",
    )
    assert receipt["norm_modules"] == 161 and receipt["gdn_norm_modules"] == 48
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    norm = torch.compile(inventory[0][1].forward, backend=backend, fullgraph=True)
    for rows, expected in ((1, 6), (8, 3), (9, 6)):
        x = torch.ones(rows, 5120)
        assert torch.equal(norm(x), torch.full_like(x, expected))
    x = torch.ones(8, 5120)
    a, b = norm(x, x)
    assert torch.equal(a, x + 20) and torch.equal(b, x + 30)
    for rows in (1, 9):
        x = torch.ones(rows, 5120)
        a, b = norm(x, x)
        assert torch.equal(a, x + 5) and torch.equal(b, x + 6)
    gated = next(item for name, item in inventory if "linear_attn.norm" in name)
    compiled_gdn = torch.compile(gated.forward, backend=backend, fullgraph=True)
    for rows, expected in ((48, 9), (384, 6), (432, 9)):
        x = torch.ones(rows, 128)
        assert torch.equal(compiled_gdn(x, x), torch.full_like(x, expected))
    assert ("norm", 8) in calls and ("gdn", 384) in calls
    assert ("fast-residual", 8) in calls
    assert all(1 < rows <= 8 for kind, rows in calls if kind == "fast-residual")
    assert all(rows <= 8 for kind, rows in calls if kind == "norm")
    assert any("qwen_d7_qualified" in str(g.graph) for g in graphs)


def test_compiled_gdn_entry_point_binding_validates_before_mutating():
    path = Path(__file__).parents[1] / "experiments/radiance-public/optimized_stock_norm.py"
    spec = importlib.util.spec_from_file_location("optimized_norm_preparation_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class RMSNormGated:
        def __init__(self):
            self._forward_method = self.forward_native

        def forward_native(self):
            raise AssertionError("binding validation must not execute the operator")

        def forward_hip(self):
            raise AssertionError("binding validation must not execute the operator")

        def unexpected(self):
            raise AssertionError("unknown arithmetic")

    modules = [RMSNormGated() for _ in range(48)]
    model = types.SimpleNamespace(
        named_modules=lambda: [
            (f"model.layers.{i}.linear_attn.norm", item) for i, item in enumerate(modules)
        ]
    )
    modules[-1]._forward_method = modules[-1].unexpected
    with pytest.raises(module.DiagnosticError, match="entry point"):
        module.prepare_native_gdn_norms(model)
    assert all(item._forward_method.__name__ == "forward_native" for item in modules[:-1])
    modules[-1]._forward_method = modules[-1].forward_native
    receipt = module.prepare_native_gdn_norms(model)
    assert receipt["previous_entry_points"] == {"forward_native": 48}
    assert all(item._forward_method.__name__ == "forward_hip" for item in modules)

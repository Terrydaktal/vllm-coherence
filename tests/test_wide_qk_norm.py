import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_authenticated_gemma_widens_only_attention_and_restores(monkeypatch):
    import torch

    sources = Path(__file__).parents[1] / "experiments/radiance-public"
    monkeypatch.syspath_prepend(str(sources))
    import wide_qk_norm as repair

    assert (
        hashlib.sha256((sources / "optimized_stock_norm.py").read_bytes()).hexdigest()
        == repair.DISPATCH_SHA256
    )
    assert (
        hashlib.sha256((sources / "stock_m1_gemma_norm.hip").read_bytes()).hexdigest()
        == repair.KERNEL_SHA256
    )

    class Native:
        def __init__(self):
            self.manifest = {
                "source_sha256": repair.KERNEL_SHA256,
                "binary_sha256": repair.BINARY_SHA256,
            }

        def __call__(self, x, residual, weight, eps):
            assert residual is None
            return "native"

    candidate = Native()

    def qualified(x, weight, eps, key):
        if 1 < x.shape[0] <= 8:
            return candidate(x, None, weight, eps)
        return "old"

    def current_dispatch(x, weight, eps, key):
        return "untouched"

    monkeypatch.setattr(
        repair.inspect, "getfile", lambda fn: str(sources / "optimized_stock_norm.py")
    )
    op = SimpleNamespace(_init_fn=qualified, _backend_fns={None: current_dispatch})
    monkeypatch.setattr(
        torch._library.custom_ops, "OPDEFS", {"qwen_d7_qualified::gemma": op}
    )
    worker = SimpleNamespace()
    assert repair.install(worker)["binary_sha256"] == repair.BINARY_SHA256
    for shape, key, device, want in (
        ((1, 4, 256), "model.layers.3.self_attn.k_norm", "cuda", "untouched"),
        ((8, 24, 256), "model.layers.3.self_attn.q_norm", "cuda", "untouched"),
        ((9, 24, 256), "model.layers.3.self_attn.q_norm", "cuda", "native"),
        ((3296, 4, 256), "model.layers.63.self_attn.k_norm", "cuda", "native"),
        ((6592, 5120), "model.layers.3.input_layernorm", "cuda", "untouched"),
        ((9, 4, 256), "model.layers.3.self_attn.q_norm", "cuda", "untouched"),
        ((9, 4, 256), "model.layers.3.self_attn.k_norm", "cpu", "untouched"),
    ):
        x = SimpleNamespace(shape=shape, device=SimpleNamespace(type=device))
        assert op._backend_fns[None](x, None, 1e-6, key) == want
    with pytest.raises(ValueError, match="already installed"):
        repair.install(worker)
    repair.remove(worker)
    assert op._backend_fns[None] is current_dispatch
    candidate.manifest["source_sha256"] = "different"
    with pytest.raises(ValueError, match="numerical reference"):
        repair.install(worker)

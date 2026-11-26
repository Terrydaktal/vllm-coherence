from pathlib import Path
from types import SimpleNamespace


def test_symbolic_graph_obeys_gdn_actual_runtime_rows(tmp_path, monkeypatch):
    import pytest
    import torch

    pytest.importorskip("triton.experimental.gluon")
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "experiments/radiance-public"))
    from gdn_prefill_direct import _project_for_runtime_rows

    calls = {"projection_direct": 0, "direct_rows": {}, "fallback_rows": {}}
    weight = torch.tensor([1.0])
    originals = {weight.data_ptr(): lambda x, z: x + z}

    @torch.library.custom_op("coherence_test::symbolic_projection_dispatch", mutates_args=())
    def project(x: torch.Tensor, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return _project_for_runtime_rows(x, z, w, originals, calls, lambda: x - z)

    @project.register_fake
    def fake(x, z, w):
        return x.new_empty(x.shape)

    # Trace a prefill first, then execute decode and both sides of the cutoff
    # through the same dynamically shaped compiled entry point.
    compiled = torch.compile(
        lambda x, z: project(x, z, weight),
        backend="eager",
        dynamic=True,
        fullgraph=True,
    )
    for m in (128, 8, 63, 64, 1, 1648):
        x, z = torch.full((m, 4), 3.0), torch.ones(m, 4)
        assert torch.equal(compiled(x, z), x - z if m >= 64 else x + z)
    assert calls["fallback_rows"] == {1: 1, 8: 1, 63: 1}
    assert calls["direct_rows"] == {64: 1, 128: 1, 1648: 1}


def test_wide_norm_build_changes_only_native_admission(tmp_path, monkeypatch):
    import hashlib

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "experiments/radiance-public"))
    import build_stock_fp8_epilogue as builder
    from wide_input_norm import WIDE_SOURCE

    def compile_stub(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"compiled-test-output")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(builder.subprocess, "run", compile_stub)
    for rows in (2048, 4096):
        out = tmp_path / str(rows)
        assert builder.build(out, rows) == 0
    default = (tmp_path / "2048/stock_fp8_epilogue.hip").read_bytes()
    wide = (tmp_path / "4096/stock_fp8_epilogue.hip").read_bytes()
    assert wide == default.replace(b"rows > 2048", b"rows > 4096")
    assert hashlib.sha256(wide).hexdigest() == WIDE_SOURCE


def test_wide_input_norm_routes_real_widths_and_restores_exact_callable(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "experiments/radiance-public"))
    import wide_input_norm as repair

    def original(*args):
        return "baseline"

    op = SimpleNamespace(_init_fn=original, _backend_fns={None: original})
    narrow = SimpleNamespace(manifest={"binary_sha256": "old"})
    wide = SimpleNamespace(manifest={"binary_sha256": "new"}, norm=lambda *args: "wide")
    monkeypatch.setattr(repair, "operators", lambda build: (op, narrow, wide))
    worker = SimpleNamespace()
    repair.install(worker, "frozen")
    for rows, expected in (
        (8, "baseline"),
        (880, "baseline"),
        (1648, "baseline"),
        (2048, "baseline"),
        (2049, "wide"),
        (2528, "wide"),
        (4096, "wide"),
        (4097, "baseline"),
    ):
        assert (
            op._backend_fns[None](
                SimpleNamespace(shape=(rows, 5120), device=SimpleNamespace(type="cuda")),
                None,
                None,
                1e-6,
                "layer",
            )
            == expected
        )
    assert repair.remove(worker)["wide_rows"] == {2049: 1, 2528: 1, 4096: 1}
    assert op._backend_fns[None] is original

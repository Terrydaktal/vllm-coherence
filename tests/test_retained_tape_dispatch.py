"""The stage tape must see both real constituent ops on an opaque decode fallback."""

from pathlib import Path

import pytest
import torch


def test_tape_records_constituents_and_checks_full_opaque_result(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "experiments" / "radiance-public"
    monkeypatch.syspath_prepend(str(source))
    from retained_tape_dispatch import RetainedTapeDispatch

    def original(x, z):
        return (x + z) * 2

    library = torch.library.Library("coherence_gdn", "FRAGMENT")
    library.define("direct_tile_projection(Tensor x, Tensor z, Tensor n, float eps, Tensor w, Tensor s, Tensor r) -> Tensor")
    library.impl("direct_tile_projection", lambda x, z, n, eps, w, s, r: original(x, z), "CPU")

    class Recorder:
        busy = False

        def __init__(self):
            self.calls = []

        def invoke(self, name, fn, args, kwargs):
            self.calls.append(name)
            return fn(*args, **kwargs)

    recorder = Recorder()
    x = torch.arange(8, dtype=torch.bfloat16)
    weight = torch.ones(1)
    args = x, x, weight, 0.0, weight, weight, weight
    with RetainedTapeDispatch(recorder, {weight.data_ptr(): original}):
        result = torch.ops.coherence_gdn.direct_tile_projection(*args)
    assert torch.equal(result, original(x, x))
    assert recorder.calls == ["aten.add.Tensor", "aten.mul.Tensor"]
    with pytest.raises(ValueError, match="differs from"):
        with RetainedTapeDispatch(recorder, {weight.data_ptr(): lambda x, z: x - z}):
            torch.ops.coherence_gdn.direct_tile_projection(*args)

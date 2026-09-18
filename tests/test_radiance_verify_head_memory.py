from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize("live_row", [None, 0, 4999])
def test_verify_liveness_scan_is_exact_and_bounded(live_row):
    source = (
        Path(__file__).resolve().parents[1]
        / "experiments/radiance-public/patch_verify_head_memory.py"
    )
    spec = importlib.util.spec_from_file_location("verify_memory_patch", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    namespace = {}
    exec(module.LIVE_WEIGHT, namespace)
    allocations = []

    class Chunk:
        def __init__(self, start, stop):
            self.start, self.stop = start, stop

        def detach(self):
            return self

        def abs(self):
            allocations.append(self.stop - self.start)
            return self

        def max(self):
            return int(live_row is not None and self.start <= live_row < self.stop)

    class Weight:
        shape = (5000, 5120)

        def __getitem__(self, key):
            return Chunk(key.start, min(key.stop, self.shape[0]))

    assert namespace["_qwen_has_live_weight"](Weight()) == (live_row is not None)
    assert max(allocations) <= 1024
    if live_row == 0:
        assert len(allocations) == 1
    else:
        assert sum(allocations) == 5000

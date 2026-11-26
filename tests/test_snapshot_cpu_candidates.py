import importlib.util
import sys
from pathlib import Path

import pytest
from qwen_r9700_lab import radiance_cache

BASE = Path(__file__).resolve().parents[1] / "experiments/radiance-public"
sys.path.insert(0, str(BASE))
from materialize_snapshot_candidates import build


@pytest.fixture()
def variant(tmp_path, monkeypatch):
    build(Path(radiance_cache.__file__).parent, tmp_path)
    spec = importlib.util.spec_from_file_location(
        "snapshot_test_candidate", tmp_path / "radiance_cache.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_four_width_and_aggregate_byte_reservations(variant):
    size = 27_000_832
    charge = 2 * size + variant.HEADER.size + 1
    assert variant.read_parallelism(size, 159) == 4
    assert variant.read_parallelism(size, 159, max_inflight_bytes=3 * charge - 1) == 2
    pool = variant._BUFFER_POOL
    with pool.reserve(4 * charge):
        assert pool.remaining == variant.READ_INFLIGHT_BYTES - 4 * charge
    assert pool.remaining == pool.capacity

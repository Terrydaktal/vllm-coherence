import importlib.util
import sys
from array import array
from pathlib import Path

import pytest
from qwen_r9700_lab import radiance_cache, radiance_prefix_lineage

BASE = Path(__file__).resolve().parents[1] / "experiments/radiance-public"
sys.path.insert(0, str(BASE))
from materialize_snapshot_candidates import build
from snapshot_cpu_candidates import fast_comparison


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


@pytest.mark.parametrize(
    "factory", [str, lambda v: array("i", v), list, lambda v: array("d", v)]
)
def test_prefix_semantics(factory):
    reference = radiance_prefix_lineage.compare_sequences
    compare = fast_comparison(reference)
    for seq in ([], [1], [2, 3], [1, 2, 3], [1, 4, 3]):
        for seq2 in ([], [1, 2, 3], seq + [4]):
            for prefix in (False, True):
                assert compare(factory(seq), factory(seq2), prefix=prefix) == reference(
                    factory(seq), factory(seq2), prefix=prefix
                )


def test_four_width_and_aggregate_byte_reservations(variant):
    size = 27_000_832
    charge = 2 * size + variant.HEADER.size + 1
    assert variant.read_parallelism(size, 159) == 4
    assert variant.read_parallelism(size, 159, max_inflight_bytes=3 * charge - 1) == 2
    pool = variant._BUFFER_POOL
    with pool.reserve(4 * charge):
        assert pool.remaining == variant.READ_INFLIGHT_BYTES - 4 * charge
    assert pool.remaining == pool.capacity


def test_written_bytes_and_verification_of_mixed_head(variant, tmp_path):
    raw = __import__("random").Random(84).randbytes(8192)
    compressed = bytes(len(raw))
    store = variant.ChatStore(tmp_path, {"id": "1" * 64, "generation": "2" * 64})
    store.activate()
    keys = [f"g0-{idx:064x}.qkv" for idx in range(6)]
    for key, data in zip(keys[:4], (raw, compressed, raw, compressed)):
        assert store.write(key, memoryview(bytearray(data)))
        assert store.path(key).read_bytes() == radiance_cache.encode_block(data)
    for key in keys[2:4]:
        store.path(key).write_bytes(store.path(key).read_bytes()[:-2])
    actual = store.prepare_publication(list(reversed(keys)), len(raw))
    assert list(actual["verified"]) == keys[:2]
    assert actual["invalid"] == keys[2:4]
    assert actual["missing"] == keys[4:]
    assert list(store.read_many(keys[:2], len(raw))) == [raw, compressed]
    assert type(store.read(keys[0], len(raw))) is bytes

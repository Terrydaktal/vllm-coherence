import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "stock_attention_padding",
    Path(__file__).resolve().parents[1] / "experiments/radiance-public/stock_m1_attention.py",
)
attention = importlib.util.module_from_spec(spec)
spec.loader.exec_module(attention)


class Buffer:
    def __init__(self, rows, storage=None):
        self.shape = (rows, 24, 256)
        self.storage = storage if storage is not None else object()

    def is_contiguous(self):
        return True

    def __getitem__(self, region):
        assert region.start is None and region.step is None
        return Buffer(region.stop, self.storage)


@pytest.mark.parametrize("live", range(1, 9))
def test_graph_padding_never_enters_native_attention_launches(live):
    query, output = Buffer(8), Buffer(8)
    active_query, active_output = attention.active_decode_buffers(query, output, live)
    assert active_query.shape == active_output.shape == (live, 24, 256)
    assert active_query.storage is query.storage
    assert active_output.storage is output.storage
    assert query.shape == output.shape == (8, 24, 256)


def test_short_output_cannot_be_hidden_by_query_padding():
    with pytest.raises(attention.DiagnosticError, match="layout changed"):
        attention.active_decode_buffers(Buffer(8), Buffer(2), 3)

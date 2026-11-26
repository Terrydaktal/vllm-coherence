import sys
from pathlib import Path


SOURCES = Path(__file__).resolve().parents[1] / "experiments" / "radiance-public"
sys.path.insert(0, str(SOURCES))
from mxfp4_prefill_tiles import _MEASURED_TAIL_ROWS, eligible_tail


def test_only_measured_partial_rows_use_thin_kernel():
    picked = _MEASURED_TAIL_ROWS
    assert picked == {129, 160, 320, 384, 513}
    assert all(eligible_tail(m) for m in picked)

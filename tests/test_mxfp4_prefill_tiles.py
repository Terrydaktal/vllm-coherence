import sys
from pathlib import Path

import pytest

SOURCES = Path(__file__).resolve().parents[1] / "experiments" / "radiance-public"
sys.path.insert(0, str(SOURCES))
from mxfp4_prefill_tiles import eligible_tail, measured_config


@pytest.mark.parametrize("mode", (0, 1, 3))
@pytest.mark.parametrize("n", (14336, 16480, 34816))
def test_only_measured_partial_rows_use_thin_kernel(mode, n):
    picked = {m for m in range(1, 4097) if measured_config(m, n, mode, 2)[1]}
    assert picked == {129, 160, 320, 384, 513}
    assert all(eligible_tail(m) for m in picked)


@pytest.mark.parametrize("n", (14336, 16480, 34816))
def test_packed_slab_qualification_is_distinct_from_plain_weights(n):
    chosen = {m for m in range(65, 2048) if measured_config(m, n, 3, 2)[0] == 4}
    assert chosen == ({1024} if n != 16480 else set())
    for m in (2048, 3296):
        assert measured_config(m, n, 3, 4) == (4, False)


def test_projection_specific_tn_for_unprocessed_and_packed_weights():
    for n, plain, packed in (
        (14336, {512, 1024}, {512, 1024, 1648}),
        (16480, {1648}, {1648}),
        (34816, {512, 1024}, {512, 1024}),
    ):
        for mode, wanted in ((0, plain), (1, packed)):
            assert {
                m for m in range(65, 2048) if measured_config(m, n, mode, 2)[0] == 4
            } == wanted

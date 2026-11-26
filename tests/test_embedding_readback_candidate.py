import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/radiance-public"))
import embedding_readback_candidate as candidate
import host_embedding


def test_exact_bits_last_partial_chunk():
    bits = torch.tensor([[0, -32768], [0x7FC1, 0x7FFF], [-1, 2]], dtype=torch.int16)
    weight = bits.view(torch.bfloat16)
    view = weight.clone()
    candidate.check_mapped_table(weight, view, rows=2)
    view.view(torch.int16)[-1, -1] ^= 1
    with pytest.raises(RuntimeError, match="row 2"):
        candidate.check_mapped_table(weight, view, rows=2)


def test_actual_placement_checks_before_swapping(monkeypatch):
    original = host_embedding._replace_storage
    monkeypatch.setattr(host_embedding, "_replace_storage", original)
    monkeypatch.setattr(host_embedding,"_coherence_compare_table", None, raising=False)
    candidate.install(host_embedding)
    w = torch.nn.Parameter(torch.tensor([[1, -0.]],dtype=torch.bfloat16),requires_grad=False)
    before = w.data_ptr()
    with pytest.raises(RuntimeError,match="readback differs"):
        host_embedding._replace_storage(w, lambda _: torch.zeros_like(w))
    assert w.data_ptr() == before
    host_embedding._replace_storage(w, lambda x: x.clone())
    assert w.data_ptr() != before and torch.signbit(w[0,1]).item()

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "experiments/radiance-public")
)
from dflash_prompt_lookup import lookup_source


def history():
    match = list(range(10, 18))
    return match + [4] * 7 + match + [5] * 7 + [6] * 40 + match


def test_lookup_requires_distant_full_continuation_and_prefers_latest_tie():
    tokens = history()
    assert lookup_source(tokens, 40) == (8, 23)
    assert lookup_source(tokens, 56) == (8, 8)
    assert lookup_source(tokens, len(tokens)) == (0, 0)
    assert lookup_source(list(range(10, 18)) * 2, 2) == (8, 8)
    assert lookup_source(list(range(10, 18)) * 2, 9) == (0, 0)


@pytest.mark.skipif(
    os.getenv("TRITON_INTERPRET") != "1", reason="enable the Triton CPU interpreter"
)
def test_entire_coupled_proposal_and_probability_cache_replaced():
    import torch
    from dflash_prompt_lookup_kernels import apply, scan

    tokens = history()
    rows = torch.tensor([[0] * len(tokens), tokens], dtype=torch.int32)
    length = torch.tensor([0, len(tokens)], dtype=torch.int32)
    slot = torch.tensor([1], dtype=torch.int32)
    sampled = torch.tensor([1], dtype=torch.int32)
    best = torch.full((5,), -123, dtype=torch.int32)
    previous = torch.tensor([-1], dtype=torch.int32)
    stats = torch.zeros(3, dtype=torch.int32)
    proposals = torch.full((1, 7), 3, dtype=torch.int64)
    logits = torch.full((2, 7, 32), -float("inf"), dtype=torch.float32)
    logits[:, :, 3:5] = 10.0
    # Include the proposed token among the old entries to exercise clear-and-write.
    logits[:, :, 5] = 8.0
    cached = torch.tensor([3, 4, 5] * 14, dtype=torch.int64).reshape(2, 7, 3)

    hit = torch.empty_like(previous)

    def run(window, same_request=0):
        scan[(5,)](
            rows, rows.stride(0), length, slot, sampled, best, window, 7, 24, 20, 16
        )
        apply[(1,)](
            rows,
            rows.stride(0),
            length,
            slot,
            sampled,
            best,
            5,
            previous,
            stats,
            same_request,
            proposals,
            logits,
            logits.stride(0),
            logits.stride(1),
            cached,
            3,
            7,
            8,
            3,
            3,
            20,
            8,
            4,
            hit,
            num_warps=4,
        )

    run(100)
    assert hit.item() == 0
    assert proposals.tolist() == [[3] * 7]
    assert stats.tolist() == [1, 0, 0]
    run(40)
    assert hit.item() == 1
    assert proposals.tolist() == [[5] * 7]
    assert cached[1].tolist() == [[3, 4, 5]] * 7
    assert torch.isfinite(logits[1]).sum().item() == 7
    assert logits[1, :, 5].tolist() == [0.0] * 7
    assert logits[0, :, 3:6].tolist() == [[10.0, 10.0, 8.0]] * 7
    assert stats.tolist() == [2, 1, 0]

    # A fresh token absent from the old shortlist must be recorded for clearing
    # next round, and the previous point mass must disappear now.
    rows[1, 23:30] = 7
    run(40)
    assert proposals.tolist() == [[7] * 7]
    assert cached[1].tolist() == [[7, 4, 5]] * 7
    assert torch.isfinite(logits[1]).sum().item() == 7
    assert logits[1, :, 7].tolist() == [0.0] * 7

    # Prefill chunks that emitted no token must never inspect or update a proposal.
    sampled.zero_()
    proposals.fill_(9)
    run(40)
    assert hit.item() == 0
    assert proposals.tolist() == [[9] * 7]
    assert stats.tolist() == [4, 2, 0]

    # Reuse of a request slot must never carry over another request's hot state.
    short_match = [20, 21, 22] + [8] * 7 + [19] * 40 + [20, 21, 22]
    rows[1, :len(short_match)] = torch.tensor(short_match)
    length[1] = len(short_match)
    sampled.fill_(4)
    previous.fill_(len(short_match) - 4)
    run(40)
    assert proposals.tolist() == [[9] * 7]
    previous.fill_(len(short_match) - 4)
    run(40, same_request=1)
    assert proposals.tolist() == [[8] * 7]
    assert stats.tolist() == [6, 3, 1]

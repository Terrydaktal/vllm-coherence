"""Negative controls for tentative sampler results; these use CPU tensors."""

import importlib.util
import os
from dataclasses import make_dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.parametrize(
    "fault,accepted",
    [
        ("none", True),
        ("padding_only", True),
        ("logit", False),
        ("valid_token", False),
        ("accepted", False),
        ("rejected", False),
    ],
)
def test_sampler_gate_detects_wrong_logits_tokens_and_commit_width(fault, accepted):
    path = Path(
        os.environ.get(
            "SPEED_SAMPLER_SOURCE",
            str(
                Path(__file__).parents[1]
                / "experiments/radiance-public/speed_graph_sampler.py"
            ),
        )
    )
    spec = importlib.util.spec_from_file_location("sampler_gate_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gate = module.GraphSampler(SimpleNamespace(sample=None))
    expected_tokens = torch.tensor([[11, 22, 33, -1, -1, -1, -1, -1]])
    tokens = expected_tokens.clone()
    expected_count = torch.tensor([3], dtype=torch.int32)
    count = expected_count.clone()
    expected_rejected = torch.tensor([5], dtype=torch.int32)
    rejected = expected_rejected.clone()
    logits = torch.arange(8 * 256, dtype=torch.float32).view(8, 256).to(torch.bfloat16)
    gate.logits = logits.clone()
    if fault == "logit":
        gate.logits.view(torch.int16)[4, 100] ^= 1
    elif fault == "valid_token":
        tokens[0, 2] = 99
    elif fault == "padding_only":
        tokens[0, 7] = 99
    elif fault == "accepted":
        count[0] = 4
    elif fault == "rejected":
        rejected[0] = 4
    gate.outputs = (SimpleNamespace(sampled_token_ids=tokens), count, rejected)
    expected = (
        SimpleNamespace(sampled_token_ids=expected_tokens),
        expected_count,
        expected_rejected,
    )
    if accepted:
        gate.check(expected, logits)
        assert gate.checked_rounds == 1
    else:
        with pytest.raises(RuntimeError):
            gate.check(expected, logits)
        assert gate.checked_rounds == 0


def test_replay_buffers_refresh_after_source_allocations_change():
    path = Path(
        os.environ.get(
            "SPEED_SAMPLER_SOURCE",
            str(
                Path(__file__).parents[1]
                / "experiments/radiance-public/speed_graph_sampler.py"
            ),
        )
    )
    spec = importlib.util.spec_from_file_location("sampler_refresh_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fields = module.GraphSampler.batch_fields
    batch_type = make_dataclass("TestBatch", [(name, object) for name in fields])
    batch = batch_type(
        **{
            name: torch.arange(8, dtype=torch.int64 if i % 2 else torch.int32)
            for i, name in enumerate(fields)
        }
    )
    hidden = torch.ones((8, 32), dtype=torch.bfloat16)
    proposal = SimpleNamespace(draft_logits=torch.ones((7, 256), dtype=torch.bfloat16))
    gate = module.GraphSampler(SimpleNamespace(sample=None, speculator=proposal))
    owners = [SimpleNamespace(gpu=torch.arange(4, dtype=torch.int32)) for _ in range(5)]
    gate.parameter_owners = lambda: owners
    gate.refresh_batch(hidden, batch)
    pointers = {name: getattr(gate.batch, name).data_ptr() for name in fields}
    hidden_pointer, draft_pointer = gate.hidden.data_ptr(), gate.draft.data_ptr()
    parameter_pointers = [value.data_ptr() for value in gate.parameters]
    for name in fields:
        setattr(batch, name, getattr(batch, name) + 100)
    hidden = hidden + 2
    proposal.draft_logits = proposal.draft_logits + 3
    for owner in owners:
        owner.gpu = owner.gpu + 23
    gate.refresh_batch(hidden, batch)
    for name in fields:
        retained, current = getattr(gate.batch, name), getattr(batch, name)
        assert retained.data_ptr() == pointers[name]
        assert retained.data_ptr() != current.data_ptr()
        assert torch.equal(retained, current)
    assert gate.hidden.data_ptr() == hidden_pointer != hidden.data_ptr()
    assert gate.draft.data_ptr() == draft_pointer != proposal.draft_logits.data_ptr()
    assert torch.equal(gate.hidden, hidden)
    assert torch.equal(gate.draft, proposal.draft_logits)
    for owner, retained, pointer in zip(owners, gate.parameters, parameter_pointers):
        assert retained.data_ptr() == pointer != owner.gpu.data_ptr()
        assert torch.equal(retained, owner.gpu)

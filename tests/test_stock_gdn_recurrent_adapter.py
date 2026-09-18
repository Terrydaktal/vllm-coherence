"""Test the real call adapter on CPU; the fake transition is intentionally not GDN."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip(
    "torch", reason="requires CPU Torch; qualification uses the pinned CPU-only stack image"
)

DIRECTORY = Path(__file__).resolve().parents[1] / "experiments/radiance-public"


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    spec = importlib.util.spec_from_file_location(
        "stock_adapter", DIRECTORY / "stock_gdn_recurrent_adapter.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    class Sequence:
        device = torch.device("cpu")

        def run(self, state, qkv, a, b, alog, bias, *, retain_rows):
            self.initial = state.clone()
            self.qkv = qkv.clone()
            count = len(qkv)
            assert list(retain_rows) == list(range(count))
            states = {i: state + i + 1 for i in range(count)}
            return SimpleNamespace(
                rows=count,
                outputs=torch.arange(count, dtype=torch.bfloat16)[:, None, None]
                .expand(count, 48, 128)
                .clone(),
                after_rows=states,
            )

    seq = Sequence()
    args = {
        "q": torch.ones((8, 16, 128), dtype=torch.bfloat16),
        "k": torch.full((8, 16, 128), 2, dtype=torch.bfloat16),
        "v": torch.full((8, 48, 128), 3, dtype=torch.bfloat16),
        "a": torch.zeros((8, 48), dtype=torch.bfloat16),
        "b": torch.ones((8, 48), dtype=torch.bfloat16),
        "A_log": torch.zeros(48),
        "dt_bias": torch.ones(48),
        "ssm_state": torch.arange(10, dtype=torch.float32)[:, None, None, None]
        .expand(10, 48, 128, 128)
        .clone(),
        "o": torch.full((8, 48, 128), -17, dtype=torch.bfloat16),
        "cu": torch.tensor([0, 8], dtype=torch.int32),
        "sidx": torch.tensor([[8, 1, 6, 2, 5, 3, 7, 4]], dtype=torch.int32),
        "num_accepted": torch.tensor([1], dtype=torch.int32),
        "num_seqs": 1,
        "H": 48,
        "Hg": 16,
        "scale": 128**-0.5,
    }
    return m, seq, m.StockRecurrentAdapter(torch, seq), args


@pytest.mark.parametrize("previous_accepted", range(1, 9))
def test_uses_previous_accepted_slot_and_publishes_each_after_row(setup, previous_accepted):
    _, seq, adapter, args = setup
    args["num_accepted"][0] = previous_accepted
    original = {k: v.clone() for k, v in args.items() if isinstance(v, torch.Tensor)}
    selected = int(args["sidx"][0, previous_accepted - 1])
    adapter(**args)
    assert torch.equal(seq.initial, original["ssm_state"][selected])
    assert seq.qkv.shape == (8, 10240)
    assert bool((seq.qkv[:, :2048] == 1).all())
    assert bool((seq.qkv[:, 2048:4096] == 2).all())
    assert bool((seq.qkv[:, 4096:] == 3).all())
    for row, slot in enumerate(args["sidx"][0]):
        assert bool((args["ssm_state"][slot] == selected + row + 1).all())
        assert bool((args["o"][row] == row).all())
    assert torch.equal(args["ssm_state"][[0, 9]], original["ssm_state"][[0, 9]])
    for name in original.keys() - {"ssm_state", "o"}:
        assert torch.equal(args[name], original[name])
    assert (adapter.calls, adapter.rows) == (1, 8)


@pytest.mark.parametrize(
    "fault", ["duplicate", "null", "range", "acceptance", "bias", "fused", "boundaries"]
)
def test_invalid_request_cannot_publish_anything(setup, fault):
    m, _, adapter, args = setup
    if fault == "duplicate":
        args["sidx"][0, 1] = 8
    elif fault == "null":
        args["sidx"][0, 0] = 0
    elif fault == "range":
        args["sidx"][0, 0] = 10
    elif fault == "acceptance":
        args["num_accepted"][0] = 0
    elif fault == "bias":
        args["dt_bias"][0] = 1.001
    elif fault == "fused":
        args["norm"] = (None, 0, 0)
    else:
        args["cu"][1] = 7
    initial = args["ssm_state"].clone()
    output = args["o"].clone()
    with pytest.raises(m.DiagnosticError):
        adapter(**args)
    assert torch.equal(args["ssm_state"], initial) and torch.equal(args["o"], output)
    assert adapter.calls == 0


@pytest.mark.parametrize("fault", ["exception", "incomplete", "nonfinite"])
def test_tentative_computation_failure_cannot_publish_partial_state(setup, fault):
    m, seq, adapter, args = setup
    original = seq.run

    def broken(*a, **kw):
        if fault == "exception":
            raise RuntimeError("injected failure")
        result = original(*a, **kw)
        if fault == "incomplete":
            del result.after_rows[7]
        else:
            result.after_rows[7][0, 0, 0] = float("nan")
        return result

    seq.run = broken
    initial = args["ssm_state"].clone()
    output = args["o"].clone()
    with pytest.raises((m.DiagnosticError, RuntimeError)):
        adapter(**args)
    assert torch.equal(args["ssm_state"], initial) and torch.equal(args["o"], output)


@pytest.mark.parametrize("count", [1, 2, 7])
def test_short_current_batch_can_start_from_last_previous_candidate(setup, count):
    _, seq, adapter, args = setup
    for name in ("q", "k", "v", "o"):
        args[name] = args[name][:count]
    args["cu"][1] = count
    args["num_accepted"][0] = 8
    original = args["ssm_state"].clone()
    adapter(**args)
    assert bool((seq.initial == 4).all())
    touched = args["sidx"][0, :count].tolist()
    for slot in range(10):
        if slot in touched:
            assert bool((args["ssm_state"][slot] == 5 + touched.index(slot)).all())
        else:
            assert torch.equal(args["ssm_state"][slot], original[slot])

"""A state failure must retain the separate causal-boundary diagnosis as well."""

import numpy as np
import pytest

from qwen_r9700_lab import conformance_scenarios as scenarios
from qwen_r9700_lab.conformance_boundaries import BoundaryRecorder
from qwen_r9700_lab.conformance_state import FrameWriter
from qwen_r9700_lab.diagnostic_contract import (
    DiagnosticError,
    digest,
    private_json,
    seal,
    write_private,
)


def capture(root, *, state_delta, boundary_delta):
    root.mkdir(mode=0o700)
    identities = {name: digest(name) for name in ("contract", "execution", "adapter")}
    frame = FrameWriter(
        root / "frame",
        **identities,
        input_digest=digest("input"),
        phase="commit",
        consumed=1,
        pending=2,
        expected=["gdn"],
    )
    frame.array("gdn", np.asarray([state_delta], dtype=np.float32))
    frame.finish()
    write_private(
        root / "schedule.json",
        seal(
            {
                "initial_state": "independent_zero_state",
                "frames": [{"name": "frame", "consumed": 1, "pending": 2}],
            }
        ),
    )
    boundaries = BoundaryRecorder(
        root / "boundaries",
        **identities,
        positions=[0],
        layers=1,
        layer_stages=[["input_norm", "post_attention_norm"]],
        input_digests={0: digest("input")},
    )
    boundaries.record(0, 0, "input_norm", np.asarray([0], dtype=np.float32))
    boundaries.record(0, 0, "post_attention_norm", np.asarray([boundary_delta], dtype=np.float32))
    boundaries.finish()


@pytest.mark.parametrize("state_delta,boundary_delta", [(1, 1), (1, 0), (0, 1), (0, 0)])
def test_state_failure_does_not_omit_boundary_comparison(
    tmp_path, monkeypatch, state_delta, boundary_delta
):
    capture(tmp_path / "serial", state_delta=0, boundary_delta=0)
    capture(tmp_path / "d7", state_delta=state_delta, boundary_delta=boundary_delta)
    # Substitute only model execution. Comparators authenticate and read actual
    # independent tensor artifacts, including the injected semantic differences.
    monkeypatch.setattr(scenarios, "tokens", lambda spec, count, seed: [1] * count)
    monkeypatch.setattr(scenarios, "plan_for", lambda *args, **kwargs: {})
    monkeypatch.setattr(scenarios, "native", lambda spec, plan, root, **kwargs: root)
    if state_delta or boundary_delta:
        with pytest.raises(DiagnosticError, match="M1/D7"):
            scenarios.forced_d7({}, {"axes": {"accepted": 3}, "seed": 17}, tmp_path)
    else:
        assert scenarios.forced_d7({}, {"axes": {"accepted": 3}, "seed": 17}, tmp_path)
    state = private_json(tmp_path / "aligned-comparison.json")
    boundary = private_json(tmp_path / "boundary-comparison/report.json")
    assert all(x["equal"] for x in state["comparisons"]) == (state_delta == 0)
    assert boundary["equal"] == (boundary_delta == 0)
    assert boundary["observations"] == 2
    if boundary_delta:
        assert boundary["first_difference"]["name"] == "p000000000-l000-post_attention_norm"

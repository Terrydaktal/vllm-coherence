"""Do not misattribute unrelated or disconnected tensors to an MLP activation."""

from copy import deepcopy

import numpy as np
import pytest

from qwen_r9700_lab.conformance_mode_boundaries import silu_cut
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, seal


def capture(eager=False):
    values = {}

    def value(key, width):
        values[key] = np.zeros((2, width), dtype=np.int16)
        return {"key": key, "dtype": "torch.bfloat16", "shape": [2, width]}

    before = "1.before.args.1" if eager else "1.before.args.0"
    after = "1.after.mutable.result" if eager else "1.after.out_ptr0"
    events = [
        {
            "index": 0,
            "operation": "radiance.mxfp4_linear.default",
            "logical_identities": ["language_model.model.layers.0.mlp.gate_up_proj.weight"],
            "before": [],
            "after": [value("0.after.result", 34816)],
        },
        {
            "index": 1,
            "operation": "_C.silu_and_mul.default"
            if eager
            else "inductor/triton_poi_fused_mul_mxfp4_linear_silu_slice_0",
            "before": [value(before, 34816)],
            "after": [value(after, 17408)],
        },
        {
            "index": 2,
            "operation": "radiance.mxfp4_linear.default",
            "logical_identities": ["language_model.model.layers.0.mlp.down_proj.weight"],
            "before": [value("2.before.args.0", 17408)],
            "after": [],
        },
    ]
    return {"positions": [60000, 60001], "events": events}, values


@pytest.mark.parametrize("eager", [False, True])
def test_identifies_actual_input_and_output_roles(eager):
    metadata, tensors = capture(eager)
    cut = silu_cut(seal(metadata), tensors, 0)
    assert cut["input"] == ("1.before.args.1" if eager else "1.before.args.0")
    assert cut["output"] == ("1.after.mutable.result" if eager else "1.after.out_ptr0")


@pytest.mark.parametrize(
    "fault",
    [
        "other_layer",
        "duplicate_owner",
        "intervening_op",
        "wrong_input",
        "wrong_output",
        "dtype",
        "missing_output",
    ],
)
def test_rejects_misleading_activation_cuts(fault):
    metadata, tensors = capture()
    if fault == "other_layer":
        metadata["events"][2]["logical_identities"][0] = (
            "language_model.model.layers.1.mlp.down_proj.weight"
        )
    elif fault == "duplicate_owner":
        metadata["events"].append(deepcopy(metadata["events"][0]))
    elif fault == "intervening_op":
        metadata["events"].insert(
            1, {"index": 0.5, "operation": "unknown", "before": [], "after": []}
        )
    elif fault == "wrong_input":
        tensors["1.before.args.0"][0, 0] ^= 1
    elif fault == "wrong_output":
        tensors["2.before.args.0"][0, 0] ^= 1
    elif fault == "dtype":
        metadata["events"][1]["before"][0]["dtype"] = "torch.float32"
    elif fault == "missing_output":
        tensors.pop("1.after.out_ptr0")
    with pytest.raises(DiagnosticError):
        silu_cut(seal(metadata), tensors, 0)

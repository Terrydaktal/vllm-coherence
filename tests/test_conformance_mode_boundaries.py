import numpy as np
import pytest

from qwen_r9700_lab.conformance_mode_boundaries import admit_bridge, compare_group, summarize
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, seal


def fixture(*, index=0, owner="model.layers.0.input_layernorm", copies=1):
    event = {
        "index": index,
        "operation": "qualified.norm",
        "logical_identities": [owner],
        "before": [{"key": "x"}],
        "after": [{"key": "y"}],
    }
    return seal({"positions": [60000, 60001], "events": [event] * copies})


def test_event_numbers_do_not_define_correspondence():
    tensors = {"x": np.zeros((2, 4), np.float32), "y": np.ones((2, 4), np.float32)}
    group = compare_group(fixture(), fixture(index=97), tensors, tensors)
    result = summarize([group], expected_positions=[60000, 60001], sources={})
    assert result["first_observed_different_boundary"] is None
    assert result["boundaries"][0]["exact_output_positions"] == 2


def test_injected_output_fault_has_exact_inputs():
    before = {"x": np.zeros((2, 4), np.float32), "y": np.ones((2, 4), np.float32)}
    after = {k: v.copy() for k, v in before.items()}
    after["y"].view(np.uint32)[1, 0] ^= 1
    group = compare_group(fixture(), fixture(index=97), before, after)
    result = summarize([group], expected_positions=[60000, 60001], sources={})
    assert result["first_observed_different_boundary"]["position"] == 60001
    assert result["first_observed_different_boundary"]["all_captured_inputs_exact"]
    assert result["boundaries"][0]["different_output_elements"] == 1


def test_repeated_owners_are_not_paired_arbitrarily():
    with pytest.raises(DiagnosticError, match="unambiguous"):
        compare_group(fixture(copies=2), fixture(), {}, {})


def test_different_semantic_owners_are_not_paired():
    with pytest.raises(DiagnosticError, match="unambiguous"):
        compare_group(fixture(), fixture(owner="model.layers.1.input_layernorm"), {}, {})


def test_missing_or_duplicate_positions_are_rejected():
    tensors = {"x": np.zeros((2, 4), np.float32), "y": np.ones((2, 4), np.float32)}
    group = compare_group(fixture(), fixture(), tensors, tensors)
    for groups, expected in [([group], [60000]), ([group, group], [60000, 60001] * 2)]:
        with pytest.raises(DiagnosticError, match="coverage"):
            summarize(groups, expected_positions=expected, sources={})


def bridge_records():
    capture = seal({"positions": 320})
    prefill = seal({"decode_capture": capture["sha256"], "positions": [0, 59999]})
    result = seal(
        {
            "observation": {
                "isolated_capture": {
                    "sha256": capture["sha256"],
                    "prefill_capture": prefill["sha256"],
                }
            }
        }
    )
    bridge = seal(
        {
            "admission": seal(
                {"captures": [False, True], "receipts": [["control"], [result["sha256"]]]}
            ),
            "decode": {"positions": 320, "full_logits_exact": 320},
            "prefill": {"positions": 1, "full_logits_exact": 1},
        }
    )
    return bridge, result, capture, prefill


def test_only_bridged_captures_are_admitted():
    bridge, result, capture, prefill = bridge_records()
    admit_bridge(bridge, result, capture, prefill)
    with pytest.raises(DiagnosticError, match="tensor capture"):
        admit_bridge(bridge, result, seal({"positions": 319}), prefill)
    with pytest.raises(DiagnosticError, match="prefill capture"):
        admit_bridge(bridge, result, capture, seal({"decode_capture": capture["sha256"]}))


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_changed_outputs_invalidate_the_observer(phase):
    bridge, result, capture, prefill = bridge_records()
    bridge.pop("sha256")
    bridge[phase]["full_logits_exact"] -= 1
    with pytest.raises(DiagnosticError, match="observer changed"):
        admit_bridge(seal(bridge), result, capture, prefill)


def test_another_pass_cannot_borrow_a_bridge():
    bridge, result, capture, prefill = bridge_records()
    result.pop("sha256")
    result["different_pass"] = True
    with pytest.raises(DiagnosticError, match="another captured pass"):
        admit_bridge(bridge, seal(result), capture, prefill)

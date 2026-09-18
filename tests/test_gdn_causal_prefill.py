"""Fail-closed checks for native GDN repair evidence and source binding."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "experiments/radiance-public"


def module(name):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


probe = module("probe_gdn_causal_prefill")
builder = module("build_gdn_causal_prefill")
transition = module("capture_gdn_decode_transition")


def records():
    return [
        {"kind": kind, "controls_ok": True, "exact": True}
        for kind in ("partition", "prefix", "native_recurrent", "future_gates", "empty_sequence")
    ]


def test_complete_exact_evidence_is_tested_not_proved():
    assert probe.classify(records()) == "TESTED"


@pytest.mark.parametrize(
    "kind", ["partition", "prefix", "native_recurrent", "future_gates", "empty_sequence"]
)
def test_missing_semantic_boundary_cannot_pass(kind):
    assert probe.classify([r for r in records() if r["kind"] != kind]) == "INCOMPLETE"


@pytest.mark.parametrize("key", ["exact", "controls_ok"])
def test_fault_or_broken_control_cannot_pass(key):
    rows = records()
    rows[2][key] = False
    assert probe.classify(rows) == ("DISCREPANCY" if key == "exact" else "INVALID_CONTROL")


def test_no_gpu_cases_is_not_a_pass():
    assert probe.classify([]) == "INCOMPLETE"


def test_unknown_source_cannot_be_repaired_by_text_replacement():
    with pytest.raises(ValueError, match="preimage mismatch"):
        builder.raw_convolution(b"// changed source with coincidentally matching text")


def test_build_rejects_unbound_input_before_publishing(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    (source / "r4d_gdn_conv_w4_h128_bf16.hip").write_text("// unbound source\n")
    with pytest.raises(ValueError, match="preimage mismatch"):
        builder.build(source, output)
    assert not output.exists()


def test_decode_capture_selects_only_the_first_transition_of_layer_zero():
    root = {"site": transition.LAYER, "parent": None}
    projection = {"site": transition.LAYER + ".in_proj_qkvz", "parent": 0}
    recurrence = {"site": "radiance_gdn.recurrent_update", "parent": 0}
    recorder = SimpleNamespace(rows={0: root, 1: projection, 2: recurrence})
    context = {"phase": "step", "consumed": 2050}
    for row in recorder.rows.values():
        assert transition.selected(recorder, row, context)
        assert not transition.selected(recorder, row, {**context, "consumed": 2051})
        assert not transition.selected(recorder, row, {**context, "phase": "prefill"})
    other_layer = {"site": transition.LAYER.replace("layers.0", "layers.1"), "parent": None}
    recorder.rows[3] = other_layer
    assert not transition.selected(recorder, {**recurrence, "parent": 3}, context)
    assert not transition.selected(recorder, {"site": "target.lm_head", "parent": 0}, context)


def test_decode_capture_rejects_unbound_conv_before_installing_hooks(tmp_path):
    candidate = tmp_path / "changed-conv.py"
    candidate.write_text("# not the qualified convolution implementation\n")
    original = (
        transition.CallRecorder._capture,
        transition.RadianceProbe.attach,
        transition.RadianceProbe.detach,
    )
    with pytest.raises(ValueError, match="candidate hash mismatch"):
        transition.install_transition_hook(candidate)
    assert original == (
        transition.CallRecorder._capture,
        transition.RadianceProbe.attach,
        transition.RadianceProbe.detach,
    )

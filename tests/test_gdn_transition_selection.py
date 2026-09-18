import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def capture():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/radiance-public/capture_gdn_decode_transition.py"
    )
    spec = importlib.util.spec_from_file_location("selected_gdn_transition", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_layer_four_capture_is_limited_to_the_declared_transition_and_ancestry(capture):
    ancestor = {"site": "target.language_model.model.layers.4.linear_attn", "parent": None}
    child = {"site": "radiance_gdn.recurrent_update", "parent": 0}
    recorder = SimpleNamespace(rows={0: ancestor, 1: child})
    assert capture.selected(
        recorder, child, {"phase": "step", "consumed": 130}, layer=4, consumed=130
    )
    assert not capture.selected(
        recorder, child, {"phase": "step", "consumed": 130}, layer=0, consumed=130
    )
    assert not capture.selected(
        recorder, child, {"phase": "step", "consumed": 131}, layer=4, consumed=130
    )
    assert not capture.selected(
        recorder, child, {"phase": "prefill", "consumed": 130}, layer=4, consumed=130
    )


def test_legacy_layer_zero_selection_is_unchanged(capture):
    row = {"site": capture.LAYER, "parent": None}
    assert capture.selected(SimpleNamespace(rows={}), row, {"phase": "step", "consumed": 2050})


def test_attention_capture_selects_only_one_transition_and_layer(capture):
    recorder = SimpleNamespace(rows={})
    context = {"phase": "step", "consumed": 2057}
    for suffix in ("", ".qkv_proj", ".q_norm", ".k_norm", ".rotary_emb", ".attn", ".o_proj"):
        row = {"site": "target.language_model.model.layers.19.self_attn" + suffix}
        assert capture.selected(recorder, row, context, layer=19, consumed=2057)
        assert not capture.selected(recorder, row, context, layer=15, consumed=2057)
        assert not capture.selected(recorder, row, context, layer=19, consumed=2058)
        assert not capture.selected(
            recorder, row, {**context, "phase": "prefill"}, layer=19, consumed=2057
        )


def test_residual_capture_includes_both_terms_but_not_weight_or_cache_modules(capture):
    recorder = SimpleNamespace(rows={})
    context = {"phase": "step", "consumed": 130}
    for name in (
        "target.language_model.model.layers.6",
        "target.language_model.model.layers.7.input_layernorm",
        "target.language_model.model.layers.32.post_attention_layernorm",
        "target.language_model.model.norm",
        "target.language_model.logits_processor",
    ):
        row = {"site": name, "parent": None}
        assert capture.selected(recorder, row, context, layer=4, consumed=130, residuals=True)
        assert not capture.selected(recorder, row, context, layer=4, consumed=130)
    for name in (
        "target.language_model.lm_head",
        "target.language_model.model.embed_tokens",
        "target.language_model.model.layers.7.self_attn",
    ):
        assert not capture.selected(
            recorder, {"site": name, "parent": None}, context, layer=4, consumed=130, residuals=True
        )


@pytest.mark.parametrize("layer,consumed", [(-1, 130), (64, 130), (True, 130), (4, 0), (4, True)])
def test_bad_selection_is_rejected_before_installing_any_hook(capture, tmp_path, layer, consumed):
    before = (
        capture.RadianceProbe.attach,
        capture.RadianceProbe.detach,
        capture.CallRecorder._capture,
    )
    with pytest.raises(ValueError, match="selected"):
        capture.install_transition_hook(tmp_path / "absent", layer=layer, consumed=consumed)
    assert before == (
        capture.RadianceProbe.attach,
        capture.RadianceProbe.detach,
        capture.CallRecorder._capture,
    )

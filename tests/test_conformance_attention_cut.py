import numpy as np
import pytest

from qwen_r9700_lab.conformance_attention_cut import (
    attention_cut,
    bf16_float,
    rotary_formula,
    round_bf16,
    selected_rotary_coefficients,
)
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, seal


def fixture(compiled):
    values, events = {}, []
    rows = 2
    qkv = np.arange(rows * 14336, dtype=np.int16).reshape(rows, 14336)
    qg = qkv[:, :12288].reshape(rows, 24, 512)
    q, gate = qg[:, :, :256].copy(), qg[:, :, 256:].reshape(rows, 6144).copy()
    k, v = [x.reshape(rows, 4, 256).copy() for x in (qkv[:, 12288:13312], qkv[:, 13312:])]
    attended = np.full_like(q, 17)
    sigmoid = np.full_like(gate, 23)
    gated = np.full_like(gate, 41)

    def event(operation, owner, before, after):
        index = len(events)
        e = {"index": index, "operation": operation, "logical_identities": []}
        if owner:
            e["logical_identities"] = [f"language_model.model.layers.3.self_attn.{owner}.weight"]
        for phase, fields in (("before", before), ("after", after)):
            e[phase] = []
            for suffix, value in fields.items():
                key = f"{index}.{phase}.{suffix}"
                values[key] = value.copy()
                e[phase].append({"key": key, "shape": list(value.shape), "dtype": "torch.bfloat16"})
        events.append(e)

    event("radiance.mxfp4_linear.default", "qkv_proj", {}, {"result": qkv})
    event("qwen_d7_qualified.gemma.default", "q_norm", {"args.0": q}, {"result": q})
    event("qwen_d7_qualified.gemma.default", "k_norm", {"args.0": k}, {"result": k})
    event(
        "vllm.unified_attention_with_output.default",
        None,
        {"args.0": q, "args.1": k, "args.2": v, "args.3": np.full_like(q, -99)},
        {"mutable.output": attended},
    )
    if compiled:
        event(
            "inductor/triton_poi_fused_mul_mxfp4_linear_sigmoid_view_0",
            None,
            {"args.0": attended, "args.1": gate},
            {"out_ptr0": gated},
        )
    else:
        event("aten.sigmoid.default", None, {"args.0": gate}, {"result": sigmoid})
        event(
            "aten.mul.Tensor",
            None,
            {"args.0": attended.reshape(rows, 6144), "args.1": sigmoid},
            {"result": gated},
        )
    event("radiance.mxfp4_linear.default", "o_proj", {"args.0": gated}, {})
    return seal({"positions": [60000, 60001], "events": events}), values


@pytest.mark.parametrize("compiled", [False, True])
def test_correct_attention_roles_exclude_uninitialized_output_argument(compiled):
    metadata, values = fixture(compiled)
    cut = attention_cut(metadata, values, 3)
    assert set(cut) == {
        "qkv_projection",
        "query_after_normalization",
        "key_after_normalization",
        "query_after_rotation",
        "key_after_rotation",
        "value",
        "attention_output",
        "gate_input",
        "gated_attention_output",
    }
    assert np.array_equal(cut["attention_output"], np.full((2, 24, 256), 17, dtype=np.int16))
    assert not np.array_equal(cut["gate_input"], cut["attention_output"].reshape(2, 6144))


@pytest.mark.parametrize(
    "fault", ["gate_role", "query_wire", "gate_wire", "output_wire", "value_wire", "sigmoid_wire"]
)
def test_detects_wrong_argument_mapping_and_disconnected_tensors(fault):
    metadata, values = fixture(compiled=fault != "sigmoid_wire")
    if fault == "gate_role":
        values["4.before.args.1"] = values["3.after.mutable.output"].reshape(2, 6144).copy()
    else:
        key = {
            "query_wire": "1.before.args.0",
            "gate_wire": "4.before.args.1",
            "output_wire": "4.after.out_ptr0",
            "value_wire": "3.before.args.2",
            "sigmoid_wire": "5.before.args.1",
        }[fault]
        values[key].flat[0] ^= 1
    with pytest.raises(DiagnosticError, match=r"wiring|connect"):
        attention_cut(metadata, values, 3)


def test_bf16_oracle_rounding_ties_and_negative_truncation():
    value = np.array([1.00390625, 1.01171875, -1.00390625, -1.01171875, -0.0], dtype=np.float32)
    rne = bf16_float(round_bf16(value, "rne"))
    rtz = bf16_float(round_bf16(value, "rtz"))
    assert np.array_equal(rne, [1, 1.015625, -1, -1.015625, -0.0])
    assert np.array_equal(rtz, [1, 1.0078125, -1, -1.0078125, -0.0])
    assert np.signbit(rne[-1]) and np.signbit(rtz[-1])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, np.finfo(np.float32).max])
def test_bf16_oracle_rejects_unhandled_nonfinite_values_and_overflow(value):
    with pytest.raises(DiagnosticError):
        round_bf16(np.array([value], dtype=np.float32), "rne")


def test_rotary_selector_disambiguates_shared_cache_by_attention_interval():
    owner = "language_model.model.layers.3.self_attn.rotary_emb.cos_sin_cache"
    events = [
        {
            "index": 10,
            "operation": "norm",
            "logical_identities": ["language_model.model.layers.3.self_attn.k_norm.weight"],
        },
        {"index": 11, "operation": "inductor/select", "logical_identities": [owner]},
        {"index": 12, "operation": "inductor/rotate_k"},
        {"index": 13, "operation": "inductor/rotate_q"},
        {"index": 14, "operation": "vllm.unified_attention_with_output.default"},
        {"index": 30, "operation": "inductor/select", "logical_identities": [owner]},
    ]
    cosine, sine = np.ones((2, 32), dtype=np.int16), np.full((2, 32), 2, dtype=np.int16)
    values = {"11.after.out_ptr0": cosine, "11.after.out_ptr1": sine}
    for index in (12, 13):
        values[f"{index}.before.args.1"] = cosine.copy()
        values[f"{index}.before.args.2"] = sine.copy()
    metadata = seal({"positions": [60000, 60001], "events": events})
    actual = selected_rotary_coefficients(metadata, values, 3)
    assert np.array_equal(actual[0], cosine) and np.array_equal(actual[1], sine)
    values["12.before.args.1"].flat[0] ^= 1
    with pytest.raises(DiagnosticError, match="coefficient inputs"):
        selected_rotary_coefficients(metadata, values, 3)


def test_rotary_oracle_distinguishes_a_product_tie_and_preserves_nonrotary_tail():
    q = round_bf16(np.full((1, 4, 256), 1.5, dtype=np.float32), "rne")
    q[:, :, 32:64] = 0
    cos = round_bf16(np.full((1, 32), 1.0078125, dtype=np.float32), "rne")
    sin = np.zeros_like(cos)
    for mode, expected in (("rne", 1.515625), ("rtz", 1.5078125)):
        output = rotary_formula(q, cos, sin, mode)
        assert (bf16_float(output[:, :, :32]) == expected).all()
        assert (output[:, :, 32:64] == 0).all()
        assert np.array_equal(output[:, :, 64:], q[:, :, 64:])

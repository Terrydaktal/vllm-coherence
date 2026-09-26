import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from qwen_r9700_lab.conformance_topk import summarize_logits
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, seal


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "current_confirmations",
    ROOT / "experiments/radiance-public/benchmark_current_confirmations.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def report(width):
    logits = summarize_logits(np.arange(64, dtype=np.float32))
    return seal({
        "schema": "urn:qwen:d7-equivalence-private-rows:v1",
        "continuation": "same synthetic inputs",
        "prefill": logits,
        "rows": [{"position": i, "absolute_position": 60000 + i,
                  "target_rows": width, "logits": copy.deepcopy(logits)} for i in range(320)],
    })


def reseal(value):
    return seal({k: v for k, v in value.items() if k != "sha256"})


def test_eager_control_retains_the_existing_rotary_repair():
    base = lambda *a, **kw: {"worker_cls": "original"}
    assert MODULE.current_config(base, {}, "fixed-bf16", execution_mode="eager")[
        "worker_cls"
    ] == "rotary_mode_d7_worker.RotaryRneWorker"
    assert MODULE.current_config(base, {}, "fixed-bf16", execution_mode="compiled")[
        "worker_cls"
    ] == "original"


def test_banked_speed_config_changes_only_candidate_m8_and_its_stage_replay():
    base = lambda *a, **kw: {"worker_cls": "original", "compilation_config": {"cudagraph_mode": "PIECEWISE"}}
    args = {"speed_candidate": True, "execution_mode": "compiled", "speculation": True}
    candidate = MODULE.current_config(base, {}, "fixed-bf16", **args)
    assert candidate["worker_cls"] == "speed_matched_stage_worker.SpeedMatchedStageWorker"
    assert candidate["compilation_config"] == {"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [1, 2, 4, 8]}
    reference = MODULE.current_config(base, {}, "fixed-bf16", **{**args, "speculation": False})
    assert reference["worker_cls"] == "original"
    assert reference["compilation_config"]["cudagraph_mode"] == "PIECEWISE"
    assert MODULE.current_config(base, {}, "fixed-bf16", **{**args, "execution_mode": "eager"})["worker_cls"] == "rotary_mode_d7_worker.RotaryRneWorker"
    assert MODULE.current_config(base, {}, "fixed-bf16", **{**args, "execution_mode": "compiled-no-graphs"})["worker_cls"] == "speed_confirmation_worker.SpeedTapeWorker"


def test_complete_pair_and_late_negative_control():
    left, right = report(1), report(8)
    result = MODULE.compare_pair(left, right, left_width=1, right_width=8)
    assert result["decode"]["full_logits_exact"] == 320
    assert result["decode"]["20"]["ranked_exact"] == 320
    logits = np.arange(64, dtype=np.float32)
    logits[0] = 128
    right["rows"][319]["logits"] = summarize_logits(logits)
    result = MODULE.compare_pair(left, reseal(right), left_width=1, right_width=8)
    assert result["first_different_prediction"] == 319
    assert result["decode"]["1"]["set_exact"] == 319
    assert result["decode"]["20"]["set_exact"] == 319
    assert result["prefill"]["full_logits_exact"]


@pytest.mark.parametrize("defect", ["missing", "position", "width", "fixture"])
def test_incomplete_or_mismatched_execution_cannot_pass(defect):
    left, right = report(1), report(8)
    if defect == "missing":
        right["rows"].pop()
    elif defect == "position":
        right["rows"][10]["absolute_position"] += 1
    elif defect == "width":
        right["rows"][10]["target_rows"] = 1
    else:
        right["continuation"] = "other inputs"
    with pytest.raises(DiagnosticError):
        MODULE.compare_pair(left, reseal(right), left_width=1, right_width=8)


def test_same_top_twenty_cannot_conceal_lower_logit_corruption():
    left, right = report(8), report(8)
    logits = np.arange(64, dtype=np.float32)
    logits[0] += 0.25
    right["rows"][7]["logits"] = summarize_logits(logits)
    result = MODULE.compare_pair(left, reseal(right), left_width=8, right_width=8)
    assert result["decode"]["20"]["ranked_exact"] == 320
    assert result["decode"]["full_logits_exact"] == 319
    assert result["first_different_prediction"] == 7


def test_operator_report_rejects_missing_or_hidden_failure(tmp_path):
    audit = {
        "rows_per_site": 320,
        "checks": [{"stage": name, "site": str(i), "passed": True}
                   for name, count in MODULE.OPERATOR_INVENTORY.items() for i in range(count)],
        "failed_checks": 0, "errors": [], "status": "SAMPLE_CHECKED",
        "release_sha256": "release", "probe_sha256": "probe", "oracle_sha256": "oracle",
        "sources": {}, "elapsed_seconds": 1, "reference": "synthetic",
        "limits": [], "private_chat_read": False,
    }
    source, output = tmp_path / "audit.json", tmp_path / "public.json"
    source.write_text(json.dumps(audit))
    MODULE.summarize_operators(source, output)
    assert json.loads(output.read_text())["checks"] == 2817
    broken = copy.deepcopy(audit)
    broken["checks"].pop()
    source.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="inventory"):
        MODULE.summarize_operators(source, output)
    audit["checks"][-1]["passed"] = False
    audit["failed_checks"] = 1
    source.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="conceals"):
        MODULE.summarize_operators(source, output)


def test_serial_quantization_checks_every_row_and_keeps_inplace_outputs(monkeypatch):
    import importlib
    import torch

    monkeypatch.syspath_prepend(str(ROOT / "experiments/radiance-public"))
    stage = importlib.import_module("current_d7_stage_matrix")
    source = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
    output, scale = torch.empty_like(source), torch.empty(8, 1)
    seen = []
    def quant(out, x, scales, upper):
        assert upper is None and x.shape == (1, 4)
        seen.append(x[0, 0].item())
        out.copy_(x + 1)
        scales.copy_(x[:, :1] + 2)
    assert stage.serial_rows(quant, (output, source, scale, None), {}) is None
    assert seen == list(range(0, 32, 4))
    assert torch.equal(output, source + 1)
    assert torch.equal(scale, source[:, :1] + 2)


def test_current_pointwise_adapter_accepts_explicit_launch_numel(monkeypatch):
    import importlib
    from types import SimpleNamespace
    import torch

    monkeypatch.syspath_prepend(str(ROOT / "experiments/radiance-public"))
    stage = importlib.import_module("current_d7_stage_matrix")
    matrix = stage.CurrentStageMatrix.__new__(stage.CurrentStageMatrix)
    matrix.silu_index = 0
    x, out = torch.ones(8, 34816, dtype=torch.bfloat16), torch.empty(8, 17408, dtype=torch.bfloat16)
    arguments = (x, out, 8 * 17408)
    thawer = SimpleNamespace(thaw=lambda cut: (arguments, {"stream": 0}))
    call = SimpleNamespace(name="inductor/triton_poi_fused_silu_slice_0", cut=((thawer,),), function=lambda *a, **kw: None)
    name, layer, variants = next(matrix.current_cases(call))
    variants["final_eager_m8"](*arguments, stream=0)
    assert name == "MLP SiLU and gating" and layer == "0"
    assert torch.equal(out, torch.nn.functional.silu(x[:, :17408]) * x[:, 17408:])

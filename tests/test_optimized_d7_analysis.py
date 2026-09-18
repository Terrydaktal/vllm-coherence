import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from qwen_r9700_lab.conformance_topk import summarize_logits
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, seal, write_private


@pytest.fixture
def analysis(monkeypatch):
    root = Path(__file__).parents[1] / "experiments/radiance-public"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location(
        "optimized_analysis_test", root / "analyze_optimized_d7.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def evidence(fixture="a", changed=False):
    logits = np.arange(64, dtype=np.float32)
    first = summarize_logits(logits)
    if changed:
        logits[0] = 1000
    second = summarize_logits(logits)
    return seal(
        {
            "continuation": fixture,
            "prefill": first,
            "rows": [
                {"position": 0, "absolute_position": 60000, "logits": first},
                {"position": 1, "absolute_position": 60001, "logits": second},
            ],
        }
    )


def test_reports_first_divergence_without_token_ids(analysis, tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    write_private(a, evidence())
    write_private(b, evidence(changed=True))
    result = analysis.compare(a, b)
    assert result["decode"]["positions"] == 2
    assert result["decode"]["full_logits_exact"] == 1
    assert result["decode"]["1"]["set_exact"] == 1
    assert result["first_different_position"] == 1
    assert result["prefill"]["full_logits_exact"]
    assert '"ids"' not in json.dumps(result)


def test_rejects_different_forced_input_fixture(analysis, tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    write_private(a, evidence())
    write_private(b, evidence(fixture="b"))
    with pytest.raises(DiagnosticError, match="fixture differs"):
        analysis.compare(a, b)

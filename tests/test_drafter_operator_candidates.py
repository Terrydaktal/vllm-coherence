import ast
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "drafter_operator_candidates", ROOT / "experiments/radiance-public/drafter_operator_candidates.py"
)
candidates = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(candidates)


def test_other_drafter_head_consumers_retain_dense_kernel_default(tmp_path):
    source = ROOT / "radiance_drafthead.py"
    generated = tmp_path / "variant.py"
    candidates.write_draft_head(source, generated)
    tree = ast.parse(generated.read_text())
    functions = {item.name: item for item in ast.walk(tree) if isinstance(item, ast.FunctionDef)}
    kernel = functions["_draft_head_int2"]
    assert kernel.args.args[-1].arg == "STORE_DENSE"
    assert ast.literal_eval(kernel.args.defaults[-1]) is True
    unchanged = {item.name: item for item in ast.walk(ast.parse(source.read_text())) if isinstance(item, ast.FunctionDef)}
    for name in ("_emit", "_rerank_exact", "_quantize_head_now"):
        assert ast.dump(functions[name]) == ast.dump(unchanged[name])
    with pytest.raises(ValueError):
        candidates.write_draft_head(source, source)
    with pytest.raises(ValueError):
        candidates.draft_head_source(generated.read_text())

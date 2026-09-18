import ast
import gzip
import importlib.util
from pathlib import Path

import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "experiments/radiance-public"


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    spec = importlib.util.spec_from_file_location(
        "interventions", DIRECTORY / "gdn_intermediate_interventions.py"
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_interventions_are_cumulative_and_only_write_original_destinations(module):
    original = gzip.decompress(
        (
            Path(__file__).with_name("fixtures") / "radiance_stock_gdn_packed_decode.py.gz"
        ).read_bytes()
    )
    for index, stage in enumerate(module.STAGES):
        result = module.intervention_source(original, stage)
        tree = ast.parse(result)
        kernel = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == module.KERNEL
        )
        refs = [
            n
            for n in ast.walk(kernel)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "load"
            and any(isinstance(v, ast.Name) and v.id.startswith("reference_") for v in ast.walk(n))
        ]
        assert len(refs) == (2, 4, 6, 7, 8, 9)[index]
        for call in ast.walk(kernel):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "store"
            ):
                assert not any(
                    isinstance(n, ast.Name) and n.id.startswith("reference_")
                    for n in ast.walk(call.args[0])
                )
        assert "beta_val = tl.sigmoid(b_val).to(tl.float32)" in result.decode()


def test_invalid_boundary_and_unreviewed_source_rejected(module):
    with pytest.raises(ValueError, match="unknown"):
        module.intervention_source(b"anything", "wrong")
    with pytest.raises(ValueError, match="unreviewed"):
        module.intervention_source(b"anything", "qk")

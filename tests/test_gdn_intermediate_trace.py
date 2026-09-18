"""The narrow diagnostic must preserve its source and reject invalid controls."""

import ast
import copy
import gzip
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "experiments/radiance-public"
FIXTURES = Path(__file__).with_name("fixtures")


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    result = []
    for name in ("gdn_intermediate_trace", "probe_gdn_intermediates"):
        spec = importlib.util.spec_from_file_location(name, DIRECTORY / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result.append(module)
    return result


class RemoveTrace(ast.NodeTransformer):
    def visit_Expr(self, node):
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "store"
            and any(isinstance(n, ast.Name) and n.id == "trace" for n in ast.walk(node))
        ):
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        self.generic_visit(node)
        return node if node.body else None


@pytest.mark.parametrize("beta", [False, True])
def test_stock_trace_only_adds_stores_not_arithmetic(modules, beta):
    generator, probe = modules
    original = gzip.decompress((FIXTURES / "radiance_stock_gdn_packed_decode.py.gz").read_bytes())
    candidate = generator.stock_source(original, beta_fp32=beta)
    before = ast.parse(probe.ablation_source(original) if beta else original)
    after = RemoveTrace().visit(ast.parse(candidate))
    for node in after.body:
        if isinstance(node, ast.FunctionDef) and node.name == probe.KERNEL:
            node.args.args = [a for a in node.args.args if a.arg != "trace"]
    assert ast.dump(before) == ast.dump(after)


def test_r4d_trace_retains_ordered_arithmetic_statements(modules):
    generator, _ = modules
    original = gzip.decompress((FIXTURES / "radiance_r4d_gdn_recurrent.hip.gz").read_bytes())
    traced = generator.r4d_source(original, traced=True).decode()
    plain = generator.r4d_source(original, traced=False).decode()
    # These original statements span normalization, gates, both reductions and
    # the persistent update. Their text and order must survive instrumentation.
    source = original.decode()
    block = source[source.index("    // l2 norms:") : source.index("    if (NORM) {")]
    fragments = [
        line.strip()
        for line in block.splitlines()
        if line.strip() and not line.strip().startswith(("//", "#"))
    ]
    for variant in (plain, traced):
        cursor = 0
        for fragment in fragments:
            position = variant.index(fragment, cursor)
            cursor = position + len(fragment)
    assert "if (N != 1 || H != 48 || Hg != 16" in traced
    assert "if (T != 1) return;" in traced


def test_wrong_source_preimage_is_rejected(modules):
    generator, _ = modules
    for function in (
        lambda: generator.stock_source(b"wrong"),
        lambda: generator.r4d_source(b"wrong", traced=True),
    ):
        with pytest.raises(ValueError, match="unreviewed"):
            function()


def test_missing_or_ambiguous_insertion_anchor_rejected(modules):
    generator, _ = modules
    for text in ("nothing", "twice twice"):
        with pytest.raises(ValueError, match="missing or ambiguous"):
            generator.replace_once(text, "twice", "new")


def test_trace_slots_are_complete_disjoint_and_in_range(modules):
    generator, _ = modules
    end = 0
    for name, shape in generator.SHAPES.items():
        assert generator.OFFSETS[name] == end
        length = 1
        for dimension in shape:
            length *= dimension
        end += length
    assert end == generator.ELEMENTS == 817296


@pytest.fixture
def controls():
    names = (
        "stock",
        "stock_repeat",
        "stock_trace",
        "beta_fp32",
        "beta_fp32_trace",
        "r4d_control",
        "r4d_trace",
    )
    return {
        name: {
            "guards_unchanged": True,
            "inputs_unchanged": True,
            "trace_complete": True,
            **{
                key: {"bit_equal": True}
                for key in (
                    "state_to_captured",
                    "output_to_captured",
                    "state_to_untraced",
                    "output_to_untraced",
                )
            },
        }
        for name in names
    }


def test_complete_neutral_controls_admit_diagnosis_only(modules, controls):
    _, probe = modules
    assert probe.classify(controls) == "DIAGNOSTIC_MEASURED"
    del controls["r4d_control"]
    assert probe.classify(controls) == "INCOMPLETE"


@pytest.mark.parametrize("name", ["stock_trace", "beta_fp32_trace", "r4d_trace"])
@pytest.mark.parametrize("field", ["state_to_untraced", "output_to_untraced"])
def test_observer_changes_cannot_be_reported_as_a_backend_difference(
    modules, controls, name, field
):
    _, probe = modules
    controls[name][field]["bit_equal"] = False
    assert probe.classify(controls) == "INVALID_CONTROL"


@pytest.mark.parametrize("name", ["stock", "stock_repeat", "r4d_control"])
def test_original_must_reproduce_preserved_transition(modules, controls, name):
    _, probe = modules
    controls[name]["state_to_captured"]["bit_equal"] = False
    assert probe.classify(controls) == "INVALID_CONTROL"


@pytest.mark.parametrize("field", ["guards_unchanged", "inputs_unchanged", "trace_complete"])
def test_incomplete_or_corrupt_trace_is_rejected(modules, controls, field):
    _, probe = modules
    bad = copy.deepcopy(controls)
    bad["r4d_trace"][field] = False
    assert probe.classify(bad) == "INVALID_CONTROL"


def test_native_probe_cannot_bypass_campaign_gpu_lease(modules, monkeypatch):
    _, probe = modules
    args = SimpleNamespace(allow_gpu=True, build=Path("frozen-build"))
    monkeypatch.delenv("QWEN_CONFORMANCE_GPU_LOCK", raising=False)
    with pytest.raises(ValueError, match="campaign GPU lease"):
        probe.require_native_admission(args)
    monkeypatch.setenv("QWEN_CONFORMANCE_GPU_LOCK", "/qualification/gpu.lock")
    probe.require_native_admission(args)
    args.allow_gpu = False
    with pytest.raises(ValueError, match="explicit admission"):
        probe.require_native_admission(args)

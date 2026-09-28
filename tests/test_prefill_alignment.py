"""Release admission fails closed on numerical or evidence-identity mismatch."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_r9700_lab.diagnostic_contract import seal

SOURCE = (
    Path(__file__).parents[1]
    / "experiments/radiance-public/prefill_alignment_runtime.py"
)
spec = importlib.util.spec_from_file_location("prefill_alignment_test", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path):
    entry = {"sources": {SOURCE.name: module.digest(SOURCE)}}
    builds = {}
    for name in ("attention", "projection"):
        root = tmp_path / name
        root.mkdir()
        build = seal({"name": name})
        (root / "build.json").write_text(json.dumps(build))
        entry[name] = {"build": str(root), "build_sha256": build["sha256"]}
        builds[name] = build["sha256"]

    def model(first, n):
        return {
            "first_position": first,
            "positions": n,
            "hidden_exact_rows": n,
            "hidden_different_elements": 0,
            "logits": {
                "full_logits_exact": n,
                **{str(k): {"set_exact": n, "ranked_exact": n} for k in (1, 10, 20)},
            },
        }

    report = {
        "schema": "urn:coherence:prefill-alignment:v1",
        "status": "SAMPLE_CHECKED",
        "sources": entry["sources"],
        "builds": builds,
        "attention_cases": [
            {"context": c, "differences": 0, "elements": 6144, "finite": True}
            for c in [
                17,
                79,
                1025,
                1030,
                2033,
                60000,
                200000,
                253792,
                1030,
                60000,
                200000,
            ]
        ],
        "projection_cases": [{"different": 0, "elements": 5120} for _ in range(10)],
        "model_comparisons": [
            model(1651, 1000),
            model(60000, 1000),
            model(200000, 320),
        ],
    }
    entry["qualification"] = str(tmp_path / "evidence.json")

    def write():
        path = Path(entry["qualification"])
        path.write_text(json.dumps(seal(report)))
        entry["qualification_sha256"] = module.digest(path)

    return entry, report, write


def test_accepts_only_complete_declared_sample(tmp_path):
    entry, _report, write = fixture(tmp_path)
    write()
    assert module.validate(entry)["status"] == "SAMPLE_CHECKED"


@pytest.mark.parametrize("feature", ["prepared_scan", "input_tiles", "dynamic_conv"])
def test_speed_path_cannot_borrow_old_alignment_evidence(tmp_path, feature):
    entry, _report, write = fixture(tmp_path)
    entry[feature] = True
    write()
    with pytest.raises(ValueError, match="optimization differs"):
        module.validate(entry)


@pytest.mark.parametrize("fault", [None, "state", "output", "source", "coverage"])
def test_prepared_scan_gate_checks_state_as_well_as_output(tmp_path, fault):
    entry, report, write = fixture(tmp_path)
    entry["prepared_scan"] = True
    name = "prepared_prefill_scan.py"
    entry["sources"][name] = module.digest(SOURCE.with_name(name))
    scan = {
        "status": "SAMPLE_CHECKED",
        "sources": {name: entry["sources"][name]},
        "cases": [
            {
                "tiles": {
                    "prepared-4": {
                        "output_different": 0,
                        "state_different": 0,
                        "finite": True,
                    }
                }
            }
            for _ in range(40)
        ],
    }
    report["prepared_scan"] = scan
    if fault in ("state", "output"):
        scan["cases"][0]["tiles"]["prepared-4"][fault + "_different"] = 1
    elif fault == "source":
        scan["sources"][name] = "0" * 64
    elif fault == "coverage":
        scan["cases"].pop()
    write()
    if fault is None:
        assert module.validate(entry)["status"] == "SAMPLE_CHECKED"
    else:
        with pytest.raises(ValueError, match="prepared GDN"):
            module.validate(entry)


def test_dynamic_convolution_retains_decode_and_bulk_kernels():
    path = SOURCE.with_name("prefill_dynamic_conv.py")
    spec = importlib.util.spec_from_file_location("dynamic_conv_test", path)
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            return lambda *args, **kwargs: (self.name, grid, args, kwargs)

    dispatch = conv.PrefillDispatch(Kernel("native"), Kernel("dynamic"))
    for rows, speculative, width, expected in (
        (1, False, 4, "native"),
        (8, True, 4, "native"),
        (9, False, 4, "dynamic"),
        (975, False, 4, "dynamic"),
        (975, True, 4, "native"),
        (975, False, 3, "native"),
        (1648, False, 4, "native"),
        (2048, False, 4, "native"),
        (2049, False, 4, "dynamic"),
        (3296, False, 4, "dynamic"),
        (4096, False, 4, "dynamic"),
        (4097, False, 4, "native"),
    ):
        args = tuple(range(12)) + (rows,)
        kwargs = {"IS_SPEC_DECODING": speculative, "KERNEL_WIDTH": width}
        result = dispatch[(1, 40)](*args, **kwargs)
        assert result == (expected, (1, 40), args, kwargs)


@pytest.mark.parametrize("fault", [None, "state", "source", "native", "coverage"])
def test_dynamic_convolution_requires_its_own_state_evidence(tmp_path, fault):
    entry, report, write = fixture(tmp_path)
    name = "prefill_dynamic_conv.py"
    entry["sources"][name] = module.digest(SOURCE.with_name(name))
    entry.update(dynamic_conv=True, conv_native_sha256="1" * 64)
    evidence = {
        "status": "SAMPLE_CHECKED",
        "source_sha256": entry["sources"][name],
        "native_source_sha256": "1" * 64,
        "cases": [
            {"output_different": 0, "state_different": 0, "finite": True}
            for _ in range(36)
        ],
    }
    report["dynamic_conv"] = evidence
    if fault == "state":
        evidence["cases"][0]["state_different"] = 1
    elif fault == "source":
        evidence["source_sha256"] = "0" * 64
    elif fault == "native":
        evidence["native_source_sha256"] = "0" * 64
    elif fault == "coverage":
        evidence["cases"].pop()
    write()
    if fault is None:
        assert module.validate(entry)["status"] == "SAMPLE_CHECKED"
    else:
        with pytest.raises(ValueError, match="convolution"):
            module.validate(entry)


@pytest.mark.parametrize(
    "fault",
    [
        "attention",
        "projection",
        "full_logits",
        "ordering",
        "state",
        "long",
        "source",
        "build",
        "status",
    ],
)
def test_numerical_and_identity_negative_controls(tmp_path, fault):
    entry, report, write = fixture(tmp_path)
    if fault == "attention":
        report["attention_cases"][0]["differences"] = 1
    if fault == "projection":
        report["projection_cases"][0]["different"] = 1
    if fault == "full_logits":
        report["model_comparisons"][0]["logits"]["full_logits_exact"] -= 1
    if fault == "ordering":
        report["model_comparisons"][0]["logits"]["20"]["ranked_exact"] -= 1
    if fault == "state":
        report["model_comparisons"][0]["hidden_different_elements"] = 1
    if fault == "long":
        report["model_comparisons"].pop()
    if fault == "source":
        report["sources"] = {SOURCE.name: "0" * 64}
    if fault == "build":
        report["builds"]["attention"] = "0" * 64
    if fault == "status":
        report["status"] = "UNPROVED"
    write()
    with pytest.raises(ValueError):
        module.validate(entry)


def test_detects_modified_receipt_and_source(tmp_path):
    entry, _report, write = fixture(tmp_path)
    write()
    Path(entry["qualification"]).write_text("{}")
    with pytest.raises(ValueError, match="qualification changed"):
        module.validate(entry)
    write()
    entry["sources"][SOURCE.name] = "0" * 64
    with pytest.raises(ValueError, match="source changed"):
        module.validate(entry)


def test_only_single_causal_prefill_uses_aligned_kernel():
    impl = SimpleNamespace(num_heads=24, num_kv_heads=4, head_size=256, scale=1 / 16)
    md = SimpleNamespace(r4d_plan=((0, 1, 1648, 0),), causal=True, r4d_max_ctx=200000)
    assert module.prefill_admitted(impl, md)
    for plan in (
        ((0, 1, 8, 0),),
        ((0, 2, 1648, 0),),
        ((0, 1, 4097, 0),),
        ((0, 1, 1648, 1),),
    ):
        md.r4d_plan = plan
        assert not module.prefill_admitted(impl, md)
    for width in (2049, 3296, 4096):
        md.r4d_plan = ((0, 1, width, 0),)
        assert module.prefill_admitted(impl, md)
    md.r4d_plan = ((0, 1, 1648, 0),)
    md.causal = False
    assert not module.prefill_admitted(impl, md)
    md.causal = True
    assert not module.prefill_admitted(impl, md, output_scale=object())


def test_cannot_widen_admission_using_narrow_evidence(tmp_path):
    entry, report, write = fixture(tmp_path)
    entry["max_prefill_rows"] = 4096
    write()
    with pytest.raises(ValueError, match="row admission"):
        module.validate(entry)
    report["max_prefill_rows"] = 4096
    for case in report["attention_cases"]:
        case["rows"] = 1648
    write()
    with pytest.raises(ValueError, match="attention boundaries"):
        module.validate(entry)


def test_normalization_guard_extension_rejects_arithmetic_edits(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "prefill_alignment_runtime", module)
    path = SOURCE.with_name("prepare_prefill_alignment_release.py")
    spec = importlib.util.spec_from_file_location("prefill_admission_test", path)
    freeze = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(freeze)
    parent, current = tmp_path / "parent", tmp_path / "current"
    parent.mkdir()
    current.mkdir()
    for name, guard in (
        ("optimized_stock_norm.py", "(prefill_aligned and 8 < x.shape[0] <= 2048)"),
        ("stock_gdn_norm_quant.py", "or not 1 <= x.shape[0] <= 2048"),
    ):
        text = f"{guard}\nunchanged arithmetic body\n"
        (parent / name).write_text(text)
        (current / name).write_text(text.replace("2048", "4096"))
    assert len(freeze.normalization_admission_revision(parent, current)) == 2
    target = current / "stock_gdn_norm_quant.py"
    target.write_text(target.read_text().replace("unchanged", "changed"))
    with pytest.raises(ValueError, match="row guard"):
        freeze.normalization_admission_revision(parent, current)


@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("stale_tiles", [False, True])
def test_freeze_preserves_parent_and_binds_new_arithmetic(
    tmp_path, monkeypatch, preserve, stale_tiles
):
    monkeypatch.setitem(sys.modules, "prefill_alignment_runtime", module)
    path = SOURCE.with_name("prepare_prefill_alignment_release.py")
    spec = importlib.util.spec_from_file_location("prefill_freeze_test", path)
    freeze = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(freeze)
    entry, evidence, _write = fixture(tmp_path)
    evidence["sources"] = {
        name: module.digest(SOURCE.with_name(name)) for name in freeze.RUNTIME
    }
    evidence["tool_continuations"] = [
        {"status": "PASS_FOR_DECLARED_SCOPE", "cases": [{"same_tokens": True}]}
    ]
    for name in ("attention", "projection"):
        root = Path(entry[name]["build"])
        (root / "candidate.so").write_bytes(b"test-only-native-placeholder")
        build = seal({"files": {"candidate.so": module.digest(root / "candidate.so")}})
        (root / "build.json").write_text(json.dumps(build))
        evidence["builds"][name] = build["sha256"]
    qualification = tmp_path / "qualification.json"
    qualification.write_text(json.dumps(seal(evidence)))
    parent = tmp_path / "parent"
    parent.mkdir()
    repair = seal({"sources": {"optimized_stock_norm.py": "old"}})
    performance = seal({"sources": {}, "reference_repair": repair["sha256"]})
    if stale_tiles:
        performance.pop("sha256")
        performance["activation_tiles"] = {
            "sources": {"prefill_activation_tiles.py": "unqualified-prior-source"}
        }
        performance = seal(performance)
    for name, value in (("repair.json", repair), ("performance.json", performance)):
        (parent / name).write_text(json.dumps(value))
    manifest = {
        "files": {
            name: module.digest(parent / name)
            for name in ("repair.json", "performance.json")
        },
        "repair": "/qualification/repair.json",
        "performance": "/qualification/performance.json",
    }
    original_arithmetic = {}
    if preserve:
        reference = {
            "attention": "unchanged-reference",
            "attention_binary_sha256": "reference-only",
        }
        manifest["prefill_alignment"] = {
            "contract": reference,
            "sources": evidence["sources"],
        }
        original_arithmetic = {
            "prefill_alignment": reference,
            "repair_sha256": "prior-repair",
            "performance_sha256": "prior-performance",
        }
    (parent / "optimized-release.json").write_text(json.dumps(manifest))
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "optimized_d7": {
                    "manifest_sha256": module.digest(parent / "optimized-release.json"),
                    "arithmetic": original_arithmetic,
                }
            }
        )
    )
    output = tmp_path / "frozen"
    if stale_tiles:
        with pytest.raises(ValueError, match="fresh activation-layout qualification"):
            freeze.prepare(
                parent,
                tmp_path / "attention",
                tmp_path / "projection",
                qualification,
                profile,
                output,
                preserve_snapshot_contract=preserve,
            )
        assert not output.exists()
        return
    result = freeze.prepare(
        parent,
        tmp_path / "attention",
        tmp_path / "projection",
        qualification,
        profile,
        output,
        preserve_snapshot_contract=preserve,
    )
    assert result["status"] == "QUALIFIED_NOT_DEPLOYED"
    assert json.loads((parent / "repair.json").read_text()) == repair
    final = json.loads((output / "optimized-release.json").read_text())
    for name, expected in final["files"].items():
        assert module.digest(output / name) == expected
    repaired = json.loads((output / "repair.json").read_text())
    perf = json.loads((output / "performance.json").read_text())
    module.authenticate(repaired)
    module.authenticate(perf)
    current = json.loads((output / "runtime-radiance-1.0.16.json").read_text())[
        "optimized_d7"
    ]
    assert current["manifest_sha256"] == module.digest(
        output / "optimized-release.json"
    )
    if preserve:
        assert current["arithmetic"] == original_arithmetic
        assert final["prefill_alignment"]["arithmetic_reference"][
            "parent_manifest_sha256"
        ] == module.digest(parent / "optimized-release.json")
        assert final["performance_sha256"] != original_arithmetic["performance_sha256"]
        # Changing the core arithmetic cannot borrow the old snapshot identity.
        manifest["prefill_alignment"]["sources"] = dict(
            evidence["sources"], **{"optimized_stock_norm.py": "changed"}
        )
        (parent / "optimized-release.json").write_text(json.dumps(manifest))
        altered = json.loads(profile.read_text())
        altered["optimized_d7"]["manifest_sha256"] = module.digest(
            parent / "optimized-release.json"
        )
        profile.write_text(json.dumps(altered))
        with pytest.raises(ValueError, match="unchanged corrected decode"):
            freeze.prepare(
                parent,
                tmp_path / "attention",
                tmp_path / "projection",
                qualification,
                profile,
                tmp_path / "rejected",
                preserve_snapshot_contract=True,
            )
        assert not (tmp_path / "rejected").exists()
    else:
        assert current["arithmetic"]["performance_sha256"] == perf["sha256"]
        assert current["arithmetic"]["repair_sha256"] == repaired["sha256"]
    assert (
        current["arithmetic"]["prefill_alignment"]
        == final["prefill_alignment"]["contract"]
    )


def test_prefill_install_wraps_decode_in_a_separate_hook_scope(monkeypatch):
    from qwen_r9700_lab.conformance_instrumentation import HookSet

    path = SOURCE.with_name("optimized_d7_performance.py")
    spec = importlib.util.spec_from_file_location("prefill_performance_test", path)
    performance = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(performance)
    owner = SimpleNamespace(forward=lambda: "original")
    initial = owner.forward

    def decode():
        return "decode"

    def prefill():
        return "prefill"

    repair = performance.PerformanceRepairs.__new__(performance.PerformanceRepairs)
    repair.manifest = {"prefill_alignment": {"qualified": True}}
    repair.hooks, repair.prefill_hooks = HookSet(), HookSet()
    repair.hooks.replace(owner, "forward", decode)

    def install(entry, hooks):
        assert entry["qualified"]
        assert hooks is repair.prefill_hooks and hooks is not repair.hooks
        assert owner.forward is decode
        hooks.replace(owner, "forward", prefill)
        return {"installed": True}

    monkeypatch.setitem(
        sys.modules, "prefill_alignment_runtime", SimpleNamespace(install=install)
    )
    repair.install_producers(None)
    assert repair.prefill_alignment == {"installed": True}
    assert owner.forward is prefill
    repair.prefill_hooks.close()
    assert owner.forward is decode
    repair.hooks.close()
    assert owner.forward is initial


def test_dynamic_convolution_patches_the_bound_stock_module(monkeypatch):
    from qwen_r9700_lab.conformance_instrumentation import HookSet

    path = SOURCE.with_name("prefill_dynamic_conv.py")
    spec = importlib.util.spec_from_file_location("prefill_dynamic_conv_test", path)
    dynamic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dynamic)
    monkeypatch.delitem(sys.modules, "qwen_stock_runtime_convolution", raising=False)
    with pytest.raises(ValueError, match="not installed"):
        dynamic.native_module()
    original, candidate = object(), object()
    bound = SimpleNamespace(_causal_conv1d_update_kernel=original)
    ordinary = SimpleNamespace(_causal_conv1d_update_kernel=object())
    monkeypatch.setitem(sys.modules, "qwen_stock_runtime_convolution", bound)
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.layers.mamba.ops.causal_conv1d", ordinary
    )

    def clone(native, sha):
        assert native is original and sha == "pinned-kernel"
        return candidate, sha

    monkeypatch.setattr(dynamic, "clone", clone)
    hooks = HookSet()
    dynamic.install(hooks, "pinned-kernel")
    dispatch = bound._causal_conv1d_update_kernel
    assert dispatch.native is original and dispatch.candidate is candidate
    assert not isinstance(
        ordinary._causal_conv1d_update_kernel, dynamic.PrefillDispatch
    )
    hooks.close()
    assert bound._causal_conv1d_update_kernel is original

import hashlib
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).parents[1] / "experiments/radiance-public"
    monkeypatch.syspath_prepend(str(root))
    return importlib.import_module("stock_gdn_lazy_patches"), importlib.import_module(
        "tp1_lazy_backports"
    )


@pytest.fixture
def recipe(modules, monkeypatch, tmp_path):
    patcher, _ = modules
    source = tmp_path / "source"
    source.mkdir()
    (source / "sub").mkdir()
    (source / "sub/cache.py").write_text("blocks = 9\n")
    (source / "untouched.py").write_text("retained = True\n")

    def digest(value):
        return hashlib.sha256(value.encode()).hexdigest()

    record = {
        "preimages": {"sub/cache.py": digest("blocks = 9\n")},
        "postimages": {"sub/cache.py": digest("blocks = 3\n")},
        "patches": [
            {
                "file": "sub/cache.py",
                "before": "blocks = 9\n",
                "after": "blocks = 3\n",
                "purpose": "lazy window",
            }
        ],
        "upstream": "test fixture, not kernel qualification",
    }
    path = tmp_path / "recipes.json"
    path.write_text(json.dumps(record))
    monkeypatch.setattr(patcher, "__file__", str(path.with_suffix(".py")))
    return source, tmp_path / "overlay", path


def test_overlay_preserves_reference_and_replaces_only_pinned_files(modules, recipe):
    source, target, _ = recipe
    modules[0].overlay(source, target)
    assert (source / "sub/cache.py").read_text() == "blocks = 9\n"
    assert (target / "sub/cache.py").read_text() == "blocks = 3\n"
    assert not (target / "sub/cache.py").is_symlink()
    assert (target / "untouched.py").is_symlink()
    with pytest.raises(FileExistsError):
        modules[0].overlay(source, target)


@pytest.mark.parametrize("corruption", ["source", "anchor", "postimage"])
def test_partial_patch_never_creates_an_overlay(modules, recipe, corruption):
    source, target, recipe_path = recipe
    if corruption == "source":
        (source / "sub/cache.py").write_text("blocks = 17\n")
    else:
        record = json.loads(recipe_path.read_text())
        if corruption == "anchor":
            record["patches"][0]["before"] = "missing"
        else:
            record["postimages"]["sub/cache.py"] = "invalid"
        recipe_path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        modules[0].overlay(source, target)
    assert not target.exists()


@pytest.mark.parametrize(
    "change",
    [
        {"status": "FAILED"},
        {"checks": []},
        {"sources": {}},
        {"negative_controls": False},
        {"alignment_cases": 0},
        {"width_pairs": 0},
        {"canonical_snapshot_roundtrip": False},
    ],
)
def test_incomplete_or_failed_state_evidence_cannot_activate(modules, tmp_path, change):
    _, module = modules
    sources = {
        name: module.digest(Path(module.__file__).with_name(name))
        for name in (
            "stock_gdn_lazy_kernel.py",
            "stock_gdn_scan_kernel.py",
            "probe_stock_gdn_lazy.py",
        )
    }
    names = {f"step-{step}-prefix-{width}" for step in range(40) for width in range(1, 9)}
    names |= {
        f"width-{current}-accepted-{previous}-state"
        for current in range(1, 9)
        for previous in range(1, 9)
    }
    names |= {
        f"align-{mode}-inplace-{same}-extra-{extra}"
        for mode, same in ((0, False), (1, False), (1, True))
        for extra in range(8)
    }
    report = {
        "status": "SAMPLE_CHECKED",
        "rows": 320,
        "alignment_cases": 24,
        "width_pairs": 64,
        "canonical_snapshot_roundtrip": True,
        "negative_controls": True,
        "checks": [{"name": name, "equal": True, "bytes": 4, "unequal_bytes": 0} for name in names],
        "sources": sources,
    }
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(report))
    module.evidence(
        {"qualification": str(path), "qualification_sha256": module.digest(path)}, kind="lazy"
    )
    report.update(change)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        module.evidence(
            {"qualification": str(path), "qualification_sha256": module.digest(path)}, kind="lazy"
        )


def test_tampered_report_cannot_reuse_its_digest(modules, tmp_path):
    module = modules[1]
    path = tmp_path / "result.json"
    path.write_text("{}")
    entry = {"qualification": str(path), "qualification_sha256": module.digest(path)}
    path.write_text('{"status":"SAMPLE_CHECKED"}')
    with pytest.raises(ValueError, match="qualification changed"):
        module.evidence(entry, kind="lazy")


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("ours,upstream", [(None, None), ("0", "0"), ("1", "0"), ("0", "1")])
def test_inherited_flags_cannot_change_the_bundled_layout(modules, lazy, ours, upstream):
    env = {"QWEN_STOCK_GDN_LAZY": ours, "RADIANCE_GDN_LAZY": upstream}
    manifest = {"lazy_gdn": {"state_abi": "stock-fp32-lazy-v1"}} if lazy else {}
    if (ours == "1") == lazy and upstream != "1":
        modules[1].validate_runtime_layout(manifest, env)
    else:
        with pytest.raises(ValueError, match="state-layout flags"):
            modules[1].validate_runtime_layout(manifest, env)


@pytest.fixture
def bundle_builder(modules, monkeypatch, tmp_path):
    from qwen_r9700_lab.diagnostic_contract import seal, write_private

    module = importlib.import_module("prepare_tp1_lazy_backports")
    source = tmp_path / "parent-runtime"
    source.mkdir()
    (source / "parent.py").write_text("state_slots = 9\n")
    parent = tmp_path / "parent.json"
    write_private(
        parent,
        seal(
            {
                "sources": {"parent.py": modules[1].digest(source / "parent.py")},
                "gemm_dispatch": {"test_parent": True},
            }
        ),
    )
    fp8 = tmp_path / "fp8.json"
    fp8.write_text("{}")
    args = module.argument_parser().parse_args(
        [
            "--performance",
            str(parent),
            "--runtime-sources",
            str(source),
            "--build",
            str(tmp_path / "build"),
            "--fp8",
            str(fp8),
            "--output",
            str(tmp_path / "bundle"),
        ]
    )
    checked = []
    monkeypatch.setattr(module, "evidence", lambda entry, *, kind: checked.append(kind))
    return module, args, checked


def test_default_bundle_keeps_existing_layout_without_vllm_overlay(bundle_builder, monkeypatch):
    from qwen_r9700_lab.diagnostic_contract import authenticate

    module, args, checked = bundle_builder

    def forbidden_overlay(*args):
        pytest.fail("existing layout must not patch the allocator or metadata")

    monkeypatch.setattr(module, "overlay", forbidden_overlay)
    assert not args.enable_lazy_gdn and args.lazy is None and args.vllm is None
    module.prepare(args)
    launch = json.loads((args.output / "launch.json").read_text())
    manifest = json.loads((args.output / "performance.json").read_text())
    authenticate(manifest)
    assert checked == ["fp8"]
    assert launch["cache_abi"] == "unchanged"
    assert launch["environment"]["QWEN_STOCK_GDN_LAZY"] == "0"
    assert launch["environment"]["RADIANCE_GDN_LAZY"] == "0"
    assert launch["pythonpath_prepend"] == [str(args.output / "runtime")]
    assert not (args.output / "python").exists()
    assert not (args.output / "runtime/stock_gdn_lazy_runtime.py").exists()
    assert (args.output / "runtime/parent.py").read_text() == "state_slots = 9\n"
    assert "lazy_gdn" not in manifest
    assert manifest["gemm_dispatch"] == {"test_parent": True}
    assert manifest["tp1_fp8"]["silu_enabled"] is False


def test_old_lazy_arguments_cannot_silently_enable_the_new_layout(bundle_builder):
    module, args, _ = bundle_builder
    args.lazy = args.output / "old-lazy-receipt.json"
    with pytest.raises(ValueError, match="explicit --enable-lazy-gdn"):
        module.prepare(args)
    assert not args.output.exists()


def test_lazy_parent_cannot_be_mislabeled_as_existing_layout(bundle_builder):
    from qwen_r9700_lab.diagnostic_contract import seal, write_private

    module, args, _ = bundle_builder
    parent = json.loads(args.performance.read_text())
    parent.pop("sha256")
    parent["lazy_gdn"] = {"state_abi": "stock-fp32-lazy-v1"}
    args.performance = args.performance.with_name("lazy-parent.json")
    write_private(args.performance, seal(parent))
    with pytest.raises(ValueError, match="original GEMM bundle"):
        module.prepare(args)
    assert not args.output.exists()

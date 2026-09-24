"""Independent finite-format and gate-stability regressions; no GPU access."""

import hashlib
import importlib
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def precision(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiments/radiance-public"))
    return importlib.import_module("mxfp4_fold_precision")


def test_reference_exponent_changes_only_rows_that_need_it(precision):
    scales = torch.tensor([[120, 120, 110], [127, 129, 124]], dtype=torch.uint8)
    assert precision.make_row_ref(scales).tolist() == [127, 128, 118]
    # The shifted row permits all E2M1 codes without quantizing any again.
    d = precision.make_row_ref(scales).short()[None] - scales.short()
    assert d.min() >= -6 and d.max() <= 8


@pytest.mark.parametrize("gap", range(15))
def test_all_admitted_exponent_spans_fit_the_exact_window(precision, gap):
    scale = torch.tensor([[110] * 3, [110 + gap] * 3], dtype=torch.uint8)
    ref = precision.make_row_ref(scale)
    assert ref.numel() == 3
    assert ((ref.short() - scale.short()) >= -6).all()
    assert ((ref.short() - scale.short()) <= 8).all()
    if gap <= 8:
        assert torch.equal(ref, scale.amax(0))


def test_wider_scales_fall_back_instead_of_losing_coefficients(precision):
    for gap in (15, 32, 254):
        scale = torch.tensor([[0] * 3, [gap] * 3], dtype=torch.uint8)
        assert precision.make_row_ref(scale).tolist() == [0, 0]
    with pytest.raises(ValueError, match="non-finite"):
        precision.make_row_ref(torch.tensor([[255] * 3], dtype=torch.uint8))
    assert precision.make_row_ref(torch.zeros((4, 3), dtype=torch.uint8)).tolist() == [
        1,
        1,
        1,
    ]


def test_generated_native_lookup_preserves_all_signed_codes_in_its_domain(precision):
    hip, wrapper = precision.patched_sources(
        (ROOT / "radiance_mxfp4_fp8.hip").read_text(),
        (ROOT / "radiance_mxfp4.py").read_text(),
    )
    compile(wrapper, "candidate_wrapper", "exec")
    table = hip.split("unsigned int kMag[16][2] = {", 1)[1].split("};", 1)[0]
    words = [int(s, 16) for s in re.findall(r"0x([a-f0-9]+)u", table)]
    assert len(words) == 32
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float64)
    for d in range(-6, 9):
        pair = words[2 * (d + 6) : 2 * (d + 7)]
        for sign in (0, 128):
            codes = [(word >> (i * 8) & 255) | sign for word in pair for i in range(4)]
            decoded = (
                torch.tensor(codes, dtype=torch.uint8)
                .view(torch.float8_e4m3fn)
                .double()
            )
            assert torch.equal(decoded, levels * (-1 if sign else 1) * 2 ** (-d))
    # Explicit negative control: the original gap-nine coefficient really loses
    # the .5 code. Changing the expectation to the old table must fail.
    assert (
        float(torch.tensor([0], dtype=torch.uint8).view(torch.float8_e4m3fn)[0])
        != 0.5 * 2**-9
    )


def test_source_changes_cannot_silently_reuse_the_patch(precision):
    with pytest.raises(ValueError, match="source changed"):
        precision.patched_sources(
            (ROOT / "radiance_mxfp4_fp8.hip").read_text() + "\n",
            (ROOT / "radiance_mxfp4.py").read_text(),
        )


@pytest.fixture
def softplus(precision):
    return importlib.import_module("patch_gdn_stable_softplus")


def test_gate_repair_removes_cancellation_and_is_idempotent(softplus, monkeypatch):
    name = "fused_recurrent.py"
    source = "def gate(x, tl):\n    return tl.log(1.0 + tl.exp(x))\n"
    monkeypatch.setitem(
        softplus.PREIMAGES, name, hashlib.sha256(source.encode()).hexdigest()
    )
    patched = softplus.patched_source(name, source)
    assert softplus.patched_source(name, patched) == patched
    tl = SimpleNamespace(
        log=torch.log,
        exp=torch.exp,
        extra=SimpleNamespace(libdevice=SimpleNamespace(log1p=torch.log1p)),
    )
    old, new = {}, {}
    exec(source, old)  # noqa: S102 - fixed synthetic source, no external input
    exec(patched, new)  # noqa: S102 - apply the actual patch to that source
    x = torch.tensor(-18.015625)
    assert old["gate"](x, tl).item() == 0
    expected = math.log1p(math.exp(x.item()))
    assert new["gate"](x, tl).item() == pytest.approx(expected, rel=1e-7)
    with pytest.raises(ValueError, match="preimage"):
        softplus.patched_source(name, patched + "\n# different source\n")


def test_gate_install_validates_all_files_before_writing(
    softplus, monkeypatch, tmp_path
):
    package = tmp_path / softplus.BASE
    package.mkdir(parents=True)
    before = {}
    for name, replacements in softplus.REPLACEMENTS.items():
        source = "\n".join(f"# {old}" for old, _ in replacements) + "\n"
        before[name] = source
        (package / name).write_text(source)
        monkeypatch.setitem(
            softplus.PREIMAGES, name, hashlib.sha256(source.encode()).hexdigest()
        )
    invalid = package / "fused_sigmoid_gating.py"
    invalid.write_text("unknown version\n")
    with pytest.raises(ValueError, match="preimage"):
        softplus.install(tmp_path)
    for name in ("fused_recurrent.py", "fused_gdn_prefill_post_conv.py"):
        assert (package / name).read_text() == before[name]
    invalid.write_text(before[invalid.name])
    receipt = softplus.install(tmp_path)
    assert len(receipt) == 3
    assert softplus.install(tmp_path) == receipt


@pytest.fixture
def admission(precision, tmp_path):
    """Tiny metadata stand-ins test rejection, not numerical qualification."""
    module = importlib.import_module("m1_arithmetic_release")
    report = json.loads(
        (ROOT / "benchmarks/results/eager-m1-repairs-20260924.json").read_text()
    )
    build = report.pop("native_build")
    for variant, metadata in build["variants"].items():
        folder = tmp_path / variant
        folder.mkdir()
        for name, key in (
            ("radiance_mxfp4_fp8.so", "binary_sha256"),
            ("radiance_mxfp4_fp8.hip", "source_sha256"),
            ("radiance_mxfp4.py", "python_sha256"),
        ):
            (folder / name).write_text(variant + name)
            metadata[key] = module.digest(folder / name)
    (tmp_path / "build.json").write_text(json.dumps(build))
    report["checks"]["gemm"]["build_sha256"] = module.digest(tmp_path / "build.json")
    evidence = tmp_path / "result.json"
    evidence.write_text(json.dumps(report))
    entry = {
        "build": str(tmp_path),
        "build_sha256": module.digest(tmp_path / "build.json"),
        "qualification": str(evidence),
        "qualification_sha256": module.digest(evidence),
    }
    return module, entry, report


@pytest.mark.parametrize(
    "fault", [None, "binary", "source", "witness", "coverage", "coefficient", "build"]
)
def test_m1_admission_rejects_unqualified_or_incomplete_repairs(admission, fault):
    module, entry, report = admission
    if fault == "binary":
        (Path(entry["build"]) / "candidate/radiance_mxfp4_fp8.so").write_bytes(
            b"old kernel"
        )
    elif fault == "source":
        report["source_sha256"]["stock_gdn_scan_kernel.py"] = "old scan"
    elif fault == "witness":
        report["checks"]["gdn"]["cancellation_witness"]["fixed"] = 1
    elif fault == "coverage":
        report["checks"]["gdn"]["prefill_checks"].pop()
    elif fault == "coefficient":
        report["checks"]["gemm"]["checks"][0]["candidate"]["different"] = 1
    elif fault == "build":
        report["checks"]["gemm"]["build_sha256"] = "another tested binary"
    path = Path(entry["qualification"])
    path.write_text(json.dumps(report))
    entry["qualification_sha256"] = module.digest(path)
    if fault:
        with pytest.raises(ValueError):
            module.validate(entry)
    else:
        module.validate(entry)


def test_dispatch_keeps_precise_kernel_and_rejects_old_wrapper(admission, monkeypatch):
    gate, precision, _ = admission
    build, _ = gate.validate(precision)
    dispatch = importlib.import_module("mxfp4_dispatch")
    monkeypatch.setattr(dispatch, "validate", lambda _: build)
    root = Path(precision["build"])
    calls = []
    native = SimpleNamespace(
        __file__=root / "candidate/radiance_mxfp4_fp8.so",
        set_decode_scratch=lambda *args: calls.append(args),
    )
    scratch = SimpleNamespace(device="fake", numel=lambda: 17, data_ptr=lambda: 4)
    kernel = SimpleNamespace(
        __file__=root / "candidate/radiance_mxfp4.py",
        _ext=native,
        DECODE_MAX_M=64,
        WPERM=True,
        _decode_scratch_ready=[True],
        _decode_scratch=[scratch, scratch],
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_current_stream_capturing=lambda: False),
        empty=lambda *a, **kw: scratch,
        zeros=lambda *a, **kw: scratch,
        float32="float32",
        int32="int32",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "radiance_mxfp4", kernel)
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed",
        SimpleNamespace(get_tensor_model_parallel_world_size=lambda: 1),
    )
    entry = {"m1_arithmetic": precision}
    receipt = dispatch.install(entry)
    assert kernel._ext is native and len(calls) == 1
    assert receipt["binary_sha256"] == build["variants"]["candidate"]["binary_sha256"]
    kernel.__file__ = root / "control/radiance_mxfp4.py"
    with pytest.raises(Exception, match="wrapper differs"):
        dispatch.install(entry)
    assert len(calls) == 1

"""Admit the independent M1 repairs using their actual operator evidence.

This is an operator qualification gate, not full-model or universal proof.
Installation runs before weight loading: new reference exponents and the new
fold table must become active together. Existing snapshots need a new ABI.
"""

import json
import shutil
from pathlib import Path

from mxfp4_fold_precision import digest
from patch_gdn_stable_softplus import BASE, PREIMAGES, patched_source


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(entry):
    root = Path(entry["build"])
    require(digest(root / "build.json") == entry["build_sha256"], "M1 build changed")
    build = json.loads((root / "build.json").read_text())
    require(
        build.get("schema") == "coherence-mxfp4-fold-precision-v1", "unknown M1 build"
    )
    require(
        build["patch_sha256"]
        == digest(Path(__file__).with_name("mxfp4_fold_precision.py")),
        "M1 fold generator changed",
    )
    require(
        set(build["variants"]) == {"control", "candidate"}, "M1 build is incomplete"
    )
    for name, metadata in build["variants"].items():
        for file, key in (
            ("radiance_mxfp4_fp8.so", "binary_sha256"),
            ("radiance_mxfp4_fp8.hip", "source_sha256"),
            ("radiance_mxfp4.py", "python_sha256"),
        ):
            require(digest(root / name / file) == metadata[key], "M1 artifact changed")
    path = Path(entry["qualification"])
    require(digest(path) == entry["qualification_sha256"], "M1 qualification changed")
    report = json.loads(path.read_text())
    require(
        report.get("status") == "OPERATOR_SAMPLE_CHECKED_NOT_FULL_MODEL_QUALIFIED",
        "M1 operator qualification failed",
    )
    require(
        report["probe_sha256"]
        == digest(Path(__file__).with_name("probe_m1_arithmetic_repairs.py")),
        "M1 probe changed",
    )
    for name in (
        "mxfp4_fold_precision.py",
        "patch_gdn_stable_softplus.py",
        "stock_gdn_scan_kernel.py",
    ):
        require(
            report["source_sha256"].get(name) == digest(Path(__file__).with_name(name)),
            "M1 tested source changed",
        )
    gemm, gdn = report["checks"]["gemm"], report["checks"]["gdn"]
    require(gemm["build_sha256"] == entry["build_sha256"], "M1 tested another binary")
    cases = {case["case"]: case for case in gemm["checks"]}
    for name, rows in (("one_hot", 800), ("normal320", 320)):
        case = cases.get(name, {})
        require(
            case.get("rows") == rows and case["control"]["different"] > 0,
            "M1 coefficient negative control missing",
        )
        require(
            case["candidate"]["different"] == 0
            and case["unaffected_vs_control"]["different"] == 0,
            "M1 coefficient comparison failed",
        )
    for name in ("M8_vs_M1", "M32_vs_M1", "M65_vs_M1", "tiled_prefill_vs_M1"):
        require(
            cases["normal320"][name]["different"] == 0, "M1 width comparison failed"
        )
    for section, names in (
        ("fallback", ("M1", "M8", "M65", "M128", "tiled")),
        ("fold_window", ("0", "8", "9", "14")),
    ):
        require(
            all(gemm[section][name]["different"] == 0 for name in names),
            "M1 layout/exponent comparison failed",
        )
    witness = gdn["cancellation_witness"]
    require(
        gdn["negative_control_detected"]
        and witness["old"] == 1
        and witness["fixed"] == witness["expected"] < 1,
        "M1 cancellation control failed",
    )
    require(gdn["matched_state_and_output_rows"] >= 320, "M1 state coverage incomplete")
    checks = {(c["tokens"], c["tile"]): c["exact"] for c in gdn["prefill_checks"]}
    require(
        all(
            checks.get((rows, tile)) is True
            for rows in (1, 8, 64, 320, 1000, 1648, 2048)
            for tile in (8, 16, 32, "selected")
        ),
        "M1 prefill coverage incomplete",
    )
    return build, report


def install(entry, package):
    """Validate all inputs before replacing any package file; no GPU use."""
    build, report = validate(entry)
    package = Path(package)
    changes = {
        package / BASE / name: patched_source(name, (package / BASE / name).read_text())
        for name in PREIMAGES
    }
    require(
        digest(Path(__file__).with_name("stock_gdn_scan_kernel.py"))
        == report["checks"]["gdn"]["scan_sha256"],
        "M8 scan differs from checked M1",
    )
    import hashlib

    require(
        hashlib.sha256(
            changes[package / BASE / "fused_recurrent.py"].encode()
        ).hexdigest()
        == report["checks"]["gdn"]["native_source_sha256"],
        "M1 native source differs",
    )
    for path, source in changes.items():
        path.write_text(source)
    candidate = Path(entry["build"]) / "candidate"
    for name in ("radiance_mxfp4.py", "radiance_mxfp4_fp8.so"):
        shutil.copy2(candidate / name, package / name)
    shutil.copy2(
        Path(__file__).with_name("mxfp4_fold_precision.py"),
        package / "mxfp4_fold_precision.py",
    )
    return {
        "binary_sha256": build["variants"]["candidate"]["binary_sha256"],
        "scope": report["status"],
    }

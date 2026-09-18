"""Admission for the independently checked TP1 FP8 and lazy GDN backports."""

import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_runtime_layout(manifest, environ=None):
    """An inherited lazy flag must never alter an existing-layout bundle."""
    if environ is None:
        import os

        environ = os.environ
    expected = bool(manifest.get("lazy_gdn"))
    if (environ.get("QWEN_STOCK_GDN_LAZY") == "1") != expected or environ.get(
        "RADIANCE_GDN_LAZY"
    ) == "1":
        raise ValueError("GDN state-layout flags do not match the qualified bundle")


def evidence(entry, *, kind):
    path = Path(entry["qualification"])
    if digest(path) != entry["qualification_sha256"]:
        raise ValueError(f"{kind} qualification changed")
    report = json.loads(path.read_text())
    if (
        report["status"] != "SAMPLE_CHECKED"
        or not report["checks"]
        or not all(
            x["equal"] is True and x["bytes"] > 0 and x["unequal_bytes"] == 0
            for x in report["checks"]
        )
    ):
        raise ValueError(f"{kind} has no successful exact comparison")
    required_sources = {
        "lazy": {"stock_gdn_lazy_kernel.py", "stock_gdn_scan_kernel.py", "probe_stock_gdn_lazy.py"},
        "fp8": {"stock_fp8_epilogue.py", "stock_fp8_epilogue.hip", "probe_stock_fp8_epilogue.py"},
    }
    if kind not in required_sources or set(report.get("sources", {})) != required_sources[kind]:
        raise ValueError(f"{kind} evidence has incomplete source binding")
    for name, expected in report["sources"].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"{kind} qualified source changed: {name}")
    if kind == "lazy":
        if (
            report.get("rows") != 320
            or report.get("alignment_cases") != 24
            or report.get("width_pairs") != 64
            or not report.get("canonical_snapshot_roundtrip")
            or not report.get("negative_controls")
        ):
            raise ValueError("lazy state acceptance/migration qualification is incomplete")
        names = {c["name"] for c in report["checks"]}
        expected = {f"step-{step}-prefix-{width}" for step in range(40) for width in range(1, 9)}
        expected |= {
            f"width-{current}-accepted-{previous}-state"
            for current in range(1, 9)
            for previous in range(1, 9)
        }
        expected |= {
            f"align-{mode}-inplace-{same}-extra-{extra}"
            for mode, same in ((0, False), (1, False), (1, True))
            for extra in range(8)
        }
        if not expected <= names:
            raise ValueError("lazy state comparison cases are missing")
    elif kind == "fp8":
        build = json.loads((Path(entry["build"]) / "build.json").read_text())
        if (
            report["build"] != build
            or not report.get("negative_control")
            or not report.get("silu_reference_compiled")
        ):
            raise ValueError("FP8 build or compiled reference qualification differs")
        if digest(Path(entry["build"]) / "candidate.so") != build["binary_sha256"]:
            raise ValueError("FP8 binary changed")
        if build["source_sha256"] != report["sources"]["stock_fp8_epilogue.hip"]:
            raise ValueError("FP8 build and checked source differ")
        expected = {
            f"norm-m{m}-r{residual}-row{row}/q"
            for m in (1, 8)
            for residual in (0, 1)
            for row in range(0, 320, m)
        }
        if not expected <= {c["name"] for c in report["checks"]}:
            raise ValueError("FP8 comparison cases are missing")
        if entry.get("prefill_enabled"):
            if report.get("prefill_sites") != 128 or report.get("prefill_rows_per_site") != 1000:
                raise ValueError("prefill FP8 requires all 128 sites and 1000 rows")
            expected = {
                f"prefill-site{site}-m1000-r{residual}/{part}"
                for site in range(128)
                for residual in (0, 1)
                for part in (("q", "scale", "carry") if residual else ("q", "scale"))
            }
            if not expected <= {c["name"] for c in report["checks"]}:
                raise ValueError("prefill FP8 comparison cases are missing")
    else:
        raise ValueError(kind)
    return report


def validate_overlay(entry):
    import vllm

    root = Path(vllm.__file__).parent
    for name, expected in entry["overlay"]["postimages"].items():
        if digest(root / name) != expected:
            raise ValueError(f"lazy state layout patch is missing or changed: {name}")


def install_lazy(entry, repairs, hooks, convolution):
    evidence(entry, kind="lazy")
    validate_overlay(entry)
    from stock_gdn_lazy_runtime import install

    return install(repairs, hooks, convolution)


def install_fp8(entry, model, hooks):
    evidence(entry, kind="fp8")
    from stock_fp8_stream import install

    return install(model, hooks, entry)

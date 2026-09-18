"""Source-bound admission for the byte-layout prefill optimization."""

from pathlib import Path

from pi_prefill_admission import checked_report, digest

SHAPES = frozenset({(34816, 5120), (5120, 17408), (16384, 5120), (5120, 6144)})
ROWS = (1000, 1648, 2048)


def evidence(entry):
    report = checked_report(entry)
    if (
        report.get("status") != "SAMPLE_CHECKED"
        or report.get("negative_control_detected") is not True
        or report.get("pack_sha256") != entry["sources"].get("prefill_activation_tiles.py")
        or report.get("probe_sha256") != entry["sources"].get("probe_prefill_activation_tiles.py")
        or report.get("binary_sha256") != entry.get("binary_sha256")
    ):
        raise ValueError("activation-tile source or native qualification differs")
    actual = {(c["M"], c["N"], c["K"]): c for c in report["cases"]}
    for m in ROWS:
        for n, k in SHAPES:
            case = actual.get((m, n, k), {})
            if not (
                case.get("layout_bytes_equal") is True
                and case.get("unequal_elements") == 0
                and case.get("equal_rows") == m
                and case.get("canaries") is True
                and case.get("finite") is True
                and case.get("adapter_exact") is True
            ):
                raise ValueError("activation-tile exact native/adapter coverage incomplete")
    return report


def install(entry):
    import radiance_mxfp4 as kernel
    import torch
    from mxfp4_dispatch import PYTHON_SHA256
    from prefill_activation_tiles import install_consumer

    evidence(entry)
    if digest(kernel.__file__) != PYTHON_SHA256 or kernel.A_TILED_MIN_M != 0:
        raise ValueError("activation consumer wrapper or layout contract changed")
    if digest(Path(kernel._ext.__file__)) != entry["binary_sha256"]:
        raise ValueError("activation consumer binary differs from qualified GEMM")
    if torch.cuda.is_current_stream_capturing():
        raise ValueError("activation consumer must precede graph capture")
    return install_consumer(kernel, SHAPES, ROWS)

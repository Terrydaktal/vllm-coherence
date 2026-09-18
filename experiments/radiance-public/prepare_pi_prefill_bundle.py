"""Extend a frozen corrected bundle only after exact operator qualification."""

import argparse
import json
import shutil
from pathlib import Path

from pi_prefill_admission import digest, gdn_evidence, scan_evidence
from tp1_lazy_backports import evidence

from qwen_r9700_lab.diagnostic_contract import authenticate, seal, write_private

SOURCES = (
    "optimized_d7_performance.py",
    "stock_fp8_stream.py",
    "stock_fp8_epilogue.py",
    "stock_fp8_epilogue.hip",
    "probe_stock_fp8_epilogue.py",
    "build_stock_fp8_epilogue.py",
    "tp1_lazy_backports.py",
    "stock_gdn_norm_quant.py",
    "probe_stock_gdn_norm_quant.py",
    "pi_prefill_admission.py",
    "optimized_prefill_scan.py",
    "probe_prefill_tiles.py",
    "prefill_activation_tiles.py",
    "prefill_tiles_admission.py",
    "probe_prefill_activation_tiles.py",
)


def prepare(base, output, job, activation_tiles=False):
    parent = json.loads((base / "performance.json").read_text())
    authenticate(parent)
    for name, expected in parent["sources"].items():
        if digest(base / "runtime" / name) != expected:
            raise ValueError("base performance source changed")
    fp8 = {
        "build": str(job / "prefill-fp8-build"),
        "qualification": str(job / "evidence/fp8-prefill-qualified.json"),
        "qualification_sha256": digest(job / "evidence/fp8-prefill-qualified.json"),
        "silu_enabled": False,
        "prefill_enabled": True,
    }
    gdn = {
        "qualification": str(job / "evidence/gdn-final-1000.json"),
        "qualification_sha256": digest(job / "evidence/gdn-final-1000.json"),
        "sources": {
            n: digest(Path(__file__).with_name(n))
            for n in ("stock_gdn_norm_quant.py", "probe_stock_gdn_norm_quant.py")
        },
    }
    scan = {
        "qualification": str(job / "evidence/scan-final.json"),
        "qualification_sha256": digest(job / "evidence/scan-final.json"),
        "sources": {
            n: digest(Path(__file__).with_name(n))
            for n in ("optimized_prefill_scan.py", "probe_prefill_tiles.py")
        },
    }
    evidence(fp8, kind="fp8")
    gdn_evidence(gdn)
    scan_evidence(scan)
    output.mkdir(mode=0o700)
    shutil.copytree(base / "runtime", output / "runtime")
    for name in SOURCES:
        shutil.copy2(Path(__file__).with_name(name), output / "runtime" / name)
        parent["sources"][name] = digest(output / "runtime" / name)
    parent["prefill_parent"] = parent.pop("sha256")
    parent.update(tp1_fp8=fp8, gdn_norm_quant=gdn, prefill_scan=scan)
    if activation_tiles:
        from prefill_tiles_admission import evidence as tile_evidence

        report = job / "evidence/activation-tiles-v2.json"
        entry = {
            "qualification": str(report),
            "qualification_sha256": digest(report),
            "binary_sha256": json.loads(report.read_text())["binary_sha256"],
            "sources": {
                name: digest(Path(__file__).with_name(name))
                for name in ("prefill_activation_tiles.py", "probe_prefill_activation_tiles.py")
            },
        }
        tile_evidence(entry)
        parent["activation_tiles"] = entry
    write_private(output / "performance.json", seal(parent))
    launch = json.loads((base / "launch.json").read_text())
    launch["pythonpath_prepend"] = [str(output / "runtime")]
    launch["environment"]["QWEN_OPTIMIZED_PERFORMANCE"] = str(output / "performance.json")
    write_private(output / "launch.json", launch)
    print(str(output / "performance.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "output", "job"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--activation-tiles", action="store_true")
    args = parser.parse_args()
    prepare(args.base, args.output, args.job, args.activation_tiles)

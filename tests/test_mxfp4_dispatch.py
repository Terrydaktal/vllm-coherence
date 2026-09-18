import importlib
import json
from pathlib import Path

import pytest

from qwen_r9700_lab.diagnostic_contract import DiagnosticError


@pytest.fixture
def adapter(monkeypatch):
    source = Path(__file__).parents[1] / "experiments/radiance-public"
    monkeypatch.syspath_prepend(str(source))
    return importlib.import_module("mxfp4_dispatch")


@pytest.fixture
def entry(adapter, tmp_path):
    """Small metadata fixtures exercise admission; they do not qualify a kernel."""
    variants = {}
    for name in ("control", "candidate"):
        folder = tmp_path / name
        folder.mkdir()
        variants[name] = {}
        for file, key in (
            ("radiance_mxfp4_fp8.so", "binary_sha256"),
            ("radiance_mxfp4_fp8.hip", "source_sha256"),
            ("radiance_mxfp4.py", "python_sha256"),
        ):
            (folder / file).write_text(name + file)
            variants[name][key] = adapter.digest(folder / file)
    (tmp_path / "build.json").write_text(
        json.dumps({"original_binary_sha256": adapter.ORIGINAL_BINARY_SHA256, "variants": variants})
    )
    entry = {
        "build": str(tmp_path),
        "build_sha256": adapter.digest(tmp_path / "build.json"),
        "qualifications": {},
    }
    for name, count, split in (("automatic", 35, "automatic"), ("split4", 20, 4)):
        path = tmp_path / (name + ".json")
        case = {
            "finite": True,
            "canaries": True,
            "counters_zero": True,
            "candidate_vs_original": {"equal": True},
            "control_vs_original": {"equal": True},
        }
        path.write_text(
            json.dumps(
                {
                    "build": entry["build_sha256"],
                    "status": "SAMPLE_CHECKED",
                    "negative_control_detected": True,
                    "split_k": split,
                    "probe_sha256": adapter.digest(
                        Path(adapter.__file__).with_name("probe_mxfp4_dispatch.py")
                    ),
                    "cases": [case] * count,
                }
            )
        )
        entry["qualifications"][name] = {"path": str(path), "sha256": adapter.digest(path)}
    return entry


def test_changed_kernel_cannot_reuse_qualification(adapter, entry):
    adapter.validate(entry)
    binary = Path(entry["build"]) / "candidate/radiance_mxfp4_fp8.so"
    binary.write_bytes(b"different binary")
    with pytest.raises(DiagnosticError, match="artifact changed"):
        adapter.validate(entry)


@pytest.mark.parametrize(
    "change",
    [
        {"status": "FAILED"},
        {"build": "wrong"},
        {"negative_control_detected": False},
        {"cases": []},
        {"probe_sha256": "wrong"},
        {"split_k": 4},
    ],
)
def test_invalid_receipt_never_activates(adapter, entry, change):
    evidence = entry["qualifications"]["automatic"]
    path = Path(evidence["path"])
    report = json.loads(path.read_text())
    report.update(change)
    path.write_text(json.dumps(report))
    evidence["sha256"] = adapter.digest(path)
    with pytest.raises(DiagnosticError):
        adapter.validate(entry)


def test_mismatch_blocks_activation_even_with_success_status(adapter, entry):
    evidence = entry["qualifications"]["automatic"]
    path = Path(evidence["path"])
    report = json.loads(path.read_text())
    report["cases"][0]["candidate_vs_original"]["equal"] = False
    path.write_text(json.dumps(report))
    evidence["sha256"] = adapter.digest(path)
    with pytest.raises(DiagnosticError, match="mismatch"):
        adapter.validate(entry)


def test_builder_rejects_unknown_source():
    import build_mxfp4_dispatch

    with pytest.raises(ValueError, match="source changed"):
        build_mxfp4_dispatch.patched_sources("new upstream source", "unknown wrapper")

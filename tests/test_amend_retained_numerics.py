"""The diagnostic correction build must preserve the measured model and its source identities."""

import json
import sys
from pathlib import Path

import pytest

from qwen_r9700_lab.diagnostic_contract import authenticate, seal

SOURCE = Path(__file__).resolve().parents[1] / "experiments/radiance-public"
sys.path.insert(0, str(SOURCE))
from amend_retained_numerics import amend, sha
from wide_qk_norm import BINARY_SHA256, KERNEL_SHA256


def test_separate_corrected_build_requires_intact_parent(tmp_path):
    old, new, norm = (tmp_path / n for n in ("original", "amended", "norm"))
    old.mkdir()
    norm.mkdir()
    initializer = old / "fourteen_retained.py"
    initializer.write_text("# synthetic parent\n")
    manifest = seal(
        {"selection": ["existing"], "files": {initializer.name: sha(initializer)}}
    )
    (old / "manifest.json").write_text(json.dumps(manifest))
    (norm / "candidate.so").write_bytes(b"synthetic model kernel")
    (norm / "build.json").write_text(
        json.dumps({"max_norm_rows": 4096, "binary_sha256": sha(norm / "candidate.so")})
    )
    amended = amend(old, new, norm=norm)
    authenticate(amended)
    assert amended["numerical_parent"] == manifest["sha256"]
    assert amended["selection"] == ["existing", "wide_input_norm"]
    assert (
        amended["files"]["fourteen_base_retained.py"]
        == manifest["files"]["fourteen_retained.py"]
    )
    assert json.loads((old / "manifest.json").read_bytes()) == manifest
    initializer.write_text("# changed\n")
    with pytest.raises(ValueError, match="parent source differs"):
        amend(old, tmp_path / "other", norm=norm)
    assert not (tmp_path / "other").exists()


def test_qk_correction_retains_authenticated_operator_evidence(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    initializer = parent / "fourteen_retained.py"
    initializer.write_text("# parent")
    (parent / "manifest.json").write_text(
        json.dumps(seal({"selection": [], "files": {initializer.name: sha(initializer)}}))
    )
    path = tmp_path / "operator.json"
    metadata = {"passed": True, "binary_sha256": BINARY_SHA256, "source_sha256": KERNEL_SHA256}
    path.write_text(json.dumps(seal(metadata)))
    overlay = tmp_path / "measured"
    result = amend(parent, overlay, qk_norm=path)
    assert result["numerical_corrections"] == {"wide_qk_norm": BINARY_SHA256}
    assert result["files"]["qk-norm-qualification.json"] == sha(path)
    path.write_text(json.dumps(seal({**metadata, "passed": False})))
    with pytest.raises(ValueError, match="verified operator run"):
        amend(parent, tmp_path / "rejected", qk_norm=path)

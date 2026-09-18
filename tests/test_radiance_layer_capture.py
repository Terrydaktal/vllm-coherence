from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/radiance-public/capture_radiance_layers.py"
LEGACY = ROOT / "experiments/m8-layer-diagnostic/layer_diagnostic.py"
spec = importlib.util.spec_from_file_location("capture_radiance_layers", SOURCE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def request():
    return {"schema": "qwen-radiance-layer-capture-request-v1", "capture_id": "isolated-test",
            "positions": [1234], "expected_layers": 64, "input_sha256": "a" * 64}


def test_unchanged_legacy_helpers_load_without_installing_old_model_hooks(monkeypatch):
    import sys

    monkeypatch.setenv("QWEN_M8_LAYER_DIAGNOSTIC_POSITIONS", "17,23")
    hooks = list(sys.meta_path)
    helpers = module.load_legacy_helpers(LEGACY)
    assert frozenset({1}) == helpers._SELECTED_POSITIONS
    assert sys.meta_path == hooks
    import os
    assert os.environ["QWEN_M8_LAYER_DIAGNOSTIC_POSITIONS"] == "17,23"
    assert all(callable(getattr(helpers, name)) for name in (
        "_tensor_sha256", "_tensor_layout", "_storage_relation", "_write_tensor_capsule"))


def test_modified_legacy_helpers_cannot_be_used(tmp_path):
    source = tmp_path / "legacy.py"
    source.write_text("raise AssertionError('must never execute')")
    with pytest.raises(ValueError, match="recorded source"):
        module.load_legacy_helpers(source)


def test_capture_requires_private_directory_and_authenticated_identity(tmp_path):
    tmp_path.chmod(0o700)
    assert module.validate_request(request(), tmp_path)["expected_layers"] == 64
    tmp_path.chmod(0o755)
    with pytest.raises(ValueError, match="private RAM"):
        module.validate_request(request(), tmp_path)
    tmp_path.chmod(0o700)
    for key, value in (("capture_id", "../public"), ("input_sha256", ""),
                       ("expected_layers", 0), ("positions", [2, 1])):
        with pytest.raises(ValueError):
            module.validate_request(dict(request(), **{key: value}), tmp_path)


def test_installer_refuses_unrecognized_decoder_before_mutating_it(tmp_path):
    package = tmp_path / "package"
    target = package / module.MODEL_SOURCE
    target.parent.mkdir(parents=True)
    target.write_text("class DifferentDecoder: pass\n")
    with pytest.raises(ValueError, match="qualified capture interface"):
        module.install(package, SOURCE, LEGACY)
    assert target.read_text() == "class DifferentDecoder: pass\n"
    assert not (package / "qwen_radiance_layer_capture.py").exists()

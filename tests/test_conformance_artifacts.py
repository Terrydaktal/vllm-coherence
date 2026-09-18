import hashlib
import sys
from types import SimpleNamespace

import pytest

from qwen_r9700_lab.conformance_artifacts import capture_runtime, compiler_settings
from qwen_r9700_lab.diagnostic_contract import authenticate


def test_artifact_identity_changes_without_gpu_imports_or_secret_environment(tmp_path):
    library = tmp_path / "libr4d.so"
    library.write_bytes(b"original")
    maps = tmp_path / "maps"
    maps.write_text(
        f"1000-2000 r-xp 00000000 00:00 1 {library}\n2000-3000 r--p 00000000 00:00 1 {library}\n"
    )
    modules = set(sys.modules)
    before = capture_runtime(
        maps_path=maps, environ={"VLLM_API_KEY": "SECRET", "RADIANCE_VERIFY_HEAD": "0"}
    )
    authenticate(before)
    assert len(before["mapped_file_bytes"]) == 1
    assert (
        before["mapped_file_bytes"][str(library)]["sha256"]
        == hashlib.sha256(b"original").hexdigest()
    )
    assert "SECRET" not in str(before)
    assert not before["artifact_inventory_complete"]
    assert not before["exact_device_binary_attested"]
    library.write_bytes(b"future optimization")
    after = capture_runtime(maps_path=maps, environ={})
    assert before["sha256"] != after["sha256"]
    library.unlink()
    missing = capture_runtime(maps_path=maps, environ={})
    assert missing["unavailable_files"][str(library)] == "FileNotFoundError"
    assert not ({"torch", "vllm", "triton"} & (set(sys.modules) - modules))


def test_reference_runtime_receipt_does_not_alias_cached_file_identities():
    from qwen_r9700_lab.conformance_artifacts import reference_runtime_identity

    report = reference_runtime_identity()
    expected = report["sha256"]
    for entry in report["files"].values():
        entry["sha256"] = "0" * 64
    assert reference_runtime_identity()["sha256"] == expected


@pytest.mark.parametrize("value", [True, False, None, "1", 1])
def test_compiler_settings_preserves_observed_bool_without_guessing(value):
    before = set(sys.modules)
    observed = compiler_settings(
        {"torch._inductor.config": SimpleNamespace(emulate_precision_casts=value)}
    )
    assert observed == {"emulate_precision_casts": value if type(value) is bool else None}
    assert set(sys.modules) == before


def test_compiler_settings_missing_is_unknown_not_false():
    assert compiler_settings({}) == {"emulate_precision_casts": None}
    assert compiler_settings({"torch._inductor.config": SimpleNamespace()}) == {
        "emulate_precision_casts": None
    }


def test_runtime_binds_precision_environment_and_actual_loaded_setting(tmp_path, monkeypatch):
    maps = tmp_path / "maps"
    maps.write_text("")
    monkeypatch.setitem(
        sys.modules, "torch._inductor.config", SimpleNamespace(emulate_precision_casts=False)
    )
    observed = capture_runtime(
        maps_path=maps,
        environ={"TORCHINDUCTOR_EMULATE_PRECISION_CASTS": "1", "VLLM_API_KEY": "SECRET"},
    )
    authenticate(observed)
    assert observed["flags"]["TORCHINDUCTOR_EMULATE_PRECISION_CASTS"] == "1"
    # Environment intent and an already-loaded compiler can disagree; retain both.
    assert observed["compiler_settings"] == {"emulate_precision_casts": False}
    assert "SECRET" not in str(observed)

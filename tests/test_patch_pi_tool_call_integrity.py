from __future__ import annotations

import hashlib
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "scripts" / "patch-pi-tool-call-integrity"
USAGE_PATCHER = ROOT / "scripts" / "patch-pi-streaming-usage"


def load_api() -> dict[str, object]:
    return runpy.run_path(str(PATCHER), run_name="patch_pi_tool_call_integrity_test")


def configure_fixture(api: dict[str, object], root: Path) -> Path:
    relative = "node_modules/example/runtime.js"
    old = b"repair partial JSON\n"
    new = b"require complete JSON\n"
    patch_type = api["FilePatch"]
    patch = patch_type(
        relative_path=relative,
        preimage_sha256=hashlib.sha256(old).hexdigest(),
        old=old,
        new=new,
    )
    api["run"].__globals__["PATCHES"] = (patch,)
    target = root / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(old)
    target.chmod(0o644)
    return target


def test_apply_is_authenticated_and_idempotent(tmp_path: Path) -> None:
    api = load_api()
    target = configure_fixture(api, tmp_path)

    api["run"](tmp_path, apply=True)
    assert target.read_bytes() == b"require complete JSON\n"
    api["run"](tmp_path, apply=False)
    api["run"](tmp_path, apply=True)


def test_ordered_same_file_patch_chain_is_authenticated_and_idempotent(tmp_path: Path) -> None:
    api = load_api()
    relative = "node_modules/example/runtime.js"
    base = b"first old\nsecond old\n"
    first_new = base.replace(b"first old", b"first new")
    patch_type = api["FilePatch"]
    api["run"].__globals__["PATCHES"] = (
        patch_type(
            relative_path=relative,
            preimage_sha256=hashlib.sha256(base).hexdigest(),
            old=b"first old",
            new=b"first new",
        ),
        patch_type(
            relative_path=relative,
            preimage_sha256=hashlib.sha256(first_new).hexdigest(),
            old=b"second old",
            new=b"second new",
        ),
    )
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(base)
    target.chmod(0o644)

    api["run"](tmp_path, apply=True)
    assert target.read_bytes() == b"first new\nsecond new\n"
    api["run"](tmp_path, apply=False)
    api["run"](tmp_path, apply=True)


def test_unknown_runtime_bytes_fail_closed(tmp_path: Path) -> None:
    api = load_api()
    target = configure_fixture(api, tmp_path)
    target.write_bytes(b"unexpected\n")

    with pytest.raises(api["PatchError"], match="differs from both pinned patch states"):
        api["run"](tmp_path, apply=True)


def test_usage_patcher_compatibility_bytes_match_this_patch_exactly() -> None:
    integrity_api = load_api()
    usage_api = runpy.run_path(str(USAGE_PATCHER), run_name="patch_pi_usage_compatibility_test")
    patch = integrity_api["PATCHES"][0]

    assert usage_api["TOOL_CALL_INTEGRITY_RELATIVE"] == patch.relative_path
    assert usage_api["TOOL_CALL_INTEGRITY_OLD"] == patch.old
    assert usage_api["TOOL_CALL_INTEGRITY_NEW"] == patch.new
    raw_patch = integrity_api["PATCHES"][1]
    assert raw_patch.relative_path == patch.relative_path
    assert usage_api["TOOL_CALL_RAW_IDS_OLD"] == raw_patch.old
    assert usage_api["TOOL_CALL_RAW_IDS_NEW"] == raw_patch.new
    stream_error_patch = integrity_api["PATCHES"][2]
    assert stream_error_patch.relative_path == patch.relative_path
    assert usage_api["TOOL_CALL_STREAM_ERROR_OLD"] == stream_error_patch.old
    assert usage_api["TOOL_CALL_STREAM_ERROR_NEW"] == stream_error_patch.new
    output_patch = integrity_api["PATCHES"][3]
    assert output_patch.relative_path == patch.relative_path
    assert usage_api["RADIANCE_OUTPUT_OLD"] == output_patch.old
    assert usage_api["RADIANCE_OUTPUT_NEW"] == output_patch.new


def test_streaming_provider_error_patch_surfaces_bounded_cause() -> None:
    api = load_api()
    patch = api["PATCHES"][2]

    assert b"chunk.error !== undefined" in patch.new
    assert b"Provider streaming error" in patch.new
    assert b"slice(0, 1024)" in patch.new
    assert patch.new.count(patch.old) == 1


def test_optional_raw_token_transport_accepts_standard_null_field() -> None:
    api = load_api()
    patch = api["PATCHES"][1]

    assert b"if (choice.token_ids != null)" in patch.new
    assert b"if (choice.token_ids !== undefined)" not in patch.new


def test_patch_chain_accepts_only_a_terminal_newline_normalization(tmp_path: Path) -> None:
    api = load_api()
    patch_type = api["FilePatch"]
    relative = "node_modules/@earendil-works/pi-ai/dist/api/openai-completions.js"
    base = b"first old\nsecond old"
    first_new = base.replace(b"first old", b"first new")
    patches = (
        patch_type(
            relative_path=relative,
            preimage_sha256=hashlib.sha256(base).hexdigest(),
            old=b"first old",
            new=b"first new",
        ),
        patch_type(
            relative_path=relative,
            preimage_sha256=hashlib.sha256(first_new).hexdigest(),
            old=b"second old",
            new=b"second new",
        ),
    )
    candidate = b"first new\nsecond new"

    assert api["classify_chain"](candidate, patches) == len(patches)
    assert api["classify_chain"](candidate + b"\n", patches) == len(patches)
    with pytest.raises(api["PatchError"], match="differs from both pinned patch states"):
        api["classify_chain"](candidate + b"\n\n", patches)

    # Upgrading must use the same canonical bytes as verification and retain LF.
    api["run"].__globals__["PATCHES"] = patches
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(first_new + b"\n")
    api["run"](tmp_path, apply=True)
    assert target.read_bytes() == candidate + b"\n"
    api["run"](tmp_path, apply=False)

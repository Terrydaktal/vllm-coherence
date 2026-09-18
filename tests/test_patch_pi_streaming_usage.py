from __future__ import annotations

import hashlib
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "scripts" / "patch-pi-streaming-usage"


def load_api() -> dict[str, object]:
    return runpy.run_path(str(PATCHER), run_name="patch_pi_streaming_usage_test")


def configure_fixture(api: dict[str, object], root: Path) -> Path:
    relative = "node_modules/example/runtime.js"
    old = b"before\n"
    new = b"before\nusage-update\n"
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
    assert target.read_bytes() == b"before\nusage-update\n"
    api["run"](tmp_path, apply=False)
    api["run"](tmp_path, apply=True)


def test_unknown_runtime_bytes_fail_closed(tmp_path: Path) -> None:
    api = load_api()
    target = configure_fixture(api, tmp_path)
    target.write_bytes(b"unexpected\n")

    with pytest.raises(api["PatchError"], match="differs from both pinned patch states"):
        api["run"](tmp_path, apply=True)


def test_upgrade_authenticates_the_previous_usage_patch(tmp_path: Path) -> None:
    api = load_api()
    target = configure_fixture(api, tmp_path)
    from dataclasses import replace

    patch = replace(api["run"].__globals__["PATCHES"][0], previous=b"before\nold-usage\n")
    api["run"].__globals__["PATCHES"] = (patch,)
    target.write_bytes(patch.previous)
    with pytest.raises(api["PatchError"], match="patch is absent"):
        api["run"](tmp_path, apply=False)
    api["run"](tmp_path, apply=True)
    assert target.read_bytes() == patch.new
    api["run"](tmp_path, apply=False)
    target.write_bytes(patch.previous + b"unexpected change\n")
    with pytest.raises(api["PatchError"], match="differs from both pinned patch states"):
        api["run"](tmp_path, apply=True)

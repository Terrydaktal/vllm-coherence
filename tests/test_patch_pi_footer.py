from __future__ import annotations

import hashlib
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "scripts" / "patch-pi-footer"


def load_api() -> dict[str, object]:
    return runpy.run_path(str(PATCHER), run_name="patch_pi_footer_test")


def configure_fixture(api: dict[str, object], root: Path) -> tuple[Path, bytes]:
    relative = "node_modules/example/footer.js"
    context_old = b"context-old\n"
    context_new = b"context-new\n"
    transform_old = b"truncate-footer\n"
    transform_new = b"wrap-footer\ninline-temperature\n"
    base = context_old + transform_old
    legacy = context_new + transform_old
    wrapped = context_new + transform_new
    unified = b"one-continuous-footer\ninline-cache-then-temperature\n"
    patched = context_new + unified
    globals_ = api["run"].__globals__
    globals_.update(
        {
            "RELATIVE_PATH": relative,
            "BASE_SHA256": hashlib.sha256(base).hexdigest(),
            "LEGACY_SHA256": hashlib.sha256(legacy).hexdigest(),
            "WRAPPED_SHA256": hashlib.sha256(wrapped).hexdigest(),
            "PATCHED_SHA256": hashlib.sha256(patched).hexdigest(),
            "CONTEXT_OLD": context_old,
            "CONTEXT_NEW": context_new,
            "REPLACEMENTS": ((transform_old, transform_new),),
            "UNIFIED_REPLACEMENTS": ((transform_new, unified),),
        }
    )
    target = root / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(base)
    target.chmod(0o644)
    return target, patched


def test_footer_patch_upgrades_base_and_legacy_states_idempotently(tmp_path: Path) -> None:
    api = load_api()
    target, expected = configure_fixture(api, tmp_path)

    api["run"](tmp_path, apply=True)
    assert target.read_bytes() == expected

    target.write_bytes(b"context-new\nwrap-footer\ninline-temperature\n")
    api["run"](tmp_path, apply=True)
    assert target.read_bytes() == expected
    api["run"](tmp_path, apply=False)
    api["run"](tmp_path, apply=True)

    target.write_bytes(b"context-new\ntruncate-footer\n")
    api["run"](tmp_path, apply=True)
    assert target.read_bytes() == expected


def test_footer_patch_rejects_unknown_runtime_bytes(tmp_path: Path) -> None:
    api = load_api()
    target, _ = configure_fixture(api, tmp_path)
    target.write_bytes(b"unknown footer\n")

    with pytest.raises(api["PatchError"], match="differs from every pinned patch state"):
        api["run"](tmp_path, apply=True)


def test_pinned_footer_transform_contains_wrapping_and_inline_temperature() -> None:
    api = load_api()
    transformed = b"\n".join(new for _, new in api["REPLACEMENTS"])

    assert b"wrapTextWithAnsi" in transformed
    assert b"wrappedRight" in transformed
    assert b"qwen-gpu-temperature" in transformed
    assert b'.filter(([key]) => key !== "qwen-gpu-temperature")' in transformed

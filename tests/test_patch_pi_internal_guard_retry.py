from __future__ import annotations

import hashlib
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "scripts" / "patch-pi-internal-guard-retry"


def load_api() -> dict[str, object]:
    return runpy.run_path(str(PATCHER), run_name="patch_pi_internal_guard_retry_test")


def configure_fixture(api: dict[str, object], root: Path) -> Path:
    relative = "node_modules/example/agent-session.js"
    old = b"emit and persist retry sentinel\n"
    new = b"hide authenticated internal retry sentinel\n"
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
    assert target.read_bytes() == b"hide authenticated internal retry sentinel\n"
    api["run"](tmp_path, apply=False)
    api["run"](tmp_path, apply=True)


def test_unknown_runtime_bytes_fail_closed(tmp_path: Path) -> None:
    api = load_api()
    target = configure_fixture(api, tmp_path)
    target.write_bytes(b"unexpected\n")

    with pytest.raises(
        api["PatchError"], match="differs from both pinned patch states"
    ):
        api["run"](tmp_path, apply=True)


def test_real_patch_requires_retry_policy_and_retryable_classification() -> None:
    api = load_api()
    patch = api["PATCHES"][0]

    assert (
        b'event.message?.[Symbol.for("qwen-r9700:internal-guard-retry:v1")]'
        in patch.new
    )
    assert b"retrySettings.enabled" in patch.new
    assert b"this._retryAttempt < retrySettings.maxRetries" in patch.new
    assert b"this._isRetryableError(event.message)" in patch.new
    assert b"if (!internalGuardRetry)" in patch.new
    assert b"if (!internalGuardRetry && event.message.role" in patch.new


def test_later_compaction_patch_is_authenticated_without_reverting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = load_api()
    target = configure_fixture(api, tmp_path)
    api["run"](tmp_path, apply=True)
    guard_bytes = target.read_bytes()
    following = runpy.run_path(
        str(PATCHER.with_name("patch-pi-precontinuation-compaction"))
    )
    later = following["FilePatch"](
        relative_path="node_modules/example/agent-session.js",
        preimage_sha256=hashlib.sha256(guard_bytes).hexdigest(),
        old=guard_bytes,
        new=guard_bytes.replace(b"sentinel", b"marker")
        + b"compact before continuing the tool loop\n",
    )
    following["PATCHES"] = (later,)
    monkeypatch.setattr(api["runpy"], "run_path", lambda _path: following)
    target.write_bytes(later.new)
    api["run"](tmp_path, apply=False)
    api["run"](tmp_path, apply=True)
    assert target.read_bytes() == later.new
    target.write_bytes(later.new + b"unexpected edit\n")
    with pytest.raises(
        api["PatchError"], match="differs from both pinned patch states"
    ):
        api["run"](tmp_path, apply=True)


@pytest.mark.parametrize(
    "applied", [(False, False), (True, False), (False, True), (True, True)]
)
def test_later_compaction_group_authenticates_complete_and_partial_upgrades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, applied: tuple[bool, bool]
) -> None:
    api = load_api()
    relative = "node_modules/example/agent-session.js"
    base = b"emit retry sentinel\ncompaction boundary old\ncontext usage old\n"
    guard = api["FilePatch"](
        relative_path=relative,
        preimage_sha256=hashlib.sha256(base).hexdigest(),
        old=b"emit retry sentinel",
        new=b"hide retry sentinel",
    )
    api["run"].__globals__["PATCHES"] = (guard,)
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(base)
    api["run"](tmp_path, apply=True)
    guard_bytes = target.read_bytes()

    following = runpy.run_path(
        str(PATCHER.with_name("patch-pi-precontinuation-compaction"))
    )
    later_group = tuple(
        following["FilePatch"](
            relative_path=relative,
            preimage_sha256=hashlib.sha256(guard_bytes).hexdigest(),
            old=old,
            new=new,
        )
        for old, new in (
            (b"compaction boundary old", b"compaction boundary new"),
            (b"context usage old", b"context usage new"),
        )
    )
    following["PATCHES"] = later_group
    monkeypatch.setattr(api["runpy"], "run_path", lambda _path: following)
    installed = guard_bytes
    for enabled, later in zip(applied, later_group, strict=True):
        if enabled:
            installed = installed.replace(later.old, later.new)
    target.write_bytes(installed)

    # Checking or reapplying the earlier guard must preserve all later patches,
    # including the currently installed subset during an in-place upgrade.
    api["run"](tmp_path, apply=False)
    api["run"](tmp_path, apply=True)
    assert target.read_bytes() == installed

    target.write_bytes(installed + b"unrelated source edit\n")
    corrupted = target.read_bytes()
    for apply in (False, True):
        with pytest.raises(
            api["PatchError"], match="differs from both pinned patch states"
        ):
            api["run"](tmp_path, apply=apply)
        assert target.read_bytes() == corrupted, (
            "authentication failure must not rewrite the runtime"
        )

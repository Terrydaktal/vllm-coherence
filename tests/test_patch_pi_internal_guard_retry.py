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

    with pytest.raises(api["PatchError"], match="differs from both pinned patch states"):
        api["run"](tmp_path, apply=True)


def test_real_patch_requires_retry_policy_and_retryable_classification() -> None:
    api = load_api()
    patch = api["PATCHES"][0]

    assert b'event.message?.[Symbol.for("qwen-r9700:internal-guard-retry:v1")]' in patch.new
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
    later = api["FilePatch"](
        relative_path="node_modules/example/agent-session.js",
        preimage_sha256=hashlib.sha256(guard_bytes).hexdigest(),
        old=guard_bytes,
        new=guard_bytes + b"compact before continuing the tool loop\n",
    )
    monkeypatch.setattr(api["runpy"], "run_path", lambda _path: {"PATCHES": (later,)})
    target.write_bytes(later.new)
    api["run"](tmp_path, apply=False)
    api["run"](tmp_path, apply=True)
    assert target.read_bytes() == later.new
    target.write_bytes(later.new + b"unexpected edit\n")
    with pytest.raises(api["PatchError"], match="differs from both pinned patch states"):
        api["run"](tmp_path, apply=True)

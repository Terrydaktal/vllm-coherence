from __future__ import annotations

import hashlib
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "scripts" / "patch-pi-precontinuation-compaction"


def load_api() -> dict[str, object]:
    return runpy.run_path(str(PATCHER), run_name="patch_pi_precontinuation_compaction_test")


def configure_fixture(api: dict[str, object], root: Path) -> Path:
    relative = "node_modules/example/agent-session.js"
    old = b"continue tool loop without checking context\n"
    new = b"compact before continuing the tool loop\n"
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
    assert target.read_bytes() == b"compact before continuing the tool loop\n"
    api["run"](tmp_path, apply=False)
    api["run"](tmp_path, apply=True)


def test_unknown_runtime_bytes_fail_closed(tmp_path: Path) -> None:
    api = load_api()
    target = configure_fixture(api, tmp_path)
    target.write_bytes(b"unexpected\n")

    with pytest.raises(api["PatchError"], match="differs from both pinned patch states"):
        api["run"](tmp_path, apply=True)


def test_partially_patched_runtime_is_authenticated_before_upgrade(tmp_path: Path) -> None:
    api = load_api()
    relative = "node_modules/example/interactive-mode.js"
    base = b"spinner old\nsummary old\n"
    patch_type = api["FilePatch"]
    patches = (
        patch_type(relative, hashlib.sha256(base).hexdigest(), b"spinner old", b"spinner new"),
        patch_type(relative, hashlib.sha256(base).hexdigest(), b"summary old", b"summary new"),
    )
    api["run"].__globals__["PATCHES"] = patches
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b"spinner new\nsummary old\n")

    api["run"](tmp_path, apply=True)
    assert target.read_bytes() == b"spinner new\nsummary new\n"
    api["run"](tmp_path, apply=False)


def test_real_patch_compacts_at_the_inter_tool_boundary_and_fails_closed() -> None:
    api = load_api()
    patch = api["PATCHES"][0]

    assert b'turn.message.stopReason === "toolUse"' in patch.new
    assert b"estimateMessagesTokens(turn.toolResults)" in patch.new
    assert b"shouldCompact(projectedTokens, contextWindow, settings)" in patch.new
    assert b'await this._runAutoCompaction("threshold", false)' in patch.new
    assert b"afterCompactionId === beforeCompactionId" in patch.new
    assert b"did not commit" in patch.new
    assert b"this.sessionManager.buildSessionContext()" in patch.new
    combined = b"\n".join(item.new for item in api["PATCHES"])
    assert b"qwen-radiance-compaction-timing-v1" in combined
    assert b"component.setElapsedMs(timing.elapsedMs)" in combined
    assert b"Compacted from ${tokenStr} tokens${timing}" in combined
    assert b"event.result.details?.elapsedMs" in combined
    assert b"this.activeStatusIndicator === undefined" in combined
    assert b"this.workingVisible && this.session.isStreaming" in combined
    assert b"this.showStatusIndicator(new WorkingStatusIndicator" in combined

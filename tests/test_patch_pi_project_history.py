from __future__ import annotations

import hashlib
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "scripts" / "patch-pi-project-history"


def load_api() -> dict[str, object]:
    return runpy.run_path(str(PATCHER), run_name="patch_pi_project_history_test")


def configure_fixture(api: dict[str, object], root: Path) -> list[Path]:
    patch_type = api["FilePatch"]
    patches = []
    targets = []
    for index, (old, new) in enumerate(
        (
            (b"model-specific history\n", b"project-local history\n"),
            (b"match a UUID\n", b"match last or a UUID\n"),
            (b"path or ID help\n", b"path, ID, or last help\n"),
        )
    ):
        relative = f"node_modules/example/file-{index}.js"
        patches.append(
            patch_type(
                relative_path=relative,
                preimage_sha256=hashlib.sha256(old).hexdigest(),
                old=old,
                new=new,
            )
        )
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(old)
        target.chmod(0o644)
        targets.append(target)
    api["run"].__globals__["PATCHES"] = tuple(patches)
    return targets


def test_apply_is_authenticated_and_idempotent(tmp_path: Path) -> None:
    api = load_api()
    targets = configure_fixture(api, tmp_path)

    api["run"](tmp_path, apply=True)
    assert [target.read_bytes() for target in targets] == [
        b"project-local history\n",
        b"match last or a UUID\n",
        b"path, ID, or last help\n",
    ]
    api["run"](tmp_path, apply=False)
    api["run"](tmp_path, apply=True)


def test_unknown_runtime_bytes_fail_closed(tmp_path: Path) -> None:
    api = load_api()
    targets = configure_fixture(api, tmp_path)
    targets[1].write_bytes(b"unexpected\n")

    with pytest.raises(api["PatchError"], match="differs from both pinned patch states"):
        api["run"](tmp_path, apply=True)


def test_real_patch_makes_history_model_neutral_and_adds_last() -> None:
    api = load_api()
    session_manager, main, help_patch = api["PATCHES"]

    assert b'return join(resolvePath(cwd), ".pi", "sessions")' in session_manager.new
    assert b"mode: 0o700" in session_manager.new
    assert b'sessionArg === "last"' in main.new
    assert b"localSessions[0]" in main.new
    assert b"--session <path|id|last>" in help_patch.new


def real_session_manager_patches():
    history_api = load_api()
    count_api = runpy.run_path(
        str(ROOT / "scripts" / "patch-pi-precontinuation-compaction"),
        run_name="composed_count_patch_test",
    )
    relative = "node_modules/@earendil-works/pi-coding-agent/dist/core/session-manager.js"
    history = next(patch for patch in history_api["PATCHES"] if patch.relative_path == relative)
    projections = tuple(patch for patch in count_api["PATCHES"] if patch.relative_path == relative)
    installed = Path.home() / ".local/share/qwen-r9700/pi/0.84.2" / relative
    if not installed.is_file():
        pytest.skip("pinned public Pi SDK is not installed")
    baseline = installed.read_bytes()
    for patch in reversed(projections):
        for known in (patch.new, *patch.legacy):
            if known in baseline:
                baseline = baseline.replace(known, patch.old)
                break
    baseline = baseline.replace(history.new, history.old)
    assert hashlib.sha256(baseline).hexdigest() == history.preimage_sha256
    assert all(history.preimage_sha256 in patch.alternate_preimage_sha256 for patch in projections)
    history_api["run"].__globals__["PATCHES"] = (history,)
    count_api["run"].__globals__["PATCHES"] = projections
    return history_api, count_api, history, projections, baseline


@pytest.mark.parametrize("first_patch", ["count", "history"])
@pytest.mark.parametrize(
    "initial_state",
    ["stock", "count_only", "history_only", "both", "tail_only", "count_tail", "history_tail", "all"],
)
def test_real_session_manager_patches_compose_in_both_orders(
    tmp_path: Path, first_patch: str, initial_state: str
) -> None:
    history_api, count_api, history, projections, baseline = real_session_manager_patches()
    count = next(patch for patch in projections if b"historicalTokens" in patch.new)
    tail = next(patch for patch in projections if b"emptyRetainedTail" in patch.new)
    data = baseline
    if initial_state in {"count_only", "both", "count_tail", "all"}:
        data = data.replace(count.old, count.new)
    if initial_state in {"history_only", "both", "history_tail", "all"}:
        data = data.replace(history.old, history.new)
    if initial_state in {"tail_only", "count_tail", "history_tail", "all"}:
        data = data.replace(tail.old, tail.new)
    target = tmp_path / history.relative_path
    target.parent.mkdir(parents=True)
    target.write_bytes(data)
    target.chmod(0o644)
    apis = [count_api, history_api] if first_patch == "count" else [history_api, count_api]
    for api in apis:
        api["run"](tmp_path, apply=True)
    final = target.read_bytes()
    assert count.new in final
    assert tail.new in final
    assert history.new in final
    restored = final.replace(history.new, history.old)
    for patch in reversed(projections):
        restored = restored.replace(patch.new, patch.old)
    assert restored == baseline
    for api in apis:
        api["run"](tmp_path, apply=False)
        api["run"](tmp_path, apply=True)
    assert target.read_bytes() == final


@pytest.mark.parametrize("mutation", ["outside_both", "inside_count", "inside_history", "duplicated_count", "inside_tail", "duplicated_tail"])
def test_composed_real_session_manager_authentication_rejects_unknown_changes(
    tmp_path: Path, mutation: str
) -> None:
    history_api, count_api, history, projections, baseline = real_session_manager_patches()
    count = next(patch for patch in projections if b"historicalTokens" in patch.new)
    tail = next(patch for patch in projections if b"emptyRetainedTail" in patch.new)
    data = baseline.replace(history.old, history.new)
    for patch in projections:
        data = data.replace(patch.old, patch.new)
    if mutation == "outside_both":
        data += b"\n// unrecognized unrelated SDK change\n"
    elif mutation == "inside_count":
        data = data.replace(b"historicalTokens >= 0", b"historicalTokens >= -1")
    elif mutation == "inside_history":
        data = data.replace(b'mode: 0o700', b'mode: 0o777')
    elif mutation == "duplicated_count":
        data += count.new
    elif mutation == "inside_tail":
        data = data.replace(b"emptyRetainedTail === true", b"emptyRetainedTail !== false")
    else:
        data += tail.new
    target = tmp_path / history.relative_path
    target.parent.mkdir(parents=True)
    target.write_bytes(data)
    target.chmod(0o644)
    for api in [history_api, count_api]:
        for apply in [False, True]:
            with pytest.raises(api["PatchError"], match="differs from both pinned patch states"):
                api["run"](tmp_path, apply=apply)
    assert target.read_bytes() == data

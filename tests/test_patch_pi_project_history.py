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

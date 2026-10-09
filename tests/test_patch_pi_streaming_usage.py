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


def real_loop_patches():
    usage_api = load_api()
    compaction_api = runpy.run_path(
        str(ROOT / "scripts" / "patch-pi-precontinuation-compaction"),
        run_name="composed_stream_compaction_patch_test",
    )
    relative = "node_modules/@earendil-works/pi-agent-core/dist/agent-loop.js"
    usage = next(patch for patch in usage_api["PATCHES"] if patch.relative_path == relative)
    compaction = tuple(patch for patch in compaction_api["PATCHES"] if patch.relative_path == relative)
    installed = Path.home() / ".local/share/qwen-r9700/pi/0.84.2" / relative
    if not installed.is_file():
        pytest.skip("pinned public Pi SDK is not installed")
    baseline = installed.read_bytes()
    for patch in reversed(compaction):
        for known in (patch.new, *patch.legacy):
            if known in baseline:
                baseline = baseline.replace(known, patch.old)
                break
    baseline = baseline.replace(usage.new, usage.old)
    assert hashlib.sha256(baseline).hexdigest() == usage.preimage_sha256
    usage_api["run"].__globals__["PATCHES"] = (usage,)
    compaction_api["run"].__globals__["PATCHES"] = compaction
    return usage_api, compaction_api, usage, compaction, baseline


@pytest.mark.parametrize("initial_state", ["stock", "usage_only", "composed"])
def test_real_stream_usage_and_compaction_loop_installation_remains_authenticated(
    tmp_path: Path, initial_state: str
) -> None:
    usage_api, compaction_api, usage, compaction, baseline = real_loop_patches()
    data = baseline
    if initial_state != "stock":
        data = data.replace(usage.old, usage.new)
    if initial_state == "composed":
        for patch in compaction:
            data = data.replace(patch.old, patch.new)
    target = tmp_path / usage.relative_path
    target.parent.mkdir(parents=True)
    target.write_bytes(data)
    target.chmod(0o644)
    # Continuous-usage dispatch is the prerequisite. Reverification after the
    # guarded continuation patch must authenticate both features in either order.
    usage_api["run"](tmp_path, apply=True)
    compaction_api["run"](tmp_path, apply=True)
    final = target.read_bytes()
    assert usage.new in final
    assert all(patch.new in final for patch in compaction)
    restored = final
    for patch in reversed(compaction):
        restored = restored.replace(patch.new, patch.old)
    assert restored.replace(usage.new, usage.old) == baseline
    for apis in [(usage_api, compaction_api), (compaction_api, usage_api)]:
        for api in apis:
            api["run"](tmp_path, apply=False)
            api["run"](tmp_path, apply=True)
    assert target.read_bytes() == final


@pytest.mark.parametrize("mutation", ["unrelated_tail", "guard_body", "duplicate_guard", "usage_dispatch"])
def test_composed_stream_usage_and_compaction_loop_rejects_unknown_or_ambiguous_bytes(
    tmp_path: Path, mutation: str
) -> None:
    usage_api, compaction_api, usage, compaction, baseline = real_loop_patches()
    data = baseline.replace(usage.old, usage.new)
    for patch in compaction:
        data = data.replace(patch.old, patch.new)
    if mutation == "unrelated_tail":
        data += b"\n// unrecognized generated runtime change\n"
    elif mutation == "guard_body":
        data = data.replace(b'interrupted === lastMessage', b'interrupted !== lastMessage')
    elif mutation == "duplicate_guard":
        data += compaction[0].new
    else:
        data = data.replace(b'case "usage_update":', b'case "unknown_usage_update":')
    target = tmp_path / usage.relative_path
    target.parent.mkdir(parents=True)
    target.write_bytes(data)
    target.chmod(0o644)
    for api in [usage_api, compaction_api]:
        for apply in [False, True]:
            with pytest.raises(api["PatchError"], match="differs from both pinned patch states"):
                api["run"](tmp_path, apply=apply)
    assert target.read_bytes() == data

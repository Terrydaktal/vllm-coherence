from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "scripts/pi-session-history"
SESSION_ID = "01a067d9-69cf-761f-8503-6554ccfd7703"


def encoded(path: Path) -> str:
    return f"--{str(path.resolve()).lstrip('/').replace('/', '-').replace(':', '-')}--"


def legacy(root: Path, cwd: Path, name: str, profile: str = "old") -> Path:
    path = root / profile / "agent-8012" / "sessions" / encoded(cwd) / name
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(PROGRAM), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_prepare_unifies_sessions_with_hardlinks_and_content_free_identity(tmp_path: Path) -> None:
    cwd = tmp_path / "tasks/money"
    cwd.mkdir(parents=True)
    state = tmp_path / "state"
    source = legacy(
        state,
        cwd,
        f"2026-09-03T15-18-16-783Z_{SESSION_ID}.jsonl",
    )
    source.write_bytes(b"synthetic transcript bytes\n")

    result = run("prepare", "--cwd", str(cwd), "--source-root", str(state))

    assert result.returncode == 0, result.stderr
    history = Path(result.stdout.strip())
    destination = history / source.name
    assert destination.stat().st_ino == source.stat().st_ino
    assert destination.stat().st_dev == source.stat().st_dev
    index = json.loads((history / ".identity.json").read_text())
    assert index["sessions"][source.name]["identity_path"] == str(source)
    assert "synthetic transcript" not in (history / ".identity.json").read_text()

    verified_local = run("verify", str(destination), "--cwd", str(cwd), "--source-root", str(state))
    verified_legacy = run("verify", str(source), "--cwd", str(cwd), "--source-root", str(state))
    assert verified_local.returncode == 0, verified_local.stderr
    assert verified_legacy.returncode == 0, verified_legacy.stderr
    assert Path(verified_local.stdout.strip()) == destination
    assert Path(verified_legacy.stdout.strip()) == destination


def test_resolve_last_and_uuid_are_project_local_and_ignore_qualifications(tmp_path: Path) -> None:
    cwd = tmp_path / "tasks/money"
    cwd.mkdir(parents=True)
    state = tmp_path / "state"
    older = legacy(
        state, cwd, "2026-09-01T00-00-00-000Z_aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
    )
    newer = legacy(state, cwd, f"2026-09-03T00-00-00-000Z_{SESSION_ID}.jsonl")
    older.write_bytes(b"older")
    newer.write_bytes(b"newer")
    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer, ns=(2_000_000_000, 2_000_000_000))
    qualification = state / "qualification" / "sessions" / newer.name
    qualification.parent.mkdir(parents=True)
    qualification.write_bytes(b"must be ignored")

    common = ("--cwd", str(cwd), "--source-root", str(state))
    last = run("resolve", *common, "last")
    by_id = run("resolve", *common, SESSION_ID)

    assert last.returncode == 0, last.stderr
    assert Path(last.stdout.strip()).name == newer.name
    assert by_id.returncode == 0, by_id.stderr
    assert Path(by_id.stdout.strip()).name == newer.name
    assert Path(by_id.stdout.strip()).stat().st_ino == newer.stat().st_ino


def test_divergent_legacy_filename_collision_fails_before_publication(tmp_path: Path) -> None:
    cwd = tmp_path / "tasks/money"
    cwd.mkdir(parents=True)
    state = tmp_path / "state"
    name = f"2026-09-03T00-00-00-000Z_{SESSION_ID}.jsonl"
    first = legacy(state, cwd, name, profile="one")
    second = legacy(state, cwd, name, profile="two")
    first.write_bytes(b"one")
    second.write_bytes(b"two")

    result = run("prepare", "--cwd", str(cwd), "--source-root", str(state))

    assert result.returncode == 3
    assert "divergent legacy session filename collision" in result.stderr
    assert not (cwd / ".pi/sessions" / name).exists()


def test_verify_rejects_an_unindexed_extra_hard_link(tmp_path: Path) -> None:
    cwd = tmp_path / "tasks/money"
    cwd.mkdir(parents=True)
    state = tmp_path / "state"
    source = legacy(
        state,
        cwd,
        f"2026-09-03T00-00-00-000Z_{SESSION_ID}.jsonl",
    )
    source.write_bytes(b"synthetic")
    prepared = run("prepare", "--cwd", str(cwd), "--source-root", str(state))
    assert prepared.returncode == 0, prepared.stderr
    destination = Path(prepared.stdout.strip()) / source.name
    extra = tmp_path / "extra.jsonl"
    extra.hardlink_to(source)

    result = run("verify", str(destination), "--cwd", str(cwd), "--source-root", str(state))

    assert result.returncode == 3
    assert "unauthenticated hard-link count" in result.stderr

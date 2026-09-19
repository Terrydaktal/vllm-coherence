"""Exercise destructive boundaries against disposable source trees, never the GPU."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "reconcile_lab", Path(__file__).resolve().parents[1] / "tools/reconcile_lab.py"
)
reconcile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reconcile)


@pytest.fixture
def trees(tmp_path, monkeypatch):
    canonical, lab = tmp_path / "canonical", tmp_path / "lab"
    for root in (canonical, lab):
        (root / "scripts").mkdir(parents=True)
    (canonical / "scripts/run").write_text("canonical\n")
    (canonical / "scripts/run").chmod(0o755)
    (canonical / "scripts/new").write_text("new helper\n")
    (lab / "scripts/run").write_text("uncommitted original\n")
    (lab / "scripts/run").chmod(0o751)
    (lab / "scripts/experiment").write_text("lab-only\n")
    (lab / "artifacts").mkdir()
    (lab / "artifacts/private").write_bytes(b"private fixture")
    monkeypatch.setattr(reconcile, "git", lambda *_: "synthetic-git-metadata")
    monkeypatch.setattr(
        reconcile, "source_paths", lambda _: ["scripts/new", "scripts/run"]
    )
    return canonical, lab, tmp_path / "backups"


def test_link_once_back_up_dirty_files_and_restore_exact_original(trees):
    canonical, lab, backups = trees
    plan = reconcile.make_plan(canonical, lab)
    manifest = reconcile.apply_plan(plan, backups)
    assert manifest.stat().st_mode & 0o777 == 0o600
    assert manifest.parent.stat().st_mode & 0o777 == 0o700
    assert (lab / "scripts/run").resolve() == canonical / "scripts/run"
    assert (lab / "scripts/run").stat().st_mode & 0o777 == 0o755
    assert reconcile.check(canonical, lab)["drift"] == []
    assert (lab / "scripts/experiment").read_text() == "lab-only\n"
    assert (lab / "artifacts/private").read_bytes() == b"private fixture"
    # Once linked, subsequent edits change exactly the source that is tested.
    (lab / "scripts/run").write_text("edited via compatibility path\n")
    assert (canonical / "scripts/run").read_text() == "edited via compatibility path\n"
    assert reconcile.apply_plan(reconcile.make_plan(canonical, lab), backups) is None
    assert reconcile.restore(manifest) == 2
    assert not (lab / "scripts/run").is_symlink()
    assert (lab / "scripts/run").read_text() == "uncommitted original\n"
    assert (lab / "scripts/run").stat().st_mode & 0o777 == 0o751
    assert not (lab / "scripts/new").exists()
    assert reconcile.restore(manifest) == 0


@pytest.mark.parametrize("side", ["lab", "canonical"])
def test_stale_plan_cannot_replace_concurrent_work(trees, side):
    canonical, lab, backups = trees
    plan = reconcile.make_plan(canonical, lab)
    changed = (lab if side == "lab" else canonical) / "scripts/run"
    changed.write_text("new work\n")
    with pytest.raises(ValueError, match="changed since plan"):
        reconcile.apply_plan(plan, backups)
    assert changed.read_text() == "new work\n"
    assert not (lab / "scripts/new").exists()


def test_restore_refuses_to_overwrite_a_replaced_link(trees):
    canonical, lab, backups = trees
    manifest = reconcile.apply_plan(reconcile.make_plan(canonical, lab), backups)
    path = lab / "scripts/run"
    path.unlink()
    path.write_text("new standalone work\n")
    assert reconcile.check(canonical, lab)["drift"] == ["scripts/run"]
    with pytest.raises(ValueError, match="subsequent lab work"):
        reconcile.restore(manifest)
    assert (lab / "scripts/new").is_symlink()
    assert path.read_text() == "new standalone work\n"


def test_interrupted_apply_is_recoverable(trees, monkeypatch):
    canonical, lab, backups = trees
    real = reconcile.replace_link

    def fail_second(path, target):
        if path.name == "run":
            raise OSError("simulated interruption")
        real(path, target)

    monkeypatch.setattr(reconcile, "replace_link", fail_second)
    with pytest.raises(RuntimeError, match="originals retained"):
        reconcile.apply_plan(reconcile.make_plan(canonical, lab), backups)
    (manifest,) = backups.glob("*/manifest.json")
    assert json.loads(manifest.read_text())["state"] == "applying"
    assert reconcile.restore(manifest) == 1
    assert (lab / "scripts/run").read_text() == "uncommitted original\n"


def test_indirect_directories_and_private_paths_are_rejected(trees):
    canonical, lab, _ = trees
    (lab / "tools").symlink_to(canonical / "scripts", target_is_directory=True)
    with pytest.raises(ValueError, match="indirect"):
        reconcile.safe_path(lab, "tools/run")
    for name in (
        "../source.py",
        "/tmp/source.py",
        "artifacts/capture",
        "benchmarks/results/upstream-run.json",
        "tests/sessions/chat.jsonl",
    ):
        with pytest.raises(ValueError, match="admitted"):
            reconcile.safe_path(lab, name)


def test_corrupt_backup_cannot_be_restored(trees):
    canonical, lab, backups = trees
    manifest = reconcile.apply_plan(reconcile.make_plan(canonical, lab), backups)
    (manifest.parent / "originals/scripts/run").write_text("damaged backup\n")
    with pytest.raises(ValueError, match="changed backup"):
        reconcile.restore(manifest)
    assert (lab / "scripts/run").is_symlink()

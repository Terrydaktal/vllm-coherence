"""One-slot retention, admission and failure behavior using real tensor evidence."""

import errno
import hashlib
import os
import shutil

import pytest
from conformance_fixture import tiny_plan

from qwen_r9700_lab.conformance_boundaries import compare_boundaries
from qwen_r9700_lab.conformance_reference_store import SCHEMA, SerialReferenceStore
from qwen_r9700_lab.conformance_replay import run_reference
from qwen_r9700_lab.conformance_state import compare_frames
from qwen_r9700_lab.diagnostic_contract import (
    DiagnosticError,
    digest,
    private_json,
    seal,
    write_private,
)

REFLINK = os.environ.get("QWEN_REQUIRE_REFLINK") == "1"


def reseal(value, **changes):
    return seal({**{k: v for k, v in value.items() if k != "sha256"}, **changes})


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    root = tmp_path_factory.mktemp("store-reference")
    plan = reseal(tiny_plan(root / "checkpoint"), forced_tokens=list(range(1, 7)))
    run_reference(plan, root / "capture")
    return plan, root / "capture"


@pytest.fixture
def source(baseline, tmp_path):
    plan, capture = baseline
    origin = tmp_path / "case-00000"
    origin.mkdir(mode=0o700)
    case = {"id": "forced_d7.fixture", "family": "forced_d7"}
    campaign = seal({"cases": [case]})
    write_private(origin / "input.json", {"campaign": campaign, "case": case})
    target = origin / "serial/capture"
    target.parent.mkdir(mode=0o700)
    shutil.copytree(capture, target)
    identity = seal(
        {
            "schema": SCHEMA + "/identity",
            "campaign": campaign["sha256"],
            "configuration": {"eager": True},
            "environment": {"flag": "1"},
        }
    )
    return identity, plan, target, origin


def populate(store, source):
    identity, plan, capture, origin = source
    return store.publish(identity, plan, capture, origin)


def first_blob(root):
    frame = private_json(root / "frame-000000/frame.json")
    return root / "frame-000000" / next(iter(frame["components"].values()))["file"]


def archive_origin(source):
    _, _, capture, origin = source
    payload = private_json(origin / "input.json")
    result = seal({"campaign": payload["campaign"]["sha256"], "case": payload["case"]})
    write_private(origin / "result.json", result)
    locator = {
        "schema": "urn:qwen:conformance-archive-locator:v1",
        "archive_path": f"/{origin.name}.tar",
        "snapshot_id": digest("archived snapshot"),
        "tar_sha256": digest("tar"),
        "manifest_sha256": digest("manifest"),
        "identity": {
            "scope": "finished-case-v1",
            "input_sha256": hashlib.sha256((origin / "input.json").read_bytes()).hexdigest(),
            "campaign": payload["campaign"]["sha256"],
            "result": result["sha256"],
        },
    }
    write_private(origin / "archive-locator.json", locator)
    write_private(origin / "archive-retirement.json", {**locator, "metadata_retained": True})
    shutil.rmtree(capture)
    return locator


def test_hit_survives_reopen_and_matches_original_byte_for_byte(source, tmp_path):
    identity, plan, capture, _ = source
    with SerialReferenceStore(tmp_path / "store", reflink=REFLINK) as store:
        assert store.project(identity, plan, plan, tmp_path / "absent") is None
        populate(store, source)
    with SerialReferenceStore(tmp_path / "store", reflink=REFLINK) as store:
        output = tmp_path / "hit"
        result = store.project(identity, plan, plan, output)
        assert result["states"] == len(plan["forced_tokens"])
    for frame in private_json(capture / "schedule.json")["frames"]:
        assert compare_frames(capture / frame["name"], output / frame["name"])["equal"]
    assert compare_boundaries(capture / "boundaries", output / "boundaries", tmp_path / "diff")[
        "equal"
    ]
    original = first_blob(capture).read_bytes()
    first_blob(output).write_bytes(b"consumer mutation")
    assert first_blob(capture).read_bytes() == original
    assert first_blob(tmp_path / "store/current/capture").read_bytes() == original


@pytest.mark.parametrize("field", ["campaign", "configuration", "environment"])
def test_runtime_identity_change_is_a_miss_not_a_stale_hit(source, tmp_path, field):
    identity, plan, _, _ = source
    with SerialReferenceStore(tmp_path / "store", reflink=REFLINK) as store:
        populate(store, source)
        changed = reseal(
            identity,
            **{field: digest("another campaign") if field == "campaign" else {"changed": "yes"}},
        )
        assert store.project(changed, plan, plan, tmp_path / "changed") is None
        assert not (tmp_path / "changed").exists()


@pytest.mark.parametrize("field", ["prefix", "forced_tokens", "contract"])
def test_plan_change_is_a_miss(source, tmp_path, field):
    identity, plan, _, _ = source
    changed = reseal(plan, **{field: digest("new contract") if field == "contract" else [0]})
    with SerialReferenceStore(tmp_path / "store", reflink=REFLINK) as store:
        populate(store, source)
        assert store.project(identity, changed, changed, tmp_path / "changed") is None


def test_matching_but_corrupt_cache_is_an_error_not_a_miss(source, tmp_path):
    identity, plan, _, _ = source
    with SerialReferenceStore(tmp_path / "store", reflink=REFLINK) as store:
        populate(store, source)
        first_blob(tmp_path / "store/current/capture").write_bytes(b"corrupt")
        with pytest.raises(DiagnosticError):
            store.project(identity, plan, plan, tmp_path / "bad")
        assert not (tmp_path / "bad/reference-projection.json").exists()


def test_current_reference_must_be_retired_before_population(source, tmp_path):
    with SerialReferenceStore(tmp_path / "store", reflink=REFLINK) as store:
        first = populate(store, source)
        with pytest.raises(DiagnosticError, match="retire the previous"):
            populate(store, source)
        assert private_json(tmp_path / "store/current/entry.json") == first
        assert not (tmp_path / "store/pending").exists()


def test_retirement_preserves_live_original_and_durable_receipt(source, tmp_path):
    _, _, capture, _ = source
    before = first_blob(capture).read_bytes()
    with SerialReferenceStore(tmp_path / "store", reflink=REFLINK) as store:
        populate(store, source)
        result = store.retire()
        assert result["retained"]["kind"] == "verified-original-capture"
        assert not (tmp_path / "store/current").exists()
        assert first_blob(capture).read_bytes() == before
        assert len(list((tmp_path / "store/retired").glob("*.json"))) == 1
        populate(store, source)
        second = store.retire()
        assert result["entry"]["publication_id"] != second["entry"]["publication_id"]
        assert len(list((tmp_path / "store/retired").glob("*.json"))) == 2


def test_retirement_uses_completed_archive_receipt_after_original_removal(source, tmp_path):
    with SerialReferenceStore(tmp_path / "store", reflink=REFLINK) as store:
        populate(store, source)
        locator = archive_origin(source)
        receipt = store.retire()
        assert receipt["retained"] == {"kind": "verified-case-archive", "locator": locator}


@pytest.mark.parametrize("fault", ["missing", "corrupt", "changed_input"])
def test_last_good_copy_is_preserved_when_origin_is_not_retained(source, tmp_path, fault):
    _, _, capture, origin = source
    with SerialReferenceStore(tmp_path / "store", reflink=REFLINK) as store:
        populate(store, source)
        if fault == "missing":
            first_blob(capture).unlink()
        elif fault == "corrupt":
            first_blob(capture).write_bytes(b"corrupt original")
        else:
            (origin / "input.json").write_text("{}\n")
        with pytest.raises((DiagnosticError, FileNotFoundError)):
            store.retire()
        assert first_blob(tmp_path / "store/current/capture").exists()
        assert not list((tmp_path / "store/retired").iterdir())


@pytest.mark.parametrize("fault", ["incomplete", "wrong_case", "wrong_archive", "bad_hash"])
def test_unrelated_or_incomplete_archive_does_not_authorize_eviction(source, tmp_path, fault):
    _, _, _, origin = source
    with SerialReferenceStore(tmp_path / "store", reflink=REFLINK) as store:
        populate(store, source)
        locator = archive_origin(source)
        if fault == "incomplete":
            (origin / "archive-retirement.json").unlink()
        else:
            if fault == "wrong_case":
                locator["identity"]["input_sha256"] = digest("another case")
            elif fault == "wrong_archive":
                locator["archive_path"] = "/another.tar"
            else:
                locator["snapshot_id"] = "not a digest"
            for name, value in [
                ("archive-locator.json", locator),
                ("archive-retirement.json", {**locator, "metadata_retained": True}),
            ]:
                (origin / name).unlink()
                write_private(origin / name, value)
        with pytest.raises((DiagnosticError, FileNotFoundError)):
            store.retire()
        assert (tmp_path / "store/current").is_dir()


def test_second_controller_cannot_enter_the_store(source, tmp_path):
    with (
        SerialReferenceStore(tmp_path / "store", reflink=REFLINK),
        pytest.raises(BlockingIOError),
        SerialReferenceStore(tmp_path / "store", reflink=REFLINK),
    ):
        pytest.fail("second controller acquired the same store")


def test_unlocked_operations_are_refused(source, tmp_path):
    store = SerialReferenceStore(tmp_path / "store", reflink=REFLINK)
    with pytest.raises(DiagnosticError, match="requires its lock"):
        populate(store, source)


def test_failed_publication_is_not_reused_and_keeps_original(source, tmp_path, monkeypatch):
    _, _, capture, _ = source
    original = first_blob(capture).read_bytes()

    def fail(*_):
        raise OSError(errno.ENOSPC, "injected full filesystem")

    with SerialReferenceStore(tmp_path / "store", reflink=REFLINK) as store:
        monkeypatch.setattr("qwen_r9700_lab.conformance_reference_store.sync", fail)
        with pytest.raises(OSError, match="full filesystem"):
            populate(store, source)
        assert not (tmp_path / "store/current").exists()
        assert (tmp_path / "store/pending").exists()
        assert first_blob(capture).read_bytes() == original
    with (
        pytest.raises(DiagnosticError, match="unfinished"),
        SerialReferenceStore(tmp_path / "store", reflink=REFLINK),
    ):
        pytest.fail("unfinished publication was silently reused")


def test_resealed_incomplete_schedule_cannot_authorize_retirement(source, tmp_path):
    with SerialReferenceStore(tmp_path / "store", reflink=REFLINK) as store:
        populate(store, source)
        path = tmp_path / "store/current/capture/schedule.json"
        document = private_json(path)
        path.unlink()
        write_private(path, reseal(document, frames=[]))
        with pytest.raises(DiagnosticError, match="completion changed"):
            store.retire()
        assert first_blob(tmp_path / "store/current/capture").exists()


@pytest.mark.parametrize("field", ["schema", "campaign", "configuration", "environment"])
def test_incomplete_identity_cannot_populate_a_store(source, tmp_path, field):
    identity, plan, capture, origin = source
    broken = seal({k: v for k, v in identity.items() if k not in {"sha256", field}})
    with SerialReferenceStore(tmp_path / "store", reflink=REFLINK) as store:
        with pytest.raises(DiagnosticError, match="identity requires"):
            store.publish(broken, plan, capture, origin)
        assert not (tmp_path / "store/pending").exists()


def test_producer_campaign_must_match_lookup_identity(source, tmp_path):
    identity, plan, capture, origin = source
    with SerialReferenceStore(tmp_path / "store", reflink=REFLINK) as store:
        with pytest.raises(DiagnosticError, match="another campaign"):
            store.publish(reseal(identity, campaign=digest("another")), plan, capture, origin)
        assert not (tmp_path / "store/pending").exists()

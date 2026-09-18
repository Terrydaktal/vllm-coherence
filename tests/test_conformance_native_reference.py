"""Serial-prefix projections retain exact observations and reject stale sources."""

import pytest
from conformance_fixture import tiny_plan

from qwen_r9700_lab.conformance_boundaries import compare_boundaries
from qwen_r9700_lab.conformance_native_reference import project_serial_reference
from qwen_r9700_lab.conformance_replay import run_reference
from qwen_r9700_lab.conformance_session import compare_campaign
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, private_json, seal, write_private


def changed(plan, **values):
    return seal({**{k: v for k, v in plan.items() if k != "sha256"}, **values})


@pytest.fixture
def serial(tmp_path):
    plan = changed(tiny_plan(tmp_path / "checkpoint"), forced_tokens=list(range(1, 12)))
    root = tmp_path / "full"
    run_reference(plan, root)
    return plan, root


@pytest.mark.parametrize("width", range(8))
def test_projected_prefix_equals_independent_shorter_execution(serial, tmp_path, width):
    source, root = serial
    requested = changed(source, forced_tokens=source["forced_tokens"][: width + 4])
    result = project_serial_reference(source, requested, root, tmp_path / "projected")
    run_reference(requested, tmp_path / "fresh")
    assert result["states"] == width + 4
    assert compare_campaign(tmp_path / "fresh", tmp_path / "projected", tmp_path / "states")[
        "equal"
    ]
    assert compare_boundaries(
        tmp_path / "fresh/boundaries", tmp_path / "projected/boundaries", tmp_path / "boundaries"
    )["equal"]
    assert (
        private_json(tmp_path / "projected/schedule.json")["reference_projection"]["source_plan"]
        == source["sha256"]
    )


@pytest.mark.parametrize(
    "fault",
    [
        "prefix",
        "forced",
        "longer",
        "contract",
        "execution",
        "accepted",
        "observation",
        "schedule",
        "missing_boundary",
        "tensor",
    ],
)
def test_incompatible_or_damaged_source_never_publishes_projection(serial, tmp_path, fault):
    source, root = serial
    requested = changed(source, forced_tokens=source["forced_tokens"][:4])
    if fault in ["prefix", "forced", "longer", "contract", "execution", "accepted", "observation"]:
        replacement = {
            "prefix": {"prefix": [0, *source["prefix"][1:]]},
            "forced": {"forced_tokens": [0, 2, 3, 4]},
            "longer": {"forced_tokens": source["forced_tokens"] + [12]},
            "contract": {"contract": "0" * 64},
            "execution": {"execution": "0" * 64},
            "accepted": {"accepted_widths": [0, 0, 0]},
            "observation": {"observation_positions": [len(source["prefix"]) + 4]},
        }[fault]
        requested = changed(requested, **replacement)
    elif fault == "schedule":
        d = private_json(root / "schedule.json")
        d["frames"].pop()
        (root / "schedule.json").unlink()
        write_private(root / "schedule.json", changed(d))
    elif fault == "missing_boundary":
        d = private_json(root / "boundaries/boundaries.json")
        d["frames"].pop(0)
        (root / "boundaries/boundaries.json").unlink()
        write_private(root / "boundaries/boundaries.json", changed(d))
    else:
        frame = private_json(root / "frame-000000/frame.json")
        blob = next(iter(frame["components"].values()))["file"]
        path = root / "frame-000000" / blob
        data = bytearray(path.read_bytes())
        data[0] ^= 1
        path.write_bytes(data)
    with pytest.raises(DiagnosticError):
        project_serial_reference(source, requested, root, tmp_path / "projected")
    assert not (tmp_path / "projected/reference-projection.json").exists()


def test_projection_owns_copies_and_cannot_mutate_the_baseline(serial, tmp_path):
    source, root = serial
    requested = changed(source, forced_tokens=source["forced_tokens"][:4])
    project_serial_reference(source, requested, root, tmp_path / "projected")
    frame = private_json(root / "frame-000000/frame.json")
    blob = next(iter(frame["components"].values()))["file"]
    original = root / "frame-000000" / blob
    projected = tmp_path / "projected/frame-000000" / blob
    before = original.read_bytes()
    assert original.stat().st_ino != projected.stat().st_ino
    projected.write_bytes(b"consumer corruption")
    assert original.read_bytes() == before


def test_reference_must_cover_every_requested_observation(serial, tmp_path):
    source, _ = serial
    restricted = changed(source, observation_positions=[0])
    root = tmp_path / "restricted"
    run_reference(restricted, root)
    requested = changed(source, forced_tokens=source["forced_tokens"][:4])
    with pytest.raises(DiagnosticError, match="missing required boundary"):
        project_serial_reference(restricted, requested, root, tmp_path / "projected")


def test_reflink_projection_verifies_every_state_and_boundary(serial, tmp_path, monkeypatch):
    from test_conformance_reflink import simulated_clone

    from qwen_r9700_lab.conformance_state import archive_frame

    calls = []

    def record(source, destination, *, reflink=False):
        assert reflink is True
        result = archive_frame(source, destination, reflink=reflink)
        calls.append(result["sha256"])
        return result

    monkeypatch.setattr("qwen_r9700_lab.conformance_state.fcntl.ioctl", simulated_clone)
    monkeypatch.setattr("qwen_r9700_lab.conformance_native_reference.archive_frame", record)
    source, root = serial
    requested = changed(source, forced_tokens=source["forced_tokens"][:4])
    result = project_serial_reference(source, requested, root, tmp_path / "projected", reflink=True)
    assert result["storage"] == "independent verified reflinks"
    assert len(calls) == result["states"] + result["observations"]
    run_reference(requested, tmp_path / "fresh")
    assert compare_campaign(tmp_path / "fresh", tmp_path / "projected", tmp_path / "states")[
        "equal"
    ]
    assert compare_boundaries(
        tmp_path / "fresh/boundaries", tmp_path / "projected/boundaries", tmp_path / "boundaries"
    )["equal"]

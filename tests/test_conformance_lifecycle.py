from copy import deepcopy

import numpy as np
import pytest
from conformance_fixture import tiny_plan

from qwen_r9700_lab.conformance_lifecycle import (
    FrameTransport,
    RecoveryCapture,
    compare_recovery,
    pack_frame,
    unpack_frame,
)
from qwen_r9700_lab.conformance_model import Checkpoint, QuantizedQwenReference, state_names
from qwen_r9700_lab.conformance_state import compare_frames
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, digest, seal, write_private
from qwen_r9700_lab.radiance_cache import RetiredGenerationError


def reference(plan):
    from pathlib import Path

    return QuantizedQwenReference(
        Checkpoint(Path(plan["checkpoint"]), plan["checkpoint_files"]),
        kv_scales=plan["kv_scales"],
        **{k: plan[k] for k in ("contract", "execution", "adapter")},
    )


@pytest.fixture
def prepared(tmp_path):
    plan = tiny_plan(tmp_path / "checkpoint")
    model = reference(plan)
    for token in plan["prefix"]:
        model.step(token)
    model.frame(tmp_path / "saved", phase="commit", pending=7)
    (tmp_path / "disk").mkdir(mode=0o700)
    transport = FrameTransport(
        tmp_path / "disk", {"id": digest("chat A"), "generation": digest("generation 1")}
    )
    try:
        yield plan, model, transport
    finally:
        model.close()


@pytest.mark.parametrize("route", ["ram", "disk", "eviction", "restart"])
def test_actual_compressed_transport_preserves_all_state_and_next_logits(tmp_path, prepared, route):
    plan, model, transport = prepared
    transport.save_ram(tmp_path / "saved")
    receipt = transport.save_disk(tmp_path / "saved")
    if route == "eviction":
        receipt = transport.evict_ram(tmp_path / "saved")
        assert transport.ram is None
    if route == "restart":
        transport = FrameTransport(tmp_path / "disk", transport.store.chat)
        assert transport.ram is None
    if route == "ram":
        transport.restore_ram(tmp_path / "restored")
    else:
        transport.restore_disk(receipt, tmp_path / "restored")
    assert compare_frames(tmp_path / "saved", tmp_path / "restored")["equal"]
    restored = reference(plan)
    try:
        restored.restore(tmp_path / "restored")
        for token in plan["forced_tokens"]:
            np.testing.assert_array_equal(restored.step(token), model.step(token))
    finally:
        restored.close()
    assert transport.store.io_totals()["written_blocks"] > 0


@pytest.mark.parametrize("component", ["gdn", "conv", "keys", "values"])
def test_bad_live_state_good_restore_is_localized_without_reading_text(
    tmp_path, prepared, component
):
    _plan, model, transport = prepared
    receipt = transport.save_disk(tmp_path / "saved")
    layer = 0 if component in {"gdn", "conv"} else 3
    model.state[layer][component].flat[0] += 1
    model.frame(tmp_path / "bad-live", phase="step", pending=7)
    transport.restore_disk(receipt, tmp_path / "restored")
    result = compare_recovery(
        tmp_path / "saved",
        tmp_path / "bad-live",
        tmp_path / "restored",
        tmp_path / "comparison",
        required=state_names(model.config),
    )
    assert result["classification"] == "live_differs_restored_matches_reference"
    assert result["comparisons"]["reference_live"]["first_difference"]["boundary"] == (
        f"layer.{layer:03d}.{component}"
    )
    assert not result["equal"]
    assert result["corruption_cause"].startswith("UNPROVED")


@pytest.mark.parametrize("persist", [False, True])
def test_restore_damage_and_already_saved_damage_are_distinguished(tmp_path, prepared, persist):
    _, model, transport = prepared
    if not persist:
        model.frame(tmp_path / "live", phase="step", pending=7)
    model.state[1]["gdn"].flat[-1] += 2
    model.frame(tmp_path / "bad", phase="step", pending=7)
    if persist:
        model.frame(tmp_path / "live", phase="step", pending=7)
    receipt = transport.save_disk(tmp_path / "bad")
    transport.restore_disk(receipt, tmp_path / "restored")
    result = compare_recovery(
        tmp_path / "saved",
        tmp_path / "live",
        tmp_path / "restored",
        tmp_path / "comparison",
        required=state_names(model.config),
    )
    assert result["classification"] == (
        "same_difference_survives_restore" if persist else "restore_introduced_difference"
    )


def test_failed_publication_keeps_previous_head_and_ram_copy(tmp_path, prepared, monkeypatch):
    _, model, transport = prepared
    old = transport.save_disk(tmp_path / "saved")
    transport.save_ram(tmp_path / "saved")
    previous_ram = transport.ram
    model.step(7)
    model.frame(tmp_path / "successor", phase="commit", pending=9)
    staged = transport.stage_disk(tmp_path / "successor")
    new_key = next(k for k in staged["keys"] if k not in old["keys"])
    transport.store.path(new_key).unlink()
    assert transport.publish(staged) is False
    transport.restore_disk(old, tmp_path / "recovered")
    assert compare_frames(tmp_path / "saved", tmp_path / "recovered")["equal"]
    monkeypatch.setattr(transport, "publish", lambda receipt: False)
    with pytest.raises(DiagnosticError, match="previous disk head retained"):
        transport.evict_ram(tmp_path / "successor")
    assert transport.ram == previous_ram


def test_disk_corruption_and_other_chat_receipt_are_rejected(tmp_path, prepared):
    _, _, transport = prepared
    receipt = transport.save_disk(tmp_path / "saved")
    wrong = deepcopy(receipt)
    wrong["chat"]["id"] = digest("chat B")
    wrong = seal({k: v for k, v in wrong.items() if k != "sha256"})
    with pytest.raises(DiagnosticError, match="different chat"):
        transport.restore_disk(wrong, tmp_path / "wrong")
    path = transport.store.path(receipt["keys"][0])
    data = bytearray(path.read_bytes())
    data[-1] ^= 0x80
    path.write_bytes(data)
    with pytest.raises((ValueError, RuntimeError)):
        transport.restore_disk(receipt, tmp_path / "corrupted")


def test_compaction_generation_retires_old_writers_without_reusing_their_state(tmp_path, prepared):
    _, model, previous = prepared
    previous.save_disk(tmp_path / "saved")
    next_chat = {**previous.store.chat, "generation": digest("generation 2")}
    successor = FrameTransport(tmp_path / "disk", next_chat)
    assert successor.store.metadata()["fallback"]["generation"] == previous.store.chat["generation"]
    with pytest.raises(DiagnosticError, match="superseded"):
        previous.stage_disk(tmp_path / "saved")
    model.reset()
    model.step(12)
    model.frame(tmp_path / "compacted", phase="commit", pending=9)
    receipt = successor.save_disk(tmp_path / "compacted")
    assert successor.store.metadata()["fallback"] is None
    assert not previous.store.generation.exists()
    successor.restore_disk(receipt, tmp_path / "restored")
    assert compare_frames(tmp_path / "compacted", tmp_path / "restored")["equal"]
    with pytest.raises(RetiredGenerationError):
        FrameTransport(tmp_path / "disk", previous.store.chat)


def test_two_chat_ram_handover_has_no_mutable_alias(tmp_path, prepared):
    plan, a, ta = prepared
    b = reference(plan)
    tb = FrameTransport(tmp_path / "disk", {"id": digest("B"), "generation": digest("gen B")})
    try:
        for token in [15, 16, 17]:
            b.step(token)
        b.frame(tmp_path / "B", phase="commit", pending=18)
        ta.save_ram(tmp_path / "saved")
        tb.save_ram(tmp_path / "B")
        expected_a = ta.ram
        for _ in range(3):
            a.state[0]["conv"].fill(999)  # same GPU allocation overwritten by another owner
            tb.restore_ram(tmp_path / f"B-restore-{_}")
            b.restore(tmp_path / f"B-restore-{_}")
            b.step(18)
            b.frame(tmp_path / f"B-next-{_}", phase="commit", pending=18)
            tb.save_ram(tmp_path / f"B-next-{_}")
            ta.restore_ram(tmp_path / f"A-restore-{_}")
            a.restore(tmp_path / f"A-restore-{_}")
            assert ta.ram == expected_a
            assert compare_frames(tmp_path / "saved", tmp_path / f"A-restore-{_}")["equal"]
    finally:
        b.close()


def test_recovery_capture_refuses_missing_duplicate_cross_generation_and_inflight(
    tmp_path, prepared
):
    plan, model, transport = prepared
    cap = RecoveryCapture(
        tmp_path / "capture",
        contract=plan["contract"],
        chat=transport.store.chat["id"],
        generation=transport.store.chat["generation"],
        required=state_names(model.config),
    )
    kwargs = {"chat": transport.store.chat["id"], "generation": transport.store.chat["generation"]}
    with pytest.raises(DiagnosticError, match="completed state transition"):
        cap.record("live", tmp_path / "saved", **kwargs, quiescent=False)
    with pytest.raises(DiagnosticError, match="compaction generation"):
        cap.record(
            "live", tmp_path / "saved", **{**kwargs, "generation": digest("wrong")}, quiescent=True
        )
    cap.record("live", tmp_path / "saved", **kwargs, quiescent=True)
    with pytest.raises(DiagnosticError, match="duplicate"):
        cap.record("live", tmp_path / "saved", **kwargs, quiescent=True)
    with pytest.raises(DiagnosticError, match="missing a capture"):
        cap.compare()
    for name in ("reference", "restored"):
        cap.record(name, tmp_path / "saved", **kwargs, quiescent=True)
    assert cap.compare()["equal"]


@pytest.mark.parametrize("change", ["truncated", "extra", "metadata", "payload"])
def test_envelope_negative_controls(tmp_path, prepared, change):
    data = pack_frame(tmp_path / "saved")
    if change == "truncated":
        data = data[:-1]
    elif change == "extra":
        data += b"extra"
    else:
        data = bytearray(data)
        data[20 if change == "metadata" else -1] ^= 0x40
        data = bytes(data)
    with pytest.raises((DiagnosticError, ValueError)):
        unpack_frame(data, tmp_path / "unpack")


def test_publication_verified_then_payload_changed_keeps_previous_checkpoint(tmp_path, prepared):
    _, model, transport = prepared
    old = transport.save_disk(tmp_path / "saved")
    model.step(7)
    model.frame(tmp_path / "next", phase="commit", pending=9)
    staged = transport.stage_disk(tmp_path / "next")
    prepared_head = transport.store.prepare_publication(staged["keys"], transport.block_size)
    key = next(k for k in staged["keys"] if k not in old["keys"])
    transport.store.path(key).write_bytes(b"truncated after verification")
    assert (
        transport.store.publish(
            staged["keys"], staged["consumed"], transport.block_size, prepared=prepared_head
        )
        is False
    )
    transport.restore_disk(old, tmp_path / "restored")
    assert compare_frames(tmp_path / "saved", tmp_path / "restored")["equal"]


def test_real_process_restart_restores_receipt_and_preserves_pending_token(tmp_path, prepared):
    import os
    import subprocess
    import sys

    _, _, transport = prepared
    receipt = transport.save_disk(tmp_path / "saved")
    write_private(tmp_path / "receipt.json", receipt)
    code = """
import json,sys
from pathlib import Path
from qwen_r9700_lab.conformance_lifecycle import FrameTransport
r=Path(sys.argv[1]);receipt=json.load(open(r/'receipt.json'))
t=FrameTransport(r/'disk',receipt['chat'])
f=t.restore_disk(receipt,r/'child-restored')
assert f['pending']==7 and f['consumed']==3
assert 'torch' not in sys.modules and 'vllm' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=20,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
        },
    )
    assert result.returncode == 0, result.stderr
    assert compare_frames(tmp_path / "saved", tmp_path / "child-restored")["equal"]

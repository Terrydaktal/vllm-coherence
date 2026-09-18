"""CPU scheduler regression: preserve response state while changing GPU ownership."""

from types import SimpleNamespace

import pytest
from test_radiance_fair_scheduler import (
    BANK_A,
    BANK_A_NEXT,
    BANK_B,
    CHAT_A,
    CHAT_B,
    load_module,
    new_scheduler,
    tool_boundary,
)


def control(chat=CHAT_A, priority=1, sequence=1, active=True, **extra):
    return dict(
        chat_id=chat,
        priority=priority,
        sequence=sequence,
        active=active,
        client="c" * 32,
        answer="d" * 32,
        **extra,
    )


@pytest.mark.parametrize("priority", [0, 1, 2])
def test_equal_priorities_preserve_response_and_two_second_tool_grace(monkeypatch, priority):
    module = load_module(monkeypatch)
    now = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    scheduler = tool_boundary(module)
    for chat in (CHAT_A, CHAT_B):
        scheduler.priorities.update(control(chat, priority))
    scheduler.response_outcome("1" * 32, True)
    for now[0] in (100.0, 101.999):
        assert scheduler._choose() == BANK_A
        assert scheduler.priority_hold is None
    now[0] = 102.0
    assert scheduler._choose() == BANK_B


@pytest.mark.parametrize("priority", [1, 2])
def test_higher_owner_keeps_long_tool_despite_old_fairness_deadline(monkeypatch, priority):
    module = load_module(monkeypatch)
    now = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    scheduler = tool_boundary(module)
    scheduler.priorities.update(control(priority=priority))
    scheduler.response_outcome("1" * 32, True)
    for now[0] in (100.0, 103.0, 131.0, 159.0):
        assert scheduler._choose() == BANK_A
        assert scheduler.priority_hold["priority"] == priority
    scheduler.priorities.update(control(priority=priority, sequence=2, active=False))
    assert scheduler._choose() == BANK_B
    assert scheduler.priority_hold is None


def test_priority_one_waits_for_response_then_takes_natural_boundary(monkeypatch):
    module = load_module(monkeypatch)
    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
    scheduler = tool_boundary(module)
    request = scheduler.response_request
    finished = [False]
    request.is_finished = lambda: finished[0]
    request.status.name = "RUNNING"
    scheduler.running = [request]
    scheduler.priorities.update(control(CHAT_B, 1))
    assert scheduler._choose() == BANK_A
    finished[0] = True
    scheduler.running.clear()
    scheduler.response_outcome("1" * 32, True)
    assert scheduler._choose() == BANK_B  # no grace delays for unequal priorities


@pytest.mark.parametrize("state", ["decode", "prefill", "tool_arguments", "thinking"])
def test_priority_two_parks_and_resumes_without_preempting_or_resetting_model_state(
    monkeypatch, state
):
    module = load_module(monkeypatch)
    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
    scheduler = new_scheduler(module)

    def request(rid, computed):
        r = SimpleNamespace(
            request_id=rid,
            done=False,
            status=SimpleNamespace(name="RUNNING"),
            num_computed_tokens=computed,
            num_in_flight_tokens=0,
            spec_token_ids=[3, 4, 5],
            kv_transfer_params={},
            stage=state,
        )
        r.is_finished = lambda: r.done
        return r

    a, b = request("a", 150800 if state != "prefill" else 16000), request("b", 0)
    owners = {"a": BANK_A, "b": BANK_B}
    banks = SimpleNamespace(
        active=BANK_A,
        owners=owners,
        managers={BANK_A: object(), BANK_B: object()},
        live_blocks=lambda: [7, 8, 9],
    )
    banks.activate = lambda key: setattr(banks, "active", key)
    scheduler.banks = banks
    scheduler.running, scheduler.waiting, scheduler.skipped_waiting = [a], [b], []
    scheduler.requests, scheduler.parked, scheduler.finished_req_ids = {"a": a, "b": b}, {}, set()
    scheduler.response_request = a
    scheduler.pending_switch, scheduler.pending_discard, scheduler.drop_banks = None, False, []
    scheduler.last_served, scheduler.max_banks, scheduler.switch_count = {BANK_A: 90.0}, 2, 0
    scheduler._publish_status = lambda **kw: None
    scheduler.status_path = "/unused-cpu-fixture"
    frames = []

    def parent(throttle_prefills=False, *, barrier=False):
        scheduler.finished_req_ids.clear()
        if not barrier and not scheduler.running:
            admitted = next(
                (r for r in scheduler.waiting if owners[r.request_id] == banks.active), None
            )
            if admitted:
                scheduler.waiting.remove(admitted)
                scheduler.running.append(admitted)
        frame = SimpleNamespace(qwen_fair=None)
        frames.append(frame)
        return frame

    scheduler._parent_step = parent
    scheduler.priorities.update(control(CHAT_B, 2))
    original = (a.num_computed_tokens, a.spec_token_ids, a.status, banks.managers[BANK_A])
    first = scheduler.schedule()
    assert first.qwen_fair["barrier"] and first.qwen_fair["bank"] == BANK_A
    assert scheduler.parked == {BANK_A: a} and scheduler.running == []
    assert scheduler.response_request is b
    second = scheduler.schedule()
    assert second.qwen_fair["save_blocks"] == [7, 8, 9]
    assert second.qwen_fair["bank"] == BANK_B
    assert not second.qwen_fair["discard_active"]
    assert scheduler.running == [b]
    # Repeated requests to keep the same priority cannot cause another handover.
    scheduler.priorities.update(control(CHAT_B, 2, sequence=2))
    assert not scheduler.schedule().qwen_fair["barrier"]
    assert scheduler.switch_count == 1
    b.done = True
    scheduler.running.clear()
    scheduler.requests.pop("b")
    scheduler.finished_req_ids.add("b")
    assert not scheduler.schedule().qwen_fair["barrier"]  # tool still executing
    scheduler.priorities.update(control(CHAT_B, 2, sequence=3, active=False))
    assert scheduler.schedule().qwen_fair["barrier"]
    assert scheduler.schedule().qwen_fair["bank"] == BANK_A
    assert scheduler.running == [a] and not scheduler.parked
    assert (a.num_computed_tokens, a.spec_token_ids, a.status, banks.managers[BANK_A]) == original
    assert not a.done and a.status.name == "RUNNING"
    assert scheduler.switch_count == 2


@pytest.mark.parametrize("blocked", ["in_flight", "restore", "slots"])
def test_immediate_handover_waits_for_safe_state_and_available_runner_slot(monkeypatch, blocked):
    module = load_module(monkeypatch)
    scheduler = tool_boundary(module)
    a = scheduler.response_request
    a.is_finished = lambda: False
    a.status.name = "RUNNING"
    a.num_in_flight_tokens = 1 if blocked == "in_flight" else 0
    scheduler.running = [a]
    if blocked == "restore":
        scheduler.running = []
        a.status.name = "WAITING_FOR_REMOTE_KVS"
        scheduler.skipped_waiting.append(a)
    if blocked == "slots":
        scheduler.runner_state_slots = 1
    scheduler.priorities.update(control(CHAT_B, 2))
    assert scheduler._choose() == BANK_A
    a.num_in_flight_tokens = 0
    a.status.name = "RUNNING"
    scheduler.runner_state_slots = 2
    assert scheduler._choose() == BANK_B


def test_expired_or_released_answer_cannot_hold_or_interrupt(monkeypatch):
    module = load_module(monkeypatch)
    now = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    levels = module.AnswerPriorities()
    value = control(priority=2)
    levels.update(value)
    assert levels.level(BANK_A) == levels.level(BANK_A_NEXT) == 2
    now[0] = 159.0
    levels.update(value)  # duplicate is idempotent, not a renewed lease
    now[0] = 160.0
    assert levels.level(BANK_A) == 0
    levels.update(control(priority=2, sequence=2))
    assert levels.level(BANK_A) == 2
    levels.update(control(priority=2, sequence=3, active=False))
    with pytest.raises(ValueError, match="stale"):
        levels.update(control(priority=2, sequence=2))
    assert levels.level(BANK_A) == 0


def test_another_window_cannot_replace_live_owner_or_release_its_answer(monkeypatch):
    module = load_module(monkeypatch)
    levels = module.AnswerPriorities()
    levels.update(control(priority=2))
    for active in (True, False):
        with pytest.raises(ValueError, match="another Pi"):
            levels.update({**control(active=active), "client": "e" * 32})
    assert levels.level(BANK_A) == 2


@pytest.mark.parametrize("owner", [0, 1, 2])
@pytest.mark.parametrize("incoming", [0, 1, 2])
def test_all_priority_pairs_during_generation(monkeypatch, owner, incoming):
    module = load_module(monkeypatch)
    scheduler = tool_boundary(module)
    a = scheduler.response_request
    a.is_finished = lambda: False
    a.status.name = "RUNNING"
    scheduler.running = [a]
    scheduler.priorities.update(control(CHAT_A, owner))
    scheduler.priorities.update(control(CHAT_B, incoming))
    assert scheduler._choose() == (BANK_B if incoming == 2 and incoming > owner else BANK_A)


@pytest.mark.parametrize("level", [1, 2])
def test_compaction_replaces_owned_generation_without_deadlocking_the_answer(monkeypatch, level):
    module = load_module(monkeypatch)
    scheduler = tool_boundary(module)
    scheduler.priorities.update(control(priority=level))
    successor = SimpleNamespace(
        request_id="a-next",
        is_finished=lambda: False,
        status=SimpleNamespace(name="WAITING"),
        kv_transfer_params={},
    )
    scheduler.banks.owners["a-next"] = BANK_A_NEXT
    scheduler.waiting.append(successor)
    scheduler.requests["a-next"] = successor
    # Old generation is still draining connector work: stay until it is safe.
    assert scheduler._choose() == BANK_A
    scheduler.requests.pop("a")
    scheduler.finished_req_ids.clear()
    assert scheduler._choose() == BANK_A_NEXT
    assert scheduler.response_request is successor
    assert scheduler.priorities.level(BANK_A_NEXT) == level


def test_killed_client_releases_the_waiting_chat_when_lease_expires(monkeypatch):
    module = load_module(monkeypatch)
    now = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    scheduler = tool_boundary(module)
    scheduler.priorities.update(control())
    assert scheduler._choose() == BANK_A
    now[0] = 160.0
    assert scheduler._choose() == BANK_B


def test_cancelling_a_parked_request_cannot_resume_it_later(monkeypatch):
    module = load_module(monkeypatch)
    scheduler = tool_boundary(module)
    request = scheduler.waiting.pop()
    scheduler.parked[BANK_B] = request

    def cancel(_self, rid):
        assert rid == request.request_id
        request.is_finished = lambda: True
        return [request]

    monkeypatch.setattr(module.Scheduler, "finish_requests", cancel, raising=False)
    assert scheduler.finish_requests(request.request_id) == [request]
    assert scheduler.parked == {}


@pytest.mark.parametrize(
    "field,value",
    [
        ("chat_id", "../other"),
        ("client", ""),
        ("answer", None),
        ("sequence", True),
        ("sequence", 0),
        ("sequence", 2**53),
        ("priority", True),
        ("priority", -1),
        ("priority", 3),
        ("priority", "2"),
        ("active", 1),
        ("prompt", "private"),
    ],
)
def test_priority_controls_reject_malformed_or_extra_data(monkeypatch, field, value):
    module = load_module(monkeypatch)
    with pytest.raises(ValueError):
        module.AnswerPriorities().update({**control(), field: value})

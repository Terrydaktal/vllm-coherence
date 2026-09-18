import io
import json
import threading
import urllib.error

import pytest

from qwen_r9700_lab.conformance_priority import PriorityLease
from qwen_r9700_lab.diagnostic_contract import DiagnosticError


class Reply(io.BytesIO):
    status = 200


class Client:
    def __init__(self):
        self.updates = []
        self.fail = set()
        self.block = None
        self.entered = threading.Event()
        self.proceed = threading.Event()
        self.bad_reply = False

    def request(self, path, body, *, timeout):
        assert path == "/qwen-radiance/priority"
        assert timeout == 6.5
        self.updates.append(dict(body))
        if self.block == (body["chat_id"], body["sequence"]):
            self.entered.set()
            assert self.proceed.wait(3), "test did not release blocked request"
        if body["sequence"] in self.fail:
            raise urllib.error.HTTPError(
                "http://127.0.0.1/priority", 503, "unconfirmed", {}, io.BytesIO(b"private error")
            )
        return Reply(
            json.dumps({"applied": not self.bad_reply, "priority": body["priority"]}).encode()
        )


@pytest.fixture
def leases(tmp_path):
    created = []

    def make(client=None, name="a"):
        value = PriorityLease(
            client or Client(), {"id": name * 64}, "f" * 64, tmp_path / name, automatic=False
        )
        created.append(value)
        return value

    yield make
    for value in created:
        value.client.proceed.set()
        value.close()


def test_default_chat_sends_no_start_heartbeat_or_release(leases):
    lease = leases()
    lease.start(0).result()
    for _ in range(3):
        lease.heartbeat().result()
    lease.release().result()
    assert lease.client.updates == []


def test_heartbeats_use_current_priority_until_explicit_change(leases):
    lease = leases()
    lease.start(0).result()
    lease.heartbeat().result()
    assert not lease.client.updates
    lease.change(1).result()
    lease.heartbeat().result()
    lease.change(2).result()
    lease.heartbeat().result()
    lease.change(0).result()
    lease.heartbeat().result()
    assert [u["priority"] for u in lease.client.updates] == [1, 1, 2, 2, 0]
    assert [u["sequence"] for u in lease.client.updates] == [1, 2, 3, 4, 5]


def test_failed_heartbeat_is_preserved_and_next_heartbeat_still_runs(leases):
    lease = leases()
    lease.start(1).result()
    lease.client.fail.add(2)
    with pytest.raises(urllib.error.HTTPError):
        lease.heartbeat().result()
    lease.heartbeat().result()
    lease.release().result()
    assert [u["sequence"] for u in lease.client.updates] == [1, 2, 3, 4]
    assert len(lease.failures) == 1
    failure = lease.failures[0]
    assert failure["purpose"] == "heartbeat"
    assert failure["http_status"] == 503 and failure["error_type"] == "HTTPError"
    assert failure["finished_ns"] >= failure["started_ns"] >= failure["queued_ns"]
    receipt = (lease.root / "00002.result.json").read_text()
    assert "private error" not in receipt
    assert "error_body_sha256" in receipt
    assert json.loads(receipt) == failure


def test_release_is_ordered_after_inflight_heartbeat_and_disables_new_ones(leases):
    lease = leases()
    lease.start(1).result()
    lease.client.block = (lease.chat["id"], 2)
    heartbeat = lease.heartbeat()
    assert lease.client.entered.wait(1)
    # Pi skips ticks while a control request is pending.
    assert lease.heartbeat().result() is None
    release = lease.release()
    assert lease.heartbeat().result() is None
    lease.client.proceed.set()
    heartbeat.result(timeout=1)
    release.result(timeout=1)
    assert [u["active"] for u in lease.client.updates] == [True, True, False]
    assert [u["sequence"] for u in lease.client.updates] == [1, 2, 3]


def test_blocked_chat_control_does_not_block_another_chat(leases):
    client = Client()
    a, b = leases(client, "a"), leases(client, "b")
    client.block = (a.chat["id"], 1)
    started = a.start(1)
    assert client.entered.wait(1)
    b.start(2).result(timeout=1)
    assert not started.done()
    client.proceed.set()
    started.result(timeout=1)


def test_false_confirmation_cannot_be_promoted_to_success(leases):
    lease = leases()
    lease.client.bad_reply = True
    with pytest.raises(DiagnosticError, match="not applied"):
        lease.start(1).result()
    assert not lease.confirmations
    assert lease.failures[0]["http_status"] == 200
    assert lease.failures[0]["status"] == "failed"


@pytest.mark.parametrize("priority", [True, False, -1, 3, 1.0, "1", None])
def test_invalid_levels_never_send_control(leases, priority):
    lease = leases()
    with pytest.raises(ValueError):
        lease.start(priority)
    with pytest.raises(ValueError):
        lease.change(priority)
    assert not lease.client.updates

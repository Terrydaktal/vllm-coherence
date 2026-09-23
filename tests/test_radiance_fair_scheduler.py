from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/radiance-public/radiance_fair_scheduler.py"
CHAT_A = "a" * 64
CHAT_B = "b" * 64
GEN_A = "1" * 64
GEN_B = "2" * 64
GEN_A_NEXT = "3" * 64
BANK_A = f"{CHAT_A}:{GEN_A}"
BANK_B = f"{CHAT_B}:{GEN_B}"
BANK_A_NEXT = f"{CHAT_A}:{GEN_A_NEXT}"


def load_module(monkeypatch):
    scheduler_module = ModuleType("vllm.v1.core.sched.scheduler")

    class Scheduler:
        def _build_kv_connector_meta(self, connector, scheduler_output):
            return connector(scheduler_output)

        def update_from_output(self, scheduler_output, model_runner_output):
            del scheduler_output
            if model_runner_output.finish:
                self.requests.pop(model_runner_output.request_id)
            return "updated"

    scheduler_module.Scheduler = Scheduler
    manager_module = ModuleType("vllm.v1.core.kv_cache_manager")
    manager_module.KVCacheManager = type("KVCacheManager", (), {})
    for name in ("vllm", "vllm.v1", "vllm.v1.core", "vllm.v1.core.sched"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, scheduler_module.__name__, scheduler_module)
    monkeypatch.setitem(sys.modules, manager_module.__name__, manager_module)
    spec = importlib.util.spec_from_file_location("radiance_fair_scheduler_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def new_scheduler(module):
    instance = module.FairScheduler.__new__(module.FairScheduler)
    instance.tool_handover = module.ToolHandover()
    instance.max_tool_deferral_seconds = 30.0
    instance.queued_at = {}
    instance.grace_status = None
    instance.priorities = module.AnswerPriorities()
    instance.priority_hold = None
    instance.runner_state_slots = 2
    return instance


@pytest.mark.parametrize("lookahead", [None, 1])
def test_new_chat_bank_preserves_release_prefill_lookahead(monkeypatch, lookahead):
    module = load_module(monkeypatch)

    def initialize(self):
        self.vllm_config = SimpleNamespace(additional_config={}, max_in_flight_tokens=2048)
        self.scheduler_config = SimpleNamespace(async_scheduling=False, watermark=0.01)
        self.parallel_config = SimpleNamespace(world_size=1)
        self.max_num_running_reqs = 2
        self.use_v2_model_runner = True
        self.defer_block_free = False
        self.kv_cache_manager = object()
        self.kv_cache_config = object()
        self.max_model_len = 253792
        self.cache_config = SimpleNamespace(enable_prefix_caching=True)
        self.use_eagle = True
        self.log_stats = False
        self.enable_kv_cache_events = False
        self.dcp_world_size = 1
        self.block_size = self.hash_block_size = 16
        self.kv_metrics_collector = None
        if lookahead is not None:
            self.num_prefill_lookahead = lookahead

    monkeypatch.setattr(module.Scheduler, "__init__", initialize)
    monkeypatch.setattr(module.FairScheduler, "_publish_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "KVCacheManager", lambda **kwargs: kwargs)
    scheduler = module.FairScheduler()
    scheduler.banks.activate(BANK_A)
    scheduler.banks.activate(BANK_B)
    manager = scheduler.banks.managers[BANK_B]
    if lookahead is None:
        assert "num_prefill_lookahead" not in manager
    else:
        assert manager["num_prefill_lookahead"] == lookahead


def tool_boundary(module):
    scheduler = new_scheduler(module)
    completed = SimpleNamespace(
        request_id="a",
        is_finished=lambda: True,
        status=SimpleNamespace(name="FINISHED_STOPPED"),
        kv_transfer_params={module.HANDOVER_TOKEN: "1" * 32},
    )
    waiting = SimpleNamespace(
        request_id="b",
        is_finished=lambda: False,
        status=SimpleNamespace(name="WAITING"),
        kv_transfer_params={},
    )
    scheduler.banks = SimpleNamespace(
        active=BANK_A,
        owners={"a": BANK_A, "b": BANK_B},
        managers={BANK_A: object(), BANK_B: object()},
    )
    scheduler.running = []
    scheduler.response_request = completed
    scheduler.tool_handover.select(completed)
    scheduler.tool_handover.finish()
    scheduler.parked = {}
    scheduler.waiting = [waiting]
    scheduler.skipped_waiting = []
    scheduler.requests = {"a": completed, "b": waiting}
    scheduler.finished_req_ids = {"a"}
    scheduler.last_served = {BANK_A: 100.0}
    scheduler.queued_at = {"b": 99.0}
    scheduler.max_banks = 2
    scheduler.drop_banks = []
    return scheduler


def test_short_tool_continues_without_a_cache_swap(monkeypatch):
    module = load_module(monkeypatch)
    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
    scheduler = tool_boundary(module)
    assert scheduler.response_outcome("1" * 32, True)
    assert scheduler._choose() == BANK_A
    assert scheduler.grace_status["phase"] == "tool_grace"
    assert scheduler.grace_status["remaining_seconds"] == 2
    continuation = SimpleNamespace(
        request_id="a2", is_finished=lambda: False, kv_transfer_params={}
    )
    scheduler.banks.owners["a2"] = BANK_A
    scheduler.waiting.append(continuation)
    scheduler.requests["a2"] = continuation
    monkeypatch.setattr(module.time, "monotonic", lambda: 100.4)
    assert scheduler._choose() == BANK_A
    assert scheduler.response_request is continuation
    assert scheduler.grace_status is None
    assert scheduler.queued_at["b"] == 99


def test_long_tool_hands_over_at_two_seconds_and_cannot_reclaim_gpu(monkeypatch):
    module = load_module(monkeypatch)
    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
    scheduler = tool_boundary(module)
    assert scheduler.response_outcome("1" * 32, True)
    monkeypatch.setattr(module.time, "monotonic", lambda: 101.999)
    assert scheduler._choose() == BANK_A
    # A duplicate parser notification cannot extend the grace period.
    assert scheduler.response_outcome("1" * 32, True)
    monkeypatch.setattr(module.time, "monotonic", lambda: 102.0)
    assert scheduler._choose() == BANK_B
    scheduler.banks.active = BANK_B
    assert scheduler.response_request.request_id == "b"
    assert scheduler.grace_status is None
    continuation = SimpleNamespace(request_id="a2", is_finished=lambda: False)
    scheduler.banks.owners["a2"] = BANK_A
    scheduler.waiting.append(continuation)
    assert not scheduler.response_outcome("1" * 32, True)
    assert scheduler._choose() == BANK_B


@pytest.mark.parametrize(
    "finish", ["final", "FINISHED_ABORTED", "FINISHED_LENGTH_CAPPED", "FINISHED_ERROR"]
)
def test_final_answer_cancellation_and_failure_skip_tool_grace(monkeypatch, finish):
    module = load_module(monkeypatch)
    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
    scheduler = tool_boundary(module)
    if finish == "final":
        assert scheduler.response_outcome("1" * 32, False)
    else:
        scheduler.response_request.status.name = finish
    assert scheduler._choose() == BANK_B
    assert scheduler.grace_status is None


def test_missing_parser_acknowledgement_has_a_bounded_fallback(monkeypatch):
    module = load_module(monkeypatch)
    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
    scheduler = tool_boundary(module)
    assert scheduler._choose() == BANK_A
    assert scheduler.grace_status["phase"] == "response_outcome"
    monkeypatch.setattr(module.time, "monotonic", lambda: 100.25)
    assert scheduler._choose() == BANK_B


def test_late_acknowledgement_cannot_rearm_grace_while_engine_was_idle(monkeypatch):
    module = load_module(monkeypatch)
    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
    scheduler = tool_boundary(module)
    monkeypatch.setattr(module.time, "monotonic", lambda: 103.0)
    assert scheduler.response_outcome("1" * 32, True)
    assert scheduler._choose() == BANK_B


def test_overdue_chat_gets_next_boundary_even_when_tool_returns_quickly(monkeypatch):
    module = load_module(monkeypatch)
    monkeypatch.setattr(module.time, "monotonic", lambda: 130.0)
    scheduler = tool_boundary(module)
    scheduler.queued_at["b"] = 100.0
    scheduler.response_outcome("1" * 32, True)
    continuation = SimpleNamespace(
        request_id="a2", is_finished=lambda: False, status=SimpleNamespace(name="WAITING")
    )
    scheduler.banks.owners["a2"] = BANK_A
    scheduler.waiting.append(continuation)
    assert scheduler._choose() == BANK_B
    assert scheduler.grace_status is None


@pytest.mark.parametrize(
    "disabled", [None, "no_tools", "no_chat", "multiple_choices", "tool_none", "beam"]
)
def test_only_supported_tool_requests_register_for_grace(monkeypatch, disabled):
    module = load_module(monkeypatch)
    request = SimpleNamespace(
        kv_transfer_params={
            "qwen_chat": {"id": CHAT_A, "generation": GEN_A},
            module.HANDOVER_TOKEN: "untrusted-client-value",
        },
        tools=[object()],
        tool_choice="auto",
        n=1,
        use_beam_search=False,
    )
    if disabled == "no_tools":
        request.tools = []
    if disabled == "no_chat":
        request.kv_transfer_params.pop("qwen_chat")
    if disabled == "multiple_choices":
        request.n = 2
    if disabled == "tool_none":
        request.tool_choice = "none"
    if disabled == "beam":
        request.use_beam_search = True
    module.prepare_tool_handover(request)
    params = request.kv_transfer_params or {}
    if disabled:
        assert module.HANDOVER_TOKEN not in params
    else:
        token = params[module.HANDOVER_TOKEN]
        assert len(token) == 32 and token != "untrusted-client-value"
        module.prepare_tool_handover(request)
        assert request.kv_transfer_params[module.HANDOVER_TOKEN] != token


def test_parser_outcome_uses_existing_ipc_without_blocking_response(monkeypatch):
    module = load_module(monkeypatch)

    async def exercise():
        calls = []
        release = asyncio.Event()

        async def utility(*args):
            calls.append(args)
            await release.wait()

        client = SimpleNamespace(engine_core=SimpleNamespace(call_utility_async=utility))
        request = SimpleNamespace(kv_transfer_params={module.HANDOVER_TOKEN: "1" * 32})
        assert module.report_tool_handover(client, request, True) is None
        tasks = list(module._outcome_tasks)
        assert len(tasks) == 1 and not tasks[0].done()
        release.set()
        await asyncio.gather(*tasks)
        assert calls == [("qwen_response_outcome", "1" * 32, True)]
        assert not module._outcome_tasks

    asyncio.run(exercise())


class ByteView:
    def __init__(self, data, begin, end):
        self.data = data
        self.begin = begin
        self.end = end

    def copy_(self, source, non_blocking=False):
        del non_blocking
        values = source.values() if isinstance(source, ByteView) else list(source)
        assert len(values) == self.end - self.begin
        self.data[self.begin : self.end] = values
        return self

    def values(self):
        return self.data[self.begin : self.end]

    def __iter__(self):
        return iter(self.values())


class ByteTensor:
    def __init__(self, values):
        self.data = list(values)

    def __getitem__(self, item):
        assert isinstance(item, slice) and item.step is None
        return ByteView(self.data, item.start or 0, item.stop)

    def numel(self):
        return len(self.data)


@pytest.fixture
def constructed_worker(tmp_path, monkeypatch):
    module = load_module(monkeypatch)
    allocations = []
    memory = {"available": 16 * 1024**3}
    original_read_text = Path.read_text

    def read_text(path, *args, **kwargs):
        if path == Path("/proc/meminfo"):
            return f"MemAvailable: {memory['available'] // 1024} kB\n"
        return original_read_text(path, *args, **kwargs)

    class StorageTensor(ByteTensor):
        def untyped_storage(self):
            return self

        def data_ptr(self):
            return id(self.data)

        def set_(self, storage):
            self.data = storage.data
            return self

    def empty(amount, **kwargs):
        if kwargs["device"] == "cpu":
            assert kwargs["pin_memory"] is True
            allocations.append(amount)
        return StorageTensor([0] * amount)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            Tensor=StorageTensor,
            uint8=object(),
            empty=empty,
            cuda=SimpleNamespace(synchronize=lambda: None),
        ),
    )
    gpu = StorageTensor(range(32))
    runner = SimpleNamespace(
        kv_caches=[gpu],
        kv_cache_config=SimpleNamespace(num_blocks=8),
        device="cuda",
    )

    def create():
        return module.WorkerBanks(runner, {"max_banks": 2, "status_path": str(tmp_path / "fair")})

    return SimpleNamespace(create=create, memory=memory, allocations=allocations, gpu=gpu)


@pytest.mark.parametrize("available", [0, 8 * 1024**3 - 1024])
def test_single_chat_and_compaction_do_not_require_unused_ram_arena(constructed_worker, available):
    fixture = constructed_worker
    fixture.memory["available"] = available
    worker = fixture.create()
    worker.before({"bank": BANK_A, "save_blocks": [], "drop_banks": [], "barrier": False})
    worker.before(
        {
            "bank": BANK_A_NEXT,
            "save_blocks": [],
            "drop_banks": [BANK_A],
            "barrier": False,
            "discard_active": True,
        }
    )

    assert worker.active == BANK_A_NEXT
    assert fixture.allocations == []
    assert fixture.gpu.data == list(range(32))
    assert worker.allocated_bytes == 0
    assert worker.stage is None
    assert worker.images == {}


def test_second_chat_checks_current_ram_before_allocating_and_can_retry(constructed_worker):
    fixture = constructed_worker
    worker = fixture.create()
    worker.before({"bank": BANK_A, "save_blocks": [], "drop_banks": [], "barrier": False})
    handover = {"bank": BANK_B, "save_blocks": [0, 1], "drop_banks": [], "barrier": False}
    fixture.memory["available"] = 8 * 1024**3

    with pytest.raises(MemoryError, match="8 GiB system headroom"):
        worker.before(handover)

    assert worker.active == BANK_A
    assert fixture.allocations == []
    assert fixture.gpu.data == list(range(32))
    assert worker.images == {}
    assert worker.free_buffers == []
    assert worker.stage is None
    assert worker.allocated_bytes == 0
    assert worker.allocation_events == 0

    fixture.memory["available"] += 1024
    worker.before(handover)
    assert worker.active == BANK_B
    assert fixture.allocations == [32, 32]
    assert worker.images[BANK_A]["buffer"].data[:8] == list(range(8))
    assert worker.allocated_bytes == 64
    assert worker.allocation_events == 1

    fixture.memory["available"] = 0
    assert worker._ensure_buffers() == (0, 0.0)
    assert fixture.allocations == [32, 32]


def test_worker_copies_each_unique_storage_region_into_one_ram_image(monkeypatch):
    module = load_module(monkeypatch)
    worker = module.WorkerBanks.__new__(module.WorkerBanks)
    first = ByteTensor(range(8))
    second = ByteTensor(range(20, 24))
    worker.regions = [(first, 0, 2), (second, 8, 1)]
    image = ByteTensor([0] * 12)

    assert worker._copy_to_buffer(image, [(1, 3)]) == 6
    assert image.data == [0, 0, 2, 3, 4, 5, 0, 0, 0, 21, 22, 0]
    first.data = [0] * 8
    second.data = [0] * 4
    assert worker._copy_from_buffer(image, [(1, 3)]) == 6
    assert first.data == [0, 0, 2, 3, 4, 5, 0, 0]
    assert second.data == [0, 21, 22, 0]


def test_worker_uses_one_inactive_image_and_round_trips_overlapping_pages(tmp_path, monkeypatch):
    module = load_module(monkeypatch)
    worker = module.WorkerBanks.__new__(module.WorkerBanks)
    worker.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
    worker.runner = SimpleNamespace()
    worker.active = BANK_A
    worker.images = {}
    worker.transferred_bytes = 0
    worker.transfer_seconds = 0.0
    worker.allocation_events = 0
    worker.allocation_seconds = 0.0
    worker.generation_replacements = 0
    worker.switches = 0
    worker.status_path = str(tmp_path / "fair")
    worker.capacity = 32
    worker.stage_capacity = 8
    worker.stage = ByteTensor([0] * 8)
    worker.free_buffers = [ByteTensor([0] * 32)]
    worker.allocated_bytes = 40
    worker.reserved_capacity_bytes = 40
    gpu = ByteTensor(range(32))
    worker.regions = [(gpu, 0, 4)]

    worker.before(
        {
            "bank": BANK_B,
            "save_blocks": [0, 1, 4],
            "drop_banks": [],
            "barrier": False,
        }
    )
    assert worker.active == BANK_B
    assert set(worker.images) == {BANK_A}
    assert not worker.free_buffers
    assert worker.images[BANK_A]["buffer"].data[:8] == list(range(8))
    assert worker.images[BANK_A]["buffer"].data[16:20] == list(range(16, 20))
    status = json.loads((tmp_path / "fair-worker.json").read_text())
    assert status["residency"] == {
        "active": {"chat_id": CHAT_B, "generation": GEN_B},
        "images": [
            {
                "chat_id": CHAT_A,
                "generation": GEN_A,
                "data_bytes": 12,
                "allocated_bytes": 32,
            }
        ],
        "free_buffer_bytes": 0,
        "staging_buffer_bytes": 8,
    }

    # Simulate B writing live blocks, including pages overlapping A's image.
    gpu.data = [100 + value for value in range(32)]
    worker.before(
        {
            "bank": BANK_A,
            "save_blocks": [1, 2, 5],
            "drop_banks": [],
            "barrier": False,
        }
    )
    assert worker.active == BANK_A
    assert set(worker.images) == {BANK_B}
    assert gpu.data[:8] == list(range(8))
    assert gpu.data[16:20] == list(range(16, 20))
    image_b = worker.images[BANK_B]["buffer"].data
    assert image_b[4:12] == [104, 105, 106, 107, 108, 109, 110, 111]
    assert image_b[20:24] == [120, 121, 122, 123]
    assert worker.switches == 2


def test_first_gpu_bank_publishes_zero_allocation_residency(tmp_path, monkeypatch):
    module = load_module(monkeypatch)
    worker = module.WorkerBanks.__new__(module.WorkerBanks)
    worker.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
    worker.active = None
    worker.images = {}
    worker.free_buffers = []
    worker.stage = None
    worker.transferred_bytes = 0
    worker.transfer_seconds = 0.0
    worker.allocation_events = 0
    worker.allocation_seconds = 0.0
    worker.generation_replacements = 0
    worker.switches = 0
    worker.status_path = str(tmp_path / "fair")
    worker.capacity = 32
    worker.stage_capacity = 8
    worker.allocated_bytes = 0
    worker.reserved_capacity_bytes = 40
    worker.regions = [(ByteTensor(range(32)), 0, 4)]

    worker.before(
        {
            "bank": BANK_A,
            "save_blocks": [],
            "drop_banks": [],
            "barrier": False,
        }
    )

    status = json.loads((tmp_path / "fair-worker.json").read_text())
    assert status["allocated_bytes"] == 0
    assert status["cached_chats"] == 1
    assert status["residency"] == {
        "active": {"chat_id": CHAT_A, "generation": GEN_A},
        "images": [],
        "free_buffer_bytes": 0,
        "staging_buffer_bytes": 0,
    }


def test_same_chat_successor_discards_gpu_bank_without_allocating_ram(tmp_path, monkeypatch):
    module = load_module(monkeypatch)
    worker = module.WorkerBanks.__new__(module.WorkerBanks)
    worker.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
    worker.active = BANK_A
    worker.images = {}
    worker.free_buffers = []
    worker.stage = None
    worker.transferred_bytes = 0
    worker.transfer_seconds = 0.0
    worker.allocation_events = 0
    worker.allocation_seconds = 0.0
    worker.generation_replacements = 0
    worker.switches = 0
    worker.status_path = str(tmp_path / "fair")
    worker.capacity = 32
    worker.stage_capacity = 8
    worker.allocated_bytes = 0
    worker.reserved_capacity_bytes = 40
    worker.regions = [(ByteTensor(range(32)), 0, 4)]

    def unexpected_allocation():
        raise AssertionError("same-chat replacement allocated pinned RAM")

    worker._ensure_buffers = unexpected_allocation
    worker.before(
        {
            "bank": BANK_A_NEXT,
            "save_blocks": [],
            "drop_banks": [BANK_A],
            "barrier": False,
            "discard_active": True,
        }
    )

    status = json.loads((tmp_path / "fair-worker.json").read_text())
    assert worker.active == BANK_A_NEXT
    assert worker.images == {}
    assert worker.free_buffers == []
    assert worker.stage is None
    assert status["allocated_bytes"] == 0
    assert status["last_allocation_bytes"] == 0
    assert status["last_transfer_bytes"] == 0
    assert status["generation_replacements"] == 1
    assert status["last_handover"] == "generation-replace"


def test_worker_times_pinned_allocation_separately_from_gpu_transfer(tmp_path, monkeypatch):
    module = load_module(monkeypatch)
    worker = module.WorkerBanks.__new__(module.WorkerBanks)
    original_read_text = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *args, **kwargs: (
            "MemAvailable: 16777216 kB\n"
            if path == Path("/proc/meminfo")
            else original_read_text(path, *args, **kwargs)
        ),
    )
    clock = iter((10.0, 12.5, 20.0, 20.4))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    worker.torch = SimpleNamespace(
        uint8=object(),
        cuda=SimpleNamespace(synchronize=lambda: None),
        empty=lambda amount, **kwargs: ByteTensor([0] * amount),
    )
    worker.active = BANK_A
    worker.images = {}
    worker.free_buffers = []
    worker.stage = None
    worker.transferred_bytes = 0
    worker.transfer_seconds = 0.0
    worker.allocation_events = 0
    worker.allocation_seconds = 0.0
    worker.generation_replacements = 0
    worker.switches = 0
    worker.status_path = str(tmp_path / "fair")
    worker.capacity = 32
    worker.max_banks = 2
    worker.stage_capacity = 8
    worker.allocated_bytes = 0
    worker.reserved_capacity_bytes = 40
    worker.regions = [(ByteTensor(range(32)), 0, 4)]

    worker.before(
        {
            "bank": BANK_B,
            "save_blocks": [0, 1],
            "drop_banks": [],
            "barrier": False,
        }
    )

    status = json.loads((tmp_path / "fair-worker.json").read_text())
    assert status["last_allocation_bytes"] == 40
    assert status["last_allocation_seconds"] == 2.5
    assert status["allocation_events"] == 1
    assert status["allocation_seconds"] == 2.5
    assert status["last_transfer_bytes"] == 8
    assert abs(status["last_transfer_seconds"] - 0.4) < 1e-9
    assert status["last_handover"] == "swap"


def test_cache_banks_route_request_operations_to_their_owner(monkeypatch):
    module = load_module(monkeypatch)

    class Manager:
        def __init__(self, name):
            self.name = name
            self.calls = []
            self.block_pool = SimpleNamespace(blocks=[], cached_block_hashes_by_block={})

        def allocate_slots(self, request, count):
            self.calls.append((request.request_id, count))
            return self.name

        def remove_skipped_blocks(self, request_id, processed_computed_tokens):
            self.calls.append((request_id, processed_computed_tokens))

        def reset_prefix_cache(self):
            self.calls.append("reset")
            return True

        def take_events(self):
            return [self.name]

    made = []

    def factory():
        manager = Manager(f"manager-{len(made) + 2}")
        made.append(manager)
        return manager

    initial = Manager("manager-1")
    banks = module.CacheBanks(initial, factory)
    req_a = SimpleNamespace(request_id="a")
    req_b = SimpleNamespace(request_id="b")
    banks.owners = {"a": "chat-a", "b": "chat-b"}
    banks.activate("chat-a")
    banks.activate("chat-b")

    assert banks.allocate_slots(req_a, 3) == "manager-1"
    assert banks.allocate_slots(req_b, 4) == "manager-2"
    banks.remove_skipped_blocks(request_id="a", processed_computed_tokens=11)
    assert initial.calls[-1] == ("a", 11)
    assert banks.reset_prefix_cache()
    assert banks.take_events() == ["manager-1", "manager-2"]


def test_scheduler_handover_after_completion_has_a_flush_barrier_and_preserves_cache(
    monkeypatch,
):
    module = load_module(monkeypatch)
    scheduler = new_scheduler(module)

    class Request:
        def __init__(self, request_id):
            self.request_id = request_id

        def is_finished(self):
            return False

    req_b = Request("b")
    manager = SimpleNamespace(
        block_pool=SimpleNamespace(
            blocks=[SimpleNamespace(block_id=7, ref_cnt=1, block_hash=None)],
            cached_block_hashes_by_block={},
        )
    )
    banks = SimpleNamespace(
        active="chat-a",
        owners={"a": "chat-a", "b": "chat-b"},
        managers={"chat-a": manager, "chat-b": manager},
        activate=lambda key: setattr(banks, "active", key),
        live_blocks=lambda: [7],
    )
    scheduler.banks = banks
    scheduler.running = []
    scheduler.parked = {"chat-b": req_b}
    scheduler.pending_switch = None
    scheduler.pending_discard = False
    scheduler.drop_banks = []
    scheduler.max_banks = 2
    scheduler.status_path = "/dev/shm/test-radiance-fair"
    scheduler.last_served = {}
    scheduler.switch_count = 0
    scheduler._publish_status = lambda *args, **kwargs: None
    outputs = []

    def parent_step(throttle_prefills=False, *, barrier=False):
        output = SimpleNamespace(qwen_fair=None)
        outputs.append((throttle_prefills, barrier, output))
        return output

    scheduler._parent_step = parent_step
    scheduler._choose = lambda: "chat-b"

    barrier = scheduler.schedule(True)
    assert outputs[-1][:2] == (True, True)
    assert scheduler.pending_switch == "chat-b"
    assert barrier.qwen_fair["barrier"]
    assert scheduler.running == []
    seen = scheduler._build_kv_connector_meta(
        lambda output: output.qwen_fair, SimpleNamespace(qwen_fair=None)
    )
    assert seen["barrier"]

    switched = scheduler.schedule(True)
    assert outputs[-1][:2] == (True, False)
    assert scheduler.running == [req_b]
    assert scheduler.parked == {}
    assert switched.qwen_fair["save_blocks"] == [7]
    assert switched.qwen_fair["bank"] == "chat-b"
    assert not switched.qwen_fair["discard_active"]


@pytest.mark.parametrize("computed_tokens", [0, 512, 1024, 12000])
@pytest.mark.parametrize("admission_delayed", [False, True])
@pytest.mark.parametrize("continuation_during_barrier", [False, True])
def test_scheduler_keeps_prefill_and_all_output_on_gpu_until_response_finishes(
    monkeypatch, computed_tokens, admission_delayed, continuation_during_barrier
):
    module = load_module(monkeypatch)
    scheduler = new_scheduler(module)

    class Request:
        def __init__(self, request_id):
            self.request_id = request_id
            self.status = SimpleNamespace(name="WAITING")
            self.num_prompt_tokens = 1024
            self.num_computed_tokens = 0
            self.finished = False

        def is_finished(self):
            return self.finished

    req_a, req_b, continuation_a = (Request(rid) for rid in ("a", "b", "a-tool-result"))
    req_a.num_computed_tokens = computed_tokens
    banks = SimpleNamespace(
        active=BANK_A,
        owners={"a": BANK_A, "b": BANK_B, "a-tool-result": BANK_A},
        managers={BANK_A: object(), BANK_B: object()},
        activate=lambda key: setattr(banks, "active", key),
        live_blocks=lambda: [7, 8],
    )
    scheduler.banks = banks
    scheduler.requests = {r.request_id: r for r in (req_a, req_b)}
    scheduler.running = [req_a]
    scheduler.waiting = [req_b]
    scheduler.skipped_waiting = []
    scheduler.parked = {}
    scheduler.response_request = None
    scheduler.finished_req_ids = set()
    scheduler.last_served = {BANK_A: 10.0}
    scheduler.max_banks = 2
    scheduler.pending_switch = None
    scheduler.pending_discard = False
    scheduler.drop_banks = []
    scheduler.switch_count = 0
    scheduler.status_path = "/unused-synthetic-status"
    scheduler._publish_status = lambda **kwargs: None

    lookup_pending = admission_delayed

    def parent_step(throttle_prefills=False, *, barrier=False):
        scheduler.finished_req_ids.clear()
        if not barrier and not scheduler.running:
            for request in scheduler.waiting:
                if scheduler._key(request) == banks.active:
                    # vLLM leaves the request WAITING when connector lookup
                    # returns ext_tokens=None, before WAITING_FOR_REMOTE_KVS.
                    if request is req_b and lookup_pending:
                        break
                    scheduler.waiting.remove(request)
                    scheduler.running.append(request)
                    break
        return SimpleNamespace(qwen_fair=None)

    scheduler._parent_step = parent_step
    for elapsed in (15, 30, 120, 3600):
        monkeypatch.setattr(module.time, "monotonic", lambda elapsed=elapsed: elapsed + 10)
        output = scheduler.schedule()
        assert scheduler.running == [req_a]
        assert output.qwen_fair["bank"] == BANK_A
        assert not output.qwen_fair["barrier"]
        assert scheduler.switch_count == 0
        assert req_a.num_computed_tokens == computed_tokens

    # Emitting a complete tool call releases the GPU immediately, while the
    # client is still executing that tool and has sent no follow-up request.
    req_a.finished = True
    scheduler.running.clear()
    scheduler.requests.pop("a")
    scheduler.finished_req_ids.add("a")
    assert scheduler.schedule().qwen_fair["barrier"]
    assert scheduler.pending_switch == BANK_B
    if continuation_during_barrier:
        scheduler.waiting.append(continuation_a)
        scheduler.requests[continuation_a.request_id] = continuation_a
    switched = scheduler.schedule()
    assert scheduler.running == ([] if admission_delayed else [req_b])
    assert switched.qwen_fair["save_blocks"] == [7, 8]
    assert scheduler.switch_count == 1
    if admission_delayed:
        assert BANK_B not in scheduler.last_served

    # A subsecond tool result arrives while B is still doing asynchronous
    # lookup. Neither elapsed time nor A's return may reverse the paid swap.
    if not continuation_during_barrier:
        scheduler.waiting.append(continuation_a)
        scheduler.requests[continuation_a.request_id] = continuation_a
    if admission_delayed:
        # A deferred lookup moves into vLLM's skipped queue while still WAITING.
        scheduler.waiting.remove(req_b)
        scheduler.skipped_waiting.append(req_b)
    for now in (3610.1, 3610.2, 3611, 3620, 8000):
        monkeypatch.setattr(module.time, "monotonic", lambda now=now: now)
        output = scheduler.schedule()
        assert output.qwen_fair["bank"] == BANK_B
        assert not output.qwen_fair["barrier"]
        assert scheduler.pending_switch is None
        assert scheduler.switch_count == 1
    lookup_pending = False
    if admission_delayed:
        scheduler.skipped_waiting.remove(req_b)
        scheduler.waiting.append(req_b)
    scheduler.schedule()
    assert scheduler.running == [req_b]
    assert scheduler.switch_count == 1
    req_b.finished = True
    scheduler.running.clear()
    scheduler.requests.pop("b")
    scheduler.finished_req_ids.add("b")
    assert scheduler.schedule().qwen_fair["barrier"]
    assert scheduler.schedule().qwen_fair["bank"] == BANK_A
    assert scheduler.running == [continuation_a]
    assert scheduler.switch_count == 2
    assert set(banks.managers) == {BANK_A, BANK_B}
    assert not scheduler.parked


def test_pending_handover_cannot_park_a_live_response(monkeypatch):
    module = load_module(monkeypatch)
    scheduler = new_scheduler(module)
    scheduler.running = [object()]
    scheduler.pending_switch = BANK_B
    with pytest.raises(RuntimeError, match="before the current response finishes"):
        scheduler.schedule()


@pytest.mark.parametrize("status", ["WAITING_FOR_REMOTE_KVS", "PREEMPTED"])
def test_restore_or_memory_retry_is_not_a_response_boundary(monkeypatch, status):
    module = load_module(monkeypatch)
    scheduler = new_scheduler(module)
    scheduler.banks = SimpleNamespace(active=BANK_A, owners={"a": BANK_A, "b": BANK_B})
    scheduler.parked = {}
    scheduler.response_request = None
    scheduler.running = []
    scheduler.waiting = [
        SimpleNamespace(request_id="a", status=SimpleNamespace(name=status)),
        SimpleNamespace(request_id="b", status=SimpleNamespace(name="WAITING")),
    ]
    scheduler.skipped_waiting = []
    assert scheduler._choose() == BANK_A


@pytest.mark.parametrize("retained_for_cleanup", [False, True])
def test_finished_or_cancelled_reservation_releases_gpu_even_during_connector_cleanup(
    monkeypatch, retained_for_cleanup
):
    module = load_module(monkeypatch)
    scheduler = new_scheduler(module)
    completed = SimpleNamespace(request_id="a", is_finished=lambda: True)
    next_request = SimpleNamespace(
        request_id="b", is_finished=lambda: False, status=SimpleNamespace(name="WAITING")
    )
    scheduler.banks = SimpleNamespace(
        active=BANK_A,
        owners={"a": BANK_A, "b": BANK_B},
        managers={BANK_A: object(), BANK_B: object()},
    )
    scheduler.running = []
    scheduler.response_request = completed
    scheduler.parked = {}
    scheduler.waiting = [next_request]
    scheduler.skipped_waiting = []
    scheduler.requests = {"b": next_request}
    if retained_for_cleanup:
        scheduler.requests["a"] = completed
    scheduler.finished_req_ids = {"a"}
    scheduler.last_served = {BANK_A: 10.0}
    scheduler.max_banks = 2
    scheduler.drop_banks = []

    assert scheduler._choose() == BANK_B
    assert scheduler.response_request is next_request


def test_scheduler_replaces_finished_generation_without_saving_its_gpu_bank(monkeypatch):
    module = load_module(monkeypatch)
    scheduler = new_scheduler(module)

    class Request:
        request_id = "new"

        def is_finished(self):
            return False

    request = Request()

    def unexpected_live_blocks():
        raise AssertionError("superseded generation was read for a RAM save")

    banks = SimpleNamespace(
        active=BANK_A,
        owners={"old": BANK_A, "new": BANK_A_NEXT},
        managers={BANK_A: object()},
        activate=lambda key: (
            banks.managers.setdefault(key, object()),
            setattr(banks, "active", key),
        ),
        live_blocks=unexpected_live_blocks,
    )
    scheduler.banks = banks
    scheduler.requests = {"new": request}
    scheduler.finished_req_ids = set()
    scheduler.running = []
    scheduler.parked = {}
    scheduler.pending_switch = BANK_A_NEXT
    scheduler.pending_discard = True
    scheduler.drop_banks = []
    scheduler.max_banks = 2
    scheduler.status_path = "/dev/shm/test-radiance-fair"
    scheduler.last_served = {BANK_A: 1}
    scheduler.switch_count = 0
    scheduler._publish_status = lambda *args, **kwargs: None
    scheduler._parent_step = lambda *args, **kwargs: SimpleNamespace(qwen_fair=None)

    output = scheduler.schedule()

    assert scheduler.banks.active == BANK_A_NEXT
    assert set(scheduler.banks.managers) == {BANK_A_NEXT}
    assert scheduler.banks.owners == {"new": BANK_A_NEXT}
    assert output.qwen_fair["save_blocks"] == []
    assert output.qwen_fair["drop_banks"] == [BANK_A]
    assert output.qwen_fair["discard_active"]


def test_scheduler_waits_for_live_old_generation_before_replacing_it(monkeypatch):
    module = load_module(monkeypatch)
    scheduler = new_scheduler(module)

    class Request:
        def __init__(self, request_id):
            self.request_id = request_id

        def is_finished(self):
            return False

    old = Request("old")
    new = Request("new")
    scheduler.banks = SimpleNamespace(
        active=BANK_A,
        owners={"old": BANK_A, "new": BANK_A_NEXT},
        managers={BANK_A: object(), BANK_B: object()},
    )
    scheduler.requests = {"old": old, "new": new}
    scheduler.finished_req_ids = set()
    scheduler.parked = {}
    scheduler.drop_banks = []
    scheduler.last_served = {}
    scheduler.max_banks = 2

    assert not scheduler._make_room(BANK_A_NEXT)
    assert set(scheduler.banks.managers) == {BANK_A, BANK_B}
    assert scheduler.drop_banks == []


def test_scheduler_forces_final_idle_status_after_request_completion(monkeypatch):
    module = load_module(monkeypatch)
    scheduler = new_scheduler(module)
    scheduler.requests = {"request": object()}
    published = []
    scheduler._publish_status = lambda *, force=False: published.append(force)

    assert (
        scheduler.update_from_output(object(), SimpleNamespace(finish=False, request_id="request"))
        == "updated"
    )
    assert published == [False]
    assert (
        scheduler.update_from_output(object(), SimpleNamespace(finish=True, request_id="request"))
        == "updated"
    )
    assert published == [False, True]


def phase_fixture(module, tmp_path):
    scheduler = new_scheduler(module)
    request = SimpleNamespace(
        request_id="synthetic-only",
        num_prompt_tokens=150800,
        num_computed_tokens=0,
        num_output_tokens=0,
        is_finished=lambda: False,
        status=SimpleNamespace(name="WAITING"),
        kv_transfer_params={"qwen_chat": {"id": CHAT_A, "generation": GEN_A}},
    )
    scheduler.requests = {request.request_id: request}
    scheduler.running = []
    scheduler.parked = {}
    scheduler.banks = SimpleNamespace(
        active=BANK_A, owners={request.request_id: BANK_A}, managers={BANK_A: object()}
    )
    scheduler.status_path = str(tmp_path / "status")
    scheduler.last_status = 0
    scheduler.quantum = 0
    scheduler.switch_count = 0
    scheduler.max_banks = 2
    scheduler.pending_switch = None
    scheduler.response_request = request
    scheduler.request_phases = module.RequestPhases()
    scheduler.request_phases.set(request, "admission")
    return scheduler, request


def test_running_transition_is_published_immediately_but_counters_are_throttled(
    tmp_path, monkeypatch
):
    module = load_module(monkeypatch)
    clock = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    scheduler, request = phase_fixture(module, tmp_path)
    path = Path(scheduler.status_path + "-scheduler.json")
    scheduler._publish_status(force=True)
    assert json.loads(path.read_text())["requests"][0]["state"] == "queued"
    clock[0] += 0.004
    scheduler.running.append(request)
    request.num_computed_tokens = 150272
    scheduler._publish_status()
    assert json.loads(path.read_text())["requests"][0]["state"] == "running"
    old = path.stat().st_mtime_ns
    clock[0] += 0.005
    request.num_computed_tokens += 16
    scheduler._publish_status()
    assert path.stat().st_mtime_ns == old


def test_generation_phase_counters_publish_each_round_without_rewriting_scheduler_status(
    tmp_path, monkeypatch
):
    module = load_module(monkeypatch)
    clock = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    scheduler, request = phase_fixture(module, tmp_path)
    scheduler_path = Path(scheduler.status_path + "-scheduler.json")
    phases_path = Path(scheduler.status_path + "-phases.json")
    rounds_path = Path(scheduler.status_path + "-rounds.jsonl")

    scheduler._publish_status(force=True)
    scheduler_bytes = scheduler_path.read_bytes()
    assert json.loads(phases_path.read_text())["requests"][0]["generation_rounds"] == 0

    scheduler.request_phases.set(request, "generate")
    scheduler.request_phases.observe_generation(
        request,
        draft_tokens=7,
        accepted_tokens=3,
        scheduled_shape={"mode": "decode", "scheduled_tokens": 8},
        now=clock[0],
    )
    clock[0] += 0.0437
    scheduler._publish_status()
    first = json.loads(phases_path.read_text())["requests"][0]
    assert scheduler_path.read_bytes() == scheduler_bytes
    assert first["generation_rounds"] == 1
    assert first["last_round_ms"] is None
    assert first["acceptance_rate"] == pytest.approx(3 / 7)
    assert first["last_acceptance_rate"] == pytest.approx(3 / 7)
    first_round = json.loads(rounds_path.read_text().splitlines()[0])
    assert first_round["schema"] == "urn:qwen-r9700:decode-rounds:v1"
    assert first_round["round"] == 1
    assert first_round["round_ms"] is None
    assert first_round["draft_tokens"] == 7
    assert first_round["accepted_tokens"] == 3
    assert first_round["scheduled_shape"] == {"mode": "decode", "scheduled_tokens": 8}

    scheduler.request_phases.observe_generation(
        request, draft_tokens=7, accepted_tokens=4, now=clock[0]
    )
    clock[0] += 0.005
    scheduler._publish_status()
    second = json.loads(phases_path.read_text())["requests"][0]
    assert scheduler_path.read_bytes() == scheduler_bytes
    assert second["generation_rounds"] == 2
    assert second["last_round_ms"] == pytest.approx(43.7)
    assert second["acceptance_rate"] == pytest.approx(7 / 14)
    assert second["last_acceptance_rate"] == pytest.approx(4 / 7)
    rounds = [json.loads(line) for line in rounds_path.read_text().splitlines()]
    assert [event["round"] for event in rounds] == [1, 2]
    assert rounds[1]["round_ms"] == pytest.approx(43.7)


def test_decode_round_log_rotates_before_it_exceeds_bounded_size(tmp_path, monkeypatch):
    module = load_module(monkeypatch)
    path = tmp_path / "rounds.jsonl"
    monkeypatch.setattr(module, "ROUND_LOG_MAX_BYTES", 80)
    event = {
        "schema": "urn:qwen-r9700:decode-rounds:v1",
        "round": 1,
        "round_ms": 45.2,
    }
    module.append_round_log(path, [event])
    first = path.read_bytes()
    module.append_round_log(path, [{**event, "round": 2}])
    assert path.exists()
    assert path.with_name("rounds.jsonl.1").read_bytes() == first
    assert json.loads(path.read_text())["round"] == 2


def test_cache_update_lookup_and_restore_keep_separate_durations_and_return_values(
    tmp_path, monkeypatch
):
    module = load_module(monkeypatch)
    clock = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    scheduler, request = phase_fixture(module, tmp_path)
    old = SimpleNamespace(req=SimpleNamespace(is_finished=lambda: True))
    states = {"old": old, request.request_id: SimpleNamespace(transfer_jobs=set())}
    result = [None, False]
    scheduler.connector = SimpleNamespace(
        connector_scheduler=SimpleNamespace(_req_status=states, _snapshot_settled_tail_only=True),
        get_num_new_matched_tokens=lambda *_: tuple(result),
    )
    scheduler._install_phase_hooks()
    assert scheduler.connector.get_num_new_matched_tokens(request, 149000) == (None, False)
    assert scheduler.request_phases.row(request)["phase"] == "cache_update"
    clock[0] += 0.237
    states.pop("old")
    assert scheduler.connector.get_num_new_matched_tokens(request, 149000) == (None, False)
    assert scheduler.request_phases.row(request)["phase"] == "cache_lookup"
    clock[0] += 0.003
    result[:] = [1024, True]
    assert scheduler.connector.get_num_new_matched_tokens(request, 149000) == (1024, True)
    row = scheduler.request_phases.row(request)
    assert row["phase"] == "cache_restore"
    assert row["cached_tokens"] == 150024
    assert scheduler._decode_sync_epochs[request.request_id] == 1
    assert row["timings_ms"]["cache_update"] == 237
    assert row["timings_ms"]["cache_lookup"] == 3
    clock[0] += 0.2
    scheduler._phase(request, "prefill")
    clock[0] += 0.1
    scheduler._phase(request, "generate")
    scheduler.request_phases.finish(request)
    recent = next(iter(scheduler.request_phases.recent.values()))
    assert recent["timings_ms"]["cache_restore"] == 200
    assert recent["timings_ms"]["prefill"] == 100
    assert recent["first_token_ms"] == pytest.approx(540)
    assert "synthetic-only" not in json.dumps(recent)
    assert not scheduler.request_phases.live


def test_request_phase_generation_metrics_track_consecutive_rounds_and_acceptance(
    tmp_path, monkeypatch
):
    module = load_module(monkeypatch)
    scheduler, request = phase_fixture(module, tmp_path)
    clock = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    scheduler.request_phases.set(request, "generate")
    scheduler.request_phases.observe_generation(
        request, draft_tokens=7, accepted_tokens=3, now=clock[0]
    )
    clock[0] += 0.0437
    scheduler.request_phases.observe_generation(
        request, draft_tokens=7, accepted_tokens=4, now=clock[0]
    )
    row = scheduler.request_phases.row(request)
    assert row["last_round_ms"] == pytest.approx(43.7)
    assert row["generation_rounds"] == 2
    assert row["draft_tokens"] == 14
    assert row["accepted_tokens"] == 7
    assert row["acceptance_rate"] == pytest.approx(0.5)
    assert row["last_acceptance_rate"] == pytest.approx(4 / 7)

    # A target-only final/EOS round has no speculative denominator.  It must
    # not erase the last actual round acceptance shown to Pi.
    clock[0] += 0.006
    scheduler.request_phases.observe_generation(
        request, draft_tokens=0, accepted_tokens=0, now=clock[0]
    )
    final_round = scheduler.request_phases.row(request)
    assert final_round["generation_rounds"] == 3
    assert final_round["acceptance_rate"] == pytest.approx(7 / 14)
    assert final_round["last_acceptance_rate"] == pytest.approx(4 / 7)

    # A handover breaks the consecutive-round interval; resumed output must
    # not include time spent while another chat owned the GPU.
    scheduler.request_phases.set(request, "gpu_queue", module.bank_identity(BANK_B))
    scheduler.request_phases.set(request, "generate")
    clock[0] += 10
    scheduler.request_phases.observe_generation(
        request, draft_tokens=7, accepted_tokens=2, now=clock[0]
    )
    resumed = scheduler.request_phases.row(request)
    # The retained duration is the target-only final round immediately before
    # handover; the ten seconds parked in the other phase are excluded.
    assert resumed["last_round_ms"] == pytest.approx(6.0)
    assert resumed["last_acceptance_rate"] == pytest.approx(2 / 7)


def test_queue_phase_requires_another_selected_response(tmp_path, monkeypatch):
    module = load_module(monkeypatch)
    scheduler, request = phase_fixture(module, tmp_path)
    scheduler._refresh_request_phases()
    assert scheduler.request_phases.row(request)["phase"] == "admission"
    other = SimpleNamespace(request_id="other", is_finished=lambda: False)
    scheduler.banks.owners[other.request_id] = BANK_B
    scheduler.response_request = other
    scheduler._refresh_request_phases()
    row = scheduler.request_phases.row(request)
    assert row["phase"] == "gpu_queue"
    assert row["blocker"] == {"chat_id": CHAT_B, "generation": GEN_B}
    scheduler.response_request = request
    scheduler.pending_switch = BANK_A
    scheduler._refresh_request_phases()
    assert scheduler.request_phases.row(request)["phase"] == "handover"
    assert scheduler.request_phases.row(request)["blocker"] is None


def test_decode_transition_stream_sync_is_once_per_request_transition(monkeypatch):
    module = load_module(monkeypatch)
    calls = []

    class Stream:
        def synchronize(self):
            calls.append("sync")

    class Cuda:
        @staticmethod
        def current_stream():
            return Stream()

    fake_torch = SimpleNamespace(cuda=Cuda())
    banks = SimpleNamespace(
        active=BANK_A,
        torch=fake_torch,
        before=lambda metadata: None,
    )
    runner = SimpleNamespace(qwen_banks=banks)
    metadata = {
        "bank": BANK_A,
        "barrier": False,
        "drop_banks": [],
        "decode_sync_key": "bank-a-request-1",
    }

    prefill = SimpleNamespace(total_num_scheduled_tokens=2048, qwen_fair=metadata)
    module.before_forward(runner, prefill)
    assert calls == []

    decode = SimpleNamespace(total_num_scheduled_tokens=8, qwen_fair=metadata)
    module.before_forward(runner, decode)
    module.before_forward(runner, decode)
    assert calls == []
    assert module.after_forward_prepare(runner, decode)
    module.after_forward_prepare(runner, decode)
    assert calls == ["sync"]

    # A new Pi turn gets a new request identity while retaining the same
    # chat-generation bank. It must be fenced on its own cadence rather than
    # inheriting the previous turn's latch.
    next_request = {**metadata, "decode_sync_key": "bank-a-request-2"}
    module.before_forward(
        runner,
        SimpleNamespace(total_num_scheduled_tokens=8, qwen_fair=next_request),
    )
    module.before_forward(
        runner,
        SimpleNamespace(total_num_scheduled_tokens=8, qwen_fair=next_request),
    )
    assert calls == ["sync"]
    assert module.after_forward_prepare(
        runner,
        SimpleNamespace(total_num_scheduled_tokens=8, qwen_fair=next_request),
    )
    assert calls == ["sync", "sync"]

    # A second prefill/cache-restore transition in the same request re-arms
    # the first subsequent decode without fencing the prefill itself.
    module.before_forward(
        runner,
        SimpleNamespace(total_num_scheduled_tokens=2048, qwen_fair=next_request),
    )
    assert calls == ["sync", "sync"]
    module.before_forward(
        runner,
        SimpleNamespace(total_num_scheduled_tokens=8, qwen_fair=next_request),
    )
    assert calls == ["sync", "sync"]
    assert module.after_forward_prepare(
        runner,
        SimpleNamespace(total_num_scheduled_tokens=8, qwen_fair=next_request),
    )
    assert calls == ["sync", "sync", "sync"]

    next_generation = {
        **metadata,
        "bank": BANK_A_NEXT,
        "decode_sync_key": "bank-a-next-request-1",
    }
    module.before_forward(
        runner,
        SimpleNamespace(total_num_scheduled_tokens=8, qwen_fair=next_generation),
    )
    assert calls == ["sync", "sync", "sync"]
    assert module.after_forward_prepare(
        runner,
        SimpleNamespace(total_num_scheduled_tokens=8, qwen_fair=next_generation),
    )
    assert calls == ["sync", "sync", "sync", "sync"]


def test_decode_transition_key_uses_scheduled_request_identity(monkeypatch):
    module = load_module(monkeypatch)
    scheduler = module.FairScheduler.__new__(module.FairScheduler)
    scheduler.running = [SimpleNamespace(request_id="request-a")]
    scheduler.response_request = None
    scheduler._decode_sync_epochs = {"request-a": 1}
    metadata = {"bank": BANK_A}

    scheduler._attach_decode_sync_key(
        SimpleNamespace(num_scheduled_tokens={"request-a": 8}), metadata
    )
    first = metadata["decode_sync_key"]
    assert isinstance(first, str) and len(first) == 64

    scheduler._decode_sync_epochs["request-a"] = 2
    scheduler._attach_decode_sync_key(
        SimpleNamespace(num_scheduled_tokens={"request-a": 8}), metadata
    )
    assert metadata["decode_sync_key"] != first

    scheduler.running = [SimpleNamespace(request_id="request-b")]
    scheduler._attach_decode_sync_key(
        SimpleNamespace(num_scheduled_tokens={"request-b": 8}), metadata
    )
    assert metadata["decode_sync_key"] != first


def test_scheduled_shape_is_content_free_and_preserves_batch_widths(tmp_path, monkeypatch):
    module = load_module(monkeypatch)
    shape = module.FairScheduler._scheduled_shape(
        SimpleNamespace(
            total_num_scheduled_tokens=10,
            num_scheduled_tokens={"private-a": 8, "private-b": 2},
            scheduled_spec_decode_tokens={"private-a": [1, 2, 3]},
        )
    )
    assert shape == {
        "mode": "decode",
        "scheduled_tokens": 10,
        "request_count": 2,
        "tokens_per_request": [2, 8],
        "draft_widths": [3],
    }
    assert "private-a" not in json.dumps(shape)

def test_decode_transition_retires_only_observed_slow_queue_episodes(monkeypatch):
    module = load_module(monkeypatch)
    calls = []

    class Stream:
        def synchronize(self):
            calls.append("sync")

    class Cuda:
        @staticmethod
        def current_stream():
            return Stream()

    runner = SimpleNamespace(
        qwen_banks=SimpleNamespace(
            active=BANK_A,
            torch=SimpleNamespace(cuda=Cuda()),
            before=lambda metadata: None,
        )
    )
    metadata = {
        "bank": BANK_A,
        "barrier": False,
        "drop_banks": [],
        "decode_sync_key": "bank-a-long-response",
        "last_round_ms": None,
    }

    first = SimpleNamespace(total_num_scheduled_tokens=8, qwen_fair=metadata)
    module.before_forward(runner, first)
    assert module.after_forward_prepare(runner, first)
    # A stable fast run does not receive a periodic fence. The old fixed
    # thirty-round policy created a visible latency spike even in this case.
    for _ in range(8):
        metadata["last_round_ms"] = 44.0
        step = SimpleNamespace(total_num_scheduled_tokens=8, qwen_fair=metadata)
        module.before_forward(runner, step)
        assert not module.after_forward_prepare(runner, step)
    assert calls == ["sync"]

    # A material jump is detected before the next forward and the recovery
    # fence is deferred until after connector preparation.
    metadata["last_round_ms"] = 53.5
    step = SimpleNamespace(total_num_scheduled_tokens=8, qwen_fair=metadata)
    module.before_forward(runner, step)
    assert calls == ["sync"]
    assert module.after_forward_prepare(runner, step)
    assert calls == ["sync", "sync"]

    # A persistent slow episode is not fenced on every round. One recovered
    # fast sample re-arms the detector for a later slow episode.
    metadata["last_round_ms"] = 53.5
    step = SimpleNamespace(total_num_scheduled_tokens=8, qwen_fair=metadata)
    module.before_forward(runner, step)
    assert not module.after_forward_prepare(runner, step)
    metadata["last_round_ms"] = 44.0
    step = SimpleNamespace(total_num_scheduled_tokens=8, qwen_fair=metadata)
    module.before_forward(runner, step)
    assert not module.after_forward_prepare(runner, step)
    for _ in range(4):
        metadata["last_round_ms"] = 44.0
        step = SimpleNamespace(total_num_scheduled_tokens=8, qwen_fair=metadata)
        module.before_forward(runner, step)
        assert not module.after_forward_prepare(runner, step)
    assert calls == ["sync", "sync"]
    metadata["last_round_ms"] = 53.5
    step = SimpleNamespace(total_num_scheduled_tokens=8, qwen_fair=metadata)
    module.before_forward(runner, step)
    assert calls == ["sync", "sync"]
    assert module.after_forward_prepare(runner, step)
    assert calls == ["sync", "sync", "sync"]

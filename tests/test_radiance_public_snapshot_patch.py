from __future__ import annotations

import gzip

import ast
import hashlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
from enum import Enum
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_SCRIPT = REPO_ROOT / "experiments" / "radiance-public" / "patch_streaming_snapshot.py"
QUALIFIED_SCHEDULER = REPO_ROOT / "tests/fixtures/vllm_027_scheduler_before_tail_publication.py.gz"

# The two release paths from the pinned MambaManager, with a minimal block pool
# supplied by the test. Preserve these anchors to catch upstream source changes.
MAMBA_RELEASE = """
class MambaManager(Parent):
    def remove_skipped_blocks(self, request_id, processed_computed_tokens, num_prompt_tokens=None):
        super().remove_skipped_blocks(
            request_id, processed_computed_tokens, num_prompt_tokens
        )
        if self.mamba_cache_mode == "align":
            last_state_block_idx = self.last_state_block_idx.get(request_id)
            if (
                last_state_block_idx is not None
                and last_state_block_idx
                < cdiv(processed_computed_tokens, self.block_size) - 1
            ):
                blocks = self.req_to_blocks[request_id]
                if blocks[last_state_block_idx] != self._null_block:
                    self.block_pool.free_blocks([blocks[last_state_block_idx]])
                    blocks[last_state_block_idx] = self._null_block

    def get_num_skipped_tokens(self, num_computed_tokens):
        return num_computed_tokens - 1

    def cache_blocks(self):
        pass
"""


def make_mamba_fixture(root):
    path = root / "vllm/v1/core/single_type_kv_cache_manager.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(MAMBA_RELEASE)
    return path


def make_fair_runtime_fixtures(root):
    output = root / "vllm/v1/core/sched/output.py"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "class SchedulerOutput:\n"
        + "\n".join(
            f"    {line}" if line else ""
            for line in (
                "# Dynamic speculative decoding: optimal K chosen by scheduler.\n"
                "# Number of spec tokens to schedule for the next step.\n"
                "num_spec_tokens_to_schedule: int = 0\n"
            ).splitlines()
        )
        + "\n"
    )
    runner = root / "vllm/v1/worker/gpu/model_runner.py"
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text(
        "def execute_model(self, scheduler_output, dummy_run=False):\n"
        "        if not dummy_run:\n"
        "            # Update the request states.\n"
    )
    return output, runner


def run_patch(target: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["QWEN_SNAPSHOT_SCHEDULER"] = str(target)
    return subprocess.run(
        [sys.executable, str(PATCH_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_patch_closes_finished_tail_publication_race(tmp_path: Path) -> None:
    target = tmp_path / "scheduler.py"
    target.write_bytes(gzip.decompress(QUALIFIED_SCHEDULER.read_bytes()))

    first = run_patch(target)
    assert first.returncode == 0, first.stderr

    patched = target.read_text()
    barrier = patched.index("A finished predecessor remains in _req_status")
    clear_groups = patched.index("for group_state in req_status.group_states:", barrier)
    update_keys = patched.index("req_status.update_offload_keys()", clear_groups)
    assert barrier < clear_groups < update_keys
    assert "if self._snapshot_settled_tail_only:" in patched[barrier:clear_groups]
    assert "req_id != request.request_id and status.req.is_finished()" in patched
    assert "return None, False" in patched[barrier:clear_groups]
    assert patched.count("A finished predecessor remains in _req_status") == 1

    second = run_patch(target)
    assert second.returncode == 0, second.stderr
    assert target.read_text() == patched
    assert "already applied" in second.stdout


@pytest.mark.parametrize("aligned,expected_hit", [(False, 0), (True, 70864)])
def test_v028_saved_target_and_draft_states_converge_on_one_prefix(tmp_path, aligned, expected_hit):
    # Execute the upstream lookup itself against the observed 74K snapshot
    # shape. Counting files alone missed this incompatible set of saved states.
    source = tmp_path / "scheduler.py"
    source.write_bytes(gzip.decompress(QUALIFIED_SCHEDULER.read_bytes()))
    if aligned:
        result = run_patch(source)
        assert result.returncode == 0, result.stderr
    methods = {
        node.name: ast.unparse(node)
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.FunctionDef)
        and node.name
        in ("storable_chunks", "_lookup", "_maximal_prefix_lookup", "_sliding_window_lookup")
    }
    lookup_result = Enum("LookupResult", "HIT HIT_PENDING RETRY MISS")
    namespace = {
        "cdiv": lambda a, b: (a + b - 1) // b,
        "round_down": lambda a, b: a // b * b,
        "LookupResult": lookup_result,
        "logger": logging.getLogger(__name__),
    }
    exec("from __future__ import annotations\n" + "\n\n".join(methods.values()), namespace)
    configs = [
        SimpleNamespace(
            tokens_per_chunk=1648,
            sliding_window_size_in_chunks=(1 if group < 6 else 2 if group == 8 else None),
            requires_cow_source=group < 6,
            is_eagle_group=group == 8,
        )
        for group in range(9)
    ]
    states = [
        SimpleNamespace(offload_keys=[(group, i) for i in range(45)], block_ids=list(range(46)))
        for group in range(9)
    ]
    config = SimpleNamespace(kv_group_configs=configs, blocks_per_chunk=1)
    finished = SimpleNamespace(config=config, req=SimpleNamespace(num_prompt_tokens=74303))
    stored = set()
    for group, (cfg, state) in enumerate(zip(configs, states, strict=True)):
        count = namespace["storable_chunks"](finished, cfg, state, 74559)
        tail = 2 if group < 6 else 3 if group == 8 else count
        stored.update(state.offload_keys[max(0, count - tail) : count])
    scheduler = SimpleNamespace(
        config=config,
        _lookup_groups=(6, 7, 0, 1, 2, 3, 4, 5, 8),
        _sliding_window_groups=(0, 1, 2, 3, 4, 5, 8),
        _mamba_align_size=1648,
        _chunks_being_loaded=set(),
        _events_tracker=SimpleNamespace(record_lookup=lambda *args, **kwargs: None),
        manager=SimpleNamespace(
            lookup=lambda key, _: lookup_result.HIT if key in stored else lookup_result.MISS
        ),
    )
    for name in ("_maximal_prefix_lookup", "_sliding_window_lookup"):
        setattr(scheduler, name, MethodType(namespace[name], scheduler))
    request = SimpleNamespace(
        req=SimpleNamespace(num_tokens=74303, request_id="synthetic"),
        num_locally_computed_tokens=0,
        req_context=None,
        group_states=states,
    )
    assert namespace["_lookup"](scheduler, request) == expected_hit


@pytest.mark.parametrize("release", ["0.27", "0.28"])
def test_chat_storage_patch_is_idempotent_and_rejects_unknown_scheduler(
    tmp_path, monkeypatch, release
):
    installer_path = PATCH_SCRIPT.with_name("patch_chat_snapshot.py")
    spec = importlib.util.spec_from_file_location("chat_snapshot_patch", installer_path)
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    serving = tmp_path / "vllm/entrypoints/openai/chat_completion/serving.py"
    serving.parent.mkdir(parents=True)
    fixture = (
        "class Serving:\n"
        "    async def create(self, request):\n"
        + installer.TOOL_HANDOVER_SERVING[0][0]
        + "    async def stream(self, request):\n"
        + "        for _ in []:\n"
        + "".join("    " * i + "if True:\n" for i in range(3, 6))
    )
    fixture += installer.BUFFERED_USAGE_OLD
    fixture += installer.TOOL_HANDOVER_SERVING[1][0]
    fixture += "    async def complete(self, request):\n        for _ in []:\n"
    fixture += installer.TOOL_HANDOVER_SERVING[2][0]
    serving.write_text(fixture)
    monkeypatch.setattr(
        installer, "STREAMING_CHAT_SHA256", hashlib.sha256(fixture.encode()).hexdigest()
    )
    engine = tmp_path / "vllm/v1/engine/core.py"
    engine.parent.mkdir(parents=True)
    engine_fixture = (
        "import queue\nclass Engine:\n"
        + installer.TOOL_HANDOVER_CORE[0][0]
        + "        pass\n    def step(self):\n"
        + installer.TOOL_HANDOVER_CORE[1][0]
    )
    engine.write_text(engine_fixture)
    monkeypatch.setattr(
        installer, "ENGINE_CORE_SHA256", hashlib.sha256(engine_fixture.encode()).hexdigest()
    )
    scheduler = tmp_path / "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
    scheduler.parent.mkdir(parents=True)
    scheduler.write_bytes(gzip.decompress(QUALIFIED_SCHEDULER.read_bytes()))
    if release == "0.28":
        source = (
            scheduler.read_text()
            .replace(
                "        meta = OffloadingConnectorMetadata(\n",
                "        partial_store_jobs = self._build_partial_tail_store_jobs("
                "scheduler_output)\n"
                "        normal_store_jobs = self._build_store_jobs(scheduler_output)\n"
                "        meta = OffloadingConnectorMetadata(\n",
            )
            .replace(
                "store_jobs=self._build_store_jobs(scheduler_output),",
                "store_jobs=partial_store_jobs | normal_store_jobs,",
            )
        )
        scheduler.write_text(source)
    factory = tmp_path / "vllm/v1/kv_offload/tiering/factory.py"
    factory.parent.mkdir(parents=True)
    factory.write_text("# factory fixture\n")
    core = REPO_ROOT / "src/qwen_r9700_lab/radiance_cache.py"
    tier = PATCH_SCRIPT.with_name("radiance_chat_tier.py")
    mamba = make_mamba_fixture(tmp_path)
    output, runner = make_fair_runtime_fixtures(tmp_path)
    worker = tmp_path / "vllm/v1/worker/gpu_worker.py"
    memory_hooks = (
        installer.V028_MEMORY_REPORT_HOOKS if release == "0.28" else installer.MEMORY_REPORT_HOOKS
    )
    worker_fixture = (
        "class Worker:\n    def warmup(self):\n"
        + memory_hooks[0][0]
        + "        )\n\n"
        + memory_hooks[1][0]
    )
    worker.write_text(worker_fixture)
    monkeypatch.setattr(
        installer,
        "V028_GPU_WORKER_SHA256" if release == "0.28" else "GPU_WORKER_SHA256",
        hashlib.sha256(worker_fixture.encode()).hexdigest(),
    )
    installer.install(tmp_path, core, tier)
    first = scheduler.read_bytes()
    first_mamba = mamba.read_bytes()
    first_serving = serving.read_bytes()
    first_output = output.read_bytes()
    first_runner = runner.read_bytes()
    first_engine = engine.read_bytes()
    first_worker = worker.read_bytes()
    installer.install(tmp_path, core, tier)
    assert first == scheduler.read_bytes()
    assert first_mamba == mamba.read_bytes()
    assert first_serving == serving.read_bytes()
    assert first_output == output.read_bytes()
    assert first_runner == runner.read_bytes()
    assert first_engine == engine.read_bytes()
    assert first_worker == worker.read_bytes()
    assert worker.read_text().count("start_memory_report(self.model_runner)") == 1
    assert worker.read_text().count("stop_memory_report(getattr(self, 'model_runner', None))") == 1
    assert (tmp_path / "qwen_radiance_memory.py").read_bytes() == core.with_name(
        "radiance_memory.py"
    ).read_bytes()
    assert engine.read_text().count("def qwen_response_outcome(") == 1
    assert engine.read_text().count("def qwen_answer_priority(") == 1
    assert engine.read_text().count("input_queue.get(timeout=0.02)") == 1
    assert "getattr(self.scheduler, 'priority_hold', None)" in engine.read_text()
    assert serving.read_text().count("prepare_tool_handover(request)") == 1
    assert serving.read_text().count("report_tool_handover(") == 2
    assert installer.BUFFERED_USAGE_NEW in serving.read_text()
    assert scheduler.read_text().count("request.cache_salt != cache_salt(chat)") == 1
    assert (
        scheduler.read_text().count("tier.set_snapshot_head(req_status, num_offloadable_tokens)")
        == 1
    )
    assert factory.read_text().count("qwen_chat_fs") == 1
    assert output.read_text().count("qwen_fair: dict | None = None") == 1
    assert runner.read_text().count("before_forward(self, scheduler_output)") == 1
    assert scheduler.read_text().count("fair.get('barrier')") == 1
    fair_scheduler = scheduler.read_text()
    fair_start = fair_scheduler.index("store_jobs = self._build_store_jobs")
    fair_flush = fair_scheduler.index("fair.get('barrier')", fair_start)
    fair_meta = fair_scheduler.index("store_jobs=", fair_flush)
    if release == "0.28":
        assert "store_jobs=partial_store_jobs | normal_store_jobs," in fair_scheduler
    assert fair_start < fair_flush < fair_meta
    assert (tmp_path / "qwen_radiance_fair_scheduler.py").read_bytes() == (
        PATCH_SCRIPT.with_name("radiance_fair_scheduler.py").read_bytes()
    )
    serving.write_text("# unexpected serving source\n")
    with pytest.raises(ValueError, match="serving source differs"):
        installer.install(tmp_path, core, tier)
    assert scheduler.read_bytes() == first
    serving.write_bytes(first_serving)
    scheduler.write_text("# unexpected source\n")

    with pytest.raises(ValueError, match="anchor changed"):
        installer.install(tmp_path, core, tier)
    assert scheduler.read_text() == "# unexpected source\n"


def test_chat_storage_abi_authenticates_every_runtime_module_and_launcher():
    base = PATCH_SCRIPT.parent
    path = base / "snapshot-abi-chat-cache-v1.json"
    abi = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads(path.read_text())
    assert manifest["storage"]["secondary_tier"] == "qwen_chat_fs"
    for name, expected in manifest["runtime"].get("release_files", {}).items():
        assert hashlib.sha256((base / name).read_bytes()).hexdigest() == expected
    assert manifest["storage"]["data_abi"] != manifest["storage"]["previous_data_abi"]
    assert manifest["runtime"]["memory_report"]["compatible_runtime_abis"] == [
        "a0fbc562276e24fcce8ddf48d71634920ec722e27dedeb3b1e78995a3da34832"
    ]
    for name, expected in manifest["runtime"]["chat_storage"]["modules"].items():
        source = (
            REPO_ROOT / "src/qwen_r9700_lab"
            if name in ("radiance_cache.py", "radiance_memory.py")
            else base
        ) / name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    launcher = base / "launch_public_clean_snapshot_server.sh"
    frontend = (REPO_ROOT / "scripts/pi-remote-qwen-radiance").read_text()
    assert f"readonly abi_id={abi}" in launcher.read_text()
    assert f"readonly SNAPSHOT_ABI={abi}" in frontend
    data_abi = manifest["storage"]["data_abi"]
    assert f"readonly data_abi={data_abi}" in launcher.read_text()
    assert f"readonly SNAPSHOT_DATA_ABI={data_abi}" in frontend
    assert 'QWEN_RADIANCE_CACHE_ABI="$SNAPSHOT_DATA_ABI"' in frontend
    assert 'for candidate in "$runtime_abi"' in frontend
    assert "qwen-radiance-public-clean-${candidate:0:16}" in frontend
    assert (
        f"readonly REMOTE_LAUNCHER_SHA256={hashlib.sha256(launcher.read_bytes()).hexdigest()}"
        in frontend
    )
    expected = manifest["runtime"]["bounded_sliding_lookup_and_settled_tail_patch"]["sha256"]
    assert hashlib.sha256(PATCH_SCRIPT.read_bytes()).hexdigest() == expected
    launcher_text = launcher.read_text()
    assert manifest["storage"]["cpu_primary_bytes"] == 18 * 1024**3
    assert "cpu_bytes_to_use: 19327352832" in launcher_text
    assert (
        "offload_mmaps=(/dev/shm/vllm_offload_qwen-radiance-public-clean-*.mmap)" in launcher_text
    )
    assert "((${#offload_mmaps[@]} <= 8))" in launcher_text
    assert "600:$(id -u):1" in launcher_text
    assert "for _ in {1..100}" in launcher_text
    assert "sleep 0.1" in launcher_text


def test_snapshot_states_survive_both_mamba_release_paths_without_accumulating():
    spec = importlib.util.spec_from_file_location(
        "chat_patch", PATCH_SCRIPT.with_name("patch_chat_snapshot.py")
    )
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)

    class Parent:
        def remove_skipped_blocks(self, request_id, processed_computed_tokens, _):
            count = self.get_num_skipped_tokens(processed_computed_tokens) // self.block_size
            for idx in range(max(0, count)):
                self.req_to_blocks[request_id][idx] = None

    scope = {"Parent": Parent, "cdiv": lambda a, b: (a + b - 1) // b}
    exec(installer.retain_settled_mamba_tail(MAMBA_RELEASE), scope)
    manager = scope["MambaManager"]()
    manager.block_size = 1648
    manager.mamba_cache_mode = "align"
    manager._null_block = None
    manager.block_pool = SimpleNamespace(free_blocks=lambda _: None)
    manager.req_to_blocks = {"cold": []}
    manager.last_state_block_idx = {}
    for chunk in range(129):
        blocks = manager.req_to_blocks["cold"]
        blocks.append(chunk + 1)
        # The direct CoW release must not free a state still inside the tail.
        manager.last_state_block_idx["cold"] = max(0, chunk - 2)
        manager.remove_skipped_blocks("cold", chunk * 1648 + 100)
        assert len([block for block in blocks if block is not None]) <= 4
        if chunk >= 3:
            assert blocks[chunk - 3] is not None
            assert blocks[chunk - 2] is not None
    manager.mamba_cache_mode = "all"
    assert manager.get_num_skipped_tokens(211152) == 211151
    manager.mamba_cache_mode = "align"
    assert manager.get_num_skipped_tokens(211152) == 211151

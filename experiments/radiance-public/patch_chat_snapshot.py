"""Install the pinned chat storage and streaming-progress runtime repairs."""

import hashlib
from pathlib import Path

STREAMING_CHAT_SHA256 = "f3fb82d6eeb956e0e5e9b8e17e135769b3378421ad0fb7958daaf774adc57079"
ENGINE_CORE_SHA256 = "6fdd067f54e5d42c57ff413e292685f5bdf6f343498d85c9262567f1eb746916"
GPU_WORKER_SHA256 = "50134ca3a147f5470e556437cbd40a13ea05b098efbac6553d5ea2f3e5f7ff57"
V028_STREAMING_CHAT_SHA256 = "a2440b6b76ad87de86bcf6fc7061ad5d02ab67c1c3c89ca7de3eeb29aa12aa2f"
V028_ENGINE_CORE_SHA256 = "f4b1e07b6d91fac1549bc74a5c804e5a88793d663b5dc045c9e455db6550f5ad"
V028_GPU_WORKER_SHA256 = "5e8faf3e00649289c81c2917dd4557a80b0b351c33295d8e0c4b204a304e6552"
MEMORY_REPORT_HOOKS = (
    (
        "        enable_gpu_sync_check()\n\n        return CompilationTimes(\n",
        "        enable_gpu_sync_check()\n\n"
        "        # Shared CPU allocator accounting, after all GPU warmup.\n"
        "        from qwen_radiance_memory import start_memory_report\n"
        "        start_memory_report(self.model_runner)\n\n"
        "        return CompilationTimes(\n",
    ),
    (
        "    def shutdown(self) -> None:\n        gc.unfreeze()\n",
        "    def shutdown(self) -> None:\n"
        "        from qwen_radiance_memory import stop_memory_report\n"
        "        stop_memory_report(getattr(self, 'model_runner', None))\n"
        "        gc.unfreeze()\n",
    ),
)
V028_MEMORY_REPORT_HOOKS = (
    (
        "        set_torch_threads_for_runtime()\n\n        return CompilationTimes(\n",
        "        set_torch_threads_for_runtime()\n\n"
        "        # Shared CPU allocator accounting, after all GPU warmup.\n"
        "        from qwen_radiance_memory import start_memory_report\n"
        "        start_memory_report(self.model_runner)\n\n"
        "        return CompilationTimes(\n",
    ),
    MEMORY_REPORT_HOOKS[1],
)
BUFFERED_USAGE_OLD = """                        if output.finish_reason is None and (
                            not request.return_token_ids or hide_stream_metadata
                        ):
                            continue
                        delta_message = DeltaMessage()
"""
BUFFERED_USAGE_NEW = """                        # Report usage while a structured argument is
                        # buffered. Keep the choice index and the existing rules
                        # for hidden reasoning and optional raw token IDs.
                        if (
                            output.finish_reason is None
                            and not include_continuous_usage
                            and (not request.return_token_ids or hide_stream_metadata)
                        ):
                            continue
                        delta_message = DeltaMessage()
"""
FAIR_OUTPUT_OLD = """    # Dynamic speculative decoding: optimal K chosen by scheduler.
    # Number of spec tokens to schedule for the next step.
    num_spec_tokens_to_schedule: int = 0
"""
FAIR_OUTPUT_NEW = (
    FAIR_OUTPUT_OLD
    + """
    # Opaque scheduler/worker handover metadata. None for the stock scheduler.
    qwen_fair: dict | None = None
"""
)
FAIR_RUNNER_OLD = """        if not dummy_run:
            # Update the request states.
"""
FAIR_RUNNER_NEW = """        if not dummy_run:
            # The preceding zero-token frame drained connector stores. Swap the
            # packed cache before request state updates, block zeroing, or loads.
            from qwen_radiance_fair_scheduler import before_forward
            before_forward(self, scheduler_output)
            # Update the request states.
"""
FAIR_PREPARE_OLD = """                self.kv_connector.pre_forward(scheduler_output)
"""
FAIR_PREPARE_NEW = """                self.kv_connector.pre_forward(scheduler_output)
                from qwen_radiance_fair_scheduler import after_forward_prepare
                after_forward_prepare(self, scheduler_output)
"""
TOOL_HANDOVER_SERVING = (
    (
        "        # Streaming response\n        tokenizer = self.renderer.tokenizer\n",
        "        from qwen_radiance_fair_scheduler import prepare_tool_handover\n"
        "        prepare_tool_handover(request)\n"
        "        # Streaming response\n        tokenizer = self.renderer.tokenizer\n",
    ),
    (
        "                        finish_reason_sent[i] = True\n",
        "                        from qwen_radiance_fair_scheduler import report_tool_handover\n"
        "                        report_tool_handover(\n"
        "                            self.engine_client, request,\n"
        "                            bool(tools_streamed[i] or tool_choice_function_name),\n"
        "                        )\n"
        "                        finish_reason_sent[i] = True\n",
    ),
    (
        "            choices.append(choice_data)\n",
        "            from qwen_radiance_fair_scheduler import report_tool_handover\n"
        "            report_tool_handover(\n"
        "                self.engine_client, request, bool(choice_data.message.tool_calls)\n"
        "            )\n"
        "            choices.append(choice_data)\n",
    ),
)
TOOL_HANDOVER_CORE = (
    (
        "    def execute_dummy_batch(self):\n",
        "    def qwen_response_outcome(self, token: str, is_tool: bool) -> bool:\n"
        "        handler = getattr(self.scheduler, 'response_outcome', None)\n"
        "        return bool(handler and handler(token, is_tool))\n\n"
        "    def execute_dummy_batch(self):\n",
    ),
    (
        "        if not model_executed and self.scheduler.has_requests():\n"
        "            time.sleep(0.001)\n",
        "        if not model_executed and self.scheduler.has_requests():\n"
        "            if getattr(self.scheduler, 'grace_status', None) is not None:\n"
        "                # Wake immediately for a quick tool result or parser acknowledgement.\n"
        "                # Otherwise check the grace deadline at 50 Hz, without busy spinning.\n"
        "                try:\n"
        "                    req = self.input_queue.get(timeout=0.02)\n"
        "                except queue.Empty:\n"
        "                    pass\n"
        "                else:\n"
        "                    self._handle_client_request(*req)\n"
        "            else:\n"
        "                time.sleep(0.001)\n",
    ),
)
PRIORITY_CORE = (
    (
        "    def execute_dummy_batch(self):\n",
        "    def qwen_answer_priority(self, value: dict) -> dict:\n"
        "        handler = getattr(self.scheduler, 'answer_priority', None)\n"
        "        if handler is None:\n"
        "            raise ValueError('scheduler does not support chat priority')\n"
        "        return handler(value)\n\n"
        "    def execute_dummy_batch(self):\n",
    ),
    (
        "            if getattr(self.scheduler, 'grace_status', None) is not None:\n",
        "            if (getattr(self.scheduler, 'grace_status', None) is not None\n"
        "                    or getattr(self.scheduler, 'priority_hold', None) is not None):\n",
    ),
)


def apply_replacements(text, replacements):
    for old, new in replacements:
        if text.count(new) == 1:
            continue
        if text.count(old) != 1:
            raise ValueError("tool handover runtime anchor changed")
        text = text.replace(old, new)
    return text


def tool_handover_core(text):
    for old, new in reversed(PRIORITY_CORE):
        if text.count(new) == 1:
            text = text.replace(new, old)
    for old, new in TOOL_HANDOVER_CORE:
        if text.count(new) == 1:
            text = text.replace(new, old)
    if hashlib.sha256(text.encode()).hexdigest() not in (
        ENGINE_CORE_SHA256,
        V028_ENGINE_CORE_SHA256,
    ):
        raise ValueError("tool handover engine source differs from the pinned runtime")
    return apply_replacements(apply_replacements(text, TOOL_HANDOVER_CORE), PRIORITY_CORE)


def memory_report_worker(text):
    for old, new in (*MEMORY_REPORT_HOOKS, *V028_MEMORY_REPORT_HOOKS):
        if text.count(new) == 1:
            text = text.replace(new, old)
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest == GPU_WORKER_SHA256:
        hooks = MEMORY_REPORT_HOOKS
    elif digest == V028_GPU_WORKER_SHA256:
        hooks = V028_MEMORY_REPORT_HOOKS
    else:
        raise ValueError("memory report worker source differs from the pinned runtime")
    return apply_replacements(text, hooks)


def stream_buffered_tool_usage(text: str) -> str:
    """Let empty parser deltas reach the normal per-choice continuous-usage path."""
    for old, new in TOOL_HANDOVER_SERVING:
        if text.count(new) == 1:
            text = text.replace(new, old)
    if text.count(BUFFERED_USAGE_NEW) == 1:
        original = text.replace(BUFFERED_USAGE_NEW, BUFFERED_USAGE_OLD)
    else:
        original = text
    if (
        hashlib.sha256(original.encode()).hexdigest()
        not in (STREAMING_CHAT_SHA256, V028_STREAMING_CHAT_SHA256)
        or original.count(BUFFERED_USAGE_OLD) != 1
    ):
        raise ValueError("buffered usage chat serving source differs from the pinned runtime")
    return apply_replacements(
        original.replace(BUFFERED_USAGE_OLD, BUFFERED_USAGE_NEW), TOOL_HANDOVER_SERVING
    )


def retain_settled_mamba_tail(text: str) -> str:
    """Keep the finished snapshot's two settled states away from live-state reuse."""
    prefix, marker, body = text.partition("class MambaManager(")
    if not marker:
        raise ValueError("snapshot Mamba retention anchor changed: MambaManager")
    replacements = [
        (
            "        super().remove_skipped_blocks(\n"
            "            request_id, processed_computed_tokens, num_prompt_tokens\n"
            "        )\n",
            "        # Retain snapshot states on release; keep allocation's normal window.\n"
            "        release_tokens = processed_computed_tokens\n"
            "        if self.mamba_cache_mode == 'align':\n"
            "            release_tokens = max(0, release_tokens - 3 * self.block_size)\n"
            "        super().remove_skipped_blocks(\n"
            "            request_id, release_tokens, num_prompt_tokens\n"
            "        )\n",
            "# Retain snapshot states on release; keep allocation's normal window.",
        ),
        (
            "                < cdiv(processed_computed_tokens, self.block_size) - 1\n",
            "                # Keep the same bounded snapshot tail as the prefix release path.\n"
            "                < cdiv(processed_computed_tokens, self.block_size) - 4\n",
            "# Keep the same bounded snapshot tail as the prefix release path.",
        ),
    ]
    for old, new, marker in replacements:
        if marker in body:
            continue
        if body.count(old) != 1:
            raise ValueError(f"snapshot Mamba retention anchor changed: {marker}")
        body = body.replace(old, new)
    return prefix + "class MambaManager(" + body


def add_fair_output(text: str) -> str:
    if FAIR_OUTPUT_NEW in text:
        return text
    if text.count(FAIR_OUTPUT_OLD) != 1:
        raise ValueError("fair scheduler output anchor changed")
    return text.replace(FAIR_OUTPUT_OLD, FAIR_OUTPUT_NEW)


def add_fair_runner_hooks(text: str) -> str:
    if FAIR_RUNNER_NEW not in text:
        if text.count(FAIR_RUNNER_OLD) != 1:
            raise ValueError("fair scheduler runner anchor changed")
        text = text.replace(FAIR_RUNNER_OLD, FAIR_RUNNER_NEW)
    if FAIR_PREPARE_NEW not in text:
        if text.count(FAIR_PREPARE_OLD) < 1:
            raise ValueError("fair scheduler cache-prepare anchor changed")
        text = text.replace(FAIR_PREPARE_OLD, FAIR_PREPARE_NEW)
    return text


def transformed_sources(
    package_root: Path,
    cache_source: Path,
    tier_source: Path,
    fair_source: Path | None = None,
) -> dict[Path, str]:
    fair_source = fair_source or tier_source.with_name("radiance_fair_scheduler.py")
    scheduler = (
        package_root / "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
    )
    factory = package_root / "vllm/v1/kv_offload/tiering/factory.py"
    mamba = package_root / "vllm/v1/core/single_type_kv_cache_manager.py"
    serving = package_root / "vllm/entrypoints/openai/chat_completion/serving.py"
    output = package_root / "vllm/v1/core/sched/output.py"
    runner = package_root / "vllm/v1/worker/gpu/model_runner.py"
    engine = package_root / "vllm/v1/engine/core.py"
    worker = package_root / "vllm/v1/worker/gpu_worker.py"
    text = scheduler.read_text()
    replacements = [
        (
            "        req_context = _create_req_context(request)\n"
            "        offloading_context = self.manager.on_new_request(req_context)",
            "        req_context = _create_req_context(request)\n"
            "        # Authenticate the cache namespace before any GPU/CPU reuse.\n"
            "        chat = (request.kv_transfer_params or {}).get('qwen_chat')\n"
            "        if chat is not None:\n"
            "            from qwen_radiance_cache import cache_salt\n"
            "            if request.cache_salt != cache_salt(chat):\n"
            "                raise ValueError('chat snapshot cache salt mismatch')\n"
            "        offloading_context = self.manager.on_new_request(req_context)",
            "# Authenticate the cache namespace before any GPU/CPU reuse.",
        ),
        (
            "            num_offloadable_tokens = self._calc_num_offloadable_tokens(\n"
            "                req_status, num_tokens_after_batch\n            )\n",
            "            num_offloadable_tokens = self._calc_num_offloadable_tokens(\n"
            "                req_status, num_tokens_after_batch\n            )\n"
            "            # Record the final head before the settled-tail store cascade.\n"
            "            if req.is_finished():\n"
            "                for tier in getattr(self.manager, 'secondary_tiers', ()):\n"
            "                    if hasattr(tier, 'set_snapshot_head'):\n"
            "                        tier.set_snapshot_head(req_status, num_offloadable_tokens)\n",
            "# Record the final head before the settled-tail store cascade.",
        ),
        (
            "        meta = OffloadingConnectorMetadata(\n"
            "            load_jobs=self._current_batch_load_jobs,\n"
            "            store_jobs=self._build_store_jobs(scheduler_output),\n"
            "            jobs_to_flush=self._current_batch_jobs_to_flush,\n"
            "        )\n",
            "        # A zero-token handover frame must finish every prior GPU\n"
            "        # store before the worker swaps the packed cache image.\n"
            "        store_jobs = self._build_store_jobs(scheduler_output)\n"
            "        fair = getattr(scheduler_output, 'qwen_fair', None)\n"
            "        if fair is not None and fair.get('barrier'):\n"
            "            self._current_batch_jobs_to_flush.update(\n"
            "                job_id for job_id, status in self._jobs.items() if status.is_store\n"
            "            )\n\n"
            "        meta = OffloadingConnectorMetadata(\n"
            "            load_jobs=self._current_batch_load_jobs,\n"
            "            store_jobs=store_jobs,\n"
            "            jobs_to_flush=self._current_batch_jobs_to_flush,\n"
            "        )\n",
            "# A zero-token handover frame must finish every prior GPU",
        ),
    ]
    for old, new, marker in replacements:
        if marker in text:
            continue
        if (
            marker == "# A zero-token handover frame must finish every prior GPU"
            and "        partial_store_jobs = self._build_partial_tail_store_jobs(" in text
        ):
            # v0.28 builds partial-tail jobs separately. Preserve both sets and
            # fence every store before our physical cache-bank handover.
            old = (
                "        normal_store_jobs = self._build_store_jobs(scheduler_output)\n"
                "        meta = OffloadingConnectorMetadata(\n"
            )
            new = (
                "        normal_store_jobs = self._build_store_jobs(scheduler_output)\n"
                "        # A zero-token handover frame must finish every prior GPU\n"
                "        # store, including v0.28 partial-tail stores, before the swap.\n"
                "        fair = getattr(scheduler_output, 'qwen_fair', None)\n"
                "        if fair is not None and fair.get('barrier'):\n"
                "            self._current_batch_jobs_to_flush.update(\n"
                "                job_id for job_id, status in self._jobs.items() "
                "if status.is_store\n"
                "            )\n"
                "        meta = OffloadingConnectorMetadata(\n"
            )
        if text.count(old) != 1:
            raise ValueError(f"chat snapshot scheduler anchor changed: {marker}")
        text = text.replace(old, new)
    factory_text = factory.read_text()
    registration = (
        '\nSecondaryTierFactory.register_tier("qwen_chat_fs", '
        '"qwen_radiance_chat_tier", "ChatFileSystemTierManager")\n'
    )
    if registration not in factory_text:
        factory_text += registration
    return {
        scheduler: text,
        factory: factory_text,
        mamba: retain_settled_mamba_tail(mamba.read_text()),
        serving: stream_buffered_tool_usage(serving.read_text()),
        output: add_fair_output(output.read_text()),
        runner: add_fair_runner_hooks(runner.read_text()),
        engine: tool_handover_core(engine.read_text()),
        worker: memory_report_worker(worker.read_text()),
        package_root / "qwen_radiance_cache.py": cache_source.read_text(),
        package_root / "qwen_radiance_chat_tier.py": tier_source.read_text(),
        package_root / "qwen_radiance_fair_scheduler.py": fair_source.read_text(),
        package_root / "qwen_radiance_memory.py": cache_source.with_name(
            "radiance_memory.py"
        ).read_text(),
    }


def install(
    package_root: Path,
    cache_source: Path,
    tier_source: Path,
    fair_source: Path | None = None,
) -> None:
    # Validate every transformed source before writing any runtime module.
    sources = transformed_sources(package_root, cache_source, tier_source, fair_source)
    for path, source in sources.items():
        compile(source, str(path), "exec")
    for path, source in sources.items():
        path.write_text(source)

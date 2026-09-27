"""DFlash caches target context, not EAGLE's shifted lookahead state.

The V2 DFlash input kernel masks rejected rows and writes query KV strictly at
positions >= the processed target prefix. Full blocks ending at that prefix
are reusable. Keep the EAGLE/MTP exclusion for those algorithms; do not change
physical block sizing or any numerical kernel.
"""

from pathlib import Path


def replace_once(text, old, new):
    if new in text:
        return text
    if text.count(old) != 1:
        raise ValueError("DFlash response-cache source anchor changed")
    return text.replace(old, new)


def transformed_sources(package: Path) -> dict[Path, str]:
    utils = package / "vllm/v1/core/kv_cache_utils.py"
    scheduler = package / "vllm/v1/core/sched/scheduler.py"
    offload = (
        package / "vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
    )
    annotated = replace_once(
        utils.read_text(),
        '    """Flag only groups that contain volatile draft-attention state."""\n'
        "    spec_config = vllm_config.speculative_config\n",
        '    """Flag only groups that contain volatile draft-attention state."""\n'
        "    spec_config = vllm_config.speculative_config\n"
        "    # DFlash writes valid target-derived context before its query suffix.\n"
        "    # It has no EAGLE-style shifted volatile token inside that prefix.\n"
        "    if spec_config is not None and spec_config.method == 'dflash':\n"
        "        return\n",
    )
    scheduled = replace_once(
        scheduler.read_text(),
        "            use_eagle=self.use_eagle,\n",
        "            use_eagle=self.use_eagle and not (\n"
        "                speculative_config is not None\n"
        "                and speculative_config.method == 'dflash'\n"
        "            ),\n",
    )
    offloaded = replace_once(
        offload.read_text(),
        "            and vllm_config.speculative_config.use_eagle()\n",
        "            and vllm_config.speculative_config.use_eagle()\n"
        "            and vllm_config.speculative_config.method != 'dflash'\n",
    )
    offloaded = replace_once(
        offloaded,
        "            elif req.is_finished():\n"
        "                num_tokens_after_batch = req.num_tokens\n",
        "            elif req.is_finished():\n"
        "                # The emitted bonus/correction token may still be pending.\n"
        "                num_tokens_after_batch = min(req.num_tokens, req.num_computed_tokens)\n",
    )
    offloaded = replace_once(
        offloaded,
        "        hash_idx = boundary_tokens // self.config.tokens_per_hash - 1\n",
        "        endpoint = getattr(request, 'qwen_response_end_lookup', None)\n"
        "        if endpoint is not None and endpoint['tokens'] == boundary_tokens:\n"
        "            from qwen_radiance_response_offload import endpoint_key\n"
        "            return endpoint_key(endpoint, group_idx)\n"
        "        hash_idx = boundary_tokens // self.config.tokens_per_hash - 1\n",
    )
    offloaded = replace_once(
        offloaded,
        "    def _lookup(self, req_status: RequestOffloadState) -> int | None:\n",
        "    def _lookup(self, req_status: RequestOffloadState) -> int | None:\n"
        "        if getattr(req_status.req, 'qwen_response_end_local', 0):\n"
        "            # A pinned, matching GPU endpoint needs no external reads.\n"
        "            req_status.partial_tail_boundary = None\n"
        "            return 0\n"
        "        from qwen_radiance_response_offload import lookup_response_end\n"
        "        handled, hit = lookup_response_end(self, req_status)\n"
        "        if handled:\n"
        "            return hit\n",
    )
    offloaded = replace_once(
        offloaded,
        "        normal_store_jobs = self._build_store_jobs(scheduler_output)\n",
        "        normal_store_jobs = self._build_store_jobs(scheduler_output)\n"
        "        from qwen_radiance_response_offload import store_response_ends\n"
        "        normal_store_jobs.update(store_response_ends(self, scheduler_output))\n",
    )
    offloaded = replace_once(
        offloaded,
        "        if self._snapshot_settled_tail_only:\n"
        "            unpublished_finished_req_id = next(\n",
        "        # GPU reuse has pinned its matching endpoint and uses ordinary CoW.\n"
        "        # It cannot race snapshot publication because it reads no snapshot.\n"
        "        # Existing block-transfer fences still protect every writable page.\n"
        "        if self._snapshot_settled_tail_only and not getattr(request, 'qwen_response_end_local', 0):\n"
        "            unpublished_finished_req_id = next(\n",
    )
    return {utils: annotated, scheduler: scheduled, offload: offloaded}


def install(package: Path):
    changes = transformed_sources(package)
    for path, text in changes.items():
        compile(text, str(path), "exec")
    for path, text in changes.items():
        if path.read_text() != text:
            path.write_text(text)

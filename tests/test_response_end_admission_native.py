"""CPU-only admission regression against the deployed vLLM allocator.

Run inside the pinned backend image. No model, tensors or GPU execution needed.
The counts reproduce the metadata captured in the 30 September cache stall.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("vllm.v1.core.kv_cache_manager")


def test_changed_long_prompt_admits_after_unused_checkpoint_reclaim(tmp_path):
    import torch
    from vllm import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec,
        SlidingWindowSpec,
    )
    from vllm.v1.request import Request

    root = Path(__file__).parents[1] / "experiments/radiance-public"
    modules = []
    for name, filename in (
        ("qwen_radiance_response_end", "radiance_response_end.py"),
        ("admission_fair_scheduler", "radiance_fair_scheduler.py"),
    ):
        spec = importlib.util.spec_from_file_location(name, root / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    response_end, fair = modules
    attention = dict(block_size=1648, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
    specs = [MambaSpec(
        block_size=1648, shapes=((4, 8), (8, 8)),
        dtypes=(torch.bfloat16, torch.float32), mamba_cache_mode="align",
        num_speculative_blocks=7,
    ) for _ in range(6)] + [
        FullAttentionSpec(**attention), FullAttentionSpec(**attention),
        SlidingWindowSpec(**attention, sliding_window=256),
    ]
    manager = KVCacheManager(
        KVCacheConfig(370, [], [KVCacheGroupSpec([f"synthetic.{i}"], spec)
                                for i, spec in enumerate(specs)]),
        max_model_len=253792, scheduler_block_size=1648, hash_block_size=1648,
        max_in_flight_tokens=2048, use_eagle=False,
    )
    endpoint = response_end.ResponseEndCache(manager)
    manager.qwen_response_end = endpoint
    endpoint.entry = {
        "tokens": 152868, "identity": ("different-prefix", None, ()),
        "pins": manager.block_pool.get_new_blocks(194), "blocks": (),
    }
    request = Request(
        request_id="synthetic-changed-prefix", prompt_token_ids=[1] * 152910,
        sampling_params=SamplingParams(max_tokens=1), pooling_params=None,
    )
    request.block_hashes = [i.to_bytes(32, "big") for i in range(152910 // 1648)]
    assert endpoint.lookup(request) is None
    assert manager.block_pool.get_num_free_blocks() == 175
    kwargs = dict(full_sequence_must_fit=True, has_scheduled_reqs=False)
    # The pre-fix policy cannot release anything and fails on every retry.
    for _ in range(3):
        assert manager.allocate_slots(request, 1648, **kwargs) is None
        assert not endpoint.release_after_progress(request)
        assert manager.block_pool.get_num_free_blocks() == 175

    scheduler = fair.FairScheduler.__new__(fair.FairScheduler)
    scheduler.response_end_enabled = True
    scheduler.status_path = str(tmp_path / "scheduler")
    scheduler.connector = None
    banks = fair.CacheBanks(manager, lambda: manager)
    banks.pressure_handler = scheduler._reclaim_cache_pressure
    assert banks.allocate_slots(request, 1648, **kwargs) is not None
    assert endpoint.entry is None
    assert manager.block_pool.get_num_free_blocks() > 0

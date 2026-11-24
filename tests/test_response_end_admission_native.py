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
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
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
    attention = {"block_size": 1648, "num_kv_heads": 1, "head_size": 128, "dtype": torch.bfloat16}
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
    kwargs = {"full_sequence_must_fit": True, "has_scheduled_reqs": False}
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


def test_earlier_exact_endpoint_admits_with_native_partial_hit_allocator():
    """An old recurrent state remains an ordinary, independently pinned hit."""
    import torch
    from vllm import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
        SlidingWindowSpec,
    )
    from vllm.v1.request import Request

    root = Path(__file__).parents[1] / "experiments/radiance-public"
    spec = importlib.util.spec_from_file_location(
        "native_fallback_response_end", root / "radiance_response_end.py"
    )
    response_end = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = response_end
    spec.loader.exec_module(response_end)
    attention = {"block_size": 1648, "num_kv_heads": 1, "head_size": 128, "dtype": torch.bfloat16}
    specs = [MambaSpec(
        block_size=1648, shapes=((4, 8), (8, 8)),
        dtypes=(torch.bfloat16, torch.float32), mamba_cache_mode="align",
        num_speculative_blocks=7,
    ) for _ in range(6)] + [
        FullAttentionSpec(**attention), FullAttentionSpec(**attention),
        SlidingWindowSpec(**attention, sliding_window=256),
    ]
    manager = KVCacheManager(
        KVCacheConfig(370, [], [KVCacheGroupSpec([f"synthetic.{i}"], item)
                                for i, item in enumerate(specs)]),
        max_model_len=253792, scheduler_block_size=1648, hash_block_size=1648,
        max_in_flight_tokens=2048, use_eagle=False,
    )
    endpoint = response_end.ResponseEndCache(manager)
    manager.qwen_response_end = endpoint
    request = Request(
        request_id="synthetic-changed-generated-tail", prompt_token_ids=[1] * 1800,
        sampling_params=SamplingParams(max_tokens=1), pooling_params=None,
    )
    request.block_hashes = [b"a" * 32]
    old_blocks = manager.block_pool.get_new_blocks(len(specs))
    earlier = {
        "tokens": 1640,
        "identity": response_end.prefix_identity(request, 1640, 1648),
        "pins": old_blocks,
        "blocks": tuple([block] for block in old_blocks),
    }
    recent_blocks = manager.block_pool.get_new_blocks(len(specs))
    endpoint.entry = {
        "tokens": 1700,
        "identity": response_end.prefix_identity(request, 1700, 1648),
        "pins": recent_blocks,
        "blocks": (),
    }
    endpoint.previous_entry = earlier
    changed_prompt = [1] * 1800
    changed_prompt[1699] = 2
    request = Request(
        request_id="synthetic-changed-generated-tail", prompt_token_ids=changed_prompt,
        sampling_params=SamplingParams(max_tokens=1), pooling_params=None,
    )
    request.block_hashes = [b"a" * 32]
    hit = endpoint.lookup(request)
    assert hit[1] == 1640
    assert endpoint.entry is earlier and endpoint.previous_entry is None
    assert request._qwen_response_end_lease is endpoint.entry
    assert all(block.ref_cnt == 0 for block in recent_blocks)
    assert manager.allocate_slots(
        request, 8, num_new_computed_tokens=hit[1], new_computed_blocks=hit[0],
        num_lookahead_tokens=7, full_sequence_must_fit=False,
        has_scheduled_reqs=False,
    ) is not None
    # Admission owns the reused pages/CoW sources independently of the optional
    # cache lease, exactly as on the existing latest-endpoint path.
    assert all(block.ref_cnt > 1 for block in old_blocks)

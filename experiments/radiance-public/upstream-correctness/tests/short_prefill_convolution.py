# Regression adapted from the upstream PR recorded in ../manifest.json.
# This module is collected only by the explicit isolated-GPU probe runner.
import os
import pytest
if os.environ.get("QWEN_UPSTREAM_GPU_TESTS") != "1":
    pytest.skip("requires the isolated GPU qualification runner", allow_module_level=True)
import torch
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.platforms import current_platform
DEVICE = "cuda:0"
def set_random_seed(seed):
    torch.manual_seed(seed)


def _apc_prefill_chunk(
    x_chunk,
    weight,
    bias,
    conv_states,
    cache_indices,
    initial_state_idx,
    block_idx_last,
    num_computed_tokens,
    block_size,
    has_initial_state,
):
    """Run one prefill chunk of a single sequence through the prefix-caching path."""
    device = conv_states.device
    i32 = {"dtype": torch.int32, "device": device}
    return causal_conv1d_fn(
        x_chunk,
        weight,
        bias,
        conv_states=conv_states,
        query_start_loc=torch.tensor([0, x_chunk.shape[-1]], **i32),
        cache_indices=torch.tensor([cache_indices], **i32),
        has_initial_state=torch.tensor([has_initial_state], device=device),
        activation="silu",
        block_idx_first_scheduled_token=torch.tensor([block_idx_last], **i32),
        block_idx_last_scheduled_token=torch.tensor([block_idx_last], **i32),
        initial_state_idx=torch.tensor([initial_state_idx], **i32),
        num_computed_tokens=torch.tensor([num_computed_tokens], **i32),
        block_size_to_align=block_size,
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="the prefix-caching conv-state path is exercised on CUDA-alike only",
)
@pytest.mark.parametrize("dim", [64, 4096])
@pytest.mark.parametrize("width", [4])
def test_causal_conv1d_apc_short_chunk_writes_last_scheduled_block(dim, width):
    """A prefill chunk shorter than the conv state must store to its own block.

    With prefix caching the initial conv state is read from a shared prefix
    block, which is not the block of the sequence's last scheduled token. A
    chunk shorter than ``width - 1`` takes the "shift left" branch; if that
    branch stores through the pointer it read the initial state from, the
    sequence's own block is never written and the shared prefix block is
    overwritten. The next chunk then convolves whatever bytes its block holds,
    and every later sequence that reuses the prefix reads a corrupted state.
    """
    device = DEVICE
    set_random_seed(0)
    state_len = width - 1
    block_size = 16
    n_blocks = 4
    dtype = torch.float32
    # Blocks: 0 is the null block, 1 is the shared prefix block holding the
    # initial state, 2 is the sequence's own block for the short chunk, 3 is
    # its block for the chunk after that.
    prefix_block, own_block, next_block = 1, 2, 3

    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype)
    initial_state = torch.randn(dim, state_len, device=device, dtype=dtype)
    # (dim, tokens): the kernel takes x channel-last.
    x = torch.randn(5, dim, device=device, dtype=dtype).T.contiguous()

    def fresh_states():
        states = torch.zeros(n_blocks, dim, state_len, device=device, dtype=dtype)
        states[prefix_block] = initial_state
        return states

    # Reference: the same five tokens as one chunk long enough to take the
    # regular branch, from the same initial state.
    ref_states = fresh_states()
    out_ref = _apc_prefill_chunk(
        x,
        weight,
        bias,
        ref_states,
        cache_indices=[prefix_block, next_block],
        initial_state_idx=0,
        block_idx_last=1,
        num_computed_tokens=block_size,
        block_size=block_size,
        has_initial_state=True,
    )

    states = fresh_states()
    # Seed the sequence's own block with the bytes a released block can hold, so
    # a missing store is loud rather than plausible.
    states[own_block] = float("nan")

    # Chunk A: two tokens, shorter than the three-token conv state.
    out_a = _apc_prefill_chunk(
        x[:, :2],
        weight,
        bias,
        states,
        cache_indices=[prefix_block, own_block],
        initial_state_idx=0,
        block_idx_last=1,
        num_computed_tokens=block_size,
        block_size=block_size,
        has_initial_state=True,
    )

    assert torch.equal(states[prefix_block], initial_state), (
        "the shared prefix block must not be written by a later sequence's chunk"
    )
    expected_own = torch.cat([initial_state[:, 2:], x[:, :2]], dim=-1)
    assert torch.allclose(states[own_block], expected_own), (
        "the short chunk's conv state must land in its own block"
    )
    assert torch.allclose(out_a, out_ref[:, :2], rtol=1e-4, atol=1e-4)

    # Chunk B: the next three tokens, whose initial state is chunk A's block.
    out_b = _apc_prefill_chunk(
        x[:, 2:],
        weight,
        bias,
        states,
        cache_indices=[own_block, next_block],
        initial_state_idx=0,
        block_idx_last=1,
        num_computed_tokens=block_size + 2,
        block_size=block_size,
        has_initial_state=True,
    )
    assert torch.isfinite(out_b).all()
    assert torch.allclose(out_b, out_ref[:, 2:], rtol=1e-4, atol=1e-4), (
        "two chunks over a cached prefix must equal the same tokens in one chunk"
    )

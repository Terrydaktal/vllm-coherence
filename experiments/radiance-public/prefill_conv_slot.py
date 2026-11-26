"""Initialize the scheduler-selected GDN convolution history on the current stream."""

import torch
import triton
import triton.language as tl

# Both prepared scan adapters are derived from the same pinned convolution
# preamble. Requiring an exact match preserves its scheduler metadata checks.
CONV_REPLACEMENT = (
    """    old_conv = conv_state.index_select(0, conv_ids.long())
    conv_state.index_copy_(
        0,
        conv_ids.long(),
        t.where(md.has_initial_state[:, None, None], old_conv, t.zeros_like(old_conv)),
    )""",
    """    conv_initializer(conv_state, conv_ids, md.has_initial_state)""",
)


@triton.jit
def _initialize_slot(
    state,
    slots,
    resumed,
    slot_stride,
    capacity,
    elements: tl.constexpr,
    block: tl.constexpr,
):
    slot = tl.load(slots).to(tl.int64)
    valid = (slot >= 0) & (slot < capacity)
    tl.device_assert(valid, "GDN prefill index outside convolution bank")
    if valid & (tl.load(resumed) == 0):
        offset = tl.program_id(0) * block + tl.arange(0, block)
        tl.store(state + slot * slot_stride + offset, 0, mask=offset < elements)


def initialize_conv_slot(state, slots, resumed):
    """Keep resumed history verbatim and zero the chosen cold slot before convolution."""
    if state.ndim != 3:
        raise ValueError(
            "expected convolution state with batch, feature and history dimensions"
        )
    n, features, history = state.shape
    if (
        features != 10240
        or history not in (3, 10)
        or state.dtype != torch.bfloat16
        or state.device.type != "cuda"
        or state.stride()[1:] not in ((1, features), (history, 1))
        or state.stride(0) < features * history
        or slots.shape != (1,)
        or slots.dtype not in (torch.int32, torch.int64)
        or resumed.shape != (1,)
        or resumed.dtype != torch.bool
        or slots.device != state.device
        or resumed.device != state.device
    ):
        # Preserve the stock slot lookup, bounds handling and strided stores.
        selected = state.index_select(0, slots.long())
        state.index_copy_(
            0,
            slots.long(),
            torch.where(resumed[:, None, None], selected, torch.zeros_like(selected)),
        )
        return False
    size = features * history
    _initialize_slot[(triton.cdiv(size, 512),)](
        state, slots, resumed, state.stride(0), n, size, 512, num_warps=4, debug=True
    )
    return True

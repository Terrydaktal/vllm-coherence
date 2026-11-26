"""Expose a checked small-row custom-op fallback to the 320-position native tape."""

import inspect

from native_d7_replay_tape import TapeDispatch


def projection_fallbacks(model):
    fallbacks = {}
    for layer in model.modules():
        if type(layer).__name__ != "QwenGatedDeltaNetAttention":
            continue
        variables = inspect.getclosurevars(layer._output_projection).nonlocals
        if "original" in variables and "direct_projection" in variables:
            fallbacks[layer.out_proj.weight.data_ptr()] = variables["original"]
    if len(fallbacks) != 48:
        raise ValueError("expected 48 retained GDN routing boundaries")
    return fallbacks


class RetainedTapeDispatch(TapeDispatch):
    def __init__(self, tape, fallbacks):
        super().__init__(tape)
        self.fallbacks = fallbacks

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        if self.tape.busy or name != "coherence_gdn.direct_tile_projection.default":
            return super().__torch_dispatch__(func, types, args, kwargs)
        return self.capture_projection(func, args, kwargs or {})

    def capture_projection(self, func, args, kwargs):
        import torch

        x, z, _, _, weight, _, _ = args
        if kwargs or x.shape[0] != 8 or x.shape != z.shape:
            raise ValueError("retained native stage tape requires an eight-row GDN cut")
        # Our mode is popped while __torch_dispatch__ is running. Execute the
        # actual retained op once to establish its full BF16 output, then
        # record the exact loaded low-row fallback's constituent operations.
        expected = func(*args)
        original = self.fallbacks[weight.data_ptr()]
        with self:
            recorded = original(x, z)
        if (expected.shape != recorded.shape or expected.dtype != recorded.dtype
                or not torch.equal(expected.contiguous().view(torch.uint8), recorded.contiguous().view(torch.uint8))):
            raise ValueError("recorded GDN fallback differs from the retained native op")
        return recorded

"""Opt-in single-sequence prefill and indexed-decode data-movement experiments."""

import inspect
from types import MethodType

import torch


def indexed_scan_into(bank, slot, resumed, mixed, a, b, alog, bias, output):
    from prefill_indexed_kernel import indexed_scan
    from prepared_prefill_scan import prepare_inputs

    rows = mixed.shape[0]
    if (
        not 128 <= rows <= 4096
        or bank.dtype != torch.float32
        or bank.shape[1:] != (48, 128, 128)
        or bank.stride()[1:] != (16384, 128, 1)
        or bank.stride(0) < 48 * 128 * 128
    ):
        raise ValueError("unsupported indexed prefill bank")
    if (
        slot.shape != (1,)
        or slot.dtype not in (torch.int32, torch.int64)
        or resumed.shape != (1,)
        or resumed.dtype != torch.bool
    ):
        raise ValueError("unsupported indexed prefill metadata")
    for arg, shape, dtype in (
        (mixed, (rows, 10240), torch.bfloat16),
        (a, (rows, 48), torch.bfloat16),
        (b, (rows, 48), torch.bfloat16),
        (alog, (48,), torch.float32),
        (bias, (48,), torch.bfloat16),
        (output, (rows, 48, 128), torch.bfloat16),
    ):
        if arg.shape != shape or arg.dtype != dtype or arg.stride(-1) != 1:
            raise ValueError("unsupported indexed prefill activation layout")
    if (
        not output.is_contiguous()
        or bank.device.type != "cuda"
        or any(
            arg.device != bank.device
            for arg in (slot, resumed, mixed, a, b, alog, bias, output)
        )
    ):
        raise ValueError(
            "indexed prefill requires a contiguous output on the same device"
        )
    qk = torch.empty((rows, 32, 128), dtype=torch.float32, device=mixed.device)
    gates = torch.empty((rows, 48, 2), dtype=torch.float32, device=mixed.device)
    prepare_inputs[(rows, 48)](
        mixed,
        a,
        b,
        alog,
        bias,
        qk,
        gates,
        128**-0.5,
        mixed.stride(0),
        a.stride(0),
        b.stride(0),
        rows,
        False,
        num_warps=1,
        enable_fp_fusion=True,
        allow_flush_denorm=False,
    )
    indexed_scan[(32, 48)](
        bank,
        mixed,
        qk,
        gates,
        output,
        slot,
        resumed,
        bank.stride(0),
        bank.shape[0],
        rows,
        mixed.stride(0),
        4,
        2,
        False,
        num_warps=1,
        num_stages=3,
        enable_fp_fusion=True,
        allow_flush_denorm=False,
        debug=True,
    )


def install_bank(worker, *, conv_initializer=None):
    from gdn_prefill_direct import _replace_function
    from stock_gdn_prefill_adapter import StockPrefillAdapter

    import radiance_gdn
    from qwen_r9700_lab.conformance_instrumentation import HookSet

    performance = worker._qwen_performance_repairs
    adapter = worker._qwen_persistent_repairs.prefill
    previous = radiance_gdn.forward_core_fused
    if (
        type(adapter) is not StockPrefillAdapter
        or getattr(previous, "__self__", None) is not adapter
        or not performance.manifest["prefill_alignment"]["prepared_scan"]
    ):
        raise ValueError("indexed bank requires the qualified prefill adapter")
    counts = {"indexed_state": 0}
    conv_replacements = ()
    if conv_initializer is not None:
        from prefill_conv_slot import CONV_REPLACEMENT

        conv_replacements = (CONV_REPLACEMENT,)
    updated = _replace_function(
        StockPrefillAdapter.__call__,
        (
            (
                "        return self.original(layer, mixed_qkv, b, a, core_attn_out)",
                "        return previous(layer, mixed_qkv, b, a, core_attn_out)",
            ),
            (
                "    old_conv = ",
                """    if not 128 <= count <= 1024 or not core_attn_out[:count].is_contiguous():
        return previous(layer, mixed_qkv, b, a, core_attn_out)
    old_conv = """,
            ),
            (
                """    selected = ssm_state.index_select(0, md.prefill_state_indices.long())
    initial = t.where(
        md.prefill_has_initial_state[:, None, None, None], selected, t.zeros_like(selected)
    )[0].contiguous()
""",
                "",
            ),
            (
                """    result = self.scan.run(initial, packed, a[:count], b[:count], layer.A_log, layer.dt_bias)
    core_attn_out[:count].view(count, 48, 128).copy_(result.outputs)
    ssm_state.index_copy_(0, md.prefill_state_indices.long(), result.final_state.unsqueeze(0))""",
                """    counts["indexed_state"] += 1
    indexed_scan_into(ssm_state, md.prefill_state_indices, md.prefill_has_initial_state, packed,
                      a[:count], b[:count], layer.A_log, layer.dt_bias, core_attn_out[:count].view(count, 48, 128))""",
            ),
            *conv_replacements,
        ),
        previous=previous,
        counts=counts,
        indexed_scan_into=indexed_scan_into,
        conv_initializer=conv_initializer,
    )
    hooks = HookSet()
    hooks.replace(radiance_gdn, "forward_core_fused", MethodType(updated, adapter))
    performance._indexed_bank_hooks = hooks
    return counts


def hoisted_recurrent(adapter, stable_parameters):
    """Round each immutable FP32 gate parameter once, prior to graph capture."""
    from gdn_prefill_direct import _replace_function

    # Keep strong references to both sides. Unrecognised arguments retain the
    # normal conversion, including transient tensors whose addresses get reused.
    cached = {id(bias): (bias, bias.to(torch.bfloat16)) for bias in stable_parameters}

    def narrow(bias):
        known = cached.get(id(bias))
        return (
            known[1]
            if known is not None and known[0] is bias
            else bias.to(torch.bfloat16)
        )

    return _replace_function(
        type(adapter).__call__,
        (("bias = dt_bias.to(t.bfloat16)", "bias = narrow(dt_bias)"),),
        narrow=narrow,
    )


def install_bias(worker):
    import radiance_gdn
    from qwen_r9700_lab.conformance_instrumentation import HookSet

    captures = inspect.getclosurevars(radiance_gdn.recurrent_update).nonlocals
    adapter = captures.get("recur")
    modules = [
        m
        for m in worker.model_runner.model.modules()
        if type(m).__name__ == "QwenGatedDeltaNetAttention"
    ]
    if adapter is None or len(modules) != 48:
        raise ValueError("bias hoist needs the pinned packed GDN recurrence")
    widened = [radiance_gdn._gate_params(m)[1] for m in modules]
    hooks = HookSet()
    hooks.replace(type(adapter), "__call__", hoisted_recurrent(adapter, widened))
    worker._qwen_performance_repairs._decode_bias_hooks = hooks
    return {"cached_layer_biases": len(widened)}

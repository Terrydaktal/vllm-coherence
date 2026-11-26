"""Opt-in GDN prefill experiments for the corrected TP1 worker."""

import inspect
import textwrap
from types import MethodType

def _replace_function(function, replacements, **bindings):
    """Derive a Python wrapper from the installed qualified implementation."""
    source = textwrap.dedent(inspect.getsource(function))
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(f"unsupported GDN prefill wrapper {function.__qualname__}")
        source = source.replace(old, new, 1)
    namespace = dict(function.__globals__, **bindings)
    exec(compile(source, inspect.getfile(function), "exec"), namespace)
    return namespace[function.__name__]


def _prepared_scan_into():
    import prepared_prefill_scan

    return _replace_function(
        prepared_prefill_scan.run,
        (
            ("transposed=False,", "transposed=False,\n    output,"),
            (
                "out = torch.empty((rows, 48, 128), device=mixed.device, dtype=torch.bfloat16)",
                """if (
        output.shape != (rows, 48, 128)
        or output.dtype != mixed.dtype
        or output.device != mixed.device
        or not output.is_contiguous()
    ):
        raise ValueError("invalid prepared scan destination")
    out = output""",
            ),
        ),
    )


def install_scan(worker):
    """Write the prepared scan into the layer's own output allocation."""
    import radiance_gdn
    from optimized_prefill_scan import PrefillScan
    from qwen_r9700_lab.conformance_instrumentation import HookSet
    from stock_gdn_prefill_adapter import StockPrefillAdapter
    from stock_gdn_sequence import SequenceResult

    performance = worker._qwen_performance_repairs
    runtime = worker._qwen_persistent_repairs
    adapter = runtime.prefill
    if (
        type(adapter) is not StockPrefillAdapter
        or type(adapter.scan) is not PrefillScan
        or not performance.manifest["prefill_alignment"]["prepared_scan"]
        or radiance_gdn.forward_core_fused != adapter.__call__
    ):
        raise ValueError("direct GDN scan requires the installed prepared-scan route")

    run_into = _prepared_scan_into()
    calls = {"scan_direct": 0}

    def produce(adapter, initial, packed, a, b, a_log, dt_bias, output):
        if 127 < packed.shape[0] <= 4096 and output.is_contiguous():
            written, state = run_into(initial, packed, a, b, a_log, dt_bias, output=output)
            calls["scan_direct"] += 1
            return SequenceResult(written, state, {}, packed.shape[0])
        result = adapter.scan.run(initial, packed, a, b, a_log, dt_bias)
        output.copy_(result.outputs)
        return result

    direct = _replace_function(
        StockPrefillAdapter.__call__,
        ((
            "result = self.scan.run(initial, packed, a[:count], b[:count], layer.A_log, layer.dt_bias)\n"
            "    core_attn_out[:count].view(count, 48, 128).copy_(result.outputs)",
            "output = core_attn_out[:count].view(count, 48, 128)\n"
            "    result = produce(self, initial, packed, a[:count], b[:count], layer.A_log, layer.dt_bias, output)",
        ),),
        produce=produce,
    )
    hooks = HookSet()
    hooks.replace(radiance_gdn, "forward_core_fused", MethodType(direct, adapter))
    performance._gdn_direct_scan_hooks = hooks
    return calls

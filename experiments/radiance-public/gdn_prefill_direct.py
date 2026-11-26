"""Opt-in GDN prefill experiments for the corrected TP1 worker."""

import functools
import inspect
import textwrap
from types import MethodType

from triton.experimental import gluon
from triton.experimental.gluon import language as tl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout


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


def install_scan(worker, *, conv_initializer=None):
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

    conv_replacements = ()
    if conv_initializer is not None:
        from prefill_conv_slot import CONV_REPLACEMENT

        conv_replacements = (CONV_REPLACEMENT,)
    direct = _replace_function(
        StockPrefillAdapter.__call__,
        ((
            "result = self.scan.run(initial, packed, a[:count], b[:count], layer.A_log, layer.dt_bias)\n"
            "    core_attn_out[:count].view(count, 48, 128).copy_(result.outputs)",
            "output = core_attn_out[:count].view(count, 48, 128)\n"
            "    result = produce(self, initial, packed, a[:count], b[:count], layer.A_log, layer.dt_bias, output)",
        ), *conv_replacements),
        produce=produce,
        conv_initializer=conv_initializer,
    )
    hooks = HookSet()
    hooks.replace(radiance_gdn, "forward_core_fused", MethodType(direct, adapter))
    performance._gdn_direct_scan_hooks = hooks
    return calls


@gluon.jit
def _tiled_norm_quant(
    x_ptr, z_ptr, w_ptr, q_ptr, s_ptr, rows, sx: tl.constexpr, sz: tl.constexpr, eps
):
    # Match the stock M1-layout norm reduction and all rounding steps.
    row = tl.program_id(0)
    layout: tl.constexpr = BlockedLayout([1, 4], [1, 32], [8, 1], [1, 0])
    heads = tl.arange(0, 64, layout=SliceLayout(1, layout))
    cols = tl.arange(0, 128, layout=SliceLayout(0, layout))
    off = heads[:, None] * 128 + cols[None, :]
    active = (heads[:, None] < 48) & (row < rows)
    x = tl.load(x_ptr + row * sx + off, active, 0).to(tl.float32)
    z = tl.load(z_ptr + row * sz + off, active, 0).to(tl.float32)
    w = tl.load(w_ptr + cols).to(tl.float32)
    var = tl.sum(x * x, 1) / 128
    rstd = tl.rsqrt(var + eps)
    y = (x * rstd[:, None]) * w[None, :]
    y *= z * (1 / (1 + tl.exp(-z)))
    y = y.to(tl.bfloat16).to(tl.float32)
    maximum = tl.max(tl.max(tl.where(heads[:, None] < 48, tl.abs(y), 0.0), 1), 0)
    scale = tl.maximum(maximum / 448.0, 1.0 / (448.0 * 512.0))
    value = tl.minimum(tl.maximum(tl.div_rn(y, scale), -448.0), 448.0)
    # Invert prefill_activation_tiles._pack's offset calculation. The last
    # tile contains positive zero bytes in rows beyond the input.
    index = (row // 16) * (16 * 6144) + (off // 16) * 256
    index += ((off // 8) % 2) * 128 + (row % 16) * 8 + off % 8
    tl.store(q_ptr + index, tl.where(row < rows, value, 0.0), heads[:, None] < 48)
    tl.store(s_ptr + row, scale, row < rows)


def tiled_norm_quant(x, z, weight, eps):
    import torch

    rows = x.shape[0]
    padded_rows = (rows + 15) // 16 * 16
    tiled = torch.empty(padded_rows * 6144, dtype=torch.uint8, device=x.device)
    scale = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    _tiled_norm_quant[(padded_rows,)](
        x, z, weight, tiled.view(torch.float8_e4m3fn), scale, rows,
        x.stride(0), z.stride(0), eps, num_warps=8,
    )
    return tiled, scale


def aligned_tiled_projection(projection, tiled, scale, weight, weight_scale, ref, wperm):
    import torch

    rows = scale.shape[0]
    out = torch.empty((rows, 5120), device=tiled.device, dtype=torch.bfloat16)
    scratch = (
        (1,) if projection.manifest.get("register_partials")
        else (4, min(rows, projection.manifest["window"]), 5120)
    )
    partials = torch.empty(scratch, device=tiled.device, dtype=torch.float32)
    code = projection.launch(
        tiled.data_ptr(), weight.data_ptr(), weight_scale.data_ptr(),
        ref.data_ptr(), scale.data_ptr(), out.data_ptr(), partials.data_ptr(),
        rows, 5120, 6144, 1, int(wperm), torch.cuda.current_stream().cuda_stream,
    )
    if code:
        raise RuntimeError(f"tiled GDN projection launch failed: {code}")
    return out


def _project_for_runtime_rows(x, z, weight, originals, calls, direct):
    # vLLM reuses one symbolic Inductor graph across prefill and decode.
    # A size condition outside this opaque operator can be specialized away.
    rows = x.shape[0]
    if not 64 <= rows <= 4096:
        calls["fallback_rows"][rows] = calls["fallback_rows"].get(rows, 0) + 1
        return originals[weight.data_ptr()](x, z)
    calls["projection_direct"] += 1
    calls["direct_rows"][rows] = calls["direct_rows"].get(rows, 0) + 1
    return direct()


def install_projection(worker):
    """Bind the tiled producer only to the qualified GDN output GEMM."""
    import torch
    import radiance_mxfp4 as gemm
    from qwen_r9700_lab.conformance_instrumentation import HookSet
    from prefill_gemm_alignment import AlignedPrefillGemm

    performance = worker._qwen_performance_repairs
    if not performance.gdn_norm_quant or not performance.prefill_alignment:
        raise ValueError("direct GDN projection requires the qualified norm and GEMM routes")
    entry = performance.manifest["prefill_alignment"]["projection"]
    projection = AlignedPrefillGemm(entry["build"])
    if projection.manifest["sha256"] != entry["build_sha256"]:
        raise ValueError("direct GDN projection build changed")

    sites = [
        layer for layer in worker.model_runner.model.modules()
        if type(layer).__name__ == "QwenGatedDeltaNetAttention"
    ]
    if len(sites) != 48:
        raise ValueError("direct GDN projection requires the pinned 48-site inventory")
    eligible = []
    for layer in sites:
        linear = layer.out_proj
        if (
            linear.tp_size == 1
            and linear.input_is_parallel
            and linear.bias is None
            and all(
                t.dtype == torch.uint8
                and t.is_contiguous()
                and t.device == linear.weight.device
                and t.shape == shape
                for t, shape in (
                    (linear.weight, (5120, 3072)),
                    (linear.weight_scale, (192, 5120)),
                    (linear.radiance_wref, (5120,)),
                )
            )
        ):
            eligible.append(layer)
    if not eligible:
        raise ValueError("no qualified direct GDN projection weights")

    calls = {"projection_direct": 0, "direct_rows": {}, "fallback_rows": {}, "sites": len(eligible)}
    originals = {layer.out_proj.weight.data_ptr(): layer._output_projection for layer in eligible}
    if len(originals) != len(eligible):
        raise ValueError("GDN output projection weights must identify distinct layers")

    @torch.library.custom_op("coherence_gdn::direct_tile_projection", mutates_args=())
    def direct_projection(
        x: torch.Tensor, z: torch.Tensor, norm_weight: torch.Tensor, eps: float,
        weight: torch.Tensor, weight_scale: torch.Tensor, ref: torch.Tensor,
    ) -> torch.Tensor:
        def direct():
            tiled, scale = tiled_norm_quant(x, z, norm_weight, eps)
            return aligned_tiled_projection(projection, tiled, scale, weight, weight_scale, ref, gemm.WPERM)

        return _project_for_runtime_rows(x, z, weight, originals, calls, direct)

    @direct_projection.register_fake
    def fake(x, z, norm_weight, eps, weight, weight_scale, ref):
        return x.new_empty((x.shape[0], 5120))

    def wrap(layer):
        original = layer._output_projection

        @functools.wraps(original)
        def output_projection(x, z):
            if (
                x.ndim != 3
                or x.shape[1:] != (48, 128)
                or x.dtype != torch.bfloat16
                or z.dtype != x.dtype
                or x.shape != z.shape
                or x.device.type != "cuda"
                or x.device != z.device
                or x.device != layer.out_proj.weight.device
                or x.device != layer.norm.weight.device
                or x.stride()[1:] != (128, 1)
                or z.stride()[1:] != (128, 1)
            ):
                return original(x, z)
            out = layer.out_proj
            return direct_projection(
                x, z, layer.norm.weight, layer.norm.eps,
                out.weight, out.weight_scale, out.radiance_wref,
            )

        return output_projection

    hooks = HookSet()
    for layer in eligible:
        hooks.replace(layer, "_output_projection", wrap(layer))
    performance._gdn_direct_projection_hooks = hooks
    return calls

"""TP1 FP8 producer/consumer wiring with the corrected norm arithmetic.

Only decoder norms feeding known Radiance linears are replaced. The residual
and final hidden state keep their BF16 contract; the final model norm and
    drafter taps are untouched. Qualified prefill preserves its own reduction order.
"""


def install(model, hooks, entry):
    import radiance_mxfp4
    import torch
    from stock_fp8_epilogue import StockFP8Epilogue
    from vllm import _custom_ops as ops
    from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size

    if get_tensor_model_parallel_world_size() != 1 or get_pp_group().world_size != 1:
        raise RuntimeError("this FP8 backport admits TP1/PP1 only")
    if radiance_mxfp4.MIN_M > 0 or radiance_mxfp4.TRACED_QUANT or radiance_mxfp4.PURE_QUANT:
        raise RuntimeError("FP8 backport requires the qualified native per-token quantizer")
    cores = [
        m for m in model.modules() if hasattr(m, "layers") and hasattr(m, "aux_hidden_state_layers")
    ]
    if len(cores) != 1 or type(cores[0]).__name__ != "Qwen3_5Model":
        raise RuntimeError("FP8 stream requires the pinned Qwen3.5 decoder")
    # Qwen3NextModel captures DFlash taps between complete decoder layers.
    # Both the MLP output and residual are still BF16 there: FP8 tuples exist
    # only between each replaced norm and its immediate linear consumer.
    # Rejecting auxiliary taps here would reject the production DFlash model.
    layers = list(cores[0].layers)
    if len(layers) != 64:
        raise RuntimeError("FP8 stream layer count changed")
    for layer in layers:
        if (
            getattr(layer, "layer_scale", False)
            or layer.use_attn_reduce_scatter_for_moe
            or getattr(layer.mlp, "expert_gate", None) is not None
        ):
            raise RuntimeError("unsupported FP8 decoder branch")
        if layer.layer_type == "linear_attention":
            if not getattr(layer.linear_attn, "_rad_merged", False):
                raise RuntimeError("FP8 stream needs tuple-aware merged GDN projections")
        elif getattr(layer.self_attn.qkv_proj, "radiance_wref", None) is None:
            raise RuntimeError("FP8 attention consumer is not the Radiance prequantized linear")
        if getattr(layer.mlp.gate_up_proj, "radiance_wref", None) is None:
            raise RuntimeError("FP8 MLP consumer is not the Radiance prequantized linear")
    native = StockFP8Epilogue(entry["build"])
    originals = {}
    counts = {
        "norm": 0,
        "prefill": 0,
        "silu": 0,
        "silu_enabled": bool(entry.get("silu_enabled")),
        "bf16_auxiliary_taps": list(cores[0].aux_hidden_state_layers or ()),
    }

    def quant(t):
        return ops.scaled_fp8_quant(t, scale=None, use_per_token_if_dynamic=True)

    @torch.library.custom_op(
        "qwen_stock_fp8::norm",
        mutates_args=(),
        schema=(
            "(Tensor x, Tensor? residual, Tensor w, float eps, str key)"
            " -> (Tensor, Tensor, Tensor?)"
        ),
    )
    def norm(
        x: torch.Tensor, residual: torch.Tensor | None, w: torch.Tensor, eps: float, key: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if 1 <= x.shape[0] <= (2048 if entry.get("prefill_enabled") else 8):
            counts["norm" if x.shape[0] <= 8 else "prefill"] += 1
            return native.norm(x, residual, w, eps)
        counts["prefill"] += 1
        value = originals[key](x, residual)
        y, carry = value if residual is not None else (value, None)
        q, scale = quant(y)
        return q, scale, carry

    @norm.register_fake
    def _(x, residual, w, eps, key):
        return (
            torch.empty_like(x, dtype=torch.float8_e4m3fn, memory_format=torch.contiguous_format),
            torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32),
            torch.empty_like(x, memory_format=torch.contiguous_format)
            if residual is not None
            else None,
        )

    def wrap(module, key):
        def forward(x, residual=None):
            q, scale, carry = norm(x, residual, module.weight, module.variance_epsilon, key)
            return ((q, scale), carry) if residual is not None else (q, scale)

        return forward

    # Install only after all admission checks. Saved callables are the corrected
    # compiled norm wrappers, and remain the reference for the larger shapes.
    for i, layer in enumerate(layers):
        for name in ("input_layernorm", "post_attention_layernorm"):
            module = getattr(layer, name)
            key = f"{i}/{name}"
            originals[key] = module.forward
            hooks.replace(module, "forward", wrap(module, key))
    if entry.get("silu_enabled"):

        @torch.library.custom_op("qwen_stock_fp8::silu", mutates_args=())
        def silu(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if 1 <= x.shape[0] <= 8:
                counts["silu"] += 1
                return native.silu(x)
            return quant(torch.nn.functional.silu(x[:, :17408]) * x[:, 17408:])

        @silu.register_fake
        def _(x):
            return (
                torch.empty((x.shape[0], 17408), device=x.device, dtype=torch.float8_e4m3fn),
                torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32),
            )

        for layer in layers:
            hooks.replace(layer.mlp.act_fn, "forward", silu)
    return counts

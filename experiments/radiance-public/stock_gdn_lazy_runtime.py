"""Corrected FP32 lazy GDN hooks for the pinned V2 align-mode runtime.

The patch manifest shrinks the state window; this adapter never falls back to
an eager speculative kernel that would write into absent snapshot columns.
"""

import os


def enabled():
    return os.environ.get("QWEN_STOCK_GDN_LAZY") == "1"


class Tables:
    def __init__(self, ctx):
        import torch
        from vllm.model_executor.layers.mamba.mamba_utils import get_temporal_copy_spec

        cfg, fwd = ctx._radiance_kv_cfg, ctx._radiance_fwd_ctx
        self.keep = []
        columns = [[], [], [], [], []]
        for local, gid in enumerate(ctx.mamba_group_ids):
            for name in cfg.kv_cache_groups[gid].layer_names:
                layer = fwd[name]
                alog, bias = layer.A_log, layer.dt_bias
                if alog.dtype != torch.float32 or bias.dtype != torch.bfloat16:
                    raise RuntimeError("lazy GDN gate parameter dtypes changed")
                self.keep.append((layer, alog, bias))
                for index, state in enumerate(layer.kv_cache):
                    if ctx._radiance_copy_funcs[index] is not get_temporal_copy_spec:
                        continue
                    if (
                        state.dtype != torch.float32
                        or state.shape[1:] != (48, 128, 128)
                        or state.stride()[1:] != (16384, 128, 1)
                        or state.data_ptr() % 64
                        or state.stride(0) % 128
                    ):
                        raise RuntimeError("lazy GDN supports the corrected FP32 TP1 state only")
                    for col, value in zip(
                        columns,
                        (
                            state.data_ptr(),
                            state.stride(0),
                            local,
                            alog.data_ptr(),
                            bias.data_ptr(),
                        ),
                        strict=True,
                    ):
                        col.append(value)
        if not columns[0]:
            raise RuntimeError("lazy GDN has no temporal states")
        self.tensors = [
            torch.tensor(values, dtype=torch.int64, device="cuda") for values in columns
        ]
        self.errors = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.count = len(columns[0])


def materialize(ctx, mode, num_reqs, a, b, c, idx_mapping):
    from stock_gdn_lazy_kernel import materialize_align

    table = getattr(ctx, "_qwen_lazy_tables", None)
    if table is None:
        table = ctx._qwen_lazy_tables = Tables(ctx)
    states, strides, groups, alogs, biases = table.tensors
    if mode not in (0, 1):
        raise ValueError("unsupported lazy GDN copy mode")
    dummy = a
    state_indices, sources, offsets, accepted, computed = (
        (a, b, c, dummy, dummy) if mode == 0 else (b, dummy, dummy, a, c)
    )
    materialize_align[(192, num_reqs, table.count)](
        states,
        strides,
        groups,
        alogs,
        biases,
        ctx.block_table_ptrs,
        idx_mapping if idx_mapping is not None else dummy,
        state_indices,
        sources,
        offsets,
        accepted,
        computed,
        table.errors,
        128**-0.5,
        ctx.block_table_stride_req,
        ctx.block_size,
        MODE=mode,
        MAPPED=idx_mapping is not None,
        num_warps=1,
        num_stages=3,
        enable_fp_fusion=True,
        allow_flush_denorm=False,
        debug=True,
    )


def install(repairs, hooks, packed_convolution):
    import radiance_gdn as native
    import torch
    from stock_gdn_lazy_kernel import invalidate, lazy_update

    if not enabled():
        raise RuntimeError("lazy GDN adapter needs the matched allocator/metadata patches")
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    if "radiance_stash_indices" not in GDNAttentionMetadata.__dataclass_fields__:
        raise RuntimeError("lazy GDN metadata patch is missing")
    if repairs.prefill is None:
        raise RuntimeError("lazy GDN requires corrected prefill")
    original = native.forward_core_fused
    errors = torch.zeros(1, dtype=torch.int32, device="cuda")
    counts = {"decode": 0, "prefill_invalidations": 0, "abi": "stock-fp32-lazy-v1"}

    def forward(layer, mixed_qkv, b, a, output):
        md = native._metadata(layer)
        if md is None or md.num_actual_tokens == 0:
            return original(layer, mixed_qkv, b, a, output)
        plan = native._plan(layer, mixed_qkv, b, a, output)
        if plan is None:
            raise RuntimeError(
                "unsupported lazy GDN metadata; eager fallback would corrupt the cache"
            )
        kind, rows, conv, state, md, (indices, _) = plan
        if (
            state.dtype != torch.float32
            or tuple(state.shape[1:]) != (48, 128, 128)
            or state.stride()[1:] != (16384, 128, 1)
            or state.data_ptr() % 64
            or state.stride(0) % 128
        ):
            raise RuntimeError("lazy GDN state geometry changed")
        if kind == "prefill":
            slots = md.radiance_stash_indices
            if slots is None or slots.numel() != 1:
                raise RuntimeError("lazy prefill requires one stash index")
            invalidate[(48, 1)](state, slots.contiguous(), state.stride(0))
            counts["prefill_invalidations"] += 1
            return original(layer, mixed_qkv, b, a, output)
        if (
            kind != "decode"
            or md.num_spec_decodes != 1
            or rows < 1
            or rows > 8
            or indices.shape != (1, 2)
            or layer.num_spec != 7
            or layer.tp_size != 1
            or a.dtype != torch.bfloat16
            or b.dtype != torch.bfloat16
        ):
            raise RuntimeError("lazy GDN admits one unmixed TP1 D7 decode sequence")
        cu = md.spec_query_start_loc[:2]
        weights = layer.conv1d.weight.view(10240, 4)
        q, k, v = packed_convolution(
            mixed_qkv[:rows],
            weights,
            layer.conv1d.bias,
            conv,
            10,
            indices[:, 0],
            md.num_accepted_tokens,
            cu,
            1,
            rows,
            48,
            16,
            8,
        )
        # The qualified transport adapter returns views of this exact packed
        # allocation. Never synthesize a pointer from an unrelated Q tensor.
        if (
            q.untyped_storage().data_ptr() != k.untyped_storage().data_ptr()
            or q.untyped_storage().data_ptr() != v.untyped_storage().data_ptr()
            or q.stride(0) != 10240
            or q.storage_offset() != 0
        ):
            raise RuntimeError("lazy GDN requires the qualified packed convolution")
        packed = q.as_strided((rows, 10240), (10240, 1))
        alog, bias = layer.A_log, layer.dt_bias
        lazy_update[(4, 48, 1)](
            packed,
            a[:rows],
            b[:rows],
            alog,
            bias,
            state,
            output[:rows].view(rows, 48, 128),
            indices,
            md.num_accepted_tokens[:1],
            cu,
            errors,
            128**-0.5,
            10240,
            a.stride(0),
            b.stride(0),
            state.stride(0),
            indices.stride(0),
            num_warps=1,
            num_stages=3,
            enable_fp_fusion=True,
            allow_flush_denorm=False,
            debug=True,
        )
        layer.__dict__.pop("_radiance_z", None)
        if output.shape[0] > rows:
            output[rows:].zero_()
        counts["decode"] += 1
        return True

    hooks.replace(native, "forward_core_fused", forward)
    return counts

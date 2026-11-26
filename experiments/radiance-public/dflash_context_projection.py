"""Fixed reduction for the DFlash2 BF16 context inner/V projection."""

import torch
import triton
import triton.language as tl


@triton.jit
def _project(
    inputs,
    weight,
    bias,
    output,
    m,
    inner: tl.constexpr,
    width: tl.constexpr,
    block_rows: tl.constexpr,
    block_cols: tl.constexpr,
    block_inner: tl.constexpr,
    has_bias: tl.constexpr,
):
    row_tile = tl.program_id(0)
    col_tile = tl.program_id(1)
    rows = row_tile * block_rows + tl.arange(0, block_rows)
    cols = col_tile * block_cols + tl.arange(0, block_cols)
    inner_offsets = tl.arange(0, block_inner)
    accumulator = tl.full((block_rows, block_cols), 0.0, tl.float32)
    for k in range(inner // block_inner):
        a = tl.load(
            inputs + rows[:, None] * inner + (k * block_inner + inner_offsets)[None, :],
            rows[:, None] < m,
            0.0,
        )
        w = tl.load(weight + cols[:, None] * inner + (k * block_inner + inner_offsets)[None, :])
        accumulator = tl.dot(a, tl.trans(w), accumulator)
    if has_bias:
        accumulator += tl.load(bias + cols)[None, :].to(tl.float32)
    tl.store(
        output + rows[:, None] * width + cols[None, :],
        accumulator.to(tl.bfloat16),
        rows[:, None] < m,
    )


def project(x, weight, bias, *, bm=32, bn=128, bk=64):
    m, k = x.shape
    n = weight.shape[0]
    if (
        (k, n) != (5120, 10240)
        or weight.shape[1] != k
        or (
            x.dtype != torch.bfloat16
            or weight.dtype != x.dtype
            or weight.device != x.device
            or not (x.is_cuda and x.is_contiguous() and weight.is_contiguous())
        )
    ):
        raise ValueError("unexpected DFlash2 context projection layout")
    if bias is not None and (
        bias.shape != (n,)
        or bias.dtype != x.dtype
        or not bias.is_contiguous()
        or bias.device != x.device
    ):
        raise ValueError("unexpected DFlash2 context bias")
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if not m:
        return out
    _project[(triton.cdiv(m, bm), n // bn)](
        x,
        weight,
        bias if bias is not None else x,
        out,
        m,
        k,
        n,
        bm,
        bn,
        bk,
        bias is not None,
        num_warps=4,
        num_stages=2,
    )
    return out


DRAFT_PROJECTION_SOURCE_SHA256 = "e3bcee30722064e6fcf86b389f0a2c40a1a228ec7ea9efc0a7a31ef055b07577"


def install(worker):
    """Bind only the fused KV projection in this pinned DFlash2 model instance."""
    import hashlib
    import inspect
    from types import FunctionType, MethodType, SimpleNamespace

    runner = worker.model_runner
    model = runner.speculator.model.model
    original = model._project_context_kv
    fn = original.__func__
    source_hash = hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
    if (
        source_hash != DRAFT_PROJECTION_SOURCE_SHA256
        or fn.__globals__["F"] is not torch.nn.functional
    ):
        raise ValueError("unrecognized DFlash context projection producer")
    if hasattr(model, "_fixed_context_projection"):
        raise ValueError("DFlash context projection is already bound")
    device = next(model.parameters()).device
    arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "").split(":")[0]
    if arch != "gfx1201":
        raise ValueError("the fixed BF16 context projection is qualified for gfx1201")
    stock = fn.__globals__["F"].linear

    def linear(x, weight, bias=None):
        if (
            x.ndim == 2
            and weight.ndim == 2
            and x.shape[1] == 5120
            and tuple(weight.shape) == (10240, 5120)
            and x.dtype == torch.bfloat16
            and weight.dtype == x.dtype
            and x.device == weight.device
            and x.is_cuda
            and x.is_contiguous()
            and weight.is_contiguous()
            and (
                bias is None
                or (
                    bias.shape == (10240,)
                    and bias.dtype == x.dtype
                    and bias.device == x.device
                    and bias.is_contiguous()
                )
            )
        ):
            return project(x, weight, bias, bm=64, bn=128, bk=64)
        return stock(x, weight, bias)

    # This exact upstream bytecode uses F only for its single fused linear call.
    # Keep the upstream lazy weight binding, normalization and KV layout identical.
    namespace = dict(fn.__globals__, F=SimpleNamespace(linear=linear))
    bound = FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__, fn.__closure__)
    bound.__kwdefaults__ = fn.__kwdefaults__
    model._project_context_kv = MethodType(bound, model)
    model._fixed_context_projection = original
    return {"producer_sha256": source_hash, "arch": arch, "tile": [64, 128, 64]}


def remove(worker):
    model = worker.model_runner.speculator.model.model
    model._project_context_kv = model._fixed_context_projection
    del model._fixed_context_projection
    return {"restored": True}

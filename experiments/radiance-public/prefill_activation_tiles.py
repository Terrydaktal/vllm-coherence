"""Experimental byte-only activation layout for the existing native tiled GEMM.

This module changes storage, never FP8 values or scales. Installation requires
source-bound native evidence. The consumer explicitly uses launch_at; ordinary
tensors are never relabelled using a data-pointer registry.
"""

import triton
import triton.language as tl


@triton.jit
def _pack(source, target, m: tl.constexpr, k: tl.constexpr, block: tl.constexpr):
    i = tl.program_id(0) * block + tl.arange(0, block)
    size = triton.cdiv(m, 16) * 16 * k
    row = (i // (16 * k)) * 16 + (i // 8) % 16
    col = ((i // 256) % (k // 16)) * 16 + ((i // 128) % 2) * 8 + i % 8
    value = tl.load(source + row * k + col, (i < size) & (row < m), 0)
    tl.store(target + i, value, i < size)


def pack(q, storage=None):
    import torch

    if q.ndim != 2 or not q.is_contiguous() or q.element_size() != 1 or q.shape[1] % 128:
        raise ValueError("activation tiling needs contiguous bytes and K divisible by 128")
    m, k = q.shape
    size = triton.cdiv(m, 16) * 16 * k
    if storage is None:
        storage = torch.empty(size, dtype=torch.uint8, device=q.device)
    if storage.dtype != torch.uint8 or storage.numel() != size or not storage.is_contiguous():
        raise ValueError("invalid tiled activation storage")
    if storage.device != q.device or q.device.type != "cuda":
        raise ValueError("activation tiling requires the same GPU")
    _pack[(triton.cdiv(size, 1024),)](q.view(torch.uint8), storage, m, k, 1024)
    return storage


def install_consumer(kernel, shapes, rows):
    """Install once per worker, before compilation, using custom-op dispatch.

    The original implementation remains the fallback, including every decode
    shape. The worker's process lifetime owns this registration and captured
    graphs; reinstallation while graphs are alive is deliberately rejected.
    """
    import torch

    op = kernel.mxfp4_linear_pq
    if "cuda" in op._backend_fns:
        raise ValueError("activation consumer already has a CUDA override")
    original = op._init_fn
    calls = {"tiled": 0, "fallback": 0, "rows": list(rows), "shapes": sorted(shapes)}

    @op.register_kernel("cuda")
    def consumer(q, scale, weight, weight_scale, weight_ref):
        m, n, k = q.shape[0], weight.shape[0], weight_scale.shape[0] * 32
        if (
            m not in rows
            or (n, k) not in shapes
            or q.shape != (m, k)
            or q.dtype != torch.float8_e4m3fn
            or not q.is_contiguous()
            or scale.numel() != m
            or not scale.is_contiguous()
        ):
            calls["fallback"] += 1
            return original(q, scale, weight, weight_scale, weight_ref)
        calls["tiled"] += 1
        tiled = pack(q)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=q.device)
        kernel._ext.launch_at(
            tiled.data_ptr(),
            weight.data_ptr(),
            weight_scale.data_ptr(),
            weight_ref.data_ptr(),
            scale.data_ptr(),
            out.data_ptr(),
            m,
            n,
            k,
            torch.cuda.current_stream().cuda_stream,
        )
        return out

    return calls

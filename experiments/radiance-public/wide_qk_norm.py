"""Use the authenticated M1 Gemma reduction for wide Qwen attention Q/K rows."""

import hashlib
import inspect
from pathlib import Path

DISPATCH_SHA256 = "6a84b3fd0fdc37adfb1ef598f61b77f2fac23bad884d91de1e5ec5e9943366d4"
KERNEL_SHA256 = "e987085f5488e62f4946135f7b13246a40e05c4a0cf12c17c5c588d0c0024ce2"
BINARY_SHA256 = "b9dc169e0f75f9f21e0e6d6e361e8dc18687781863f5ee4105f88cdfcf6ac5c6"


def matches(shape, key):
    if len(shape) != 3 or shape[0] <= 8 or shape[2] != 256:
        return False
    return (shape[1] == 4 and key.endswith(".self_attn.k_norm")) or (
        shape[1] == 24 and key.endswith(".self_attn.q_norm")
    )


def install(worker):
    import torch

    op = torch._library.custom_ops.OPDEFS["qwen_d7_qualified::gemma"]
    fn = op._init_fn
    if hashlib.sha256(Path(inspect.getfile(fn)).read_bytes()).hexdigest() != DISPATCH_SHA256:
        raise ValueError("unsupported compiled Gemma norm dispatcher")
    cells = inspect.getclosurevars(fn).nonlocals
    native = cells["candidate"]
    if (
        native.manifest["source_sha256"] != KERNEL_SHA256
        or native.manifest["binary_sha256"] != BINARY_SHA256
    ):
        raise ValueError("unsupported numerical reference for Q/K normalization")
    if hasattr(worker, "_wide_qk_norm"):
        raise ValueError("wide Q/K norm already installed")
    dispatch = "cuda" if "cuda" in op._backend_fns else None
    previous = op._backend_fns[dispatch]

    def forward(x, weight, eps, key):
        if x.device.type == "cuda" and matches(x.shape, key):
            return native(x, None, weight, eps)
        return previous(x, weight, eps, key)

    op._backend_fns[dispatch] = forward
    worker._wide_qk_norm = (op, dispatch, previous)
    return {"source_sha256": KERNEL_SHA256, "binary_sha256": BINARY_SHA256}


def remove(worker):
    op, dispatch, previous = worker._wide_qk_norm
    op._backend_fns[dispatch] = previous
    del worker._wide_qk_norm

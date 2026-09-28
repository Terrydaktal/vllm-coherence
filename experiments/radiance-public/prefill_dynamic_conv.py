"""Reuse the native convolution kernel across changing prefill lengths.

Only integer launch dimensions/strides stop being compile-time constants. The
native arithmetic, state copies, masks and decode/speculation kernel are kept.
The native source identity is part of release admission, not inferred by shape.
"""

import hashlib
import inspect
import linecache
import sys

DIMENSIONS = ("seqlen", "stride_x_seq", "stride_x_dim", "stride_o_seq", "stride_o_dim")


def native_module():
    # RuntimeRepairs loads a source-bound stock copy and retains its function in
    # StockPrefillAdapter. Patching vLLM's ordinary import misses that executable.
    module = sys.modules.get("qwen_stock_runtime_convolution")
    if module is None or not hasattr(module, "_causal_conv1d_update_kernel"):
        raise ValueError("qualified stock convolution module is not installed")
    return module


def clone(native, expected_sha256=None):
    source = inspect.getsource(native.fn)
    sha = hashlib.sha256(source.encode()).hexdigest()
    if expected_sha256 is not None and sha != expected_sha256:
        raise ValueError("native convolution source differs from qualification")
    decorator = '@triton.jit(do_not_specialize_on_alignment=["num_cache_lines"])'
    if source.count(decorator) != 1:
        raise ValueError("native convolution decorator changed")
    changed = source.replace(
        decorator,
        f"@triton.jit(do_not_specialize={list(DIMENSIONS)!r}, "
        'do_not_specialize_on_alignment=["num_cache_lines"])',
    ).replace(
        "def _causal_conv1d_update_kernel(", "def _coherence_prefill_dynamic_conv("
    )
    for name in DIMENSIONS:
        old = f"    {name}: tl.constexpr,"
        if changed.count(old) != 1:
            raise ValueError("native convolution signature changed")
        changed = changed.replace(old, f"    {name},")
    filename = (
        f"<coherence-prefill-conv-{hashlib.sha256(changed.encode()).hexdigest()}>"
    )
    linecache.cache[filename] = (len(changed), None, changed.splitlines(True), filename)
    namespace = dict(native.fn.__globals__)
    # The serving installer pins this native source; no prompt or file input is executed.
    exec(compile(changed, filename, "exec"), namespace)  # noqa: S102
    return namespace["_coherence_prefill_dynamic_conv"], sha


class PrefillDispatch:
    def __init__(self, native, candidate):
        self.native, self.candidate = native, candidate
        self.calls = {"prefill": 0, "fallback": 0}

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            rows = kwargs.get("seqlen", args[12] if len(args) > 12 else 0)
            use = (
                isinstance(rows, int)
                and 8 < rows <= 4096
                and rows not in (1648, 2048)
                and kwargs.get("IS_SPEC_DECODING") is False
                and kwargs.get("KERNEL_WIDTH") == 4
            )
            self.calls["prefill" if use else "fallback"] += 1
            return (self.candidate if use else self.native)[grid](*args, **kwargs)

        return launch


def install(hooks, expected_sha256):
    native = native_module()
    candidate, sha = clone(native._causal_conv1d_update_kernel, expected_sha256)
    dispatch = PrefillDispatch(native._causal_conv1d_update_kernel, candidate)
    hooks.replace(native, "_causal_conv1d_update_kernel", dispatch)
    return {"native_source_sha256": sha, "calls": dispatch.calls}

"""Use the qualified one-row FP8 producer for wide target-prefill batches."""

from pathlib import Path

OLD_SOURCE = "12123cc8c1e5b837551782ba4affb8238f65969df6531bbceb7139d79a728ec0"
WIDE_SOURCE = "8c044b1c1a07e9ab3bcd137197fa2758a061a71d8166108456c5c10204603089"


def operators(build):
    import torch

    op = torch._library.custom_ops.OPDEFS["qwen_stock_fp8::norm"]
    fn = op._init_fn
    cells = {
        name: cell.cell_contents
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__, strict=True)
    }
    narrow = cells["native"]
    if (
        not cells["entry"].get("prefill_enabled")
        or narrow.manifest["source_sha256"] != OLD_SOURCE
    ):
        raise ValueError(
            "expected the actual qualified native row-invariant normalization"
        )

    class WideNorm(type(narrow)):
        @staticmethod
        def admit(x, width, *, max_rows=8):
            type(narrow).admit(
                x, width, max_rows=4096 if max_rows == 2048 else max_rows
            )

    wide = WideNorm(Path(build))
    if (
        wide.manifest.get("upstream_source_sha256") != OLD_SOURCE
        or wide.manifest["source_sha256"] != WIDE_SOURCE
        or wide.manifest.get("max_norm_rows") != 4096
    ):
        raise ValueError("unexpected wide normalization build")
    return op, narrow, wide


def install(worker, build):
    op, narrow, wide = operators(build)
    if getattr(worker, "_wide_native_norm", None):
        raise ValueError("wide native normalization already installed")
    key = "cuda" if "cuda" in op._backend_fns else None
    prior = op._backend_fns[key]
    counts = {}

    def forward(x, residual, weight, eps, key):
        rows = x.shape[0]
        if 2048 < rows <= 4096 and x.device.type == "cuda":
            counts[rows] = counts.get(rows, 0) + 1
            return wide.norm(x, residual, weight, eps)
        return prior(x, residual, weight, eps, key)

    # The already-registered torch dispatcher resolves this callable by key on
    # every launch. Replacing it keeps the original dispatch-key registration.
    op._backend_fns[key] = forward
    worker._wide_native_norm = (op, key, prior, wide, counts)
    return {
        "binary_sha256": wide.manifest["binary_sha256"],
        "reference_sha256": narrow.manifest["binary_sha256"],
        "wide_rows": counts,
    }


def remove(worker):
    op, key, prior, _, counts = worker._wide_native_norm
    op._backend_fns[key] = prior
    del worker._wide_native_norm
    return {
        "wide_rows": counts,
        "restored_original_callable": op._backend_fns[key] is prior,
    }

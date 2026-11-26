"""Compare mapped input embedding bits, synchronizing once on success."""

import torch


@torch.no_grad()
def check_mapped_table(weight, view, *, rows=4096):
    starts = range(0, weight.shape[0], rows)
    # Scalar comparisons stay on device until all chunks have completed. Use
    # exact signed 16-bit representations even for NaNs and negative zero.
    flags = torch.stack(
        [
            torch.eq(
                view[i : i + rows].view(torch.int16),
                weight[i : i + rows].view(torch.int16),
            ).all()
            for i in starts
        ]
    )
    if not flags.all().item():
        index = flags.logical_not().nonzero()[0].item()
        raise RuntimeError(
            f"pinned host embedding readback differs at row {index * rows} or later"
        )


def install(module):
    from cache_buffer_views import substitute

    method = module._replace_storage
    if hasattr(method, "__wrapped__"):
        method = method.__wrapped__
    import inspect

    source = inspect.getsource(method)
    start = source.index("    for row in range(")
    end = source.index("    # Preserve the Parameter", start)
    module._coherence_compare_table = check_mapped_table
    module._replace_storage = substitute(
        method,
        source[start:end],
        "    _coherence_compare_table(weight, view, rows=CHECK_ROWS)\n\n",
    )

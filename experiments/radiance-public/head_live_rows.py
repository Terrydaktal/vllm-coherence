"""Run Global-512 selection and BF16 rescoring only on the live rows."""

import inspect


def live_rows_variant(module):
    # Keep the installed coarse kernel, precision gate and telemetry. Both
    # shipped implementations share the selection and rerank expressions.
    if module.GLOBAL_TOPK != 512:
        raise ValueError("live-row reranking requires the Global-512 target head")
    original = module._apply_head_global
    source = inspect.getsource(original)
    replacements = (
        ("    ids = logits.topk(", "    logits = logits[:rows]\n    ids = logits.topk("),
        ("torch.empty(padded, GLOBAL_TOPK,", "torch.empty(rows, GLOBAL_TOPK,"),
        ("_rerank_exact[(padded, GLOBAL_TOPK)]", "_rerank_exact[(rows, GLOBAL_TOPK)]"),
        ("    return logits[:rows]", "    return logits"),
    )
    for expected, replacement in replacements:
        if source.count(expected) != 1:
            raise ValueError(f"unexpected Global-512 source at {expected}")
        source = source.replace(expected, replacement, 1)
    namespace = {}
    exec(compile(source, inspect.getfile(original), "exec"), original.__globals__, namespace)
    return namespace[original.__name__]


def install():
    import radiance_verifyhead as module

    if module._state["armed"]:
        raise ValueError("install live-row reranking before the target head is armed")
    module._apply_head_global = live_rows_variant(module)

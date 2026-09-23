"""Defer draft-head packing until DFlash has shared the real target weights.

Backport of GGZ14/vllm-mxfp4 commit 1d76c82699c24ffe537e543dc4da575500d4e639.
The pinned image must match exactly; no quantizer or projection kernel changes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

SOURCE = "radiance_drafthead.py"
PREIMAGE = "0de6f82cbf22d7ef6ac254748df60c72b329834fbc839cdb7d8f3cbf337bba92"
POSTIMAGE = "33ac79c763df528a28674c7f9d2bf0b8e1a3645757e6182210b09137be69adb3"
UPSTREAM = "https://github.com/GGZ14/vllm-mxfp4/commit/1d76c82699c24ffe537e543dc4da575500d4e639"
OLD = """    # A drafter whose checkpoint carries no lm_head (DFlash2) gets the target's tensor shared in
    # AFTER load_weights returns, so at this point the parameter is still allocated-but-empty.
    # Quantising that yields an all-zero head, and the failure is silent and total: the serve comes
    # up, text stays coherent because the TARGET is fine, and only acceptance collapses to ~1.0 --
    # which reads as a plausible accuracy verdict on the quantisation. Defer instead.
    if float(w.data.abs().max()) == 0.0:
        lp._apply_head = types.MethodType(_apply_head_lazy, lp)
        return "lm_head empty at load_weights (shared in later); quantising on first use"
    return _quantize_head_now(lp, lm_head)
"""
NEW = """    # DFlash shares the target's head AFTER load_weights. torch.empty can contain nonzero
    # allocator leftovers, so checking for zeros cannot tell whether weights were loaded.
    # Always defer: the first invocation supplies the actual shared head. Packing and
    # the existing empty-head fallback stay unchanged; later calls use the fast path directly.
    # Backport: GGZ14/vllm-mxfp4 commit 1d76c82699c24ffe537e543dc4da575500d4e639.
    lp._apply_head = types.MethodType(_apply_head_lazy, lp)
    return "deferred to first use (the weight a drafter scores against is shared in later)"
"""


def patched_source(source: str) -> str:
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest == POSTIMAGE:
        return source
    if digest != PREIMAGE:
        raise ValueError("Draft-head source differs from the pinned preimage")
    if source.count(OLD) != 1:
        raise ValueError("Draft-head initialization anchor is ambiguous")
    result = source.replace(OLD, NEW)
    if hashlib.sha256(result.encode()).hexdigest() != POSTIMAGE:
        raise ValueError("Draft-head repair differs from the expected postimage")
    return result


def install(package: Path) -> dict:
    path = package / SOURCE
    before = path.read_text()
    after = patched_source(before)
    compile(after, str(path), "exec")
    if after != before:
        path.write_text(after)
    return {
        "source": SOURCE,
        "before": hashlib.sha256(before.encode()).hexdigest(),
        "after": hashlib.sha256(after.encode()).hexdigest(),
        "upstream_commit": UPSTREAM,
    }

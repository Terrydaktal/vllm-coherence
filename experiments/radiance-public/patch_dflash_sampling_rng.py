"""Backport the DFlash2 noise separation from upstream vLLM PR #54282.

Qualified against the installed native GPU sampling kernels and opaque long
requests. This repairs sampling probabilities; it is not a repetition filter.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

SOURCE = "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py"
PREIMAGE = "e6b65df587092ba10f7a9e802fa963e19db0a7cc60b70b5eb97352eb13d18628"
POSTIMAGE = "54e5c70b389a9e3d067fdf21fc28f615ee083fc0374d266bfd486eadbc36fe35"
DRAFT_NOISE_SALT = 1 << 30
OLD = "        position = tl.load(sample_pos_ptr + flat) - 1\n"
NEW = """        # Upstream #54282: keep the proposal's Philox offset separate from
        # the target's residual draw. Sharing noise conditions the replacement
        # on the rejected proposal and biases the target distribution.
        # This DFlash2 sampling-position buffer is not clamped to model length.
        position = tl.load(sample_pos_ptr + flat) - 1 + (1 << 30)
"""


def patched_source(source: str) -> str:
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest == POSTIMAGE:
        return source
    if digest != PREIMAGE:
        raise ValueError("DFlash2 sampler source differs from the qualified preimage")
    if source.count(OLD) != 1:
        raise ValueError("DFlash2 sampler position anchor is ambiguous")
    return source.replace(OLD, NEW).replace(
        "# Candidate ids key the noise, matching the target's own sampling.",
        "# Candidate ids key the proposal's independent sampling noise.",
    )


def install(package: Path) -> dict:
    path = package / SOURCE
    before = path.read_text()
    after = patched_source(before)
    if after != before:
        path.write_text(after)
    compile(after, str(path), "exec")
    return {"source": SOURCE, "before": hashlib.sha256(before.encode()).hexdigest(),
            "after": hashlib.sha256(after.encode()).hexdigest(), "draft_noise_salt": DRAFT_NOISE_SALT,
            "upstream_pr": 54282}

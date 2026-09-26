"""Initialize GDN state for a fresh, single-token prompt in pinned Radiance.

The length-only metadata split calls it a decode, whose convolution/recurrent
path assumes initialized history. Attention-block zeroing excludes Mamba state.
Use the existing prefill path, which respects has_initial_state=False, when
the whole batch is at its first position. No decode arithmetic is changed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

SOURCE = "vllm/v1/attention/backends/gdn_attn.py"
PREIMAGE = "212e3c96c46d18dafa8db54256f449d22495c68d1264ae4401849d1dbb62785c"
POSTIMAGE = "f1007fa33e1fb7d182cb588f11f0f7314cc076ebd87961b5d523dd54c4e1f6cc"
OLD = """            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(m, decode_threshold=1)
            )
"""
NEW = (
    OLD
    + """            # A fresh one-token prompt has no convolution/recurrent history.
            # The ordinary decode kernels assume that history already exists;
            # only prefill honors has_initial_state=False. max_seq_len is a CPU
            # upper bound, so ==1 guarantees every live row starts at position 0.
            if num_decodes and m.max_seq_len == 1:
                num_prefills += num_decodes
                num_prefill_tokens += num_decode_tokens
                num_decodes = 0
                num_decode_tokens = 0
"""
)


def patched_source(source: str) -> str:
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest == POSTIMAGE:
        return source
    if digest != PREIMAGE or source.count(OLD) != 1:
        raise ValueError("GDN metadata differs from the pinned preimage")
    result = source.replace(OLD, NEW)
    if hashlib.sha256(result.encode()).hexdigest() != POSTIMAGE:
        raise ValueError("GDN initial-prefill repair has an unexpected postimage")
    return result


def install(package: Path) -> dict:
    path = package / SOURCE
    before = path.read_text()
    after = patched_source(before)
    compile(after, str(path), "exec")
    if before != after:
        path.write_text(after)
    return {
        "source": SOURCE,
        "before": hashlib.sha256(before.encode()).hexdigest(),
        "after": hashlib.sha256(after.encode()).hexdigest(),
    }

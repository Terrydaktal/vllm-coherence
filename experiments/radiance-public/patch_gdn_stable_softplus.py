"""Source-pinned stable softplus for ordinary GDN decode and prefill gates."""

import hashlib
from pathlib import Path

BASE = "vllm/third_party/flash_linear_attention/ops/"
PREIMAGES = {
    "fused_recurrent.py": "00a3b971b0dbb6ed26e246970a0e1a21a9a174030974685bdd3f0c8ab5fb4bfe",
    "fused_gdn_prefill_post_conv.py": "590717b2107abc90d72548de06b5d315229eb2775c94614f3f545df11cc2b5df",
    "fused_sigmoid_gating.py": "000ab8996af9788fdb8843a6a3b91833e7a14c8acc0e1ea073a536330f64cb6f",
}
REPLACEMENTS = {
    "fused_recurrent.py": [
        ("tl.log(1.0 + tl.exp(x))", "tl.extra.libdevice.log1p(tl.exp(x))")
    ],
    "fused_gdn_prefill_post_conv.py": [
        ("tl.log(1.0 + tl.exp(-x))", "tl.extra.libdevice.log1p(tl.exp(-x))"),
        ("tl.log(1.0 + tl.exp(x))", "tl.extra.libdevice.log1p(tl.exp(x))"),
    ],
    "fused_sigmoid_gating.py": [
        ("tl.log(1 + tl.exp(beta * x))", "tl.extra.libdevice.log1p(tl.exp(beta * x))")
    ],
}


def patched_source(name, source):
    if name not in PREIMAGES:
        raise ValueError("unsupported GDN gate source")
    original = source
    # Idempotence still validates the complete source, not only a marker.
    for old, new in REPLACEMENTS[name]:
        original = original.replace(new, old)
    if hashlib.sha256(original.encode()).hexdigest() != PREIMAGES[name]:
        raise ValueError("GDN gate source differs from its pinned preimage")
    result = original
    for old, new in REPLACEMENTS[name]:
        if result.count(old) != 1:
            raise ValueError("ambiguous GDN gate replacement")
        result = result.replace(old, new)
    compile(result, name, "exec")
    return result


def install(package):
    # Validate every preimage before changing any file.
    changes = []
    for name in PREIMAGES:
        path = Path(package) / BASE / name
        before = path.read_text()
        after = patched_source(name, before)
        changes.append((path, before, after))
    for path, before, after in changes:
        if before != after:
            path.write_text(after)
    return {
        str(path.relative_to(package)): hashlib.sha256(after.encode()).hexdigest()
        for path, _, after in changes
    }

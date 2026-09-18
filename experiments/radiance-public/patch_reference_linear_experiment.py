"""Make the image's diagnostic linear reference safe for WPERM and limited VRAM.

This is an experimental oracle, not the production sampler or a loop guard.
It retains packed checkpoint weights but returns BF16 linear algebra on BF16
activations, bypassing the information loss of their extra FP8 conversion.
"""

import hashlib
from pathlib import Path

SOURCE = "radiance_mxfp4.py"
PREIMAGE = "7fd10b2d5b6a6c3853583eb71ca10dfcaf6784694c7941bd1e28968addcb88f2"
OLD = '''        codes = torch.stack([weight & 0x0F, (weight >> 4) & 0x0F], -1).reshape(N, K)
        w = _E2M1[codes.long()] * torch.pow(
            2.0, weight_scale.float() - 127.0).T.repeat_interleave(32, dim=1)
        return (x.float() @ w.T.float()).to(torch.bfloat16)
'''
NEW = '''        # Diagnostic only: checkpoint-order weights, bounded dequantization,
        # and original BF16 activations. Do not reinterpret fragment-order data.
        reference_weight = unpermute_w(weight, N, K) if WPERM else weight
        reference_out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
        for first in range(0, N, 2048):
            last = min(first + 2048, N)
            packed = reference_weight[first:last]
            codes = torch.stack([packed & 0x0F, (packed >> 4) & 0x0F], -1).reshape(last - first, K)
            scales = torch.pow(2.0, weight_scale[:, first:last].float() - 127.0).T.repeat_interleave(32, dim=1)
            weights_bf16 = (_E2M1[codes.long()] * scales).to(torch.bfloat16)
            reference_out[:, first:last] = torch.nn.functional.linear(x, weights_bf16)
        return reference_out
'''


def patched_source(source):
    assert hashlib.sha256(source.encode()).hexdigest() == PREIMAGE
    assert source.count(OLD) == 1
    updated = source.replace(OLD, NEW)
    compile(updated, SOURCE, "exec")
    return updated


def install(package):
    path = Path(package) / SOURCE
    updated = patched_source(path.read_text())
    path.write_text(updated)
    return {"source_sha256": hashlib.sha256(updated.encode()).hexdigest(),
            "activation_reference": "BF16", "dequantization_rows": 2048}

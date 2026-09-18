"""Efficient weight expansion for the diagnostic BF16 linear reference.

The reference still uses the original packed checkpoint, BF16 activations and
PyTorch BF16 matrix multiplication. A fused pointwise kernel expands the weights
without the slow reference's many temporary int64/FP32 tensors and row-wise GEMMs.
This is an experiment only, never part of the production overlay.
"""

import hashlib
from pathlib import Path

from patch_reference_linear_experiment import OLD, PREIMAGE, SOURCE

ANCHOR = '@torch.library.custom_op("radiance::mxfp4_linear", mutates_args=())\n'
HELPER = '''from vllm.triton_utils import triton as _reference_triton, tl as _reference_tl


@_reference_triton.jit
def _reference_expand_bf16(packed, scales, levels, output,
                           N: _reference_tl.constexpr, K: _reference_tl.constexpr,
                           BLOCK: _reference_tl.constexpr):
    offsets = _reference_tl.program_id(0) * BLOCK + _reference_tl.arange(0, BLOCK)
    mask = offsets < N * K
    row, col = offsets // K, offsets % K
    byte = _reference_tl.load(packed + row * (K // 2) + col // 2, mask, other=0)
    code = (byte >> ((col % 2) * 4)) & 15
    exponent = _reference_tl.load(scales + (col // 32) * N + row, mask, other=0)
    level = _reference_tl.load(levels + code)
    value = level * _reference_tl.exp2(exponent.to(_reference_tl.float32) - 127.0)
    _reference_tl.store(output + offsets, value, mask)


'''
NEW = '''        # Diagnostic only: return BF16 linear math on the original activations.
        reference_weight = unpermute_w(weight, N, K) if WPERM else weight
        weights_bf16 = torch.empty((N, K), device=x.device, dtype=torch.bfloat16)
        _reference_expand_bf16[(_reference_triton.cdiv(N * K, 2048),)](
            reference_weight, weight_scale, _E2M1, weights_bf16, N, K, BLOCK=2048)
        return torch.nn.functional.linear(x, weights_bf16)
'''


def patched_source(source):
    if hashlib.sha256(source.encode()).hexdigest() != PREIMAGE:
        raise ValueError("BF16 reference source differs from the qualified image")
    if source.count(OLD) != 1 or source.count(ANCHOR) != 1:
        raise ValueError("BF16 reference anchors are ambiguous")
    updated = source.replace(OLD, NEW).replace(ANCHOR, HELPER + ANCHOR)
    compile(updated, SOURCE, "exec")
    return updated


def install(package):
    path = Path(package) / SOURCE
    updated = patched_source(path.read_text())
    path.write_text(updated)
    return {"source_sha256": hashlib.sha256(updated.encode()).hexdigest(),
            "activation_reference": "BF16", "fused_weight_expansion": True}

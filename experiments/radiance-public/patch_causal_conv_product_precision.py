"""Preserve FP32 products in the reviewed stock convolution update kernel.

The BF16 operands otherwise produce a BF16 product before the FP32 addition.
On the qualified ROCm build that product truncates toward zero. This changes
the M1 recurrence input relative to Radiance's FP32-product speculative path.

This patch has finite operator evidence; it is not a full-model equivalence
certificate. Installation is explicit and rejects unreviewed source versions.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

SOURCE = "vllm/model_executor/layers/mamba/ops/causal_conv1d.py"
PREIMAGE = "85d3715c80905579d9b031e4d4b43c3e041b152064cd3e2388b472a2f80c1a32"
POSTIMAGE = "da2d1183c29f68497d0166960c3ca7bedd143b90eead1de3d98e90af0c0f8a4a"
KERNEL = "_causal_conv1d_update_kernel"
OLD = "acc += matrix_x * matrix_w  # [BLOCK_N]"
NEW = "acc += matrix_x.to(tl.float32) * matrix_w.to(tl.float32)  # [BLOCK_N]"


def patched_source(source: str) -> str:
    before = hashlib.sha256(source.encode()).hexdigest()
    if before == POSTIMAGE:
        return source
    if before != PREIMAGE:
        raise ValueError("causal convolution source differs from the reviewed preimage")
    nodes = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == KERNEL
    ]
    if len(nodes) != 1:
        raise ValueError("causal convolution update kernel is ambiguous")
    node = nodes[0]
    lines = source.splitlines(keepends=True)
    body = "".join(lines[node.lineno - 1 : node.end_lineno])
    if body.count(OLD) != 1 or NEW in body:
        raise ValueError("causal convolution product anchor is ambiguous")
    result = (
        "".join(lines[: node.lineno - 1])
        + body.replace(OLD, NEW)
        + "".join(lines[node.end_lineno :])
    )
    if hashlib.sha256(result.encode()).hexdigest() != POSTIMAGE:
        raise ValueError("causal convolution patch differs from the reviewed postimage")
    compile(result, SOURCE, "exec")
    return result


def install(package: Path) -> dict:
    path = package / SOURCE
    before = path.read_text()
    after = patched_source(before)
    if after != before:
        path.write_text(after)
    return {
        "source": SOURCE,
        "before": hashlib.sha256(before.encode()).hexdigest(),
        "after": hashlib.sha256(after.encode()).hexdigest(),
        "kernel": KERNEL,
        "arithmetic": "convert operands to FP32 before multiplication",
    }

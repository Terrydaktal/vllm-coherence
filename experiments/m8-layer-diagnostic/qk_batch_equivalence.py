"""Check fused Q/K normalization and RoPE for M8 versus serial M1 calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _tensor_sha256(value: Any) -> str:
    import torch

    payload = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _comparison(actual: Any, expected: Any) -> dict[str, Any]:
    import torch

    actual = actual.detach()
    expected = expected.detach()
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise RuntimeError("QK batch comparison tensor contract differs")
    unequal = actual != expected
    mismatch_count = int(torch.count_nonzero(unequal).cpu().item())
    first_mismatch = None
    max_abs_error = 0.0
    if mismatch_count:
        first_mismatch = int(torch.nonzero(unequal.reshape(-1), as_tuple=False)[0].cpu().item())
        max_abs_error = float(torch.max(torch.abs(actual.float() - expected.float())).cpu().item())
    return {
        "actual_sha256": _tensor_sha256(actual),
        "expected_sha256": _tensor_sha256(expected),
        "first_mismatch_flat_index": first_mismatch,
        "max_abs_error": max_abs_error,
        "mismatch_count": mismatch_count,
    }


def _tensor_layout(value: Any) -> dict[str, Any]:
    """Describe the physical tensor view without treating addresses as identity.

    Logical hashes deliberately make strided and contiguous tensors look the same.
    This record preserves the storage contract needed to diagnose aliasing, view,
    and workspace-lifetime defects in the fused Q/K path.
    """

    element_size = int(value.element_size())
    shape = [int(item) for item in value.shape]
    stride = [int(item) for item in value.stride()]
    if any(item < 0 for item in stride):
        raise RuntimeError("QK equivalence does not support negative-stride tensors")
    storage_offset = int(value.storage_offset())
    maximum_element = storage_offset
    for extent, step in zip(shape, stride, strict=True):
        if extent:
            maximum_element += (extent - 1) * step
    storage = value.untyped_storage()
    return {
        "contiguous": bool(value.is_contiguous()),
        "data_ptr": int(value.data_ptr()),
        "dtype": str(value.dtype),
        "element_size": element_size,
        "shape": shape,
        "storage_byte_interval": [
            storage_offset * element_size,
            (maximum_element + 1) * element_size,
        ],
        "storage_data_ptr": int(storage.data_ptr()),
        "storage_nbytes": int(storage.nbytes()),
        "storage_offset": storage_offset,
        "stride": stride,
    }


def _shares_storage(left: Any, right: Any) -> bool:
    return int(left.untyped_storage().data_ptr()) == int(right.untyped_storage().data_ptr())


def _write_create_only(path: Path, document: dict[str, Any]) -> None:
    path = path.expanduser().absolute()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or path.parent.is_symlink()
        or parent.st_uid != os.getuid()
        or stat.S_IMODE(parent.st_mode) & 0o077
    ):
        raise RuntimeError("QK evidence parent must be an owned private directory")
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def run(*, seeds: tuple[int, ...], start_position: int) -> dict[str, Any]:
    import torch
    from vllm.model_executor.layers.fused_qk_norm_rope import (
        fused_qk_rmsnorm_rope_gate,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("QK batch equivalence requires a CUDA/HIP device")
    rows = 8
    q_heads = 24
    kv_heads = 4
    head_dim = 256
    rotary_dim = 64
    q_size = q_heads * head_dim
    kv_size = kv_heads * head_dim
    eps = 1e-6
    cases: list[dict[str, Any]] = []
    for seed in seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        q_weight = (torch.randn(head_dim, generator=generator) * 0.05 + 1.0).cuda()
        k_weight = (torch.randn(head_dim, generator=generator) * 0.05 + 1.0).cuda()
        cos_sin = torch.randn(
            start_position + rows + 1,
            rotary_dim,
            dtype=torch.bfloat16,
            generator=generator,
        ).cuda()
        positions = torch.arange(
            start_position, start_position + rows, dtype=torch.int64, device="cuda"
        )

        for layout in ("separate_contiguous", "combined_qkv_split_views"):
            if layout == "separate_contiguous":
                q_gate = torch.randn(
                    rows, q_size * 2, dtype=torch.bfloat16, generator=generator
                ).cuda()
                key = torch.randn(rows, kv_size, dtype=torch.bfloat16, generator=generator).cuda()
                combined = None
            else:
                combined = torch.randn(
                    rows,
                    q_size * 2 + kv_size * 2,
                    dtype=torch.bfloat16,
                    generator=generator,
                ).cuda()
                q_gate, key, _value = combined.split([q_size * 2, kv_size, kv_size], dim=-1)

            batch = fused_qk_rmsnorm_rope_gate(
                q_gate,
                key,
                q_weight,
                k_weight,
                cos_sin,
                positions,
                eps,
                q_heads,
                kv_heads,
                head_dim,
                rotary_dim,
            )
            serial_rows = [
                fused_qk_rmsnorm_rope_gate(
                    q_gate[row : row + 1],
                    key[row : row + 1],
                    q_weight,
                    k_weight,
                    cos_sin,
                    positions[row : row + 1],
                    eps,
                    q_heads,
                    kv_heads,
                    head_dim,
                    rotary_dim,
                )
                for row in range(rows)
            ]
            serial = tuple(
                torch.cat([values[index] for values in serial_rows]) for index in range(3)
            )

            permutation = torch.tensor([7, 0, 5, 2, 6, 1, 4, 3], device="cuda")
            inverse = torch.argsort(permutation)
            permuted = fused_qk_rmsnorm_rope_gate(
                q_gate.index_select(0, permutation),
                key.index_select(0, permutation),
                q_weight,
                k_weight,
                cos_sin,
                positions.index_select(0, permutation),
                eps,
                q_heads,
                kv_heads,
                head_dim,
                rotary_dim,
            )
            unpermuted = tuple(value.index_select(0, inverse) for value in permuted)
            torch.cuda.synchronize()
            labels = ("q", "k", "gate")
            cases.append(
                {
                    "layout": layout,
                    "seed": seed,
                    "batch_vs_serial": {
                        label: _comparison(batch[index], serial[index])
                        for index, label in enumerate(labels)
                    },
                    "batch_vs_row_permuted": {
                        label: _comparison(batch[index], unpermuted[index])
                        for index, label in enumerate(labels)
                    },
                    "inputs": {
                        "combined_qkv": (
                            _tensor_layout(combined) if combined is not None else None
                        ),
                        "k_weight_sha256": _tensor_sha256(k_weight),
                        "key_layout": _tensor_layout(key),
                        "key_sha256": _tensor_sha256(key),
                        "positions_sha256": _tensor_sha256(positions),
                        "q_gate_layout": _tensor_layout(q_gate),
                        "q_gate_sha256": _tensor_sha256(q_gate),
                        "q_gate_shares_key_storage": _shares_storage(q_gate, key),
                        "q_weight_sha256": _tensor_sha256(q_weight),
                    },
                    "outputs": {
                        label: _tensor_layout(batch[index]) for index, label in enumerate(labels)
                    },
                }
            )
    passed = all(
        comparison["mismatch_count"] == 0
        for case in cases
        for group in ("batch_vs_serial", "batch_vs_row_permuted")
        for comparison in case[group].values()
    )
    source = Path(__file__).resolve()
    return {
        "schema": "urn:qwen-r9700:qk-batch-equivalence:v2",
        "claim": "bounded_component_qualification",
        "configuration": {
            "dtype": "bfloat16",
            "eps": eps,
            "head_dim": head_dim,
            "kv_heads": kv_heads,
            "q_heads": q_heads,
            "rotary_dim": rotary_dim,
            "rows": rows,
            "seeds": list(seeds),
            "start_position": start_position,
            "storage_layouts": [
                "separate_contiguous",
                "combined_qkv_split_views",
            ],
        },
        "environment": {
            "device": torch.cuda.get_device_name(0),
            "hip": torch.version.hip,
            "torch": torch.__version__,
        },
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cases": cases,
        "passed": passed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwen-qk-batch-equivalence")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--start-position", default=36_477, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        seeds = tuple(int(value) for value in args.seeds.split(","))
        if not seeds or len(seeds) > 64 or len(set(seeds)) != len(seeds):
            raise RuntimeError("seeds must contain 1..64 unique integers")
        if args.start_position < 0:
            raise RuntimeError("start position must be nonnegative")
        document = run(seeds=seeds, start_position=args.start_position)
        _write_create_only(args.output, document)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"qwen-qk-batch-equivalence: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0 if document["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Replay private actual GDN inputs against the shared FP64 recurrence on CPU."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import re
from itertools import pairwise
from pathlib import Path

from capture_gdn_operator import SOURCE_SHA256
from probe_gdn_numerics import sequential_reference


def replay(root, private, layer):
    import torch

    torch.set_grad_enabled(False)
    torch.set_num_threads(2)
    report = json.loads((root / f"gdn-operator-layer{layer}.json").read_text())
    if (
        report.get("schema") != "qwen-gdn-operator-capture-v1"
        or not report.get("complete")
        or report.get("source_sha256") != SOURCE_SHA256
        or not re.fullmatch(r"[a-z0-9_-]{1,140}", report.get("capsule_id", ""))
    ):
        raise ValueError("operator capsule identity is invalid")
    path = private / "layer-capsules" / (report["capsule_id"] + ".pt")
    if path.is_symlink() or path.stat().st_mode & 0o077:
        raise ValueError("operator capsule is not private")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != report["capsule_sha256"]:
        raise ValueError("operator capsule differs from its captured hash")
    capsule = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if (
        capsule["schema"] != "qwen-gdn-operator-capsule-v1"
        or capsule["source_sha256"] != SOURCE_SHA256
        or capsule["layer"] != layer
        or capsule["request"]["input_sha256"] != report["input_sha256"]
    ):
        raise ValueError("operator capsule metadata differs")
    values = {}
    for name, record in capsule["tensors"].items():
        tensor = record["tensor"]
        raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("captured tensor hash differs")
        if not torch.isfinite(tensor).all():
            raise ValueError("operator capsule has nonfinite input or output")
        values[name] = tensor
    q, k, v = (values[name][0] for name in ("q", "k", "v"))
    cumulative = values["g"][0].double()
    steps = torch.empty_like(cumulative)
    boundaries = values["cu_seqlens"].tolist()
    chunk = capsule["chunk"]
    if chunk != 64 or boundaries[0] != 0 or boundaries[-1] != len(q):
        raise ValueError("unsupported GDN sequence geometry")
    for begin, end in pairwise(boundaries):
        for start in range(begin, end, chunk):
            stop = min(start + chunk, end)
            steps[start] = cumulative[start]
            steps[start + 1 : stop] = cumulative[start + 1 : stop] - cumulative[start : stop - 1]
    output, final = sequential_reference(
        q, k, v, steps, values["beta"][0], values["initial_state"], boundaries, capsule["scale"]
    )
    comparisons = {}
    for name, actual, reference in (
        ("output", values["output"][0], output),
        ("final_state", values["final_state"], final),
    ):
        delta = actual.double() - reference
        comparisons[name] = {
            "max_absolute_error": delta.abs().max().item(),
            "relative_l2_error": (delta.norm() / reference.norm().clamp_min(1e-30)).item(),
            "rmse": delta.square().mean().sqrt().item(),
            "reference_finite": bool(torch.isfinite(reference).all()),
        }
    return {
        "schema": "qwen-gdn-actual-input-reference-v1",
        "complete": True,
        "capsule_sha256": report["capsule_sha256"],
        "input_sha256": report["input_sha256"],
        "layer": layer,
        "tokens": len(q),
        "heads": v.shape[-2],
        "query_heads": q.shape[-2],
        "comparisons": comparisons,
        "scope": "one actual native invocation, identical captured inputs and initial state",
        "reference": "shared independent sequential FP64 recurrence",
        "full_model_or_bit_exact_equivalence_proven": False,
    }


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--private", type=Path, default=Path("/private-fixtures"))
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()
    try:
        result = replay(args.root, args.private, args.layer)
        (args.root / f"gdn-operator-reference-layer{args.layer}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps(result))
    except Exception as error:
        print(json.dumps({"error_type": type(error).__name__}))
        raise SystemExit(1) from None

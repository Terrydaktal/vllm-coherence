"""Compare authenticated decoder capsules inside their private RAM directory.

Only hashes, dimensions and aggregate numerical errors leave the capsules.
Difference thresholds are descriptive, not model-correctness pass/fail limits.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import re
from pathlib import Path

from capture_radiance_layers import LEGACY_SHA256


def load_capsule(root, private):
    import torch

    report = json.loads((Path(root) / "layer-capture-result.json").read_text())
    if (report.get("schema") != "qwen-radiance-layer-capture-v1"
            or not report.get("complete") or report.get("legacy_helpers_sha256") != LEGACY_SHA256
            or not re.fullmatch(r"[a-z0-9_-]{1,100}", report.get("capture_id", ""))):
        raise ValueError("decoder capture report is not authenticated")
    capsule = Path(private) / "layer-capsules" / (report["capture_id"] + ".pt")
    if capsule.is_symlink() or capsule.stat().st_mode & 0o077:
        raise ValueError("decoder capsule is not private")
    data = capsule.read_bytes()
    if hashlib.sha256(data).hexdigest() != report["capsule_sha256"]:
        raise ValueError("decoder capsule differs from its recorded hash")
    value = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if (value.get("schema") != "qwen-radiance-layer-capsule-v1"
            or value["request"]["input_sha256"] != report["input_sha256"]
            or value["request"]["capture_id"] != report["capture_id"]
            or value["request"]["positions"] != report["positions"]
            or value["request"]["expected_layers"] != report["layers"]
            or value.get("legacy_helpers_sha256") != LEGACY_SHA256):
        raise ValueError("decoder capsule identity does not match its report")
    return report, value


def stage_order(key):
    layer, stage, position = key.split(":")
    sequence = ["input_hidden", "input_residual", "input_layernorm", "linear_attn",
                "self_attn", "post_attention_layernorm", "mlp", "output", "model_output"]
    return (int(layer) if int(layer) >= 0 else 64,
            sequence.index(stage.split(".")[0]), stage, int(position))


def compare(left_root, right_root, private):
    import torch

    private = Path(private)
    if private.is_symlink() or not private.is_dir() or private.stat().st_mode & 0o077:
        raise ValueError("decoder comparison requires protected RAM storage")
    left_report, left = load_capsule(left_root, private)
    right_report, right = load_capsule(right_root, private)
    for key in ("input_sha256", "positions", "layers", "legacy_helpers_sha256"):
        if left_report[key] != right_report[key]:
            raise ValueError("decoder captures do not describe the same input position")
    if left["records"].keys() != right["records"].keys():
        raise ValueError("decoder captures cover different stages")
    rows = []
    for key in sorted(left["records"], key=stage_order):
        a, b = left["records"][key], right["records"][key]
        x, y = a["tensor"], b["tensor"]
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError("decoder tensor representations differ")
        for row, tensor in ((a, x), (b, y)):
            raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != row["sha256"]:
                raise ValueError("decoder tensor differs from its recorded hash")
        finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        measures = {"max_absolute_error": None, "relative_l2_error": None,
                    "cosine_similarity": None}
        if finite:
            x, y = x.double().flatten(), y.double().flatten()
            delta = x - y
            x_norm, y_norm = x.norm().item(), y.norm().item()
            measures = {"max_absolute_error": delta.abs().max().item(),
                        "relative_l2_error": delta.norm().item() / max(x_norm, 1e-30),
                        "cosine_similarity": torch.dot(x, y).item() / max(x_norm * y_norm, 1e-30)}
        rows.append({"key": key, "shape": list(a["tensor"].shape),
                     "dtype": str(a["tensor"].dtype), "bit_equal": a["sha256"] == b["sha256"],
                     "both_finite": finite, **measures})
    first_above = {}
    for threshold in (0.01, 0.05, 0.1):
        first_above[str(threshold)] = next((row for row in rows
            if row["relative_l2_error"] is not None and row["relative_l2_error"] > threshold), None)
    return {"schema": "qwen-radiance-layer-comparison-v1",
            "left_capture": left_report["capture_id"], "right_capture": right_report["capture_id"],
            "input_sha256": left_report["input_sha256"], "tensors": len(rows),
            "bit_equal_tensors": sum(row["bit_equal"] for row in rows),
            "nonfinite_tensor_pairs": sum(not row["both_finite"] for row in rows),
            "first_bit_difference": next((row for row in rows if not row["bit_equal"]), None),
            "first_relative_l2_above": first_above, "rows": rows}


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--private", type=Path, default=Path("/private-fixtures"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(args.left, args.right, args.private)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "rows"}))
    except Exception as error:
        print(json.dumps({"error_type": type(error).__name__}))
        raise SystemExit(1) from None

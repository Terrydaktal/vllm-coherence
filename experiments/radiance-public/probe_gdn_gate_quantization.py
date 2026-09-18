"""Compare checkpoint GDN control projections with the preserved BF16 weights.

CPU only, public model tensors only. Does not read conversations or change either
checkpoint. Writes a small BF16 reference shard only if scale/identity checks pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def probe(quantized: Path, reference: Path, output: Path, official_verification: Path | None = None):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    output.mkdir(parents=True, exist_ok=True)
    qi = json.loads((quantized / "model.safetensors.index.json").read_text())["weight_map"]
    ri = json.loads((reference / "model.safetensors.index.json").read_text())["weight_map"]

    def tensor(root, index, name):
        with safe_open(root / index[name], framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    def digest(value):
        return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    # These tensors establish that the comparison uses the same input scaling,
    # convolution and gate parameterization, rather than mixing rescaled models.
    controls = [name for name in qi if name.startswith("model.language_model.layers.")
                and name.endswith(("input_layernorm.weight", ".linear_attn.A_log",
                                   ".linear_attn.dt_bias", ".linear_attn.conv1d.weight"))]
    control_rows = []
    for name in sorted(controls):
        actual, expected = tensor(quantized, qi, name), tensor(reference, ri, name)
        control_rows.append({"name": name, "identical": torch.equal(actual, expected),
                             "quantized_sha256": digest(actual), "reference_sha256": digest(expected)})
    names = sorted(name for name in qi if name.endswith(".weight")
                   and any(key in name for key in (".linear_attn.in_proj_a.", ".linear_attn.in_proj_b.")))
    if len(names) != 96:
        raise ValueError("expected both control projections in each of 48 GDN layers")
    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
    rows, restored = [], {}
    for name in names:
        packed = tensor(quantized, qi, name)
        scales = tensor(quantized, qi, name.removesuffix(".weight") + ".weight_scale")
        original = tensor(reference, ri, name)
        if packed.dtype != torch.uint8 or scales.dtype != torch.uint8 or original.dtype != torch.bfloat16:
            raise ValueError("unexpected gate tensor precision")
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(packed.shape[0], -1)
        scale = torch.exp2(scales.float() - 127).repeat_interleave(32, dim=1)
        dequantized = levels[codes.long()] * scale
        if dequantized.shape != original.shape or original.shape != (48, 5120):
            raise ValueError("unexpected gate shape")
        original_float = original.float()
        normalized = original_float / scale
        nearest = torch.full_like(normalized, float("inf"))
        for level in levels:
            torch.minimum(nearest, (normalized - level).abs(), out=nearest)
        actual_distance = (normalized - levels[codes.long()]).abs()
        nearest_fraction = float((actual_distance <= nearest + 1e-6).float().mean())
        rows.append({"name": name, "elements": original.numel(),
                     "relative_weight_error": float((dequantized - original_float).norm() / original_float.norm()),
                     "cosine_similarity": float(torch.nn.functional.cosine_similarity(
                         dequantized.flatten(), original_float.flatten(), dim=0)),
                     "matches_nearest_mxfp4_value_fraction": nearest_fraction,
                     "reference_sha256": digest(original),
                     "quantized_weight_sha256": digest(packed), "scale_sha256": digest(scales)})
        restored[name] = original.contiguous()
    report = {"quantized_model": quantized.name, "bf16_reference_model": reference.name,
              "all_controls_identical": all(row["identical"] for row in control_rows),
              "minimum_nearest_quantization_fraction": min(row["matches_nearest_mxfp4_value_fraction"] for row in rows),
              "maximum_weight_relative_error": max(row["relative_weight_error"] for row in rows),
              "gate_tensors": len(rows), "bf16_bytes": sum(t.numel() * t.element_size() for t in restored.values()),
              "controls": control_rows, "gates": rows,
              "source_identity_limit": "BF16 tensors come from the preserved official-base AutoRound build; this is not a direct hash comparison with the gated OrcaRouter BF16 repository"}
    # Existing BF16 gates are suitable for a controlled precision experiment only
    # after validating that they explain the current quantized coefficients.
    qualified = report["all_controls_identical"] and report["minimum_nearest_quantization_fraction"] > .999
    if official_verification is not None:
        proof = json.loads(official_verification.read_text())
        verified = {row["name"]: row for row in proof["gates"]}
        if (proof["model"] != "Qwen/Qwen3.8-27B"
                or proof["revision"] != "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
                or not proof["all_match"] or set(verified) != set(restored)):
            raise ValueError("official BF16 verification does not cover this reference")
        for name, original in restored.items():
            if verified[name]["sha256"] != digest(original) or not verified[name]["matches_preserved_reference"]:
                raise ValueError("official BF16 tensor identity check failed")
        # This establishes direct identity with the public original weights.
        # OrcaRouter documents that only residual-writing matrices were edited;
        # a/b gates are outside that edit. Keep that provenance inference explicit.
        report["official_reference_verification_sha256"] = hashlib.sha256(official_verification.read_bytes()).hexdigest()
        report["identity_basis"] = "96 exact official BF16 hashes plus unchanged input/gate controls; OrcaRouter's documented edit excludes these gates"
        qualified = report["all_controls_identical"]
    report["precision_experiment_identity_checks_passed"] = qualified
    if qualified:
        path = output / "gdn-gates-bf16.safetensors"
        if path.exists():
            raise ValueError("refusing to overwrite an existing reference shard")
        save_file(restored, path, metadata={"format": "pt", "purpose": "isolated GDN gate precision comparison"})
        report["reference_shard_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "gate-quantization.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("gates", "controls")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantized", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-verification", type=Path)
    args = parser.parse_args()
    probe(args.quantized, args.reference, args.output, args.official_verification)

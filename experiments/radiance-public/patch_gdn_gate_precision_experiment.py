"""Isolated BF16 control-gate comparison; never rewrites model checkpoints.

The pinned public BF16 gate reference is authenticated before loading. Every
replacement is checked again in the actual target model after its weight loader
has mapped and packed the tensors. No generation or sampling rules are changed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil


MODEL = "/models/Qwen3.8-27B-Uncensored-MXFP4-awq"
CONTRACT = Path("/benchmark/gate-reference-contract.json")
QUARK = "vllm/model_executor/layers/quantization/quark/quark.py"
LOADER = "vllm/model_executor/model_loader/default_loader.py"
PREIMAGES = {
    QUARK: "1eed93af5c03cc148f1d28071990e2c42786007364f5ad38d02c85c178c9b73c",
    LOADER: "7610702c528052f176ee8380300cf966e2531f3c9553dbaaa28259295efe00d4",
}


def tensor_digest(tensor):
    import torch

    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def load_reference():
    import torch
    from safetensors.torch import load_file

    contract = json.loads(CONTRACT.read_text())
    shard = Path(contract["shard_path"])
    if hashlib.sha256(shard.read_bytes()).hexdigest() != contract["shard_sha256"]:
        raise ValueError("BF16 gate reference shard hash mismatch")
    weights = load_file(shard, device="cpu")
    if len(weights) != 96 or set(weights) != set(contract["tensors"]):
        raise ValueError("BF16 gate reference does not cover all 48 layers")
    for name, weight in weights.items():
        if (not re.fullmatch(r"model\.language_model\.layers\.\d+\.linear_attn\.in_proj_[ab]\.weight", name)
                or weight.dtype != torch.bfloat16 or tuple(weight.shape) != (48, 5120)
                or tensor_digest(weight) != contract["tensors"][name]):
            raise ValueError("BF16 gate reference tensor identity mismatch")
    return weights, contract


def replace_gate_weights(weights, model_path):
    if model_path != MODEL:
        yield from weights
        return
    reference, contract = load_reference()
    scales = {name.removesuffix(".weight") + ".weight_scale" for name in reference}
    seen_weights, seen_scales = set(), set()
    for name, weight in weights:
        if name in reference:
            if name in seen_weights or tensor_digest(weight) != contract["quantized_tensors"][name]:
                raise ValueError("unexpected source gate weight during load")
            seen_weights.add(name)
            yield name, reference[name]
        elif name in scales:
            if name in seen_scales or tensor_digest(weight) != contract["quantized_scales"][name]:
                raise ValueError("unexpected source gate scale during load")
            seen_scales.add(name)
        else:
            yield name, weight
    if seen_weights != set(reference) or seen_scales != scales:
        raise ValueError("incomplete source gate replacement")


def attest_loaded_gates(model, model_path):
    if model_path != MODEL:
        return
    import torch

    reference, contract = load_reference()
    rows = []
    for name, module in model.named_modules():
        if not name.endswith(".linear_attn.in_proj_ba"):
            continue
        match = re.search(r"\.layers\.(\d+)\.linear_attn\.in_proj_ba$", name)
        if not match:
            raise ValueError("unexpected mapped gate module")
        prefix = f"model.language_model.layers.{match[1]}.linear_attn.in_proj_"
        expected = torch.cat([reference[prefix + part + ".weight"] for part in ("b", "a")])
        actual = module.weight.detach().cpu()
        if (module.quant_method.__class__.__name__ != "UnquantizedLinearMethod"
                or actual.dtype != torch.bfloat16 or not torch.equal(actual, expected)):
            raise ValueError("loaded GPU gate is not the exact BF16 reference")
        rows.append({"layer": int(match[1]), "sha256": tensor_digest(actual),
                     "elements": actual.numel(), "dtype": str(actual.dtype)})
    if len(rows) != 48 or len({row["layer"] for row in rows}) != 48:
        raise ValueError("not every target gate was verified after model loading")
    report = {"gate_modules": len(rows), "all_exact_bf16": True,
              "reference_shard_sha256": contract["shard_sha256"], "layers": rows}
    Path("/benchmark/gate-loaded-attestation.json").write_text(json.dumps(report, indent=2) + "\n")


def install(package: Path):
    changes = {}
    for relative, expected in PREIMAGES.items():
        path = package / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"gate precision experiment source mismatch: {relative}")
        changes[relative] = path.read_text()
    anchor = "        # Check if the layer is skipped for quantization.\n"
    if changes[QUARK].count(anchor) != 1:
        raise ValueError("quantization construction anchor is ambiguous")
    changes[QUARK] = changes[QUARK].replace(anchor,
        "        # Isolated, verified GDN gate precision experiment.\n"
        "        if prefix.endswith('.linear_attn.in_proj_ba'):\n"
        "            return UnquantizedLinearMethod()\n" + anchor)
    before = "        yield from self._get_weights_iterator(primary_weights)\n"
    after = ("        from qwen_gdn_gate_precision_experiment import replace_gate_weights\n"
             "        yield from replace_gate_weights(self._get_weights_iterator(primary_weights), model_config.model)\n")
    if changes[LOADER].count(before) != 1:
        raise ValueError("primary weight iteration anchor is ambiguous")
    changes[LOADER] = changes[LOADER].replace(before, after)
    before = "        loaded_weights = model.load_weights(self.get_all_weights(model_config, model))\n"
    if changes[LOADER].count(before) != 1:
        raise ValueError("loaded model attestation anchor is ambiguous")
    changes[LOADER] = changes[LOADER].replace(before, before +
        "        from qwen_gdn_gate_precision_experiment import attest_loaded_gates\n"
        "        attest_loaded_gates(model, model_config.model)\n")
    result = {}
    for relative, source in changes.items():
        compile(source, relative, "exec")
        (package / relative).write_text(source)
        result[relative] = hashlib.sha256(source.encode()).hexdigest()
    shutil.copyfile(__file__, package / "qwen_gdn_gate_precision_experiment.py")
    return result

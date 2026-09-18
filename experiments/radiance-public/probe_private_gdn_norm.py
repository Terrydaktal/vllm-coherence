"""Replay captured private GDN norms; publish only identities and numerical statistics."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import (
    DiagnosticError,
    authenticate,
    private_json,
    seal,
    write_private,
)


def qualify(args):
    import torch
    from analyze_private_d7_rows import checked_bytes, difference
    from safetensors import safe_open
    from stock_m1_gdn_norm import StockM1GdnNorm

    candidate = StockM1GdnNorm()
    roots = [args.corpus / args.revision / arm / "000/trace" for arm in ("m1", "m8")]
    docs = [private_json(root / "manifest.json") for root in roots]
    for doc in docs:
        authenticate(doc)
    if docs[0]["positions"] != docs[1]["positions"] or len(docs[0]["positions"]) != 9:
        raise DiagnosticError("norm replay requires the paired eight-position capture")
    positions = docs[0]["positions"][1:]
    tables = [
        {(r["position"], r["module"], r["phase"], r["path"]): r for r in d["records"]} for d in docs
    ]
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    config = json.loads((args.model / "config.json").read_text())
    eps = config.get("text_config", config)["rms_norm_eps"]
    checks = []

    def tensor(arm, module, phase, path):
        result = []
        for position in positions:
            row = tables[arm][position, module, phase, path]
            if row["dtype"] != "torch.bfloat16" or row["shape"] != [48, 128]:
                raise DiagnosticError("unexpected private GDN norm capture layout")
            raw = checked_bytes(roots[arm], row)
            result.append(torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(48, 128))
        return torch.cat(result).cuda()

    def compare(label, left, right):
        a, b = [
            t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes() for t in (left, right)
        ]
        item = {
            "check": label,
            "equal": a == b,
            "elements": left.numel(),
            "left_sha256": hashlib.sha256(a).hexdigest(),
            "right_sha256": hashlib.sha256(b).hexdigest(),
        }
        if a != b:
            item.update(difference(a, b, str(left.dtype)))
        checks.append(item)

    for layer in (i for i in range(64) if i % 4 != 3):
        module = f"language_model.model.layers.{layer}.linear_attn.norm"
        name = f"model.language_model.layers.{layer}.linear_attn.norm.weight"
        with safe_open(args.model / index[name], framework="pt", device="cpu") as f:
            weight = f.get_tensor(name).cuda()
        inputs = [
            (
                tensor(arm, module, "before", "call.args.0"),
                tensor(arm, module, "before", "call.args.1"),
            )
            for arm in (0, 1)
        ]
        expected = [tensor(arm, module, "after", "call.result") for arm in (0, 1)]

        def native(x, z, weight=weight):
            return candidate.native.layer_norm_fwd(
                x,
                weight,
                None,
                eps,
                z=z,
                norm_before_gate=True,
                is_rms_norm=True,
                activation="silu",
            )[0]

        x, z = inputs[0]
        reference = torch.cat([native(x[i : i + 48], z[i : i + 48]) for i in range(0, 384, 48)])
        compare(f"layer-{layer}-reproduce-m1", reference, expected[0])
        compare(f"layer-{layer}-reproduce-m8", native(*inputs[1]), expected[1])
        compare(f"layer-{layer}-original-batch-same-input", native(x, z), reference)
        compare(f"layer-{layer}-candidate-same-input", candidate(x, z, weight, eps), reference)
    return {
        "checks": checks,
        "trace_sha256": [d["sha256"] for d in docs],
        "m1_rows_per_block": candidate.native.calc_rows_per_block(48, torch.device("cuda", 0)),
        "m8_rows_per_block": candidate.native.calc_rows_per_block(384, torch.device("cuda", 0)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    if not args.allow_gpu or not os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"):
        raise DiagnosticError("private norm replay requires admission and shared GPU lease")
    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease

    os.umask(0o077)
    args.output.mkdir(mode=0o700)
    with gpu_lease(args.output / "gpu-lease"):
        result = qualify(args)
    report = seal(
        {
            "status": "MEASURED",
            **result,
            "formal_equivalence": "UNPROVED",
            "scope": (
                "48 gated norms on eight captured private Pi positions; "
                "actual outputs authenticated before intervention."
            ),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    write_private(args.output / "probe-result.json", report)
    groups = {
        suffix: sum(r["equal"] for r in report["checks"] if r["check"].endswith(suffix))
        for suffix in (
            "reproduce-m1",
            "reproduce-m8",
            "original-batch-same-input",
            "candidate-same-input",
        )
    }
    print(
        json.dumps(
            {
                "sha256": report["sha256"],
                "groups": groups,
                "row_tiles": [report["m1_rows_per_block"], report["m8_rows_per_block"]],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

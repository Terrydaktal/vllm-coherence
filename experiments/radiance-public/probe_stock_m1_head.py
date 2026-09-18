"""Measure supported small-batch BF16 head paths against the pinned M1 operator.

Only synthetic full-model captures are admitted by this diagnostic. It does not
decode tokens, alter a model file, or install a candidate into a server.
"""

import argparse
import hashlib
import json
import os
import statistics
import types
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import DiagnosticError, authenticate, seal, write_private


def qualify(args):
    import torch
    from analyze_gdn_decode_transition import array, call
    from probe_stock_gdn_sequence import compare_tensors
    from safetensors import safe_open
    from vllm import envs
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

    torch.set_num_threads(2)
    model_report = json.loads((args.capture / "probe-result.json").read_text())
    authenticate(model_report)
    if not model_report["scope"].startswith("Synthetic 129-token prefix"):
        raise DiagnosticError("head probe requires the declared synthetic capture")
    site = "target.language_model.logits_processor"
    captures = [call(args.capture, arm, site) for arm in ("serial", "d7")]
    hidden = torch.from_numpy(array(captures[1] / "before", "args.1")).to(
        device="cuda", dtype=torch.bfloat16
    )
    expected_first = torch.from_numpy(array(captures[0] / "after", "result")).to(
        device="cuda", dtype=torch.bfloat16
    )
    model = Path(json.loads(args.spec.read_text())["native_config"]["model"])
    index = json.loads((model / "model.safetensors.index.json").read_text())
    with safe_open(str(model / index["weight_map"]["lm_head.weight"]), framework="pt") as source:
        weight = source.get_tensor("lm_head.weight").to("cuda")
    if weight.dtype != torch.bfloat16 or weight.shape != (248320, 5120):
        raise DiagnosticError("unexpected pinned vocabulary head")
    head = types.SimpleNamespace(weight=weight, quant_method=UnquantizedEmbeddingMethod())
    state = types.SimpleNamespace(head_dtype=None)
    exact = types.MethodType(LogitsProcessor._apply_head, state)

    def grouped(x, size):
        return torch.cat([exact(head, x[i : i + size], None) for i in range(0, len(x), size)])

    reproduced = compare_tensors(
        "captured-stock-m1", grouped(hidden[:1], 1), expected_first, args.output
    )
    if not reproduced["equal"]:
        raise DiagnosticError("head dispatch did not reproduce the captured stock M1 output")
    modes = {
        "serial-m1": lambda x: grouped(x, 1),
        "groups-2": lambda x: grouped(x, 2),
        "groups-4": lambda x: grouped(x, 4),
        "groups-5": lambda x: grouped(x, 5),
        "original-m8": lambda x: exact(head, x, None),
        "strided-bmm-m1": lambda x: torch.bmm(
            x.unsqueeze(1), weight.t().unsqueeze(0).expand(len(x), -1, -1)
        ).squeeze(1),
    }
    torch.manual_seed(1937)
    inputs = [
        hidden,
        hidden[:1].expand(8, -1).clone(),
        (hidden.float() + torch.randn_like(hidden.float()) * 0.03125).bfloat16(),
        (hidden.float() * 0.5).bfloat16(),
        (hidden.float() * 2).bfloat16(),
        torch.randn_like(hidden),
    ]
    checks = [reproduced]
    for index, x in enumerate(inputs):
        reference = grouped(x, 1)
        for name, function in modes.items():
            checks.append(
                compare_tensors(f"input-{index}-{name}", function(x), reference, args.output)
            )
    timings = {}
    for name, function in modes.items():
        for _ in range(3):
            function(hidden)
        samples = []
        for _ in range(15):
            before, after = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            before.record()
            function(hidden)
            after.record()
            after.synchronize()
            samples.append(before.elapsed_time(after))
        timings[name] = {"median_ms": statistics.median(samples), "samples_ms": samples}
    return {
        "checks": checks,
        "timings": timings,
        "stock_skinny_gemm": bool(envs.VLLM_ROCM_USE_SKINNY_GEMM),
        "capture": model_report["sha256"],
        "weight_shape": list(weight.shape),
        "positions_per_mode": sum(len(x) for x in inputs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "capture", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    if not args.allow_gpu or not os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"):
        raise DiagnosticError("head probe requires admission and the shared GPU lease")
    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease

    os.umask(0o077)
    args.output.mkdir(mode=0o700)
    result, error = {}, None
    try:
        with gpu_lease(args.output / "gpu-lease"):
            result = qualify(args)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    report = seal(
        {
            "status": "MEASURED" if error is None else "FAILED",
            "error": error,
            **result,
            "scope": "48 synthetic hidden vectors; full BF16 logits and head-only event timings.",
            "formal_equivalence": "UNPROVED",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    write_private(args.output / "probe-result.json", report)
    print(json.dumps({k: report[k] for k in ("status", "error", "sha256")}), flush=True)
    return 0 if error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())

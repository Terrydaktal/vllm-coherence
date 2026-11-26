"""Replay numerical divergence at original Q/K RMSNorm inputs and measure both paths."""

import argparse
import ast
import hashlib
import json
import statistics
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from stock_m1_norm import StockM1Norm
from wide_qk_norm import BINARY_SHA256, KERNEL_SHA256

from qwen_r9700_lab.diagnostic_contract import seal

IR_FILE = Path("/opt/vllm/lib/python3.12/site-packages/vllm/ir/ops/layernorm.py")
IR_SHA = "65d33dcb96404ddde273acf84ef901151a8155a2cffc144bdd0c49fe1d576a22"
GEMMA_FILE = Path(
    "/opt/vllm/lib/python3.12/site-packages/vllm/model_executor/layers/layernorm.py"
)
GEMMA_SHA = "3460eba01758849027bb59b8592626406e33ab25b23d2def729bb0ec42afce67"


def original():
    # Execute the two installed native method bodies directly, detached only from the
    # vLLM registration machinery. The captured original outputs independently verify
    # that the reference has reconstructed this exact route on the affected inputs.
    ns = {"torch": torch, "Tensor": torch.Tensor}
    for path, digest, name in (
        (IR_FILE, IR_SHA, "rms_norm"),
        (GEMMA_FILE, GEMMA_SHA, "GemmaRMSNorm"),
    ):
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != digest:
            raise ValueError("installed reference source changed")
        node = next(
            n for n in ast.parse(source).body if getattr(n, "name", None) == name
        )
        if name == "GemmaRMSNorm":
            node = next(
                n for n in node.body if getattr(n, "name", None) == "forward_native"
            )
        node.decorator_list.clear()
        # Only the exact upstream function authenticated above is admitted.
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)  # noqa: S102
        if name == "rms_norm":
            ns["ir"] = SimpleNamespace(ops=SimpleNamespace(rms_norm=ns["rms_norm"]))
    return ns["forward_native"]


def capture(base, event, arm):
    file = base / arm / "stages/group-0000.pt"
    raw = torch.load(file, map_location="cpu", weights_only=True, mmap=True)
    return raw[f"{event}.before.args.0"], raw[f"{event}.after.result"]


def mismatches(a, b):
    return int((a.view(torch.int16) != b.view(torch.int16)).count_nonzero())


def timings(arms):
    samples = {name: [] for name in arms}
    for repeat in range(7):
        for name in list(arms) if repeat % 2 == 0 else reversed(arms):
            run = arms[name]
            run()
            torch.cuda.synchronize()
            start = time.monotonic_ns()
            for _ in range(9):
                run()
            torch.cuda.synchronize()
            samples[name].append((time.monotonic_ns() - start) / 9000)
    return {name: statistics.median(series) for name, series in samples.items()}


def run(args):
    torch.manual_seed(4711)
    ref = original()
    native = StockM1Norm(args.build)
    if (
        native.manifest["source_sha256"] != KERNEL_SHA256
        or native.manifest["binary_sha256"] != BINARY_SHA256
    ):
        raise ValueError("wrong native kernel")
    index = json.loads((args.model / "model.safetensors.index.json").read_bytes())[
        "weight_map"
    ]
    evidence = []
    times = []
    for width, layer, k_event, root in (
        (256, 55, 621, args.short),
        (1024, 51, 576, args.long),
    ):
        for head, event in (("q", k_event - 1), ("k", k_event)):
            key = f"model.language_model.layers.{layer}.self_attn.{head}_norm.weight"
            with safe_open(args.model / index[key], framework="pt", device="cpu") as f:
                weight = f.get_tensor(key).to(device="cuda", dtype=torch.bfloat16)
            module = SimpleNamespace(weight=weight, variance_epsilon=1e-6)
            cpu, cold = capture(root, event, "prefill")
            other, decode = capture(root, event, "decode")
            if mismatches(cpu, other):
                raise ValueError("captured inputs differ")
            x = cpu.cuda()
            h = x.shape[1]
            one = ref(module, x)
            native_one = native(x, None, weight, 1e-6)
            binding = {
                "width": width,
                "head": head,
                "event": event,
                "captured_cold_vs_decode": mismatches(cold, decode),
                "original_m1_vs_decode": mismatches(one, decode.cuda()),
                "native_m1_vs_decode": mismatches(native_one, decode.cuda()),
                "cases": [],
            }
            # Replicate the *actual row* for an independent reference across every output,
            # including the two adversarial values absent from typical random corpora.
            for n in (8, 9, 64, width + 65, 1648, 3296):
                rows = x.expand(n, h, 256).contiguous()
                cooked = native(rows, None, weight, 1e-6)
                old = ref(module, rows)
                expected = one.expand_as(rows)
                binding["cases"].append(
                    {
                        "rows": n,
                        "values": rows.numel(),
                        "native_vs_m1": mismatches(cooked, expected),
                        "old_vs_m1": mismatches(old, expected),
                        "old_vs_captured_cold": mismatches(old[:1], cold.cuda()),
                    }
                )
            evidence.append(binding)
            if width != 256:
                continue
            for n in (64, 1648, 3296):
                for layout in ("contiguous", "strided"):
                    stride = h * 256 if layout == "contiguous" else 56 * 256
                    x_b = torch.empty_strided(
                        (n, h, 256),
                        (stride, 256, 1),
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    x_b.normal_()
                    candidate = partial(native, x_b, None, weight, 1e-6)
                    baseline = partial(ref, module, x_b)
                    fixed = candidate()
                    samples = sorted({0, n - 1, n // 2, 1, n // 3})
                    errors = sum(
                        mismatches(fixed[i : i + 1], ref(module, x_b[i : i + 1]))
                        for i in samples
                    )
                    times.append(
                        {
                            "head": head,
                            "rows": n,
                            "layout": layout,
                            "row_stride": stride,
                            "sampled_m1_mismatches": errors,
                            "microseconds": timings(
                                {"old": baseline, "candidate": candidate}
                            ),
                        }
                    )
    good = all(
        not b["original_m1_vs_decode"]
        and not b["native_m1_vs_decode"]
        and all(c["native_vs_m1"] == 0 for c in b["cases"])
        for b in evidence
    )
    proved = all(
        b["captured_cold_vs_decode"] == 1
        and any(
            c["old_vs_captured_cold"] == 0 and c["old_vs_m1"] > 0 for c in b["cases"]
        )
        for b in evidence
        if b["head"] == "k"
    )
    return seal(
        {
            "source_sha256": KERNEL_SHA256,
            "binary_sha256": BINARY_SHA256,
            "captured": evidence,
            "timings": times,
            "passed": good
            and proved
            and not any(t["sampled_m1_mismatches"] for t in times),
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("build", "model", "short", "long", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = p.parse_args()
    result = run(args)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "sha256": result["sha256"]}))

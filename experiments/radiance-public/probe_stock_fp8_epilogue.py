"""Exact FP8 bytes/scales/residuals versus the frozen corrected norm + native quant.

No conversation data. Measures graph replay; allocations and Python dispatch are
outside the timed graph. The reference native quantizer is never replaced.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def run(args):
    os.environ["TORCHINDUCTOR_EMULATE_PRECISION_CASTS"] = "1"
    import torch
    from stock_fp8_epilogue import StockFP8Epilogue
    from stock_m1_norm import StockM1Norm
    from vllm import _custom_ops as ops

    torch.set_num_threads(4)
    torch.manual_seed(190918)
    candidate = StockFP8Epilogue(args.build)
    frozen = StockM1Norm(args.reference)
    report = {
        "status": "RUNNING",
        "checks": [],
        "timings": [],
        "private_chat_read": False,
        "input": "320 deterministic synthetic BF16 rows plus zero/tiny/adversarial values",
        "formal_equivalence": "UNPROVED",
        "build": candidate.manifest,
    }

    def compare(name, a, b):
        x, y = a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        row = {
            "name": name,
            "bytes": x.numel(),
            "equal": torch.equal(x, y),
            "unequal_bytes": int((x != y).sum()),
        }
        report["checks"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if not row["equal"]:
            raise RuntimeError(f"{name}: {row['unequal_bytes']} unequal bytes")

    def quant(x):
        return ops.scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=True)

    x = torch.randn(320, 5120, device="cuda", dtype=torch.bfloat16)
    r = torch.randn_like(x)
    w = torch.randn(5120, device="cuda", dtype=torch.bfloat16) * 0.1
    x[0].zero_()
    r[0].zero_()
    x[1] *= 1e-20
    r[1] *= 1e-20
    x[2] *= 128
    # A near-cancelling residual tests that the variance uses unrounded FP32 sums.
    r[3] = -x[3]
    r[3, ::7] += 0.00390625
    for m in (1, 8):
        for residual in (False, True):
            for start in range(0, 320, m):
                inp, res = (
                    x[start : start + m],
                    r[start : start + m] if residual else None,
                )
                expected = frozen(inp, res, w, 1e-6)
                normed, carry = expected if residual else (expected, None)
                eq, es = quant(normed)
                cq, cs, cr = candidate.norm(inp, res, w, 1e-6)
                label = f"norm-m{m}-r{int(residual)}-row{start}"
                compare(label + "/q", cq, eq)
                compare(label + "/scale", cs, es)
                if residual:
                    compare(label + "/carry", cr, carry)
    gu = torch.randn(320, 34816, device="cuda", dtype=torch.bfloat16) * 3
    gu[0].zero_()
    gu[1] *= 1e-20
    # Exercise all finite BF16 gate encodings, with a bounded up projection.
    values = (
        torch.arange(65536, device="cuda", dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
    )
    values = values[torch.isfinite(values)]
    gu[2:6, :17408].copy_(values.repeat(2)[: 4 * 17408].reshape(4, 17408))
    gu[2:6, 17408:].fill_(0.25)

    def silu_ref(t):
        return quant(torch.nn.functional.silu(t[:, :17408]) * t[:, 17408:])

    eager_silu_ref = silu_ref
    silu_ref = torch.compile(silu_ref, fullgraph=True, dynamic=False)
    report["silu_reference_compiled"] = True
    report["norm_reference"] = "same opaque qualified HIP norm used by compiled M8"
    for start in range(0, 320, 8):
        eq, es = silu_ref(gu[start : start + 8])
        if start == 0:
            oq, oscale = eager_silu_ref(gu[start : start + 8])
            compare("compiled-reference/q", eq, oq)
            compare("compiled-reference/scale", es, oscale)
        cq, cs = candidate.silu(gu[start : start + 8])
        compare(f"silu-{start}/q", cq, eq)
        compare(f"silu-{start}/scale", cs, es)
    for m in (1, 8):
        xx, rr, gg = x[8 : 8 + m], r[8 : 8 + m], gu[8 : 8 + m]

        def norm_ref(xx=xx, rr=rr):
            yy, carry = frozen(xx, rr, w, 1e-6)
            q, s = quant(yy)
            return q, s, carry

        functions = {
            "norm-reference": norm_ref,
            "norm-fused": lambda xx=xx, rr=rr: candidate.norm(xx, rr, w, 1e-6),
            "silu-reference": lambda gg=gg: silu_ref(gg),
            "silu-fused": lambda gg=gg: candidate.silu(gg),
        }
        outputs = {}
        for name, fn in functions.items():
            fn()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = fn()
            for t in result:
                t.view(torch.uint8).fill_(42)
            graph.replay()
            outputs[name] = tuple(t.clone() for t in result)
            # Many operators inside ONE replay avoid timing the Python launch
            # gaps as if they were GPU work.
            packed_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(packed_graph):
                for _ in range(100):
                    fn()
            for _ in range(3):
                packed_graph.replay()
            samples = []
            for _ in range(5):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                packed_graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) / 100)
            report["timings"].append(
                {"stage": name, "rows": m, "ms": sorted(samples)[2]}
            )
        for stage in ("norm", "silu"):
            for index, (got, expected) in enumerate(
                zip(
                    outputs[stage + "-fused"],
                    outputs[stage + "-reference"],
                    strict=True,
                )
            ):
                compare(f"graph/{stage}-m{m}-{index}", got, expected)
    bad = cq.view(torch.uint8).clone()
    bad.flatten()[0] ^= 1
    if torch.equal(bad, cq.view(torch.uint8)):
        raise RuntimeError("negative control failed")
    qualify_prefill(args, candidate, compare, report)
    report.update(
        status="SAMPLE_CHECKED",
        negative_control=True,
        sources={
            n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
            for n in (
                "stock_fp8_epilogue.py",
                "stock_fp8_epilogue.hip",
                "probe_stock_fp8_epilogue.py",
            )
        },
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "timings")}), flush=True)


def qualify_prefill(args, candidate, compare, report):
    """Check every decoder weight against the explicitly selected norm contract."""
    import torch
    from safetensors import safe_open
    from vllm import _custom_ops as ops
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from stock_m1_norm import StockM1Norm

    model = Path("/models/Qwen3.8-27B-Uncensored-MXFP4-awq")
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    names = sorted(
        n
        for n in index
        if n.startswith("model.language_model.layers.")
        and n.endswith((".input_layernorm.weight", ".post_attention_layernorm.weight"))
    )
    if len(names) != 128:
        raise RuntimeError("prefill decoder norm inventory changed")
    with set_current_vllm_config(VllmConfig()):
        module = GemmaRMSNorm(5120, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    canonical = StockM1Norm(args.reference) if args.row_invariant else None
    report["row_invariant"] = args.row_invariant
    torch.manual_seed(20260918)
    x = torch.randn(2048, 5120, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    x[0].zero_()
    residual[0].zero_()
    x[1] *= 1e-20
    residual[1] *= 1e-20
    residual[3] = -x[3]
    residual[3, ::7] += 0.00390625
    for site, name in enumerate(names):
        with safe_open(model / index[name], framework="pt", device="cpu") as f:
            module.weight.data.copy_(f.get_tensor(name).to("cuda"))
        for width in (9, 15, 16, 320, 1000, 1648, 2048) if site == 0 else (1000,):
            for has_residual in (False, True):
                xx, rr = x[:width], residual[:width] if has_residual else None
                result = (
                    canonical(xx, rr, module.weight, 1e-6)
                    if canonical is not None
                    else module.forward_native(xx, rr)
                )
                y, carry = result if has_residual else (result, None)
                q, scale = ops.scaled_fp8_quant(
                    y, scale=None, use_per_token_if_dynamic=True
                )
                actual = candidate.norm(xx, rr, module.weight, 1e-6)
                prefix = f"prefill-site{site}-m{width}-r{int(has_residual)}"
                for label, got, expected in zip(
                    ("q", "scale", "carry"), actual, (q, scale, carry), strict=True
                ):
                    if expected is not None:
                        compare(prefix + "/" + label, got, expected)
    report["prefill_sites"] = 128
    report["prefill_rows_per_site"] = 1000
    report["prefill_widths"] = [9, 15, 16, 320, 1000, 1648, 2048]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("build", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument(
        "--row-invariant",
        action="store_true",
        help="Use the declared M1 arithmetic for prefill rows too",
    )
    args = parser.parse_args()
    if not args.allow_gpu or not os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"):
        raise RuntimeError("explicit GPU admission and lease required")
    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease

    os.umask(0o077)
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with gpu_lease(args.output.parent / ("lease-" + str(time.time_ns()))):
        run(args)

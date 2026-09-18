"""Exact native/graph regression for the isolated GEMM dispatch backport.

Uses checkpoint weights and deterministic synthetic BF16 activations, never
conversation text. The shipped extension is the frozen oracle; a rebuilt
unmodified control detects compiler/source drift independently of the patch.
"""

import argparse
import importlib.util
import json
import os
import statistics
import time
from pathlib import Path

from build_mxfp4_dispatch import ORIGINAL_BINARY_SHA256, digest


def load_extension(path, label):
    spec = importlib.util.spec_from_file_location(label + ".radiance_mxfp4_fp8", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(args):
    for key, value in {
        "RADIANCE_MXFP4": "1",
        "RADIANCE_MXFP4_W4A8": "1",
        "RADIANCE_MXFP4_WPERM": "1",
        "RADIANCE_MXFP4_DECODE_MAX_M": "64",
        "RADIANCE_MXFP4_DECODE_NT": "1",
        "RADIANCE_MXFP4_TN4_MIN_M": "2048",
        "RADIANCE_MXFP4_EPIFAST": "1",
        "RADIANCE_MXFP4_W4A8_MIN_M": "0",
    }.items():
        os.environ[key] = value
    if args.split_k:
        os.environ["RADIANCE_MXFP4_DECODE_KS"] = str(args.split_k)
    else:
        os.environ.pop("RADIANCE_MXFP4_DECODE_KS", None)
    import radiance_mxfp4 as kernel
    import torch
    from safetensors import safe_open
    from vllm import _custom_ops as ops

    torch.set_num_threads(4)
    torch.manual_seed(20260918)
    report = {
        "status": "RUNNING",
        "build": digest(args.build / "build.json"),
        "probe_sha256": digest(__file__),
        "split_k": args.split_k or "automatic",
        "cases": [],
        "timings": [],
        "reference": "frozen shipped extension",
        "private_chat_read": False,
        "input": "deterministic synthetic BF16 activations; actual checkpoint weights",
    }
    build = json.loads((args.build / "build.json").read_text())
    original = Path(kernel._ext.__file__)
    if digest(original) != ORIGINAL_BINARY_SHA256:
        raise ValueError("frozen oracle binary changed")
    modules = {"original": kernel._ext}
    for name in ("control", "candidate"):
        binary = args.build / name / "radiance_mxfp4_fp8.so"
        if digest(binary) != build["variants"][name]["binary_sha256"]:
            raise ValueError("built binary changed")
        modules[name] = load_extension(binary, name)
    guard = 512
    workspaces = []
    for name, module in modules.items():
        width = 36864 if name == "candidate" else 32768
        size = 4 * 64 * width
        partial = torch.full((size + 2 * guard,), 12345.0, dtype=torch.float32, device="cuda")
        count = torch.full((width // 128 + 8 + 2 * guard,), 12345, dtype=torch.int32, device="cuda")
        count[guard:-guard].zero_()
        module.set_decode_scratch(partial[guard:].data_ptr(), size * 4, count[guard:].data_ptr())
        workspaces.append((partial, count))
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]

    def weight(name):
        with safe_open(args.model / index[name], framework="pt", device="cpu") as file:
            return file.get_tensor(name)

    def record(case):
        report["cases"].append(case)
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(case), flush=True)

    def bits_equal(a, b):
        return torch.equal(a.view(torch.int16), b.view(torch.int16))

    def compare(a, b):
        return {
            "equal": bits_equal(a, b),
            "unequal_elements": int((a.view(torch.int16) != b.view(torch.int16)).sum()),
            "elements": a.numel(),
            "max_abs_error": float((a.float() - b.float()).abs().max()),
        }

    def check(raw, scale, *, layer, rows, width, graph=False, timing=False):
        # Repeating actual rows tests both sides of the width boundary without
        # inventing a different quantization scheme or allocating a whole model.
        repeat = (width + raw.shape[0] - 1) // raw.shape[0]
        raw = raw.repeat(repeat, 1)[:width].contiguous().cuda()
        scale = scale.repeat(repeat, 1)[:width].T.contiguous().cuda()
        k = raw.shape[1] * 2
        packed = kernel.permute_w(raw, width, k)
        ref = kernel.make_row_ref(scale)
        x = torch.randn((rows, k), device="cuda", dtype=torch.bfloat16)
        xq, xs = ops.scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=True)
        xs = xs.view(-1).contiguous()
        outputs, slabs, graphs = {}, [], {}

        def launch(module, out, start, count):
            module.launch(
                xq[start:].data_ptr(),
                packed.data_ptr(),
                scale.data_ptr(),
                ref.data_ptr(),
                xs[start:].data_ptr(),
                out[start:].data_ptr(),
                count,
                width,
                k,
                torch.cuda.current_stream().cuda_stream,
            )

        bands = (1, 8) if rows == 320 else (rows,)
        for m in bands:
            for name, module in modules.items():
                slab = torch.full(
                    (rows * width + 2 * guard,), 42.0, device="cuda", dtype=torch.bfloat16
                )
                out = slab[guard:-guard].view(rows, width)
                slabs.append(slab)
                for start in range(0, rows, m):
                    launch(module, out, start, min(m, rows - start))
                if graph:
                    # Capture the native optimized release kernel, not Python timing.
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        for start in range(0, rows, m):
                            launch(module, out, start, min(m, rows - start))
                    out.fill_(float("nan"))
                    for _ in range(3):
                        g.replay()
                    graphs[name] = g
                outputs[(m, name)] = out
            torch.cuda.synchronize()
            a, b, c = (outputs[(m, name)] for name in modules)
            case = {
                "layer": layer,
                "M": m,
                "N": width,
                "K": k,
                "rows": rows,
                "graph": graph,
                "control_vs_original": compare(a, b),
                "candidate_vs_original": compare(a, c),
                "finite": bool(torch.isfinite(a).all() and torch.isfinite(c).all()),
                "canaries": all(
                    bool((s[:guard] == v).all() and (s[-guard:] == v).all())
                    for s, v in [(s, 42) for s in slabs]
                    + [(s, 12345) for pair in workspaces for s in pair]
                ),
                "counters_zero": all(
                    bool((pair[1][guard:-guard] == 0).all()) for pair in workspaces
                ),
            }
            record(case)
            if not (
                case["control_vs_original"]["equal"]
                and case["candidate_vs_original"]["equal"]
                and case["finite"]
                and case["canaries"]
                and case["counters_zero"]
            ):
                raise AssertionError("exact dispatch regression failed; see result.json")
            if timing:
                samples = {name: [] for name in modules}
                for trial in range(9):
                    for name in list(modules) if trial % 2 == 0 else reversed(modules):
                        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                        start.record()
                        for _ in range(20):
                            graphs[name].replay()
                        end.record()
                        end.synchronize()
                        samples[name].append(start.elapsed_time(end) / 20)
                report["timings"].append(
                    {
                        "M": m,
                        "N": width,
                        "K": k,
                        "method": "GPU events, graph replay, 9 alternating batches of 20",
                        "median_ms": {
                            name: statistics.median(values) for name, values in samples.items()
                        },
                    }
                )
        if rows == 320:
            case = {
                "layer": layer,
                "cross_width": "M1 versus M8",
                "rows": rows,
                **{name: compare(outputs[(1, name)], outputs[(8, name)]) for name in modules},
            }
            record(case)
            if not all(case[name]["equal"] for name in modules):
                raise AssertionError("M1/M8 dispatch regression")
        # A deliberately flipped BF16 bit must be detected by the same comparator.
        changed = next(iter(outputs.values())).clone()
        changed.view(torch.int16).view(-1)[0] ^= 1
        if compare(next(iter(outputs.values())), changed)["equal"]:
            raise AssertionError("negative control not detected")

    started = time.monotonic()
    try:
        for layer in (0,) if args.split_k else (0, 31, 63):
            prefix = f"model.language_model.layers.{layer}.mlp."
            raw = torch.cat(
                [weight(prefix + part + ".weight") for part in ("gate_proj", "up_proj")]
            )
            scale = torch.cat(
                [weight(prefix + part + ".weight_scale") for part in ("gate_proj", "up_proj")]
            )
            if not args.split_k:
                check(raw, scale, layer=layer, rows=320, width=34816)
                for m in (1, 8):
                    check(
                        raw, scale, layer=layer, rows=m, width=34816, graph=True, timing=layer == 0
                    )
            if layer == 0:
                for n in (32768, 32784, 34816, 36864, 36880):
                    for m in (1, 8, 64, 65):
                        check(raw, scale, layer=layer, rows=m, width=n, graph=True)
        report["status"] = "SAMPLE_CHECKED"
        report["negative_control_detected"] = True
    except Exception:
        report["status"] = "FAILED"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "timings": report["timings"],
                "elapsed_seconds": report["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model", type=Path, default=Path("/models/Qwen3.8-27B-Uncensored-MXFP4-awq")
    )
    parser.add_argument("--split-k", type=int, choices=(0, 4), default=0)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease

    with gpu_lease(args.output / "lease"):
        run(args)

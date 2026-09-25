"""Exact operator qualification and DRAM-fed timing for decode staging changes.

No private transcript is read. Tests use checkpoint weights and seeded BF16
activations. All timing kernels are graph captured; events surround replay
batches, never individual kernels. These are operator, not serving timings.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extension(path, name):
    spec = importlib.util.spec_from_file_location(name + ".radiance_mxfp4_fp8", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(args):
    for key, value in {
        "RADIANCE_MXFP4_W4A8": "1",
        "RADIANCE_MXFP4_W4A8_MIN_M": "0",
        "RADIANCE_MXFP4_WPERM": "1",
        "RADIANCE_MXFP4_DECODE_NT": "1",
        "RADIANCE_MXFP4_DECODE_MAX_M": "64",
        "RADIANCE_MXFP4_R4D_DECODE_MAX_M": "0",
        "RADIANCE_MXFP4_TN4_MIN_M": "2048",
        "RADIANCE_MXFP4_EPIFAST": "1",
    }.items():
        os.environ[key] = value
    os.environ.pop("RADIANCE_MXFP4_DECODE_KS", None)
    sys.path.insert(0, str(args.source))
    import torch
    from safetensors import safe_open
    from vllm import _custom_ops as ops

    import radiance_mxfp4 as wrapper

    torch.set_num_threads(4)
    torch.manual_seed(20260925)
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    build = json.loads((args.build / "build.json").read_text())
    if build["parent"]["radiance_mxfp4_fp8.so"] != digest(
        args.source / "radiance_mxfp4_fp8.so"
    ):
        raise ValueError("qualified parent changed")
    modules = {"release": extension(args.source / "radiance_mxfp4_fp8.so", "release")}
    selected = (
        set(args.variants.split(",")) if args.variants else set(build["variants"])
    )
    if not selected <= set(build["variants"]) or "control" not in selected:
        raise ValueError("requested variants must exist and include rebuilt control")
    for name, entry in build["variants"].items():
        if name not in selected:
            continue
        path = args.build / name / "radiance_mxfp4_fp8.so"
        if digest(path) != entry["binary_sha256"]:
            raise ValueError("candidate binary changed")
        modules[name] = extension(path, name)
    guard, width = 512, 36864
    scratch = []
    for module in modules.values():
        partial = torch.full((4 * 64 * width + guard * 2,), 12345.0, device="cuda")
        counters = torch.full(
            (width // 16 + 8 + guard * 2,), 12345, dtype=torch.int32, device="cuda"
        )
        counters[guard:-guard].zero_()
        module.set_decode_scratch(
            partial[guard:].data_ptr(),
            (partial.numel() - guard * 2) * 4,
            counters[guard:].data_ptr(),
        )
        scratch.append((partial, counters))
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    def tensor(name):
        with safe_open(args.model / index[name], framework="pt", device="cpu") as file:
            return file.get_tensor(name)

    report = {
        "status": "RUNNING",
        "build_sha256": digest(args.build / "build.json"),
        "probe_sha256": digest(__file__),
        "cases": [],
        "timings": [],
        "private_chat_read": False,
        "scope": "operator samples, not full-model qualification",
        "selected_variants": sorted(selected),
        "excluded_variants": sorted(set(build["variants"]) - selected),
    }

    def save():
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    def compare(a, b):
        different = int((a.view(torch.int16) != b.view(torch.int16)).sum())
        return {
            "different": different,
            "elements": a.numel(),
            "max_abs": float((a.float() - b.float()).abs().max()),
        }

    shapes = {
        "gate_up": (0, "mlp", ("gate_proj", "up_proj"), 64),
        "down": (0, "mlp", ("down_proj",), 64),
        "attention_in": (3, "self_attn", ("q_proj", "k_proj", "v_proj"), 16),
        "attention_out": (3, "self_attn", ("o_proj",), 16),
        "gdn_in": (0, "linear_attn", ("in_proj_qkv", "in_proj_z"), 48),
        "gdn_out": (0, "linear_attn", ("out_proj",), 48),
    }
    try:
        for label in args.shapes.split(","):
            layer, component, names, count = shapes[label]
            prefix = f"model.language_model.layers.{layer}.{component}."
            raw = torch.cat([tensor(prefix + name + ".weight") for name in names])
            scales = torch.cat(
                [tensor(prefix + name + ".weight_scale") for name in names]
            )
            n, k = raw.shape[0], raw.shape[1] * 2
            w = wrapper.permute_w(raw, n, k).cuda()
            ws = scales.T.contiguous().cuda()
            wr = wrapper.make_row_ref(ws)
            x = torch.randn((args.rows, k), device="cuda", dtype=torch.bfloat16)
            xq, xs = ops.scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=True)
            xs = xs.view(-1).contiguous()

            def launch(
                module,
                out,
                start=0,
                m=8,
                weights=w,
                xq=xq,
                xs=xs,
                ws=ws,
                wr=wr,
                n=n,
                k=k,
            ):
                module.launch(
                    xq[start:].data_ptr(),
                    weights.data_ptr(),
                    ws.data_ptr(),
                    wr.data_ptr(),
                    xs[start:].data_ptr(),
                    out[start:].data_ptr(),
                    m,
                    n,
                    k,
                    torch.cuda.current_stream().cuda_stream,
                )

            outputs = {}
            for m in (1, 8):
                for name, module in modules.items():
                    slab = torch.full(
                        (args.rows * n + guard * 2,),
                        42.0,
                        dtype=torch.bfloat16,
                        device="cuda",
                    )
                    out = slab[guard:-guard].view(args.rows, n)
                    # Warm the exact native entry before graph capture.
                    for start in range(0, args.rows, m):
                        launch(module, out, start, min(m, args.rows - start))
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for start in range(0, args.rows, m):
                            launch(module, out, start, min(m, args.rows - start))
                    out.fill_(float("nan"))
                    for _ in range(3):
                        graph.replay()
                    torch.cuda.synchronize()
                    if not bool(torch.isfinite(out).all()):
                        raise AssertionError("non-finite operator output")
                    if not bool(
                        (slab[:guard] == 42).all() and (slab[-guard:] == 42).all()
                    ):
                        raise AssertionError("output guard damaged")
                    outputs[(m, name)] = out.clone()
                row = {
                    "shape": label,
                    "N": n,
                    "K": k,
                    "M": m,
                    "rows": args.rows,
                    "comparisons": {
                        name: compare(outputs[(m, "release")], outputs[(m, name)])
                        for name in modules
                        if name != "release"
                    },
                }
                report["cases"].append(row)
                save()
                print(json.dumps(row), flush=True)
                if any(v["different"] for v in row["comparisons"].values()):
                    raise AssertionError(
                        "candidate or rebuilt control differs from qualified release"
                    )
            for name in modules:
                equality = compare(outputs[(1, name)], outputs[(8, name)])
                report["cases"].append({"shape": label, "M1_vs_M8": name, **equality})
                if equality["different"]:
                    raise AssertionError("M1/M8 equality lost")
            corrupt = outputs[(8, "release")].clone()
            corrupt.view(torch.int16)[0, 0] ^= 1
            if compare(outputs[(8, "release")], corrupt)["different"] != 1:
                raise AssertionError("negative control not detected")
            # Rotate >=192 MiB of identical weight storage. A single weight
            # can reside in Infinity Cache and make a bandwidth optimization
            # look faster than it is in a complete decoder.
            copies = max(6, (192 * 1024 * 1024 + w.numel() - 1) // w.numel())
            copies = min(copies, 24)
            weights = [w] + [w.clone() for _ in range(copies - 1)]
            graphs = {}
            out = torch.empty((8, n), dtype=torch.bfloat16, device="cuda")
            for name, module in modules.items():
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for weight in weights:
                        launch(module, out, weights=weight)
                graphs[name] = graph
            for _ in range(20):
                for graph in graphs.values():
                    graph.replay()
            samples = {name: [] for name in modules}
            for trial in range(9):
                order = list(modules) if trial % 2 == 0 else list(reversed(modules))
                for name in order:
                    start, end = (
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    )
                    start.record()
                    for _ in range(32):
                        graphs[name].replay()
                    end.record()
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end) / (32 * copies))
            row = {
                "shape": label,
                "M": 8,
                "N": n,
                "K": k,
                "layers": count,
                "weight_bytes": w.numel(),
                "rotating_weight_bytes": w.numel() * copies,
                "median_ms": {
                    name: statistics.median(vals) for name, vals in samples.items()
                },
                "samples_ms": samples,
            }
            report["timings"].append(row)
            print(json.dumps(row), flush=True)
            save()
            del weights, graphs, outputs, corrupt, w, ws, wr, x, xq, xs, out
            torch.cuda.empty_cache()
        for partial, counters in scratch:
            if not bool(
                (partial[:guard] == 12345).all()
                and (partial[-guard:] == 12345).all()
                and (counters[:guard] == 12345).all()
                and (counters[-guard:] == 12345).all()
                and (counters[guard:-guard] == 0).all()
            ):
                raise AssertionError("scratch guard or completion counters damaged")
        report["status"] = "OPERATOR_SAMPLE_CHECKED"
        report["negative_control_detected"] = True
        save()
    except Exception:
        report["status"] = "FAILED"
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=320)
    parser.add_argument("--variants", help="explicit variant subset including control")
    parser.add_argument(
        "--shapes", default="gate_up,down,attention_in,attention_out,gdn_in,gdn_out"
    )
    run(parser.parse_args())

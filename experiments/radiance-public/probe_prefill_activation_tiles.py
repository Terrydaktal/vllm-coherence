"""Native tiled-prefill experiment: exact bytes, GEMM output, graphs and latency.

Uses public checkpoint weights and synthetic activations. No serving hooks or
private conversation input. A failed or slower experiment is not deployable.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    for k, v in {
        "RADIANCE_MXFP4": "1",
        "RADIANCE_MXFP4_W4A8": "1",
        "RADIANCE_MXFP4_WPERM": "1",
        "RADIANCE_MXFP4_DECODE_MAX_M": "64",
        "RADIANCE_MXFP4_DECODE_NT": "1",
        "RADIANCE_MXFP4_TN4_MIN_M": "2048",
        "RADIANCE_MXFP4_EPIFAST": "1",
        "RADIANCE_MXFP4_W4A8_MIN_M": "0",
    }.items():
        os.environ[k] = v
    import radiance_mxfp4 as kernel
    import torch
    from prefill_activation_tiles import install_consumer, pack
    from prefill_tiles_admission import ROWS, SHAPES
    from probe_mxfp4_dispatch import load_extension
    from safetensors import safe_open
    from vllm import _custom_ops as ops

    torch.set_num_threads(4)
    torch.manual_seed(20260918)
    build = json.loads((args.build / "build.json").read_text())
    binary = args.build / "candidate/radiance_mxfp4_fp8.so"
    if digest(binary) != build["variants"]["candidate"]["binary_sha256"]:
        raise ValueError("candidate binary changed")
    ext = load_extension(binary, "prefill_tiles")
    kernel._ext = ext
    calls = install_consumer(kernel, SHAPES, ROWS)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    report = {
        "status": "RUNNING",
        "binary_sha256": digest(binary),
        "probe_sha256": digest(__file__),
        "pack_sha256": digest(Path(__file__).with_name("prefill_activation_tiles.py")),
        "private_chat_read": False,
        "cases": [],
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def weight(name):
        with safe_open(args.model / index[name], framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    def timing(fn):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(8):
                fn()
        values = []
        for _ in range(7):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end) / 8)
        return statistics.median(values)

    started = time.monotonic()
    try:
        # Attention, GDN and both MLP shapes, from actual layer-zero weights.
        for parts in (
            ("mlp.gate_proj", "mlp.up_proj"),
            ("mlp.down_proj",),
            ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"),
            ("linear_attn.out_proj",),
        ):
            prefix = "model.language_model.layers.0."
            names = [prefix + part for part in parts]
            # Qwen checkpoints can store split qkv/z projections. Report absence
            # explicitly rather than silently claiming a shape was checked.
            if any(name + ".weight" not in index for name in names):
                report.setdefault("absent_checkpoint_modules", []).append(list(parts))
                continue
            raw = torch.cat([weight(n + ".weight") for n in names]).cuda()
            scale = torch.cat([weight(n + ".weight_scale") for n in names]).T.contiguous().cuda()
            n, k = raw.shape[0], raw.shape[1] * 2
            packed = kernel.permute_w(raw, n, k)
            ref = kernel.make_row_ref(scale)
            for m in (1000, 1648, 2048):
                x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
                q, s = ops.scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=True)
                s = s.flatten().contiguous()
                at = pack(q)
                padded = torch.zeros(((m + 15) // 16 * 16, k), dtype=torch.uint8, device="cuda")
                padded[:m].copy_(q.view(torch.uint8))
                independent = padded.view(-1, 16, k // 16, 2, 8).permute(0, 2, 3, 1, 4).contiguous()
                layout_equal = torch.equal(at, independent.flatten())
                slabs = [
                    torch.full((m * n + 1024,), 42, dtype=torch.bfloat16, device="cuda")
                    for _ in range(2)
                ]
                a, b = [t[512:-512].view(m, n) for t in slabs]

                def launch(
                    tiled,
                    q=q,
                    at=at,
                    packed=packed,
                    scale=scale,
                    ref=ref,
                    s=s,
                    a=a,
                    b=b,
                    m=m,
                    n=n,
                    k=k,
                ):
                    if tiled:
                        pack(q, at)
                    (ext.launch_at if tiled else ext.launch)(
                        (at if tiled else q).data_ptr(),
                        packed.data_ptr(),
                        scale.data_ptr(),
                        ref.data_ptr(),
                        s.data_ptr(),
                        (b if tiled else a).data_ptr(),
                        m,
                        n,
                        k,
                        torch.cuda.current_stream().cuda_stream,
                    )

                launch(False)
                launch(True)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    launch(True)
                b.fill_(float("nan"))
                at.fill_(255)
                graph.replay()
                torch.cuda.synchronize()
                adapter = kernel.mxfp4_linear_pq(q, s, packed, scale, ref)
                adapter_exact = torch.equal(adapter.view(torch.int16), a.view(torch.int16))
                mismatch = int((a.view(torch.int16) != b.view(torch.int16)).sum())
                row_equal = int((a.view(torch.int16) == b.view(torch.int16)).all(1).sum())
                case = {
                    "modules": list(parts),
                    "M": m,
                    "N": n,
                    "K": k,
                    "layout_bytes_equal": layout_equal,
                    "unequal_elements": mismatch,
                    "equal_rows": row_equal,
                    "total_elements": a.numel(),
                    "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                    "canaries": all(
                        bool((t[:512] == 42).all() and (t[-512:] == 42).all()) for t in slabs
                    ),
                    "adapter_exact": adapter_exact,
                    "ordinary_ms": timing(lambda: launch(False)),
                    "tiled_including_reorder_ms": timing(lambda: launch(True)),
                }
                report["cases"].append(case)
                print(json.dumps(case), flush=True)
                save()
                if not (
                    layout_equal
                    and mismatch == 0
                    and case["finite"]
                    and case["canaries"]
                    and adapter_exact
                ):
                    raise AssertionError("tiled prefill is not exact")
                # Same comparator must detect a one-bit defect.
                b.view(torch.int16).flatten()[0] ^= 1
                if torch.equal(a.view(torch.int16), b.view(torch.int16)):
                    raise AssertionError("negative control missed")
            del raw, scale, packed, ref
        if not report["cases"]:
            raise AssertionError("empty coverage")
        report["negative_control_detected"] = True
        report["consumer_calls"] = calls
        report["status"] = "SAMPLE_CHECKED"
    except Exception:
        report["status"] = "FAILED"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())

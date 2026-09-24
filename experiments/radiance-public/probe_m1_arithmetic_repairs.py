"""Native operator regression and graph timing for the independent M1 repairs.

Public checkpoint weights and seeded synthetic inputs only. Does not install
anything in the server. This is not full-model or snapshot qualification.
"""

import argparse
import fcntl
import gc
import json
import os
import statistics
import time
from functools import partial
from pathlib import Path

from mxfp4_fold_precision import make_row_ref
from patch_gdn_stable_softplus import patched_source
from probe_eager_m1_independent import (
    checkpoint_reader,
    compare,
    decode_mxfp4_rows,
    device_memory,
    digest,
    idle,
    load_module,
)


def graph_us(torch, call, api):
    idle(api)
    for _ in range(8):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(16):
            call()
    samples = []
    for _ in range(7):
        idle(api)
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(32):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / (16 * 32))
    return {
        "median_us": statistics.median(samples),
        "samples_us": samples,
        "invocations_per_sample": 512,
        "mode": "uninstrumented kernels in HIP graph; device events around replay batch",
    }


def gemm(args, torch, tensor):
    import numpy as np

    import radiance_mxfp4 as wrapper

    build = json.loads((args.build / "build.json").read_text())
    extensions = {}
    for name, entry in build["variants"].items():
        path = args.build / name / "radiance_mxfp4_fp8.so"
        if digest(path) != entry["binary_sha256"]:
            raise ValueError("GEMM binary changed")
        extensions[name] = load_module(path, name + ".radiance_mxfp4_fp8")
    # The public fused qkv shape uses KS=1, so no global scratch is required.
    for module in extensions.values():
        module.set_decode_scratch(0, 0, 0)
    base = "model.language_model.layers.15.self_attn."
    weights = torch.cat(
        [tensor(base + p + ".weight") for p in ("q_proj", "k_proj", "v_proj")]
    )
    scales = torch.cat(
        [tensor(base + p + ".weight_scale") for p in ("q_proj", "k_proj", "v_proj")]
    )
    n, k = weights.shape[0], weights.shape[1] * 2
    assert (n, k) == (14336, 5120)
    selected = [9544, 10568, 10711, 12048, 0, 9543]
    exact = decode_mxfp4_rows(torch, weights[selected], scales[selected])
    w = wrapper.permute_w(weights, n, k).cuda()
    ws = scales.T.contiguous().cuda()
    refs = {
        "control": scales.amax(1).cuda(),
        "candidate": make_row_ref(scales.T).cuda(),
    }
    assert refs["candidate"].numel() == n
    record = {
        "build_sha256": digest(args.build / "build.json"),
        "changed_reference_rows": int((refs["control"] != refs["candidate"]).sum()),
        "checks": [],
    }

    def launch(
        module,
        x,
        scale,
        out,
        ref,
        m,
        tiled=False,
        weight=w,
        weight_scale=ws,
        nn=n,
        kk=k,
    ):
        fn = module.launch_at if tiled else module.launch
        fn(
            x.data_ptr(),
            weight.data_ptr(),
            weight_scale.data_ptr(),
            ref.data_ptr() if ref is not None else 0,
            scale.data_ptr(),
            out.data_ptr(),
            m,
            nn,
            kk,
            torch.cuda.current_stream().cuda_stream,
        )

    columns = sorted(
        {
            v["block"] * 32 + j
            for v in args.samples["fold_blocks"]["blocks"]
            for j in range(32)
        }
    )
    acts = torch.zeros((len(columns), k))
    acts[torch.arange(len(columns)), columns] = 1
    gen = torch.Generator().manual_seed(20260924)
    for label, source in (
        ("one_hot", acts),
        ("normal320", torch.randn((320, k), generator=gen).bfloat16().float()),
    ):
        scale = (source.abs().amax(-1) / 448).clamp_min(1 / (448 * 512))
        q = (source / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        expected = torch.from_numpy(
            np.asarray((q.double() * scale.double()[:, None]).numpy() @ exact.numpy().T)
        ).bfloat16()
        x, xs = q.cuda(), scale.cuda()
        outputs = {}
        for name, module in extensions.items():
            output = torch.empty((len(source), n), dtype=torch.bfloat16, device="cuda")
            for i in range(len(source)):
                if i % 32 == 0:
                    idle(args.api)
                launch(module, x[i:], xs[i:], output[i:], refs[name], 1)
            outputs[name] = output.cpu()
            del output
        row = {"case": label, "rows": len(source)}
        for name in outputs:
            row[name] = compare(torch, outputs[name][:, selected], expected)
            row[name + "_gates"] = compare(
                torch,
                outputs[name][:, selected[:4]].sigmoid(),
                expected[:, :4].sigmoid(),
            )
        assert row["control"]["different"] > 0, (
            "negative control did not detect original weight loss"
        )
        if label == "one_hot":
            assert row["candidate"]["different"] == 0
        unaffected = torch.ones(n, dtype=torch.bool)
        unaffected[selected[:4]] = False
        row["unaffected_vs_control"] = compare(
            torch,
            outputs["candidate"][:, unaffected],
            outputs["control"][:, unaffected],
        )
        assert row["unaffected_vs_control"]["different"] == 0
        if label == "normal320":
            for width in (8, 32, 65):
                idle(args.api)
                output = torch.empty(
                    (len(source), n), dtype=torch.bfloat16, device="cuda"
                )
                for i in range(0, len(source), width):
                    launch(
                        extensions["candidate"],
                        x[i:],
                        xs[i:],
                        output[i:],
                        refs["candidate"],
                        min(width, len(source) - i),
                    )
                row[f"M{width}_vs_M1"] = compare(
                    torch, output.cpu(), outputs["candidate"]
                )
                assert row[f"M{width}_vs_M1"]["different"] == 0
                del output
            # Prefill's fragment-tiled activation layout uses padded 16-row tiles.
            tiled = x.reshape(20, 16, k // 16, 2, 8).permute(0, 2, 3, 1, 4).contiguous()
            output = torch.empty((320, n), dtype=torch.bfloat16, device="cuda")
            launch(
                extensions["candidate"], tiled, xs, output, refs["candidate"], 320, True
            )
            row["tiled_prefill_vs_M1"] = compare(
                torch, output.cpu(), outputs["candidate"]
            )
            assert row["tiled_prefill_vs_M1"]["different"] == 0
            record["timing"] = {}
            for width in (1, 8):
                record["timing"][str(width)] = {}
                for name in ("control", "candidate", "control"):
                    entry = graph_us(
                        torch,
                        partial(
                            launch, extensions[name], x, xs, output, refs[name], width
                        ),
                        args.api,
                    )
                    record["timing"][str(width)].setdefault(name, []).append(entry)
            del tiled, output
        record["checks"].append(row)
        del x, xs, outputs
    del w, ws, weights, scales, refs
    gc.collect()
    torch.cuda.empty_cache()
    # Independent one-hot oracle checks the fallback, all layouts, and partial M.
    n, k = 48, 128
    codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=gen)
    scales = torch.randint(110, 141, (n, k // 32), dtype=torch.uint8, generator=gen)
    refs = make_row_ref(scales.T)
    assert refs.numel() == 2
    expected = decode_mxfp4_rows(torch, codes, scales).T.bfloat16()
    packed = wrapper.permute_w(codes, n, k).cuda()
    ws = scales.T.contiguous().cuda()
    x = torch.eye(k).to(torch.float8_e4m3fn).cuda()
    xs = torch.ones(k, device="cuda")
    out = torch.empty((k, n), dtype=torch.bfloat16, device="cuda")
    # This process is WPERM=1, matching the actual loader's permuted fallback.
    record["fallback"] = {}
    for m in (1, 8, 65, 128):
        for i in range(0, k, m):
            launch(
                extensions["candidate"],
                x[i:],
                xs[i:],
                out[i:],
                None,
                min(m, k - i),
                weight=packed,
                weight_scale=ws,
                nn=n,
                kk=k,
            )
        row = compare(torch, out, expected)
        assert row["different"] == 0
        record["fallback"][f"M{m}"] = row
    tiled = x.reshape(8, 16, 8, 2, 8).permute(0, 2, 3, 1, 4).contiguous()
    launch(
        extensions["candidate"],
        tiled,
        xs,
        out,
        None,
        128,
        True,
        weight=packed,
        weight_scale=ws,
        nn=n,
        kk=k,
    )
    record["fallback"]["tiled"] = compare(torch, out, expected)
    assert record["fallback"]["tiled"]["different"] == 0
    # Exercise the entire expanded folding window on hardware, including d=-6.
    for gap in (0, 8, 9, 14):
        synthetic_scales = torch.full((n, k // 32), 110, dtype=torch.uint8)
        synthetic_scales[:, 1::2] += gap
        reference = make_row_ref(synthetic_scales.T).cuda()
        local_scales = synthetic_scales.T.contiguous().cuda()
        expected = decode_mxfp4_rows(torch, codes, synthetic_scales).T.bfloat16()
        launch(
            extensions["candidate"],
            x,
            xs,
            out,
            reference,
            128,
            weight=packed,
            weight_scale=local_scales,
            nn=n,
            kk=k,
        )
        row = compare(torch, out, expected)
        assert row["different"] == 0
        record.setdefault("fold_window", {})[str(gap)] = row
    return record


def gdn(args, torch, tensor):
    import vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv as prep_old
    import vllm.third_party.flash_linear_attention.ops.fused_recurrent as old
    from stock_gdn_scan_kernel import stock_gdn_scan_kernel as scan

    def patched(module):
        src = Path(module.__file__)
        path = args.output / ("fixed_" + src.name)
        path.write_text(patched_source(src.name, src.read_text()))
        return load_module(
            path, "vllm.third_party.flash_linear_attention.ops.fixed_" + src.stem
        )

    fixed, prep = patched(old), patched(prep_old)
    base = "model.language_model.layers.12.linear_attn."
    al = tensor(base + "A_log").float().cuda()
    bias = tensor(base + "dt_bias").bfloat16().cuda()
    gen = torch.Generator().manual_seed(20260924)
    total_rows = 2048
    mixed = torch.randn((total_rows, 10240), generator=gen).bfloat16().cuda()
    a = (torch.rand((total_rows, 48), generator=gen) * 32 - 24).bfloat16().cuda()
    b = torch.randn((total_rows, 48), generator=gen).bfloat16().cuda()
    state = torch.zeros((2, 48, 128, 128), device="cuda")
    state[1].copy_(torch.randn((48, 128, 128), generator=gen) * 0.05)
    output = torch.empty((1, 1, 48, 128), dtype=torch.bfloat16, device="cuda")
    indices = torch.tensor([1], dtype=torch.int32, device="cuda")
    sequence_out = torch.empty((8, 48, 128), dtype=torch.bfloat16, device="cuda")
    states = torch.empty((8, 48, 128, 128), device="cuda")

    def serial(module, i=0):
        module.fused_recurrent_gated_delta_rule_packed_decode(
            mixed[i : i + 1],
            a[i : i + 1],
            b[i : i + 1],
            al,
            bias,
            128**-0.5,
            state,
            output,
            indices,
            True,
        )

    def run_scan(start, count, initial, out, history, save, tile=32, kernel=scan):
        kernel[(128 // tile, 48)](
            mixed[start:],
            a[start:],
            b[start:],
            al,
            bias,
            initial,
            out,
            history,
            None,
            None,
            None,
            count,
            128**-0.5,
            10240,
            48,
            48,
            H=16,
            HV=48,
            K=128,
            V=128,
            BK=128,
            BV=tile,
            SAVE_ROWS=save,
            INDEXED=False,
            stride_state=0,
            stride_index_seq=0,
            stride_index_row=0,
            num_warps=1,
            num_stages=3,
            enable_fp_fusion=True,
            allow_flush_denorm=False,
        )

    rows = 0
    first_state = state[1].clone()
    serial_outputs = torch.empty(
        (total_rows, 48, 128), dtype=torch.bfloat16, device="cuda"
    )
    prefix_lengths = (1, 8, 64, 320, 1000, 1648, 2048)
    expected_states = {}
    for start in range(0, 320, 8):
        idle(args.api)
        initial = state[1].clone()
        run_scan(start, 8, initial, sequence_out, states, True)
        for j in range(8):
            serial(fixed, start + j)
            serial_outputs[start + j].copy_(output[0, 0])
            if not torch.equal(state[1], states[j]) or not torch.equal(
                output[0, 0], sequence_out[j]
            ):
                raise AssertionError(
                    f"GDN M1/M8 state or output mismatch at row {start + j}"
                )
            rows += 1
            if rows in prefix_lengths:
                expected_states[rows] = state[1].clone()
    for i in range(320, total_rows):
        if i % 32 == 0:
            idle(args.api)
        serial(fixed, i)
        serial_outputs[i].copy_(output[0, 0])
        if i + 1 in prefix_lengths:
            expected_states[i + 1] = state[1].clone()
    # Prefill final-state publication must match the retained-state scan too.
    prefill_out = torch.empty_like(serial_outputs)
    final = torch.empty_like(state[1])
    prefill_checks = []
    for count in prefix_lengths:
        for tile in (8, 16, 32, "selected"):
            idle(args.api)
            size = (
                (8 if count >= 2048 else (16 if count > 8 else 32))
                if tile == "selected"
                else tile
            )
            run_scan(0, count, first_state, prefill_out, final, False, size)
            exact = torch.equal(final, expected_states[count]) and torch.equal(
                prefill_out[:count], serial_outputs[:count]
            )
            prefill_checks.append({"tokens": count, "tile": tile, "exact": exact})
            assert exact, f"prefill {count} rows / tile {tile} differs from serial M1"
    cpu_a = a.cpu().double()
    cpu_bias = bias.cpu().double()
    cpu_al = al.cpu().double()
    expected = -cpu_al.exp() * torch.nn.functional.softplus(cpu_a + cpu_bias)
    actual = prep.fused_post_conv_prep(mixed, a, b, al, bias, 16, 128, 128)[3].cpu()
    torch.testing.assert_close(actual, expected.float(), rtol=2e-6, atol=0)
    record = {
        "matched_state_and_output_rows": rows,
        "state_elements_compared": rows * 48 * 128 * 128,
        "prefill_final_state_equal": True,
        "prefill_rows": total_rows,
        "prefill_spatial_tiles": [8, 16, 32],
        "prefill_checks": prefill_checks,
        "prefill_gate_max_relative_error": float(
            ((actual.double() - expected).abs() / expected.abs()).max()
        ),
        "native_source_sha256": digest(Path(fixed.__file__)),
        "scan_sha256": digest(Path(__file__).with_name("stock_gdn_scan_kernel.py")),
    }
    saved_mixed, saved_a = mixed[0].clone(), a[0].clone()
    mixed[0].zero_()
    mixed[0, torch.arange(16, device="cuda") * 128] = 1
    a[0].copy_((-18 - bias.float()).bfloat16())
    witness = {}
    for name, module in (("old", old), ("fixed", fixed)):
        state[1].zero_()
        state[1, :, :, 0] = 1
        serial(module)
        witness[name] = float(state[1, 29, 0, 0])
    x = float(a[0, 29]) + float(bias[29])
    target = torch.tensor(-float(al[29]), dtype=torch.float64)
    witness["expected"] = float(
        (
            -(-target).exp()
            * torch.nn.functional.softplus(torch.tensor(x, dtype=torch.float64))
        )
        .exp()
        .float()
    )
    assert witness["old"] == 1 and witness["fixed"] == witness["expected"] < 1
    record["cancellation_witness"] = witness
    record["negative_control_detected"] = True
    mixed[0].copy_(saved_mixed)
    a[0].copy_(saved_a)
    state[1].copy_(first_state)
    record["timing"] = {
        name: graph_us(torch, partial(serial, module), args.api)
        for name, module in [("old", old), ("fixed", fixed)]
    }
    old_scan_file = args.output / "old_stock_gdn_scan_kernel.py"
    old_scan_file.write_text(
        Path(__file__)
        .with_name("stock_gdn_scan_kernel.py")
        .read_text()
        .replace(
            "        # Keep tiny nonzero gates: adding exp(x) to 1 first cancels them in FP32.\n",
            "",
        )
        .replace("tl.extra.libdevice.log1p(tl.exp(x))", "tl.log(1.0 + tl.exp(x))")
    )
    old_hash = json.loads((args.audit / "sources.json").read_text())[
        "coherence/stock_gdn_scan_kernel.py"
    ]
    assert digest(old_scan_file) == old_hash
    old_scan = load_module(old_scan_file, "old_gdn_scan_timing").stock_gdn_scan_kernel
    record["M8_timing"] = {
        name: graph_us(
            torch,
            partial(
                run_scan, 0, 8, first_state, sequence_out, states, True, kernel=kernel
            ),
            args.api,
        )
        for name, kernel in (("old", old_scan), ("fixed", scan))
    }
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "build", "audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--api", default="http://127.0.0.1:8080")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument(
        "--stages", nargs="+", choices=("gemm", "gdn"), default=["gemm", "gdn"]
    )
    args = parser.parse_args()
    if not args.allow_gpu or not os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"):
        raise RuntimeError("explicit GPU admission and shared lease required")
    with open(os.environ["QWEN_CONFORMANCE_GPU_LOCK"], "a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        idle(args.api)
        total, free = device_memory()
        if free < 512 * 1024**2:
            raise RuntimeError(
                "less than 512 MiB free; no GPU initialization attempted"
            )
        args.output.mkdir(mode=0o700)
        os.environ["TRITON_CACHE_DIR"] = str(args.output / "triton")
        os.environ.update(
            RADIANCE_MXFP4_W4A8="1",
            RADIANCE_MXFP4_W4A8_MIN_M="0",
            RADIANCE_MXFP4_WPERM="1",
            RADIANCE_MXFP4_DECODE_MAX_M="64",
            RADIANCE_MXFP4_DECODE_NT="1",
            PYTORCH_ALLOC_CONF="expandable_segments:False",
        )
        import torch

        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        torch.cuda.set_per_process_memory_fraction(224 * 1024**2 / total)
        args.samples = json.loads((args.audit / "checkpoint-samples.json").read_text())
        tensor = checkpoint_reader(args.model, args.samples["small_tensors"])
        report = {
            "status": "RUNNING",
            "probe_sha256": digest(__file__),
            "source_sha256": {
                p: digest(Path(__file__).with_name(p))
                for p in (
                    "mxfp4_fold_precision.py",
                    "patch_gdn_stable_softplus.py",
                    "stock_gdn_scan_kernel.py",
                )
            },
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "checks": {},
        }
        start = time.monotonic()
        try:
            for name, fn in [("gemm", gemm), ("gdn", gdn)]:
                if name not in args.stages:
                    continue
                print(json.dumps({"stage": name, "status": "starting"}), flush=True)
                report["checks"][name] = fn(args, torch, tensor)
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                (args.output / "result.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
                print(json.dumps({"stage": name, "status": "checked"}), flush=True)
            report["status"] = "OPERATOR_SAMPLE_CHECKED_NOT_FULL_MODEL_QUALIFIED"
        except BaseException as exc:
            report["status"] = "FAILED"
            report["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            report["elapsed_seconds"] = time.monotonic() - start
            report["peak_allocator_MiB"] = torch.cuda.max_memory_allocated() / 1024**2
            (args.output / "result.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()

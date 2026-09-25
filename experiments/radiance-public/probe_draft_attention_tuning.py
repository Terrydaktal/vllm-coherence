"""Compare exact drafter attention outputs before tuning launch scheduling.

Keep the query/KV tiles and arithmetic fixed; vary warp count, pipeline depth
and occupancy hints only. This does not alter draft context or target sampling.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def partition_query_rows(source, rows):
    """Launch more independent CTAs while keeping each row's original KV loop.

    Padding stays at BLOCK_M=16 for the matrix instruction. Mask unused query
    rows to avoid overlapping writes. Keep the original four-query group's
    loop bounds and V mask so no zero-only tiles or rounding points change.
    """
    if rows not in (1, 2):
        raise ValueError("unsupported query partition")

    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError("drafter query partition anchor changed: " + old[:65])
        source = source.replace(old, new)

    replace(
        "\n    BLOCK_Q = BLOCK_M // num_queries_per_kv\n",
        "\n    BLOCK_Q = BLOCK_M // num_queries_per_kv\n"
        "    if head_size == 128 and max_seqlen_q == 8 and num_queries_per_kv == 4 "
        "and q.shape[0] == 8:\n"
        f"        BLOCK_Q = {rows}\n",
    )
    replace(
        "    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)",
        "    query_mask_0 = ((query_pos < cur_batch_query_len)\n"
        "                    & (offs_m // num_queries_per_kv < BLOCK_Q))",
    )
    start = source.index(
        "    loop_lo, loop_hi, max_seq_prefix_len = compute_tile_loop_bounds("
    )
    end = source.index("\n    )", start) + len("\n    )")
    original = source[start:end]
    changed = original.replace(
        "        q_block_local_idx,",
        "        (q_block_local_idx * BLOCK_Q) // (BLOCK_M // num_queries_per_kv),",
    ).replace("        BLOCK_Q,", "        BLOCK_M // num_queries_per_kv,")
    if changed == original:
        raise ValueError("drafter tile bounds changed")
    source = source[:start] + changed + source[end:]
    replace(
        "            qpos_lo = q_block_local_idx * BLOCK_Q",
        "            qpos_lo = ((q_block_local_idx * BLOCK_Q)\n"
        "                        // (BLOCK_M // num_queries_per_kv))\n"
        "            qpos_lo *= BLOCK_M // num_queries_per_kv",
    )
    return source


def specialize_unit_scales(source):
    """Avoid multiplying by one, with the original path for every other scale."""
    old = "        return (data.to(tl.float32) * tl.load(tensor_scale)).to(Q.dtype)"
    new = """        scale = tl.load(tensor_scale)
        if scale == 1.0:
            result = data.to(Q.dtype)
        else:
            result = (data.to(tl.float32) * scale).to(Q.dtype)
        return result"""
    if source.count(old) != 1:
        raise ValueError("KV dequantization anchor changed")
    return source.replace(old, new)


def main(args):
    import torch
    import vllm.v1.attention.ops.triton_unified_attention as control

    torch.set_num_threads(4)
    torch.manual_seed(20260925)
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    source_path = Path(control.__file__)
    source = source_path.read_text()
    anchor = "    launch_kwargs: dict[str, int] = {}"
    if source.count(anchor) != 1:
        raise ValueError("drafter attention launch source changed")
    modules = {"control": control}
    variants = {
        "w4_s1": {"num_warps": 4, "num_stages": 1},
        "w4_s2": {"num_warps": 4, "num_stages": 2},
        "w8_s1": {"num_warps": 8, "num_stages": 1},
        "w8_s2": {"num_warps": 8, "num_stages": 2},
        "w4_s1_occ1": {"num_warps": 4, "num_stages": 1, "waves_per_eu": 1},
        "w4_s1_occ2": {"num_warps": 4, "num_stages": 1, "waves_per_eu": 2},
    }
    if args.query_partition:
        variants = {
            "w4_s1_occ2": {"num_warps": 4, "num_stages": 1, "waves_per_eu": 2},
            "q1_w4_s1": {"num_warps": 4, "num_stages": 1},
            "q1_w4_s1_occ2": {"num_warps": 4, "num_stages": 1, "waves_per_eu": 2},
            "q2_w4_s1": {"num_warps": 4, "num_stages": 1},
            "q2_w4_s1_occ2": {"num_warps": 4, "num_stages": 1, "waves_per_eu": 2},
        }
    if args.unit_scales:
        variants = {
            "w4_s1": {"num_warps": 4, "num_stages": 1},
            "w4_s1_occ2": {"num_warps": 4, "num_stages": 1, "waves_per_eu": 2},
            "w4_s1_occ4": {"num_warps": 4, "num_stages": 1, "waves_per_eu": 4},
            "w2_s1": {"num_warps": 2, "num_stages": 1},
            "unit_w4_s1": {"num_warps": 4, "num_stages": 1},
            "unit_w4_s1_occ2": {"num_warps": 4, "num_stages": 1, "waves_per_eu": 2},
        }
    for name, kwargs in variants.items():
        path = args.output / (name + ".py")
        candidate = source.replace(
            anchor, f"    launch_kwargs: dict[str, int] = {kwargs!r}"
        )
        if name.startswith(("q1_", "q2_")):
            candidate = partition_query_rows(candidate, int(name[1]))
        if name.startswith("unit_"):
            candidate = specialize_unit_scales(candidate)
        path.write_text(candidate)
        spec = importlib.util.spec_from_file_location(
            "vllm.v1.attention.ops." + name, path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    block_size = 1648 if args.layout == "hnd1648" else 16
    blocks = (200128 + block_size - 1) // block_size
    # Shape matches the deployed five-layer DFlash2: 32 Q / 8 KV heads,
    # head width 128, FP8 KV and 2048-token sliding attention.
    shape = (
        (blocks, 8, block_size, 256)
        if args.layout == "hnd1648"
        else (2, blocks, block_size, 8, 128)
    )
    kv = torch.randint(32, 82, shape, device="cuda", dtype=torch.uint8)
    signs = torch.randint(0, 2, kv.shape, device="cuda", dtype=torch.uint8)
    kv.bitwise_or_(signs * 128)
    del signs
    if args.layout == "hnd1648":
        view = kv.view(torch.float8_e4m3fn).permute(0, 2, 1, 3)
        k, v = view[..., :128], view[..., 128:]
    else:
        k, v = kv.view(torch.float8_e4m3fn).unbind(0)
    query = torch.randn((8, 32, 128), device="cuda", dtype=torch.bfloat16)
    lengths = torch.tensor([60008], dtype=torch.int32, device="cuda")
    table = torch.randperm(blocks, device="cuda", dtype=torch.int32).reshape(1, -1)
    # The full model visits many GB of weights between these attention calls.
    # Rotate physical pages so the microbenchmark does not repeatedly time one
    # small sliding window already resident in the last-level GPU cache.
    graph_calls = 32 if args.rotate_pages else 5
    page_step = (2048 + block_size - 1) // block_size + 1
    timing_tables = [
        table.roll(i * page_step, dims=1) if args.rotate_pages else table
        for i in range(graph_calls)
    ]
    cumulative = torch.tensor([0, 8], dtype=torch.int32, device="cuda")
    scale_base = torch.ones((), device="cuda")
    scale = scale_base.expand(1, 8)
    outputs = {name: torch.empty_like(query) for name in modules}
    report = {
        "status": "RUNNING",
        "source_sha256": digest(source_path),
        "probe_sha256": digest(__file__),
        "variants": variants,
        "cases": {},
        "private_chat_read": False,
        "layout": args.layout,
        "kv_shape": list(kv.shape),
        "key_strides": list(k.stride()),
        "rotate_pages": args.rotate_pages,
        "timing_graph_calls": graph_calls,
        "kv_quant_mode": args.kv_quant_mode,
        "query_partition": args.query_partition,
        "unit_scale_specialization": args.unit_scales,
    }

    def save():
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    for context in (2048, 60000, 200000):
        for causal in (True, False):
            label = f"{context}_{'causal' if causal else 'noncausal'}"

            def launch(name, context=context, causal=causal, block_table=None):
                modules[name].unified_attention(
                    q=query,
                    k=k,
                    v=v,
                    out=outputs[name],
                    cu_seqlens_q=cumulative,
                    max_seqlen_q=8,
                    seqused_k=lengths,
                    max_seqlen_k=context + 64,
                    softmax_scale=128**-0.5,
                    causal=causal,
                    window_size=(2047, 0),
                    block_table=table if block_table is None else block_table,
                    softcap=0.0,
                    q_descale=None,
                    k_descale=scale,
                    v_descale=scale,
                    kv_quant_mode=args.kv_quant_mode,
                )

            checks = {name: {"different": 0, "elements": 0} for name in modules}
            for sample in range(args.samples):
                query.normal_(std=(0.25, 1.0, 4.0, 12.0)[sample % 4])
                scale_base.fill_((1.0, 0.5, 1.25, 0.003)[sample % 4])
                lengths.fill_(context + 8 + sample)
                for name in modules:
                    launch(name)
                torch.cuda.synchronize()
                for name in modules:
                    checks[name]["different"] += int(
                        (
                            outputs[name].view(torch.int16)
                            != outputs["control"].view(torch.int16)
                        ).sum()
                    )
                    checks[name]["elements"] += query.numel()
                    if not bool(torch.isfinite(outputs[name]).all()):
                        raise AssertionError("drafter attention output is non-finite")
            eligible = [
                name for name, check in checks.items() if check["different"] == 0
            ]
            corrupted = outputs["control"].clone()
            corrupted.view(torch.int16).reshape(-1)[0].bitwise_xor_(1)
            if torch.equal(
                corrupted.view(torch.int16), outputs["control"].view(torch.int16)
            ):
                raise AssertionError(
                    "bitwise checker failed its injected-fault control"
                )
            query.normal_()
            scale_base.fill_(1.0)
            lengths.fill_(context + 8)
            graphs = {}
            for name in eligible:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for timing_table in timing_tables:
                        launch(name, block_table=timing_table)
                graphs[name] = graph
            for _ in range(8):
                for graph in graphs.values():
                    graph.replay()
            samples = {name: [] for name in eligible}
            for trial in range(7):
                for name in eligible if trial % 2 == 0 else reversed(eligible):
                    start, end = (
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    )
                    start.record()
                    for _ in range(32):
                        graphs[name].replay()
                    end.record()
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end) / (graph_calls * 32))
            report["cases"][label] = {
                "positions": args.samples * 8,
                "checks": checks,
                "negative_control_detected": True,
                "median_ms_per_layer": {
                    name: statistics.median(values) for name, values in samples.items()
                },
                "samples_ms": samples,
            }
            save()
            print(
                json.dumps(
                    {
                        "case": label,
                        "eligible": eligible,
                        "timings": report["cases"][label]["median_ms_per_layer"],
                    }
                ),
                flush=True,
            )
    report["status"] = "SAMPLE_CHECKED_VARIANTS_RECORDED"
    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--layout", choices=("nhd16", "hnd1648"), default="hnd1648")
    parser.add_argument("--rotate-pages", action="store_true")
    parser.add_argument("--kv-quant-mode", type=int, choices=(0, 1), default=1)
    parser.add_argument("--query-partition", action="store_true")
    parser.add_argument("--unit-scales", action="store_true")
    main(parser.parse_args())

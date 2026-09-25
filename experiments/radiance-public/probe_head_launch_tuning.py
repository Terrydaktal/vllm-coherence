"""Tune INT2-head execution without changing its quantization or candidate rule.

Compare every coarse-logit byte and every emitted candidate/score against the
installed release on 320 input rows. Kernel timing uses alternating graph
replays. Neither live model sessions nor their private contents are accessed.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path
from types import SimpleNamespace


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_candidate_selection(source, mode):
    start = source.index(
        '        masked = tl.where(mask_n[None, :], acc, float("-inf"))'
    )
    end = source.index("\n    @triton.jit", start)
    original = source[start:end]
    # Preserve the original float-reduction behavior for NaNs and negative
    # zero. Ordinary finite scores have unique sortable (score, -index) keys.
    prefix = """        if KC > 0:
            masked = tl.where(mask_n[None, :], acc, float("-inf"))
            bits = masked.to(tl.uint32, bitcast=True)
            special = (masked != masked) | (bits == 0x80000000)
            exceptional = tl.sum(tl.sum(special.to(tl.int32), axis=1), axis=0) > 0
            if exceptional:
"""
    fallback = "\n".join("        " + line for line in original.rstrip().splitlines())
    if mode == "bitonic":
        normal = """                ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
                columns = tl.arange(0, BLOCK_N)
                keys = (ordered.to(tl.uint64) << 6) | (63 - columns[None, :]).to(tl.uint64)
                selected = tl.topk(keys, KC)
                selected_indices = (63 - (selected & 63)).to(tl.int32)
                selected_scores = tl.gather(masked, selected_indices, axis=1)
                offsets = offs_m[:, None] * (NBLK * KC) + pid * KC + tl.arange(0, KC)[None, :]
                tl.store(BM + offsets, selected_scores)
                tl.store(BI + offsets, (pid * BLOCK_N + selected_indices).to(tl.int32))
"""
    elif mode == "paired_reduce":
        normal = (
            "\n".join(
                "        " + line
                for line in original.rstrip()
                .replace(
                    "            mx = tl.max(masked, axis=1)\n            am = tl.argmax(masked, axis=1)",
                    "            mx, am = tl.max(masked, axis=1, return_indices=True)",
                )
                .splitlines()
            )
            + "\n"
        )
    else:
        raise ValueError("unknown candidate selector")
    return (
        source[:start]
        + prefix
        + fallback
        + "\n            else:\n"
        + normal
        + source[end:]
    )


def main(args):
    import torch
    from safetensors import safe_open
    from triton.compiler.errors import CompilationError
    from triton.runtime.errors import OutOfResources

    import radiance_drafthead as original

    torch.set_num_threads(4)
    torch.manual_seed(20260925)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    key = "lm_head.weight"
    if key not in index:
        keys = [k for k in index if k.endswith("lm_head.weight")]
        if len(keys) != 1:
            raise ValueError("expected exactly one explicit head weight")
        key = keys[0]
    with safe_open(args.model / index[key], framework="pt", device="cpu") as file:
        weight = file.get_tensor(key).cuda()
    state = SimpleNamespace()
    original._quantize_head_now(state, SimpleNamespace(weight=weight))
    n, k = weight.shape
    blocks = (n + 63) // 64
    if k % 512 or n % 64:
        raise ValueError("probe admits only complete pinned head tiles")
    x = torch.zeros((16, k), device="cuda", dtype=torch.bfloat16)
    xs = torch.zeros((16, k // 128), device="cuda", dtype=torch.float32)
    inputs = torch.randn((args.rows, k), device="cuda", dtype=torch.bfloat16)
    variants = {"control": (original, original._cfg_for(16))}
    for warps, stages, occupancy in (
        (1, 1, None),
        (2, 2, None),
        (4, 1, None),
        (4, 2, None),
        (8, 1, None),
        (2, 1, 2),
        (2, 1, 4),
        (2, 1, 8),
        (4, 1, 2),
        (4, 1, 4),
    ):
        config = {"num_warps": warps, "num_stages": stages}
        if occupancy is not None:
            config["waves_per_eu"] = occupancy
        variants[f"w{warps}_s{stages}_occ{occupancy or 0}"] = (original, config)
    source = Path(original.__file__).read_text()
    selection_variants = {"control"}
    for selection in ("bitonic", "paired_reduce"):
        path = args.output / (selection + ".py")
        path.write_text(replace_candidate_selection(source, selection))
        spec = importlib.util.spec_from_file_location(
            "coherence_head_" + selection, path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for warps in (2, 4):
            name = f"{selection}_w{warps}"
            variants[name] = (module, {"num_warps": warps, "num_stages": 1})
            selection_variants.add(name)
    for name, change in (
        (
            "cg",
            (
                "mask=mask_n[None, :], other=0).to(tl.uint16)",
                'mask=mask_n[None, :], other=0, cache_modifier=".cg").to(tl.uint16)',
            ),
        ),
        (
            "unroll2",
            (
                "for g in range(0, NG):",
                "for g in tl.range(0, NG, loop_unroll_factor=2):",
            ),
        ),
        (
            "pipeline2",
            ("for g in range(0, NG):", "for g in tl.range(0, NG, num_stages=2):"),
        ),
    ):
        old, new = change
        if source.count(old) != 1:
            raise ValueError("INT2 head source anchor changed")
        path = args.output / (name + ".py")
        path.write_text(source.replace(old, new))
        spec = importlib.util.spec_from_file_location("coherence_head_" + name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        variants[name] = (module, original._cfg_for(16))
    # Larger physical N tiles retain the original eight candidates from each
    # logical 64-token block. Neither local capacity nor tie order is relaxed.
    start = source.index(
        '        masked = tl.where(mask_n[None, :], acc, float("-inf"))'
    )
    end = source.index("\n    @triton.jit", start)
    emit = """        masked = tl.where(mask_n[None, :], acc, float("-inf"))
        pieces: tl.constexpr = BLOCK_N // 64
        masked = tl.reshape(masked, (acc.shape[0], pieces, 64))
        block = pid * pieces + tl.arange(0, pieces)
        for c in tl.static_range(KC):
            mx = tl.max(masked, axis=2)
            am = tl.argmax(masked, axis=2)
            offsets = offs_m[:, None] * (NBLK * KC) + block[None, :] * KC + c
            tl.store(BM + offsets, mx)
            tl.store(BI + offsets, (block[None, :] * 64 + am).to(tl.int32))
            masked = tl.where(tl.arange(0, 64)[None, None, :] == am[:, :, None], float("-inf"), masked)
"""
    path = args.output / "larger_n_tiles.py"
    path.write_text(source[:start] + emit + source[end:])
    spec = importlib.util.spec_from_file_location("coherence_head_larger_tiles", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for block_n, warps in ((128, 2), (128, 4), (128, 8), (256, 4), (256, 8)):
        variants[f"n{block_n}_w{warps}"] = (
            module,
            {"num_warps": warps, "num_stages": 1, "block_n": block_n},
        )
    scale_transposed = state._radiance_scale.T.contiguous()
    zs_transposed = state._radiance_zs.T.contiguous()
    weight_grouped = (
        state._radiance_wq.reshape(n, k // 512, 128).permute(1, 0, 2).contiguous()
    )
    for layout in ("scale", "weight", "both"):
        altered = source
        if layout in ("scale", "both"):
            for symbol in ("S", "ZS"):
                old = "tl.load(" + symbol + " + offs_n * stride_s + gi"
                if altered.count(old) != 1:
                    raise ValueError("head scale indexing changed")
                altered = altered.replace(
                    old, "tl.load(" + symbol + " + gi * stride_s + offs_n"
                )
        if layout in ("weight", "both"):
            old = "Wq + offs_n[None, :] * stride_wq + (g * G + offs_k)[:, None]"
            if altered.count(old) != 1:
                raise ValueError("head weight indexing changed")
            altered = altered.replace(
                old, "Wq + g * stride_wq + offs_n[None, :] * G + offs_k[:, None]"
            )
        path = args.output / ("layout_" + layout + ".py")
        path.write_text(altered)
        spec = importlib.util.spec_from_file_location(
            "coherence_head_layout_" + layout, path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for warps in (2, 4):
            variants[f"layout_{layout}_w{warps}"] = (
                module,
                {
                    "num_warps": warps,
                    "num_stages": 1,
                    "scale_transposed": layout in ("scale", "both"),
                    "weight_grouped": layout in ("weight", "both"),
                },
            )
    if args.selection_only:
        variants = {
            name: item for name, item in variants.items() if name in selection_variants
        }
    report = {
        "schema": "coherence-int2-launch-tuning-v1",
        "status": "RUNNING",
        "source_sha256": digest(original.__file__),
        "probe_sha256": digest(__file__),
        "shape": [16, n, k],
        "rows": args.rows,
        "scope": "INT2 operator; no full-model claim or sampler change",
        "modes": {},
    }

    def save():
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        for kc in (0, 8):
            results = {}
            buffers = {}
            for name in variants:
                buffers[name] = (
                    torch.empty((16, n), device="cuda", dtype=torch.bfloat16),
                    torch.empty((16, max(1, blocks * kc)), device="cuda"),
                    torch.empty(
                        (16, max(1, blocks * kc)), device="cuda", dtype=torch.int32
                    ),
                )

            def launch(name, buffers=buffers, kc=kc, k=k, n=n):
                module, config = variants[name]
                config = dict(config)
                block_n = config.pop("block_n", 64)
                if config.pop("scale_transposed", False):
                    scale, zs = scale_transposed, zs_transposed
                else:
                    scale, zs = state._radiance_scale, state._radiance_zs
                packed = (
                    weight_grouped
                    if config.pop("weight_grouped", False)
                    else state._radiance_wq
                )
                y, bm, bi = buffers[name]
                module._draft_head_int2[((n + block_n - 1) // block_n,)](
                    x,
                    xs,
                    packed,
                    scale,
                    zs,
                    y,
                    bm,
                    bi,
                    k,
                    n,
                    packed.stride(0),
                    scale.stride(0),
                    xs.stride(0),
                    blocks,
                    kc,
                    G=128,
                    BLOCK_M=16,
                    BLOCK_N=block_n,
                    **config,
                )

            eligible = []
            for name in variants:
                try:
                    launch(name)
                    torch.cuda.synchronize()
                except (CompilationError, OutOfResources, ValueError) as exc:
                    results[name] = {
                        "status": "UNSUPPORTED",
                        "error": type(exc).__name__ + ": " + str(exc)[:1000],
                    }
                    continue
                eligible.append(name)
                results[name] = {
                    "status": "CHECKING",
                    "checked_rows": 0,
                    "logit_differences": 0,
                    "candidate_score_differences": 0,
                    "candidate_id_differences": 0,
                }
            if "control" not in eligible:
                raise RuntimeError("release head failed")
            for offset in range(0, args.rows, 8):
                x.zero_()
                x[:8].copy_(inputs[offset : offset + 8])
                xs.copy_(x.reshape(16, k // 128, 128).float().sum(-1))
                launch("control")
                reference = buffers["control"]
                for name in eligible:
                    if name != "control":
                        launch(name)
                    y, bm, bi = buffers[name]
                    row = results[name]
                    row["logit_differences"] += int(
                        (y.view(torch.int16) != reference[0].view(torch.int16)).sum()
                    )
                    if kc:
                        row["candidate_score_differences"] += int(
                            (
                                bm.view(torch.int32) != reference[1].view(torch.int32)
                            ).sum()
                        )
                        row["candidate_id_differences"] += int(
                            (bi != reference[2]).sum()
                        )
                    row["checked_rows"] += 8
            negative = reference[0].clone()
            negative.view(torch.int16)[0, 0] ^= 1
            if torch.equal(negative, reference[0]):
                raise AssertionError("bit-flip negative control was not detected")
            good = []
            for name in eligible:
                row = results[name]
                row["status"] = (
                    "EXACT_SAMPLE"
                    if not any(
                        row[k]
                        for k in (
                            "logit_differences",
                            "candidate_score_differences",
                            "candidate_id_differences",
                        )
                    )
                    else "REJECTED_MISMATCH"
                )
                if row["status"] == "EXACT_SAMPLE":
                    good.append(name)
            graphs = {}
            for name in good:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(8):
                        launch(name)
                graphs[name] = graph
            times = {name: [] for name in good}
            for _ in range(5):
                for graph in graphs.values():
                    graph.replay()
            for trial in range(9):
                for name in good if trial % 2 == 0 else good[::-1]:
                    start, end = (
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    )
                    start.record()
                    for _ in range(8):
                        graphs[name].replay()
                    end.record()
                    end.synchronize()
                    times[name].append(start.elapsed_time(end) / 64)
            for name, samples in times.items():
                results[name]["median_ms"] = statistics.median(samples)
                results[name]["samples_ms"] = samples
            report["modes"][str(kc)] = {
                "results": results,
                "negative_control_detected": True,
            }
            save()
            print(
                json.dumps(
                    {
                        "kc": kc,
                        "results": {
                            n: {k: v for k, v in r.items() if k != "samples_ms"}
                            for n, r in results.items()
                        },
                    }
                ),
                flush=True,
            )
            del graphs, buffers
            torch.cuda.empty_cache()
        report["status"] = "OPERATOR_SAMPLE_CHECKED"
        save()
    except Exception:
        report["status"] = "FAILED"
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=320)
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args()
    if args.rows <= 0 or args.rows % 8:
        parser.error("rows must be a positive multiple of eight")
    main(args)

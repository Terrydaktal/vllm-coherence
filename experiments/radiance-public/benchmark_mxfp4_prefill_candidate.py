"""Paired synthetic-activation timing of the four TP1 MXFP4 input GEMM arms."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from statistics import median

from mxfp4_prefill_candidate import (
    SHAPES,
    checked_launch,
    fold_indices,
    pack_words,
    sha256,
)

LAYERS = {
    "attention": (3, "self_attn", ("q_proj", "k_proj", "v_proj")),
    "gdn": (0, "linear_attn", ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")),
    "mlp": (0, "mlp", ("gate_proj", "up_proj")),
}
ARMS = ("control", "preprocess")
# Each arm occupies every ordinal once. The reversed pairs counter thermal drift.
ORDERS = ((0, 1), (1, 0), (0, 1), (1, 0))


def weights(model, index, layer):
    import torch
    from mxfp4_fold_precision import make_row_ref
    from safetensors import safe_open

    from radiance_mxfp4 import permute_w

    layer_number, component, names = LAYERS[layer]
    base = f"model.language_model.layers.{layer_number}.{component}."

    def tensor(suffix):
        parts = []
        for name in names:
            key = base + name + suffix
            with safe_open(model / index[key], framework="pt", device="cpu") as shard:
                parts.append(shard.get_tensor(key))
        return torch.cat(parts)

    w = tensor(".weight")
    s = tensor(".weight_scale")
    n, k = w.shape[0], 2 * w.shape[1]
    if (
        (n, k) not in SHAPES
        or s.shape != (n, k // 32)
        or not (s.dtype == w.dtype == torch.uint8)
    ):
        raise ValueError(f"{layer} has an unexpected checkpoint layout")
    digest = hashlib.sha256()
    for t in (w, s):
        digest.update(memoryview(t.numpy()).cast("B"))
    w = permute_w(w.cuda(), n, k)
    s = s.cuda().T.contiguous()
    return w, s, make_row_ref(s), digest.hexdigest()


def measure(launch, tensors, m, n, k, tn, repeats, stream, eviction_buffer):
    import torch

    a, weight, ws, ref, scale, psw, indices = tensors
    control = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(control)

    def run(mode):
        w, s = (psw, indices) if mode & 1 else (weight, ws)
        dest = out if mode else control
        rc = launch(
            a.data_ptr(),
            w.data_ptr(),
            s.data_ptr(),
            ref.data_ptr(),
            scale.data_ptr(),
            dest.data_ptr(),
            m,
            n,
            k,
            stream.cuda_stream,
            tn,
            mode,
        )
        if rc:
            raise RuntimeError(f"native launch {ARMS[mode]} returned {rc}")

    run(0)
    # Poison the candidate destination uniquely; failure to store is observable.
    bad = {}
    for mode in (1,):
        out.view(torch.int16).copy_(control.view(torch.int16).bitwise_not())
        run(mode)
        different = 0
        for first in range(0, m, 256):
            end = min(first + 256, m)
            different += int(
                torch.count_nonzero(
                    out[first:end].view(torch.int16)
                    != control[first:end].view(torch.int16)
                )
            )
        bad[ARMS[mode]] = different
    if any(bad.values()):
        return {
            "status": "DIFFERENT",
            "different_bf16_elements": bad,
            "elements": m * n,
        }

    samples = {arm: [] for arm in ARMS}
    after_eviction = {arm: [] for arm in ARMS}
    for order in ORDERS:
        events = []
        for mode in order:
            eviction_buffer.add_(1)
            cold = torch.cuda.Event(enable_timing=True)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            cold.record(stream)
            run(mode)
            start.record(stream)
            for _ in range(repeats):
                run(mode)
            end.record(stream)
            events.append((mode, cold, start, end))
        stream.synchronize()
        for mode, cold, start, end in events:
            after_eviction[ARMS[mode]].append(cold.elapsed_time(start))
            samples[ARMS[mode]].append(start.elapsed_time(end) / repeats)
    med = {mode: median(values) for mode, values in samples.items()}
    return {
        "status": "MATCH",
        "elements": m * n,
        "different_bf16_elements": bad,
        "after_eviction_samples_ms": after_eviction,
        "after_eviction_median_ms": {
            arm: median(values) for arm, values in after_eviction.items()
        },
        "samples_ms": samples,
        "median_ms": med,
        "change_percent": {a: 100 * (med[a] / med["control"] - 1) for a in ARMS[1:]},
    }


def main():
    import torch

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--widths", type=int, nargs="+", default=[256, 2049, 3296])
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    if not 1 <= args.repeats <= 10 or any(not 64 < m <= 4096 for m in args.widths):
        p.error("repeats must be 1..10 and widths must be 65..4096")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1201":
        raise ValueError("this benchmark needs one gfx1201 device")
    stream = torch.cuda.current_stream()
    native = checked_launch(args.build)
    eviction_buffer = torch.zeros(128 << 20, device="cuda", dtype=torch.uint8)
    index_file = args.model / "model.safetensors.index.json"
    index = json.loads(index_file.read_text())["weight_map"]
    seed = 839211
    rng = torch.Generator(device="cuda").manual_seed(seed)
    report = {
        "status": "RUNNING",
        "scope": "paired isolated input-projection kernel replay",
        "probe_sha256": sha256(__file__),
        "build_sha256": sha256(args.build / "build.json"),
        "index_sha256": sha256(index_file),
        "device": torch.cuda.get_device_name(0),
        "hip": torch.version.hip,
        "activation_seed": seed,
        "repeats": args.repeats,
        "eviction_bytes_before_single_launch": eviction_buffer.nbytes,
        "orders": [list(order) for order in ORDERS],
        "cases": [],
        "ineligible": [],
        "elapsed_seconds": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        tmp = args.output.with_suffix(".partial.json")
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        os.replace(tmp, args.output)

    save()
    try:
        for label in LAYERS:
            w, s, ref, identity = weights(args.model, index, label)
            n, k = w.shape[0], 2 * w.shape[1]
            idx = fold_indices(s, ref)
            if idx is None:
                report["ineligible"].append(
                    {"layer": label, "source_weight_sha256": identity}
                )
                continue
            packed = pack_words(w)
            for m in args.widths:
                # Valid signed E4M3 values in a physical fragment-tiled buffer; all
                # arms consume the very same bytes and per-token scales.
                padded = (m + 15) // 16 * 16
                codes = torch.randint(
                    0, 128, (padded, k), dtype=torch.uint8, device="cuda", generator=rng
                )
                signs = (codes & 1) << 7
                a = ((codes >> 1) + 16) | signs
                scale = torch.rand(m, device="cuda", generator=rng) + 0.125
                tn = (
                    4
                    if m >= int(os.environ.get("RADIANCE_MXFP4_TN4_MIN_M", "2048"))
                    else 2
                )
                row = measure(
                    native,
                    (a, w, s, ref, scale, packed, idx),
                    m,
                    n,
                    k,
                    tn,
                    args.repeats,
                    stream,
                    eviction_buffer,
                )
                row.update(
                    layer=label, rows=m, n=n, k=k, tn=tn, source_weight_sha256=identity
                )
                report["cases"].append(row)
                save()
                print(
                    json.dumps(
                        {
                            v: row[v]
                            for v in ("layer", "rows", "status", "median_ms")
                            if v in row
                        }
                    ),
                    flush=True,
                )
                if row["status"] != "MATCH":
                    raise ValueError(
                        "a candidate differs from the corrected native control"
                    )
            del w, s, ref, idx, packed
        if not report["cases"]:
            raise ValueError("no corrected folded input weights were eligible")
        report["status"] = "COMPLETE"
    except BaseException as error:
        report["status"] = "FAILED"
        report["error"] = str(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()

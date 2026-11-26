"""Paired quick timings and bit-exactness of thin GEMM tails and shape-specific TN2/TN4."""

import argparse
import json
import random
from pathlib import Path
from statistics import median

from benchmark_mxfp4_prefill_candidate import LAYERS, weights
from mxfp4_prefill_candidate import fold_indices, pack_words, sha256
from mxfp4_prefill_tiles import eligible_tail, load


def measure(native, layer, m, a, scale, weight, scales, ref, *, packed, indices, repeat, order):
    import torch

    n, k = weight.shape[0], weight.shape[1] * 2
    tn = 2 if m < 2048 else 4
    options = [("default", tn, 0)]
    if eligible_tail(m):
        options.append(("thin", tn, 1))
    if m in (512, 768, 1024, 1648, 2048):
        options.append(("other_tn", 6 - tn, 0))

    reference = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    dest = torch.empty_like(reference)
    stream = torch.cuda.current_stream()

    def launch(w, s, out, tile, prep, thin):
        status = native(
            a.data_ptr(),
            w.data_ptr(),
            s.data_ptr(),
            ref.data_ptr(),
            scale.data_ptr(),
            out.data_ptr(),
            m,
            n,
            k,
            stream.cuda_stream,
            tile,
            prep,
            thin,
        )
        if status:
            raise RuntimeError(f"native launch failed with status {status}")

    launch(weight, scales, reference, tn, 0, 0)
    records = []
    for encoding in (0, 1, 3) if indices is not None else (0,):
        w, s = (packed, indices) if encoding else (weight, scales)
        for _, tile, thin in options:
            dest.view(torch.int16).copy_(reference.view(torch.int16).bitwise_not())
            launch(w, s, dest, tile, encoding, thin)
            if not torch.equal(dest.view(torch.int16), reference.view(torch.int16)):
                raise ValueError(f"mismatch at {layer=} {m=} {tile=} {encoding=} {thin=}")
        samples = {key: [] for key, *_ in options}
        for _ in range(repeat):
            # AB/BA pairs cancel first-in-pair and slow temperature/frequency drift.
            paired = options.copy()
            order.shuffle(paired)
            for seq in (paired, paired[::-1]):
                for name, tile, thin in seq:
                    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    start.record(stream)
                    launch(w, s, dest, tile, encoding, thin)
                    launch(w, s, dest, tile, encoding, thin)
                    end.record(stream)
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end) / 2)
        # Use alternating pairs for tuning and untouched confirmation estimates.
        fit = {
            key: median(v for i, v in enumerate(values) if (i // 2) % 2 == 0)
            for key, values in samples.items()
        }
        heldout = {
            key: median(v for i, v in enumerate(values) if (i // 2) % 2 == 1)
            for key, values in samples.items()
        }
        records.append(
            {
                "layer": layer,
                "m": m,
                "n": n,
                "k": k,
                "encoding": encoding,
                "default_tn": tn,
                "elements": m * n,
                "all_byte_exact": True,
                "ms": samples,
                "fit_ms": fit,
                "heldout_ms": heldout,
            }
        )
    return records


def run(args):
    import torch

    device = torch.cuda.get_device_properties(0)
    if device.gcnArchName.split(":")[0] != "gfx1201":
        raise ValueError("expected one gfx1201 device")
    torch.manual_seed(12931)
    order = random.Random(12931)
    native = load(args.library)
    index_path = args.model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    report = {
        "device": device.name,
        "hip": torch.version.hip,
        "source": sha256(__file__),
        "library": sha256(args.library),
        "index": sha256(index_path),
        "seed": 12931,
        "status": "RUNNING",
        "weights": {},
        "cases": [],
    }

    def save():
        tmp = args.output.with_suffix(".partial.json")
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(args.output)

    save()
    for label in LAYERS:
        w, s, ref, digest = weights(args.model, index, label)
        report["weights"][label] = digest
        indices = fold_indices(s, ref)
        packed = pack_words(w) if indices is not None else None
        for m in args.widths:
            k = w.shape[1] * 2
            codes = torch.randint(
                0, 128, ((m + 15) // 16 * 16, k), dtype=torch.uint8, device="cuda"
            )
            a = ((codes >> 1) + 16) | ((codes & 1) << 7)
            scale = torch.rand(m, device="cuda") + 0.125
            report["cases"].extend(
                measure(
                    native,
                    label,
                    m,
                    a,
                    scale,
                    w,
                    s,
                    ref,
                    packed=packed,
                    indices=indices,
                    repeat=args.repeat,
                    order=order,
                )
            )
            save()
        del packed, indices, w, s, ref
    report["status"] = "COMPLETE"
    save()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeat", type=int, default=4, choices=range(2, 13))
    p.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[129, 160, 161, 256, 320, 384, 512, 513, 768, 1024, 1648, 2048],
    )
    args = p.parse_args()
    if any(m <= 64 or m > 4096 for m in args.widths):
        p.error("expected tiled input rows in [65,4096]")
    run(args)

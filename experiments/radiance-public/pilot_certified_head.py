"""CPU tightness pilot for variable-size target-head candidate certification.

Requires a raw BF16 head slice with identity metadata and explicit synthetic
capture arrays. Uses no GPU library and never reads chat/session files. Timings
are CPU analysis costs, not native head latency or a tokens/second prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from qwen_r9700_lab import certified_head as ch


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def distribution(values):
    values = np.asarray(values)
    return {
        "min": int(values.min()),
        "median": float(np.median(values)),
        "max": int(values.max()),
        "per_row": values.tolist(),
    }


def support_metrics(lo, hi, approx, reference, k):
    """Measure both the guaranteed cutoff and an explicitly optimistic refinement.

    Captured native scores are evidence, not a reference-compatible rescorer.
    The seed experiment assumes a perfect rescorer solely to bound likely work.
    """
    m, n = lo.shape
    cutoff = np.partition(lo, n - k, axis=1)[:, n - k]
    survivors = hi >= cutoff[:, None]
    reference_cutoff = np.partition(reference, n - k, axis=1)[:, n - k]
    true_support = reference >= reference_cutoff[:, None]
    missed = true_support & ~survivors
    seeds = {}
    for size in (128, 256):
        size = min(size, n)
        indices = np.argsort(-approx, axis=1, kind="stable")[:, :size]
        seeded_lo, seeded_hi = lo.copy(), hi.copy()
        rows = np.arange(m)[:, None]
        seeded_lo[rows, indices] = reference[rows, indices]
        seeded_hi[rows, indices] = reference[rows, indices]
        revised = np.partition(seeded_lo, n - k, axis=1)[:, n - k]
        unresolved = seeded_hi >= revised[:, None]
        unresolved[rows, indices] = False
        seeds[str(size)] = {
            "unresolved_after_ideal_seed": distribution(unresolved.sum(1)),
            "captured_topk_tokens_missing_from_seed": distribution(
                true_support.sum(1) - true_support[rows, indices].sum(1)
            ),
        }
    return {
        "k": k,
        "survivors": distribution(survivors.sum(1)),
        "captured_topk_misses": int(missed.sum()),
        "cutoffs": cutoff.tolist(),
        "ideal_seed_diagnostic": seeds,
    }


def replay_refinement(arrays, head, hidden, reference, *, group):
    """Run the actual selector with its CPU score-fidelity refiner.

    The fallback is the saved full-head result for this exact input. This is an
    offline replay, not a new GPU evaluation or latency measurement. The INT2
    and INT4 paths each start with their own complete-vocabulary intervals.
    """
    outcomes = []
    for row in range(len(hidden)):
        identity = f"saved-head-row-{row}/group-{group}/{ch.FP32_DOT_CONTRACT}"

        def rescore(ids, *, row=row, identity=identity):
            weights = (head[ids].astype(np.uint32) << 16).view(np.float32)
            lower, upper = ch.reference_score_intervals(
                hidden[row : row + 1], weights, group=group, contract=ch.FP32_DOT_CONTRACT
            )
            return ch.Refinement(ids, ch.IntervalRow(lower[0], upper[0], identity))

        intervals = ch.IntervalRow(arrays["lo"][row], arrays["hi"][row], identity)
        result = ch.refine_topk(
            intervals,
            arrays["center"][row],
            20,
            (rescore,),
            lambda row=row: reference[row],
            seed_size=256,
            max_work=1024,
        )
        captured = reference[row, result.ids]
        if not np.array_equal(captured.view(np.uint64), result.scores.view(np.uint64)):
            raise ValueError("CPU singleton disagrees with saved full-head result")
        outcomes.append(
            {
                "status": result.status,
                "reason": result.reason,
                "rescored": result.rescored,
                "rounds": result.rounds,
            }
        )
    counts = {}
    for outcome in outcomes:
        key = outcome["status"] + (":" + outcome["reason"] if outcome["reason"] else "")
        counts[key] = counts.get(key, 0) + 1
    return {
        "counts": counts,
        "per_row": outcomes,
        "scope": "CPU selector; saved full-head fallback; conditional arithmetic contract",
    }


def run(weights: Path, metadata: Path, rows: Path, output: Path, *, group=128, chunk=1024):
    meta = json.loads(metadata.read_text())
    row_receipt = json.loads(rows.with_suffix(".json").read_text())
    if digest(weights) != meta["sha256"] or digest(rows) != row_receipt["output_sha256"]:
        raise ValueError("input identity mismatch")
    vocab, width = meta["shape"]
    if meta["dtype"] != "BF16" or weights.stat().st_size != vocab * width * 2:
        raise ValueError("weight shape/dtype/size mismatch")
    if group < 1 or width % group or chunk < 1:
        raise ValueError("invalid group or chunk size")
    with np.load(rows, allow_pickle=False) as source:
        hidden = source["hidden"].astype(np.float64)
        reference = source["reference"].astype(np.float64)
    if hidden.ndim != 2 or hidden.shape[1] != width or reference.shape != (len(hidden), vocab):
        raise ValueError("capture shape differs from head")
    if not np.isfinite(reference).all() or not np.array_equal(ch.bf16(reference), reference):
        raise ValueError("reference must contain finite BF16-valued scores")
    output.mkdir(parents=True, exist_ok=False)
    sources = {"pilot": digest(Path(__file__)), "intervals": digest(Path(ch.__file__))}
    weight_before = weights.stat()
    head = np.memmap(weights, dtype="<u2", mode="r", shape=(vocab, width))
    shape = reference.shape
    arrays = {
        bits: {name: np.empty(shape, dtype=np.float64) for name in ("center", "lo", "hi")}
        for bits in (2, 4)
    }
    metrics = {
        bits: {"captured_scores_outside_intervals": 0, "cpu_seconds": 0.0} for bits in (2, 4)
    }
    ref_outside = np.zeros(len(hidden), dtype=np.int64)
    score_singletons = np.zeros(len(hidden), dtype=np.int64)
    score_singletons_disagree = 0
    topk_singletons = {k: np.zeros(len(hidden), dtype=np.int64) for k in (1, 20)}
    topk_counts = {}
    topk_cutoffs = {}
    for k in topk_singletons:
        topk_cutoffs[k] = np.partition(reference, vocab - k, axis=1)[:, vocab - k]
        topk_counts[k] = (reference >= topk_cutoffs[k][:, None]).sum(1)
    started = time.monotonic()
    score_seconds = 0.0
    for first in range(0, vocab, chunk):
        last = min(vocab, first + chunk)
        w = (head[first:last].astype(np.uint32) << 16).view(np.float32)
        for bits in (2, 4):
            tick = time.monotonic()
            search = ch.quantize_search(w, bits, group)
            model = ch.prepare_bounds(w, search)
            center, lo, hi = ch.search_intervals(hidden, model, contract=ch.FP32_DOT_CONTRACT)
            for name, value in (("center", center), ("lo", lo), ("hi", hi)):
                arrays[bits][name][:, first:last] = value
            metrics[bits]["captured_scores_outside_intervals"] += int(
                ((reference[:, first:last] < lo) | (reference[:, first:last] > hi)).sum()
            )
            metrics[bits]["cpu_seconds"] += time.monotonic() - tick
        tick = time.monotonic()
        lo, hi = ch.reference_score_intervals(hidden, w, group=group, contract=ch.FP32_DOT_CONTRACT)
        score_seconds += time.monotonic() - tick
        captured = reference[:, first:last]
        ref_outside += ((captured < lo) | (captured > hi)).sum(1)
        singleton = lo.view(np.uint64) == hi.view(np.uint64)
        score_singletons += singleton.sum(1)
        score_singletons_disagree += int((singleton & (lo != captured)).sum())
        for k, cutoff in topk_cutoffs.items():
            topk_singletons[k] += (singleton & (captured >= cutoff[:, None])).sum(1)
        if first % (32 * chunk) == 0 or last == vocab:
            progress = {
                "tokens_processed": last,
                "vocab": vocab,
                "rows": len(hidden),
                "cpu_seconds": round(time.monotonic() - started, 1),
            }
            print(json.dumps(progress), flush=True)
            (output / "progress.json").write_text(json.dumps(progress) + "\n")
    for bits in arrays:
        metrics[bits]["selection"] = {
            str(k): support_metrics(
                arrays[bits]["lo"], arrays[bits]["hi"], arrays[bits]["center"], reference, k
            )
            for k in (1, 20)
        }
        metrics[bits]["refinement_replay"] = replay_refinement(
            arrays[bits], head, hidden, reference, group=group
        )
    intersect_lo = np.maximum(arrays[2]["lo"], arrays[4]["lo"])
    intersect_hi = np.minimum(arrays[2]["hi"], arrays[4]["hi"])
    disjoint = int((intersect_lo > intersect_hi).sum())
    intersection = (
        None
        if disjoint
        else {
            str(k): support_metrics(intersect_lo, intersect_hi, arrays[4]["center"], reference, k)
            for k in (1, 20)
        }
    )
    weight_after = weights.stat()
    if (weight_before.st_ino, weight_before.st_size, weight_before.st_mtime_ns) != (
        weight_after.st_ino,
        weight_after.st_size,
        weight_after.st_mtime_ns,
    ) or sources != {"pilot": digest(Path(__file__)), "intervals": digest(Path(ch.__file__))}:
        raise ValueError("input or implementation changed during pilot")
    result = {
        "status": "CPU_BOUND_TIGHTNESS_PILOT",
        "gpu_used": False,
        "scope": row_receipt["scope"],
        "native_certificate": False,
        "arithmetic_contract": ch.FP32_DOT_CONTRACT,
        "contract_native_binding": "UNPROVED",
        "head_shape": [vocab, width],
        "rows": len(hidden),
        "group": group,
        "weights_sha256": meta["sha256"],
        "capture_arrays_sha256": row_receipt["output_sha256"],
        "source_sha256": sources,
        "variants": metrics,
        "int2_then_int4_intersection": intersection,
        "disjoint_intersections": disjoint,
        "score_fidelity": {
            "captured_scores_outside_reference_envelope": distribution(ref_outside),
            "singleton_scores": distribution(score_singletons),
            "singleton_disagreements_with_capture": score_singletons_disagree,
            "topk_singletons": {str(k): distribution(v) for k, v in topk_singletons.items()},
            "topk_support_including_ties": {
                str(k): distribution(v) for k, v in topk_counts.items()
            },
            "cpu_seconds": score_seconds,
        },
        "cpu_total_seconds": time.monotonic() - started,
        "limitations": [
            "Synthetic discrepancy captures are not representative of workload frequency.",
            "CPU search reconstruction and FP64 center are not native INT2/INT4 execution.",
            "FP32 accumulator envelope is conditional; native reduction binding is unproved.",
            "Ideal seed uses saved native scores only to estimate work, not to authorize output.",
            "CPU timings do not predict GPU head latency or end-to-end throughput.",
        ],
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    brief = {
        "rows": len(hidden),
        "gpu_used": False,
        "seconds": result["cpu_total_seconds"],
        "variants": {
            str(b): {
                str(k): {
                    name: metrics[b]["selection"][str(k)]["survivors"][name]
                    for name in ("min", "median", "max")
                }
                for k in (1, 20)
            }
            for b in metrics
        },
    }
    print(json.dumps(brief), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", type=int, default=128)
    parser.add_argument("--chunk", type=int, default=1024)
    args = parser.parse_args()
    run(args.weights, args.metadata, args.rows, args.output, group=args.group, chunk=args.chunk)


if __name__ == "__main__":
    main()

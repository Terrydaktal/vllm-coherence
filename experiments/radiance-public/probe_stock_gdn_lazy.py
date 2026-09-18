"""Short exact deferred-state test against the frozen corrected indexed scan.

Synthetic tensors only. All candidate outputs and every accepted-prefix state
are compared as bytes, including graph replay and rejected-suffix isolation.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def run(args):
    import torch
    from stock_gdn_lazy_kernel import invalidate, lazy_update, materialize, materialize_align
    from stock_gdn_scan_kernel import stock_gdn_scan_kernel

    torch.set_num_threads(4)
    torch.manual_seed(190918)
    options = {
        "num_warps": 1,
        "num_stages": 3,
        "enable_fp_fusion": True,
        "allow_flush_denorm": False,
    }
    dev = "cuda"
    storage = torch.full((13 * 802816,), 17.0, device=dev, dtype=torch.float32)
    pool = storage.as_strided((13, 48, 128, 128), (802816, 16384, 128, 1))
    reference_storage = storage.clone()
    ref = reference_storage.as_strided(pool.shape, pool.stride())
    lazy_ids = torch.tensor([[3, 5]], device=dev, dtype=torch.int32)
    ref_ids = torch.tensor([[3, 7, 9, 2, 4, 6, 8, 1]], device=dev, dtype=torch.int32)
    accepted = torch.ones(1, device=dev, dtype=torch.int32)
    cu = torch.tensor([0, 8], device=dev, dtype=torch.int32)
    dest = torch.tensor([11], device=dev, dtype=torch.int32)
    width = accepted.clone()
    errors = torch.zeros(1, device=dev, dtype=torch.int32)
    alog = torch.linspace(-3, 1, 48, device=dev)
    bias = torch.randn(48, device=dev, dtype=torch.bfloat16)
    qkv = torch.randn(8, 10240, device=dev, dtype=torch.bfloat16)
    a = torch.randn(8, 96, device=dev, dtype=torch.bfloat16)[:, :48]
    b = torch.randn_like(a)
    output = torch.empty(8, 48, 128, device=dev, dtype=torch.bfloat16)
    expected = torch.empty_like(output)
    initial = torch.randn_like(pool[3]) * 0.1
    pool[3].copy_(initial)
    ref[3].copy_(initial)
    invalidate[(48, 1)](pool, lazy_ids[:, 1].contiguous(), pool.stride(0))
    report = {
        "status": "RUNNING",
        "rows": 0,
        "checks": [],
        "timings": [],
        "private_chat_read": False,
        "formal_equivalence": "UNPROVED",
    }

    def check(name, actual, target):
        aa, bb = actual.contiguous().view(torch.uint8), target.contiguous().view(torch.uint8)
        row = {
            "name": name,
            "bytes": aa.numel(),
            "equal": torch.equal(aa, bb),
            "unequal_bytes": int((aa != bb).sum()),
        }
        if not row["equal"]:
            row["max_abs_error"] = float((actual.float() - target.float()).abs().max())
            if actual.shape == (48, 128, 128):
                row["unequal_per_part"] = [
                    (
                        actual[:, p : p + 32].view(torch.int32)
                        != target[:, p : p + 32].view(torch.int32)
                    )
                    .sum()
                    .item()
                    for p in range(0, 128, 32)
                ]
        report["checks"].append(row)
        if not row["equal"]:
            report["status"] = "FAILED"
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            raise RuntimeError(f"exact comparison failed: {name}: {row['unequal_bytes']} bytes")

    def oracle():
        stock_gdn_scan_kernel[(4, 48, 1)](
            qkv,
            a,
            b,
            alog,
            bias,
            ref,
            expected,
            ref,
            ref_ids,
            accepted,
            cu,
            8,
            128**-0.5,
            qkv.stride(0),
            a.stride(0),
            b.stride(0),
            H=16,
            HV=48,
            K=128,
            V=128,
            BK=128,
            BV=32,
            SAVE_ROWS=True,
            INDEXED=True,
            stride_state=ref.stride(0),
            stride_index_seq=8,
            stride_index_row=1,
            **options,
        )

    def candidate():
        lazy_update[(4, 48, 1)](
            qkv,
            a,
            b,
            alog,
            bias,
            pool,
            output,
            lazy_ids,
            accepted,
            cu,
            errors,
            128**-0.5,
            qkv.stride(0),
            a.stride(0),
            b.stride(0),
            pool.stride(0),
            2,
            **options,
        )

    def restore():
        return materialize[(4, 48, 1)](
            pool, lazy_ids, width, dest, alog, bias, errors, 128**-0.5, pool.stride(0), 2, **options
        )

    # Compile before capture, then reset authoritative state. Inputs remain live
    # graph buffers and change each step; comparing only one captured input is weak.
    oracle()
    candidate()
    compiled_restore = restore()
    args.output.with_suffix(".restore.ttgir").write_text(compiled_restore.asm["ttgir"])
    graphs = {}
    for name, fn in (("oracle", oracle), ("candidate", candidate), ("restore", restore)):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        graphs[name] = graph
    ref[3].copy_(initial)
    pool[3].copy_(initial)
    invalidate[(48, 1)](pool, lazy_ids[:, 1].contiguous(), pool.stride(0))
    untouched = pool[[0, 2, 4, 6, 7, 8, 9, 10, 12]].clone()
    for step in range(40):
        accepted.fill_(1 if step == 0 else 1 + (step - 1) % 8)
        qkv.normal_()
        a.normal_()
        b.normal_()
        output.fill_(float("nan"))
        expected.fill_(float("nan"))
        graphs["oracle"].replay()
        graphs["candidate"].replay()
        check(f"step-{step}-output", output, expected)
        for count, slot in enumerate((3, 7, 9, 2, 4, 6, 8, 1), 1):
            width.fill_(count)
            pool[11].fill_(float("nan"))
            graphs["restore"].replay()
            check(f"step-{step}-prefix-{count}", pool[11], ref[slot])
        report["rows"] += 8
    check("untouched-slots", pool[[0, 2, 4, 6, 7, 8, 9, 10, 12]], untouched)
    check("device-error-flag", errors, torch.zeros_like(errors))
    saved = pool.clone()
    reference_saved = ref.clone()
    for current in range(1, 9):
        for previous in range(1, 9):
            pool.copy_(saved)
            ref.copy_(reference_saved)
            accepted.fill_(previous)
            cu[1] = current
            oracle()
            candidate()
            check(f"width-{current}-accepted-{previous}", output[:current], expected[:current])
            width.fill_(current)
            restore()
            check(
                f"width-{current}-accepted-{previous}-state",
                pool[11],
                ref[(3, 7, 9, 2, 4, 6, 8, 1)[current - 1]],
            )
    report["width_pairs"] = 64
    pool.copy_(saved)
    ref.copy_(reference_saved)
    cu[1] = 8
    pointers = [
        torch.tensor([v], device=dev, dtype=torch.int64)
        for v in (pool.data_ptr(), pool.stride(0), 0, alog.data_ptr())
    ]
    bias_pointer = torch.tensor([bias.data_ptr()], device=dev, dtype=torch.int64)
    mapping = torch.tensor([4], device=dev, dtype=torch.int32)
    state_index = torch.zeros(6, device=dev, dtype=torch.int32)
    source_index = torch.zeros_like(state_index)
    off = torch.zeros_like(state_index)
    acc = torch.zeros_like(state_index)
    computed = torch.zeros_like(state_index)
    # Physical IDs differ from the reference, and destination may alias the
    # old stash. All parts must finish reading their OWN region before writing.
    for mode, same_base in ((0, False), (1, False), (1, True)):
        bt = torch.tensor(
            [[0, 0, 3, 5, 9]] if mode == 0 or same_base else [[0, 0, 11, 3, 5]],
            device=dev,
            dtype=torch.int32,
        )
        bt_pointer = torch.tensor([bt.data_ptr()], device=dev, dtype=torch.int64)
        for extra in range(8):
            pool.copy_(saved)
            state_index[4] = 3 if not same_base else 2
            source_index[4] = 2
            off[4] = extra
            acc[4] = 8
            computed[4] = 48 + 7 - extra
            compiled_align = materialize_align[(192, 1, 1)](
                *pointers,
                bias_pointer,
                bt_pointer,
                mapping,
                state_index,
                source_index,
                off,
                acc,
                computed,
                errors,
                128**-0.5,
                5,
                16,
                MODE=mode,
                MAPPED=True,
                **options,
                debug=True,
            )
            args.output.with_suffix(".align.ttgir").write_text(compiled_align.asm["ttgir"])
            destination = 5 if mode == 0 else (3 if same_base else 11)
            check(
                f"align-{mode}-inplace-{same_base}-extra-{extra}",
                pool[destination],
                ref[(3, 7, 9, 2, 4, 6, 8, 1)[extra]],
            )
            if mode == 0:
                headers = pool[9].view(torch.int32).reshape(48, -1)[:, ::4096]
                check(
                    f"migration-invalidates-new-stash-{extra}", headers, torch.zeros_like(headers)
                )
        # An inactive mapping must not read the per-request decision buffers.
        mapping.fill_(-1)
        before = pool.clone()
        materialize_align[(192, 1, 1)](
            *pointers,
            bias_pointer,
            bt_pointer,
            mapping,
            state_index,
            source_index,
            off,
            acc,
            computed,
            errors,
            128**-0.5,
            5,
            16,
            MODE=mode,
            MAPPED=True,
            **options,
            debug=True,
        )
        check(f"inactive-align-{mode}-{same_base}", pool, before)
        mapping.fill_(4)
    pool.copy_(saved)
    check("alignment-device-error-flag", errors, torch.zeros_like(errors))
    report["alignment_cases"] = 24
    # Canonical prefix state roundtrip into different physical slots: no raw
    # speculative stash is treated as an independently resumable state.
    width.fill_(4)
    restore()
    canonical = pool[11].cpu().clone().to(dev)
    pool[10].copy_(canonical)
    ref[3].copy_(canonical)
    lazy_ids[0, 0] = 10
    lazy_ids[0, 1] = 12
    accepted.fill_(1)
    invalidate[(48, 1)](pool, lazy_ids[:, 1].contiguous(), pool.stride(0))
    oracle()
    candidate()
    check("canonical-snapshot-resume-output", output, expected)
    for count, slot in enumerate((3, 7, 9, 2, 4, 6, 8, 1), 1):
        width.fill_(count)
        restore()
        check(f"canonical-snapshot-resume-state-{count}", pool[11], ref[slot])
    report["canonical_snapshot_roundtrip"] = True
    padding = storage.view(13, 802816)[:, 48 * 128 * 128 :]
    check("state-allocation-padding", padding, torch.full_like(padding, 17))
    # A stale/incomplete stash must be detected, not silently clamped.
    invalidate[(48, 1)](pool, lazy_ids[:, 1].contiguous(), pool.stride(0))
    accepted.fill_(8)
    before = pool.clone()
    candidate()
    if int(errors.item()) != 2:
        raise RuntimeError("negative control did not reject an invalid replay width")
    check("invalid-replay-no-state-write", pool, before)
    errors.zero_()
    accepted.fill_(1)
    # Negative numerical control proves the comparison sees one flipped bit.
    bad = output.clone()
    bad.view(torch.int16).flatten()[0] ^= 1
    if torch.equal(output.view(torch.uint8), bad.view(torch.uint8)):
        raise RuntimeError("byte comparator failed its negative control")
    for previous in (1, 4, 8):
        candidate()
        accepted.fill_(previous)
        for name, fn in (("oracle", oracle), ("candidate", candidate)):
            timing_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(timing_graph):
                for _ in range(100):
                    fn()
            for _ in range(3):
                timing_graph.replay()
            samples = []
            for _ in range(5):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                timing_graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) / 100)
            report["timings"].append({"path": name, "previous": previous, "ms": sorted(samples)[2]})
    report.update(
        status="SAMPLE_CHECKED",
        negative_controls=True,
        sources={
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "stock_gdn_lazy_kernel.py",
                "stock_gdn_scan_kernel.py",
                "probe_stock_gdn_lazy.py",
            )
        },
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "rows", "timings")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    if not args.allow_gpu or not os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"):
        raise RuntimeError("explicit GPU admission and shared lease are required")
    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease

    os.umask(0o077)
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with gpu_lease(args.output.parent / ("lease-" + str(time.time_ns()))):
        run(args)


if __name__ == "__main__":
    main()

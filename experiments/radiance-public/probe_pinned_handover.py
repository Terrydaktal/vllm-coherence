#!/usr/bin/env python3
"""Check native pinned-buffer ownership and three-bank DMA using synthetic bytes."""

from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace


def probe():
    # This standalone process must never publish into the live Pi status files.
    os.environ["QWEN_CACHE_JOB_TELEMETRY"] = "0"
    os.environ["QWEN_GENERATION_ROUND_TELEMETRY"] = "0"
    import torch
    from qwen_radiance_fair_scheduler import WorkerBanks
    from qwen_radiance_pinned_memory import allocate_pinned_bytes

    torch.cuda.set_device(0)
    started = time.monotonic()
    a, mapping = allocate_pinned_bytes(torch, 1024**2 + 1)
    b, _ = allocate_pinned_bytes(torch, a.numel())
    expected = torch.arange(a.numel(), dtype=torch.int64).remainder(251).to(torch.uint8)
    a.copy_(expected)
    view = a[16:]
    del a
    gc.collect()
    stream = torch.cuda.Stream()
    for _ in range(64):
        with torch.cuda.stream(stream):
            device = view.to("cuda", non_blocking=True)
            b[16:].copy_(device, non_blocking=True)
        stream.synchronize()
        assert torch.equal(b[16:], expected[16:])

    with tempfile.TemporaryDirectory(prefix="coherence-handover-probe-") as temporary:
        status = str(Path(temporary) / "fair")
        os.environ["QWEN_ROUND_EVENT_STATUS_PATH"] = status
        gpus = [torch.empty(size, dtype=torch.uint8, device="cuda") for size in (1024**2, 512 * 1024)]
        runner = SimpleNamespace(
            kv_caches=[gpus[0], gpus[1], gpus[0]],
            kv_cache_config=SimpleNamespace(num_blocks=64), device="cuda",
        )
        worker = WorkerBanks(runner, {"max_banks": 3, "status_path": status})
        worker._ensure_buffers()
        assert worker.stage is not None and len(worker.free_buffers) == 2
        assert worker.host_page_mappings
        banks = [f"{letter * 64}:{number * 64}" for letter, number in zip("abc", "123", strict=True)]
        selected = [0, 2, 3, 17, 55, 56, 63]
        reference = {}
        verified = 0

        def activate(bank):
            worker.before({"bank": bank, "save_blocks": selected, "drop_banks": [], "barrier": False})

        for index, bank in enumerate(banks):
            activate(bank)
            reference[bank] = []
            for gpu in gpus:
                values = torch.arange(gpu.numel(), dtype=torch.int64).add(index * 61).remainder(251).to(torch.uint8)
                gpu.copy_(values)
                reference[bank].append(values)
        for bank in banks * 20:
            activate(bank)
            for gpu, values in zip(gpus, reference[bank], strict=True):
                stride = gpu.numel() // 64
                for block in selected:
                    begin, end = block * stride, (block + 1) * stride
                    assert torch.equal(gpu[begin:end].cpu(), values[begin:end])
            verified += 1
        assert worker.allocation_events == 1
        assert len(worker.images) == 2

    return {
        "pinned": view.is_pinned(), "view_and_async_copy_cycles": 64,
        "three_chat_handover_checks": verified,
        "unique_gpu_storage_regions": len(worker.regions),
        "requested_view_buffer_bytes": expected.numel(),
        "view_buffer_backing_bytes": mapping[1] - mapping[0],
        "equal": True, "elapsed_seconds": time.monotonic() - started,
        "scope": "Synthetic uint8 data only; existing DMA fences, two parked banks, overlapping/noncontiguous saved blocks, storage-view lifetime. No model inference or session data.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = probe()
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")

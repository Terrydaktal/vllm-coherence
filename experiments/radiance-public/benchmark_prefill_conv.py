"""Check dynamic prefill convolution against unchanged native output and state."""

import gc
import hashlib
import json
import statistics
import time
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import seal, write_private


def run(options):
    import torch
    from prefill_dynamic_conv import PrefillDispatch, clone, native_module

    conv = native_module()
    original = conv._causal_conv1d_update_kernel
    candidate, parent = clone(original)
    dispatch = PrefillDispatch(original, candidate)
    report = {
        "status": "RUNNING",
        "native_source_sha256": parent,
        "source_sha256": hashlib.sha256(
            Path(__file__).with_name("prefill_dynamic_conv.py").read_bytes()
        ).hexdigest(),
        "cases": [],
    }
    torch.manual_seed(902734)
    try:
        for dtype in (torch.bfloat16, torch.float32):
            for varlen in (False, True):
                for rows in options["rows"]:
                    dim = 10240
                    x = torch.randn((rows, dim), dtype=dtype, device="cuda")
                    initial = torch.randn((4, dim, 3), dtype=dtype, device="cuda")
                    weight = torch.randn((dim, 4), dtype=dtype, device="cuda")
                    bias = (
                        torch.randn(dim, dtype=dtype, device="cuda")
                        if options.get("bias", True)
                        else None
                    )
                    indices = torch.tensor(
                        [[1, 2]] if varlen else [1], dtype=torch.int32, device="cuda"
                    )
                    starts = (
                        torch.tensor([0, rows], dtype=torch.int32, device="cuda")
                        if varlen
                        else None
                    )
                    first = (
                        torch.tensor([0], dtype=torch.int32, device="cuda")
                        if varlen
                        else None
                    )
                    last = (
                        torch.tensor([1], dtype=torch.int32, device="cuda")
                        if varlen
                        else None
                    )
                    outputs, states, first_ms = {}, {}, {}

                    def launch(
                        kernel,
                        initial=initial,
                        x=x,
                        varlen=varlen,
                        weight=weight,
                        bias=bias,
                        indices=indices,
                        starts=starts,
                        rows=rows,
                        first=first,
                        last=last,
                    ):
                        conv._causal_conv1d_update_kernel = kernel
                        state = initial.clone()
                        inp = (
                            x.clone()
                            if varlen
                            else x.T.unsqueeze(0).clone(
                                memory_format=torch.preserve_format
                            )
                        )
                        out = conv.causal_conv1d_update(
                            inp,
                            state,
                            weight,
                            bias,
                            activation="silu",
                            conv_state_indices=indices,
                            query_start_loc=starts,
                            max_query_len=rows,
                            initial_state_idx=first,
                            block_idx_last_scheduled_token=last,
                        )
                        return out, state

                    for name, kernel in (("native", original), ("dynamic", dispatch)):
                        torch.cuda.synchronize()
                        before = time.perf_counter()
                        outputs[name], states[name] = launch(kernel)
                        torch.cuda.synchronize()
                        first_ms[name] = (time.perf_counter() - before) * 1000
                    case = {
                        "rows": rows,
                        "has_bias": bias is not None,
                        "path": "native_static" if rows in (1648, 2048) else "dynamic",
                        "dtype": str(dtype),
                        "varlen": varlen,
                        "output_different": int(
                            torch.count_nonzero(
                                outputs["native"].cpu().contiguous().view(torch.uint8)
                                != outputs["dynamic"]
                                .cpu()
                                .contiguous()
                                .view(torch.uint8)
                            )
                        ),
                        "state_different": int(
                            torch.count_nonzero(
                                states["native"].cpu().contiguous().view(torch.uint8)
                                != states["dynamic"]
                                .cpu()
                                .contiguous()
                                .view(torch.uint8)
                            )
                        ),
                        "finite": bool(
                            torch.isfinite(outputs["dynamic"]).all()
                            and torch.isfinite(states["dynamic"]).all()
                        ),
                        "output_elements": outputs["dynamic"].numel(),
                        "state_elements": states["dynamic"].numel(),
                        "first_call_ms": first_ms,
                    }
                    report["cases"].append(case)
                    if (
                        case["output_different"]
                        or case["state_different"]
                        or not case["finite"]
                    ):
                        report["status"] = "MISMATCH"
                        write_private(Path(options["output"]), seal(report))
                        raise RuntimeError("dynamic convolution changed output/state")
                    samples = {"native": [], "dynamic": []}
                    for iteration in range(options.get("repeats", 5)):
                        order = [("native", original), ("dynamic", dispatch)]
                        for name, kernel in (
                            order if iteration % 2 == 0 else reversed(order)
                        ):
                            begin, end = (
                                torch.cuda.Event(enable_timing=True) for _ in range(2)
                            )
                            begin.record()
                            unused = launch(kernel)
                            end.record()
                            end.synchronize()
                            samples[name].append(begin.elapsed_time(end))
                            del unused
                    case["median_ms"] = {
                        k: statistics.median(v) for k, v in samples.items()
                    }
                    print(json.dumps(case), flush=True)
                    del outputs, states, x, initial, weight, bias, launch
                    gc.collect()
        report["status"] = "SAMPLE_CHECKED"
        report["calls"] = dispatch.calls
        write_private(Path(options["output"]), seal(report))
        return report
    finally:
        conv._causal_conv1d_update_kernel = original

"""Check whether shared GDN preparation pays without changing nine-slot states.

The installed recurrence and arithmetic are the reference. Every FP32 after-row
state and BF16 output is compared; only exact samples proceed to GPU timing.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
from pathlib import Path

PREPARE = """
@triton.jit
def prepare_qk(mixed, qk, stride_m: tl.constexpr, scale):
    row, head = tl.program_id(0), tl.program_id(1)
    i = tl.arange(0, 128)
    q = tl.load(mixed + row * stride_m + head * 128 + i).to(tl.float32)
    k = tl.load(mixed + row * stride_m + (16 + head) * 128 + i).to(tl.float32)
    q = q / tl.sqrt(tl.sum(q * q) + 1e-6)
    k = k / tl.sqrt(tl.sum(k * k) + 1e-6)
    q = q * scale
    tl.store(qk + row * 4096 + head * 128 + i, q)
    tl.store(qk + row * 4096 + (16 + head) * 128 + i, k)

@triton.jit
def prepare_gates(a, b, alog, bias, gates, sa: tl.constexpr, sb: tl.constexpr):
    row = tl.program_id(0)
    h = tl.arange(0, 64)
    av = tl.load(a + row * sa + h, h < 48, other=0).to(tl.float32)
    bv = tl.load(b + row * sb + h, h < 48, other=0).to(tl.float32)
    al = tl.load(alog + h, h < 48, other=0).to(tl.float32)
    db = tl.load(bias + h, h < 48, other=0).to(tl.float32)
    x = av + db
    sp = tl.where(x <= 20.0, tl.extra.libdevice.log1p(tl.exp(x)), x)
    g = -tl.exp(al) * sp
    beta = tl.sigmoid(bv).to(b.dtype.element_ty).to(tl.float32)
    tl.store(gates + row * 96 + h, exp(g), h < 48)
    tl.store(gates + row * 96 + 48 + h, beta, h < 48)
"""


def main(args):
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(20260925)
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    source = args.source.read_text()
    expected = "be8533624e435fe138237b06cecd12848773e88bf0b36e48a921a032111d887b"
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != expected:
        raise ValueError("pinned GDN arithmetic source changed")
    modules = {}
    variants = (
        ("control", False, False, 4),
        ("gates", False, True, 4),
        ("qk_w1", True, False, 1),
        ("qk_w4", True, False, 4),
        ("both_w1", True, True, 1),
        ("both_w4", True, True, 4),
    )
    geometries = {"control": (32, 4, 0)}
    if args.geometry:
        geometries.update({f"bv{bv}_w4": (bv, 4, 0) for bv in (8, 16, 64, 128)})
        geometries.update({f"bv32_w4_occ{occ}": (32, 4, occ) for occ in (2, 4, 8)})
        variants = tuple((name, False, False, 4) for name in geometries)
    for name, qk, gates, warps in variants:
        changed = source.replace(
            "    initial,\n", "    initial,\n    prepared_qk,\n    prepared_gates,\n"
        )
        if qk:
            changed = changed.replace(
                "p_mixed + q_off", "prepared_qk + row * 4096 + q_off"
            )
            changed = changed.replace(
                "p_mixed + k_off", "prepared_qk + row * 4096 + k_off"
            )
            changed = changed.replace(
                "        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)\n", ""
            )
            changed = changed.replace(
                "        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)\n", ""
            )
            changed = changed.replace("        b_q = b_q * scale\n", "")
        if gates:
            start = changed.index("        a_val = tl.load(a + row * stride_a")
            end = changed.index("        b_v -= tl.sum", start)
            changed = (
                changed[:start]
                + "        beta_val = tl.load(prepared_gates + row * 96 + 48 + i_hv)\n        b_h *= tl.load(prepared_gates + row * 96 + i_hv)\n"
                + changed[end:]
            )
        p = args.output / (name + ".py")
        p.write_text((source if name == "control" else changed) + PREPARE)
        spec = importlib.util.spec_from_file_location("gdn_prepare_" + name, p)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = (module, qk, gates, warps)
    mixed = torch.empty((8, 10240), dtype=torch.bfloat16, device="cuda")
    a = torch.empty((8, 48), dtype=torch.bfloat16, device="cuda")
    b = torch.empty_like(a)
    alog = torch.randn(48, device="cuda") * 0.3
    bias = torch.randn(48, dtype=torch.bfloat16, device="cuda")
    initial = torch.empty((48, 128, 128), device="cuda")
    qk = torch.empty((8, 4096), device="cuda")
    gates = torch.empty((8, 96), device="cuda")
    outputs = {
        n: torch.empty((8, 48, 128), dtype=torch.bfloat16, device="cuda")
        for n in modules
    }
    states = {n: torch.empty((8, 48, 128, 128), device="cuda") for n in modules}
    metadata = torch.ones(16, dtype=torch.int32, device="cuda")
    initial_pool = [initial]
    state_pools = {name: [state] for name, state in states.items()}
    report = {
        "source_sha256": expected,
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": "RUNNING",
        "results": {
            n: {"rows": 0, "state_differences": 0, "output_differences": 0}
            for n in modules
        },
    }

    def save():
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    def launch(name, slot=0):
        module, use_qk, use_gates, warps = modules[name]
        if use_qk:
            module.prepare_qk[(8, 16)](mixed, qk, 10240, 128**-0.5, num_warps=warps)
        if use_gates:
            module.prepare_gates[(8,)](a, b, alog, bias, gates, 48, 48, num_warps=4)
        inputs = [mixed, a, b, alog, bias, initial_pool[slot]]
        if name != "control":
            inputs.extend((qk, gates))
        bv, recurrence_warps, occupancy = geometries.get(name, (32, 4, 0))
        options = {"waves_per_eu": occupancy} if occupancy else {}
        module.stock_gdn_scan_kernel[(128 // bv, 48, 1)](
            *inputs,
            outputs[name],
            state_pools[name][slot],
            metadata,
            metadata,
            metadata,
            8,
            128**-0.5,
            stride_qkv=10240,
            stride_a=48,
            stride_b=48,
            H=16,
            HV=48,
            K=128,
            V=128,
            BK=128,
            BV=bv,
            SAVE_ROWS=True,
            INDEXED=False,
            stride_state=48 * 128 * 128,
            stride_index_seq=8,
            stride_index_row=1,
            num_warps=recurrence_warps,
            num_stages=1,
            **options,
        )

    save()
    try:
        for offset in range(0, args.rows, 8):
            mixed.normal_()
            a.normal_()
            b.normal_()
            initial.normal_(std=0.1)
            launch("control")
            for name in modules:
                if name != "control":
                    launch(name)
                record = report["results"][name]
                record["rows"] += 8
                record["state_differences"] += int(
                    (
                        states[name].view(torch.int32)
                        != states["control"].view(torch.int32)
                    ).sum()
                )
                record["output_differences"] += int(
                    (
                        outputs[name].view(torch.int16)
                        != outputs["control"].view(torch.int16)
                    ).sum()
                )
        fault = states["control"].clone()
        fault.view(torch.int32)[3, 11, 2, 57] ^= 1
        if torch.equal(fault, states["control"]):
            raise AssertionError("state corruption negative control missed")
        report["negative_control_detected"] = True
        good = [
            n
            for n, r in report["results"].items()
            if not r["state_differences"] and not r["output_differences"]
        ]
        # Rotate 192 MiB of state destinations per arm, exceeding GPU L2;
        # repeatedly updating one layer's cached state is not a model timing.
        initial_pool.extend(initial.clone() for _ in range(7))
        for name in good:
            state_pools[name].extend(torch.empty_like(states[name]) for _ in range(7))
        report["timing_state_destination_bytes_per_arm"] = (
            states["control"].numel() * 4 * 8
        )
        graphs = {}
        for name in good:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for step in range(16):
                    launch(name, step % 8)
            graphs[name] = graph
        times = {name: [] for name in good}
        for _ in range(5):
            for graph in graphs.values():
                graph.replay()
        for trial in range(9):
            for name in good if trial % 2 == 0 else good[::-1]:
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                for _ in range(16):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                times[name].append(start.elapsed_time(end) / 256)
        for name, values in times.items():
            report["results"][name]["median_ms"] = statistics.median(values)
            report["results"][name]["samples_ms"] = values
        report["status"] = "OPERATOR_SAMPLES_RECORDED"
        save()
        print(json.dumps(report), flush=True)
    except Exception:
        report["status"] = "FAILED"
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=320)
    parser.add_argument("--geometry", action="store_true")
    args = parser.parse_args()
    if args.rows <= 0 or args.rows % 8:
        parser.error("rows must be a positive multiple of eight")
    main(args)

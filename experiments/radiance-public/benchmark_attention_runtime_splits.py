"""Compare the released FULL graph attention kernel and runtime partition against independent M1."""

import argparse
import ctypes
import json
import random
from pathlib import Path
from statistics import median

import torch
from attention_runtime_splits import load, sha
from stock_m1_attention import m1_splits
from stock_m1_attention_shared import R4DArgs, SharedM1Attention

from qwen_r9700_lab.diagnostic_contract import authenticate, seal


def benchmark(stock, candidate, captures, quick, alternative=None):
    parent = json.loads((stock / "build.json").read_bytes())
    authenticate(parent)
    if sha(stock / "native.so") != parent["files"]["native.so"]:
        raise ValueError("unqualified M1 attention binary")
    release, repaired = SharedM1Attention(stock), load(candidate)
    launches = {"released": release.launch, "repaired": repaired}
    if alternative is not None:
        launches["alternative"] = load(alternative)
    lib = ctypes.CDLL(str(stock / "native.so"))
    oracle = (lib.r4d_attn_decode_h256_gqa6_fp8kv, lib.r4d_attn_decode_h256_gqa6_bf16kv)
    for native in oracle:
        native.argtypes = [ctypes.POINTER(R4DArgs), ctypes.c_void_p]
        native.restype = ctypes.c_int
    torch.manual_seed(27049)
    stream = torch.cuda.current_stream().cuda_stream
    scratch = torch.full(
        (8 * 24 * 32 * release.partial_bytes + 2048,),
        0xA5,
        device="cuda",
        dtype=torch.uint8,
    )
    body = scratch[1024:-1024]

    def check(prefix, q, kv, table, scales, label, timed, prefill=None):
        total = prefix + 8
        lens = torch.arange(prefix + 1, total + 1, device="cuda", dtype=torch.int32)
        ref = torch.empty_like(q)
        targets = {name: torch.empty_like(q) for name in launches}
        kind = int(kv.dtype == torch.bfloat16)
        root = oracle[kind]

        def args(row, out, width, splits):
            return R4DArgs(
                q[row].data_ptr(),
                kv.data_ptr(),
                table.data_ptr(),
                lens[row + width - 1].data_ptr(),
                out[row].data_ptr(),
                scales[0].data_ptr(),
                scales[1].data_ptr(),
                0,
                body.data_ptr(),
                1,
                width,
                24,
                4,
                256,
                16,
                table.shape[1],
                kv.stride(0),
                kv.stride(1),
                1.0 / 16,
                splits,
                327952,
            )

        for row in range(8):
            if root(
                ctypes.byref(args(row, ref, 1, m1_splits(prefix + row + 1))), stream
            ):
                raise ValueError("M1 oracle launch failed")

        def run(fn, arg):
            if fn(ctypes.byref(arg), kind, stream):
                raise ValueError("shared attention launch failed")

        variants = {
            name: (fn, args(0, targets[name], 8, 32)) for name, fn in launches.items()
        }
        for fn, arg in variants.values():
            run(fn, arg)
        mismatches = {
            label: (out.view(torch.int16) != ref.view(torch.int16)).sum((1, 2)).tolist()
            for label, out in targets.items()
        }
        if prefill is not None:
            mismatches["captured_prefill"] = (
                (prefill.view(torch.int16) != ref.view(torch.int16))
                .sum((1, 2))
                .tolist()
            )
        if not torch.all(scratch[:1024].eq(0xA5)) or not torch.all(
            scratch[-1024:].eq(0xA5)
        ):
            raise ValueError("scratch guard modified")
        timings = {name: [] for name in variants}
        if timed:
            repetitions = 3 if quick else 10
            for i in range(9 if quick else 21):
                names = tuple(variants)
                order = names[i % len(names) :] + names[: i % len(names)]
                for name in order:
                    fn, arg = variants[name]
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    for _ in range(repetitions):
                        run(fn, arg)
                    end.record()
                    end.synchronize()
                    timings[name].append(start.elapsed_time(end) / repetitions)
        return {
            "prefix": prefix,
            "source": label,
            "kv_dtype": str(kv.dtype),
            "different_elements": mismatches,
            "median_ms": {k: median(v) for k, v in timings.items() if v},
            "samples_ms": timings,
        }

    cases = []
    for path in captures:
        d = torch.load(path, map_location="cpu", weights_only=True)
        n = d["length"]
        logical = d["kv_bytes"].view(torch.uint8).cuda()
        table = torch.arange(logical.shape[0], dtype=torch.int32, device="cuda")[None]
        scales = [
            torch.full((4,), float(d[k]), dtype=torch.float32, device="cuda")
            for k in ("k_scale", "v_scale")
        ]
        rows = {p: i for i, p in enumerate(d["positions"])}
        capture = "capture:" + sha(path)[:12]
        for p in (248, 255, 256, 264, 272, 997, 1001, 1008, 1016, 1024, 1072, 1080):
            if p + 8 > n or not all(i in rows for i in range(p, p + 8)):
                continue
            indices = [rows[i] for i in range(p, p + 8)]
            q = d["query"][indices].cuda().contiguous()
            expected = d["output"][indices].reshape_as(q).cuda().contiguous()
            cases.append(check(p, q, logical, table, scales, capture, False, expected))

    prefixes = [
        0,
        7,
        8,
        119,
        120,
        127,
        128,
        241,
        247,
        248,
        249,
        254,
        255,
        256,
        257,
        263,
        264,
        265,
        271,
        272,
        505,
        511,
        996,
        999,
        1000,
        1001,
        1002,
        1003,
        1004,
        1005,
        1006,
        1007,
        1008,
        1009,
        1016,
        1023,
        1024,
        1072,
        1080,
        59999,
        60000,
        199999,
        200000,
    ]
    pages = (max(prefixes) + 8 + 15) // 16
    kv = torch.empty((pages, 4, 16, 512), device="cuda", dtype=torch.uint8)
    for i in range(0, pages, 192):
        data = kv[i : i + 192]
        data.copy_(
            torch.randn(data.shape, device="cuda", dtype=torch.float16)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
    table = torch.randperm(pages, device="cuda", dtype=torch.int32)[None]
    scales = [
        torch.tensor(v, device="cuda", dtype=torch.float32)
        for v in ((0.5, 1, 2, 1.25), (1, 0.75, 2, 0.25))
    ]
    random.Random(518).shuffle(prefixes)
    for p in prefixes:
        q = torch.randn((8, 24, 256), dtype=torch.bfloat16, device="cuda")
        q[1:] *= 4
        cases.append(
            check(p, q, kv, table, scales, "seed27049", p in {256, 1024, 60000, 200000})
        )
    short_pages = 128
    bf16 = kv[:short_pages].view(torch.float8_e4m3fn).to(torch.bfloat16)
    layout = torch.randperm(short_pages, device="cuda", dtype=torch.int32)[None]
    for p in (128, 248, 256, 264, 996, 1000, 1004, 1008, 1016, 1024, 1080, 2000):
        q = torch.randn((8, 24, 256), dtype=torch.bfloat16, device="cuda")
        cases.append(check(p, q, bf16, layout, scales, "seed27049-bf16", False))
    return seal(
        {
            "schema": "coherence-M8-dynamic-split-qualification-v1",
            "driver_sha256": sha(Path(__file__)),
            "device": torch.cuda.get_device_name(),
            "torch_version": str(torch.__version__),
            "hip_version": torch.version.hip,
            "stock": parent["sha256"],
            "repair": json.loads((candidate / "build.json").read_bytes())["sha256"],
            "alternative": json.loads((alternative / "build.json").read_bytes())[
                "sha256"
            ]
            if alternative
            else None,
            "alternative_differences": sum(
                sum(c["different_elements"]["alternative"]) for c in cases
            )
            if alternative
            else None,
            "cases": cases,
            "query_elements": 8 * 24 * 256 * len(cases),
            "repaired_differences": sum(
                sum(c["different_elements"]["repaired"]) for c in cases
            ),
            "released_differences": sum(
                sum(c["different_elements"]["released"]) for c in cases
            ),
            "captured_prefill_differences": sum(
                sum(c["different_elements"].get("captured_prefill", ())) for c in cases
            ),
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stock", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--alternative", type=Path)
    p.add_argument("--capture", action="append", type=Path, default=[])
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = benchmark(
        args.stock, args.candidate, args.capture, args.quick, args.alternative
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}))

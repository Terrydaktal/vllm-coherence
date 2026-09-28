"""Compare spatial GDN scan tiles while retaining chronological arithmetic."""

import hashlib
import json
import random
import statistics
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import seal, write_private


def run(options):
    import torch
    from stock_gdn_scan_kernel import stock_gdn_scan_kernel

    torch.manual_seed(902730)
    order = random.Random(902730)
    report = {
        "schema": "urn:coherence:prefill-scan-speed:v1",
        "status": "RUNNING",
        "cases": [],
        "sources": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("benchmark_prefill_scan.py", "prepared_prefill_scan.py")
        },
    }
    jobs = [
        (kind, rows)
        for kind in options.get("patterns", ["random"])
        for rows in options["rows"]
    ]
    jobs += [(str(path), None) for path in options.get("captures", [])]
    for kind, rows in jobs:
        if rows is None:
            data = torch.load(kind, weights_only=True)
            rows = data["packed"].shape[0]
        else:
            data = None
        initial = torch.randn((48, 128, 128), device="cuda") * 0.01
        qkv = (torch.randn((rows, 10240), device="cuda") * 0.2).bfloat16()
        a = torch.randn((rows, 48), device="cuda").bfloat16()
        b = torch.randn_like(a)
        a_log = torch.randn(48, device="cuda") * 0.1
        dt = torch.randn(48, device="cuda").bfloat16()
        if kind == "gates":
            a.copy_(
                torch.tensor(
                    [-104, -90, -30, -20, -15, -1, 0, 19.875, 20, 20.125, 30, 80],
                    device="cuda",
                ).repeat(rows, 4)
            )
            b.copy_(
                torch.tensor([-80, -15, -1, 0, 1, 15, 80, 2], device="cuda").repeat(
                    rows, 6
                )
            )
            dt.zero_()
        elif kind == "tiny":
            qkv[:, :4096] *= 1e-15
            qkv[::2, :2048] = 0
        elif kind == "large":
            qkv *= 80
            initial *= 100
        elif data is not None:
            initial, qkv, a, b, a_log, dt = (
                data[key].cuda()
                for key in ("initial", "packed", "a", "b", "a_log", "dt_bias")
            )
        elif kind != "random":
            raise ValueError("unknown numerical pattern")
        outputs = {
            tile: (
                torch.empty((rows, 48, 128), device="cuda", dtype=torch.bfloat16),
                torch.empty((1, 48, 128, 128), device="cuda"),
            )
            for tile in options["tiles"]
        }

        def launch(
            tile,
            outputs=outputs,
            qkv=qkv,
            a=a,
            b=b,
            a_log=a_log,
            dt=dt,
            initial=initial,
            rows=rows,
        ):
            out, state = outputs[tile]
            stock_gdn_scan_kernel[(128 // tile, 48)](
                qkv,
                a,
                b,
                a_log,
                dt,
                initial,
                out,
                state,
                None,
                None,
                None,
                rows,
                128**-0.5,
                qkv.stride(0),
                a.stride(0),
                b.stride(0),
                H=16,
                HV=48,
                K=128,
                V=128,
                BK=128,
                BV=tile,
                SAVE_ROWS=False,
                INDEXED=False,
                stride_state=0,
                stride_index_seq=0,
                stride_index_row=0,
                num_warps=1,
                num_stages=3,
                enable_fp_fusion=True,
                allow_flush_denorm=False,
            )

        for tile in outputs:
            launch(tile)
        reference = outputs[32]
        if options.get("prepared"):
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "prepared_prefill_candidate",
                Path(__file__).with_name("prepared_prefill_scan.py"),
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            prepared_launch = module.run
            native_launch = launch

            def launch(
                tile,
                initial=initial,
                qkv=qkv,
                a=a,
                b=b,
                a_log=a_log,
                dt=dt,
                native_launch=native_launch,
                outputs=outputs,
                prepared_launch=prepared_launch,
            ):
                if isinstance(tile, str):
                    geometry = [int(v) for v in tile.split("-")[1:]]
                    outputs[tile] = prepared_launch(
                        initial,
                        qkv,
                        a,
                        b,
                        a_log,
                        dt,
                        tile=geometry[0],
                        **(
                            dict(
                                zip(
                                    ("warps", "pipeline", "batch", "transposed")[
                                        : len(geometry) - 1
                                    ],
                                    geometry[1:],
                                    strict=True,
                                )
                            )
                            if len(geometry) > 1
                            else {}
                        ),
                    )
                else:
                    native_launch(tile)

            for tile in options.get("prepared_tiles", [8, 16, 32]):
                launch(f"prepared-{tile}")
            for geometry in options.get("prepared_geometries", []):
                launch("prepared-" + "-".join(str(v) for v in geometry))
        cases = {}
        for tile, actual in outputs.items():
            cases[str(tile)] = {
                "output_different": int(
                    torch.count_nonzero(
                        actual[0].view(torch.int16) != reference[0].view(torch.int16)
                    ).item()
                ),
                "state_different": int(
                    torch.count_nonzero(
                        actual[1].view(torch.int32) != reference[1].view(torch.int32)
                    ).item()
                ),
                "output_elements": actual[0].numel(),
                "state_elements": actual[1].numel(),
                "finite": all(bool(torch.isfinite(t).all()) for t in actual),
            }
            if (
                cases[str(tile)]["output_different"]
                or cases[str(tile)]["state_different"]
                or not cases[str(tile)]["finite"]
            ):
                report["cases"].append({"pattern": kind, "rows": rows, "tiles": cases})
                report["status"] = "MISMATCH"
                write_private(Path(options["output"]), seal(report))
                raise RuntimeError("spatial scan tile changed arithmetic")
        samples = {tile: [] for tile in outputs}
        for _ in range(options["repeats"]):
            tiles = list(outputs)
            order.shuffle(tiles)
            for tile in tiles:
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                launch(tile)
                end.record()
                end.synchronize()
                samples[tile].append(start.elapsed_time(end))
        report["cases"].append(
            {
                "rows": rows,
                "pattern": Path(kind).parent.parent.name if data is not None else kind,
                "tiles": cases,
                "samples_ms": {str(tile): v for tile, v in samples.items()},
                "median_ms": {
                    str(tile): statistics.median(v) for tile, v in samples.items()
                },
            }
        )
    report["status"] = "SAMPLE_CHECKED"
    report = seal(report)
    write_private(Path(options["output"]), report)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--rows", nargs="+", type=int, default=[65, 672, 976, 1648, 2048]
    )
    parser.add_argument("--tiles", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--prepared", action="store_true")
    print(json.dumps(run(vars(parser.parse_args()))))

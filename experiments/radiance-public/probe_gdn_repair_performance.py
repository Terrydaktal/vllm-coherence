"""Measure the qualified GDN correction with random tensors and fixed buffers.

This is an isolated kernel timing experiment, not an end-to-end token-rate test.
It must run between servers, never alongside a measured model generation.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import statistics
import time
from pathlib import Path


def probe(root: Path):
    import radiance_gdn as native
    import r4d
    import torch

    torch.set_grad_enabled(False)
    torch.manual_seed(149314)
    manifest = json.loads((root / "manifest.json").read_text())
    profile = json.loads((root / "production-profile.json").read_text())
    library_path = root / "gdn_extreme_decay_reference.so"
    assert hashlib.sha256(Path(r4d.__file__).read_bytes()).hexdigest() == profile["kernel_hashes"]["r4d.so"]
    assert hashlib.sha256(library_path.read_bytes()).hexdigest() == manifest["gdn_repair_sha256"]
    assert native.ENABLED and native.CHUNK == 64
    # This process imports the unpatched image. The comparison explicitly adds
    # the qualified correction, without changing module state between samples.
    assert not hasattr(native, "_qwen_gdn_original_scan")
    library = ctypes.CDLL(str(library_path))
    repair = library.qwen_gdn_extreme_decay_reference
    repair.argtypes = ([ctypes.c_void_p] * 9 + [ctypes.c_int] * 4
                       + [ctypes.c_float] * 2 + [ctypes.c_void_p])
    repair.restype = ctypes.c_int
    heads, query_heads, width = 48, 16, 128
    scale = width ** -0.5
    started = time.monotonic()
    rows = []
    for tokens in (8, 64, 512, 2048):
        for affected_heads in (0, 1, 8, 48):
            q = torch.nn.functional.normalize(torch.randn(tokens, query_heads, width,
                                                         device="cuda"), dim=-1).bfloat16()
            k = torch.nn.functional.normalize(torch.randn_like(q.float()), dim=-1).bfloat16()
            v = torch.randn(tokens, heads, width, device="cuda", dtype=torch.bfloat16)
            steps = torch.full((tokens, heads), -0.02, device="cuda")
            # Ensure the selected heads cross the repair threshold, including
            # short sequences; the required span is already covered by the
            # independent mathematical correctness probe.
            steps[:, :affected_heads] = -32.0
            cumulative = torch.cat([steps[i:i + 64].cumsum(0)
                                    for i in range(0, tokens, 64)]).contiguous()
            beta = torch.full_like(steps, 0.5)
            cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
            initial = torch.randn(1, heads, width, width, device="cuda") * 0.01
            matrix = native.kkt_solve(k, beta, cumulative, cu, 1, tokens, heads, query_heads)
            output = torch.empty_like(v)
            final = torch.empty_like(initial)
            scan_args = (q.data_ptr(), k.data_ptr(), v.data_ptr(), matrix.data_ptr(),
                         cumulative.data_ptr(), beta.data_ptr(), initial.data_ptr(),
                         output.data_ptr(), final.data_ptr(), cu.data_ptr(), 1, heads,
                         query_heads, width, width, 64, scale)
            repair_args = (q.data_ptr(), k.data_ptr(), v.data_ptr(), cumulative.data_ptr(),
                           beta.data_ptr(), initial.data_ptr(), output.data_ptr(),
                           final.data_ptr(), cu.data_ptr(), 1, heads, query_heads,
                           64, scale, 128.0)

            def launch(corrected):
                active_stream = torch.cuda.current_stream().cuda_stream
                native._CHUNK_SCAN(*scan_args, active_stream)
                if corrected and repair(*repair_args, active_stream):
                    raise RuntimeError("qualified GDN correction launch failed")

            for _ in range(5):
                launch(False)
                launch(True)
            torch.cuda.synchronize()
            graphs = {}
            for corrected in (False, True):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    launch(corrected)
                graphs[corrected] = graph
            measured = {False: [], True: []}
            # Alternate ordering to avoid assigning warm-up or clock drift to
            # one side. Graph replay excludes Python's extra ctypes-call cost;
            # the result is explicitly a device-work comparison.
            for sample in range(30):
                for corrected in ((False, True) if sample % 2 == 0 else (True, False)):
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                    graphs[corrected].replay()
                    end.record()
                    end.synchronize()
                    measured[corrected].append(begin.elapsed_time(end))
            launch(True)
            torch.cuda.synchronize()
            assert bool(torch.isfinite(output).all() and torch.isfinite(final).all())
            original = statistics.median(measured[False])
            corrected = statistics.median(measured[True])
            rows.append({"tokens": tokens, "heads": heads,
                         "affected_heads": affected_heads, "samples_each": 30,
                         "original_median_ms": original,
                         "corrected_median_ms": corrected,
                         "extra_median_ms": corrected - original,
                         "original_p95_ms": sorted(measured[False])[28],
                         "corrected_p95_ms": sorted(measured[True])[28]})
    report = {"complete": True, "elapsed_seconds": time.monotonic() - started,
              "method": "alternating GPU event timings of captured scan/corrected-scan",
              "scope": "one GDN layer; fixed random tensors; excludes Python overhead, other layers, and decoding",
              "repair_sha256": manifest["gdn_repair_sha256"], "cases": rows}
    (root / "gdn-repair-performance.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    probe(parser.parse_args().root)

"""Compare native GDN prefill with an independent FP64 sequential recurrence.

Random tensors only. This diagnoses numerical behavior, including extreme decay
spans; it never opens a model checkpoint or a conversation. A completed report
can contain numerical failures without making the diagnostic process fail.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import time
from pathlib import Path


def sequential_reference(q, k, v, steps, beta, initial, boundaries, scale):
    """Independent FP64 recurrence shared by synthetic and captured-input replay.

    This is a numerical diagnostic oracle, not a bit-exact full-model contract.
    It never uses the native WY factorization, matrix solve or scan output.
    """
    import torch

    heads, query_heads = v.shape[-2], q.shape[-2]
    qr = q.double().repeat_interleave(heads // query_heads, dim=1)
    kr = k.double().repeat_interleave(heads // query_heads, dim=1)
    vr, br, gr = v.double(), beta.double(), steps.double().exp()
    oracle = torch.empty_like(vr)
    oracle_final = torch.empty_like(initial, dtype=torch.float64)
    for sequence, (begin, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        state = initial[sequence].double().clone()
        for position in range(begin, end):
            state *= gr[position, :, None, None]
            residual = vr[position] - (state * kr[position, :, None, :]).sum(-1)
            state += (br[position, :, None] * residual)[:, :, None] * kr[position, :, None, :]
            oracle[position] = (state * qr[position, :, None, :]).sum(-1) * scale
        oracle_final[sequence] = state
    return oracle, oracle_final


def probe(root: Path):
    import torch
    import radiance_gdn as native
    import r4d

    torch.set_grad_enabled(False)
    torch.manual_seed(14929)
    device = "cuda"
    heads, query_heads, width = 48, 16, 128
    scale = width ** -0.5
    expected = json.loads((root / "production-profile.json").read_text())["kernel_hashes"]["r4d.so"]
    observed = hashlib.sha256(Path(r4d.__file__).read_bytes()).hexdigest()
    assert observed == expected and native.ENABLED
    manifest = json.loads((root / "manifest.json").read_text())
    repair = None
    if manifest.get("gdn_repair_sha256"):
        library_path = root / "gdn_extreme_decay_reference.so"
        assert hashlib.sha256(library_path.read_bytes()).hexdigest() == manifest["gdn_repair_sha256"]
        library = ctypes.CDLL(str(library_path))
        repair = library.qwen_gdn_extreme_decay_reference
        repair.argtypes = ([ctypes.c_void_p] * 9 + [ctypes.c_int] * 4
                           + [ctypes.c_float] * 2 + [ctypes.c_void_p])
        repair.restype = ctypes.c_int
    started = time.monotonic()
    rows = []
    cases = [(65, 0.02, 1, [65], False), (128, 1.0, 1, [128], False),
             (128, 2.5, 1, [128], False), (128, 3.2, 1, [128], False),
             (128, 8.0, 1, [128], False), (257, 0.02, 1, [257], False),
             (257, 3.2, 1, [257], False), (1024, 0.02, 1, [1024], False),
             (128, 32.0, 1, [128], False), (128, 2.5, 10000, [128], False),
             (195, 0.02, 1, [65, 130], False), (195, 3.2, 1, [65, 130], True)]
    for tokens, decay, amplitude, lengths, mixed_heads in cases:
        q = torch.randn(tokens, query_heads, width, device=device)
        k = torch.randn_like(q)
        q = torch.nn.functional.normalize(q, dim=-1).to(torch.bfloat16)
        k = torch.nn.functional.normalize(k, dim=-1).to(torch.bfloat16)
        v = torch.randn(tokens, heads, width, device=device, dtype=torch.bfloat16)
        v *= amplitude
        steps = torch.full((tokens, heads), -decay, device=device, dtype=torch.float32)
        if mixed_heads:
            steps[:, 1:-1] = -0.02
        beta = torch.full_like(steps, 0.5)
        boundaries = [0]
        for length in lengths:
            boundaries.append(boundaries[-1] + length)
        assert boundaries[-1] == tokens
        cumulative = torch.cat([
            steps[i:min(i + native.CHUNK, end)].cumsum(0)
            for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True)
            for i in range(begin, end, native.CHUNK)
        ], dim=0).contiguous()
        cu = torch.tensor(boundaries, device=device, dtype=torch.int32)
        initial = torch.randn(len(lengths), heads, width, width, device=device) * 0.01
        matrix = native.kkt_solve(k, beta, cumulative, cu, len(lengths), tokens, heads, query_heads)
        # Guard the caller-owned output, including non-full final chunks.
        count = v.numel()
        slab = torch.full((count + 512,), 37.0, device=device, dtype=torch.bfloat16)
        output = slab[256:256 + count].view(1, tokens, heads, width)
        actual, final = native.fused_prefill(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), matrix.unsqueeze(0),
            cumulative.unsqueeze(0), beta.unsqueeze(0), scale, initial, True, cu,
            None, out=output,
        )
        torch.cuda.synchronize()
        # No WY factorization, chunk scan, kernel output or native intermediate
        # participates in this reference. It follows the defining recurrence.
        oracle, oracle_final = sequential_reference(q, k, v, steps, beta, initial, boundaries, scale)
        out_error = ((actual[0].double() - oracle).norm() / oracle.norm()).item()
        state_error = ((final.double() - oracle_final).norm() / oracle_final.norm()).item()
        finite = bool(torch.isfinite(actual).all() and torch.isfinite(final).all())
        guards = bool((slab[:256] == 37).all() and (slab[-256:] == 37).all())
        row = {"tokens": tokens, "heads": heads, "query_heads": query_heads,
               "sequence_lengths": lengths, "mixed_head_decay": mixed_heads,
               "value_amplitude": amplitude,
               "negative_log_decay_per_token": decay,
               "maximum_chunk_decay_span": min(tokens - 1, 63) * decay,
               "output_relative_error": out_error if math.isfinite(out_error) else None,
               "state_relative_error": state_error if math.isfinite(state_error) else None,
               "finite": finite, "guards_intact": guards,
               "within_one_percent": finite and guards and max(out_error, state_error) < 0.01}
        if repair is not None:
            original_output, original_final = actual.clone(), final.clone()
            error = repair(
                q.data_ptr(), k.data_ptr(), v.data_ptr(), cumulative.data_ptr(),
                beta.data_ptr(), initial.data_ptr(), actual.data_ptr(), final.data_ptr(),
                cu.data_ptr(), len(lengths), heads, query_heads, native.CHUNK, scale, 128.0,
                torch.cuda.current_stream().cuda_stream,
            )
            assert error == 0, "native numerical repair launch failed"
            torch.cuda.synchronize()
            repaired_out = ((actual[0].double() - oracle).norm() / oracle.norm()).item()
            repaired_state = ((final.double() - oracle_final).norm() / oracle_final.norm()).item()
            repaired_finite = bool(torch.isfinite(actual).all() and torch.isfinite(final).all())
            repaired_guards = bool((slab[:256] == 37).all() and (slab[-256:] == 37).all())
            row["experimental_repair"] = {
                "output_relative_error": repaired_out if math.isfinite(repaired_out) else None,
                "state_relative_error": repaired_state if math.isfinite(repaired_state) else None,
                "finite": repaired_finite, "guards_intact": repaired_guards,
                "within_one_percent": repaired_finite and repaired_guards
                and max(repaired_out, repaired_state) < 0.01,
                "unchanged_below_threshold": (bool(torch.equal(actual, original_output)
                    and torch.equal(final, original_final))
                    if row["maximum_chunk_decay_span"] <= 128 else None),
                "ordinary_heads_unchanged": (bool(torch.equal(actual[:, :, 1:-1], original_output[:, :, 1:-1])
                    and torch.equal(final[:, 1:-1], original_final[:, 1:-1]))
                    if mixed_heads else None),
            }
        rows.append(row)
        (root / "gdn-probe.json").write_text(json.dumps({"complete": False, "cases": rows}, indent=2))
        print(json.dumps(row), flush=True)
    graph_report = None
    if repair is not None:
        # The last case has two unequal sequence lengths and mixed affected /
        # unaffected heads. Capture both launches and replay with changed q data.
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            graph_output, graph_final = native.fused_prefill(
                q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), matrix.unsqueeze(0),
                cumulative.unsqueeze(0), beta.unsqueeze(0), scale, initial, True, cu,
                None, out=output,
            )
            assert repair(q.data_ptr(), k.data_ptr(), v.data_ptr(), cumulative.data_ptr(),
                beta.data_ptr(), initial.data_ptr(), graph_output.data_ptr(), graph_final.data_ptr(),
                cu.data_ptr(), len(lengths), heads, query_heads, native.CHUNK, scale, 128.0,
                torch.cuda.current_stream().cuda_stream) == 0
        q.mul_(-1)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        graph_out_error = ((graph_output[0].double() + oracle).norm() / oracle.norm()).item()
        graph_state_error = ((graph_final.double() - oracle_final).norm() / oracle_final.norm()).item()
        graph_report = {"replays": 3, "changed_query_inputs": True,
                        "output_relative_error": graph_out_error,
                        "state_relative_error": graph_state_error,
                        "within_one_percent": max(graph_out_error, graph_state_error) < 0.01}
    report = {"complete": True, "reference": "independent FP64 sequential gated delta recurrence",
              "r4d_sha256": observed, "elapsed_seconds": time.monotonic() - started,
              "all_within_one_percent": all(row["within_one_percent"] for row in rows),
              "experimental_repair_sha256": manifest.get("gdn_repair_sha256"),
              "graph_capture": graph_report, "cases": rows}
    (root / "gdn-probe.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = probe(args.root)
        print(json.dumps({key: value for key, value in result.items() if key != "cases"}))
        if result["experimental_repair_sha256"] and any(
            not row["experimental_repair"]["within_one_percent"]
            or row["experimental_repair"]["unchanged_below_threshold"] is False
            or row["experimental_repair"]["ordinary_heads_unchanged"] is False
            for row in result["cases"]
        ):
            raise SystemExit(2)
        if result["experimental_repair_sha256"] and not result["graph_capture"]["within_one_percent"]:
            raise SystemExit(2)
    except Exception as error:
        print(json.dumps({"error_type": type(error).__name__}))
        raise SystemExit(1) from None

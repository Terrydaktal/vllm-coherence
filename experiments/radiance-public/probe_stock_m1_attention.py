"""Check R4D query batching and rejected-query noninterference with synthetic KV."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import DiagnosticError, seal, write_private


def qualify(args):
    spec = json.loads(args.spec.read_text())
    os.environ.update(spec["environment"])
    import torch  # isort: skip

    import r4d
    import radiance_r4d_attn as attention
    from probe_stock_gdn_sequence import compare_tensors

    wrapper_hash = hashlib.sha256(Path(attention.__file__).read_bytes()).hexdigest()
    if wrapper_hash != spec["binding"]["files"]["radiance_r4d_attn.py"]:
        raise DiagnosticError("attention wrapper differs from pinned qualification")
    torch.manual_seed(71439)
    heads, kv_heads, dim, block, width, maximum = 24, 4, 256, 16, 8, 253792
    max_blocks = (maximum + block - 1) // block
    needed = r4d.attn_decode_h256_gqa6_scratch_bytes(
        1, attention.MAX_DECODE_QLEN, heads, kv_heads, dim, maximum, 0
    )
    scratch = torch.empty(needed, device="cuda", dtype=torch.uint8)
    checks = []
    for dtype_index, dtype in enumerate((torch.float8_e4m3fn, torch.bfloat16)):
        for prefix in (129, 2049, 8193, 60001):
            blocks = (prefix + width + block - 1) // block
            kv = torch.randn((blocks, kv_heads, block, 2 * dim), device="cuda").to(dtype)
            table = torch.zeros((1, max_blocks), device="cuda", dtype=torch.int32)
            table[0, :blocks] = torch.arange(blocks, device="cuda", dtype=torch.int32)
            lengths = torch.empty((1,), device="cuda", dtype=torch.int32)
            query = torch.randn((width, heads, dim), device="cuda").bfloat16()

            def launch(
                q,
                length,
                bound,
                *,
                kv=kv,
                table=table,
                lengths=lengths,
                function=attention._DECODE[dtype_index],
            ):
                lengths.fill_(length)
                out = torch.empty_like(q)
                function(
                    q.data_ptr(),
                    kv.data_ptr(),
                    table.data_ptr(),
                    lengths.data_ptr(),
                    out.data_ptr(),
                    0,
                    0,
                    scratch.data_ptr(),
                    1,
                    len(q),
                    heads,
                    kv_heads,
                    dim,
                    block,
                    max_blocks,
                    kv.stride(0),
                    kv.stride(1),
                    dim**-0.5,
                    0,
                    bound,
                    torch.cuda.current_stream().cuda_stream,
                )
                return out

            for bound_mode in ("actual", "graph"):
                serial = torch.cat(
                    [
                        launch(
                            query[i : i + 1],
                            prefix + i + 1,
                            maximum if bound_mode == "graph" else prefix + i + 1,
                        )
                        for i in range(width)
                    ]
                )
                bound = maximum if bound_mode == "graph" else prefix + width
                batch = launch(query, prefix + width, bound)
                label = f"{dtype}-prefix-{prefix}-{bound_mode}"
                checks.append(compare_tensors(label + "-m8-vs-m1", batch, serial, args.output))

                def independent(
                    q,
                    *,
                    bound=bound,
                    table=table,
                    prefix=prefix,
                    kv=kv,
                    function=attention._DECODE[dtype_index],
                ):
                    rows = len(q)
                    # Keep the M1 split law while expressing each query as an
                    # independent logical sequence sharing the immutable KV.
                    tiles = (bound + 15) // 16
                    splits = 32
                    while splits > 16 and splits > tiles // 2:
                        splits //= 2
                    splits = max(1, min(splits, tiles))
                    out = torch.empty_like(q)
                    repeated_table = table.expand(rows, -1).contiguous()
                    row_lengths = torch.arange(
                        prefix + 1, prefix + rows + 1, device="cuda", dtype=torch.int32
                    )
                    function(
                        q.data_ptr(),
                        kv.data_ptr(),
                        repeated_table.data_ptr(),
                        row_lengths.data_ptr(),
                        out.data_ptr(),
                        0,
                        0,
                        scratch.data_ptr(),
                        rows,
                        1,
                        heads,
                        kv_heads,
                        dim,
                        block,
                        max_blocks,
                        kv.stride(0),
                        kv.stride(1),
                        dim**-0.5,
                        splits,
                        bound,
                        torch.cuda.current_stream().cuda_stream,
                    )
                    return out

                independent_output = independent(query)
                checks.append(
                    compare_tensors(
                        label + "-independent-query-vs-m1", independent_output, serial, args.output
                    )
                )
                for factor in (0.0, 16.0):
                    changed = query.clone()
                    changed[1:] = (changed[1:].float() * factor).bfloat16()
                    result = launch(changed, prefix + width, bound)
                    checks.append(
                        compare_tensors(
                            label + f"-future-query-{factor}", result[:1], batch[:1], args.output
                        )
                    )
                    checks.append(
                        compare_tensors(
                            label + f"-independent-future-query-{factor}",
                            independent(changed)[:1],
                            independent_output[:1],
                            args.output,
                        )
                    )
            del kv, query, table
    return {
        "checks": checks,
        "wrapper_sha256": wrapper_hash,
        "library_sha256": hashlib.sha256(Path(r4d.__file__).read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    if not args.allow_gpu or not os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"):
        raise DiagnosticError("attention diagnostic requires admission and shared GPU lease")
    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease

    os.umask(0o077)
    args.output.mkdir(mode=0o700)
    result, error = {}, None
    try:
        with gpu_lease(args.output / "gpu-lease"):
            result = qualify(args)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    report = seal(
        {
            "status": "MEASURED" if error is None else "FAILED",
            "error": error,
            **result,
            "scope": (
                "Synthetic M1/M8 attention; four prefixes and two KV formats, eager/graph bounds."
            ),
            "formal_equivalence": "UNPROVED",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    write_private(args.output / "probe-result.json", report)
    print(json.dumps({k: report[k] for k in ("status", "error", "sha256")}), flush=True)
    return 0 if error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())

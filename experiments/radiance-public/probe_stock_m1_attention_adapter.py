"""Native eager/boundary/graph checks of the actual independent-query adapter.

All inputs are synthetic. A passing result applies to these invocations only.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import MethodType, SimpleNamespace

from qwen_r9700_lab.conformance_instrumentation import HookSet
from qwen_r9700_lab.diagnostic_contract import DiagnosticError, seal, write_private


def qualify(args):
    spec = json.loads(args.spec.read_text())
    os.environ.update(spec["environment"])
    import torch  # isort: skip

    import radiance_r4d_attn as native
    import stock_m1_attention as candidate
    from probe_stock_gdn_sequence import compare_tensors

    source = hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
    if source != spec["binding"]["files"]["radiance_r4d_attn.py"]:
        raise DiagnosticError("attention source differs from the pinned qualification")
    original = native.R4DAttentionImpl.forward
    maximum, heads, kv_heads, dim, block = 253792, 24, 4, 256, 16
    prefixes = (0, 15, 249, 1005, 2049, 60001)
    blocks = (max(prefixes) + 8 + block - 1) // block
    max_blocks = (maximum + block - 1) // block
    scratch_bytes = native.r4d.attn_decode_h256_gqa6_scratch_bytes(
        1, native.MAX_DECODE_QLEN, heads, kv_heads, dim, maximum, 0
    )
    # Interior scratch preserves alignment and exposes writes beyond capacity.
    guarded = torch.full((scratch_bytes + 1024,), 0xA5, device="cuda", dtype=torch.uint8)
    scratch = guarded[512:-512]
    checks = []
    hooks = HookSet()
    torch.manual_seed(829301)
    try:
        repair = candidate.StockM1Attention(hooks)
        for dtype_index, dtype in enumerate((torch.float8_e4m3fn, torch.bfloat16)):
            kv = torch.randn((blocks, kv_heads, block, 2 * dim), device="cuda").to(dtype)
            table = torch.zeros((1, max_blocks), device="cuda", dtype=torch.int32)
            table[0, :blocks] = torch.randperm(blocks, device="cuda", dtype=torch.int32)
            for scales in ((1.0, 1.0), (0.5, 1.5)):
                layer = SimpleNamespace(_k_scale_float=scales[0], _v_scale_float=scales[1])
                impl = SimpleNamespace(
                    num_heads=heads,
                    num_kv_heads=kv_heads,
                    head_size=dim,
                    scale=dim**-0.5,
                    _kv_geometry=None,
                    _descales=None,
                    _descale_len=4,
                )
                for name in ("_geometry", "_descale_ptrs"):
                    setattr(impl, name, MethodType(getattr(native.R4DAttentionImpl, name), impl))

                def metadata(width, bound, length, *, table=table):
                    return SimpleNamespace(
                        r4d_plan=((0, 1, width, 0),),
                        causal=True,
                        r4d_max_ctx=bound,
                        r4d_scratch=scratch,
                        block_table=table,
                        seq_lens=torch.tensor([length], device="cuda", dtype=torch.int32),
                    )

                def reference(q, prefix, graph_bound=False, *, impl=impl, layer=layer, kv=kv):
                    rows = []
                    for i in range(len(q)):
                        length = prefix + i + 1
                        md = metadata(1, maximum if graph_bound else length, length)
                        out = torch.empty_like(q[i : i + 1])
                        original(impl, layer, q[i : i + 1], None, None, kv, md, out)
                        rows.append(out)
                    return torch.cat(rows)

                def compare(label, out, ref):
                    checks.append(compare_tensors(label, out, ref, args.output))
                    guard = torch.cat((guarded[:512], guarded[-512:]))
                    checks.append(
                        compare_tensors(
                            label + "-scratch-canary",
                            guard,
                            torch.full_like(guard, 0xA5),
                            args.output,
                        )
                    )

                for width in range(1, 9):
                    label = f"dtype-{dtype_index}-scales-{scales[0]}-{scales[1]}-width-{width}"
                    q = torch.randn((width, heads, dim), device="cuda").bfloat16()
                    q[1:] *= 16  # Exercise the original wave-wide rescaling defect.
                    out = torch.empty_like(q)
                    for prefix in prefixes:
                        md = metadata(width, prefix + width, prefix + width)
                        kv_before = kv.clone()
                        native.R4DAttentionImpl.forward(impl, layer, q, None, None, kv, md, out)
                        if not torch.equal(kv.view(torch.uint8), kv_before.view(torch.uint8)):
                            raise DiagnosticError("eager attention mutated shared KV")
                        compare(label + f"-eager-prefix-{prefix}", out, reference(q, prefix))

                    # Capture once, then change query values, causal lengths and
                    # physical page order at stable addresses before each replay.
                    md = metadata(width, maximum, 15 + width)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(2):
                            native.R4DAttentionImpl.forward(impl, layer, q, None, None, kv, md, out)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        native.R4DAttentionImpl.forward(impl, layer, q, None, None, kv, md, out)
                    for iteration, prefix in enumerate((15, 1005, 2049, 60001)):
                        q.copy_(torch.randn_like(q))
                        q[1:] *= 16
                        md.seq_lens.fill_(prefix + width)
                        table[0, :blocks].copy_(
                            torch.randperm(blocks, device="cuda", dtype=torch.int32)
                        )
                        # Also change a KV block without changing its allocation.
                        kv[0].copy_(torch.randn(kv[0].shape, device="cuda").to(dtype))
                        kv_before = kv.clone()
                        graph.replay()
                        if not torch.equal(kv.view(torch.uint8), kv_before.view(torch.uint8)):
                            raise DiagnosticError("captured attention mutated shared KV")
                        compare(
                            label + f"-graph-replay-{iteration}-prefix-{prefix}",
                            out,
                            reference(q, prefix, graph_bound=True),
                        )
                    del graph
        receipt = repair.receipt()
    finally:
        hooks.close()
    return {
        "checks": checks,
        "dispatch": receipt,
        "wrapper_sha256": source,
        "adapter_sha256": hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(Path(native.r4d.__file__).read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    if not args.allow_gpu or not os.environ.get("QWEN_CONFORMANCE_GPU_LOCK"):
        raise DiagnosticError("adapter qualification requires admission and shared GPU lease")
    from qwen_r9700_lab.conformance_gpu_lease import gpu_lease

    os.umask(0o077)
    args.output.mkdir(mode=0o700)
    result, error = {}, None
    try:
        with gpu_lease(args.output / "gpu-lease"):
            result = qualify(args)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    equal = (
        error is None
        and bool(result.get("checks"))
        and all(row["equal"] for row in result["checks"])
    )
    report = seal(
        {
            "status": "TESTED" if equal else "FAILED_OR_DISCREPANT",
            "error": error,
            **result,
            "formal_equivalence": "UNPROVED",
            "scope": (
                "Synthetic native adapter; widths 1..8, FP8/BF16 KV, descales, "
                "split boundaries, shared-KV preservation and graph replay."
            ),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    write_private(args.output / "probe-result.json", report)
    print(json.dumps({k: report[k] for k in ("status", "error", "sha256")}), flush=True)
    return 0 if equal else 2


if __name__ == "__main__":
    raise SystemExit(main())

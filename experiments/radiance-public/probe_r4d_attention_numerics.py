"""Check native paged attention over long contexts against FP32 attention.

Random inputs only. Includes shuffled physical blocks and the larger context
bound used during graph capture, with BF16 and FP8 cache storage.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def run(root):
    profile = json.loads((root / "production-profile.json").read_text())
    os.environ.update(profile["kernel_environment"])
    import torch
    import r4d
    import radiance_r4d_attn as attention

    library_hash = hashlib.sha256(Path(r4d.__file__).read_bytes()).hexdigest()
    assert library_hash == profile["kernel_hashes"]["r4d.so"]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(1789373017)
    maximum = 253792
    heads, kv_heads, dim, block = 24, 4, 256, 16
    max_blocks = (maximum + block - 1) // block
    guard = 512
    needed = r4d.attn_decode_h256_gqa6_scratch_bytes(
        1, attention.MAX_DECODE_QLEN, heads, kv_heads, dim, maximum, 0)
    scratch = torch.full((needed + 2 * guard,), 17, device="cuda", dtype=torch.uint8)
    rows = []
    started = time.monotonic()
    for variant, dtype in enumerate((torch.float8_e4m3fn, torch.bfloat16)):
        for length in (127, 128, 129, 8191, 8192, 8193, 32767, 65536, 86142, 132739, 200000, 239265, maximum):
            blocks = (length + block - 1) // block
            logical = torch.randn((blocks, kv_heads, block, 2 * dim), device="cuda", dtype=torch.bfloat16).to(dtype)
            cache = torch.empty_like(logical)
            permutation = torch.randperm(blocks, device="cuda")
            cache.view(torch.uint8)[permutation] = logical.view(torch.uint8)
            table = torch.zeros((1, max_blocks), device="cuda", dtype=torch.int32)
            table[0, :blocks] = permutation.int()
            lengths = torch.tensor([length], device="cuda", dtype=torch.int32)
            for query_len in (1, 7, 8, 9, 16, 63, 64, 65, 128):
                if query_len > length:
                    continue
                query = torch.randn((query_len, heads, dim), device="cuda", dtype=torch.bfloat16)
                expected = torch.empty((query_len, heads, dim), device="cuda", dtype=torch.float32)
                key_positions = torch.arange(length, device="cuda")
                query_positions = torch.arange(length - query_len, length, device="cuda")
                mask = key_positions[None, :] > query_positions[:, None]
                for head in range(kv_heads):
                    keys = logical[:, head, :, :dim].reshape(-1, dim)[:length].float()
                    values = logical[:, head, :, dim:].reshape(-1, dim)[:length].float()
                    q = query[:, head * 6:(head + 1) * 6].transpose(0, 1).float()
                    logits = q @ keys.T / dim ** .5
                    logits.masked_fill_(mask[None], float("-inf"))
                    expected[:, head * 6:(head + 1) * 6] = (logits.softmax(-1) @ values).transpose(0, 1)
                    del keys, values, q, logits
                bounds = sorted({length, maximum}) if query_len <= attention.MAX_DECODE_QLEN else [maximum]
                for bound in bounds:
                    slab = torch.full((query_len * heads * dim + 2 * guard,), 42, device="cuda", dtype=torch.bfloat16)
                    out = slab[guard:-guard].view(query_len, heads, dim)
                    fn = attention._DECODE[variant] if query_len <= attention.MAX_DECODE_QLEN else attention._PREFILL[variant]
                    fn(query.data_ptr(), cache.data_ptr(), table.data_ptr(), lengths.data_ptr(),
                       out.data_ptr(), 0, 0, scratch[guard:].data_ptr(), 1, query_len,
                       heads, kv_heads, dim, block, max_blocks, cache.stride(0), cache.stride(1),
                       dim ** -.5, 0, bound, torch.cuda.current_stream().cuda_stream)
                    relative = float((out.float() - expected).norm() / expected.norm())
                    intact = bool((scratch[:guard] == 17).all() & (scratch[-guard:] == 17).all()
                                  & (slab[:guard] == 42).all() & (slab[-guard:] == 42).all())
                    row = {"cache_dtype": str(dtype), "context_tokens": length, "query_tokens": query_len,
                           "launch_context_bound": bound, "relative_error_vs_fp32": relative,
                           "guard_regions_intact": intact, "finite": bool(torch.isfinite(out).all())}
                    rows.append(row)
                    (root / "attention-probe.json").write_text(json.dumps({"rows": rows}, indent=2))
                    print(json.dumps(row), flush=True)
                    assert relative < .02 and intact and row["finite"]
                    del out, slab
                del query, expected, mask, key_positions, query_positions
            del logical, cache, permutation, table, lengths
    (root / "attention-probe.json").write_text(json.dumps({"rows": rows, "all_passed": True,
        "elapsed_seconds": time.monotonic() - started, "library_sha256": library_hash}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)

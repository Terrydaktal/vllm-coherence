"""Build an isolated candidates-only INT2 DFlash2 head from the pinned implementation."""

from pathlib import Path


def draft_head_source(stock):
    """Keep stock argument defaults for every consumer sharing this Triton kernel."""
    changes = (
        ("BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):",
         "BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, STORE_DENSE: tl.constexpr = True):"),
        ("        tl.store(Y + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16),\n"
         "                 mask=mask_n[None, :])",
         "        if STORE_DENSE:\n"
         "            tl.store(Y + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16),\n"
         "                     mask=mask_n[None, :])"),
        ("    y = torch.empty(M, n, dtype=torch.bfloat16, device=x.device)",
         "    w = getattr(lm_head, 'weight', None)\n"
         "    exact = w is not None and w.dtype in (torch.bfloat16, torch.float16) and w.shape == (n, k)\n"
         "    candidates_only = exact and getattr(self, '_radiance_topk_only', False)\n"
         "    output_rows = m if candidates_only else M\n"
         "    y = torch.empty(output_rows, n, dtype=torch.bfloat16, device=x.device)"),
        ("nblk, KCAND, G=GROUP, BLOCK_M=M, BLOCK_N=BLOCK_N, **_cfg_for(M))",
         "nblk, KCAND, G=GROUP, BLOCK_M=M, BLOCK_N=BLOCK_N, STORE_DENSE=not candidates_only, **_cfg_for(M))"),
        ('    w = getattr(lm_head, "weight", None)\n'
         "    if w is not None and w.dtype in (torch.bfloat16, torch.float16) and w.shape == (n, k):\n"
         "        idx = bi.gather(1, bm.topk(RERANK, dim=1).indices).contiguous()",
         "    if exact:\n"
         "        idx = bi[:output_rows].gather(1, bm[:output_rows].topk(RERANK, dim=1).indices).contiguous()"),
        ("ex = torch.empty(M, RERANK,", "ex = torch.empty(output_rows, RERANK,"),
        ("_rerank_exact[(M, RERANK)]", "_rerank_exact[(output_rows, RERANK)]"),
    )
    for before, after in changes:
        if stock.count(before) != 1:
            raise ValueError(f"unexpected draft head source near {before!r}")
        stock = stock.replace(before, after, 1)
    compile(stock, "radiance_drafthead.py", "exec")
    return stock


def write_draft_head(source, destination):
    """Write a standalone importable candidate without editing the installed module."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("write the candidate to a separate overlay")
    destination.write_text(draft_head_source(source.read_text()))

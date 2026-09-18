"""Isolated target-head benchmark hook; never imported by production launchers.

Only a benchmark-owned Python overlay imports this module. Captures are hidden
vectors in private tmpfs; reports contain counts, shapes and timings, not text.
The drafter is unchanged. Global variants are explicitly uncertified experiments.
"""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
from pathlib import Path


def global_head(state, lm_head, hidden, candidates):
    import radiance_drafthead as dh
    import torch

    x = hidden.reshape(-1, hidden.shape[-1])
    rows, width = x.shape
    padded = dh._pow2_at_least(rows)
    if padded != rows:
        x = torch.cat((x, x.new_zeros(padded - rows, width)))
    x = x.contiguous()
    vocab, blocks = state._radiance_n, state._radiance_nblk
    sums = x.reshape(padded, width // dh.GROUP, dh.GROUP).float().sum(-1).contiguous()
    logits = torch.empty(padded, vocab, dtype=torch.bfloat16, device=x.device)
    dummy_score = torch.empty(1, dtype=torch.float32, device=x.device)
    dummy_id = torch.empty(1, dtype=torch.int32, device=x.device)
    dh._draft_head_int2[(blocks,)](
        x,
        sums,
        state._radiance_wq,
        state._radiance_scale,
        state._radiance_zs,
        logits,
        dummy_score,
        dummy_id,
        width,
        vocab,
        state._radiance_wq.stride(0),
        state._radiance_scale.stride(0),
        sums.stride(0),
        blocks,
        0,
        G=dh.GROUP,
        BLOCK_M=padded,
        BLOCK_N=dh.BLOCK_N,
        **dh._cfg_for(padded),
    )
    ids = logits.topk(candidates, dim=-1).indices.to(torch.int32).contiguous()
    rescored = torch.empty(padded, candidates, dtype=torch.float32, device=x.device)
    dh._rerank_exact[(padded, candidates)](
        x,
        lm_head.weight,
        ids,
        rescored,
        width,
        lm_head.weight.stride(0),
        R=candidates,
        BLOCK_K=512,
        num_warps=4,
    )
    logits.fill_(-float("inf"))
    logits.scatter_(1, ids.long(), rescored.to(torch.bfloat16))
    return logits[:rows]


def apply_variant(state, head, hidden, bias, mode):
    if bias is not None:
        raise ValueError("head benchmark does not admit an embedding bias")
    if mode == "full":
        return state._radiance_exact_head(head, hidden, bias)
    if mode == "block80":
        return state._radiance_fast_head(head, hidden, bias)
    if mode in ("global128", "global256"):
        return global_head(state, head, hidden, int(mode.removeprefix("global")))
    raise ValueError("unsupported benchmark head mode")


def install():
    root = Path(os.environ["QWEN_PRIVATE_HEAD_ROOT"])
    if not str(root).startswith("/dev/shm/qwen-private-head-") or root.stat().st_mode & 0o077:
        raise ValueError("head capture root must be private tmpfs")

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "radiance_verifyhead":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            original = spec.loader

            class Loader(importlib.abc.Loader):
                def create_module(self, specification):
                    return original.create_module(specification)

                def exec_module(self, module):
                    original.exec_module(module)
                    original_gated = module._apply_head_gated
                    captured = 0
                    trial_stats = None

                    def gated(state, lm_head, hidden, embedding_bias=None):
                        nonlocal captured, trial_stats
                        import torch

                        if torch.cuda.is_current_stream_capturing():
                            return original_gated(state, lm_head, hidden, embedding_bias)
                        control = json.loads((root / "control.json").read_text())
                        mode = control["mode"]
                        trial = control.get("trial", "startup")
                        if trial_stats is None or trial_stats["trial"] != trial:
                            if trial_stats is not None:
                                (
                                    root / f"trial-{os.getpid()}-{trial_stats['trial']}.json"
                                ).write_text(json.dumps(trial_stats))
                            trial_stats = {
                                "trial": trial,
                                "mode": mode,
                                "calls": 0,
                                "shapes": {},
                                "first_rows": int(hidden.shape[0]),
                            }
                        rows = str(hidden.shape[0])
                        trial_stats["calls"] += 1
                        trial_stats["shapes"][rows] = trial_stats["shapes"].get(rows, 0) + 1
                        result = apply_variant(state, lm_head, hidden, embedding_bias, mode)
                        if control.get("capture") and captured < control.get("capture_calls", 256):
                            value = hidden.detach().to("cpu", copy=True).contiguous()
                            target = root / f"head-{os.getpid()}-{captured:04d}.pt"
                            with target.open("xb") as stream:
                                reference = result.detach().to("cpu", copy=True).contiguous()
                                saved = {
                                    "hidden": value,
                                    "mode": mode,
                                    "call": captured,
                                    "trial": trial,
                                }
                                if control.get("compact_capture"):
                                    # Keep all hidden rows but bind the full in-model reference
                                    # with a digest, avoiding tens of GiB of saved logits.
                                    saved.update(
                                        reference_sha256=hashlib.sha256(
                                            reference.view(torch.uint8).numpy().tobytes()
                                        ).hexdigest(),
                                        reference_shape=list(reference.shape),
                                        reference_dtype=str(reference.dtype),
                                    )
                                else:
                                    saved["reference"] = reference
                                torch.save(saved, stream)
                            captured += 1
                        # A marker proves the owned target hook actually ran; no logits or text.
                        marker = root / f"hook-{os.getpid()}.json"
                        if not marker.exists():
                            marker.write_text(
                                json.dumps(
                                    {
                                        "pid": os.getpid(),
                                        "hidden_width": hidden.shape[-1],
                                        "head_shape": list(lm_head.weight.shape),
                                        "head_dtype": str(getattr(state, "head_dtype", None)),
                                        "weight_dtype": str(lm_head.weight.dtype),
                                        "quant_method": type(lm_head.quant_method).__name__,
                                        "created_ns": time.time_ns(),
                                    }
                                )
                            )
                        return result

                    module._apply_head_gated = gated

            spec.loader = Loader()
            return spec

    sys.meta_path.insert(0, Finder())

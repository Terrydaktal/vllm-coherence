"""Read-only paired synthetic cache-path diagnostics for one isolated worker."""

import hashlib
import json
from functools import wraps
from pathlib import Path

from prefill_divergence_probe import PrefillDivergenceProbe


def tensor_digest(value):
    import torch

    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
    return hashlib.sha256(raw.tobytes()).hexdigest()


def logical_draft_cache(spec, first, end, destination=None):
    """Hash ordered logical context slots; physical page numbers may differ."""
    import torch

    records = []
    for index, layer in enumerate(spec.model.model.layers):
        kv = layer.self_attn.attn.kv_cache
        gid = spec.draft_kv_cache_group_ids[
            (spec._layer_group_idx or [0] * (index + 1))[index]
        ]
        block_size = spec.block_tables.kernel_block_sizes[gid]
        table = spec.block_tables.input_block_tables[gid]
        # The active Triton backend packs K and V into the final dimension:
        # [blocks, heads, tokens, 2 * head_size]. The five-dimensional
        # variants keep an explicit K/V axis. Physical strides need not match.
        packed = kv.ndim == 4 and kv.shape[2] == block_size
        explicit = kv.ndim == 5 and 2 in kv.shape[:2] and kv.shape[2] == block_size
        if not (packed or explicit):
            records.append(
                {
                    "layer": index,
                    "unrecognized_layout": list(kv.shape),
                    "block_size": block_size,
                }
            )
            continue
        if explicit and kv.shape[0] == 2:
            kv = kv.transpose(0, 1)
        logical = torch.arange(first, end, device=kv.device)
        pages = table[0, logical // block_size].long()
        if (pages < 0).any() or (pages >= kv.shape[0]).any():
            raise ValueError("invalid logical drafter page in sampled replay")
        contents = kv.view(torch.uint8)[pages, :, logical % block_size].cpu()
        if destination is not None:
            # Private synthetic bytes allow an exact first-difference reduction
            # after both requests, without a second expensive cold prefill.
            torch.save(contents, destination / f"layer-{index}.pt")
        records.append(
            {
                "layer": index,
                "first": first,
                "end": end,
                "null_pages": int((pages == 0).sum()),
                "chunks_of_32": [tensor_digest(t) for t in contents.split(32)],
            }
        )
    return records


class SampledCacheProbe(PrefillDivergenceProbe):
    def qwen_sampled_cache_begin(self, destination, prompt_tokens):
        from qwen_r9700_lab.conformance_instrumentation import HookSet

        path = Path(destination).resolve()
        if not path.is_relative_to("/prefill-diagnosis/refresh") or path.exists():
            raise ValueError("new private isolated sample path required")
        if not (prompt_tokens == 2049 or 60_000 <= prompt_tokens < 61_000) or getattr(
            self, "_sampled_replay_probe", None
        ):
            raise ValueError("expected one bounded synthetic replay at a qualified context")
        path.mkdir(mode=0o700, parents=True)
        spec = self.model_runner.speculator
        original = spec.propose
        hooks, records = HookSet(), []
        self._sampled_replay_probe = (hooks, records, path)

        target = self.model_runner.model
        compute_logits = target.compute_logits
        target_packets = []

        @wraps(compute_logits)
        def target_head(*args, **kwargs):
            logits = compute_logits(*args, **kwargs)
            positions = self._prefill_capture["positions"]
            if (
                len(target_packets) < 3
                and positions
                and positions[-1] >= prompt_tokens - 1
            ):
                target_packets.append([tensor_digest(row) for row in logits])
            return logits

        @wraps(original)
        def capture(*args, **kwargs):
            batch, aux, num_sampled = args[0], args[4], args[5]
            if len(records) >= 3 or not int(num_sampled[0].item()):
                return original(*args, **kwargs)
            counters = getattr(self, "_fourteen_lookup", None)
            lookup_before = counters.cpu().tolist() if counters is not None else None
            proposals = original(*args, **kwargs)
            lookup_delta = (
                [
                    after - before
                    for after, before in zip(
                        counters.cpu().tolist(), lookup_before, strict=True
                    )
                ]
                if lookup_before is not None
                else None
            )
            slot = int(batch.idx_mapping[0].item())
            steps = spec.num_speculative_steps
            record = {
                "target_input_rows": batch.num_tokens,
                "lookup_scanned_used_hot": lookup_delta,
                "sampled_target_logits_sha256": target_packets[len(records)]
                if len(target_packets) > len(records)
                else None,
                "num_sampled": int(num_sampled[0].item()),
                "positions": spec.sample_pos[:steps].cpu().tolist(),
                "seed": int(spec.seeds[slot].item()),
                "temperature": float(spec.temperature[slot].item()),
                "anchor": tensor_digest(
                    spec.input_buffers.input_ids[spec._anchor_indices[:1]]
                ),
                "proposal_sha256": [tensor_digest(t) for t in proposals[0]],
                "candidate_ids_sha256": [
                    tensor_digest(row) for row in spec._cached_candidate_ids[slot]
                ],
                "scores_sha256": [
                    tensor_digest(row) for row in spec._selector_scores[0]
                ],
                "full_distribution_sha256": [
                    tensor_digest(row) for row in spec.draft_logits[slot]
                ],
                "auxiliary_last_row_sha256": [
                    tensor_digest(x[batch.num_tokens - 1]) for x in (aux or [])
                ],
            }
            if not records:
                record["logical_draft_cache"] = logical_draft_cache(
                    spec, prompt_tokens - 2047, prompt_tokens, path
                )
            records.append(record)
            return proposals

        hooks.replace(target, "compute_logits", target_head)
        hooks.replace(spec, "propose", capture)
        return {"active": True}

    def qwen_sampled_cache_finish(self):
        hooks, records, path = self._sampled_replay_probe
        hooks.close()
        self._sampled_replay_probe = None
        (path / "proposals.json").write_text(json.dumps(records, indent=2) + "\n")
        return {"rounds": len(records)}

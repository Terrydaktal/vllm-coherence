"""Test-only observation of the qualified worker's graph dispatch.

Default dispatch observation uses CPU metadata only. The separately enabled
single-token sample probe reads tensors and is never a timing mode. Run the
functional suite again with SpeedCandidateWorker itself for the release gate.
"""

from collections import Counter
from dataclasses import asdict

from speed_candidate_worker import SpeedCandidateWorker


class SpeedLifecycleWorker(SpeedCandidateWorker):
    def qwen_lifecycle_sample_probe(self, enabled):
        """Opt-in diagnostic for synthetic single-token requests, never timing."""
        import torch

        if not hasattr(self, "_lifecycle_sample_original"):
            runner = self.model_runner
            original = runner.sample
            compute = runner.model.compute_logits
            self._lifecycle_sample_original = original

            def observed(*args, **kwargs):
                if (
                    not self._lifecycle_sample_enabled
                    or len(self._lifecycle_sample_rows) >= 16
                ):
                    return original(*args, **kwargs)
                row = {}

                def logits(hidden):
                    out = compute(hidden)
                    row["hidden"] = hidden.detach().clone()
                    row["logits"] = out.detach().clone()
                    return out

                runner.model.compute_logits = logits
                try:
                    result = original(*args, **kwargs)
                    sampled = result[0]
                    for name, value in vars(sampled).items():
                        if isinstance(value, torch.Tensor):
                            row[name] = value.detach().clone()
                    self._lifecycle_sample_rows.append(row)
                    return result
                finally:
                    runner.model.compute_logits = compute

            runner.sample = observed
        self._lifecycle_sample_enabled = enabled
        self._lifecycle_sample_rows = []
        return {"enabled": enabled, "diagnostic_only": True}

    def qwen_lifecycle_sample_report(self):
        import hashlib

        import torch

        rows = []
        for row in self._lifecycle_sample_rows:
            value = {}
            for name, tensor in row.items():
                tensor = tensor.cpu().contiguous()
                value[name] = {
                    "shape": list(tensor.shape),
                    "sha256": hashlib.sha256(
                        tensor.view(torch.uint8).numpy().tobytes()
                    ).hexdigest(),
                }
                if name == "logits":
                    scores, ids = tensor.float().topk(10)
                    value[name].update(
                        ids=ids.tolist(),
                        scores=scores.tolist(),
                        nans=int(tensor.isnan().sum()),
                    )
                elif tensor.numel() <= 32:
                    value[name]["values"] = tensor.tolist()
            rows.append(value)
        return rows

    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        manager = self.model_runner.cudagraph_manager
        original = manager.dispatch
        self._lifecycle_dispatch_counts = Counter()

        def observed(
            num_reqs,
            num_tokens,
            uniform_token_count,
            num_active_loras,
            max_query_len=None,
        ):
            desc = original(
                num_reqs,
                num_tokens,
                uniform_token_count,
                num_active_loras,
                max_query_len,
            )
            self._lifecycle_dispatch_counts[
                (
                    num_reqs,
                    num_tokens,
                    uniform_token_count,
                    max_query_len,
                    desc.cg_mode.name,
                    desc.num_reqs,
                    desc.num_tokens,
                )
            ] += 1
            return desc

        manager.dispatch = observed
        self._lifecycle_original_dispatch = original
        return result

    def qwen_lifecycle_report(self):
        manager = self.model_runner.cudagraph_manager

        def descriptor(desc):
            value = asdict(desc)
            value["cg_mode"] = desc.cg_mode.name
            return value

        # Exercise the real dispatcher on shape metadata; this does not execute
        # a fabricated batch or claim numerical coverage of two-row ownership.
        grid = []
        for requests in (1, 2):
            for width in range(1, 10):
                total = requests * width
                desc = self._lifecycle_original_dispatch(
                    requests, total, width, 0, width
                )
                grid.append(
                    {"requests": requests, "width": width, "selected": descriptor(desc)}
                )
        return {
            "captured": [descriptor(d) for d in manager.graphs],
            "dispatch_grid": grid,
            "observed": [
                {
                    "requests": key[0],
                    "tokens": key[1],
                    "uniform": key[2],
                    "max_query_len": key[3],
                    "mode": key[4],
                    "graph_requests": key[5],
                    "graph_tokens": key[6],
                    "count": count,
                }
                for key, count in self._lifecycle_dispatch_counts.items()
            ],
            "observation": "CPU metadata only; no GPU synchronization",
        }

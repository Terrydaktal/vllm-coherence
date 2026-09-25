"""Experimental replay of the unchanged Global-512 head and D7 sampler.

Only the pinned single-request, eight-row, T=1/p=.95/k=40 path is admitted.
The first forty replayed rounds compare complete head-logit bytes and every
observable sampler result before returning. State updates remain outside this
graph. Unsupported metadata and changed allocations call the original path.
"""

from collections import Counter
from dataclasses import replace


class GraphSampler:
    batch_fields = (
        "idx_mapping",
        "expanded_idx_mapping",
        "expanded_local_pos",
        "input_ids",
        "positions",
        "logits_indices",
        "cu_num_logits",
        "seq_lens",
    )

    def __init__(self, runner):
        self.runner = runner
        self.original = runner.sample
        self.enabled = True
        self.graph = None
        self.key = None
        self.keep = None
        self.outputs = None
        self.logits = None
        self.batch = None
        self.hidden = None
        self.draft = None
        self.parameters = None
        self.checked_rounds = 0
        self.warmups = 0
        self.counts = Counter()

    @staticmethod
    def tensor_key(value):
        return (
            value.data_ptr(),
            tuple(value.shape),
            tuple(value.stride()),
            str(value.dtype),
            str(value.device),
        )

    def signature(self, hidden, batch, grammar):
        import torch

        import radiance_verifyhead

        if grammar is not None or torch.cuda.is_current_stream_capturing():
            return None
        if (
            tuple(hidden.shape) != (8, 5120)
            or hidden.dtype != torch.bfloat16
            or not hidden.is_cuda
            or batch.num_reqs != 1
            or batch.num_tokens != 8
            or batch.num_draft_tokens != 7
            or batch.has_prefill
            or batch.has_structured_output_reqs
            or radiance_verifyhead.GLOBAL_TOPK != 512
            or not radiance_verifyhead._batch_is_safe(self.runner, batch, grammar)
        ):
            return None
        sampler = self.runner.sampler
        rejection = self.runner.rejection_sampler
        if (
            sampler.compute_nans
            or sampler.return_sampling_mask
            or rejection.synthetic_conditional_rates is not None
            or rejection.use_block_verification
        ):
            return None
        idx = batch.idx_mapping_np[:1]
        ss = sampler.sampling_states
        if not (
            (ss.temperature.np[idx] == 1.0).all()
            and (ss.top_p.np[idx] == 0.95).all()
            and (ss.top_k.np[idx] == 40).all()
        ):
            return None
        dynamic = [getattr(batch, name) for name in self.batch_fields]
        dynamic.extend((hidden, self.runner.speculator.draft_logits))
        tensors = [owner.gpu for owner in self.parameter_owners()]
        if any(value is None or not value.is_cuda for value in tensors + dynamic):
            return None
        return (
            tuple(self.tensor_key(value)[1:] for value in tensors),
            tuple(self.tensor_key(value)[1:] for value in dynamic),
            tuple(int(x) for x in idx),
            tuple(int(x) for x in batch.cu_num_logits_np),
            bool(sampler.needs_logits_processing[idx].any()),
            bool(sampler.use_fp64_gumbel),
        )

    def parameter_owners(self):
        ss = self.runner.sampler.sampling_states
        return (
            ss.temperature,
            ss.top_k,
            ss.top_p,
            ss.seeds,
            self.runner.req_states.prefill_len,
        )

    def refresh_batch(self, hidden, batch):
        """Refresh new per-round indices without replaying a stale allocation."""
        import torch

        if self.batch is None:
            self.batch = replace(
                batch,
                **{name: getattr(batch, name).clone() for name in self.batch_fields},
            )
            self.hidden = hidden.clone()
            self.draft = self.runner.speculator.draft_logits.clone()
            self.parameters = [owner.gpu.clone() for owner in self.parameter_owners()]
            return
        groups = {}
        for name in self.batch_fields:
            source, target = getattr(batch, name), getattr(self.batch, name)
            destinations, sources = groups.setdefault(source.dtype, ([], []))
            destinations.append(target)
            sources.append(source)
        for source, target in (
            (hidden, self.hidden),
            (self.runner.speculator.draft_logits, self.draft),
        ):
            destinations, sources = groups.setdefault(source.dtype, ([], []))
            destinations.append(target)
            sources.append(source)
        for owner, target in zip(self.parameter_owners(), self.parameters, strict=True):
            source = owner.gpu
            destinations, sources = groups.setdefault(source.dtype, ([], []))
            destinations.append(target)
            sources.append(source)
        for destinations, sources in groups.values():
            torch._foreach_copy_(destinations, sources)

    def invoke_with_logits(self, hidden, batch, grammar):
        model = self.runner.model
        original = model.compute_logits
        retained = []

        def compute(*args, **kwargs):
            result = original(*args, **kwargs)
            retained.append(result)
            return result

        model.compute_logits = compute
        try:
            outputs = self.original(hidden, batch, grammar)
        finally:
            model.compute_logits = original
        if len(retained) != 1:
            raise RuntimeError("sampler graph did not call exactly one target head")
        return outputs, retained[0]

    def check(self, reference, reference_logits):
        import torch

        candidate, sampled, rejected = self.outputs
        expected, expected_sampled, expected_rejected = reference
        if not torch.equal(
            self.logits.view(torch.int16), reference_logits.view(torch.int16)
        ):
            raise RuntimeError("sampler graph changed complete target head-logit bytes")
        if not (
            torch.equal(sampled, expected_sampled)
            and torch.equal(rejected, expected_rejected)
        ):
            raise RuntimeError("sampler graph changed accepted/rejected prefix lengths")
        for field in ("logprobs_tensors", "num_nans", "sampling_mask_tensors"):
            if (
                getattr(candidate, field, None) is not None
                or getattr(expected, field, None) is not None
            ):
                raise RuntimeError("sampler graph acquired an unsupported observable")
        # Rejected/padded token slots are unspecified scratch, not published output.
        columns = torch.arange(
            candidate.sampled_token_ids.shape[1], device=sampled.device
        )
        live = columns[None, :] < sampled[:, None]
        if not torch.equal(
            torch.where(live, candidate.sampled_token_ids, 0),
            torch.where(live, expected.sampled_token_ids, 0),
        ):
            raise RuntimeError("sampler graph changed a published token")
        self.checked_rounds += 1

    def __call__(self, hidden, batch, grammar):
        import torch

        if not self.enabled:
            self.counts["disabled"] += 1
            return self.original(hidden, batch, grammar)
        key = self.signature(hidden, batch, grammar)
        if key is None:
            self.counts["unsupported"] += 1
            return self.original(hidden, batch, grammar)
        if self.graph is not None and key != self.key:
            self.counts["allocation_or_metadata_change"] += 1
            for index, name in enumerate(
                (
                    "persistent_tensor",
                    "dynamic_layout",
                    "request_slot",
                    "row_boundaries",
                    "processing",
                    "fp64",
                )
            ):
                if key[index] != self.key[index]:
                    self.counts["changed_" + name] += 1
            return self.original(hidden, batch, grammar)
        if self.warmups < 2:
            self.warmups += 1
            self.counts["warmup"] += 1
            return self.original(hidden, batch, grammar)
        if self.graph is None:
            self.key = key
            self.keep = (hidden, batch)
            self.refresh_batch(hidden, batch)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            live_draft = self.runner.speculator.draft_logits
            self.runner.speculator.draft_logits = self.draft
            owners = self.parameter_owners()
            live_parameters = [owner.gpu for owner in owners]
            try:
                for owner, retained in zip(owners, self.parameters, strict=True):
                    owner.gpu = retained
                with torch.cuda.graph(self.graph):
                    self.outputs, self.logits = self.invoke_with_logits(
                        self.hidden, self.batch, grammar
                    )
            finally:
                self.runner.speculator.draft_logits = live_draft
                for owner, live in zip(owners, live_parameters, strict=True):
                    owner.gpu = live
        else:
            self.refresh_batch(hidden, batch)
        self.graph.replay()
        self.counts["replay"] += 1
        if self.checked_rounds < 40:
            reference, logits = self.invoke_with_logits(hidden, batch, grammar)
            self.check(reference, logits)
        return self.outputs

    def receipt(self):
        return {
            "enabled": self.enabled,
            "captured": self.graph is not None,
            "checked_rounds": self.checked_rounds,
            "checked_logit_rows": 8 * self.checked_rounds,
            "counts": dict(self.counts),
            "scope": "sampled Global-512 logits and committed token prefix; state update excluded",
        }

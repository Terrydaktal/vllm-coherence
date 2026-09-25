"""Isolated graph replay of the existing admitted Global-512 target head.

This adds no selector or numerical operation. The original admission gate
still decides whether the approximate head may run. Every warmup comparison
checks the complete output bytes before returning them to the sampler.
"""


class GraphHead:
    def __init__(self, original, *, checks=40):
        if checks < 40:
            raise ValueError("head experiment requires at least 320 checked rows")
        self.original = original
        self.required_checks = checks
        self.enabled = True
        self.graph = None
        self.input = None
        self.output = None
        self.weight_identity = None
        self.warmups = 0
        self.replays = 0
        self.checked_rows = 0
        self.full_vector_equal_rows = 0
        self.fallbacks = 0

    def receipt(self):
        return {
            "enabled": self.enabled,
            "captured": self.graph is not None,
            "replays": self.replays,
            "checked_rows": self.checked_rows,
            "full_vector_equal_rows": self.full_vector_equal_rows,
            "fallbacks": self.fallbacks,
            "scope": "same admitted Global-512 implementation; exact complete output bytes",
        }

    @staticmethod
    def identity(weight):
        return (
            weight.data_ptr(),
            tuple(weight.shape),
            weight.stride(),
            weight.dtype,
            weight.device,
        )

    def __call__(self, lm_head, hidden_states, embedding_bias=None):
        import torch

        from qwen_r9700_lab.conformance_topk import require

        admitted = (
            self.enabled
            and embedding_bias is None
            and tuple(hidden_states.shape) == (8, 5120)
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.is_cuda
            and hidden_states.is_contiguous()
            and not torch.cuda.is_current_stream_capturing()
        )
        if not admitted or self.warmups < 2:
            self.warmups += int(admitted)
            self.fallbacks += 1
            return self.original(lm_head, hidden_states, embedding_bias)
        identity = self.identity(lm_head.weight)
        if self.graph is not None:
            require(identity == self.weight_identity, "captured head weights changed")
            require(
                hidden_states.device == self.input.device,
                "captured head input device changed",
            )
        else:
            self.weight_identity = identity
            self.input = torch.empty_like(hidden_states)
            self.input.copy_(hidden_states)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = self.original(lm_head, self.input, None)
            require(
                tuple(self.output.shape) == (8, lm_head.weight.shape[0]),
                "captured head output shape changed",
            )
        self.input.copy_(hidden_states)
        self.graph.replay()
        self.replays += 1
        if self.checked_rows < 8 * self.required_checks:
            reference = self.original(lm_head, hidden_states, None)
            same = (reference.view(torch.int16) == self.output.view(torch.int16)).all(1)
            count = int(same.sum())
            self.checked_rows += 8
            self.full_vector_equal_rows += count
            require(count == 8, "graph replay changed Global-512 logits")
        return self.output

"""Opt-in CPU qualification helper: parallelize independent reference output rows.

The current campaign is unchanged. Every dot product still uses the same bound
serial C primitive, in the same increasing-k FP32 order. No GPU is used here.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from qwen_r9700_lab.diagnostic_contract import seal
from qwen_r9700_lab.ordered_reference_linear import OrderedLinear


def partitions(rows, workers):
    if rows < 0 or workers < 1 or workers > 32:
        raise ValueError("invalid independent-row partition")
    count = min(rows, workers)
    return [(rows * i // count, rows * (i + 1) // count) for i in range(count)]


class ParallelOrderedLinear(OrderedLinear):
    def __init__(self, reference, binding, *, workers=4):
        partitions(1, workers)
        super().__init__(reference, binding)
        self.workers = workers
        self.native = self.function
        self.function = self.multiply
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ordered-row")
        self.closed = False
        self.parallel_calls = 0
        self.execution_binding = seal(
            {
                "serial_implementation": binding["sha256"],
                "workers": workers,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "scope": (
                    "Disjoint output rows; unchanged serial multiply/add order within every row."
                ),
                "formal_equivalence": "UNPROVED",
            }
        )

    def multiply(self, x, w, output, batches, rows, width):
        if self.closed:
            raise RuntimeError("parallel reference is closed")
        if self.workers == 1 or rows < 2 or batches == 0:
            return self.native(x, w, output, batches, rows, width)
        inputs, result = x.reshape(batches, width), output.reshape(batches, rows)

        def interval(start, end):
            weight = w[start:end]
            for batch in range(batches):
                status = self.native(
                    inputs[batch],
                    weight,
                    result[batch, start:end],
                    1,
                    end - start,
                    width,
                )
                if status:
                    return status
            return 0

        jobs = [self.executor.submit(interval, a, b) for a, b in partitions(rows, self.workers)]
        statuses = [job.result() for job in jobs]
        self.parallel_calls += 1
        return next((status for status in statuses if status), 0)

    def close(self):
        self.executor.shutdown(wait=True)
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

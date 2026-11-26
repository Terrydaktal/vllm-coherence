"""Exact CPU-side candidates for the signed snapshot and lineage modules."""

from contextlib import contextmanager
from threading import Condition


class ByteBudget:
    def __init__(self, capacity):
        self.capacity = capacity
        self.remaining = capacity
        self.condition = Condition()

    @contextmanager
    def reserve(self, size):
        size = min(size, self.capacity)
        with self.condition:
            self.condition.wait_for(lambda: self.remaining >= size)
            self.remaining -= size
        try:
            yield
        finally:
            with self.condition:
                self.remaining += size
                self.condition.notify_all()

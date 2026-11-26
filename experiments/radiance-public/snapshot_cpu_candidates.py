"""Exact CPU-side candidates for the signed snapshot and lineage modules."""

from array import array
from contextlib import contextmanager
from threading import Condition


def fast_comparison(original):
    def compare(previous, current, *, prefix=False):
        n = min(len(previous), len(current))
        if type(previous) is type(current) is str:
            same = previous.startswith(current) or current.startswith(previous)
        elif (
            type(previous) is type(current) is array
            and previous.typecode == current.typecode
            and previous.typecode in "bBhHiIlLqQ"
        ):
            # Both operands have the same native integer representation.
            same = (
                memoryview(previous)[:n].tobytes() == memoryview(current)[:n].tobytes()
            )
        else:
            return original(previous, current, prefix=prefix)
        if not same:
            return original(previous, current, prefix=prefix)
        equal = len(previous) == len(current) or (prefix and n == len(previous))
        result = dict(
            equal=equal,
            compared_tokens=n,
            previous_tokens=len(previous),
            current_tokens=len(current),
        )
        if not equal:
            result["first_difference"] = n
        return result

    return compare


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

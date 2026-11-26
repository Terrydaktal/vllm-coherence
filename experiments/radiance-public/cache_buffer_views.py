"""Keep raw snapshot blocks as views until the destination buffer copy."""

import inspect
from textwrap import dedent


def substitute(method, original, replacement):
    source = dedent(inspect.getsource(method))
    if source.count(original) != 1:
        raise ValueError(f"snapshot buffer view patch did not match {method.__qualname__}")
    namespace = {}
    source = source.replace(original, replacement, 1)
    # Both callers transform a single checked expression in the pinned release.
    exec(compile(source, inspect.getfile(method), "exec"), method.__globals__, namespace)  # noqa: S102
    return namespace[method.__name__]


def reader_variant(read_many):
    # stream.read() owns immutable bytes. The decoder already accepts this
    # buffer protocol through struct, Zstandard, and SHA-256. Only slicing a
    # bytes object materializes another raw payload.
    return substitute(
        read_many,
        "data = decode_block(encoded, expected_size)",
        "data = decode_block(memoryview(encoded), expected_size)",
    )

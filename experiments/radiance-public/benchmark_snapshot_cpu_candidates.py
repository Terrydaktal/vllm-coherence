"""Time verified snapshot restores."""

import json, os, statistics, tempfile, time
from pathlib import Path
from qwen_r9700_lab import radiance_cache as c
from materialize_snapshot_candidates import build
import importlib.util
import sys
from cache_buffer_views import reader_variant


def timed(f):
    t = time.perf_counter()
    v = f()
    return (time.perf_counter() - t) * 1000, v


def alternating(functions):
    result = {key: [] for key in functions}
    for repeat in range(5):
        for key in functions if repeat % 2 == 0 else reversed(functions):
            ms, _ = timed(functions[key])
            result[key].append(ms)
    return {key: statistics.median(times) for key, times in result.items()}


def run():
    result = {}
    n = 27000832
    keys = [f"g0-{i:064x}.qkv" for i in range(8)]
    read3 = reader_variant(c.ChatStore.read_many)
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "build"
        build(Path(c.__file__).parent, directory)
        spec = importlib.util.spec_from_file_location(
            "benchmark_snapshot_variant", directory / "radiance_cache.py"
        )
        candidate = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = candidate
        spec.loader.exec_module(candidate)
        read4 = candidate.ChatStore.read_many
        store = c.ChatStore(tmp, {"id": "a" * 64, "generation": "b" * 64})
        store.activate()
        for raw in (False, True):
            data = os.urandom(n if raw else n // 2).ljust(n, b"\0")
            for key in keys:
                store.path(key).unlink(missing_ok=True)
            assert store.write_many((key, memoryview(data)) for key in keys)

            dest = bytearray(n)

            def read(f):
                for chunk in f(store, keys, n):
                    dest[:] = chunk
                assert dest == data

            result[f"read_raw={raw}"] = alternating(
                {k: lambda f=f: read(f) for k, f in (("three", read3), ("four", read4))}
            )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2), flush=True)

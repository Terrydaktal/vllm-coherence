"""Time verified snapshot restores and native prefix comparisons."""

import json, os, statistics, tempfile, time
from array import array
from pathlib import Path
from qwen_r9700_lab import radiance_cache as c, radiance_prefix_lineage as lineage
from snapshot_cpu_candidates import fast_comparison
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
    fast = fast_comparison(lineage.compare_sequences)
    for n in (60000, 200000, 320000):
        a = array("i", range(n))
        b = array("i", a)
        b.append(123)
        for diff in (None, 0, n // 2, n - 1):
            if diff is not None:
                b[diff] = -1
            assert fast(a, b, prefix=True) == lineage.compare_sequences(
                a, b, prefix=True
            )
            if diff is not None:
                b[diff] = a[diff]
        result[f"prefix_{n}"] = alternating(
            {
                name: lambda f=f: f(a, b, prefix=True)
                for name, f in (
                    ("reference", lineage.compare_sequences),
                    ("candidate", fast),
                )
            }
        )
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

"""Compare the two parallel snapshot readers on owned synthetic disk blocks."""

import argparse
import hashlib
import json
import os
import platform
import statistics
import tempfile
import time
from pathlib import Path

from qwen_r9700_lab import radiance_cache as cache

from cache_buffer_views import reader_variant


def benchmark(block_size, count, repeats, raw):
    stock = cache.ChatStore.read_many
    arms = {"copied_payload": stock, "buffer_view": reader_variant(stock)}
    report = {
        "scope": "synthetic storage read, decompression, SHA-256, destination copy",
        "python": platform.python_version(),
        "workers": cache.read_parallelism(block_size, count),
        "block_size": block_size,
        "block_count": count,
        "raw": raw,
        "paired_repeats": repeats,
        "source_sha256": {
            Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (cache.__file__, __file__, reader_variant.__code__.co_filename)
        },
    }
    with tempfile.TemporaryDirectory(prefix="coherence-buffer-views-") as root:
        digest = lambda value: hashlib.sha256(value).hexdigest()
        store = cache.ChatStore(root, {
            "id": digest(b"benchmark"),
            "generation": digest(b"cache buffer views"),
        })
        store.activate()
        keys = [f"g0-{index:064x}.qkv" for index in range(count)]
        expected = []
        for key in keys:
            noise_size = block_size if raw else block_size // 2
            block = os.urandom(noise_size) + bytes(block_size - noise_size)
            assert store.write(key, memoryview(block))
            expected.append(digest(block))
        report["file_bytes"] = sum(store.path(key).stat().st_size for key in keys)
        destination = memoryview(bytearray(block_size * count))
        slots = [destination[index:index + block_size] for index in range(0, len(destination), block_size)]
        samples = {name: [] for name in arms}
        for repeat in range(repeats):
            for name in arms if repeat % 2 == 0 else reversed(arms):
                start = time.perf_counter()
                for slot, data in zip(slots, arms[name](store, keys, block_size), strict=True):
                    slot[:] = data
                samples[name].append((time.perf_counter() - start) * 1000)
                assert [digest(slot) for slot in slots] == expected
    report["samples_ms"] = samples
    report["median_ms"] = {name: statistics.median(times) for name, times in samples.items()}
    report["exact_destination_checks"] = repeats * len(arms) * count
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--block-size", type=int, default=27000832)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()
    if min(args.block_size, args.count, args.repeats) < 1:
        parser.error("block size, count and repeats must be positive")
    args.output.write_text(json.dumps(
        benchmark(args.block_size, args.count, args.repeats, args.raw), indent=2,
    ) + "\n")

"""Owned CPU-only loader comparison; no model, GPU or chat fixtures."""

from __future__ import annotations

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


def run(repetitions=5, count=8, block_size=27 * 1024**2):
    for name, value in (("repetitions", repetitions), ("count", count), ("block_size", block_size)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    result = {
        "schema": "urn:coherence:synthetic-parallel-cache-load:v1",
        "python": platform.python_version(),
        "block_count": count,
        "block_size": block_size,
        "raw_bytes": count * block_size,
        "repetitions": repetitions,
        "parallel_workers": cache.read_parallelism(block_size, count),
        "scope": "CPU synthetic lossless disk read + decompression + SHA + destination copy",
        "page_cache_policy": "production per-payload POSIX_FADV_DONTNEED; alternating repeated reads",
        "gpu_measured": False,
    }
    with tempfile.TemporaryDirectory(prefix="coherence-parallel-load-") as directory:
        chat = {"id": hashlib.sha256(b"synthetic-loader").hexdigest(),
                "generation": hashlib.sha256(b"fixture-generation").hexdigest()}
        current = cache.ChatStore(directory, chat)
        current.activate()
        names, expected = [], []
        for index in range(count):
            # Half incompressible bytes and half compressible bytes retain the
            # same 27MiB decoded/SHA/copy cost as real blocks, using owned data.
            data = os.urandom(block_size // 2) + bytes(block_size - block_size // 2)
            key = f"g0-{index:064x}.qkv"
            assert current.write(key, memoryview(data))
            names.append(key)
            expected.append(hashlib.sha256(data).digest())
        result["file_bytes"] = sum(current.path(name).stat().st_size for name in names)
        destination = bytearray(count * block_size)
        view = memoryview(destination)
        samples = {"serial": [], "parallel": []}
        for repeat in range(repetitions):
            for mode in (("serial", "parallel") if repeat % 2 == 0 else ("parallel", "serial")):
                started = time.perf_counter()
                workers = 1 if mode == "serial" else cache.READ_WORKERS
                for index, data in enumerate(current.read_many(names, block_size, workers=workers)):
                    view[index * block_size:(index + 1) * block_size] = data
                samples[mode].append((time.perf_counter() - started) * 1000)
                # Independently check every CPU slot, outside the timed span.
                assert [hashlib.sha256(view[index * block_size:(index + 1) * block_size]).digest()
                        for index in range(count)] == expected
        result["samples_ms"] = samples
        result["median_ms"] = {name: statistics.median(values) for name, values in samples.items()}
        result["speedup"] = result["median_ms"]["serial"] / result["median_ms"]["parallel"]
        result["exact_destination_checks"] = repetitions * 2 * count
    result["fixture_removed"] = True
    return result


if __name__ == "__main__":
    def positive_integer(text):
        try:
            value = int(text)
        except ValueError as error:
            raise argparse.ArgumentTypeError("must be a positive integer") from error
        if value < 1:
            raise argparse.ArgumentTypeError("must be a positive integer")
        return value

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=positive_integer, default=5,
                        help="alternating serial/parallel repetitions (default: 5)")
    parser.add_argument("--count", type=positive_integer, default=8,
                        help="number of independent blocks (default: 8)")
    parser.add_argument("--block-size", type=positive_integer, default=27 * 1024**2,
                        help="decoded bytes per block (default: 28311552)")
    args = parser.parse_args()
    report = run(repetitions=args.repetitions, count=args.count, block_size=args.block_size)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), **report}))

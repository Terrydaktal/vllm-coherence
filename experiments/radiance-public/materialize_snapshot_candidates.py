"""Generate the pinned cache/lineage module variants for isolated qualification."""

import argparse
import hashlib
import json
from pathlib import Path


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"unexpected snapshot source at {old[:70]}")
    return source.replace(old, new, 1)


def cache_source(source):
    source = replace_once(
        source, "3 * expected_size + HEADER.size", "2 * expected_size + HEADER.size + 1"
    )
    source = replace_once(
        source,
        "_READ_SLOTS = threading.BoundedSemaphore(READ_WORKERS)",
        "from snapshot_cpu_candidates import ByteBudget\n"
        "_BUFFER_POOL = ByteBudget(READ_INFLIGHT_BYTES)\n"
        "_READ_SLOTS = threading.BoundedSemaphore(READ_WORKERS)",
    )
    source = replace_once(
        source,
        "        with _READ_BATCH_SLOTS, self.lock():",
        "        with _READ_BATCH_SLOTS, self.lock(), _BUFFER_POOL.reserve(width * (2 * expected_size + HEADER.size + 1)):",
    )
    source = replace_once(
        source,
        "            return next(blocks)",
        "            return bytes(next(blocks))",
    )
    source = replace_once(
        source,
        "data = decode_block(encoded, expected_size)",
        "data = decode_block(memoryview(encoded), expected_size)",
    )
    return source


def lineage_source(source):
    return replace_once(
        source,
        "def _get(value, name, default=None):",
        "from snapshot_cpu_candidates import fast_comparison\n"
        "compare_sequences = fast_comparison(compare_sequences)\n\n\ndef _get(value, name, default=None):",
    )


def build(src, dest):
    dest.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, transform in (
        ("radiance_cache.py", cache_source),
        ("radiance_prefix_lineage.py", lineage_source),
    ):
        original = (src / name).read_bytes()
        encoded = transform(original.decode()).encode()
        compile(encoded, name, "exec")
        (dest / name).write_bytes(encoded)
        report[name] = dict(
            original=hashlib.sha256(original).hexdigest(),
            candidate=hashlib.sha256(encoded).hexdigest(),
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.source_directory, args.destination), indent=2))

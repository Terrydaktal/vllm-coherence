"""Exact CPU-side candidates for the signed snapshot and lineage modules."""

from array import array
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
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


def parallel_verifier(cache, budget=None):
    import zstandard

    budget = budget or ByteBudget(cache.READ_INFLIGHT_BYTES)

    def prepare(self, keys, block_size):
        with self.lock():
            prior = self.metadata()
            known = (
                prior.get("verified_head", {})
                if prior.get("verified_block_size") == block_size
                else {}
            )
            result = dict(keys=sorted(set(keys)), missing=[], invalid=[], verified={})
            if prior.get("generation") != self.chat["generation"]:
                result["missing"] = result["keys"]
                return result

            def verify(job):
                key, path, before = job
                size = 0
                try:
                    with path.open("rb") as stream:
                        encoded = stream.read(block_size + cache.HEADER.size + 1)
                        if len(encoded) > block_size + cache.HEADER.size:
                            raise ValueError("snapshot exceeds expected block size")
                        cache._release_payload_page_cache(stream.fileno(), len(encoded))
                    cache.decode_block(memoryview(encoded), block_size)
                    size = len(encoded)
                    if before != self._fingerprint(path):
                        raise ValueError("snapshot changed during verification")
                    return key, "verified", before, size
                except FileNotFoundError:
                    return key, "missing", None, size
                except (OSError, ValueError, zstandard.ZstdError):
                    return key, "invalid", None, size

            jobs = []
            for key in result["keys"]:
                path = self.path(key)
                try:
                    fingerprint = self._fingerprint(path)
                except FileNotFoundError:
                    result["missing"].append(key)
                    continue
                except (OSError, ValueError):
                    result["invalid"].append(key)
                    continue
                if fingerprint in (
                    known.get(key),
                    self._verified.get((key, block_size)),
                ):
                    result["verified"][key] = fingerprint
                else:
                    jobs.append((key, path, fingerprint))

            verified_bytes = 0
            captured = cache.cache_telemetry.capture_context()

            def measured(job):
                return cache.cache_telemetry.run_with_context(
                    lambda: verify(job), captured
                )

            width = cache.read_parallelism(block_size, len(jobs))
            with budget.reserve(width * (2 * block_size + cache.HEADER.size + 1)):
                with (
                    ThreadPoolExecutor(max_workers=width)
                    if width > 1
                    else nullcontext() as pool
                ):
                    for key, status, fp, size in (
                        pool.map(measured, jobs) if pool else map(verify, jobs)
                    ):
                        verified_bytes += size
                        if status == "verified":
                            result[status][key] = fp
                        else:
                            result[status].append(key)
            for key, fp in result["verified"].items():
                self._verified[key, block_size] = fp
            for status in ("missing", "invalid"):
                result[status].sort()
            self._record_io({"verification_file_bytes": verified_bytes})
            return result

    return prepare

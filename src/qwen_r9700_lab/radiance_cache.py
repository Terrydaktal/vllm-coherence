"""Chat-owned, lossless Radiance snapshots. Also runs over SSH using only Python.

The model's cache salt isolates chats and compaction generations. The lock lives
outside the generations: retiring one waits for its readers/writers, and late
writes cannot recreate it. Legacy shared caches are never adopted or deleted.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import struct
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

try:
    import qwen_radiance_cache_telemetry as cache_telemetry
except ModuleNotFoundError as error:
    if error.name != "qwen_radiance_cache_telemetry":
        raise
    try:
        from qwen_r9700_lab import radiance_cache_telemetry as cache_telemetry
    except ModuleNotFoundError as missing:
        if missing.name not in {"qwen_r9700_lab", "qwen_r9700_lab.radiance_cache_telemetry"}:
            raise
        try:
            # The legacy flat /patches tree supports the standalone cache CLI.
            import radiance_cache_telemetry as cache_telemetry
        except ModuleNotFoundError as standalone:
            if standalone.name != "radiance_cache_telemetry":
                raise

            # SSH sends this CLI as source on stdin, and lifecycle control loads
            # its pinned flush helper by filename. Neither host tool needs a
            # recorder or has to import the serving package. The backend's
            # installer authenticates and installs the real telemetry module.
            class _HostCacheTelemetry:
                @staticmethod
                def span(*args, **kwargs):
                    return contextlib.nullcontext()

                @staticmethod
                def lock(mutex, *args, **kwargs):
                    return mutex

                @staticmethod
                def measured(*args, **kwargs):
                    return lambda operation: operation

                @staticmethod
                def emit(*args, **kwargs):
                    return None

            cache_telemetry = _HostCacheTelemetry()

DEFAULT_ROOT = "/home/lewis/.cache/qwen-radiance-public-clean-snapshot-v1"
FORMAT = "qwen-chat-cache-v1"
HEADER = struct.Struct(">8sQ32s")
COMPRESSED = b"QWENKV1Z"
RAW = b"QWENKV1R"
ID = re.compile(r"[0-9a-f]{64}\Z")
KEY = re.compile(r"g[0-9]+-[0-9a-f]{16,128}\.qkv\Z")
CONTROL_DIRECTORY = Path("/dev/shm/qwen-radiance-snapshot-control-v1")
CONTROL_SCHEMA = "urn:qwen-r9700:radiance-snapshot-control:v1"
CONTROL_FILE = re.compile(r"([0-9a-f]{32})\.(request|response)\.json\Z")
_WRITE_LOCKS = tuple(threading.Lock() for _ in range(64))
RETIREMENT_SCHEMA = "urn:coherence:snapshot-retirements:v1"
TEST_CHAT_TITLES = {"Synthetic release smoke", "Synthetic relay probe"}


class RetiredGenerationError(ValueError):
    """A valid chat identity names a generation durably superseded by compaction."""


class SnapshotIntegrityError(ValueError):
    """Stored bytes fail their own encoding or checksum, independently of the ABI."""


def identity(value: dict) -> dict:
    if not isinstance(value, dict) or any(
        not isinstance(value.get(k), str) or not ID.fullmatch(value[k])
        for k in ("id", "generation")
    ):
        raise ValueError("chat id and generation must be SHA256 identifiers")
    return {
        k: str(value.get(k, ""))[:4096]
        for k in ("id", "generation", "title", "cwd", "session_file")
    }


def cache_salt(chat: dict) -> str:
    chat = identity(chat)
    return f"{FORMAT}:{chat['id']}:{chat['generation']}"


def real_directory(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"not a real directory: {path}")


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with cache_telemetry.span("directory_fsync"):
            os.fsync(fd)
    finally:
        os.close(fd)


def _release_payload_page_cache(fd: int, size: int, *, stage: str = "disk_page_cache_release") -> None:
    """Release only the redundant OS copy of a durable snapshot payload.

    The primary offload tier and parked chat images retain their own state.
    This hint neither deletes the file nor changes durability. Unsupported or
    failed advice must not turn a successful snapshot write into a failure.
    """
    advise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    with cache_telemetry.span(stage, bytes=size) as observation:
        status, error_code = "unsupported", None
        if advise is not None and dontneed is not None:
            try:
                advise(fd, 0, 0, dontneed)
            except OSError as error:
                status, error_code = "failed", error.errno or 0
            else:
                status = "complete"
        if observation is not None:
            observation.values["diagnostic_status"] = status
            if error_code is not None:
                observation.values["status_code"] = error_code


def release_model_file_cache(runner) -> None:
    """Release checkpoint-file residue after all target/drafter GPU warmup.

    Called once, before allocating the parking pool. Only regular safetensors
    files in the configured local model directories are advised. Existing CPU
    mappings and device weights are untouched; advice cannot change file bytes.
    """
    if getattr(runner, "_qwen_model_file_cache_released", False):
        return
    config = getattr(runner, "vllm_config", None)
    speculative = getattr(config, "speculative_config", None)
    candidates = (
        getattr(getattr(config, "model_config", None), "model", None),
        getattr(speculative, "model", None),
        getattr(getattr(speculative, "draft_model_config", None), "model", None),
    )
    directories = set()
    for model in candidates:
        if not isinstance(model, (str, Path)):
            continue
        try:
            directory = Path(model).resolve(strict=True)
            if not directory.is_dir() or directory in directories:
                continue
            directories.add(directory)
            with os.scandir(directory) as entries:
                for index, entry in enumerate(entries):
                    if index >= 4096:
                        cache_telemetry.emit("model_page_cache_release", diagnostic_status="incomplete")
                        break
                    if not entry.name.endswith(".safetensors") or not entry.is_file(follow_symlinks=False):
                        continue
                    fd = os.open(entry.path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
                    try:
                        info = os.fstat(fd)
                        if stat.S_ISREG(info.st_mode):
                            _release_payload_page_cache(fd, info.st_size, stage="model_page_cache_release")
                    finally:
                        os.close(fd)
        except (OSError, RuntimeError) as error:
            cache_telemetry.emit("model_page_cache_release", success=False,
                                 status_code=getattr(error, "errno", None) or 0)
    runner._qwen_model_file_cache_released = True


def atomic_write(path: Path, content: bytes, *, release_page_cache: bool = False) -> None:
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            with cache_telemetry.span("disk_write", bytes=len(content)):
                stream.write(content)
                stream.flush()
            with cache_telemetry.span("file_fsync"):
                os.fsync(stream.fileno())
            if release_page_cache:
                _release_payload_page_cache(stream.fileno(), len(content))
        with cache_telemetry.span("file_publish"):
            temporary.replace(path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def private_control_directory(path: Path = CONTROL_DIRECTORY, *, create: bool = False) -> Path:
    """Return the owner-only tmpfs control directory used by the live backend."""
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if path.is_symlink() or not path.is_dir() or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("snapshot control directory must be private and owner-owned")
    return path


def request_tail_flush(
    chat: dict,
    *,
    timeout: float = 120.0,
    control_directory: Path = CONTROL_DIRECTORY,
) -> dict:
    """Ask the live backend to make one in-RAM tail durably publishable."""
    chat = identity(chat)
    if not 0 < timeout <= 600:
        raise ValueError("snapshot flush timeout must be between 0 and 600 seconds")
    directory = private_control_directory(Path(control_directory))
    nonce = uuid.uuid4().hex
    request = directory / f"{nonce}.request.json"
    response = directory / f"{nonce}.response.json"
    payload = {
        "schema": CONTROL_SCHEMA,
        "nonce": nonce,
        "action": "flush",
        "chat": chat,
    }
    atomic_write(request, json.dumps(payload, sort_keys=True).encode())
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            try:
                info = response.lstat()
                if response.is_symlink() or not response.is_file() or info.st_size > 65536:
                    raise ValueError("unsafe snapshot flush response")
                result = json.loads(response.read_text())
                if result.get("schema") != CONTROL_SCHEMA or result.get("nonce") != nonce:
                    raise ValueError("snapshot flush response identity mismatch")
                if result.get("status") == "error":
                    raise RuntimeError(result.get("error") or "backend snapshot flush failed")
                if result.get("status") not in ("flushed", "already_durable"):
                    raise RuntimeError("backend rejected the snapshot tail flush")
                return result
            except FileNotFoundError:
                time.sleep(0.05)
        raise TimeoutError("live backend did not complete the snapshot tail flush")
    finally:
        request.unlink(missing_ok=True)
        response.unlink(missing_ok=True)


def encode_block(data: memoryview | bytes) -> bytes:
    import zstandard

    with cache_telemetry.span("compression", bytes=len(data)):
        compressed = zstandard.ZstdCompressor(level=1, write_checksum=True).compress(data)
    magic, payload = (COMPRESSED, compressed) if len(compressed) < len(data) else (RAW, data)
    with cache_telemetry.span("checksum", bytes=len(data)):
        digest = hashlib.sha256(data).digest()
    with cache_telemetry.span("encoded_buffer_copy", bytes=len(payload)):
        return HEADER.pack(magic, len(data), digest) + bytes(payload)


def decode_block(data: bytes, expected_size: int) -> bytes:
    import zstandard

    if len(data) < HEADER.size:
        raise SnapshotIntegrityError("truncated snapshot header")
    magic, size, digest = HEADER.unpack(data[: HEADER.size])
    if size != expected_size:
        raise ValueError("snapshot block size differs from the runtime")
    payload = data[HEADER.size :]
    if magic == COMPRESSED:
        # Check the frame size before allocating, even if its header is corrupt.
        try:
            if zstandard.frame_content_size(payload) != expected_size:
                raise SnapshotIntegrityError("snapshot frame size differs from its header")
            with cache_telemetry.span("decompression", bytes=expected_size):
                payload = zstandard.ZstdDecompressor().decompress(
                    payload, max_output_size=expected_size, allow_extra_data=False
                )
        except zstandard.ZstdError as error:
            raise SnapshotIntegrityError("invalid compressed snapshot payload") from error
    elif magic != RAW:
        raise SnapshotIntegrityError("unknown snapshot encoding")
    with cache_telemetry.span("checksum", bytes=len(payload)):
        valid = len(payload) == expected_size and hashlib.sha256(payload).digest() == digest
    if not valid:
        raise SnapshotIntegrityError("snapshot checksum mismatch")
    return payload


class ChatStore:
    def __init__(self, data_root: Path | str, chat: dict):
        self.chat = identity(chat)
        self.root = Path(data_root)
        real_directory(self.root)
        self.managed = self.root / FORMAT
        self.managed.mkdir(exist_ok=True, mode=0o700)
        real_directory(self.managed)
        self.directory = self.managed / self.chat["id"]
        self.directory.mkdir(exist_ok=True, mode=0o700)
        real_directory(self.directory)
        self.generations = self.directory / "generations"
        self.generations.mkdir(exist_ok=True, mode=0o700)
        real_directory(self.generations)
        self.generation = self.generations / self.chat["generation"]
        self._verified = {}

    @contextlib.contextmanager
    def lock(self, *, exclusive: bool = False):
        fd = os.open(self.directory / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            with cache_telemetry.span("snapshot_lock", minimum_ms=0.05):
                fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield
        finally:
            os.close(fd)

    def metadata(self) -> dict:
        path = self.directory / "chat.json"
        if path.is_symlink():
            raise ValueError("chat metadata is a symlink")
        return json.loads(path.read_text()) if path.exists() else {}

    def save_metadata(self, value: dict) -> None:
        value = {**value, "updated_at": datetime.now(UTC).isoformat()}
        atomic_write(self.directory / "chat.json", json.dumps(value, sort_keys=True).encode())

    def current(self) -> bool:
        return self.metadata().get("generation") == self.chat["generation"]

    def activate(self) -> dict:
        """Retire old writers, retaining one complete fallback until publication."""
        with self.lock(exclusive=True):
            prior = self.metadata()
            retired = prior.get("retired_generations", [])
            if self.chat["generation"] in retired:
                raise RetiredGenerationError(
                    "stale request targets a retired compaction generation"
                )
            recreated = not self.generation.exists()
            self.generation.mkdir(exist_ok=True, mode=0o700)
            real_directory(self.generation)
            # Persist the new directory before the tombstone can retire its
            # predecessor. Retrying an interrupted activation also repairs it.
            sync_directory(self.generations)
            if prior.get("generation") != self.chat["generation"]:
                if prior.get("generation"):
                    retired = [*retired, prior["generation"]]
                fallback = prior.get("fallback")
                if prior.get("head"):
                    fallback = {k: prior[k] for k in ("generation", "head", "tokens")}
                # The durable tombstone precedes deletion. Retrying after a crash
                # finishes collection; old queued writers see current() == False.
                self.save_metadata(
                    {
                        **self.chat,
                        "format": FORMAT,
                        "retired_generations": retired,
                        "status": "empty",
                        "tokens": 0,
                        "head": [],
                        "fallback": fallback,
                    }
                )
            else:
                if recreated:
                    prior = {**prior, "status": "empty", "tokens": 0, "head": []}
                self.save_metadata({**prior, **self.chat})
            # An exclusive lock proves no live writer owns these temporary files.
            for temporary in self.generation.glob(".pending-*"):
                temporary.unlink()
            info = self.metadata()
            fallback = info.get("fallback") or {}
            removed = retained = 0
            for path in self.generations.iterdir():
                if path.name == self.chat["generation"]:
                    continue
                if not ID.fullmatch(path.name):
                    raise ValueError(f"unexpected generation path: {path}")
                real_directory(path)
                if path.name == fallback.get("generation"):
                    keep = set(fallback["head"])
                    for block in path.iterdir():
                        if block.name in keep:
                            retained += block.lstat().st_size
                        elif KEY.fullmatch(block.name) or block.name.startswith(".pending-"):
                            removed += block.lstat().st_size
                            block.unlink()
                    sync_directory(path)
                    continue
                removed += sum(p.stat().st_size for p in path.iterdir() if p.is_file())
                shutil.rmtree(path)
            sync_directory(self.generations)
            return {
                "chat_id": self.chat["id"],
                "removed_file_bytes": removed,
                "retained_file_bytes": retained,
                "generation": self.chat["generation"],
            }

    def path(self, key: str) -> Path:
        if not KEY.fullmatch(key):
            raise ValueError("invalid snapshot object key")
        real_directory(self.generation)
        path = self.generation / key
        if path.is_symlink():
            raise ValueError("snapshot object is a symlink")
        return path

    def exists(self, key: str) -> bool:
        return self.exists_many([key])[0]

    def exists_many(self, keys: list[str]) -> list[bool]:
        # A hybrid prefix lookup can inspect hundreds of keys. Read the chat
        # generation once for the batch, rather than reopening its manifest for
        # each key on the scheduler thread.
        with self.lock():
            if not self.current():
                return [False] * len(keys)
            return [self.path(key).is_file() for key in keys]

    def io_totals(self) -> dict:
        path = self.directory / "io.json"
        if path.is_symlink():
            raise ValueError("snapshot I/O counters are a symlink")
        return json.loads(path.read_text()) if path.exists() else {"available": False}

    def _record_io(self, changes: dict) -> None:
        """Persist one counter update per transfer batch, outside generations.

        These are completed snapshot payload writes, including objects later
        collected. They exclude metadata and failed partial writes; device
        counters account for those too. A crash can lose the unfinished batch's
        counters, so this is a lower bound rather than a NAND wear estimate.
        """
        if not any(changes.values()):
            return
        fd = os.open(self.directory / ".io.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            with cache_telemetry.span("io_counter_lock", minimum_ms=0.05):
                fcntl.flock(fd, fcntl.LOCK_EX)
            prior = self.io_totals()
            now = datetime.now(UTC).isoformat()
            info = {
                **prior,
                "available": True,
                "since": prior.get("since", now),
                "updated_at": now,
                "scope": "completed_snapshot_payload_io_lower_bound",
            }
            for name, amount in changes.items():
                info[name] = prior.get(name, 0) + amount
            atomic_write(self.directory / "io.json", json.dumps(info, sort_keys=True).encode())
        finally:
            os.close(fd)

    def write(self, key: str, data: memoryview) -> bool:
        return self.write_many([(key, data)])

    def write_many(self, blocks) -> bool:
        # Compression holds the read lock too: compaction cannot return before
        # all old writes have stopped, and a late writer cannot recreate a folder.
        with self.lock():
            if not self.current():
                return False
            counts = {
                "written_file_bytes": 0,
                "written_raw_bytes": 0,
                "written_blocks": 0,
                "reused_blocks": 0,
                "compression_seconds": 0.0,
                "write_failures": 0,
            }
            try:
                for key, data in blocks:
                    path = self.path(key)
                    # The engine has one process but several filesystem workers.
                    # Coalesce simultaneous requests for the same immutable key.
                    with cache_telemetry.lock(
                        _WRITE_LOCKS[hash(str(path)) % len(_WRITE_LOCKS)], "block_write_lock"
                    ):
                        if path.exists():
                            counts["reused_blocks"] += 1
                            continue
                        started = time.monotonic()
                        encoded = encode_block(data)
                        counts["compression_seconds"] += time.monotonic() - started
                        atomic_write(path, encoded, release_page_cache=True)
                        counts["written_file_bytes"] += len(encoded)
                        counts["written_raw_bytes"] += len(data)
                        counts["written_blocks"] += 1
            except Exception:
                counts["write_failures"] += 1
                raise
            finally:
                self._record_io(counts)
            return True

    def read(self, key: str, expected_size: int) -> bytes:
        with contextlib.closing(self.read_many([key], expected_size)) as blocks:
            return next(blocks)

    def read_many(self, keys: list[str], expected_size: int):
        with self.lock():
            if not self.current():
                raise ValueError("snapshot generation was retired")
            counts = {
                "read_file_bytes": 0,
                "read_raw_bytes": 0,
                "read_blocks": 0,
                "read_failures": 0,
                "invalidated_blocks": 0,
                "invalidated_file_bytes": 0,
            }
            try:
                for key in keys:
                    path = self.path(key)
                    # Exclude a simultaneous repair of this immutable key. A
                    # reader must not remove a newer valid replacement after
                    # detecting damage in the old file. Release before yield.
                    with cache_telemetry.lock(
                        _WRITE_LOCKS[hash(str(path)) % len(_WRITE_LOCKS)], "block_read_lock"
                    ):
                        with cache_telemetry.span("disk_read"), path.open("rb") as stream:
                            encoded = stream.read()
                            _release_payload_page_cache(stream.fileno(), len(encoded))
                        try:
                            data = decode_block(encoded, expected_size)
                        except SnapshotIntegrityError:
                            # A confirmed corrupt object is not a usable disk
                            # head. Make it a miss so recomputed data can replace
                            # it; preserve valid files on ABI or I/O errors.
                            path.unlink()
                            sync_directory(path.parent)
                            counts["invalidated_blocks"] += 1
                            counts["invalidated_file_bytes"] += len(encoded)
                            raise
                    counts["read_file_bytes"] += len(encoded)
                    counts["read_raw_bytes"] += len(data)
                    counts["read_blocks"] += 1
                    yield data
            except Exception:
                counts["read_failures"] += 1
                raise
            finally:
                self._record_io(counts)

    @staticmethod
    def _fingerprint(path: Path) -> list[int]:
        value = path.stat()
        return [value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]

    def prepare_publication(self, keys: list[str], block_size: int) -> dict:
        """Verify new/changed payloads away from the scheduler's critical path.

        Published immutable files retain their verified fingerprints. Normal
        continuations verify only newly written blocks, not the whole prefix.
        """
        import zstandard

        with self.lock():
            prior = self.metadata()
            known = (
                prior.get("verified_head", {})
                if prior.get("verified_block_size") == block_size
                else {}
            )
            result = {"keys": sorted(set(keys)), "missing": [], "invalid": [], "verified": {}}
            if prior.get("generation") != self.chat["generation"]:
                result["missing"] = result["keys"]
                return result
            verified_bytes = 0
            for key in result["keys"]:
                path = self.path(key)
                try:
                    before = self._fingerprint(path)
                    if known.get(key) != before and self._verified.get((key, block_size)) != before:
                        with path.open("rb") as stream:
                            encoded = stream.read()
                            _release_payload_page_cache(stream.fileno(), len(encoded))
                        decode_block(encoded, block_size)
                        verified_bytes += len(encoded)
                        if before != self._fingerprint(path):
                            raise ValueError("snapshot changed during verification")
                    result["verified"][key] = before
                    self._verified[key, block_size] = before
                except FileNotFoundError:
                    result["missing"].append(key)
                except (OSError, ValueError, zstandard.ZstdError):
                    result["invalid"].append(key)
            self._record_io({"verification_file_bytes": verified_bytes})
            return result

    def publish(self, keys: list[str], tokens: int, block_size: int, *, prepared=None,
                response_end=None, immutable_keys=None) -> bool:
        """Commit a complete head or roll back its abandoned writes, then collect.

        The caller must first drain every request and disk job for this chat.
        A rejected successor cannot be repaired after those jobs have finished;
        retain the previous head and discard the failed candidate instead.
        """
        if immutable_keys is not None and not set(immutable_keys).issubset(keys):
            raise ValueError("immutable snapshot prefix differs from publication candidate")
        if response_end is not None and (
            not isinstance(response_end, dict)
            or response_end.get("schema") != "urn:coherence:response-end:v1"
            or response_end.get("tokens") != tokens
            or not isinstance(response_end.get("prefix_sha256"), str)
            or not ID.fullmatch(response_end["prefix_sha256"])
        ):
            raise ValueError("invalid exact response-end manifest")
        if prepared is None:
            with self.lock():
                if not self.current():
                    return False
            prepared = self.prepare_publication(keys, block_size)
        with self.lock(exclusive=True):
            if not self.current():
                return False
            prior = self.metadata()
            keys = sorted(set(keys))
            if keys != prepared["keys"]:
                raise ValueError("verified snapshot differs from publication candidate")
            missing, invalid = list(prepared["missing"]), list(prepared["invalid"])
            for key, expected in prepared["verified"].items():
                try:
                    if self._fingerprint(self.path(key)) != expected:
                        invalid.append(key)
                except FileNotFoundError:
                    missing.append(key)
                except OSError:
                    invalid.append(key)
            complete = not missing and not invalid
            info = {
                **prior,
                "status": "ready" if complete else "incomplete",
                "publication": {
                    "tokens": tokens,
                    "expected_blocks": len(keys),
                    "missing_keys": missing,
                    "invalid_keys": invalid,
                    "result": "committed" if complete else "rolled_back",
                },
            }
            if complete:
                info.update(
                    head=keys,
                    tokens=tokens,
                    verified_head=prepared["verified"],
                    verified_block_size=block_size,
                    fallback=None,
                    response_end=response_end,
                    immutable_head=sorted(set(immutable_keys)) if immutable_keys is not None else None,
                )
                self._verified = {
                    (key, block_size): value for key, value in prepared["verified"].items()
                }
            self._collect(info)
            return complete

    def _collect(self, info: dict) -> dict:
        """Under the exclusive chat lock, persist intent before deleting anything."""
        keep = set(info["head"])
        if any(not KEY.fullmatch(key) for key in keep):
            raise ValueError("invalid published head manifest")
        info = {**info, "gc": {"status": "pending"}}
        self.save_metadata(info)
        removed_files = removed_bytes = 0
        try:
            for path in self.generation.iterdir():
                if path.name not in keep and (
                    KEY.fullmatch(path.name) or path.name.startswith(".pending-")
                ):
                    size = path.lstat().st_size
                    path.unlink()
                    removed_files += 1
                    removed_bytes += size
            fallback_generation = (info.get("fallback") or {}).get("generation")
            for directory in self.generations.iterdir():
                if directory.name in {self.chat["generation"], fallback_generation}:
                    continue
                if not ID.fullmatch(directory.name):
                    raise ValueError("invalid snapshot generation directory")
                real_directory(directory)
                for path in directory.iterdir():
                    removed_files += 1
                    removed_bytes += path.lstat().st_size
                shutil.rmtree(directory)
            sync_directory(self.generation)
            sync_directory(self.generations)
        except OSError as error:
            # If even this write fails, the durable pending record still causes
            # startup recovery. Never report a completed GC before directory fsync.
            self.save_metadata({**info, "gc": {"status": "failed", "errno": error.errno}})
            raise
        result = {
            "status": "complete",
            "removed_files": removed_files,
            "removed_file_bytes": removed_bytes,
        }
        self.save_metadata({**info, "gc": result})
        return result

    def collect(self, *, failure: str | None = None) -> dict:
        """Recover abandoned writes with no active requests, preserving the head."""
        with self.lock(exclusive=True):
            if not self.current():
                return {"status": "retired"}
            info = self.metadata()
            if failure is not None:
                info = {**info, "status": "incomplete", "publication": {"result": failure}}
            return self._collect(info)


def _maintenance_json(path: Path, *, limit=8 * 1024 * 1024) -> dict:
    """Read bounded regular metadata without following links or opening pipes."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("cache metadata is not a regular file")
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("cache metadata exceeds its size limit")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("cache metadata must be an object")  # noqa: TRY004 -- invalid serialized metadata
    return value


def _retirement_ledger(path: Path) -> dict:
    try:
        ledger = _maintenance_json(path)
    except FileNotFoundError:
        ledger = {"schema": RETIREMENT_SCHEMA, "entries": {}}
    if ledger.get("schema") != RETIREMENT_SCHEMA or not isinstance(ledger.get("entries"), dict):
        raise ValueError("invalid snapshot retirement ledger")
    for key, entry in ledger["entries"].items():
        parts = key.split("/")
        if (len(parts) != 2 or any(not ID.fullmatch(part) for part in parts)
                or not isinstance(entry, dict) or entry.get("abi") != parts[0]
                or entry.get("chat_id") != parts[1]
                or entry.get("status") not in {"pending", "complete"}):
            raise ValueError("invalid archived snapshot identity")
        counters = entry.get("io", {})
        if not isinstance(counters, dict):
            raise ValueError("invalid archived snapshot counters")  # noqa: TRY004 -- invalid serialized counters
        _merge_io_history({}, counters)
    return ledger


def _merge_io_history(prior: dict, current: dict) -> dict:
    # A pending deletion overlaps the still-present io.json. These are high-water
    # copies of the same counter, not two independent transfers. A test identity
    # reused with a new generation retains io.json and continues this same counter.
    def epochs(value):
        since = value.get("since")
        if since is not None and (not isinstance(since, str) or not since):
            raise ValueError("invalid snapshot traffic start time")
        if "epochs" in value:
            result = value["epochs"]
            if (not isinstance(result, dict) or not result or any(
                    not isinstance(key, str) or not isinstance(row, dict) or "epochs" in row
                    or key != (row.get("since") or "legacy")
                    for key, row in result.items())):
                raise ValueError("invalid snapshot traffic epochs")
            return result
        return {since or "legacy": value} if value.get("available") else {}

    previous, latest = epochs(prior), epochs(current)
    if len(set(previous) | set(latest)) > 1 or "epochs" in prior or "epochs" in current:
        combined = {key: _merge_io_history(previous.get(key, {}), latest.get(key, {}))
                    for key in set(previous) | set(latest)}
        # Recreating a retired data ABI can restart io.json. Keep separate epochs
        # so its new writes are added, while a pending deletion is deduplicated.
        merged = {**prior, **current, "epochs": combined, "available": True}
        for key in ("written_file_bytes", "written_raw_bytes", "written_blocks", "reused_blocks",
                    "verification_file_bytes", "write_failures", "compression_seconds"):
            if any(key in row for row in combined.values()):
                merged[key] = sum(row.get(key, 0) for row in combined.values())
        starts = [row["since"] for row in combined.values() if row.get("since")]
        if starts:
            merged["since"] = min(starts)
        return merged
    merged = {**prior, **current}
    for key in ("written_file_bytes", "written_raw_bytes", "written_blocks", "reused_blocks",
                "verification_file_bytes", "write_failures", "compression_seconds"):
        if key in prior or key in current:
            values = (prior.get(key, 0), current.get(key, 0))
            if any(type(value) not in (int, float) or not 0 <= value < float("inf") for value in values):
                raise ValueError("invalid snapshot traffic counters")
            merged[key] = max(values)
    merged["available"] = bool(prior.get("available") or current.get("available"))
    starts = [value for value in (prior.get("since"), current.get("since")) if value]
    if starts:
        merged["since"] = min(starts)
    return merged


def _test_chats_in_use(prefix: Path, tail_path: Path) -> set[str] | None:
    """Fail closed for a live namespace if its activity metadata is unavailable."""
    protected = set()
    samples = []
    for path, kind in ((Path(str(prefix) + "-scheduler.json"), "scheduler"),
                       (Path(str(prefix) + "-worker.json"), "worker"), (tail_path, "tail")):
        try:
            value = _maintenance_json(path, limit=1024 * 1024)
            samples.append(value)
            # Scheduler/worker records change at admission/handover, so they can
            # be old while an idle engine is healthy. The tail reporter supplies
            # the heartbeat; matching PIDs bind the event records to that engine.
            if kind == "tail" and not 0 <= time.time() - float(value["updated_at"]) <= 10:
                return None
            if kind == "scheduler":
                rows = value.get("requests", [])
            elif kind == "worker":
                residency = value["residency"]
                rows = [residency.get("active"), *residency.get("images", [])]
            else:
                rows = value.get("chats", [])
            for row in rows:
                if row is None:
                    continue
                chat_id = row.get("chat_id", "")
                if not isinstance(chat_id, str) or not ID.fullmatch(chat_id):
                    return None
                protected.add(chat_id)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None
    pid = samples[-1].get("pid")
    if type(pid) is not int or pid <= 0 or any(value.get("pid") != pid for value in samples):
        return None
    return protected


def _owned_test_files(directory: Path) -> list[Path]:
    files = []
    for path in directory.rglob("*"):
        relative = path.relative_to(directory)
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            if relative.parts[0] != "generations" or len(relative.parts) > 2:
                raise ValueError("unknown directory in test snapshot")
            if len(relative.parts) == 2 and not ID.fullmatch(path.name):
                raise ValueError("unknown test generation")
        elif stat.S_ISREG(mode) and (
            (len(relative.parts) == 1 and path.name in {"chat.json", "io.json", ".lock", ".io.lock"})
            or (len(relative.parts) == 1 and path.name.startswith(".pending-"))
            or (len(relative.parts) == 3 and relative.parts[0] == "generations"
                and ID.fullmatch(relative.parts[1])
                and (KEY.fullmatch(path.name) or path.name.startswith(".pending-")))
        ):
            files.append(path)
        else:
            raise ValueError("unknown file or symlink in test snapshot")
    return files


def purge_test_chats(cache_root: Path, *, abi=None, dry_run=False,
                     status_prefix=Path("/dev/shm/qwen-radiance-fair-public"),
                     tail_path=Path("/dev/shm/qwen-radiance-snapshot-tail.json")) -> dict:
    """Remove only labelled qualification snapshots, retaining traffic and tombstones.

    An exclusive chat lock drains readers/writers. A durable replacement generation
    and retired-generation list then stop late writes even in a live namespace.
    The tiny lock, deletion marker and io.json remain; their counters continue if a
    future test uses the same chat ID. No transcript or model payload is read.
    """
    root = Path(cache_root).expanduser().absolute()
    if abi is not None and (not isinstance(abi, str) or not ID.fullmatch(abi)):
        raise ValueError("invalid snapshot ABI")
    real_directory(root)
    if root.resolve() != root:
        raise ValueError("cache root must not contain symlinks")
    snapshots = root / "snapshots"
    real_directory(snapshots)
    result = {"schema": "urn:coherence:cache-purge-tests:v1", "dry_run": dry_run,
              "removed_file_bytes": 0, "chats": [], "skipped": []}
    ledger_path = root / "snapshot-retirements.json"
    ledger = _retirement_ledger(ledger_path)
    roots = [snapshots / abi] if abi else sorted(snapshots.iterdir())
    lock_fd = None
    if not dry_run:
        lock_fd = os.open(root / ".snapshot-retirement.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(lock_fd)
            result["skipped"].append({"reason": "retirement_busy"})
            return result
    try:
        if not dry_run:
            # The ledger may have changed while the maintenance lock was acquired.
            ledger = _retirement_ledger(ledger_path)
        for namespace in roots:
            if not ID.fullmatch(namespace.name) or namespace.is_symlink():
                continue
            real_directory(namespace)
            data = namespace / "data"
            if abi and (data.is_symlink() or not data.is_dir()):
                manifest = _maintenance_json(namespace / "abi.json")
                data_abi = manifest.get("storage", {}).get("data_abi", "")
                if not isinstance(data_abi, str) or not ID.fullmatch(data_abi):
                    raise ValueError("invalid snapshot data ABI reference")
                namespace = snapshots / data_abi
                real_directory(namespace)
                data = namespace / "data"
            if data.is_symlink() or not data.is_dir():
                continue  # Runtime aliases are not additional copies.
            real_directory(data)
            managed = data / FORMAT
            if not managed.exists():
                continue
            real_directory(managed)
            engine_fd = None
            protected = None
            try:
                engine_fd = os.open(managed / ".engine.lock", os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
                if not stat.S_ISREG(os.fstat(engine_fd).st_mode):
                    raise ValueError("invalid engine lease")
                try:
                    fcntl.flock(engine_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    protected = set()
                except BlockingIOError:
                    protected = _test_chats_in_use(status_prefix, tail_path)
            except (OSError, ValueError):
                pass
            try:
                for directory in sorted(managed.iterdir()):
                    if not ID.fullmatch(directory.name):
                        continue
                    real_directory(directory)
                    try:
                        info = _maintenance_json(directory / "chat.json", limit=1024 * 1024)
                    except (OSError, ValueError):
                        result["skipped"].append({"abi": namespace.name, "chat_id": directory.name,
                                                  "reason": "metadata_unreadable"})
                        continue
                    if (info.get("title") not in TEST_CHAT_TITLES
                            or info.get("cwd") not in {"/qualification", "/workspace/qualification"}):
                        continue
                    item = {"abi": namespace.name, "chat_id": directory.name, "title": info["title"]}
                    if protected is None or directory.name in protected:
                        result["skipped"].append({**item, "reason": "activity_unknown" if protected is None else "test_in_use"})
                        continue
                    descriptor = None
                    try:
                        descriptor = os.open(directory / ".lock", os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
                        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                            raise ValueError("invalid chat lock")
                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        info = _maintenance_json(directory / "chat.json", limit=1024 * 1024)
                        if (info.get("id") != directory.name or info.get("format") != FORMAT
                                or info.get("title") != item["title"]
                                or info.get("cwd") not in {"/qualification", "/workspace/qualification"}):
                            raise ValueError("test snapshot identity changed")
                        identity(info)
                        files = _owned_test_files(directory)
                        payloads = [path for path in files if path.parent != directory or path.name.startswith(".pending-")]
                        key = f"{namespace.name}/{directory.name}"
                        if info.get("status") == "purged" and not payloads and ledger["entries"].get(key, {}).get("status") == "complete":
                            continue
                        try:
                            current_io = _maintenance_json(directory / "io.json", limit=1024 * 1024)
                        except FileNotFoundError:
                            current_io = {"available": False}
                        item.update(tokens=info.get("tokens", 0), file_bytes=sum(p.stat().st_size for p in payloads), files=len(payloads))
                        if dry_run:
                            result["chats"].append(item)
                            continue
                        prior = ledger["entries"].get(key, {})
                        entry = {**item, "reason": "purged_test", "status": "pending",
                                 "retired_at": datetime.now(UTC).isoformat(),
                                 "io": _merge_io_history(prior.get("io", {}), current_io)}
                        ledger["entries"][key] = entry
                        atomic_write(ledger_path, json.dumps(ledger, sort_keys=True).encode())
                        generations = directory / "generations"
                        real_directory(generations)
                        if info.get("status") != "purged":
                            retired = set(info.get("retired_generations", [])) | {info["generation"]}
                            retired.update(p.name for p in generations.iterdir())
                            tombstone = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
                            (generations / tombstone).mkdir(mode=0o700)
                            sync_directory(generations)
                            marker = {**identity(info), "generation": tombstone, "format": FORMAT,
                                      "status": "purged", "tokens": 0, "head": [],
                                      "retired_generations": sorted(retired), "purged_at": entry["retired_at"]}
                            atomic_write(directory / "chat.json", json.dumps(marker, sort_keys=True).encode())
                            info = marker
                        for generation in generations.iterdir():
                            if generation.name != info["generation"]:
                                shutil.rmtree(generation)
                        for path in payloads:
                            if path.parent == directory:
                                path.unlink()
                        sync_directory(generations)
                        sync_directory(directory)
                        entry["status"] = "complete"
                        atomic_write(ledger_path, json.dumps(ledger, sort_keys=True).encode())
                        result["removed_file_bytes"] += item["file_bytes"]
                        result["chats"].append(item)
                    except (OSError, ValueError, TypeError) as error:
                        result["skipped"].append({**item, "reason": type(error).__name__})
                    finally:
                        if descriptor is not None:
                            os.close(descriptor)
            finally:
                if engine_fd is not None:
                    os.close(engine_fd)
        return result
    finally:
        if lock_fd is not None:
            os.close(lock_fd)


def retire_incompatible_snapshots(data_root: Path, *, apply=False, chat_ids=None) -> dict:
    """Retire owned snapshots in inactive, incompatible data namespaces.

    Runtime aliases of the current data directory are not copies. Other live
    engines retain their namespaces. The per-engine lock is held throughout
    deletion; a momentarily idle chat lock alone cannot prove safety. Only the
    production root's managed chat directories are eligible, never benchmark
    subdirectories, unknown files, or unlabelled legacy cache objects.
    """
    data_root = Path(data_root).resolve(strict=True)
    real_directory(data_root)
    namespace = data_root.parent
    snapshots = namespace.parent
    result = {"current_data_abi": namespace.name, "applied": apply,
              "removed_file_bytes": 0, "chats": [], "skipped": []}
    if data_root.name != "data" or snapshots.name != "snapshots" or not ID.fullmatch(namespace.name):
        return result
    if chat_ids is not None and any(not ID.fullmatch(value) for value in chat_ids):
        raise ValueError("invalid retirement chat identity")
    ledger_path = snapshots.parent / "snapshot-retirements.json"
    if ledger_path.is_symlink():
        raise ValueError("snapshot retirement ledger is a symlink")
    # One maintenance worker or operator at a time, without blocking a startup.
    lock_fd = os.open(snapshots.parent / ".snapshot-retirement.lock",
                      os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            result["skipped"].append({"reason": "retirement_busy"})
            return result
        ledger = _retirement_ledger(ledger_path)
        for old in sorted(snapshots.iterdir()):
            if old == namespace or not ID.fullmatch(old.name) or old.is_symlink():
                continue
            old_data = old / "data"
            if old_data.is_symlink() or not old_data.is_dir():
                continue
            managed = old_data / FORMAT
            if not managed.exists():
                continue
            real_directory(managed)
            engine_fd = None
            try:
                # Older/unknown layouts without a lifetime lease require an
                # explicit migration; do not infer ownership from file ages.
                engine_fd = os.open(managed / ".engine.lock", os.O_RDWR | os.O_NOFOLLOW)
                fcntl.flock(engine_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (OSError, ValueError) as error:
                if engine_fd is not None:
                    os.close(engine_fd)
                result["skipped"].append({"abi": old.name, "reason": type(error).__name__})
                continue
            try:
                for directory in sorted(managed.iterdir()):
                    if not ID.fullmatch(directory.name) or (
                        chat_ids is not None and directory.name not in chat_ids
                    ):
                        continue
                    real_directory(directory)
                    meta_path = directory / "chat.json"
                    if meta_path.is_symlink() or not meta_path.is_file():
                        key = f"{old.name}/{directory.name}"
                        pending = ledger["entries"].get(key, {})
                        if apply and pending.get("status") == "pending" and not any(directory.iterdir()):
                            # Crash after metadata unlink but before final rmdir.
                            directory.rmdir()
                            sync_directory(managed)
                            pending["status"] = "complete"
                            atomic_write(ledger_path, json.dumps(ledger, sort_keys=True).encode())
                            continue
                        result["skipped"].append({"abi": old.name, "chat_id": directory.name,
                                                  "reason": "unknown_metadata"})
                        continue
                    info = json.loads(meta_path.read_text())
                    if info.get("id") != directory.name or info.get("format") != FORMAT:
                        raise ValueError("retirement metadata identity mismatch")
                    store = ChatStore(old_data, info)
                    with store.lock(exclusive=True):
                        # Refuse unexpected material instead of recursively
                        # deleting something this cache implementation does not own.
                        files = []
                        for path in directory.rglob("*"):
                            relative = path.relative_to(directory)
                            if path.is_symlink():
                                raise ValueError("symlink in retired snapshot")
                            if path.is_dir():
                                if relative.parts[0] != "generations" or len(relative.parts) > 2:
                                    raise ValueError("unknown directory in retired snapshot")
                                if len(relative.parts) == 2 and not ID.fullmatch(path.name):
                                    raise ValueError("unknown retired generation")
                            elif path.is_file() and (
                                (len(relative.parts) == 1 and path.name in
                                 {"chat.json", "io.json", ".lock", ".io.lock"})
                                or (len(relative.parts) == 3 and relative.parts[0] == "generations"
                                    and ID.fullmatch(relative.parts[1])
                                    and (KEY.fullmatch(path.name) or path.name.startswith(".pending-")))
                                or (len(relative.parts) == 1 and path.name.startswith(".pending-"))
                            ):
                                files.append(path)
                            else:
                                raise ValueError("unknown file in retired snapshot")
                        entry = {"abi": old.name, "chat_id": directory.name,
                                 "tokens": info.get("tokens", 0),
                                 "file_bytes": sum(path.stat().st_size for path in files),
                                 "files": len(files), "replaced_by": namespace.name}
                        result["chats"].append(entry)
                        if apply:
                            key = f"{old.name}/{directory.name}"
                            # Keep numeric write-traffic history, not old KV payloads.
                            prior_entry = ledger["entries"].get(key, {})
                            entry["io"] = _merge_io_history(prior_entry.get("io", {}), store.io_totals())
                            ledger["entries"][key] = {**entry, "status": "pending"}
                            atomic_write(ledger_path, json.dumps(ledger, sort_keys=True).encode())
                            # Keep ownership metadata until all payload removal
                            # succeeds, so a crash/ENOSPC retry can still identify
                            # and collect the remaining files.
                            for generation in store.generations.iterdir():
                                shutil.rmtree(generation)
                            store.generations.rmdir()
                            for path in directory.iterdir():
                                if path != meta_path:
                                    path.unlink()
                            meta_path.unlink()
                            directory.rmdir()
                            sync_directory(managed)
                            ledger["entries"][key]["status"] = "complete"
                            atomic_write(ledger_path, json.dumps(ledger, sort_keys=True).encode())
                            result["removed_file_bytes"] += entry["file_bytes"]
            finally:
                os.close(engine_fd)
        return result
    finally:
        os.close(lock_fd)


def report(data_root: Path) -> dict:
    chats = []
    managed = data_root / FORMAT
    if managed.exists():
        real_directory(managed)
        for directory in sorted(managed.iterdir()):
            if not ID.fullmatch(directory.name):
                continue
            real_directory(directory)
            metadata_path = directory / "chat.json"
            if not metadata_path.exists():
                continue
            info = json.loads(metadata_path.read_text())
            if info.get("id") != directory.name:
                raise ValueError("chat metadata identity differs from its directory")
            store = ChatStore(data_root, info)
            with store.lock():
                info = store.metadata()
                sizes = {"files": 0, "file_bytes": 0, "allocated_bytes": 0, "raw_bytes": 0}
                for path in store.generations.rglob("*.qkv"):
                    if path.is_symlink():
                        raise ValueError(f"snapshot object is a symlink: {path}")
                    stat = path.stat()
                    with path.open("rb") as stream:
                        header = stream.read(HEADER.size)
                    sizes["files"] += 1
                    sizes["file_bytes"] += stat.st_size
                    sizes["allocated_bytes"] += stat.st_blocks * 512
                    if len(header) == HEADER.size:
                        sizes["raw_bytes"] += HEADER.unpack(header)[1]
                chats.append(
                    {
                        **{
                            k: v
                            for k, v in info.items()
                            if k not in ("head", "retired_generations")
                        },
                        **sizes,
                    }
                )
    legacy_files = legacy_bytes = 0
    for path in data_root.rglob("*.bin"):
        if path.is_symlink():
            continue
        legacy_files += 1
        legacy_bytes += path.stat().st_size
    return {
        "data_root": str(data_root),
        "chats": chats,
        "legacy_unassigned_files": legacy_files,
        "legacy_unassigned_bytes": legacy_bytes,
    }


def human(size: int) -> str:
    return f"{size / 1024**3:.2f} GiB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="ai")
    parser.add_argument("--cache-root", default=DEFAULT_ROOT)
    parser.add_argument("--abi", help="snapshot ABI directory; required for mutation commands")
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser(
        "list", help="list chats, compressed sizes, and unassigned legacy cache"
    )
    listing.add_argument("--json", action="store_true")
    compact = sub.add_parser("compact", help="retire a chat generation after Pi commits compaction")
    compact.add_argument("--identity-json", required=True)
    flush = sub.add_parser("flush", help="force the live backend to publish a chat's buffered tail")
    flush.add_argument("--identity-json", required=True)
    flush.add_argument("--timeout", type=float, default=120.0)
    purge = sub.add_parser("purge-tests", help="remove labelled qualification snapshots, preserving lifetime traffic")
    purge.add_argument("--dry-run", action="store_true")
    purge.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.host not in ("local", "localhost", "127.0.0.1"):
        remote_args = ["--host", "local", "--cache-root", args.cache_root]
        if args.abi:
            remote_args += ["--abi", args.abi]
        remote_args += [args.command]
        remote_args += ["--json"] if args.command == "list" and args.json else []
        if args.command in ("compact", "flush"):
            remote_args += ["--identity-json", args.identity_json]
        if args.command == "flush":
            remote_args += ["--timeout", str(args.timeout)]
        if args.command == "purge-tests":
            if args.dry_run:
                remote_args.append("--dry-run")
            if args.json:
                remote_args.append("--json")
        return subprocess.run(
            [
                "ssh",
                "-T",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                "--",
                args.host,
                shlex.join(["python3", "-", *remote_args]),
            ],
            input=Path(__file__).read_text(),
            text=True,
            check=False,
        ).returncode
    snapshots = Path(args.cache_root).expanduser() / "snapshots"
    if args.abi and not ID.fullmatch(args.abi):
        parser.error("--abi must be a SHA256 identifier")
    if args.command in ("compact", "flush") and not args.abi:
        parser.error(f"{args.command} requires --abi")
    if args.command == "flush":
        print(json.dumps(request_tail_flush(json.loads(args.identity_json), timeout=args.timeout)))
        return 0
    if args.command == "compact":
        result = ChatStore(snapshots / args.abi / "data", json.loads(args.identity_json)).activate()
        print(json.dumps(result))
        return 0
    if args.command == "purge-tests":
        try:
            result = purge_test_chats(Path(args.cache_root), abi=args.abi, dry_run=args.dry_run)
        except (OSError, ValueError) as error:
            print(json.dumps({"error": str(error)}))
            return 2
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            verb = "Would purge" if args.dry_run else "Purged"
            size = sum(row["file_bytes"] for row in result["chats"]) if args.dry_run else result["removed_file_bytes"]
            print(f"{verb} {len(result['chats'])} labelled test cache(s): {human(size)}")
            for row in result["chats"]:
                print(f"  {row['chat_id'][:12]}/{row['abi'][:8]}  {row['title']}  {human(row['file_bytes'])}")
            for row in result["skipped"]:
                print(f"  Skipped {row.get('chat_id', row.get('abi', 'maintenance'))[:12]}: {row['reason']}")
            print("Lifetime write counters and tiny deletion markers are retained; transcripts are untouched.")
        return 2 if result["skipped"] else 0
    roots = [snapshots / args.abi] if args.abi else sorted(snapshots.iterdir())
    reports = [report(root / "data") for root in roots if (root / "data").is_dir()]
    if args.json:
        print(json.dumps(reports, indent=2))
        return 0
    for value in reports:
        print(value["data_root"])
        print(
            f"{'CHAT':12}  {'DISK FILES':>12}  {'RAW CACHE':>12}  "
            f"{'TOKENS':>8}  {'STATE':10}  TITLE / DIRECTORY"
        )
        for chat in value["chats"]:
            print(
                f"{chat['id'][:12]}  {human(chat['file_bytes']):>12}  "
                f"{human(chat['raw_bytes']):>12}  "
                f"{chat.get('tokens', 0):>8}  {chat.get('status', 'unknown'):10}  "
                f"{chat.get('title') or chat.get('cwd')}"
            )
        print(
            f"Unassigned legacy cache: {human(value['legacy_unassigned_bytes'])} "
            f"({value['legacy_unassigned_files']} files; preserved)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Crash-safe, fresh-process checkpoints for pinned AutoRound quantization.

This module deliberately has no AutoRound or Torch import at module import time.  The
launcher installs :class:`RestartableAutoRoundRuntime` only after authenticating the
exact AutoRound sources and supplies Torch serialization callbacks at runtime.
"""

from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import os
import stat
import time
from collections import OrderedDict
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

CHECKPOINT_SCHEMA = "urn:qwen-r9700:autoround-segment-checkpoint:v1"
JOB_SCHEMA = "urn:qwen-r9700:autoround-restartable-job:v1"
POINTER_SCHEMA = "urn:qwen-r9700:autoround-restartable-pointer:v1"
RETIREMENT_SCHEMA = "urn:qwen-r9700:autoround-checkpoint-payload-retirement:v1"


class RestartContractError(RuntimeError):
    """Raised when resumable state does not satisfy the authenticated contract."""


class SegmentCompleteError(RuntimeError):
    """Private control-flow signal used to restart in a clean worker process."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def protected_directory(path: Path) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise RestartContractError(f"missing protected directory: {path}") from exc
    if not stat.S_ISDIR(info.st_mode) or path.is_symlink():
        raise RestartContractError(f"protected path is not a real directory: {path}")
    if info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise RestartContractError(f"protected directory has unsafe ownership or mode: {path}")
    return info


def protected_regular_file(path: Path) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise RestartContractError(f"missing protected file: {path}") from exc
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise RestartContractError(f"protected path is not a regular file: {path}")
    if info.st_uid != os.getuid() or info.st_nlink != 1 or info.st_mode & 0o022:
        raise RestartContractError(f"protected file has unsafe ownership, links, or mode: {path}")
    return info


def stable_file_record(path: Path, relative_to: Path) -> dict[str, Any]:
    before = protected_regular_file(path)
    digest = sha256_file(path)
    after = path.lstat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise RestartContractError(f"file changed while it was authenticated: {path}")
    try:
        relative = path.relative_to(relative_to).as_posix()
    except ValueError as exc:
        raise RestartContractError(f"checkpoint file escapes its root: {path}") from exc
    return {"path": relative, "bytes": before.st_size, "sha256": digest}


def verify_file_record(root: Path, record: dict[str, Any]) -> Path:
    if set(record) != {"path", "bytes", "sha256"}:
        raise RestartContractError("checkpoint file record has an unexpected shape")
    relative = Path(str(record["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise RestartContractError("checkpoint file record escapes its root")
    path = root / relative
    before = protected_regular_file(path)
    if before.st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
        raise RestartContractError(f"checkpoint file differs from its sealed identity: {path}")
    after = path.lstat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RestartContractError(f"checkpoint file changed during verification: {path}")
    return path


def _write_new_file(path: Path, payload: bytes, mode: int = 0o600) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)


def atomic_replace_bytes(path: Path, payload: bytes, mode: int = 0o600) -> None:
    temporary = path.parent / f".{path.name}.{os.getpid()}.{os.urandom(12).hex()}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            mode,
        )
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        temporary.replace(path)
        fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            temporary.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    before = protected_regular_file(path)
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RestartContractError(f"invalid checkpoint JSON: {path}: {exc}") from exc
    after = path.lstat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RestartContractError(f"checkpoint JSON changed while it was read: {path}")
    if not isinstance(value, dict):
        raise RestartContractError(f"checkpoint JSON is not an object: {path}")
    return value


class DurableCheckpointStore:
    """Append-only checkpoint chain with one atomically replaced head pointer."""

    def __init__(self, root: Path, stage: Path, job: dict[str, Any]) -> None:
        self.root = root
        self.stage = stage
        self.checkpoints = root / "checkpoints"
        self.orphans = root / "orphans"
        self.job_path = root / "job.json"
        self.pointer_path = root / "current.json"
        self.job = {"schema": JOB_SCHEMA, **job}
        self.job_bytes = canonical_json(self.job) + b"\n"
        self.job_sha256 = sha256_bytes(self.job_bytes)

    def initialize(self) -> None:
        if not self.root.exists() and not self.root.is_symlink():
            protected_directory(self.root.parent)
            self.root.mkdir(mode=0o700)
            fsync_directory(self.root.parent)
        protected_directory(self.root)
        for directory in (self.checkpoints, self.orphans):
            if not directory.exists() and not directory.is_symlink():
                directory.mkdir(mode=0o700)
                fsync_directory(self.root)
            protected_directory(directory)
        if not self.job_path.exists() and not self.job_path.is_symlink():
            _write_new_file(self.job_path, self.job_bytes)
        elif self.job_path.read_bytes() != self.job_bytes:
            raise RestartContractError("restartable job identity differs from the existing job")
        protected_regular_file(self.job_path)

    def load(self) -> dict[str, Any] | None:
        if not self.pointer_path.exists() and not self.pointer_path.is_symlink():
            return None
        pointer = _read_json(self.pointer_path)
        if pointer.get("schema") != POINTER_SCHEMA or pointer.get("job_sha256") != self.job_sha256:
            raise RestartContractError("checkpoint pointer does not belong to this job")
        relative = Path(str(pointer.get("manifest", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise RestartContractError("checkpoint pointer manifest path escapes its root")
        manifest_path = self.root / relative
        before = protected_regular_file(manifest_path)
        payload = manifest_path.read_bytes()
        after = manifest_path.lstat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RestartContractError("checkpoint manifest changed while it was read")
        if sha256_bytes(payload) != pointer.get("manifest_sha256"):
            raise RestartContractError("checkpoint pointer manifest digest differs")
        try:
            manifest = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RestartContractError("checkpoint manifest is invalid JSON") from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema") != CHECKPOINT_SCHEMA
            or manifest.get("job_sha256") != self.job_sha256
            or manifest.get("sequence") != pointer.get("sequence")
        ):
            raise RestartContractError("checkpoint manifest identity differs")
        expected_keys = {
            "schema",
            "sequence",
            "job_sha256",
            "previous_manifest_sha256",
            "phase",
            "completed_blocks",
            "group_progress",
            "writer",
            "shards",
            "continuation",
            "memory",
            "sealed_at_ns",
        }
        if set(manifest) != expected_keys:
            raise RestartContractError("checkpoint manifest has an unexpected field set")
        if manifest.get("phase") not in {"quantizing", "group-complete", "finalizing"}:
            raise RestartContractError("checkpoint phase is invalid")
        completed = manifest.get("completed_blocks")
        if (
            not isinstance(completed, list)
            or not all(isinstance(name, str) and name for name in completed)
            or len(completed) != len(set(completed))
        ):
            raise RestartContractError("checkpoint completed-block ledger is invalid")
        if not isinstance(manifest.get("group_progress"), dict) or not isinstance(
            manifest.get("writer"), dict
        ):
            raise RestartContractError("checkpoint progress or writer state is invalid")
        continuation = manifest.get("continuation")
        if continuation is not None:
            verify_file_record(self.root, continuation)
        if (manifest["phase"] == "quantizing") != (continuation is not None):
            raise RestartContractError("checkpoint phase and continuation presence differ")
        self._verify_shards(manifest)
        return manifest

    def _verify_shards(self, manifest: dict[str, Any]) -> None:
        shards = manifest.get("shards")
        if not isinstance(shards, list):
            raise RestartContractError("checkpoint shard ledger is not a list")
        seen: set[str] = set()
        for record in shards:
            if not isinstance(record, dict) or set(record) != {
                "path",
                "bytes",
                "sha256",
                "final_path",
            }:
                raise RestartContractError("checkpoint shard record has an unexpected shape")
            path_value = str(record["path"])
            final_value = str(record["final_path"])
            if path_value in seen:
                raise RestartContractError("checkpoint shard ledger contains a duplicate")
            seen.add(path_value)
            candidates = [path_value]
            if manifest.get("phase") == "finalizing":
                candidates.append(final_value)
            matches = []
            for relative_value in candidates:
                relative = Path(relative_value)
                if relative.is_absolute() or ".." in relative.parts:
                    raise RestartContractError("checkpoint shard path escapes the stage")
                candidate = self.stage / relative
                if candidate.exists() or candidate.is_symlink():
                    matches.append(candidate)
            if len(matches) != 1:
                raise RestartContractError(
                    f"checkpoint shard must have exactly one authenticated location: {path_value}"
                )
            observed = stable_file_record(matches[0], self.stage)
            if observed["bytes"] != record["bytes"] or observed["sha256"] != record["sha256"]:
                raise RestartContractError(f"checkpoint shard differs: {matches[0]}")

    def continuation_path(self, manifest: dict[str, Any]) -> Path:
        record = manifest.get("continuation")
        if not isinstance(record, dict):
            raise RestartContractError("active checkpoint has no continuation state")
        return verify_file_record(self.root, record)

    def quarantine_uncommitted_shards(self, manifest: dict[str, Any] | None) -> list[Path]:
        """Move only unreferenced AutoRound temporary shards into recoverable storage."""

        referenced = set()
        if manifest is not None:
            referenced = {str(record["path"]) for record in manifest["shards"]}
        uncommitted: list[Path] = []
        for directory, directory_names, file_names in os.walk(self.stage, followlinks=False):
            directory_path = Path(directory)
            for name in list(directory_names):
                candidate = directory_path / name
                if candidate.is_symlink():
                    raise RestartContractError(
                        f"symlink directory is forbidden in stage: {candidate}"
                    )
            for name in file_names:
                suffix = next(
                    (
                        candidate
                        for candidate in (".safetensors", ".bin")
                        if name.endswith(candidate)
                    ),
                    None,
                )
                ordinal = (
                    ""
                    if suffix is None or not name.startswith("model-shard-")
                    else name[len("model-shard-") : -len(suffix)]
                )
                if len(ordinal) != 5 or not ordinal.isdigit():
                    continue
                candidate = directory_path / name
                relative = candidate.relative_to(self.stage).as_posix()
                if relative not in referenced:
                    protected_regular_file(candidate)
                    uncommitted.append(candidate)
        if not uncommitted:
            return []
        destination_root = self.orphans / f"{time.time_ns()}-{os.urandom(8).hex()}"
        destination_root.mkdir(mode=0o700)
        fsync_directory(self.orphans)
        moved: list[Path] = []
        for source in uncommitted:
            relative = source.relative_to(self.stage)
            destination = destination_root / relative
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                raise RestartContractError(f"orphan destination already exists: {destination}")
            source.rename(destination)
            fsync_directory(source.parent)
            fsync_directory(destination.parent)
            moved.append(destination)
        return moved

    def seal(
        self,
        *,
        phase: str,
        completed_blocks: list[str],
        group_progress: dict[str, Any],
        writer: dict[str, Any],
        shards: list[dict[str, Any]],
        write_continuation: Callable[[Path], None] | None,
        memory: dict[str, int],
    ) -> dict[str, Any]:
        previous = self.load()
        sequence = 1 if previous is None else int(previous["sequence"]) + 1
        checkpoint_dir = self.checkpoints / f"{sequence:06d}-{os.urandom(12).hex()}"
        checkpoint_dir.mkdir(mode=0o700)
        fsync_directory(self.checkpoints)
        continuation = None
        if write_continuation is not None:
            temporary = checkpoint_dir / "continuation.pt.part"
            final = checkpoint_dir / "continuation.pt"
            write_continuation(temporary)
            descriptor = os.open(temporary, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            temporary.replace(final)
            fsync_directory(checkpoint_dir)
            continuation = stable_file_record(final, self.root)
        previous_digest = None
        if previous is not None:
            previous_path = self.root / str(_read_json(self.pointer_path)["manifest"])
            previous_digest = sha256_file(previous_path)
        manifest = {
            "schema": CHECKPOINT_SCHEMA,
            "sequence": sequence,
            "job_sha256": self.job_sha256,
            "previous_manifest_sha256": previous_digest,
            "phase": phase,
            "completed_blocks": completed_blocks,
            "group_progress": group_progress,
            "writer": writer,
            "shards": shards,
            "continuation": continuation,
            "memory": memory,
            "sealed_at_ns": time.time_ns(),
        }
        manifest_path = checkpoint_dir / "manifest.json"
        payload = canonical_json(manifest) + b"\n"
        _write_new_file(manifest_path, payload)
        relative = manifest_path.relative_to(self.root).as_posix()
        pointer = {
            "schema": POINTER_SCHEMA,
            "job_sha256": self.job_sha256,
            "sequence": sequence,
            "manifest": relative,
            "manifest_sha256": sha256_bytes(payload),
        }
        atomic_replace_bytes(self.pointer_path, canonical_json(pointer) + b"\n")
        return manifest

    def retire_old_continuations(self, retain: int) -> list[Path]:
        """Retire only non-head continuation payloads after a newer head is durable."""

        if retain < 1:
            raise RestartContractError("at least one checkpoint payload must be retained")
        head = self.load()
        if head is None:
            return []
        manifests: list[tuple[int, Path, dict[str, Any]]] = []
        for checkpoint_dir in sorted(self.checkpoints.iterdir()):
            protected_directory(checkpoint_dir)
            manifest_path = checkpoint_dir / "manifest.json"
            if not manifest_path.exists() and not manifest_path.is_symlink():
                continue
            manifest = _read_json(manifest_path)
            if (
                manifest.get("schema") != CHECKPOINT_SCHEMA
                or manifest.get("job_sha256") != self.job_sha256
                or not isinstance(manifest.get("sequence"), int)
            ):
                raise RestartContractError("checkpoint retirement found a foreign manifest")
            continuation = manifest.get("continuation")
            if continuation is not None:
                manifests.append((int(manifest["sequence"]), checkpoint_dir, continuation))
        manifests.sort(key=lambda item: item[0])
        keep_sequences = {item[0] for item in manifests[-retain:]}
        retired: list[Path] = []
        for sequence, checkpoint_dir, record in manifests:
            if sequence in keep_sequences:
                continue
            if sequence >= int(head["sequence"]):
                raise RestartContractError("checkpoint retirement selected the active head")
            continuation_path = self.root / str(record["path"])
            receipt_path = checkpoint_dir / "continuation.retired.json"
            receipt = {
                "schema": RETIREMENT_SCHEMA,
                "sequence": sequence,
                "job_sha256": self.job_sha256,
                "continuation": record,
            }
            receipt_bytes = canonical_json(receipt) + b"\n"
            if receipt_path.exists() or receipt_path.is_symlink():
                if _read_json(receipt_path) != receipt:
                    raise RestartContractError("checkpoint retirement receipt differs")
            else:
                if not continuation_path.exists() and not continuation_path.is_symlink():
                    raise RestartContractError(
                        "checkpoint continuation disappeared before retirement was recorded"
                    )
                verify_file_record(self.root, record)
                _write_new_file(receipt_path, receipt_bytes)
            if continuation_path.exists() or continuation_path.is_symlink():
                verify_file_record(self.root, record)
                continuation_path.unlink()
                fsync_directory(checkpoint_dir)
                retired.append(continuation_path)
        return retired


class RestartableAutoRoundRuntime:
    """State adapter invoked by a source-authenticated AutoRound method overlay."""

    def __init__(
        self,
        store: DurableCheckpointStore,
        *,
        segment_blocks: int,
        max_rss_bytes: int,
        min_free_bytes: int,
        tensor_save: Callable[[dict[str, Any], Path], None],
        tensor_load: Callable[[Path], dict[str, Any]],
        expected_blocks: list[str] | None = None,
    ) -> None:
        if segment_blocks < 1:
            raise RestartContractError("segment_blocks must be positive")
        self.store = store
        self.segment_blocks = segment_blocks
        self.max_rss_bytes = max_rss_bytes
        self.min_free_bytes = min_free_bytes
        self.tensor_save = tensor_save
        self.tensor_load = tensor_load
        self.expected_blocks = None if expected_blocks is None else list(expected_blocks)
        self.manifest = store.load()
        store.quarantine_uncommitted_shards(self.manifest)
        self.completed_blocks = (
            [] if self.manifest is None else list(self.manifest["completed_blocks"])
        )
        self.group_progress = {} if self.manifest is None else dict(self.manifest["group_progress"])
        self._restored = False
        self._worker_blocks = 0
        self._active_group: str | None = None
        self._active_names: list[str] = []

    @staticmethod
    def group_key(block_names: list[str]) -> str:
        return sha256_bytes(canonical_json(block_names))

    @staticmethod
    def _writer_state(writer: Any, stage: Path) -> dict[str, Any]:
        shard_meta = []
        for item in writer.shard_meta:
            directory = Path(item.get("dir", writer.output_dir))
            try:
                relative_dir = directory.relative_to(stage).as_posix()
            except ValueError as exc:
                raise RestartContractError("AutoRound shard directory escapes the stage") from exc
            shard_meta.append(
                {
                    "tmp_file": str(item["tmp_file"]),
                    "params": list(item["params"]),
                    "dir": relative_dir,
                }
            )
        state = {
            "shard_counter": int(writer.shard_counter),
            "shard_meta": shard_meta,
            "all_saved": sorted(writer._all_saved),
            "global_weight_map": dict(writer.global_weight_map),
            "total_param_elems": int(writer.total_param_elems),
            "total_param_size_bytes": int(writer.total_param_size_bytes),
            "skipped_meta_tensors": list(writer.skipped_meta_tensors),
            "shard_suffix": str(writer.shard_suffix),
        }
        if state["shard_counter"] != len(shard_meta):
            raise RestartContractError("AutoRound writer counter and shard metadata differ")
        params = [name for item in shard_meta for name in item["params"]]
        if len(params) != len(set(params)) or set(params) != set(state["all_saved"]):
            raise RestartContractError("AutoRound writer saved-parameter ledger is inconsistent")
        return state

    @staticmethod
    def _validate_writer_state(state: dict[str, Any]) -> None:
        expected_keys = {
            "shard_counter",
            "shard_meta",
            "all_saved",
            "global_weight_map",
            "total_param_elems",
            "total_param_size_bytes",
            "skipped_meta_tensors",
            "shard_suffix",
        }
        if set(state) != expected_keys:
            raise RestartContractError("saved writer state has an unexpected field set")
        if state["shard_suffix"] not in {"safetensors", "bin"}:
            raise RestartContractError("saved writer shard suffix is invalid")
        if (
            not isinstance(state["shard_meta"], list)
            or not isinstance(state["all_saved"], list)
            or not isinstance(state["global_weight_map"], dict)
        ):
            raise RestartContractError("saved writer collections are invalid")
        if (
            not all(isinstance(name, str) and name for name in state["all_saved"])
            or len(state["all_saved"]) != len(set(state["all_saved"]))
            or not all(
                isinstance(name, str) and isinstance(shard, str)
                for name, shard in state["global_weight_map"].items()
            )
            or not isinstance(state["total_param_elems"], int)
            or state["total_param_elems"] < 0
            or not isinstance(state["total_param_size_bytes"], int)
            or state["total_param_size_bytes"] < 0
        ):
            raise RestartContractError("saved writer totals or name maps are invalid")
        params: list[str] = []
        for item in state["shard_meta"]:
            if not isinstance(item, dict) or set(item) != {"tmp_file", "params", "dir"}:
                raise RestartContractError("saved writer shard metadata is invalid")
            if (
                not isinstance(item["tmp_file"], str)
                or not isinstance(item["dir"], str)
                or not isinstance(item["params"], list)
                or not all(isinstance(name, str) and name for name in item["params"])
            ):
                raise RestartContractError("saved writer shard metadata values are invalid")
            temporary_name = Path(item["tmp_file"])
            if (
                temporary_name.name != item["tmp_file"]
                or not item["tmp_file"].startswith("model-shard-")
                or not item["tmp_file"].endswith(f".{state['shard_suffix']}")
            ):
                raise RestartContractError("saved writer temporary shard name is invalid")
            params.extend(item["params"])
        if (
            not isinstance(state["shard_counter"], int)
            or state["shard_counter"] != len(state["shard_meta"])
            or len(params) != len(set(params))
            or set(params) != set(state["all_saved"])
        ):
            raise RestartContractError("saved writer parameter ledger is inconsistent")

    @staticmethod
    def _restore_writer(writer: Any, stage: Path, state: dict[str, Any]) -> None:
        RestartableAutoRoundRuntime._validate_writer_state(state)
        if writer.current_shard_tensors or writer.current_shard_size:
            raise RestartContractError("AutoRound writer is nonempty before checkpoint restoration")
        restored_meta = []
        for item in state["shard_meta"]:
            relative = Path(str(item["dir"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise RestartContractError("saved shard directory escapes the stage")
            restored_meta.append(
                {
                    "tmp_file": str(item["tmp_file"]),
                    "params": list(item["params"]),
                    "dir": str(stage / relative),
                }
            )
        writer.shard_counter = int(state["shard_counter"])
        writer.shard_meta = restored_meta
        writer._all_saved = set(state["all_saved"])
        writer.global_weight_map = dict(state["global_weight_map"])
        writer.total_param_elems = int(state["total_param_elems"])
        writer.total_param_size_bytes = int(state["total_param_size_bytes"])
        writer.skipped_meta_tensors = list(state["skipped_meta_tensors"])
        writer.current_shard_tensors = OrderedDict()
        writer.current_shard_size = 0

    @staticmethod
    def _memory_record() -> dict[str, int]:
        rss_pages = int(Path("/proc/self/statm").read_text().split()[1])
        return {"rss_bytes": rss_pages * os.sysconf("SC_PAGE_SIZE"), "pid": os.getpid()}

    def _shard_records(self, writer: Any, finalizing: bool) -> list[dict[str, Any]]:
        records = []
        count = int(writer.shard_counter)
        for index, item in enumerate(writer.shard_meta, start=1):
            directory = Path(item.get("dir", writer.output_dir))
            old_path = directory / item["tmp_file"]
            new_name = (
                f"model.{writer.shard_suffix}"
                if count == 1
                else f"model-{index:05d}-of-{count:05d}.{writer.shard_suffix}"
            )
            new_path = directory / new_name
            candidates = [old_path, new_path] if finalizing else [old_path]
            existing = [path for path in candidates if path.exists() or path.is_symlink()]
            if len(existing) != 1:
                raise RestartContractError("AutoRound shard has an ambiguous or missing location")
            record = stable_file_record(existing[0], self.store.stage)
            record["path"] = old_path.relative_to(self.store.stage).as_posix()
            record["final_path"] = new_path.relative_to(self.store.stage).as_posix()
            records.append(record)
        return records

    def _restore_once(self, compressor: Any, model: Any) -> None:
        if self._restored:
            return
        if self.manifest is not None:
            self._restore_writer(compressor.shard_writer, self.store.stage, self.manifest["writer"])
            get_module = __import__("auto_round.utils", fromlist=["get_module"]).get_module
            for name in self.completed_blocks:
                module = get_module(model, name)
                if module is None:
                    raise RestartContractError(f"completed AutoRound block is missing: {name}")
                module.to("meta")
            compressor.shard_writer._offload_to_meta(list(compressor.shard_writer._all_saved))
        self._restored = True

    def before_group(
        self,
        compressor: Any,
        model: Any,
        block_names: list[str],
        input_ids: Any,
        q_input: Any,
        input_others: dict[str, Any],
        nblocks: int,
    ) -> tuple[int, Any, Any, dict[str, Any]]:
        if nblocks != 1:
            raise RestartContractError("restartable AutoRound requires nblocks=1")
        if not compressor.compress_context.is_immediate_saving:
            raise RestartContractError("restartable AutoRound requires immediate shard saving")
        if not compressor.compress_context.is_immediate_packing:
            raise RestartContractError("restartable AutoRound requires immediate weight packing")
        self._restore_once(compressor, model)
        key = self.group_key(block_names)
        self._active_group = key
        self._active_names = list(block_names)
        progress = self.group_progress.get(key, {"block_names": list(block_names), "next_index": 0})
        if progress.get("block_names") != list(block_names):
            raise RestartContractError("AutoRound block group changed across restart")
        start = int(progress.get("next_index", 0))
        if start < 0 or start > len(block_names):
            raise RestartContractError("AutoRound checkpoint block index is out of range")
        if start == 0 or start == len(block_names):
            return start, input_ids, q_input, input_others
        if self.manifest is None or self.manifest.get("phase") != "quantizing":
            raise RestartContractError("active block group has no quantization continuation")
        payload = self.tensor_load(self.store.continuation_path(self.manifest))
        if payload.get("group_key") != key or payload.get("next_index") != start:
            raise RestartContractError("tensor continuation identity differs from block progress")
        return start, payload["input_ids"], payload.get("q_input"), payload["input_others"]

    def after_block(
        self,
        compressor: Any,
        model: Any,
        block_names: list[str],
        index: int,
        input_ids: Any,
        q_input: Any,
        input_others: dict[str, Any],
        nblocks: int,
    ) -> None:
        if self._active_group != self.group_key(block_names) or block_names != self._active_names:
            raise RestartContractError("AutoRound active block group changed unexpectedly")
        completed_now = block_names[index : index + nblocks]
        for name in completed_now:
            if name in self.completed_blocks:
                raise RestartContractError(f"AutoRound attempted a completed block again: {name}")
            self.completed_blocks.append(name)
        next_index = min(index + nblocks, len(block_names))
        self.group_progress[self._active_group] = {
            "block_names": list(block_names),
            "next_index": next_index,
        }
        self._worker_blocks += len(completed_now)
        gc.collect()
        trim_process_heap()
        memory = self._memory_record()
        boundary = (
            self._worker_blocks >= self.segment_blocks
            or next_index == len(block_names)
            or (self.max_rss_bytes > 0 and memory["rss_bytes"] >= self.max_rss_bytes)
        )
        if not boundary:
            return
        free_bytes = os.statvfs(self.store.root).f_bavail * os.statvfs(self.store.root).f_frsize
        if free_bytes < self.min_free_bytes:
            raise RestartContractError(
                f"checkpoint filesystem has {free_bytes} free bytes; need {self.min_free_bytes}"
            )
        compressor.shard_writer._flush_shard()
        writer_state = self._writer_state(compressor.shard_writer, self.store.stage)
        shards = self._shard_records(compressor.shard_writer, finalizing=False)

        def save(path: Path) -> None:
            self.tensor_save(
                {
                    "group_key": self._active_group,
                    "next_index": next_index,
                    "input_ids": input_ids,
                    "q_input": q_input,
                    "input_others": input_others,
                },
                path,
            )

        group_complete = next_index == len(block_names)
        self.manifest = self.store.seal(
            phase="group-complete" if group_complete else "quantizing",
            completed_blocks=list(self.completed_blocks),
            group_progress=dict(self.group_progress),
            writer=writer_state,
            shards=shards,
            write_continuation=None if group_complete else save,
            memory=memory,
        )
        raise SegmentCompleteError(
            f"sealed segment {self.manifest['sequence']} through "
            f"{len(self.completed_blocks)} blocks"
        )

    def before_final_renames(self, writer: Any) -> None:
        if not self._restored:
            raise RestartContractError("AutoRound finalization began before checkpoint restoration")
        if self.expected_blocks is not None and self.completed_blocks != self.expected_blocks:
            raise RestartContractError(
                "AutoRound reached finalization without the exact ordered block ledger"
            )
        current_state = self._writer_state(writer, self.store.stage)
        if self.manifest is not None and self.manifest.get("phase") == "finalizing":
            if current_state != self.manifest.get("writer"):
                raise RestartContractError(
                    "AutoRound finalizing writer state changed across restart"
                )
            return
        self.manifest = self.store.seal(
            phase="finalizing",
            completed_blocks=list(self.completed_blocks),
            group_progress=dict(self.group_progress),
            writer=current_state,
            shards=self._shard_records(writer, finalizing=False),
            write_continuation=None,
            memory=self._memory_record(),
        )

    def finalize_rename(self, old_path: str, new_path: str) -> None:
        old = Path(old_path)
        new = Path(new_path)
        if old == new:
            protected_regular_file(old)
            return
        if old.exists() and not old.is_symlink() and not (new.exists() or new.is_symlink()):
            old.rename(new)
            fsync_directory(new.parent)
            return
        if not (old.exists() or old.is_symlink()) and new.exists() and not new.is_symlink():
            if self.manifest is None or self.manifest.get("phase") != "finalizing":
                raise RestartContractError(
                    "final shard exists without a sealed finalization ledger"
                )
            expected = next(
                (
                    item
                    for item in self.manifest["shards"]
                    if self.store.stage / item["path"] == old
                    and self.store.stage / item["final_path"] == new
                ),
                None,
            )
            if expected is None:
                raise RestartContractError("renamed shard is absent from finalization ledger")
            observed = stable_file_record(new, self.store.stage)
            if observed["bytes"] != expected["bytes"] or observed["sha256"] != expected["sha256"]:
                raise RestartContractError("renamed shard differs from finalization ledger")
            return
        raise RestartContractError("final shard rename has an ambiguous filesystem state")


def instrument_data_driven_source(source: str) -> str:
    """Return the exact pinned method source with three explicit runtime hooks."""

    replacements = {
        "        input_ids, input_others = self._preprocess_block_inputs(inputs)\n": (
            "        input_ids, input_others = self._preprocess_block_inputs(inputs)\n"
            "        _qwen_start, input_ids, q_input, input_others = "
            "_qwen_restartable_runtime.before_group(\n"
            "            self, model, block_names, input_ids, q_input, input_others, nblocks\n"
            "        )\n"
        ),
        "        for i in range(0, len(block_names), nblocks):\n": (
            "        for i in range(_qwen_start, len(block_names), nblocks):\n"
        ),
        (
            "            if self.compress_context.low_cpu_mem_usage and not "
            "self.compress_context.is_immediate_saving:\n"
        ): (
            "            _qwen_restartable_runtime.after_block(\n"
            "                self, model, block_names, i, input_ids, q_input, "
            "input_others, nblocks\n"
            "            )\n\n"
            "            if self.compress_context.low_cpu_mem_usage and not "
            "self.compress_context.is_immediate_saving:\n"
        ),
    }
    result = source
    for needle, replacement in replacements.items():
        if result.count(needle) != 1:
            raise RestartContractError(f"pinned AutoRound method anchor count differs: {needle!r}")
        result = result.replace(needle, replacement)
    return result


def instrument_shard_writer_source(source: str) -> str:
    replacements = {
        "        self._flush_shard()\n\n        total_skipped =": (
            "        self._flush_shard()\n"
            "        _qwen_restartable_runtime.before_final_renames(self)\n\n"
            "        total_skipped ="
        ),
        "            os.rename(old_path, new_path)\n": (
            "            _qwen_restartable_runtime.finalize_rename(old_path, new_path)\n"
        ),
    }
    result = source
    for needle, replacement in replacements.items():
        if result.count(needle) != 1:
            raise RestartContractError(
                f"pinned ShardWriter method anchor count differs: {needle!r}"
            )
        result = result.replace(needle, replacement)
    return result


def trim_process_heap() -> None:
    """Best-effort release of free glibc arenas between quantized blocks."""

    with suppress(AttributeError):
        ctypes.CDLL(None).malloc_trim(0)

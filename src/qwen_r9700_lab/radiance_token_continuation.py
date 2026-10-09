"""Private generated-token continuation journals, independent of KV ownership.

This implements an explicit *generated-token* session contract. It does not
claim equality to a chat API which retokenizes all historical assistant text.
The adapter remains responsible for checking actual message/configuration
identity and computing canonical_previous_ids with its authenticated tokenizer.
Only then may an exact canonical prefix identify where to append new tokens to
the original generated token history. Text equality alone never admits reuse.

No token values, content hashes, salts, paths or exception text are logged.
Journal payloads are immutable packed CPU int32 values. Durable storage is an
optional private cache; a missing, corrupt, unsafe or contended cache falls back.
The model cache, not this emitted-token journal, owns the processed boundary.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
import struct
import threading
from array import array
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = "urn:coherence:generated-token-continuation:v1"
CONTRACT = "generated_tokens_v1"
DEFAULT_MAX_TOKENS = 253_792
MAX_TOKENS = 300_000
MAX_BYTES = 32 * 1024 * 1024
MAX_HEADER = 16 * 1024
MAX_DIRECTORY_ENTRIES = 256
MAGIC = b"COHTOK01"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX32 = re.compile(r"[0-9a-f]{32}\Z")
FILE_NAME = re.compile(r"[0-9a-f]{64}\.ctok\Z")


class _InvalidJournal(ValueError):
    pass


@dataclass(frozen=True, repr=False)
class PackedTokenIds(Sequence[int]):
    """An immutable CPU-only sequence with four retained bytes per token."""

    _data: bytes = field(repr=False)

    def __post_init__(self):
        if not isinstance(self._data, bytes) or len(self._data) % 4:
            raise ValueError("invalid packed token storage")

    def __len__(self):
        return len(self._data) // 4

    def __iter__(self) -> Iterator[int]:
        return (row[0] for row in struct.iter_unpack("<I", self._data))

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step == 1:
                return PackedTokenIds(self._data[start * 4 : stop * 4])
            return _pack([self[i] for i in range(start, stop, step)], len(self))
        if not isinstance(index, int):
            raise TypeError("token index must be an integer")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("token index outside journal")
        return struct.unpack_from("<I", self._data, index * 4)[0]

    def __repr__(self):
        return f"PackedTokenIds(count={len(self)})"


def _pack(values, limit):
    # Never invoke a tensor/array protocol or copy data from a device.
    if not isinstance(values, (list, tuple, array, range, PackedTokenIds)):
        raise _InvalidJournal("unsupported token container")
    if len(values) > limit:
        raise _InvalidJournal("token budget exceeded")
    if any(type(value) is not int or not 0 <= value < 2**31 for value in values):
        raise _InvalidJournal("invalid token value")
    if isinstance(values, PackedTokenIds):
        return values
    packed = array("I", values)
    if packed.itemsize != 4:
        raise _InvalidJournal("unsupported CPU integer size")
    if struct.pack("=I", 1) != struct.pack("<I", 1):
        packed.byteswap()
    return PackedTokenIds(packed.tobytes())


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _identity_digest(value):
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise _InvalidJournal("invalid identity digest")
    return value


@dataclass(frozen=True, repr=False)
class ContinuationIdentity:
    cache_salt: str = field(repr=False)
    model: str = field(repr=False)
    tokenizer: str = field(repr=False)
    template: str = field(repr=False)
    configuration: str = field(repr=False)

    def __post_init__(self):
        for value in (
            self.cache_salt,
            self.model,
            self.tokenizer,
            self.template,
            self.configuration,
        ):
            if not isinstance(value, str) or not value or len(value.encode()) > 4096:
                raise ValueError("invalid continuation identity")

    @property
    def key(self):
        return _digest([SCHEMA, self.cache_salt])

    @property
    def digest(self):
        return _digest(
            [
                SCHEMA,
                self.cache_salt,
                self.model,
                self.tokenizer,
                self.template,
                self.configuration,
            ]
        )

    def __repr__(self):
        return "ContinuationIdentity(private)"


@dataclass(frozen=True)
class ContinuationLease:
    identity: ContinuationIdentity = field(repr=False)
    token: str = field(repr=False)
    base_version: str | None = field(repr=False)


@dataclass(frozen=True)
class TokenJournal:
    tokens: PackedTokenIds = field(repr=False)
    identity_digest: str = field(repr=False)
    history_identity: str = field(repr=False)
    output_identity: str = field(repr=False)
    version: str = field(repr=False)
    processed_tokens: int | None = None
    message_count: int | None = None
    contract: str = CONTRACT

    @property
    def raw_ids(self):
        return self.tokens


@dataclass(frozen=True)
class ContinuationProposal:
    tokens: PackedTokenIds | None = field(default=None, repr=False)
    reason: str = "unavailable"
    version: str | None = field(default=None, repr=False)
    processed_tokens: int | None = None
    contract: str = CONTRACT

    @property
    def applied(self):
        return self.tokens is not None


@dataclass
class _Active:
    leases: set[str] = field(default_factory=set)
    conflicted: bool = False


class TokenContinuationLedger:
    """One completed journal per salt, with bounded nonauthoritative metadata.

    configuration must bind the cache generation, runtime/data ABI and any
    other adapter configuration affecting token construction. Unknown processed
    counts stay None. No caller may infer KV validity from a journal alone.
    """

    def __init__(
        self,
        root=None,
        *,
        max_chats=8,
        max_tokens=DEFAULT_MAX_TOKENS,
        max_bytes=MAX_BYTES,
        max_pending=8,
    ):
        if not (
            type(max_chats) is int
            and 1 <= max_chats <= 8
            and type(max_tokens) is int
            and 1 <= max_tokens <= MAX_TOKENS
            and type(max_bytes) is int
            and 4 <= max_bytes <= MAX_BYTES
            and type(max_pending) is int
            and 1 <= max_pending <= 8
        ):
            raise ValueError("invalid continuation bounds")
        configured = (
            root if root is not None else os.environ.get("QWEN_TOKEN_CONTINUATION_ROOT")
        )
        self.root = Path(configured) if configured else None
        self.max_chats, self.max_tokens = max_chats, max_tokens
        self.max_bytes, self.max_pending = max_bytes, max_pending
        self._records = OrderedDict()
        self._active = {}
        self._lock = threading.RLock()
        self._health = {"read_errors": 0, "write_errors": 0, "rejections": 0}
        self._last_reason = "ready"

    def _reject(self, reason):
        self._health["rejections"] += 1
        self._last_reason = reason
        return ContinuationProposal(reason=reason)

    @property
    def health(self):
        with self._lock:
            return {
                **self._health,
                "durable": self.root is not None,
                "records": len(self._records),
                "retained_bytes": sum(
                    len(row.tokens) * 4 for row in self._records.values()
                ),
                "pending": sum(len(row.leases) for row in self._active.values()),
                "last_reason": self._last_reason,
            }

    @contextmanager
    def _directory(self):
        if self.root is None or not self.root.is_absolute() or ".." in self.root.parts:
            raise _InvalidJournal("invalid private root")
        fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            for component in self.root.parts[1:]:
                try:
                    child = os.open(
                        component,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=fd,
                    )
                except FileNotFoundError:
                    os.mkdir(component, 0o700, dir_fd=fd)
                    child = os.open(
                        component,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=fd,
                    )
                os.close(fd)
                fd = child
            info = os.fstat(fd)
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise _InvalidJournal("unsafe private root")
            yield fd
        finally:
            os.close(fd)

    @staticmethod
    def _regular(fd):
        info = os.fstat(fd)
        if not (
            stat.S_ISREG(info.st_mode)
            and info.st_uid == os.getuid()
            and info.st_nlink == 1
            and not stat.S_IMODE(info.st_mode) & 0o077
        ):
            raise _InvalidJournal("unsafe private file")
        return info

    @contextmanager
    def _disk_lock(self, directory):
        fd = os.open(
            ".lock",
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
            0o600,
            dir_fd=directory,
        )
        try:
            self._regular(fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        finally:
            os.close(fd)

    def _read(self, directory, key):
        try:
            fd = os.open(
                key + ".ctok",
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                dir_fd=directory,
            )
        except FileNotFoundError:
            return None
        try:
            info = self._regular(fd)
            limit = min(self.max_tokens * 4, self.max_bytes) + MAX_HEADER + 44
            if info.st_size > limit or info.st_size < 44:
                raise _InvalidJournal("invalid journal size")
            parts, remaining = [], info.st_size + 1
            while remaining:
                part = os.read(fd, min(remaining, 64 * 1024))
                if not part:
                    break
                parts.append(part)
                remaining -= len(part)
            encoded = b"".join(parts)
            if len(encoded) != info.st_size:
                raise _InvalidJournal("journal changed during read")
        finally:
            os.close(fd)
        if (
            encoded[:8] != MAGIC
            or hashlib.sha256(encoded[:-32]).digest() != encoded[-32:]
        ):
            raise _InvalidJournal("invalid journal integrity")
        size = struct.unpack_from("<I", encoded, 8)[0]
        if not 0 < size <= MAX_HEADER or 12 + size > len(encoded) - 32:
            raise _InvalidJournal("invalid journal header")
        header = json.loads(encoded[12 : 12 + size])
        expected = {
            "schema",
            "contract",
            "key",
            "identity",
            "history",
            "output",
            "version",
            "token_count",
            "processed_tokens",
            "message_count",
        }
        if not isinstance(header, dict) or set(header) != expected:
            raise _InvalidJournal("unsupported journal metadata")
        if (
            header["schema"] != SCHEMA
            or header["contract"] != CONTRACT
            or header["key"] != key
        ):
            raise _InvalidJournal("journal binding differs")
        for name in ("identity", "history", "output"):
            _identity_digest(header[name])
        if not isinstance(header["version"], str) or not HEX32.fullmatch(
            header["version"]
        ):
            raise _InvalidJournal("invalid journal version")
        tokens = _pack(PackedTokenIds(encoded[12 + size : -32]), self.max_tokens)
        if (
            type(header["token_count"]) is not int
            or header["token_count"] != len(tokens)
            or not tokens
            or len(tokens) * 4 > self.max_bytes
        ):
            raise _InvalidJournal("invalid journal token count")
        processed = header["processed_tokens"]
        if processed is not None and (
            type(processed) is not int or not 0 <= processed <= len(tokens)
        ):
            raise _InvalidJournal("invalid processed boundary")
        messages = header["message_count"]
        if messages is not None and (
            type(messages) is not int or not 0 <= messages <= 1024
        ):
            raise _InvalidJournal("invalid history boundary")
        return TokenJournal(
            tokens,
            header["identity"],
            header["history"],
            header["output"],
            header["version"],
            processed,
            messages,
        )

    def _remember(self, key, record):
        self._records[key] = record
        self._records.move_to_end(key)
        while self._records and (
            len(self._records) > self.max_chats
            or sum(len(row.tokens) * 4 for row in self._records.values())
            > self.max_bytes
        ):
            self._records.popitem(last=False)

    def get_record(self, identity):
        with self._lock:
            if not isinstance(identity, ContinuationIdentity):
                self._last_reason = "invalid_identity"
                return None
            if self.root is not None:
                try:
                    with self._directory() as directory, self._disk_lock(directory):
                        record = self._read(directory, identity.key)
                except (OSError, ValueError, TypeError):
                    self._health["read_errors"] += 1
                    self._last_reason = "storage_unavailable"
                    self._records.pop(identity.key, None)
                    return None
                if record is None:
                    self._records.pop(identity.key, None)
                else:
                    self._remember(identity.key, record)
            record = self._records.get(identity.key)
            if record is None:
                self._last_reason = "no_record"
            elif record.identity_digest != identity.digest:
                self._last_reason = "identity_changed"
                return None
            else:
                self._records.move_to_end(identity.key)
                self._last_reason = "ready"
            return record

    def begin(self, identity):
        with self._lock:
            if not isinstance(identity, ContinuationIdentity):
                self._reject("invalid_identity")
                return None
            record = self.get_record(identity)
            if self.root is not None and self._last_reason == "storage_unavailable":
                self._reject("storage_unavailable")
                return None
            # A changed construction must not reuse the previous token history,
            # but a successful fresh request may replace it. Keep only its CAS
            # version here; get_record/propose still reject its identity.
            if record is None and self._last_reason == "identity_changed":
                record = self._records.get(identity.key)
            active = self._active.get(identity.key)
            if active is not None:
                active.conflicted = True
            if (
                sum(len(row.leases) for row in self._active.values())
                >= self.max_pending
            ):
                self._reject("pending_budget")
                return None
            if active is None:
                active = self._active[identity.key] = _Active()
            token = secrets.token_hex(16)
            active.leases.add(token)
            return ContinuationLease(
                identity, token, record.version if record else None
            )

    def _lease_valid(self, lease):
        if not isinstance(lease, ContinuationLease):
            return False
        active = self._active.get(lease.identity.key)
        return (
            active is not None
            and lease.token in active.leases
            and not active.conflicted
        )

    def abort(self, lease):
        with self._lock:
            if not isinstance(lease, ContinuationLease):
                return
            active = self._active.get(lease.identity.key)
            if active is not None:
                active.leases.discard(lease.token)
                if not active.leases:
                    self._active.pop(lease.identity.key, None)

    def propose(
        self,
        identity,
        current_ids,
        *,
        canonical_previous_ids,
        history_identity,
        output_identity,
        boundary_token_ids,
        history_unchanged,
        output_unchanged,
        configuration_unchanged,
        lease=None,
    ):
        with self._lock:
            if any(
                value is not True
                for value in (
                    history_unchanged,
                    output_unchanged,
                    configuration_unchanged,
                )
            ):
                return self._reject("construction_changed")
            if lease is not None and (
                not self._lease_valid(lease) or lease.identity != identity
            ):
                return self._reject("overlapping_or_stale_request")
            if (
                lease is None
                and isinstance(identity, ContinuationIdentity)
                and identity.key in self._active
            ):
                return self._reject("overlapping_or_stale_request")
            record = self.get_record(identity)
            if record is None:
                return self._reject(self._last_reason)
            if lease is not None and lease.base_version != record.version:
                return self._reject("journal_changed")
            if record.message_count is None:
                return self._reject("history_boundary_missing")
            try:
                if (
                    _identity_digest(history_identity) != record.history_identity
                    or _identity_digest(output_identity) != record.output_identity
                ):
                    return self._reject("history_or_output_changed")
                current = _pack(current_ids, self.max_tokens)
                canonical = _pack(canonical_previous_ids, self.max_tokens)
                if (
                    not isinstance(boundary_token_ids, (list, tuple, set, frozenset))
                    or not 1 <= len(boundary_token_ids) <= 64
                ):
                    return self._reject("unsupported_boundary")
                if any(
                    type(value) is not int or not 0 <= value < 2**31
                    for value in boundary_token_ids
                ):
                    return self._reject("unsupported_boundary")
                if not canonical or not current or len(current) <= len(canonical):
                    return self._reject("prompt_not_extended")
                if (
                    record.tokens[-1] != canonical[-1]
                    or record.tokens[-1] not in boundary_token_ids
                ):
                    return self._reject("unsupported_boundary")
                if current._data[: len(canonical._data)] != canonical._data:
                    return self._reject("canonical_prefix_changed")
                payload = record.tokens._data + current._data[len(canonical._data) :]
                if len(payload) // 4 > self.max_tokens or len(payload) > self.max_bytes:
                    return self._reject("token_budget")
            except (TypeError, ValueError, OverflowError):
                return self._reject("unsupported_tokens")
            self._last_reason = "generated_prefix_preserved"
            return ContinuationProposal(
                PackedTokenIds(payload),
                "generated_prefix_preserved",
                record.version,
                record.processed_tokens,
            )

    def append_verified_suffix(
        self,
        identity,
        suffix_ids,
        *,
        history_identity,
        output_identity,
        expected_version,
        boundary_token_ids,
        lease,
    ):
        """Append a suffix authenticated by the request's trusted renderer.

        The caller must check unchanged message/configuration identities and
        that decoding the stored raw history exactly prefixes the actual full
        rendering, then encode only the remaining suffix. This method does not
        establish those tokenizer/text facts. It rechecks the journal, lease and
        boundary before returning an immutable proposal; it never mutates KV.
        """
        with self._lock:
            if not self._lease_valid(lease) or lease.identity != identity:
                return self._reject("overlapping_or_stale_request")
            record = self.get_record(identity)
            if record is None:
                return self._reject(self._last_reason)
            if (
                not isinstance(expected_version, str)
                or not HEX32.fullmatch(expected_version)
                or expected_version != record.version
                or lease.base_version != record.version
            ):
                return self._reject("journal_changed")
            if record.message_count is None:
                return self._reject("history_boundary_missing")
            try:
                if (
                    _identity_digest(history_identity) != record.history_identity
                    or _identity_digest(output_identity) != record.output_identity
                ):
                    return self._reject("history_or_output_changed")
                if (
                    not isinstance(boundary_token_ids, (list, tuple, set, frozenset))
                    or not 1 <= len(boundary_token_ids) <= 64
                    or any(
                        type(value) is not int or not 0 <= value < 2**31
                        for value in boundary_token_ids
                    )
                    or record.tokens[-1] not in boundary_token_ids
                ):
                    return self._reject("unsupported_boundary")
                suffix = _pack(suffix_ids, self.max_tokens)
                if not suffix:
                    return self._reject("prompt_not_extended")
                payload = record.tokens._data + suffix._data
                if len(payload) // 4 > self.max_tokens or len(payload) > self.max_bytes:
                    return self._reject("token_budget")
            except (TypeError, ValueError, OverflowError):
                return self._reject("unsupported_tokens")
            self._last_reason = "generated_prefix_preserved"
            return ContinuationProposal(
                PackedTokenIds(payload),
                "generated_prefix_preserved",
                record.version,
                record.processed_tokens,
            )

    def _write(self, directory, key, record):
        header = json.dumps(
            {
                "schema": SCHEMA,
                "contract": CONTRACT,
                "key": key,
                "identity": record.identity_digest,
                "history": record.history_identity,
                "output": record.output_identity,
                "version": record.version,
                "token_count": len(record.tokens),
                "processed_tokens": record.processed_tokens,
                "message_count": record.message_count,
            },
            separators=(",", ":"),
        ).encode()
        payload = MAGIC + struct.pack("<I", len(header)) + header + record.tokens._data
        payload += hashlib.sha256(payload).digest()
        temporary = ".tmp-" + secrets.token_hex(16)
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=directory,
        )
        try:
            try:
                self._regular(fd)
                pending = memoryview(payload)
                while pending:
                    written = os.write(fd, pending)
                    if written <= 0:
                        raise OSError("incomplete journal write")
                    pending = pending[written:]
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(
                temporary, key + ".ctok", src_dir_fd=directory, dst_dir_fd=directory
            )
            os.fsync(directory)
        finally:
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass
        self._prune(directory, key)

    def _prune(self, directory, keep):
        entries = []
        with os.scandir(directory) as scan:
            for ordinal, entry in enumerate(scan):
                if ordinal >= MAX_DIRECTORY_ENTRIES:
                    raise _InvalidJournal("journal directory budget exceeded")
                if not FILE_NAME.fullmatch(entry.name):
                    continue
                fd = os.open(
                    entry.name,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                    dir_fd=directory,
                )
                try:
                    info = self._regular(fd)
                    if info.st_size > self.max_tokens * 4 + MAX_HEADER + 44:
                        raise _InvalidJournal("journal directory size exceeded")
                    entries.append((entry.name, info.st_mtime_ns, info.st_size))
                finally:
                    os.close(fd)
        entries.sort(key=lambda row: (row[0] != keep + ".ctok", -row[1], row[0]))
        used = 0
        removed = False
        disk_budget = self.max_bytes + self.max_chats * (MAX_HEADER + 44)
        for ordinal, (name, _, size) in enumerate(entries):
            if ordinal < self.max_chats and used + size <= disk_budget:
                used += size
                continue
            os.unlink(name, dir_fd=directory)
            removed = True
            self._records.pop(name[:-5], None)
        if removed:
            os.fsync(directory)

    def record_completed(
        self,
        lease,
        raw_prefix,
        output_ids,
        *,
        history_identity,
        output_identity,
        processed_tokens=None,
        message_count=None,
        success=True,
    ):
        with self._lock:
            try:
                if success is not True or not self._lease_valid(lease):
                    self._reject("cancelled_or_overlapping_request")
                    return False
                prefix = _pack(raw_prefix, self.max_tokens)
                output = _pack(output_ids, self.max_tokens)
                payload = prefix._data + output._data
                count = len(payload) // 4
                if (
                    not prefix
                    or not output
                    or count > self.max_tokens
                    or len(payload) > self.max_bytes
                ):
                    self._reject("token_budget")
                    return False
                if processed_tokens is not None and (
                    type(processed_tokens) is not int
                    or not 0 <= processed_tokens <= count
                ):
                    self._reject("invalid_processed_boundary")
                    return False
                if message_count is not None and (
                    type(message_count) is not int or not 0 <= message_count <= 1024
                ):
                    self._reject("invalid_history_boundary")
                    return False
                record = TokenJournal(
                    PackedTokenIds(payload),
                    lease.identity.digest,
                    _identity_digest(history_identity),
                    _identity_digest(output_identity),
                    secrets.token_hex(16),
                    processed_tokens,
                    message_count,
                )
                key = lease.identity.key
                if self.root is not None:
                    with self._directory() as directory, self._disk_lock(directory):
                        previous = self._read(directory, key)
                        if (
                            previous.version if previous else None
                        ) != lease.base_version:
                            self._reject("journal_changed")
                            return False
                        self._write(directory, key, record)
                else:
                    previous = self._records.get(key)
                    if (previous.version if previous else None) != lease.base_version:
                        self._reject("journal_changed")
                        return False
                self._remember(key, record)
                self._last_reason = "recorded"
                return True
            except (OSError, ValueError, TypeError, OverflowError):
                self._health["write_errors"] += 1
                self._last_reason = "storage_or_input_unavailable"
                return False
            finally:
                self.abort(lease)

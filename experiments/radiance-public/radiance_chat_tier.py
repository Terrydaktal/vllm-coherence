"""vLLM filesystem tier with isolated chat generations and lossless Zstd blocks."""

from __future__ import annotations

import fcntl
import functools
import json
import logging
import os
import threading
import time
from pathlib import Path

from qwen_radiance_cache import (
    CONTROL_DIRECTORY,
    CONTROL_FILE,
    CONTROL_SCHEMA,
    FORMAT,
    ID,
    ChatStore,
    RetiredGenerationError,
    atomic_write,
    identity,
    private_control_directory,
    real_directory,
)
from vllm.v1.kv_offload.base import OffloadPolicy, get_offload_block_hash, get_offload_group_idx
from vllm.v1.kv_offload.tiering.base import JobResult
from vllm.v1.kv_offload.tiering.fs.manager import (
    FileSystemTierManager,
    FsAsyncLookupManager,
)

logger = logging.getLogger(__name__)
TAIL_FLUSH_TOKENS = 8192
TAIL_RAM_MAX_BYTES = 6 * 1024**3
TAIL_RAM_MAX_CHATS = 5
TAIL_BLOCK_LIMIT = 15
TAIL_STATUS = Path("/dev/shm/qwen-radiance-snapshot-tail.json")


def object_key(key):
    return f"g{get_offload_group_idx(key)}-{get_offload_block_hash(key).hex()}.qkv"


def tail_overlap(config):
    # v0.28 marks only the draft group as EAGLE. Its one-block rollback also
    # needs the preceding aligned target Mamba state for a common cache hit.
    return int(config.is_eagle_group or getattr(config, "requires_cow_source", False))


def head_keys(status, num_tokens):
    """Keep the same full prefix / settled windows that the scheduler can store."""
    keys = []
    for config, state in zip(status.config.kv_group_configs, status.group_states, strict=True):
        count = status.storable_chunks(config, state, num_tokens)
        window = config.sliding_window_size_in_chunks
        if window is None:
            # During decoding the EAGLE exclusion can reduce storable_chunks
            # below a prefix already stored during prefill. Those full-attention
            # chunks remain valid and are needed for hybrid lookup convergence.
            count = min(max(count, state.next_stored_chunk_idx), len(state.offload_keys))
        first = 0 if window is None else max(0, count - window - tail_overlap(config))
        keys.extend(state.offload_keys[first:count])
    return keys


def tail_keys(status, num_tokens):
    """Return only the changing recurrent/sliding portion of a complete head."""
    keys = []
    for config, state in zip(status.config.kv_group_configs, status.group_states, strict=True):
        window = config.sliding_window_size_in_chunks
        if window is None:
            continue
        count = status.storable_chunks(config, state, num_tokens)
        first = max(0, count - window - tail_overlap(config))
        keys.extend(state.offload_keys[first:count])
    return keys


def synchronized(method):
    @functools.wraps(method)
    def guarded(self, *args, **kwargs):
        with self._chat_mutex:
            return method(self, *args, **kwargs)

    return guarded


class ChatLookup(FsAsyncLookupManager):
    def batch_lookup(self, keys, req_context):
        state = self._tier._chat_requests.get(req_context.req_id)
        if state is None:
            return [False] * len(keys)
        names = [object_key(key) for key in keys]
        disk = state["store"].exists_many(names)
        ram = self._tier._ram_blocks(state["store"], names)
        return [present or name in ram for name, present in zip(names, disk, strict=True)]


class ChatFileSystemTierManager(FileSystemTierManager):
    def __init__(
        self,
        *args,
        root_dir,
        tail_flush_tokens=TAIL_FLUSH_TOKENS,
        tail_ram_max_bytes=TAIL_RAM_MAX_BYTES,
        tail_ram_max_chats=TAIL_RAM_MAX_CHATS,
        tail_block_limit=TAIL_BLOCK_LIMIT,
        control_directory=str(CONTROL_DIRECTORY),
        tail_status_path=str(TAIL_STATUS),
        **kwargs,
    ):
        super().__init__(*args, root_dir=root_dir, **kwargs)
        if self.file_mapper.rank != 0:
            raise ValueError("chat snapshots currently require the qualified single-rank lane")
        self._root = Path(root_dir)
        managed = self._root / FORMAT
        managed.mkdir(exist_ok=True, mode=0o700)
        real_directory(managed)
        self._engine_lock = os.open(
            managed / ".engine.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
        )
        # A second backend must not collect files being used by the first one.
        fcntl.flock(self._engine_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self._lookup_manager.shutdown()
        self._lookup_manager = ChatLookup(self, self.tier_type)
        self._chat_requests = {}
        self._chat_jobs = {}
        self._chat_stores = {}
        self._tail_heads = {}
        self._ignored_jobs = []
        self._sequence = 0
        self._chat_mutex = threading.RLock()
        self._tail_flush_tokens = int(tail_flush_tokens)
        self._tail_ram_max_bytes = int(tail_ram_max_bytes)
        self._tail_ram_max_chats = int(tail_ram_max_chats)
        self._tail_block_limit = int(tail_block_limit)
        if (
            self._tail_flush_tokens < 1
            or self._tail_ram_max_bytes < self._block_size
            or self._tail_ram_max_chats < 1
            or self._tail_block_limit < 1
        ):
            raise ValueError("invalid Radiance tail journal limits")
        self._control_directory = private_control_directory(Path(control_directory), create=True)
        self._tail_status_path = Path(tail_status_path)
        self._tail_status_signature = None
        self._tail_status_written = 0.0
        self._last_tail_flush = None
        for path in self._control_directory.iterdir():
            if CONTROL_FILE.fullmatch(path.name):
                if path.is_symlink() or not path.is_file():
                    raise ValueError("unsafe snapshot control file")
                path.unlink()
        # The engine lock proves the previous engine has no outstanding jobs.
        # Recover both interrupted GC and orphan writes from older runtimes.
        for directory in managed.iterdir():
            if ID.fullmatch(directory.name):
                real_directory(directory)
                metadata = directory / "chat.json"
                if metadata.is_symlink():
                    raise ValueError("chat metadata is a symlink")
                if metadata.exists():
                    chat = json.loads(metadata.read_text())
                    if chat.get("id") != directory.name:
                        raise ValueError("chat metadata identity differs from its directory")
                    store = ChatStore(self._root, chat)
                    self._chat_stores[chat["id"]] = store
                    try:
                        store.collect()
                    except OSError:
                        self._sequence += 1
                        self._chat_requests[f"gc-recovery-{directory.name}"] = {
                            "store": store,
                            "jobs": set(),
                            "finished": True,
                            "failed": False,
                            "head": None,
                            "sequence": self._sequence,
                            "collect_only": True,
                        }
                        logger.exception("Chat snapshot startup cleanup failed: %s", directory.name)
        self._retry_stop = threading.Event()
        self._publish_wake = threading.Event()
        self._retry_thread = threading.Thread(
            target=self._retry_gc, name="chat-snapshot-gc", daemon=True
        )
        self._retry_thread.start()

    def _retry_gc(self):
        # Engine scheduler polling can stop entirely once a request completes.
        # This worker also services explicit tmpfs flush requests without doing
        # compression, verification, or collection on the scheduler thread.
        while not self._retry_stop.is_set():
            self._publish_wake.wait(0.25)
            self._publish_wake.clear()
            if self._retry_stop.is_set():
                return
            try:
                self._publish_ready()
                self._process_control_requests()
            except Exception:
                logger.exception("Chat snapshot publication worker failed; retrying")

    @synchronized
    def on_new_request(self, req_context):
        chat = (req_context.kv_transfer_params or {}).get("qwen_chat")
        store = None
        if chat is not None:
            store = self._chat_stores.get(chat["id"])
            if store is None or store.chat["generation"] != chat["generation"]:
                candidate = ChatStore(self._root, chat)
                try:
                    candidate.activate()
                except RetiredGenerationError:
                    # vLLM treats connector on_new_request exceptions as fatal to
                    # the whole engine. A stale client may run with its isolated
                    # cache salt, but it cannot read, write, or reactivate this
                    # tombstoned durable generation.
                    logger.warning(
                        "Retired chat generation bypasses durable snapshots: %s", chat["id"]
                    )
                    store = None
                else:
                    store = candidate
                    for key in [key for key in self._tail_heads if key[0] == chat["id"]]:
                        # The committed compaction hook flushes the predecessor before
                        # activation. A stale in-process copy must never cross salts.
                        del self._tail_heads[key]
            elif not store.current():
                # A compaction can retire this generation through the CLI while
                # an older request is still queued. Bypass durable storage without
                # turning a safe cache miss into an engine-wide failure.
                logger.warning(
                    "Queued retired chat generation bypasses durable snapshots: %s", chat["id"]
                )
                store = None
        result = super().on_new_request(req_context)
        if store is not None:
            # A RAM hit does not prove the durable block still exists. Request
            # all prefix blocks so the primary tier can repair missing disk
            # objects; the writer skips objects already present on disk.
            result.policy = OffloadPolicy.REQUEST_LEVEL
            self._chat_stores[chat["id"]] = store
            self._sequence += 1
            self._chat_requests[req_context.req_id] = {
                "store": store,
                "jobs": set(),
                "finished": False,
                "failed": False,
                "head": None,
                "tail_keys": set(),
                "tail_blocks": {},
                "force_flush": False,
                "sequence": self._sequence,
            }
        return result

    @synchronized
    def set_snapshot_head(self, status, num_tokens):
        state = self._chat_requests.get(status.req_context.req_id)
        if state is not None:
            changing = {object_key(key) for key in tail_keys(status, num_tokens)}
            if len(changing) > self._tail_block_limit:
                raise ValueError(
                    f"settled snapshot tail has {len(changing)} blocks; limit is "
                    f"{self._tail_block_limit}"
                )
            state["head"] = ([object_key(key) for key in head_keys(status, num_tokens)], num_tokens)
            state["tail_keys"] = changing
            params = status.req_context.kv_transfer_params or {}
            state["force_flush"] = params.get("qwen_snapshot_force_flush") is True
            # A cancelled request must never supersede a complete earlier head.
            if (
                status.req.status.name != "FINISHED_STOPPED"
                and status.req.status.name != "FINISHED_LENGTH_CAPPED"
            ):
                state["failed"] = True

    def _store(self, state, keys, offsets):
        view = self._primary_kv_view.cast("B")
        disk_blocks = []
        tail_blocks = {}
        for key, offset in zip(keys, offsets, strict=True):
            if offset < 0 or offset + self._block_size > len(view):
                raise ValueError("snapshot store offset out of bounds")
            name = object_key(key)
            block = view[offset : offset + self._block_size]
            if name in state["tail_keys"]:
                # The primary slot is pinned only for this async job. Copy the
                # changing tail before acknowledging it so later CPU eviction
                # cannot force another GPU prefill or lose the flush source.
                tail_blocks[name] = bytes(block)
            else:
                disk_blocks.append((name, block))
        if disk_blocks and not state["store"].write_many(disk_blocks):
            raise ValueError("snapshot generation was retired during store")
        if tail_blocks:
            with self._chat_mutex:
                state["tail_blocks"].update(tail_blocks)

    def _load(self, state, keys, offsets):
        view = self._primary_kv_view.cast("B")
        for offset in offsets:
            if offset < 0 or offset + self._block_size > len(view):
                raise ValueError("snapshot load offset out of bounds")
        names = [object_key(key) for key in keys]
        ram = self._ram_blocks(state["store"], names)
        disk_names = [name for name in names if name not in ram]
        disk = iter(state["store"].read_many(disk_names, self._block_size))
        for name, offset in zip(names, offsets, strict=True):
            data = ram.get(name)
            if data is None:
                data = next(disk)
            # Verify before touching the destination. Corruption cannot expose
            # a partly decoded recurrent state as a successful restored block.
            view[offset : offset + self._block_size] = data

    @synchronized
    def _submit(self, job, is_store):
        state = self._chat_requests[job.req_context.req_id]
        state["jobs"].add(job.job_id)
        self._chat_jobs[job.job_id] = (job.req_context.req_id, is_store)
        keys = list(job.keys)
        if is_store and self.events is not None:
            self._store_job_keys[job.job_id] = keys
        if not is_store:
            # The v0.28 filesystem parent consumes this map on completion.
            # Without it, a failed read remains a cached HIT and is promoted
            # repeatedly. Failed blocks must become misses before another
            # lookup; none of their partially written CPU slots is published.
            self._load_job_keys[job.job_id] = keys
        operation = self._store if is_store else self._load
        task = functools.partial(
            operation, state, keys, [int(bid) * self._block_size for bid in job.block_ids]
        )
        enqueue = self._pool.enqueue_store if is_store else self._pool.enqueue_load
        enqueue(job.job_id, 1, [task])

    def submit_store(self, job_metadata):
        if job_metadata.req_context.req_id not in self._chat_requests:
            # The primary RAM tier cascades new blocks to every secondary tier.
            # Acknowledge unlabelled stores without creating shared .bin files
            # that cannot later be attributed or safely collected.
            with self._chat_mutex:
                self._ignored_jobs.append(JobResult(job_id=job_metadata.job_id, success=True))
            return None
        self._submit(job_metadata, True)

    def submit_load(self, job_metadata):
        if job_metadata.req_context.req_id not in self._chat_requests:
            # ChatLookup always misses these. Fail an unexpected load so stale
            # unlabelled data can never reach the model.
            with self._chat_mutex:
                self._ignored_jobs.append(JobResult(job_id=job_metadata.job_id, success=False))
            return None
        self._submit(job_metadata, False)

    @synchronized
    def on_request_finished(self, req_context):
        super().on_request_finished(req_context)
        if req_context.req_id in self._chat_requests:
            self._chat_requests[req_context.req_id]["finished"] = True
            self._publish_wake.set()

    @synchronized
    def get_finished_jobs(self):
        results = super().get_finished_jobs()
        results.extend(self._ignored_jobs)
        self._ignored_jobs = []
        for result in results:
            owner = self._chat_jobs.pop(result.job_id, None)
            if owner is not None:
                req_id, is_store = owner
                state = self._chat_requests[req_id]
                state["jobs"].remove(result.job_id)
                # The parent invalidates failed loads and the engine recomputes
                # their tokens. A subsequent successfully completed request can
                # publish its verified replacement. Failed writes, cancellation
                # and unfinished requests still prevent publication.
                state["failed"] |= is_store and not result.success
        self._publish_wake.set()
        return results

    def _ram_blocks(self, store, names):
        wanted = set(names)
        result = {}
        key = (store.chat["id"], store.chat["generation"])
        with self._chat_mutex:
            record = self._tail_heads.get(key)
            if record is not None:
                record["last_access"] = time.monotonic()
                result.update(
                    (name, data) for name, data in record["tail_blocks"].items() if name in wanted
                )
            states = sorted(
                (
                    state
                    for state in self._chat_requests.values()
                    if (
                        state["store"].chat["id"],
                        state["store"].chat["generation"],
                    )
                    == key
                ),
                key=lambda state: state["sequence"],
            )
            for state in states:
                result.update(
                    (name, data)
                    for name, data in state.get("tail_blocks", {}).items()
                    if name in wanted
                )
        return result

    def _build_tail_record(self, state, *, force=False):
        keys, tokens = state["head"]
        tail_names = sorted(state["tail_keys"])
        ram = self._ram_blocks(state["store"], tail_names)
        on_disk = dict(zip(tail_names, state["store"].exists_many(tail_names), strict=True))
        missing = [name for name in tail_names if name not in ram and not on_disk[name]]
        if missing:
            raise ValueError(f"settled snapshot tail is missing {len(missing)} RAM blocks")
        durable_tokens = state["store"].metadata().get("tokens", 0)
        return {
            "store": state["store"],
            "keys": list(keys),
            "tokens": tokens,
            "durable_tokens": durable_tokens if isinstance(durable_tokens, int) else 0,
            "tail_keys": set(tail_names),
            "tail_blocks": {name: ram[name] for name in tail_names if name in ram},
            "sequence": state["sequence"],
            "force": force or state["force_flush"],
            "last_access": time.monotonic(),
            "retry_after": 0.0,
        }

    def _record_due(self, record):
        return time.monotonic() >= record.get("retry_after", 0) and (
            record["force"]
            or record["tokens"] - record["durable_tokens"] >= self._tail_flush_tokens
        )

    def _flush_record(self, record, reason):
        store = record["store"]
        key = (store.chat["id"], store.chat["generation"])
        with self._chat_mutex:
            if self._tail_heads.get(key) is not record or self._states_for_chat(key[0]):
                return None
            tail_blocks = list(record["tail_blocks"].items())
        if tail_blocks and not store.write_many(tail_blocks):
            raise ValueError("snapshot generation retired before tail flush")
        prepared = store.prepare_publication(record["keys"], self._block_size)
        with self._chat_mutex:
            # Keep the old manifest valid through tail writes and verification.
            # The short atomic publish/GC section excludes a newer request so it
            # can never delete blocks that request is currently producing.
            if self._tail_heads.get(key) is not record or self._states_for_chat(key[0]):
                return None
            if not store.publish(
                record["keys"], record["tokens"], self._block_size, prepared=prepared
            ):
                logger.warning("Chat snapshot rejected; previous head retained: %s", key[0])
                del self._tail_heads[key]
                return {"status": "rejected", "tokens": record["durable_tokens"]}
            del self._tail_heads[key]
            self._last_tail_flush = {
                "chat_id": record["store"].chat["id"],
                "generation": record["store"].chat["generation"],
                "tokens": record["tokens"],
                "reason": reason,
            }
        self._publish_tail_status(force=True)
        return {
            "status": "flushed",
            "tokens": record["tokens"],
            "tail_blocks": len(record["tail_keys"]),
            "tail_raw_bytes": sum(len(data) for data in record["tail_blocks"].values()),
            "reason": reason,
        }

    def _publish_ready(self, *, force_identities=frozenset(), force_all=False):
        with self._chat_mutex:
            chats = {state["store"].chat["id"] for state in self._chat_requests.values()}
        for chat_id in chats:
            with self._chat_mutex:
                states = self._states_for_chat(chat_id)
                if not states or any(not s["finished"] or s["jobs"] for s in states.values()):
                    continue
                latest = max(states.values(), key=lambda state: state["sequence"])
                if time.monotonic() < latest.get("retry_after", 0):
                    continue
                successful = [
                    state
                    for state in states.values()
                    if not state.get("collect_only")
                    and not state["failed"]
                    and state["head"] is not None
                ]
                candidate = (
                    max(successful, key=lambda state: state["sequence"]) if successful else None
                )
                candidate_key = (
                    (candidate["store"].chat["id"], candidate["store"].chat["generation"])
                    if candidate is not None
                    else None
                )
            try:
                if latest.get("collect_only"):
                    record = None
                elif candidate is not None:
                    record = self._build_tail_record(
                        candidate,
                        force=force_all or candidate_key in force_identities,
                    )
                else:
                    record = None
                with self._chat_mutex:
                    current = self._states_for_chat(chat_id)
                    if current.keys() != states.keys() or any(
                        not state["finished"] or state["jobs"] for state in current.values()
                    ):
                        continue
                    if latest.get("collect_only"):
                        latest["store"].collect()
                    elif record is not None:
                        self._tail_heads[candidate_key] = record
                    elif not any(key[0] == chat_id for key in self._tail_heads):
                        latest["store"].collect(failure="request_or_write_failed")
                    for req_id in states:
                        del self._chat_requests[req_id]
            except OSError:
                latest["collect_only"] = True
                latest["retry_after"] = time.monotonic() + 5
                logger.exception("Chat snapshot cleanup pending; retrying: %s", chat_id)
                continue
            except ValueError:
                logger.exception("Chat snapshot tail rejected; previous head retained: %s", chat_id)
                with self._chat_mutex:
                    for req_id in states:
                        self._chat_requests.pop(req_id, None)

        with self._chat_mutex:
            if force_all:
                # Completed requests have already moved out of _chat_requests.
                # Shutdown must flush their retained RAM heads too, including
                # a tail whose ordinary background retry has not fallen due.
                for record in self._tail_heads.values():
                    record["force"] = True
                    record["retry_after"] = 0.0
            due = [record for record in self._tail_heads.values() if self._record_due(record)]
        for record in sorted(due, key=lambda value: value["sequence"]):
            try:
                self._flush_record(record, "forced" if record["force"] else "token_interval")
            except (OSError, ValueError):
                record["retry_after"] = time.monotonic() + 5
                logger.exception(
                    "Chat snapshot tail flush pending; retrying: %s",
                    record["store"].chat["id"],
                )
        self._enforce_tail_budget()
        self._publish_tail_status()

    def _enforce_tail_budget(self):
        while True:
            with self._chat_mutex:
                records = list(self._tail_heads.values())
                total = sum(
                    sum(len(data) for data in record["tail_blocks"].values()) for record in records
                )
                if len(records) <= self._tail_ram_max_chats and total <= self._tail_ram_max_bytes:
                    return
                eligible = [
                    record
                    for record in records
                    if not self._states_for_chat(record["store"].chat["id"])
                ]
                if not eligible:
                    return
                oldest = min(eligible, key=lambda record: record["last_access"])
                oldest["force"] = True
            try:
                if self._flush_record(oldest, "ram_eviction") is None:
                    return
            except (OSError, ValueError):
                oldest["retry_after"] = time.monotonic() + 5
                logger.exception(
                    "RAM eviction is waiting for a durable snapshot tail: %s",
                    oldest["store"].chat["id"],
                )
                return

    def _force_identity(self, chat):
        chat = identity(chat)
        key = (chat["id"], chat["generation"])
        self._publish_ready(force_identities={key})
        with self._chat_mutex:
            if self._states_for_chat(chat["id"]):
                return None
            record = self._tail_heads.get(key)
            store = self._chat_stores.get(chat["id"])
        if record is not None:
            record["force"] = True
            return self._flush_record(record, "explicit")
        if store is None or store.chat["generation"] != chat["generation"]:
            raise ValueError("live backend has no matching chat generation")
        info = store.metadata()
        if not store.current():
            raise ValueError("chat generation was retired before tail flush")
        return {"status": "already_durable", "tokens": info.get("tokens", 0)}

    def _process_control_requests(self):
        for path in sorted(self._control_directory.glob("*.request.json")):
            match = CONTROL_FILE.fullmatch(path.name)
            if match is None or path.is_symlink() or not path.is_file():
                continue
            nonce = match.group(1)
            response = self._control_directory / f"{nonce}.response.json"
            if response.exists():
                path.unlink(missing_ok=True)
                continue
            try:
                if path.stat().st_size > 65536:
                    raise ValueError("oversized snapshot control request")
                request = json.loads(path.read_text())
                if (
                    request.get("schema") != CONTROL_SCHEMA
                    or request.get("nonce") != nonce
                    or request.get("action") != "flush"
                ):
                    raise ValueError("invalid snapshot control request")
                result = self._force_identity(request.get("chat"))
                if result is None:
                    continue
                payload = {"schema": CONTROL_SCHEMA, "nonce": nonce, **result}
            except Exception as error:
                payload = {
                    "schema": CONTROL_SCHEMA,
                    "nonce": nonce,
                    "status": "error",
                    "error": str(error),
                }
            atomic_write(response, json.dumps(payload, sort_keys=True).encode())
            path.unlink(missing_ok=True)

    def _publish_tail_status(self, *, force=False):
        with self._chat_mutex:
            records = list(self._tail_heads.values())
            body = {
                "schema": "urn:qwen-r9700:radiance-tail-residency:v1",
                "pid": os.getpid(),
                "flush_tokens": self._tail_flush_tokens,
                "max_bytes": self._tail_ram_max_bytes,
                "max_chats": self._tail_ram_max_chats,
                "chats": [
                    {
                        "chat_id": record["store"].chat["id"],
                        "generation": record["store"].chat["generation"],
                        "tokens": record["tokens"],
                        "durable_tokens": record["durable_tokens"],
                        "blocks": len(record["tail_blocks"]),
                        "bytes": sum(len(data) for data in record["tail_blocks"].values()),
                    }
                    for record in records
                ],
            }
            if self._last_tail_flush is not None:
                body["last_flush"] = dict(self._last_tail_flush)
            signature = json.dumps(body, sort_keys=True)
            now = time.monotonic()
            if (
                not force
                and signature == self._tail_status_signature
                and now - self._tail_status_written < 5
            ):
                return
            self._tail_status_signature = signature
            self._tail_status_written = now
            payload = {**body, "updated_at": time.time()}
        path = self._tail_status_path
        if path.exists() and (path.is_symlink() or not path.is_file()):
            raise ValueError("unsafe snapshot tail status path")
        atomic_write(path, json.dumps(payload, sort_keys=True).encode())

    def _states_for_chat(self, chat_id):
        return {
            rid: state
            for rid, state in self._chat_requests.items()
            if state["store"].chat["id"] == chat_id
        }

    def shutdown(self):
        self._retry_stop.set()
        self._publish_wake.set()
        self._retry_thread.join()
        try:
            # Finish every acknowledged RAM copy, then publish each complete
            # candidate before the general CPU tier and its mmap disappear.
            self.drain_jobs()
            self.get_finished_jobs()
            with self._chat_mutex:
                # The server can begin an orderly shutdown while a generation is
                # still active. It has no settled head to publish, but it must not
                # prevent the preceding complete RAM tail from becoming durable.
                for state in self._chat_requests.values():
                    if not state["finished"]:
                        state["finished"] = True
                        state["failed"] = True
            self._publish_ready(force_all=True)
            self._process_control_requests()
        except Exception:
            logger.exception("Clean backend shutdown could not flush every snapshot tail")
        try:
            super().shutdown()
        finally:
            self._tail_status_path.unlink(missing_ok=True)
            os.close(self._engine_lock)

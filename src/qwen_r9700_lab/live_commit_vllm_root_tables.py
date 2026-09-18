"""Atomic scheduler block-table roots for the fixed Qwen3.8/DFlash topology.

The production scheduler stores one mutable ``request_id -> block list`` mapping
per KV-cache group.  Updating 69 ordinary dictionaries cannot be an atomic
publication.  This module replaces only the protected request's lookup with a
shared root bank: all 69 views resolve through one canonical root identifier,
while a context-local override exposes a private candidate or serial root.

Immutable prefix blocks are shared.  Each private root owns fresh blocks for
the 48 recurrent states and every attention page that an eight-token round can
write.  Publication is one pointer assignment after all validation.  The
module deliberately does not execute either model arm; it is the allocator and
root-publication primitive consumed by the live commit coordinator.
"""

from __future__ import annotations

import contextvars
import re
import threading
from collections.abc import Callable, Iterator, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from qwen_r9700_lab.live_commit_scheduler_authority import PhysicalRoot

MAMBA_GROUPS = 48
TARGET_ATTENTION_GROUPS = 16
DRAFT_ATTENTION_GROUPS = 5
TOTAL_GROUPS = MAMBA_GROUPS + TARGET_ATTENTION_GROUPS + DRAFT_ATTENTION_GROUPS
MAX_COMMIT_WIDTH = 8

_ROOT_ID = re.compile(r"^[A-Za-z0-9._:+/-]{1,256}$")
_MISSING = object()


class VllmRootTableError(RuntimeError):
    """A cache topology, ownership operation, or root publication is unsafe."""


def _root_id(value: object, label: str) -> str:
    if not isinstance(value, str) or _ROOT_ID.fullmatch(value) is None:
        raise VllmRootTableError(f"{label} is invalid")
    return value


def _block_id(block: object, label: str) -> int:
    value = getattr(block, "block_id", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise VllmRootTableError(f"{label} has an invalid block ID")
    return value


def _manager_kind(manager: object) -> str:
    """Classify without importing vLLM into the controller environment."""

    name = type(manager).__name__
    if name == "MambaManager":
        return "mamba"
    if name == "FullAttentionManager":
        return "target"
    if name == "SlidingWindowManager":
        return "draft"
    raise VllmRootTableError(f"unsupported KV manager class: {name}")


@dataclass(frozen=True)
class QwenGroupTopology:
    """Authenticated structural facts needed to address one cache group."""

    index: int
    kind: str
    block_size: int


@dataclass(frozen=True)
class PrivateRootAllocation:
    """The three roots and the atomic table bank that owns their mappings."""

    canonical: PhysicalRoot
    candidate: PhysicalRoot
    serial: PhysicalRoot
    bank: AtomicRootTableBank
    topology: tuple[QwenGroupTopology, ...]
    new_block_ids_to_zero: tuple[int, ...]


class AtomicRootTableBank:
    """One-pointer canonical publication with context-local private selection."""

    def __init__(
        self,
        *,
        request_id: str,
        canonical_root_id: str,
        canonical_tables: Sequence[Sequence[object]],
    ) -> None:
        if not isinstance(request_id, str) or not request_id:
            raise VllmRootTableError("request ID is invalid")
        canonical = _root_id(canonical_root_id, "canonical root ID")
        tables = self._normalize_tables(canonical_tables, "canonical tables")
        self.request_id = request_id
        self._tables: dict[str, tuple[list[object], ...]] = {canonical: tables}
        self._canonical_root_id = canonical
        self._canonical_root_getter: Callable[[], str] | None = None
        self._selection: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            f"qwen_root_{id(self):x}", default=None
        )
        self._lock = threading.Lock()

    @staticmethod
    def _normalize_tables(
        tables: Sequence[Sequence[object]], label: str
    ) -> tuple[list[object], ...]:
        if not isinstance(tables, Sequence) or len(tables) != TOTAL_GROUPS:
            raise VllmRootTableError(f"{label} group count differs")
        normalized: list[list[object]] = []
        for index, group in enumerate(tables):
            if not isinstance(group, Sequence):
                raise VllmRootTableError(f"{label}[{index}] is not a sequence")
            normalized.append(list(group))
        return tuple(normalized)

    @property
    def canonical_root_id(self) -> str:
        getter = self._canonical_root_getter
        if getter is None:
            return self._canonical_root_id
        try:
            root = _root_id(getter(), "external canonical root ID")
        except VllmRootTableError:
            raise
        except BaseException as error:
            raise VllmRootTableError("external canonical root lookup failed") from error
        if root not in self._tables:
            raise VllmRootTableError("external canonical root is not registered")
        return root

    @property
    def canonical_is_external(self) -> bool:
        return self._canonical_root_getter is not None

    @property
    def selected_root_id(self) -> str:
        selected = self._selection.get()
        return self.canonical_root_id if selected is None else selected

    @property
    def private_selection_active(self) -> bool:
        return self._selection.get() is not None

    @property
    def root_ids(self) -> frozenset[str]:
        return frozenset(self._tables)

    def register(self, root_id: str, tables: Sequence[Sequence[object]]) -> None:
        root = _root_id(root_id, "registered root ID")
        normalized = self._normalize_tables(tables, f"root {root} tables")
        with self._lock:
            if root in self._tables:
                raise VllmRootTableError("root is already registered")
            self._tables[root] = normalized

    def bind_canonical_root(self, getter: Callable[[], str]) -> None:
        """Bind lookups to an external composite authority's sole state pointer.

        Binding is a startup-only operation.  Once bound, this bank cannot
        publish independently: the external authority atomically changes its
        complete immutable state and every block-table proxy observes that same
        root through ``getter``.
        """

        if not callable(getter):
            raise VllmRootTableError("external canonical root getter is invalid")
        with self._lock:
            if self._canonical_root_getter is not None:
                raise VllmRootTableError("external canonical root is already bound")
            try:
                observed = _root_id(getter(), "external canonical root ID")
            except VllmRootTableError:
                raise
            except BaseException as error:
                raise VllmRootTableError("external canonical root lookup failed") from error
            if observed != self._canonical_root_id or observed not in self._tables:
                raise VllmRootTableError("external canonical root differs at binding")
            self._canonical_root_getter = getter

    def table(self, group_index: int, root_id: str | None = None) -> list[object]:
        if (
            isinstance(group_index, bool)
            or not isinstance(group_index, int)
            or not 0 <= group_index < TOTAL_GROUPS
        ):
            raise VllmRootTableError("group index is invalid")
        root = self.selected_root_id if root_id is None else _root_id(root_id, "root ID")
        try:
            return self._tables[root][group_index]
        except KeyError as error:
            raise VllmRootTableError("root is not registered") from error

    def tables(self, root_id: str | None = None) -> tuple[list[object], ...]:
        root = self.selected_root_id if root_id is None else _root_id(root_id, "root ID")
        try:
            return self._tables[root]
        except KeyError as error:
            raise VllmRootTableError("root is not registered") from error

    @contextmanager
    def select_private(self, root_id: str) -> Iterator[None]:
        root = _root_id(root_id, "private root ID")
        if root == self.canonical_root_id or root not in self._tables:
            raise VllmRootTableError("private root selection is invalid")
        token = self._selection.set(root)
        try:
            yield
        finally:
            self._selection.reset(token)

    def publish(self, *, expected_canonical: str, selected_root: str) -> str:
        """Publish by one assignment; validation cannot fail after the write."""

        expected = _root_id(expected_canonical, "expected canonical root ID")
        selected = _root_id(selected_root, "selected root ID")
        with self._lock:
            if self._canonical_root_getter is not None:
                raise VllmRootTableError(
                    "externally bound canonical root cannot publish independently"
                )
            if self._selection.get() is not None:
                raise VllmRootTableError("cannot publish inside a private-root context")
            if self.canonical_root_id != expected:
                raise VllmRootTableError("canonical root changed before publication")
            if selected == expected or selected not in self._tables:
                raise VllmRootTableError("selected publication root is invalid")
            previous = self._canonical_root_id
            self._canonical_root_id = selected
            return previous


class RootedRequestBlockMap(MutableMapping[str, list[object]]):
    """Per-manager map proxy whose protected key follows ``AtomicRootTableBank``."""

    def __init__(
        self,
        delegate: MutableMapping[str, list[object]],
        bank: AtomicRootTableBank,
        group_index: int,
    ) -> None:
        if not isinstance(delegate, MutableMapping):
            raise VllmRootTableError("request block-map delegate is invalid")
        bank.table(group_index)
        self.delegate = delegate
        self.bank = bank
        self.group_index = group_index

    def __getitem__(self, key: str) -> list[object]:
        if key == self.bank.request_id:
            return self.bank.table(self.group_index)
        return self.delegate[key]

    def __setitem__(self, key: str, value: list[object]) -> None:
        if key == self.bank.request_id:
            if not isinstance(value, list):
                raise VllmRootTableError("protected request blocks must be a list")
            current = self.bank.table(self.group_index)
            current[:] = value
            return
        self.delegate[key] = value

    def __delitem__(self, key: str) -> None:
        if key == self.bank.request_id:
            self.bank.table(self.group_index).clear()
            return
        del self.delegate[key]

    def __iter__(self) -> Iterator[str]:
        yielded_protected = False
        for key in self.delegate:
            if key == self.bank.request_id:
                if not yielded_protected:
                    yielded_protected = True
                    yield key
            else:
                yield key
        if not yielded_protected:
            yield self.bank.request_id

    def __len__(self) -> int:
        return len(set(self.delegate) | {self.bank.request_id})

    def get(self, key: str, default: object = None) -> Any:
        if key == self.bank.request_id:
            return self.bank.table(self.group_index)
        return self.delegate.get(key, default)

    def pop(self, key: str, default: object = _MISSING) -> Any:
        if key == self.bank.request_id:
            value = self.bank.table(self.group_index).copy()
            self.bank.table(self.group_index).clear()
            return value
        if default is _MISSING:
            return self.delegate.pop(key)
        return self.delegate.pop(key, default)


class QwenSchedulerRootAllocator:
    """Reserve and install three exact roots over vLLM's shared block pool."""

    def __init__(
        self,
        coordinator: object,
        *,
        request_id: str,
        logical_length: int,
        allocator_generation: int,
        canonical_root_id: str,
        candidate_root_id: str,
        serial_root_id: str,
        max_commit_width: int = MAX_COMMIT_WIDTH,
    ) -> None:
        if not isinstance(request_id, str) or not request_id:
            raise VllmRootTableError("request ID is invalid")
        for value, label in (
            (logical_length, "logical length"),
            (allocator_generation, "allocator generation"),
            (max_commit_width, "maximum commit width"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise VllmRootTableError(f"{label} is invalid")
        if not 1 <= max_commit_width <= MAX_COMMIT_WIDTH:
            raise VllmRootTableError("maximum commit width is outside 1..8")
        self.coordinator = coordinator
        self.request_id = request_id
        self.logical_length = logical_length
        self.allocator_generation = allocator_generation
        self.root_ids = (
            _root_id(canonical_root_id, "canonical root ID"),
            _root_id(candidate_root_id, "candidate root ID"),
            _root_id(serial_root_id, "serial root ID"),
        )
        if len(set(self.root_ids)) != 3:
            raise VllmRootTableError("root IDs are not unique")
        self.max_commit_width = max_commit_width
        self.managers, self.topology, self.block_pool, self.null_block = (
            self._validate_topology()
        )
        self._original_maps: tuple[MutableMapping[str, list[object]], ...] | None = None
        self._allocation: PrivateRootAllocation | None = None
        self._private_ownership: tuple[object, ...] = ()

    def _validate_topology(
        self,
    ) -> tuple[
        tuple[object, ...],
        tuple[QwenGroupTopology, ...],
        object,
        object,
    ]:
        managers_raw = getattr(self.coordinator, "single_type_managers", None)
        if not isinstance(managers_raw, Sequence) or len(managers_raw) != TOTAL_GROUPS:
            raise VllmRootTableError("KV coordinator does not have exactly 69 groups")
        managers = tuple(managers_raw)
        expected = (
            ("mamba",) * MAMBA_GROUPS
            + ("target",) * TARGET_ATTENTION_GROUPS
            + ("draft",) * DRAFT_ATTENTION_GROUPS
        )
        pools = {id(getattr(manager, "block_pool", None)) for manager in managers}
        if len(pools) != 1 or None in (getattr(managers[0], "block_pool", None),):
            raise VllmRootTableError("KV groups do not share one block pool")
        block_pool = managers[0].block_pool
        null_block = getattr(block_pool, "null_block", None)
        if null_block is None:
            raise VllmRootTableError("block pool has no null block")
        topology: list[QwenGroupTopology] = []
        real_ids: set[int] = set()
        for index, (manager, expected_kind) in enumerate(
            zip(managers, expected, strict=True)
        ):
            if getattr(manager, "kv_cache_group_id", None) != index:
                raise VllmRootTableError("KV group IDs are not canonical")
            kind = _manager_kind(manager)
            if kind != expected_kind:
                raise VllmRootTableError("KV group ordering differs from Qwen/DFlash")
            block_size = getattr(manager, "block_size", None)
            if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
                raise VllmRootTableError("KV group block size is invalid")
            if kind == "mamba" and getattr(manager, "mamba_cache_mode", None) != "none":
                raise VllmRootTableError("GDN manager is not in the qualified mode-none lane")
            mapping = getattr(manager, "req_to_blocks", None)
            if not isinstance(mapping, MutableMapping):
                raise VllmRootTableError("KV manager request table is invalid")
            blocks = mapping.get(self.request_id)
            if not isinstance(blocks, list) or not blocks:
                raise VllmRootTableError("protected request has no allocated KV blocks")
            if kind == "mamba":
                non_null = [block for block in blocks if block is not null_block]
                if len(non_null) != 1:
                    raise VllmRootTableError(
                        "mode-none GDN group must own exactly one physical state block"
                    )
            for block in blocks:
                if block is null_block:
                    continue
                identifier = _block_id(block, f"group {index} block")
                if identifier in real_ids:
                    raise VllmRootTableError("physical cache block is owned by two groups")
                real_ids.add(identifier)
            topology.append(QwenGroupTopology(index, kind, block_size))
        return managers, tuple(topology), block_pool, null_block

    def _write_indices(self, block_size: int) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    position // block_size
                    for position in range(
                        self.logical_length,
                        self.logical_length + self.max_commit_width,
                    )
                }
            )
        )

    def _private_block_count(self) -> int:
        count = MAMBA_GROUPS
        for group in self.topology[MAMBA_GROUPS:]:
            count += len(self._write_indices(group.block_size))
        return count

    def _extend_canonical_capacity(
        self,
        canonical_tables: tuple[list[object], ...],
        reserved: Iterator[object],
    ) -> None:
        """Reserve capacity for the full M8 write span before a round opens."""

        for topology, group in zip(self.topology, canonical_tables, strict=True):
            if topology.kind == "mamba":
                continue
            history_pages = (
                self.logical_length + topology.block_size - 1
            ) // topology.block_size
            if len(group) < history_pages:
                raise VllmRootTableError(
                    "attention root is missing committed logical history"
                )
            required_pages = self._write_indices(topology.block_size)[-1] + 1
            while len(group) < required_pages:
                group.append(next(reserved))

    def _canonical_capacity_count(self) -> int:
        count = 0
        for topology, manager in zip(self.topology, self.managers, strict=True):
            if topology.kind == "mamba":
                continue
            blocks = manager.req_to_blocks[self.request_id]
            history_pages = (
                self.logical_length + topology.block_size - 1
            ) // topology.block_size
            if len(blocks) < history_pages:
                raise VllmRootTableError(
                    "attention root is missing committed logical history"
                )
            required_pages = self._write_indices(topology.block_size)[-1] + 1
            count += max(0, required_pages - len(blocks))
        return count

    def _allocate_blocks(self, count: int) -> list[object]:
        getter = getattr(self.block_pool, "get_new_blocks", None)
        if not callable(getter):
            raise VllmRootTableError("block pool allocation API is unavailable")
        blocks = getter(count)
        if not isinstance(blocks, list) or len(blocks) != count:
            raise VllmRootTableError("block pool returned an incomplete reservation")
        identifiers = [_block_id(block, "reserved block") for block in blocks]
        if len(identifiers) != len(set(identifiers)):
            raise VllmRootTableError("block pool returned duplicate reservations")
        return blocks

    def _free_blocks(self, blocks: Sequence[object]) -> None:
        free = getattr(self.block_pool, "free_blocks", None)
        if not callable(free):
            raise VllmRootTableError("block pool free API is unavailable")
        free(reversed(tuple(blocks)))

    def _acquire_shared_blocks(
        self,
        canonical_tables: tuple[list[object], ...],
        private_tables: tuple[list[object], ...],
        acquired: list[object],
    ) -> None:
        """Add one allocator reference for every prefix page a root shares."""

        touch = getattr(self.block_pool, "touch", None)
        if not callable(touch):
            raise VllmRootTableError("block pool touch API is unavailable")
        canonical_objects = {
            id(block)
            for group in canonical_tables
            for block in group
            if block is not self.null_block
        }
        shared = [
            block
            for group in private_tables
            for block in group
            if block is not self.null_block and id(block) in canonical_objects
        ]
        if len(shared) != len({id(block) for block in shared}):
            raise VllmRootTableError("a private root references one real block twice")
        for block in shared:
            before = getattr(block, "ref_cnt", None)
            if isinstance(before, bool) or not isinstance(before, int) or before <= 0:
                raise VllmRootTableError("shared block has an invalid reference count")
            try:
                touch((block,))
            except BaseException as error:
                after_error = getattr(block, "ref_cnt", None)
                if after_error == before + 1:
                    acquired.append(block)
                elif after_error != before:
                    raise VllmRootTableError(
                        "shared-block acquisition failed with an unknown ownership state"
                    ) from error
                raise
            if getattr(block, "ref_cnt", None) != before + 1:
                raise VllmRootTableError(
                    "shared-block acquisition did not add exactly one reference"
                )
            acquired.append(block)

    def _build_private_tables(
        self,
        canonical_tables: tuple[list[object], ...],
        reserved: Iterator[object],
    ) -> tuple[tuple[list[object], ...], tuple[tuple[int, ...], ...]]:
        private = tuple(list(group) for group in canonical_tables)
        writable: list[tuple[int, ...]] = []
        for topology, group in zip(self.topology, private, strict=True):
            if topology.kind == "mamba":
                real_indexes = [
                    index for index, block in enumerate(group) if block is not self.null_block
                ]
                if len(real_indexes) != 1:
                    raise VllmRootTableError("GDN root shape changed during reservation")
                group[real_indexes[0]] = next(reserved)
                writable.append((_block_id(group[real_indexes[0]], "private GDN block"),))
                continue
            group_writable: list[int] = []
            for logical_index in self._write_indices(topology.block_size):
                if logical_index >= len(group):  # pragma: no cover - capacity is prebuilt.
                    raise VllmRootTableError("attention root capacity preparation failed")
                block = next(reserved)
                group[logical_index] = block
                group_writable.append(_block_id(block, "private attention block"))
            writable.append(tuple(group_writable))
        return private, tuple(writable)

    def _physical_root(
        self,
        root_id: str,
        tables: tuple[list[object], ...],
        writable: tuple[tuple[int, ...], ...],
    ) -> PhysicalRoot:
        mamba_ids: list[int] = []
        for group in tables[:MAMBA_GROUPS]:
            non_null = [block for block in group if block is not self.null_block]
            if len(non_null) != 1:
                raise VllmRootTableError("GDN root no longer has exactly one state block")
            mamba_ids.append(_block_id(non_null[0], "GDN state block"))
        target_tables = tables[
            MAMBA_GROUPS : MAMBA_GROUPS + TARGET_ATTENTION_GROUPS
        ]
        draft_tables = tables[MAMBA_GROUPS + TARGET_ATTENTION_GROUPS :]
        target_writable = writable[
            MAMBA_GROUPS : MAMBA_GROUPS + TARGET_ATTENTION_GROUPS
        ]
        draft_writable = writable[MAMBA_GROUPS + TARGET_ATTENTION_GROUPS :]
        return PhysicalRoot(
            root_id=root_id,
            request_id=self.request_id,
            allocator_generation=self.allocator_generation,
            target_blocks=tuple(
                tuple(_block_id(block, "target block") for block in group)
                for group in target_tables
            ),
            draft_blocks=tuple(
                tuple(
                    _block_id(block, "draft block")
                    for block in group
                )
                for group in draft_tables
            ),
            writable_target_blocks=tuple(target_writable),
            writable_draft_blocks=tuple(draft_writable),
            gdn_slots=tuple(mamba_ids),
            convolution_slots=tuple(mamba_ids),
            position_state_id=f"{root_id}:position",
            decoding_state_id=f"{root_id}:decoding",
        )

    def reserve(self) -> PrivateRootAllocation:
        if self._allocation is not None:
            raise VllmRootTableError("private roots are already reserved")
        canonical_tables = tuple(
            list(manager.req_to_blocks[self.request_id])
            for manager in self.managers
        )
        per_root = self._private_block_count()
        canonical_capacity = self._canonical_capacity_count()
        allocated: list[object] = []
        acquired: list[object] = []
        try:
            allocated = self._allocate_blocks(canonical_capacity + per_root * 2)
            acquired.extend(allocated)
            iterator = iter(allocated)
            self._extend_canonical_capacity(canonical_tables, iterator)
            candidate_tables, candidate_writable = self._build_private_tables(
                canonical_tables, iterator
            )
            serial_tables, serial_writable = self._build_private_tables(
                canonical_tables, iterator
            )
            try:
                next(iterator)
            except StopIteration:
                pass
            else:  # pragma: no cover - count calculation and builder share one source.
                raise VllmRootTableError("private reservation has unused blocks")
            self._acquire_shared_blocks(
                canonical_tables, candidate_tables, acquired
            )
            self._acquire_shared_blocks(canonical_tables, serial_tables, acquired)

            canonical_writable: list[tuple[int, ...]] = []
            for topology, group in zip(self.topology, canonical_tables, strict=True):
                if topology.kind == "mamba":
                    canonical_writable.append(
                        tuple(
                            _block_id(block, "canonical GDN block")
                            for block in group
                            if block is not self.null_block
                        )
                    )
                    continue
                indexes = self._write_indices(topology.block_size)
                canonical_writable.append(
                    tuple(
                        _block_id(group[index], "canonical writable attention block")
                        for index in indexes
                        if index < len(group) and group[index] is not self.null_block
                    )
                )
            bank = AtomicRootTableBank(
                request_id=self.request_id,
                canonical_root_id=self.root_ids[0],
                canonical_tables=canonical_tables,
            )
            bank.register(self.root_ids[1], candidate_tables)
            bank.register(self.root_ids[2], serial_tables)
            allocation = PrivateRootAllocation(
                canonical=self._physical_root(
                    self.root_ids[0], canonical_tables, tuple(canonical_writable)
                ),
                candidate=self._physical_root(
                    self.root_ids[1], candidate_tables, candidate_writable
                ),
                serial=self._physical_root(
                    self.root_ids[2], serial_tables, serial_writable
                ),
                bank=bank,
                topology=self.topology,
                new_block_ids_to_zero=tuple(
                    _block_id(block, "new root block") for block in allocated
                ),
            )
        except BaseException:
            if acquired:
                self._free_blocks(acquired)
            raise
        self._allocation = allocation
        self._private_ownership = tuple(acquired)
        return allocation

    def install(self) -> PrivateRootAllocation:
        allocation = self.reserve() if self._allocation is None else self._allocation
        assert allocation is not None
        if self._original_maps is not None:
            raise VllmRootTableError("root table bank is already installed")
        original = tuple(manager.req_to_blocks for manager in self.managers)
        installed: list[int] = []
        try:
            for index, (manager, mapping) in enumerate(
                zip(self.managers, original, strict=True)
            ):
                observed = list(mapping[self.request_id])
                prepared = allocation.bank.table(index)
                if len(observed) > len(prepared) or observed != prepared[: len(observed)]:
                    raise VllmRootTableError(
                        "canonical request table changed before proxy installation"
                    )
                manager.req_to_blocks = RootedRequestBlockMap(
                    mapping, allocation.bank, index
                )
                installed.append(index)
        except BaseException:
            for index in reversed(installed):
                self.managers[index].req_to_blocks = original[index]
            if self._private_ownership:
                self._free_blocks(self._private_ownership)
            self._private_ownership = ()
            self._allocation = None
            raise
        self._original_maps = original
        return allocation

    def close(self) -> None:
        """Restore ordinary manager maps and release every noncanonical root ref.

        The current canonical table retains exactly one reference per real
        block so the coordinator's normal request-finalization path can free it.
        All spare-root and additional shared-prefix references are released.
        """

        if self._original_maps is None or self._allocation is None:
            raise VllmRootTableError("root table bank is not installed")
        bank = self._allocation.bank
        if bank.private_selection_active:
            raise VllmRootTableError("cannot close inside a private-root context")
        canonical = bank.tables(bank.canonical_root_id)
        all_occurrences = [
            block
            for root_id in sorted(bank.root_ids)
            for group in bank.tables(root_id)
            for block in group
            if block is not self.null_block
        ]
        canonical_objects = {
            id(block) for group in canonical for block in group if block is not self.null_block
        }
        kept: set[int] = set()
        to_free: list[object] = []
        for block in all_occurrences:
            identity = id(block)
            if identity in canonical_objects and identity not in kept:
                kept.add(identity)
            else:
                to_free.append(block)

        original_values = tuple(
            list(mapping.get(self.request_id, [])) for mapping in self._original_maps
        )
        updated: list[int] = []
        try:
            for index, mapping in enumerate(self._original_maps):
                mapping[self.request_id] = list(canonical[index])
                updated.append(index)
        except BaseException:
            for index in reversed(updated):
                self._original_maps[index][self.request_id] = original_values[index]
            raise
        for index, manager in enumerate(self.managers):
            manager.req_to_blocks = self._original_maps[index]
        self._free_blocks(to_free)
        self._original_maps = None
        self._private_ownership = ()
        self._allocation = None

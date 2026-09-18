"""Pure host-side continuity policy for Quest historical page-set reuse."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass


def block_table_extends(previous: tuple[int, ...], current: tuple[int, ...]) -> bool:
    """Return whether ``current`` preserves every prior physical block."""

    return len(current) >= len(previous) and current[: len(previous)] == previous


@dataclass(frozen=True, slots=True)
class PageReuseHit[PagePayload]:
    historical_pages: PagePayload
    recent_start: int
    total_pages: int


@dataclass(slots=True)
class _Entry[PagePayload]:
    request_token: object
    block_table_host: tuple[int, ...]
    sequence_length: int
    budget_pages: int
    recent_pages: int
    historical_pages: PagePayload
    historical_count: int
    reuse_count: int = 0


class HistoricalPageReusePolicy[PagePayload]:
    """Track tiny per-layer page payloads without consulting a device tensor."""

    def __init__(
        self,
        *,
        refresh_interval: int = 4,
        max_sequence_advance: int = 9,
        logical_page_tokens: int = 64,
    ) -> None:
        if refresh_interval < 1:
            raise ValueError("refresh_interval must be positive")
        if max_sequence_advance < 1:
            raise ValueError("max_sequence_advance must be positive")
        if logical_page_tokens < 1:
            raise ValueError("logical_page_tokens must be positive")
        self.refresh_interval = refresh_interval
        self.max_sequence_advance = max_sequence_advance
        self.logical_page_tokens = logical_page_tokens
        self._entries: dict[Hashable, _Entry[PagePayload]] = {}

    def clear(self) -> None:
        self._entries.clear()

    def contains(self, layer_identity: Hashable) -> bool:
        return layer_identity in self._entries

    def forget(self, layer_identity: Hashable) -> None:
        self._entries.pop(layer_identity, None)

    def remember(
        self,
        *,
        layer_identity: Hashable,
        request_token: object | None,
        block_table_host: tuple[int, ...] | None,
        sequence_length: int,
        budget_pages: int,
        recent_pages: int,
        historical_pages: PagePayload,
        historical_count: int,
    ) -> bool:
        """Record a fresh ranking, or forget the layer when metadata is unsafe."""

        if request_token is None or block_table_host is None or historical_count <= 0:
            self.forget(layer_identity)
            return False
        self._entries[layer_identity] = _Entry(
            request_token=request_token,
            block_table_host=block_table_host,
            sequence_length=int(sequence_length),
            budget_pages=int(budget_pages),
            recent_pages=int(recent_pages),
            historical_pages=historical_pages,
            historical_count=int(historical_count),
        )
        return True

    def reuse(
        self,
        *,
        layer_identity: Hashable,
        request_token: object | None,
        block_table_host: tuple[int, ...] | None,
        sequence_length: int,
        budget_pages: int,
        recent_pages: int,
    ) -> PageReuseHit[PagePayload] | None:
        """Return a reusable history and current tail bounds, otherwise forget it."""

        if self.refresh_interval <= 1:
            return None
        entry = self._entries.get(layer_identity)
        if entry is None:
            return None
        if request_token is None or block_table_host is None:
            self.forget(layer_identity)
            return None
        sequence_advance = int(sequence_length) - entry.sequence_length
        continuous = (
            request_token is entry.request_token
            and 0 < sequence_advance <= self.max_sequence_advance
            and block_table_extends(entry.block_table_host, block_table_host)
            and int(budget_pages) == entry.budget_pages
            and int(recent_pages) == entry.recent_pages
        )
        if not continuous or entry.reuse_count >= self.refresh_interval - 1:
            self.forget(layer_identity)
            return None

        total_pages = (
            int(sequence_length) + self.logical_page_tokens - 1
        ) // self.logical_page_tokens
        selected_count = min(int(budget_pages), total_pages)
        recent_count = min(max(0, int(recent_pages)), selected_count)
        historical_count = selected_count - recent_count
        if entry.historical_count != historical_count:
            self.forget(layer_identity)
            return None

        entry.sequence_length = int(sequence_length)
        entry.block_table_host = block_table_host
        entry.reuse_count += 1
        return PageReuseHit(
            historical_pages=entry.historical_pages,
            recent_start=total_pages - recent_count,
            total_pages=total_pages,
        )

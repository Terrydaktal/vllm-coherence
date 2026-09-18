"""Reference mapping for the compiled FP8 logical-page attention ABI.

The production cache is paged in large physical blocks (1728 tokens in the
current hybrid profile), while the sparse selector operates on 64-token
logical pages.  This dependency-free module is the executable contract used by
host-side tests and review tooling; the HIP implementation must produce the
same addresses and valid-token counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


DEFAULT_PAGE_TOKENS = 64


@dataclass(frozen=True)
class LogicalPageAddress:
    """Physical address and valid range for one logical page."""

    logical_page: int
    physical_block_index: int
    physical_block_id: int
    token_offset: int
    valid_tokens: int


def pages_per_block(block_size: int, page_tokens: int = DEFAULT_PAGE_TOKENS) -> int:
    """Return the number of logical pages in one physical cache block."""

    if block_size <= 0 or page_tokens <= 0 or block_size % page_tokens:
        raise ValueError("physical block size must be a positive page multiple")
    return block_size // page_tokens


def logical_page_count(sequence_length: int, page_tokens: int = DEFAULT_PAGE_TOKENS) -> int:
    """Return the number of logical pages needed for an occupied prefix."""

    if sequence_length < 0 or page_tokens <= 0:
        raise ValueError("sequence length and page size must be non-negative/positive")
    return (sequence_length + page_tokens - 1) // page_tokens


def map_logical_page(
    logical_page: int,
    *,
    sequence_length: int,
    block_table: Sequence[int],
    block_size: int,
    page_tokens: int = DEFAULT_PAGE_TOKENS,
) -> LogicalPageAddress:
    """Map a logical page to its physical block and valid token span.

    ``block_table`` contains physical cache IDs in logical block order.  The
    returned ``token_offset`` is the offset within that physical block; it is
    not the global token position.  This distinction is the source of several
    easy-to-miss errors when a 64-token selector is layered over a 1728-token
    hybrid cache.
    """

    if sequence_length < 0:
        raise ValueError("sequence length must be non-negative")
    page_count = logical_page_count(sequence_length, page_tokens)
    if logical_page < 0 or logical_page >= page_count:
        raise IndexError("logical page is outside the occupied prefix")
    per_block = pages_per_block(block_size, page_tokens)
    physical_index = logical_page // per_block
    if physical_index >= len(block_table):
        raise IndexError("logical page has no physical block-table entry")
    token_offset = (logical_page % per_block) * page_tokens
    remaining = sequence_length - logical_page * page_tokens
    valid_tokens = min(page_tokens, block_size - token_offset, max(0, remaining))
    return LogicalPageAddress(
        logical_page=logical_page,
        physical_block_index=physical_index,
        physical_block_id=int(block_table[physical_index]),
        token_offset=token_offset,
        valid_tokens=valid_tokens,
    )


def map_tree_node(
    node_index: int,
    *,
    prefix_length: int,
    block_table: Sequence[int],
    block_size: int,
) -> tuple[int, int]:
    """Map an appended one-token tree node to ``(physical_id, offset)``."""

    if node_index < 0 or prefix_length < 0 or block_size <= 0:
        raise ValueError("tree node and cache geometry must be non-negative/positive")
    position = prefix_length + node_index
    physical_index, token_offset = divmod(position, block_size)
    if physical_index >= len(block_table):
        raise IndexError("tree node has no physical block-table entry")
    return int(block_table[physical_index]), token_offset


__all__ = [
    "DEFAULT_PAGE_TOKENS",
    "LogicalPageAddress",
    "logical_page_count",
    "map_logical_page",
    "map_tree_node",
    "pages_per_block",
]

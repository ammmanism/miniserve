"""miniserve.blocks — KV-cache block management and paged memory allocation.

Provides the BlockManager that tracks physical and logical block mappings,
handles allocation, freeing, and copy-on-write for PagedAttention.
"""

from typing import TypedDict


class OutOfBlocksError(Exception):
    """Raised when there are insufficient free blocks for allocation."""
    pass


class UsageDict(TypedDict):
    total: int
    free: int
    used: int
    refcounted: int


class BlockManager:
    """Manages physical KV-cache blocks with reference counting and copy-on-write."""

    def __init__(self, num_blocks: int, block_size: int = 16) -> None:
        """Initialize the block manager.

        Args:
            num_blocks: Total number of physical blocks available.
            block_size: Size of each block in tokens (default 16).
        """
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        self._num_blocks = num_blocks
        self._block_size = block_size
        self._free: list[int] = list(range(num_blocks))
        self._refcount: list[int] = [0] * num_blocks
        self._owner: dict[str, set[int]] = {}

    def allocate(self, num_blocks: int, request_id: str) -> list[int]:
        """Allocate a contiguous block table for a request.

        Args:
            num_blocks: Number of blocks to allocate.
            request_id: Unique identifier for the request.

        Returns:
            List of physical block IDs (the block table).

        Raises:
            OutOfBlocksError: If insufficient free blocks available.
        """
        if num_blocks < 0:
            raise ValueError("num_blocks cannot be negative")
        if num_blocks == 0:
            return []

        if len(self._free) < num_blocks:
            raise OutOfBlocksError(f"Insufficient free blocks: need {num_blocks}, have {len(self._free)}")

        # Atomic allocation: take blocks from free list
        block_table = self._free[:num_blocks]
        self._free = self._free[num_blocks:]

        # Update refcount and owner
        for block_id in block_table:
            self._refcount[block_id] = 1

        if request_id not in self._owner:
            self._owner[request_id] = set()
        self._owner[request_id].update(block_table)

        return block_table

    def free(self, block_table: list[int], request_id: str) -> None:
        """Release one reference on each block in ``block_table``.

        The physical block returns to the free list only when its refcount
        reaches zero.  Ownership is validated by refcount, not by the
        ``_owner`` map: fork-shared holders (e.g. the prefix cache, beam
        copies) release their own references here without having ever
        called ``allocate``.  A block whose refcount is already zero is a
        double-free, and the whole table is validated before any mutation
        so a failed free leaves state untouched.

        Args:
            block_table: List of physical block IDs to release.
            request_id: Identifier of the releasing holder (bookkeeping on
                the ``_owner`` map only).

        Raises:
            ValueError: On double-free (refcount already zero) or a table
                containing the same block id twice.
        """
        if not block_table:
            return

        if len(set(block_table)) != len(block_table):
            raise ValueError(f"Duplicate block id in table: {block_table}")

        # Validate the whole table first: a free() must be atomic.
        for block_id in block_table:
            if self._refcount[block_id] <= 0:
                raise ValueError(f"Block {block_id} not in use (double-free?)")

        for block_id in block_table:
            self._refcount[block_id] -= 1
            if self._refcount[block_id] == 0:
                self._free.append(block_id)

        # Owner bookkeeping: drop released ids when this holder was tracked
        # via allocate().  Fork-shared holders have no entry; that is fine.
        if request_id in self._owner:
            owned = self._owner[request_id]
            owned.difference_update(block_table)
            if not owned:
                del self._owner[request_id]

    def fork(self, block_table: list[int]) -> list[int]:
        """Create a copy-on-write fork of a block table.

        Increments refcount on all blocks and returns a new table (copy).

        Args:
            block_table: List of physical block IDs to fork.

        Returns:
            New block table (copy of input) with incremented refcounts.
        """
        if not block_table:
            return []

        new_table = block_table.copy()
        for block_id in block_table:
            self._refcount[block_id] += 1

        return new_table

    def evict_one(self) -> int | None:
        """Evict one free block (refcount == 0) and return its ID.

        The evicted block is moved to the front of the free list so that
        the next allocation will reuse it.

        Returns:
            Physical block ID that was evicted, or None if no evictable block.
        """
        for i, block_id in enumerate(self._free):
            if self._refcount[block_id] == 0:
                # Move to front of free list for immediate reuse
                self._free.pop(i)
                self._free.insert(0, block_id)
                return block_id
        return None

    def usage(self) -> UsageDict:
        """Return current usage statistics.

        Returns:
            Dictionary with total, free, used, and refcounted block counts.
        """
        used = sum(1 for rc in self._refcount if rc > 0)
        refcounted = sum(self._refcount)
        free = len(self._free)
        return {
            "total": self._num_blocks,
            "free": free,
            "used": used,
            "refcounted": refcounted,
        }

    def assert_invariants(self) -> None:
        """Verify internal consistency invariants.

        Raises:
            AssertionError: If any invariant is violated.
        """
        # free + used == total
        usage = self.usage()
        assert usage["free"] + usage["used"] == usage["total"], \
            f"free({usage['free']}) + used({usage['used']}) != total({usage['total']})"

        # sum of refcounts >= used (equality when no forks)
        assert usage["refcounted"] >= usage["used"], \
            f"sum(refcounts)({usage['refcounted']}) < used({usage['used']})"

        # no negative refcounts
        assert all(rc >= 0 for rc in self._refcount), "negative refcount found"

        # no block both free-listed and refcounted
        for block_id in self._free:
            assert self._refcount[block_id] == 0, \
                f"Block {block_id} in free list but refcount={self._refcount[block_id]}"

        # all refcounted blocks are not in free list
        for block_id, rc in enumerate(self._refcount):
            if rc > 0:
                assert block_id not in self._free, \
                    f"Block {block_id} has refcount={rc} but is in free list"

        # owner tracking consistency
        all_owned: set[int] = set()
        for blocks in self._owner.values():
            all_owned.update(blocks)
        for block_id in all_owned:
            assert self._refcount[block_id] > 0, \
                f"Block {block_id} owned but refcount={self._refcount[block_id]}"
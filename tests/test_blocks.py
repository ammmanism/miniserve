"""Tests for miniserve.blocks — KV-cache block management."""
import pytest
from miniserve.blocks import BlockManager, OutOfBlocksError


class TestBlockManager:
    """Test cases for BlockManager."""

    def test_alloc_free_symmetry(self) -> None:
        """alloc/free symmetry: usage() returns to initial after free."""
        mgr = BlockManager(num_blocks=10, block_size=16)
        initial = mgr.usage()

        table = mgr.allocate(3, "req1")
        after_alloc = mgr.usage()
        assert after_alloc["used"] == 3
        assert after_alloc["free"] == 7

        mgr.free(table, "req1")
        after_free = mgr.usage()
        assert after_free == initial
        mgr.assert_invariants()

    def test_allocate_zero(self) -> None:
        """allocate(0) returns []; free([]) is a no-op."""
        mgr = BlockManager(num_blocks=10, block_size=16)

        table = mgr.allocate(0, "req1")
        assert table == []
        assert mgr.usage()["free"] == 10

        mgr.free([], "req1")  # Should not raise
        assert mgr.usage()["free"] == 10
        mgr.assert_invariants()

    def test_allocate_over_capacity_raises(self) -> None:
        """allocate more than capacity raises OutOfBlocksError, leaves usage unchanged."""
        mgr = BlockManager(num_blocks=5, block_size=16)
        initial = mgr.usage()

        with pytest.raises(OutOfBlocksError):
            mgr.allocate(6, "req1")

        # Usage should be unchanged
        assert mgr.usage() == initial
        mgr.assert_invariants()

    def test_fork_increments_refcount(self) -> None:
        """fork increments refcount; freeing one copy does not free physical blocks."""
        mgr = BlockManager(num_blocks=10, block_size=16)

        table1 = mgr.allocate(4, "req1")
        assert mgr.usage()["refcounted"] == 4

        table2 = mgr.fork(table1)
        assert table2 == table1  # Same physical blocks
        assert mgr.usage()["refcounted"] == 8  # Each block now has refcount 2
        assert mgr.usage()["used"] == 4  # Still only 4 physical blocks used

        # Free first copy
        mgr.free(table1, "req1")
        assert mgr.usage()["refcounted"] == 4  # Refcounts back to 1
        assert mgr.usage()["used"] == 4  # Blocks still in use
        assert mgr.usage()["free"] == 6  # Not freed yet

        # Fork-held references are released by the holder's own id,
        # without registration in _owner (see BlockManager.free docs).
        mgr.free(table2, "req2-holder")
        assert mgr.usage()["used"] == 0
        assert mgr.usage()["free"] == 10
        mgr.assert_invariants()

    def test_fork_free_second_copy(self) -> None:
        """fork increments refcount; freeing one copy does not free physical blocks (proper owner)."""
        mgr = BlockManager(num_blocks=10, block_size=16)

        table1 = mgr.allocate(4, "req1")
        table2 = mgr.fork(table1)

        # Free first copy
        mgr.free(table1, "req1")
        assert mgr.usage()["used"] == 4
        assert mgr.usage()["free"] == 6

        # Free second copy
        mgr.free(table2, "req2")
        assert mgr.usage()["used"] == 0
        assert mgr.usage()["free"] == 10
        mgr.assert_invariants()

    def test_double_free_raises(self) -> None:
        """double-free raises."""
        mgr = BlockManager(num_blocks=10, block_size=16)

        table = mgr.allocate(3, "req1")
        mgr.free(table, "req1")

        with pytest.raises(ValueError):
            mgr.free(table, "req1")
        mgr.assert_invariants()

    def test_evict_one_none_when_all_refcounted(self) -> None:
        """evict_one returns None when every block has refcount > 0."""
        mgr = BlockManager(num_blocks=5, block_size=16)

        table = mgr.allocate(5, "req1")  # All blocks allocated
        assert mgr.evict_one() is None
        mgr.assert_invariants()

    def test_evict_one_reuses_block(self) -> None:
        """evict_one then allocate reuses the same physical block ID."""
        mgr = BlockManager(num_blocks=5, block_size=16)

        table = mgr.allocate(3, "req1")
        mgr.free(table, "req1")  # 3 blocks now free, plus 2 never allocated = 5 free

        evicted = mgr.evict_one()
        assert evicted is not None
        assert evicted in table + [3, 4]  # Could be any free block
        assert mgr.usage()["free"] == 5  # Still 5 free (evicted block moved to front)

        # Allocate again - should reuse the evicted block (now at front of free list)
        new_table = mgr.allocate(1, "req2")
        assert new_table[0] == evicted
        assert mgr.usage()["free"] == 4
        mgr.assert_invariants()

    def test_oversubscription(self) -> None:
        """oversubscription: allocate to full, fork all, free one, evict, realloc succeeds."""
        mgr = BlockManager(num_blocks=4, block_size=16)

        # Allocate all blocks
        table1 = mgr.allocate(4, "req1")
        assert mgr.usage()["free"] == 0
        mgr.assert_invariants()

        # Fork all
        table2 = mgr.fork(table1)
        assert mgr.usage()["refcounted"] == 8
        mgr.assert_invariants()

        # Free one copy
        mgr.free(table1, "req1")
        assert mgr.usage()["refcounted"] == 4
        assert mgr.usage()["used"] == 4
        assert mgr.usage()["free"] == 0
        mgr.assert_invariants()

        # Evict one (should be None since all refcounted)
        assert mgr.evict_one() is None
        mgr.assert_invariants()

        # Free second copy
        mgr.free(table2, "req2")
        assert mgr.usage()["free"] == 4
        mgr.assert_invariants()

        # Evict one
        evicted = mgr.evict_one()
        assert evicted is not None
        assert mgr.usage()["free"] == 4  # Still 4 free (evicted block moved to front)
        mgr.assert_invariants()

        # Reallocate
        table3 = mgr.allocate(1, "req3")
        assert table3[0] == evicted
        assert mgr.usage()["free"] == 3
        mgr.assert_invariants()

    def test_usage_exactness(self) -> None:
        """usage() exactness: known small scenario, assert exact integers."""
        mgr = BlockManager(num_blocks=6, block_size=16)

        # Initial state
        assert mgr.usage() == {"total": 6, "free": 6, "used": 0, "refcounted": 0}

        # Allocate 2 blocks
        table1 = mgr.allocate(2, "req1")
        assert mgr.usage() == {"total": 6, "free": 4, "used": 2, "refcounted": 2}

        # Fork them
        table2 = mgr.fork(table1)
        assert mgr.usage() == {"total": 6, "free": 4, "used": 2, "refcounted": 4}

        # Allocate 2 more
        table3 = mgr.allocate(2, "req3")
        assert mgr.usage() == {"total": 6, "free": 2, "used": 4, "refcounted": 6}

        # Free req1
        mgr.free(table1, "req1")
        assert mgr.usage() == {"total": 6, "free": 2, "used": 4, "refcounted": 4}

        # Free req2
        mgr.free(table2, "req2")
        assert mgr.usage() == {"total": 6, "free": 4, "used": 2, "refcounted": 2}

        # Free req3
        mgr.free(table3, "req3")
        assert mgr.usage() == {"total": 6, "free": 6, "used": 0, "refcounted": 0}
        mgr.assert_invariants()

    def test_fuzz_random_operations(self) -> None:
        """fuzz: 500 random alloc/free/fork/evict ops; assert_invariants after each."""
        import random

        mgr = BlockManager(num_blocks=20, block_size=16)
        request_counter = 0
        active_requests: dict[str, list[int]] = {}

        for _ in range(500):
            op = random.choice(["alloc", "free", "fork", "evict"])

            if op == "alloc" and mgr.usage()["free"] > 0:
                num_blocks = random.randint(1, min(5, mgr.usage()["free"]))
                request_counter += 1
                req_id = f"req{request_counter}"
                try:
                    table = mgr.allocate(num_blocks, req_id)
                    active_requests[req_id] = table
                except OutOfBlocksError:
                    pass  # Can happen in race between check and alloc

            elif op == "free" and active_requests:
                req_id = random.choice(list(active_requests.keys()))
                table = active_requests.pop(req_id)
                try:
                    mgr.free(table, req_id)
                except ValueError:
                    # Might fail if already freed via fork path
                    pass

            elif op == "fork" and active_requests:
                req_id = random.choice(list(active_requests.keys()))
                table = active_requests[req_id]
                new_table = mgr.fork(table)
                request_counter += 1
                new_req_id = f"req{request_counter}"
                active_requests[new_req_id] = new_table

            elif op == "evict":
                mgr.evict_one()

            mgr.assert_invariants()

        # Clean up remaining
        for req_id, table in list(active_requests.items()):
            try:
                mgr.free(table, req_id)
            except ValueError:
                pass
        mgr.assert_invariants()

    def test_fork_empty_table(self) -> None:
        """fork on empty table returns empty list."""
        mgr = BlockManager(num_blocks=10, block_size=16)
        assert mgr.fork([]) == []
        mgr.assert_invariants()

    def test_allocate_negative_raises(self) -> None:
        """allocate with negative num_blocks raises ValueError."""
        mgr = BlockManager(num_blocks=10, block_size=16)
        with pytest.raises(ValueError):
            mgr.allocate(-1, "req1")
        mgr.assert_invariants()

    def test_init_invalid_params(self) -> None:
        """BlockManager init validates parameters."""
        with pytest.raises(ValueError):
            BlockManager(0, 16)
        with pytest.raises(ValueError):
            BlockManager(10, 0)
        with pytest.raises(ValueError):
            BlockManager(-1, 16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
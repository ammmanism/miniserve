"""Tests for miniserve.prefix_cache — token-level radix prefix cache.

Convention: the test plays the engine — it allocates a table under a
request id, hands it to insert(), then requests free() on the mirror copy
to simulate request completion.  After that the cache holds the sole
reference, so eviction must return the physical blocks to the free list.
block_size=4 keeps token lists tiny.
"""
from miniserve.blocks import BlockManager
from miniserve.prefix_cache import RadixPrefixCache

BS = 4


def mkseq(start: int, n: int) -> list[int]:
    return [start + i for i in range(n)]


def make_cache(num_blocks: int = 64) -> tuple[BlockManager, RadixPrefixCache]:
    bm = BlockManager(num_blocks=num_blocks, block_size=BS)
    return bm, RadixPrefixCache(bm, block_size=BS)


def cache_with(bm: BlockManager, cache: RadixPrefixCache,
               tokens: list[int], req_id: str) -> list[int]:
    """Engine flow: allocate, insert, release the engine mirror ref."""
    table = bm.allocate(len(tokens) // BS, req_id)
    cache.insert(tokens, table)
    bm.free(table, req_id)
    bm.assert_invariants()
    return table


def test_exact_sequence_hit() -> None:
    """1. exact sequence hit returns full length + block table."""
    bm, cache = make_cache()
    seq = mkseq(100, 8)
    table = cache_with(bm, cache, seq, "r1")
    matched, blocks = cache.match(seq)
    assert matched == 8 and blocks == table
    bm.assert_invariants()


def test_partial_prefix_hit() -> None:
    """2. cache B, query B+C -> matched_len == len(B)."""
    bm, cache = make_cache()
    base = mkseq(200, 8)
    table = cache_with(bm, cache, base, "rB")
    matched, blocks = cache.match(base + mkseq(900, 4))
    assert matched == 8 and blocks == table
    bm.assert_invariants()


def test_longest_match_preferred() -> None:
    """3. cache A and A+B, query A+B+C matches the full A+B path."""
    bm, cache = make_cache()
    a, ab = mkseq(100, 4), mkseq(100, 8)
    ta = bm.allocate(1, "rA")
    cache.insert(a, ta)
    bm.free(ta, "rA")
    bm.assert_invariants()
    # Engine B reused A's block via fork, allocated one more for its tail.
    tab = list(bm.fork(ta)) + bm.allocate(1, "rB")
    cache.insert(ab, tab)
    bm.free(tab, "rB")
    bm.assert_invariants()
    matched, blocks = cache.match(mkseq(100, 12))
    assert matched == 8 and blocks == tab
    bm.assert_invariants()


def test_no_match() -> None:
    """4. no match returns (0, [])."""
    bm, cache = make_cache()
    cache_with(bm, cache, mkseq(100, 4), "r1")
    matched, blocks = cache.match(mkseq(777, 4))
    assert (matched, blocks) == (0, [])
    bm.assert_invariants()


def test_lru_order() -> None:
    """5. cache A, B, C; access A; evict one -> B evicted first."""
    bm, cache = make_cache()
    cache_with(bm, cache, mkseq(100, 4), "rA")
    cache_with(bm, cache, mkseq(200, 4), "rB")
    cache_with(bm, cache, mkseq(300, 4), "rC")
    cache.match(mkseq(100, 4))  # refresh A's LRU stamp
    released = cache.evict_until(1)
    assert released == 1
    assert bm.assert_invariants() is None
    assert cache.match(mkseq(100, 4))[0] == 4   # A survived (freshly used)
    assert cache.match(mkseq(300, 4))[0] == 4   # C survived (older than A)
    assert cache.match(mkseq(200, 4))[0] == 0   # B evicted (LRU)
    bm.assert_invariants()


def test_eviction_frees_blocks() -> None:
    """6. eviction frees blocks via block_manager.free; usage() reflects it."""
    bm, cache = make_cache()
    initial = bm.usage()
    cache_with(bm, cache, mkseq(100, 8), "r1")  # 2 blocks, engine ref released
    assert bm.usage()["used"] == 2              # held by the cache's fork
    released = cache.evict_until(2)
    assert released == 2
    usage = bm.usage()
    assert (usage["free"], usage["used"]) == (initial["free"], 0)
    bm.assert_invariants()


def test_reinsert_identical_no_double_consume() -> None:
    """7. re-insert identical sequence does not double-consume blocks."""
    bm, cache = make_cache()
    seq = mkseq(100, 8)
    table = cache_with(bm, cache, seq, "r1")
    first_usage = bm.usage()
    first_cached = cache.stats()["blocks_cached"]
    cache.insert(seq, table)  # same tokens, same physical table
    bm.assert_invariants()
    assert bm.usage()["free"] == first_usage["free"]
    assert bm.usage()["refcounted"] == first_usage["refcounted"]
    assert cache.stats()["blocks_cached"] == first_cached


def test_evict_until_zero_is_noop() -> None:
    """8. evict_until(0) is a no-op."""
    bm, cache = make_cache()
    cache_with(bm, cache, mkseq(100, 4), "r1")
    before = bm.usage(), cache.stats()
    assert cache.evict_until(0) == 0
    assert (bm.usage(), cache.stats()) == before
    bm.assert_invariants()


def test_stats_exact() -> None:
    """9. stats(): hits/queries/evictions counts are exact after a scripted
    sequence."""
    bm, cache = make_cache()
    cache_with(bm, cache, mkseq(100, 4), "rX")
    cache_with(bm, cache, mkseq(200, 4), "rY")
    cache.match(mkseq(100, 4))   # hit X
    cache.match(mkseq(777, 4))   # miss
    released = cache.evict_until(1)  # LRU leaf: Y (X was refreshed)
    assert released == 1
    assert cache.stats() == {"hits": 1, "queries": 2, "evictions": 1,
                             "blocks_cached": 1}  # X survives, Y was evicted
    assert cache.match(mkseq(200, 4)) == (0, [])     # Y gone (miss raises queries)
    assert cache.match(mkseq(100, 4))[0] == 4        # X still cached
    bm.assert_invariants()


def test_hit_rate_exact() -> None:
    """10. hit_rate == hits / queries; 0.0 when queries == 0."""
    bm, cache = make_cache()
    assert cache.hit_rate() == 0.0
    cache_with(bm, cache, mkseq(100, 4), "r1")
    cache.match(mkseq(100, 4))  # hit
    cache.match(mkseq(900, 4))  # miss
    assert cache.stats()["hits"] == 1 and cache.stats()["queries"] == 2
    assert cache.hit_rate() == 0.5
    bm.assert_invariants()


def test_previously_hit_misses_after_eviction() -> None:
    """11. after eviction, a previously-hit sequence misses."""
    bm, cache = make_cache()
    cache_with(bm, cache, mkseq(100, 4), "r1")
    assert cache.match(mkseq(100, 4))[0] == 4
    assert cache.evict_until(1) == 1
    assert cache.match(mkseq(100, 4)) == (0, [])
    bm.assert_invariants()


def test_leaf_to_root_cascade() -> None:
    """12. chained A -> A+B: eviction takes the leaf, then its parent."""
    bm, cache = make_cache()
    a, ab = mkseq(100, 4), mkseq(100, 8)
    ta = bm.allocate(1, "rA")
    cache.insert(a, ta)
    bm.free(ta, "rA")
    bm.assert_invariants()
    tab = list(bm.fork(ta)) + bm.allocate(1, "rB")
    cache.insert(ab, tab)
    bm.free(tab, "rB")
    bm.assert_invariants()
    released = cache.evict_until(2)
    assert released == 2                      # B's block, then A's block
    assert cache.stats()["evictions"] == 2    # leaf first, then cascade to root
    assert cache.stats()["blocks_cached"] == 0
    assert bm.usage()["free"] == 64
    bm.assert_invariants()


def test_partial_trailing_block_dropped_on_insert() -> None:
    """13. insert of a sequence ending mid-block caches only whole blocks."""
    bm, cache = make_cache()
    table = bm.allocate(2, "rX")  # engine prefilled 6 tokens = 2 blocks
    cache.insert(mkseq(50, 6), table)  # usable = 4 tokens -> 1 block cached
    bm.free(table, "rX")
    bm.assert_invariants()
    assert cache.stats()["blocks_cached"] == 1
    assert bm.usage()["used"] == 1           # second block returned to free list
    matched, blocks = cache.match(mkseq(50, 4))
    assert matched == 4 and len(blocks) == 1
    matched_extra, blocks_extra = cache.match(mkseq(50, 6))
    assert matched_extra == 4 and blocks_extra == [table[0]]  # trailing 2 tokens span an uncached block -> dropped
    bm.assert_invariants()

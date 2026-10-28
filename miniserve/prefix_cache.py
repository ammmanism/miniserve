"""miniserve.prefix_cache — token-level radix tree caching of KV block tables.

A compressed radix tree over token ids: each edge carries a token chunk
whose length is a multiple of ``block_size`` and each node pins the physical
block ids covering its chunk.  Every node takes ONE extra reference on its
blocks via ``BlockManager.fork``; eviction releases it via
``BlockManager.free``, so the real block manager's refcounts decide which
physical blocks actually return to the free list.

Documented choices:
- Hits are block granular.  ``match`` stops at the first block-level
  divergence; a trailing partial block of a cached chunk is never reported
  as a hit, so ``matched_len`` is always a multiple of ``block_size``.
- ``last_access`` is a strictly increasing logical clock (as float), not
  wall time: LRU eviction order is reproducible under tests.
- ``stats()["hits"]`` counts queries that matched at least one block, so
  ``hit_rate == hits / queries`` is well-behaved.
- ``stats()["evictions"]`` counts evicted tree nodes.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from miniserve.blocks import BlockManager

_CACHE_ID = "prefix-cache"  # holder id used for every cache-held refcount


class _Node:
    """Radix node.  ``children`` is keyed by each child chunk's first token."""

    __slots__ = ("children", "token_chunk", "block_ids", "last_access", "refcount")

    def __init__(
        self,
        token_chunk: Tuple[int, ...],
        block_ids: List[int],
        last_access: float,
    ) -> None:
        self.children: Dict[int, "_Node"] = {}
        self.token_chunk = token_chunk            # path-compressed tokens
        self.block_ids = list(block_ids)          # physical blocks for the chunk
        self.last_access = last_access
        self.refcount = 0                         # structural dependents (children)


class RadixPrefixCache:
    """Token-level prefix cache over paged KV block tables, LRU-evicted."""

    def __init__(self, block_manager: BlockManager, block_size: int = 16) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self._bm = block_manager
        self._block_size = block_size
        self._root = _Node((), [], 0.0)
        self._clock = 0
        # blocks_cached is maintained on node attach/evict, never guessed.
        self._stats = {"hits": 0, "queries": 0, "evictions": 0, "blocks_cached": 0}

    def _tick(self) -> float:
        """Advance the logical clock (float tick, LRU key)."""
        self._clock += 1
        return float(self._clock)

    def match(self, token_ids: List[int]) -> Tuple[int, List[int]]:
        """Return ``(matched_len, block_table_prefix)`` for the longest cached
        prefix whose blocks the query fully covers; a diverging trailing
        block is not counted.  Refreshes ``last_access`` along the hit path."""
        self._stats["queries"] += 1
        now = self._tick()
        node = self._root
        blocks: List[int] = []
        i = 0
        while i < len(token_ids):
            child = node.children.get(token_ids[i])
            if child is None:
                break
            chunk = child.token_chunk
            c = 0  # shared head length between the query remainder and chunk
            limit = min(len(token_ids) - i, len(chunk))
            while c < limit and token_ids[i + c] == chunk[c]:
                c += 1
            full_blocks = c // self._block_size  # only complete blocks hit
            if full_blocks:
                blocks = blocks + child.block_ids[:full_blocks]
                i += full_blocks * self._block_size
                node = child
                node.last_access = now
            if c < len(chunk):
                break  # diverged inside the chunk: stop without partials
        matched = len(blocks) * self._block_size
        if matched > 0:
            self._stats["hits"] += 1
        return matched, list(blocks)

    def insert(self, token_ids: List[int], block_table: List[int]) -> None:
        """Cache the block-aligned head of ``token_ids`` under ``block_table``.

        Only ``len(tokens) - len(tokens) % block_size`` tokens are cached; a
        partial trailing block is dropped (its physical block carries tokens
        the cache never saw).  New nodes fork their block slice so the cache
        holds its own reference; existing nodes along the path are shared
        structurally.  A prefix whose cached physical blocks differ from the
        incoming table (engine recomputed) is kept as-is, untouched.
        """
        usable = len(token_ids) - len(token_ids) % self._block_size
        if usable == 0 or not block_table:
            return
        blocks = block_table[: usable // self._block_size]
        node = self._root
        i = 0  # tokens consumed from token_ids
        b = 0  # blocks consumed from blocks
        while i < usable:
            rem = tuple(token_ids[i:usable])
            child = node.children.get(rem[0])
            if child is None:
                k = len(rem) // self._block_size
                new = _Node(rem, blocks[b : b + k], self._tick())
                self._bm.fork(new.block_ids)  # cache pins its own refcount
                node.children[new.token_chunk[0]] = new
                node.refcount += 1
                self._stats["blocks_cached"] += k
                return
            chunk = child.token_chunk
            c = 0  # shared head between the insertion remainder and the chunk
            limit = min(len(rem), len(chunk))
            while c < limit and rem[c] == chunk[c]:
                c += 1
            kb = c // self._block_size  # complete shared blocks
            if kb > 0 and child.block_ids[:kb] != blocks[b : b + kb]:
                return  # recomputed prefix; keep the cached physicals intact
            if intact := c - c % self._block_size:  # block-aligned shared head
                if intact == len(rem):
                    child.last_access = self._tick()  # re-insert / endpoint
                    return
                if c < len(chunk):
                    # Split at the block boundary so both sequences diverge
                    # as siblings under a shared parent node.
                    mid = _Node(chunk[:intact], child.block_ids[:kb], child.last_access)
                    tail = _Node(chunk[intact:], child.block_ids[kb:], child.last_access)
                    mid.children[tail.token_chunk[0]] = tail
                    mid.refcount = 1
                    node.children[rem[0]] = mid
                    node = mid
                else:
                    node = child
                i += intact
                b += kb
                continue
            return  # diverges inside the child's first block: uncacheable

    def _leaves(self, node: _Node, parent: "_Node | None", key: "int | None",
               out: List[Tuple[_Node, "_Node | None", "int | None"]]) -> None:
        if node.children:
            for k in node.children:
                self._leaves(node.children[k], node, k, out)
        elif node is not self._root:
            out.append((node, parent, key))

    def evict_until(self, free_blocks_needed: int) -> int:
        """Evict LRU leaf nodes (and cascade leaf-to-root) until at least
        ``free_blocks_needed`` physical blocks return to the free list.
        Returns the number of blocks actually released."""
        released = 0
        while released < free_blocks_needed:
            leaves: List[Tuple[_Node, "_Node | None", "int | None"]] = []
            self._leaves(self._root, self._root, None, leaves)
            if not leaves:
                break
            leaf = min(leaves, key=lambda t: t[0].last_access)
            node, parent, key = leaf
            before = self._bm.usage()["free"]
            self._bm.free(list(node.block_ids), _CACHE_ID)
            released += self._bm.usage()["free"] - before
            self._stats["evictions"] += 1
            self._stats["blocks_cached"] -= len(node.block_ids)
            if key is not None:
                del parent.children[key]  # type: ignore[index]
                parent.refcount -= 1  # structural dependent gone
        return released

    def stats(self) -> Dict[str, int]:
        """Accounting so far: hits, queries, evictions, blocks_cached."""
        return dict(self._stats)

    def hit_rate(self) -> float:
        """Fraction of queries that matched at least one cached block."""
        return self._stats["hits"] / self._stats["queries"] if self._stats["queries"] else 0.0

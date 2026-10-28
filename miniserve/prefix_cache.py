"""miniserve.prefix_cache — Automatic prefix/prompt caching for KV blocks.

Detects shared prefixes across requests and reuses previously computed
KV-cache blocks to skip redundant prefill computation.

TODO: Implement PrefixCache with trie-based lookup.
"""

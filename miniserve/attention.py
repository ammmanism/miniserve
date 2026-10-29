"""miniserve.attention -- SDPA-backed paged attention.

``PagedAttention`` computes attention directly over a paged KV cache using
``torch.nn.functional.scaled_dot_product_attention``:

* block-table gather -- logical positions map through ``block_table`` into
  physical blocks of shape [num_blocks, block_size, num_kv_heads, head_dim];
* ragged batch -- one ``context_lens`` entry per row; shorter rows are
  zero-padded and masked out;
* GQA/MQA -- when ``num_kv_heads < num_heads`` the cache holds fewer K/V
  heads and they are expanded with ``repeat_interleave`` at attention time
  (the FlashAttention-2 GQA path);
* sliding window -- ``window`` keeps only the last ``window`` key positions
  per query, ANDed with the causal mask;
* FP8 KV cache -- ``kv_dtype=torch.float8_e4m3fn`` stores the cache in FP8
  and dequantizes to the query dtype immediately before SDPA.  FP8 is
  lossy by design: expect ~1e-2..5e-2 absolute output error versus an FP16
  cache.  This is a storage-bandwidth trade, not a correctness guarantee;
* RoPE -- ``apply_rope`` is the fused rotary interface; Part 9 swaps the
  body for a Triton kernel behind the same signature.

Query layout is [batch, q_len, num_heads, head_dim].  The q_len query
tokens are the trailing positions of each row's context (a decode step is
q_len=1); prompts are written into the cache first with ``write_kv``.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[x1, x2] -> [-x2, x1] along the last axis (HF rotate_half)."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fused rotary embedding; Part 9 swaps the body for one Triton kernel.

    ``q`` is [batch, heads, seq, head_dim] and ``k`` (optional) is
    [batch, kv_heads, seq, head_dim]; ``cos``/``sin`` are [max_pos, head_dim]
    and ``position_ids`` is [seq] or [batch, seq].  Returns
    ``(q_rot, k_rot | None)`` with the same shapes and dtype as the inputs.
    """
    cos = cos[position_ids].to(q.dtype)
    sin = sin[position_ids].to(q.dtype)
    if cos.ndim == 2:  # [seq, d]: shared by every batch row
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)  # [batch, 1, seq, d]
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = None if k is None else k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


class PagedAttention:
    """SDPA paged attention over a block-table-mapped KV cache."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        *,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
        window: int | None = None,
        kv_dtype: torch.dtype = torch.float16,
    ) -> None:
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_kv_heads <= 0 or num_heads % num_kv_heads:
            raise ValueError("num_heads must be a multiple of num_kv_heads")
        if head_dim is None or head_dim <= 0 or head_dim % 2:
            raise ValueError("head_dim must be a positive even integer")
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        if window is not None and window <= 0:
            raise ValueError("window must be positive")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.dtype = dtype
        self.device = torch.device(device)
        self.window = window
        self.kv_dtype = kv_dtype
        self.group = num_heads // num_kv_heads

    def allocate_kv_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate (k_cache, v_cache) shaped [blocks, block_size, kv_heads, dim]."""
        shape = (self.num_blocks, self.block_size, self.num_kv_heads, self.head_dim)
        k = torch.zeros(shape, dtype=self.kv_dtype, device=self.device)
        v = torch.zeros(shape, dtype=self.kv_dtype, device=self.device)
        return k, v

    def write_kv(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_table: torch.Tensor,
        start: int | Sequence[int] = 0,
    ) -> None:
        """Scatter new k/v [batch, tokens, kv_heads, head_dim] into the cache.

        ``start`` is the absolute position of each row's first new token
        (an int, or one entry per batch row).  A trailing partial block is
        written in place; paged layout is always token-addressed so the
        block-table gather never needs a contiguous tail block.
        """
        starts = start if isinstance(start, Sequence) else [start] * k.shape[0]
        for b in range(k.shape[0]):
            for t in range(k.shape[1]):
                pos = int(starts[b]) + t
                phys = int(block_table[b, pos // self.block_size])
                k_cache[phys, pos % self.block_size] = k[b, t].to(k_cache.dtype)
                v_cache[phys, pos % self.block_size] = v[b, t].to(v_cache.dtype)

    def build_mask(
        self,
        context_lens: torch.Tensor | Sequence[int],
        q_len: int,
        window: int | None = None,
        causal: bool = True,
    ) -> torch.Tensor:
        """Boolean keep-mask [batch, q_len, max_ctx] (True = attend)."""
        lens = torch.as_tensor(context_lens, device=self.device).reshape(-1)
        max_ctx = int(lens.max()) if lens.numel() else 0
        key_pos = torch.arange(max_ctx, device=self.device)[None, None, :]
        q_pos = lens[:, None] - q_len + torch.arange(q_len, device=self.device)[None, :]
        mask = torch.ones(lens.numel(), q_len, max_ctx, dtype=torch.bool, device=self.device)
        if causal:
            mask &= key_pos <= q_pos[:, :, None]
        mask &= key_pos < lens[:, None, None]
        if window is not None:
            mask &= key_pos > (q_pos - window)[:, :, None]
        return mask

    def _gather(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        lens: torch.Tensor,
        max_ctx: int,
    ) -> torch.Tensor:
        """Gather [batch, max_ctx, kv_heads, head_dim] from the paged cache."""
        pos = torch.arange(max_ctx, device=self.device)
        blk = block_table[:, pos // self.block_size]
        off = pos % self.block_size
        valid = pos[None, :] < lens[:, None]
        blk = torch.where(valid, blk, torch.zeros_like(blk))
        return cache[blk, off]

    def forward(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        context_lens: torch.Tensor | Sequence[int],
        causal: bool = True,
        window: int | None = None,
    ) -> torch.Tensor:
        """Paged SDPA.  ``window`` overrides the constructor value per call."""
        window = self.window if window is None else window
        lens = torch.as_tensor(context_lens, device=self.device).reshape(-1)
        q_len = query.shape[1]
        max_ctx = int(lens.max())
        block_table = torch.as_tensor(block_table, device=self.device, dtype=torch.long)
        if block_table.ndim == 1:
            block_table = block_table[None, :]
        k = self._gather(k_cache, block_table, lens, max_ctx).to(query.dtype)
        v = self._gather(v_cache, block_table, lens, max_ctx).to(query.dtype)
        q = query.transpose(1, 2)  # SDPA layout: [batch, heads, seq, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.group > 1:  # GQA/MQA: one K/V head serves `group` query heads
            k = k.repeat_interleave(self.group, dim=1)
            v = v.repeat_interleave(self.group, dim=1)
        mask = self.build_mask(lens, q_len, window, causal)[:, None]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        return out.transpose(1, 2)

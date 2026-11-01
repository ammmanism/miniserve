"""kernels.paged_attn_triton -- split-K flash decoding with fused RoPE (Part 9 B).

Three layers, deliberately separated:

1. **Triton kernels** (CUDA only): ``_split_kernel`` computes per-split
   partials (max, sumexp, weighted V) with ``tl.dot`` tensor-core tiling
   (BLOCK_M=16 query heads, BLOCK_N=64 keys, fp16 in / fp32 accumulate);
   ``_merge_kernel`` combines split partials with a log-sum-exp merge.
2. **Torch reference** (any device): ``flash_decode_reference`` implements the
   same split decomposition, masking and merge in plain torch.  This is what
   the CPU test suite checks against SDPA, so the *algorithm* is verified
   even where Triton cannot launch.
3. **Dispatcher**: ``flash_decode`` picks the Triton path on CUDA and raises
   a clear error elsewhere (no silent fallback -- a "Triton" number must
   never come from a PyTorch path).

Layout note: the M axis packs the ``GROUP = num_heads / num_kv_heads`` query
heads that share one KV head, so a single K/V tile feeds the whole ``tl.dot``
tile (the FlashDecoding GQA packing).  Rows beyond GROUP are masked garbage
and never stored.

Autotune configurations are persisted in ``kernels/tuned_configs.json`` and
loaded at import; tests never trigger a tuning run.  Override the path with
``MINISERVE_TRITON_CONFIGS``.

Honesty: developed on a CPU-only host.  The Triton launch path could not be
executed here; it is parity-tested only when CUDA + triton are present
(``tests/test_triton_parity.py`` skips otherwise, loudly).  The reference
algorithm is fully verified on CPU.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Sequence

import torch

try:  # Triton ships CUDA/ROCm wheels only; the import must survive on CPU.
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on CPU-only hosts
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    TRITON_AVAILABLE = False

BLOCK_M = 16
BLOCK_N = 64
MAX_SPLITS = 8
MIN_SPLIT_CTX = 2048

_DEFAULT_CONFIGS = [
    {"block_size": 16, "num_warps": 4, "num_stages": 2},
    {"block_size": 32, "num_warps": 8, "num_stages": 3},
]


def load_tuned_configs(path: Optional[str] = None) -> List[dict]:
    """Read persisted autotune configs (never tune during tests)."""
    p = Path(path or os.getenv("MINISERVE_TRITON_CONFIGS")
             or Path(__file__).with_name("tuned_configs.json"))
    if not p.exists():
        return [dict(c) for c in _DEFAULT_CONFIGS]
    data = json.loads(p.read_text())
    configs = data.get("configs", data)
    out = []
    for c in configs:
        out.append({"block_size": int(c.get("block_size", 16)),
                    "num_warps": int(c.get("num_warps", 4)),
                    "num_stages": int(c.get("num_stages", 2))})
    return out or [dict(c) for c in _DEFAULT_CONFIGS]


TUNED_CONFIGS = load_tuned_configs()


def num_splits_for(context_len: int) -> int:
    """Split only when the context is long enough to amortize the merge."""
    if context_len <= MIN_SPLIT_CTX:
        return 1
    return max(1, min(MAX_SPLITS, (context_len + MIN_SPLIT_CTX - 1) // MIN_SPLIT_CTX))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _gather_paged(
    cache: torch.Tensor, table: torch.Tensor, length: int
) -> torch.Tensor:
    """[L, KVH, D] rows for one sequence through its block table."""
    bs = cache.shape[1]
    pos = torch.arange(length, device=table.device)
    idx = table[pos // bs] * bs + pos % bs
    return cache.reshape(-1, cache.shape[2], cache.shape[3]).index_select(0, idx)


def flash_decode_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: Sequence[int] | torch.Tensor,
    *,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    position: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    num_splits: Optional[int] = None,
) -> torch.Tensor:
    """Split-K decode in pure torch: [B,1,H,D] -> [B,1,H,D].

    Caches are [blocks, block_size, KVH, D].  When ``cos``/``sin``
    ([max_pos, D]) and ``position`` ([B] or [B,1]) are given, RoPE is fused
    into the Q path (the KV cache is assumed to hold rotated keys already,
    as written by the model runner).
    """
    if q.shape[1] != 1:
        raise ValueError("flash decode expects q_len == 1")
    B, _, H, D = q.shape
    KVH = k_cache.shape[2]
    if H % KVH:
        raise ValueError("num_heads must be a multiple of num_kv_heads")
    group = H // KVH
    lens = torch.as_tensor(context_lens).reshape(-1).to(torch.long)
    table = torch.as_tensor(block_table, dtype=torch.long)
    if table.ndim == 1:
        table = table[None, :]
    scale = softmax_scale if softmax_scale is not None else (1.0 / (D ** 0.5))
    qf = q.reshape(B, H, D).to(torch.float32)
    if cos is not None and sin is not None and position is not None:
        pos = torch.as_tensor(position).reshape(-1).to(torch.long)[:B]
        cs = cos.to(q.device).index_select(0, pos).to(torch.float32)
        sn = sin.to(q.device).index_select(0, pos).to(torch.float32)
        qf = qf * cs[:, None, :] + _rotate_half(qf) * sn[:, None, :]
    out = torch.zeros(B, 1, H, D, dtype=torch.float32, device=q.device)
    for b in range(B):
        L = int(lens[b])
        if L <= 0:
            continue
        k = _gather_paged(k_cache, table[b], L).to(torch.float32)
        v = _gather_paged(v_cache, table[b], L).to(torch.float32)
        if group > 1:
            k = k.repeat_interleave(group, dim=1)
            v = v.repeat_interleave(group, dim=1)
        splits = max(1, min(num_splits or num_splits_for(L), L))
        per = (L + splits - 1) // splits
        best_m = torch.full((H,), float("-inf"), device=q.device)
        best_acc = torch.zeros(H, D, dtype=torch.float32, device=q.device)
        best_l = torch.zeros(H, dtype=torch.float32, device=q.device)
        for s in range(splits):
            lo, hi = s * per, min(L, (s + 1) * per)
            if hi <= lo:
                continue
            scores = torch.einsum("hd,nhd->hn", qf[b], k[lo:hi]) * scale
            m_s = scores.max(dim=-1).values
            p = torch.exp(scores - m_s[:, None])
            l_s = p.sum(dim=-1)
            m = torch.maximum(best_m, m_s)
            alpha = torch.exp(best_m - m)
            beta = torch.exp(m_s - m)
            acc_s = torch.einsum("hn,nhd->hd", p, v[lo:hi])
            best_acc = best_acc * alpha[:, None] + acc_s * beta[:, None]
            best_l = best_l * alpha + l_s * beta
            best_m = m
        out[b, 0] = best_acc / best_l[:, None]
    return out.to(q.dtype)


if TRITON_AVAILABLE:  # pragma: no cover - requires CUDA + triton

    @triton.autotune(configs=TUNED_CONFIGS, key=["HEAD_DIM", "BLOCK_SIZE"])
    @triton.jit
    def _split_kernel(
        q_ptr, k_ptr, v_ptr, bt_ptr, ctx_ptr,
        cos_ptr, sin_ptr, pos_ptr,
        acc_ptr, m_ptr, l_ptr,
        stride_qb, stride_qh,
        stride_kb, stride_ks, stride_kh,
        stride_btb, stride_p,
        GROUP: tl.constexpr, HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr, NUM_SPLITS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        USE_ROPE: tl.constexpr, SCALE: tl.constexpr,
    ):
        """Partials for one (sequence, kv head, split).

        Grid: (num_kv_heads * num_splits, batch).  M packs the GROUP query
        heads sharing this KV head; N walks the split's key range.
        """
        pid = tl.program_id(0)
        b = tl.program_id(1)
        NUM_KV_HEADS = tl.num_programs(0) // NUM_SPLITS
        kvh = pid // NUM_SPLITS
        split = pid % NUM_SPLITS
        ctx = tl.load(ctx_ptr + b)
        per = (ctx + NUM_SPLITS - 1) // NUM_SPLITS
        lo = split * per
        hi = tl.minimum(ctx, lo + per)

        rows = tl.arange(0, BLOCK_M)
        row_mask = rows < GROUP
        head = kvh * GROUP + rows
        dims = tl.arange(0, HEAD_DIM)
        q = tl.load(q_ptr + b * stride_qb + head[:, None] * stride_qh
                    + dims[None, :], mask=row_mask[:, None], other=0.0)
        if USE_ROPE:
            pos = tl.load(pos_ptr + b)
            c = tl.load(cos_ptr + pos * HEAD_DIM + dims)
            s = tl.load(sin_ptr + pos * HEAD_DIM + dims)
            half = HEAD_DIM // 2
            shifted = (dims + half) % HEAD_DIM
            q_shift = tl.load(q_ptr + b * stride_qb + head[:, None] * stride_qh
                              + shifted[None, :], mask=row_mask[:, None], other=0.0)
            rot = tl.where(dims[None, :] < half, -q_shift, q_shift)
            q = q * c[None, :] + rot * s[None, :]

        acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for start in range(lo, hi, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            keep = offs < hi
            blk = tl.load(bt_ptr + b * stride_btb + offs // BLOCK_SIZE,
                          mask=offs < ctx, other=-1)
            safe = tl.where(blk < 0, 0, blk)
            phys = safe * stride_kb + (offs % BLOCK_SIZE) * stride_ks
            k = tl.load(k_ptr + phys[:, None] + kvh * stride_kh
                        + dims[None, :], mask=keep[:, None], other=0.0)
            scores = tl.dot(q, tl.trans(k)) * SCALE
            scores = tl.where(keep[None, :], scores, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(scores, 1))
            p = tl.exp(scores - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(v_ptr + phys[:, None] + kvh * stride_kh
                        + dims[None, :], mask=keep[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
        base = (pid * stride_p + b) * BLOCK_M
        tl.store(m_ptr + base + rows, m_i, mask=row_mask)
        tl.store(l_ptr + base + rows, l_i, mask=row_mask)
        tl.store(acc_ptr + base * HEAD_DIM + rows[:, None] * HEAD_DIM
                 + dims[None, :], acc, mask=row_mask[:, None])

    @triton.jit
    def _merge_kernel(acc_ptr, m_ptr, l_ptr, out_ptr,
                      stride_p, stride_qh,
                      NUM_SPLITS: tl.constexpr, GROUP: tl.constexpr,
                      HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr):
        """Log-sum-exp merge of one sequence's (kv head, split) partials."""
        kvh = tl.program_id(0)
        b = tl.program_id(1)
        NUM_KV_HEADS = tl.num_programs(0)
        rows = tl.arange(0, BLOCK_M)
        row_mask = rows < GROUP
        dims = tl.arange(0, HEAD_DIM)

        m = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        for s in range(NUM_SPLITS):
            pid = kvh * NUM_SPLITS + s
            m = tl.maximum(m, tl.load(m_ptr + (pid * stride_p + b) * BLOCK_M + rows,
                                      mask=row_mask, other=float("-inf")))
        acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
        denom = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for s in range(NUM_SPLITS):
            pid = kvh * NUM_SPLITS + s
            base = (pid * stride_p + b) * BLOCK_M
            m_s = tl.load(m_ptr + base + rows, mask=row_mask, other=0.0)
            l_s = tl.load(l_ptr + base + rows, mask=row_mask, other=0.0)
            a_s = tl.load(acc_ptr + base * HEAD_DIM + rows[:, None] * HEAD_DIM
                          + dims[None, :], mask=row_mask[:, None], other=0.0)
            beta = tl.exp(m_s - m)
            acc += a_s * beta[:, None]
            denom += l_s * beta
        out = acc / denom[:, None]
        head = kvh * GROUP + rows
        tl.store(out_ptr + b * (NUM_KV_HEADS * GROUP) * stride_qh
                 + head[:, None] * stride_qh + dims[None, :], out,
                 mask=row_mask[:, None])

    def flash_decode(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        context_lens: Sequence[int] | torch.Tensor,
        *,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        position: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        num_splits: Optional[int] = None,
    ) -> torch.Tensor:
        """Launch split-K flash decoding (CUDA + Triton only)."""
        if q.shape[1] != 1:
            raise ValueError("flash decode expects q_len == 1")
        if not q.is_cuda:
            raise RuntimeError("triton_flash requires CUDA tensors")
        B, _, H, D = q.shape
        KVH = k_cache.shape[2]
        if H % KVH:
            raise ValueError("num_heads must be a multiple of num_kv_heads")
        group = H // KVH
        lens = torch.as_tensor(context_lens, device=q.device).reshape(-1)
        table = torch.as_tensor(block_table, device=q.device, dtype=torch.long)
        if table.ndim == 1:
            table = table[None, :]
        splits = max(1, min(num_splits or num_splits_for(int(lens.max())),
                            max(1, int(lens.min()))))
        scale = softmax_scale if softmax_scale is not None else (1.0 / (D ** 0.5))
        q3 = q.reshape(B, H, D).contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        nprog = KVH * splits
        acc = torch.empty(nprog * B * BLOCK_M * D, dtype=torch.float32, device=q.device)
        ms = torch.empty(nprog * B * BLOCK_M, dtype=torch.float32, device=q.device)
        ls = torch.empty(nprog * B * BLOCK_M, dtype=torch.float32, device=q.device)
        out = torch.empty(B, H * D, dtype=q.dtype, device=q.device)
        dummy = torch.zeros(1, D, device=q.device, dtype=q.dtype)
        cos_t = cos if cos is not None else dummy
        sin_t = sin if sin is not None else dummy
        pos_t = (torch.as_tensor(position, device=q.device).reshape(-1) if position is not None
                 else torch.zeros(B, dtype=torch.long, device=q.device))
        _split_kernel[(nprog, B)](
            q3, k_cache, v_cache, table, lens,
            cos_t, sin_t, pos_t,
            acc, ms, ls,
            q3.stride(0), q3.stride(1),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            table.stride(0), B,
            GROUP=group, HEAD_DIM=D,
            BLOCK_SIZE=k_cache.shape[1], NUM_SPLITS=splits,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            USE_ROPE=cos is not None, SCALE=scale,
        )
        _merge_kernel[(KVH, B)](acc, ms, ls, out, B, D,
                                NUM_SPLITS=splits, GROUP=group,
                                HEAD_DIM=D, BLOCK_M=BLOCK_M)
        return out.view(B, 1, H, D)
else:  # pragma: no cover - CPU-only host

    def flash_decode(*args, **kwargs) -> torch.Tensor:
        raise RuntimeError(
            "triton_flash requires CUDA and the triton package; "
            "use flash_decode_reference for the CPU algorithm path"
        )

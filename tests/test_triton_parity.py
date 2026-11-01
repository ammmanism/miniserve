"""Parity tests for the Part 9 split-K flash-decode kernel.

Two tiers:

* **CPU (always runs)**: the Triton kernel's *algorithm* -- split decomposition,
  masked paged gather, log-sum-exp merge, fused RoPE -- is implemented in
  pure torch (``flash_decode_reference``) and checked against SDPA here.
* **CUDA (skipped loudly elsewhere)**: the actual Triton launch is compared to
  the same reference.  A skipped Triton launch is reported as a skip, never
  as a pass, so no GPU number is implied by a green CPU run.
"""
import math

import pytest
import torch

from kernels.paged_attn_triton import (
    MAX_SPLITS,
    TRITON_AVAILABLE,
    flash_decode_reference,
    load_tuned_configs,
    num_splits_for,
)
from miniserve.attention import PagedAttention, apply_rope

ATOL = 5e-3
ROPE_ATOL = 1e-4


def _case(B, H, KVH, D, ctx, block_size=16, seed=0, device="cpu"):
    """Build a paged decode case plus its SDPA reference output."""
    g = torch.Generator().manual_seed(seed)
    nb = (ctx + block_size - 1) // block_size
    attn = PagedAttention(H, KVH, D, num_blocks=nb * B, block_size=block_size,
                          dtype=torch.float32, device=device)
    k_cache, v_cache = attn.allocate_kv_cache()
    q = torch.randn(B, 1, H, D, generator=g)
    k = torch.randn(B, ctx, KVH, D, generator=g)
    v = torch.randn(B, ctx, KVH, D, generator=g)
    table = torch.stack([torch.arange(nb) + i * nb for i in range(B)])
    attn.write_kv(k_cache, v_cache, k, v, table, start=0)
    expected = attn.forward(q, k_cache, v_cache, table, [ctx] * B)
    return q, k_cache, v_cache, table, expected


def _rope_tables(ctx, D, seed=1):
    g = torch.Generator().manual_seed(seed)
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    pos = torch.arange(ctx + 1, dtype=torch.float32)
    freqs = torch.outer(pos, inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    return torch.cos(emb), torch.sin(emb)


@pytest.mark.parametrize("H,KVH,D,ctx", [(4, 4, 16, 1), (8, 2, 32, 300), (4, 1, 16, 77)])
def test_reference_matches_sdpa(H, KVH, D, ctx):
    q, k, v, table, expected = _case(1, H, KVH, D, ctx)
    got = flash_decode_reference(q, k, v, table, [ctx])
    torch.testing.assert_close(got, expected, atol=ATOL, rtol=0)


def test_split_k_long_context():
    ctx = 4096
    q, k, v, table, expected = _case(1, 8, 2, 64, ctx, block_size=32)
    got = flash_decode_reference(q, k, v, table, [ctx], num_splits=MAX_SPLITS)
    assert num_splits_for(ctx) == 2
    assert num_splits_for(16384) == MAX_SPLITS
    torch.testing.assert_close(got, expected, atol=ATOL, rtol=0)


def test_fused_rope_matches_unfused():
    ctx, D = 512, 64
    q, k, v, table, _ = _case(1, 4, 2, D, ctx)
    cos, sin = _rope_tables(ctx, D)
    pos = torch.tensor([ctx - 1])
    q_rot, _ = apply_rope(q.transpose(1, 2), None, cos, sin, pos.reshape(1, 1))
    attn = PagedAttention(4, 2, D, num_blocks=table.shape[1], block_size=16,
                          dtype=torch.float32)
    expected = attn.forward(q_rot.transpose(1, 2), k, v, table, [ctx])
    fused = flash_decode_reference(q, k, v, table, [ctx], cos=cos, sin=sin, position=pos)
    torch.testing.assert_close(fused, expected, atol=ROPE_ATOL, rtol=0)


def test_num_splits_policy():
    assert num_splits_for(1) == 1
    assert num_splits_for(2048) == 1
    assert num_splits_for(2049) == 2
    assert num_splits_for(100000) == MAX_SPLITS


def test_tuned_configs_are_persisted():
    configs = load_tuned_configs()
    assert len(configs) >= 2
    assert {c["block_size"] for c in configs} <= {16, 32}
    for c in configs:
        assert c["num_warps"] in (4, 8) and c["num_stages"] in (2, 3)


def test_ragged_batch():
    q, k, v, table, _ = _case(3, 4, 2, 16, 64)
    lens = [64, 17, 33]
    attn = PagedAttention(4, 2, 16, num_blocks=table.shape[1], block_size=16,
                          dtype=torch.float32)
    q = torch.randn(3, 1, 4, 16)
    expected = attn.forward(q, k, v, table, lens)
    got = flash_decode_reference(q, k, v, table, lens)
    torch.testing.assert_close(got, expected, atol=ATOL, rtol=0)


def test_shape_fuzz_50():
    for i in range(50):
        gen = torch.Generator().manual_seed(1000 + i)
        B = int(torch.randint(1, 4, (1,), generator=gen))
        KVH = int(torch.randint(1, 3, (1,), generator=gen))
        factor = int(torch.randint(1, 5, (1,), generator=gen))
        H, D = KVH * factor, int(torch.randint(2, 5, (1,), generator=gen)) * 8
        ctx = int(torch.randint(1, 65, (1,), generator=gen))
        bs = (8, 16, 32)[int(torch.randint(0, 3, (1,), generator=gen))]
        q, k, v, table, expected = _case(B, H, KVH, D, ctx, block_size=bs, seed=i)
        got = flash_decode_reference(q, k, v, table, [ctx] * B)
        torch.testing.assert_close(got, expected, atol=ATOL, rtol=0,
                                   msg=f"shape #{i}: B={B} H={H} KVH={KVH} D={D} "
                                       f"ctx={ctx} bs={bs}")


@pytest.mark.skipif(torch.cuda.is_available(), reason="CPU-only error path")
def test_triton_backend_fails_loudly_without_cuda():
    q, k, v, table, _ = _case(1, 4, 4, 16, 8)
    attn = PagedAttention(4, 4, 16, num_blocks=table.shape[1], block_size=16,
                          dtype=torch.float32)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        attn.forward(q, k, v, table, [8], backend="triton_flash")


def test_triton_backend_validates_contract():
    q, k, v, table, _ = _case(1, 4, 4, 16, 8)
    attn = PagedAttention(4, 4, 16, num_blocks=table.shape[1], block_size=16,
                          dtype=torch.float32)
    with pytest.raises(ValueError, match="decode-only"):
        attn.forward(torch.randn(1, 2, 4, 16), k, v, table, [8],
                     backend="triton_flash")
    with pytest.raises(ValueError, match="sliding window"):
        attn.forward(q, k, v, table, [8], window=4, backend="triton_flash")
    with pytest.raises(ValueError, match="unknown attention backend"):
        attn.forward(q, k, v, table, [8], backend="nope")


needs_triton = pytest.mark.skipif(
    not (TRITON_AVAILABLE and torch.cuda.is_available()),
    reason="needs CUDA + triton; the Triton launch is UNVERIFIED on this host",
)


@needs_triton
@pytest.mark.parametrize("H,KVH,D,ctx", [(4, 4, 16, 64), (8, 2, 64, 4096)])
def test_triton_kernel_matches_reference(H, KVH, D, ctx):
    q, k, v, table, _ = _case(2, H, KVH, D, ctx, block_size=32, device="cuda")
    got = PagedAttention(H, KVH, D, num_blocks=table.shape[1], block_size=32,
                         dtype=torch.float32, device="cuda").forward(
        q, k, v, table, [ctx] * 2, backend="triton_flash")
    ref = flash_decode_reference(q, k, v, table, [ctx] * 2)
    torch.testing.assert_close(got, ref, atol=ATOL, rtol=0)


@needs_triton
def test_triton_kernel_fused_rope():
    ctx, D = 300, 64
    q, k, v, table, _ = _case(1, 4, 2, D, ctx, device="cuda")
    cos, sin = _rope_tables(ctx, D)
    pos = torch.tensor([ctx - 1], device="cuda")
    fused = PagedAttention(4, 2, D, num_blocks=table.shape[1], block_size=16,
                           dtype=torch.float32, device="cuda").forward(
        q, k, v, table, [ctx], backend="triton_flash",
        cos=cos.to("cuda"), sin=sin.to("cuda"), position=pos)
    q_rot, _ = apply_rope(q.transpose(1, 2), None, cos, sin, pos.reshape(1, 1))
    ref = flash_decode_reference(q_rot.transpose(1, 2), k, v, table, [ctx])
    torch.testing.assert_close(fused, ref, atol=ROPE_ATOL, rtol=0)

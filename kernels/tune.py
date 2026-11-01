"""Tune the split-K flash-decode kernel on a GPU and persist the winners.

Run on a CUDA machine::

    python kernels/tune.py --head-dim 64 --block-size 16 --block-size 32

The output overwrites ``kernels/tuned_configs.json`` with the measured
best-config-per-(head_dim, block_size) matrix used by
``kernels/paged_attn_triton.py``.  Nothing in this file runs during tests;
the authoring host has no GPU, so the committed JSON is a *declared* space,
not a measured one (see its ``measured`` flag).
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time
from pathlib import Path

import torch

from kernels.paged_attn_triton import BLOCK_N, _split_kernel, flash_decode


def _case(ctx: int, H: int, KVH: int, D: int, block_size: int, device="cuda"):
    nb = (ctx + block_size - 1) // block_size
    q = torch.randn(1, 1, H, D, device=device, dtype=torch.float16)
    k = torch.randn(nb, block_size, KVH, D, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    table = torch.arange(nb, device=device).reshape(1, -1)
    return q, k, v, table, [ctx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-dim", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--block-size", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--num-warps", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--num-stages", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default=str(Path(__file__).with_name("tuned_configs.json")))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("tuning requires a CUDA GPU")
    winners = []
    for D, bs in itertools.product(args.head_dim, args.block_size):
        q, k, v, table, lens = _case(args.ctx, 8, 2, D, bs)
        best = None
        for warps, stages in itertools.product(args.num_warps, args.num_stages):
            _split_kernel.configs = [triton_config(bs, warps, stages)]
            times = []
            for _ in range(args.iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                flash_decode(q, k, v, table, lens)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            ms = statistics.median(times) * 1e3
            print(f"head_dim={D} block_size={bs} warps={warps} stages={stages}: {ms:.3f} ms")
            if best is None or ms < best[0]:
                best = (ms, warps, stages)
        winners.append({"head_dim": D, "block_size": bs,
                        "num_warps": best[1], "num_stages": best[2],
                        "median_ms": round(best[0], 4)})
    Path(args.out).write_text(json.dumps(
        {"measured": True, "key": ["head_dim", "block_size"],
         "device": torch.cuda.get_device_name(0), "configs": winners}, indent=2))
    print(f"wrote {args.out}")
    return 0


def triton_config(block_size, num_warps, num_stages):
    import triton
    return triton.Config({"block_size": block_size}, num_warps=num_warps,
                         num_stages=num_stages)


if __name__ == "__main__":
    raise SystemExit(main())

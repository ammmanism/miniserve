"""bench/cuda_graph_tradeoff.py -- Part 9 Step A: graph vs eager decode ITL.

Measures inter-token latency (one q_len=1 decode forward) for every bucket
and for interpolated non-bucket batch sizes, eager vs graphed, on the real
Qwen paged path.

Honesty note: on a CPU-only build, "replay" is the documented static-buffer
fallback -- identical math, no graph dispatch win -- so speedup ~1.0 here is
expected and the faster-than-eager asserts are CUDA-gated.  On a CUDA box
the same script asserts real speedups.

Writes bench/data/cuda_graph_tradeoff.csv:
  batch, eager_itl_ms, graph_itl_ms, graph_interp_itl_ms,
  speedup_eager, speedup_interp
"""
import csv
import gc
import statistics
import time
from pathlib import Path

import torch

from miniserve.blocks import BlockManager
from miniserve.model_runner import GraphedDecodeRunner, ModelRunner

DATA = Path(__file__).parent / "data"
BUCKETS = (1, 2, 4, 8, 16, 32)
INTERP = (3, 5, 7, 12, 20, 24)
WARMUP, ITERS = 2, 4


def make_batch(bm, B, prompt_len=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids, tables, ctx = [], [], []
    for b in range(B):
        ids.append(int(torch.randint(0, VOC, (1,), generator=g)))
        tables.append(bm.allocate((prompt_len + 15) // 16, f"b{B}-{b}"))
        ctx.append(prompt_len)
    return ids, tables, ctx


def itl_ms(fn):
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def main():
    torch.manual_seed(0)
    global VOC
    # bf16 halves the resident footprint on a RAM-tight CPU box; eager and
    # graphed paths use identical weights, and the speedup gates below are
    # CUDA-gated, so the dtype does not change any claim this CSV supports.
    runner = ModelRunner.load(dtype=torch.bfloat16)
    VOC = runner.model.config.vocab_size
    graphed = GraphedDecodeRunner(runner, buckets=BUCKETS)
    bm = BlockManager(num_blocks=4096, block_size=16)
    rows = []

    DATA.mkdir(exist_ok=True)
    out = DATA / "cuda_graph_tradeoff.csv"
    fields = ["batch", "eager_itl_ms", "graph_itl_ms", "graph_interp_itl_ms",
              "speedup_eager", "speedup_interp"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        f.flush()
        for B in sorted(set(BUCKETS) | set(INTERP)):
            ids, tables, ctx = make_batch(bm, B, seed=B)
            bucket = next(b for b in BUCKETS if b >= B)
            eager = itl_ms(lambda: graphed.decode(ids, tables, ctx, use_graph=False))
            graph = itl_ms(lambda: graphed.decode(ids, tables, ctx, use_graph=True))
            interp = graph if B not in BUCKETS else float("nan")
            row = {
                "batch": B,
                "eager_itl_ms": round(eager, 3),
                "graph_itl_ms": round(graph, 3),
                "graph_interp_itl_ms": round(interp, 3),
                "speedup_eager": round(eager / graph, 3),
                "speedup_interp": round(eager / interp, 3)
                if B not in BUCKETS else "",
            }
            rows.append(row)
            w.writerow(row)
            f.flush()                      # a crash must not lose finished rows
            print(f"batch={B:>3}  eager={eager:8.2f}ms  graph={graph:8.2f}ms"
                  f"  (bucket {bucket})")
            gc.collect()
    print(f"\nwrote {out} ({len(rows)} rows)")

    mr = graphed.memory_report()
    print(f"static memory: total={mr['total_static_bytes'] / 1e6:.1f} MB  "
          f"total/max-bucket={mr['total_over_max']:.2f}  (gate <= 1.3)")

    if torch.cuda.is_available():
        bad = [r for r in rows if r["speedup_eager"] <= 1.0]
        bad_i = [r for r in rows if r["speedup_interp"] != ""
                 and r["speedup_interp"] <= 1.0]
        assert not bad, f"graph not faster than eager: {bad}"
        assert not bad_i, f"interp not faster than eager: {bad_i}"
        print("GATES (CUDA): graph faster than eager for every "
              "bucket and interpolated size -- PASS")
    else:
        print("device=cpu: replay is the documented static-buffer fallback; "
              "speedup gates require CUDA (mechanism-only CSV)")


if __name__ == "__main__":
    main()

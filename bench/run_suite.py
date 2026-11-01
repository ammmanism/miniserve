"""bench.run_suite -- benchmark orchestration and CSV result writer (Part 10).

Three modes, one file:

  engine (default)  Real paged path in-process: sequential prefill, batched
                    decode through GraphedDecodeRunner.  Every latency is
                    measured; nothing is modeled.  Writes bench/data/
                    all_results.csv and bench/data/roofline.csv.
  --url URL         Async HTTP/SSE sweep against a running server via
                    bench.client.BenchClient (per-token timestamps).
  --microbench      CPU-side component throughput -> bench/data/microbench.csv.

Cost: if $GPU_HOURLY_USD is set, cost_per_1m_tokens is emitted per row
        = USD / (tok/s * 3600) * 1e6.
Power: if pynvml is importable, GPU power is sampled at 10 Hz during each
        point; tokens_per_joule is emitted.  Missing pynvml -> NaN, no fail.

The FastAPI server in this repo is a stub generator, so committed engine
numbers come from the in-process mode.  HTTP rows are labelled source=http.
"""
from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import platform
import statistics
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

DATA = Path(__file__).parent / "data"
SEED = 42
DTYPE_BYTES = 4  # fp32 weights/KV on this reference build
GRID = [(1, 128, 32), (1, 512, 64), (2, 512, 64),
        (4, 512, 64), (2, 1024, 64), (4, 128, 32)]

FIELDS = ["source", "device", "hardware", "model", "seed", "concurrency",
          "isl", "osl", "requests", "total_tokens", "wall_s",
          "throughput_tok_s", "ttft_p50_ms", "ttft_p99_ms", "itl_p50_ms",
          "itl_p99_ms", "gpu_hourly_usd", "cost_per_1m_tokens", "power_w",
          "tokens_per_joule"]


def hardware() -> str:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = platform.processor() or platform.machine()
    return f"{dev}:{name}".replace(",", ";")


def q(values: Sequence[float], frac: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return float(s[min(len(s) - 1, int(round(frac * (len(s) - 1))))])


class PowerSampler:
    """10 Hz GPU power sampling via pynvml; silent NaN when unavailable."""

    def __init__(self) -> None:
        self.watts: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handle = None
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._pynvml = None

    def _loop(self) -> None:
        while not self._stop.wait(0.1):
            self._poll()

    def _poll(self) -> None:
        try:
            mw = self._pynvml.nvmlDeviceGetPowerUsage(self._handle)
            self.watts.append(mw / 1000.0)
        except Exception:
            pass

    def __enter__(self) -> "PowerSampler":
        if self._pynvml is not None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._poll()
        try:
            if self._pynvml is not None:
                self._pynvml.nvmlShutdown()
        except Exception:
            pass

    @property
    def mean_w(self) -> float:
        return statistics.fmean(self.watts) if self.watts else float("nan")


class PagedEngine:
    """Sequential prefill + batched decode over the real paged KV cache."""

    def __init__(self, runner, max_rows: int = 32) -> None:
        from miniserve.blocks import BlockManager
        from miniserve.model_runner import GraphedDecodeRunner

        self.runner = runner
        self.vocab = runner.model.config.vocab_size
        self.bs = runner.block_size
        self.bm = BlockManager(runner.max_blocks, block_size=self.bs)
        buckets = [b for b in (1, 2, 4, 8, 16, 32) if b <= max_rows] or [1]
        self.graphed = GraphedDecodeRunner(runner, buckets=buckets)
        self.params = sum(p.numel() for p in runner.model.parameters())
        cfg = runner.model.config
        self.kv_bytes_per_token = (2 * cfg.num_hidden_layers
                                   * runner.num_kv_heads * runner.head_dim
                                   * DTYPE_BYTES)

    def prompt_ids(self, isl: int, tag: int) -> List[int]:
        hi = self.vocab - 1000
        return [1000 + ((tag * 7919 + i * 104729) % hi) for i in range(isl)]

    def _table(self, ids: Sequence[int]) -> torch.Tensor:
        need = (len(ids) + self.bs - 1) // self.bs
        return torch.tensor([self.bm.allocate(need, "bench")], dtype=torch.long)

    @torch.no_grad()
    def prefill(self, ids: List[int]) -> int:
        r = self.runner
        table = self._table(ids)  # one table shared by every layer
        for state in r.layers:
            state.block_table = table
        r.start = 0
        r.row_starts = None
        r._active = True
        try:
            out = r.model(input_ids=torch.tensor([ids]),
                          position_ids=torch.arange(len(ids))[None],
                          use_cache=False)
        finally:
            r._active = False
        return int(out.logits[0, -1].argmax())

    @torch.no_grad()
    def decode(self, tables: List[List[int]], ctx: List[int],
               last: List[int]) -> List[int]:
        logits = self.graphed.decode(last, tables, ctx)
        return logits.argmax(-1).tolist()

    def run_point(self, n: int, isl: int, osl: int) -> dict:
        torch.manual_seed(SEED)
        from miniserve.blocks import BlockManager
        self.bm = BlockManager(self.runner.max_blocks, block_size=self.bs)
        t0 = time.perf_counter()
        tables: List[List[int]] = []
        ctx: List[int] = []
        last: List[int] = []
        ttft_ms: List[float] = []
        itl_ms: List[float] = []
        for i in range(n):
            ids = self.prompt_ids(isl, i + 1)
            token = self.prefill(ids)
            tables.append(self.runner.layers[0].block_table[0].tolist())
            ctx.append(len(ids))
            last.append(token)
            ttft_ms.append((time.perf_counter() - t0) * 1000.0)
        for _ in range(max(0, osl - 1)):
            step_t0 = time.perf_counter()
            tokens = self.decode(tables, ctx, last)
            dt = (time.perf_counter() - step_t0) * 1000.0
            itl_ms.extend([dt] * n)
            last = tokens
            ctx = [c + 1 for c in ctx]
        wall = time.perf_counter() - t0
        total = n * osl
        return {
            "requests": n, "total_tokens": total, "wall_s": wall,
            "throughput_tok_s": total / wall if wall > 0 else float("nan"),
            "ttft_p50_ms": q(ttft_ms, 0.50), "ttft_p99_ms": q(ttft_ms, 0.99),
            "itl_p50_ms": q(itl_ms, 0.50) if itl_ms else float("nan"),
            "itl_p99_ms": q(itl_ms, 0.99) if itl_ms else float("nan"),
            "tokens_processed": n * (isl + osl),
            "batch": n, "mean_ctx": isl + osl / 2,
        }


def roofline_row(res: dict, params: int, kv_bytes: int) -> dict:
    b = res["batch"]
    bytes_read = params * DTYPE_BYTES + kv_bytes * b * res["mean_ctx"]
    bytes_written = kv_bytes * b
    intensity = (2.0 * params * b) / max(1.0, bytes_read + bytes_written)
    tflops = 2.0 * params * res["tokens_processed"] / res["wall_s"] / 1e12
    return {"concurrency": b, "isl": res["isl"], "osl": res["osl"],
            "arithmetic_intensity": round(intensity, 4),
            "achieved_tflops": round(tflops, 6),
            "wall_s": round(res["wall_s"], 4),
            "tokens_processed": res["tokens_processed"]}


def emit_row(res: dict, cfg: dict) -> dict:
    price = os.getenv("GPU_HOURLY_USD")
    price_f = float(price) if price else float("nan")
    cost = (price_f / (res["throughput_tok_s"] * 3600.0) * 1e6
            if price and res["throughput_tok_s"] > 0 else float("nan"))
    joules = res["power_w"] * res["wall_s"]
    tpj = res["total_tokens"] / joules if joules and not math.isnan(joules) else float("nan")
    row = {"source": cfg["source"], "device": cfg["device"],
           "hardware": cfg["hardware"], "model": cfg["model"],
           "seed": SEED, "concurrency": cfg["concurrency"],
           "isl": cfg["isl"], "osl": cfg["osl"]}
    row.update({k: res[k] for k in ("requests", "total_tokens", "wall_s",
                                    "throughput_tok_s", "ttft_p50_ms",
                                    "ttft_p99_ms", "itl_p50_ms", "itl_p99_ms")})
    row.update({"gpu_hourly_usd": price_f if price else "",
                "cost_per_1m_tokens": round(cost, 4) if price else "",
                "power_w": round(res["power_w"], 2) if not math.isnan(res["power_w"]) else "",
                "tokens_per_joule": round(tpj, 4) if not math.isnan(tpj) else ""})
    return row


def run_engine(args) -> None:
    from miniserve.model_runner import ModelRunner

    # 256 blocks = 4096 KV tokens: enough for the grid's largest point
    # (4 x (1024 prompt + generation)), with headroom for one extra step.
    runner = ModelRunner.load(args.model, max_blocks=256)
    eng = PagedEngine(runner, max_rows=max(32, args.concurrency))
    points = GRID if args.matrix else [(args.concurrency, args.isl, args.osl)]
    rows, roof = [], []
    if args.append and (DATA / "all_results.csv").exists():
        with (DATA / "all_results.csv").open(newline="") as f:
            rows = [r for r in csv.DictReader(f)
                    if not (r["concurrency"] == str(args.concurrency)
                            and r["isl"] == str(args.isl)
                            and r["osl"] == str(args.osl))]
    if args.append and (DATA / "roofline.csv").exists():
        with (DATA / "roofline.csv").open(newline="") as f:
            roof = [r for r in csv.DictReader(f)
                    if not (r["concurrency"] == str(args.concurrency)
                            and r["isl"] == str(args.isl)
                            and r["osl"] == str(args.osl))]
    for n, isl, osl in points:
        with PowerSampler() as ps:
            res = eng.run_point(n, isl, osl)
        res["power_w"] = ps.mean_w
        res["isl"], res["osl"] = isl, osl
        row = emit_row(res, {"source": "engine", "device": row_device(),
                             "hardware": hardware(), "model": args.model,
                             "concurrency": n, "isl": isl, "osl": osl})
        rows.append(row)
        roof.append(roofline_row(res, eng.params, eng.kv_bytes_per_token))
        print(f"c={n:<3} isl={isl:<5} osl={osl:<4} "
              f"ttft_p99={row['ttft_p99_ms']:8.1f}ms "
              f"itl_p99={row['itl_p99_ms']:8.1f}ms "
              f"tok/s={row['throughput_tok_s']:8.2f}")
        gc.collect()

    DATA.mkdir(exist_ok=True)
    write_csv(DATA / "all_results.csv", FIELDS, rows)
    write_csv(DATA / "roofline.csv", list(roof[0].keys()), roof)
    print(f"wrote {DATA / 'all_results.csv'} ({len(rows)} rows)")
    print(f"wrote {DATA / 'roofline.csv'} ({len(roof)} rows)")


def run_http(args) -> None:
    import asyncio

    from bench.client import BenchClient

    prompts = ["The quick brown fox jumps over the lazy dog. " * 64] * args.concurrency
    client = BenchClient(args.url)
    start = time.perf_counter()
    results = asyncio.run(client.run(prompts, args.osl, args.concurrency))
    wall = time.perf_counter() - start
    ok = [r for r in results if r.ok]
    ttft = [r.ttft_ms for r in ok if r.ttft_ms is not None]
    itls = [v for r in ok for v in r.itl_ms]
    total = sum(r.n_tokens for r in ok)
    res = {"requests": len(results), "total_tokens": total, "wall_s": wall,
           "throughput_tok_s": total / wall if wall else float("nan"),
           "ttft_p50_ms": q(ttft, 0.5), "ttft_p99_ms": q(ttft, 0.99),
           "itl_p50_ms": q(itls, 0.5), "itl_p99_ms": q(itls, 0.99),
           "power_w": float("nan")}
    row = emit_row(res, {"source": "http", "device": row_device(),
                         "hardware": hardware(), "model": args.model,
                         "concurrency": args.concurrency, "isl": 0,
                         "osl": args.osl})
    write_csv(DATA / "all_results.csv", FIELDS, [row])
    print(f"http: {len(ok)}/{len(results)} ok, {total} tokens, "
          f"{res['throughput_tok_s']:.2f} tok/s")


def row_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def write_csv(path: Path, fields: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run_microbench(args) -> None:
    from miniserve.blocks import BlockManager
    from miniserve.prefix_cache import RadixPrefixCache
    from miniserve.sampler import Sampler
    from miniserve.scheduler import Request, Scheduler

    hw = hardware()
    rows = []
    bench_t0 = time.perf_counter()
    t = time.perf_counter()
    bm = BlockManager(4096, block_size=16)
    iters = 2000
    for i in range(iters):
        table = bm.allocate(1, f"m{i}")
        bm.free(table, f"m{i}")
    dt = time.perf_counter() - t
    rows.append({"component": "block_manager", "op": "allocate+free",
                 "ops_per_sec": round(iters / dt, 1),
                 "us_per_op": round(dt / iters * 1e6, 3),
                 "n": iters, "hardware": hw})

    cache = RadixPrefixCache(bm, block_size=16)
    seqs = [[1000 + (i * 31 + j) % 5000 for j in range(64)] for i in range(200)]
    for i, seq in enumerate(seqs):
        need = (len(seq) + 15) // 16
        cache.insert(seq, bm.allocate(need, f"c{i}"))
    cold = [[9000 + j for j in range(64)] for _ in range(500)]
    t = time.perf_counter()
    for seq in cold:
        cache.match(seq)
    dt = time.perf_counter() - t
    rows.append({"component": "prefix_cache", "op": "match (cold)",
                 "ops_per_sec": round(len(cold) / dt, 1),
                 "us_per_op": round(dt / len(cold) * 1e6, 3),
                 "n": len(cold), "hardware": hw})
    hot = [seqs[i % 200] for i in range(500)]
    t = time.perf_counter()
    for seq in hot:
        cache.match(seq)
    dt = time.perf_counter() - t
    rows.append({"component": "prefix_cache", "op": "match (90% hit)",
                 "ops_per_sec": round(len(hot) / dt, 1),
                 "us_per_op": round(dt / len(hot) * 1e6, 3),
                 "n": len(hot), "hardware": hw})

    sampler = Sampler()
    logits = [float(i % 97) for i in range(151936)]
    t = time.perf_counter()
    for _ in range(100):
        sampler.sample(logits, temperature=0.0)
    dt = time.perf_counter() - t
    rows.append({"component": "sampler", "op": "sample greedy",
                 "ops_per_sec": round(100 / dt, 1),
                 "us_per_op": round(dt / 100 * 1e6, 3),
                 "n": 100, "hardware": hw})

    sched = Scheduler(BlockManager(4096, block_size=16), max_batch=32,
                      prefill_chunk=64)
    for i in range(32):
        sched.add(Request(f"s{i}", [1000 + (i + j) % 5000 for j in range(64)],
                          max_new_tokens=16))
    steps = 200
    t = time.perf_counter()
    for _ in range(steps):
        sched.step()
    dt = time.perf_counter() - t
    rows.append({"component": "scheduler", "op": "step (32 running)",
                 "ops_per_sec": round(steps / dt, 1),
                 "us_per_op": round(dt / steps * 1e6, 3),
                 "n": steps, "hardware": hw})

    write_csv(DATA / "microbench.csv",
              ["component", "op", "ops_per_sec", "us_per_op", "n", "hardware"],
              rows)
    print(f"wrote {DATA / 'microbench.csv'} ({len(rows)} rows) "
          f"in {time.perf_counter() - bench_t0:.1f}s")


def run_prefix_granularity() -> None:
    """Measure matched prefix length vs shared-prefix length (block granularity).

    A radix cache can store token-granular keys, but the KV it hands back is
    block-granular: only whole 16-token blocks can be reused.  This CSV
    quantifies the tokens lost to that rounding -- the evidence behind the
    GAP_ANALYSIS claim about sub-block shared prompts.
    """
    from miniserve.blocks import BlockManager
    from miniserve.prefix_cache import RadixPrefixCache

    bs = 16
    bm = BlockManager(4096, block_size=bs)
    cache = RadixPrefixCache(bm, block_size=bs)
    base = [1000 + (i * 131 + 7) % 20000 for i in range(512)]
    cache.insert(base, bm.allocate(len(base) // bs, "sys"))
    rows = []
    for shared in (8, 15, 16, 17, 30, 31, 32, 48, 64, 128):
        matched, _ = cache.match(base[:shared])
        rows.append({"shared_prefix_tokens": shared,
                     "matched_tokens": matched,
                     "wasted_tokens": shared - matched,
                     "block_size": bs,
                     "hardware": hardware()})
        print(f"shared={shared:<4} matched={matched:<4} wasted={shared - matched}")
    write_csv(DATA / "prefix_granularity.csv",
              ["shared_prefix_tokens", "matched_tokens", "wasted_tokens",
               "block_size", "hardware"], rows)
    print(f"wrote {DATA / 'prefix_granularity.csv'} ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--isl", type=int, default=128)
    ap.add_argument("--osl", type=int, default=32)
    ap.add_argument("--matrix", action="store_true")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--microbench", action="store_true")
    ap.add_argument("--prefix-granularity", action="store_true")
    ap.add_argument("--url", default="")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    if args.microbench:
        run_microbench(args)
    elif args.prefix_granularity:
        run_prefix_granularity()
    elif args.url:
        run_http(args)
    else:
        run_engine(args)


if __name__ == "__main__":
    main()

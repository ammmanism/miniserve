"""bench/overlap_proof.py -- Part 8 gate: overlap v2 proof + TTFT (ADR-006).

Runs the overlapped pipeline against a synthetic forward/sample pair
(forward sleep --forward-ms, sample sleep --sample-ms).  The mechanism is
real (side-thread sampling, plan-thread scheduling, draft hooks, double
buffers); the latencies are modeled, so the CSV proves scheduling overlap,
not kernel speed.

Writes:
  bench/data/overlap_proof_v2.csv         per-step trace (32-way decode)
  bench/data/overlap_with_chunked_prefill.csv  median TTFT, ON vs sync OFF

Gates (exit 1 on failure): median overlap_ratio < 1,
median sample_overlap_ratio >= 0.5, prefill-during-forward fraction >= 0.3.
"""
import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

from miniserve.blocks import BlockManager
from miniserve.overlap import OverlapEngine, StepPlan
from miniserve.scheduler import Request, Scheduler

V = 32000
DATA = Path(__file__).parent / "data"


def make_fns(fwd_ms, samp_ms):
    def forward(rows):
        time.sleep(fwd_ms / 1000.0)
        return [[(row[-1] * 1103515245 + 12345) % V] for row in rows]

    def sample(out, plan):
        time.sleep(samp_ms / 1000.0)
        return [o[0] for o in out]

    return forward, sample


def make_batch(n, isl, osl, arrival):
    return [Request(f"q{i}", [(i * 31 + t) % V for t in range(isl)],
                    max_new_tokens=osl, arrival_time=arrival) for i in range(n)]


def run_overlap(args, fwd_ms, samp_ms):
    forward, sample = make_fns(fwd_ms, samp_ms)
    bm = BlockManager(num_blocks=2048, block_size=16)
    sched = Scheduler(bm, max_batch=args.concurrency, max_chunked=4,
                      prefill_chunk=args.chunk)
    eng = OverlapEngine(sched, forward, sample, chunk_size=args.chunk)
    t0 = time.perf_counter()
    reqs = make_batch(args.concurrency, args.isl, args.osl, t0)
    for r in reqs:
        assert sched.add(r)
    eng.run(reqs, args.steps)
    active = [r for r in eng.rows_log
              if r["n_decoding"] or r["n_prefilling"]]
    ttft = {rid: (ts[0] - t0) * 1000.0 for rid, ts in eng.ship_times.items()
            if ts}
    return eng, active, ttft


def run_sync(args, fwd_ms, samp_ms):
    """No-overlap baseline: join everything between phases, every step."""
    forward, sample = make_fns(fwd_ms, samp_ms)
    bm = BlockManager(num_blocks=2048, block_size=16)
    sched = Scheduler(bm, max_batch=args.concurrency, max_chunked=4,
                      prefill_chunk=args.chunk)
    off = OverlapEngine(sched, forward, sample, chunk_size=args.chunk)
    t0 = time.perf_counter()
    reqs = make_batch(args.concurrency, args.isl, args.osl, t0)
    for r in reqs:
        assert sched.add(r)
        off.submit(r)
    done = 0
    ttft = {}
    while done < args.concurrency:
        ev = sched.step()
        rows, rids = off.rows_for(ev)
        out = forward(rows)
        toks = sample(out, StepPlan(rows=rows, row_rids=rids, ev=ev))
        for j, rid in enumerate(rids):
            if rid not in ev["decoding"]:
                continue
            off.outputs[rid].append(toks[j])
            off._last[rid] = toks[j]
            if rid not in ttft:
                ttft[rid] = (time.perf_counter() - t0) * 1000.0
        done += len(ev["finished"])
    return ttft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--isl", type=int, default=128)
    ap.add_argument("--osl", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--forward-ms", type=float, default=20.0)
    ap.add_argument("--sample-ms", type=float, default=3.0)
    args = ap.parse_args()

    eng, active, ttft_on = run_overlap(args, args.forward_ms, args.sample_ms)
    med = statistics.median
    gate_or = med(r["overlap_ratio"] for r in active)
    gate_sr = med(r["sample_overlap_ratio"] for r in active)
    frac = sum(r["plan_includes_new_prefill"] for r in active) / len(active)

    DATA.mkdir(exist_ok=True)
    trace = DATA / "overlap_proof_v2.csv"
    with trace.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(active[0].keys()))
        w.writeheader()
        w.writerows(active)

    ttft_off = run_sync(args, args.forward_ms, args.sample_ms)
    ttcsv = DATA / "overlap_with_chunked_prefill.csv"
    with ttcsv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "median_ttft_ms", "p90_ttft_ms", "n_requests"])
        for mode, t in (("chunked+overlap", ttft_on), ("sync-no-overlap", ttft_off)):
            vals = sorted(t.values())
            w.writerow([mode, round(med(vals), 2),
                        round(vals[int(0.9 * len(vals))], 2), len(vals)])

    print(f"active steps: {len(active)}  rows: bench/data/{trace.name}")
    print(f"median overlap_ratio        = {gate_or:.3f}  (gate < 1)")
    print(f"median sample_overlap_ratio = {gate_sr:.3f}  (gate >= 0.5)")
    print(f"prefill-during-forward frac = {frac:.3f}  (gate >= 0.3)")
    print(f"TTFT chunked+overlap = {med(ttft_on.values()):.1f} ms, "
          f"sync = {med(ttft_off.values()):.1f} ms")
    ok = gate_or < 1.0 and gate_sr >= 0.5 and frac >= 0.3
    print("GATES:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

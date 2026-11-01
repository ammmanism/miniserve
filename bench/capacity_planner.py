"""bench.capacity_planner -- max concurrency under an SLO (Part 10).

Given an SLO (P99 TTFT, P99 ITL) and a workload (ISL/OSL), find the largest
concurrency whose *measured* P99s satisfy it.

Search: probe the ladder 1,2,4,8,... until a probe fails (or --max-concurrency
is hit), then binary-search the gap, measuring midpoints until the gap is 1
or --max-measurements is spent.  Every returned number comes from a real
run_point() measurement -- there is no analytical model here.

``--from-csv`` reuses bench/data/all_results.csv instead of loading the model,
which is how the committed slo_capacity.csv is produced reproducibly.

Output row: slo_ttft_ms, slo_itl_ms, max_concurrency, achieved_throughput.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

DATA = Path(__file__).parent / "data"
FIELDS = ["slo_ttft_ms", "slo_itl_ms", "isl", "osl", "max_concurrency",
          "achieved_throughput", "measured_points"]


def meets(res: dict, slo_ttft: float, slo_itl: float) -> bool:
    ttft = res["ttft_p99_ms"]
    itl = res["itl_p99_ms"]
    if math.isnan(ttft) or math.isnan(itl):
        return False
    return ttft <= slo_ttft and itl <= slo_itl


def search(results: Dict[int, dict], slo_ttft: float,
           slo_itl: float) -> Tuple[int, dict]:
    """Largest measured concurrency meeting the SLO (0 when none does)."""
    ok = [c for c, r in results.items() if meets(r, slo_ttft, slo_itl)]
    if not ok:
        return 0, {}
    best = max(ok)
    return best, results[best]


def from_csv(path: Path, isl: int, osl: int) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("source") != "engine":
                continue
            if int(row["isl"]) != isl or int(row["osl"]) != osl:
                continue
            c = int(row["concurrency"])
            out[c] = {
                "ttft_p99_ms": float(row["ttft_p99_ms"]),
                "itl_p99_ms": float(row["itl_p99_ms"]),
                "throughput_tok_s": float(row["throughput_tok_s"]),
            }
    return out


def measure(isl: int, osl: int, max_c: int, max_measurements: int,
            model: str, slo_ttft: float, slo_itl: float) -> Dict[int, dict]:
    from bench.run_suite import PagedEngine
    from miniserve.model_runner import ModelRunner

    runner = ModelRunner.load(model)
    eng = PagedEngine(runner, max_rows=max_c)
    results: Dict[int, dict] = {}

    def probe(c: int) -> dict:
        r = eng.run_point(c, isl, osl)
        results[c] = r
        print(f"  measured c={c:<3} p99ttft={r['ttft_p99_ms']:9.1f}ms "
              f"p99itl={r['itl_p99_ms']:9.1f}ms "
              f"tok/s={r['throughput_tok_s']:.3f}")
        return r

    lo, hi = 0, None
    c = 1
    while c <= max_c and len(results) < max_measurements:
        r = probe(c)
        if not meets(r, slo_ttft, slo_itl):
            hi = c
            break
        lo = c
        c *= 2
    if hi is None:
        hi = max_c
    while lo + 1 < hi and len(results) < max_measurements:
        mid = (lo + hi) // 2
        if mid in results:
            lo = mid
            continue
        r = probe(mid)
        if meets(r, slo_ttft, slo_itl):
            lo = mid
        else:
            hi = mid
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slo-ttft", type=float, required=True,
                    help="P99 TTFT budget in ms")
    ap.add_argument("--slo-itl", type=float, required=True,
                    help="P99 ITL budget in ms")
    ap.add_argument("--isl", type=int, default=512)
    ap.add_argument("--osl", type=int, default=128)
    ap.add_argument("--max-concurrency", type=int, default=32)
    ap.add_argument("--max-measurements", type=int, default=6)
    ap.add_argument("--from-csv", default="")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--out", default=str(DATA / "slo_capacity.csv"))
    args = ap.parse_args()

    if args.from_csv:
        results = from_csv(Path(args.from_csv), args.isl, args.osl)
    else:
        results = measure(args.isl, args.osl, args.max_concurrency,
                          args.max_measurements, args.model,
                          args.slo_ttft, args.slo_itl)

    best, res = search(results, args.slo_ttft, args.slo_itl)
    points = ";".join(f"{c}:{results[c]['throughput_tok_s']:.4f}"
                      for c in sorted(results))
    row = {"slo_ttft_ms": args.slo_ttft, "slo_itl_ms": args.slo_itl,
           "isl": args.isl, "osl": args.osl, "max_concurrency": best,
           "achieved_throughput": round(res["throughput_tok_s"], 4) if res else 0,
           "measured_points": points}

    out = Path(args.out)
    rows: List[dict] = []
    if args.append and out.exists():
        with out.open(newline="") as f:
            rows = [r for r in csv.DictReader(f)
                    if not (float(r["slo_ttft_ms"]) == args.slo_ttft
                            and float(r["slo_itl_ms"]) == args.slo_itl
                            and int(r["isl"]) == args.isl
                            and int(r["osl"]) == args.osl)]
    rows.append(row)
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"SLO: p99_ttft<{args.slo_ttft:g}ms, p99_itl<{args.slo_itl:g}ms")
    print(f"workload: isl={args.isl}, osl={args.osl}")
    print(f"max_concurrency: {best}")
    print(f"achieved_throughput_tok_s: {row['achieved_throughput']}")
    print(f"measured points: {points}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

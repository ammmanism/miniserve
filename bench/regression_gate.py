"""bench.regression_gate -- fail a PR when committed-CSV numbers regress.

Compares current bench/data/all_results.csv against a committed baseline
(bench/data/baseline.csv, engine rows only) per (concurrency, isl, osl):

    throughput_tok_s   regression if current < baseline * (1 - 5%)
    ttft_p99_ms        regression if current > baseline * (1 + 10%)
    itl_p99_ms         regression if current > baseline * (1 + 10%)

Exit 0 = pass, 1 = regression (or no overlapping configs / unreadable CSV).
A current metric that is missing/NaN counts as a regression: silence is not
evidence of health.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

DATA = Path(__file__).parent / "data"
TOL = {"throughput_tok_s": 0.05, "ttft_p99_ms": 0.10, "itl_p99_ms": 0.10}
HIGHER_IS_BETTER = {"throughput_tok_s"}


def load(path: Path) -> Dict[Tuple[int, int, int], Dict[str, float]]:
    out: Dict[Tuple[int, int, int], Dict[str, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("source", "engine") != "engine":
                continue
            key = (int(row["concurrency"]), int(row["isl"]), int(row["osl"]))
            out[key] = {name: _num(row.get(name)) for name in TOL}
    return out


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def check(base: Dict, cur: Dict) -> List[Tuple[Tuple[int, int, int], str,
                                               float, float, float, bool]]:
    rows = []
    for key in sorted(set(base) & set(cur)):
        for metric, tol in TOL.items():
            b, c = base[key][metric], cur[key][metric]
            if math.isnan(b):
                continue
            if math.isnan(c):
                rows.append((key, metric, b, c, float("-inf"), False))
                continue
            if metric in HIGHER_IS_BETTER:
                limit = b * (1.0 - tol)
                ok = c >= limit
            else:
                limit = b * (1.0 + tol)
                ok = c <= limit
            delta = (c - b) / b * 100.0 if b else float("nan")
            rows.append((key, metric, b, c, delta, ok))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(DATA / "baseline.csv"))
    ap.add_argument("--results", default=str(DATA / "all_results.csv"))
    ap.add_argument("--throughput-tol", type=float, default=0.05)
    ap.add_argument("--ttft-tol", type=float, default=0.10)
    ap.add_argument("--itl-tol", type=float, default=0.10)
    args = ap.parse_args()
    TOL["throughput_tok_s"] = args.throughput_tol
    TOL["ttft_p99_ms"] = args.ttft_tol
    TOL["itl_p99_ms"] = args.itl_tol

    base_path, cur_path = Path(args.baseline), Path(args.results)
    for p in (base_path, cur_path):
        if not p.exists():
            print(f"error: {p} not found", file=sys.stderr)
            sys.exit(1)
    base, cur = load(base_path), load(cur_path)
    overlap = set(base) & set(cur)
    if not overlap:
        print(f"error: no overlapping (concurrency,isl,osl) configs between "
              f"{base_path.name} and {cur_path.name}", file=sys.stderr)
        sys.exit(1)

    rows = check(base, cur)
    print(f"{'config':<22}{'metric':<18}{'baseline':>12}{'current':>12}"
          f"{'delta':>9}  verdict")
    failures = 0
    for (c, isl, osl), metric, b, cur_v, delta, ok in rows:
        failures += 0 if ok else 1
        tag = f"c={c} isl={isl} osl={osl}"
        d = "n/a" if delta == float("-inf") else f"{delta:+.1f}%"
        print(f"{tag:<22}{metric:<18}{b:>12.4f}{cur_v:>12.4f}{d:>9}  "
              f"{'PASS' if ok else 'FAIL'}")
    skipped = set(base) - overlap
    for key in sorted(skipped):
        print(f"c={key[0]} isl={key[1]} osl={key[2]}  (not in current run, "
              f"skipped)")
    verdict = "PASS" if failures == 0 else f"FAIL ({failures} regression(s))"
    print(f"\nGATE: {verdict} -- {len(overlap)} configs, {len(rows)} checks")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()

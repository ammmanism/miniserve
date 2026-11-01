"""tests/test_bench_v2.py -- Part 10 bench tooling (fast, model-free).

Covers: regression-gate pass/fail logic and exit codes, capacity search,
roofline PNG rendering, and the microbench (<60 s, >= 4 rows).  All file
outputs go to tmp_path; committed bench/data is never modified by tests.
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import pytest

from bench import plot, run_suite
from bench.capacity_planner import from_csv, search

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["source", "device", "hardware", "model", "seed", "concurrency",
          "isl", "osl", "requests", "total_tokens", "wall_s",
          "throughput_tok_s", "ttft_p50_ms", "ttft_p99_ms", "itl_p50_ms",
          "itl_p99_ms", "gpu_hourly_usd", "cost_per_1m_tokens", "power_w",
          "tokens_per_joule"]


def _row(**kw):
    base = {"source": "engine", "device": "cpu", "hardware": "test",
            "model": "m", "seed": 42, "concurrency": 1, "isl": 128, "osl": 32,
            "requests": 1, "total_tokens": 32, "wall_s": 1.0,
            "throughput_tok_s": 32.0, "ttft_p50_ms": 10.0, "ttft_p99_ms": 12.0,
            "itl_p50_ms": 5.0, "itl_p99_ms": 6.0, "gpu_hourly_usd": "",
            "cost_per_1m_tokens": "", "power_w": "",
            "tokens_per_joule": ""}
    base.update(kw)
    return base


def _write(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def _gate(base_csv, cur_csv):
    return subprocess.run(
        [sys.executable, str(ROOT / "bench" / "regression_gate.py"),
         "--baseline", str(base_csv), "--results", str(cur_csv)],
        capture_output=True, text=True, cwd=ROOT,
    )


def test_regression_gate_identical_passes(tmp_path):
    base = tmp_path / "base.csv"
    _write(base, [_row(), _row(concurrency=4, throughput_tok_s=100.0)])
    proc = _gate(base, base)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "GATE: PASS" in proc.stdout


def test_regression_gate_catches_regressions(tmp_path):
    base, cur = tmp_path / "base.csv", tmp_path / "cur.csv"
    _write(base, [_row()])
    _write(cur, [_row(throughput_tok_s=28.0,      # -12.5% > 5%
                      ttft_p99_ms=14.0,           # +16.7% > 10%
                      itl_p99_ms=6.5)])           # +8.3% ok
    proc = _gate(base, cur)
    assert proc.returncode == 1
    assert "FAIL (2 regression(s))" in proc.stdout


def test_regression_gate_tolerates_small_noise(tmp_path):
    base, cur = tmp_path / "base.csv", tmp_path / "cur.csv"
    _write(base, [_row()])
    _write(cur, [_row(throughput_tok_s=31.0, ttft_p99_ms=12.5,
                      itl_p99_ms=6.3)])
    assert _gate(base, cur).returncode == 0


def test_capacity_search_sane_and_csv_roundtrip(tmp_path):
    path = tmp_path / "results.csv"
    _write(path, [
        _row(concurrency=1, throughput_tok_s=32.0),
        _row(concurrency=2, throughput_tok_s=55.0),
        _row(concurrency=4, throughput_tok_s=70.0, ttft_p99_ms=900.0),
    ])
    measured = from_csv(path, 128, 32)
    assert set(measured) == {1, 2, 4}

    c, row = search(measured, slo_ttft=500.0, slo_itl=50.0)
    assert c == 2
    assert 0 < c < 1000
    assert row["throughput_tok_s"] == 55.0

    assert search(measured, slo_ttft=1.0, slo_itl=1.0) == (0, {})


def test_roofline_png_generated(tmp_path, monkeypatch):
    monkeypatch.setattr(plot, "DATA", tmp_path)
    rows = [{"concurrency": c, "isl": 128, "osl": 32,
             "arithmetic_intensity": 40.0 + c, "achieved_tflops": 0.05 + c / 100,
             "wall_s": 1.0, "tokens_processed": 160 * c}
            for c in (1, 2, 4, 8, 16, 32)]
    with (tmp_path / "roofline.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    out = plot.roofline_plot(out=tmp_path / "roofline.png")
    assert out.exists() and out.stat().st_size > 50_000


def test_microbench_fast_and_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(run_suite, "DATA", tmp_path)
    import time
    t0 = time.perf_counter()
    run_suite.run_microbench(argparse.Namespace())
    assert time.perf_counter() - t0 < 60.0
    with (tmp_path / "microbench.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 4
    assert all(float(r["ops_per_sec"]) > 0 for r in rows)

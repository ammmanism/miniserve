"""bench.plot -- figures for the committed benchmark CSVs (Part 10).

All figures are rendered from bench/data/*.csv; no number is typed into a
plot.  Hardware roofline peaks are read from the device when CUDA is present,
otherwise from MINISERVE_MEM_BW_GBS / MINISERVE_PEAK_TFLOPS with documented
CPU assumptions (51.2 GB/s dual-channel DDR4-3200 theoretical, 0.72 TFLOP/s
fp32 AVX2 at 8 threads).  The plot annotates which source was used.
"""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = Path(__file__).parent / "data"


def load(name: str) -> List[Dict[str, str]]:
    path = DATA / name
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _f(row: Dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def peaks() -> tuple[float, float, str]:
    """(memory GB/s, compute TFLOP/s, provenance string)."""
    mem = os.getenv("MINISERVE_MEM_BW_GBS")
    flop = os.getenv("MINISERVE_PEAK_TFLOPS")
    if mem and flop:
        return float(mem), float(flop), "env override"
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            bw = (getattr(p, "memory_clock_rate", 0) * 1e3
                  * getattr(p, "memory_bus_width", 0) / 8 / 1e9)
            fps = (p.multi_processor_count * 128 * 2
                   * getattr(p, "clock_rate", 0) * 1e3 / 1e12)
            if bw > 0:
                return bw, fps, f"cuda:{p.name}"
    except Exception:
        pass
    return 51.2, 0.72, "assumed CPU DDR4-3200 / AVX2"


def roofline_plot(out: Path | None = None) -> Path:
    rows = load("roofline.csv")
    out = out or DATA / "roofline.png"
    mem, flop, prov = peaks()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    xs = [max(0.01, _f(r, "arithmetic_intensity")) for r in rows]
    ys = [max(1e-6, _f(r, "achieved_tflops")) for r in rows]
    ax.scatter(xs, ys, s=46, zorder=3, label="miniserve points (measured)")
    for r, x, y in zip(rows, xs, ys):
        ax.annotate(f"c={r['concurrency']},isl={r['isl']}", (x, y),
                    textcoords="offset points", xytext=(4, 4), fontsize=6)
    grid = [10 ** (i / 20) for i in range(-40, 61)]
    mem_line = [x * mem / 1000.0 for x in grid]      # GB/s -> TFLOP/s at 1 flop/byte
    ax.plot(grid, mem_line, "--", lw=1.2,
            label=f"memory roofline {mem:.1f} GB/s")
    ax.axhline(flop, linestyle="--", lw=1.2,
               label=f"compute peak {flop:.2f} TFLOP/s")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("achieved TFLOP/s")
    ax.set_title(f"Roofline -- miniserve ({prov})", fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def throughput_vs_concurrency(out: Path | None = None) -> Path:
    rows = [r for r in load("all_results.csv") if r.get("source") == "engine"]
    out = out or DATA / "throughput_vs_concurrency.png"
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    keys = sorted({(r["isl"], r["osl"]) for r in rows})
    for isl, osl in keys:
        pts = sorted((int(r["concurrency"]), _f(r, "throughput_tok_s"))
                     for r in rows if r["isl"] == isl and r["osl"] == osl)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    marker="o", label=f"isl={isl},osl={osl}")
    ax.set_xlabel("concurrency")
    ax.set_ylabel("throughput (tok/s)")
    ax.set_title("Throughput vs concurrency (oversubscription)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def latency_vs_concurrency(out: Path | None = None) -> Path:
    rows = [r for r in load("all_results.csv") if r.get("source") == "engine"]
    out = out or DATA / "latency_vs_concurrency.png"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
    for isl, osl in sorted({(r["isl"], r["osl"]) for r in rows}):
        pts = sorted((int(r["concurrency"]), r) for r in rows
                     if r["isl"] == isl and r["osl"] == osl)
        xs = [p[0] for p in pts]
        a1.plot(xs, [_f(r, "ttft_p99_ms") for _, r in pts], marker="o",
                label=f"isl={isl},osl={osl}")
        a2.plot(xs, [_f(r, "itl_p99_ms") for _, r in pts], marker="o",
                label=f"isl={isl},osl={osl}")
    for ax, title in ((a1, "P99 TTFT (ms)"), (a2, "P99 ITL (ms)")):
        ax.set_xlabel("concurrency")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    a1.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def slo_capacity_plot(out: Path | None = None) -> Path:
    rows = load("slo_capacity.csv")
    out = out or DATA / "slo_capacity.png"
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    labels: List[str] = []
    xs, ys = [], []
    for r in rows:
        xs.append(_f(r, "max_concurrency"))
        ys.append(_f(r, "achieved_throughput"))
        labels.append(f"p99ttft<{r['slo_ttft_ms']}ms\np99itl<{r['slo_itl_ms']}ms")
    if xs:
        ax.bar(range(len(xs)), ys, color="#5b8def")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(labels, fontsize=7)
        for i, (x, y) in enumerate(zip(xs, ys)):
            ax.annotate(f"c={x:.0f}", (i, y), ha="center", va="bottom",
                        fontsize=8)
    ax.set_ylabel("achieved throughput (tok/s)")
    ax.set_title("SLO capacity: max concurrency meeting each SLO")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def microbench_plot(out: Path | None = None) -> Path:
    rows = load("microbench.csv")
    out = out or DATA / "microbench.png"
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    names = [f"{r['component']}\n{r['op']}" for r in rows]
    vals = [max(1e-3, _f(r, "ops_per_sec")) for r in rows]
    ax.barh(range(len(rows)), vals, color="#2bb673")
    ax.set_xscale("log")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names, fontsize=7)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:,.0f} ops/s", (v, i), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=7)
    ax.set_xlabel("ops/s (log scale)")
    ax.set_title("CPU-side component throughput")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def cost_plot(out: Path | None = None) -> Path | None:
    rows = [r for r in load("all_results.csv")
            if r.get("cost_per_1m_tokens") not in ("", None)]
    if not rows:
        return None
    out = out or DATA / "cost_per_million_tokens.png"
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for isl, osl in sorted({(r["isl"], r["osl"]) for r in rows}):
        pts = sorted((int(r["concurrency"]), _f(r, "cost_per_1m_tokens"))
                     for r in rows if r["isl"] == isl and r["osl"] == osl)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                label=f"isl={isl},osl={osl}")
    ax.set_xlabel("concurrency")
    ax.set_ylabel("USD / 1M generated tokens")
    ax.set_title("Cost per million tokens (GPU_HOURLY_USD)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def all_plots() -> List[Path]:
    made = [roofline_plot(), throughput_vs_concurrency(),
            latency_vs_concurrency(), slo_capacity_plot(), microbench_plot()]
    cost = cost_plot()
    if cost:
        made.append(cost)
    return made


if __name__ == "__main__":
    for p in all_plots():
        print(f"wrote {p}")

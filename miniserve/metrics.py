"""miniserve.metrics -- dependency-free Prometheus text exposition.

The server exposes ``/metrics`` and the scheduler feeds queue-wait into the
same process-wide registry, so a single scrape shows both the HTTP layer and
the batching layer.  Implemented by hand (counters, gauges, histograms) so
the format is explicit and testable: ``render()`` emits valid Prometheus
exposition text (version 0.0.4).

Naming: every series is ``miniserve_<metric>``; histograms expand to
``_bucket``/``_sum``/``_count`` exactly as Prometheus expects.  Bucket counts
are cumulative (each observation increments every bucket whose bound it is
<=), so rendering is a straight dump.
"""
from __future__ import annotations

import threading
from typing import Dict, Iterable, Optional, Tuple

Labels = Tuple[Tuple[str, str], ...]

DEFAULT_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
                   1.0, 2.5, 5.0, 10.0, 30.0, 60.0)

HELP = {
    "miniserve_ttft_seconds": "Time to first token, end-to-end (seconds).",
    "miniserve_itl_seconds": "Inter-token latency per decode step (seconds).",
    "miniserve_queue_wait_seconds": "Scheduler admission wait (seconds).",
    "miniserve_requests_total": "Completion requests by status.",
    "miniserve_tokens_total": "Generated tokens by kind.",
    "miniserve_queue_depth": "Requests waiting for admission.",
    "miniserve_running": "Requests currently decoding.",
}


def _key(name: str, labels: Optional[Dict[str, str]]) -> Tuple[str, Labels]:
    return name, tuple(sorted((labels or {}).items()))


def _fmt_labels(labels: Labels) -> str:
    if not labels:
        return ""
    return "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}"


class _Histogram:
    __slots__ = ("buckets", "counts", "total", "count")

    def __init__(self, buckets: Iterable[float]) -> None:
        self.buckets = tuple(sorted(buckets))
        self.counts = [0] * len(self.buckets)  # cumulative per bucket
        self.total = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        for i, edge in enumerate(self.buckets):
            if value <= edge:
                self.counts[i] += 1


class Metrics:
    """Thread-safe counters, gauges and histograms with Prometheus rendering."""

    def __init__(self, buckets: Iterable[float] = DEFAULT_BUCKETS) -> None:
        self._lock = threading.Lock()
        self._buckets = tuple(sorted(buckets))
        self._counters: Dict[Tuple[str, Labels], float] = {}
        self._gauges: Dict[Tuple[str, Labels], float] = {}
        self._hist: Dict[Tuple[str, Labels], _Histogram] = {}

    # -- writers ---------------------------------------------------------
    def counter(self, name: str, value: float = 1.0,
                labels: Optional[Dict[str, str]] = None) -> None:
        k = _key(name, labels)
        with self._lock:
            self._counters[k] = self._counters.get(k, 0.0) + value

    def gauge(self, name: str, value: float,
              labels: Optional[Dict[str, str]] = None) -> None:
        with self._lock:
            self._gauges[_key(name, labels)] = float(value)

    def observe(self, name: str, value: float,
                labels: Optional[Dict[str, str]] = None,
                buckets: Optional[Iterable[float]] = None) -> None:
        k = _key(name, labels)
        with self._lock:
            hist = self._hist.get(k)
            if hist is None:
                hist = _Histogram(buckets or self._buckets)
                self._hist[k] = hist
            hist.observe(float(value))

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._hist.clear()

    # -- renderer --------------------------------------------------------
    def render(self) -> str:
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            hists = {k: (h.buckets, list(h.counts), h.total, h.count)
                     for k, h in self._hist.items()}

        def header(name: str, kind: str) -> list[str]:
            out = []
            if name in HELP:
                out.append(f"# HELP {name} {HELP[name]}")
            out.append(f"# TYPE {name} {kind}")
            return out

        out: list[str] = []
        for name in sorted({n for n, _ in counters}):
            out += header(name, "counter")
            for (n, labels), value in sorted(counters.items()):
                if n == name:
                    out.append(f"{name}{_fmt_labels(labels)} {value:g}")

        for name in sorted({n for n, _ in gauges}):
            out += header(name, "gauge")
            for (n, labels), value in sorted(gauges.items()):
                if n == name:
                    out.append(f"{name}{_fmt_labels(labels)} {value:g}")

        for name in sorted({n for n, _ in hists}):
            out += header(name, "histogram")
            for (n, labels), (edges, counts, total, count) in sorted(hists.items()):
                if n != name:
                    continue
                for edge, seen in zip(edges, counts):
                    merged = dict(labels)
                    merged.update({"le": str(edge)})
                    out.append(
                        f"{name}_bucket{_fmt_labels(tuple(sorted(merged.items())))} "
                        f"{seen}"
                    )
                merged = dict(labels)
                merged.update({"le": "+Inf"})
                out.append(
                    f"{name}_bucket{_fmt_labels(tuple(sorted(merged.items())))} {count}"
                )
                out.append(f"{name}_sum{_fmt_labels(labels)} {total:g}")
                out.append(f"{name}_count{_fmt_labels(labels)} {count}")
        return "\n".join(out) + "\n"


METRICS = Metrics()


def export() -> str:
    """Prometheus exposition text for the process-wide registry."""
    return METRICS.render()

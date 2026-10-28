"""miniserve.metrics — Prometheus metrics collection and export.

Tracks request latency (TTFT, TPOT, e2e), throughput, queue depth,
cache hit rates, and GPU utilization for monitoring.

TODO: Implement MetricsCollector with prometheus-client gauges/histograms.
"""

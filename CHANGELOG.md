# Changelog

All notable changes to miniserve are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning is [SemVer](https://semver.org/).

## [1.0.0] — 2026-11-02

First tagged release. Every engine primitive below is unit- or parity-tested;
the linked test file is the ground truth for what "works" means.

### Engine primitives

| Primitive | Owned by | Verified by |
|---|---|---|
| Paged KV block manager (refcounts, copy-on-write forks, eviction) | [`miniserve/blocks.py`](miniserve/blocks.py) | [`tests/test_blocks.py`](tests/test_blocks.py) |
| Radix tree prefix cache (longest-match reuse, LRU eviction, block reinsert) | [`miniserve/prefix_cache.py`](miniserve/prefix_cache.py) | [`tests/test_prefix_cache.py`](tests/test_prefix_cache.py) |
| Continuous-batching scheduler (priority, chunked prefill, SLO admission, preemption) | [`miniserve/scheduler.py`](miniserve/scheduler.py) | [`tests/test_scheduler.py`](tests/test_scheduler.py) |
| Overlap scheduler (prefill/decode pipelining, double buffers, overlap-ratio oracle) | [`miniserve/overlap.py`](miniserve/overlap.py) | [`tests/test_overlap.py`](tests/test_overlap.py) |
| End-to-end engine loop and stats | [`miniserve/engine.py`](miniserve/engine.py) | [`tests/test_engine.py`](tests/test_engine.py) |
| PagedAttention + RoPE (MHA / GQA / MQA / sliding window / FP8 KV) | [`miniserve/attention.py`](miniserve/attention.py), [`miniserve/model_runner.py`](miniserve/model_runner.py) | [`tests/test_parity.py`](tests/test_parity.py) |
| Greedy parity vs HuggingFace (baseline runner + INT8 quantized weights) | [`miniserve/model_runner.py`](miniserve/model_runner.py) | [`tests/test_parity.py`](tests/test_parity.py) |
| Bucketed CUDA-graph decode (double buffers, pool persistence, ragged→bucket interpolation) | [`miniserve/model_runner.py`](miniserve/model_runner.py) | [`tests/test_cuda_graph.py`](tests/test_cuda_graph.py) |
| Split-K Triton flash-decode kernel (tensor-core tiling, log-sum-exp split merge, fused RoPE, persisted autotune) | [`kernels/paged_attn_triton.py`](kernels/paged_attn_triton.py) | [`tests/test_triton_parity.py`](tests/test_triton_parity.py) |
| HTTP API (SSE streaming, queue backpressure, rate limiting, request dedup) | [`miniserve/server.py`](miniserve/server.py) | [`tests/test_server.py`](tests/test_server.py), [`tests/test_client.py`](tests/test_client.py) |
| Grammar-constrained JSON output (logits-processor hook) | [`miniserve/sampler.py`](miniserve/sampler.py) | [`tests/test_parity.py`](tests/test_parity.py) |
| Prometheus metrics registry (counters, gauges, cumulative histograms, scheduler-batching series) | [`miniserve/metrics.py`](miniserve/metrics.py) | [`tests/test_metrics.py`](tests/test_metrics.py) |
| OpenTelemetry tracing hooks (prefill → decode → detokenize spans, dedup flag) | [`miniserve/server.py`](miniserve/server.py) | [`tests/test_server.py`](tests/test_server.py) |
| Streaming client layer (SSE re-connect, idle-keepalive, backpressure) | [`miniserve/stream.py`](miniserve/stream.py) | [`tests/test_client.py`](tests/test_client.py) |
| Benchmark suite + empirical CSVs + plot/capacity planner | [`bench/run_suite.py`](bench/run_suite.py), [`bench/capacity_planner.py`](bench/capacity_planner.py) | [`tests/test_bench_v2.py`](tests/test_bench_v2.py) |
| Performance regression gate (5% throughput / 10% P99 TTFT·ITL tolerances) | [`bench/regression_gate.py`](bench/regression_gate.py) | [`tests/test_bench_v2.py`](tests/test_bench_v2.py) |

### CI / release / supply chain (this phase)

- **ci.yml**: matrix OS (ubuntu, macos) × Python (3.10 / 3.11 / 3.12) × torch (2.3.1 / 2.4.0 / 2.5.0), each running `pytest -m "not gpu and not slow"`.
- **ci.yml**: per-PR dependency audit — `pip-audit` via uv; SBOM with `syft` (SPDX upload as artifact); vuln scan with `grype --fail-on high`.
- **ci.yml**: performance regression gate on every PR to `main` (c=4 / ISL=512 / OSL=64 smoke → `regression_gate.py` on the committed baseline).
- **release.yml**: multi-platform wheels via cibuildwheel (linux-x86_64, linux-aarch64 via QEMU, macOS-arm64) + SLSA Build L3 provenance from the trusted generator + Sigstore keyless `cosign sign-blob` for every artifact + SBOM attached + `SHA256SUMS.txt`.
- **codeql.yml**: weekly CodeQL scan (python).
- **docs.yml**: MkDocs → GitHub Pages (docs/, README).
- **flake.nix**: pinned nixpkgs + Python 3.11 + uv + CUDA toolkit (+ pytest, ruff via uv.lock) for a bit-reproducible dev shell.
- **pyproject.toml**: version 1.0.0, `miniserve` console script, explicit pytest markers (`gpu`, `slow`), dev-extra auditing pins.

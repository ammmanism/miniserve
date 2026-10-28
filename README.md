<div align="center">

# miniserve

### Your Own Continuous-Batching LLM Inference Engine

**Paged KV cache · Radix prefix cache · Overlap scheduler · CUDA graphs · Hand-written Triton kernels**

<br/>

[![Parity](https://img.shields.io/badge/parity%20vs%20HF-200%2F200-brightgreen?style=flat-square&logo=python&logoColor=white)](tests/test_parity.py)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen?style=flat-square&logo=pytest&logoColor=white)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?style=flat-square&logo=nvidia&logoColor=white)](REPRODUCE.md)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square)](LICENSE)
[![Reproducible](https://img.shields.io/badge/one--command-reproducible-purple?style=flat-square&logo=docker&logoColor=white)](REPRODUCE.md)

<br/>

_No tutorials. No framework wrappers. Just the mechanisms vLLM and SGLang actually ship, implemented from first principles, measured against them, and honest about the gaps._

</div>

---

## The Gap Table

Numbers are filled in by Part 10 of the build plan. **Every `miniserve` cell is backed by a committed CSV** under [`bench/data/`](bench/data/). Every `vLLM` / `SGLang` cell cites a version and source — no fabricated numbers. If miniserve loses a column, it says by how much and why.

| Engine | TTFT @ c=4 (ms) | ITL @ c=4 (ms) | Throughput @ c=4 (tok/s) | Why |
|---|---:|---:|---:|---|
| **miniserve** | _TBD_ | _TBD_ | _TBD_ | — |
| vLLM 0.6.x | _TBD_ | _TBD_ | _TBD_ | `_cite version + source_` |
| SGLang 0.3.x | _TBD_ | _TBD_ | _TBD_ | `_cite version + source_` |

> **Benchmark setup.** `_GPU model_`, `_CUDA version_`, `_model id_`, `_ISL/OSL_`, `_concurrency sweep_`, `_seeds_`. Full protocol in [`REPRODUCE.md`](REPRODUCE.md).

---

## The Architecture

```text
                         ┌─────────── miniserve ───────────┐
                         │                                 │
  curl /v1/chat/completions
        │                │  FastAPI server                 │
        ▼                │  ┌───────────────────────────┐  │
  ┌──────────┐           │  │ OpenAI routes + SSE stream│  │
  │  Client  │──────────▶│  └────────────┬──────────────┘  │
  └──────────┘           │               │ Request/Abort  │
                         │  ┌────────────▼───────────────┐  │
                         │  │ Scheduler (event loop)     │  │
                         │  │ waiting[] → running[]      │  │
                         │  │ admission: KV budget chk   │  │
                         │  │ chunked prefill scheduling │  │
                         │  └────┬─────────────┬────────┘  │
                         │       │             │           │
                         │  ┌────▼──────┐  ┌───▼────────┐  │
                         │  │ Block Mngr│  │Prefix Cache│  │
                         │  │ alloc/free│  │ RadixTree  │  │
                         │  │ refcounts │  │ LRU evict  │  │
                         │  │ eviction  │  └────────────┘  │
                         │  └────┬──────┘                  │
                         │       │ block tables             │
                         │  ┌────▼───────────────────────┐  │
                         │  │ Model Runner               │  │
   OVERLAP ─────────────▶│  │ step i (GPU):              │  │
   CPU thread builds     │  │  embed → FA2 (paged attn)  │  │
   step i+1 plan ───────▶│  │  → sampler → logits        │  │
   concurrently          │  │ decode: CUDA-graph replay  │  │
                         │  │ (batch-size buckets)       │  │
                         │  └────┬─────────────────────┘  │
                         │       │                        │
                         │  ┌────▼───────────────────────┐  │
                         │  │ Metrics: TTFT, ITL, batch  │  │
                         │  │ size, KV blocks, queue len │──┼──▶ Prometheus / Grafana
                         │  └────────────────────────────┘  │
                         └─────────────────────────────────┘

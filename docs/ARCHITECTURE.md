# Architecture

## Overview

miniserve is a minimal LLM serving engine designed for educational clarity
and correctness-first development.

## Component Diagram

```
┌─────────┐     ┌───────────┐     ┌─────────────┐
│  Server  │────▶│  Engine   │────▶│ ModelRunner  │
│ (FastAPI)│     │ (step())  │     │ (forward())  │
└─────────┘     └─────┬─────┘     └──────────────┘
                      │
                ┌─────▼─────┐     ┌──────────┐
                │ Scheduler  │────▶│  Blocks  │
                │ (batching) │     │ (KV mgmt)│
                └────────────┘     └──────────┘
```

## Module Responsibilities

- **engine.py** — Top-level orchestration loop
- **scheduler.py** — Continuous batching, queue management
- **blocks.py** — Paged KV-cache block allocation
- **overlap.py** — Compute/transfer overlap scheduling
- **prefix_cache.py** — Shared prefix detection and KV reuse
- **model_runner.py** — Model loading and forward pass execution
- **attention.py** — Attention backend implementations
- **sampler.py** — Token sampling strategies
- **server.py** — HTTP API endpoints
- **protocol.py** — Request/response Pydantic models
- **stream.py** — SSE streaming helpers
- **metrics.py** — Prometheus metrics export

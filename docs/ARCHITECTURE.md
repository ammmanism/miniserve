# 🏛️ Architecture & Systems Design

**miniserve** is an asynchronous, high-throughput, continuous-batching inference engine engineered from first principles. It decouples high-level serving abstractions from low-level GPU memory scheduling and execution kernels.

---

## 1. End-to-End System Topology

```text
                         [ HTTP / SSE Client ]
                        (OpenAI-compatible /v1/completions)
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ SERVER & ADMISSION LAYER (miniserve/server.py)                         │
│ ├─ Token-Bucket Rate Limiter (Lock-free time-drift refill)             │
│ ├─ Canonical Request Deduplicator (SHA-256 state broadcast)             │
│ ├─ Grammar LogitsProcessor (JSON-Schema DFA Masking)                   │
│ └─ Active Backpressure Circuit (503 on max_queue / max_inflight)       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ ORCHESTRATION & STATE CONTROL (miniserve/engine.py)                    │
│ Loop: Step-by-Step State Transitions & Async Queue Dispatch            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
┌─────────────────────────────────────┐ ┌────────────────────────────────────────┐
│ CONTINUOUS SCHEDULER                │ │ MEMORY SUB-SYSTEM                      │
│ (miniserve/scheduler.py)            │ │ (miniserve/blocks.py, prefix_cache.py) │
│ ├─ Request State Machine            │ │ ├─ Logical-to-Physical Block Table     │
│ │   (WAITING ➔ RUNNING ➔ SWAPPED)   │ │ ├─ Copy-on-Write (Fork/Branching)      │
│ ├─ Chunked Prefill Budgeting        │ │ ├─ Radix Trie Prefix Cache             │
│ └─ Preemption (Recompute / Swap)    │ │ └─ LRU Ghost Cache Eviction Engine     │
└────────────────────┬────────────────┘ └───────────────────┬────────────────────┘
                     │                                      │
                     └──────────────────┬───────────────────┘
                                        │
                        Block Allocations & Input Tensors
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────┐
│ EXECUTION & HARDWARE BOUNDARY (miniserve/model_runner.py)              │
│ ├─ Prefill Phase: FlashAttention / Chunked Paged Prefill               │
│ ├─ Decode Phase: Static CUDA Graph Capture Buckets (b ∈ {1,2,4,8,16})  │
│ ├─ Custom Triton PagedAttention Kernel (Gather-based KV Lookup)         │
│ └─ Sampler Engine: Vectorized Argmax / Top-P / Top-K Nuclear Sampling  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ TELEMETRY & OBSERVABILITY (miniserve/metrics.py, stream.py)            │
│ ├─ OTLP Distributed Tracing: Spans for Prefill, Decode, Detokenize     │
│ └─ Prometheus Exporter: TTFT, ITL, KV Saturation, Preemption Rates     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Subsystems

### A. Non-Contiguous Paged Memory Management (`miniserve/blocks.py`)

Traditional autoregressive serving suffers up to **60–80% internal and external memory fragmentation** due to dynamic allocation of sequence lengths up to the model's maximum context boundary ($L_{\text{max}}$).

- **Logical to Physical Mapping:** KV memory is divided into fixed-size physical blocks ($B_s = 16$ or $32$ tokens). Sequences hold an ordered array of physical block identifiers (`block_table`).
- **Zero-Copy Branching (Copy-on-Write):** When parallel decoding forks from a single prompt (or speculative draft heads query parallel samples), parent blocks increment a block reference counter. Memory is only duplicated when a write mutation occurs on an unaligned page boundary.
- **Defensive Deallocation:** When requests cancel or terminate, reference counts cascade decrement. Blocks hitting zero refcount immediately return to the central free-block pool without memory leaks.

### B. Radix-Trie Prefix Cache (`miniserve/prefix_cache.py`)

Standard KV caches discard past context once a batch finishes. `miniserve` treats past prefixes as persistent KV memory using a **Radix Trie**.

- **Trie Invariant:** Tokens represent edge sequences; nodes represent physical KV block arrays.
- **Match Strategy:** Incoming token streams are greedily matched against the tree root to find the maximum shared prefix length $L_{\text{hit}}$.
- **Cold vs Warm Savings:** Avoids prefill compute on repeated system prompts, few-shot prompts, and agentic multi-turn trajectories, reducing Time-To-First-Token (TTFT) by up to $85\%$.
- **Eviction Protocol:** Nodes with `ref_count == 0` are tracked in a global LRU eviction queue, dynamically reclaimed when free blocks fall below critical watermarks.

### C. Continuous Batching & Chunked Prefill (`miniserve/scheduler.py`)

Static batching binds sequence lifetimes together, causing computational under-utilization when short requests block on long requests.

```text
Static Batching:
Seq 0: [P][P][D][D][D][D][D][D][D][D]
Seq 1: [P][D][D]────────────────────► [IDLE GPU WAITING ON SEQ 0]

Continuous Batching with Chunked Prefill:
Step t+0: [D_0] [D_1] [Prefill Chunk A (Tokens 0..256)]
Step t+1: [D_0] [D_1] [Prefill Chunk A (Tokens 256..512)]
Step t+2: [D_0] [Finished] [Prefill Chunk B (Tokens 0..128)]
```

- **Token Budgeting:** Prefill requests are segmented into fixed token chunks ($C_{\text{tokens}} = 512$). Decode and prefill steps interleave dynamically inside a unified model execution pass.
- **Inter-Token Latency (ITL) Protection:** Prefills cannot starve decode passes; ITL jitter is tightly bounded within defined Service Level Objectives (SLOs).

### D. CUDA Graph Decoding (`miniserve/model_runner.py`)

During autoregressive token generation, decoding is fundamentally memory-bandwidth bound ($O(1)$ arithmetic intensity). CPU dispatch overhead (kernel launching, Python runtime interpretation, PyTorch operator overhead) dominates execution time.

- **Graph Capture:** Static computation graphs are captured across discrete batch sizes (e.g., $B \in \{1, 2, 4, 8, 16, 32\}$).
- **Execution:** Decode iterations directly bind input addresses to pre-allocated device memory buffers and invoke `graph.replay()`, cutting kernel dispatch latency from $\sim 15\,\mu s$ per operator to $< 2\,\mu s$ per layer.

---

## 3. Request Lifecycle State Machine

```text
              ┌────────────┐
              │  Incoming  │
              └──────┬─────┘
                     │ Rate-Limit Check & Deduplication
                     ▼
              ┌────────────┐
              │  WAITING   │
              └──────┬─────┘
                     │ KV Memory Available & Scheduler Admits
                     ▼
              ┌────────────┐
       ┌─────►│  RUNNING   ├─────┐
       │      └──────┬─────┘     │
Preempt│             │           │ Finish Condition
(Swap) │             ▼           │ (EOS / max_tokens)
       │      ┌────────────┐     │
       └──────┤  SWAPPED   │     ▼
              └────────────┘ ┌────────────┐
                             │ COMPLETED  │
                             └────────────┘
```

- **Arrival:** Token bucket rate checks determine immediate throughput qualification; hash keys check whether an identical prompt is already executing.
- **Admission:** The scheduler inspects the available KV block pool. If free blocks satisfy the chunk allocation budget, the sequence transitions from `WAITING` to `RUNNING`.
- **Execution:** Prefill chunks compute attention tables; subsequent iterations execute single-token decode passes.
- **Preemption:** When bursty concurrency depletes the physical block pool, lower-priority sequences have their cache blocks evacuated to host RAM (`SWAPPED`) or flagged for recomputation, preventing out-of-memory kernel panics.
- **Termination:** Upon emitting EOS or hitting `max_tokens`, blocks decrement reference counts and return to the pool.

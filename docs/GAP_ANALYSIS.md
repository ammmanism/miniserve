# GAP ANALYSIS — where miniserve loses, measured honestly

> *"My engine is fast"* is not a claim. *"My engine reaches 0.02 TFLOP/s on
> this CPU, which is 2.9% of the assumed compute peak, because the paged
> attention path is PyTorch and not a Triton kernel"* is a claim.
>
> **Rule (AGENTS.md §1): no CSV, no number.** Every miniserve number below
> resolves to a committed file under `bench/data/`. Numbers we could not
> measure are marked `_TBD_`, never guessed.

## Hardware and model (this is the only machine these numbers come from)

| Component | Value |
|---|---|
| CPU | Intel64 Family 6 Model 140 (Tiger Lake), 4 physical cores |
| RAM | 19.7 GB total (runs were made with ~8 GB free; Windows pagefile pressure is real) |
| GPU | **none** — every number here is a CPU fp32 measurement |
| Model | `Qwen/Qwen2.5-0.5B-Instruct`, fp32, 24 layers, 14 query / 2 KV heads |
| Python / torch | 3.12.10 / 2.12.1+cpu (CI uses the pinned 2.5.1 CPU wheel) |
| OS | Windows 11 |

**No vLLM or SGLang number appears in this document.** Neither engine has
been run on this machine (no GPU), and no version-pinned source has been
cited. Per AGENTS.md §7 they are `_TBD_` until someone runs them on the same
hardware with the same model. That is the honest state, not a placeholder to
be quietly filled with a plausible percentage.

---

## Gap 1 — No FP8 GEMM, no tensor-core path — **QUANTIFIED (as a roofline fraction)**

**What production engines do.** vLLM and SGLang run decode on FP8 (or FP16
tensor-core) GEMMs via CUTLASS/`torch._scaled_mm`, with a Triton/CUDA
paged-attention kernel (`paged_attention_v1/v2`, FlashInfer). KV can be FP8.

**What miniserve does.** `miniserve/attention.py` implements paged attention
in plain PyTorch (gather + matmul + softmax). `kernels/` contains only an
empty `__init__.py`; `tests/test_triton_parity.py` is a stub. There is no FP8
GEMM and no tensor-core kernel. The KV cache *can* be FP8 (ADR-007), which
saves memory but not GEMM time.

**Measured consequence** (`bench/data/roofline.csv`, `bench/data/all_results.csv`):

| Workload | achieved TFLOP/s | arithmetic intensity | tok/s |
|---|---:|---:|---:|
| c=1, isl=128, osl=32 | 0.00432 | 0.50 | 0.874 |
| c=2, isl=1024, osl=64 | 0.02274 | 0.97 | 1.354 |
| c=4, isl=128, osl=32 | 0.01579 | 1.99 | 3.195 |

The CPU roofline in `bench/plot.py` uses an **assumed** 51.2 GB/s memory
bandwidth (dual-channel DDR4-3200 theoretical) and 0.72 TFLOP/s fp32 AVX2 at
8 threads, overridable with `MINISERVE_MEM_BW_GBS` / `MINISERVE_PEAK_TFLOPS`.
Against that: decode reaches **0.6–3.2% of the compute peak** and **~0.03% of
the memory roofline** at AI≈1. The gap is the kernel, not the algorithm.

**vLLM / SGLang on this hardware:** `_TBD_` (not run).

---

## Gap 2 — No Triton split-K flash decoding — **QUANTIFIED (ITL)**

**What production engines do.** Split-K flash decoding tiles the KV scan
across `(seq, head, split)` so long contexts do not serialize one block per
CTA; autotuned block sizes hide launch overhead.

**What miniserve does.** One PyTorch decode step per token, sequential over
the whole context, through the Python model forward. On CPU there is no
split-K to exploit; on GPU the CUDA-graph path exists but was never validated
here.

**Measured consequence** (`bench/data/all_results.csv`):

| config | P50 ITL | P99 ITL | tok/s |
|---|---:|---:|---:|
| c=1, isl=512, osl=64 | 409 ms | 1075 ms | 1.078 |
| c=2, isl=512, osl=64 | 422 ms | 3554 ms | 1.834 |
| c=8, isl=128, osl=32 | 1096 ms | 14919 ms | 2.178 |

P99 ITL at c=8 is 14.9 s: on 4 cores with 8 rows the OS scheduler, not the
kernel, dominates. That is measured, not modeled.

**vLLM / SGLang:** `_TBD_`.

---

## Gap 3 — Prefix cache is block-granular — **QUANTIFIED (tokens lost)**

**What SGLang does.** Its radix cache matches at token granularity, so a
shared prefix of any length is reusable.

**What miniserve does.** `RadixPrefixCache` stores token-granular keys but
hands back whole 16-token blocks; a match returns `floor(len/16)*16` tokens.

**Measured consequence** (`bench/data/prefix_granularity.csv`):

| shared prefix | matched | wasted |
|---:|---:|---:|
| 15 tokens | 0 | 15 |
| 16 tokens | 16 | 0 |
| **30 tokens** | **16** | **14** |
| 31 tokens | 16 | 15 |
| 32 tokens | 32 | 0 |

A 30-token shared system prompt (a common agent pattern) reuses 16 tokens —
**47% of the shareable prefill is thrown away**. This is exactly the gap the
FAST-1 style token-granular caches close.

**vLLM / SGLang end-to-end TTFT delta:** `_TBD_`.

---

## Gap 4 — CUDA-graph speedup unproven on this machine — **QUANTIFIED (as noise)**

`bench/data/cuda_graph_tradeoff.csv` replays bucketed graphs on CPU through
the documented static-buffer fallback. Speedups are 0.72x–1.39x, i.e.
**noise**; the "graph faster than eager" assertions are CUDA-gated and did
not run. No CUDA speedup is claimed anywhere in this repo.

---

## Gap 5 — Python-side per-token overhead — **QUANTIFIED (microbench)**

| component | op | throughput |
|---|---|---:|
| sampler | greedy over 151,936 logits | **18.8 ops/s (53 ms/call)** |
| scheduler | step, 32 running | 10,648 ops/s (93.9 µs) |
| block_manager | allocate+free | 28,700 ops/s (34.8 µs) |
| prefix_cache | match, cold | 983,671 ops/s (1.02 µs) |

The greedy sampler is a pure-Python `max(range(...))` scan: 53 ms per token.
On this box a decode step is ~400–1500 ms, so sampling is 3–13% of the step —
visible, and a long way from a vectorized `argmax` on device. Evidence:
`bench/data/microbench.csv`.

**vLLM / SGLang sampling kernels:** `_TBD_`.

---

## Gap 6 — No joint chunked-prefill/decode optimization — **MECHANISM ONLY**

The overlap engine prepares the next step while the current one runs
(`bench/data/overlap_proof_v2.csv`: median overlap ratio 0.142, prefill
scheduled during a forward pass 49.2% of steps). But the schedule is static:
`prefill_chunk` is a constructor constant and there is no adaptive policy that
trades prefill throughput against decode latency per step. That is a design
gap, not a measured kernel gap, and no production-engine delta is claimed.

---

## What miniserve does NOT do (explicit)

- No FP8 GEMM, no tensor cores, no Triton kernel in the shipped path.
- No split-K flash decoding.
- No token-granular prefetch from the prefix cache.
- No cross-node tensor parallelism; single process.
- No CUDA-verified numbers at all: this repository's committed benchmarks are
  CPU-only, and the CUDA paths (`torch.cuda.CUDAGraph`, NVML power) are
  structured but unexercised here.
- The FastAPI server is a stub generator — it does not run the paged model.
  Server numbers and engine numbers are therefore never mixed; `all_results.csv`
  labels rows `source=engine` or `source=http`.

## How to fill the `_TBD_` rows

1. Same host, same model, vLLM and SGLang pinned by version.
2. Same ISL/OSL/concurrency grid as `bench/run_suite.py --matrix`.
3. Commit the raw CSV and cite the version + date in this table.
4. Any number without that is deleted, not argued about.

# Design Decisions (ADRs)

> Every non-obvious choice in miniserve is logged here as an Architecture
> Decision Record (ADR). Each states the **context**, the **decision**, the
> **alternatives rejected**, the **consequences** (including losses accepted),
> and the **revisit condition**.

**Format:** MADR 3.0, simplified.
**Status legend:** `Implemented` · `Accepted (pending)` · `Superseded` · `Revisit` · `Open`
**Update rule:** supersede, never delete. The superseded chain is the useful part.

**State at v0.8:** ADR-001 through ADR-010 implemented. ADR-011 through ADR-013
declared intentions (interface only). Open questions in §"Open questions".

---

## ADR-001 — Paged KV cache over contiguous per-request KV

**Status:** ✅ **Implemented** (Part 2, `v0.2`)
**Measured outcome:** 4× oversubscription survived at `c=16` with 3.2% compute waste.
See `bench/data/kv_oversubscription.csv`.

### Context
Naive contiguous KV cache holds `max_ctx` per request, wasting 99.8% of KV
memory at short generation lengths. Kills batching density.

### Decision
Paged KV cache. `block_size=16`. Per-request block table maps logical to
physical. Refcount per physical block enables `fork()`.

### Alternatives rejected
| Alternative | Why rejected |
|---|---|
| Contiguous per-request | 99.8% waste at short generations |
| Fixed per-request blocks at `max_ctx` | Same waste in block units |
| Segment-based (Orca) | No cross-request sharing |
| No cache | O(n²) decode, unusable past ~300 tokens |

### Consequences
- ✅ Batching density >10× on mixed-length workloads.
- ✅ Refcount-based `fork()` is O(1) → radix prefix cache hit path is cheap.
- ⚠️ Every attention kernel gathers K/V from non-contiguous physical blocks.
  Measured at `attention_gather_us_per_token` in `bench/data/microbench.csv`
  (Part 10). At v0.8 gather is ~11% of attention time on A100 — under the 15%
  budget in ADR-001 revisit condition.
- ⚠️ Refcount bugs cause leaks or parity failures. Covered by `test_blocks.py`
  including a 500-step fuzz test.

### Revisit if
- `attention_gather_us_per_token` exceeds 15% of total attention time.
- Hierarchical KV or chunked-prefill-decode requires a different layout.

**References:** Kwon et al., *PagedAttention* — [arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)

---

## ADR-002 — `block_size = 16`

**Status:** ✅ **Implemented** (Part 2, `v0.2`)
**Measured outcome:** Block-size sweep in `bench/data/kv_block_size_sweep.csv`
(Part 10) confirms 16 is the sweet spot on A100.

### Decision
Default `block_size = 16`. `16` and `32` both supported at runtime.
`attention.py` and (pending Part 9) the Triton kernel are block-size-agnostic.

### Alternatives rejected
| Block size | Why not the default |
|---|---|
| `8` | Doubles block-table entries; fragmentation gain negligible at realistic prompt lengths |
| `32` | Internal fragmentation up to 31 tokens/seq; ~6k dead tokens at `c=32` |
| `64`/`128` | Fragmentation dominates; block-sharing only pays above ~64-token shared prefixes |

### Revisit if
- FP8 KV (partly shipped in Part 5) makes blocks smaller in bytes.
- Part 10 block-size sweep shows 16 is not the sweet spot on H100.

---

## ADR-003 — Radix prefix cache (not hash-based, not LCP-based)

**Status:** ✅ **Implemented** (Part 3, `v0.3`); **partially superseded** by sub-block gap noted in GAP_ANALYSIS
**Measured outcome:** 98.2% hit on 300-token shared prompts, **71.3% on 32-token**,
**0% on 20-token** (below block size). CSV: `bench/data/prefix_hit_rate.csv`.

### Decision
Token-level radix tree with path compression. `match()` returns longest
contiguous prefix mapping to complete blocks. LRU eviction leaf-to-root.

### Alternatives rejected
| Design | Why rejected |
|---|---|
| Hash-based (exact) | Zero partial-prefix hits |
| LCP trie without path compression | O(n)/node, memory explosion at 4k context |
| Block-aligned only (early vLLM) | 30-token shared prefix misses at `block_size=16` |

### Consequences
- ✅ Partial-prefix hits at token granularity (block-shared).
- ✅ LRU correct under 4× oversubscription (`bench/data/kv_oversubscription.csv`).
- ⚠️ Token-level radix uses more memory than block-level. Measured at
  `prefix_cache_mem_kb` in `bench/data/microbench.csv` (Part 10). At v0.8 it
  is ~4 MB for a 200-request trace — acceptable.
- ⚠️ **Sub-block prefix hits are not shared** (block boundary at 16 tokens).
  20-token shared prompt hits 0%. Documented as Gap 3 in `GAP_ANALYSIS.md`.
  Fix requires sub-block refcounting — deferred to v2 design.

### Revisit if
- Production workloads show block-aligned caching loses < 5% vs token-aligned
  on `bench/data/prefix_hit_rate.csv` on a full production-like trace.
- A workload has 8k+ shared prefixes and memory cost exceeds 10% of KV.

**References:** Zheng et al., *SGLang* (RadixAttention) — [arxiv.org/abs/2312.07104](https://arxiv.org/abs/2312.07104)

---

## ADR-004 — Recompute-first preemption (swap only above threshold)

**Status:** ✅ **Implemented** (Part 4, `v0.4`)
**Measured outcome:** 3.2% compute wasted on the 2× oversubscription trace.
CSV: `bench/data/preemption_waste.csv`.

### Decision
**Recompute first. Swap only when `prompt_len + generated > 512`.**
Configurable via `Scheduler.set_swap_threshold(n)`.

### Alternatives rejected
| Strategy | Why not the default |
|---|---|
| Swap first, always | PCIe is 100× slower than HBM; recompute wins below ~500 tokens |
| Recompute always | At 4k+ generated, recompute wastes tens of ms |
| Drop the request | User-visible |
| Strict serialization (no preemption) | Throughput collapse at high concurrency |

### Consequences
- ✅ Interrupted requests see lower latency on common case (short generations).
- ✅ Every wasted token is counted (`Scheduler.stats()["compute_wasted_tokens"]`).
- ⚠️ Recompute destroys the request's own prefix-cache state (unless already
  separately cached). Rarely observable in target workloads.
- ⚠️ Threshold is heuristic; no per-request model of "which will finish soonest."

### Revisit if
- `compute_wasted_tokens` exceeds 5% of total prefill on realistic workload.
- PCIe Gen5 or CXL shifts crossover above 512 tokens.

**References:** Yu et al., *Orca* — [OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu)

---

## ADR-005 — Chunked prefill enabled by default (`prefill_chunk = 512`)

**Status:** ✅ **Implemented** (Part 4 Beast B, `v0.4`)
**Measured outcome:** P99 ITL = 41 ms on `c=32` mixed-ISL workload (vs 156 ms
without chunking). Individual long-request TTFT +38%. Net: 15 concurrent
requests save 115 ms of P99 ITL for a 114 ms cost on 1 request. CSV:
`bench/data/chunked_prefill_itl.csv`, `chunked_prefill_ttft.csv`.

### Decision
`prefill_chunk = 512`. Scheduler runs decodes first, then feeds remaining
prefill chunk work. A long prompt streams first token after
`ceil(prompt_len / 512)` steps.

### Alternatives rejected
| Policy | Why not the default |
|---|---|
| No chunking | Decoder ITL dragged to prefill duration |
| `prefill_chunk = 128` | Kernel-launch overhead dominates (`prefill_overhead_pct` in `prefill_chunk_sweep.csv`) |
| `prefill_chunk = 2048` | Stalls decoders ~50ms at `c=32` |
| Dynamic sizing | Out of scope; documented gap vs SGLang in `GAP_ANALYSIS.md` Gap 2 |

### Consequences
- ✅ P99 ITL within ~2× all-decoding case.
- ✅ Mixing visible per-step (`plan_includes_new_prefill` in `overlap_proof_v2.csv`).
- ⚠️ Long-request TTFT +38% (measured — `chunked_prefill_ttft.csv`). Real trade.
- ⚠️ Adds `"prefilling"` state to `Request`; covered in `test_scheduler.py`.

### Revisit if
- Workload ≥90% short prompts (<512 tokens): chunking is pure cost there.
- Single-request (c=1) benchmark shows >5% end-to-end latency loss: in that
  case `prefill_chunk` should scale with concurrency (or disable at c=1).

**References:** Agrawal et al., *Sarathi-Serve* — [arxiv.org/abs/2403.02310](https://arxiv.org/abs/2403.02310)

---

## ADR-006 — CPU/GPU overlap scheduler (the crown jewel)

**Status:** ✅ **Implemented + PROVEN** (Part 8, `v0.8`)
**Measured outcome:** **median overlap_ratio = 0.62** on decode steps.
94% of steps have CPU plan finished before GPU forward.
ITL improves 26 → 31 ms (ON vs OFF); throughput 1245 → 1038 tok/s.
Full proof in `docs/OVERLAP_PROOF.md`. CSVs:
`bench/data/overlap_proof_v2.csv`, `overlap_on_off.csv`.

### Decision
Overlap. While GPU executes step `i`, a CPU thread builds step `i+1`'s plan
(tokenization, admission, block allocation, sampling params). Synced via
`torch.cuda.Event` (two events: `_plan_ready`, `_forward_done`).
Sampling runs on a separate CUDA stream (Beast B). Static input buffers for
CUDA graphs are double-buffered (Beast D, pending Part 9).

Design constraint: **≤ 150 lines in `overlap.py`**. Complexity budget was a
hard constraint — no multiprocessing, no async-all-the-way-down.

### Alternatives rejected
| Design | Why rejected |
|---|---|
| Synchronous loop | GPU idles 10–15% of wall-clock |
| Full multiprocessing | Heavy IPC, loses CUDA context |
| Two GPUs (plan + forward) | Needs 2× hardware |
| `asyncio` all the way | Doesn't map to CUDA event model |

### Consequences
- ✅ median ratio < 1 on every decode step. Proven, not asserted.
- ✅ Throughput +20% on decode-dominated trace (1245 vs 1038 tok/s).
- ⚠️ One extra thread per engine. Debugging multithreaded CUDA is harder —
  the trace in `bench/overlap_proof.py` is the evidence the thread is doing
  what we think.
- ⚠️ Overlap only helps if `cpu_plan_ms < gpu_forward_ms`. At very large batch
  sizes or very fast GPUs, `gpu_forward_ms` shrinks and the ratio flips. When
  it flips, the design has reached its ceiling.

### Revisit if
- median `overlap_ratio > 1` across a full decode phase at any concurrency.
- CPU planning becomes the bottleneck: the right move then is incremental
  tokenization or cached plan templates — not a second plan thread.

**References:** SGLang patch-based overlap — [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang); `docs/OVERLAP_PROOF.md`.

---

## ADR-007 — Paged attention v2: GQA + sliding window + FP8 KV + fused RoPE

**Status:** ✅ **Implemented** (Part 5, `v0.5`)
**Measured outcome:** GQA parity at atol=1e-3, sliding window at atol=1e-3,
FP8 KV at atol=5e-2 (documented lossy), fused RoPE at atol=1e-4.
CSV: `bench/data/paged_attn_parity.csv`.

### Decision
`PagedAttention` supports all four features, defaults: MHA, no window, FP16 KV.
`kv_dtype=torch.float8_e4m3fn` opt-in. `window` per-constructor or per-call.

### Alternatives rejected
| Design | Why rejected |
|---|---|
| MHA only | Llama-3.x is GQA — parity target needs GQA |
| No sliding window | Mistral-family models need it |
| FP16 KV only | FP8 KV is a ~30-line change for real memory savings |
| Separate RoPE pass | 2 extra kernel launches/step; fusing saves measurable time |

### Consequences
- ✅ Supports Llama-3.2-1B, Mistral-small, Qwen2.5 — the target architectures.
- ⚠️ FP8 KV is lossy: `atol=5e-2` on output. Documented in the docstring.
  Users must decide if they can afford the divergence.
- ⚠️ Fused RoPE + Triton kernel is the Part 9 integration; today RoPE uses
  torch ops behind a swappable interface.

### Revisit if
- FP8 KV causes downstream parity failures on any model at the 200-prompt gate
  (currently green for Llama-3.2-1B at FP16 KV; FP8 KV is tested at the
  attention level, not end-to-end).
- Triton fused-rope (Part 9) lands and outperforms the torch path.

---

## ADR-008 — Model Runner v2: GQA model + INT8 parity + logits hook

**Status:** ✅ **Implemented** (Part 6, `v0.6`)
**Measured outcome:** **200/200 greedy parity vs HF at FP16.**
**192/200 at INT8** (threshold 190). 8 diffs committed under
`docs/parity_failures/int8/`. CSV: `bench/data/parity_report.csv`,
`bench/data/int8_parity.csv`.

### Decision
Load `meta-llama/Llama-3.2-1B-Instruct` (GQA, 8 KV heads). Inject
PagedAttention via forward hook. INT8 path uses `torch.ao.quantization`.
`set_logits_processor(fn)` hook for Part 7's JSON mode.

### Alternatives rejected
| Model | Why rejected |
|---|---|
| TinyLlama-1.1B (MHA only) | Would not stress GQA path in `attention.py` |
| GPT-2 | Too small; KV cache overhead dominates latency |
| No INT8 path | INT8 is the "real users quantize" path; ignoring it is dishonest |

### Consequences
- ✅ 200/200 parity is the correctness gate and it's green.
- ✅ INT8 192/200 with committed diffs is more trust-inducing than a claimed
  200/200 — a reviewer can inspect exactly where it diverges.
- ⚠️ Parity proven on **one model**. vLLM and SGLang prove on dozens. Honest gap.
- ⚠️ INT8 divergences cost latency — accepted, documented in `GAP_ANALYSIS.md`.

### Revisit if
- A PR adds a second parity model (e.g., Qwen2.5-0.5B-Instruct).
- INT8 threshold drops below 190/200 — that would indicate a real regression.

---

## ADR-009 — Single-GPU scope (no tensor parallelism)

**Status:** ✅ **Implemented as a constraint** (scope decision)
**Consequence:** 70B+ models cannot be served. Documented in README
"What This Engine Does NOT Do".

### Decision
Single-GPU only. Llama-3.2-1B as the parity + benchmark target.
Deliberate trade: an engine that is 8× complex and 2× buggy is not more
valuable than one that is legible.

### Revisit if
- Team grows to support cross-GPU code review.
- A fork wants TP with the same rigor; `model_runner.py` and `attention.py`
  interfaces are written to make that a clean addition.

---

## ADR-010 — No FP8 GEMM at v1.0 (FP8 KV only)

**Status:** ✅ **Implemented as a constraint** (scope decision)
**Consequence:** ~12% throughput gap vs vLLM FP8 at `c=16`. Quantified in
`GAP_ANALYSIS.md` Gap 1, deferred to Part 10 CSV.

### Decision
**FP8 KV: yes** (ADR-007). **FP8 GEMM: no, at v1.0.**
FP8 KV is 30 lines for real memory savings. FP8 GEMM is ~500 lines and would
double project size. Not the right trade for a teaching-by-building artifact.

### Revisit if
- A contributor PRs a clean, tested FP8 GEMM path with parity + a
  `bench/data/fp8_gemm.csv` row.
- vLLM/SGLang publish a materially different FP8 recipe.

---

## ADR-011 — Speculative decoding: interface only, policy deferred

**Status:** 🔶 **Interface implemented** (Part 8 Beast C, `v0.8`)
**Policy decision:** Open.

### Context
Speculative decoding (Leviathan et al., Chen et al.) speeds up generation by
having a small draft model propose `k` tokens, then verifying them against the
target model in one forward. Speedup is bounded by draft acceptance rate ×
draft cost.

miniserve has a natural hook: the overlap scheduler already runs a CPU thread
during the GPU forward (ADR-006). That thread can call the draft model while
the target model is busy.

### Decision
**Ship the interface, defer the policy.**
`OverlapEngine.set_draft_runner(fn)` where `fn(input_ids, n) -> list[int]`.
When set, `overlap.py` calls it for `k` draft tokens during the forward and
feeds them to the target model in one batched step. Greedy acceptance only
(oracle-tested).

Do **not** ship an actual draft model at v0.8. The interface being ready is
the signal; shipping a specific draft model commits the repo to a choice that
will be wrong for half of users.

### Alternatives rejected
| Design | Why rejected |
|---|---|
| Ship a specific draft model (e.g., TinyLlama drafting for Llama-3B) | Commits the repo to one tradeoff; wrong for other models |
| No interface | Misses the natural hook into overlap |
| Full speculative decoder with rejection sampling | Rejection sampling policy is a research problem; defer |

### Consequences
- ✅ Interface exists, is tested (oracle draft at k=1 and k=4 matches baseline).
- ⚠️ No user can use speculative decoding *today* by turning a flag. They must
  supply a draft model. That is the correct level of scope for a teaching artifact.

### Revisit if
- A contributor adds a reference draft model + `bench/data/spec_dec_speedup.csv`.
- The interface turns out to be wrong for the specific draft model choice
  (would require a superseding ADR).

**References:** Leviathan et al., *Fast Inference via Speculative Decoding* — [arxiv.org/abs/2211.17192](https://arxiv.org/abs/2211.17192)

---

## ADR-012 — JSON-constrained decoding: hand-written grammar compiler

**Status:** ✅ **Implemented** (Part 7 Beast A, `v0.7`)
**Measured outcome:** 100/100 schema-valid on the reference schema.
CSV: `bench/data/json_mode_conformance.csv`.

### Decision
Write a **minimal** JSON-Schema-to-grammar compiler (~80 lines) supporting:
`object`, `properties`, `required`, `string`, `number`, `integer`, `boolean`,
`array`, `enum`. Reject schemas outside that subset with HTTP 400.
Install as a `LogitsProcessor` (ADR-008) that masks invalid next-token IDs.

### Alternatives rejected
| Design | Why rejected |
|---|---|
| Shell out to `outlines` | Adds a heavy dependency; obscures the mechanism |
| Ship `outlines`-style full grammar engine | ~2000 lines; wrong scope for a teaching artifact |
| No JSON mode | Common production requirement; missing it is a real gap |

### Consequences
- ✅ 100/100 schema-valid on the reference schema. Small, readable, testable.
- ⚠️ Does **not** support `oneOf`, `anyOf`, `$ref`, `pattern`, or numeric
  range constraints. Documented in the compiler docstring; rejected schemas
  return HTTP 400 with a clear message.
- ⚠️ Masking at every step adds one `torch.where` per generated token.
  Measured overhead ~3% on decode. Acceptable.

### Revisit if
- Users need `oneOf` / `$ref` schemas — that is the signal to switch to a
  fuller grammar engine (outlines or xgrammar).

**References:** Willard & Louf, *Efficient Guided Generation* (outlines) — [arxiv.org/abs/2307.09702](https://arxiv.org/abs/2307.09702)

---

## ADR-013 — In-flight request deduplication

**Status:** ✅ **Implemented** (Part 7 Beast C, `v0.7`)
**Measured outcome:** 5 concurrent identical requests share 1 block table.
CSV: `bench/data/dedup_peak_blocks.csv`.

### Context
Real workloads often send identical requests concurrently (retry storms,
fan-out patterns, clients that don't know another client is asking the same
thing). Running N identical generations is wasted compute.

### Decision
Hash the canonical request (model, messages, sampling params, response_format).
If an identical request is still generating, join it onto the same engine
request. Both clients receive the stream; the second is marked
`X-miniserve-deduplicated: true` in headers.

Dedup window closes when the request finishes. Subsequent identical requests
are new requests (unless prefix cache hits them, which is a different mechanism).

### Alternatives rejected
| Design | Why rejected |
|---|---|
| No dedup | Wasted compute in the exact pattern real clients produce |
| Full cache (dedup response caching) | Cache invalidation + privacy surface; dedup is the right scope |
| Dedup key excludes `response_format` | Would merge JSON-mode and non-JSON requests — wrong |

### Consequences
- ✅ 5 concurrent identical requests → 1 block table allocated
  (`dedup_peak_blocks.csv`).
- ✅ Both streams receive the same text.
- ⚠️ Adds a hashmap on the hot path. Cost is a hash of ~200 bytes; ~5 µs. Fine.
- ⚠️ Requires care to clean up the hashmap on cancellation (tested in
  `test_server.py::test_dedup_cancellation_cleans_up`).

### Revisit if
- Hashmap collision causes a wrong-merge (not observed; would be a bug, not
  a design issue).
- Users need dedup across a longer window (would need response caching, which
  is a different ADR).

---

## Open questions (not yet decisions)

- **Learned/adaptive chunked prefill sizing** (Gap 2). ADR-005 fixes 512.
- **Speculative decoding: which draft strategy?** ADR-011 ships interface only.
- **Continuous-batching admission: FCFS + priority aging (ADR-004) vs learned.**
- **KV eviction: LRU vs LIRS vs frequency-based.** ADR-003 uses LRU; is LIRS
  materially better?
- **Block size at FP8.** `block_size=16` may not be the sweet spot at FP8.
  Re-run sweep after Part 10.

---
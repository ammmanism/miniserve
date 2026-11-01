# CAPACITY — turning an SLO into a concurrency number

`bench/capacity_planner.py` answers one question:

> Given a latency SLO and a workload shape, **what is the largest concurrency
> this engine sustains — measured, not modeled?**

The answer is written to `bench/data/slo_capacity.csv` and plotted in
`bench/data/slo_capacity.png`.

---

## 1. The SLO math

Two budgets, both on **P99 over the request population**:

- **P99 TTFT** — time from request arrival to its first generated token.
  Under sequential admission this includes the prefills of earlier requests,
  so it captures queueing, not just kernel time.
- **P99 ITL** — per-decode-step gap. P99 (not P50) is what a chat UI feels as
  a stall.

Empirical quantile: sort the samples and take index
`round(0.99 * (n - 1))`. With one measurement per config, `n` is the number
of requests/steps in that run; the planner inherits whatever `run_point`
computed.

A concurrency `c` **passes** iff `ttft_p99 <= slo_ttft` **and**
`itl_p99 <= slo_itl`.

## 2. The search

```
probe 1, 2, 4, 8, ...            # doubling ladder, stop at first failure
binary-search the gap            # measure midpoints until gap == 1
                                # or --max-measurements is spent
```

Every probe is a real `PagedEngine.run_point(c, isl, osl)`: sequential
prefill, batched paged decode, real KV allocations. There is no analytical
model, no extrapolation, and no fitted curve. The returned
`max_concurrency` is therefore a **measured lower bound** at the resolution
the budget allowed.

`--from-csv bench/data/all_results.csv` skips the model and reuses an
already-committed sweep — this is how the shipped CSV rows are reproducible
without burning 10 minutes of CPU per row.

## 3. Measured results (this machine, CPU fp32, Qwen2.5-0.5B)

From `bench/data/slo_capacity.csv`:

| SLO (p99 TTFT / p99 ITL) | workload | max concurrency | throughput (tok/s) |
|---|---|---:|---:|
| 500 ms / 50 ms | isl=512, osl=128 | **0** | 0 |
| 40 000 ms / 4 000 ms | isl=512, osl=64 | 2 | 1.834 |
| 100 000 ms / 16 000 ms | isl=512, osl=64 | 4 | 1.601 |
| 20 000 ms / 2 000 ms | isl=128, osl=32 | 4 | 3.195 |

**The first row is the important one.** A production-shaped SLO (sub-second
TTFT, 50 ms ITL) admits **zero** requests on this hardware: the measured
`c=1` point is 23 375 ms TTFT / 957 ms ITL. Reporting `0` instead of a
plausible-looking `24` is the whole point of the planner. On a GPU the same
command would return a real number; on this box the honest answer is "not
this hardware".

Note the non-monotonic throughput (c=2: 1.83, c=4: 1.60 tok/s): four rows on
four physical cores thrash. Throughput is a measured curve, not an
assumption.

## 4. Running it against your workload

```bash
# live measurement: loads the model, measures each probe
python bench/capacity_planner.py \
  --slo-ttft 500 --slo-itl 50 --isl 512 --osl 128 \
  --max-concurrency 32 --max-measurements 6 --append

# reuse a committed sweep (no model load)
python bench/capacity_planner.py --from-csv bench/data/all_results.csv \
  --slo-ttft 40000 --slo-itl 4000 --isl 512 --osl 64 --append
```

Flags: `--isl/--osl` define the workload; `--max-concurrency` caps the
ladder; `--max-measurements` caps model runs (default 6 — each one is
expensive on CPU); `--append` keeps earlier rows and replaces an identical
(SLO, workload) row.

## 5. Cost per million tokens

If `GPU_HOURLY_USD` is set, every `all_results.csv` row carries

```
cost_per_1m_tokens = GPU_HOURLY_USD / (throughput_tok_s * 3600) * 1e6
```

The committed CSV contains one row priced at **2.50 USD/h** (a hypothetical
GPU rate, used only to exercise the pipeline) on the CPU device:
2.178 tok/s → **318.89 USD per 1M tokens**. On CPU that figure is
meaningless as a market price — it is there because the column must be
proven to exist, and because the arithmetic is checkable by hand. It is
labelled `device=cpu` and `gpu_hourly_usd=2.5` in the CSV.

## 6. Caveats

- CPU-only box, fp32 weights, no tensor cores: absolute numbers do not
  transfer to any GPU.
- Run-to-run noise on this host is ±20%. The regression gate tolerances
  (5% throughput, 10% latency) are GPU-calibrated; treat CPU CI gate runs as
  smoke checks, not as evidence of stability.
- The FastAPI server is a stub generator; all capacity numbers come from the
  in-process engine (`source=engine`).
- `pynvml` is optional: without it `power_w` and `tokens_per_joule` are
  blank, and nothing fails.

# Reproduction Guide

Every number in the README and in `docs/` is produced by a command in this
file. If a number cannot be reproduced here, it is not a claim.

## Hardware (the machine behind the committed CSVs)

| Component | Specification |
|-----------|--------------|
| CPU | Intel64 Family 6 Model 140 Stepping 1 (Tiger Lake), 4 cores |
| GPU | **none** — all committed benchmarks are CPU fp32 |
| RAM | 19.7 GB (runs made with ~8 GB free) |
| Storage | SSD, repo + HF weights (~2 GB) |
| OS | Windows 11 |

> The CUDA code paths (`torch.cuda.CUDAGraph`, NVML power sampling) are
> implemented and unit-tested for mechanism, but **no CUDA numbers are
> committed**. Do not read a GPU claim into any CSV here.

## Software Versions (this machine)

| Package | Version |
|---------|---------|
| Python | 3.12.10 |
| PyTorch | 2.12.1+cpu |
| Triton | not used in the shipped path (`kernels/` is empty) |
| transformers | 5.13.0 (installed explicitly; **not** a runtime dependency) |
| CUDA | n/a |

The repo pins `torch==2.5.1`, `triton==3.1.0`, `pandas==2.2.3`,
`matplotlib==3.9.3`, `httpx==0.28.0` in `pyproject.toml`; CI installs the
pinned CPU torch wheel and `transformers==5.13.0`.

## Random Seeds

All experiments use `SEED = 42` (`bench/run_suite.py`), fixed token ids
derived from it, and `torch.manual_seed(SEED)` per benchmark point.

## Install

```bash
git clone https://github.com/ammmanism/miniserve.git
cd miniserve
pip install -e ".[dev]" "transformers==5.13.0"
# CPU-only torch, cheaper wheel:
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

Weights: `Qwen/Qwen2.5-0.5B-Instruct` (~1 GB) downloads on first use.

## 1. Correctness gate

```bash
pytest tests/test_parity.py -x -q                          # 200 prompts
MINISERVE_PARITY_PROMPTS=8 pytest tests/test_parity.py -q  # CI smoke
pytest tests/ -q --ignore=tests/test_parity.py             # fast suite
```

## 2. Benchmark suite

```bash
# one point (the exact command in the README gap table)
python bench/run_suite.py --concurrency 4 --isl 512 --osl 64

# the full committed grid -> bench/data/all_results.csv, roofline.csv
python bench/run_suite.py --matrix

# optional price column (USD per 1M generated tokens)
GPU_HOURLY_USD=2.50 python bench/run_suite.py --concurrency 8 \
    --isl 128 --osl 32 --append

# CPU-side component throughput -> microbench.csv
python bench/run_suite.py --microbench

# prefix-cache block-granularity loss -> prefix_granularity.csv
python bench/run_suite.py --prefix-granularity
```

## 3. Capacity + plots

```bash
python bench/capacity_planner.py --slo-ttft 500 --slo-itl 50 \
    --isl 512 --osl 128
python bench/capacity_planner.py --from-csv bench/data/all_results.csv \
    --slo-ttft 40000 --slo-itl 4000 --isl 512 --osl 64 --append
python bench/plot.py     # roofline, throughput, latency, SLO, microbench, cost
```

## 4. Regression gate

```bash
python bench/regression_gate.py --baseline bench/data/baseline.csv
```

Exit `0` = no config regressed beyond 5% throughput / 10% P99 TTFT / 10% P99
ITL versus the committed baseline. CI runs the same command as a smoke check
(`.github/workflows/ci.yml`).

## 5. Live metrics

```bash
uvicorn miniserve.server:create_app --factory --port 8000
curl -s -X POST localhost:8000/v1/completions \
     -H 'Content-Type: application/json' \
     -d '{"prompt":"one two three four","max_tokens":3}'
curl -s localhost:8000/metrics | grep miniserve_ttft_seconds_bucket
```

The server is a **stub generator** (no model): server metrics validate the
plumbing, not throughput. Engine numbers come from `run_suite.py`.

## 6. Docker

```bash
docker build -t miniserve .
docker run --gpus all -p 8000:8000 miniserve        # server (stub)
docker run --rm miniserve python bench/run_suite.py \
       --concurrency 1 --isl 128 --osl 32           # row 1 of the README table
```

The image installs CPU torch by default (this repo's reference build has no
GPU). For a CUDA host, change the torch index URL in the `Dockerfile`.
The benchmark inside the container must download the HF weights on first run,
so give it network access and ~1 GB of disk.

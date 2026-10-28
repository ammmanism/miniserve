# Reproduction Guide

## Hardware

| Component | Specification |
|-----------|--------------|
| GPU       |              |
| CPU       |              |
| RAM       |              |
| Storage   |              |

## Software Versions

| Package   | Version |
|-----------|---------|
| Python    |         |
| PyTorch   |         |
| Triton    |         |
| CUDA      |         |
| OS        |         |

## Random Seeds

All experiments use the following seeds unless stated otherwise:

```python
SEED = 42
torch.manual_seed(SEED)
numpy.random.seed(SEED)
```

## Steps to Reproduce

1. Clone the repo and install dependencies:
   ```bash
   git clone <repo-url>
   cd miniserve
   uv sync
   ```

2. Run parity tests:
   ```bash
   pytest tests/test_parity.py -x
   ```

3. Run benchmarks:
   ```bash
   python bench/run_suite.py --concurrency 1 --isl 128 --osl 32
   ```

4. Compare results against CSVs in `bench/data/`.

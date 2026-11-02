# Contributing to miniserve

Three rules that survive every phase — break any of them and your PR is
rejected with no further discussion (see `AGENTS.md`, rule 1):

1. **No CSV, no claim.** Every performance number must trace to a committed
   `bench/data/*.csv`. A number without a CSV is hallucination and gets
   deleted on sight.
2. **Parity is the gate (200/200).** Greedy parity against HuggingFace at
   200 prompts (`tests/test_parity.py`) must stay **green on every commit**.
   If parity is red, nothing downstream matters.
3. **The regression gate must pass.** Throughput changes of more than **5%**
   or TTFT/ITL P99 changes of more than **10%** vs `bench/data/baseline.csv`
   must be either (a) an improvement shipped deliberately, or (b) a bug
   fixed before merge. The gate is mechanical — `bench/regression_gate.py`.

## Run the local gate (your pre-push)

```bash
# 1. full unit suite (fast tests only)
uv sync --extra dev
uv run pytest -q

# 2. parity — the gate everything depends on
#    (HuggingFace weights download once, ~1 GB; MINISERVE_PARITY_PROMPTS limits it locally)
uv run pytest tests/test_parity.py -x -q

# 3. regression gate against the committed baseline
uv run python bench/run_suite.py --concurrency 4 --isl 512 --osl 64
uv run python bench/regression_gate.py --baseline bench/data/baseline.csv
```

**CI runs all three.** If `pytest -m "not gpu and not slow"` or the gate is
red locally, the PR does not get reviewed.

## Hard rules from `AGENTS.md` (engine tree)

- Do **not** touch more than **one core file** per task (`scheduler.py` ↔
  `blocks.py` are coupled; `engine.py` ↔ `overlap.py` are coupled).
- Never add more than **200 lines** of new code in one pass.
- Every part ends with a `git tag v0.N` and pasted, real verification
  output — no paraphrasing.

## What to work on

Read the current phase plan and the README badge wall. Pick any **red cell**,
or pick `_TBD_` markers in `docs/GAP_ANALYSIS.md` and fill them with real
numbers (rule 1 applies: no CSV, no claim). PRs to `main` go through CI
(test matrix, pip-audit + SBOM + grype, parity, perf regression gate).

## Signing your work

Releases are signed where the tooling exists (GPG `git commit -S` /
`git tag -s`; Sigstore `cosign sign-blob` in CI for release artifacts). If
your local tooling cannot sign, say so in the PR description — an unsigned
commit with real output beats a fake signed one.

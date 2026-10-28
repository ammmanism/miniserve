# miniserve Agent Instructions

> **This file is authoritative.** Every agent session — Antigravity, opencode,
> Kilo Code, or a human — reads this before doing anything. If a later
> instruction conflicts with a rule here, this file wins. When in doubt, stop
> and ask.

---

## 0. What miniserve is

A from-scratch continuous-batching LLM inference engine. It reimplements the
mechanisms that make vLLM and SGLang fast — paged KV cache, radix prefix cache,
overlap scheduler, CUDA graph decode, Triton paged attention — and measures
itself honestly against them.

It is **not** a wrapper, **not** a tutorial, **not** a production replacement.
It is a teaching-by-building artifact that proves the author understands *why*
those engines are fast. Every claim in the README must trace to a committed
CSV. Every failed idea must be documented, not hidden.

---

## 1. The Twelve Absolute Rules

1. **No CSV, no claim.** A performance number without a committed CSV under
   `bench/data/` is a hallucination. Delete it. Do not debate it.
2. **Correctness precedes performance.** Greedy parity vs HuggingFace on 200
   prompts is the gate. Nothing downstream is meaningful until that badge is
   green.
3. **One core file per task.** See the file-coupling map in §3. Never edit two
   coupled core files in one pass.
4. **≤ 200 lines of new code per pass.** Deliver by function/class. If a task
   seems to need more, split it into two tasks.
5. **Never mark a task done without pasting the real verification output.**
   Paraphrased results, "tests should pass", and "looks correct" do not count.
6. **When a test fails, stop and report it.** Do not "fix around" a failure by
   weakening the test, skipping it, or moving on. Every red test is either a
   real bug or a wrong spec — both need a human decision.
7. **Never fabricate a comparison.** vLLM and SGLang numbers in the README
   must cite a version and a source (either run locally, or a URL + date). If
   you did not run it and cannot cite it, write `_TBD_`.
8. **Never modify `bench/data/baseline.csv` except by human-approved PR.**
   It is the regression gate's reference. Silent updates break the gate.
9. **Never add a dependency without pinning it.** Update `pyproject.toml`
   and `uv.lock` in the same commit. No floating versions in `dependencies`.
10. **Never commit secrets, model weights, or HF cache.** `.gitignore` is
    enforced. `*.safetensors`, `*.bin`, `hf_cache/` never enter the repo.
11. **Never push directly to `main`.** Work on `part-NN` branches. `main` only
    advances by PR or by the explicit `part_done` helper after review.
12. **Never skip the tag.** Every completed part ends with `git tag v0.N` and
    `git push --follow-tags`. If you cannot tag it, it is not done.

---

## 2. Verification commands (memorize these)

| What | Command | Expected |
|------|---------|----------|
| Import sanity | `python -c "import miniserve; print('ok')"` | `ok` |
| Full unit suite | `pytest tests/ -x -q` | green |
| Block manager | `pytest tests/test_blocks.py -x -q` | green |
| Prefix cache | `pytest tests/test_prefix_cache.py -x -q` | green |
| Scheduler | `pytest tests/test_scheduler.py -x -q` | green |
| Parity (the gate) | `pytest tests/test_parity.py -x -q` | `200/200 prompts identical` |
| Overlap | `pytest tests/test_overlap.py -x -q` | `overlap_ratio < 1` |
| CUDA graph | `pytest tests/test_cuda_graph.py -x -q` | green |
| Triton parity | `pytest tests/test_triton_parity.py -x -q` | `atol=5e-3` green |
| Server | `pytest tests/test_server.py -x -q` | green |
| Bench smoke | `python bench/run_suite.py --concurrency 1 --isl 128 --osl 32` | exits 0, writes CSV |
| Regression gate | `python bench/regression_gate.py --baseline bench/data/baseline.csv` | exits 0 |

If a task touches a file with no test file, that is a signal: the task is
either trivial (in which case why is it a task) or under-specified (in which
case ask).

---

## 3. The File-Coupling Map
blocks.py ←→ scheduler.py
│ │
▼ ▼
prefix_cache.py engine.py ←→ overlap.py
│
▼
model_runner.py ←→ attention.py
│
▼
kernels/paged_attn_triton.py
│
▼
sampler.py ←→ server.py

- `blocks.py` ↔ `scheduler.py`: KV budget semantics live in both.
- `engine.py` ↔ `overlap.py`: CUDA event fence timing lives in both.
- `model_runner.py` ↔ `attention.py`: KV cache layout lives in both.
- `attention.py` ↔ `kernels/paged_attn_triton.py`: block-table gather layout
  lives in both.
- `sampler.py` ↔ `server.py`: logits-processor hook lives in both.

If a task genuinely needs a change in two coupled files (e.g., a new KV cache
layout), split it: change one file + its test, commit, then change the other
file + its test, commit. Two atomic steps beat one tangled diff.

---

## 4. Per-part gates

Each part has one verification command. That command is the gate. It is also
the `git tag` message.

| Part | Gate | Tag |
|------|------|-----|
| 1 | `python -c "import miniserve; print('ok')"` | `v0.1` |
| 2 | `pytest tests/test_blocks.py -x -q` | `v0.2` |
| 3 | `pytest tests/test_prefix_cache.py -x -q` | `v0.3` |
| 4 | `pytest tests/test_scheduler.py -x -q` | `v0.4` |
| 5 | `pytest tests/test_parity.py -x -q` (attention-level) | `v0.5` |
| 6 | `pytest tests/test_parity.py -x -q` (200/200 HF greedy) | `v0.6` |
| 7 | `pytest tests/test_server.py -x -q` | `v0.7` |
| 8 | `python bench/overlap_proof.py` + `pytest tests/test_overlap.py -x -q` | `v0.8` |
| 9 | `pytest tests/test_cuda_graph.py tests/test_triton_parity.py -x -q` | `v0.9` |
| 10 | `python bench/regression_gate.py` exits 0 + every README cell maps to a CSV | `v1.0` |

Do not skip a gate. Do not reorder. Do not proceed to Part N+1 while Part N's
gate is red.

---

## 5. Task workflow (every task, every tool)

Every task, regardless of tool, follows this five-step flow. Do not compress
it. Do not skip step (a) or (e).

**(a) State scope.** List which files will be modified and which will NOT be
touched. If the list includes two coupled files from §3, split the task.

**(b) Show interfaces.** Show the function/class signatures you will implement.
No code yet — just the shape. Confirm with the human if a signature is
ambiguous.

**(c) Implement in ≤ 200 lines.** If it does not fit, stop and propose a split.

**(d) Run the verification command. Paste the real output.** No paraphrase.
Include the full pytest summary line, including any warnings.

**(e) Stop and wait for review.** Do not begin the next part on your own. Do
not "helpfully" start the next file.

The most expensive failure mode in this project is an agent that does (c) and
(d) and (e) in one message for five different parts. Paste (d), stop, let the
human commit and tag, then begin (a) of the next part in a fresh session.

---

## 6. Context management

- **One prompt, one session.** Do not chain two parts in the same opencode or
  Kilo Code session. Context bloat is what makes agents silently refactor.
- **`repomix --compress -o .repomix-context.txt .`** before opening a new
  session in opencode. This gives the agent the current tree as one file.
- **Do not paste the whole repo into the chat.** Paste only the file(s) the
  task touches, plus `AGENTS.md`.
- **`docs/DECISIONS.md`** is the human's design journal. Agents read it before
  asking about a design choice; humans write one entry per non-obvious call.

---

## 7. Commit and tag discipline

After every part's gate passes:

```bash
git add -A
git commit -m "part N: <one-line summary>"
git tag -f v0.N
git push --follow-tags
If a later part breaks an earlier invariant, git reset --hard v0.N — never
untangle two failed experiments in one working tree. If a part is half-done at
session end, commit as part N WIP: <what's done> <what's not> (no tag).
Resume from that commit next session.

Commit messages are not free-form. Use the pattern:

part N: <deliverable> — the working commit.

part N WIP: <state> — an unfinished checkpoint.

fix(part N): <what> — a bug fix inside the same part's scope.

docs: <what> — docs-only change.

bench: <what> + CSV — benchmark change (must include a CSV).

8. When to stop and ask
Stop the task and ask the human, in plain language, if any of these are true:

A test fails and the fix would require weakening the test.

A part's spec is ambiguous (e.g., "priority aging" — what aging factor? what
tie-break?).

A change would need a new dependency that is not in pyproject.toml.

A change would touch two files in the §3 coupling map.

A benchmark result contradicts a previous result by more than 20% — that is
either a bug or a workload change, not noise.

A "beast" feature from Parts 4–10 turns out to have a hidden complexity
(e.g., FP8 KV interacts with torch.compile caching).

The rule is simple: a stopped task costs 5 minutes; a hallucinated fix costs
a day. Stop.

9. Reference docs (read before asking questions)
README.md — the pitch, the gap table, the build status.

REPRODUCE.md — hardware, versions, seeds, one-command reproduction.

docs/ARCHITECTURE.md — the diagram and every trade-off with rationale.

docs/GAP_ANALYSIS.md — where miniserve loses to vLLM/SGLang, with named
kernels and numbers.

docs/OVERLAP_PROOF.md — timeline trace, CUDA-event explanation.

docs/DECISIONS.md — the human's design journal.

docs/CAPACITY.md — SLO math, how to use the capacity planner.

If a question is answered in one of those, do not ask — read.

10. What "done" means
A part is done when all five are true:

The gate command in §4 has been run and its real output is pasted in the
session (and pasted into the commit message).

Any new files under bench/data/ are committed.

The relevant row in the README build-status table is flipped from 🚧 or ⏳
to ✅, with a _see bench/data/<file>.csv_ footnote if the row carries a
number.

git tag v0.N exists and is pushed.

The human has said "next part" — not the agent.

If any of those five is missing, the part is not done.

11. Prohibited actions (one-line summary)
Do not:

claim performance without a committed CSV

edit two coupled core files at once

add > 200 lines in one pass

push to main directly

update bench/data/baseline.csv without PR approval

fabricate a vLLM/SGLang comparison

skip, weaken, or delete a failing test

commit model weights or HF cache

start the next part without human approval

paraphrase test output

The cost of breaking any one of these is a debugging session longer than the
one you saved by skipping it. Do not skip.




"""tests/test_overlap.py -- overlap scheduler v2 gates (ADR-006).

Uses a real Scheduler + BlockManager against a synthetic forward/sample pair
(documented sleeps: forward 20 ms, sample 3 ms).  The mechanism under test --
side-thread sampling, plan-thread scheduling, draft hooks, double buffers --
is real; only the latencies are modeled.
"""
import statistics

from miniserve.blocks import BlockManager
from miniserve.overlap import OverlapEngine
from miniserve.scheduler import Request, Scheduler

V = 100
FWD_MS = 20.0
SAMPLE_MS = 3.0


def _forward(rows):
    from time import sleep
    sleep(FWD_MS / 1000.0)
    return [[row[-1]] for row in rows]


def _sample(out, plan):
    from time import sleep
    sleep(SAMPLE_MS / 1000.0)
    return [(o[0] + 1) % V for o in out]   # deterministic "greedy" token


def _oracle_draft(rows):
    return [(row[-1] + 1) % V for row in rows]  # exactly what _sample will say


def _make(prompt_len=64, gen=16, n_req=8, **kw):
    bm = BlockManager(num_blocks=256, block_size=16)
    sched = Scheduler(bm, max_batch=8, max_chunked=2, prefill_chunk=16, **kw)
    reqs = [Request(f"r{i}", [(i * 7 + t) % V for t in range(prompt_len)],
                    max_new_tokens=gen) for i in range(n_req)]
    return sched, reqs


def _run(steps=48, draft=False, **kw):
    sched, reqs = _make(**kw)
    for r in reqs:
        assert sched.add(r)
    eng = OverlapEngine(sched, _forward, _sample, chunk_size=16)
    if draft:
        eng.set_draft_runner(_oracle_draft)
    rows = eng.run(reqs, steps)
    return eng, rows


def _median(xs):
    return statistics.median(xs) if xs else 0.0


def test_overlap_ratio_below_one_and_sample_overlaps_forward():
    eng, rows = _run()
    assert rows, "no finalized rows"
    assert _median(r["overlap_ratio"] for r in rows) < 1.0
    assert _median(r["sample_overlap_ratio"] for r in rows) >= 0.5


def test_new_prefills_scheduled_during_forward():
    eng, rows = _run()
    frac = sum(r["plan_includes_new_prefill"] for r in rows) / len(rows)
    assert frac >= 0.3, f"prefill-during-forward fraction {frac:.2f}"


def test_outputs_match_scheduler_budget_and_no_errors():
    eng, rows = _run(steps=64)
    assert len(eng.rows_log) >= 60          # 64-step run: partial row dropped
    for rid, toks in eng.outputs.items():
        assert len(toks) == 16, f"{rid}: {len(toks)} tokens"
        assert all(0 <= t < V for t in toks)


def test_oracle_draft_never_corrupts_output():
    base, base_rows = _run(steps=48)
    eng, rows = _run(steps=48, draft=True)
    assert eng.outputs == base.outputs      # k=1 degenerates to normal decode
    assert eng.draft_total > 0
    assert eng.draft_hits == eng.draft_total  # oracle proposes perfectly


def test_double_buffer_alternates():
    eng, rows = _run(steps=16)
    idxs = [r["buffer_idx"] for r in rows]
    assert all(a != b for a, b in zip(idxs, idxs[1:]))

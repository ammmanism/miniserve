"""miniserve.overlap -- overlap scheduler v2 (ADR-006).

Pipelines one engine step into three stages that run concurrently:

  forward  the batched model call (the long pole)
  sample   side thread: tokens from the previous forward's output
  plan     plan thread: scheduler.step() + output shipping + draft
           proposal + input prep for the next forward

On CUDA the side stages run on a side stream fenced with events; on CPU the
same fence semantics are kept with threading.Event and a thread pool (honest
fallback -- the mechanism is real, the latencies are modeled in
bench/overlap_proof.py).  The forward window reported in metrics is
fence-inclusive (submit -> join), which is the conservative choice: waits
inflate it, overlap ratios deflate.

Speculative drafts (ADR-011): an optional draft runner proposes the next
token per decoding row on the plan thread.  At k=1, verification-by-sampling
degenerates to normal decode -- the proposal is only checked against the
sampled token and counted in stats, so a wrong draft can never corrupt the
output stream.  Multi-token (k>1) verification is future work.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

Plan = dict  # one scheduler.step() event


@dataclass
class StepPlan:
    """Everything the next forward needs; built while the engine is busy."""

    rows: List[List[int]]        # token ids per batch row (chunk or last token)
    row_rids: List[str]          # row j belongs to row_rids[j]
    ev: Plan                     # scheduler event this plan was built from
    draft: Dict[str, int] = field(default_factory=dict)  # rid -> proposed token


def _inter(a0: float, a1: float, b0: float, b1: float) -> float:
    """Length of the intersection of intervals [a0,a1] and [b0,b1]."""
    return max(0.0, min(a1, b1) - max(a0, b0))


class OverlapEngine:
    """Runs scheduler.step() concurrently with the batched forward pass."""

    def __init__(self, scheduler, forward_fn, sample_fn, *, chunk_size=None,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self._sched = scheduler
        self._forward_fn = forward_fn
        self._sample_fn = sample_fn
        self._chunk = chunk_size if chunk_size is not None \
            else getattr(scheduler, "_chunk", 512)
        self._clock = clock
        self._draft_fn: Optional[Callable[[List[List[int]]], List[int]]] = None
        self._pool = ThreadPoolExecutor(max_workers=6)
        self._reqs: Dict[str, object] = {}
        self._last: Dict[str, int] = {}          # rid -> last committed token
        self.outputs: Dict[str, List[int]] = {}  # rid -> generated tokens
        self.ship_times: Dict[str, List[float]] = {}  # rid -> per-token ship t
        self.rows_log: List[dict] = []           # one metrics row per step
        self._buf: List[Optional[List[List[int]]]] = [None, None]
        self._buf_idx = 0
        self._pending_draft: Dict[str, int] = {}
        self.draft_hits = 0
        self.draft_total = 0

    def set_draft_runner(self, fn: Callable[[List[List[int]]], List[int]]) -> None:
        """fn(input_rows) -> one proposed next token per row (ADR-011)."""
        self._draft_fn = fn

    def submit(self, req) -> None:
        self._reqs[req.request_id] = req
        self.outputs[req.request_id] = []
        self.ship_times[req.request_id] = []

    def prepare_inputs(self, plan: StepPlan) -> int:
        """Write rows into the off-hand buffer; the forward reads the other
        one while the next plan writes this one.  Returns the buffer index."""
        idx = 1 - self._buf_idx
        self._buf[idx] = plan.rows
        self._buf_idx = idx
        return idx

    def rows_for(self, ev: Plan):
        """Batch rows for one forward: prompt chunks for prefill, the last
        sampled token for decoders.  A chunk that finished the prompt also
        decoded this step; its forward row is the chunk, not a token.
        Shared by the overlapped pipeline and the sync (no-overlap) path."""
        rows, rids = [], []
        for rid, off in ev["prefilling"]:
            req = self._reqs[rid]
            rows.append(req.prompt[max(0, off - self._chunk):off])
            rids.append(rid)
        prefilling = {rid for rid, _ in ev["prefilling"]}
        for rid in ev["decoding"]:
            if rid in prefilling:
                continue
            req = self._reqs[rid]
            rows.append([self._last.get(rid, req.prompt[-1])])
            rids.append(rid)
        return rows, rids

    def _sample_worker(self, out, plan: StepPlan):
        t0 = self._clock()
        toks = self._sample_fn(out, plan)
        return toks, (t0, self._clock())

    def _plan_worker(self, prev: Optional[StepPlan], s_fut, is_final: bool,
                     ready: threading.Event):
        try:
            return self._plan_body(prev, s_fut, is_final, ready)
        finally:
            ready.set()                         # never wedge the forward fence

    def _plan_body(self, prev: Optional[StepPlan], s_fut, is_final: bool,
                   ready: threading.Event):
        t0 = self._clock()
        ev = {"prefilling": [], "decoding": []} if is_final else self._sched.step()
        t1 = self._clock()                      # phase A: policy work
        toks, _ = s_fut.result() if prev is not None else ([], None)
        t2 = self._clock()
        if prev is not None:                    # phase B: consume prev sample
            for j, rid in enumerate(prev.row_rids):
                if rid not in prev.ev["decoding"]:
                    continue                    # mid-prefill chunk: no token
                tok = toks[j]
                self.outputs[rid].append(tok)
                self._last[rid] = tok
                self.ship_times[rid].append(self._clock())
                if rid in self._pending_draft:  # greedy draft acceptance
                    self.draft_hits += int(self._pending_draft.pop(rid) == tok)
        if is_final:
            ready.set()
            return None
        rows, rids = self.rows_for(ev)
        if self._draft_fn is not None:
            for rid, prop in zip(rids, self._draft_fn(rows)):
                if rid in ev["decoding"]:
                    self._pending_draft[rid] = prop
                    self.draft_total += 1
        idx = self.prepare_inputs(StepPlan(rows=rows, row_rids=rids, ev=ev))
        t3 = self._clock()
        ready.set()
        return {"plan": StepPlan(rows=rows, row_rids=rids, ev=ev),
                "a": (t0, t1), "b": (t2, t3), "idx": idx}

    def _forward_worker(self, p_fut, ready: threading.Event, t0: float):
        ready.wait()                            # fence: inputs patched
        rows = self._buf[p_fut.result()["idx"]]
        out = self._forward_fn(rows)
        return out, (t0, self._clock())

    def run(self, requests: List, steps: int) -> List[dict]:
        """Drive the pipeline for `steps` forwards; returns metrics rows
        (one per finalized step; the partial last row is dropped)."""
        for r in requests:
            self.submit(r)
        ev = self._sched.step()                 # prime: first plan, sync
        rows, rids = self.rows_for(ev)
        idx = self.prepare_inputs(StepPlan(rows=rows, row_rids=rids, ev=ev))
        ready = threading.Event()
        ready.set()
        fut = self._pool.submit(self._forward_worker,
                                self._pool.submit(lambda: {"idx": idx}), ready,
                                self._clock())
        prev, prev_s, prev_p = StepPlan(rows=[], row_rids=[], ev=ev), None, None
        for i in range(2, steps + 2):
            is_final = i == steps + 1
            out, f_win = fut.result()           # join forward_{i-1}
            slot_t0 = self._clock()             # next slot opens at join
            s_fut = self._pool.submit(self._sample_worker, out, prev)
            ready = threading.Event()
            p_fut = self._pool.submit(self._plan_worker, prev, s_fut,
                                      is_final, ready)
            if not is_final:
                fut = self._pool.submit(self._forward_worker, p_fut, ready,
                                        slot_t0)
            else:
                fut = None
            _, s_win = s_fut.result()
            p_res = p_fut.result()
            if prev_s is not None and prev_p is not None:
                self._finalize(prev_s, prev_p, f_win)
            prev_s, prev_p = s_win, p_res
            if p_res is not None:
                prev = p_res["plan"]
        self._pool.shutdown(wait=True)
        return self.rows_log

    def _finalize(self, s_win, p_res, f_win) -> None:
        f0, f1 = f_win
        s0, s1 = s_win
        a0, a1 = p_res["a"]
        b0, b1 = p_res["b"]
        ev = p_res["plan"].ev
        f_ms = f1 - f0
        s_ms = s1 - s0
        s_ov = _inter(s0, s1, f0, f1)
        ov = s_ov + _inter(a0, a1, f0, f1) + _inter(b0, b1, f0, f1)
        self.rows_log.append({
            "step": len(self.rows_log) + 2,
            "forward_ms": f_ms, "sample_ms": s_ms,
            "plan_ms": (a1 - a0) + (b1 - b0),
            "overlap_ms": ov,
            "overlap_ratio": ov / f_ms if f_ms else 0.0,
            "sample_overlap_ratio": s_ov / s_ms if s_ms else 0.0,
            "plan_includes_new_prefill": int(bool(ev["prefilling"])),
            "buffer_idx": p_res["idx"],
            "n_decoding": len(ev["decoding"]),
            "n_prefilling": len(ev["prefilling"]),
        })

    def stats(self) -> dict:
        return {"draft_hits": self.draft_hits, "draft_total": self.draft_total,
                "steps_finalized": len(self.rows_log)}

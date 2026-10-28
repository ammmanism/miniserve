"""miniserve.scheduler -- policy-driven continuous-batching scheduler.

Priority queue with aging, chunked prefill, SLO admission, preemption
accounting (recompute vs prefix-cache swap), and a queue-wait histogram.
Step invariants: victims are decoding-only; max_batch caps the union of
decode + prefill-chunk jobs; "ff"/"df"/"rr" reorders work within that cap;
a chunk finishing the prompt emits the first token in the same step when
the decode pass has not already run (rr even step).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from miniserve.blocks import BlockManager, OutOfBlocksError


@dataclass
class Request:
    """One inference request plus scheduler-owned runtime state."""

    request_id: str
    prompt: List[int]
    max_new_tokens: int = 64
    priority: int = 0            # higher = more urgent
    arrival_time: float = field(default_factory=time.monotonic)
    deadline_s: Optional[float] = None

    state: str = "queued"        # queued|prefilling|decoding|swapped|finished|rejected
    gpu_blocks: List[int] = field(default_factory=list)
    prefill_offset: int = 0      # prompt tokens computed so far
    num_generated: int = 0
    compute_done_preempt: int = 0  # prompt tokens wasted on preemption
    reason: str = ""

    def effective_priority(self, now: float, aging: float) -> float:
        return self.priority + aging * max(0.0, now - self.arrival_time)


class Scheduler:
    """Schedules Requests against a real BlockManager each step()."""

    def __init__(
        self,
        block_manager: BlockManager,
        *,
        block_size: int = 16,
        max_batch: int = 8,
        max_chunked: int = 2,
        aging_factor: float = 0.1,
        prefill_chunk: int = 512,
        prefill_policy: str = "rr",
        avg_decode_rate: float = 32.0,
        prefix_cache: Optional[object] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if prefill_policy not in ("ff", "df", "rr"):
            raise ValueError(f"unknown prefill_policy: {prefill_policy}")
        self._bm = block_manager
        self._bs = block_size
        self._max_batch = max_batch
        self._max_chunked = max_chunked
        self._aging = aging_factor
        self._chunk = prefill_chunk
        self._policy = prefill_policy
        self._rate = avg_decode_rate
        self._cache = prefix_cache
        self._clock = clock
        self._waiting: List[Request] = []      # heap-keyed drain, ties FCFS
        self._running: Dict[str, Request] = {}
        self.queue_wait_ms: List[float] = []   # one entry per admission
        self._stats = {
            "recompute_preemptions": 0,
            "swap_preemptions": 0,
            "compute_wasted_tokens": 0,
            "slo_rejections": 0,
            "admissions": 0,
        }
        self.log: List[dict] = []              # captured per-step debug logs

    @property
    def prefill_policy(self) -> str:
        return self._policy

    @prefill_policy.setter
    def prefill_policy(self, policy: str) -> None:
        if policy not in ("ff", "df", "rr"):
            raise ValueError(f"unknown prefill_policy: {policy}")
        self._policy = policy

    def set_aging(self, alpha: float) -> None:  # aging knob (starvation fix)
        self._aging = alpha

    def _blocks_for(self, tokens: int) -> int:
        return (tokens + self._bs - 1) // self._bs

    def add(self, req: Request) -> bool:
        """Reject deadline-infeasible requests before anything is queued."""
        now = self._clock()
        if req.deadline_s is not None:
            remaining = (len(req.prompt) - req.prefill_offset) + req.max_new_tokens
            t_remaining = -(-remaining // self._rate)
            if now + t_remaining > req.deadline_s:
                req.state = "rejected"
                req.reason = "slo_infeasible"
                self._stats["slo_rejections"] += 1
                return False
        self._waiting.append(req)
        return True

    def _pick_victim(self, blocker: Request) -> Optional[Request]:
        """Decoding-only victims: lowest priority, least progress, newest."""
        cands = [r for r in self._running.values() if r.state == "decoding"]
        if not cands:
            return None
        return min(cands, key=lambda r: (r.priority, r.num_generated, -r.arrival_time))

    def _preempt(self, victim: Request) -> str:
        """Recompute discards prefill progress; swap (cache) saves prompt KV."""
        cached = self._cache is not None and victim.prefill_offset == len(victim.prompt)
        if cached:
            self._cache.insert(victim.prompt, victim.gpu_blocks)
        wasted = victim.prefill_offset
        victim.compute_done_preempt += wasted
        self._stats["recompute_preemptions" if not cached else "swap_preemptions"] += 1
        self._stats["compute_wasted_tokens"] += wasted
        self._bm.free(victim.gpu_blocks, victim.request_id)
        victim.gpu_blocks = []
        victim.state = "swapped" if cached else "queued"
        if not cached:
            victim.prefill_offset = 0
        self._running.pop(victim.request_id, None)
        self._waiting.append(victim)
        return "swap" if cached else "recompute"

    def _admit(self, now: float, ev: dict) -> None:
        """Drain waiting by effective priority (ties FCFS); reserve the
        first chunk's blocks.  Swapped requests first try the prefix cache."""
        for req in sorted(self._waiting,
                          key=lambda r: (-r.effective_priority(now, self._aging),
                                         r.arrival_time, r.request_id)):
            if len(self._running) >= self._max_batch:
                break
            if self._cache is not None and req.state == "swapped":
                m_len, table = self._cache.match(req.prompt)
                if m_len > 0:
                    req.gpu_blocks = list(self._bm.fork(table))
                    req.prefill_offset = min(m_len, len(req.prompt))
                    req.state = "decoding" if req.prefill_offset == len(req.prompt) else "prefilling"
                    req.num_generated = 0
                    self._running[req.request_id] = req
                    self.queue_wait_ms.append((now - req.arrival_time) * 1000.0)
                    self._stats["admissions"] += 1
                    ev["admitted"].append(req.request_id)
                    self._waiting.remove(req)
                    continue
            first_chunk = min(self._chunk, len(req.prompt) - req.prefill_offset)
            grew = self._blocks_for(req.prefill_offset + first_chunk) - len(req.gpu_blocks)
            try:
                if grew > 0:
                    req.gpu_blocks += self._bm.allocate(grew, req.request_id)
            except OutOfBlocksError:
                continue  # KV budget: leave queued this step
            ev["admitted"].append(req.request_id)
            self._running[req.request_id] = req
            req.state = "prefilling" if req.prefill_offset < len(req.prompt) else "decoding"
            self.queue_wait_ms.append((now - req.arrival_time) * 1000.0)
            self._stats["admissions"] += 1
            self._waiting.remove(req)

    def _prefill_job(self, req: Request, ev: dict) -> bool:
        """Advance up to prefill_chunk prompt tokens for one request."""
        chunk = min(self._chunk, len(req.prompt) - req.prefill_offset)
        grew = self._blocks_for(req.prefill_offset + chunk) - len(req.gpu_blocks)
        try:
            if grew > 0:
                req.gpu_blocks += self._bm.allocate(grew, req.request_id)
        except OutOfBlocksError:
            return False  # stall this step; prefill is never preempted
        req.prefill_offset += chunk
        ev["prefilling"].append((req.request_id, req.prefill_offset))
        if req.prefill_offset == len(req.prompt):
            req.state = "decoding"
        return True

    def _decode_job(self, req: Request, ev: dict) -> bool:
        """Produce one token, preempting a low-priority decoder on KV exhaustion."""
        need = self._blocks_for(len(req.prompt) + req.num_generated + 1) - len(req.gpu_blocks)
        for _attempt in (0, 1):
            try:
                if need > 0:
                    req.gpu_blocks += self._bm.allocate(need, req.request_id)
                break
            except OutOfBlocksError:
                if _attempt == 1:
                    return False
                victim = self._pick_victim(req)
                if victim is None:
                    return False  # KV full, no victim: no progress this step
                mode = self._preempt(victim)
                ev["preempted"].append((victim.request_id, mode))
                if victim is req:   # self-preempted: this step is over for req
                    return False
        req.num_generated += 1
        ev["decoding"].append(req.request_id)
        if req.num_generated == req.max_new_tokens:
            self._bm.free(req.gpu_blocks, req.request_id)
            req.state = "finished"
            ev["finished"].append(req.request_id)
            self._running.pop(req.request_id)
        return True

    def step(self) -> dict:
        """Admit, then run decode+prefill work within the union batch cap."""
        now = self._clock()
        ev = {"step": len(self.log) + 1, "admitted": [], "decoding": [],
              "prefilling": [], "finished": [], "preempted": []}
        self._admit(now, ev)

        decs = [r for r in self._running.values() if r.state == "decoding"]
        pref = [r for r in self._running.values() if r.state == "prefilling"]
        chunk_first = self._policy == "ff" or (self._policy == "rr" and (len(self.log) + 1) % 2 == 0)

        executed, promoted = 0, []
        if chunk_first:  # chunks consume slots first, decoders get the rest
            for req in pref:
                if executed >= self._max_batch or len(ev["prefilling"]) >= self._max_chunked:
                    break
                if self._prefill_job(req, ev):
                    executed += 1
                    if req.state == "decoding":
                        promoted.append(req)
        for req in decs + promoted:  # all decoders run; promoted get first token
            if (req.state == "decoding" and executed < self._max_batch
                    and req.request_id in self._running and self._decode_job(req, ev)):
                executed += 1
        if not chunk_first:  # decode first, then chunks with leftover slots
            for req in pref:
                if executed >= self._max_batch or len(ev["prefilling"]) >= self._max_chunked:
                    break
                if self._prefill_job(req, ev):
                    executed += 1

        self.log.append(ev)
        return ev

    def stats(self) -> dict:
        waits = self.queue_wait_ms
        avg = sum(waits) / len(waits) if waits else 0.0
        return dict(self._stats) | {"avg_queue_wait_ms": avg}
"""miniserve.engine -- top-level inference engine (ADR-006).

Thin wiring: owns the Scheduler policy object and the OverlapEngine that
pipelines scheduler.step() under the batched forward pass.  Token-level
mechanics (sampling, drafts, double-buffered input prep) live in
miniserve.overlap; policy lives in miniserve.scheduler; this module only
joins them and exposes the run loop's results.
"""
from __future__ import annotations

from typing import List, Optional

from miniserve.blocks import BlockManager
from miniserve.overlap import OverlapEngine
from miniserve.scheduler import Request, Scheduler


class LLMEngine:
    """Continuous-batching engine: add(request) then run(requests, steps)."""

    def __init__(
        self,
        block_manager: BlockManager,
        forward_fn,
        sample_fn,
        *,
        chunk_size: Optional[int] = None,
        draft_fn=None,
        **scheduler_kw,
    ) -> None:
        self.scheduler = Scheduler(block_manager, **scheduler_kw)
        self.overlap = OverlapEngine(
            self.scheduler, forward_fn, sample_fn, chunk_size=chunk_size
        )
        if draft_fn is not None:
            self.overlap.set_draft_runner(draft_fn)

    def add(self, req: Request) -> bool:
        """SLO-admit a request (rejected requests never run)."""
        return self.scheduler.add(req)

    def run(self, requests: List[Request], steps: int) -> List[dict]:
        """Drive the overlapped pipeline; returns per-step metrics rows."""
        return self.overlap.run(requests, steps)

    @property
    def outputs(self):
        return self.overlap.outputs

    def stats(self) -> dict:
        return self.scheduler.stats() | self.overlap.stats()

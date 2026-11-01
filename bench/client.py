"""bench.client -- async HTTP load generator with per-token timestamps.

Streams ``/v1/completions`` (SSE) so TTFT is the arrival time of the first
delta and ITL is the gap between deltas.  Used by ``run_suite --url`` to
measure a running server.  The committed engine numbers come from the
in-process runner (see run_suite) because the FastAPI server in this repo is
a stub generator, not the paged model path -- that distinction is stated in
README and docs/CAPACITY.md.

A transport can be injected for tests (httpx.ASGITransport), so CI never
needs a live port.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

import httpx


@dataclass
class RequestResult:
    ok: bool
    ttft_ms: Optional[float]
    itl_ms: List[float] = field(default_factory=list)
    total_ms: float = 0.0
    n_tokens: int = 0
    error: str = ""

    def itl_p50(self) -> float:
        if not self.itl_ms:
            return float("nan")
        s = sorted(self.itl_ms)
        return s[len(s) // 2]


class BenchClient:
    """Concurrent SSE client: one result per request, token-timestamped."""

    def __init__(self, base_url: str = "http://localhost:8000", *,
                 api_key: Optional[str] = None, timeout_s: float = 600.0,
                 transport: Optional[Any] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self._transport = transport

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def one(self, client: httpx.AsyncClient, prompt: str,
                  max_tokens: int, model: str) -> RequestResult:
        payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
                   "stream": True, "temperature": 0.0}
        stamp = time.perf_counter()
        first = None
        last = stamp
        itls: List[float] = []
        n = 0
        try:
            async with client.stream("POST", f"{self.base_url}/v1/completions",
                                     json=payload,
                                     headers=self._headers()) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    return RequestResult(False, None, total_ms=0.0,
                                         error=f"HTTP {resp.status_code}")
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    text = "".join(
                        c.get("text", "")
                        for c in chunk.get("choices", [])
                    )
                    if not text:
                        continue
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    else:
                        itls.append((now - last) * 1000.0)
                    last = now
                    n += 1
        except Exception as exc:  # network/timeout: report, never raise
            return RequestResult(False, None, total_ms=(time.perf_counter() - stamp) * 1000.0,
                                 error=repr(exc))
        total = (time.perf_counter() - stamp) * 1000.0
        return RequestResult(ok=n > 0,
                             ttft_ms=None if first is None else (first - stamp) * 1000.0,
                             itl_ms=itls, total_ms=total, n_tokens=n)

    async def run(self, prompts: Sequence[str], max_tokens: int,
                  concurrency: int, model: str = "miniserve") -> List[RequestResult]:
        sem = asyncio.Semaphore(max(1, concurrency))
        kwargs = {"timeout": self.timeout_s}
        if self._transport is not None:
            kwargs["transport"] = self._transport
            kwargs["base_url"] = self.base_url
        async with httpx.AsyncClient(**kwargs) as client:
            async def bounded(p: str) -> RequestResult:
                async with sem:
                    return await self.one(client, p, max_tokens, model)

            return await asyncio.gather(*(bounded(p) for p in prompts))

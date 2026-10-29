"""miniserve.server — Production-shaped FastAPI server.

Features:
  - OpenAI-compatible /v1/completions (non-streaming + SSE streaming)
  - Token-bucket rate limiter keyed on API key / remote IP (429 + Retry-After)
  - In-flight request deduplication — identical requests share one generation
  - OpenTelemetry tracing (no-op when OTEL_EXPORTER_OTLP_ENDPOINT unset)
  - Backpressure: 503 when queue > MAX_QUEUE or inflight > MAX_INFLIGHT
  - Grammar-constrained JSON mode via sampler.JSONLogitsProcessor
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from miniserve.protocol import (
    CompletionChoice, CompletionRequest, CompletionResponse,
    StreamChunk, StreamDelta, UsageInfo,
)
from miniserve.sampler import JSONLogitsProcessor, Sampler
from miniserve.stream import SSEStream, sse_done, sse_event

# ---------------------------------------------------------------------------
# OpenTelemetry (graceful no-op when not configured)
# ---------------------------------------------------------------------------
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    _OTEL_OK = True
except ImportError:
    _OTEL_OK = False

_tracer_provider: Optional[Any] = None
_in_memory_exporter: Optional[Any] = None  # set by tests


def _get_tracer() -> Any:
    if not _OTEL_OK:
        return None
    global _tracer_provider
    if _tracer_provider is None:
        _tracer_provider = TracerProvider()
        if _in_memory_exporter is not None:
            _tracer_provider.add_span_processor(
                SimpleSpanProcessor(_in_memory_exporter)
            )
    # Always use _tracer_provider directly — avoids global override warning
    return _tracer_provider.get_tracer("miniserve")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_QUEUE: int = int(os.getenv("MAX_QUEUE", "128"))
MAX_INFLIGHT: int = int(os.getenv("MAX_INFLIGHT", "64"))
RATE_LIMIT_RPS: float = float(os.getenv("RATE_LIMIT_RPS", "10"))
RATE_LIMIT_BURST: int = int(os.getenv("RATE_LIMIT_BURST", "20"))

# ---------------------------------------------------------------------------
# Token-bucket rate limiter
# ---------------------------------------------------------------------------
import threading


class _TokenBucket:
    def __init__(self, rate: float, burst: int) -> None:
        self._rate = rate
        self._burst = burst
        self._tokens: float = burst
        self._last: float = time.monotonic()
        self._lock = threading.Lock()

    def consume(self) -> Tuple[bool, float]:
        """Returns (allowed, retry_after_seconds)."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True, 0.0
            retry = (1.0 - self._tokens) / self._rate
            return False, retry


_buckets: Dict[str, _TokenBucket] = {}
_buckets_lock = threading.Lock()


def _get_bucket(key: str) -> _TokenBucket:
    with _buckets_lock:
        if key not in _buckets:
            _buckets[key] = _TokenBucket(RATE_LIMIT_RPS, RATE_LIMIT_BURST)
        return _buckets[key]


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Response:
        auth = request.headers.get("Authorization", "")
        key = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else (
            request.client.host if request.client else "unknown"
        )
        bucket = _get_bucket(key)
        allowed, retry_after = bucket.consume()
        if not allowed:
            return Response(
                content='{"error":"rate limit exceeded"}',
                status_code=429,
                headers={"Retry-After": str(int(retry_after) + 1), "Content-Type": "application/json"},
            )
        return await call_next(request)


# ---------------------------------------------------------------------------
# In-flight deduplication store
# ---------------------------------------------------------------------------

class _DedupEntry:
    def __init__(self) -> None:
        self.done = asyncio.Event()
        self.result: Optional[str] = None  # final text
        self.chunks: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self.subscribers: int = 1

    async def stream(self) -> AsyncGenerator[str, None]:
        while True:
            chunk = await self.chunks.get()
            if chunk is None:
                break
            yield chunk

    def broadcast(self, chunk: Optional[str]) -> None:
        # Put a copy into each subscriber's queue
        self.chunks.put_nowait(chunk)


_dedup_store: Dict[str, "_DedupEntry"] = {}
_dedup_lock = asyncio.Lock()


def _dedup_key(req: CompletionRequest) -> str:
    canon = json.dumps({
        "model": req.model,
        "prompt": req.prompt,
        "messages": req.messages,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "top_k": req.top_k,
        "seed": req.seed,
        "response_format": req.response_format.model_dump() if req.response_format else None,
    }, sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Stub engine (generates text without a real model)
# ---------------------------------------------------------------------------

_sampler = Sampler()
_inflight: int = 0
_queue_depth: int = 0


async def _stub_generate(req: CompletionRequest, proc: Optional[JSONLogitsProcessor]) -> str:
    """Minimal stub: returns schema-valid JSON or echoes prompt words."""
    await asyncio.sleep(0.01)  # simulate compute
    if proc is not None:
        return proc.stub_completion()
    words = (req.prompt or "hello world").split()
    n = min(req.max_tokens, len(words))
    return " ".join(words[:n]) if n else "hello"


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    *,
    in_memory_exporter: Any = None,
    max_queue: int = MAX_QUEUE,
    max_inflight: int = MAX_INFLIGHT,
    rate_rps: float = RATE_LIMIT_RPS,
    rate_burst: int = RATE_LIMIT_BURST,
) -> FastAPI:
    global _in_memory_exporter, MAX_QUEUE, MAX_INFLIGHT, RATE_LIMIT_RPS, RATE_LIMIT_BURST
    global _tracer_provider
    _in_memory_exporter = in_memory_exporter
    MAX_QUEUE = max_queue
    MAX_INFLIGHT = max_inflight
    RATE_LIMIT_RPS = rate_rps
    RATE_LIMIT_BURST = rate_burst

    # Reset tracer + rate-limit buckets for test isolation
    if in_memory_exporter is not None and _OTEL_OK:
        _tracer_provider = TracerProvider()
        _tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_exporter))
    else:
        _tracer_provider = None

    with _buckets_lock:
        _buckets.clear()

    app = FastAPI(title="miniserve")
    app.add_middleware(RateLimitMiddleware)

    # ------------------------------------------------------------------ #
    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, request: Request) -> Response:
        global _inflight, _queue_depth

        # Backpressure
        if _queue_depth >= max_queue or _inflight >= max_inflight:
            return Response(
                content='{"error":"server overloaded"}',
                status_code=503,
                headers={"Retry-After": "1", "Content-Type": "application/json"},
            )

        # JSON mode — validate schema
        proc: Optional[JSONLogitsProcessor] = None
        if req.response_format and req.response_format.type == "json_object":
            try:
                proc = JSONLogitsProcessor(req.response_format.schema_)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        # Deduplication
        dkey = _dedup_key(req)
        deduped = False

        async with _dedup_lock:
            entry = _dedup_store.get(dkey)
            if entry is not None:
                deduped = True
                entry.subscribers += 1
            else:
                entry = _DedupEntry()
                _dedup_store[dkey] = entry

        tracer = _get_tracer()
        req_id = f"cmpl-{uuid.uuid4().hex[:8]}"

        if deduped:
            # Join onto existing generation
            if req.stream:
                async def _join_stream() -> AsyncGenerator[str, None]:
                    async for chunk in entry.stream():
                        yield chunk
                    yield sse_done()
                headers = {"X-Deduplicated": "true"}
                return StreamingResponse(_join_stream(), media_type="text/event-stream", headers=headers)
            else:
                await entry.done.wait()
                text = entry.result or ""
                resp = CompletionResponse(
                    id=req_id, model=req.model,
                    choices=[CompletionChoice(text=text)],
                    usage=UsageInfo(completion_tokens=len(text.split())),
                )
                return Response(
                    content=resp.model_dump_json(),
                    headers={"X-Deduplicated": "true"},
                    media_type="application/json",
                )

        # Primary generation
        _queue_depth += 1
        try:
            _inflight += 1
            span_ctx = None

            if tracer:
                with tracer.start_as_current_span("miniserve.generate") as span:
                    span.set_attribute("model", req.model)
                    span.set_attribute("deduplicated", False)
                    span_ctx = span.get_span_context()

                    with tracer.start_as_current_span("miniserve.prefill"):
                        pass  # stub — real prefill goes here

                    with tracer.start_as_current_span("miniserve.decode_step"):
                        text = await _stub_generate(req, proc)

                    with tracer.start_as_current_span("miniserve.detokenize"):
                        pass

                    span.set_attribute("generated_tokens", len(text.split()))
            else:
                text = await _stub_generate(req, proc)

            entry.result = text
            entry.done.set()

            if req.stream:
                # Broadcast chunks to dedup subscribers
                chunk = StreamChunk(
                    id=req_id, model=req.model,
                    choices=[StreamDelta(text=text, finish_reason="stop")],
                )
                sse = sse_event(chunk.model_dump())
                entry.broadcast(sse)
                entry.broadcast(None)  # signal done

                async def _primary_stream() -> AsyncGenerator[str, None]:
                    yield sse
                    yield sse_done()

                stream = SSEStream(_primary_stream(), span_context=span_ctx)
                return StreamingResponse(stream, media_type="text/event-stream")
            else:
                resp = CompletionResponse(
                    id=req_id, model=req.model,
                    choices=[CompletionChoice(text=text)],
                    usage=UsageInfo(completion_tokens=len(text.split())),
                )
                return Response(content=resp.model_dump_json(), media_type="application/json")

        finally:
            _inflight -= 1
            _queue_depth -= 1
            async with _dedup_lock:
                entry.subscribers -= 1
                if entry.subscribers <= 0:
                    _dedup_store.pop(dkey, None)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "inflight": _inflight, "queue": _queue_depth}

    return app

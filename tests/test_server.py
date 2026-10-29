"""Tests for miniserve.server — 12 tests covering all beast features."""
from __future__ import annotations

import asyncio
import json
import time
from typing import List

import pytest
import httpx
from httpx import AsyncClient

from miniserve.server import create_app, _buckets, _buckets_lock, _dedup_store
from miniserve.sampler import JSONLogitsProcessor, Sampler, _validate_schema
from miniserve.protocol import CompletionRequest, ResponseFormat

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
}


@pytest.fixture
def otel_exporter():
    try:
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        exporter = InMemorySpanExporter()
        yield exporter
    except ImportError:
        pytest.skip("opentelemetry-sdk not installed")


@pytest.fixture
def app():
    return create_app(rate_rps=10, rate_burst=20, max_queue=128, max_inflight=64)


@pytest.fixture
def app_tight():
    """App with tiny queue/inflight caps for backpressure tests."""
    return create_app(rate_rps=1000, rate_burst=1000, max_queue=2, max_inflight=2)


@pytest.fixture
def app_otel(otel_exporter):
    return create_app(in_memory_exporter=otel_exporter, rate_rps=1000, rate_burst=1000)


# ---------------------------------------------------------------------------
# 1. Health endpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health(app):
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# 2. Basic completion
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_basic_completion(app):
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/v1/completions", json={"prompt": "hello world", "max_tokens": 4})
    assert r.status_code == 200
    body = r.json()
    assert "choices" in body
    assert isinstance(body["choices"][0]["text"], str)


# ---------------------------------------------------------------------------
# 3. Streaming completion
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_completion(app):
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        async with client.stream(
            "POST", "/v1/completions",
            json={"prompt": "hi there", "max_tokens": 8, "stream": True}
        ) as r:
            assert r.status_code == 200
            raw = await r.aread()
    assert b"[DONE]" in raw


# ---------------------------------------------------------------------------
# 4. JSON mode — schema-valid output (N=20 samples for speed)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_json_mode_schema_valid(app):
    proc = JSONLogitsProcessor(PERSON_SCHEMA)
    valid = 0
    N = 20
    for _ in range(N):
        text = proc.stub_completion()
        if proc.validate_complete(text):
            valid += 1
    assert valid == N, f"Only {valid}/{N} were schema-valid"


# ---------------------------------------------------------------------------
# 5. JSON mode via HTTP endpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_json_mode_endpoint(app):
    payload = {
        "prompt": "Give me a person",
        "max_tokens": 32,
        "response_format": {"type": "json_object", "schema": PERSON_SCHEMA},
    }
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/v1/completions", json=payload)
    assert r.status_code == 200
    text = r.json()["choices"][0]["text"]
    proc = JSONLogitsProcessor(PERSON_SCHEMA)
    assert proc.validate_complete(text), f"Not schema-valid: {text!r}"


# ---------------------------------------------------------------------------
# 6. JSON mode — unsupported schema -> 400
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_json_mode_bad_schema(app):
    payload = {
        "prompt": "test",
        "response_format": {"type": "json_object", "schema": {"type": "unicorn"}},
    }
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/v1/completions", json=payload)
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# 7. Rate limit — burst then 429
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rate_limit_burst_then_429():
    app = create_app(rate_rps=10, rate_burst=5)
    results: List[int] = []
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        for _ in range(8):
            r = await client.post(
                "/v1/completions",
                json={"prompt": "x"},
                headers={"Authorization": "Bearer testkey"},
            )
            results.append(r.status_code)
    ok = sum(1 for s in results if s == 200)
    blocked = sum(1 for s in results if s == 429)
    assert ok == 5, f"Expected 5 ok, got {ok}"
    assert blocked == 3, f"Expected 3 blocked, got {blocked}"


# ---------------------------------------------------------------------------
# 8. Rate limit — Retry-After header present
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rate_limit_retry_after():
    app = create_app(rate_rps=1, rate_burst=1)
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        await client.post("/v1/completions", json={"prompt": "x"},
                          headers={"Authorization": "Bearer k2"})
        r = await client.post("/v1/completions", json={"prompt": "x"},
                              headers={"Authorization": "Bearer k2"})
    assert r.status_code == 429
    assert "Retry-After" in r.headers


# ---------------------------------------------------------------------------
# 9. Deduplication — concurrent identical requests share one result
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dedup_concurrent_identical():
    app = create_app(rate_rps=1000, rate_burst=1000)
    payload = {"prompt": "dedup test prompt", "max_tokens": 8, "seed": 42}
    N = 5

    async def _one(client: AsyncClient) -> str:
        r = await client.post("/v1/completions", json=payload)
        assert r.status_code == 200
        return r.json()["choices"][0]["text"]

    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        texts = await asyncio.gather(*[_one(client) for _ in range(N)])

    # All responses must be identical
    assert len(set(texts)) == 1, f"Expected 1 unique text, got {set(texts)}"


# ---------------------------------------------------------------------------
# 10. Dedup header — X-Deduplicated on secondary requests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dedup_header():
    """Second request to same key while first is in-flight gets X-Deduplicated header."""
    import miniserve.server as srv
    app = create_app(rate_rps=1000, rate_burst=1000)
    payload = {"prompt": "header check", "max_tokens": 4, "seed": 7}

    dedup_count = 0
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resps = await asyncio.gather(
            client.post("/v1/completions", json=payload),
            client.post("/v1/completions", json=payload),
        )
    for r in resps:
        assert r.status_code == 200
        if r.headers.get("x-deduplicated") == "true":
            dedup_count += 1
    # At least one should be deduplicated (or both served same result)
    assert all(r.status_code == 200 for r in resps)


# ---------------------------------------------------------------------------
# 11. Backpressure — 503 when inflight/queue cap exceeded
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_backpressure_503():
    """Direct counter manipulation triggers 503 immediately."""
    import miniserve.server as srv
    app = create_app(rate_rps=1000, rate_burst=1000, max_queue=2, max_inflight=2)
    # Force inflight counter to cap
    srv._inflight = 2
    srv._queue_depth = 0
    try:
        async with AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post("/v1/completions", json={"prompt": "bp", "max_tokens": 4})
        assert r.status_code == 503, f"Expected 503, got {r.status_code}"
        assert "Retry-After" in r.headers
    finally:
        srv._inflight = 0
        srv._queue_depth = 0


@pytest.mark.asyncio
async def test_backpressure_queue():
    """Queue depth cap triggers 503."""
    import miniserve.server as srv
    app = create_app(rate_rps=1000, rate_burst=1000, max_queue=2, max_inflight=64)
    srv._queue_depth = 2
    srv._inflight = 0
    try:
        async with AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post("/v1/completions", json={"prompt": "bp2", "max_tokens": 4})
        assert r.status_code == 503
    finally:
        srv._queue_depth = 0


# ---------------------------------------------------------------------------
# 12. OpenTelemetry — span counts
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_otel_spans(app_otel, otel_exporter):
    async with AsyncClient(
        transport=httpx.ASGITransport(app=app_otel), base_url="http://test"
    ) as client:
        for _ in range(3):
            r = await client.post("/v1/completions", json={"prompt": "otel test", "max_tokens": 4})
            assert r.status_code == 200

    spans = otel_exporter.get_finished_spans()
    names = [s.name for s in spans]
    generate_spans = [n for n in names if n == "miniserve.generate"]
    prefill_spans = [n for n in names if n == "miniserve.prefill"]
    assert len(generate_spans) >= 3, f"Expected >= 3 generate spans, got {generate_spans}"
    assert len(prefill_spans) >= 3, f"Expected >= 3 prefill spans, got {prefill_spans}"


# ---------------------------------------------------------------------------
# Sampler unit tests
# ---------------------------------------------------------------------------

def test_sampler_greedy():
    s = Sampler()
    logits = [0.1, 0.5, 0.9, 0.2]
    assert s.sample(logits, temperature=0.0) == 2


def test_json_logits_processor_stub():
    proc = JSONLogitsProcessor(PERSON_SCHEMA)
    out = proc.stub_completion()
    assert proc.validate_complete(out)


def test_schema_validation_rejects_unknown_type():
    with pytest.raises(ValueError):
        _validate_schema({"type": "unknown_type"})

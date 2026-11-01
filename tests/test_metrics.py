"""tests/test_metrics.py -- Prometheus layering + wiring (Part 10).

Covers the exposition format itself, the scheduler's queue-wait wiring, and
the server's /metrics endpoint (httpx ASGI transport, no network).
"""
import httpx
import pytest

from miniserve.blocks import BlockManager
from miniserve.metrics import METRICS
from miniserve.scheduler import Request, Scheduler
from miniserve.server import create_app


@pytest.fixture(autouse=True)
def _fresh():
    METRICS.reset()
    yield
    METRICS.reset()


def test_render_counters_gauges_histograms():
    METRICS.counter("miniserve_requests_total", 2, {"status": "ok"})
    METRICS.gauge("miniserve_running", 3)
    METRICS.observe("miniserve_ttft_seconds", 0.02)
    METRICS.observe("miniserve_ttft_seconds", 0.5)
    text = METRICS.render()

    assert '# TYPE miniserve_requests_total counter' in text
    assert 'miniserve_requests_total{status="ok"} 2' in text
    assert "# TYPE miniserve_running gauge" in text
    assert "miniserve_running 3" in text

    assert "# TYPE miniserve_ttft_seconds histogram" in text
    assert 'miniserve_ttft_seconds_bucket{le="0.025"} 1' in text
    assert 'miniserve_ttft_seconds_bucket{le="1.0"} 2' in text
    assert 'miniserve_ttft_seconds_bucket{le="+Inf"} 2' in text
    assert "miniserve_ttft_seconds_count 2" in text


def test_histogram_buckets_are_cumulative():
    for v in (0.0005, 0.02, 100.0):
        METRICS.observe("miniserve_itl_seconds", v)
    text = METRICS.render()
    assert 'miniserve_itl_seconds_bucket{le="0.001"} 1' in text
    assert 'miniserve_itl_seconds_bucket{le="0.025"} 2' in text
    assert 'miniserve_itl_seconds_bucket{le="60.0"} 2' in text
    assert 'miniserve_itl_seconds_bucket{le="+Inf"} 3' in text


def test_scheduler_feeds_metrics():
    sched = Scheduler(BlockManager(num_blocks=16, block_size=4), max_batch=2,
                      prefill_chunk=8)
    for i in range(3):
        assert sched.add(Request(f"r{i}", [i + 1] * 8, max_new_tokens=2))
    for _ in range(8):
        sched.step()
    text = METRICS.render()
    assert "miniserve_queue_wait_seconds_count" in text
    assert "miniserve_queue_depth" in text
    assert 'miniserve_tokens_total{kind="generated"}' in text


async def test_server_metrics_endpoint():
    app = create_app(rate_rps=1000, rate_burst=1000)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/completions",
            json={"model": "m", "prompt": "one two three", "max_tokens": 2},
        )
        assert resp.status_code == 200
        metrics = await client.get("/metrics")
        assert metrics.status_code == 200
        assert "text/plain" in metrics.headers["content-type"]
        body = metrics.text
        assert "miniserve_ttft_seconds_bucket" in body
        assert 'miniserve_requests_total{status="ok"} 1' in body

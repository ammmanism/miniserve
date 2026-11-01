import asyncio, httpx, pytest
from bench.client import BenchClient
from miniserve.server import create_app

async def test_client_stream_per_token():
    app = create_app(rate_rps=1000, rate_burst=1000)
    c = BenchClient("http://test", transport=httpx.ASGITransport(app=app))
    rs = await c.run(["alpha beta gamma delta"] * 3, max_tokens=4, concurrency=2)
    assert len(rs) == 3
    assert all(r.ok for r in rs)
    assert all(r.ttft_ms is not None and r.ttft_ms >= 0 for r in rs)
    assert all(r.n_tokens >= 1 for r in rs)

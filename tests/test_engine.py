"""tests/test_engine.py -- LLMEngine wiring over the overlap pipeline."""
from miniserve.blocks import BlockManager
from miniserve.engine import LLMEngine
from miniserve.scheduler import Request
from tests.test_overlap import V, _forward, _oracle_draft, _sample


def test_engine_runs_pipeline_and_reports_stats():
    bm = BlockManager(num_blocks=256, block_size=16)
    eng = LLMEngine(bm, _forward, _sample, draft_fn=_oracle_draft,
                    max_batch=8, max_chunked=2, prefill_chunk=16)
    reqs = [Request(f"e{i}", [(i * 3 + t) % V for t in range(64)],
                    max_new_tokens=16) for i in range(8)]
    assert all(eng.add(r) for r in reqs)
    rows = eng.run(reqs, 48)
    assert rows and all(0 < r["overlap_ratio"] < 1 for r in rows)
    assert all(len(t) == 16 for t in eng.outputs.values())
    st = eng.stats()
    assert st["admissions"] == 8
    assert st["draft_hits"] == st["draft_total"] > 0

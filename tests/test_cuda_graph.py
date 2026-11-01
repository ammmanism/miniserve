"""tests/test_cuda_graph.py -- Part 9 Step A: bucketed graph decode.

Parity, double-buffer alternation, bucket interpolation, ragged
real/padding mix, and pool-persistence memory accounting, all against the
real Qwen paged path (fp32 CPU; graph replay uses the documented CPU
fallback -- same static-buffer machinery, no CUDA timings here).
"""
import pytest
import torch

from miniserve.blocks import BlockManager
from miniserve.model_runner import GraphedDecodeRunner, ModelRunner

VOCAB = 32000


@pytest.fixture(scope="module")
def runner():
    return ModelRunner.load()


@pytest.fixture(scope="module")
def graphed(runner):
    return GraphedDecodeRunner(runner)


def make_batch(bm: BlockManager, B: int, prompt_len: int = 24, seed: int = 0):
    """B real rows: distinct fake prompts, real block tables, ctx = prompt."""
    g = torch.Generator().manual_seed(seed)
    ids, tables, ctx = [], [], []
    for b in range(B):
        ids.append(int(torch.randint(0, VOCAB, (1,), generator=g)))
        need = (prompt_len + bm._block_size - 1) // bm._block_size
        tables.append(bm.allocate(need, f"row{b}-{seed}"))
        ctx.append(prompt_len)
    return ids, tables, ctx


def test_bucket_parity(runner, graphed):
    bm = BlockManager(num_blocks=512, block_size=16)
    for B in (1, 2):
        ids, tables, ctx = make_batch(bm, B, seed=B)
        eager = graphed.decode(ids, tables, ctx, use_graph=False)
        graph = graphed.decode(ids, tables, ctx, use_graph=True)
        assert torch.allclose(eager, graph, atol=1e-5), f"bucket {B}"


def test_double_buffer_alternation(graphed):
    bm = BlockManager(num_blocks=512, block_size=16)
    ids, tables, ctx = make_batch(bm, 2, seed=7)
    start = len(graphed.replay_log)
    for _ in range(4):
        graphed.decode(ids, tables, ctx)
    idxs = [i for _, i in graphed.replay_log[start:]]
    assert idxs == [0, 1, 0, 1]


@pytest.mark.parametrize("B", [3, 5, 7, 12, 20, 24])
def test_interpolation_matches_eager(runner, graphed, B):
    bm = BlockManager(num_blocks=2048, block_size=16)
    ids, tables, ctx = make_batch(bm, B, seed=B * 3)
    eager = graphed.decode(ids, tables, ctx, use_graph=False)
    graph = graphed.decode(ids, tables, ctx, use_graph=True)
    assert graph.shape == (B, runner.model.config.vocab_size)
    assert torch.allclose(eager, graph, atol=1e-3), f"interp batch {B}"


def test_ragged_real_pad_mix_bucket16(runner, graphed):
    """5 real rows + 11 padding rows replayed at bucket 16: the 5 real
    logits must match eager on the same padded batch AND eager on 5 rows."""
    bm = BlockManager(num_blocks=2048, block_size=16)
    ids, tables, ctx = make_batch(bm, 5, seed=42)
    pad_ids = ids + [0] * 11
    pad_tables = tables + [[-1]] * 11          # padding marked with -1
    pad_ctx = ctx + [0] * 11
    graph = graphed.decode(pad_ids, pad_tables, pad_ctx, use_graph=True)
    eager_padded = graphed.decode(pad_ids, pad_tables, pad_ctx, use_graph=False)
    eager_small = graphed.decode(ids, tables, ctx, use_graph=False)
    assert torch.allclose(graph, eager_padded, atol=1e-3)
    assert torch.allclose(graph[:5], eager_small, atol=1e-3)


def test_graph_pool_persistence_memory(graphed):
    report = graphed.memory_report()
    assert report["total_over_max"] <= 1.3     # max-bucket footprint bound
    assert report["total_static_bytes"] > 0

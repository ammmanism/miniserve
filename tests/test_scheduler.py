"""Tests for miniserve.scheduler — priority aging, chunked prefill, SLO.

The test plays the engine: Request state is advanced purely by the
scheduler against a REAL BlockManager (no mocks).  A fake clock makes
every assertion deterministic; a 500-step fuzz still exercises natural
timing.  assert_invariants() runs after every step of every test that
mutates the block pool.
"""
import random
from typing import Callable

import pytest

from miniserve.blocks import BlockManager
from miniserve.prefix_cache import RadixPrefixCache
from miniserve.scheduler import Request, Scheduler


def mk(num_blocks: int = 64, block_size: int = 4, **kw) -> tuple[BlockManager, Scheduler]:
    bm = BlockManager(num_blocks, block_size)
    fake = {"now": 100.0}
    clock: Callable[[], float] = lambda: fake["now"]
    sched = Scheduler(bm, block_size=block_size, clock=clock, **kw)
    return bm, sched


def mkc(num_blocks: int = 64, block_size: int = 4, **kw) -> tuple[BlockManager, Scheduler, dict]:
    """Same as mk but returns the time-state dict for explicit control."""
    bm = BlockManager(num_blocks, block_size)
    fake = {"now": 100.0}
    sched = Scheduler(bm, block_size=block_size, clock=lambda: fake["now"], **kw)
    return bm, sched, fake


def add(sched: Scheduler, rid: str, n_tokens: int, *, at: float = 100.0,
        priority: int = 0, max_new: int = 3, deadline: float | None = None) -> Request:
    req = Request(rid, [100 + i for i in range(n_tokens)], max_new_tokens=max_new,
                  priority=priority, arrival_time=at, deadline_s=deadline)
    return req if sched.add(req) else req  # caller inspects state on False


def test_fcfs_admission_on_equal_priority() -> None:
    bm, sch, fake = mkc(max_batch=1)
    add(sch, "late", 4, at=100.0)
    add(sch, "early", 4, at=90.0)
    ev = sch.step()
    # Equal priority: earlier arrival wins the only slot (FCFS through aging).
    assert ev["admitted"] == ["early"]
    assert set(sch._running) == {"early"}
    bm.assert_invariants()


def test_kv_budget_gates_admission() -> None:
    bm, sch = mk(num_blocks=4)
    add(sch, "r1", 8)          # 2 blocks
    add(sch, "r2", 8)          # 2 blocks -> pool full
    add(sch, "r3", 8)          # cannot fit its first chunk
    ev = sch.step()
    bm.assert_invariants()
    assert set(ev["admitted"]) == {"r1", "r2"}
    assert "r3" not in sch._running
    r3 = sch._waiting[0]
    assert r3.state == "queued"
    later = sch.step()
    assert "r3" not in later["admitted"]          # still budget-blocked
    bm.assert_invariants()


def test_decode_growth_and_finish_frees_blocks() -> None:
    bm, sch = mk(num_blocks=16)
    req = add(sch, "r1", 8, max_new=2)
    for _ in range(6):
        sch.step()
        bm.assert_invariants()
    assert req.state == "finished"
    assert bm.usage()["used"] == 0               # blocks returned on finish
    bm.assert_invariants()


def test_preemption_never_targets_prefilling() -> None:
    bm, sch, fake = mkc(num_blocks=5, max_chunked=1, prefill_chunk=16)
    keeper = Request("keeper", [1, 2, 3, 4] * 10)   # 40-token prompt: needs 10
    keeper.arrival_time = 99.0                      # blocks > pool of 5, so it
    sch.add(keeper)                                 # stays prefilling forever
    short = Request("short", [9, 9, 9, 9], max_new_tokens=9, arrival_time=99.5)
    sch.add(short)
    fake["now"] = 100.0
    for step_no in range(6):
        ev = sch.step()
        bm.assert_invariants()
        if ev["preempted"]:
            # only the decoding request may be preempted
            assert [t[0] for t in ev["preempted"]] == ["short"]
            k = (sch._running.get("keeper") or
                 next(r for r in sch._waiting if r.request_id == "keeper"))
            assert k.state == "prefilling"            # its blocks stayed hot
            break
    else:
        pytest.fail("expected a preempt to pressure the prefilling request")


def test_preempted_request_requeued_then_completes() -> None:
    bm, sch = mk(num_blocks=5, max_chunked=4)
    big = add(sch, "big", 12, max_new=5, at=99.0)   # prompt+5 -> 5 blocks, pool 5
    pal = add(sch, "pal", 4, max_new=9, at=99.5)    # feed/preempt partner
    for _ in range(60):
        sch.step()
        bm.assert_invariants()
        if big.state == "finished" and pal.state == "finished":
            break
    assert big.state == "finished" and pal.state == "finished"
    assert pal.compute_done_preempt > 0   # pal absorbed the pressure
    assert sch.stats()["recompute_preemptions"] >= 1
    assert bm.usage()["used"] == 0


def test_priority_aging_overtake() -> None:
    bm, sch, fake = mkc(max_batch=1, aging_factor=0.1)
    low = Request("low", [7, 7, 7, 7], priority=0, arrival_time=0.0, max_new_tokens=99)
    hi = Request("hi", [8, 8, 8, 8], priority=5, arrival_time=99.9, max_new_tokens=99)
    fake["now"] = 100.0
    sch.add(low)
    sch.add(hi)
    # low waited 100s: 0 + 0.1*100 = 10 beats fresh priority 5
    ev = sch.step()
    assert ev["admitted"] == ["low"]
    bm.assert_invariants()
    # Without the wait, the priority-5 request wins instead
    bm2, sch2, fake2 = mkc(max_batch=1, aging_factor=0.1)
    lo2 = Request("low2", [7, 7, 7, 7], priority=0, arrival_time=99.9)
    hi2 = Request("hi2", [8, 8, 8, 8], priority=5, arrival_time=99.9)
    fake2["now"] = 100.0
    sch2.add(hi2)
    sch2.add(lo2)
    ev2 = sch2.step()
    assert ev2["admitted"] == ["hi2"]
    bm2.assert_invariants()


def test_chunked_prefill_first_token_at_step8() -> None:
    bm, sch = mk(num_blocks=512, block_size=16, max_chunked=3, prefill_chunk=512)
    long_req = Request("longA", list(range(4096)), max_new_tokens=2, arrival_time=99.0)
    sch.add(long_req)
    shorts = [Request(f"s{i}", list(range(900 + i, 916)), max_new_tokens=8,
                      arrival_time=99.5 + i / 1000) for i in range(3)]
    for r in shorts:
        sch.add(r)
    first_decode_step = None
    short_decoded_steps = 0
    for step_no in range(1, 13):
        ev = sch.step()
        bm.assert_invariants()
        if "longA" in ev["decoding"] and first_decode_step is None:
            first_decode_step = step_no
        if long_req.state == "prefilling":
            assert "longA" not in ev["decoding"]     # prefilling -> no token yet
        short_decoded_steps += len(set(ev["decoding"]) & {s.request_id for s in shorts})
    assert first_decode_step == 8, "4096/512 -> first token on step 8, not 1"
    assert short_decoded_steps >= 12, "short decoders interleave during prefill"
    # capture the mixed prefill/decode step for the VERIFY log paste
    print("CAPTURED step log:", sch.log[7])


def test_slo_rejection_before_allocation() -> None:
    bm, sch, fake = mkc(avg_decode_rate=10.0)
    feasible = Request("ok", [1] * 8, deadline_s=1e9)
    infeasible = Request("doomed", [1] * 8, deadline_s=100.0, arrival_time=100.0)
    fake["now"] = 100.0
    assert sch.add(feasible) is True
    before = bm.usage()
    assert sch.add(infeasible) is False
    assert infeasible.state == "rejected" and infeasible.reason == "slo_infeasible"
    assert infeasible not in sch._waiting
    assert bm.usage() == before               # nothing allocated, nothing queued
    sch.step()
    bm.assert_invariants()
    assert sch.stats()["slo_rejections"] == 1


def test_preemption_waste_accounting() -> None:
    bm, sch = mk(num_blocks=63, block_size=16, max_chunked=4)
    big = add(sch, "big", 1000, max_new=99)   # 63-blocks prompt, pool 63:
    # big preempts ITSELF (recompute) every 9 decode steps; each cycle
    # discards its full prefilled prompt offset (1000 tokens).
    for _ in range(200):
        if sch._stats["recompute_preemptions"] >= 3:
            break
        sch.step()
        bm.assert_invariants()
    assert sch._stats["compute_wasted_tokens"] == 3 * 1000, (
        "each recompute preempt discards the full prompt offset (1000)")
    assert big.compute_done_preempt == 3 * 1000


def test_prefill_policy_ff_vs_df_differ() -> None:
    bm1, sch_ff = mk(num_blocks=64, max_chunked=2, prefill_chunk=8)
    bm2, sch_df = mk(num_blocks=64, max_chunked=2, prefill_chunk=8)
    sch_ff.prefill_policy = "ff"
    assert sch_df.prefill_policy == "rr"
    sch_df.prefill_policy = "df"

    def trace(sch) -> tuple[Request, Request]:
        long_req = Request("A", list(range(24)), max_new_tokens=3, arrival_time=99.0)
        short = Request("B", list(range(31, 35)), max_new_tokens=2, arrival_time=99.5)
        sch.add(long_req)
        sch.add(short)
        return long_req, short

    a_ff, b_ff = trace(sch_ff)
    a_df, b_df = trace(sch_df)
    ev_ff1 = sch_ff.step()  # ff: chunks first -> B promoted AND decoded step 1
    ev_df1 = sch_df.step()  # df: decode pass ran first -> B decodes next step
    bm1.assert_invariants(); bm2.assert_invariants()
    ev_ff2 = sch_ff.step()
    ev_df2 = sch_df.step()
    assert "B" in ev_ff1["decoding"]              # ff: first token same step
    assert "B" not in ev_df1["decoding"]          # df: decode pass was first
    assert "B" in ev_df2["decoding"]              # df: first token one step later
    assert a_ff.prefill_offset == a_df.prefill_offset == 16  # same read-progress


def test_queue_wait_histogram() -> None:
    bm, sch = mk(num_blocks=64)
    add(sch, "w1", 4, at=100.0)
    add(sch, "w2", 4, at=99.0)
    add(sch, "w3", 4, at=98.0)
    sch.step()       # all three (max_batch 8)
    bm.assert_invariants()
    assert sch.stats()["admissions"] == 3
    assert len(sch.queue_wait_ms) == 3                  # one entry per admission
    assert sorted(sch.queue_wait_ms) == [0.0, 1000.0, 2000.0]
    # waits reflect arrival latency: now-arrival = 0s, 1s, 2s -- exactly.


def test_swap_resume_via_prefix_cache() -> None:
    bm = BlockManager(16, block_size=4)
    cache = RadixPrefixCache(bm, block_size=4)
    fake = {"now": 100.0}
    sch = Scheduler(bm, block_size=4, prefix_cache=cache, clock=lambda: fake["now"],
                    max_chunked=4, prefill_chunk=8)
    a = add(sch, "swapper", 8, max_new=60)   # 68 tokens -> 17 blocks, pool 16:
    # A self-preempts (swap); the cache rescues the prompt KV, so re-admission
    # forks the same blocks back with prefill_offset intact (zero re-prefill).
    resumed = False
    saw_swap = False
    for steps in range(1, 300):
        ev = sch.step()
        bm.assert_invariants()
        if ev["preempted"] and ev["preempted"][0][1] == "swap":
            saw_swap = True
            swaps, waste = sch._stats["swap_preemptions"], sch._stats["compute_wasted_tokens"]
            assert waste == 8 * swaps            # one full-prompt discard per swap
        if saw_swap and a.state == "decoding" and a.prefill_offset == 8 and ev["decoding"]:
            resumed = True
            break   # post-swap resume: decoding on the forked cached blocks
    bm.assert_invariants()
    assert resumed, "A must resume decoding from the cache after its swap"
    assert sch._stats["swap_preemptions"] >= 1
    assert sch._stats["recompute_preemptions"] == 0
    assert sch._stats["compute_wasted_tokens"] == 8 * sch._stats["swap_preemptions"]
    assert a.prefill_offset == 8                 # prompt was never re-prefilled
    assert cache.stats()["blocks_cached"] == 2   # its pins stay in the cache


def test_max_batch_cap() -> None:
    bm, sch = mk(num_blocks=64, max_batch=2)
    add(sch, "r1", 4); add(sch, "r2", 4); add(sch, "r3", 4)
    ev = sch.step()
    bm.assert_invariants()
    assert len(ev["admitted"]) == 2
    assert "r3" not in ev["admitted"]


def test_policy_validation() -> None:
    bm, sch = mk()
    with pytest.raises(ValueError):
        Scheduler(bm, block_size=4, prefill_policy="bogus")
    with pytest.raises(ValueError):
        sch.prefill_policy = "bogus"  # type: ignore[assignment]
    sch.prefill_policy = "ff"
    assert sch.prefill_policy == "ff"
    sch.set_aging(0.5)  # no crash; starvation knob live


def test_stats_keys_exact() -> None:
    bm, sch, fake = mkc(num_blocks=32)
    add(sch, "r1", 4, max_new=1, at=100.0)
    add(sch, "r2", 4, max_new=1, at=100.0)
    fake["now"] = 100.5
    sch.step(); sch.step(); sch.step()
    bm.assert_invariants()
    expected = {"recompute_preemptions": 0, "swap_preemptions": 0,
                "compute_wasted_tokens": 0, "slo_rejections": 0,
                "admissions": 2, "avg_queue_wait_ms": 500.0}
    assert sch.stats() == expected


def test_fuzz_500_steps_invariants() -> None:
    bm, sch = mk(num_blocks=64, max_chunked=2)
    rng = random.Random(20261028)
    counter = 0
    states = {"queued", "prefilling", "decoding", "finished", "rejected", "swapped"}
    for step_no in range(500):
        if rng.random() < 0.35 and counter < 40:
            counter += 1
            req = Request(f"f{counter}", [rng.randrange(5) for _ in range(rng.randint(1, 64))],
                          max_new_tokens=rng.randint(1, 8), priority=rng.randint(0, 4),
                          arrival_time=rng.uniform(0, 100),
                          deadline_s=(3000 + step_no) if rng.random() < 0.3 else None)
            req.prefill_offset = rng.choice([0, 0, 0])
            sch.add(req)
        ev = sch.step()
        bm.assert_invariants()
        for r in list(sch._running.values()) + sch._waiting:
            assert r.state in states, f"idle-state violation {r.request_id}: {r.state}"
            assert r.num_generated <= r.max_new_tokens
            assert 0 <= r.prefill_offset <= len(r.prompt)
    sch.stats()
    bm.assert_invariants()

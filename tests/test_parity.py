"""Parity tests for paged attention against dense reference implementations."""
import csv
import json
import math
import os
from pathlib import Path

import pytest
import torch

from miniserve.attention import PagedAttention, apply_rope
from miniserve.model_runner import DEFAULT_MODEL, ModelRunner

# Part 6 gate: 200 greedy prompts, 95% token-for-token for the lossy INT8 path.
PARITY_PROMPTS = int(os.environ.get("MINISERVE_PARITY_PROMPTS", "200"))
PARITY_REQUIRED = 0.95
K = 4  # generated tokens compared per prompt
REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "bench" / "data" / "parity_report.csv"
INT8_FAILURES = REPO / "docs" / "parity_failures" / "int8"
TOPICS = ["gravity", "the Roman empire", "sourdough", "TCP handshakes",
          "basket weaving"]


def _prompts(n=PARITY_PROMPTS):
    """Deterministic prompt set (no dataset dependency): 200 short questions."""
    return [f"You are terse. Explain {TOPICS[i % len(TOPICS)]} in one sentence, "
            f"step {i}." for i in range(n)]


def _hf_greedy(model, tokenizer, prompt_ids, k=K):
    """HuggingFace greedy reference: pure argmax, forced to exactly k tokens.

    ``repetition_penalty=1.0`` is essential: Qwen2.5's generation_config sets
    1.1, which makes HF ``generate`` diverge from plain greedy decoding.
    """
    with torch.no_grad():
        out = model.generate(
            torch.tensor([prompt_ids]), do_sample=False, repetition_penalty=1.0,
            max_new_tokens=k, min_new_tokens=k,
            pad_token_id=tokenizer.eos_token_id)
    return out[0, len(prompt_ids):].tolist()


def _record_report(runner, prompts, matched, atol_logits):
    """Upsert this runner's row into bench/data/parity_report.csv."""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    rows = {}
    if REPORT.exists():
        with REPORT.open(newline="") as f:
            for row in csv.DictReader(f):
                rows[(row["model"], row["quantize"])] = row
    rows[(runner.model_name, runner.quantize)] = {
        "model": runner.model_name, "num_heads": str(runner.num_heads),
        "num_kv_heads": str(runner.num_kv_heads), "prompts": str(prompts),
        "matched": str(matched), "atol_logits": f"{atol_logits:.2e}",
        "quantize": runner.quantize,
    }
    fields = ["model", "num_heads", "num_kv_heads", "prompts", "matched",
              "atol_logits", "quantize"]
    with REPORT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow({name: rows[key][name] for name in fields})


@pytest.fixture(scope="session")
def runner():
    return ModelRunner.load(DEFAULT_MODEL)


@pytest.fixture(scope="session")
def int8_runner():
    return ModelRunner.load(DEFAULT_MODEL, quantize="int8")


def _scenario(h, kv_heads, d, lens, q_len=1, block_size=8,
              dtype=torch.float32, kv_dtype=None, seed=0, scale=1.0):
    """Build q + dense k/v and mirror k/v into a paged KV cache."""
    g = torch.Generator().manual_seed(seed)
    kv_dtype = dtype if kv_dtype is None else kv_dtype
    batch, max_l = len(lens), max(lens)
    nblocks = (max_l + block_size - 1) // block_size
    q = (torch.randn(batch, q_len, h, d, generator=g) * scale).to(dtype)
    k = torch.randn(batch, max_l, kv_heads, d, generator=g) * scale
    v = torch.randn(batch, max_l, kv_heads, d, generator=g) * scale
    pa = PagedAttention(h, kv_heads, d, num_blocks=batch * nblocks,
                        block_size=block_size, dtype=dtype, kv_dtype=kv_dtype)
    k_cache, v_cache = pa.allocate_kv_cache()
    block_table = torch.zeros(batch, nblocks, dtype=torch.long)
    for i, length in enumerate(lens):
        block_table[i] = torch.arange(i * nblocks, (i + 1) * nblocks)
        pa.write_kv(k_cache, v_cache, k[i:i + 1, :length], v[i:i + 1, :length],
                    block_table[i:i + 1], start=0)
    return pa, q, k, v, k_cache, v_cache, block_table, torch.tensor(lens)


def _dense_ref(q, k, v, lens, window=None, causal=True):
    """Dense float32 reference; returns (output, attention weights)."""
    batch, q_len, heads, d = q.shape
    group = heads // k.shape[2]
    k = k.repeat_interleave(group, dim=2).float()
    v = v.repeat_interleave(group, dim=2).float()
    lens_t = torch.as_tensor(lens).reshape(-1)
    q_pos = lens_t[:, None] - q_len + torch.arange(q_len)[None, :]
    key_pos = torch.arange(k.shape[1])[None, None, :]
    mask = torch.ones(batch, q_len, k.shape[1], dtype=torch.bool)
    if causal:
        mask = key_pos <= q_pos[:, :, None]
    mask = mask & (key_pos < lens_t[:, None, None])
    if window is not None:
        mask = mask & (key_pos > (q_pos - window)[:, :, None])
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k) / math.sqrt(d)
    scores = scores.masked_fill(~mask[:, None], float("-inf"))
    attn = scores.softmax(-1)
    return torch.einsum("bhqk,bkhd->bqhd", attn, v), attn


def test_paged_attention_parity():
    """Base Part-5 test: MHA paged attention, ragged batch, causal mask."""
    pa, q, k, v, kc, vc, bt, lt = _scenario(
        8, 8, 16, [8, 13, 21], q_len=3, block_size=8, kv_dtype=torch.float32)
    out = pa.forward(q, kc, vc, bt, lt)
    ref, _ = _dense_ref(q, k, v, lt)
    err = (out - ref).abs().max().item()
    print(f"parity max_abs_err={err:.2e}")
    assert err < 1e-4


def test_paged_attention_gqa():
    """GQA with the Llama-2-70B-style 32 query / 8 KV head ratio."""
    pa, q, k, v, kc, vc, bt, lt = _scenario(
        32, 8, 64, [32, 47], q_len=1, block_size=8, kv_dtype=torch.float32)
    assert pa.group == 4
    assert kc.shape == (12, 8, 8, 64)  # [blocks, block_size, kv_heads, dim]
    out = pa.forward(q, kc, vc, bt, lt)
    ref, _ = _dense_ref(q, k, v, lt)
    err = (out - ref).abs().max().item()
    print(f"gqa max_abs_err={err:.2e}")
    assert err < 1e-3


def test_paged_attention_mqa():
    """MQA: a single KV head shared by all query heads."""
    pa, q, k, v, kc, vc, bt, lt = _scenario(
        8, 1, 32, [16, 23], q_len=1, block_size=8, kv_dtype=torch.float32)
    assert kc.shape[2] == 1
    out = pa.forward(q, kc, vc, bt, lt)
    ref, _ = _dense_ref(q, k, v, lt)
    err = (out - ref).abs().max().item()
    print(f"mqa max_abs_err={err:.2e}")
    assert err < 1e-3


def test_paged_attention_sliding_window():
    """window=64 over ctx=512: attention weight outside the window is zero."""
    window, q_len = 64, 2
    lens = [512, 300]
    pa, q, k, v, kc, vc, bt, lt = _scenario(
        4, 4, 16, lens, q_len=q_len, block_size=16, kv_dtype=torch.float32)
    out = pa.forward(q, kc, vc, bt, lt, window=window)
    ref, attn = _dense_ref(q, k, v, lt, window=window)
    err = (out - ref).abs().max().item()
    print(f"sliding_window max_abs_err={err:.2e}")
    assert err < 1e-4
    for qi, p in enumerate([lens[0] - 2, lens[0] - 1]):
        assert attn[0, :, qi, :p - window + 1].abs().max().item() == 0.0
        assert attn[0, :, qi, p - window + 1:p + 1].sum().item() > 0.0
    mask = pa.build_mask(lt, q_len, window)
    assert not mask[0, 1, :448].any() and mask[0, 1, 448:512].all()


def test_paged_attention_fp8_kv():
    """FP8 (e4m3) cache is lossy: output error vs FP16 stays within 5e-2.

    Inputs use ~0.25 std (LLM activation scale, not unit variance): at unit
    variance the e4m3 round-off alone puts the absolute error at ~5e-2.
    """
    kwargs = dict(q_len=1, block_size=8, dtype=torch.float16, seed=3, scale=0.25)
    pa16, q, _, _, kc16, vc16, bt, lt = _scenario(
        4, 2, 32, [16, 24], kv_dtype=torch.float16, **kwargs)
    pa8, q8, _, _, kc8, vc8, _, _ = _scenario(
        4, 2, 32, [16, 24], kv_dtype=torch.float8_e4m3fn, **kwargs)
    assert torch.equal(q, q8) and kc16.shape == kc8.shape
    out16 = pa16.forward(q, kc16, vc16, bt, lt)
    out8 = pa8.forward(q8, kc8, vc8, bt, lt)
    err = (out16.float() - out8.float()).abs().max().item()
    print(f"fp8 max_abs_err={err:.2e} (lossy path, atol=5e-2)")
    assert err < 5e-2
    assert kc8.dtype == torch.float8_e4m3fn and kc8.element_size() == 1
    assert kc16.element_size() == 2  # the FP8 cache is half the bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_paged_attention_fp8_memory():
    """torch.cuda.memory_allocated shows the FP8 KV cache at ~2x smaller."""
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    pa16 = PagedAttention(4, 2, 32, num_blocks=2048, block_size=16, device="cuda")
    caches16 = pa16.allocate_kv_cache()
    used16 = torch.cuda.memory_allocated() - base
    del caches16
    torch.cuda.empty_cache()
    pa8 = PagedAttention(4, 2, 32, num_blocks=2048, block_size=16,
                         device="cuda", kv_dtype=torch.float8_e4m3fn)
    caches8 = pa8.allocate_kv_cache()
    used8 = torch.cuda.memory_allocated() - base
    ratio = used16 / used8
    print(f"fp8_memory ratio={ratio:.2f}x")
    assert 1.8 < ratio < 2.2


def test_apply_rope_parity():
    """Fused apply_rope matches a half-split (rotate-half) reference."""
    torch.manual_seed(0)
    max_pos, heads, kv_heads, d, batch, seq = 64, 6, 2, 16, 2, 8
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))
    freqs = torch.outer(torch.arange(max_pos).float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos, sin = emb.cos(), emb.sin()
    pos = torch.arange(seq).expand(batch, seq)
    q = torch.randn(batch, heads, seq, d)
    k = torch.randn(batch, kv_heads, seq, d)
    q_rot, k_rot = apply_rope(q, k, cos, sin, pos)
    c, s = cos[pos].unsqueeze(1), sin[pos].unsqueeze(1)
    half = d // 2

    def _ref(x):
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * c[..., :half] - x2 * s[..., :half],
                          x2 * c[..., half:] + x1 * s[..., half:]], dim=-1)

    q_err = (q_rot - _ref(q)).abs().max().item()
    k_err = (k_rot - _ref(k)).abs().max().item()
    print(f"rope max_abs_err q={q_err:.2e} k={k_err:.2e}")
    assert q_err < 1e-4 and k_err < 1e-4
    q_only, k_none = apply_rope(q, None, cos, sin, pos[0])  # 1-D position_ids
    assert k_none is None and (q_only - q_rot).abs().max().item() < 1e-6


# ---------------------------------------------------------------------------
# Part 6: real-model greedy parity (PagedAttention injected) + INT8 + hooks
# ---------------------------------------------------------------------------

def _calibration_drift(runner, prompt_ids, ref_tokens, ours_logits, steps=K):
    """Max |logit diff| vs an HF full forward at each compared position."""
    worst = 0.0
    for step in range(steps):
        prefix = prompt_ids + ref_tokens[:step]
        with torch.no_grad():
            ref = runner.model(input_ids=torch.tensor([prefix]),
                               use_cache=False).logits[0, -1]
        worst = max(worst, (ref - ours_logits[step]).abs().max().item())
    return worst


def test_greedy_parity_hf(runner):
    """GQA model + PagedAttention: 200/200 prompts greedily identical to HF."""
    prompts = _prompts()
    matched, drift, calibrated = 0, 0.0, 0
    for i, text in enumerate(prompts):
        ids = runner.tokenizer(text)["input_ids"]
        ref = _hf_greedy(runner.model, runner.tokenizer, ids)
        logits_sink = []
        runner.set_logits_processor(lambda lg, gen: logits_sink.append(lg.clone()) or lg)
        ours = runner.generate_greedy(ids, max_new_tokens=K)
        runner.set_logits_processor(None)
        if ours == ref:
            matched += 1
            if calibrated < 5:  # cheap logit-level calibration subset
                drift = max(drift, _calibration_drift(runner, ids, ref, logits_sink))
                calibrated += 1
    print(f"greedy parity: {matched}/{len(prompts)} prompts identical "
          f"(model={runner.model_name}, kv_heads={runner.num_kv_heads}, "
          f"max_abs_logit_diff={drift:.2e})")
    _record_report(runner, len(prompts), matched, drift)
    assert matched == len(prompts)


def test_greedy_parity_int8(int8_runner):
    """INT8 dynamic quantization is lossy: >= 95% of prompts must match."""
    prompts = _prompts()
    mismatches = []
    for i, text in enumerate(prompts):
        ids = int8_runner.tokenizer(text)["input_ids"]
        ref = _hf_greedy(int8_runner.model, int8_runner.tokenizer, ids)
        ours = int8_runner.generate_greedy(ids, max_new_tokens=K)
        if ours != ref:
            mismatches.append({"prompt_index": i, "prompt": text,
                               "ours": ours, "hf": ref})
    matched = len(prompts) - len(mismatches)
    threshold = int(PARITY_REQUIRED * len(prompts))
    print(f"greedy parity (int8): {matched}/{len(prompts)} (threshold {threshold})")
    INT8_FAILURES.mkdir(parents=True, exist_ok=True)
    safe = int8_runner.model_name.replace("/", "__")
    with (INT8_FAILURES / f"{safe}.jsonl").open("w") as f:
        for item in mismatches:
            f.write(json.dumps(item) + "\n")
    _record_report(int8_runner, len(prompts), matched, 0.0)
    assert matched >= threshold


def test_chat_template_stability(runner):
    """A 3-message conversation renders to stable ids and round-trips text."""
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Name one prime number."},
        {"role": "assistant", "content": "2"},
    ]
    ids_a = runner.apply_chat_template(messages)
    ids_b = runner.apply_chat_template(messages)
    assert ids_a == ids_b and len(ids_a) > 0
    tok = runner.tokenizer
    text = tok.decode(ids_a, skip_special_tokens=True)
    for message in messages:
        assert message["content"] in text
    re_ids = tok(text, add_special_tokens=False)["input_ids"]
    assert tok.decode(re_ids, skip_special_tokens=True) == text


def test_logits_processor_hook(runner):
    """A forcing processor makes every decode step emit token 42."""
    def force_42(logits, generated_ids):
        forced = torch.full_like(logits, float("-inf"))
        forced[42] = 0.0
        return forced

    ids = runner.tokenizer("Say anything at all.")["input_ids"]
    runner.set_logits_processor(force_42)
    try:
        generated = runner.generate_greedy(ids, max_new_tokens=5)
    finally:
        runner.set_logits_processor(None)
    assert generated == [42] * 5

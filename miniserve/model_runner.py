"""miniserve.model_runner -- HF model hosting with PagedAttention injected.

One ``ModelRunner`` wraps a causal-LM (Qwen2/Llama-family) and replaces every
attention module's ``forward`` with the same math running over our paged KV
cache (``miniserve.attention.PagedAttention``).  The original forward is kept
and used whenever ``_active`` is False, so the very same weights can produce
the HuggingFace reference output for parity tests.

Documented choices:
- Model: ``Qwen/Qwen2.5-0.5B-Instruct`` (GQA, 14 query / 2 KV heads).  It is
  the specified fallback: ``meta-llama/Llama-3.2-1B-Instruct`` is gated (401
  without an HF token).  Head counts are read from ``model.config``.
- Weights load in float32: CPU has no fast bf16 matmul, and parity wants the
  highest-fidelity path.  The paged KV cache is float32 for the same reason.
- Positions: ``generate_greedy`` passes explicit ``position_ids`` so the
  model's RoPE sees absolute positions even though no HF cache is used.
- INT8: ``torch.ao.quantization.quantize_dynamic`` on every ``nn.Linear``.
  Dynamic INT8 is lossy for greedy decoding; the parity gate tolerates
  ``PARITY_REQUIRED`` mismatches (see tests/test_parity.py).
- Chat template: the tokenizer's own template is used when present;
  otherwise a documented ``<|role|>`` fallback format is applied.  This is
  NOT claimed to be HF-template-identical on tokenizers without a template.
- Structured output: ``set_logits_processor(fn)`` installs the Part 7 hook
  ``fn(logits, generated_ids) -> logits``, applied before every argmax.
"""

from __future__ import annotations

import types
from typing import Callable, List, Optional, Sequence

import torch

from miniserve.attention import PagedAttention
from miniserve.blocks import BlockManager

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # Llama-3.2-1B is gated


def _paged_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[object] = None,
    **kwargs,
):
    """Drop-in HF attention forward: identical math, paged KV cache."""
    runner: "ModelRunner" = self._paged_runner
    if not runner._active:  # HuggingFace reference path (own cache, ground truth)
        return self._hf_forward(hidden_states, position_embeddings,
                                attention_mask, past_key_values, **kwargs)
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    batch, seq, _ = hidden_states.shape
    shape = (batch, seq, -1, self.head_dim)
    q = self.q_proj(hidden_states).view(shape).transpose(1, 2)
    k = self.k_proj(hidden_states).view(shape).transpose(1, 2)
    v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
    state = runner.layers[self.layer_idx]
    state.pa.write_kv(state.k_cache, state.v_cache, k.transpose(1, 2),
                      v.transpose(1, 2), state.block_table, start=runner.start)
    ctx_len = runner.start + seq
    out = state.pa.forward(q.transpose(1, 2), state.k_cache, state.v_cache,
                           state.block_table, [ctx_len], window=runner.window)
    out = out.reshape(batch, seq, -1).contiguous()  # [b, seq, heads * dim]
    return self.o_proj(out), None


class _LayerState:
    """Per-layer paged attention + its KV tensors and block table."""

    __slots__ = ("pa", "k_cache", "v_cache", "block_table")

    def __init__(self, pa: PagedAttention, k_cache: torch.Tensor, v_cache: torch.Tensor):
        self.pa, self.k_cache, self.v_cache = pa, k_cache, v_cache
        self.block_table: torch.Tensor | None = None


class ModelRunner:
    """Causal LM with PagedAttention injected into every attention layer."""

    def __init__(self, model, tokenizer, model_name: str = DEFAULT_MODEL,
                 quantize: str = "none", block_size: int = 16,
                 max_blocks: int = 128, window: Optional[int] = None) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.quantize = quantize
        cfg = model.config
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", None) or self.num_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.num_heads
        self.block_size, self.max_blocks = block_size, max_blocks
        self.window = window
        self.start = 0
        self._active = False
        self.logits_processor: Callable = lambda logits, generated: logits
        self.last_logits: torch.Tensor | None = None
        self.layers: List[_LayerState] = []
        for _ in range(cfg.num_hidden_layers):
            pa = PagedAttention(self.num_heads, self.num_kv_heads, self.head_dim,
                                num_blocks=max_blocks, block_size=block_size,
                                dtype=torch.float32, kv_dtype=torch.float32)
            self.layers.append(_LayerState(pa, *pa.allocate_kv_cache()))
        for layer in self.model.model.layers:
            attn = layer.self_attn
            attn._paged_runner = self
            attn._hf_forward = attn.forward
            attn.forward = types.MethodType(_paged_attention_forward, attn)

    @classmethod
    def load(cls, model_name: str = DEFAULT_MODEL, quantize: str = "none", **kwargs):
        """Load tokenizer + weights; ``quantize`` is "none" or "int8"."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if quantize not in ("none", "int8"):
            raise ValueError(f"unknown quantize: {quantize}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        if quantize == "int8":
            # lm_head stays in fp32 (standard recipe): it is a 152k-vocab
            # output head; quantizing it distorts logits far beyond any
            # tolerance and buys ~2% of the model's parameters.
            from torch.ao.quantization import default_dynamic_qconfig, quantize_dynamic

            spec = {module: default_dynamic_qconfig
                    for name, module in model.named_modules()
                    if isinstance(module, torch.nn.Linear)
                    and not name.endswith("lm_head")}
            model = quantize_dynamic(model, qconfig_spec=spec)
        return cls(model, tokenizer, model_name, quantize, **kwargs)

    @property
    def info(self) -> dict:
        """Metadata recorded in bench/data/parity_report.csv."""
        return {"model": self.model_name, "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads, "quantize": self.quantize}

    def set_logits_processor(self, fn: Optional[Callable]) -> None:
        """Install the Part 7 structured-output hook (``None`` resets it)."""
        self.logits_processor = fn or (lambda logits, generated: logits)

    def apply_chat_template(self, messages: Sequence[dict],
                            add_generation_prompt: bool = True) -> List[int]:
        """Render ``messages`` to token ids with the tokenizer's template.

        Falls back to a documented simple format when the tokenizer has no
        chat template: ``<|role|>\\ncontent\\n`` per message, plus an
        ``<|assistant|>\\n`` generation prompt.  No HF-template parity is
        claimed for that fallback.
        """
        tok = self.tokenizer
        if getattr(tok, "chat_template", None):
            kwargs = dict(tokenize=True, add_generation_prompt=add_generation_prompt)
            try:  # transformers>=5 returns BatchEncoding unless return_dict=False
                rendered = tok.apply_chat_template(list(messages),
                                                   return_dict=False, **kwargs)
            except TypeError:  # transformers 4.x has no return_dict
                rendered = tok.apply_chat_template(list(messages), **kwargs)
            if isinstance(rendered, dict):
                rendered = rendered["input_ids"]
            return [int(token) for token in rendered]
        text = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in messages)
        if add_generation_prompt:
            text += "<|assistant|>\n"
        return list(tok(text, add_special_tokens=False)["input_ids"])

    @torch.no_grad()
    def generate_greedy(self, prompt_ids: Sequence[int],
                        max_new_tokens: int = 4) -> List[int]:
        """Greedy decode through the paged path, one token at a time."""
        ids = torch.as_tensor(list(prompt_ids), dtype=torch.long).reshape(1, -1)
        total = ids.shape[1] + max_new_tokens
        bm = BlockManager(self.max_blocks, self.block_size)
        needed = (total + self.block_size - 1) // self.block_size
        table = torch.tensor(bm.allocate(needed, "generate"), dtype=torch.long)[None]
        for state in self.layers:
            state.block_table = table
        generated: List[int] = []
        self.start = 0
        self._active = True
        try:
            for step in range(max_new_tokens):
                cur = ids if step == 0 else ids[:, -1:]
                pos = torch.arange(self.start, self.start + cur.shape[1])[None]
                out = self.model(input_ids=cur, position_ids=pos, use_cache=False)
                logits = self.logits_processor(out.logits[0, -1], list(generated))
                self.last_logits = logits.detach().clone()
                token = int(logits.argmax())
                generated.append(token)
                ids = torch.cat([ids, torch.tensor([[token]])], dim=1)
                self.start += cur.shape[1]
        finally:
            self._active = False
        return generated

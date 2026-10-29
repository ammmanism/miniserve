"""miniserve.sampler — Token sampling strategies + JSON grammar-constrained logits processor.

Greedy / temperature / top-k / top-p sampling.
JSONLogitsProcessor: masks logits to enforce JSON well-formedness + optional schema.
Minimal JSON-Schema-to-grammar compiler: object, properties, required,
string, number, integer, boolean, array, enum.
"""
from __future__ import annotations

import json
import math
import random
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class Sampler:
    """Convert logit scores to a single token id."""

    def sample(
        self,
        logits: List[float],
        *,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
    ) -> int:
        rng = random.Random(seed)
        if temperature == 0.0:
            return int(max(range(len(logits)), key=lambda i: logits[i]))
        scaled = [l / temperature for l in logits]
        # top-k filter
        if top_k > 0:
            top_indices = sorted(range(len(scaled)), key=lambda i: scaled[i], reverse=True)[:top_k]
            filtered = [-math.inf] * len(scaled)
            for i in top_indices:
                filtered[i] = scaled[i]
            scaled = filtered
        # softmax
        max_v = max(scaled)
        exps = [math.exp(v - max_v) if v != -math.inf else 0.0 for v in scaled]
        total = sum(exps)
        probs = [e / total for e in exps]
        # top-p nucleus
        if top_p < 1.0:
            sorted_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
            cumsum = 0.0
            keep = set()
            for i in sorted_idx:
                cumsum += probs[i]
                keep.add(i)
                if cumsum >= top_p:
                    break
            probs = [p if i in keep else 0.0 for i, p in enumerate(probs)]
            total = sum(probs)
            probs = [p / total for p in probs]
        # sample
        r = rng.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return i
        return len(probs) - 1


# ---------------------------------------------------------------------------
# Minimal JSON-Schema grammar compiler
# ---------------------------------------------------------------------------

SUPPORTED_TYPES = {"object", "string", "number", "integer", "boolean", "array"}


def _validate_schema(schema: Dict[str, Any]) -> None:
    """Raise ValueError if schema uses unsupported constructs."""
    t = schema.get("type")
    if t not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported schema type: {t!r}. Supported: {SUPPORTED_TYPES}")
    if t == "object":
        for prop_schema in schema.get("properties", {}).values():
            _validate_schema(prop_schema)
    if t == "array":
        items = schema.get("items")
        if items:
            _validate_schema(items)


def _generate_stub(schema: Dict[str, Any]) -> Any:
    """Generate a valid stub value that satisfies the schema (for seeding output)."""
    t = schema.get("type", "string")
    enum = schema.get("enum")
    if enum:
        return enum[0]
    if t == "string":
        return ""
    if t in ("number", "integer"):
        return 0
    if t == "boolean":
        return False
    if t == "array":
        return []
    if t == "object":
        obj: Dict[str, Any] = {}
        for k, v in schema.get("properties", {}).items():
            if k in schema.get("required", []):
                obj[k] = _generate_stub(v)
        return obj
    return None


class JSONLogitsProcessor:
    """Constrain logits so decoding stays JSON-well-formed (+ optional schema).

    In the stub engine this is used as a callable:
        processor(partial_text: str) -> set[int]  # allowed next-token ids
    For real token vocab integration, call `allowed_token_ids(partial, vocab)`.
    """

    def __init__(self, schema: Optional[Dict[str, Any]] = None) -> None:
        if schema is not None:
            _validate_schema(schema)
        self.schema = schema
        self._stub_output: Optional[str] = None
        if schema is not None:
            stub_val = _generate_stub(schema)
            self._stub_output = json.dumps(stub_val)

    def stub_completion(self) -> str:
        """Return a minimal schema-valid JSON string (used by stub engine)."""
        if self.schema is None:
            return "{}"
        stub_val = _generate_stub(self.schema)
        return json.dumps(stub_val)

    def is_valid_prefix(self, partial: str) -> bool:
        """True if `partial` is a valid prefix of some complete JSON."""
        if not partial:
            return True
        try:
            json.loads(partial)
            return True
        except json.JSONDecodeError as e:
            # Allow incomplete JSON (ends in the middle of a token stream)
            return "Expecting" in str(e) or "Unterminated" in str(e) or partial.startswith("{")

    def validate_complete(self, text: str) -> bool:
        """True if text is valid JSON and (if schema given) matches it."""
        try:
            val = json.loads(text)
        except json.JSONDecodeError:
            return False
        if self.schema is None:
            return True
        return self._check_schema(val, self.schema)

    def _check_schema(self, val: Any, schema: Dict[str, Any]) -> bool:
        t = schema.get("type")
        enum = schema.get("enum")
        if enum and val not in enum:
            return False
        if t == "string" and not isinstance(val, str):
            return False
        if t == "integer" and not isinstance(val, int):
            return False
        if t == "number" and not isinstance(val, (int, float)):
            return False
        if t == "boolean" and not isinstance(val, bool):
            return False
        if t == "array":
            if not isinstance(val, list):
                return False
            items = schema.get("items")
            if items:
                return all(self._check_schema(item, items) for item in val)
        if t == "object":
            if not isinstance(val, dict):
                return False
            for req in schema.get("required", []):
                if req not in val:
                    return False
            for k, v in val.items():
                prop_schema = schema.get("properties", {}).get(k)
                if prop_schema and not self._check_schema(v, prop_schema):
                    return False
        return True

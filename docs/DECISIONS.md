# Design Decisions

## Decision Log

### 001 — Greedy-only sampling for parity gate

**Date:** TBD
**Decision:** Start with greedy (argmax) sampling only.
**Rationale:** Deterministic output enables exact parity testing vs HuggingFace.
**Status:** Active

### 002 — Single-GPU only (no tensor parallelism)

**Date:** TBD
**Decision:** Target single-GPU deployment initially.
**Rationale:** Reduces complexity; TP can be layered on later.
**Status:** Active

### 003 — Pydantic v2 for protocol models

**Date:** TBD
**Decision:** Use Pydantic v2 for all request/response schemas.
**Rationale:** Better performance, native JSON schema support.
**Status:** Active

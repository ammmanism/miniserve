# Gap Analysis: miniserve vs vLLM

## Purpose

Track which vLLM features miniserve has implemented, which are planned,
and which are explicitly out of scope.

## Feature Matrix

| Feature | vLLM | miniserve | Priority | Status |
|---------|------|-----------|----------|--------|
| Continuous batching | ✅ | ❌ | P0 | Planned |
| PagedAttention | ✅ | ❌ | P0 | Planned |
| Prefix caching | ✅ | ❌ | P1 | Planned |
| Speculative decoding | ✅ | ❌ | P2 | Planned |
| Tensor parallelism | ✅ | ❌ | P2 | Out of scope |
| CUDA graphs | ✅ | ❌ | P1 | Planned |
| Chunked prefill | ✅ | ❌ | P1 | Planned |
| Overlap scheduling | ✅ | ❌ | P1 | Planned |
| LoRA serving | ✅ | ❌ | P3 | Out of scope |
| Quantization (AWQ/GPTQ) | ✅ | ❌ | P2 | Planned |

## Notes

- P0 features are required for greedy parity gate
- No feature is "done" without passing parity tests and committed bench CSVs

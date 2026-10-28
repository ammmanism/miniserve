# miniserve

Minimal, single-file-friendly LLM serving engine — built for correctness first, speed second.

## Gap Analysis vs vLLM

| Feature | vLLM | miniserve | Status |
|---------|------|-----------|--------|
| Continuous batching | ✅ | | |
| PagedAttention | ✅ | | |
| Prefix caching | ✅ | | |
| Speculative decoding | ✅ | | |
| Tensor parallelism | ✅ | | |
| CUDA graphs | ✅ | | |
| Chunked prefill | ✅ | | |
| Overlap scheduling | ✅ | | |

## Quick Start

```bash
uv sync
python -m miniserve.server --model <model_name>
```

## Benchmarks

See `bench/data/` for raw CSV results. No performance claims without committed data.

## License

MIT

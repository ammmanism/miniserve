"""miniserve.engine — Top-level inference engine orchestrating model execution.

Coordinates the scheduler, model runner, and KV-cache management to process
incoming requests through prefill and decode stages.

TODO: Implement LLMEngine class with step() loop.
"""

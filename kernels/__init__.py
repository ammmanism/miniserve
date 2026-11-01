"""kernels -- Custom Triton kernels for miniserve.

Houses hand-written Triton kernels for fused operations.  The kernels import
on CPU (the Triton imports are guarded); launching requires CUDA, and each
module documents which parts are CPU-verified (reference algorithms) and
which are CUDA-only (kernel launches).
"""

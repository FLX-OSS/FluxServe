# flux-kernel

Standalone kernel package for FluxServe experiments.

Install in editable mode from the FluxServe repo root:

```bash
pip install -e flux-kernel/python/ --no-build-isolation
```

The install builds CUDA RMSNorm shared library by default.
Set `FLUX_KERNEL_SKIP_CUDA_BUILD=1` to skip native CUDA compilation.

The public APIs are exposed from both `flux_kernel` and `flux_kernel.ops`:

```python
from flux_kernel.ops import rmsnorm
from flux_kernel.cuda.rmsnorm import rmsnorm_fused_parallel
from flux_kernel import (
    apply_rope_with_cos_sin_cache_inplace,
    moe_align_block_size,
    moe_fused_gate,
    silu_and_mul,
)
```
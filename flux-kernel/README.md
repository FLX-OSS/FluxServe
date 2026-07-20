# flux-kernel

Standalone kernel package for FluxServe experiments.

Install in editable mode from the FluxServe repo root:

```bash
pip install -e flux-kernel/python/ --no-build-isolation
```

The install builds CUDA RMSNorm shared library by default.
Set `FLUX_KERNEL_SKIP_CUDA_BUILD=1` to skip native CUDA compilation.

The RMSNorm APIs are exposed as:

```python
from flux_kernel.ops import rmsnorm
from flux_kernel.cuda.rmsnorm import rmsnorm_fused_parallel
```

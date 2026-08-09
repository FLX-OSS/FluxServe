# Copyright (c) 2026 FLUX-OSS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from flux_kernel.ops.activation import silu_and_mul
from flux_kernel.ops.moe import moe_align_block_size, moe_fused_gate
from flux_kernel.ops.rmsnorm import qk_rmsnorm, rmsnorm
from flux_kernel.ops.rope import apply_rope_with_cos_sin_cache_inplace

__all__ = [
    "rmsnorm",
    "qk_rmsnorm",
    "silu_and_mul",
    "moe_fused_gate",
    "moe_align_block_size",
    "apply_rope_with_cos_sin_cache_inplace",
]

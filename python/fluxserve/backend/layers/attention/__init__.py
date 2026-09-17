
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

from fluxserve.backend.layers.attention.base import (
    AttentionForwardConfig,
    DenseAttention,
    apply_qk_norm,
    repeat_kv,
)
from fluxserve.backend.layers.attention.flashinfer import (
    FlashInferRaggedAttention,
    FlashInferRaggedPrefillAttention,
)
from fluxserve.backend.layers.attention.forward import (
    AttentionForward,
)
from fluxserve.backend.layers.attention.fa4 import (
    FA4PagedAttention,
    fa4_package_version,
    load_fa4_varlen_func,
    validate_fa4_runtime,
)
from fluxserve.backend.layers.attention.metadata import (
    PagedAttentionMetadata,
    build_block_diffusion_paged_metadata,
)
from fluxserve.backend.layers.attention.selector import (
    AttentionBackend,
    AttentionBackendCapability,
    AttentionBackendSelector,
    AttentionBatchCharacteristics,
    AttentionExecutionPlan,
    AttentionMaskKind,
    AttentionPhase,
    AttentionScheduleCandidate,
    AttentionSchedulePlan,
    AttentionWorkItem,
    CallableCostModel,
)

__all__ = [
    "AttentionBackend",
    "AttentionBackendCapability",
    "AttentionBackendSelector",
    "AttentionBatchCharacteristics",
    "AttentionExecutionPlan",
    "AttentionForward",
    "AttentionForwardConfig",
    "AttentionMaskKind",
    "AttentionPhase",
    "AttentionScheduleCandidate",
    "AttentionSchedulePlan",
    "AttentionWorkItem",
    "CallableCostModel",
    "DenseAttention",
    "FA4PagedAttention",
    "FlashInferRaggedAttention",
    "FlashInferRaggedPrefillAttention",
    "PagedAttentionMetadata",
    "apply_qk_norm",
    "build_block_diffusion_paged_metadata",
    "fa4_package_version",
    "load_fa4_varlen_func",
    "repeat_kv",
    "validate_fa4_runtime",
]

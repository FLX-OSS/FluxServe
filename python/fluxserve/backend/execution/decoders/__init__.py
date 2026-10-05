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

"""
    Model-specific decoder implementations.
"""

from fluxserve.backend.execution.decoders.diffusion_gemma import (
    DiffusionGemmaDecoder,
    DiffusionGemmaSamplingConfig,
)
from fluxserve.backend.execution.decoders.llada import load_decoder
from fluxserve.backend.execution.decoders.llada.base import ParallelDecoder
from fluxserve.backend.execution.decoders.llada.hierarchy import HierarchyDecoder
from fluxserve.backend.execution.decoders.llada.joint_threshold import JointThresholdDecoder
from fluxserve.backend.execution.decoders.llada.levenshtein import LevenshteinJointDecoder
from fluxserve.backend.execution.decoders.llada.static import StaticParallelDecoder
from fluxserve.backend.execution.decoders.llada.threshold import (
    CreditThresholdParallelDecoder,
    ThresholdParallelDecoder,
)
from fluxserve.backend.execution.decoders.nemotron import (
    NemotronSampling,
    NemotronThresholdDecoder,
    ThinkingBudget,
)

__all__ = [
    "CreditThresholdParallelDecoder",
    "DiffusionGemmaDecoder",
    "DiffusionGemmaSamplingConfig",
    "HierarchyDecoder",
    "JointThresholdDecoder",
    "LevenshteinJointDecoder",
    "NemotronSampling",
    "NemotronThresholdDecoder",
    "ParallelDecoder",
    "StaticParallelDecoder",
    "ThinkingBudget",
    "ThresholdParallelDecoder",
    "load_decoder",
]

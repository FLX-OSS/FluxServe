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
    I-DLM FlashInfer paged ISD runner.
"""

from fluxserve.backend.execution.runners.base import BlockDiffusionRunner
from fluxserve.backend.execution.runners.idlm.fa4 import IDLMPagedRunner
from fluxserve.backend.execution.runners.nemotron.flashinfer import (
    NemotronFlashInferDiffusionRunner,
)
from fluxserve.backend.layers.attention.nemotron_flashinfer import (
    require_flashinfer_token_paged,
)


class IDLMFlashInferRunner(IDLMPagedRunner):
    """The same ISD loop over FlashInfer token paging (Ampere and newer)."""

    paged_attention_backend = "flashinfer"

    def _init_paged_backend(self, *args, **kwargs):
        config = kwargs.get("runner_config", args[2] if len(args) > 2 else None)
        model_config = kwargs.get("model_config", args[0] if args else None)
        if config is None or config.attention_backend != "flashinfer":
            raise ValueError("I-DLM FlashInfer requires attention_backend='flashinfer'")
        if config.kv_cache_layout != "paged":
            raise ValueError("I-DLM FlashInfer requires kv_cache_layout='paged'")
        if config.enable_prefill_cuda_graph:
            raise ValueError(
                "I-DLM FlashInfer supports --use-decode-cuda-graph; prefill "
                "graph is unsupported"
            )
        if int(config.page_size or config.block_length) % 16:
            raise ValueError("I-DLM paged page_size must be a multiple of 16")
        if config.enable_decode_cuda_graph:
            NemotronFlashInferDiffusionRunner._validate_decode_graph_config(
                config, kwargs, args
            )
        self._validate_architecture(model_config)
        require_flashinfer_token_paged()
        BlockDiffusionRunner.__init__(self, *args, **kwargs)
        self._paged_request_slots = {}
        self.fa4_graph_runner = None


__all__ = ["IDLMFlashInferRunner"]

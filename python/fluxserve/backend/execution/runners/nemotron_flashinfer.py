# Copyright (c) 2026 FLUX-OSS
# SPDX-License-Identifier: MIT

"""Nemotron eager paging on the standard FlashInfer prefill API.

Reuse the tested Nemotron block and speculation loops, including per-request
prefix/seed ownership. Only backend initialization and attention dispatch differ
from FA4; LLaDA's block-extend runner is never involved.
"""

from fluxserve.backend.execution.runners.block_diffusion import BlockDiffusionRunner
from fluxserve.backend.execution.runners.nemotron_fa4 import NemotronFA4DiffusionRunner
from fluxserve.backend.execution.runners.nemotron_selfspec_paged import (
    NemotronSelfSpecPagedRunner,
)
from fluxserve.backend.layers.attention.flashinfer_token import (
    require_flashinfer_token_paged,
)


class NemotronFlashInferDiffusionRunner(NemotronFA4DiffusionRunner):
    paged_attention_backend = "flashinfer"

    def _init_paged_backend(self, *args, **kwargs):
        config = kwargs.get("runner_config", args[2] if len(args) > 2 else None)
        model_config = kwargs.get("model_config", args[0] if args else None)
        if config is None or config.attention_backend != "flashinfer":
            raise ValueError("Nemotron FlashInfer requires attention_backend='flashinfer'")
        if config.kv_cache_layout != "paged":
            raise ValueError("Nemotron FlashInfer requires kv_cache_layout='paged'")
        if (config.enable_cuda_graph or config.enable_prefill_cuda_graph
                or config.enable_decode_cuda_graph):
            raise ValueError("Nemotron FlashInfer is eager; CUDA graphs require FA4")
        if int(config.page_size or config.block_length) % 16:
            raise ValueError("Nemotron paged page_size must be a multiple of 16")
        self._validate_architecture(model_config)
        require_flashinfer_token_paged()
        BlockDiffusionRunner.__init__(self, *args, _allow_flashinfer=True, **kwargs)
        self._paged_request_slots = {}
        self.fa4_graph_runner = None


class NemotronFlashInferSelfSpecRunner(
    NemotronSelfSpecPagedRunner, NemotronFlashInferDiffusionRunner
):
    """Variable-length acceptance and rollback with FlashInfer paging."""

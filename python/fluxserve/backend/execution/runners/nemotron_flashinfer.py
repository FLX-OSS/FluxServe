# Copyright (c) 2026 FLUX-OSS
# SPDX-License-Identifier: MIT

"""Nemotron paging on the standard FlashInfer prefill API.

Reuse the tested Nemotron block and speculation loops, including per-request
prefix/seed ownership. Only backend initialization and attention dispatch differ
from FA4; LLaDA's block-extend runner is never involved.

Decode CUDA graphs are supported through the same ``NemotronCudaGraphRunner``
FA4 uses, with a fixed FlashInfer plan and refreshed page buffers per replay.
Both diffusion and self-speculation use separate causal/non-causal captures.
Prefill graphs are not supported:
prefill shapes vary per request, as on FA4.
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
        if config.enable_prefill_cuda_graph:
            raise ValueError(
                "Nemotron FlashInfer supports --use-decode-cuda-graph; prefill "
                "graph is unsupported"
            )
        if int(config.page_size or config.block_length) % 16:
            raise ValueError("Nemotron paged page_size must be a multiple of 16")
        if config.enable_decode_cuda_graph:
            self._validate_decode_graph_config(config, kwargs, args)
        self._validate_architecture(model_config)
        require_flashinfer_token_paged()
        BlockDiffusionRunner.__init__(self, *args, _allow_flashinfer=True, **kwargs)
        self._paged_request_slots = {}
        # NemotronFA4DiffusionRunner.__init__ builds the graph runner from the
        # config after this returns; nothing to inherit from the FA4 base, whose
        # constructor this path deliberately skips.
        self.fa4_graph_runner = None

    @staticmethod
    def _validate_decode_graph_config(config, kwargs, args):
        """The FA4 gates that matter here; the FA4 constructor never ran."""
        server_args = kwargs.get("server_args", args[1] if len(args) > 1 else None)
        if any(int(getattr(server_args, name, 1)) != 1 for name in ("dp_size", "pp_size")):
            raise ValueError("Nemotron decode CUDA graph requires DP/PP=1")
        tp_size = int(getattr(server_args, "tp_size", 1))
        ep_size = int(getattr(server_args, "ep_size", 1))
        if tp_size < 1 or tp_size != ep_size:
            raise ValueError("Nemotron decode CUDA graph requires TP=EP >= 1")
        if config.decode_cuda_graph_mode != "padded":
            raise ValueError(
                "Nemotron decode CUDA graph requires --cuda-graph-decode-mode padded"
            )
        if int(config.page_size or config.block_length) != int(config.block_length):
            raise ValueError("Nemotron decode graph requires page_size == block_length")
        sizes = config.cuda_graph_capture_batch_sizes or config.supported_batch_sizes
        if max(sizes) < int(getattr(server_args, "max_num_seqs", 1)):
            raise ValueError("Nemotron decode graph buckets must cover max_num_seqs")


class NemotronFlashInferSelfSpecRunner(
    NemotronSelfSpecPagedRunner, NemotronFlashInferDiffusionRunner
):
    """Variable-length acceptance and rollback with FlashInfer paging."""

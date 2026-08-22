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

import logging
import time
from typing import Optional

import torch

from fluxserve.backend.configs.model_config import ModelConfig
from fluxserve.backend.distributed import (
    get_moe_expert_parallel_world_size,
    get_tp_group,
    set_custom_all_reduce,
)
from fluxserve.backend.execution.cuda_graph_runner import CudaGraphRunner
from fluxserve.backend.execution.decoders import load_decoder
from fluxserve.backend.execution.forward_batch_info import RunnerConfig
from fluxserve.backend.execution.forward_batch_info import ForwardBatch
from fluxserve.backend.layers.dp_attention import get_attention_tp_size
from fluxserve.backend.managers.kvcache import KVCache
from fluxserve.backend.model_loader import get_model
from fluxserve.backend.utils.runtime_utils import (
    get_available_gpu_memory,
    require_nvidia_cuda,
)
from fluxserve.backend.utils.server_args import ServerArgs

logger = logging.getLogger(__name__)


def _is_flex_block_mask(attention_mask) -> bool:
    return attention_mask is not None and attention_mask.__class__.__name__ == "BlockMask"


class ModelRunner:
    def __init__(
        self,
        model_config: ModelConfig,
        server_args: ServerArgs,
        runner_config: RunnerConfig | None = None,
        device: str | torch.device = "cuda",
    ):
        self.model_config = model_config
        self.server_args = server_args
        self.runner_config = runner_config or RunnerConfig()
        require_nvidia_cuda(device)
        self.device, self.gpu_id = self._normalize_device(device)

        self.block_length = self.runner_config.block_length
        self.prefilling_limit = self.runner_config.prefilling_limit
        self.prefill_lengths = list(self.runner_config.prefill_lengths)
        self.cache_lengths = list(self.runner_config.cache_lengths)
        self.max_length = max(self.runner_config.max_length, max(self.cache_lengths))
        self.decoding_lengths = list(self.runner_config.decoding_lengths)
        self.supported_batch_sizes = list(self.runner_config.supported_batch_sizes)
        self.enable_flashinfer_attention_graph = bool(
            self.runner_config.enable_cuda_graph
            and self.runner_config.attention_backend == "flashinfer"
            and self.runner_config.kv_cache_layout == "paged"
            and self.runner_config.flashinfer_cache_mode == "paged"
            and self.runner_config.flashinfer_prefill_mode == "paged"
            and (
                (
                    get_attention_tp_size() == 1
                    and get_moe_expert_parallel_world_size() == 1
                )
                or (
                    get_attention_tp_size() == 4
                    and get_moe_expert_parallel_world_size() == 4
                )
            )
        )
        if (
            self.runner_config.attention_backend == "flashinfer"
            and self.runner_config.enable_cuda_graph
            and not self.enable_flashinfer_attention_graph
        ):
            logger.info(
                "Disabling CUDA graph because attention_backend='flashinfer' "
                "uses dynamic ragged decode metadata."
            )
            self.runner_config.enable_cuda_graph = False
        if (
            self.runner_config.kv_cache_layout == "paged"
            and self.runner_config.enable_cuda_graph
            and not self.enable_flashinfer_attention_graph
        ):
            logger.info(
                "Disabling CUDA graph because kv_cache_layout='paged' "
                "materializes dynamic dense KV views."
            )
            self.runner_config.enable_cuda_graph = False
        self.enable_cuda_graph = bool(
            self.runner_config.enable_cuda_graph
            and not self.enable_flashinfer_attention_graph
        )
        self.enable_compile = self.runner_config.enable_compile
        self.use_cross_block = self.runner_config.use_cross_block
        self.early_stop = self.runner_config.early_stop
        self.num_forwards = 0

        self.init_model()
        self.init_decoder()
        self.tp_group = get_tp_group()
        set_custom_all_reduce(True)

        self.warmup_run()
        self.init_device_graphs()

    def _normalize_device(self, device: str | torch.device):
        device_str = str(device)
        if device_str.startswith("cuda:"):
            return "cuda", int(device_str.split(":", 1)[1])
        if device_str == "cuda":
            return "cuda", torch.cuda.current_device()
        if device_str.isdigit():
            return f"cuda:{device_str}", int(device_str)
        raise RuntimeError(
            f"FluxServe backend supports NVIDIA CUDA only, got device {device_str!r}."
        )

    def init_model(self):
        quant_config = getattr(self.model_config, "quant_config", None)
        self.model = get_model(
            model_config=self.model_config,
            device=self.device,
            quant_config=quant_config,
        )

    def init_decoder(self):
        self.decoder = load_decoder(self.runner_config)

    def warmup_run(self):
        if not self.enable_cuda_graph:
            return
        x = torch.arange(
            self.block_length,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        self.forward_normal(x, use_cache=True)

    def init_device_graphs(self):
        self.graph_runner = None
        self.graph_mem_usage = 0
        if not self.enable_cuda_graph:
            return

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            "Capture %s graph begin. avail mem=%.2f GB",
            "cuda",
            before_mem,
        )
        self.graph_runner = CudaGraphRunner(self)

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        self.graph_mem_usage = before_mem - after_mem
        logger.info(
            "Capture %s graph end. Time elapsed: %.2f s. mem usage=%.2f GB. avail mem=%.2f GB.",
            "cuda",
            time.perf_counter() - tic,
            self.graph_mem_usage,
            after_mem,
        )

    def forward_normal(
        self,
        input_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[torch.Tensor] = None,
        past_key_values=None,
        replace_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
    ):
        backup_ca_comm = self.tp_group.ca_comm
        if (
            attention_mask is not None
            and not _is_flex_block_mask(attention_mask)
            and past_key_values is not None
        ):
            attention_mask_partial = torch.zeros(
                (
                    attention_mask.shape[0],
                    attention_mask.shape[1],
                    past_key_values.shape[4],
                ),
                dtype=torch.bool,
                device=attention_mask.device,
            )
            attention_mask_partial[:, :, : attention_mask.shape[2]] = attention_mask
            attention_mask = attention_mask_partial

        ret = self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
            past_key_values=past_key_values,
            replace_position=replace_position,
            use_cache=use_cache,
            attention_mask=attention_mask,
            forward_batch=forward_batch,
        )
        self.tp_group.ca_comm = backup_ca_comm
        return ret

    def forward(
        self,
        input_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[torch.Tensor] = None,
        past_key_values=None,
        replace_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        attention_mask: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
    ):
        if attention_mask is not None and not _is_flex_block_mask(attention_mask):
            attention_mask = attention_mask.bool()
        if isinstance(past_key_values, KVCache):
            past_key_values = past_key_values._data

        is_decode_phase = (
            input_ids is not None and use_cache is True and past_key_values is not None
        )
        length = input_ids.shape[1]
        cache_length = past_key_values.shape[4] if past_key_values is not None else 0

        can_run_graph = bool(
            self.graph_runner
            and self.graph_runner.can_run(
                input_ids,
                position_ids,
                past_key_values,
                is_decode_phase,
                length,
                cache_length,
            )
        )
        if can_run_graph and self.enable_cuda_graph and forward_batch is None:
            logger.debug("run cuda graph")
            return self.graph_runner.replay(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                is_decode_phase=is_decode_phase,
                length=length,
                attention_mask=attention_mask,
                cache_length=cache_length,
            )

        logger.debug("run normal")
        return self.forward_normal(
            input_ids,
            position_ids,
            inputs_embeds,
            pp_proxy_tensors,
            past_key_values,
            replace_position,
            use_cache,
            attention_mask,
            forward_batch,
        )

    @torch.no_grad()
    def prefill(self, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def decode(self, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

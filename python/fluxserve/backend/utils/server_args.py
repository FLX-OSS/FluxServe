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

from dataclasses import dataclass
from fluxserve.backend.configs.model_config import ModelConfig


@dataclass
class ServerArgs:
    model_name: str = ""
    model_config: ModelConfig = None
    device: str = "cuda"
    enable_dp_attention: bool = False
    trust_remote_code: bool = True
    tp_size: int = 1
    dp_size: int = 1
    ep_size: int = 1
    pp_size: int = 1
    moe_dense_tp_size: int | None = None
    moe_a2a_backend: str = "none"
    moe_runner_backend: str = "auto"
    deepep_mode: str = "auto"
    deepep_config: str = ""
    enable_two_batch_overlap: bool = False
    enable_single_batch_overlap: bool = False
    tbo_token_distribution_threshold: float = 0.48
    enable_cudagraph_gc: bool = False
    enable_dp_lm_head: bool = False
    enable_fp32_lm_head: bool = False
    enable_nan_detection: bool = False
    sampling_backend: str = "pytorch"
    enable_flashinfer_allreduce_fusion: bool = False
    debug_tensor_dump_output_folder: str | None = None
    multi_item_scoring_delimiter: int | None = None
    speculative_algorithm: str | None = None
    host: str = "0.0.0.0"
    port: int = 8000
    apply_template: bool = False
    max_num_seqs: int = 8
    max_scheduled_tokens: int = 512
    max_model_len: int = 2048
    stream_interval: int = 1
    enable_prefix_caching: bool = False
    scheduler_policy: str = "default"
    scheduler_page_size: int | None = None
    scheduler_num_device_pages: int = 0
    gpu_memory_utilization: float = 0.90
    gpu_memory_safety_reserve: float = 0.05


_GLOBAL_SERVER_ARGS = ServerArgs()


def set_global_server_args_for_scheduler(server_args: ServerArgs):
    global _GLOBAL_SERVER_ARGS
    _GLOBAL_SERVER_ARGS = server_args


def get_global_server_args() -> ServerArgs:
    return _GLOBAL_SERVER_ARGS

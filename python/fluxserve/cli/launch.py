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
    HTTP server startup and local worker execution.
"""

from __future__ import annotations

import asyncio
import logging
import os

import torch
from transformers import AutoConfig, AutoTokenizer

from fluxserve.backend.configs import register_configs
from fluxserve.backend.distributed.launch import (
    destroy_distributed,
    initialize_distributed,
    launch_local_workers,
    reject_external_distributed_launch,
    should_launch_local_workers,
)
from fluxserve.backend.engine import AsyncLLM
from fluxserve.backend.engine.distributed_executor import DistributedGenerationExecutor
from fluxserve.backend.engine.executor import BlockDiffusionExecutor
from fluxserve.backend.engine.scheduler_adapter import PagedSchedulerAdapter
from fluxserve.backend.entrypoints.http_server import run
from fluxserve.backend.execution.forward_batch_info import RunnerConfig
from fluxserve.backend.execution.runners import (
    BlockDiffusionRunner,
    DiffusionGemmaRunner,
    FA4DiffusionRunner,
    FlashInferDiffusionRunner,
)
from fluxserve.backend.layers.dp_attention import initialize_dp_attention
from fluxserve.backend.layers.moe.utils import initialize_moe_config
from fluxserve.backend.utils.runtime_utils import require_nvidia_cuda
from fluxserve.backend.utils.runtime_utils import profile_paged_kv_pages
from fluxserve.backend.utils.server_args import ServerArgs

from fluxserve.cli.utils import (
    _check_block_routing_alignment,
    _reject_unsupported_quantization,
    default_cuda_graph_capture_batch_sizes,
    normalize_attention_backend_args,
    normalize_diffusion_gemma_serve_args,
    set_process_title,
)

logger = logging.getLogger(__name__)


def launch(args) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    reject_external_distributed_launch()
    normalize_attention_backend_args(args)
    if should_launch_local_workers(args.tp_size):
        if args.process_name:
            set_process_title(f"{args.process_name}:supervisor")
        require_nvidia_cuda(args.device)
        launch_local_workers(_launch_worker, args)
        return
    _launch_worker(args)


def _launch_worker(args, *, init_method: str = "env://") -> None:
    if args.process_name:
        set_process_title(args.process_name)

    require_nvidia_cuda(args.device)
    register_configs()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    model_config = AutoConfig.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    is_diffusion_gemma = normalize_diffusion_gemma_serve_args(args, model_config)
    apply_template = bool(args.apply_template or is_diffusion_gemma)
    if is_diffusion_gemma and not args.apply_template:
        logger.info("Diffusion-Gemma chat requests use the checkpoint chat template.")
    if is_diffusion_gemma and args.scheduler_policy == "paged":
        raise RuntimeError(
            "Diffusion-Gemma FlashInfer does not support scheduler_policy='paged' yet."
        )
    _reject_unsupported_quantization(model_config)
    model_config.quant_config = None
    _check_block_routing_alignment(model_config, args.block_length)
    valid_paged_backend = (
        args.attention_backend == "fa4" and args.kv_cache_layout == "paged"
    ) or (
        args.attention_backend == "flashinfer"
        and args.kv_cache_layout == "paged"
        and args.flashinfer_cache_mode == "paged"
        and args.flashinfer_prefill_mode == "paged"
    )
    if args.scheduler_policy == "paged" and not valid_paged_backend:
        raise RuntimeError(
            "scheduler_policy='paged' requires either FA4 with paged KV, or "
            "FlashInfer paged prefill and paged KV."
        )
    server_args = ServerArgs(
        model_name=args.model_name,
        model_config=model_config,
        device=args.device,
        host=args.host,
        port=args.port,
        apply_template=apply_template,
        max_num_seqs=args.max_num_seqs,
        max_scheduled_tokens=args.max_scheduled_tokens,
        max_model_len=args.max_model_len,
        generation_block_size=(
            int(
                args.canvas_length
                or getattr(model_config, "canvas_length", None)
                or args.block_length
            )
            if is_diffusion_gemma
            else 1
        ),
        scheduler_policy=args.scheduler_policy,
        scheduler_page_size=args.page_size or args.block_length,
        scheduler_num_device_pages=args.scheduler_num_device_pages,
        gpu_memory_utilization=args.gpu_memory_utilization,
        gpu_memory_safety_reserve=args.gpu_memory_safety_reserve,
        trust_remote_code=args.trust_remote_code,
        tp_size=args.tp_size,
        dp_size=args.dp_size,
        ep_size=args.ep_size,
        pp_size=args.pp_size,
        enable_dp_attention=args.enable_dp_attention,
    )
    context = initialize_distributed(
        server_args,
        backend=args.distributed_backend,
        init_method=init_method,
    )
    if args.process_name and context.is_distributed:
        set_process_title(f"{args.process_name}:rank{context.rank}")
    if context.is_distributed:
        server_args.device = f"cuda:{context.local_rank}"
        args.device = server_args.device
    else:
        device_index = (
            int(args.device)
            if str(args.device).isdigit()
            else torch.device(args.device).index or 0
        )
        torch.cuda.set_device(device_index)

    try:
        initialize_dp_attention(server_args=server_args, model_config=model_config)
        initialize_moe_config(server_args)

        if args.scheduler_policy == "paged":
            page_size = int(args.page_size or args.block_length)
            if page_size != int(args.block_length):
                raise ValueError(
                    "paged block diffusion requires page_size == block_length"
                )
            if int(args.max_model_len) % int(args.block_length) != 0:
                raise ValueError(
                    "paged block diffusion requires max_model_len to be "
                    "divisible by block_length"
                )
            if int(args.max_scheduled_tokens) % int(args.block_length) != 0:
                raise ValueError(
                    "paged block diffusion requires max_scheduled_tokens to be "
                    "divisible by block_length"
                )
            num_device_pages = int(args.scheduler_num_device_pages)
            server_args.scheduler_num_device_pages = num_device_pages

        graph_capture_sizes = tuple(
            int(size)
            for size in args.cuda_graph_capture_sizes
            if 0 < int(size) <= int(args.max_model_len)
            and int(size) % int(args.block_length) == 0
        )
        if args.use_cuda_graph and not graph_capture_sizes:
            raise ValueError(
                "CUDA graph capture sizes must include at least one block-aligned "
                "length no greater than max_model_len"
            )
        graph_capture_batch_sizes = tuple(
            args.cuda_graph_capture_bs
            or default_cuda_graph_capture_batch_sizes(args.max_num_seqs)
        )
        if any(size > int(args.max_num_seqs) for size in graph_capture_batch_sizes):
            raise ValueError(
                "CUDA graph capture batch sizes cannot exceed max_num_seqs="
                f"{args.max_num_seqs}: got {graph_capture_batch_sizes}"
            )

        runner_config = RunnerConfig(
            gen_length=args.max_new_tokens,
            block_length=args.block_length,
            prefilling_limit=args.prefilling_limit,
            mini_batch_size=args.mini_batch_size,
            max_length=args.max_model_len,
            supported_batch_sizes=tuple(
                2**i for i in range(max(1, args.max_num_seqs).bit_length())
            ),
            enable_cuda_graph=args.use_cuda_graph,
            enable_prefill_cuda_graph=args.use_prefill_cuda_graph,
            enable_decode_cuda_graph=args.use_decode_cuda_graph,
            decode_cuda_graph_mode=args.cuda_graph_decode_mode,
            cuda_graph_capture_batch_sizes=graph_capture_batch_sizes,
            cuda_graph_capture_sizes=graph_capture_sizes,
            attention_backend=args.attention_backend,
            flashinfer_decode_batch_mode=args.flashinfer_decode_batch_mode,
            flashinfer_prefill_mode=args.flashinfer_prefill_mode,
            flashinfer_cache_mode=args.flashinfer_cache_mode,
            kv_cache_layout=args.kv_cache_layout,
            page_size=args.page_size,
            canvas_length=args.canvas_length,
            max_denoising_steps=args.max_denoising_steps,
            parallel_decoding=args.parallel_decoding,
            threshold=args.threshold,
            low_threshold=args.low_threshold,
            editing_threshold=args.editing_threshold,
            max_post_steps=args.max_post_steps,
            steps=args.steps,
            max_steps_per_block=args.max_steps_per_block,
            delete_token_id=int(getattr(model_config, "delete_token_id", 156930)),
            split_token_id=int(getattr(model_config, "split_token_id", 156931)),
        )
        if is_diffusion_gemma:
            runner_cls = DiffusionGemmaRunner
        else:
            if args.attention_backend == "flashinfer":
                runner_cls = FlashInferDiffusionRunner
            elif args.attention_backend == "fa4":
                runner_cls = FA4DiffusionRunner
            else:
                runner_cls = BlockDiffusionRunner
        runner = runner_cls(
            model_config=model_config,
            server_args=server_args,
            runner_config=runner_config,
            device=args.device,
        )
        if args.scheduler_policy == "paged" and int(server_args.scheduler_num_device_pages) <= 0:
            server_args.scheduler_num_device_pages = profile_paged_kv_pages(
                runner=runner, page_size=int(args.page_size or args.block_length),
                utilization=server_args.gpu_memory_utilization,
                safety_reserve=server_args.gpu_memory_safety_reserve)
        base_executor = BlockDiffusionExecutor(runner=runner, tokenizer=tokenizer)
        executor = DistributedGenerationExecutor(base_executor, context)
        if context.is_rank0:
            scheduler = None
            if args.scheduler_policy == "paged":
                page_size = int(args.page_size or args.block_length)
                num_device_pages = int(server_args.scheduler_num_device_pages)
                scheduler = PagedSchedulerAdapter(
                    max_batch_size=args.max_num_seqs,
                    max_scheduled_tokens=args.max_scheduled_tokens,
                    page_size=page_size,
                    num_device_pages=num_device_pages,
                    max_model_len=args.max_model_len,
                )
            engine = AsyncLLM(
                server_args=server_args,
                executor=executor,
                tokenizer=tokenizer,
                scheduler=scheduler,
            )
            try:
                run(
                    engine,
                    host=args.host,
                    port=args.port,
                    runner_config=runner_config,
                )
            finally:
                asyncio.run(executor.shutdown_workers())
        else:
            asyncio.run(executor.run_worker_loop())
    finally:
        destroy_distributed()

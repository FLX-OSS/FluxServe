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


from __future__ import annotations

import argparse
import asyncio
import logging

import torch
from transformers import AutoConfig, AutoTokenizer

from fluxserve.bench import add_bench_subparser
from fluxserve.bench_offline import (
    add_bench_offline_subparser,
    bench_offline,
    normalize_attention_backend_args,
)
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
    FlashInferDiffusionRunner,
)
from fluxserve.backend.layers.dp_attention import initialize_dp_attention
from fluxserve.backend.layers.moe.utils import initialize_moe_config
from fluxserve.backend.utils.runtime_utils import require_nvidia_cuda
from fluxserve.backend.utils.runtime_utils import profile_paged_kv_pages
from fluxserve.backend.utils.server_args import ServerArgs

logger = logging.getLogger(__name__)


def set_process_title(title: str) -> None:
    try:
        import setproctitle
    except ImportError:
        return
    setproctitle.setproctitle(title)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fluxserve")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help='Launch the FluxServe server')
    serve.add_argument("--model", "--model-name", dest="model_name", required=True)
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--apply-template",
        action="store_true",
        help=(
            "Render chat requests with tokenizer.apply_chat_template(). "
            "By default FluxServe uses its LLaDA-compatible prompt renderer."
        ),
    )
    serve.add_argument("--device", default="cuda", help='GPU device type')
    serve.add_argument("--max-num-seqs", type=int, default=8)
    serve.add_argument("--max-scheduled-tokens", type=int, default=512)
    serve.add_argument("--max-model-len", type=int, default=2048)
    serve.add_argument("--max-new-tokens", type=int, default=128)
    serve.add_argument(
        "--scheduler-policy",
        choices=("default", "paged"),
        default="default",
    )
    serve.add_argument("--scheduler-num-device-pages", type=int, default=0)
    serve.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    serve.add_argument("--gpu-memory-safety-reserve", type=float, default=0.05)
    serve.add_argument("--block-length", type=int, default=64)
    serve.add_argument("--prefilling-limit", type=int, default=128)
    serve.add_argument("--mini-batch-size", type=int, default=4)
    serve.add_argument(
        "--attention-backend",
        choices=("sdpa", "flex", "flashinfer"),
        default="flashinfer",
    )
    serve.add_argument(
        "--flashinfer-decode-batch-mode",
        choices=("default", "max_batch"),
        default="max_batch",
    )
    serve.add_argument(
        "--flashinfer-prefill-mode",
        choices=("dense", "ragged", "paged"),
        default="paged",
    )
    serve.add_argument(
        "--flashinfer-cache-mode",
        choices=("dense", "paged"),
        default="paged",
    )
    serve.add_argument(
        "--kv-cache-layout",
        choices=("dense", "paged"),
        default="paged",
    )
    serve.add_argument("--page-size", type=int, default=None)
    serve.add_argument("--parallel-decoding", default="threshold")
    serve.add_argument("--threshold", type=float, default=0.9)
    serve.add_argument("--low-threshold", type=float, default=0.3)
    serve.add_argument("--tp-size", type=int, default=1)
    serve.add_argument("--dp-size", type=int, default=1)
    serve.add_argument("--ep-size", type=int, default=1)
    serve.add_argument("--pp-size", type=int, default=1)
    serve.add_argument("--enable-dp-attention", action="store_true", default=False)
    serve.add_argument("--distributed-backend", default="nccl")
    serve.add_argument("--use-cuda-graph", action="store_true")
    serve.add_argument("--use-prefill-cuda-graph", action="store_true")
    serve.add_argument("--use-decode-cuda-graph", action="store_true")
    serve.add_argument("--cuda-graph-decode-mode", choices=("decomposed", "padded"), default="decomposed")
    serve.add_argument("--cuda-graph-capture-bs", "--cuda_graph_capture_bs", type=int, nargs="+", default=None, metavar="N")
    serve.add_argument(
        "--cuda-graph-capture-sizes",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024],
        metavar="N",
        help="Prefill sequence-length buckets captured by CUDA graphs.",
    )
    serve.add_argument("--trust-remote-code", action="store_true", default=True)
    serve.add_argument(
        "--process-name",
        default="fluxserve",
        help="Process title shown by ps/top for online serving.",
    )
    sub.add_parser("env", help="Print environment and dependency information.")
    add_bench_subparser(sub)
    add_bench_offline_subparser(sub)
    return parser


def _reject_unsupported_quantization(model_config) -> None:
    quant_config = getattr(model_config, "quantization_config", None)
    if not isinstance(quant_config, dict):
        return

    nested = quant_config.get("quantization")
    configs = (quant_config, nested) if isinstance(nested, dict) else (quant_config,)
    for config in configs:
        quant_method = str(config.get("quant_method", "")).lower()
        quant_algo = str(config.get("quant_algo", "")).upper()
        if "fp8" in quant_method or "FP8" in quant_algo or "FP4" in quant_algo:
            raise ValueError(
                "FluxServe does not currently support FP8 or FP4 quantized checkpoints. "
                "Use an unquantized BF16/FP16 checkpoint."
            )


def serve(args) -> None:
    reject_external_distributed_launch()
    normalize_attention_backend_args(args)
    if should_launch_local_workers(args.tp_size):
        if args.process_name:
            set_process_title(f"{args.process_name}:supervisor")
        require_nvidia_cuda(args.device)
        launch_local_workers(_serve_worker, args)
        return
    _serve_worker(args)


def _serve_worker(args, *, init_method: str = "env://") -> None:
    if args.process_name:
        set_process_title(args.process_name)

    require_nvidia_cuda(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    model_config = AutoConfig.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    _reject_unsupported_quantization(model_config)
    model_config.quant_config = None
    if args.scheduler_policy == "paged" and (
        args.attention_backend != "flashinfer"
        or args.kv_cache_layout != "paged"
        or args.flashinfer_cache_mode != "paged"
        or args.flashinfer_prefill_mode != "paged"
    ):
        raise RuntimeError(
            "scheduler_policy='paged' requires attention_backend='flashinfer', "
            "kv_cache_layout='paged', flashinfer_cache_mode='paged', and "
            "flashinfer_prefill_mode='paged'."
        )
    server_args = ServerArgs(
        model_name=args.model_name,
        model_config=model_config,
        device=args.device,
        host=args.host,
        port=args.port,
        apply_template=args.apply_template,
        max_num_seqs=args.max_num_seqs,
        max_scheduled_tokens=args.max_scheduled_tokens,
        max_model_len=args.max_model_len,
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
            cuda_graph_capture_batch_sizes=args.cuda_graph_capture_bs,
            cuda_graph_capture_sizes=graph_capture_sizes,
            attention_backend=args.attention_backend,
            flashinfer_decode_batch_mode=args.flashinfer_decode_batch_mode,
            flashinfer_prefill_mode=args.flashinfer_prefill_mode,
            flashinfer_cache_mode=args.flashinfer_cache_mode,
            kv_cache_layout=args.kv_cache_layout,
            page_size=args.page_size,
            parallel_decoding=args.parallel_decoding,
            threshold=args.threshold,
            low_threshold=args.low_threshold,
        )
        runner_cls = (
            FlashInferDiffusionRunner
            if args.attention_backend == "flashinfer"
            else BlockDiffusionRunner
        )
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
                run(engine, host=args.host, port=args.port)
            finally:
                asyncio.run(executor.shutdown_workers())
        else:
            asyncio.run(executor.run_worker_loop())
    finally:
        destroy_distributed()


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "serve":
        serve(args)
    elif args.command == "env":
        from fluxserve.env import main as env_main

        env_main()
    elif args.command == "bench":
        args.dispatch_function(args)
    elif args.command == "bench_offline":
        bench_offline(args)


if __name__ == "__main__":
    main()

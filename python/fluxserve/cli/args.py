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
    Argument registration for the FluxServe commands.
"""

import argparse
import json

DEFAULT_TIMEOUT_SEC = 60 * 60
SUPPORTED_METRICS = ("E2E", "QUEUE", "EXECUTION", "HTTP_OVERHEAD")


class StoreExplicit(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, f"{self.dest}_explicit", True)


def _parse_metrics(value: str) -> tuple[str, ...]:
    requested = {item.strip().upper() for item in value.split(",") if item.strip()}
    if not requested:
        raise argparse.ArgumentTypeError("--metrics must contain at least E2E.")
    unknown = requested.difference(SUPPORTED_METRICS)
    if unknown:
        supported = ", ".join(SUPPORTED_METRICS)
        raise argparse.ArgumentTypeError(
            f"Unknown metric(s): {', '.join(sorted(unknown))}. Supported metrics: {supported}."
        )
    if "E2E" not in requested:
        raise argparse.ArgumentTypeError("--metrics must include E2E.")
    return tuple(metric for metric in SUPPORTED_METRICS if metric in requested)


def add_launch_subparser(subparsers: argparse._SubParsersAction) -> None:
    launch = subparsers.add_parser("launch", help='Launch the FluxServe server')
    launch.add_argument("--model", "--model-name", dest="model_name", required=True)
    launch.add_argument("--host", default="0.0.0.0")
    launch.add_argument("--port", type=int, default=8000)
    launch.add_argument(
        "--apply-template",
        action="store_true",
        help=(
            "Render chat requests with tokenizer.apply_chat_template(). "
            "By default FluxServe uses its LLaDA-compatible prompt renderer."
        ),
    )
    launch.add_argument("--device", default="cuda", help='GPU device type')
    launch.add_argument("--max-num-seqs", type=int, default=16)
    launch.add_argument("--max-scheduled-tokens", type=int, default=2048)
    launch.add_argument("--max-model-len", type=int, default=65536)
    launch.add_argument("--max-new-tokens", type=int, default=128)
    launch.add_argument(
        "--scheduler-policy",
        choices=("default", "paged"),
        default="default",
    )
    launch.add_argument("--scheduler-num-device-pages", type=int, default=0)
    launch.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    launch.add_argument("--gpu-memory-safety-reserve", type=float, default=0.05)
    launch.add_argument("--block-length", type=int, default=64)
    launch.add_argument(
        "--canvas-length",
        "--canvas_length",
        dest="canvas_length",
        type=int,
        default=None,
        help=(
            "Override the Diffusion-Gemma denoising canvas length. "
            "Defaults to the checkpoint configuration."
        ),
    )
    launch.add_argument(
        "--max-denoising-steps",
        type=int,
        default=None,
        help="Override checkpoint denoising steps (primarily for smoke tests).",
    )
    launch.add_argument("--prefilling-limit", type=int, default=128)
    launch.add_argument("--mini-batch-size", type=int, default=4)
    launch.add_argument(
        "--attention-backend",
        choices=("sdpa", "flex", "flashinfer", "fa4"),
        default="fa4",
        action=StoreExplicit,
    )
    launch.set_defaults(attention_backend_explicit=False)
    launch.add_argument(
        "--flashinfer-decode-batch-mode",
        choices=("default", "max_batch"),
        default="max_batch",
    )
    launch.add_argument(
        "--flashinfer-prefill-mode",
        choices=("dense", "ragged", "paged"),
        default="paged",
    )
    launch.add_argument(
        "--flashinfer-cache-mode",
        choices=("dense", "paged"),
        default="paged",
    )
    launch.add_argument(
        "--kv-cache-layout",
        choices=("dense", "paged"),
        default="paged",
    )
    launch.add_argument("--page-size", type=int, default=None)
    launch.add_argument("--parallel-decoding", default="threshold")
    launch.add_argument("--threshold", type=float, default=0.95)
    launch.add_argument("--low-threshold", type=float, default=0.3)
    launch.add_argument(
        "--editing-threshold",
        type=float,
        default=0.5,
        help="LLaDA2.1 T2T editing threshold (joint_threshold decoding). "
        "Official presets: 0.5 (Quality), 0.0 (Speed).",
    )
    launch.add_argument(
        "--max-post-steps",
        type=int,
        default=16,
        help="Max post-mask editing iterations per block (joint_threshold).",
    )
    launch.add_argument(
        "--steps",
        type=int,
        default=0,
        help="LLaDA2.2 M2T transfer-schedule steps (levenshtein_joint); "
        "0 means block_length.",
    )
    launch.add_argument(
        "--max-steps-per-block",
        type=int,
        default=1000,
        help="Hard per-block iteration cap (levenshtein_joint).",
    )
    launch.add_argument("--tp-size", type=int, default=1)
    launch.add_argument("--dp-size", type=int, default=1)
    launch.add_argument("--ep-size", type=int, default=1)
    launch.add_argument("--pp-size", type=int, default=1)
    launch.add_argument("--enable-dp-attention", action="store_true", default=False)
    launch.add_argument("--distributed-backend", default="nccl")
    launch.add_argument("--use-cuda-graph", action="store_true")
    launch.add_argument("--use-prefill-cuda-graph", action="store_true")
    launch.add_argument("--use-decode-cuda-graph", action="store_true")
    launch.add_argument("--cuda-graph-decode-mode", choices=("decomposed", "padded"), default="decomposed")
    launch.add_argument(
        "--cuda-graph-capture-bs",
        "--cuda_graph_capture_bs",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "Decode CUDA graph batch sizes. Defaults to batch size 1 and "
            "every even size up to --max-num-seqs."
        ),
    )
    launch.add_argument(
        "--cuda-graph-capture-sizes",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024],
        metavar="N",
        help="Prefill sequence-length buckets captured by CUDA graphs.",
    )
    launch.add_argument("--trust-remote-code", action="store_true", default=True)
    launch.add_argument(
        "--process-name",
        default="fluxserve",
        help="Process title shown by ps/top for online serving.",
    )


def add_bench_subparser(subparsers: argparse._SubParsersAction) -> None:
    bench = subparsers.add_parser("bench", help="Online serving benchmark.")
    bench.add_argument("--model", required=True)
    bench.add_argument("--dataset", required=True)
    bench.add_argument("--tokenizer", default=None)
    bench.add_argument("--base-url", default=None)
    bench.add_argument("--host", default="127.0.0.1")
    bench.add_argument("--port", type=int, default=8000)
    bench.add_argument("--endpoint", default="/v1/chat/completions")
    bench.add_argument("--num-prompts", type=int, default=None)
    bench.add_argument("--dataset-output-len", type=int, default=None)
    bench.add_argument("--request-rate", type=float, default=float("inf"))
    bench.add_argument("--burstiness", type=float, default=1.0)
    bench.add_argument("--max-concurrency", type=int, default=None)
    bench.add_argument("--ready-check-timeout-sec", type=int, default=600)
    bench.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SEC)
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument("--request-id-prefix", default="bench-")
    bench.add_argument("--trust-remote-code", action="store_true", default=True)
    bench.add_argument("--extra-body", type=json.loads, default={})
    bench.add_argument("--ignore-eos", action="store_true")
    bench.add_argument("--metric-percentiles", default="50,90,95,99")
    bench.add_argument(
        "--metrics",
        type=_parse_metrics,
        default=("E2E",),
        help="Comma-separated output metrics; E2E is required (default: E2E).",
    )
    bench.add_argument("--save-result", action="store_true")
    bench.add_argument("--save-detailed", action="store_true")
    bench.add_argument("--result-dir", default="bench_results")
    bench.add_argument("--output-file", default=None)


def add_bench_offline_subparser(subparsers) -> None:
    parser = subparsers.add_parser("bench_offline", help="Offline batched benchmark.")
    parser.add_argument("--model", "--model-name", "--model_name", dest="model_name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--max-model-length",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Paged KV-cache capacity in tokens per sequence, including prompt "
            "and generation. Must cover block/canvas rounding and warmup. "
            "Defaults to the capacity required by the dataset."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=1)
    parser.add_argument("--mini-batch-size", "--mini_batch_size", dest="mini_batch_size", type=int, default=4)
    parser.add_argument("--use-naive-batching", "--use_naive_batching", dest="use_naive_batching", action="store_true")
    parser.add_argument("--tp-size", "--tp_size", dest="tp_size", type=int, default=1)
    parser.add_argument("--dp-size", "--dp_size", dest="dp_size", type=int, default=1)
    parser.add_argument("--ep-size", "--ep_size", dest="ep_size", type=int, default=1)
    parser.add_argument("--pp-size", "--pp_size", dest="pp_size", type=int, default=1)
    parser.add_argument("--distributed-backend", "--distributed_backend", dest="distributed_backend", default="nccl")
    parser.add_argument("--use-cuda-graph", "--use_cuda_graph", dest="use_cuda_graph", action="store_true")
    parser.add_argument("--use-prefill-cuda-graph", "--use_prefill_cuda_graph", dest="use_prefill_cuda_graph", action="store_true")
    parser.add_argument("--use-decode-cuda-graph", "--use_decode_cuda_graph", dest="use_decode_cuda_graph", action="store_true")
    parser.add_argument(
        "--cuda-graph-capture-sizes",
        "--cuda_graph_capture_sizes",
        dest="cuda_graph_capture_sizes",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024],
        metavar="N",
        help="Prefill sequence-length buckets captured by CUDA graphs.",
    )
    parser.add_argument("--prefilling-limit", "--prefilling_limit", dest="prefilling_limit", type=int, default=128)
    parser.set_defaults(attention_backend_explicit=False)
    parser.add_argument("--attention-backend", "--attention_backend", dest="attention_backend", choices=("sdpa", "flex", "flashinfer", "fa4"), default="flashinfer", action=StoreExplicit)
    parser.add_argument("--flashinfer-decode-batch-mode", "--flashinfer_decode_batch_mode", dest="flashinfer_decode_batch_mode", choices=("default", "max_batch"), default="max_batch")
    parser.add_argument("--flashinfer-prefill-mode", "--flashinfer_prefill_mode", dest="flashinfer_prefill_mode", choices=("dense", "ragged", "paged"), default="paged")
    parser.add_argument("--flashinfer-cache-mode", "--flashinfer_cache_mode", dest="flashinfer_cache_mode", choices=("dense", "paged"), default="paged")
    parser.add_argument("--kv-cache-layout", "--kv_cache_layout", dest="kv_cache_layout", choices=("dense", "paged"), default="paged")
    parser.add_argument("--page-size", "--page_size", dest="page_size", type=int)
    parser.add_argument("--gen-len", "--gen_len", dest="gen_len", type=int, default=1024)
    parser.add_argument("--block-length", "--block_length", dest="block_length", type=int, default=64)
    parser.add_argument(
        "--canvas-length", "--canvas_length", dest="canvas_length", type=int
    )
    parser.add_argument(
        "--max-denoising-steps",
        "--max_denoising_steps",
        dest="max_denoising_steps",
        type=int,
    )
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--low-threshold", "--low_threshold", dest="low_threshold", type=float, default=0.3)
    parser.add_argument("--parallel-decoding", "--parallel_decoding", dest="parallel_decoding", default="threshold")
    parser.add_argument(
        "--editing-threshold",
        "--editing_threshold",
        dest="editing_threshold",
        type=float,
        default=0.5,
        help="LLaDA2.1 T2T editing threshold (joint_threshold decoding).",
    )
    parser.add_argument(
        "--max-post-steps",
        "--max_post_steps",
        dest="max_post_steps",
        type=int,
        default=16,
        help="Max post-mask editing iterations per block (joint_threshold).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="LLaDA2.2 M2T transfer-schedule steps (levenshtein_joint); "
        "0 means block_length.",
    )
    parser.add_argument(
        "--max-steps-per-block",
        "--max_steps_per_block",
        dest="max_steps_per_block",
        type=int,
        default=1000,
        help="Hard per-block iteration cap (levenshtein_joint).",
    )
    parser.add_argument("--use-credit", "--use_credit", dest="use_credit", action="store_true")
    parser.add_argument("--dataset-format", "--dataset_format", dest="dataset_format", choices=("auto", "legacy", "openai"), default="openai")
    parser.add_argument(
        "--disable-sorting",
        "--disable_sorting",
        dest="disable_sorting",
        action="store_true",
        default=True,
    )
    parser.add_argument("--exp-name", "--exp_name", dest="exp_name", default="exp")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="runs/detailed_results")
    parser.add_argument("--log-file", "--log_file", dest="log_file", default="run.log")
    parser.add_argument("--trust-remote-code", "--trust_remote_code", dest="trust_remote_code", action="store_true", default=True)
    parser.add_argument("--process-name", "--process_name", dest="process_name", default="fluxserve")

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


import json
import os
import string
import time
from pathlib import Path

import numpy as np
import torch
import tqdm
from transformers import AutoConfig, AutoTokenizer

from fluxserve.backend.distributed.launch import (
    destroy_distributed,
    initialize_distributed,
    launch_local_workers,
    reject_external_distributed_launch,
    should_launch_local_workers,
)
from fluxserve.backend.execution.forward_batch_info import (
    GenerationBatchInfo,
    RunnerConfig,
)
from fluxserve.backend.execution.runners import (
    BlockDiffusionRunner,
    FlashInferDiffusionRunner,
)
from fluxserve.backend.layers.dp_attention import initialize_dp_attention
from fluxserve.backend.layers.moe import initialize_moe_config
from fluxserve.backend.layers.quantization import QUANTIZATION_METHODS
from fluxserve.backend.metrics import record_batch_performance_metrics
from fluxserve.backend.utils.server_args import ServerArgs
from fluxserve.backend.utils.runtime_utils import require_nvidia_cuda

os.environ["TOKENIZERS_PARALLELISM"] = "false"

BUCKET_SIZE = 32


def normalize_attention_backend_args(args) -> None:
    if args.attention_backend == "flashinfer":
        return
    args.flashinfer_prefill_mode = "dense"
    args.flashinfer_cache_mode = "dense"
    args.kv_cache_layout = "dense"
    args.page_size = None


class BenchmarkLogger:
    def __init__(self, log_file: str | None = None, rank: int = 0):
        self.log_file = log_file
        self.rank = rank
        if self.is_master and self.log_file:
            os.makedirs(os.path.dirname(self.log_file) or ".", exist_ok=True)

    @property
    def is_master(self) -> bool:
        return self.rank == 0

    def info(self, message: str) -> None:
        if not self.is_master:
            return
        timestamped_message = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(timestamped_message)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(timestamped_message + "\n")


def bucket_length(length: int) -> int:
    return BUCKET_SIZE * (length // BUCKET_SIZE)


def render_openai_messages(messages):
    rendered = []
    for message in messages:
        role = message.get("role", "").upper()
        content = message.get("content", "")
        if role == "SYSTEM":
            rendered.append(f"<role>SYSTEM</role>{content}<|role_end|>")
        elif role == "ASSISTANT":
            rendered.append(f"<role>ASSISTANT</role>{content}<|role_end|>")
        else:
            rendered.append(f"<role>HUMAN</role>{content}<|role_end|>")
    rendered.append("<role>ASSISTANT</role>")
    return "".join(rendered)


def load_openai_style_inputs(dataset, tokenizer):
    prompts = []
    questions = []
    ids = []
    all_input_ids = []
    with open(dataset, "r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list):
                raise ValueError(f"OpenAI-style JSONL row {idx} is missing messages")
            metadata = row.get("metadata") or {}
            ids.append(metadata.get("task_id", idx))
            question = "\n".join(
                message.get("content", "")
                for message in messages
                if message.get("role") == "user"
            )
            questions.append(question)
            prompt = render_openai_messages(messages)
            prompts.append(prompt)
            input_ids = torch.tensor(tokenizer(prompt)["input_ids"]).unsqueeze(0)
            all_input_ids.append(input_ids)
    return all_input_ids, prompts, questions, ids


def load_legacy_inputs(dataset, tokenizer):
    with open(dataset, "r") as f:
        data = json.load(f)
    details_data = data["judge_details"] if "judge_details" in data else data["details"]

    prompts = []
    questions = []
    ids = []
    all_input_ids = []
    for idx, judge_detail in enumerate(details_data):
        ids.append(idx)
        question = judge_detail["prompt"]
        questions.append(question)
        prompt = (
            "<role>SYSTEM</role>detailed thinking off<|role_end|>"
            f"<role>HUMAN</role>{question}<|role_end|><role>ASSISTANT</role>"
        )
        prompts.append(prompt)
        input_ids = torch.tensor(tokenizer(prompt)["input_ids"]).unsqueeze(0)
        all_input_ids.append(input_ids)
    return all_input_ids, prompts, questions, ids


def detect_dataset_format(dataset):
    with open(dataset, "r") as f:
        first_nonempty = next((line.strip() for line in f if line.strip()), "")
    if not first_nonempty:
        raise ValueError(f"Dataset is empty: {dataset}")
    row = json.loads(first_nonempty)
    if isinstance(row, dict) and "messages" in row:
        return "openai"
    return "legacy"


def load_inputs(dataset, tokenizer, dataset_format="auto"):
    if dataset_format == "auto":
        dataset_format = detect_dataset_format(dataset)
    if dataset_format == "openai":
        return load_openai_style_inputs(dataset, tokenizer)
    if dataset_format == "legacy":
        return load_legacy_inputs(dataset, tokenizer)
    raise ValueError(f"Unsupported dataset format: {dataset_format}")


def calc_padded_gen_lens(args, all_input_ids):
    return [
        bucket_length(input_ids.shape[1] + args.gen_len) - input_ids.shape[1]
        for input_ids in all_input_ids
    ]


def cut_eos(data, eos_id=156892):
    eos_indices = (data[0] == eos_id).nonzero(as_tuple=True)[0]
    if eos_indices.numel() > 0:
        return data[:, : eos_indices[0].item()]
    return data


def summarize_outputs(answers, token_numbers):
    punctuation = set(string.punctuation)
    rows = []
    whitespace_only = 0
    punctuation_only = 0
    for idx, answer in enumerate(answers):
        stripped = answer.strip()
        is_whitespace_only = len(stripped) == 0
        is_punctuation_only = bool(stripped) and all(ch in punctuation for ch in stripped)
        whitespace_only += int(is_whitespace_only)
        punctuation_only += int(is_punctuation_only)
        rows.append(
            {
                "id": idx,
                "chars": len(answer),
                "stripped_chars": len(stripped),
                "generated_length": int(token_numbers[idx]),
                "whitespace_only": is_whitespace_only,
                "punctuation_only": is_punctuation_only,
                "preview": stripped[:160],
            }
        )
    return {
        "num_answers": len(answers),
        "whitespace_only": whitespace_only,
        "punctuation_only": punctuation_only,
        "bad_answer_ids": [
            row["id"]
            for row in rows
            if row["whitespace_only"] or row["punctuation_only"]
        ],
        "rows": rows,
    }


def print_output_summary(summary, logger):
    num_answers = summary["num_answers"]
    bad_count = len(summary["bad_answer_ids"])
    logger.info(
        "[Output check] "
        f"answers={num_answers}, bad={bad_count}, "
        f"whitespace_only={summary['whitespace_only']}, "
        f"punctuation_only={summary['punctuation_only']}"
    )
    for row in summary["rows"]:
        if row["whitespace_only"] or row["punctuation_only"]:
            logger.info(
                "[Output check] "
                f"id={row['id']} generated_length={row['generated_length']} "
                f"chars={row['chars']} stripped_chars={row['stripped_chars']} "
                f"preview={row['preview']!r}"
            )


def build_server_args(args, model_config):
    return ServerArgs(
        model_name=args.model_name,
        model_config=model_config,
        quantization=args.quantization,
        modelopt_quant="fp8" if args.quantization == "modelopt_fp8" else "",
        enable_dp_attention=args.dp_size > 1,
        trust_remote_code=True,
        tp_size=args.parallel_world_size,
        dp_size=args.dp_size,
        ep_size=args.ep_size,
        pp_size=1,
        moe_dense_tp_size=1 if args.dp_size > 1 else None,
        max_num_seqs=args.batch_size,
    )


def build_runner_config(args, batch_info):
    cache_length = 128
    cache_lengths = []
    while cache_length < batch_info.max_length:
        cache_lengths.append(cache_length)
        cache_length *= 2
    cache_lengths.append(cache_length)
    return RunnerConfig(
        gen_length=args.gen_len,
        block_length=args.block_length,
        prefilling_limit=args.prefilling_limit,
        mini_batch_size=args.mini_batch_size,
        max_length=batch_info.max_length,
        prefill_lengths=batch_info.prefill_lengths,
        cache_lengths=cache_lengths,
        supported_batch_sizes=batch_info.supported_batch_sizes,
        enable_cuda_graph=args.use_cuda_graph,
        use_cross_block=args.batch_size == 1,
        cache=args.cache,
        prefix_cache_num_pages=args.prefix_cache_num_pages,
        parallel_decoding=args.parallel_decoding,
        threshold=args.threshold,
        low_threshold=args.low_threshold,
        use_credit=args.use_credit,
        attention_backend=args.attention_backend,
        flashinfer_decode_batch_mode=getattr(
            args, "flashinfer_decode_batch_mode", "max_batch"
        ),
        flashinfer_prefill_mode=getattr(args, "flashinfer_prefill_mode", "dense"),
        flashinfer_cache_mode=getattr(args, "flashinfer_cache_mode", "dense"),
        kv_cache_layout=getattr(args, "kv_cache_layout", "dense"),
        page_size=getattr(args, "page_size", None),
    )


def pad_batch(input_ids, device, mask_id):
    max_length = max(sample.shape[1] for sample in input_ids)
    batch = torch.full(
        (len(input_ids), max_length),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    for idx, sample in enumerate(input_ids):
        batch[idx, : sample.shape[1]] = sample.to(device)
    return batch


def maybe_disable_sorting(batch_info, disable_sorting):
    if disable_sorting:
        batch_info.sorted_indices = list(range(len(batch_info.input_lengths)))
    return batch_info


def percentile(values, pct):
    if not values:
        return 0
    sorted_values = sorted(int(value) for value in values)
    index = round((len(sorted_values) - 1) * pct)
    return sorted_values[index]


def log_input_shape_summary(input_lengths, batch_info, args, logger):
    logger.info(
        "[Info] Input token lengths: "
        f"count={len(input_lengths)}, min={min(input_lengths)}, "
        f"p50={percentile(input_lengths, 0.50)}, "
        f"p90={percentile(input_lengths, 0.90)}, "
        f"max={max(input_lengths)}"
    )
    logger.info(
        "[Info] Prefill lengths: "
        f"{list(batch_info.prefill_lengths)}, "
        f"sorting={'disabled' if args.disable_sorting else 'enabled'}"
    )
    if args.attention_backend == "flex" and args.disable_sorting:
        logger.info(
            "[Info] FlexAttention prefill shape reuse is best with sorting enabled."
        )


def warmup_runner(runner, args, device, logger):
    warmup_ids = torch.randint(
        0,
        100000,
        (args.mini_batch_size, args.block_length),
        dtype=torch.long,
        device=device,
    )
    original_gen_length = runner.runner_config.gen_length
    runner.runner_config.gen_length = args.block_length
    runner.generate(warmup_ids)

    if args.attention_backend == "flex":
        warmup_shapes = []
        for prefill_length in runner.prefill_lengths:
            warmup_shapes.append((args.mini_batch_size, int(prefill_length)))
            prefill_ids = torch.randint(
                0,
                100000,
                (args.mini_batch_size, int(prefill_length)),
                dtype=torch.long,
                device=device,
            )
            runner.runner_config.gen_length = args.block_length
            runner.generate(prefill_ids)
        logger.info(f"[Info] Flex prefill warmup shapes: {warmup_shapes}")

    runner.runner_config.gen_length = original_gen_length


@torch.no_grad()
def run_worker(args, *, init_method: str = "env://"):
    from fluxserve.cli import _resolve_quant_config, set_process_title

    server_args = None
    context = None
    rank = int(os.environ.get("RANK", "0"))
    gpu_id = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(args.parallel_world_size)
    if args.process_name:
        set_process_title(f"{args.process_name}:rank{rank}")
    logger = BenchmarkLogger(args.log_file, rank)
    logger.info(f"started world_size={world_size} rank={rank} gpu_id={gpu_id} args={args}")
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    all_input_ids, prompts, questions, ids = load_inputs(
        args.dataset, tokenizer, args.dataset_format
    )
    padded_gen_lens = calc_padded_gen_lens(args, all_input_ids)
    dataset_name = Path(args.dataset).stem
    os.makedirs(args.output_dir, exist_ok=True)

    input_lengths = [inp.size(-1) for inp in all_input_ids]
    batch_info = GenerationBatchInfo.from_lengths(
        input_lengths,
        padded_gen_lens,
        batch_size=args.batch_size,
        block_length=args.block_length,
        prefilling_limit=args.prefilling_limit,
        mini_batch_size=args.mini_batch_size,
        gen_length=args.gen_len,
        unbounded_prefill=(
            args.attention_backend == "flashinfer"
            and getattr(args, "flashinfer_prefill_mode", "dense") == "paged"
            and getattr(args, "flashinfer_cache_mode", "dense") == "paged"
            and getattr(args, "kv_cache_layout", "dense") == "paged"
        ),
    )
    if args.cache == "prefix":
        page_size = args.page_size or args.block_length
        pages_per_sequence = (
            batch_info.max_length + page_size - 1
        ) // page_size
        min_active_pages = args.batch_size * pages_per_sequence + 1
        if args.prefix_cache_num_pages < min_active_pages:
            raise ValueError(
                "--prefix-cache-num-pages is too small for one active batch: "
                f"need at least {min_active_pages}, got "
                f"{args.prefix_cache_num_pages}"
            )
    batch_info = maybe_disable_sorting(batch_info, args.disable_sorting)
    logger.info(
        "[Info] Input batching order: "
        + ("original dataset order" if args.disable_sorting else "sorted by input length")
    )
    log_input_shape_summary(input_lengths, batch_info, args, logger)

    logger.info("[Loading model]")
    model_config = AutoConfig.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    model_config.quant_config = _resolve_quant_config(
        model_config,
        args.quantization,
    )
    if model_config.quant_config is None:
        logger.info("[Info] No supported quantization config detected.")
    else:
        logger.info(
            f"[Info] Using quantization config: {model_config.quant_config.get_name()}"
        )

    server_args = build_server_args(args, model_config)
    server_args.device = device
    context = initialize_distributed(
        server_args,
        backend=args.distributed_backend,
        init_method=init_method,
    )
    try:
        initialize_dp_attention(server_args=server_args, model_config=model_config)
        initialize_moe_config(server_args)

        runner_config = build_runner_config(args, batch_info)
        runner_cls = (
            FlashInferDiffusionRunner
            if args.attention_backend == "flashinfer"
            else BlockDiffusionRunner
        )
        runner = runner_cls(
            model_config=model_config,
            server_args=server_args,
            runner_config=runner_config,
            device=device,
        )

        warmup_runner(runner, args, device, logger)
        if args.cache == "prefix":
            from fluxserve.backend.managers.prefix_cache import PrefixCacheManager

            runner.ensure_paged_kv_cache(
                num_device_pages=args.prefix_cache_num_pages
            )
            runner.prefix_cache_manager = PrefixCacheManager(
                page_size=int(runner.runner_config.page_size),
                num_pages=args.prefix_cache_num_pages,
            )

        sorted_input_ids = [all_input_ids[i] for i in batch_info.sorted_indices]
        sorted_padded_gen_lens = [padded_gen_lens[i] for i in batch_info.sorted_indices]
        iterator = (
            tqdm.trange(0, len(sorted_input_ids), args.batch_size)
            if rank == 0
            else range(0, len(sorted_input_ids), args.batch_size)
        )

        start = time.time()
        for i in iterator:
            input_ids = sorted_input_ids[i : i + args.batch_size]
            runner.runner_config.gen_length = max(
                sorted_padded_gen_lens[i : i + len(input_ids)]
            )
            batch_input_ids = pad_batch(input_ids, device, runner.decoder.mask_id)
            inner_start = time.time()
            prev_forwards = runner.num_forwards
            out = runner.generate(batch_input_ids)
            nfe = runner.num_forwards - prev_forwards
            sample_time = time.time() - inner_start

            for j in range(batch_input_ids.shape[0]):
                batch_info.outputs.append(out[j].unsqueeze(0))
            metrics = record_batch_performance_metrics(
                batch_info,
                out,
                sorted_input_ids,
                i,
                nfe,
                sample_time,
                runner.decoder.eos_id,
                runner.decoder.mask_id,
            )
            if rank == 0:
                logger.info(
                    f"[Iter={i:4d}]nfe={nfe:4d}, "
                    f"Token number={metrics.batch_token_number:4d}, "
                    f"Sample_time={sample_time:2.4f}, "
                    f"FPS={metrics.fps:4.2f}({np.mean(batch_info.fpss):4.2f}),"
                    f"TPF={metrics.tpf:2.2f}({np.mean(batch_info.tpfs):4.2f}), "
                    f"TPS={metrics.tps:4.2f}({np.mean(batch_info.tpss):4.2f})"
                )
        stop = time.time()

        if rank == 0:
            if args.cache == "prefix":
                logger.info(
                    f"[Prefix cache] {runner.prefix_cache_manager.snapshot()}"
                )
            _write_results(
                args,
                batch_info,
                all_input_ids,
                prompts,
                questions,
                ids,
                tokenizer,
                dataset_name,
                start,
                stop,
                logger,
            )
    finally:
        destroy_distributed()


def resolve_log_file(args) -> str:
    if args.log_file is None:
        return os.path.join(args.output_dir, f"{args.exp_name}.log")
    if not os.path.isabs(args.log_file) and os.path.dirname(args.log_file) == "":
        return os.path.join(args.output_dir, args.log_file)
    return args.log_file


def _write_results(
    args,
    batch_info,
    all_input_ids,
    prompts,
    questions,
    ids,
    tokenizer,
    dataset_name,
    start,
    stop,
    logger,
) -> None:
    outputs = batch_info.original_order(batch_info.outputs)
    tpfs = batch_info.original_order(batch_info.tpfs)
    tpss = batch_info.original_order(batch_info.tpss)
    fpss = batch_info.original_order(batch_info.fpss)
    token_numbers = batch_info.original_order(batch_info.token_numbers)
    answers = [
        tokenizer.decode(
            cut_eos(outputs[i][:, all_input_ids[i].shape[1] :])[0],
            skip_special_tokens=True,
        )
        for i in tqdm.trange(len(outputs))
    ]
    print_output_summary(summarize_outputs(answers, token_numbers), logger)
    logger.info(
        f"Forward: {batch_info.total_forward}, Time: {stop - start}, "
        f"FPS: {batch_info.total_forward / batch_info.total_time}({np.mean(fpss)}), "
        f"TPS: {batch_info.total_token / batch_info.total_time}({np.mean(tpss)}), "
        f"TPF: {batch_info.total_token / batch_info.total_forward}({np.mean(tpfs)})"
    )
    filename = os.path.join(
        args.output_dir,
        f"{args.exp_name}_{dataset_name}_{args.parallel_decoding}_{args.threshold}.jsonl",
    )
    with open(filename, "w") as f:
        for i, answer in enumerate(answers):
            json.dump(
                {
                    "id": ids[i],
                    "question": questions[i],
                    "prompt": prompts[i],
                    "answer": answer,
                    "generated_length": token_numbers[i],
                    "tpf": tpfs[i],
                    "tps": tpss[i],
                    "fps": fpss[i],
                },
                f,
            )
            f.write("\n")


def add_bench_offline_subparser(subparsers) -> None:
    parser = subparsers.add_parser("bench_offline")
    parser.add_argument("--model", "--model-name", "--model_name", dest="model_name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=1)
    parser.add_argument("--mini-batch-size", "--mini_batch_size", dest="mini_batch_size", type=int, default=4)
    parser.add_argument("--use-naive-batching", "--use_naive_batching", dest="use_naive_batching", action="store_true")
    parser.add_argument("--tp-size", "--tp_size", dest="tp_size", type=int, default=1)
    parser.add_argument("--dp-size", "--dp_size", dest="dp_size", type=int, default=1)
    parser.add_argument("--ep-size", "--ep_size", dest="ep_size", type=int, default=1)
    parser.add_argument("--pp-size", "--pp_size", dest="pp_size", type=int, default=1)
    parser.add_argument("--distributed-backend", "--distributed_backend", dest="distributed_backend", default="nccl")
    parser.add_argument(
        "--quantization",
        choices=("auto", *QUANTIZATION_METHODS),
        default="auto",
    )
    parser.add_argument("--use-quant", "--use_quant", dest="use_quant", action="store_true")
    parser.add_argument("--use-cuda-graph", "--use_cuda_graph", dest="use_cuda_graph", action="store_true")
    parser.add_argument("--prefilling-limit", "--prefilling_limit", dest="prefilling_limit", type=int, default=128)
    parser.add_argument("--attention-backend", "--attention_backend", dest="attention_backend", choices=("sdpa", "flex", "flashinfer"), default="flashinfer")
    parser.add_argument("--flashinfer-decode-batch-mode", "--flashinfer_decode_batch_mode", dest="flashinfer_decode_batch_mode", choices=("default", "max_batch"), default="max_batch")
    parser.add_argument("--flashinfer-prefill-mode", "--flashinfer_prefill_mode", dest="flashinfer_prefill_mode", choices=("dense", "ragged", "paged"), default="paged")
    parser.add_argument("--flashinfer-cache-mode", "--flashinfer_cache_mode", dest="flashinfer_cache_mode", choices=("dense", "paged"), default="paged")
    parser.add_argument("--kv-cache-layout", "--kv_cache_layout", dest="kv_cache_layout", choices=("dense", "paged"), default="paged")
    parser.add_argument("--page-size", "--page_size", dest="page_size", type=int)
    parser.add_argument("--gen-len", "--gen_len", dest="gen_len", type=int, default=1024)
    parser.add_argument("--block-length", "--block_length", dest="block_length", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--low-threshold", "--low_threshold", dest="low_threshold", type=float, default=0.3)
    parser.add_argument("--parallel-decoding", "--parallel_decoding", dest="parallel_decoding", default="threshold")
    parser.add_argument("--use-credit", "--use_credit", dest="use_credit", action="store_true")
    parser.add_argument("--cache", choices=("prefix",), default="")
    parser.add_argument(
        "--prefix-cache-num-pages",
        "--prefix_cache_num_pages",
        dest="prefix_cache_num_pages",
        type=int,
        default=0,
    )
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


def bench_offline(args) -> None:
    from fluxserve.cli import set_process_title

    reject_external_distributed_launch()
    normalize_attention_backend_args(args)
    if args.cache == "prefix":
        page_size = args.page_size or args.block_length
        if not (
            args.attention_backend == "flashinfer"
            and args.flashinfer_prefill_mode == "paged"
            and args.flashinfer_cache_mode == "paged"
            and args.kv_cache_layout == "paged"
        ):
            raise ValueError(
                "--cache prefix requires FlashInfer paged prefill, cache, and KV layout"
            )
        if args.block_length % page_size != 0:
            raise ValueError(
                "--cache prefix requires page_size to divide block_length"
            )
        if args.prefix_cache_num_pages <= 1:
            raise ValueError("--prefix-cache-num-pages must be greater than 1")
    if args.use_quant:
        args.quantization = "modelopt_fp8"
    args.log_file = resolve_log_file(args)
    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
    with open(args.log_file, "w"):
        pass
    logger = BenchmarkLogger(args.log_file, rank=0)
    logger.info(f"[Info] Writing benchmark log to {args.log_file}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.mini_batch_size <= 0:
        raise ValueError("--mini-batch-size must be positive")
    if args.tp_size <= 0:
        raise ValueError("--tp_size must be positive")
    if args.ep_size <= 0:
        raise ValueError("--ep_size must be positive")
    if args.dp_size <= 0:
        raise ValueError("--dp_size must be positive")
    if args.dp_size > 1 and args.dp_size != args.ep_size:
        raise ValueError(
            "This benchmark currently supports dp_attention with EP only when "
            f"dp_size == ep_size, got dp_size={args.dp_size}, ep_size={args.ep_size}"
        )
    args.parallel_world_size = args.tp_size * args.dp_size
    args.world_size = args.parallel_world_size
    args.enable_dp_attention = args.dp_size > 1
    if args.dp_size > 1 and args.use_cuda_graph:
        logger.info(
            "[Info] Disabling CUDA graph because dp_attention requires "
            "ForwardBatch metadata."
        )
        args.use_cuda_graph = False
    if args.batch_size == 1:
        args.use_naive_batching = True
    if args.use_naive_batching:
        args.mini_batch_size = args.batch_size
    elif args.mini_batch_size > args.batch_size:
        logger.info(
            "[Info] Clamping mini_batch_size to batch_size because benchmark "
            f"batches contain at most {args.batch_size} sequences "
            f"(requested mini_batch_size={args.mini_batch_size})."
        )
        args.mini_batch_size = args.batch_size
    
    if args.dp_size > 1:
        logger.info("[Info] Disabling model TP because dp_size > 1.")
        args.use_tp = False
    else:
        args.use_tp = args.tp_size > 1
    args.attn_tp_size = args.parallel_world_size // args.dp_size
    logger.info(
        "[Info] Effective parallelism: "
        f"world/tp_group_size={args.parallel_world_size}, "
        f"model_tp_enabled={args.use_tp}, attention_tp_size={args.attn_tp_size}, "
        f"requested_tp_size={args.tp_size}, dp_size={args.dp_size}, "
        f"ep_size={args.ep_size}"
    )

    logger.info(str(args))
    require_nvidia_cuda(args.device)
    if should_launch_local_workers(args.world_size):
        if args.process_name:
            set_process_title(f"{args.process_name}:supervisor")
        launch_local_workers(run_worker, args, world_size=args.world_size)
    else:
        os.environ.setdefault("FLUXSERVE_SUPPRESS_DEFAULT_MOE_CONFIG_WARNING", "1")
        run_worker(args)

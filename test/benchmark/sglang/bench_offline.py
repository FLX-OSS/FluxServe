#!/usr/bin/env python3
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import sglang as sgl
from transformers import AutoTokenizer


def disable_flashinfer_deepseek_topk_if_needed(mode: str) -> bool:
    if mode == "never":
        return False

    try:
        import torch
    except ImportError:
        if mode == "always":
            torch = None
        else:
            return False

    should_disable = mode == "always"
    if mode == "auto":
        should_disable = (
            torch is not None
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability(0)[0] < 9
        )
    if not should_disable:
        return False

    try:
        from sglang.srt.layers.moe import topk as sglang_topk
    except ImportError:
        return False

    if getattr(sglang_topk, "fused_topk_deepseek", None) is None:
        return False

    sglang_topk.fused_topk_deepseek = None
    return True


def configure_child_flashinfer_deepseek_topk_patch(mode: str) -> None:
    should_disable = mode == "always"
    if mode == "auto":
        try:
            import torch

            should_disable = (
                torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 9
            )
        except ImportError:
            should_disable = False

    if not should_disable:
        return

    sitecustomize_dir = Path(__file__).resolve().parent / "sglang_topk_sitecustomize"
    os.environ["SGLANG_DISABLE_FLASHINFER_DEEPSEEK_TOPK"] = "1"
    os.environ["PYTHONPATH"] = (
        str(sitecustomize_dir)
        if not os.environ.get("PYTHONPATH")
        else f"{sitecustomize_dir}{os.pathsep}{os.environ['PYTHONPATH']}"
    )
    if str(sitecustomize_dir) not in sys.path:
        sys.path.insert(0, str(sitecustomize_dir))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
    return rows


def record_to_prompt(record: dict[str, Any], tokenizer) -> str:
    messages = record.get("messages")
    if isinstance(messages, list):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return "\n".join(
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in messages
                if isinstance(msg, dict)
            )

    for key in ("prompt", "text", "content"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value

    raise ValueError("JSONL row has no supported prompt field")


def iter_batches(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start // batch_size, items[start : start + batch_size]


def mem_fraction(value: str) -> float:
    fraction = float(value)
    if not 0 < fraction <= 1:
        raise argparse.ArgumentTypeError("--mem-fraction-static must be in (0, 1]")
    return fraction


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def load_dllm_algorithm_config(path: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "Please install PyYAML to merge --dllm-algorithm-config with "
            "--dllm-block-size."
        ) from e

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"--dllm-algorithm-config must contain a mapping: {path}")
    return config


def make_dllm_algorithm_config_path(
    config_path: str, block_size: int | None
) -> tuple[str, str | None]:
    if block_size is None:
        return config_path, None

    config = load_dllm_algorithm_config(config_path) if config_path else {}
    config["block_size"] = block_size

    fd, tmp_path = tempfile.mkstemp(
        prefix="sglang_dllm_algorithm_config_", suffix=".json"
    )
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(config, f)
        f.write("\n")
    return tmp_path, tmp_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a batched JSONL benchmark through sglang.Engine."
    )
    parser.add_argument("--model-path", default="inclusionAI/LLaDA2.0-mini")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("datasets/openai_humaneval.openai.jsonl"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gen-length", type=int, default=2048)
    parser.add_argument("--result-filename", default="humaneval_bench_engine.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--dllm-algorithm", default="LowConfidence")
    parser.add_argument("--dllm-algorithm-config", default="")
    parser.add_argument(
        "--dllm-block-size",
        type=positive_int,
        default=None,
        help=(
            "Override the diffusion LLM block_size. SGLang reads this from "
            "dllm_algorithm_config, so this script writes a temporary JSON config."
        ),
    )
    parser.add_argument("--attention-backend", default="torch_native")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--disable-radix-cache", action="store_true", default=True)
    parser.add_argument("--max-running-requests", type=int, default=4)
    parser.add_argument(
        "--tp-size",
        "--tp",
        dest="tp_size",
        type=positive_int,
        default=1,
        help="SGLang tensor parallel size. For TP+EP, this is the total TP world.",
    )
    parser.add_argument(
        "--ep-size",
        "--ep",
        dest="ep_size",
        type=positive_int,
        default=1,
        help="SGLang expert parallel size inside the TP world.",
    )
    parser.add_argument(
        "--dp-size",
        "--dp",
        dest="dp_size",
        type=positive_int,
        default=1,
        help="SGLang attention data parallel size.",
    )
    parser.add_argument(
        "--moe-dp-size",
        type=positive_int,
        default=1,
        help="SGLang MoE data parallel size used when deriving MoE TP/EP ranks.",
    )
    parser.add_argument(
        "--enable-dp-attention",
        action="store_true",
        help="Enable SGLang DP attention when --dp-size is greater than 1.",
    )
    parser.add_argument(
        "--moe-a2a-backend",
        default="none",
        choices=(
            "none",
            "deepep",
            "mooncake",
            "nixl",
            "mori",
            "ascend_fuseep",
            "flashinfer",
            "megamoe",
        ),
        help="SGLang MoE all-to-all backend for expert parallelism.",
    )
    parser.add_argument("--moe-runner-backend", default="auto")
    parser.add_argument(
        "--deepep-mode",
        default="auto",
        choices=("auto", "normal", "low_latency"),
    )
    parser.add_argument("--deepep-config", default=None)
    parser.add_argument(
        "--moe-dense-tp-size",
        type=positive_int,
        default=None,
        help="Optional SGLang dense-layer TP size for MoE models.",
    )
    parser.add_argument(
        "--disable-flashinfer-deepseek-topk",
        choices=("auto", "always", "never"),
        default="always",
        help=(
            "Disable FlashInfer fused DeepSeek top-k on unsupported GPUs. "
            "auto disables it when CUDA compute capability is below 9.0."
        ),
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=mem_fraction,
        default=0.8,
        help=(
            "Fraction of GPU memory SGLang may use for model weights plus KV cache. "
            "Lower this to leave more VRAM free for other processes."
        ),
    )
    args = parser.parse_args()

    if args.ep_size > args.tp_size:
        raise ValueError(
            f"--ep-size must be <= --tp-size, got ep_size={args.ep_size}, "
            f"tp_size={args.tp_size}"
        )
    if args.tp_size % (args.ep_size * args.moe_dp_size) != 0:
        raise ValueError(
            "--tp-size must be divisible by --ep-size * --moe-dp-size, got "
            f"tp_size={args.tp_size}, ep_size={args.ep_size}, "
            f"moe_dp_size={args.moe_dp_size}"
        )
    if args.enable_dp_attention and args.dp_size <= 1:
        raise ValueError("--enable-dp-attention requires --dp-size > 1")
    if args.enable_dp_attention and args.tp_size % args.dp_size != 0:
        raise ValueError(
            f"--tp-size must be divisible by --dp-size for DP attention, got "
            f"tp_size={args.tp_size}, dp_size={args.dp_size}"
        )

    rows = load_jsonl(args.dataset_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    prompts = [record_to_prompt(row, tokenizer) for row in rows]
    prompt_lens = [len(tokenizer.encode(prompt)) for prompt in prompts]

    engine_kwargs = {
        "model_path": args.model_path,
        "trust_remote_code": args.trust_remote_code,
        "dllm_algorithm": args.dllm_algorithm,
        "attention_backend": args.attention_backend,
        "disable_cuda_graph": args.disable_cuda_graph,
        "disable_radix_cache": args.disable_radix_cache,
        "max_running_requests": args.max_running_requests,
        "tp_size": args.tp_size,
        "dp_size": args.dp_size,
        "ep_size": args.ep_size,
        "moe_dp_size": args.moe_dp_size,
        "enable_dp_attention": args.enable_dp_attention,
        "moe_a2a_backend": args.moe_a2a_backend,
        "moe_runner_backend": args.moe_runner_backend,
        "deepep_mode": args.deepep_mode,
    }
    if args.mem_fraction_static is not None:
        engine_kwargs["mem_fraction_static"] = args.mem_fraction_static
    dllm_algorithm_config_path, tmp_dllm_algorithm_config_path = (
        make_dllm_algorithm_config_path(
            args.dllm_algorithm_config, args.dllm_block_size
        )
    )
    if dllm_algorithm_config_path:
        engine_kwargs["dllm_algorithm_config"] = dllm_algorithm_config_path
    if args.deepep_config:
        engine_kwargs["deepep_config"] = args.deepep_config
    if args.moe_dense_tp_size is not None:
        engine_kwargs["moe_dense_tp_size"] = args.moe_dense_tp_size

    configure_child_flashinfer_deepseek_topk_patch(
        args.disable_flashinfer_deepseek_topk
    )
    disabled_flashinfer_deepseek_topk = disable_flashinfer_deepseek_topk_if_needed(
        args.disable_flashinfer_deepseek_topk
    )

    print(f"Using sglang from: {sgl.__file__}")
    print(
        "Parallelism: "
        f"tp_size={args.tp_size}, ep_size={args.ep_size}, "
        f"dp_size={args.dp_size}, moe_dp_size={args.moe_dp_size}, "
        f"moe_tp_size={args.tp_size // args.ep_size // args.moe_dp_size}, "
        f"enable_dp_attention={args.enable_dp_attention}"
    )
    print(
        "FlashInfer fused DeepSeek top-k disabled: "
        f"{disabled_flashinfer_deepseek_topk}"
    )
    result_path = Path(args.result_filename)
    sampling_params = {
        "temperature": 0,
        "max_new_tokens": args.gen_length,
    }

    batch_results = []
    llm = None
    try:
        print(f"Loading engine: {args.model_path}")
        llm = sgl.Engine(**engine_kwargs)
        total_start = time.perf_counter()
        with result_path.open("a", encoding="utf-8") as fout:
            for batch_index, batch_prompts in iter_batches(prompts, args.batch_size):
                start = batch_index * args.batch_size
                batch_prompt_lens = prompt_lens[start : start + len(batch_prompts)]
                tic = time.perf_counter()
                outputs = llm.generate(batch_prompts, sampling_params)
                latency = time.perf_counter() - tic

                completion_tokens = 0
                for output in outputs:
                    meta_info = output.get("meta_info") if isinstance(output, dict) else None
                    if isinstance(meta_info, dict):
                        completion_tokens += int(
                            meta_info.get("completion_tokens", args.gen_length)
                        )
                    else:
                        completion_tokens += args.gen_length

                result = {
                    "mode": "engine_jsonl_batch",
                    "batch_index": batch_index,
                    "batch_size": len(batch_prompts),
                    "gen_length": args.gen_length,
                    "latency": latency,
                    "input_tokens": sum(batch_prompt_lens),
                    "output_tokens": completion_tokens,
                    "request_throughput": len(batch_prompts) / latency,
                    "output_throughput": completion_tokens / latency,
                }
                print(json.dumps(result))
                fout.write(json.dumps(result) + "\n")
                fout.flush()
                batch_results.append(result)

            total_latency = time.perf_counter() - total_start
            total_input_tokens = sum(result["input_tokens"] for result in batch_results)
            total_output_tokens = sum(result["output_tokens"] for result in batch_results)
            aggregate = {
                "mode": "engine_jsonl_total",
                "num_prompts": len(prompts),
                "num_batches": len(batch_results),
                "batch_size": args.batch_size,
                "gen_length": args.gen_length,
                "total_latency": total_latency,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "request_throughput": len(prompts) / total_latency,
                "output_throughput": total_output_tokens / total_latency,
                "total_throughput": (total_input_tokens + total_output_tokens)
                / total_latency,
            }
            print(json.dumps(aggregate))
            fout.write(json.dumps(aggregate) + "\n")
    finally:
        if llm is not None:
            llm.shutdown()
        if tmp_dllm_algorithm_config_path is not None:
            Path(tmp_dllm_algorithm_config_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()

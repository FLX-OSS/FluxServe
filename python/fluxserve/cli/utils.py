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
    Shared command configuration helpers.
"""

def default_cuda_graph_capture_batch_sizes(max_num_seqs: int) -> tuple[int, ...]:
    """Return batch size 1 and every positive even size up to the limit."""
    max_num_seqs = int(max_num_seqs)
    if max_num_seqs <= 0:
        raise ValueError("max_num_seqs must be positive")
    return (1, *range(2, max_num_seqs + 1, 2))


def set_process_title(title: str) -> None:
    try:
        import setproctitle
    except ImportError:
        return
    setproctitle.setproctitle(title)


def normalize_attention_backend_args(args) -> None:
    if args.attention_backend == "flashinfer":
        return
    args.flashinfer_prefill_mode = "dense"
    args.flashinfer_cache_mode = "dense"
    if args.attention_backend == "fa4":
        if getattr(args, "kv_cache_layout", "paged") != "paged":
            raise ValueError("attention_backend='fa4' requires --kv-cache-layout paged")
        if getattr(args, "page_size", None) is None:
            args.page_size = int(args.block_length)
        return
    args.kv_cache_layout = "dense"
    args.page_size = None


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


def _check_block_routing_alignment(model_config, block_length: int) -> None:
    """LLaDA2.2 MoE block routing selects experts per config.block_size-token
    window; serving blocks must tile those windows exactly."""
    if not getattr(model_config, "expert_capacity", 0):
        return
    model_block_size = int(getattr(model_config, "block_size", 0) or 0)
    if model_block_size and int(block_length) % model_block_size != 0:
        raise ValueError(
            f"this checkpoint uses MoE block routing with block_size="
            f"{model_block_size}; --block-length ({block_length}) must be a "
            "multiple of it."
        )


def normalize_diffusion_gemma_serve_args(args, model_config) -> bool:
    architectures = set(getattr(model_config, "architectures", ()) or ())
    is_diffusion_gemma = (
        "DiffusionGemmaForBlockDiffusion" in architectures
        or getattr(model_config, "model_type", None) == "diffusion_gemma"
    )
    if not is_diffusion_gemma:
        return False
    if args.attention_backend == "fa4":
        raise ValueError("attention_backend='fa4' currently supports LLaDA 2.x only")
    if args.canvas_length is not None and int(args.canvas_length) <= 0:
        raise ValueError("--canvas-length must be positive")
    if args.use_cuda_graph or args.use_prefill_cuda_graph:
        raise ValueError(
            "Diffusion-Gemma supports decode CUDA graphs only; use "
            "--use-decode-cuda-graph."
        )
    if args.use_decode_cuda_graph:
        if not getattr(args, "attention_backend_explicit", False):
            args.attention_backend = "flashinfer"
        if not (
            args.attention_backend == "flashinfer"
            and args.flashinfer_prefill_mode == "paged"
            and args.flashinfer_cache_mode == "paged"
            and args.kv_cache_layout == "paged"
        ):
            raise ValueError(
                "Diffusion-Gemma decode CUDA graphs require FlashInfer paged "
                "prefill, paged cache mode, and paged KV layout."
            )
    elif not getattr(args, "attention_backend_explicit", False):
        args.attention_backend = "sdpa"
        normalize_attention_backend_args(args)
    return True

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
    FluxServe CLI helpers.
"""

from __future__ import annotations

import argparse
import logging


class StoreExplicit(argparse.Action):
    """Store an option and record that the user supplied it explicitly."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, f"{self.dest}_explicit", True)


def configure_logging(rank: int = 0) -> None:
    """Configure the FluxServe logger without enabling third-party chatter."""
    namespace = logging.getLogger("fluxserve")
    if any(getattr(handler, "_fluxserve_handler", False) for handler in namespace.handlers):
        return
    handler = logging.StreamHandler()
    handler._fluxserve_handler = True
    handler.setFormatter(
        logging.Formatter(f"[fluxserve][rank{rank}] %(levelname)s %(name)s: %(message)s")
    )
    namespace.addHandler(handler)
    namespace.setLevel(logging.INFO)


def default_cuda_graph_capture_batch_sizes(max_num_seqs: int) -> tuple[int, ...]:
    """Return batch size 1 and every positive even size up to the limit."""
    max_num_seqs = int(max_num_seqs)
    if max_num_seqs <= 0:
        raise ValueError("max_num_seqs must be positive")
    return (1, *range(2, max_num_seqs + 1, 2))


def set_process_title(title: str) -> None:
    """Set the OS process title when the optional dependency is available."""
    try:
        import setproctitle
    except ImportError:
        return
    setproctitle.setproctitle(title)


def reject_unsupported_quantization(model_config) -> None:
    """Reject checkpoint quantization modes unsupported by FluxServe."""
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


def check_block_routing_alignment(model_config, block_length: int) -> None:
    """Ensure serving blocks tile LLaDA2.2 MoE routing windows."""
    if not getattr(model_config, "expert_capacity", 0):
        return
    model_block_size = int(getattr(model_config, "block_size", 0) or 0)
    if model_block_size and int(block_length) % model_block_size != 0:
        raise ValueError(
            f"this checkpoint uses MoE block routing with block_size="
            f"{model_block_size}; --block-length ({block_length}) must be a "
            "multiple of it."
        )


def normalize_attention_backend_args(args) -> None:
    """Normalize cache and page settings for the selected attention backend."""
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

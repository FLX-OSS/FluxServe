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
    Diffusion-Gemma FA4 runner.
"""
from __future__ import annotations

import torch

from fluxserve.backend.execution.runners.diffusion_gemma.flashinfer import (
    DiffusionGemmaFlashInferRunner,
)
from fluxserve.backend.layers.attention.diffusion_gemma_flashinfer import (
    FA4_SLIDING_PAGE_SIZE,
    probe_diffusion_gemma_fa4,
)
from fluxserve.backend.layers.attention.diffusion_gemma_window import (
    validate_sliding_window_mode,
)
from fluxserve.backend.layers.attention.fa4 import validate_fa4_runtime


class DiffusionGemmaFA4Runner(DiffusionGemmaFlashInferRunner):
    """Diffusion-Gemma with FA4 sliding-window layers and FlashInfer full layers.

    The sliding-window layers (head_dim 256) run on FA4's SM100 head_dim-256
    kernel. The full-attention layers (head_dim 512, K != V) have no FA4 kernel
    yet and stay on FlashInfer. Both share one paged KV cache (page 128), the
    batch loop, and the denoise CUDA graphs.
    """

    _supports_flashinfer = False
    _supports_fa4 = True
    sliding_backend = "fa4"

    def _validate_paged_runner_config(self, runner_config) -> None:
        if runner_config is None or runner_config.attention_backend != "fa4":
            raise ValueError("DiffusionGemmaFA4Runner requires attention_backend='fa4'")
        if runner_config.kv_cache_layout != "paged":
            raise ValueError("Diffusion-Gemma FA4 requires the paged KV layout")
        if int(runner_config.page_size or 0) != FA4_SLIDING_PAGE_SIZE:
            raise ValueError(
                f"Diffusion-Gemma FA4 requires page_size={FA4_SLIDING_PAGE_SIZE} "
                f"(the SM100 head_dim-256 TMA page), got {runner_config.page_size}"
            )
        validate_sliding_window_mode(
            getattr(runner_config, "sliding_window_mode", "per_query"),
            getattr(runner_config, "sliding_window_left", None),
        )

    def _flashinfer_serves(self, attention) -> bool:
        return getattr(attention, "sliding_window", None) is None

    def _probe_sliding_backend(self) -> None:
        validate_fa4_runtime(self.device)
        device = torch.device(self.device)
        index = device.index if device.index is not None else torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        if major not in (10, 11):
            raise RuntimeError(
                "Diffusion-Gemma FA4 needs FA4's SM100/SM110 head_dim-256 kernel; "
                f"got compute capability {major}.{minor}"
            )
        geometries = []
        for layer in self.model.model.layers:
            attention = layer.self_attn
            if getattr(attention, "sliding_window", None) is None:
                continue
            if int(attention.head_dim) != 256:
                raise ValueError(
                    "Diffusion-Gemma FA4 serves head_dim-256 sliding-window layers; "
                    f"layer {getattr(attention, 'layer_id', '?')} has head_dim "
                    f"{attention.head_dim}"
                )
            geometries.append(
                (
                    int(attention.num_heads),
                    int(attention.num_kv_heads),
                    int(attention.head_dim),
                    int(attention.sliding_window),
                )
            )
        probe_diffusion_gemma_fa4(
            geometries,
            self.device,
            sliding_window_mode=getattr(
                self.runner_config, "sliding_window_mode", "per_query"
            ),
            sliding_window_left=getattr(self.runner_config, "sliding_window_left", None),
        )

    def _gemma_page_size(self) -> int:
        return FA4_SLIDING_PAGE_SIZE


__all__ = ["DiffusionGemmaFA4Runner"]

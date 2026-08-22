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

import torch.nn as nn


def get_model(
    *,
    model_config,
    device: str,
    quant_config=None,
) -> nn.Module:
    from fluxserve.backend.model_loader.loader import (
        DefaultModelLoader,
        DiffusionGemmaModelLoader,
    )

    architectures = set(getattr(model_config, "architectures", ()) or ())
    is_diffusion_gemma = (
        "DiffusionGemmaForBlockDiffusion" in architectures
        or getattr(model_config, "model_type", None) == "diffusion_gemma"
    )
    loader = DiffusionGemmaModelLoader() if is_diffusion_gemma else DefaultModelLoader()
    return loader.load_model(
        model_config=model_config,
        device=device,
        quant_config=quant_config,
    )


def __getattr__(name: str):
    if name == "DefaultModelLoader":
        from fluxserve.backend.model_loader.loader import DefaultModelLoader

        return DefaultModelLoader
    raise AttributeError(name)


__all__ = ["DefaultModelLoader", "get_model"]

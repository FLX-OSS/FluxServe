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

from .base import ModelRunner
from .block_diffusion import BlockDiffusionRunner
from .flashinfer_diffusion import FlashInferDiffusionRunner
from .fa4_diffusion import FA4DiffusionRunner
from .diffusion_gemma import DiffusionGemmaRunner
from .nemotron_diffusion import NemotronDiffusionRunner
from .nemotron_fa4 import NemotronFA4DiffusionRunner
from .nemotron_flashinfer import (
    NemotronFlashInferDiffusionRunner,
    NemotronFlashInferSelfSpecRunner,
)
from .nemotron_selfspec import NemotronSelfSpecRunner
from .nemotron_selfspec_paged import NemotronSelfSpecPagedRunner


def get_nemotron_runner(backend: str, decoding: str):
    runners = {
        "threshold": {
            "sdpa": NemotronDiffusionRunner,
            "fa4": NemotronFA4DiffusionRunner,
            "flashinfer": NemotronFlashInferDiffusionRunner,
        },
        "self_speculation": {
            "sdpa": NemotronSelfSpecRunner,
            "fa4": NemotronSelfSpecPagedRunner,
            "flashinfer": NemotronFlashInferSelfSpecRunner,
        },
    }
    try:
        return runners[decoding][backend]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported Nemotron backend/decoding: {backend!r}/{decoding!r}"
        ) from exc


__all__ = [
    "BlockDiffusionRunner",
    "DiffusionGemmaRunner",
    "FA4DiffusionRunner",
    "FlashInferDiffusionRunner",
    "ModelRunner",
    "NemotronDiffusionRunner",
    "NemotronFA4DiffusionRunner",
    "NemotronFlashInferDiffusionRunner",
    "NemotronFlashInferSelfSpecRunner",
    "NemotronSelfSpecPagedRunner",
    "NemotronSelfSpecRunner",
    "get_nemotron_runner",
]

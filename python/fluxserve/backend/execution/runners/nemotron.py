# Copyright (c) 2026 FLUX-OSS
# SPDX-License-Identifier: MIT

"""Shared server/offline runner selection for Nemotron."""

from fluxserve.backend.execution.runners.nemotron_diffusion import NemotronDiffusionRunner
from fluxserve.backend.execution.runners.nemotron_fa4 import NemotronFA4DiffusionRunner
from fluxserve.backend.execution.runners.nemotron_flashinfer import (
    NemotronFlashInferDiffusionRunner,
    NemotronFlashInferSelfSpecRunner,
)
from fluxserve.backend.execution.runners.nemotron_selfspec import NemotronSelfSpecRunner
from fluxserve.backend.execution.runners.nemotron_selfspec_paged import NemotronSelfSpecPagedRunner


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

# Copyright (c) 2026 FLUX-OSS
# SPDX-License-Identifier: MIT
"""Request-owned sampling state for Nemotron's two selection rules."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


def validate_sampling_params(params):
    temperature = params.get("temperature", 0.0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be a finite non-negative number")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be a finite non-negative number")
    seed = params.get("seed")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63
    ):
        raise ValueError("seed must be an integer in [0, 2**63)")
    if params.get("top_p", 1.0) != 1.0 or params.get("top_k", -1) not in (-1, 0):
        raise ValueError("Nemotron supports temperature sampling with top_p=1 and no top_k filter")
    for name in ("frequency_penalty", "presence_penalty"):
        if params.get(name, 0.0) != 0.0:
            raise ValueError(f"Nemotron does not support {name}")


@dataclass
class NemotronSampling:
    temperature: float = 0.0
    seed: int = 0
    _generators: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        validate_sampling_params({"temperature": self.temperature, "seed": self.seed})

    def generator(self, device):
        key = str(device)
        if key not in self._generators:
            self._generators[key] = torch.Generator(device=device).manual_seed(self.seed)
        return self._generators[key]

    def categorical(self, logits):
        if self.temperature == 0:
            return logits.argmax(dim=-1)
        probabilities = torch.softmax(logits / self.temperature, dim=-1)
        return torch.multinomial(
            probabilities.reshape(-1, probabilities.shape[-1]), 1,
            generator=self.generator(logits.device),
        ).reshape(logits.shape[:-1])

    def diffusion(self, logits):
        if self.temperature == 0:
            return logits.argmax(dim=-1)
        # Reproduce the checkpoint's float64 Gumbel-max transform. Confidence
        # still comes from unscaled logits in the threshold decoder.
        values = logits.to(torch.float64)
        noise = torch.rand(
            values.shape, dtype=torch.float64, device=values.device,
            generator=self.generator(values.device),
        )
        return (values.exp() / (-noise.log()).pow(self.temperature)).argmax(-1)


def make_sampling(config=None, params=None):
    values = {
        "temperature": getattr(config, "temperature", 0.0),
        "seed": getattr(config, "seed", 0),
    }
    values.update(params or {})
    validate_sampling_params(values)
    # Offline default is explicit and identical across TP ranks. Online input
    # processing assigns unseeded requests a seed before rank broadcasting.
    return NemotronSampling(float(values["temperature"]), values["seed"] or 0)


def sample_tokens(logits, sampling=None):
    return logits.argmax(-1) if sampling is None else sampling.categorical(logits)


class NemotronSamplingMixin:
    supports_request_sampling = True

    def _stop_on_eos_batch(self, batch_size, sampling_params=None):
        if sampling_params is None or isinstance(sampling_params, dict):
            sampling_params = [sampling_params] * batch_size
        return [
            not params["ignore_eos"] if params and "ignore_eos" in params
            else self.early_stop
            for params in sampling_params
        ]

    def _sampling_batch(self, batch_size, sampling_params=None):
        if sampling_params is None or isinstance(sampling_params, dict):
            sampling_params = [sampling_params] * batch_size
        if len(sampling_params) != batch_size:
            raise ValueError("sampling_params must contain one entry per request")
        return [make_sampling(self.runner_config, params) for params in sampling_params]

    def _sampling_for_request(self, request):
        if not hasattr(self, "_request_sampling"):
            self._request_sampling = {}
        rid = str(request.rid)
        if rid not in self._request_sampling:
            self._request_sampling[rid] = make_sampling(
                self.runner_config, getattr(request, "sampling_params", {})
            )
        return self._request_sampling[rid]

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
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import torch
from safetensors.torch import load_file


def get_model_name(config) -> str:
    model_name = getattr(config, "_name_or_path", "")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError(
            "Model config must provide a Hugging Face model identifier in "
            "PretrainedConfig._name_or_path"
        )
    return model_name.strip()


@lru_cache(maxsize=None)
def _download_model_snapshot(model_name: str) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to resolve model weights"
        ) from exc

    return snapshot_download(
        repo_id=model_name,
        allow_patterns=[
            "*.safetensors",
            "model.safetensors.index.json",
            "hf_quant_config.json",
        ],
    )


def resolve_model_snapshot(config) -> Path:
    model_name = get_model_name(config)
    model_path = Path(model_name).expanduser()
    if (model_path / "model.safetensors.index.json").is_file():
        model_dir = model_path
    else:
        model_dir = Path(_download_model_snapshot(model_name))
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Hugging Face model '{model_name}' does not provide "
            "model.safetensors.index.json"
        )
    return model_dir


def load_safetensors_index(model_dir: Path) -> dict:
    index_path = model_dir / "model.safetensors.index.json"
    with open(index_path, "r") as f:
        return json.load(f)


def get_safetensors_shard_files(model_dir: Path) -> list[str]:
    index = load_safetensors_index(model_dir)
    return sorted(set(index["weight_map"].values()))


def iter_safetensors_shards(
    model_dir: Path,
    shard_files: list[str],
) -> Iterator[tuple[str, dict[str, torch.Tensor]]]:
    for shard in shard_files:
        shard_path = model_dir / shard
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard: {shard_path}")
        with torch.inference_mode():
            yield shard, load_file(str(shard_path))


@dataclass
class ExpertMappings:
    per_gpu_expert_mapping: list[torch.Tensor]
    per_gpu_inverse_mapping: list[torch.Tensor]


def load_expert_mappings(
    *,
    config,
    expert_map_path: str,
    ep_rank: int,
    ep_size: int,
) -> ExpertMappings:
    num_experts = config.num_experts
    num_layers = config.num_hidden_layers

    if num_layers == 20:
        map_name = f"mini_expert_map_{ep_size}.pt"
    else:
        map_name = f"flash_expert_map_{ep_size}.pt"
    expert_map_file = Path(expert_map_path) / map_name

    if expert_map_file.exists():
        print("load expert_map from", expert_map_file)
        expert_map = torch.load(expert_map_file)
    else:
        expert_map = torch.zeros(num_experts, dtype=torch.int32)
        for e in range(num_experts):
            expert_map[e] = e // (num_experts // ep_size)
        expert_map = expert_map.unsqueeze(0).repeat(num_layers, 1)

    arange_experts = torch.arange(num_experts, dtype=torch.int64)
    per_gpu_expert_mapping = [
        arange_experts[expert_map[i] == ep_rank] for i in range(num_layers)
    ]
    per_gpu_inverse_mapping = [
        torch.ones(num_experts, dtype=torch.int64).mul(-1) for _ in range(num_layers)
    ]
    for layer_id in range(num_layers):
        per_gpu_inverse_mapping[layer_id][per_gpu_expert_mapping[layer_id]] = (
            torch.arange(per_gpu_expert_mapping[layer_id].shape[0])
        )

    return ExpertMappings(
        per_gpu_expert_mapping=per_gpu_expert_mapping,
        per_gpu_inverse_mapping=per_gpu_inverse_mapping,
    )


def tp_split(
    tensor: torch.Tensor,
    dim: int,
    rank: int,
    world: int,
    is_w13: bool = False,
) -> torch.Tensor:
    if world == 1:
        return tensor
    if is_w13:
        shard_size = tensor.size(dim) // 2
        size = shard_size // world
        w1 = tensor.narrow(dim, rank * size, size)
        w3 = tensor.narrow(dim, shard_size + rank * size, size)
        return torch.cat([w1, w3], dim=dim)
    size = tensor.size(dim) // world
    return tensor.narrow(dim, rank * size, size)

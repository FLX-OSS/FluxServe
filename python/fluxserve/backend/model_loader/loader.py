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
    Model loader.
"""


import json
import logging
import re
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn

from fluxserve.backend.distributed import (
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
    get_moe_tensor_parallel_rank,
    get_moe_tensor_parallel_world_size,
)
from fluxserve.backend.model_loader.weight_utils import (
    get_model_name,
    get_safetensors_shard_files,
    iter_safetensors_shards,
    load_expert_mappings,
    resolve_model_snapshot,
    tp_split,
)
from fluxserve.backend.models import (
    DiffusionGemmaForConditionalGeneration,
    LLaDA2LLM,
)
from fluxserve.backend.models.nemotron_diffusion import (
    NemotronLabsDiffusionLLM,
    is_nemotron_diffusion_config,
    nemotron_head_dim,
    nemotron_weight_plan,
)
from fluxserve.backend.utils.runtime_utils import tqdm_progress as tqdm

logger = logging.getLogger(__name__)


class DefaultModelLoader:
    def load_model(
        self,
        *,
        model_config,
        device: str,
        quant_config=None,
    ) -> nn.Module:
        model = LLaDA2LLM(
            config=model_config,
            quant_config=quant_config,
        ).eval()
        self.load_weights(model, dtype=torch.bfloat16, device=device)
        model.init_h2e_module()
        model = model.to(device)
        self.process_weights_after_loading(model)
        return model.eval()

    def load_weights(
        self,
        model: LLaDA2LLM,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        model_dir = resolve_model_snapshot(model.config)
        mappings = load_expert_mappings(
            config=model.config,
            expert_map_path=model.expert_map_path,
            ep_rank=get_moe_expert_parallel_rank(),
            ep_size=get_moe_expert_parallel_world_size(),
        )
        state_dict = self._read_state_dict_for_current_rank(
            model,
            model_dir,
            mappings,
        )

        if model.quant_config is not None:
            self._update_state_dict_for_fusemoe_quant(
                model,
                state_dict,
                model.config.num_hidden_layers,
                dtype,
                mappings.per_gpu_expert_mapping,
                mappings.per_gpu_inverse_mapping,
                device,
            )
        else:
            self._update_state_dict_for_fusemoe(
                model,
                state_dict,
                model.config.num_hidden_layers,
                dtype,
                mappings.per_gpu_expert_mapping,
                mappings.per_gpu_inverse_mapping,
                device,
            )

    def _read_state_dict_for_current_rank(self, model: LLaDA2LLM, model_dir, mappings):
        shard_files = get_safetensors_shard_files(model_dir)
        local_experts_by_layer = [
            set(expert_ids.tolist()) for expert_ids in mappings.per_gpu_expert_mapping
        ]

        state_dict = {}
        for _, file_state_dict in tqdm.tqdm(
            iter_safetensors_shards(model_dir, shard_files),
            total=len(shard_files),
        ):
            filtered_file_state_dict = {}
            for key, value in file_state_dict.items():
                if ".mlp.experts." in key:
                    layer_id = int(key.split(".mlp.experts.")[0].split(".")[-1])
                    expert_id = int(key.split(".mlp.experts.")[1].split(".")[0])
                    if expert_id in local_experts_by_layer[layer_id]:
                        filtered_file_state_dict[key] = value
                else:
                    filtered_file_state_dict[key] = value

            state_dict.update(filtered_file_state_dict)
        return state_dict

    @staticmethod
    def process_weights_after_loading(model: LLaDA2LLM) -> None:
        if model.quant_config is None:
            return
        for name, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if (
                quant_method is not None
                and hasattr(quant_method, "process_weights_after_loading")
            ):
                if hasattr(module, "weight_scale") and module.weight_scale is not None:
                    if module.weight_scale.dim() == 0:
                        print(f"Fixing scalar weight_scale for {name}")
                        module.weight_scale.data = module.weight_scale.data.unsqueeze(0)
                if hasattr(module, "input_scale") and module.input_scale is not None:
                    if module.input_scale.dim() == 0:
                        print(f"Fixing scalar input_scale for {name}")
                        module.input_scale.data = module.input_scale.data.unsqueeze_(0)
                quant_method.process_weights_after_loading(module)

    def _update_state_dict_for_fusemoe_quant(
        self,
        model: LLaDA2LLM,
        state_dict,
        num_layers,
        dtype,
        per_gpu_expert_mapping,
        per_gpu_inverse_mapping,
        device,
    ):
        new_state_dict = {}
        gate_projs = [{} for _ in range(num_layers)]
        gate_input_scales = [{} for _ in range(num_layers)]
        gate_weight_scales = [{} for _ in range(num_layers)]
        up_projs = [{} for _ in range(num_layers)]
        up_weight_scales = [{} for _ in range(num_layers)]
        down_projs = [{} for _ in range(num_layers)]
        down_input_scales = [{} for _ in range(num_layers)]
        down_weight_scales = [{} for _ in range(num_layers)]
        moe_tp_rank = get_moe_tensor_parallel_rank()
        moe_tp_size = get_moe_tensor_parallel_world_size()
        for key, value in tqdm.tqdm(state_dict.items()):
            if ".mlp.experts." in key:
                layer_id = int(key.split(".mlp.experts.")[0].split(".")[-1])
                expert_id = int(key.split(".mlp.experts.")[1].split(".")[0])
                if layer_id < num_layers:
                    if re.search(r"experts\.\d{1,4}\.gate_proj\.input_scale", key):
                        gate_input_scales[layer_id][expert_id] = value
                    elif re.search(r"experts\.\d{1,4}\.gate_proj\.weight_scale", key):
                        gate_weight_scales[layer_id][expert_id] = value
                    elif re.search(r"experts\.\d{1,4}\.up_proj\.weight_scale", key):
                        up_weight_scales[layer_id][expert_id] = value
                    elif re.search(r"experts\.\d{1,4}\.down_proj\.input_scale", key):
                        down_input_scales[layer_id][expert_id] = value
                    elif re.search(r"experts\.\d{1,4}\.down_proj\.weight_scale", key):
                        down_weight_scales[layer_id][expert_id] = value
                    elif re.search(r"experts\.\d{1,4}\.gate_proj\.weight$", key):
                        gate_projs[layer_id][expert_id] = value
                    elif re.search(r"experts\.\d{1,4}\.up_proj\.weight$", key):
                        up_projs[layer_id][expert_id] = value
                    elif re.search(r"experts\.\d{1,4}\.down_proj\.weight$", key):
                        down_projs[layer_id][expert_id] = value
            else:
                new_state_dict[key] = value

        for layer_id in tqdm.trange(num_layers):
            if f"model.layers.{layer_id}.mlp.w1" in state_dict:
                new_state_dict[f"model.layers.{layer_id}.mlp.experts.w13_weight"] = (
                    tp_split(
                        state_dict[f"model.layers.{layer_id}.mlp.w1"][
                            per_gpu_expert_mapping[layer_id]
                        ],
                        dim=1,
                        rank=moe_tp_rank,
                        world=moe_tp_size,
                        is_w13=True,
                    ).contiguous()
                )
                new_state_dict[f"model.layers.{layer_id}.mlp.experts.w2_weight"] = (
                    tp_split(
                        state_dict[f"model.layers.{layer_id}.mlp.w2"][
                            per_gpu_expert_mapping[layer_id]
                        ],
                        dim=2,
                        rank=moe_tp_rank,
                        world=moe_tp_size,
                    ).contiguous()
                )
                del new_state_dict[f"model.layers.{layer_id}.mlp.w1"]
                del new_state_dict[f"model.layers.{layer_id}.mlp.w2"]
                model.model.layers[layer_id].mlp.experts.expert_map_cpu = (
                    per_gpu_inverse_mapping[layer_id]
                )

            if len(gate_projs[layer_id]) > 0:
                w13_weight = []
                w2_weight = []
                w13_input_scale = []
                w13_weight_scale = []
                w2_input_scale = []
                w2_weight_scale = []
                for expert_id in per_gpu_expert_mapping[layer_id]:
                    expert_id = int(expert_id)
                    gate_proj = gate_projs[layer_id][expert_id].to(device)
                    up_proj = up_projs[layer_id][expert_id].to(device)
                    down_proj = down_projs[layer_id][expert_id].to(device)
                    gate_weight_scale = gate_weight_scales[layer_id][expert_id].to(device)
                    up_weight_scale = up_weight_scales[layer_id][expert_id].to(device)
                    down_weight_scale = down_weight_scales[layer_id][expert_id].to(device)
                    gate_input_scale = gate_input_scales[layer_id][expert_id].to(device)
                    down_input_scale = down_input_scales[layer_id][expert_id].to(device)

                    w13_weight.append(torch.cat([gate_proj, up_proj], dim=0))
                    w2_weight.append(down_proj)
                    w13_input_scale.append(gate_input_scale)
                    w13_weight_scale.append(
                        torch.stack([gate_weight_scale, up_weight_scale], dim=0)
                    )
                    w2_input_scale.append(down_input_scale)
                    w2_weight_scale.append(down_weight_scale)

                new_state_dict[f"model.layers.{layer_id}.mlp.experts.w13_weight"] = (
                    tp_split(
                        torch.stack(w13_weight, dim=0),
                        dim=1,
                        rank=moe_tp_rank,
                        world=moe_tp_size,
                        is_w13=True,
                    )
                    .contiguous()
                )
                new_state_dict[f"model.layers.{layer_id}.mlp.experts.w2_weight"] = (
                    tp_split(
                        torch.stack(w2_weight, dim=0),
                        dim=2,
                        rank=moe_tp_rank,
                        world=moe_tp_size,
                    )
                    .contiguous()
                )
                new_state_dict[f"model.layers.{layer_id}.mlp.experts.w13_input_scale"] = (
                    torch.stack(w13_input_scale, dim=0).contiguous()
                )
                new_state_dict[f"model.layers.{layer_id}.mlp.experts.w13_weight_scale"] = (
                    torch.stack(w13_weight_scale, dim=0).contiguous()
                )
                new_state_dict[f"model.layers.{layer_id}.mlp.experts.w2_input_scale"] = (
                    torch.stack(w2_input_scale, dim=0).contiguous()
                )
                new_state_dict[f"model.layers.{layer_id}.mlp.experts.w2_weight_scale"] = (
                    torch.stack(w2_weight_scale, dim=0).contiguous()
                )
                model.model.layers[layer_id].mlp.experts.expert_map_cpu = (
                    per_gpu_inverse_mapping[layer_id]
                )

            self._transform_common_layer_weights(model, state_dict, new_state_dict, layer_id)

        new_state_dict["model.full_word_embeddings.weight"] = state_dict[
            "model.word_embeddings.weight"
        ]
        for key, value in tqdm.tqdm(new_state_dict.items()):
            new_state_dict[key] = value.to(device)
        model.apply_state_dicts(new_state_dict)

        for name, param in model.named_parameters():
            if (
                "norm" in name
                or "embed_tokens" in name
                or "word_embeddings" in name
                or "lm_head" in name
            ):
                param.data = param.data.to(dtype)
            elif ".mlp.correction_bias" in name:
                param.data = param.data.to(torch.float32)

        for name, buf in model.named_buffers():
            if "scale" in name:
                continue
            if "cos_sin_cache" in name:
                continue
            if buf.dtype != dtype:
                buf.data = buf.data.to(dtype)

    def _update_state_dict_for_fusemoe(
        self,
        model: LLaDA2LLM,
        state_dict,
        num_layers,
        dtype,
        per_gpu_expert_mapping,
        per_gpu_inverse_mapping,
        device,
    ):
        new_state_dict = {}
        gate_projs = [{} for _ in range(num_layers)]
        up_projs = [{} for _ in range(num_layers)]
        down_projs = [{} for _ in range(num_layers)]

        moe_tp_rank = get_moe_tensor_parallel_rank()
        moe_tp_size = get_moe_tensor_parallel_world_size()
        for key, value in tqdm.tqdm(state_dict.items()):
            if ".mlp.experts." in key:
                layer_id = int(key.split(".mlp.experts.")[0].split(".")[-1])
                expert_id = int(key.split(".mlp.experts.")[1].split(".")[0])

                if layer_id < num_layers:
                    if "gate_proj" in key:
                        gate_projs[layer_id][expert_id] = value
                    elif "up_proj" in key:
                        up_projs[layer_id][expert_id] = value
                    elif "down_proj" in key:
                        down_projs[layer_id][expert_id] = value
            else:
                new_state_dict[key] = value

        for layer_id in tqdm.trange(num_layers):
            if f"model.layers.{layer_id}.mlp.w1" in state_dict:
                new_state_dict[f"model.layers.{layer_id}.mlp.experts.w13_weight"] = (
                    tp_split(
                        state_dict[f"model.layers.{layer_id}.mlp.w1"][
                            per_gpu_expert_mapping[layer_id]
                        ],
                        dim=1,
                        rank=moe_tp_rank,
                        world=moe_tp_size,
                        is_w13=True,
                    ).contiguous()
                )
                new_state_dict[f"model.layers.{layer_id}.mlp.experts.w2_weight"] = (
                    tp_split(
                        state_dict[f"model.layers.{layer_id}.mlp.w2"][
                            per_gpu_expert_mapping[layer_id]
                        ],
                        dim=2,
                        rank=moe_tp_rank,
                        world=moe_tp_size,
                    ).contiguous()
                )
                del new_state_dict[f"model.layers.{layer_id}.mlp.w1"]
                del new_state_dict[f"model.layers.{layer_id}.mlp.w2"]
                model.model.layers[layer_id].mlp.experts.expert_map_cpu = (
                    per_gpu_inverse_mapping[layer_id]
                )

            if len(gate_projs[layer_id]) > 0:
                w13_weight = []
                w2_weight = []
                for expert_id in per_gpu_expert_mapping[layer_id]:
                    expert_id = int(expert_id)
                    gate_proj = gate_projs[layer_id][expert_id].to(device)
                    up_proj = up_projs[layer_id][expert_id].to(device)
                    down_proj = down_projs[layer_id][expert_id].to(device)
                    w13_weight.append(torch.cat([gate_proj, up_proj], dim=0))
                    w2_weight.append(down_proj)
                new_state_dict[f"model.layers.{layer_id}.mlp.experts.w13_weight"] = (
                    tp_split(
                        torch.stack(w13_weight, dim=0),
                        dim=1,
                        rank=moe_tp_rank,
                        world=moe_tp_size,
                        is_w13=True,
                    )
                    .contiguous()
                )
                new_state_dict[f"model.layers.{layer_id}.mlp.experts.w2_weight"] = (
                    tp_split(
                        torch.stack(w2_weight, dim=0),
                        dim=2,
                        rank=moe_tp_rank,
                        world=moe_tp_size,
                    )
                    .contiguous()
                )
                model.model.layers[layer_id].mlp.experts.expert_map_cpu = (
                    per_gpu_inverse_mapping[layer_id]
                )

            self._transform_common_layer_weights(model, state_dict, new_state_dict, layer_id)

        new_state_dict["model.full_word_embeddings.weight"] = state_dict[
            "model.word_embeddings.weight"
        ]
        for key, value in tqdm.tqdm(new_state_dict.items()):
            new_state_dict[key] = value.to(device)
        model.apply_state_dicts(new_state_dict)
        for name, param in model.named_parameters():
            if ".mlp.correction_bias" in name or "layernorm.weight" in name:
                param.data = param.data.to(torch.float32)
            else:
                param.data = param.data.to(dtype)

    @staticmethod
    def _transform_common_layer_weights(
        model: LLaDA2LLM,
        state_dict,
        new_state_dict,
        layer_id: int,
    ) -> None:
        if f"model.layers.{layer_id}.mlp.gate.expert_bias" in state_dict:
            new_state_dict[f"model.layers.{layer_id}.mlp.correction_bias"] = state_dict[
                f"model.layers.{layer_id}.mlp.gate.expert_bias"
            ]
            del new_state_dict[f"model.layers.{layer_id}.mlp.gate.expert_bias"]

        if f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight" in state_dict:
            shared_tp_size = (
                model.model.layers[layer_id].mlp.shared_experts.gate_up_proj.tp_size
            )
            shared_tp_rank = (
                model.model.layers[layer_id].mlp.shared_experts.gate_up_proj.tp_rank
            )
            part_size = (
                state_dict[
                    f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight"
                ].shape[0]
                // shared_tp_size
            )
            part_start = shared_tp_rank * part_size
            part_end = part_start + part_size
            new_state_dict[
                f"model.layers.{layer_id}.mlp.shared_experts.gate_up_proj.weight"
            ] = torch.cat(
                [
                    state_dict[
                        f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight"
                    ][part_start:part_end],
                    state_dict[
                        f"model.layers.{layer_id}.mlp.shared_experts.up_proj.weight"
                    ][part_start:part_end],
                ],
                dim=0,
            )
            if (
                f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight_scale"
                in state_dict
            ):
                new_state_dict[
                    f"model.layers.{layer_id}.mlp.shared_experts.gate_up_proj.weight_scale"
                ] = torch.stack(
                    [
                        state_dict[
                            f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight_scale"
                        ],
                        state_dict[
                            f"model.layers.{layer_id}.mlp.shared_experts.up_proj.weight_scale"
                        ],
                    ],
                    dim=0,
                )
                new_state_dict[
                    f"model.layers.{layer_id}.mlp.shared_experts.gate_up_proj.input_scale"
                ] = torch.stack(
                    [
                        state_dict[
                            f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.input_scale"
                        ],
                        state_dict[
                            f"model.layers.{layer_id}.mlp.shared_experts.up_proj.input_scale"
                        ],
                    ],
                    dim=0,
                )
            for suffix in (
                "gate_proj.weight",
                "up_proj.weight",
                "gate_proj.weight_scale",
                "up_proj.weight_scale",
                "gate_proj.input_scale",
                "up_proj.input_scale",
            ):
                new_state_dict.pop(
                    f"model.layers.{layer_id}.mlp.shared_experts.{suffix}", None
                )

        if f"model.layers.{layer_id}.mlp.gate_proj.weight" in state_dict:
            mlp_tp_size = model.model.layers[layer_id].mlp.gate_up_proj.tp_size
            mlp_tp_rank = model.model.layers[layer_id].mlp.gate_up_proj.tp_rank
            part_size = (
                state_dict[f"model.layers.{layer_id}.mlp.gate_proj.weight"].shape[0]
                // mlp_tp_size
            )
            part_start = mlp_tp_rank * part_size
            part_end = part_start + part_size
            new_state_dict[f"model.layers.{layer_id}.mlp.gate_up_proj.weight"] = (
                torch.cat(
                    [
                        state_dict[f"model.layers.{layer_id}.mlp.gate_proj.weight"][
                            part_start:part_end
                        ],
                        state_dict[f"model.layers.{layer_id}.mlp.up_proj.weight"][
                            part_start:part_end
                        ],
                    ],
                    dim=0,
                )
            )
            if f"model.layers.{layer_id}.mlp.gate_proj.weight_scale" in state_dict:
                new_state_dict[
                    f"model.layers.{layer_id}.mlp.gate_up_proj.weight_scale"
                ] = torch.stack(
                    [
                        state_dict[
                            f"model.layers.{layer_id}.mlp.gate_proj.weight_scale"
                        ],
                        state_dict[
                            f"model.layers.{layer_id}.mlp.up_proj.weight_scale"
                        ],
                    ],
                    dim=0,
                )
                new_state_dict[
                    f"model.layers.{layer_id}.mlp.gate_up_proj.input_scale"
                ] = torch.stack(
                    [
                        state_dict[
                            f"model.layers.{layer_id}.mlp.gate_proj.input_scale"
                        ],
                        state_dict[f"model.layers.{layer_id}.mlp.up_proj.input_scale"],
                    ],
                    dim=0,
                )
            for suffix in (
                "gate_proj.weight",
                "up_proj.weight",
                "gate_proj.weight_scale",
                "up_proj.weight_scale",
                "gate_proj.input_scale",
                "up_proj.input_scale",
            ):
                new_state_dict.pop(f"model.layers.{layer_id}.mlp.{suffix}", None)


class DiffusionGemmaModelLoader:
    """Streaming loader for the official unquantized Diffusion-Gemma layout."""

    def load_model(self, *, model_config, device: str, quant_config=None) -> nn.Module:
        if quant_config is not None:
            raise ValueError("Diffusion-Gemma quantization is not supported yet")
        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device(device):
                model = DiffusionGemmaForConditionalGeneration(model_config).eval()
        finally:
            torch.set_default_dtype(old_dtype)

        model_dir = resolve_model_snapshot(model_config)
        shard_files = get_safetensors_shard_files(model_dir)
        loaded: set[str] = set()
        unexpected: set[str] = set()
        for _, shard in tqdm.tqdm(
            iter_safetensors_shards(model_dir, shard_files), total=len(shard_files)
        ):
            shard_loaded, shard_unexpected = model.load_weights(shard.items())
            loaded.update(shard_loaded)
            unexpected.update(shard_unexpected)
            del shard

        required = {
            name
            for name, _ in model.named_parameters()
            if not (name == "lm_head.weight" and model.text_config.tie_word_embeddings)
        }
        missing = sorted(required - loaded)
        if missing:
            preview = ", ".join(missing[:20])
            raise ValueError(
                f"Diffusion-Gemma checkpoint is missing {len(missing)} required "
                f"text weights: {preview}"
            )
        if unexpected:
            preview = ", ".join(sorted(unexpected)[:20])
            raise ValueError(
                f"Diffusion-Gemma checkpoint contains unmapped text weights: {preview}"
            )
        return model.eval()


# Nemotron-Labs-Diffusion ships one unsharded ``model.safetensors`` next to a
# draft LoRA adapter that must not be loaded as a base weight, so it resolves
# and iterates its checkpoint here instead of through the sharded helpers.
NEMOTRON_WEIGHT_FILE = "model.safetensors"
NEMOTRON_GENERATION_CONFIG_FILE = "generation_config.json"


def resolve_nemotron_snapshot(model_config, *, local_files_only: bool = False) -> Path:
    """Locate the snapshot directory holding a single-file checkpoint.

    ``local_files_only`` makes resolution a cache lookup that raises rather
    than fetching. Tests use it: this checkpoint is 27 GB, and a helper that
    silently downloads it would turn a unit test into a long transfer on any
    machine whose cache happens to be cold.
    """
    model_name = get_model_name(model_config)
    local = Path(model_name).expanduser()
    if (local / NEMOTRON_WEIGHT_FILE).is_file():
        return local
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "huggingface_hub is required to resolve Nemotron model weights"
        ) from exc
    # Deliberately narrow: a bare "*.safetensors" pattern would also fetch
    # linear_spec_lora/adapter_model.safetensors, which is not a base weight.
    snapshot = Path(
        snapshot_download(
            repo_id=model_name,
            revision=getattr(model_config, "_commit_hash", None),
            allow_patterns=[NEMOTRON_WEIGHT_FILE, "*.json", "*.jinja"],
            local_files_only=local_files_only,
        )
    )
    if not (snapshot / NEMOTRON_WEIGHT_FILE).is_file():
        raise FileNotFoundError(
            f"Nemotron checkpoint '{model_name}' does not provide {NEMOTRON_WEIGHT_FILE}"
        )
    return snapshot


def read_safetensors_header(path: Path | str) -> dict[str, dict]:
    """Read a safetensors header without mapping or loading any tensor data."""
    with open(path, "rb") as handle:
        length = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header


def iter_nemotron_tensors(
    weight_file: Path | str,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield one checkpoint tensor at a time.

    Reading the whole 27 GB file at once would need it resident in host memory
    before any GPU copy; ``safe_open`` keeps that bounded to one tensor.
    """
    from safetensors import safe_open

    with safe_open(str(weight_file), framework="pt") as handle:
        for name in handle.keys():
            yield name, handle.get_tensor(name)


def resolve_nemotron_eos_ids(model_config) -> tuple[int, ...]:
    """Collect EOS ids from ``generation_config.json``, falling back to config."""
    ids: list[int] = []
    try:
        # Cache-only: the caller is already loading this checkpoint, or is a
        # test. Neither wants an incidental download from an id lookup.
        snapshot = resolve_nemotron_snapshot(model_config, local_files_only=True)
        path = snapshot / NEMOTRON_GENERATION_CONFIG_FILE
        if path.is_file():
            with open(path, "r") as handle:
                raw = json.load(handle).get("eos_token_id")
            if isinstance(raw, int):
                ids = [raw]
            elif isinstance(raw, (list, tuple)):
                ids = [int(value) for value in raw]
    except (OSError, ValueError, RuntimeError):
        logger.debug("Nemotron generation_config.json unavailable", exc_info=True)
    if not ids:
        raw = getattr(model_config, "eos_token_id", None)
        if isinstance(raw, int):
            ids = [raw]
        elif isinstance(raw, (list, tuple)):
            ids = [int(value) for value in raw]
    deduplicated: list[int] = []
    for value in ids:
        if value not in deduplicated:
            deduplicated.append(int(value))
    if not deduplicated:
        raise ValueError("Nemotron checkpoint declares no EOS token id")
    return tuple(deduplicated)


def nemotron_decoding_ids(model_config) -> dict[str, object]:
    """Mask and EOS ids for this checkpoint.

    ``RunnerConfig`` defaults to LLaDA's ids, which are outside this model's
    131072-entry vocabulary; both the primary id and the tuple are set so the
    shared decoder factory keeps its existing ordering rule unchanged.
    """
    mask_id = getattr(model_config, "mask_token_id", None)
    vocab_size = int(model_config.vocab_size)
    if mask_id is None or not 0 <= int(mask_id) < vocab_size:
        raise ValueError(
            "Nemotron checkpoint must declare mask_token_id inside its "
            f"vocabulary, got {mask_id!r} for vocab_size={vocab_size}"
        )
    eos_ids = resolve_nemotron_eos_ids(model_config)
    for value in eos_ids:
        if not 0 <= value < vocab_size:
            raise ValueError(
                f"Nemotron EOS id {value} is outside vocab_size={vocab_size}"
            )
    return {
        "mask_id": int(mask_id),
        "eos_id": int(eos_ids[0]),
        "eos_ids": eos_ids,
    }


NEMOTRON_END_THINK_TOKEN = "</think>"
NEMOTRON_TOKENIZER_CONFIG_FILE = "tokenizer_config.json"


def resolve_end_think_token_id(model_config) -> int | None:
    """The checkpoint's ``</think>`` id, or ``None`` when it declares none."""
    try:
        snapshot = resolve_nemotron_snapshot(model_config, local_files_only=True)
        path = snapshot / NEMOTRON_TOKENIZER_CONFIG_FILE
        if not path.is_file():
            return None
        with open(path) as handle:
            added = json.load(handle).get("added_tokens_decoder") or {}
    except (OSError, ValueError, RuntimeError):
        logger.debug("Nemotron tokenizer_config.json unavailable", exc_info=True)
        return None
    for token_id, entry in added.items():
        if isinstance(entry, dict) and entry.get("content") == NEMOTRON_END_THINK_TOKEN:
            return int(token_id)
    return None


NEMOTRON_LORA_SUBFOLDER = "linear_spec_lora"
NEMOTRON_LORA_WEIGHT_FILE = "adapter_model.safetensors"
NEMOTRON_LORA_CONFIG_FILE = "adapter_config.json"


class NemotronDraftAdapter:
    """The ``linear_spec_lora`` draft adapter, pre-fused into weight pairs.

    The adapter targets ``o_proj`` only and is toggled on for the drafting
    forward and off for verification, within a single iteration. Evaluating a
    low-rank update on every forward would put two extra matmuls per layer on
    the critical path for an adapter that never changes, so both weights are
    materialised once and the module's parameter is pointed at one of them.

    The cost is memory: one extra ``o_proj`` per layer, about 1.7 GB in
    bfloat16 for this checkpoint. The benefit is that toggling is a pointer
    swap and the drafting forward runs at exactly base-model speed.
    """

    def __init__(self, layers):
        self.layers = list(layers)
        self.enabled = False

    def apply(self, enabled: bool) -> None:
        if enabled == self.enabled:
            return
        for module, base, draft in self.layers:
            module.weight.data = draft if enabled else base
        self.enabled = bool(enabled)


def load_nemotron_lora(model, model_config, path=None):
    """Fuse the draft adapter into per-layer weight pairs, or return ``None``.

    Returns ``None`` when the checkpoint ships no adapter: self-speculation
    works without it at a lower acceptance length, so a missing adapter is a
    configuration, not an error.
    """
    from safetensors import safe_open

    if path is None:
        try:
            snapshot = resolve_nemotron_snapshot(model_config, local_files_only=True)
        except (OSError, ValueError, RuntimeError):
            logger.debug("Nemotron snapshot unavailable for LoRA", exc_info=True)
            return None
        path = snapshot / NEMOTRON_LORA_SUBFOLDER / NEMOTRON_LORA_WEIGHT_FILE
        config_path = snapshot / NEMOTRON_LORA_SUBFOLDER / NEMOTRON_LORA_CONFIG_FILE
    else:
        path = Path(path)
        config_path = path.parent / NEMOTRON_LORA_CONFIG_FILE
    if not Path(path).is_file():
        return None

    rank, alpha, targets = 0, 0.0, ("o_proj",)
    if Path(config_path).is_file():
        with open(config_path) as handle:
            raw = json.load(handle)
        rank = int(raw.get("r", 0) or 0)
        alpha = float(raw.get("lora_alpha", 0.0) or 0.0)
        targets = tuple(raw.get("target_modules") or targets)
    if set(targets) != {"o_proj"}:
        raise ValueError(
            "Nemotron draft adapter support covers o_proj only; this adapter "
            f"targets {sorted(targets)}"
        )

    layers = []
    with safe_open(str(path), framework="pt") as handle:
        names = set(handle.keys())
        for index, block in enumerate(model.model.layers):
            prefix = f"base_model.model.encoder.layers.{index}.self_attn.o_proj"
            a_name, b_name = f"{prefix}.lora_A.weight", f"{prefix}.lora_B.weight"
            if a_name not in names or b_name not in names:
                raise ValueError(
                    f"Nemotron draft adapter is missing {a_name} / {b_name}"
                )
            lora_a = handle.get_tensor(a_name)
            lora_b = handle.get_tensor(b_name)
            if not rank:
                rank = int(lora_a.shape[0])
            if not alpha:
                alpha = float(rank)
            module = block.self_attn.o_proj
            base = module.weight.data
            # o_proj is row-parallel: its input dimension is sharded, so the
            # adapter's A matrix must be sliced the same way before fusing.
            shard_width = int(base.shape[1])
            if shard_width > lora_a.shape[1]:
                raise ValueError(
                    f"layer {index}: o_proj shard {shard_width} exceeds the "
                    f"adapter's input width {int(lora_a.shape[1])}"
                )
            shard_index = int(lora_a.shape[1]) // shard_width
            rank_id = 0 if shard_index <= 1 else _attention_tp_rank()
            start = rank_id * shard_width
            a_shard = lora_a[:, start : start + shard_width]
            delta = (alpha / rank) * (
                lora_b.to(torch.float32) @ a_shard.to(torch.float32)
            )
            draft = (base.to(torch.float32) + delta.to(base.device)).to(base.dtype)
            layers.append((module, base, draft))
    return NemotronDraftAdapter(layers)


def _attention_tp_rank() -> int:
    from fluxserve.backend.layers.dp_attention import get_attention_tp_rank

    try:
        return int(get_attention_tp_rank())
    except AssertionError:
        return 0


class NemotronModelLoader:
    """Load the single-file Nemotron checkpoint into the dense model."""

    def load_model(self, *, model_config, device: str, quant_config=None) -> nn.Module:
        if quant_config is not None:
            raise ValueError(
                "Nemotron-Labs-Diffusion quantization is not supported yet"
            )
        if not is_nemotron_diffusion_config(model_config):
            raise ValueError(
                "NemotronModelLoader received a non-Nemotron config: "
                f"architectures={getattr(model_config, 'architectures', None)!r}"
            )
        old_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device(device):
                model = NemotronLabsDiffusionLLM(model_config).eval()
        finally:
            torch.set_default_dtype(old_dtype)

        snapshot = resolve_nemotron_snapshot(model_config)
        weight_file = snapshot / NEMOTRON_WEIGHT_FILE
        plan = nemotron_weight_plan(model_config)

        with torch.inference_mode():
            consumed, unexpected = model.load_weights(
                iter_nemotron_tensors(weight_file)
            )

        if unexpected:
            preview = ", ".join(sorted(unexpected)[:20])
            raise ValueError(
                f"Nemotron checkpoint contains {len(unexpected)} unmapped "
                f"tensors: {preview}"
            )
        missing = sorted(set(plan) - consumed)
        if missing:
            preview = ", ".join(missing[:20])
            raise ValueError(
                f"Nemotron checkpoint is missing {len(missing)} required "
                f"tensors: {preview}"
            )
        logger.info(
            "Nemotron-Labs-Diffusion loaded: %d checkpoint tensors, "
            "%d layers, head_dim=%d, 0 unmatched keys",
            len(consumed),
            int(model_config.num_hidden_layers),
            nemotron_head_dim(model_config),
        )
        return model.eval()

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

import re

import torch
import torch.nn as nn

from fluxserve.backend.distributed import (
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
    get_moe_tensor_parallel_rank,
    get_moe_tensor_parallel_world_size,
)
from fluxserve.backend.model_loader.weight_utils import (
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
from fluxserve.backend.utils.runtime_utils import tqdm_progress as tqdm


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

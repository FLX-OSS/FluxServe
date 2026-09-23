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

"""Nemotron-Labs-Diffusion dense decoder.

``nvidia/Nemotron-Labs-Diffusion-14B`` is a dense Ministral-shaped decoder used
as a tri-mode language model: the same weights serve autoregressive, block
diffusion and self-speculative decoding, selected purely by the attention
pattern a forward runs under. This module owns only the architecture; causality
belongs to the caller, which supplies either a dense ``attention_mask`` or paged
attention metadata.

See ``docs/serving/nemotron/nemotron-labs-diffusion-14B.md``.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

import fluxserve.backend.distributed as flux_distributed
from fluxserve.backend.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from fluxserve.backend.execution.forward_batch_info import ForwardBatch, PPProxyTensors
from fluxserve.backend.layers.activation import SiluAndMul
from fluxserve.backend.layers.attention import AttentionForward, AttentionForwardConfig
from fluxserve.backend.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
)
from fluxserve.backend.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from fluxserve.backend.layers.norm import RMSNorm
from fluxserve.backend.layers.rotary_embedding import get_rope
from fluxserve.backend.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from fluxserve.backend.utils.runtime_utils import add_prefix, make_layers

NEMOTRON_ARCHITECTURE = "NemotronLabsDiffusionModel"
NEMOTRON_MODEL_TYPE = "nemotron_labs_diffusion"

# Below this position the query temperature scale is exactly 1.0, so serving
# stays bit-identical without it.
DEFAULT_QUERY_SCALE_MAX_POSITION = 16384


def is_nemotron_diffusion_config(model_config) -> bool:
    """Recognize a Nemotron-Labs-Diffusion checkpoint from its config alone."""
    architectures = set(getattr(model_config, "architectures", ()) or ())
    return (
        NEMOTRON_ARCHITECTURE in architectures
        or getattr(model_config, "model_type", None) == NEMOTRON_MODEL_TYPE
    )


def nemotron_rope_parameters(config) -> tuple[float, dict | None]:
    """Resolve ``(base, rope_scaling)`` for ``get_rope``.

    The checkpoint keeps YaRN settings in ``rope_parameters`` and mirrors them
    onto ``rope_scaling``; ``rope_theta`` lives inside that dict rather than at
    the top level.
    """
    params = getattr(config, "rope_parameters", None)
    if not isinstance(params, dict):
        params = getattr(config, "rope_scaling", None)
    base = float(getattr(config, "rope_theta", 10000.0) or 10000.0)
    if isinstance(params, dict):
        base = float(params.get("rope_theta", base))
    return base, params if isinstance(params, dict) else None


def nemotron_query_scale(
    positions: torch.Tensor,
    beta: float | None,
    original_max_position: int | None,
) -> torch.Tensor | None:
    """Llama-4 style query temperature scale, or ``None`` when disabled.

    Mirrors the checkpoint's ``_get_llama_4_attn_scale``:
    ``1 + beta * log(1 + floor(pos / original_max_position))``. The value is
    exactly ``1.0`` for every position below ``original_max_position``, so this
    is an identity within the supported context window and only becomes
    load-bearing for long context.
    """
    if not beta or not original_max_position:
        return None
    scale = 1.0 + float(beta) * torch.log1p(
        torch.floor(positions.float() / float(original_max_position))
    )
    return scale.unsqueeze(-1)


def nemotron_head_dim(config) -> int:
    """``head_dim`` is 128 while ``hidden_size // num_heads`` is 160 here."""
    head_dim = getattr(config, "head_dim", None)
    if head_dim:
        return int(head_dim)
    return int(config.hidden_size) // int(config.num_attention_heads)


def nemotron_weight_plan(config) -> dict[str, tuple[str, object, tuple[int, ...]]]:
    """Every checkpoint tensor this model expects, with destination and shape.

    Returns ``{source_name: (destination_param, shard_id, full_shape)}`` where
    ``shard_id`` is the fused-slice selector passed to a parameter's
    ``weight_loader`` (``None`` for unfused parameters). Shapes are the full
    unsharded checkpoint shapes, so this plan can be checked against a
    safetensors header without loading tensors or building the model.
    """
    hidden = int(config.hidden_size)
    vocab = int(config.vocab_size)
    intermediate = int(config.intermediate_size)
    head_dim = nemotron_head_dim(config)
    q_dim = int(config.num_attention_heads) * head_dim
    kv_dim = int(config.num_key_value_heads) * head_dim

    plan: dict[str, tuple[str, object, tuple[int, ...]]] = {
        "encoder.embed_tokens.weight": ("model.embed_tokens.weight", None, (vocab, hidden)),
        "encoder.norm.weight": ("model.norm.weight", None, (hidden,)),
        "diffusion_head.weight": ("lm_head.weight", None, (vocab, hidden)),
    }
    for layer_id in range(int(config.num_hidden_layers)):
        source = f"encoder.layers.{layer_id}."
        dest = f"model.layers.{layer_id}."
        plan[source + "input_layernorm.weight"] = (
            dest + "input_layernorm.weight", None, (hidden,)
        )
        plan[source + "post_attention_layernorm.weight"] = (
            dest + "post_attention_layernorm.weight", None, (hidden,)
        )
        qkv = dest + "self_attn.qkv_proj.weight"
        plan[source + "self_attn.q_proj.weight"] = (qkv, "q", (q_dim, hidden))
        plan[source + "self_attn.k_proj.weight"] = (qkv, "k", (kv_dim, hidden))
        plan[source + "self_attn.v_proj.weight"] = (qkv, "v", (kv_dim, hidden))
        plan[source + "self_attn.o_proj.weight"] = (
            dest + "self_attn.o_proj.weight", None, (hidden, q_dim)
        )
        gate_up = dest + "mlp.gate_up_proj.weight"
        plan[source + "mlp.gate_proj.weight"] = (gate_up, 0, (intermediate, hidden))
        plan[source + "mlp.up_proj.weight"] = (gate_up, 1, (intermediate, hidden))
        plan[source + "mlp.down_proj.weight"] = (
            dest + "mlp.down_proj.weight", None, (hidden, intermediate)
        )
    return plan


class NemotronDiffusionMLP(nn.Module):
    def __init__(self, config, prefix: str = ""):
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(
                "Nemotron-Labs-Diffusion expects hidden_act='silu', got "
                f"{config.hidden_act!r}"
            )
        intermediate_size = int(config.intermediate_size)
        bias = bool(getattr(config, "mlp_bias", False))
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=bias,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=bias,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class NemotronDiffusionAttention(nn.Module):
    """Dense GQA attention; the caller owns causality.

    There is no QK norm and no attention bias in this checkpoint. Unlike the
    LLaDA attention module, ``head_dim`` is read from the config because it does
    not equal ``hidden_size // num_attention_heads`` here.
    """

    def __init__(self, config, layer_id: int = 0, prefix: str = ""):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = int(config.hidden_size)
        self.head_dim = nemotron_head_dim(config)
        self.total_num_heads = int(config.num_attention_heads)
        self.total_num_kv_heads = int(config.num_key_value_heads)
        # Checkpoint loading and CPU tests construct the model outside a
        # distributed context; fall back to a single rank there.
        try:
            tp_rank = get_attention_tp_rank()
            tp_size = get_attention_tp_size()
        except AssertionError:
            tp_rank, tp_size = 0, 1
        if self.total_num_heads % tp_size:
            raise ValueError(
                f"num_attention_heads={self.total_num_heads} is not divisible by "
                f"attention tp_size={tp_size}"
            )
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim**-0.5

        bias = bool(getattr(config, "attention_bias", False))
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            prefix=add_prefix("qkv_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=bias,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

        base, rope_scaling = nemotron_rope_parameters(config)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=int(config.max_position_embeddings),
            base=base,
            rope_scaling=rope_scaling,
            dtype=torch.float32,
        )
        self.query_scale_beta = (
            float(rope_scaling.get("llama_4_scaling_beta", 0.0) or 0.0)
            if rope_scaling
            else 0.0
        )
        self.query_scale_max_position = (
            int(rope_scaling.get("original_max_position_embeddings", 0) or 0)
            if rope_scaling
            else 0
        )

        self.attention_forward = AttentionForward(
            AttentionForwardConfig(
                layer_id=layer_id,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                num_key_value_groups=self.num_key_value_groups,
                scale=self.scale,
            )
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values=None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
    ):
        bsz, q_len, _ = hidden_states.size()
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q, k = self.rotary_emb(
            positions.flatten(),
            q.flatten(0, 1),
            k.flatten(0, 1),
            fused_set_kv_buffer_arg=None,
        )
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Identity below `original_max_position_embeddings`; see guide 2.3.
        scale = nemotron_query_scale(
            positions.reshape(-1, q_len),
            self.query_scale_beta,
            self.query_scale_max_position,
        )
        if scale is not None:
            q = q * scale.unsqueeze(1).to(q.dtype)

        attn_output, present_key_values = self.attention_forward.forward(
            q,
            k,
            v,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attention_mask=attention_mask,
            forward_batch=forward_batch,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output, _ = self.o_proj(attn_output)
        return attn_output, present_key_values


class NemotronDiffusionDecoderLayer(nn.Module):
    def __init__(self, config, layer_id: int = 0, prefix: str = ""):
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = NemotronDiffusionAttention(
            config, layer_id, add_prefix("self_attn", prefix)
        )
        self.mlp = NemotronDiffusionMLP(config, add_prefix("mlp", prefix))
        eps = float(config.rms_norm_eps)
        self.input_layernorm = RMSNorm(config.hidden_size, eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values=None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_values = self.self_attn(
            positions,
            hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attention_mask=attention_mask,
            forward_batch=forward_batch,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, present_key_values


class NemotronDiffusionModel(nn.Module):
    def __init__(self, config, prefix: str = "model"):
        super().__init__()
        self.config = config
        self.pp_group = get_pp_group()
        if self.pp_group.world_size != 1:
            raise ValueError(
                "Nemotron-Labs-Diffusion pipeline parallelism is not supported yet"
            )
        self.embed_tokens = VocabParallelEmbedding(
            int(config.vocab_size),
            int(config.hidden_size),
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers, self.start_layer, self.end_layer = make_layers(
            int(config.num_hidden_layers),
            lambda idx, layer_prefix: NemotronDiffusionDecoderLayer(
                config, idx, layer_prefix
            ),
            prefix=add_prefix("layers", prefix),
        )
        self.norm = RMSNorm(int(config.hidden_size), float(config.rms_norm_eps))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        past_key_values=None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
    ):
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        all_present_key_values: list[torch.Tensor] = []
        for layer_id in range(self.start_layer, self.end_layer):
            layer = self.layers[layer_id]
            hidden_states, present_key_values = layer(
                positions,
                hidden_states,
                past_key_values=(
                    past_key_values[layer_id] if past_key_values is not None else None
                ),
                use_cache=use_cache,
                attention_mask=attention_mask,
                forward_batch=forward_batch,
            )
            if use_cache and present_key_values is not None:
                all_present_key_values.extend(present_key_values)
        return self.norm(hidden_states), all_present_key_values


class NemotronLabsDiffusionLLM(nn.Module):
    """Top-level module matching FluxServe's runner/loader contract.

    Checkpoint names are mapped by :meth:`checkpoint_name`: the transformer
    stack is ``encoder.*`` rather than ``model.*`` and the output projection is
    ``diffusion_head`` rather than ``lm_head``.
    """

    def __init__(self, config, quant_config=None, expert_map_path: str = ""):
        super().__init__()
        if quant_config is not None:
            raise ValueError(
                "Nemotron-Labs-Diffusion quantization is not supported yet"
            )
        del expert_map_path
        self.config = config
        self.quant_config = None
        self.model = NemotronDiffusionModel(config)
        self.lm_head = ParallelLMHead(
            int(config.vocab_size),
            int(config.hidden_size),
            bias=False,
            prefix="lm_head",
        )
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.tie_weights(self.model.embed_tokens)
        self._lm_head_sharded_to_full_mapping = None

    @staticmethod
    def checkpoint_name(name: str) -> str | None:
        """Map one checkpoint tensor name onto this module's namespace."""
        if name.startswith("encoder."):
            return "model." + name[len("encoder.") :]
        if name == "diffusion_head.weight":
            return "lm_head.weight"
        return name

    def weight_plan(self) -> dict[str, tuple[str, object, tuple[int, ...]]]:
        return nemotron_weight_plan(self.config)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> tuple[set[str], set[str]]:
        """Load checkpoint tensors, returning ``(consumed_sources, unexpected)``.

        Sources are reported rather than destinations: a fused destination such
        as ``qkv_proj.weight`` is written three times, so seeing it once would
        not prove every slice arrived. The caller compares the consumed source
        names against :func:`nemotron_weight_plan`.
        """
        params = dict(self.named_parameters())
        plan = self.weight_plan()
        consumed: set[str] = set()
        unexpected: set[str] = set()

        for raw_name, value in weights:
            entry = plan.get(raw_name)
            if entry is None:
                unexpected.add(raw_name)
                continue
            target, shard_id, expected_shape = entry
            if tuple(value.shape) != tuple(expected_shape):
                raise ValueError(
                    f"Nemotron checkpoint shape mismatch for {raw_name}: "
                    f"checkpoint={tuple(value.shape)} expected={tuple(expected_shape)}"
                )
            param = params.get(target)
            if param is None:
                raise ValueError(
                    f"Nemotron model has no parameter {target!r} for checkpoint "
                    f"tensor {raw_name!r}"
                )
            value = value.to(device=param.device)
            loader = getattr(param, "weight_loader", None)
            if loader is not None:
                if shard_id is None:
                    loader(param, value)
                else:
                    loader(param, value, shard_id)
            elif param.shape != value.shape:
                raise ValueError(
                    f"Nemotron weight shape mismatch for {target}: "
                    f"checkpoint={tuple(value.shape)} model={tuple(param.shape)}"
                )
            else:
                param.data.copy_(value.to(param.dtype))
            consumed.add(raw_name)
        return consumed, unexpected

    def _get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        local_logits = torch.matmul(
            hidden_states.to(self.lm_head.weight.dtype),
            self.lm_head.weight.T,
        )
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1:
            gathered = [torch.empty_like(local_logits) for _ in range(tp_size)]
            torch.distributed.all_gather(
                gathered,
                local_logits,
                group=flux_distributed.get_tensor_model_parallel_group(),
            )
            local_logits = torch.cat(gathered, dim=-1)
            mapping = self._lm_head_sharded_to_full_mapping
            if mapping is None or mapping.device != local_logits.device:
                sharded_to_full = self.lm_head.get_sharded_to_full_mapping()
                if sharded_to_full is not None:
                    mapping = torch.tensor(
                        sharded_to_full,
                        device=local_logits.device,
                        dtype=torch.long,
                    )
                    self._lm_head_sharded_to_full_mapping = mapping
            if mapping is not None:
                local_logits = local_logits.index_select(-1, mapping)
        # The checkpoint's diffusion_head keeps its projection dtype. Its
        # threshold rule computes softmax in that dtype, so an upcast here
        # changes which positions clear the threshold in bfloat16 serving.
        return local_logits[..., : int(self.config.vocab_size)]

    @torch.no_grad()
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        past_key_values=None,
        replace_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> MoeCausalLMOutputWithPast:
        del pp_proxy_tensors
        reference = input_ids if input_ids is not None else inputs_embeds
        device = reference.device
        if position_ids is None:
            batch_size, length = reference.shape[0], reference.shape[1]
            if replace_position is not None:
                start, end = int(replace_position[0]), int(replace_position[1])
                positions = torch.arange(start, end, device=device, dtype=torch.long)
            else:
                positions = torch.arange(length, device=device, dtype=torch.long)
            position_ids = positions.unsqueeze(0).repeat(batch_size, 1)

        hidden_states, present_key_values = self.model(
            input_ids,
            position_ids,
            past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            attention_mask=attention_mask,
            forward_batch=forward_batch,
        )
        return MoeCausalLMOutputWithPast(
            logits=self._get_logits(hidden_states),
            past_key_values=present_key_values,
            hidden_states=hidden_states,
        )


__all__ = [
    "DEFAULT_QUERY_SCALE_MAX_POSITION",
    "NEMOTRON_ARCHITECTURE",
    "NEMOTRON_MODEL_TYPE",
    "NemotronDiffusionAttention",
    "NemotronDiffusionDecoderLayer",
    "NemotronDiffusionMLP",
    "NemotronDiffusionModel",
    "NemotronLabsDiffusionLLM",
    "is_nemotron_diffusion_config",
    "nemotron_head_dim",
    "nemotron_query_scale",
    "nemotron_rope_parameters",
    "nemotron_weight_plan",
]

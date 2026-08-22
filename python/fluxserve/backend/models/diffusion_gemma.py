"""Text-only Diffusion-Gemma implementation for FluxServe.

This module intentionally depends only on FluxServe and Transformers.  The
encoder and decoder checkpoint trees are represented by one shared Gemma4
backbone; the runner selects causal prompt/commit attention or bidirectional
canvas attention.
"""

from __future__ import annotations

from copy import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

import fluxserve.backend.distributed as flux_distributed
from fluxserve.backend.distributed import (
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from fluxserve.backend.execution.forward_batch_info import ForwardBatch, PPProxyTensors
from fluxserve.backend.layers.attention import (
    AttentionForward,
    AttentionForwardConfig,
    apply_qk_norm,
)
from fluxserve.backend.layers.attention.diffusion_gemma_flashinfer import (
    DiffusionGemmaPagedAttention,
)
from fluxserve.backend.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
)
from fluxserve.backend.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from fluxserve.backend.layers.moe.fused_moe_triton.layer import FusedMoE
from fluxserve.backend.layers.moe.topk import StandardTopKOutput
from fluxserve.backend.layers.norm import RMSNorm
from fluxserve.backend.layers.rotary_embedding import get_rope
from fluxserve.backend.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from fluxserve.backend.utils.runtime_utils import add_prefix, make_layers


def get_diffusion_gemma_text_config(config):
    return getattr(config, "text_config", config)


def diffusion_gemma_layer_config(config, layer_idx: int):
    """Resolve heterogeneous attention fields for one Gemma4 layer."""
    if getattr(config, "is_heterogeneous", False):
        return config.per_layer_config[layer_idx]
    layer = copy(config)
    if config.layer_types[layer_idx] == "full_attention":
        layer.head_dim = getattr(config, "global_head_dim", None) or config.head_dim
        layer.num_key_value_heads = (
            getattr(config, "num_global_key_value_heads", None)
            or config.num_key_value_heads
        )
    return layer


def _rope_config(config, layer_type: str) -> tuple[float, dict | None]:
    params = getattr(config, "rope_parameters", None)
    if isinstance(params, dict) and layer_type in params:
        params = params[layer_type]
    if not isinstance(params, dict):
        params = getattr(config, "rope_scaling", None)
    base = getattr(config, "rope_theta", 10000.0)
    if isinstance(params, dict):
        base = params.get("rope_theta", base)
    return float(base), params


class DiffusionGemmaMLP(nn.Module):
    def __init__(self, config, intermediate_size: int, prefix: str = ""):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            prefix=add_prefix("down_proj", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = F.gelu(gate, approximate="tanh") * up
        x, _ = self.down_proj(x)
        return x


class DiffusionGemmaRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, has_weight=False)
        self.scale = nn.Parameter(torch.ones(config.hidden_size))
        self.proj = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.root_size = config.hidden_size**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x) * self.root_size
        return F.linear(x * self.scale.to(x.dtype), self.proj.weight).float()


class DiffusionGemmaMoE(nn.Module):
    def __init__(self, config, layer_id: int, prefix: str = ""):
        super().__init__()
        self.top_k = int(getattr(config, "top_k_experts", 2))
        self.per_expert_scale = nn.Parameter(torch.ones(config.num_experts))
        intermediate_size = getattr(
            config,
            "moe_intermediate_size",
            None,
        )
        if intermediate_size is None:
            intermediate_size = getattr(config, "expert_intermediate_size")
        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=self.top_k,
            layer_id=layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            reduce_results=True,
            prefix=add_prefix("experts", prefix),
            activation="gelu_tanh",
            inplace=False,
            use_weight_loader_fused=True,
        )

    def forward(self, x: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.reshape(-1, original_shape[-1])
        router_logits = router_logits.reshape(-1, router_logits.shape[-1])
        probs = router_logits.softmax(dim=-1)
        weights, ids = probs.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        weights = weights * self.per_expert_scale[ids]
        topk = StandardTopKOutput(weights.float(), ids.int(), router_logits)
        return self.experts(x, topk).reshape(original_shape)


class DiffusionGemmaAttention(nn.Module):
    def __init__(self, config, layer_id: int, prefix: str = ""):
        super().__init__()
        layer_config = diffusion_gemma_layer_config(config, layer_id)
        self.layer_id = layer_id
        self.layer_type = config.layer_types[layer_id]
        self.sliding_window = (
            int(config.sliding_window)
            if self.layer_type == "sliding_attention"
            else None
        )
        self.hidden_size = config.hidden_size
        self.head_dim = int(layer_config.head_dim)
        self.total_num_heads = int(config.num_attention_heads)
        self.total_num_kv_heads = int(layer_config.num_key_value_heads)
        # The engine initializes DP-attention metadata before serving.  Keep
        # direct model construction (checkpoint loading and local tests)
        # usable on a single process as well.
        try:
            tp_rank = get_attention_tp_rank()
            tp_size = get_attention_tp_size()
        except AssertionError:
            tp_rank, tp_size = 0, 1
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bool(getattr(config, "attention_bias", False)),
            prefix=add_prefix("qkv_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=bool(getattr(config, "attention_bias", False)),
            prefix=add_prefix("o_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.v_norm = RMSNorm(self.head_dim, config.rms_norm_eps, has_weight=False)
        base, rope_scaling = _rope_config(config, self.layer_type)
        partial_rotary_factor = 1.0
        if isinstance(rope_scaling, dict):
            partial_rotary_factor = float(
                rope_scaling.get("partial_rotary_factor", 1.0)
            )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=base,
            rope_scaling=rope_scaling,
            dtype=torch.float32,
            partial_rotary_factor=partial_rotary_factor,
        )
        self.attention_forward = AttentionForward(
            AttentionForwardConfig(
                layer_id=layer_id,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                num_key_value_groups=self.num_key_value_groups,
                scale=1.0,
            )
        )
        self.paged_attention = DiffusionGemmaPagedAttention(
            layer_id=layer_id,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            scale=1.0,
            sliding_window=self.sliding_window,
        )

    def _apply_sliding_window(
        self,
        attention_mask: torch.Tensor | None,
        positions: torch.Tensor,
        kv_len: int,
    ) -> torch.Tensor | None:
        if self.sliding_window is None:
            return attention_mask
        q_pos = positions.long()
        cache_len = kv_len - positions.shape[-1]
        kv_start = positions[..., :1] - cache_len
        kv_pos = kv_start.unsqueeze(-1) + torch.arange(
            kv_len, device=positions.device
        ).view(1, 1, -1)
        window = kv_pos >= q_pos.unsqueeze(-1) - self.sliding_window + 1
        window &= kv_pos <= q_pos.unsqueeze(-1) + self.sliding_window - 1
        if attention_mask is None:
            return window
        return attention_mask[..., -kv_len:].bool() & window

    def _cache_for_next_forward(self, k, v):
        if self.sliding_window is None or k.shape[2] <= self.sliding_window:
            return k, v
        return k[:, :, -self.sliding_window :, :], v[:, :, -self.sliding_window :, :]

    @staticmethod
    def _append_past_key_values(k, v, past_key_values):
        if past_key_values is None:
            return k, v
        return (
            torch.cat((past_key_values[0], k), dim=2),
            torch.cat((past_key_values[1], v), dim=2),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values=None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
        forward_batch: ForwardBatch | None = None,
    ):
        bsz, q_len, _ = hidden_states.shape
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.layer_type == "full_attention":
            # The official full-attention layers have no v_proj; V is the raw
            # key projection before K normalization and rotary embedding.
            v = k
        q, k = apply_qk_norm(
            q,
            k,
            query_layernorm=self.q_norm,
            key_layernorm=self.k_norm,
            head_dim=self.head_dim,
            alt_stream=None,
        )
        v = self.v_norm(v.view(bsz, q_len, self.num_kv_heads, self.head_dim)).flatten(-2)
        q, k = self.rotary_emb(
            positions.flatten(), q.flatten(0, 1), k.flatten(0, 1), fused_set_kv_buffer_arg=None
        )
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if getattr(past_key_values, "is_diffusion_gemma_paged_cache", False):
            if forward_batch is None or not forward_batch.diffusion_gemma_phase:
                raise RuntimeError(
                    "Diffusion-Gemma paged attention requires phase metadata"
                )
            output = self.paged_attention.forward(
                q,
                k,
                v,
                cache=past_key_values,
                metadata=forward_batch.diffusion_gemma_attention_metadata,
                graph_runner=(
                    forward_batch.flashinfer_cuda_graph_runner
                    if forward_batch.diffusion_gemma_full_decode_graph
                    else None
                ),
            )
            output = output.transpose(1, 2).reshape(bsz, q_len, -1)
            output, _ = self.o_proj(output)
            return output, past_key_values if use_cache else None

        k, v = self._append_past_key_values(k, v, past_key_values)
        kv_len = k.shape[2]
        attention_mask = self._apply_sliding_window(attention_mask, positions, kv_len)
        output, _ = self.attention_forward.forward(
            q,
            k,
            v,
            past_key_values=None,
            use_cache=False,
            attention_mask=attention_mask,
            forward_batch=forward_batch,
        )
        output = output.transpose(1, 2).reshape(bsz, q_len, -1)
        output, _ = self.o_proj(output)
        present = self._cache_for_next_forward(k, v) if use_cache else None
        return output, present


class DiffusionGemmaDecoderLayer(nn.Module):
    def __init__(self, config, layer_id: int, prefix: str = ""):
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = DiffusionGemmaAttention(config, layer_id, add_prefix("self_attn", prefix))
        first_shared = config.num_hidden_layers - int(
            getattr(config, "num_kv_shared_layers", 0)
        )
        intermediate = int(config.intermediate_size)
        if getattr(config, "use_double_wide_mlp", False) and layer_id >= first_shared:
            intermediate *= 2
        self.mlp = DiffusionGemmaMLP(config, intermediate, add_prefix("mlp", prefix))
        eps = config.rms_norm_eps
        self.input_layernorm = RMSNorm(config.hidden_size, eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps)
        self.enable_moe = bool(
            getattr(config, "enable_moe_block", False)
            or getattr(config, "use_second_mlp_block", False)
            or getattr(config, "num_experts", 0)
        )
        if self.enable_moe:
            self.router = DiffusionGemmaRouter(config)
            self.moe = DiffusionGemmaMoE(config, layer_id, add_prefix("moe", prefix))
            self.post_feedforward_layernorm_1 = RMSNorm(config.hidden_size, eps)
            self.pre_feedforward_layernorm_2 = RMSNorm(config.hidden_size, eps)
            self.post_feedforward_layernorm_2 = RMSNorm(config.hidden_size, eps)
        ple_size = int(getattr(config, "hidden_size_per_layer_input", 0) or 0)
        if ple_size:
            self.per_layer_input_gate = nn.Linear(config.hidden_size, ple_size, bias=False)
            self.per_layer_projection = nn.Linear(ple_size, config.hidden_size, bias=False)
            self.post_per_layer_input_norm = RMSNorm(config.hidden_size, eps)
        else:
            self.per_layer_input_gate = None
        self.layer_scalar = nn.Parameter(torch.ones(1))

    def forward(
        self,
        positions,
        hidden_states,
        past_key_values=None,
        use_cache=False,
        attention_mask=None,
        per_layer_input=None,
        forward_batch=None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present = self.self_attn(
            positions,
            hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attention_mask=attention_mask,
            forward_batch=forward_batch,
        )
        hidden_states = self.post_attention_layernorm(hidden_states) + residual
        residual = hidden_states
        dense = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        if self.enable_moe:
            dense = self.post_feedforward_layernorm_1(dense)
            moe_input = self.pre_feedforward_layernorm_2(residual)
            moe = self.moe(moe_input, self.router(residual))
            hidden_states = dense + self.post_feedforward_layernorm_2(moe)
        else:
            hidden_states = dense
        hidden_states = self.post_feedforward_layernorm(hidden_states) + residual
        if per_layer_input is not None and self.per_layer_input_gate is not None:
            gate = F.gelu(self.per_layer_input_gate(hidden_states), approximate="tanh")
            ple = self.per_layer_projection(gate * per_layer_input)
            hidden_states = hidden_states + self.post_per_layer_input_norm(ple)
        return hidden_states * self.layer_scalar, present


class DiffusionGemmaModel(nn.Module):
    def __init__(self, config, prefix: str = "model"):
        super().__init__()
        self.config = config
        self.pp_group = get_pp_group()
        if self.pp_group.world_size != 1:
            raise ValueError("Diffusion-Gemma pipeline parallelism is not supported yet")
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=add_prefix("embed_tokens", prefix)
        )
        self.normalizer = config.hidden_size**0.5
        ple_size = int(getattr(config, "hidden_size_per_layer_input", 0) or 0)
        self.ple_size = ple_size
        if ple_size:
            ple_vocab = int(getattr(config, "vocab_size_per_layer_input", config.vocab_size))
            self.embed_tokens_per_layer = VocabParallelEmbedding(
                ple_vocab,
                ple_size * config.num_hidden_layers,
                prefix=add_prefix("embed_tokens_per_layer", prefix),
            )
            self.per_layer_model_projection = ColumnParallelLinear(
                config.hidden_size,
                ple_size * config.num_hidden_layers,
                bias=False,
                gather_output=True,
                prefix=add_prefix("per_layer_model_projection", prefix),
            )
            self.per_layer_projection_norm = RMSNorm(ple_size, config.rms_norm_eps)
        else:
            self.embed_tokens_per_layer = None
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, layer_prefix: DiffusionGemmaDecoderLayer(
                config, idx, layer_prefix
            ),
            prefix=add_prefix("layers", prefix),
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.normalizer

    def _per_layer_inputs(self, input_ids, inputs_embeds):
        if self.embed_tokens_per_layer is None:
            return None
        ple_vocab = self.embed_tokens_per_layer.num_embeddings
        valid = (input_ids >= 0) & (input_ids < ple_vocab)
        ple_ids = torch.where(valid, input_ids, torch.zeros_like(input_ids))
        token_ple = self.embed_tokens_per_layer(ple_ids) * self.ple_size**0.5
        projected, _ = self.per_layer_model_projection(inputs_embeds)
        projected = projected * self.config.hidden_size**-0.5
        shape = (*input_ids.shape, self.config.num_hidden_layers, self.ple_size)
        projected = self.per_layer_projection_norm(projected.reshape(shape))
        return (projected + token_ple.reshape(shape)) * (2.0**-0.5)

    def forward(
        self,
        input_ids,
        positions,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=False,
        attention_mask=None,
        forward_batch=None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        per_layer_inputs = self._per_layer_inputs(input_ids, inputs_embeds)
        hidden_states = inputs_embeds
        presents = []
        paged_cache = getattr(
            past_key_values, "is_diffusion_gemma_paged_cache", False
        )
        if (
            past_key_values is not None
            and not paged_cache
            and len(past_key_values) != len(self.layers)
        ):
            raise ValueError(
                "Diffusion-Gemma past_key_values must contain one entry per layer: "
                f"expected {len(self.layers)}, got {len(past_key_values)}"
            )
        for i, layer in enumerate(self.layers):
            layer_past = (
                past_key_values
                if paged_cache
                else past_key_values[i]
                if past_key_values is not None
                else None
            )
            layer_ple = per_layer_inputs[..., i, :] if per_layer_inputs is not None else None
            hidden_states, present = layer(
                positions,
                hidden_states,
                past_key_values=layer_past,
                use_cache=use_cache,
                attention_mask=attention_mask,
                per_layer_input=layer_ple,
                forward_batch=forward_batch,
            )
            if use_cache and not paged_cache:
                presents.append(present)
        return self.norm(hidden_states), past_key_values if paged_cache else tuple(presents)


class DiffusionGemmaSelfConditioning(nn.Module):
    def __init__(self, config, size: int):
        super().__init__()
        self.pre_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_norm = RMSNorm(config.hidden_size, config.rms_norm_eps, has_weight=False)
        self.mlp = DiffusionGemmaMLP(config, size, "self_conditioning")

    def forward(self, inputs_embeds, soft_embeds):
        return self.post_norm(inputs_embeds + self.mlp(self.pre_norm(soft_embeds)))


class DiffusionGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config, quant_config=None, expert_map_path: str = ""):
        super().__init__()
        if quant_config is not None:
            raise ValueError("Diffusion-Gemma quantization is not supported yet")
        del expert_map_path
        self.config = config
        self.text_config = get_diffusion_gemma_text_config(config)
        self.text_config.attention_k_eq_v = True
        if getattr(self.text_config, "num_experts", None):
            self.text_config.enable_moe_block = True
        self.quant_config = None
        self.model = DiffusionGemmaModel(self.text_config)
        size = getattr(config, "self_conditioning_size", None) or self.text_config.intermediate_size
        self.self_conditioning = DiffusionGemmaSelfConditioning(self.text_config, int(size))
        self.lm_head = ParallelLMHead(
            self.text_config.vocab_size,
            self.text_config.hidden_size,
            bias=False,
            prefix="lm_head",
        )
        if self.text_config.tie_word_embeddings:
            self.lm_head.tie_weights(self.model.embed_tokens)
        sharded_mapping = self.lm_head.get_sharded_to_full_mapping()
        self.register_buffer(
            "_sharded_to_full_index",
            torch.tensor(
                sharded_mapping if sharded_mapping is not None else (),
                dtype=torch.long,
                device=self.lm_head.weight.device,
            ),
            persistent=False,
        )
        self.final_logit_softcapping = getattr(
            self.text_config, "final_logit_softcapping", None
        )

    @staticmethod
    def _checkpoint_name(name: str) -> str | None:
        if name.startswith(("model.encoder.vision_tower.", "model.encoder.embed_vision.")):
            return None
        if name.startswith("model.encoder.language_model."):
            return "model." + name[len("model.encoder.language_model.") :]
        if name.startswith("model.decoder.self_conditioning."):
            return "self_conditioning." + name.split("self_conditioning.", 1)[1]
        if name.startswith("model.decoder."):
            return "model." + name[len("model.decoder.") :]
        return name

    def load_weights(self, weights) -> tuple[set[str], set[str]]:
        """Load one or more checkpoint tensors using FluxServe shard loaders."""
        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        loaded: set[str] = set()
        unexpected: set[str] = set()

        def local_expert_weights(value: torch.Tensor) -> torch.Tensor:
            ep_size = get_moe_expert_parallel_world_size()
            if ep_size == 1:
                return value
            if value.shape[0] % ep_size != 0:
                raise ValueError(
                    "Diffusion-Gemma expert count must be divisible by EP size: "
                    f"checkpoint={value.shape[0]}, ep_size={ep_size}"
                )
            local_experts = value.shape[0] // ep_size
            start = get_moe_expert_parallel_rank() * local_experts
            return value.narrow(0, start, local_experts)

        def load_param(target: str, value: torch.Tensor, shard_id=None):
            param = params.get(target)
            if param is None:
                unexpected.add(target)
                return
            value = value.to(device=param.device)
            loader = getattr(param, "weight_loader", None)
            if loader is not None:
                if shard_id is None:
                    loader(param, value)
                else:
                    loader(param, value, shard_id)
            else:
                if param.shape != value.shape:
                    raise ValueError(
                        f"Diffusion-Gemma weight shape mismatch for {target}: "
                        f"checkpoint={tuple(value.shape)} model={tuple(param.shape)}"
                    )
                param.data.copy_(value.to(param.dtype))
            loaded.add(target)

        for raw_name, value in weights:
            name = self._checkpoint_name(raw_name)
            if name is None:
                continue
            name = name.replace(".router.per_expert_scale", ".moe.per_expert_scale")
            name = name.replace(".experts.gate_up_proj", ".moe.experts.w13_weight")
            name = name.replace(".experts.down_proj", ".moe.experts.w2_weight")

            if name.endswith(".self_attn.q_proj.weight"):
                load_param(name.replace("q_proj", "qkv_proj"), value, "q")
            elif name.endswith(".self_attn.k_proj.weight"):
                target = name.replace("k_proj", "qkv_proj")
                load_param(target, value, "k")
                layer_id = int(name.split(".layers.", 1)[1].split(".", 1)[0])
                if self.text_config.layer_types[layer_id] == "full_attention":
                    load_param(target, value, "v")
            elif name.endswith(".self_attn.v_proj.weight"):
                load_param(name.replace("v_proj", "qkv_proj"), value, "v")
            elif name.endswith(".mlp.gate_proj.weight"):
                load_param(name.replace("gate_proj", "gate_up_proj"), value, 0)
            elif name.endswith(".mlp.up_proj.weight"):
                load_param(name.replace("up_proj", "gate_up_proj"), value, 1)
            elif name.startswith("self_conditioning.") and name.endswith("gate_proj.weight"):
                load_param(
                    name.replace("self_conditioning.gate_proj", "self_conditioning.mlp.gate_up_proj"),
                    value,
                    0,
                )
            elif name.startswith("self_conditioning.") and name.endswith("up_proj.weight"):
                load_param(
                    name.replace("self_conditioning.up_proj", "self_conditioning.mlp.gate_up_proj"),
                    value,
                    1,
                )
            elif name.startswith("self_conditioning.") and name.endswith("down_proj.weight"):
                load_param(name.replace("self_conditioning.down_proj", "self_conditioning.mlp.down_proj"), value)
            elif ".moe.experts.w13_weight" in name:
                target = name if name.endswith("weight") else name + ".weight"
                param = params.get(target)
                if param is None:
                    unexpected.add(target)
                else:
                    loader = getattr(param, "weight_loader", None)
                    if loader is None:
                        raise ValueError(f"Missing fused MoE loader for {target}")
                    value = local_expert_weights(value)
                    loader(param, value.to(param.device), target, "w13")
                    loaded.add(target)
            elif ".moe.experts.w2_weight" in name:
                target = name if name.endswith("weight") else name + ".weight"
                param = params.get(target)
                if param is None:
                    unexpected.add(target)
                else:
                    loader = getattr(param, "weight_loader", None)
                    if loader is None:
                        raise ValueError(f"Missing fused MoE loader for {target}")
                    value = local_expert_weights(value)
                    loader(param, value.to(param.device), target, "w2")
                    loaded.add(target)
            elif name in buffers:
                buffers[name].data.copy_(value.to(buffers[name].device, buffers[name].dtype))
                loaded.add(name)
            elif name == "lm_head.weight" and self.text_config.tie_word_embeddings:
                continue
            else:
                load_param(name, value)
        return loaded, unexpected

    def embed_with_self_conditioning(self, input_ids, soft_embeds=None):
        inputs = self.model.embed_input_ids(input_ids)
        if soft_embeds is None:
            soft_embeds = torch.zeros_like(inputs)
        return self.self_conditioning(inputs, soft_embeds.to(inputs.dtype))

    def _get_logits(self, hidden_states):
        local = torch.matmul(hidden_states.to(self.lm_head.weight.dtype), self.lm_head.weight.T)
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1:
            parts = [torch.empty_like(local) for _ in range(tp_size)]
            torch.distributed.all_gather(
                parts, local, group=flux_distributed.get_tensor_model_parallel_group()
            )
            local = torch.cat(parts, dim=-1)
            if self._sharded_to_full_index.numel():
                local = local.index_select(-1, self._sharded_to_full_index)
        logits = local[..., : self.text_config.vocab_size].float()
        if self.final_logit_softcapping is not None:
            cap = float(self.final_logit_softcapping)
            logits = torch.tanh(logits / cap) * cap
        return logits

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        inputs_embeds=None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        past_key_values=None,
        replace_position=None,
        use_cache=False,
        attention_mask=None,
        forward_batch=None,
    ):
        del pp_proxy_tensors, replace_position
        if position_ids is None:
            length = input_ids.shape[1]
            position_ids = torch.arange(length, device=input_ids.device).expand(
                input_ids.shape[0], -1
            )
        hidden, presents = self.model(
            input_ids,
            position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attention_mask=attention_mask,
            forward_batch=forward_batch,
        )
        return MoeCausalLMOutputWithPast(
            logits=self._get_logits(hidden),
            past_key_values=presents,
            hidden_states=hidden,
        )


__all__ = [
    "DiffusionGemmaForConditionalGeneration",
    "DiffusionGemmaModel",
    "DiffusionGemmaSelfConditioning",
    "diffusion_gemma_layer_config",
    "get_diffusion_gemma_text_config",
]

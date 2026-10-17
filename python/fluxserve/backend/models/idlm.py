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
    I-DLM (Introspective Diffusion Language Model) implementation.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

import fluxserve.backend.distributed as flux_distributed
from fluxserve.backend.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from fluxserve.backend.execution.forward_batch_info import ForwardBatch, PPProxyTensors
from fluxserve.backend.layers.activation import SiluAndMul
from fluxserve.backend.layers.attention import (
    AttentionForward,
    AttentionForwardConfig,
    apply_qk_norm,
)
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

# I-DLM checkpoints keep SDAR's names: the architecture is SDAR's, the training
# recipe (strict causal attention, shifted logits) is I-DLM's.
IDLM_ARCHITECTURE = "SDARForCausalLM"
IDLM_MODEL_TYPE = "sdar"
# R-ISD serves the gated LoRA on top of the AR model the checkpoint came from.
QWEN3_ARCHITECTURE = "Qwen3ForCausalLM"
QWEN3_MODEL_TYPE = "qwen3"
# The MASK token of every released I-DLM checkpoint and adapter.
IDLM_DEFAULT_MASK_ID = 151669


def is_idlm_config(model_config) -> bool:
    """Recognize an SDAR-architecture checkpoint from its config alone."""
    architectures = set(getattr(model_config, "architectures", ()) or ())
    return (
        IDLM_ARCHITECTURE in architectures
        or getattr(model_config, "model_type", None) == IDLM_MODEL_TYPE
    )


def is_qwen3_config(model_config) -> bool:
    """A Qwen3 dense checkpoint: the AR base of an I-DLM gated LoRA."""
    architectures = set(getattr(model_config, "architectures", ()) or ())
    return (
        QWEN3_ARCHITECTURE in architectures
        or getattr(model_config, "model_type", None) == QWEN3_MODEL_TYPE
    )


def check_idlm_config(model_config) -> None:
    """Reject SDAR checkpoints that are not I-DLM.

    Plain SDAR denoises each block bidirectionally and predicts the token at
    the same position; I-DLM is strictly causal with shifted logits, which is
    what lets one causal forward both draft and verify. The two share an
    architecture name, so ``use_regular_causal`` is the only thing telling them
    apart, and serving a plain SDAR checkpoint causally would run a different
    model rather than fail.
    """
    if not getattr(model_config, "use_regular_causal", False):
        raise ValueError(
            "This SDAR checkpoint does not declare use_regular_causal=true. "
            "FluxServe serves I-DLM (strictly causal, shifted logits) only; "
            "block-bidirectional SDAR decoding is not supported."
        )
    mask_id = getattr(model_config, "mask_token_id", None)
    vocab_size = int(model_config.vocab_size)
    if mask_id is None or not 0 <= int(mask_id) < vocab_size:
        raise ValueError(
            "I-DLM checkpoint must declare mask_token_id inside its vocabulary, "
            f"got {mask_id!r} for vocab_size={vocab_size}"
        )
    check_idlm_architecture(model_config)


def check_idlm_architecture(model_config) -> None:
    """The Qwen3 dense layout this module implements, shared with R-ISD bases."""
    if getattr(model_config, "attention_bias", False):
        raise ValueError("I-DLM checkpoints with attention_bias are not supported")
    if getattr(model_config, "hidden_act", "silu") != "silu":
        raise ValueError(
            f"I-DLM expects hidden_act='silu', got {model_config.hidden_act!r}"
        )
    if getattr(model_config, "use_sliding_window", False):
        raise ValueError("I-DLM checkpoints with sliding-window attention are not supported")


def idlm_head_dim(config) -> int:
    head_dim = getattr(config, "head_dim", None)
    if head_dim:
        return int(head_dim)
    return int(config.hidden_size) // int(config.num_attention_heads)


def idlm_rope_parameters(config) -> tuple[float, dict | None]:
    """Resolve ``(base, rope_scaling)`` for ``get_rope``.

    Transformers 5 standardizes RoPE settings into ``rope_parameters`` (here
    ``{"rope_type": "default", "rope_theta": 1e6}``) and mirrors them onto
    ``rope_scaling``; older configs keep ``rope_theta`` at the top level.
    """
    params = getattr(config, "rope_parameters", None)
    if not isinstance(params, dict):
        params = getattr(config, "rope_scaling", None)
    base = float(getattr(config, "rope_theta", 10000.0) or 10000.0)
    if isinstance(params, dict):
        base = float(params.get("rope_theta", base))
    return base, params if isinstance(params, dict) else None


def idlm_weight_plan(config) -> dict[str, tuple[str, object, tuple[int, ...]]]:
    """Every checkpoint tensor this model expects, with destination and shape.

    Same contract as ``nemotron_weight_plan``: ``{source_name:
    (destination_param, shard_id, full_shape)}``, with full unsharded shapes so
    the plan can be checked against safetensors headers alone.
    """
    hidden = int(config.hidden_size)
    vocab = int(config.vocab_size)
    intermediate = int(config.intermediate_size)
    head_dim = idlm_head_dim(config)
    q_dim = int(config.num_attention_heads) * head_dim
    kv_dim = int(config.num_key_value_heads) * head_dim

    plan: dict[str, tuple[str, object, tuple[int, ...]]] = {
        "model.embed_tokens.weight": ("model.embed_tokens.weight", None, (vocab, hidden)),
        "model.norm.weight": ("model.norm.weight", None, (hidden,)),
    }
    if not getattr(config, "tie_word_embeddings", False):
        plan["lm_head.weight"] = ("lm_head.weight", None, (vocab, hidden))
    for layer_id in range(int(config.num_hidden_layers)):
        prefix = f"model.layers.{layer_id}."
        plan[prefix + "input_layernorm.weight"] = (
            prefix + "input_layernorm.weight", None, (hidden,)
        )
        plan[prefix + "post_attention_layernorm.weight"] = (
            prefix + "post_attention_layernorm.weight", None, (hidden,)
        )
        qkv = prefix + "self_attn.qkv_proj.weight"
        plan[prefix + "self_attn.q_proj.weight"] = (qkv, "q", (q_dim, hidden))
        plan[prefix + "self_attn.k_proj.weight"] = (qkv, "k", (kv_dim, hidden))
        plan[prefix + "self_attn.v_proj.weight"] = (qkv, "v", (kv_dim, hidden))
        plan[prefix + "self_attn.o_proj.weight"] = (
            prefix + "self_attn.o_proj.weight", None, (hidden, q_dim)
        )
        plan[prefix + "self_attn.q_norm.weight"] = (
            prefix + "self_attn.q_norm.weight", None, (head_dim,)
        )
        plan[prefix + "self_attn.k_norm.weight"] = (
            prefix + "self_attn.k_norm.weight", None, (head_dim,)
        )
        gate_up = prefix + "mlp.gate_up_proj.weight"
        plan[prefix + "mlp.gate_proj.weight"] = (gate_up, 0, (intermediate, hidden))
        plan[prefix + "mlp.up_proj.weight"] = (gate_up, 1, (intermediate, hidden))
        plan[prefix + "mlp.down_proj.weight"] = (
            prefix + "mlp.down_proj.weight", None, (hidden, intermediate)
        )
    return plan


def check_idlm_context_limit(max_positions: int, model_config=None, *, serving_limit=None) -> None:
    """Check absolute positions, including the speculative block slots."""
    limit = int(getattr(model_config, "max_position_embeddings", 0) or 0) or None
    if serving_limit is not None:
        limit = int(serving_limit) if limit is None else min(limit, int(serving_limit))
    if limit is not None and not 0 <= int(max_positions) <= limit:
        raise ValueError(
            f"I-DLM support is limited to {limit} total positions; requested "
            f"{int(max_positions)}. This includes the speculative block slots."
        )


class IDLMGatedLoRA:
    """The R-ISD gated LoRA residual, active only at MASK input positions.

    Every adapted projection computes ``W x + 1[x_j is MASK] * s * B A x``.
    Clean positions -- the anchor and the specs -- therefore run the base
    weights alone, and because attention is causal they never see a MASK
    position: their logits are the base model's exact distribution and their
    KV, the only KV a step commits, is the base model's. MASK positions get
    the adapter's proposals. The gate comes from the input ids, so a captured
    decode graph needs no extra input; prefill does not enable the adapter.
    """

    TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    def __init__(self, *, mask_id: int, scaling: float, layers: list[dict]):
        self.mask_id = int(mask_id)
        self.scaling = float(scaling)
        self.active = False
        self.gate: torch.Tensor | None = None
        # Per layer: one shrink per fused projection, then one expand per slice.
        self.layers = []
        for weights in layers:
            fused = {}
            for name, parts in (
                ("qkv", ("q_proj", "k_proj", "v_proj")),
                ("o", ("o_proj",)),
                ("gate_up", ("gate_proj", "up_proj")),
                ("down", ("down_proj",)),
            ):
                present = [part for part in parts if part in weights]
                if not present:
                    continue
                fused[name] = (
                    torch.cat([weights[part][0] for part in present], dim=0).contiguous(),
                    [(parts.index(part), weights[part][1].contiguous()) for part in present],
                )
            self.layers.append(fused)

    def set_gate(self, input_ids: torch.Tensor, dtype: torch.dtype) -> None:
        self.gate = (input_ids == self.mask_id).reshape(-1, 1).to(dtype)

    def apply(self, layer_id: int, name: str, x: torch.Tensor, out: torch.Tensor,
              slice_sizes) -> torch.Tensor:
        """Add the gated residual to ``out`` in place and return it.

        ``slice_sizes`` are the widths of the fused output's slices in order
        (one for an unfused projection); absent targets keep base weights.
        """
        entry = self.layers[layer_id].get(name)
        if entry is None:
            return out
        shrink_weight, expands = entry
        rows = x.reshape(-1, x.shape[-1])
        flat = out.view(-1, out.shape[-1])
        shrink = torch.mm(rows, shrink_weight.t()) * self.gate
        rank = shrink_weight.shape[0] // len(expands)
        offsets = [0]
        for size in slice_sizes:
            offsets.append(offsets[-1] + int(size))
        for position, (slice_index, expand_weight) in enumerate(expands):
            start, end = offsets[slice_index], offsets[slice_index + 1]
            flat[:, start:end].addmm_(
                shrink[:, position * rank : (position + 1) * rank],
                expand_weight.t(),
                alpha=self.scaling,
            )
        return out


class IDLMMLP(nn.Module):
    def __init__(self, config, prefix: str = "", layer_id: int = 0):
        super().__init__()
        self.layer_id = layer_id
        self.lora: IDLMGatedLoRA | None = None
        intermediate_size = int(config.intermediate_size)
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        lora = self.lora if self.lora is not None and self.lora.active else None
        gate_up, _ = self.gate_up_proj(hidden_states)
        if lora is not None:
            width = gate_up.shape[-1] // 2
            gate_up = lora.apply(self.layer_id, "gate_up", hidden_states, gate_up, (width, width))
        activated = self.act_fn(gate_up)
        hidden_states, _ = self.down_proj(activated)
        if lora is not None:
            hidden_states = lora.apply(
                self.layer_id, "down", activated, hidden_states, (hidden_states.shape[-1],)
            )
        return hidden_states


class IDLMAttention(nn.Module):
    """Qwen3 GQA attention with per-head QK RMSNorm; the caller owns causality.

    I-DLM is served causally in every phase, so the runner always builds causal
    metadata; nothing here depends on the phase.
    """

    def __init__(self, config, layer_id: int = 0, prefix: str = ""):
        super().__init__()
        self.layer_id = layer_id
        self.lora: IDLMGatedLoRA | None = None
        self.hidden_size = int(config.hidden_size)
        self.head_dim = idlm_head_dim(config)
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

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            prefix=add_prefix("qkv_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        eps = float(config.rms_norm_eps)
        self.q_norm = RMSNorm(self.head_dim, eps=eps)
        self.k_norm = RMSNorm(self.head_dim, eps=eps)
        base, rope_scaling = idlm_rope_parameters(config)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=int(config.max_position_embeddings),
            base=base,
            rope_scaling=rope_scaling,
            dtype=torch.float32,
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
        lora = self.lora if self.lora is not None and self.lora.active else None
        qkv, _ = self.qkv_proj(hidden_states)
        if lora is not None:
            qkv = lora.apply(
                self.layer_id, "qkv", hidden_states, qkv,
                (self.q_size, self.kv_size, self.kv_size),
            )
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Qwen3 normalizes each head before RoPE.
        q, k = apply_qk_norm(
            q,
            k,
            query_layernorm=self.q_norm,
            key_layernorm=self.k_norm,
            head_dim=self.head_dim,
            alt_stream=None,
        )
        q, k = self.rotary_emb(
            positions.flatten(),
            q.flatten(0, 1),
            k.flatten(0, 1),
            fused_set_kv_buffer_arg=None,
        )
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

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
        projected, _ = self.o_proj(attn_output)
        if lora is not None:
            projected = lora.apply(
                self.layer_id, "o", attn_output, projected, (projected.shape[-1],)
            )
        return projected, present_key_values


class IDLMDecoderLayer(nn.Module):
    def __init__(self, config, layer_id: int = 0, prefix: str = ""):
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = IDLMAttention(config, layer_id, add_prefix("self_attn", prefix))
        self.mlp = IDLMMLP(config, add_prefix("mlp", prefix), layer_id)
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


class IDLMModel(nn.Module):
    def __init__(self, config, prefix: str = "model"):
        super().__init__()
        self.config = config
        self.lora: IDLMGatedLoRA | None = None
        self.pp_group = get_pp_group()
        if self.pp_group.world_size != 1:
            raise ValueError("I-DLM pipeline parallelism is not supported yet")
        self.embed_tokens = VocabParallelEmbedding(
            int(config.vocab_size),
            int(config.hidden_size),
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers, self.start_layer, self.end_layer = make_layers(
            int(config.num_hidden_layers),
            lambda idx, layer_prefix: IDLMDecoderLayer(config, idx, layer_prefix),
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
        if self.lora is not None and self.lora.active:
            if input_ids is None:
                raise ValueError("the gated LoRA needs input ids to find MASK positions")
            self.lora.set_gate(input_ids, hidden_states.dtype)

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


class IDLMForCausalLM(nn.Module):
    """Top-level module matching FluxServe's runner/loader contract.

    The checkpoint's tensor names already match this module's, apart from the
    fused QKV and gate/up projections that :func:`idlm_weight_plan` maps.
    """

    def __init__(self, config, quant_config=None, expert_map_path: str = ""):
        super().__init__()
        if quant_config is not None:
            raise ValueError("I-DLM quantization is not supported yet")
        del expert_map_path
        check_idlm_architecture(config)
        self.config = config
        self.quant_config = None
        self.model = IDLMModel(config)
        self.lm_head = ParallelLMHead(
            int(config.vocab_size),
            int(config.hidden_size),
            bias=False,
            prefix="lm_head",
        )
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.tie_weights(self.model.embed_tokens)
        self._lm_head_sharded_to_full_mapping = None

    def weight_plan(self) -> dict[str, tuple[str, object, tuple[int, ...]]]:
        return idlm_weight_plan(self.config)

    def attach_lora(self, lora: IDLMGatedLoRA | None) -> None:
        """Route every adapted projection through ``lora`` (``None`` detaches)."""
        if lora is not None and len(lora.layers) != len(self.model.layers):
            raise ValueError(
                f"the adapter has {len(lora.layers)} layers, the model "
                f"{len(self.model.layers)}"
            )
        self.model.lora = lora
        for layer in self.model.layers:
            layer.self_attn.lora = lora
            layer.mlp.lora = lora

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> tuple[set[str], set[str]]:
        """Load checkpoint tensors, returning ``(consumed_sources, unexpected)``.

        As with Nemotron, sources are reported rather than destinations, since
        a fused destination is written once per slice.
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
                    f"I-DLM checkpoint shape mismatch for {raw_name}: "
                    f"checkpoint={tuple(value.shape)} expected={tuple(expected_shape)}"
                )
            param = params.get(target)
            if param is None:
                raise ValueError(
                    f"I-DLM model has no parameter {target!r} for checkpoint "
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
                    f"I-DLM weight shape mismatch for {target}: "
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
        # Logits stay in the projection dtype; the ISD decoder upcasts only the
        # rows it turns into probabilities.
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
    ) -> CausalLMOutputWithPast:
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
        return CausalLMOutputWithPast(
            logits=self._get_logits(hidden_states),
            past_key_values=present_key_values,
            hidden_states=hidden_states,
        )


__all__ = [
    "IDLM_ARCHITECTURE",
    "IDLM_DEFAULT_MASK_ID",
    "IDLM_MODEL_TYPE",
    "IDLMAttention",
    "IDLMDecoderLayer",
    "IDLMForCausalLM",
    "IDLMGatedLoRA",
    "IDLMMLP",
    "IDLMModel",
    "QWEN3_ARCHITECTURE",
    "QWEN3_MODEL_TYPE",
    "check_idlm_architecture",
    "check_idlm_config",
    "check_idlm_context_limit",
    "idlm_head_dim",
    "idlm_rope_parameters",
    "idlm_weight_plan",
    "is_idlm_config",
    "is_qwen3_config",
]

# Adapted from https://github.com/inclusionAI/dInfer/blob/master/python/dinfer/model/modeling_llada2_moe_sglang.py
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

import logging
import re
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

import fluxserve.backend.distributed as flux_distributed

# monkey patch
def torch_all_reduce(tensor):
    if (
        flux_distributed.get_tensor_model_parallel_world_size() > 1
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.all_reduce(
            tensor, group=flux_distributed.get_tensor_model_parallel_group()
        )
    return tensor

flux_distributed.tensor_model_parallel_all_reduce = torch_all_reduce

def torch_all_gather(input_: torch.Tensor) -> torch.Tensor:
    input_size = input_.size()
    world_size = flux_distributed.get_tensor_model_parallel_world_size()
    output_size = (input_size[0] * world_size,) + input_size[1:]
    output_tensor = torch.empty(output_size, dtype=input_.dtype, device=input_.device)

    torch.distributed.all_gather_into_tensor(output_tensor, input_)
    return output_tensor

flux_distributed.tensor_model_parallel_all_gather = torch_all_gather


from fluxserve.backend.distributed import (
    get_moe_expert_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    parallel_state,
    tensor_model_parallel_all_reduce,
)
from fluxserve.backend.eplb.expert_location import ModelConfigForExpertLocation
from fluxserve.backend.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from fluxserve.backend.layers.activation import SiluAndMul
from fluxserve.backend.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    enable_moe_dense_fully_dp,
)
from fluxserve.backend.layers.dp_attention import (
    get_attention_dp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
)
from fluxserve.backend.layers.attention import (
    AttentionForward,
    AttentionForwardConfig,
    apply_qk_norm,
)
from fluxserve.backend.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from fluxserve.backend.layers.moe import get_deepep_mode, get_moe_a2a_backend
from fluxserve.backend.layers.moe.fused_moe_triton.layer import FusedMoE
from fluxserve.backend.layers.moe.token_dispatcher import DeepEPDispatcher
from fluxserve.backend.layers.moe.topk import TopK
from fluxserve.backend.layers.norm import RMSNorm
from fluxserve.backend.layers.quantization.base_config import QuantizationConfig
from fluxserve.backend.layers.rotary_embedding import get_rope
from fluxserve.backend.layers.utils import PPMissingLayer
from fluxserve.backend.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

from fluxserve.backend.execution.forward_batch_info import ForwardBatch, PPProxyTensors
from fluxserve.backend.execution.cuda_graph_runner import get_is_capture_mode
from fluxserve.backend.utils.runtime_utils import (
    add_prefix,
    is_cuda,
    is_non_idle_and_non_empty,
    make_layers,
    tqdm_progress as tqdm,
)


logger = logging.getLogger(__name__)
_is_cuda = is_cuda()


class H2Embed:
    def __init__(self, embedding: VocabParallelEmbedding, tau: float = 1.0):
        """
        W_e : token embedding weights [V, d]
        tau : temperature; lower values yield sharper distributions
        """
        self.embedding = embedding
        self.tau = tau

    def __call__(
        self,
        x: torch.Tensor,
        mask_index: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        iter_cont_weight: float = 0.0,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L] token ids
            mask_index: [B, L] bool tensor, True where continuous embedding should be used
            logits: [B, L, V] logits used to produce continuous embeddings
            iter_cont_weight: blending weight between continuous and discrete embeddings

        Returns:
            Embedded representations [B, L, d]
        """
        # Base discrete embedding
        result = self.embedding(x)

        # Replace selected positions with continuous embeddings
        if mask_index is not None and logits is not None:
            prob = torch.softmax(logits / self.tau, dim=-1)  # [B, L, V]
            if isinstance(self.embedding, VocabParallelEmbedding):
                shard = self.embedding.shard_indices
                shard_size = shard.org_vocab_end_index - shard.org_vocab_start_index
                local_prob = prob[
                    ..., shard.org_vocab_start_index : shard.org_vocab_end_index
                ]
                local_weight = self.embedding.weight[:shard_size]
                input_embeds_h = local_prob.to(local_weight.dtype) @ local_weight
                input_embeds_h = tensor_model_parallel_all_reduce(input_embeds_h)
            else:
                input_embeds_h = prob @ self.embedding.weight  # [B, L, d]

            # Blend continuous and discrete embeddings
            result = torch.where(
                mask_index.unsqueeze(-1),
                iter_cont_weight * input_embeds_h + 1 * result,
                result,
            )
        return result


class LLaDA2MLP(nn.Module):
    def __init__(
        self,
        intermediate_size: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: Optional[bool] = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size

        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            # reduce_results=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

        if config.hidden_act != "silu":
            raise ValueError("Unsupported activation. Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        if (self.tp_size == 1) and hidden_states.shape[0] == 0:
            return hidden_states

        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.down_proj(
            hidden_states, skip_all_reduce=use_reduce_scatter
        )
        return hidden_states


class LLaDA2Gate(nn.Module):
    def __init__(
        self,
        config,
        params_dtype: Optional[torch.dtype] = None,
        prefix: str = "",
    ):
        super().__init__()
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.weight = nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                dtype=self.params_dtype,
            ),
        )
        if getattr(config, "moe_router_enable_expert_bias", False):
            self.expert_bias = nn.Parameter(
                torch.empty((config.num_experts,), dtype=torch.get_default_dtype()),
            )
        else:
            self.expert_bias = None

    def forward(self, hidden_states):
        logits = F.linear(hidden_states.to(self.weight.dtype), self.weight, None)
        return logits


class LLaDA2SparseMoeBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.tp_size = get_tensor_model_parallel_world_size()
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size
        self.num_shared_experts = config.num_shared_experts
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.score_function = getattr(config, "score_function", None)

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        # Gate always runs at half / full precision for now.
        router_dtype = getattr(config, "router_dtype", None)
        if router_dtype is None:
            self.router_dtype = None
        elif router_dtype == "fp32":
            self.router_dtype = torch.float32
        else:
            self.router_dtype = torch.bfloat16
        # self.topk_filename = 'mini_topk_gsm8k.npy'
        # with open(self.topk_filename, 'wb') as f:
        #     pass
        # check group topk
        self.num_expert_group = getattr(config, "n_group", 0)
        self.topk_group = getattr(config, "topk_group", 0)
        if self.num_expert_group > 0 or self.topk_group > 0:
            assert (
                self.num_expert_group > 0
                and 0 < self.topk_group <= self.num_expert_group
            )
            self.use_grouped_topk = True
        else:
            self.num_expert_group = self.topk_group = None
            self.use_grouped_topk = False

        self.num_experts = config.num_experts

        self.gate = LLaDA2Gate(
            config=config,
            params_dtype=self.router_dtype,
            prefix=add_prefix("gate", prefix),
        )
        self.correction_bias = (
            self.gate.expert_bias if self.gate.expert_bias is not None else None
        )

        if self.score_function is not None:
            assert (
                self.score_function == "softmax" and self.correction_bias is None
            ) or (
                self.score_function == "sigmoid" and self.correction_bias is not None
            ), (
                "score_function and correction_bias should be in 2 combination "
                "(softmax, None) or (sigmoid, not None)"
            )

        self.topk = TopK(
            top_k=self.top_k,
            renormalize=self.norm_topk_prob,
            use_grouped_topk=self.use_grouped_topk,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
            correction_bias=self.correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
        )

        self.experts = FusedMoE(
            num_experts=self.num_experts,
            top_k=self.top_k,
            layer_id=self.layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            prefix=add_prefix("experts", prefix),
            inplace=False,
        )

        # shared expert
        if config.num_shared_experts is not None:
            if hasattr(config, "moe_shared_expert_intermediate_size"):
                intermediate_size = config.moe_shared_expert_intermediate_size
            else:
                intermediate_size = config.moe_intermediate_size
            intermediate_size *= config.num_shared_experts
            # disable tp for shared experts when enable deepep moe
            self.shared_experts = LLaDA2MLP(
                intermediate_size=intermediate_size,
                config=config,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
                **(
                    dict(tp_rank=0, tp_size=1)
                    if get_moe_expert_parallel_world_size() > 1
                    else {}
                ),
            )
        # dispatcher
        if get_moe_a2a_backend().is_deepep():
            self.ep_size = get_tensor_model_parallel_world_size()

            self.deepep_dispatcher = DeepEPDispatcher(
                group=parallel_state.get_tp_group().device_group,
                router_topk=self.top_k,
                permute_fusion=True,
                num_experts=self.num_experts,
                num_local_experts=config.num_experts // self.tp_size,
                hidden_size=config.hidden_size,
                params_dtype=config.torch_dtype,
                deepep_mode=get_deepep_mode(),
                async_finish=True, 
                return_recv_hook=True,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        if not get_moe_a2a_backend().is_deepep():
            return self.forward_normal(hidden_states, use_reduce_scatter)
        else:
            return self.forward_deepep(hidden_states, forward_batch)

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
        ]

    def _forward_shared_experts(self, hidden_states: torch.Tensor):
        shared_output = None
        if self.num_shared_experts > 0:
            shared_output = self.shared_experts(hidden_states)
        return shared_output

    def _forward_router_experts(self, hidden_states: torch.Tensor):
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        return self.experts(hidden_states, topk_output)

    @torch.compiler.disable
    def forward_normal_dual_stream(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        router_output = self._forward_router_experts(hidden_states)

        with torch.cuda.stream(self.alt_stream):
            shared_output = self._forward_shared_experts(hidden_states)
        current_stream.wait_stream(self.alt_stream)

        return router_output, shared_output

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        bsz, num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)

        DUAL_STREAM_TOKEN_THRESHOLD = 1024
        normal_ep = (
            get_moe_expert_parallel_world_size() > 1
            and get_moe_a2a_backend().is_none()
        )

        if (
            self.alt_stream is not None
            and hidden_states.shape[0] > 0
            and hidden_states.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
            and get_is_capture_mode()
        ):
            router_output, shared_output = self.forward_normal_dual_stream(
                hidden_states
            )
        else:
            shared_output = self._forward_shared_experts(hidden_states)
            router_output = self._forward_router_experts(hidden_states)

        if normal_ep and not use_reduce_scatter:
            final_hidden_states = tensor_model_parallel_all_reduce(router_output)
            if self.num_shared_experts > 0:
                final_hidden_states = final_hidden_states + shared_output
        else:
            final_hidden_states = router_output
            if self.num_shared_experts > 0:
                final_hidden_states = final_hidden_states + shared_output
            if self.tp_size > 1 and not use_reduce_scatter:
                final_hidden_states = tensor_model_parallel_all_reduce(
                    final_hidden_states
                )
        return final_hidden_states.view(bsz, num_tokens, hidden_size)

    def forward_deepep(
        self, 
        hidden_states: torch.Tensor, 
        forward_batch: ForwardBatch
    ) -> torch.Tensor:
        shared_output = None
        forward_mode = forward_batch.forward_mode
        if is_non_idle_and_non_empty(forward_mode, hidden_states):
            router_logits = self.gate(hidden_states)
            if self.num_shared_experts > 0:
                shared_output = self.shared_experts(hidden_states)

            topk_weights, topk_idx, _ = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
        else:
            topk_idx = torch.full(
                (0, self.top_k), -1, dtype=torch.int, device=hidden_states.device
            )
            topk_weights = torch.empty(
                (0, self.top_k), dtype=torch.float32, device=hidden_states.device
            )

        if self.ep_size > 1:
            (
                hidden_states,
                topk_idx,
                topk_weights,
                reorder_topk_ids,
                num_recv_tokens_per_expert,
                seg_indptr,
                masked_m,
                expected_m,
            ) = self.deepep_dispatcher.dispatch(
                hidden_states,
                topk_idx,
                topk_weights,
                forward_batch=forward_batch,
            )

        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            reorder_topk_ids=reorder_topk_ids,
            seg_indptr=seg_indptr,
            masked_m=masked_m,
            expected_m=expected_m,
            num_recv_tokens_per_expert=num_recv_tokens_per_expert,
            forward_batch=forward_batch,
        )
        if self.ep_size > 1:
            final_hidden_states = self.deepep_dispatcher.combine(
                final_hidden_states,
                topk_idx,
                topk_weights,
                forward_batch=forward_batch,
            )

        final_hidden_states *= self.routed_scaling_factor

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        return final_hidden_states


class LLaDA2Attention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_kv_heads = config.num_key_value_heads
        self.dp_size = get_attention_dp_size()
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        assert self.total_num_heads % attn_tp_size == 0
        assert self.total_num_heads >= self.total_kv_heads
        if attn_tp_size > self.total_kv_heads:
            assert attn_tp_size % self.total_kv_heads == 0

        self.num_heads = self.total_num_heads // attn_tp_size
        self.head_dim = (
            config.head_dim
            if hasattr(config, "head_dim")
            else (self.hidden_size // self.total_num_heads)
        )
        self.q_size = self.head_dim * self.num_heads

        self.num_kv_heads = max(1, self.total_kv_heads // attn_tp_size)
        self.total_kv_heads = self.num_kv_heads * attn_tp_size
        self.kv_size = max(1, self.num_kv_heads * self.head_dim)

        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        self.scale = self.head_dim**-0.5

        self.use_qk_norm = True

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("query_key_value", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        if self.use_qk_norm:
            self.query_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.dense = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("dense", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        if hasattr(config, "partial_rotary_factor"):
            self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        elif hasattr(config, "rotary_dim"):
            self.rotary_dim = config.rotary_dim
        else:
            self.rotary_dim = self.head_dim
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
            dtype=torch.float32,
        )

        self.alt_stream = alt_stream
        self.attention_forward = AttentionForward(
            AttentionForwardConfig(
                layer_id=self.layer_id,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                num_key_value_groups=self.num_key_value_groups,
                scale=self.scale,
                alt_stream=self.alt_stream,
            )
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values=None,
        replace_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        bsz, q_len, _ = hidden_states.size()

        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.use_qk_norm:
            q, k = apply_qk_norm(
                q,
                k,
                query_layernorm=self.query_layernorm,
                key_layernorm=self.key_layernorm,
                head_dim=self.head_dim,
                alt_stream=self.alt_stream,
            )

        q, k = self.rotary_emb(
            positions.flatten(),
            q.flatten(0, 1),
            k.flatten(0, 1),
            fused_set_kv_buffer_arg=None,
        )

        q = q.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        attn_output, present_key_values = self.attention_forward.forward(
            q,
            k,
            v,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attention_mask=attention_mask,
            forward_batch=forward_batch,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output, _ = self.dense(attn_output)
        return attn_output, present_key_values


class LLaDA2Block(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        hidden_size = config.hidden_size

        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.dp_size = get_attention_dp_size()
        self.attention = LLaDA2Attention(
            config,
            layer_id,
            quant_config,
            reduce_results=False,
            prefix=add_prefix("attention", prefix),
            alt_stream=alt_stream,
        )
        self.layer_id = layer_id
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        self.tp_size = get_tensor_model_parallel_world_size()

        self.is_layer_sparse = self._is_layer_sparse(
            config, layer_id=layer_id, is_nextn=False
        )
        is_previous_layer_sparse = self._is_layer_sparse(
            config, layer_id=layer_id - 1, is_nextn=False
        )

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
        )

        self.is_last_layer = self.layer_id == config.num_hidden_layers - 1

        if self.is_layer_sparse:
            self.mlp = LLaDA2SparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                alt_stream=alt_stream,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = LLaDA2MLP(
                intermediate_size=config.intermediate_size,
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
        )

    def _is_layer_sparse(
        self, 
        config: PretrainedConfig, 
        layer_id: int, 
        is_nextn: bool
    ) -> bool:
        return is_nextn or (
            config.num_experts is not None and layer_id >= config.first_k_dense_replace
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        past_key_values=None,
        replace_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> torch.Tensor:
        bsz, q_len, h = hidden_states.size()
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
        )
        hidden_states, present_key_values = self.attention(
            positions=positions,
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            replace_position=replace_position,
            use_cache=use_cache,
            attention_mask=attention_mask,
            forward_batch=forward_batch,
        )
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
        )
        if self.is_layer_sparse:
            hidden_states = self.mlp(
                hidden_states,
                forward_batch=forward_batch,
                use_reduce_scatter=False,
            )
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
        )

        return hidden_states, residual, present_key_values


class LLaDA2Model(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = ".",
    ):
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.hidden_size
        self.full_word_embeddings = VocabParallelEmbedding(
            self.vocab_size,
            self.embed_dim,
            quant_config=quant_config,
            prefix=add_prefix("full_word_embeddings", prefix),
        )
        if self.pp_group.is_first_rank:
            self.word_embeddings = VocabParallelEmbedding(
                self.vocab_size,
                self.embed_dim,
                quant_config=quant_config,
                prefix=add_prefix("word_embeddings", prefix),
            )
        else:
            self.word_embeddings = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: LLaDA2Block(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(self.embed_dim, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        past_key_values=None,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        replace_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.word_embeddings(input_ids.clone())
            else:
                hidden_states = input_embeds.clone()
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        all_present_key_values = []
        for i in range(self.start_layer, self.end_layer):
            # with get_global_expert_distribution_recorder().with_current_layer(i):
            layer = self.layers[i]
            hidden_states, residual, present_key_values = layer(
                positions,
                hidden_states,
                residual,
                past_key_values[i] if past_key_values is not None else None,
                replace_position=replace_position,
                use_cache=use_cache,
                attention_mask=attention_mask,
                forward_batch=forward_batch,
            )
            if use_cache:
                all_present_key_values.extend(present_key_values)
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states = hidden_states + residual
                hidden_states = self.norm(hidden_states)
            return hidden_states, all_present_key_values


class LLaDA2LLM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        expert_map_path: str = "",
    ):
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        alt_stream = torch.cuda.Stream() if _is_cuda else None

        self.model = LLaDA2Model(
            config,
            quant_config,
            alt_stream=alt_stream,
            prefix=add_prefix("model", ""),
        )
        self.device = torch.device('cpu')
        self.expert_map_path = expert_map_path

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix="lm_head",
        )
        if config.tie_word_embeddings:
            self.lm_head.tie_weights(self.model.word_embeddings)
        self._lm_head_sharded_to_full_mapping: Optional[torch.Tensor] = None

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def get_embed_and_head(self):
        """Used by the eagle_worker."""
        return self.model.word_embeddings.weight, self.lm_head.weight
    
    def set_embed_and_head(self, embed, head):
        """Used by the eagle_worker."""
        del self.model.word_embeddings.weight
        del self.lm_head.weight
        self.model.word_embeddings.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @staticmethod
    def _is_vocab_parallel_weight(name: str) -> bool:
        return name in {
            "model.word_embeddings.weight",
            "model.full_word_embeddings.weight",
            "lm_head.weight",
        }

    def _get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.quant_config is not None and not hasattr(self.lm_head, "weight"):
            local_logits = self.lm_head.quant_method.apply(self.lm_head, hidden_states)
        else:
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
                sharded_to_full_mapping = self.lm_head.get_sharded_to_full_mapping()
                if sharded_to_full_mapping is not None:
                    mapping = torch.tensor(
                        sharded_to_full_mapping,
                        device=local_logits.device,
                        dtype=torch.long,
                    )
                    self._lm_head_sharded_to_full_mapping = mapping
            if mapping is not None:
                local_logits = local_logits.index_select(-1, mapping)

        return local_logits[..., : self.config.vocab_size].float()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        past_key_values=None,
        replace_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> MoeCausalLMOutputWithPast:
        self.device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            length = (
                input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
            )
            batch_size = (
                input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
            )
            if replace_position is not None:
                position_ids = (
                    torch.arange(
                        replace_position[0],
                        replace_position[1],
                        device=self.device,
                        dtype=torch.long,
                    )
                    .unsqueeze(0)
                    .repeat(batch_size, 1)
                )
            else:
                position_ids = (
                    torch.arange(length, device=self.device, dtype=torch.long)
                    .unsqueeze(0)
                    .repeat(batch_size, 1)
                )

        hidden_states, present_key_values = self.model(
            input_ids,
            position_ids,
            past_key_values,
            inputs_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
            replace_position=replace_position,
            use_cache=use_cache,
            attention_mask=attention_mask,
            forward_batch=forward_batch,
        )
        logits = self._get_logits(hidden_states)
        return MoeCausalLMOutputWithPast(
            logits=logits,
            past_key_values=present_key_values,
            hidden_states=hidden_states,
        )


    def apply_state_dicts(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
       
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())

        new_state_dict_keys = weights.keys()
        self_state_dict_keys = params_dict.keys()
        unused_keys = []
        for key in new_state_dict_keys:
           if key not in self_state_dict_keys:
               unused_keys.append(key)

        not_inited_keys = []
        for key in self_state_dict_keys:
           if key not in new_state_dict_keys:
               not_inited_keys.append(key) 

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        for key, value in weights.items():
            if key not in params_dict:
                print(f"not match:{key}")
                continue
            if self._is_vocab_parallel_weight(key):
                continue
            if self.quant_config is not None:
                # unsqueeze to match params shape
                if value.dim() == 0 and params_dict[key].dim() == 1:
                    if params_dict[key].shape[0] == 1:
                        # [] -> [1]
                        value = value.unsqueeze(0)    
                    elif params_dict[key].shape[0] == 2:
                        # [] -> [1] -> [2]
                        value = value.unsqueeze(0).repeat(2)    
                    elif params_dict[key].shape[0] == 3:
                        # [] -> [1] -> [2]
                        value = value.unsqueeze(0).repeat(3)    
                    weights[key] = value 
            if value.shape != params_dict[key].shape:
                if not re.search(r'query_key_value.weight$', key):
                    mismatch_dim = 0 if value.shape[0] != params_dict[key].shape[0] else 1
                    if mismatch_dim==0:
                        part_size = params_dict[key].shape[0]
                        weights[key] = (
                            value[tp_rank * part_size : (tp_rank + 1) * part_size] 
                            if self.quant_config is None 
                            else value[tp_rank * part_size : (tp_rank + 1) * part_size].contiguous()    
                        )    
                    else:
                        part_size = params_dict[key].shape[1]
                        weights[key] = (
                            value[:, tp_rank * part_size : (tp_rank + 1) * part_size] 
                            if self.quant_config is None 
                            else value[:, tp_rank * part_size : (tp_rank + 1) * part_size].contiguous() 
                        )  
                    if weights[key].shape != params_dict[key].shape:
                        print('shape mismatch fixed:', key, weights[key].shape, params_dict[key].shape)
                else:
                    attn_tp_rank = get_attention_tp_rank()
                    attn_tp_size = get_attention_tp_size()
                    hidden_size = self.config.hidden_size
                    total_num_heads = self.config.num_attention_heads
                    total_kv_heads = self.config.num_key_value_heads
                    q_dim = hidden_size
                    q_part = q_dim // attn_tp_size
                    q_weight = value[attn_tp_rank * q_part : (attn_tp_rank + 1) * q_part]
                    if attn_tp_size > total_kv_heads:
                        n_replica = attn_tp_size//total_kv_heads
                        kv_dim = hidden_size * total_kv_heads // total_num_heads
                        kv_part = kv_dim // total_kv_heads
                        k_weight = value[q_dim + (attn_tp_rank//n_replica) * kv_part : q_dim + ((attn_tp_rank//n_replica) + 1) * kv_part]
                        v_weight = value[q_dim + kv_dim + (attn_tp_rank//n_replica) * kv_part : q_dim + kv_dim + ((attn_tp_rank//n_replica) + 1) * kv_part]
                    else:
                        kv_dim = hidden_size * total_kv_heads // total_num_heads
                        kv_part = kv_dim // attn_tp_size
                        k_weight = value[q_dim + attn_tp_rank * kv_part : q_dim + (attn_tp_rank + 1) * kv_part]
                        v_weight = value[q_dim + kv_dim + attn_tp_rank * kv_part : q_dim + kv_dim + (attn_tp_rank + 1) * kv_part]
                    weights[key] = torch.cat([q_weight, k_weight, v_weight], dim=0)
                    assert weights[key].shape == params_dict[key].shape

        params_dict = dict(self.named_parameters())
        buffer_dict = dict(self.named_buffers())
        for name, loaded_weight in weights.items():
            if name in params_dict:
                param = params_dict[name]
                if self._is_vocab_parallel_weight(name) and hasattr(
                    param, "weight_loader"
                ):
                    param.weight_loader(param, loaded_weight)
                else:
                    param.data = loaded_weight
            elif name in buffer_dict:
                buffer = buffer_dict[name]
                buffer.data = loaded_weight
            else:
                print('params not matching:', name)


        if not is_nextn:
            self.routed_experts_weights_of_layer = {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if not isinstance(layer, PPMissingLayer)
                and isinstance(layer.mlp, LLaDA2SparseMoeBlock)
            }
    
    def init_h2e_module(self):
        self.h2e = H2Embed(self.model.full_word_embeddings, tau=1.0)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        num_groups = getattr(config, "n_group", 0)
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None if num_groups == 0 else num_groups,
        )
        
    def after_loading(self):
        for name, module in self.named_modules():
            if hasattr(module, "quant_method") and module.quant_method is not None and hasattr(module.quant_method, "process_weights_after_loading"):
                if hasattr(module, "weight_scale") and module.weight_scale is not None:
                    if module.weight_scale.dim() == 0:
                        print(f"Fixing scalar weight_scale for {name}")
                        module.weight_scale.data = module.weight_scale.data.unsqueeze(0)
                if hasattr(module, "input_scale") and module.input_scale is not None:
                    if module.input_scale.dim() == 0:
                        print(f"Fixing scalar input_scale for {name}")
                        module.input_scale.data = module.input_scale.data.unsqueeze_(0)
                module.quant_method.process_weights_after_loading(module)
    
    def after_processing(self):
        if self.quant_config is not None:
            self.after_loading()

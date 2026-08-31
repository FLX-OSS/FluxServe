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

import torch
import torch.distributed as dist

from fluxserve.backend.configs.model_config import ModelConfig
from fluxserve.backend.execution.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    RunnerConfig,
)
from fluxserve.backend.execution.runners.utils import (
    DecodeEditBudget,
    align_exp2,
    gather_blocks,
    select_batch_sequences_by_mask_number,
)
from fluxserve.backend.layers.dp_attention import (
    DpPaddingMode,
    get_attention_dp_size,
    get_attention_tp_size,
    set_dp_buffer_len,
)
from fluxserve.backend.managers.kvcache import PagedKVCache, TokenArray
from fluxserve.backend.execution.runners.base import ModelRunner
from fluxserve.backend.utils.server_args import ServerArgs

logger = logging.getLogger(__name__)


def block_causal_mask(block_length: int):
    def mask_mod(b, h, q_idx, kv_idx):
        del b, h
        return kv_idx // block_length <= q_idx // block_length

    return mask_mod


class BlockDiffusionRunner(ModelRunner):
    """Model runner for block diffusion generation."""

    def __init__(
        self,
        model_config: ModelConfig,
        server_args: ServerArgs,
        runner_config: RunnerConfig | None = None,
        device: str = "cuda",
        _allow_flashinfer: bool = False,
    ):
        super().__init__(model_config, server_args, runner_config, device)
        if self.runner_config.attention_backend == "flashinfer" and not _allow_flashinfer:
            raise ValueError(
                "BlockDiffusionRunner supports only attention_backend='sdpa' or "
                "'flex'. Use FlashInferDiffusionRunner for FlashInfer."
            )
        self._flex_prefill_mask_cache = {}

    def _use_paged_kv_cache(self) -> bool:
        return self.runner_config.kv_cache_layout == "paged"

    def preprocess_inputs(self, prompts):
        prompt_length = prompts.shape[1]
        block_length = self.runner_config.block_length
        gen_length = self.runner_config.gen_length
        total_length = (
            (prompt_length + gen_length + block_length - 1) // block_length
        ) * block_length
        new_gen_length = total_length - prompt_length
        if total_length > self.max_length:
            self.max_length = total_length
        max_cache_length = max(self.max_length, prompt_length + gen_length)
        mask_length = (
            (max_cache_length + block_length - 1) // block_length
        ) * block_length
        attn_mask_num_blocks = mask_length // block_length
        return total_length, new_gen_length, attn_mask_num_blocks

    def build_block_attention_mask(self, num_blocks, mini_batch_size):
        block_mask = torch.tril(
            torch.ones(num_blocks, num_blocks, device=self.device, dtype=torch.bool)
        )
        return (
            block_mask.repeat_interleave(self.block_length, dim=0)
            .repeat_interleave(self.block_length, dim=1)
            .unsqueeze(0)
            .repeat(mini_batch_size, 1, 1)
        )

    def get_flex_prefill_attention_mask(self, q_len: int, kv_len: int):
        try:
            from torch.nn.attention.flex_attention import create_block_mask
        except ImportError as exc:
            raise RuntimeError(
                "attention_backend='flex' requires "
                "torch.nn.attention.flex_attention. Use a PyTorch build with "
                "FlexAttention or run with attention_backend='sdpa'."
            ) from exc

        device = torch.device(self.device)
        cache_key = (str(device), self.block_length, int(q_len), int(kv_len))
        block_mask = self._flex_prefill_mask_cache.get(cache_key)
        if block_mask is None:
            block_mask = create_block_mask(
                block_causal_mask(self.block_length),
                B=None,
                H=None,
                Q_LEN=int(q_len),
                KV_LEN=int(kv_len),
                device=device,
            )
            self._flex_prefill_mask_cache[cache_key] = block_mask
        return block_mask

    def allocate_kv_cache(self, batch_size):
        config = self.model.model.config
        num_layers = config.num_hidden_layers
        num_kv_heads = config.num_key_value_heads
        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // num_heads
        tp_size = get_attention_tp_size()
        local_kv_heads = max(1, num_kv_heads // tp_size)
        if self._use_paged_kv_cache():
            return PagedKVCache(
                num_layers=num_layers,
                batch_size=batch_size,
                local_kv_heads=local_kv_heads,
                max_length=self.max_length,
                head_dim=head_dim,
                page_size=int(self.runner_config.page_size),
                dtype=torch.bfloat16,
                device=self.device,
            )
        return torch.zeros(
            (
                num_layers,
                2,
                batch_size,
                local_kv_heads,
                self.max_length,
                head_dim,
            ),
            dtype=torch.bfloat16,
            device=self.device,
        )

    def _make_forward_batch(
        self, num_tokens: int, is_prefill: bool
    ) -> ForwardBatch | None:
        if get_attention_dp_size() == 1:
            return None

        local_num_tokens = torch.tensor(num_tokens, dtype=torch.int64, device=self.device)
        global_num_tokens_gpu = torch.empty(
            get_attention_dp_size(), dtype=torch.int64, device=self.device
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_gather_into_tensor(global_num_tokens_gpu, local_num_tokens)
        else:
            global_num_tokens_gpu[0] = local_num_tokens

        global_num_tokens = [int(x) for x in global_num_tokens_gpu.tolist()]
        dp_padding_mode = DpPaddingMode.get_dp_padding_mode(
            is_prefill, global_num_tokens
        )
        if dp_padding_mode.is_max_len():
            local_dp_buffer_len = max(global_num_tokens)
            global_dp_buffer_len = local_dp_buffer_len * get_attention_dp_size()
        else:
            local_dp_buffer_len = int(local_num_tokens.item())
            global_dp_buffer_len = sum(global_num_tokens)

        set_dp_buffer_len(
            global_dp_buffer_len,
            local_dp_buffer_len,
            global_num_tokens,
        )
        return ForwardBatch(
            forward_mode=ForwardMode.EXTEND if is_prefill else ForwardMode.DECODE,
            global_num_tokens_gpu=global_num_tokens_gpu,
            dp_padding_mode=dp_padding_mode,
            global_dp_buffer_len=global_dp_buffer_len,
        )

    def _write_prefill_kv_cache(
        self,
        *,
        global_idx,
        local_idx: int,
        sample_len: int,
        prefilling_kv: torch.Tensor,
    ):
        if isinstance(self.past_key_values, PagedKVCache):
            self.past_key_values.write_range(
                seq_id=global_idx,
                start=0,
                kv=prefilling_kv[:, :, local_idx, :, :sample_len],
            )
            return
        self.past_key_values[:, :, global_idx, :, :sample_len] = (
            prefilling_kv[:, :, local_idx, :, :sample_len]
        )
        self.past_key_values[:, :, global_idx, :, sample_len:] = 0

    @torch.no_grad()
    def generate(self, prompts):
        batch_size = prompts.shape[0]
        mini_batch_size = self.runner_config.mini_batch_size
        total_length, new_gen_length, attn_mask_num_blocks = self.preprocess_inputs(
            prompts
        )
        if self.runner_config.cache not in {"", "prefix"}:
            raise ValueError(f"Unsupported cache mode: {self.runner_config.cache}")

        attention_mask = None
        if self.runner_config.attention_backend != "flex":
            attention_mask = self.build_block_attention_mask(
                attn_mask_num_blocks, mini_batch_size
            )
        pos_ids = torch.arange(total_length, device=self.device).unsqueeze(0).repeat(
            batch_size, 1
        )

        x = TokenArray(
            prompts,
            new_gen_length,
            self.decoder.mask_id,
            self.decoder.eos_id,
            self.device,
        )

        non_mask_number = (prompts != self.decoder.mask_id).sum(dim=-1)
        decoding_start = (non_mask_number // self.block_length) * self.block_length
        use_unbounded_prefill = bool(
            getattr(self, "_use_flashinfer_paged_prefill", lambda: False)()
        )
        if self.runner_config.attention_backend != "flex" and not use_unbounded_prefill:
            decoding_start = decoding_start.clip(0, self.prefilling_limit)
        prefilling_lengths = decoding_start.clone()

        self.past_key_values = self.allocate_kv_cache(batch_size)
        num_layers = self.model.model.config.num_hidden_layers

        self._prefill_batches(
            x,
            prefilling_lengths,
            non_mask_number,
            attention_mask,
            pos_ids,
            num_layers,
            mini_batch_size,
        )
        self._decode_batches(
            x,
            decoding_start,
            total_length,
            pos_ids,
            num_layers,
            mini_batch_size,
            prompt_lengths=non_mask_number,
        )

        logger.info("The number of diffusion iterations: %s", self.num_forwards)
        return x.get_generated_tokens()

    def _prefill_batches(
        self,
        x,
        prefilling_lengths,
        non_mask_number,
        attention_mask,
        pos_ids,
        num_layers,
        mini_batch_size,
    ):
        prefilling_flag = prefilling_lengths > 0
        while torch.any(prefilling_flag):
            seq_ids = select_batch_sequences_by_mask_number(
                x, prefilling_flag, self.decoder.mask_id, mini_batch_size
            )
            if self.runner_config.attention_backend == "flex":
                prefilling_length = int(
                    torch.max(
                        (
                            (non_mask_number[seq_ids] + self.block_length - 1)
                            // self.block_length
                        )
                        * self.block_length
                    ).item()
                )
            else:
                prefilling_length = int(torch.max(prefilling_lengths[seq_ids]).item())
            prefilling_x = x.select_seqs(seq_ids)
            forward_batch = self._make_forward_batch(
                len(seq_ids) * prefilling_length, is_prefill=True
            )
            if self.runner_config.attention_backend == "flex":
                prefill_attention_mask = self.get_flex_prefill_attention_mask(
                    prefilling_length,
                    prefilling_length,
                )
            else:
                prefill_attention_mask = attention_mask[
                    : len(seq_ids), :prefilling_length, :prefilling_length
                ].contiguous()

            output = self.model(
                prefilling_x[:, :prefilling_length].contiguous(),
                use_cache=True,
                attention_mask=prefill_attention_mask,
                position_ids=pos_ids[seq_ids, :prefilling_length].contiguous(),
                forward_batch=forward_batch,
            )

            inner_shape = output.past_key_values[0].shape
            prefilling_kv = torch.stack(output.past_key_values, dim=0).reshape(
                num_layers, 2, *inner_shape
            )
            for local_idx, sample_len in enumerate(prefilling_lengths[seq_ids]):
                if self.runner_config.attention_backend == "flex":
                    sample_len = int(
                        (
                            (non_mask_number[seq_ids[local_idx]] + self.block_length - 1)
                            // self.block_length
                            * self.block_length
                        ).item()
                    )
                else:
                    sample_len = int(sample_len.item())
                global_idx = seq_ids[local_idx]
                self._write_prefill_kv_cache(
                    global_idx=global_idx,
                    local_idx=local_idx,
                    sample_len=sample_len,
                    prefilling_kv=prefilling_kv,
                )
            self.num_forwards += 1
            prefilling_flag[seq_ids] = False

    def _make_decode_loop_state(self, num_rows):
        """Per-decode-loop editing state: iteration budget and, for the
        levenshtein_joint decoder, the per-row block-refinement state."""
        needs_row_state = getattr(self.decoder, "needs_row_state", False)
        edit_budget = DecodeEditBudget(
            num_rows,
            getattr(self.runner_config, "max_post_steps", 16),
            self.block_length,
            self.device,
            max_block_iters_override=(
                self.decoder.max_block_iters if needs_row_state else None
            ),
        )
        row_state = (
            self.decoder.make_row_state(num_rows, self.block_length, self.device)
            if needs_row_state
            else None
        )
        return edit_budget, row_state

    def _decoder_editing_kwargs(
        self, seq_ids, prompt_lengths, edit_budget, row_state
    ):
        if getattr(self.decoder, "needs_editing_inputs", False):
            if prompt_lengths is None or edit_budget is None:
                raise RuntimeError(
                    "joint_threshold decoding requires prompt_lengths and an "
                    "edit budget; the decode loop was not given them."
                )
            return dict(
                prompt_lengths=prompt_lengths[seq_ids],
                allow_edit=edit_budget.allow_edit(seq_ids),
            )
        if getattr(self.decoder, "needs_row_state", False):
            if prompt_lengths is None or row_state is None:
                raise RuntimeError(
                    "levenshtein_joint decoding requires prompt_lengths and a "
                    "row state; the decode loop was not given them."
                )
            return dict(
                prompt_lengths=prompt_lengths[seq_ids],
                row_state=row_state,
                seq_ids=seq_ids,
            )
        return {}

    def _decode_batches(
        self,
        x,
        decoding_start,
        total_length,
        pos_ids,
        num_layers,
        mini_batch_size,
        prompt_lengths=None,
    ):
        edit_budget, row_state = self._make_decode_loop_state(
            decoding_start.shape[0]
        )
        decoding_flag = (decoding_start + self.block_length) <= total_length
        while torch.any(decoding_flag):
            current_cache_length = max(
                self.runner_config.max_cache_length_align,
                align_exp2(int(torch.min(decoding_start[decoding_flag]).item()) + self.block_length),
            )
            current_cache_length = min(current_cache_length, self.max_length)
            current_cache_flag = decoding_flag & (
                (decoding_start + self.block_length) <= current_cache_length
            )
            while torch.any(current_cache_flag):
                seq_ids = select_batch_sequences_by_mask_number(
                    x, current_cache_flag, self.decoder.mask_id, mini_batch_size
                )
                decoding_x = x.select_seqs(seq_ids)
                decoding_block = gather_blocks(
                    decoding_x.data, decoding_start[seq_ids], self.block_length
                )
                if isinstance(self.past_key_values, PagedKVCache):
                    decoding_past_key_values = self.past_key_values.materialize(
                        seq_ids=seq_ids,
                        length=current_cache_length,
                    )
                else:
                    decoding_past_key_values = self.past_key_values[
                        :, :, seq_ids, :, :current_cache_length
                    ]
                decoding_pos_ids = torch.arange(
                    self.block_length, device=self.device, dtype=torch.long
                ).unsqueeze(0).repeat(seq_ids.shape[0], 1)
                decoding_pos_ids = decoding_pos_ids + decoding_start[seq_ids].unsqueeze(1)
                forward_batch = self._make_forward_batch(
                    len(seq_ids) * self.block_length, is_prefill=False
                )
                output = self.model(
                    decoding_block,
                    use_cache=True,
                    position_ids=decoding_pos_ids,
                    past_key_values=decoding_past_key_values,
                    forward_batch=forward_batch,
                )
                logits = output.logits[: len(seq_ids)]

                decoder_kwargs = self._decoder_editing_kwargs(
                    seq_ids, prompt_lengths, edit_budget, row_state
                )
                self.decoder.batch_decode(
                    logits,
                    decoding_start[seq_ids],
                    decoding_x,
                    self.block_length,
                    **decoder_kwargs,
                )

                # A block is finished on an iteration whose pre-update tokens
                # had no masks AND that applied no update. For the 2.0
                # decoders this is exactly the old `no masks` test (they only
                # ever write masked positions), and it keeps the KV-commit
                # invariant: the forward that commits a block's KV is the one
                # whose input tokens are the block's final tokens.
                after = gather_blocks(
                    decoding_x.data, decoding_start[seq_ids], self.block_length
                )
                had_mask = (decoding_block == self.decoder.mask_id).any(dim=1)
                changed = (after != decoding_block).any(dim=1)
                block_finished = (~had_mask) & (~changed)
                edit_budget.update(seq_ids, had_mask, changed, block_finished)
                self._update_finished_kv_cache(
                    output,
                    seq_ids,
                    decoding_start,
                    block_finished,
                    current_cache_length,
                    num_layers,
                )

                decoding_start[seq_ids] += block_finished.long() * self.block_length
                x[seq_ids] = decoding_x.data

                if self.early_stop:
                    # Branchless on purpose: an `if eos_mask.any()` here forces
                    # a GPU->CPU sync every decode iteration.
                    eos_mask = torch.any(
                        x[seq_ids] == self.decoder.eos_id, dim=1
                    ) & block_finished
                    decoding_start[seq_ids] = torch.where(
                        eos_mask,
                        torch.full_like(decoding_start[seq_ids], total_length),
                        decoding_start[seq_ids],
                    )
                    decoding_flag[seq_ids] = decoding_flag[seq_ids] & ~eos_mask

                self.num_forwards += 1
                decoding_flag = decoding_flag & (
                    (decoding_start + self.block_length) <= total_length
                )
                current_cache_flag = decoding_flag & (
                    (decoding_start + self.block_length) <= current_cache_length
                )

    def _update_finished_kv_cache(
        self,
        output,
        seq_ids,
        decoding_start,
        block_finished,
        current_cache_length,
        num_layers,
    ):
        if not torch.any(block_finished):
            return
        inner_shape = output.past_key_values[0].shape
        decoding_kv = torch.stack(output.past_key_values, dim=0).reshape(
            num_layers, 2, *inner_shape
        )[:, :, : block_finished.shape[0]]
        if isinstance(self.past_key_values, PagedKVCache):
            for local_idx in block_finished.nonzero(as_tuple=True)[0].tolist():
                self.past_key_values.write_range(
                    seq_id=seq_ids[local_idx],
                    start=int(decoding_start[seq_ids[local_idx]].item()),
                    kv=decoding_kv[
                        :,
                        :,
                        local_idx,
                        :,
                        current_cache_length
                        - self.block_length : current_cache_length,
                    ],
                )
            return
        finished_seq_ids = seq_ids[block_finished]
        block_positions = decoding_start[finished_seq_ids].unsqueeze(1) + torch.arange(
            self.block_length, device=self.device
        )
        kv_slice = decoding_kv.permute(2, 4, 0, 1, 3, 5)[
            block_finished,
            current_cache_length - self.block_length : current_cache_length,
        ]
        self.past_key_values[
            :, :, finished_seq_ids.unsqueeze(1), :, block_positions.long()
        ] = kv_slice

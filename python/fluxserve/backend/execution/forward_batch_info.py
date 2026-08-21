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

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Dict, Literal, Optional, Sequence


class ForwardMode(IntEnum):
    EXTEND = auto()
    DECODE = auto()
    MIXED = auto()
    IDLE = auto()
    TARGET_VERIFY = auto()
    DRAFT_EXTEND = auto()

    def is_extend(self) -> bool:
        return self == ForwardMode.EXTEND

    def is_decode(self) -> bool:
        return self == ForwardMode.DECODE

    def is_mixed(self) -> bool:
        return self == ForwardMode.MIXED

    def is_idle(self) -> bool:
        return self == ForwardMode.IDLE

    def is_extend_or_mixed(self) -> bool:
        return self == ForwardMode.EXTEND or self == ForwardMode.MIXED

    def is_target_verify(self) -> bool:
        return self == ForwardMode.TARGET_VERIFY

    def is_draft_extend(self) -> bool:
        return self == ForwardMode.DRAFT_EXTEND

    def is_speculative(self) -> bool:
        return self == ForwardMode.TARGET_VERIFY or self == ForwardMode.DRAFT_EXTEND

    def is_decode_or_idle(self) -> bool:
        return self == ForwardMode.DECODE or self == ForwardMode.IDLE

    @staticmethod
    def decode_or_target_verify(
        *,
        has_drafter: bool = False,
        use_target_verify: bool = False,
    ) -> "ForwardMode":
        return (
            ForwardMode.TARGET_VERIFY
            if has_drafter and use_target_verify
            else ForwardMode.DECODE
        )

    @staticmethod
    def from_num_extends(
        num_extends: int,
        batch_size: int,
        *,
        has_drafter: bool = False,
        use_target_verify: bool = False,
    ) -> "ForwardMode":
        if batch_size <= 0:
            return ForwardMode.IDLE
        if num_extends > 0:
            return ForwardMode.MIXED if num_extends < batch_size else ForwardMode.EXTEND
        return ForwardMode.decode_or_target_verify(
            has_drafter=has_drafter,
            use_target_verify=use_target_verify,
        )


class CaptureHiddenMode(IntEnum):
    NULL = auto()
    FULL = auto()
    LAST = auto()

    def need_capture(self) -> bool:
        return self != CaptureHiddenMode.NULL

    def is_full(self) -> bool:
        return self == CaptureHiddenMode.FULL

    def is_last(self) -> bool:
        return self == CaptureHiddenMode.LAST


@dataclass
class RunnerConfig:
    gen_length: int = 1024
    block_length: int = 64
    prefilling_limit: int = 128
    mini_batch_size: int = 1
    max_length: int = 2048
    prefill_lengths: Sequence[int] = field(default_factory=lambda: (128,))
    cache_lengths: Sequence[int] = field(default_factory=lambda: (128,))
    decoding_lengths: Sequence[int] = field(default_factory=tuple)
    supported_batch_sizes: Sequence[int] = field(default_factory=lambda: (1,))
    enable_cuda_graph: bool = False
    enable_prefill_cuda_graph: bool = False
    enable_decode_cuda_graph: bool = False
    cuda_graph_capture_sizes: Sequence[int] = field(
        default_factory=lambda: (64, 128, 256, 512, 1024)
    )
    cuda_graph_log_callback: Any = None
    enable_compile: bool = False
    use_cross_block: bool = False
    early_stop: bool = True
    cache: str = ""
    max_cache_length_align: int = 128
    parallel_decoding: str = "hierarchy"
    threshold: float = 0.9
    low_threshold: float = 0.3
    use_credit: bool = False
    mask_id: int = 156895
    eos_id: int = 156892
    attention_backend: str = "sdpa"
    flashinfer_decode_batch_mode: str = "max_batch"
    decode_cuda_graph_mode: str = "decomposed"
    cuda_graph_capture_batch_sizes: Sequence[int] | None = None
    flashinfer_prefill_mode: str = "dense"
    flashinfer_cache_mode: str = "dense"
    kv_cache_layout: Literal["dense", "paged"] = "dense"
    page_size: int | None = None
    canvas_length: int | None = None
    max_denoising_steps: int | None = None
    t_min: float | None = None
    t_max: float | None = None
    entropy_bound: float | None = None
    confidence_threshold: float | None = None
    stability_threshold: int | None = None

    def __post_init__(self):
        if self.enable_cuda_graph:
            self.enable_prefill_cuda_graph = True
            self.enable_decode_cuda_graph = True
        self.enable_cuda_graph = bool(
            self.enable_prefill_cuda_graph or self.enable_decode_cuda_graph
        )
        if self.attention_backend not in {"sdpa", "flex", "flashinfer"}:
            raise ValueError(
                "attention_backend must be one of 'sdpa', 'flex', or 'flashinfer', "
                f"got {self.attention_backend!r}"
            )
        if self.flashinfer_decode_batch_mode not in {"default", "max_batch"}:
            raise ValueError(
                "flashinfer_decode_batch_mode must be one of 'default' or "
                f"'max_batch', got {self.flashinfer_decode_batch_mode!r}"
            )
        if self.decode_cuda_graph_mode not in {"decomposed", "padded"}:
            raise ValueError("decode_cuda_graph_mode must be 'decomposed' or 'padded'")
        if self.flashinfer_prefill_mode not in {"dense", "ragged", "paged"}:
            raise ValueError(
                "flashinfer_prefill_mode must be one of 'dense', 'ragged', or 'paged', "
                f"got {self.flashinfer_prefill_mode!r}"
            )
        if self.flashinfer_cache_mode not in {"dense", "paged"}:
            raise ValueError(
                "flashinfer_cache_mode must be one of 'dense' or 'paged', "
                f"got {self.flashinfer_cache_mode!r}"
            )
        if self.kv_cache_layout not in {"dense", "paged"}:
            raise ValueError(
                "kv_cache_layout must be one of 'dense' or 'paged', "
                f"got {self.kv_cache_layout!r}"
            )
        if (
            self.kv_cache_layout == "paged"
            and self.attention_backend == "flashinfer"
            and self.flashinfer_cache_mode != "paged"
        ):
            raise ValueError(
                "kv_cache_layout='paged' does not support "
                "attention_backend='flashinfer' unless "
                "flashinfer_cache_mode='paged'."
            )
        if self.flashinfer_cache_mode == "paged" and (
            self.attention_backend != "flashinfer" or self.kv_cache_layout != "paged"
        ):
            raise ValueError(
                "flashinfer_cache_mode='paged' requires "
                "attention_backend='flashinfer' and kv_cache_layout='paged'."
            )
        if self.page_size is None and self.kv_cache_layout == "paged":
            self.page_size = int(self.block_length)
        elif self.page_size is not None:
            self.page_size = int(self.page_size)
        if self.page_size is not None and self.page_size <= 0:
            raise ValueError(f"page_size must be positive, got {self.page_size!r}")
        if (
            self.enable_prefill_cuda_graph
            and self.page_size is not None
            and any(
                size % self.page_size != 0
                for size in self.cuda_graph_capture_sizes
            )
        ):
            raise ValueError(
                "cuda_graph_capture_sizes must be multiples of page_size="
                f"{self.page_size}, got {self.cuda_graph_capture_sizes}"
            )
        if (
            self.flashinfer_prefill_mode == "ragged"
            and self.attention_backend != "flashinfer"
        ):
            raise ValueError(
                "flashinfer_prefill_mode='ragged' requires "
                "attention_backend='flashinfer'."
            )
        if self.flashinfer_prefill_mode == "paged" and (
            self.attention_backend != "flashinfer"
            or self.kv_cache_layout != "paged"
            or self.flashinfer_cache_mode != "paged"
        ):
            raise ValueError(
                "flashinfer_prefill_mode='paged' requires "
                "attention_backend='flashinfer', kv_cache_layout='paged', "
                "and flashinfer_cache_mode='paged'."
            )
        self.prefill_lengths = tuple(int(x) for x in self.prefill_lengths)
        self.cache_lengths = tuple(int(x) for x in self.cache_lengths)
        self.supported_batch_sizes = tuple(int(x) for x in self.supported_batch_sizes)
        if self.cuda_graph_capture_batch_sizes is not None:
            self.cuda_graph_capture_batch_sizes = tuple(sorted(set(int(x) for x in self.cuda_graph_capture_batch_sizes)))
            if not self.cuda_graph_capture_batch_sizes or any(
                x <= 0 or (x != 1 and x % 2 != 0)
                for x in self.cuda_graph_capture_batch_sizes
            ):
                raise ValueError(
                    "cuda_graph_capture_batch_sizes must contain batch size 1 "
                    "or positive even batch sizes"
                )
        self.cuda_graph_capture_sizes = tuple(
            sorted(set(int(x) for x in self.cuda_graph_capture_sizes))
        )
        if not self.cuda_graph_capture_sizes or any(
            size <= 0 for size in self.cuda_graph_capture_sizes
        ):
            raise ValueError("cuda_graph_capture_sizes must contain positive lengths")
        decoding_lengths = tuple(int(x) for x in self.decoding_lengths)
        if not decoding_lengths:
            decoding_lengths = (int(self.block_length),)
        elif int(self.block_length) not in decoding_lengths:
            decoding_lengths = (*decoding_lengths, int(self.block_length))
        self.decoding_lengths = decoding_lengths


@dataclass
class GenerationBatchInfo:
    input_lengths: Sequence[int]
    padded_gen_lens: Sequence[int]
    sorted_indices: Sequence[int]
    batch_size: int
    block_length: int
    max_length: int
    prefill_lengths: Sequence[int]
    supported_batch_sizes: Sequence[int]
    outputs: list = field(default_factory=list)
    token_numbers: list[int] = field(default_factory=list)
    tpfs: list[float] = field(default_factory=list)
    tpss: list[float] = field(default_factory=list)
    fpss: list[float] = field(default_factory=list)
    denoising_steps: list[int] = field(default_factory=list)
    total_forward: int = 0
    total_token: int = 0
    total_time: float = 0.0

    @classmethod
    def from_lengths(
        cls,
        input_lengths: Sequence[int],
        padded_gen_lens: Sequence[int],
        *,
        batch_size: int,
        block_length: int,
        prefilling_limit: int,
        mini_batch_size: int,
        gen_length: int,
        unbounded_prefill: bool = False,
    ) -> "GenerationBatchInfo":
        sorted_indices = sorted(range(len(input_lengths)), key=lambda i: input_lengths[i])
        max_length = max(input_lengths) + gen_length if input_lengths else gen_length
        def _prefill_len(length: int) -> int:
            block_aligned = length // block_length * block_length
            if not unbounded_prefill:
                block_aligned = min(block_aligned, prefilling_limit)
            return int(max(block_length, block_aligned))

        prefill_lengths = sorted(
            {
                _prefill_len(int(length))
                for length in input_lengths
            }
        )
        max_power = int(mini_batch_size).bit_length() - 1
        supported_batch_sizes = [2**i for i in range(max_power + 1)]
        return cls(
            input_lengths=list(input_lengths),
            padded_gen_lens=list(padded_gen_lens),
            sorted_indices=sorted_indices,
            batch_size=batch_size,
            block_length=block_length,
            max_length=max_length,
            prefill_lengths=prefill_lengths,
            supported_batch_sizes=supported_batch_sizes,
        )

    def original_order(self, values: Sequence[Any]) -> list[Any]:
        reordered = [None] * len(self.sorted_indices)
        for sorted_pos, original_idx in enumerate(self.sorted_indices):
            reordered[original_idx] = values[sorted_pos]
        return reordered


@dataclass
class ForwardBatch:
    forward_mode: ForwardMode = ForwardMode.DECODE
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL
    num_token_non_padded: Optional[int] = None
    input_ids: Any = None
    return_logprob: bool = False
    top_logprobs_nums: Optional[list] = None
    token_ids_logprobs: Optional[list] = None
    extend_seq_lens: Any = None
    extend_seq_lens_cpu: Optional[list] = None
    extend_logprob_start_lens_cpu: Optional[list] = None
    extend_input_logprob_token_ids_gpu: Any = None
    next_token_logits_buffer: Any = None
    padded_static_len: int = -1
    is_prefill_only: bool = False
    global_num_tokens_gpu: Any = None
    dp_local_start_pos: Any = None
    dp_local_num_tokens: Any = None
    global_dp_buffer_len: Optional[int] = None
    global_num_tokens_for_logprob_cpu: Any = None
    global_num_tokens_for_logprob_gpu: Any = None
    dp_padding_mode: Any = None
    flashinfer_qo_indptr: Any = None
    flashinfer_kv_indptr: Any = None
    flashinfer_kv_indptr_cpu: tuple[int, ...] = field(default_factory=tuple)
    flashinfer_kv_lens: Any = None
    flashinfer_kv_lens_cpu: tuple[int, ...] = field(default_factory=tuple)
    flashinfer_q_offsets: Any = None
    flashinfer_q_offsets_cpu: tuple[int, ...] = field(default_factory=tuple)
    flashinfer_kv_offsets: Any = None
    flashinfer_kv_offsets_cpu: tuple[int, ...] = field(default_factory=tuple)
    flashinfer_qo_indptr_cpu: tuple[int, ...] = field(default_factory=tuple)
    flashinfer_block_length: int = 0
    flashinfer_prefill_lens_cpu: tuple[int, ...] = field(default_factory=tuple)
    flashinfer_paged_kv_indices: Any = None
    flashinfer_paged_kv_last_page_len: Any = None
    flashinfer_paged_kv_indices_cpu: tuple[int, ...] = field(default_factory=tuple)
    flashinfer_paged_kv_last_page_len_cpu: tuple[int, ...] = field(default_factory=tuple)
    flashinfer_seq_ids: Any = None
    flashinfer_slot_mapping: Any = None
    flashinfer_append_indptr: Any = None
    flashinfer_append_batch_indices: Any = None
    flashinfer_append_positions: Any = None
    flashinfer_custom_mask: Any = None
    flashinfer_page_size: int = 0
    use_flashinfer_prefill: bool = False
    use_flashinfer_decode: bool = False
    use_flashinfer_paged_decode: bool = False
    use_flashinfer_paged_prefill: bool = False
    flashinfer_cuda_graph_runner: Any = None
    flashinfer_cuda_graph_dummy_page: int = -1
    flashinfer_full_prefill_graph: bool = False
    flashinfer_full_decode_graph: bool = False
    flashinfer_use_native_append: bool = False
    diffusion_gemma_phase: str | None = None
    diffusion_gemma_attention_metadata: Any = None
    diffusion_gemma_full_decode_graph: bool = False


class PPProxyTensors:
    def __init__(self, tensors: Optional[Dict[str, Any]] = None):
        self.tensors = tensors or {}

    def __getitem__(self, key: str) -> Any:
        return self.tensors[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.tensors.get(key, default)

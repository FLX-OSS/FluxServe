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

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class PagedAttentionMetadata:
    """Backend-neutral metadata for one block-diffusion attention forward.

    ``q_token_indices`` maps packed kernel output back to the padded model input.
    Prefill represents every block as a virtual varlen sequence.  Consequently
    a non-causal attention call has exactly LLaDA's block-causal semantics while
    all virtual sequences continue to share the same physical KV pages.
    """

    phase: Literal["prefill", "decode"]
    batch_size: int
    max_input_len: int
    block_length: int
    page_size: int
    q_lens_cpu: tuple[int, ...]
    q_offsets_cpu: tuple[int, ...]
    q_token_indices: torch.Tensor
    qo_indptr: torch.Tensor
    kv_lens: torch.Tensor
    page_table: torch.Tensor
    slot_mapping: torch.Tensor
    max_q_len: int
    max_kv_len: int

    @property
    def is_identity_mapping(self) -> bool:
        # The builder preserves request/token order; no padding means no gather.
        return all(length == self.max_input_len for length in self.q_lens_cpu)

    @property
    def num_tasks(self) -> int:
        return int(self.kv_lens.numel())

    @property
    def num_input_tokens(self) -> int:
        return int(self.q_token_indices.numel())


def build_block_diffusion_paged_metadata(
    *,
    phase: Literal["prefill", "decode"],
    q_offsets: torch.Tensor,
    q_lens: torch.Tensor,
    page_table: torch.Tensor,
    max_input_len: int,
    block_length: int,
    page_size: int,
) -> PagedAttentionMetadata:
    """Build one FA-style varlen plan without depending on a backend API."""

    if phase not in ("prefill", "decode"):
        raise ValueError(f"unsupported attention phase {phase!r}")
    if block_length <= 0 or page_size <= 0 or max_input_len <= 0:
        raise ValueError("block_length, page_size, and max_input_len must be positive")
    if page_size % 16 != 0:
        raise ValueError(
            f"paged FA4 requires page_size to be a multiple of 16, got {page_size}"
        )
    if q_offsets.ndim != 1 or q_lens.ndim != 1:
        raise ValueError("q_offsets and q_lens must be one-dimensional")
    if q_offsets.numel() != q_lens.numel():
        raise ValueError("q_offsets and q_lens must contain one value per request")
    batch_size = int(q_lens.numel())
    if page_table.ndim != 2 or page_table.shape[0] != batch_size:
        raise ValueError(
            "page_table must have one row per request, got "
            f"shape={tuple(page_table.shape)} and batch_size={batch_size}"
        )

    q_lens_cpu = tuple(int(x) for x in q_lens.detach().cpu().tolist())
    q_offsets_cpu = tuple(int(x) for x in q_offsets.detach().cpu().tolist())
    if not q_lens_cpu or any(length <= 0 for length in q_lens_cpu):
        raise ValueError("paged attention requires positive query lengths")
    if any(length > max_input_len for length in q_lens_cpu):
        raise ValueError(
            f"query length exceeds max_input_len={max_input_len}: {q_lens_cpu}"
        )
    if any(offset < 0 for offset in q_offsets_cpu):
        raise ValueError(f"query offsets must be non-negative: {q_offsets_cpu}")
    if any(offset % block_length for offset in q_offsets_cpu):
        raise ValueError(
            "LLaDA block attention requires block-aligned query offsets: "
            f"{q_offsets_cpu}"
        )
    if any(length % block_length for length in q_lens_cpu):
        raise ValueError(
            "LLaDA block attention requires block-aligned query lengths: "
            f"{q_lens_cpu}"
        )
    if phase == "decode" and any(length != block_length for length in q_lens_cpu):
        raise ValueError(
            "LLaDA decode requires one denoising block per request, got "
            f"q_lens={q_lens_cpu}, block_length={block_length}"
        )

    device = page_table.device
    selected_page_table = page_table.to(device=device, dtype=torch.int32).contiguous()
    q_indices: list[int] = []
    task_kv_lens: list[int] = []
    task_rows: list[int] = []
    qo_values = [0]
    slot_parts: list[torch.Tensor] = []

    for row, (q_offset, q_len) in enumerate(
        zip(q_offsets_cpu, q_lens_cpu, strict=True)
    ):
        kv_len = q_offset + q_len
        required_pages = (kv_len + page_size - 1) // page_size
        if required_pages > selected_page_table.shape[1]:
            raise ValueError(
                f"request {row} needs {required_pages} pages for kv_len={kv_len}, "
                f"but page_table has {selected_page_table.shape[1]} columns"
            )

        local_positions = torch.arange(
            q_offset,
            kv_len,
            device=device,
            dtype=torch.long,
        )
        # Validate the visible prefix too, not just newly written slots.
        if torch.any(selected_page_table[row, :required_pages] < 0):
            raise ValueError(f"request {row} contains an invalid negative page id")
        physical_pages = selected_page_table[row, local_positions // page_size].long()
        slot_parts.append(physical_pages * page_size + local_positions % page_size)

        for local_start in range(0, q_len, block_length):
            local_end = local_start + block_length
            q_indices.extend(
                range(
                    row * max_input_len + local_start,
                    row * max_input_len + local_end,
                )
            )
            qo_values.append(qo_values[-1] + block_length)
            task_kv_lens.append(q_offset + local_end)
            task_rows.append(row)

    q_token_indices = torch.tensor(q_indices, device=device, dtype=torch.long)
    qo_indptr = torch.tensor(qo_values, device=device, dtype=torch.int32)
    kv_lens_tensor = torch.tensor(task_kv_lens, device=device, dtype=torch.int32)
    task_row_indices = torch.tensor(task_rows, device=device, dtype=torch.long)
    max_kv_len = max(task_kv_lens)
    max_task_pages = (max_kv_len + page_size - 1) // page_size
    visible_page_table = selected_page_table[:, :max_task_pages]
    task_page_table = (
        visible_page_table.contiguous()
        if phase == "decode"
        else visible_page_table.index_select(0, task_row_indices).contiguous()
    )
    slot_mapping = torch.cat(slot_parts).contiguous()

    return PagedAttentionMetadata(
        phase=phase,
        batch_size=batch_size,
        max_input_len=int(max_input_len),
        block_length=int(block_length),
        page_size=int(page_size),
        q_lens_cpu=q_lens_cpu,
        q_offsets_cpu=q_offsets_cpu,
        q_token_indices=q_token_indices,
        qo_indptr=qo_indptr,
        kv_lens=kv_lens_tensor,
        page_table=task_page_table,
        slot_mapping=slot_mapping,
        max_q_len=int(block_length),
        max_kv_len=max_kv_len,
    )

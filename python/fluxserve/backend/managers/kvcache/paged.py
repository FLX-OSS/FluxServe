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

import math

import torch


class PagedKVCache:
    """Flashinfer-style Paged KV cache."""

    def __init__(
        self,
        *,
        num_layers: int,
        batch_size: int,
        local_kv_heads: int,
        max_length: int,
        head_dim: int,
        page_size: int,
        num_pages: int | None = None,
        reserve_dummy_page: bool | int = False,
        dtype: torch.dtype,
        device: str | torch.device,
    ):
        if page_size <= 0:
            raise ValueError(f"page_size must be positive, got {page_size!r}")
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length!r}")

        self.num_layers = int(num_layers)
        self.batch_size = int(batch_size)
        self.local_kv_heads = int(local_kv_heads)
        self.max_length = int(max_length)
        self.head_dim = int(head_dim)
        self.page_size = int(page_size)
        self.pages_per_sequence = math.ceil(self.max_length / self.page_size)
        default_num_pages = self.batch_size * self.pages_per_sequence
        self.num_dummy_pages = int(reserve_dummy_page)
        self.dummy_page_id = (
            (int(num_pages) if num_pages is not None else default_num_pages)
            if self.num_dummy_pages
            else -1
        )
        self.uses_external_page_table = num_pages is not None
        base_num_pages = int(num_pages) if num_pages is not None else default_num_pages
        self.num_pages = base_num_pages + self.num_dummy_pages
        if self.num_pages <= 0:
            raise ValueError(f"num_pages must be positive, got {self.num_pages!r}")
        self.device = torch.device(device)

        self.data = torch.zeros(
            (
                2,
                self.num_layers,
                self.num_pages,
                self.page_size,
                self.local_kv_heads,
                self.head_dim,
            ),
            dtype=dtype,
            device=self.device,
        )
        if num_pages is None:
            self.page_table = torch.arange(
                default_num_pages,
                dtype=torch.long,
                device=self.device,
            ).reshape(self.batch_size, self.pages_per_sequence)
        else:
            self.page_table = torch.zeros(
                (self.batch_size, self.pages_per_sequence),
                dtype=torch.long,
                device=self.device,
            )
        if self.num_dummy_pages:
            self.data[:, :, self.dummy_page_id :].zero_()

    def set_page_table(self, seq_id: int, pages: list[int] | tuple[int, ...]) -> None:
        seq_id = int(seq_id)
        if seq_id < 0 or seq_id >= self.batch_size:
            raise ValueError(
                f"seq_id must be in [0, {self.batch_size}), got {seq_id}"
            )
        if len(pages) > self.pages_per_sequence:
            raise ValueError(
                "page table row exceeds sequence capacity: "
                f"got {len(pages)} pages, capacity={self.pages_per_sequence}"
            )
        if not pages:
            return
        page_tensor = torch.as_tensor(pages, dtype=torch.long, device=self.device)
        if torch.any(page_tensor <= 0):
            raise ValueError(
                f"scheduler page ids must be positive; got {list(pages)}"
            )
        if torch.any(page_tensor >= self.num_pages):
            raise ValueError(
                f"scheduler page ids must be < num_pages={self.num_pages}; "
                f"got {list(pages)}"
            )
        self.page_table[seq_id].zero_()
        self.page_table[seq_id, : page_tensor.numel()] = page_tensor

    def set_page_tables(
        self,
        seq_ids: torch.Tensor | list[int] | tuple[int, ...],
        page_rows: list[list[int]] | tuple[tuple[int, ...], ...],
    ) -> None:
        seq_ids_list = [int(x) for x in torch.as_tensor(seq_ids).detach().cpu().tolist()]
        if len(seq_ids_list) != len(page_rows):
            raise ValueError(
                "seq_ids and page_rows must have matching lengths, "
                f"got {len(seq_ids_list)} and {len(page_rows)}"
            )
        for seq_id, pages in zip(seq_ids_list, page_rows, strict=True):
            self.set_page_table(seq_id, list(pages))

    @property
    def shape(self):
        return self.data.shape

    def layer_paged_kv(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one layer's paged K/V tensors in FlashInfer NHD layout."""
        layer = int(layer_id)
        if layer < 0 or layer >= self.num_layers:
            raise ValueError(
                f"layer_id must be in [0, {self.num_layers}), got {layer_id}"
            )
        return self.data[0, layer], self.data[1, layer]

    def flashinfer_paged_metadata(
        self,
        *,
        seq_ids: torch.Tensor,
        lengths: torch.Tensor | list[int] | tuple[int, ...],
        index_dtype: torch.dtype = torch.int32,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build FlashInfer paged KV metadata for selected sequences."""
        seq_ids = seq_ids.to(device=self.device, dtype=torch.long)
        lengths_tensor = torch.as_tensor(lengths, dtype=torch.long, device=self.device)
        if lengths_tensor.numel() != seq_ids.numel():
            raise ValueError(
                "lengths must have one entry per seq_id, "
                f"got {lengths_tensor.numel()} lengths for {seq_ids.numel()} seq_ids"
            )
        if torch.any(lengths_tensor <= 0):
            raise ValueError("FlashInfer paged metadata requires positive KV lengths.")
        if torch.any(lengths_tensor > self.max_length):
            raise ValueError(
                f"KV lengths must be <= max_length={self.max_length}, "
                f"got {lengths_tensor.detach().cpu().tolist()}"
            )

        page_counts = torch.div(
            lengths_tensor + self.page_size - 1,
            self.page_size,
            rounding_mode="floor",
        )
        indptr = torch.empty(
            int(seq_ids.numel()) + 1,
            dtype=index_dtype,
            device=self.device,
        )
        indptr[0] = 0
        indptr[1:] = torch.cumsum(page_counts.to(index_dtype), dim=0)

        indices = torch.empty(
            int(indptr[-1].item()),
            dtype=index_dtype,
            device=self.device,
        )
        cursor = 0
        for seq_id, page_count in zip(seq_ids.tolist(), page_counts.tolist(), strict=True):
            page_count = int(page_count)
            selected_pages = self.page_table[int(seq_id), :page_count]
            if self.uses_external_page_table and torch.any(selected_pages <= 0):
                raise ValueError(
                    "external page table has unset page ids for "
                    f"seq_id={int(seq_id)}, page_count={page_count}"
                )
            indices[cursor : cursor + page_count] = selected_pages.to(index_dtype)
            cursor += page_count

        last_page_len = ((lengths_tensor - 1) % self.page_size + 1).to(index_dtype)
        return indptr, indices, last_page_len

    def _check_length(self, length: int):
        if length < 0 or length > self.max_length:
            raise ValueError(
                f"length must be in [0, {self.max_length}], got {length}"
            )

    def _check_range(self, start: int, length: int):
        if start < 0 or length < 0 or start + length > self.max_length:
            raise ValueError(
                "range must be inside cache capacity, "
                f"got start={start}, length={length}, max_length={self.max_length}"
            )

    def slot_mapping(
        self,
        seq_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        seq_ids = seq_ids.to(device=self.device, dtype=torch.long)
        positions = positions.to(device=self.device, dtype=torch.long)
        logical_pages = positions // self.page_size
        offsets = positions % self.page_size
        physical_pages = self.page_table[seq_ids.unsqueeze(-1), logical_pages]
        if self.uses_external_page_table and torch.any(physical_pages <= 0):
            raise ValueError("external page table has unset page ids for slot mapping")
        return physical_pages * self.page_size + offsets

    def write_range(
        self,
        *,
        seq_id: int | torch.Tensor,
        start: int,
        kv: torch.Tensor,
    ) -> None:
        length = int(kv.shape[-2])
        self._check_range(int(start), length)
        if length == 0:
            return
        seq_idx = (
            int(seq_id.item()) if isinstance(seq_id, torch.Tensor) else int(seq_id)
        )
        positions = torch.arange(
            int(start),
            int(start) + length,
            dtype=torch.long,
            device=self.device,
        )
        pages = self.page_table[seq_idx, positions // self.page_size]
        if self.uses_external_page_table and torch.any(pages <= 0):
            raise ValueError("external page table has unset page ids for write")
        offsets = positions % self.page_size
        src = kv.to(device=self.device, dtype=self.data.dtype).permute(1, 0, 3, 2, 4)
        self.data[:, :, pages, offsets] = src

    def materialize(
        self,
        *,
        seq_ids: torch.Tensor,
        length: int,
    ) -> torch.Tensor:
        length = int(length)
        self._check_length(length)
        seq_ids = seq_ids.to(device=self.device, dtype=torch.long)
        out = torch.zeros(
            (
                self.num_layers,
                2,
                int(seq_ids.numel()),
                self.local_kv_heads,
                length,
                self.head_dim,
            ),
            dtype=self.data.dtype,
            device=self.device,
        )
        if length == 0 or seq_ids.numel() == 0:
            return out

        positions = torch.arange(length, dtype=torch.long, device=self.device)
        logical_pages = positions // self.page_size
        offsets = positions % self.page_size
        for batch_idx, seq_id in enumerate(seq_ids.tolist()):
            pages = self.page_table[int(seq_id), logical_pages]
            chunk = self.data[:, :, pages, offsets]
            out[:, :, batch_idx] = chunk.permute(1, 0, 3, 2, 4)
        return out

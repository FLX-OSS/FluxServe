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

"""Decode CUDA graphs for Nemotron-Labs-Diffusion's two block phases.

A Nemotron block alternates between a bidirectional denoising forward and a
causal commit forward. The CuTe kernel specialises on ``causal``, so the flag is
baked into whatever graph records it: one capture cannot serve both. This runner
therefore captures **two variants per batch bucket** over the same token shape,
``batch x block_length``, which adds graph variants without adding a shape
class.

The two differ in more than the flag:

* the denoise graph returns logits, which the threshold decoder consumes
  outside the graph;
* the commit graph additionally returns the argmax of its final-position
  logits, which is the next block's seed, and never runs the decoder tail.

Padding rows reuse the existing reserved-dummy-page convention, so a bucket may
be replayed with fewer live rows than it was captured for -- which happens
constantly here, because rows finish denoising at different steps and only some
are ready to commit on any given iteration.
"""

from __future__ import annotations

import bisect
import time
from types import SimpleNamespace

import torch

from fluxserve.backend.execution.cuda_graph_runner import model_capture_mode
from fluxserve.backend.execution.forward_batch_info import ForwardBatch, ForwardMode
from fluxserve.backend.layers.attention.metadata import PagedAttentionMetadata

DENOISE = False
COMMIT = True


class NemotronCudaGraphRunner:
    """Capture denoise and commit graphs per batch bucket."""

    def __init__(self, batch_sizes):
        self.batch_sizes = tuple(sorted(set(map(int, batch_sizes))))
        if not self.batch_sizes or self.batch_sizes[0] < 1:
            raise ValueError("Nemotron decode graphs need positive batch buckets")
        # Keyed by (batch_size, causal).
        self.entries: dict[tuple[int, bool], SimpleNamespace] = {}
        self.cache = None
        # Created on first capture: allocating a graph pool needs a CUDA
        # context, and this object is constructed during argument validation,
        # which also happens where there is no GPU.
        self.pool = None
        self.capture_time_s = 0.0
        self.capture_memory_bytes = 0
        self.replay_count = 0
        self.padded_rows = 0

    def invalidate(self) -> None:
        if self.entries:
            torch.cuda.synchronize(self.cache.device)
        self.entries.clear()
        self.cache = None
        self.pool = torch.cuda.graph_pool_handle() if self.pool is not None else None

    @torch.inference_mode()
    def capture(self, runner) -> None:
        cache = runner.past_key_values
        if self.cache is not cache:
            self.invalidate()
            self.cache = cache
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        started = time.perf_counter()
        before = torch.cuda.memory_allocated(cache.device)
        for batch_size in reversed(self.batch_sizes):
            for causal in (DENOISE, COMMIT):
                key = (batch_size, causal)
                if key not in self.entries:
                    self.entries[key] = self._capture(runner, batch_size, causal)
        torch.cuda.synchronize(cache.device)
        self.capture_time_s += time.perf_counter() - started
        self.capture_memory_bytes = torch.cuda.memory_allocated(cache.device) - before

    def _capture(self, runner, batch_size: int, causal: bool):
        cache = self.cache
        length = int(runner.block_length)
        if cache.num_dummy_pages < batch_size or cache.page_size != length:
            raise ValueError(
                "Nemotron decode graphs need one reserved block-sized page per "
                "bucket row and page_size == block_length"
            )
        device = cache.device
        ids = torch.full(
            (batch_size, length), runner.decoder.mask_id,
            dtype=torch.long, device=device,
        )
        positions = torch.arange(length, device=device).repeat(batch_size, 1)
        dummy_pages = torch.arange(
            cache.dummy_page_id, cache.dummy_page_id + batch_size,
            device=device, dtype=torch.int32,
        )
        table = dummy_pages[:, None].expand(-1, cache.pages_per_sequence).contiguous()
        slots = (
            dummy_pages.long()[:, None] * length
            + torch.arange(length, device=device)
        ).flatten()
        metadata = PagedAttentionMetadata(
            phase="decode",
            batch_size=batch_size,
            max_input_len=length,
            block_length=length,
            page_size=length,
            q_lens_cpu=(length,) * batch_size,
            q_offsets_cpu=(0,) * batch_size,
            q_token_indices=torch.arange(batch_size * length, device=device),
            qo_indptr=torch.arange(
                batch_size + 1, device=device, dtype=torch.int32
            ) * length,
            kv_lens=torch.full(
                (batch_size,), length, device=device, dtype=torch.int32
            ),
            page_table=table,
            slot_mapping=slots,
            max_q_len=length,
            max_kv_len=cache.max_length,
            causal=causal,
        )
        batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE, paged_attention_metadata=metadata
        )
        kv = [cache.layer_paged_kv(i) for i in range(cache.num_layers)]

        def forward():
            # The paged kernel writes keys and values through `slot_mapping`,
            # so nothing is needed from a returned cache.
            hidden, _ = runner.model.model(
                ids, positions, kv, use_cache=False,
                attention_mask=None, forward_batch=batch,
            )
            logits = runner.model._get_logits(hidden)
            seed = logits[:, -1].argmax(dim=-1) if causal else None
            return logits, seed

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with model_capture_mode(), torch.cuda.stream(stream):
            for _ in range(3):
                forward()
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        # Every rank finishes eager collective warmup before any rank records
        # NCCL operations into a graph.
        runner.tp_group.barrier()
        graph = torch.cuda.CUDAGraph()
        with model_capture_mode(), torch.cuda.graph(graph, pool=self.pool, stream=stream):
            logits, seed = forward()
        torch.cuda.synchronize(device)
        return SimpleNamespace(
            graph=graph, input_ids=ids, position_ids=positions, metadata=metadata,
            dummy_pages=dummy_pages, logits=logits, seed=seed, causal=causal,
        )

    def bucket_for(self, batch_size: int) -> int:
        index = bisect.bisect_left(self.batch_sizes, batch_size)
        if batch_size < 1 or index == len(self.batch_sizes):
            raise ValueError(
                f"no Nemotron decode graph bucket for batch {batch_size}; "
                f"buckets are {self.batch_sizes}"
            )
        return self.batch_sizes[index]

    def can_replay(self, runner, *, batch_size: int, length: int,
                   causal: bool) -> bool:
        if not self.entries or self.cache is not runner.past_key_values:
            return False
        if length != int(runner.block_length) or batch_size < 1:
            return False
        if batch_size > self.batch_sizes[-1]:
            return False
        return (self.bucket_for(batch_size), causal) in self.entries

    @torch.inference_mode()
    def replay(self, runner, *, input_ids, position_ids, seq_ids, causal: bool):
        if self.cache is not runner.past_key_values:
            raise RuntimeError(
                "Nemotron graph KV allocation changed; recapture before replay"
            )
        actual = int(input_ids.shape[0])
        size = self.bucket_for(actual)
        entry = self.entries[(size, causal)]
        length = int(runner.block_length)
        if input_ids.shape != (actual, length) or position_ids.shape != input_ids.shape:
            raise ValueError("Nemotron graphs expect one full block per row")

        entry.input_ids[:actual].copy_(input_ids)
        entry.position_ids[:actual].copy_(position_ids)
        metadata = entry.metadata
        metadata.page_table[:actual].copy_(
            self.cache.page_table.index_select(0, seq_ids)
        )
        metadata.kv_lens[:actual].copy_(position_ids[:, -1] + 1)
        # Absolute position over page_size gives the page even when a block
        # straddles two pages, which it does whenever the prompt length is not
        # a multiple of the block length.
        pages = metadata.page_table[:actual].gather(1, position_ids // length)
        metadata.slot_mapping.view(size, length)[:actual].copy_(
            pages.long() * length + position_ids % length
        )
        if actual < size:
            # Padding rows must not alias a live request's pages: reset any row
            # that carried real work on a previous replay.
            device = self.cache.device
            entry.input_ids[actual:].fill_(runner.decoder.mask_id)
            entry.position_ids[actual:].copy_(torch.arange(length, device=device))
            metadata.page_table[actual:].copy_(entry.dummy_pages[actual:, None])
            metadata.kv_lens[actual:].fill_(length)
            metadata.slot_mapping.view(size, length)[actual:].copy_(
                entry.dummy_pages[actual:, None].long() * length
                + torch.arange(length, device=device)
            )
        entry.graph.replay()
        self.replay_count += 1
        self.padded_rows += size - actual
        return SimpleNamespace(
            logits=entry.logits[:actual],
            seed=None if entry.seed is None else entry.seed[:actual],
        )

    def stats(self) -> dict:
        denoise = sum(1 for _, causal in self.entries if causal is DENOISE)
        commit = sum(1 for _, causal in self.entries if causal is COMMIT)
        return {
            "decode_capture_count": len(self.entries),
            "denoise_capture_count": denoise,
            "commit_capture_count": commit,
            "decode_replay_count": self.replay_count,
            "decode_fallback_count": 0,
            "decode_padded_rows": self.padded_rows,
            "capture_time_s": self.capture_time_s,
            "capture_memory_bytes": self.capture_memory_bytes,
        }


__all__ = ["NemotronCudaGraphRunner"]

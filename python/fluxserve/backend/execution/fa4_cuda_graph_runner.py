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


"""Native FluxServe full-model FA4 decode graphs with mutable paged KV inputs."""
from __future__ import annotations

import bisect
import time
from types import SimpleNamespace

import torch

from fluxserve.backend.execution.cuda_graph_runner import model_capture_mode
from fluxserve.backend.execution.forward_batch_info import ForwardBatch, ForwardMode
from fluxserve.backend.layers.attention.metadata import PagedAttentionMetadata


class FA4CudaGraphRunner:
    """Capture once per batch bucket and KV allocation; KV lengths stay dynamic.

    Outputs alias graph-owned buffers and must be consumed before another replay.
    Each padding row owns a distinct reserved page, outside the scheduler pool.
    Only TP1/EP1 is currently supported by the enclosing runner.
    """

    def __init__(self, batch_sizes):
        self.batch_sizes = tuple(sorted(set(map(int, batch_sizes))))
        if not self.batch_sizes or self.batch_sizes[0] < 1:
            raise ValueError("FA4 decode graph needs positive batch buckets")
        self.entries = {}
        self.cache = None
        self.pool = torch.cuda.graph_pool_handle()
        self.capture_time_s = 0.0
        self.replay_count = 0
        self.padded_rows = 0

    def invalidate(self):
        if self.entries:
            torch.cuda.synchronize(self.cache.device)
        self.entries.clear()
        self.cache = None
        self.pool = torch.cuda.graph_pool_handle()

    @torch.inference_mode()
    def capture(self, runner):
        cache = runner.past_key_values
        if self.cache is not cache:
            self.invalidate()
            self.cache = cache
        started = time.perf_counter()
        before = torch.cuda.memory_allocated(cache.device)
        for batch_size in reversed(self.batch_sizes):
            if batch_size not in self.entries:
                self.entries[batch_size] = self._capture_bucket(runner, batch_size)
        torch.cuda.synchronize(cache.device)
        self.capture_time_s += time.perf_counter() - started
        self.capture_memory_bytes = torch.cuda.memory_allocated(cache.device) - before

    def _capture_bucket(self, runner, batch_size):
        cache, length = self.cache, int(runner.block_length)
        if cache.num_dummy_pages < batch_size or cache.page_size != length:
            raise ValueError("FA4 graph requires one reserved block-sized page per bucket row")
        device = cache.device
        ids = torch.full((batch_size, length), runner.decoder.mask_id,
                         dtype=torch.long, device=device)
        positions = torch.arange(length, device=device).repeat(batch_size, 1)
        dummy_pages = torch.arange(cache.dummy_page_id, cache.dummy_page_id + batch_size,
                                   device=device, dtype=torch.int32)
        table = dummy_pages[:, None].expand(-1, cache.pages_per_sequence).contiguous()
        slots = (dummy_pages.long()[:, None] * length + torch.arange(length, device=device)).flatten()
        metadata = PagedAttentionMetadata(
            phase="decode", batch_size=batch_size, max_input_len=length,
            block_length=length, page_size=length, q_lens_cpu=(length,) * batch_size,
            q_offsets_cpu=(0,) * batch_size,
            q_token_indices=torch.arange(batch_size * length, device=device),
            qo_indptr=torch.arange(batch_size + 1, device=device, dtype=torch.int32) * length,
            kv_lens=torch.full((batch_size,), length, device=device, dtype=torch.int32),
            page_table=table, slot_mapping=slots, max_q_len=length,
            max_kv_len=cache.max_length,
        )
        batch = ForwardBatch(forward_mode=ForwardMode.DECODE,
                             paged_attention_metadata=metadata)
        kv = [cache.layer_paged_kv(i) for i in range(cache.num_layers)]
        prompt = torch.ones_like(ids, dtype=torch.bool)
        allow_edit = torch.zeros(batch_size, device=device, dtype=torch.bool)
        fused = bool(getattr(runner.decoder, "graph_fused_step", False))

        def forward():
            hidden, _ = runner.model.model(ids, positions, kv, use_cache=False,
                                           attention_mask=None, forward_batch=batch)
            logits = runner.model._get_logits(hidden)
            step = runner.decoder.graph_step(logits, ids, prompt, allow_edit) if fused else None
            return logits, step

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with model_capture_mode(), torch.cuda.stream(stream):
            for _ in range(3):
                forward()
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with model_capture_mode(), torch.cuda.graph(graph, pool=self.pool, stream=stream):
            logits, step = forward()
        torch.cuda.synchronize(device)
        return SimpleNamespace(graph=graph, input_ids=ids, position_ids=positions,
                               metadata=metadata, dummy_pages=dummy_pages,
                               prompt_positions=prompt, allow_edit=allow_edit,
                               logits=logits, step=step)

    @torch.inference_mode()
    def replay(self, runner, input_ids, position_ids, seq_ids, prompt_positions=None, allow_edit=None):
        if self.cache is not runner.past_key_values:
            raise RuntimeError("FA4 graph KV allocation changed; recapture before replay")
        actual = input_ids.shape[0]
        index = bisect.bisect_left(self.batch_sizes, actual)
        if actual < 1 or index == len(self.batch_sizes):
            raise ValueError(f"No FA4 decode graph bucket for batch {actual}")
        size = self.batch_sizes[index]
        entry = self.entries[size]
        length = runner.block_length
        if input_ids.shape != (actual, length) or position_ids.shape != input_ids.shape:
            raise ValueError("FA4 graph expects one full decode block per row")
        if entry.step is not None and (prompt_positions is None or allow_edit is None):
            raise ValueError("Joint decode graph requires prompt protection and editing budget")
        entry.input_ids[:actual].copy_(input_ids)
        entry.position_ids[:actual].copy_(position_ids)
        metadata = entry.metadata
        # The full-width table is stable even as live rows grow or recycle pages.
        metadata.page_table[:actual].copy_(self.cache.page_table.index_select(0, seq_ids))
        metadata.kv_lens[:actual].copy_(position_ids[:, -1] + 1)
        pages = metadata.page_table[:actual].gather(1, position_ids // length)
        metadata.slot_mapping.view(size, length)[:actual].copy_(pages.long() * length + position_ids % length)
        if entry.step is not None:
            entry.prompt_positions[:actual].copy_(prompt_positions)
            entry.allow_edit[:actual].copy_(allow_edit)
        if actual < size:
            # Reset rows that may have been real requests on the previous replay.
            entry.input_ids[actual:].fill_(runner.decoder.mask_id)
            entry.position_ids[actual:].copy_(torch.arange(length, device=self.cache.device))
            metadata.page_table[actual:].copy_(entry.dummy_pages[actual:, None])
            metadata.kv_lens[actual:].fill_(length)
            metadata.slot_mapping.view(size, length)[actual:].copy_(
                entry.dummy_pages[actual:, None].long() * length
                + torch.arange(length, device=self.cache.device))
            entry.prompt_positions[actual:].fill_(True)
            entry.allow_edit[actual:].fill_(False)
        entry.graph.replay()
        self.replay_count += 1
        self.padded_rows += size - actual
        return SimpleNamespace(logits=entry.logits[:actual],
                               step=None if entry.step is None else tuple(t[:actual] for t in entry.step))

    def stats(self):
        return {"decode_capture_count": len(self.entries),
                "decode_replay_count": self.replay_count,
                "decode_fallback_count": 0,
                "decode_padded_rows": self.padded_rows,
                "capture_time_s": self.capture_time_s,
                "capture_memory_bytes": getattr(self, "capture_memory_bytes", 0)}

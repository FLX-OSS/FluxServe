"""Validate full-adapter replay with changing inputs and physical page IDs."""
import os

import pytest
import torch

from fluxserve.backend.layers.attention.base import AttentionForwardConfig
from fluxserve.backend.layers.attention.fa4 import FA4PagedAttention
from fluxserve.backend.layers.attention.metadata import build_block_diffusion_paged_metadata


@pytest.mark.skipif(
    os.environ.get("FLUXSERVE_RUN_FA4_SMOKE") != "1",
    reason="set FLUXSERVE_RUN_FA4_SMOKE=1 on a supported GPU",
)
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_adapter_graph_replay_with_new_inputs_and_pages(phase):
    assert torch.cuda.is_available()
    torch.manual_seed(42)
    device, dtype = "cuda", torch.bfloat16
    lengths = (128, 64) if phase == "prefill" else (64, 64)
    offsets = (0, 0) if phase == "prefill" else (128, 192)
    table = torch.tensor([[5, 2, 7, 1], [6, 3, 0, 4]], device=device)

    def metadata_for(pages):
        return build_block_diffusion_paged_metadata(
            phase=phase, q_offsets=torch.tensor(offsets, device=device),
            q_lens=torch.tensor(lengths, device=device), page_table=pages,
            max_input_len=max(lengths), block_length=64, page_size=64,
        )

    metadata = metadata_for(table)
    config = AttentionForwardConfig(
        layer_id=0, num_heads=16, num_kv_heads=4, head_dim=128,
        num_key_value_groups=4, scale=128**-0.5,
    )
    adapter = FA4PagedAttention(config)
    q = torch.randn(2, 16, max(lengths), 128, device=device, dtype=dtype)
    k = torch.randn(2, 4, max(lengths), 128, device=device, dtype=dtype)
    v = torch.randn_like(k)
    cache = tuple(torch.randn(8, 64, 4, 128, device=device, dtype=dtype) for _ in range(2))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        for _ in range(3):
            adapter.forward(q, k, v, cache, metadata)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        output = adapter.forward(q, k, v, cache, metadata)

    for replay in range(3):
        # Addresses and shape stay fixed; values and physical cache layout change.
        q.normal_()
        k.normal_()
        v.normal_()
        for tensor in cache:
            tensor.normal_()
        updated = metadata_for((table + replay) % 8)
        metadata.page_table.copy_(updated.page_table)
        metadata.slot_mapping.copy_(updated.slot_mapping)
        eager_cache = tuple(t.clone() for t in cache)
        with torch.inference_mode():
            expected = adapter.forward(q, k, v, eager_cache, updated)
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        for actual_kv, expected_kv in zip(cache, eager_cache):
            torch.testing.assert_close(actual_kv, expected_kv, atol=0, rtol=0)

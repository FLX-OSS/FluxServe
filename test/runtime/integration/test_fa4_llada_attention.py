import os

import pytest
import torch
import torch.nn.functional as F

from fluxserve.backend.layers.attention.base import AttentionForwardConfig
from fluxserve.backend.layers.attention.fa4 import FA4PagedAttention
from fluxserve.backend.layers.attention.metadata import (
    build_block_diffusion_paged_metadata,
)


RUN_FA4_SMOKE = "FLUXSERVE_RUN_FA4_SMOKE"


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.skipif(
    os.environ.get(RUN_FA4_SMOKE) != "1",
    reason=f"set {RUN_FA4_SMOKE}=1 to compile and run the FA4 kernel",
)
def test_fa4_paged_attention_matches_llada_block_causal_reference(phase, seed):
    if not torch.cuda.is_available():
        pytest.skip("FA4 smoke test requires CUDA")
    capability = torch.cuda.get_device_capability()
    if capability[0] not in (9, 10, 11):
        pytest.skip(f"FA4 paged KV is unsupported on compute capability {capability}")
    pytest.importorskip("flash_attn.cute")

    torch.manual_seed(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    # Production LLaDA2.x attention geometry.
    batch_size, q_heads, kv_heads, head_dim = 2, 16, 4, 128
    block_length = page_size = 64
    if phase == "prefill":
        max_input_len = 128
        q_lens = torch.tensor([128, 64], device=device)
        q_offsets = torch.zeros_like(q_lens)
    else:
        max_input_len = block_length
        q_lens = torch.tensor([block_length, block_length], device=device)
        q_offsets = torch.tensor([128, 192], device=device)
    # Non-contiguous physical page IDs exercise the actual page-table lookup.
    page_table = torch.tensor(
        [[5, 2, 7, 1], [6, 3, 0, 4]], dtype=torch.long, device=device
    )
    metadata = build_block_diffusion_paged_metadata(
        phase=phase,
        q_offsets=q_offsets,
        q_lens=q_lens,
        page_table=page_table,
        max_input_len=max_input_len,
        block_length=block_length,
        page_size=page_size,
    )
    q = torch.randn(
        batch_size, q_heads, max_input_len, head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        batch_size, kv_heads, max_input_len, head_dim, device=device, dtype=dtype
    )
    v = torch.randn_like(k)
    k_cache = torch.zeros(8, page_size, kv_heads, head_dim, device=device, dtype=dtype)
    v_cache = torch.zeros_like(k_cache)
    prefix_k = []
    prefix_v = []
    for row, q_offset in enumerate(q_offsets.tolist()):
        row_prefix_k = torch.randn(
            kv_heads, q_offset, head_dim, device=device, dtype=dtype
        )
        row_prefix_v = torch.randn_like(row_prefix_k)
        prefix_k.append(row_prefix_k)
        prefix_v.append(row_prefix_v)
        if q_offset:
            positions = torch.arange(q_offset, device=device)
            pages = page_table[row, positions // page_size]
            offsets = positions % page_size
            k_cache[pages, offsets] = row_prefix_k.transpose(0, 1)
            v_cache[pages, offsets] = row_prefix_v.transpose(0, 1)

    config = AttentionForwardConfig(
        layer_id=0,
        num_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        num_key_value_groups=q_heads // kv_heads,
        scale=head_dim**-0.5,
    )
    actual = FA4PagedAttention(config).forward(q, k, v, (k_cache, v_cache), metadata)

    expected = torch.zeros_like(actual)
    groups = q_heads // kv_heads
    for row, length in enumerate(q_lens.tolist()):
        logical_k = torch.cat((prefix_k[row], k[row, :, :length]), dim=1)
        logical_v = torch.cat((prefix_v[row], v[row, :, :length]), dim=1)
        repeated_k = logical_k.repeat_interleave(groups, dim=0)
        repeated_v = logical_v.repeat_interleave(groups, dim=0)
        q_offset = int(q_offsets[row])
        for start in range(0, length, block_length):
            end = start + block_length
            kv_end = q_offset + end
            expected[row, :, start:end] = F.scaled_dot_product_attention(
                q[row, :, start:end].float(),
                repeated_k[:, :kv_end].float(),
                repeated_v[:, :kv_end].float(),
                dropout_p=0.0,
                is_causal=False,
                scale=config.scale,
            ).to(dtype)

    error = (actual.float() - expected.float()).abs()
    print(
        f"phase={phase} seed={seed} max_abs_error={error.max().item():.8f} "
        f"mean_abs_error={error.mean().item():.8f}"
    )
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)

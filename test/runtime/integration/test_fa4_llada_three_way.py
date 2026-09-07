"""GH200 correctness test for LLaDA2.x paged block attention.

This compares three computations over the same Q/K/V tensors and physical KV
pages:

* PyTorch SDPA in FP32 (the reference);
* FluxServe's standalone FA4 adapter;
* FlashInfer BatchPrefillBlockExtend (the existing FluxServe backend).

Run with::

    FLUXSERVE_RUN_FA4_THREE_WAY=1 pytest -q -s \
      test/runtime/integration/test_fa4_llada_three_way.py
"""

from __future__ import annotations

import math
import os

import pytest
import torch
import torch.nn.functional as F

from fluxserve.backend.layers.attention.base import AttentionForwardConfig
from fluxserve.backend.layers.attention.fa4 import FA4PagedAttention
from fluxserve.backend.layers.attention.metadata import (
    build_block_diffusion_paged_metadata,
)


RUN_THREE_WAY = "FLUXSERVE_RUN_FA4_THREE_WAY"
BLOCK_LENGTH = 64
PAGE_SIZE = 64
Q_HEADS = 16
KV_HEADS = 4
HEAD_DIM = 128


def _pack_valid(x: torch.Tensor, lengths: tuple[int, ...]) -> torch.Tensor:
    # Model layout [batch, heads, max_q_len, dim] -> varlen [tokens, heads, dim].
    return torch.cat(
        [x[row, :, :length].transpose(0, 1) for row, length in enumerate(lengths)]
    ).contiguous()


def _sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    prefix_k: list[torch.Tensor],
    prefix_v: list[torch.Tensor],
    q_lens: tuple[int, ...],
    q_offsets: tuple[int, ...],
    scale: float,
) -> torch.Tensor:
    """FP32 SDPA with the exact LLaDA block-causal visibility rule."""

    output = torch.zeros_like(q)
    groups = Q_HEADS // KV_HEADS
    for row, (q_len, q_offset) in enumerate(zip(q_lens, q_offsets, strict=True)):
        logical_k = torch.cat((prefix_k[row], k[row, :, :q_len]), dim=1)
        logical_v = torch.cat((prefix_v[row], v[row, :, :q_len]), dim=1)
        logical_k = logical_k.repeat_interleave(groups, dim=0).float()
        logical_v = logical_v.repeat_interleave(groups, dim=0).float()
        for block_start in range(0, q_len, BLOCK_LENGTH):
            block_end = block_start + BLOCK_LENGTH
            visible_kv = q_offset + block_end
            output[row, :, block_start:block_end] = F.scaled_dot_product_attention(
                q[row, :, block_start:block_end].float(),
                logical_k[:, :visible_kv],
                logical_v[:, :visible_kv],
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
            ).to(q.dtype)
    return _pack_valid(output, q_lens)


def _batch_prefill_block_extend(
    q: torch.Tensor,
    cache: tuple[torch.Tensor, torch.Tensor],
    page_table: torch.Tensor,
    q_lens: tuple[int, ...],
    q_offsets: tuple[int, ...],
    scale: float,
) -> torch.Tensor:
    """Call the existing FluxServe FlashInfer block-extend algorithm directly."""

    import flashinfer

    device = q.device
    page_counts = tuple(
        math.ceil((offset + length) / PAGE_SIZE)
        for length, offset in zip(q_lens, q_offsets, strict=True)
    )
    qo_indptr = torch.tensor(
        (0, *torch.tensor(q_lens).cumsum(0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    kv_indptr = torch.tensor(
        (0, *torch.tensor(page_counts).cumsum(0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    kv_indices = torch.cat(
        [page_table[row, :count] for row, count in enumerate(page_counts)]
    ).to(torch.int32)
    kv_lens = tuple(
        offset + length
        for length, offset in zip(q_lens, q_offsets, strict=True)
    )
    last_page_len = torch.tensor(
        tuple((length - 1) % PAGE_SIZE + 1 for length in kv_lens),
        dtype=torch.int32,
        device=device,
    )
    workspace = torch.empty(256 * 1024**2, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        kv_layout="NHD",
        backend="auto",
        block_extend=True,
        block_size=BLOCK_LENGTH,
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        num_qo_heads=Q_HEADS,
        num_kv_heads=KV_HEADS,
        head_dim_qk=HEAD_DIM,
        page_size=PAGE_SIZE,
        custom_mask=None,
        causal=False,
        q_data_type=q.dtype,
        kv_data_type=cache[0].dtype,
        sm_scale=scale,
        q_offsets=torch.tensor(q_offsets, dtype=torch.int32, device=device),
        kv_offsets=torch.zeros(len(q_lens), dtype=torch.int32, device=device),
    )
    return wrapper.run(_pack_valid(q, q_lens), cache, return_lse=False)


def _error(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    error = (actual.float() - expected.float()).abs()
    return float(error.max()), float(error.mean())


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.skipif(
    os.environ.get(RUN_THREE_WAY) != "1",
    reason=f"set {RUN_THREE_WAY}=1 to compile and run FA4 and FlashInfer",
)
def test_fa4_batch_prefill_block_extend_and_sdpa_agree(phase: str, seed: int):
    if not torch.cuda.is_available():
        pytest.skip("three-way attention test requires CUDA")
    capability = torch.cuda.get_device_capability()
    if capability[0] not in (9, 10, 11):
        pytest.skip(f"FA4 paged KV is unsupported on compute capability {capability}")
    pytest.importorskip("flash_attn.cute")
    pytest.importorskip("flashinfer")

    torch.manual_seed(seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    if phase == "prefill":
        max_q_len = 128
        q_lens = (128, 64)
        q_offsets = (0, 0)
    else:
        max_q_len = BLOCK_LENGTH
        q_lens = (BLOCK_LENGTH, BLOCK_LENGTH)
        q_offsets = (128, 192)

    # Deliberately non-contiguous page IDs catch accidental contiguous-cache use.
    page_table = torch.tensor(
        [[5, 2, 7, 1], [6, 3, 0, 4]], dtype=torch.int32, device=device
    )
    q = torch.randn(
        len(q_lens), Q_HEADS, max_q_len, HEAD_DIM, dtype=dtype, device=device
    )
    k = torch.randn(
        len(q_lens), KV_HEADS, max_q_len, HEAD_DIM, dtype=dtype, device=device
    )
    v = torch.randn_like(k)
    base_k_cache = torch.zeros(
        8, PAGE_SIZE, KV_HEADS, HEAD_DIM, dtype=dtype, device=device
    )
    base_v_cache = torch.zeros_like(base_k_cache)
    prefix_k: list[torch.Tensor] = []
    prefix_v: list[torch.Tensor] = []
    for row, q_offset in enumerate(q_offsets):
        row_k = torch.randn(KV_HEADS, q_offset, HEAD_DIM, dtype=dtype, device=device)
        row_v = torch.randn_like(row_k)
        prefix_k.append(row_k)
        prefix_v.append(row_v)
        if q_offset:
            positions = torch.arange(q_offset, device=device)
            pages = page_table[row, positions // PAGE_SIZE].long()
            offsets = positions % PAGE_SIZE
            base_k_cache[pages, offsets] = row_k.transpose(0, 1)
            base_v_cache[pages, offsets] = row_v.transpose(0, 1)

    metadata = build_block_diffusion_paged_metadata(
        phase=phase,
        q_offsets=torch.tensor(q_offsets, device=device),
        q_lens=torch.tensor(q_lens, device=device),
        page_table=page_table,
        max_input_len=max_q_len,
        block_length=BLOCK_LENGTH,
        page_size=PAGE_SIZE,
    )
    config = AttentionForwardConfig(
        layer_id=0,
        num_heads=Q_HEADS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        num_key_value_groups=Q_HEADS // KV_HEADS,
        scale=HEAD_DIM**-0.5,
    )

    fa4_cache = (base_k_cache.clone(), base_v_cache.clone())
    fa4_padded = FA4PagedAttention(config).forward(q, k, v, fa4_cache, metadata)
    fa4 = _pack_valid(fa4_padded, q_lens)

    # Populate an independent, byte-identical logical cache for FlashInfer.
    batch_cache = (base_k_cache.clone(), base_v_cache.clone())
    positions = metadata.slot_mapping.long()
    packed_k = _pack_valid(k, q_lens)
    packed_v = _pack_valid(v, q_lens)
    batch_cache[0].view(-1, KV_HEADS, HEAD_DIM)[positions] = packed_k
    batch_cache[1].view(-1, KV_HEADS, HEAD_DIM)[positions] = packed_v
    batch_prefill = _batch_prefill_block_extend(
        q, batch_cache, page_table, q_lens, q_offsets, config.scale
    )

    sdpa = _sdpa_reference(
        q, k, v, prefix_k, prefix_v, q_lens, q_offsets, config.scale
    )
    fa4_error = _error(fa4, sdpa)
    batch_error = _error(batch_prefill, sdpa)
    cross_error = _error(fa4, batch_prefill)
    print(
        f"phase={phase} seed={seed} "
        f"fa4_vs_sdpa(max={fa4_error[0]:.8f},mean={fa4_error[1]:.8f}) "
        f"batchprefill_vs_sdpa(max={batch_error[0]:.8f},mean={batch_error[1]:.8f}) "
        f"fa4_vs_batchprefill(max={cross_error[0]:.8f},mean={cross_error[1]:.8f})"
    )
    torch.testing.assert_close(fa4, sdpa, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(batch_prefill, sdpa, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(fa4, batch_prefill, atol=3e-2, rtol=3e-2)

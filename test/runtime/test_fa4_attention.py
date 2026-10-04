import pytest
import torch

from fluxserve.backend.execution.forward_batch_info import ForwardBatch, RunnerConfig
from fluxserve.backend.layers.attention.base import AttentionForwardConfig
from fluxserve.backend.layers.attention.fa4 import FA4PagedAttention
from fluxserve.backend.layers.attention.metadata import (
    build_block_diffusion_paged_metadata,
)


def _metadata(*, phase="prefill"):
    if phase == "prefill":
        q_offsets = torch.tensor([0, 16])
        q_lens = torch.tensor([32, 16])
    else:
        q_offsets = torch.tensor([32, 48])
        q_lens = torch.tensor([16, 16])
    return build_block_diffusion_paged_metadata(
        phase=phase,
        q_offsets=q_offsets,
        q_lens=q_lens,
        page_table=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]),
        max_input_len=int(q_lens.max()),
        block_length=16,
        page_size=16,
    )


def test_prefill_is_decomposed_into_block_causal_virtual_sequences():
    metadata = _metadata()

    assert metadata.q_lens_cpu == (32, 16)
    assert metadata.q_offsets_cpu == (0, 16)
    assert metadata.qo_indptr.tolist() == [0, 16, 32, 48]
    assert metadata.kv_lens.tolist() == [16, 32, 32]
    assert metadata.q_token_indices.tolist() == [*range(32), *range(32, 48)]
    assert metadata.page_table.tolist() == [
        [0, 1],
        [0, 1],
        [4, 5],
    ]
    assert metadata.slot_mapping.tolist() == [
        *range(32),
        *range(5 * 16, 6 * 16),
    ]
    assert metadata.max_q_len == 16
    assert metadata.max_kv_len == 32


def test_decode_uses_prefix_plus_the_entire_current_block():
    metadata = _metadata(phase="decode")

    assert metadata.qo_indptr.tolist() == [0, 16, 32]
    assert metadata.kv_lens.tolist() == [48, 64]
    assert metadata.page_table.tolist() == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]
    assert metadata.slot_mapping.tolist() == [
        *range(2 * 16, 3 * 16),
        *range(7 * 16, 8 * 16),
    ]


@pytest.mark.parametrize(
    ("q_offsets", "q_lens", "message"),
    [
        ([1], [16], "block-aligned query offsets"),
        ([0], [17], "block-aligned query lengths"),
        ([0], [32], "one denoising block"),
    ],
)
def test_metadata_rejects_invalid_llada_block_shapes(q_offsets, q_lens, message):
    phase = "decode" if q_lens == [32] else "prefill"
    with pytest.raises(ValueError, match=message):
        build_block_diffusion_paged_metadata(
            phase=phase,
            q_offsets=torch.tensor(q_offsets),
            q_lens=torch.tensor(q_lens),
            page_table=torch.tensor([[0, 1, 2, 3]]),
            max_input_len=max(q_lens),
            block_length=16,
            page_size=16,
        )


def test_adapter_uses_standalone_paged_varlen_contract_and_writes_cache():
    calls = []

    def fake_fa4(q, k, v, **kwargs):
        calls.append((q.clone(), k.clone(), v.clone(), kwargs))
        return q + 1, None

    config = AttentionForwardConfig(
        layer_id=0,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        num_key_value_groups=2,
        scale=8**-0.5,
    )
    adapter = FA4PagedAttention(config, kernel=fake_fa4)
    metadata = _metadata()
    q = torch.arange(2 * 2 * 32 * 8, dtype=torch.bfloat16).view(2, 2, 32, 8)
    k = torch.arange(2 * 1 * 32 * 8, dtype=torch.bfloat16).view(2, 1, 32, 8)
    v = k + 100
    k_cache = torch.zeros(8, 16, 1, 8, dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    batch = ForwardBatch(paged_attention_metadata=metadata)

    assert adapter.can_run(q, k, v, (k_cache, v_cache), None, batch)
    output = adapter.forward(q, k, v, (k_cache, v_cache), metadata)

    packed_q, written_k, written_v, kwargs = calls[0]
    expected_packed_q = (
        q.transpose(1, 2).reshape(64, 2, 8).index_select(0, metadata.q_token_indices)
    )
    assert torch.equal(packed_q, expected_packed_q)
    assert kwargs["cu_seqlens_q"].tolist() == [0, 16, 32, 48]
    assert kwargs["seqused_k"].tolist() == [16, 32, 32]
    assert kwargs["page_table"].dtype == torch.int32
    assert kwargs["causal"] is False
    assert kwargs["return_lse"] is False

    flattened_k = k.transpose(1, 2).reshape(64, 1, 8)
    flattened_v = v.transpose(1, 2).reshape(64, 1, 8)
    assert torch.equal(
        written_k[
            metadata.slot_mapping // 16,
            metadata.slot_mapping % 16,
        ],
        flattened_k.index_select(0, metadata.q_token_indices),
    )
    assert torch.equal(
        written_v[
            metadata.slot_mapping // 16,
            metadata.slot_mapping % 16,
        ],
        flattened_v.index_select(0, metadata.q_token_indices),
    )
    expected = torch.zeros_like(q).transpose(1, 2).reshape(64, 2, 8)
    expected.index_copy_(0, metadata.q_token_indices, expected_packed_q + 1)
    expected = expected.view(2, 32, 2, 8).transpose(1, 2).contiguous()
    assert torch.equal(output, expected)


def test_runner_config_requires_paged_cache_for_fa4():
    with pytest.raises(ValueError, match="requires kv_cache_layout='paged'"):
        RunnerConfig(attention_backend="fa4", kv_cache_layout="dense")

    config = RunnerConfig(
        attention_backend="fa4",
        kv_cache_layout="paged",
        page_size=64,
    )
    assert config.attention_backend == "fa4"
    assert config.page_size == 64

    with pytest.raises(ValueError, match="multiple of 16"):
        RunnerConfig(
            attention_backend="fa4",
            kv_cache_layout="paged",
            page_size=24,
        )


def test_decode_fast_path_preserves_token_storage_and_output_layout():
    metadata = _metadata(phase="decode")
    assert metadata.is_identity_mapping
    tokens = torch.randn(2, 16, 2, 8, dtype=torch.bfloat16)
    q = tokens.transpose(1, 2)
    k = torch.randn(2, 1, 16, 8, dtype=q.dtype)
    cache = tuple(torch.zeros(8, 16, 1, 8, dtype=q.dtype) for _ in range(2))

    def kernel(packed_q, *_args, **_kwargs):
        assert packed_q.data_ptr() == tokens.data_ptr()
        return packed_q

    config = AttentionForwardConfig(
        layer_id=0, num_heads=2, num_kv_heads=1, head_dim=8,
        num_key_value_groups=2, scale=8**-0.5,
    )
    actual = FA4PagedAttention(config, kernel=kernel).forward(q, k, k, cache, metadata)
    torch.testing.assert_close(actual, q)
    # LLaDA's following transpose/contiguous must not copy the result again.
    assert actual.transpose(1, 2).contiguous().data_ptr() == tokens.data_ptr()
    torch.testing.assert_close(
        cache[0].view(-1, 1, 8)[metadata.slot_mapping],
        k.transpose(1, 2).reshape(-1, 1, 8),
    )


def test_metadata_rejects_invalid_visible_prefix_page():
    with pytest.raises(ValueError, match="negative page"):
        build_block_diffusion_paged_metadata(
            phase="decode", q_offsets=torch.tensor([16]), q_lens=torch.tensor([16]),
            page_table=torch.tensor([[-1, 1]]), max_input_len=16,
            block_length=16, page_size=16,
        )


@pytest.mark.parametrize("phase,causal", [
    ("prefill", False), ("prefill", True),
    ("decode", False), ("decode", True),
])
def test_sm120_dense_fa4_adapter_matches_paged_reference(phase, causal):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("requires an SM120 GPU")

    torch.manual_seed(7)
    device = "cuda"
    page_size = 16
    q_lens = torch.tensor([32, 16] if phase == "prefill" else [16, 16])
    q_offsets = torch.tensor([0, 16] if phase == "prefill" else [32, 48])
    page_table = torch.tensor([[5, 2, 7, 1], [4, 0, 3, 6]], device=device)
    metadata = build_block_diffusion_paged_metadata(
        phase=phase, q_offsets=q_offsets, q_lens=q_lens,
        page_table=page_table, max_input_len=int(q_lens.max()),
        block_length=16, page_size=page_size, causal=causal,
    )
    config = AttentionForwardConfig(
        layer_id=0, num_heads=4, num_kv_heads=2, head_dim=64,
        num_key_value_groups=2, scale=64**-0.5,
    )
    q = torch.randn(
        2, 4, metadata.max_input_len, 64, device=device, dtype=torch.bfloat16
    )
    k = torch.randn(
        2, 2, metadata.max_input_len, 64, device=device, dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
    cache = tuple(
        torch.randn(8, page_size, 2, 64, device=device, dtype=torch.bfloat16)
        for _ in range(2)
    )
    actual = FA4PagedAttention(config).forward(q, k, v, cache, metadata)
    torch.cuda.synchronize()

    packed_q = q.transpose(1, 2).contiguous().view(-1, 4, 64).index_select(
        0, metadata.q_token_indices
    )
    expected_packed = []
    for task in range(metadata.num_tasks):
        q_start = int(metadata.qo_indptr[task])
        q_end = int(metadata.qo_indptr[task + 1])
        kv_len = int(metadata.kv_lens[task])
        pages = metadata.page_table[task, : (kv_len + page_size - 1) // page_size]
        keys = cache[0].index_select(0, pages.long()).reshape(-1, 2, 64)[:kv_len]
        values = cache[1].index_select(0, pages.long()).reshape(-1, 2, 64)[:kv_len]
        queries = packed_q[q_start:q_end].float().transpose(0, 1)
        keys = keys.float().repeat_interleave(2, dim=1).transpose(0, 1)
        values = values.float().repeat_interleave(2, dim=1).transpose(0, 1)
        scores = queries @ keys.transpose(-1, -2) * config.scale
        if causal:
            q_pos = (
                torch.arange(q_end - q_start, device=device)
                + kv_len - (q_end - q_start)
            )
            k_pos = torch.arange(kv_len, device=device)
            scores.masked_fill_(k_pos[None, None, :] > q_pos[None, :, None], -torch.inf)
        expected_packed.append((scores.softmax(-1) @ values).transpose(0, 1))
    expected = q.new_zeros(2 * metadata.max_input_len, 4, 64)
    expected.index_copy_(
        0, metadata.q_token_indices, torch.cat(expected_packed).to(q.dtype)
    )
    expected = expected.view(2, metadata.max_input_len, 4, 64).transpose(1, 2)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)

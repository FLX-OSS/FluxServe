from __future__ import annotations

import pytest
import torch

from flux_kernel import (
    apply_rope_with_cos_sin_cache_inplace,
    moe_align_block_size,
    moe_fused_gate,
    silu_and_mul,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_silu_and_mul_matches_sgl(dtype: torch.dtype) -> None:
    from sgl_kernel import silu_and_mul as sgl_silu_and_mul

    x = torch.randn(23, 2560, device="cuda", dtype=dtype)
    actual = torch.empty(23, 1280, device="cuda", dtype=dtype)
    expected = torch.empty_like(actual)
    assert silu_and_mul(x, actual) is actual
    sgl_silu_and_mul(x, expected)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("tokens", [1, 23, 64])
@pytest.mark.parametrize("apply_scale", [False, True])
def test_moe_fused_gate_matches_sgl(tokens: int, apply_scale: bool) -> None:
    from sgl_kernel import moe_fused_gate as sgl_moe_fused_gate

    logits = torch.randn(tokens, 256, device="cuda", dtype=torch.float32)
    bias = torch.randn(256, device="cuda", dtype=torch.float32)
    actual_w, actual_i = moe_fused_gate(logits, bias, 8, 4, 8, 0, 2.5, apply_scale)
    expected_w, expected_i = sgl_moe_fused_gate(
        logits, bias, 8, 4, 8, 0, 2.5, apply_scale
    )

    actual_i, order = actual_i.sort(dim=-1)
    actual_w = actual_w.gather(1, order)
    expected_i, order = expected_i.sort(dim=-1)
    expected_w = expected_w.gather(1, order)
    torch.testing.assert_close(actual_i, expected_i)
    torch.testing.assert_close(actual_w, expected_w, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("tokens", [1, 23, 512])
def test_moe_align_block_size_matches_sgl(dtype: torch.dtype, tokens: int) -> None:
    from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size

    topk_ids = torch.randint(0, 65, (tokens, 8), device="cuda", dtype=dtype)
    block_size = 16
    num_experts = 65
    max_padded = topk_ids.numel() + num_experts * (block_size - 1)
    nblocks = (max_padded + block_size - 1) // block_size

    def buffers():
        return (
            torch.empty(max_padded, device="cuda", dtype=torch.int32),
            torch.empty(nblocks, device="cuda", dtype=torch.int32),
            torch.empty(1, device="cuda", dtype=torch.int32),
            torch.empty(num_experts + 1, device="cuda", dtype=torch.int32),
        )

    actual = buffers()
    expected = buffers()
    fused_padding = max_padded <= 4096
    assert (
        moe_align_block_size(
            topk_ids, num_experts, block_size, *actual, fused_padding
        )
        is None
    )
    if not fused_padding:
        expected[0].fill_(topk_ids.numel())
    sgl_moe_align_block_size(
        topk_ids, num_experts, block_size, *expected, fused_padding
    )
    assert actual[2].item() == expected[2].item()
    used = actual[2].item()
    used_blocks = (used + block_size - 1) // block_size
    torch.testing.assert_close(actual[1][:used_blocks], expected[1][:used_blocks])
    # Atomic insertion makes slots nondeterministic across all blocks for an expert.
    for expert in actual[1][:used_blocks].unique().tolist():
        actual_blocks = actual[1][:used_blocks] == expert
        expected_blocks = expected[1][:used_blocks] == expert
        torch.testing.assert_close(
            actual[0][: used_blocks * block_size]
            .view(used_blocks, block_size)[actual_blocks]
            .flatten().sort().values,
            expected[0][: used_blocks * block_size]
            .view(used_blocks, block_size)[expected_blocks]
            .flatten().sort().values,
        )


@pytest.mark.parametrize("is_neox", [False, True])
@pytest.mark.parametrize("separate_outputs", [False, True])
def test_rope_matches_sgl(is_neox: bool, separate_outputs: bool) -> None:
    from sgl_kernel import apply_rope_with_cos_sin_cache_inplace as sgl_rope

    positions = torch.arange(23, device="cuda", dtype=torch.int64)
    query = torch.randn(23, 4 * 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(23, 128, device="cuda", dtype=torch.bfloat16)
    frequencies = torch.randn(128, 32, device="cuda", dtype=torch.float32)
    cache = torch.cat((frequencies.cos(), frequencies.sin()), dim=-1)
    expected_q, expected_k = query.clone(), key.clone()
    actual_q, actual_k = query.clone(), key.clone()
    out_q = torch.empty_like(actual_q) if separate_outputs else None
    out_k = torch.empty_like(actual_k) if separate_outputs else None

    assert (
        apply_rope_with_cos_sin_cache_inplace(
            positions, actual_q, actual_k, 128, cache, is_neox,
            output_q_rope=out_q, output_k_rope=out_k,
        )
        is None
    )
    sgl_rope(positions, expected_q, expected_k, 128, cache, is_neox)
    torch.testing.assert_close(
        actual_q if out_q is None else out_q,
        expected_q,
    )
    torch.testing.assert_close(
        actual_k if out_k is None else out_k,
        expected_k,
    )


def test_rope_rejects_fused_kv_scatter() -> None:
    with pytest.raises(NotImplementedError, match="KV-cache"):
        apply_rope_with_cos_sin_cache_inplace(None, None, None, 128, None, True, object())

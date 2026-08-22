from types import MethodType, SimpleNamespace

import torch

from fluxserve.backend.execution.runners.flashinfer_diffusion import (
    FlashInferDiffusionRunner,
)
from fluxserve.backend.engine.request import RequestState
from fluxserve.backend.layers.attention.utils import FlashInferPagedBlockExtendState
from fluxserve.backend.layers.attention.flashinfer import FlashInferPagedPrefillAttention
from fluxserve.backend.execution.forward_batch_info import ForwardBatch
from fluxserve.backend.managers.kvcache import PagedKVCache
import pytest


def test_flashinfer_unaligned_prompt_prefills_aligned_prefix_and_replays_partial_block():
    """A 100-token prompt with 64-token blocks prefills 0:64 and decodes 64:128."""
    runner = FlashInferDiffusionRunner.__new__(FlashInferDiffusionRunner)
    runner.device = torch.device("cpu")
    runner.block_length = 64
    runner.prefilling_limit = 128
    runner.num_forwards = 0
    runner.decoder = SimpleNamespace(mask_id=-1, eos_id=-2)
    runner.runner_config = SimpleNamespace(
        mini_batch_size=1,
        cache="prefix",
        attention_backend="flashinfer",
        kv_cache_layout="paged",
        flashinfer_cache_mode="paged",
        flashinfer_prefill_mode="paged",
    )
    runner.model = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(num_hidden_layers=1))
    )
    runner.preprocess_inputs = lambda prompts: (128, 28, 2)
    runner.allocate_kv_cache = lambda batch_size: object()

    observed = {}

    def record_prefill(
        self,
        x,
        prefilling_lengths,
        non_mask_number,
        attention_mask,
        pos_ids,
        num_layers,
        mini_batch_size,
    ):
        observed["prefill_lengths"] = prefilling_lengths.clone()
        observed["prompt_and_masks"] = x.data.clone()

    def record_decode(
        self,
        x,
        decoding_start,
        total_length,
        pos_ids,
        num_layers,
        mini_batch_size,
    ):
        observed["decode_start"] = decoding_start.clone()

    runner._prefill_batches = MethodType(record_prefill, runner)
    runner._decode_batches = MethodType(record_decode, runner)

    prompt = torch.arange(100).unsqueeze(0)
    runner.generate(prompt)

    assert observed["prefill_lengths"].tolist() == [64]
    assert observed["decode_start"].tolist() == [64]
    assert torch.equal(observed["prompt_and_masks"][0, 64:100], prompt[0, 64:100])
    assert torch.all(observed["prompt_and_masks"][0, 100:128] == -1)


def test_flashinfer_64_token_mask_is_bidirectional_within_each_block():
    # Exercise the exact custom-mask builder used by FlashInfer paged attention.
    state = SimpleNamespace(device=torch.device("cpu"))
    packed = FlashInferPagedBlockExtendState.make_mask(
        state,
        q_offsets=torch.tensor([64]),
        qo_indptr=torch.tensor([0, 64]),
        kv_lens=torch.tensor([128]),
        block_length=64,
    )
    mask = packed.reshape(64, 128)

    # The partial-prompt suffix and generated portion share block [64, 128).
    assert bool(mask[0, 127])
    assert bool(mask[63, 64])
    # A token cannot attend into a later block.
    next_block_mask = FlashInferPagedBlockExtendState.make_mask(
        state,
        q_offsets=torch.tensor([64]),
        qo_indptr=torch.tensor([0, 64]),
        kv_lens=torch.tensor([129]),
        block_length=64,
    ).reshape(64, 129)
    assert not bool(next_block_mask[63, 128])
    # It can attend to all earlier blocks.
    assert bool(mask[36, 0])


def test_online_block_start_reuses_partial_prompt_block():
    state = RequestState(rid="r", input_ids=list(range(100)), max_new_tokens=128)

    assert state.aligned_prefill_length(64) == 64
    assert state.aligned_prefill_length(64) + state.current_decode_block * 64 == 64

    state.mark_decode_block_done()
    assert state.aligned_prefill_length(64) + state.current_decode_block * 64 == 128


@pytest.mark.parametrize("external_pages", [False, True])
def test_flashinfer_native_append_matches_fluxserve_slot_mapping(external_pages):
    if not torch.cuda.is_available():
        pytest.skip("FlashInfer paged append requires CUDA")
    pytest.importorskip("flashinfer")

    cache = PagedKVCache(
        num_layers=1,
        batch_size=4,
        local_kv_heads=2,
        max_length=32,
        head_dim=128,
        page_size=8,
        num_pages=24 if external_pages else None,
        dtype=torch.bfloat16,
        device="cuda",
    )
    seq_ids = torch.tensor([3, 1], device="cuda")
    if external_pages:
        cache.set_page_tables(seq_ids, [[7, 2, 11], [5, 13]])
    q_offsets = torch.tensor([9, 3], dtype=torch.int32, device="cuda")
    q_lens = torch.tensor([6, 4], dtype=torch.int32, device="cuda")
    kv_lens = q_offsets + q_lens
    kv_indptr, kv_indices, last_page_len = cache.flashinfer_paged_metadata(
        seq_ids=seq_ids, lengths=kv_lens
    )

    runner = FlashInferDiffusionRunner.__new__(FlashInferDiffusionRunner)
    runner.device = torch.device("cuda")
    forward_batch = ForwardBatch(
        flashinfer_kv_indptr=kv_indptr,
        flashinfer_paged_kv_indices=kv_indices,
        flashinfer_paged_kv_last_page_len=last_page_len,
    )
    runner._attach_flashinfer_append_metadata(
        forward_batch, q_lens=q_lens, kv_lens=kv_lens
    )
    nnz = int(q_lens.sum())
    packed_k = torch.arange(
        nnz * 2 * 128, device="cuda", dtype=torch.float32
    ).to(torch.bfloat16).reshape(nnz, 2, 128)
    packed_v = -packed_k
    layer_cache = cache.layer_paged_kv(0)
    FlashInferPagedPrefillAttention._append_native(
        packed_k, packed_v, layer_cache, forward_batch
    )
    torch.cuda.synchronize()

    slots = torch.cat([
        cache.slot_mapping(
            seq_ids[i : i + 1],
            torch.arange(int(q_offsets[i]), int(kv_lens[i]), device="cuda").unsqueeze(0),
        ).reshape(-1)
        for i in range(2)
    ])
    actual_k = layer_cache[0][slots // cache.page_size, slots % cache.page_size]
    actual_v = layer_cache[1][slots // cache.page_size, slots % cache.page_size]
    torch.testing.assert_close(actual_k, packed_k)
    torch.testing.assert_close(actual_v, packed_v)

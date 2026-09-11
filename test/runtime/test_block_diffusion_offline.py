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
        gen_length=28,
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
    runner.preprocess_inputs = lambda prompts, **kwargs: (128, 28, 2)
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
        prompt_lengths=None,
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


def test_token_array_preserves_prompt_eos_and_batch_shape():
    from fluxserve.backend.managers.kvcache import TokenArray
    from fluxserve.backend.metrics.performance import count_completion_tokens

    for batch in (1, 2):
        prompt = torch.tensor([[13, 2]]).repeat(batch, 1)
        x = TokenArray(prompt, 4, mask_id=12, eos_id=13, device="cpu")
        x.data[:, 2:] = torch.tensor([3, 16, 4, 12])
        before = x.data.clone()
        result = x.get_generated_tokens()
        assert torch.equal(result, before)
        assert torch.equal(x.data, before)
        assert result.shape == (batch, 6)
        assert count_completion_tokens(result[0], 2, (13, 16), 12) == 2


@pytest.mark.parametrize("kind", ["dense", "default", "max_batch"])
@pytest.mark.parametrize("lengths", [[5, 3], [0, 3]])
def test_offline_generation_limits_rows_and_preserves_prompt(kind, lengths):
    """Exercise real decode loops: cap both forwards and published tokens."""
    from fluxserve.backend.execution.runners.block_diffusion import BlockDiffusionRunner
    from fluxserve.backend.execution.decoders.joint_threshold import JointThresholdDecoder
    from fluxserve.backend.metrics.performance import count_completion_tokens

    cls = BlockDiffusionRunner if kind == "dense" else FlashInferDiffusionRunner
    runner = cls.__new__(cls)
    runner.device = torch.device("cpu")
    runner.block_length = 4
    runner.max_length = 16
    runner.prefilling_limit = 16
    runner.num_forwards = 0
    runner.early_stop = True
    runner.runner_config = SimpleNamespace(
        gen_length=5, block_length=4, mini_batch_size=2, cache="prefix",
        attention_backend="sdpa" if kind == "dense" else "flashinfer",
        max_post_steps=2, max_cache_length_align=4,
        flashinfer_decode_batch_mode=kind,
    )
    runner.decoder = JointThresholdDecoder(.5, 0, mask_id=12, eos_ids=(13, 16))
    runner.flashinfer_graph_runner = None
    runner._use_flashinfer_paged_cache = lambda: False
    runner._use_flashinfer_paged_prefill = lambda: False
    runner._make_forward_batch = lambda *a, **kw: None
    runner._make_decode_forward_batch = lambda *a, **kw: None
    runner.allocate_kv_cache = lambda batch: torch.zeros(1, 2, batch, 1, 16, 1)
    runner._prefill_batches = lambda *a, **kw: None
    committed = []

    def model(tokens, **kw):
        logits = torch.full((*tokens.shape, 20), -10.)
        logits[..., 7] = 10.
        return SimpleNamespace(logits=logits, input_tokens=tokens.clone())

    model.model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=1))
    runner.model = model

    def commit(output, ids, starts, finished, *args):
        for row in finished.nonzero(as_tuple=True)[0].tolist():
            committed.append((int(ids[row]), int(starts[ids[row]]), output.input_tokens[row]))

    runner._update_finished_kv_cache = commit
    prompts = torch.tensor([[13, 2, 12, 12, 12, 12], [1, 2, 3, 4, 5, 6]])
    result = runner.generate(prompts, prompt_lengths=[2, 6], generation_lengths=lengths)
    for row, prompt_len in enumerate([2, 6]):
        assert torch.equal(result[row, :prompt_len], prompts[row, :prompt_len])
        assert result[row, prompt_len:prompt_len + lengths[row]].tolist() == [7] * lengths[row]
        assert (result[row, prompt_len + lengths[row]:] == 12).all()
        assert count_completion_tokens(result[row], prompt_len, (13, 16), 12) == lengths[row]
    expected = {(1, 4), (1, 8)}
    if lengths[0]:
        expected |= {(0, 0), (0, 4)}
    assert {(row, start) for row, start, _ in committed} == expected
    for row, start, tokens in committed:
        end = min(start + 4, [2, 6][row] + lengths[row])
        assert torch.equal(tokens[:end-start], result[row, start:end])

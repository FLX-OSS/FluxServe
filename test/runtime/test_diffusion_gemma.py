from types import SimpleNamespace

import torch
import torch.nn.functional as F

from fluxserve.backend.configs.diffusion_gemma import DiffusionGemmaConfig
from fluxserve.backend.layers.attention.base import AttentionForwardConfig, DenseAttention
from fluxserve.backend.layers.attention.diffusion_gemma_flashinfer import (
    DiffusionGemmaLayerGeometry,
    DiffusionGemmaPagedAttention,
    DiffusionGemmaPagedKVCache,
)
from fluxserve.backend.execution.decoders.diffusion_gemma import (
    DiffusionGemmaDecoder,
    DiffusionGemmaSamplingConfig,
    normalize_eos_ids,
)
from fluxserve.backend.execution.runners.diffusion_gemma import DiffusionGemmaRunner
from fluxserve.backend.models.diffusion_gemma import (
    DiffusionGemmaAttention,
    DiffusionGemmaForConditionalGeneration,
    diffusion_gemma_layer_config,
)
from fluxserve.backend.layers.moe.fused_moe_triton.fused_moe import (
    gelu_tanh_and_mul,
)


def _sampling_config(**overrides):
    values = dict(
        canvas_length=4,
        max_denoising_steps=2,
        t_min=0.0,
        t_max=0.0,
        entropy_bound=0.1,
        confidence_threshold=10.0,
        stability_threshold=2,
        vocab_size=8,
        eos_ids=(1,),
        pad_id=0,
    )
    values.update(overrides)
    return DiffusionGemmaSamplingConfig(**values)


def test_layer_config_resolves_global_attention_geometry():
    config = SimpleNamespace(
        is_heterogeneous=False,
        layer_types=["sliding_attention", "full_attention"],
        head_dim=32,
        global_head_dim=64,
        num_key_value_heads=4,
        num_global_key_value_heads=2,
    )
    sliding = diffusion_gemma_layer_config(config, 0)
    full = diffusion_gemma_layer_config(config, 1)
    assert (sliding.head_dim, sliding.num_key_value_heads) == (32, 4)
    assert (full.head_dim, full.num_key_value_heads) == (64, 2)


def test_top_level_config_exposes_runtime_text_geometry():
    config = DiffusionGemmaConfig(
        text_config={"hidden_size": 64, "dtype": "bfloat16"}
    )
    assert config.hidden_size == 64
    assert config.dtype == torch.bfloat16


def test_checkpoint_mapping_uses_decoder_as_shared_backbone():
    map_name = DiffusionGemmaForConditionalGeneration._checkpoint_name
    assert map_name("model.decoder.layers.2.self_attn.q_proj.weight") == (
        "model.layers.2.self_attn.q_proj.weight"
    )
    assert map_name("model.decoder.self_conditioning.gate_proj.weight") == (
        "self_conditioning.gate_proj.weight"
    )
    assert map_name("model.encoder.language_model.layers.2.layer_scalar") == (
        "model.layers.2.layer_scalar"
    )
    assert map_name("model.encoder.vision_tower.layer.weight") is None


def test_diffusion_masks_separate_denoise_and_commit():
    causal = DiffusionGemmaRunner._causal_mask(3, 2, torch.device("cpu"))[0]
    bidi = DiffusionGemmaRunner._bidirectional_mask(3, 2, torch.device("cpu"))[0]
    assert causal.tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, False],
        [True, True, True, True, True],
    ]
    assert bool(bidi.all())


def test_diffusion_runner_accepts_matching_tp_ep(monkeypatch):
    monkeypatch.setattr(
        "fluxserve.backend.execution.runners.block_diffusion.BlockDiffusionRunner.__init__",
        lambda self, model_config, server_args, runner_config, device, **kwargs: None,
    )
    args = SimpleNamespace(
        tp_size=2,
        ep_size=2,
        dp_size=1,
        pp_size=1,
        enable_dp_attention=False,
    )

    runner = DiffusionGemmaRunner(SimpleNamespace(), args, device="cuda")

    assert runner.last_denoising_steps == []


def test_diffusion_runner_rejects_mismatched_tp_ep():
    args = SimpleNamespace(
        tp_size=2,
        ep_size=1,
        dp_size=1,
        pp_size=1,
        enable_dp_attention=False,
    )

    try:
        DiffusionGemmaRunner(SimpleNamespace(), args, device="cuda")
    except ValueError as error:
        assert "requires TP and EP to use the same world size" in str(error)
    else:
        raise AssertionError("mismatched TP/EP topology should be rejected")


def test_diffusion_attention_appends_immutable_tuple_cache():
    past_k = torch.arange(6).reshape(1, 1, 3, 2)
    past_v = past_k + 10
    next_k = torch.arange(4).reshape(1, 1, 2, 2) + 100
    next_v = next_k + 10

    key, value = DiffusionGemmaAttention._append_past_key_values(
        next_k, next_v, (past_k, past_v)
    )

    assert torch.equal(key, torch.cat((past_k, next_k), dim=2))
    assert torch.equal(value, torch.cat((past_v, next_v), dim=2))


def test_sliding_attention_returns_bounded_cache_suffix():
    attention = object.__new__(DiffusionGemmaAttention)
    attention.sliding_window = 3
    key = torch.arange(10).reshape(1, 1, 5, 2)
    value = key + 100

    cached_key, cached_value = attention._cache_for_next_forward(key, value)

    assert torch.equal(cached_key, key[:, :, -3:])
    assert torch.equal(cached_value, value[:, :, -3:])


def test_full_attention_returns_complete_cache():
    attention = object.__new__(DiffusionGemmaAttention)
    attention.sliding_window = None
    key = torch.arange(10).reshape(1, 1, 5, 2)
    value = key + 100

    cached_key, cached_value = attention._cache_for_next_forward(key, value)

    assert cached_key is key
    assert cached_value is value


def test_sliding_mask_uses_absolute_positions_for_truncated_cache():
    attention = object.__new__(DiffusionGemmaAttention)
    attention.sliding_window = 3
    positions = torch.tensor([[5, 6]])
    logical_mask = torch.ones(1, 2, 7, dtype=torch.bool)

    mask = attention._apply_sliding_window(logical_mask, positions, kv_len=5)

    assert mask.tolist() == [
        [[False, True, True, True, True], [False, False, True, True, True]]
    ]


def test_diffusion_gemma_paged_cache_supports_heterogeneous_layers():
    cache = DiffusionGemmaPagedKVCache(
        layer_geometries=[
            DiffusionGemmaLayerGeometry(4, 256),
            DiffusionGemmaLayerGeometry(1, 512),
        ],
        max_length=17,
        page_size=8,
        dtype=torch.bfloat16,
        device="cpu",
    )

    assert cache.layer_paged_kv(0)[0].shape == (3, 8, 4, 256)
    assert cache.layer_paged_kv(1)[0].shape == (3, 8, 1, 512)
    indptr, indices, last_page_len = cache.metadata(17)
    assert indptr.tolist() == [0, 3]
    assert indices.tolist() == [0, 1, 2]
    assert last_page_len.tolist() == [1]


def test_diffusion_gemma_paged_cache_builds_variable_length_batch_metadata():
    cache = DiffusionGemmaPagedKVCache(
        layer_geometries=[DiffusionGemmaLayerGeometry(1, 8)],
        max_length=20,
        page_size=4,
        dtype=torch.bfloat16,
        device="cpu",
        batch_size=4,
    )

    metadata = cache.build_metadata(
        phase="prefill",
        seq_ids=(0, 1, 2, 3),
        q_offsets=(0, 0, 0, 0),
        q_lens=(3, 5, 7, 9),
        kv_lens=(3, 5, 7, 9),
        max_q_len=9,
    )

    assert cache.pages_per_sequence == 5
    assert cache.num_pages == 20
    assert cache.layer_paged_kv(0)[0].shape == (20, 4, 1, 8)
    assert metadata.qo_indptr.tolist() == [0, 3, 8, 15, 24]
    assert metadata.kv_indptr.tolist() == [0, 1, 3, 5, 8]
    assert metadata.kv_indices.tolist() == [0, 5, 6, 10, 11, 15, 16, 17]
    assert metadata.last_page_len.tolist() == [3, 1, 3, 1]
    assert metadata.gather_indices.tolist() == [
        0,
        1,
        2,
        9,
        10,
        11,
        12,
        13,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
    ]


def test_diffusion_gemma_batch_metadata_reuses_phase_masks():
    cache = DiffusionGemmaPagedKVCache(
        layer_geometries=[DiffusionGemmaLayerGeometry(1, 8)],
        max_length=16,
        page_size=4,
        dtype=torch.bfloat16,
        device="cpu",
        batch_size=2,
    )
    metadata = cache.build_metadata(
        phase="denoise",
        seq_ids=(0, 1),
        q_offsets=(3, 5),
        q_lens=(2, 2),
        kv_lens=(5, 7),
        max_q_len=2,
    )

    first = metadata.mask(3)
    second = metadata.mask(3)

    assert first.data_ptr() == second.data_ptr()
    assert first.numel() == 2 * 5 + 2 * 7


def test_diffusion_gemma_paged_masks_cover_all_phases_and_sliding_window():
    attention = DiffusionGemmaPagedAttention(
        layer_id=0,
        num_heads=1,
        num_kv_heads=1,
        head_dim=2,
        scale=1.0,
        sliding_window=3,
    )
    positions = torch.tensor([3, 4])

    prefill = attention._mask(positions, 5, "prefill")
    denoise = attention._mask(positions, 5, "denoise")
    commit = attention._mask(positions, 5, "commit")

    assert prefill.tolist() == [
        [False, True, True, True, False],
        [False, False, True, True, True],
    ]
    assert denoise.tolist() == [
        [False, True, True, True, True],
        [False, False, True, True, True],
    ]
    assert torch.equal(commit, prefill)


def test_dense_attention_honors_model_scale():
    config = AttentionForwardConfig(
        layer_id=0,
        num_heads=1,
        num_kv_heads=1,
        head_dim=2,
        num_key_value_groups=1,
        scale=1.0,
    )
    query = torch.tensor([[[[1.0, 0.0]]]])
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    value = torch.tensor([[[[2.0, 0.0], [0.0, 2.0]]]])

    actual = DenseAttention(config).forward(query, key, value, None)
    expected = F.scaled_dot_product_attention(query, key, value, scale=1.0)
    default_scaled = F.scaled_dot_product_attention(query, key, value)

    assert torch.allclose(actual, expected)
    assert not torch.allclose(actual, default_scaled)


def test_fused_gelu_uses_contiguous_gate_up_halves():
    gate = torch.tensor([[1.0, 2.0]])
    up = torch.tensor([[3.0, 4.0]])
    packed = torch.cat((gate, up), dim=-1)

    actual = gelu_tanh_and_mul(packed)
    expected = F.gelu(gate, approximate="tanh") * up

    assert torch.allclose(actual, expected)


def test_first_denoising_step_runs_zero_signal_self_conditioning():
    embedded = torch.tensor([[[1.0, 2.0]]])
    observed = {}

    def self_conditioning(inputs, soft_embeds):
        observed["inputs"] = inputs
        observed["soft_embeds"] = soft_embeds
        return inputs + 1

    model = SimpleNamespace(
        model=SimpleNamespace(embed_input_ids=lambda input_ids: embedded),
        self_conditioning=self_conditioning,
    )
    result = DiffusionGemmaForConditionalGeneration.embed_with_self_conditioning(
        model, torch.tensor([[3]])
    )

    assert torch.equal(observed["inputs"], embedded)
    assert torch.equal(observed["soft_embeds"], torch.zeros_like(embedded))
    assert torch.equal(result, embedded + 1)


def test_sampler_uses_categorical_sampling(monkeypatch):
    import fluxserve.backend.execution.decoders.diffusion_gemma as module

    monkeypatch.setattr(module, "tensor_model_parallel_all_reduce", lambda x: x)
    sampled_shape = {}

    def multinomial(probs, num_samples):
        sampled_shape["shape"] = tuple(probs.shape)
        return torch.zeros((probs.shape[0], num_samples), dtype=torch.long)

    monkeypatch.setattr(module.torch, "multinomial", multinomial)
    decoder = DiffusionGemmaDecoder(_sampling_config(max_denoising_steps=1))
    state = decoder.new_state(torch.device("cpu"))
    logits = torch.zeros(1, 4, 8)
    embed = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)

    decoder.step(logits, state, embed, 0, 8, 2.0)
    assert sampled_shape["shape"] == (4, 8)


def test_sampler_entropy_acceptance_matches_transformers(monkeypatch):
    import fluxserve.backend.execution.decoders.diffusion_gemma as module

    monkeypatch.setattr(module, "tensor_model_parallel_all_reduce", lambda x: x)
    decoder = DiffusionGemmaDecoder(_sampling_config(canvas_length=3, max_denoising_steps=1, entropy_bound=0.5))
    state = decoder.new_state(torch.device("cpu"))
    logits = torch.tensor([[[5.0, 0.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0]]]).expand(1, 3, 8)
    embed = torch.arange(8 * 2, dtype=torch.float32).reshape(8, 2)
    decoder.step(logits, state, embed, 0, 8, 1.0)
    # The lowest-entropy token is accepted first; the next one is rejected
    # once the entropy budget is exceeded.
    assert torch.equal(state.canvas, state.argmax_canvas)


def test_sampler_converges_at_max_steps(monkeypatch):
    import fluxserve.backend.execution.decoders.diffusion_gemma as module

    monkeypatch.setattr(module, "tensor_model_parallel_all_reduce", lambda x: x)
    decoder = DiffusionGemmaDecoder(_sampling_config(max_denoising_steps=1))
    state = decoder.new_state(torch.device("cpu"))
    logits = torch.zeros(1, 4, 8)
    logits[..., 3] = 10
    embed = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)
    assert decoder.step(logits, state, embed, 0, 8, 2.0)
    assert state.step == 1
    assert torch.all(state.argmax_canvas == 3)
    assert state.soft_embeds.shape == (1, 4, 3)


def test_sampler_requires_positive_entropy_bound():
    try:
        DiffusionGemmaDecoder(_sampling_config(entropy_bound=0.0))
    except ValueError as error:
        assert "entropy_bound" in str(error)
    else:
        raise AssertionError("expected invalid entropy bound to fail")


def test_sampler_normalizes_and_detects_all_eos_ids():
    assert normalize_eos_ids(1) == (1,)
    assert normalize_eos_ids([1, 106, 50, 1]) == (1, 106, 50)

    decoder = DiffusionGemmaDecoder(_sampling_config(eos_ids=(1, 106, 50)))
    assert decoder.eos_id == 1
    assert decoder.eos_ids == (1, 106, 50)
    for eos_id in decoder.eos_ids:
        tokens = torch.tensor([[7, eos_id, 6]])
        assert decoder.contains_eos(tokens)
        assert decoder.first_eos_index(tokens) == 1
    assert not decoder.contains_eos(torch.tensor([[7, 6, 5]]))
    assert decoder.first_eos_index(torch.tensor([[7, 6, 5]])) is None


def test_runner_batch_uses_per_row_prompt_and_generation_lengths():
    runner = object.__new__(DiffusionGemmaRunner)
    runner.decoder = SimpleNamespace(pad_id=0)
    runner.runner_config = SimpleNamespace(gen_length=4)
    runner.last_denoising_steps = []
    observed = []

    def generate_one(prompt, generation_length):
        observed.append((prompt.tolist(), generation_length))
        runner._current_denoising_steps = generation_length
        return torch.arange(10, 10 + generation_length).unsqueeze(0)

    runner._generate_one = generate_one
    prompts = torch.tensor([[2, 3, 0], [4, 5, 6]])

    output = runner.generate(
        prompts,
        prompt_lengths=[2, 3],
        generation_lengths=[2, 4],
    )

    assert observed == [([2, 3], 2), ([4, 5, 6], 4)]
    assert output.shape == (2, 7)
    assert output[0].tolist() == [2, 3, 0, 10, 11, 0, 0]
    assert output[1].tolist() == [4, 5, 6, 10, 11, 12, 13]
    assert runner.last_denoising_steps == [2, 4]

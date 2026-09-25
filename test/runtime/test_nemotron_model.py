"""Nemotron-Labs-Diffusion model, checkpoint mapping and configuration tests.

Checkpoint-dependent tests read only the safetensors header, so they run on a
login node without a GPU and without mapping 27 GB of weights. They skip when
the Hugging Face cache is not visible; set ``HF_HOME`` to a cache containing
the checkpoint to exercise them.
"""

import functools
import math
import pathlib
from types import SimpleNamespace

import pytest
import torch

from fluxserve.backend.model_loader.nemotron import (
    MAX_SUPPORTED_POSITIONS,
    NemotronModelLoader,
    check_nemotron_context_limit,
    nemotron_block_length,
    nemotron_decoding_ids,
    read_safetensors_header,
    resolve_nemotron_eos_ids,
    resolve_nemotron_snapshot,
)
from fluxserve.backend.models.nemotron_diffusion import (
    NemotronLabsDiffusionLLM,
    is_nemotron_diffusion_config,
    nemotron_head_dim,
    nemotron_query_scale,
    nemotron_rope_parameters,
    nemotron_weight_plan,
)

REPO_ID = "nvidia/Nemotron-Labs-Diffusion-14B"

# The pinned checkpoint's geometry, from config.json. Kept literal so a config
# change is a visible test failure rather than a silently rewritten expectation.
CHECKPOINT_GEOMETRY = dict(
    num_hidden_layers=40,
    hidden_size=5120,
    intermediate_size=16384,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    vocab_size=131072,
    rms_norm_eps=1e-5,
    mask_token_id=100,
    eos_token_id=11,
    block_size=32,
    max_position_embeddings=262144,
    hidden_act="silu",
    attention_bias=False,
    mlp_bias=False,
    tie_word_embeddings=False,
)
CHECKPOINT_TENSOR_COUNT = 363
ROPE_PARAMETERS = {
    "rope_type": "yarn",
    "rope_theta": 1000000000.0,
    "factor": 16.0,
    "original_max_position_embeddings": 16384,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "llama_4_scaling_beta": 0.1,
}


def checkpoint_config(**overrides):
    """A config object with the real checkpoint's fields, built offline."""
    values = dict(
        CHECKPOINT_GEOMETRY,
        architectures=["NemotronLabsDiffusionModel"],
        model_type="nemotron_labs_diffusion",
        rope_parameters=dict(ROPE_PARAMETERS),
        rope_scaling=dict(ROPE_PARAMETERS),
        rope_theta=ROPE_PARAMETERS["rope_theta"],
        _name_or_path=REPO_ID,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def tiny_config(**overrides):
    """A small model with the same structure, cheap enough to instantiate."""
    values = dict(
        architectures=["NemotronLabsDiffusionModel"],
        model_type="nemotron_labs_diffusion",
        num_hidden_layers=2,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        rms_norm_eps=1e-5,
        mask_token_id=7,
        eos_token_id=3,
        block_size=4,
        max_position_embeddings=512,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
        rope_parameters={"rope_type": "yarn", "rope_theta": 10000.0, "factor": 2.0,
                         "original_max_position_embeddings": 256,
                         "beta_fast": 32.0, "beta_slow": 1.0,
                         "llama_4_scaling_beta": 0.1},
        rope_theta=10000.0,
        _name_or_path="nemotron-tiny",
    )
    values["rope_scaling"] = dict(values["rope_parameters"])
    values.update(overrides)
    return SimpleNamespace(**values)


def checkpoint_header():
    """The real checkpoint's safetensors header, or skip.

    Cache-only on purpose: without it a runner with a cold cache would start a
    27 GB download from a unit test.
    """
    try:
        snapshot = resolve_nemotron_snapshot(
            checkpoint_config(), local_files_only=True
        )
    except Exception as error:  # noqa: BLE001 - any resolution failure skips
        pytest.skip(f"Nemotron checkpoint unavailable: {error}")
    return read_safetensors_header(snapshot / "model.safetensors")


# --------------------------------------------------------------------------
# Dispatch and configuration
# --------------------------------------------------------------------------


def test_dispatch_matches_architecture_or_model_type():
    assert is_nemotron_diffusion_config(checkpoint_config())
    assert is_nemotron_diffusion_config(
        SimpleNamespace(architectures=None, model_type="nemotron_labs_diffusion")
    )
    assert is_nemotron_diffusion_config(
        SimpleNamespace(architectures=["NemotronLabsDiffusionModel"], model_type=None)
    )
    assert not is_nemotron_diffusion_config(
        SimpleNamespace(architectures=["LLaDA2MoeModelLM"], model_type="llada2_moe")
    )


def test_head_dim_comes_from_config_not_hidden_size_division():
    config = checkpoint_config()
    assert nemotron_head_dim(config) == 128
    assert config.hidden_size // config.num_attention_heads == 160
    # Fallback stays available for configs that omit head_dim.
    assert nemotron_head_dim(checkpoint_config(head_dim=None)) == 160


def test_rope_parameters_read_theta_from_the_nested_dict():
    base, scaling = nemotron_rope_parameters(checkpoint_config())
    assert base == 1000000000.0
    assert scaling["rope_type"] == "yarn"
    assert scaling["factor"] == 16.0
    assert scaling["original_max_position_embeddings"] == 16384


def test_decoding_ids_come_from_the_checkpoint_not_llada_defaults():
    ids = nemotron_decoding_ids(checkpoint_config())
    assert ids["mask_id"] == 100
    assert ids["eos_id"] == 11
    assert ids["eos_ids"] == (11,)
    from fluxserve.backend.execution.forward_batch_info import RunnerConfig

    defaults = RunnerConfig()
    assert ids["mask_id"] != defaults.mask_id
    assert ids["eos_id"] != defaults.eos_id
    assert defaults.mask_id >= CHECKPOINT_GEOMETRY["vocab_size"]


def test_decoding_ids_reject_out_of_vocabulary_values():
    with pytest.raises(ValueError, match="mask_token_id"):
        nemotron_decoding_ids(checkpoint_config(mask_token_id=999999))
    with pytest.raises(ValueError, match="mask_token_id"):
        nemotron_decoding_ids(checkpoint_config(mask_token_id=None))
    with pytest.raises(ValueError, match="outside vocab_size"):
        nemotron_decoding_ids(
            checkpoint_config(eos_token_id=999999, _name_or_path="nemotron-absent")
        )


def test_block_length_defaults_to_checkpoint_block_size():
    config = checkpoint_config()
    assert nemotron_block_length(config) == 32
    assert nemotron_block_length(config, 64) == 64
    with pytest.raises(ValueError):
        nemotron_block_length(config, 0)


def test_context_limit_rejects_positions_beyond_the_checkpoint_window():
    check_nemotron_context_limit(MAX_SUPPORTED_POSITIONS)
    with pytest.raises(ValueError, match="262144"):
        check_nemotron_context_limit(MAX_SUPPORTED_POSITIONS + 1)


# --------------------------------------------------------------------------
# Query temperature scaling (identity inside the supported window)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("position", [0, 1, 31, 32, 16383])
def test_query_scale_is_exactly_identity_below_the_original_window(position):
    positions = torch.tensor([[position]], dtype=torch.long)
    scale = nemotron_query_scale(positions, 0.1, 16384)
    assert scale.shape == (1, 1, 1)
    assert scale.item() == 1.0


@pytest.mark.parametrize("position", [16384, 32768, 65536])
def test_query_scale_matches_the_reference_formula_beyond_the_window(position):
    positions = torch.tensor([[position]], dtype=torch.long)
    scale = nemotron_query_scale(positions, 0.1, 16384)
    expected = 1.0 + 0.1 * math.log(1.0 + math.floor(position / 16384))
    assert scale.item() == pytest.approx(expected, rel=1e-6)


def test_query_scale_disabled_without_beta_or_window():
    positions = torch.tensor([[32768]], dtype=torch.long)
    assert nemotron_query_scale(positions, 0.0, 16384) is None
    assert nemotron_query_scale(positions, 0.1, 0) is None
    assert nemotron_query_scale(positions, None, 16384) is None


# --------------------------------------------------------------------------
# Checkpoint mapping
# --------------------------------------------------------------------------


def test_checkpoint_names_map_encoder_and_diffusion_head():
    name_of = NemotronLabsDiffusionLLM.checkpoint_name
    assert name_of("encoder.layers.7.self_attn.q_proj.weight") == (
        "model.layers.7.self_attn.q_proj.weight"
    )
    assert name_of("encoder.embed_tokens.weight") == "model.embed_tokens.weight"
    assert name_of("encoder.norm.weight") == "model.norm.weight"
    assert name_of("diffusion_head.weight") == "lm_head.weight"


def test_weight_plan_covers_every_expected_tensor_exactly_once():
    plan = nemotron_weight_plan(checkpoint_config())
    assert len(plan) == CHECKPOINT_TENSOR_COUNT
    # Nine per-layer tensors plus embeddings, final norm and the head.
    assert len(plan) == 9 * CHECKPOINT_GEOMETRY["num_hidden_layers"] + 3

    qkv = "model.layers.0.self_attn.qkv_proj.weight"
    shards = {
        plan[f"encoder.layers.0.self_attn.{name}_proj.weight"][1]
        for name in ("q", "k", "v")
    }
    assert shards == {"q", "k", "v"}
    assert all(
        plan[f"encoder.layers.0.self_attn.{name}_proj.weight"][0] == qkv
        for name in ("q", "k", "v")
    )
    gate_up = "model.layers.0.mlp.gate_up_proj.weight"
    assert plan["encoder.layers.0.mlp.gate_proj.weight"][:2] == (gate_up, 0)
    assert plan["encoder.layers.0.mlp.up_proj.weight"][:2] == (gate_up, 1)


def test_weight_plan_shapes_use_head_dim_not_hidden_over_heads():
    plan = nemotron_weight_plan(checkpoint_config())
    assert plan["encoder.layers.0.self_attn.q_proj.weight"][2] == (4096, 5120)
    assert plan["encoder.layers.0.self_attn.k_proj.weight"][2] == (1024, 5120)
    assert plan["encoder.layers.0.self_attn.v_proj.weight"][2] == (1024, 5120)
    assert plan["encoder.layers.0.self_attn.o_proj.weight"][2] == (5120, 4096)
    assert plan["encoder.layers.0.mlp.gate_proj.weight"][2] == (16384, 5120)
    assert plan["encoder.layers.0.mlp.down_proj.weight"][2] == (5120, 16384)
    assert plan["diffusion_head.weight"][2] == (131072, 5120)


def test_weight_plan_destinations_exist_on_the_constructed_model():
    config = tiny_config()
    model = NemotronLabsDiffusionLLM(config)
    plan = nemotron_weight_plan(config)
    params = dict(model.named_parameters())

    destinations = {target for target, _, _ in plan.values()}
    assert destinations <= set(params)
    # Nothing in the model is left without a checkpoint source.
    assert set(params) == destinations


# --------------------------------------------------------------------------
# Real checkpoint header (no tensor data is read)
# --------------------------------------------------------------------------


def test_real_checkpoint_header_matches_the_plan_exactly():
    header = checkpoint_header()
    plan = nemotron_weight_plan(checkpoint_config())

    assert len(header) == CHECKPOINT_TENSOR_COUNT
    missing = sorted(set(plan) - set(header))
    unexpected = sorted(set(header) - set(plan))
    assert not missing, f"checkpoint is missing {len(missing)}: {missing[:5]}"
    assert not unexpected, f"checkpoint has {len(unexpected)} unmapped: {unexpected[:5]}"

    for name, (_, _, expected_shape) in plan.items():
        assert tuple(header[name]["shape"]) == tuple(expected_shape), name
        assert header[name]["dtype"] == "BF16", name


def test_real_checkpoint_eos_ids_come_from_generation_config():
    try:
        resolve_nemotron_snapshot(checkpoint_config(), local_files_only=True)
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"Nemotron checkpoint unavailable: {error}")
    assert resolve_nemotron_eos_ids(checkpoint_config()) == (11,)


def test_loader_rejects_foreign_configs_and_quantization():
    loader = NemotronModelLoader()
    with pytest.raises(ValueError, match="non-Nemotron"):
        loader.load_model(
            model_config=SimpleNamespace(architectures=["LLaDA2MoeModelLM"]),
            device="cpu",
        )
    with pytest.raises(ValueError, match="quantization"):
        loader.load_model(
            model_config=checkpoint_config(), device="cpu", quant_config={"x": 1}
        )


# --------------------------------------------------------------------------
# Architecture behavior: causality is owned by the caller
# --------------------------------------------------------------------------


def initialized_tiny_model(seed: int = 0):
    """A small model whose attention actually discriminates.

    Norm weights stay at one and projections use a wide init on purpose: a
    small init drives every attention score toward zero, the softmax toward
    uniform, and the output toward the mean of V, which would hide exactly the
    position and attention-pattern effects these tests exist to check.
    """
    torch.manual_seed(seed)
    model = NemotronLabsDiffusionLLM(tiny_config()).eval()
    for name, parameter in model.named_parameters():
        if parameter.dim() == 1:
            torch.nn.init.ones_(parameter)
        else:
            torch.nn.init.normal_(parameter, std=0.5)
    return model


def causal_mask(batch_size: int, length: int) -> torch.Tensor:
    return (
        torch.tril(torch.ones(length, length, dtype=torch.bool))
        .unsqueeze(0)
        .expand(batch_size, -1, -1)
    )


def test_forward_runs_in_both_attention_modes():
    model = initialized_tiny_model()
    ids = torch.randint(0, 64, (2, 6))

    bidirectional = model(input_ids=ids, use_cache=False)
    assert tuple(bidirectional.logits.shape) == (2, 6, 64)
    assert bidirectional.logits.dtype == torch.float32
    assert torch.isfinite(bidirectional.logits).all()

    causal = model(input_ids=ids, attention_mask=causal_mask(2, 6), use_cache=True)
    assert tuple(causal.logits.shape) == (2, 6, 64)
    # The same weights under two attention patterns must not agree.
    assert not torch.allclose(bidirectional.logits, causal.logits)


def test_logits_keep_checkpoint_projection_dtype_for_threshold_sampling():
    model = initialized_tiny_model()
    model.lm_head.to(torch.bfloat16)
    hidden = torch.randn(1, 4, tiny_config().hidden_size, dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(hidden, model.lm_head.weight)
    logits = model._get_logits(hidden)
    assert logits.dtype == torch.bfloat16
    assert torch.equal(logits, expected[..., :tiny_config().vocab_size])


def test_present_key_values_follow_the_runner_convention():
    model = initialized_tiny_model()
    ids = torch.randint(0, 64, (2, 6))
    output = model(input_ids=ids, attention_mask=causal_mask(2, 6), use_cache=True)

    config = tiny_config()
    assert len(output.past_key_values) == 2 * config.num_hidden_layers
    assert tuple(output.past_key_values[0].shape) == (
        2,
        config.num_key_value_heads,
        6,
        config.head_dim,
    )


def test_only_the_first_layer_kv_is_attention_pattern_independent():
    """The premise behind G1: a causal commit forward cannot be replaced.

    Layer 0 projects the embeddings, so its K/V are mask-independent. Every
    later layer consumes hidden states that depend on the attention pattern, so
    bidirectional-forward KV is not the causal KV the model expects to reuse.
    """
    model = initialized_tiny_model()
    ids = torch.randint(0, 64, (2, 6))

    bidirectional = model(input_ids=ids, use_cache=True)
    causal = model(input_ids=ids, attention_mask=causal_mask(2, 6), use_cache=True)

    # Layer 0 key and value.
    assert torch.allclose(bidirectional.past_key_values[0], causal.past_key_values[0])
    assert torch.allclose(bidirectional.past_key_values[1], causal.past_key_values[1])
    # Layer 1 key and value.
    assert not torch.allclose(
        bidirectional.past_key_values[2], causal.past_key_values[2]
    )
    assert not torch.allclose(
        bidirectional.past_key_values[3], causal.past_key_values[3]
    )


def max_abs_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max())


# Float32 cos/sin accumulation at large positions leaves ~1e-5 of noise on
# logits of order 1e0, so "unchanged" is asserted against a tolerance and
# "changed" against a floor well above it. Measured separation is >4 orders.
POSITION_NOISE = 1e-3
POSITION_SIGNAL = 1e-2


def test_position_ids_default_to_absolute_positions():
    model = initialized_tiny_model()
    ids = torch.randint(0, 64, (1, 4))

    implicit = model(input_ids=ids, use_cache=False)
    explicit = model(
        input_ids=ids,
        position_ids=torch.arange(4).unsqueeze(0),
        use_cache=False,
    )
    assert torch.equal(implicit.logits, explicit.logits)

    # RoPE is relative, so a uniform shift is a no-op by construction. The
    # query temperature scale is the only absolute-position term, and it is an
    # identity below `original_max_position_embeddings`. Spacing must matter.
    shifted = model(
        input_ids=ids,
        position_ids=torch.arange(8, 12).unsqueeze(0),
        use_cache=False,
    )
    assert max_abs_difference(implicit.logits, shifted.logits) < POSITION_NOISE

    spread = model(
        input_ids=ids,
        position_ids=(torch.arange(4) * 2).unsqueeze(0),
        use_cache=False,
    )
    assert max_abs_difference(implicit.logits, spread.logits) > POSITION_SIGNAL


def test_query_scale_breaks_rope_shift_invariance_beyond_the_window():
    """Long-context guard: the absolute-position term must reach attention.

    Inside the window the model is shift-invariant; the first block of
    positions at or past `original_max_position_embeddings` must not be.
    """
    config = tiny_config()
    window = config.rope_parameters["original_max_position_embeddings"]
    model = initialized_tiny_model()
    ids = torch.randint(0, 64, (1, 4))

    def logits_at(start: int) -> torch.Tensor:
        return model(
            input_ids=ids,
            position_ids=torch.arange(start, start + 4).unsqueeze(0),
            use_cache=False,
        ).logits

    inside = logits_at(0)
    assert max_abs_difference(inside, logits_at(window - 4)) < POSITION_NOISE
    assert max_abs_difference(inside, logits_at(window)) > POSITION_SIGNAL


# --------------------------------------------------------------------------
# AR execution contract (Phase 1 harness plumbing, exercised on CPU)
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def ar_harness_module():
    """Import the Phase 1 harness, which lives outside the test namespace."""
    import importlib.util
    import sys

    path = (
        pathlib.Path(__file__).resolve().parent
        / "integration"
        / "nemotron_ar_harness.py"
    )
    spec = importlib.util.spec_from_file_location("nemotron_ar_harness", path)
    module = importlib.util.module_from_spec(spec)
    # `@dataclass` resolves annotations through sys.modules during exec.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ar_harness(model, config):
    """The Phase 1 harness's FluxServe side, wrapped around a small model."""
    module = ar_harness_module()
    return module, module.FluxServeAR(model, config, "cpu", max_length=64)


def test_teacher_forced_decode_matches_full_sequence_prefill():
    """The KV cache window convention must reproduce a single causal prefill.

    A decode step writes its K/V into the final slot of the window it is
    handed, so the window has to be sized to include the new token. If that
    convention is wrong the cached path silently attends to stale or shifted
    keys, which this catches without a GPU or a reference model.
    """
    config = tiny_config()
    model = initialized_tiny_model()
    _, harness = ar_harness(model, config)

    ids = torch.randint(0, config.vocab_size, (1, 12))
    full, _, _ = harness.prefill(ids)

    split = 5
    _, cache, cache_length = harness.prefill(ids[:, :split])
    for offset in range(split, ids.shape[1]):
        step_logits, cache_length = harness.decode_step(
            ids[:, offset : offset + 1], offset, cache, cache_length
        )
        expected = full[:, offset]
        assert cache_length == offset + 1
        assert max_abs_difference(step_logits, expected) < 2e-3
        assert torch.equal(step_logits.argmax(-1), expected.argmax(-1))


def test_ar_prefill_is_causal():
    """Changing a later token must not move an earlier position's logits."""
    config = tiny_config()
    model = initialized_tiny_model()
    _, harness = ar_harness(model, config)

    ids = torch.randint(0, config.vocab_size, (1, 8))
    baseline, _, _ = harness.prefill(ids)

    edited = ids.clone()
    edited[0, -1] = (int(edited[0, -1]) + 1) % config.vocab_size
    changed, _, _ = harness.prefill(edited)

    assert max_abs_difference(baseline[:, :-1], changed[:, :-1]) < 1e-5
    assert max_abs_difference(baseline[:, -1], changed[:, -1]) > POSITION_SIGNAL


def test_teacher_forced_split_leaves_requested_steps():
    module, _ = ar_harness(initialized_tiny_model(), tiny_config())
    assert module.teacher_forced_split(32, 8) == 24
    assert module.teacher_forced_split(1, 8) == 1
    assert module.teacher_forced_split(4, 8) == 1


def test_metrics_report_the_protocol_quantities():
    module, _ = ar_harness(initialized_tiny_model(), tiny_config())
    reference = torch.tensor([[[0.0, 2.0, 1.0], [3.0, 0.0, 0.0]]])
    identical = module.tensor_metrics(reference.clone(), reference)
    assert identical["max_abs"] == 0.0
    assert identical["nrms"] == 0.0
    assert identical["top1_agreement"] == 1.0
    assert identical["disagreement_positions"] == []
    # Reference top1/top2 margins: 2-1 = 1 and 3-0 = 3.
    assert identical["reference_margin_min"] == pytest.approx(1.0)

    flipped = reference.clone()
    flipped[0, 0] = torch.tensor([0.0, 1.0, 2.0])
    changed = module.tensor_metrics(flipped, reference)
    assert changed["top1_agreement"] == pytest.approx(0.5)
    assert changed["disagreement_positions"] == [0]
    assert changed["disagreement_margins"] == [pytest.approx(1.0)]

    assert not module.tensor_metrics(
        torch.tensor([[[float("nan"), 0.0]]]), torch.tensor([[[0.0, 1.0]]])
    )["finite"]


# --------------------------------------------------------------------------
# Entry-point normalization (shared by the server and the offline bench)
# --------------------------------------------------------------------------


def serve_args(**overrides):
    values = dict(
        attention_backend="sdpa",
        kv_cache_layout="dense",
        use_cuda_graph=False,
        use_prefill_cuda_graph=False,
        use_decode_cuda_graph=False,
        scheduler_policy="",
        parallel_decoding="threshold",
        block_length=None,
        max_model_len=4096,
        max_thinking_tokens=None,
        end_think_token_id=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_normalization_is_the_dispatch_predicate():
    from fluxserve.backend.model_loader.nemotron import normalize_nemotron_args

    assert normalize_nemotron_args(serve_args(), checkpoint_config()) is True
    assert (
        normalize_nemotron_args(
            serve_args(),
            SimpleNamespace(architectures=["LLaDA2MoeModelLM"], model_type="llada2_moe"),
        )
        is False
    )


def test_normalization_defaults_block_length_to_the_checkpoint_block_size():
    from fluxserve.backend.model_loader.nemotron import normalize_nemotron_args

    args = serve_args()
    normalize_nemotron_args(args, checkpoint_config())
    assert args.block_length == 32

    doubled = serve_args(block_length=64)
    normalize_nemotron_args(doubled, checkpoint_config())
    assert doubled.block_length == 64

    with pytest.raises(ValueError, match="multiple of"):
        normalize_nemotron_args(serve_args(block_length=48), checkpoint_config())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"attention_backend": "flex"}, "attention_backend"),
        ({"attention_backend": "flashinfer"}, "kv-cache-layout paged"),
        ({"attention_backend": "fa4"}, "kv-cache-layout paged"),
        ({"use_decode_cuda_graph": True}, "paged FA4 or FlashInfer paths"),
        ({"use_cuda_graph": True}, "paged FA4 or FlashInfer paths"),
        ({"use_prefill_cuda_graph": True}, "not captured"),
        ({"scheduler_policy": "paged"}, "requires"),
        ({"parallel_decoding": "levenshtein_joint"}, "supports --parallel-decoding"),
        # Continuous scheduling needs the paged FA4 path; the dense
        # self-speculation runner serves one request at a time.
        ({"parallel_decoding": "self_speculation", "scheduler_policy": "paged"},
         "attention-backend fa4"),
        ({"max_model_len": 262145}, "262144"),
    ],
)
def test_normalization_rejects_unsupported_combinations(overrides, message):
    from fluxserve.backend.model_loader.nemotron import normalize_nemotron_args

    with pytest.raises(ValueError, match=message):
        normalize_nemotron_args(serve_args(**overrides), checkpoint_config())


def test_paged_fa4_combination_is_accepted():
    from fluxserve.backend.model_loader.nemotron import normalize_nemotron_args

    args = serve_args(attention_backend="fa4", kv_cache_layout="paged")
    assert normalize_nemotron_args(args, checkpoint_config()) is True


def test_runner_config_ids_are_replaced_with_the_checkpoints():
    from fluxserve.backend.execution.forward_batch_info import RunnerConfig
    from fluxserve.backend.model_loader.nemotron import (
        apply_nemotron_runner_config,
    )

    runner_config = RunnerConfig()
    assert runner_config.mask_id == 156895  # LLaDA's, outside this vocabulary
    apply_nemotron_runner_config(runner_config, checkpoint_config())
    assert (runner_config.mask_id, runner_config.eos_id) == (100, 11)
    assert runner_config.eos_ids == (11,)


def test_self_speculation_is_selected_through_the_existing_flag():
    from fluxserve.backend.model_loader.nemotron import (
        NEMOTRON_DECODING_MODES,
        SELF_SPECULATION,
        normalize_nemotron_args,
    )

    assert SELF_SPECULATION in NEMOTRON_DECODING_MODES
    args = serve_args(parallel_decoding=SELF_SPECULATION)
    assert normalize_nemotron_args(args, checkpoint_config()) is True
    assert args.block_length == 32


def test_decode_graphs_are_accepted_on_the_paged_path():
    from fluxserve.backend.model_loader.nemotron import normalize_nemotron_args

    args = serve_args(
        attention_backend="fa4", kv_cache_layout="paged",
        use_decode_cuda_graph=True,
    )
    assert normalize_nemotron_args(args, checkpoint_config()) is True


def test_paged_scheduling_is_accepted_on_the_paged_path():
    from fluxserve.backend.model_loader.nemotron import normalize_nemotron_args

    args = serve_args(
        attention_backend="fa4", kv_cache_layout="paged",
        scheduler_policy="paged",
    )
    assert normalize_nemotron_args(args, checkpoint_config()) is True


def test_snapshot_resolution_can_refuse_to_download(monkeypatch):
    """A cache lookup must not become a 27 GB transfer inside a unit test."""
    import huggingface_hub

    from fluxserve.backend.model_loader import nemotron as loader

    seen = {}

    def fake_download(**kwargs):
        seen.update(kwargs)
        raise FileNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    with pytest.raises(FileNotFoundError):
        loader.resolve_nemotron_snapshot(
            checkpoint_config(_name_or_path="org/absent"), local_files_only=True
        )
    assert seen["local_files_only"] is True
    assert seen["allow_patterns"] == ["model.safetensors", "*.json", "*.jinja"], (
        "a bare *.safetensors pattern would also pull the LoRA adapter"
    )


def test_dense_serving_does_not_advertise_a_paged_cache():
    """`--kv-cache-layout` defaults to paged; the dense runner ignores it."""
    from fluxserve.backend.model_loader.nemotron import normalize_nemotron_args

    args = serve_args(attention_backend="sdpa", kv_cache_layout="paged")
    assert normalize_nemotron_args(args, checkpoint_config()) is True
    assert args.kv_cache_layout == "dense"

    paged = serve_args(attention_backend="fa4", kv_cache_layout="paged")
    normalize_nemotron_args(paged, checkpoint_config())
    assert paged.kv_cache_layout == "paged", "the FA4 path keeps its paged cache"


def test_paged_self_speculation_is_supported():
    """Stage 4c: the offline paged path preallocates a page table per sequence,
    so rollback is a prefix length there, not page accounting."""
    from fluxserve.backend.model_loader.nemotron import normalize_nemotron_args

    args = serve_args(
        parallel_decoding="self_speculation",
        attention_backend="fa4",
        kv_cache_layout="paged",
    )
    assert normalize_nemotron_args(args, checkpoint_config()) is True

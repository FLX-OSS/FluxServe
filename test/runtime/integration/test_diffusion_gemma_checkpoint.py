import gc
import os

import pytest
import torch

import fluxserve  # noqa: F401 - registers the native Transformers config
from transformers import AutoConfig, AutoTokenizer

from fluxserve.backend.distributed.launch import (
    destroy_distributed,
    initialize_distributed,
)
from fluxserve.backend.execution.forward_batch_info import RunnerConfig
from fluxserve.backend.execution.runners.diffusion_gemma import DiffusionGemmaRunner
from fluxserve.backend.layers.dp_attention import initialize_dp_attention
from fluxserve.backend.layers.moe import initialize_moe_config
from fluxserve.backend.models.diffusion_gemma import (
    DiffusionGemmaForConditionalGeneration,
)
from fluxserve.backend.utils.server_args import ServerArgs

MODEL_ID = "google/diffusiongemma-26B-A4B-it"
RUN_SMOKE = "FLUXSERVE_RUN_DIFFUSION_GEMMA_SMOKE"


@pytest.mark.skipif(
    os.environ.get(RUN_SMOKE) != "1",
    reason=f"set {RUN_SMOKE}=1 to load the official 26B checkpoint",
)
def test_official_diffusion_gemma_checkpoint_smoke():
    if not torch.cuda.is_available():
        pytest.skip("Diffusion-Gemma checkpoint smoke test requires CUDA")

    model_name = os.environ.get("DIFFUSION_GEMMA_MODEL", MODEL_ID)
    device = "cuda:0"
    torch.cuda.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
    )
    model_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
    )
    model_config.quant_config = None
    prompt = tokenizer(
        "Explain why the sky is blue in one sentence.", return_tensors="pt"
    )["input_ids"].to(device)

    server_args = ServerArgs(
        model_name=model_name,
        model_config=model_config,
        device=device,
        tp_size=1,
        dp_size=1,
        ep_size=1,
        pp_size=1,
    )
    runner_config = RunnerConfig(
        gen_length=8,
        block_length=8,
        canvas_length=8,
        max_denoising_steps=2,
        mini_batch_size=1,
        max_length=prompt.shape[1] + 8,
        prefill_lengths=(prompt.shape[1],),
        cache_lengths=(prompt.shape[1] + 8,),
        supported_batch_sizes=(1,),
        attention_backend="sdpa",
        kv_cache_layout="dense",
    )

    runner = None
    initialize_distributed(server_args)
    try:
        initialize_dp_attention(server_args=server_args, model_config=model_config)
        initialize_moe_config(server_args)
        runner = DiffusionGemmaRunner(
            model_config=model_config,
            server_args=server_args,
            runner_config=runner_config,
            device=device,
        )

        assert isinstance(runner.model, DiffusionGemmaForConditionalGeneration)
        non_bf16 = {
            (name, parameter.dtype)
            for name, parameter in runner.model.named_parameters()
            if parameter.is_floating_point() and parameter.dtype != torch.bfloat16
        }
        assert not non_bf16
        assert runner.decoder.eos_ids == (1, 106, 50)

        forwards = []
        forward_model = runner._forward_model

        def traced_forward(**kwargs):
            result = forward_model(**kwargs)
            attention_mask = kwargs["attention_mask"]
            forwards.append(
                {
                    "self_conditioned": kwargs.get("inputs_embeds") is not None,
                    "has_prefix_cache": kwargs.get("past_key_values") is not None,
                    "all_visible": bool(attention_mask.all()),
                    "logits_dtype": result.logits.dtype,
                    "logits_finite": bool(torch.isfinite(result.logits).all()),
                    "hidden_dtype": result.hidden_states.dtype,
                    "cache_dtypes": {
                        tensor.dtype
                        for layer_cache in result.past_key_values
                        for tensor in layer_cache
                    },
                }
            )
            return result

        runner._forward_model = traced_forward
        torch.manual_seed(0)
        output = runner.generate(prompt, prompt_lengths=[prompt.shape[1]])

        assert output.shape == (1, prompt.shape[1] + 8)
        generated = output[0, prompt.shape[1] :]
        valid_ids = (generated >= 0) & (generated < runner.decoder.config.vocab_size)
        assert bool(valid_ids.all())
        assert isinstance(tokenizer.decode(generated, skip_special_tokens=True), str)

        assert forwards[0]["has_prefix_cache"] is False
        assert forwards[0]["self_conditioned"] is False
        assert forwards[0]["all_visible"] is False
        denoise = [item for item in forwards if item["self_conditioned"]]
        assert 1 <= len(denoise) <= runner.decoder.config.max_denoising_steps
        assert all(item["has_prefix_cache"] and item["all_visible"] for item in denoise)
        commit = forwards[-1]
        assert commit["has_prefix_cache"] is True
        assert commit["self_conditioned"] is False
        assert commit["all_visible"] is False
        assert all(item["hidden_dtype"] == torch.bfloat16 for item in forwards)
        assert all(item["cache_dtypes"] == {torch.bfloat16} for item in forwards)
        assert all(item["logits_dtype"] == torch.float32 for item in forwards)
        assert all(item["logits_finite"] for item in forwards)
    finally:
        runner = None
        gc.collect()
        torch.cuda.empty_cache()
        destroy_distributed()

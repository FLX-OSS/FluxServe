"""Opt-in parity check against the Transformers Diffusion-Gemma reference."""

import gc
import os

import pytest
import torch

import fluxserve  # noqa: F401
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DiffusionGemmaForBlockDiffusion,
    DiffusionGemmaGenerationConfig,
)

from fluxserve.backend.distributed.launch import destroy_distributed, initialize_distributed
from fluxserve.backend.execution.forward_batch_info import RunnerConfig
from fluxserve.backend.execution.runners.diffusion_gemma import DiffusionGemmaRunner
from fluxserve.backend.layers.dp_attention import initialize_dp_attention
from fluxserve.backend.layers.moe import initialize_moe_config
from fluxserve.backend.utils.server_args import ServerArgs

MODEL_ID = "google/diffusiongemma-26B-A4B-it"
RUN_PARITY = "FLUXSERVE_RUN_DIFFUSION_GEMMA_PARITY"
CANVAS_LENGTH = 256
MAX_DENOISING_STEPS = 48
GENERATION_LENGTH = 256


def _cosine(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def _capture(store, name):
    def hook(_module, _inputs, output):
        if name not in store:
            if isinstance(output, tuple):
                output = output[0]
            store[name] = output.detach()
    return hook


@pytest.mark.skipif(
    os.environ.get(RUN_PARITY) != "1",
    reason=f"set {RUN_PARITY}=1 to load the official 26B reference and FluxServe models",
)
def test_diffusion_gemma_transformers_parity():
    if not torch.cuda.is_available():
        pytest.skip("parity requires CUDA")
    model_name = os.environ.get("DIFFUSION_GEMMA_MODEL", MODEL_ID)
    device = "cuda:0"
    torch.cuda.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    prompts = [
        "Explain why the sky is blue in one sentence.",
        "What is 17 times 19? Give only the number.",
        "Write a Python function that returns the larger of two numbers.",
    ]
    prompt_ids = [
        tokenizer(item, return_tensors="pt")["input_ids"].to(device)
        for item in prompts
    ]

    reference = DiffusionGemmaForBlockDiffusion.from_pretrained(
        model_name, config=config, torch_dtype=torch.bfloat16,
        trust_remote_code=True, local_files_only=True,
    ).eval().to(device)
    generation_config = DiffusionGemmaGenerationConfig.from_pretrained(
        model_name, local_files_only=True
    )
    generation_config.max_new_tokens = GENERATION_LENGTH
    config.quant_config = None
    args = ServerArgs(model_name=model_name, model_config=config, device=device, tp_size=1, dp_size=1, ep_size=1, pp_size=1)
    runner_config = RunnerConfig(
        gen_length=GENERATION_LENGTH, block_length=CANVAS_LENGTH,
        canvas_length=CANVAS_LENGTH, max_denoising_steps=MAX_DENOISING_STEPS,
        mini_batch_size=1, max_length=max(item.shape[1] for item in prompt_ids) + GENERATION_LENGTH,
        prefill_lengths=(max(item.shape[1] for item in prompt_ids),),
        cache_lengths=(max(item.shape[1] for item in prompt_ids) + GENERATION_LENGTH,),
        supported_batch_sizes=(1,), attention_backend="sdpa", kv_cache_layout="dense",
    )
    runner = None
    initialize_distributed(args)
    try:
        initialize_dp_attention(server_args=args, model_config=config)
        initialize_moe_config(args)
        runner = DiffusionGemmaRunner(config, args, runner_config, device=device)
        ref_layer = reference.model.decoder.layers[0]
        our_layer = runner.model.model.layers[0]
        for case_index, prompt in enumerate(prompt_ids):
            torch.manual_seed(case_index)
            canvas = torch.full((1, CANVAS_LENGTH), runner.decoder.pad_id, dtype=torch.long, device=device)
            positions = torch.arange(prompt.shape[1], prompt.shape[1] + canvas.shape[1], device=device).unsqueeze(0)
            with torch.no_grad():
                ref = reference(
                    input_ids=prompt, attention_mask=torch.ones_like(prompt),
                    decoder_input_ids=canvas,
                    decoder_attention_mask=torch.ones(1, prompt.shape[1] + canvas.shape[1], dtype=torch.bool, device=device),
                    decoder_position_ids=positions,
                )
                prefill = runner._forward_model(
                    input_ids=prompt,
                    position_ids=torch.arange(prompt.shape[1], device=device).unsqueeze(0),
                    use_cache=True, attention_mask=runner._causal_mask(prompt.shape[1], 0, device),
                )
                ours = runner._forward_model(
                    input_ids=canvas, inputs_embeds=runner.model.embed_with_self_conditioning(canvas),
                    position_ids=positions, past_key_values=prefill.past_key_values,
                    use_cache=False, attention_mask=runner._bidirectional_mask(canvas.shape[1], prompt.shape[1], device),
                )
            agreement = (ours.logits.argmax(-1) == ref.logits.argmax(-1)).float().mean().item()
            cosine = _cosine(ours.logits, ref.logits)
            ref_tokens = ref.logits.argmax(-1)
            our_tokens = ours.logits.argmax(-1)
            ref_text = tokenizer.decode(ref_tokens[0].tolist(), skip_special_tokens=False)
            our_text = tokenizer.decode(our_tokens[0].tolist(), skip_special_tokens=False)
            print(
                f"parity case={case_index} prompt={prompts[case_index]!r} "
                f"logits_cosine={cosine:.6f} argmax_agreement={agreement:.3f}\n"
                f"  reference_ids={ref_tokens[0].tolist()}\n"
                f"  fluxserve_ids={our_tokens[0].tolist()}\n"
                f"  reference_text={ref_text!r}\n"
                f"  fluxserve_text={our_text!r}"
            )
            # Compare the complete autoregressive generation loop as a
            # separate diagnostic from the single-canvas tensor probe above.
            torch.manual_seed(case_index)
            with torch.no_grad():
                ref_generation = reference.generate(
                    input_ids=prompt,
                    generation_config=generation_config,
                )
                torch.manual_seed(case_index)
                flux_generation = runner.generate(
                    prompt, prompt_lengths=[prompt.shape[1]]
                )
            ref_output = ref_generation.sequences if hasattr(ref_generation, "sequences") else ref_generation
            ref_output = ref_output[:, -GENERATION_LENGTH:]
            flux_output = flux_generation[:, -GENERATION_LENGTH:]
            generation_agreement = (ref_output == flux_output).float().mean().item()
            ref_decoded = tokenizer.decode(ref_output[0].tolist(), skip_special_tokens=True)
            flux_decoded = tokenizer.decode(flux_output[0].tolist(), skip_special_tokens=True)
            print(
                f"  generation_agreement={generation_agreement:.3f}\n"
                f"  reference_generation={ref_decoded!r}\n"
                f"  fluxserve_generation={flux_decoded!r}"
            )
    finally:
        runner = None
        del reference
        gc.collect()
        torch.cuda.empty_cache()
        destroy_distributed()

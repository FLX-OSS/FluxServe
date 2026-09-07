import gc
import os

import pytest
import torch
import torch.nn.functional as F

import fluxserve  # noqa: F401 - registers the native Transformers config
from transformers import AutoConfig

from fluxserve.backend.distributed.launch import (
    destroy_distributed,
    initialize_distributed,
)
from fluxserve.backend.execution.forward_batch_info import RunnerConfig
from fluxserve.backend.execution.runners.fa4_diffusion import FA4DiffusionRunner
from fluxserve.backend.layers.dp_attention import initialize_dp_attention
from fluxserve.backend.layers.moe import initialize_moe_config
from fluxserve.backend.utils.server_args import ServerArgs


MODEL_ID = "inclusionAI/LLaDA2.1-mini"
RUN_FA4_MODEL_PARITY = "FLUXSERVE_RUN_FA4_MODEL_PARITY"


@pytest.mark.skipif(
    os.environ.get(RUN_FA4_MODEL_PARITY) != "1",
    reason=f"set {RUN_FA4_MODEL_PARITY}=1 to load LLaDA2.x and run model parity",
)
def test_llada2_model_fa4_prefill_matches_dense_sdpa():
    if not torch.cuda.is_available():
        pytest.skip("LLaDA2.x FA4 model parity requires CUDA")
    capability = torch.cuda.get_device_capability()
    if capability[0] not in (9, 10, 11):
        pytest.skip(f"FA4 paged KV is unsupported on compute capability {capability}")
    pytest.importorskip("flash_attn.cute")

    device = "cuda:0"
    torch.cuda.set_device(0)
    model_name = os.environ.get("LLADA2_FA4_MODEL", MODEL_ID)
    model_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
    )
    model_config.quant_config = None
    server_args = ServerArgs(
        model_name=model_name,
        model_config=model_config,
        device=device,
        max_num_seqs=1,
        max_model_len=256,
        tp_size=1,
        dp_size=1,
        ep_size=1,
        pp_size=1,
    )
    runner_config = RunnerConfig(
        gen_length=64,
        block_length=64,
        mini_batch_size=1,
        max_length=256,
        prefill_lengths=(128,),
        cache_lengths=(128,),
        supported_batch_sizes=(1,),
        attention_backend="fa4",
        kv_cache_layout="paged",
        page_size=64,
    )

    runner = None
    initialize_distributed(server_args)
    try:
        initialize_dp_attention(server_args=server_args, model_config=model_config)
        initialize_moe_config(server_args)
        runner = FA4DiffusionRunner(
            model_config=model_config,
            server_args=server_args,
            runner_config=runner_config,
            device=device,
        )
        torch.manual_seed(0)
        input_ids = torch.randint(
            0,
            int(model_config.vocab_size),
            (1, 128),
            dtype=torch.long,
            device=device,
        )
        position_ids = torch.arange(128, device=device).unsqueeze(0)
        block_mask = runner.build_block_attention_mask(2, 1)[:, :128, :128]

        capture_phase = [None]
        attention_outputs = {"reference": [], "fa4": []}

        def capture_attention(_module, _args, output):
            phase = capture_phase[0]
            if phase is not None:
                attention_outputs[phase].append(output[0].detach().clone())

        hook_handles = [
            layer.attention.register_forward_hook(capture_attention)
            for layer in runner.model.model.layers
        ]

        capture_phase[0] = "reference"
        reference = runner.model(
            input_ids,
            use_cache=False,
            attention_mask=block_mask,
            position_ids=position_ids,
        )
        capture_phase[0] = None
        reference_repeat = runner.model(
            input_ids,
            use_cache=False,
            attention_mask=block_mask,
            position_ids=position_ids,
        )
        dense_hidden_error = (
            reference_repeat.hidden_states.float() - reference.hidden_states.float()
        ).abs()
        dense_logits_error = (reference_repeat.logits - reference.logits).abs()
        dense_hidden_relative_l2 = (
            torch.linalg.vector_norm(dense_hidden_error)
            / torch.linalg.vector_norm(reference.hidden_states.float())
        ).item()
        dense_logits_relative_l2 = (
            torch.linalg.vector_norm(dense_logits_error)
            / torch.linalg.vector_norm(reference.logits)
        ).item()
        dense_top1_agreement = (
            (reference_repeat.logits.argmax(dim=-1) == reference.logits.argmax(dim=-1))
            .float()
            .mean()
            .item()
        )
        print(
            "dense_repeat "
            f"hidden_max_abs={dense_hidden_error.max().item():.8f} "
            f"hidden_mean_abs={dense_hidden_error.mean().item():.8f} "
            f"hidden_relative_l2={dense_hidden_relative_l2:.8f} "
            f"logits_max_abs={dense_logits_error.max().item():.8f} "
            f"logits_mean_abs={dense_logits_error.mean().item():.8f} "
            f"logits_relative_l2={dense_logits_relative_l2:.8f} "
            f"top1_agreement={dense_top1_agreement:.8f}"
        )

        def make_paged_forward():
            runner.past_key_values = runner.allocate_kv_cache(1)
            batch = runner._make_paged_batch(
                seq_ids=torch.tensor([0], device=device),
                q_offsets=torch.tensor([0], device=device),
                q_lens=torch.tensor([128], device=device),
                max_input_len=128,
                is_prefill=True,
            )
            cache = [
                runner.past_key_values.layer_paged_kv(layer_id)
                for layer_id in range(model_config.num_hidden_layers)
            ]
            return batch, cache

        def paged_sdpa_reference(q, k_cache, v_cache, **kwargs):
            cu_q = kwargs["cu_seqlens_q"]
            kv_lens = kwargs["seqused_k"]
            page_rows = kwargs["page_table"]
            outputs = []
            groups = q.shape[1] // k_cache.shape[2]
            for task in range(kv_lens.numel()):
                q_start = int(cu_q[task])
                q_end = int(cu_q[task + 1])
                kv_len = int(kv_lens[task])
                page_count = (kv_len + k_cache.shape[1] - 1) // k_cache.shape[1]
                pages = page_rows[task, :page_count].long()
                task_k = k_cache.index_select(0, pages).flatten(0, 1)[:kv_len]
                task_v = v_cache.index_select(0, pages).flatten(0, 1)[:kv_len]
                task_k = task_k.transpose(0, 1).repeat_interleave(groups, dim=0)
                task_v = task_v.transpose(0, 1).repeat_interleave(groups, dim=0)
                outputs.append(
                    F.scaled_dot_product_attention(
                        q[q_start:q_end].transpose(0, 1).unsqueeze(0),
                        task_k.unsqueeze(0),
                        task_v.unsqueeze(0),
                        dropout_p=0.0,
                        is_causal=False,
                        scale=kwargs["softmax_scale"],
                    )[0].transpose(0, 1)
                )
            return torch.cat(outputs, dim=0)

        for layer in runner.model.model.layers:
            layer.attention.attention_forward.fa4_paged._kernel = paged_sdpa_reference
        reference_paged_batch, reference_paged_cache = make_paged_forward()
        reference_paged = runner.model(
            input_ids,
            use_cache=True,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=reference_paged_cache,
            forward_batch=reference_paged_batch,
        )
        paged_hidden_error = (
            reference_paged.hidden_states.float() - reference.hidden_states.float()
        ).abs()
        paged_logits_error = (reference_paged.logits - reference.logits).abs()
        paged_hidden_relative_l2 = (
            torch.linalg.vector_norm(paged_hidden_error)
            / torch.linalg.vector_norm(reference.hidden_states.float())
        ).item()
        paged_logits_relative_l2 = (
            torch.linalg.vector_norm(paged_logits_error)
            / torch.linalg.vector_norm(reference.logits)
        ).item()
        print(
            "paged_sdpa "
            f"hidden_max_abs={paged_hidden_error.max().item():.8f} "
            f"hidden_mean_abs={paged_hidden_error.mean().item():.8f} "
            f"hidden_relative_l2={paged_hidden_relative_l2:.8f} "
            f"logits_max_abs={paged_logits_error.max().item():.8f} "
            f"logits_mean_abs={paged_logits_error.mean().item():.8f} "
            f"logits_relative_l2={paged_logits_relative_l2:.8f}"
        )

        for layer in runner.model.model.layers:
            layer.attention.attention_forward.fa4_paged._kernel = None
        runner.past_key_values = runner.allocate_kv_cache(1)
        forward_batch = runner._make_paged_batch(
            seq_ids=torch.tensor([0], device=device),
            q_offsets=torch.tensor([0], device=device),
            q_lens=torch.tensor([128], device=device),
            max_input_len=128,
            is_prefill=True,
        )
        capture_phase[0] = "fa4"
        actual = runner.model(
            input_ids,
            use_cache=True,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=[
                runner.past_key_values.layer_paged_kv(layer_id)
                for layer_id in range(model_config.num_hidden_layers)
            ],
            forward_batch=forward_batch,
        )
        capture_phase[0] = None
        for handle in hook_handles:
            handle.remove()

        for layer_id, (reference_attention, fa4_attention) in enumerate(
            zip(
                attention_outputs["reference"],
                attention_outputs["fa4"],
                strict=True,
            )
        ):
            layer_error = (fa4_attention.float() - reference_attention.float()).abs()
            layer_relative_l2 = (
                torch.linalg.vector_norm(layer_error)
                / torch.linalg.vector_norm(reference_attention.float())
            ).item()
            print(
                f"layer={layer_id} attention_max_abs={layer_error.max().item():.8f} "
                f"attention_mean_abs={layer_error.mean().item():.8f} "
                f"attention_relative_l2={layer_relative_l2:.8f}"
            )

        hidden_error = (
            actual.hidden_states.float() - reference.hidden_states.float()
        ).abs()
        logits_error = (actual.logits - reference.logits).abs()
        hidden_relative_l2 = (
            torch.linalg.vector_norm(hidden_error)
            / torch.linalg.vector_norm(reference.hidden_states.float())
        ).item()
        logits_relative_l2 = (
            torch.linalg.vector_norm(logits_error)
            / torch.linalg.vector_norm(reference.logits)
        ).item()
        top1_agreement = (
            (actual.logits.argmax(dim=-1) == reference.logits.argmax(dim=-1))
            .float()
            .mean()
            .item()
        )
        print(
            "model_prefill "
            f"hidden_max_abs={hidden_error.max().item():.8f} "
            f"hidden_mean_abs={hidden_error.mean().item():.8f} "
            f"hidden_relative_l2={hidden_relative_l2:.8f} "
            f"logits_max_abs={logits_error.max().item():.8f} "
            f"logits_mean_abs={logits_error.mean().item():.8f} "
            f"logits_relative_l2={logits_relative_l2:.8f} "
            f"top1_agreement={top1_agreement:.8f}"
        )

        assert hidden_error.max().item() <= 0.125
        assert hidden_error.mean().item() <= 0.01
        assert hidden_relative_l2 <= 0.02
        assert logits_error.mean().item() <= 0.05
        assert logits_relative_l2 <= 0.02
        assert top1_agreement >= 0.99
    finally:
        runner = None
        gc.collect()
        torch.cuda.empty_cache()
        destroy_distributed()

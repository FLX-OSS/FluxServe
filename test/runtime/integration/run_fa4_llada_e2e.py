"""Real-weight LLaDA2.1-mini generation through FluxServe and dense SDPA."""
import argparse
import copy
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import fluxserve  # noqa: F401
from transformers import AutoConfig, AutoTokenizer
from fluxserve.backend.distributed.launch import initialize_distributed, destroy_distributed
from fluxserve.backend.execution.forward_batch_info import RunnerConfig
from fluxserve.backend.execution.runners.fa4_diffusion import FA4DiffusionRunner
from fluxserve.backend.execution.runners.block_diffusion import BlockDiffusionRunner
from fluxserve.backend.layers.dp_attention import initialize_dp_attention
from fluxserve.backend.layers.moe import initialize_moe_config
from fluxserve.backend.utils.server_args import ServerArgs
from fluxserve.backend.layers.attention.fa4 import load_fa4_varlen_func


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="inclusionAI/LLaDA2.1-mini")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-token-match", action="store_true")
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    config.quant_config = None
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    server = ServerArgs(model_name=args.model, model_config=config, device="cuda:0",
                        max_num_seqs=1, max_model_len=512, tp_size=1, dp_size=1, ep_size=1, pp_size=1)
    rc = RunnerConfig(gen_length=192, block_length=64, max_length=512,
                      prefill_lengths=(256,), cache_lengths=(512,), prefilling_limit=256,
                      mini_batch_size=1, supported_batch_sizes=(1,), parallel_decoding="threshold",
                      early_stop=False, attention_backend="fa4", kv_cache_layout="paged", page_size=64)
    initialize_distributed(server)
    initialize_dp_attention(server_args=server, model_config=config)
    initialize_moe_config(server)
    records = []
    try:
        runner = FA4DiffusionRunner(model_config=config, server_args=server, runner_config=rc, device="cuda:0")
        kernel = load_fa4_varlen_func()
        attention_checks = {"calls": 0, "max_abs": 0.0, "max_relative_l2": 0.0,
                            "max_bf16_sdpa_relative_l2": 0.0,
                            "criterion": "FP32 relative L2 <= 1%; max and L2 errors <= 2x BF16 SDPA error + 1e-5"}
        def checked_fa4(q, k, v, **kw):
            actual = kernel(q, k, v, **kw)
            output = actual[0] if isinstance(actual, tuple) else actual
            groups = q.shape[1] // k.shape[2]
            with sdpa_kernel(SDPBackend.MATH):
                for task, length in enumerate(kw["seqused_k"].tolist()):
                    lo, hi = kw["cu_seqlens_q"][task:task+2].tolist()
                    pages = kw["page_table"][task, :(length + k.shape[1] - 1)//k.shape[1]].long()
                    kk = k[pages].flatten(0, 1)[:length].transpose(0, 1).repeat_interleave(groups, 0).float()
                    vv = v[pages].flatten(0, 1)[:length].transpose(0, 1).repeat_interleave(groups, 0).float()
                    expected = F.scaled_dot_product_attention(q[lo:hi].transpose(0, 1).float(), kk, vv,
                                                             scale=kw["softmax_scale"]).transpose(0, 1)
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        bf16_reference = F.scaled_dot_product_attention(
                            q[lo:hi].transpose(0, 1).unsqueeze(0), kk.to(q.dtype).unsqueeze(0),
                            vv.to(q.dtype).unsqueeze(0), scale=kw["softmax_scale"])[0].transpose(0, 1)
                    # Bound error against independent FP32 math and the existing
                    # BF16 kernel's numerical error, not near-zero output values.
                    # Fixed pointwise atol=.02 fails even the BF16 SDPA control
                    # on real activations with cancellation and large V values.
                    error = (output[lo:hi].float() - expected).abs()
                    control_error = (bf16_reference.float() - expected).abs()
                    assert torch.isfinite(error).all()
                    assert error.max() <= 2 * control_error.max() + 1e-5
                    assert error.norm() <= 2 * control_error.norm() + 1e-5
                    norm = expected.norm().clamp_min(1e-12)
                    relative = ((output[lo:hi].float() - expected).norm() / norm).item()
                    bf_relative = ((bf16_reference.float() - expected).norm() / norm).item()
                    assert relative <= .01 and bf_relative <= .01, (relative, bf_relative)
                    attention_checks["max_relative_l2"] = max(attention_checks["max_relative_l2"], relative)
                    attention_checks["max_bf16_sdpa_relative_l2"] = max(attention_checks["max_bf16_sdpa_relative_l2"], bf_relative)
                    attention_checks["max_abs"] = max(attention_checks["max_abs"],
                                                      (output[lo:hi].float() - expected).abs().max().item())
            attention_checks["calls"] += 1
            return actual
        for layer in runner.model.model.layers:
            layer.attention.attention_forward.fa4_paged._kernel = checked_fa4
        # The baseline uses the existing dense runner and exactly the same weights.
        reference = object.__new__(BlockDiffusionRunner)
        reference.__dict__ = runner.__dict__.copy()
        reference.runner_config = copy.deepcopy(rc)
        reference.runner_config.attention_backend = "sdpa"
        reference.runner_config.kv_cache_layout = "dense"
        reference.decoder = copy.deepcopy(runner.decoder)
        for prompt, answer in [
            ("What is 2 + 2? Answer with only the number.", "4"),
            ("Read the following context carefully. " + "This is a simple geography question. " * 12
             + "What is the capital of France? Answer with only the city name.", "Paris"),
            ("Return the integers from 1 to 40 in order, separated by a comma and a space. "
             "Do not add any other text.", "count40"),
        ]:
            rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                      tokenize=False, add_generation_prompt=True)
            ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
            x = torch.tensor([ids], device="cuda:0")
            record = {"prompt": prompt, "expected": answer, "prompt_tokens": len(ids)}
            for name, engine in [("sdpa", reference), ("fa4", runner)]:
                engine.num_forwards = 0
                starts = []
                def record_forward(_model, _inputs, kwargs):
                    positions = kwargs.get("position_ids")
                    if positions is not None:
                        starts.append(int(positions.min()))
                handle = engine.model.register_forward_pre_hook(record_forward, with_kwargs=True)
                output = engine.generate(x.clone())
                handle.remove()
                # TokenArray's single-request return removes EOS globally.
                prefix_len = sum(token != engine.decoder.eos_id for token in ids)
                tokens = output[0, prefix_len:].tolist()
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                record[name] = {"text": text, "tokens": tokens, "forwards": engine.num_forwards,
                                "forward_starts": starts}
                print(name, json.dumps(record[name], ensure_ascii=False), flush=True)
                assert tokens and engine.decoder.mask_id not in tokens
                if answer == "count40":
                    assert [int(n) for n in re.findall(r"\d+", text)] == list(range(1, 41)), (name, text)
                    assert len(tokens) > 64
                else:
                    assert text.strip() == answer, (name, text)
                assert len(set(starts)) >= 2, starts
            record["exact_token_match"] = record["fa4"]["tokens"] == record["sdpa"]["tokens"]
            # Both trajectories must solve the task. Exact cross-backend token
            # identity is reported separately: BF16/MoE can change decisions.
            if args.require_token_match:
                assert record["exact_token_match"], record
            records.append(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        assert attention_checks["calls"] > 0
        args.output.write_text(json.dumps({"model": args.model, "dtype": "bfloat16",
                                          "attention_checks": attention_checks, "cases": records},
                                         ensure_ascii=False, indent=2) + "\n")
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()

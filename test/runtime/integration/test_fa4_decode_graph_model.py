"""Real-weight graph/eager parity, including page recycling and padded batches."""
import os

import pytest
import torch
import fluxserve  # noqa: F401
from transformers import AutoConfig, AutoTokenizer

from fluxserve.backend.distributed.launch import initialize_distributed, destroy_distributed
from fluxserve.backend.execution.forward_batch_info import RunnerConfig
from fluxserve.backend.execution.runners.fa4_diffusion import FA4DiffusionRunner
from fluxserve.backend.layers.dp_attention import initialize_dp_attention
from fluxserve.backend.layers.moe import initialize_moe_config
from fluxserve.backend.utils.server_args import ServerArgs


@pytest.mark.skipif(os.getenv("FLUXSERVE_RUN_FA4_GRAPH_MODEL") != "1", reason="real GH200 model test")
@torch.inference_mode()
def test_real_model_decode_graph():
    model = "inclusionAI/LLaDA2.1-mini"
    config = AutoConfig.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    config.quant_config = None
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=True)
    server = ServerArgs(model_name=model, model_config=config, device="cuda:0",
                        max_num_seqs=16, max_model_len=65536, scheduler_num_device_pages=1100,
                        tp_size=1, ep_size=1, dp_size=1, pp_size=1)
    rc = RunnerConfig(max_length=65536, block_length=64, gen_length=2048,
                      supported_batch_sizes=(1, 2, 4, 8, 16),
                      attention_backend="fa4", kv_cache_layout="paged", page_size=64,
                      parallel_decoding="joint_threshold", threshold=.7, editing_threshold=.5,
                      enable_decode_cuda_graph=True, decode_cuda_graph_mode="padded",
                      cuda_graph_capture_batch_sizes=(1, 2, 4, 8, 10, 12, 16))
    initialize_distributed(server)
    runner = None
    try:
        initialize_dp_attention(server_args=server, model_config=config)
        initialize_moe_config(server)
        runner = FA4DiffusionRunner(model_config=config, server_args=server, runner_config=rc, device="cuda:0")
        runner.prepare_online_cuda_graphs()
        graph = runner.fa4_graph_runner
        cache = runner.past_key_values
        torch.manual_seed(123)
        # Real token inputs, with protected prompt positions and masked suffix.
        prompt_ids = tokenizer.encode("Solve carefully: Janet has 16 apples and sells 7. How many remain?")
        base = torch.tensor((prompt_ids * 64)[:64], device="cuda")
        for case, batch_size in enumerate((*range(1, 17), 3, 1, 1)):
            seq_ids = torch.arange(batch_size - 1, -1, -1, device="cuda")
            # Every row gets disjoint physical pages, shuffled on each replay.
            page_rows = torch.randperm(1088, device="cuda").reshape(16, 68) + 1
            cache.page_table.zero_()
            cache.page_table[:, :68].copy_(page_rows)
            cache.data.normal_(std=.1)
            lengths = torch.tensor([64, 128, 512, 1024, 2048, 4096], device="cuda")
            offsets = lengths[(torch.arange(batch_size, device="cuda") + case) % len(lengths)]
            if case == 18:
                # Exercise the actual configured 64K boundary, not just capacity.
                cache.page_table[0].copy_(torch.randperm(1024, device="cuda") + 1)
                offsets.fill_(65536 - 64)
            positions = offsets[:, None] + torch.arange(64, device="cuda")
            ids = base.repeat(batch_size, 1)
            ids[:, 17 + case % 16:] = runner.decoder.mask_id
            prompt = torch.arange(64, device="cuda")[None, :] < 17
            prompt = prompt.expand(batch_size, -1).contiguous()
            allow = torch.arange(batch_size, device="cuda") % 2 == 0
            # Snapshot entire real pool: catches padding writes into other rows.
            original = cache.data.clone()
            fb = runner._make_paged_batch(seq_ids=seq_ids, q_offsets=offsets,
                q_lens=torch.full_like(offsets, 64), max_input_len=64, is_prefill=False)
            kv = [cache.layer_paged_kv(i) for i in range(cache.num_layers)]
            hidden, _ = runner.model.model(ids, positions, kv, use_cache=False,
                                            attention_mask=None, forward_batch=fb)
            expected_logits = runner.model._get_logits(hidden)
            expected = runner.decoder.graph_step(expected_logits, ids, prompt, allow)
            expected_cache = cache.data[:, :, :cache.dummy_page_id].clone()
            cache.data.copy_(original)
            actual = graph.replay(runner, ids, positions, seq_ids, prompt, allow)
            torch.cuda.synchronize()
            # Both fused steps suppress the mask column to -inf.
            torch.testing.assert_close(actual.logits, expected_logits, atol=0, rtol=0)
            for observed, reference in zip(actual.step, expected):
                torch.testing.assert_close(observed, reference, atol=0, rtol=0)
            torch.testing.assert_close(cache.data[:, :, :cache.dummy_page_id], expected_cache,
                                       atol=0, rtol=0)
            finite = torch.isfinite(expected_logits)
            max_error = (actual.logits[finite] - expected_logits[finite]).abs().max().item()
            print(f"PASS case={case} batch={batch_size} kv={offsets.tolist()} logits_max_abs={max_error} tokens/flags exact", flush=True)
            del original, expected_cache, hidden, expected_logits, expected, actual
        assert graph.replay_count == 19
        assert len(graph.entries) == 7
        assert graph.padded_rows > 0
        print(graph.stats(), flush=True)
        runner.ensure_paged_kv_cache(num_device_pages=1101)
        assert not graph.entries and graph.cache is None
        with pytest.raises(RuntimeError, match="allocation changed"):
            graph.replay(runner, ids, positions, seq_ids, prompt, allow)
    finally:
        if runner is not None:
            runner.shutdown_cuda_graphs()
        destroy_distributed()

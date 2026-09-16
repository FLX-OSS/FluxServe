"""Four-GPU regression for the standard MoE EP path captured by FA4 graphs."""
from datetime import timedelta

import pytest
import torch


def _ep_graph_worker(rank, rendezvous):
    import torch.distributed as dist
    from transformers import PretrainedConfig
    from fluxserve.backend.distributed import initialize_model_parallel
    from fluxserve.backend.execution.cuda_graph_runner import model_capture_mode
    from fluxserve.backend.models.llada2 import LLaDA2SparseMoeBlock

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank,
                            world_size=4, timeout=timedelta(seconds=180))
    try:
        initialize_model_parallel(tensor_model_parallel_size=4,
                                  expert_model_parallel_size=4)
        config = PretrainedConfig(
            hidden_size=128, moe_intermediate_size=128,
            num_experts=8, num_experts_per_tok=2, num_shared_experts=1,
            norm_topk_prob=True, hidden_act="silu",
        )
        block = LLaDA2SparseMoeBlock(
            0, config, alt_stream=torch.cuda.Stream()
        ).cuda().to(torch.bfloat16)
        with torch.no_grad():
            torch.manual_seed(17)
            for name, parameter in block.named_parameters():
                parameter.uniform_(-0.05, 0.05)
                if name.startswith("experts."):
                    parameter.mul_(1 + rank * 0.1)
            x = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), model_capture_mode():
                for _ in range(3):
                    block(x)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with model_capture_mode(), torch.cuda.graph(graph, stream=stream):
                captured = block(x)
            for seed in (31, 47):
                torch.manual_seed(seed)
                x.copy_(torch.randn_like(x))
                eager = block(x)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(captured, eager, atol=0.01, rtol=0.01)
                replicas = [torch.empty_like(captured) for _ in range(4)]
                dist.all_gather(replicas, captured)
                for replica in replicas:
                    torch.testing.assert_close(replica, captured, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four CUDA GPUs")
def test_standard_moe_tp4_ep4_graph_matches_eager(tmp_path):
    torch.multiprocessing.spawn(
        _ep_graph_worker, args=(f"file://{tmp_path / 'nccl-init'}",),
        nprocs=4, join=True,
    )

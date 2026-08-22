from types import SimpleNamespace

import pytest
import torch

from fluxserve.backend.metrics import count_completion_tokens
from fluxserve.bench_offline import (
    bucket_length,
    calc_padded_gen_lens,
    compact_batch_output,
    cut_eos,
    load_openai_style_inputs,
    normalize_diffusion_gemma_args,
    warmup_runner,
)
from fluxserve.cli import build_parser, normalize_diffusion_gemma_serve_args
from fluxserve.backend.execution.runners.diffusion_gemma import DiffusionGemmaRunner


def test_generation_bucket_rounds_up_without_shortening_request():
    args = SimpleNamespace(gen_len=8)
    inputs = [torch.zeros(1, 10), torch.zeros(1, 31)]

    assert bucket_length(1) == 32
    assert bucket_length(32) == 32
    assert bucket_length(33) == 64
    assert calc_padded_gen_lens(args, inputs) == [22, 33]
    assert all(length >= args.gen_len for length in calc_padded_gen_lens(args, inputs))


def test_compact_batch_output_removes_padding_before_generation():
    inputs = [torch.tensor([[2, 3]]), torch.tensor([[4, 5, 6, 7]])]
    # Generated tokens begin after the common padded prompt width of four.
    output = torch.tensor(
        [
            [2, 3, 0, 0, 10, 106, 12, 0],
            [4, 5, 6, 7, 20, 21, 22, 23],
        ]
    )

    compacted = compact_batch_output(output, inputs, [3, 4], mask_id=0)

    assert compacted.tolist() == [
        [2, 3, 10, 106, 12, 0, 0, 0],
        [4, 5, 6, 7, 20, 21, 22, 23],
    ]
    assert count_completion_tokens(compacted[0], 2, (1, 106, 50), 0) == 2


def test_benchmark_eos_helpers_use_earliest_configured_id():
    generated = torch.tensor([[7, 106, 8, 1, 50]])
    assert cut_eos(generated, (1, 106, 50)).tolist() == [[7]]

    row = torch.tensor([20, 21, 7, 50, 8, 1])
    assert count_completion_tokens(row, 2, (1, 106, 50), mask_id=0) == 2
    assert count_completion_tokens(row, 2, 1, mask_id=0) == 4
    with pytest.raises(ValueError, match="at least one EOS"):
        cut_eos(generated, ())


def test_diffusion_gemma_cli_options_reach_effective_canvas_config():
    args = build_parser().parse_args(
        [
            "bench_offline",
            "--model",
            "model",
            "--dataset",
            "dataset.jsonl",
            "--block-length",
            "64",
            "--canvas-length",
            "8",
            "--max-denoising-steps",
            "2",
        ]
    )
    config = SimpleNamespace(
        model_type="diffusion_gemma",
        architectures=["DiffusionGemmaForBlockDiffusion"],
        canvas_length=256,
    )

    assert normalize_diffusion_gemma_args(args, config)
    assert args.canvas_length == 8
    assert args.block_length == 8
    assert args.max_denoising_steps == 2
    assert args.attention_backend == "sdpa"
    assert args.kv_cache_layout == "dense"


def test_diffusion_gemma_cli_uses_checkpoint_canvas_by_default():
    args = build_parser().parse_args(
        ["bench_offline", "--model", "model", "--dataset", "dataset.jsonl"]
    )
    config = SimpleNamespace(
        model_type="diffusion_gemma",
        architectures=["DiffusionGemmaForBlockDiffusion"],
        canvas_length=256,
    )

    assert normalize_diffusion_gemma_args(args, config)
    assert args.canvas_length == 256
    assert args.block_length == 256


def test_diffusion_gemma_allows_decode_only_cuda_graph():
    args = build_parser().parse_args(
        [
            "bench_offline", "--model", "model", "--dataset", "dataset.jsonl",
            "--attention-backend", "flashinfer",
            "--use-decode-cuda-graph",
        ]
    )
    config = SimpleNamespace(
        model_type="diffusion_gemma",
        architectures=["DiffusionGemmaForBlockDiffusion"],
        canvas_length=256,
    )
    assert normalize_diffusion_gemma_args(args, config)
    assert args.use_decode_cuda_graph
    assert not args.use_prefill_cuda_graph


def test_diffusion_gemma_online_decode_graph_selects_flashinfer_paged_defaults():
    args = build_parser().parse_args(
        ["serve", "--model", "model", "--use-decode-cuda-graph"]
    )
    config = SimpleNamespace(
        model_type="diffusion_gemma",
        architectures=["DiffusionGemmaForBlockDiffusion"],
    )

    assert normalize_diffusion_gemma_serve_args(args, config)
    assert args.attention_backend == "flashinfer"
    assert args.flashinfer_prefill_mode == "paged"
    assert args.flashinfer_cache_mode == "paged"
    assert args.kv_cache_layout == "paged"


def test_diffusion_gemma_online_accepts_canvas_override():
    args = build_parser().parse_args(
        ["serve", "--model", "model", "--canvas-length", "32"]
    )
    config = SimpleNamespace(
        model_type="diffusion_gemma",
        architectures=["DiffusionGemmaForBlockDiffusion"],
        canvas_length=256,
    )

    assert normalize_diffusion_gemma_serve_args(args, config)
    assert args.canvas_length == 32


def test_diffusion_gemma_online_rejects_invalid_canvas_override():
    args = build_parser().parse_args(
        ["serve", "--model", "model", "--canvas-length", "0"]
    )
    config = SimpleNamespace(model_type="diffusion_gemma", architectures=[])

    with pytest.raises(ValueError, match="canvas-length must be positive"):
        normalize_diffusion_gemma_serve_args(args, config)


@pytest.mark.parametrize("flag", ["--use-cuda-graph", "--use-prefill-cuda-graph"])
def test_diffusion_gemma_online_rejects_non_decode_graphs(flag):
    args = build_parser().parse_args(["serve", "--model", "model", flag])
    config = SimpleNamespace(model_type="diffusion_gemma", architectures=[])

    with pytest.raises(ValueError, match="decode CUDA graphs only"):
        normalize_diffusion_gemma_serve_args(args, config)


def test_diffusion_gemma_online_graph_startup_preallocates_stable_cache(monkeypatch):
    calls = []
    logs = []

    class GraphRunner:
        gemma_max_entries = 1
        gemma_capture_enabled = False
        capture_time_s = 0.0
        decode_capture_batch_sizes = (1, 2, 4, 6)

        def log(self, message, *args):
            logs.append(message % args)

        def reset_serving_counts(self):
            pass

        def stats(self):
            return {
                "gemma_decode_max_entries": self.gemma_max_entries,
                "capture_time_s": self.capture_time_s,
            }

    runner = SimpleNamespace(
        flashinfer_graph_runner=GraphRunner(),
        server_args=SimpleNamespace(max_num_seqs=6),
        supported_batch_sizes=[1, 2, 4],
        runner_config=SimpleNamespace(
            supported_batch_sizes=(1, 2, 4),
            cuda_graph_capture_batch_sizes=(1, 2, 4, 6),
        ),
        max_length=2048,
        device=torch.device("cuda:0"),
        _paged_cache=lambda max_length, batch_size: calls.append(
            (max_length, batch_size)
        ),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)

    stats = DiffusionGemmaRunner.prepare_online_cuda_graphs(runner)

    assert calls == [(2048, 6)]
    assert runner.supported_batch_sizes == [1, 2, 4, 6]
    assert runner.runner_config.supported_batch_sizes == (1, 2, 4, 6)
    assert runner.flashinfer_graph_runner.gemma_capture_enabled
    assert stats["gemma_decode_max_entries"] == 4
    assert "lazy_batch_buckets=1,2,4,6" in logs[0]


def test_diffusion_gemma_graph_warmup_skips_homogeneous_cache_allocation(
    monkeypatch,
):
    graph_runner = SimpleNamespace(
        supports_llada2_graphs=False,
        supports_diffusion_gemma_graphs=True,
        gemma_capture_enabled=True,
        gemma_capture_count=1,
        gemma_max_entries=1,
        record_capture_memory=lambda *_: None,
    )
    generated_shapes = []
    runner = SimpleNamespace(
        runner_config=SimpleNamespace(gen_length=32),
        flashinfer_graph_runner=graph_runner,
        generate=lambda ids: generated_shapes.append(tuple(ids.shape)),
        allocate_kv_cache=lambda *_: pytest.fail(
            "Diffusion-Gemma must not use the homogeneous KV allocator"
        ),
    )
    args = SimpleNamespace(
        use_cuda_graph=False,
        use_prefill_cuda_graph=False,
        use_decode_cuda_graph=True,
        attention_backend="flashinfer",
        flashinfer_prefill_mode="paged",
        flashinfer_cache_mode="paged",
        kv_cache_layout="paged",
        batch_size=4,
        mini_batch_size=2,
        block_length=256,
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *_: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *_: 0)

    warmup_runner(
        runner,
        args,
        torch.device("cpu"),
        SimpleNamespace(info=lambda *_: None),
    )

    assert generated_shapes == [(4, 256)]
    assert not graph_runner.gemma_capture_enabled
    assert runner.runner_config.gen_length == 32


def test_diffusion_gemma_benchmark_uses_tokenizer_chat_template(tmp_path):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text(
        '{"messages": [{"role": "user", "content": "Hello"}], '
        '"metadata": {"task_id": "sample"}}\n'
    )

    class Tokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert messages == [{"role": "user", "content": "Hello"}]
            assert tokenize is False
            assert add_generation_prompt is True
            return "<bos><|turn>user\nHello<turn|>\n<|turn>model\n"

        def __call__(self, prompt):
            assert prompt.startswith("<bos><|turn>user")
            return {"input_ids": [2, 105, 2364, 107]}

    input_ids, prompts, questions, ids = load_openai_style_inputs(
        dataset,
        Tokenizer(),
        apply_chat_template=True,
    )

    assert input_ids[0].tolist() == [[2, 105, 2364, 107]]
    assert prompts[0].startswith("<bos><|turn>user")
    assert questions == ["Hello"]
    assert ids == ["sample"]

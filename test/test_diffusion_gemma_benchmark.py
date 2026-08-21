from types import SimpleNamespace

import pytest
import torch

from fluxserve.backend.metrics import count_completion_tokens
from fluxserve.bench_offline import (
    bucket_length,
    calc_padded_gen_lens,
    cut_eos,
    load_openai_style_inputs,
    normalize_diffusion_gemma_args,
)
from fluxserve.cli import build_parser


def test_generation_bucket_rounds_up_without_shortening_request():
    args = SimpleNamespace(gen_len=8)
    inputs = [torch.zeros(1, 10), torch.zeros(1, 31)]

    assert bucket_length(1) == 32
    assert bucket_length(32) == 32
    assert bucket_length(33) == 64
    assert calc_padded_gen_lens(args, inputs) == [22, 33]
    assert all(length >= args.gen_len for length in calc_padded_gen_lens(args, inputs))


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

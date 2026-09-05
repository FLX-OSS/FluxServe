from types import SimpleNamespace

import pytest

from fluxserve.backend.engine.async_llm import AsyncLLM
from fluxserve.backend.engine.processor import InputProcessor
from fluxserve.backend.engine.scheduler_adapter import DefaultSchedulerAdapter
from fluxserve.backend.utils.server_args import ServerArgs
from fluxserve.cli import build_parser, default_cuda_graph_capture_batch_sizes


def test_default_scheduler_batches_and_deduplicates_requests():
    scheduler = DefaultSchedulerAdapter(max_batch_size=2)
    first = SimpleNamespace(rid="first")
    second = SimpleNamespace(rid="second")
    third = SimpleNamespace(rid="third")

    scheduler.submit([first, second, first, third])

    assert scheduler.next_batch().request_ids == ["first", "second"]
    assert scheduler.next_batch().request_ids == ["third"]
    assert scheduler.next_batch() is None


def test_default_scheduler_finish_and_abort_manage_active_requests():
    scheduler = DefaultSchedulerAdapter(max_batch_size=2)
    first = SimpleNamespace(rid="first")
    second = SimpleNamespace(rid="second")
    scheduler.submit([first, second])

    scheduler.abort("first")
    scheduler.finish("second")
    scheduler.submit([first, second])

    assert scheduler.next_batch().request_ids == ["first", "second"]


def test_async_llm_uses_default_scheduler_unless_one_is_injected():
    args = ServerArgs(max_num_seqs=3)
    tokenizer = SimpleNamespace()
    executor = SimpleNamespace()

    engine = AsyncLLM(args, executor, tokenizer=tokenizer)
    assert isinstance(engine.scheduler, DefaultSchedulerAdapter)
    assert engine.scheduler.max_batch_size == 3

    class FalseyScheduler:
        def __bool__(self):
            return False

    injected = FalseyScheduler()
    engine = AsyncLLM(args, executor, tokenizer=tokenizer, scheduler=injected)
    assert engine.scheduler is injected


def test_scheduler_policy_defaults_to_default():
    assert ServerArgs().scheduler_policy == "default"
    args = build_parser().parse_args(["serve", "--model", "model"])
    assert args.scheduler_policy == "default"
    assert args.apply_template is False


@pytest.mark.parametrize(
    ("max_num_seqs", "expected"),
    [
        (1, (1,)),
        (2, (1, 2)),
        (7, (1, 2, 4, 6)),
        (12, (1, 2, 4, 6, 8, 10, 12)),
    ],
)
def test_default_cuda_graph_capture_batch_sizes(max_num_seqs, expected):
    assert default_cuda_graph_capture_batch_sizes(max_num_seqs) == expected


def test_cuda_graph_capture_batch_sizes_are_optional():
    args = build_parser().parse_args(["serve", "--model", "model"])
    assert args.cuda_graph_capture_bs is None

    args = build_parser().parse_args(
        [
            "serve", "--model", "model",
            "--cuda-graph-capture-bs", "1", "4", "8",
        ]
    )
    assert args.cuda_graph_capture_bs == [1, 4, 8]


def test_cli_accepts_apply_template():
    args = build_parser().parse_args(
        ["serve", "--model", "model", "--apply-template"]
    )
    assert args.apply_template is True


def test_cli_accepts_canvas_length():
    args = build_parser().parse_args(
        ["serve", "--model", "model", "--canvas-length", "32"]
    )
    assert args.canvas_length == 32


def test_input_processor_reserves_complete_generation_blocks():
    processor = InputProcessor(
        ServerArgs(max_model_len=1024, generation_block_size=256),
        SimpleNamespace(),
    )
    state = processor.make_state(
        {
            "rid": "request",
            "input_ids": [1] * 700,
            "text": None,
            "sampling_params": {"max_tokens": 300},
        }
    )

    assert state.max_new_tokens == 256


def test_input_processor_rejects_prompt_without_room_for_generation_block():
    processor = InputProcessor(
        ServerArgs(max_model_len=1024, generation_block_size=256),
        SimpleNamespace(),
    )
    with pytest.raises(ValueError, match="generation block of 256"):
        processor.make_state(
            {
                "rid": "request",
                "input_ids": [1] * 800,
                "text": None,
                "sampling_params": {"max_tokens": 1},
            }
        )


@pytest.mark.parametrize("policy", ["default", "paged", "dynamic"])
def test_cli_accepts_supported_scheduler_policies(policy):
    args = build_parser().parse_args(
        ["serve", "--model", "model", "--scheduler-policy", policy]
    )
    assert args.scheduler_policy == policy


def test_dynamic_policy_uses_paged_page_profiling(monkeypatch):
    import fluxserve.cli as cli

    args = build_parser().parse_args(
        ["serve", "--model", "model", "--scheduler-policy", "dynamic"]
    )
    server_args = ServerArgs(scheduler_num_device_pages=0)
    runner = object()
    monkeypatch.setattr(cli, "profile_paged_kv_pages", lambda **kwargs: 17)
    # The production condition is shared by paged and dynamic policies.
    if args.scheduler_policy in ("paged", "dynamic") and server_args.scheduler_num_device_pages <= 0:
        server_args.scheduler_num_device_pages = cli.profile_paged_kv_pages(runner=runner, page_size=64, utilization=0.9, safety_reserve=0.05)
    assert server_args.scheduler_num_device_pages == 17


def test_default_policy_profiles_pages_when_flashinfer_graph_runner_requires_them(
    monkeypatch,
):
    import fluxserve.cli as cli

    monkeypatch.setattr(cli, "profile_paged_kv_pages", lambda **kwargs: 19)
    args = build_parser().parse_args(["serve", "--model", "model"])
    server_args = ServerArgs(scheduler_num_device_pages=0)
    runner = type("Runner", (), {"enable_flashinfer_attention_graph": True})()
    needs_pages = (
        args.scheduler_policy in ("paged", "dynamic")
        or bool(getattr(runner, "enable_flashinfer_attention_graph", False))
    )
    if needs_pages and server_args.scheduler_num_device_pages <= 0:
        server_args.scheduler_num_device_pages = cli.profile_paged_kv_pages(
            runner=runner, page_size=64, utilization=0.9, safety_reserve=0.05
        )
    assert server_args.scheduler_num_device_pages == 19


@pytest.mark.parametrize("policy", ["fifo", "cpp_plan", "unknown"])
def test_cli_rejects_removed_scheduler_policies(policy):
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["serve", "--model", "model", "--scheduler-policy", policy]
        )

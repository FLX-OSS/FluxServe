from types import SimpleNamespace

import pytest

from fluxserve.backend.engine.async_llm import AsyncLLM
from fluxserve.backend.engine.scheduler_adapter import DefaultSchedulerAdapter
from fluxserve.backend.utils.server_args import ServerArgs
from fluxserve.cli import build_parser


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


def test_cli_accepts_apply_template():
    args = build_parser().parse_args(
        ["serve", "--model", "model", "--apply-template"]
    )
    assert args.apply_template is True


@pytest.mark.parametrize("policy", ["default", "paged"])
def test_cli_accepts_supported_scheduler_policies(policy):
    args = build_parser().parse_args(
        ["serve", "--model", "model", "--scheduler-policy", policy]
    )
    assert args.scheduler_policy == policy


@pytest.mark.parametrize("policy", ["fifo", "cpp_plan", "unknown"])
def test_cli_rejects_removed_scheduler_policies(policy):
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["serve", "--model", "model", "--scheduler-policy", policy]
        )

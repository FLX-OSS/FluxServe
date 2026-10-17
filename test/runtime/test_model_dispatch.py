"""Checkpoint config + CLI flags -> runner, scheduler and server settings.

``fluxserve launch`` and ``fluxserve bench_offline`` each decide, inline, which
runner class serves a checkpoint and how the engine around it is configured.
These tests drive both commands end to end with every GPU-touching dependency
stubbed out, and record what they would have built. They pin today's dispatch
table so it can be moved into a single registry without changing behaviour.

LLaDA2 must not be used in tests, so only Nemotron and Diffusion-Gemma
checkpoint configs are dispatched here. The LLaDA runner classes are still
stubbed: the CLI imports them and the Nemotron runners inherit from them.
"""

from types import SimpleNamespace

import pytest
import torch

import fluxserve.backend.execution.runners as runners_pkg
from fluxserve.backend.distributed.launch import DistributedContext
from fluxserve.backend.engine.scheduler_adapter import DefaultSchedulerAdapter
from fluxserve.cli import bench_offline, launch
from test_nemotron_model import checkpoint_config as nemotron_config

RUNNER_NAMES = (
    "BlockDiffusionRunner",
    "DiffusionGemmaFlashInferRunner",
    "DiffusionGemmaRunner",
    "FA4DiffusionRunner",
    "FlashInferDiffusionRunner",
    "NemotronDiffusionRunner",
    "NemotronFA4DiffusionRunner",
    "NemotronFlashInferDiffusionRunner",
    "NemotronFlashInferSelfSpecRunner",
    "NemotronSelfSpecPagedRunner",
    "NemotronSelfSpecRunner",
)


def gemma_config():
    return SimpleNamespace(
        architectures=["DiffusionGemmaForBlockDiffusion"],
        model_type="diffusion_gemma",
        canvas_length=32,
    )


CONFIGS = {"gemma": gemma_config, "nemotron": nemotron_config}


class _Dispatched(Exception):
    """Stops ``bench_offline`` once the runner exists, before any generation."""


class _StubRunner:
    built: list

    def __init__(self, *, model_config, server_args, runner_config, device):
        self.model_config = model_config
        self.server_args = server_args
        self.runner_config = runner_config
        self.device = device
        self.block_length = runner_config.block_length
        self.decoder = SimpleNamespace(
            mask_id=runner_config.mask_id,
            eos_id=runner_config.eos_id,
            eos_ids=tuple(runner_config.eos_ids) or (runner_config.eos_id,),
        )
        self.draft_adapter_loaded = False
        self.built.append(self)

    def load_draft_adapter(self):
        self.draft_adapter_loaded = True


def _stub_runner_classes(built):
    """Stand-ins named like the real runners, with the same inheritance.

    The CLI uses ``isinstance`` on runner classes (e.g. to load the draft
    adapter), so a stub must subclass the stubs of its real exported bases.
    """
    base = type("_RecordingStubRunner", (_StubRunner,), {"built": built})
    stubs = {}

    def make(name):
        if name not in stubs:
            real = getattr(runners_pkg, name)
            bases = tuple(
                make(parent.__name__)
                for parent in real.__bases__
                if getattr(runners_pkg, parent.__name__, None) is parent
            )
            stubs[name] = type(name, bases or (base,), {})
        return stubs[name]

    for name in RUNNER_NAMES:
        make(name)
    return stubs


@pytest.fixture
def wiring(monkeypatch, tmp_path):
    """Stub everything below the dispatch decision and record what is built."""
    record = SimpleNamespace(
        runners=[], schedulers=[], served=[], profiled=[], model_config=None
    )
    stubs = _stub_runner_classes(record.runners)
    real_get_nemotron_runner = runners_pkg.get_nemotron_runner
    tokenizer = SimpleNamespace()

    def from_pretrained_config(*args, **kwargs):
        return record.model_config

    for module in (launch, bench_offline):
        for name in RUNNER_NAMES:
            if hasattr(module, name):
                monkeypatch.setattr(module, name, stubs[name])
        monkeypatch.setattr(
            module,
            "get_nemotron_runner",
            lambda backend, decoding: stubs[
                real_get_nemotron_runner(backend, decoding).__name__
            ],
        )
        monkeypatch.setattr(module, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer))
        monkeypatch.setattr(module, "AutoConfig", SimpleNamespace(from_pretrained=from_pretrained_config))
        monkeypatch.setattr(module, "reject_external_distributed_launch", lambda *a, **k: None)
        monkeypatch.setattr(module, "should_launch_local_workers", lambda *a, **k: False)
        monkeypatch.setattr(module, "require_nvidia_cuda", lambda *a, **k: None)
        monkeypatch.setattr(module, "set_process_title", lambda *a, **k: None)
        monkeypatch.setattr(module, "configure_logging", lambda *a, **k: None)
        monkeypatch.setattr(module, "initialize_distributed", lambda *a, **k: DistributedContext())
        monkeypatch.setattr(module, "destroy_distributed", lambda *a, **k: None)
        monkeypatch.setattr(module, "initialize_dp_attention", lambda *a, **k: None)
        monkeypatch.setattr(module, "initialize_moe_config", lambda *a, **k: None)
        monkeypatch.setattr(module, "resolve_checkpoint_eos_ids", lambda *a, **k: ())
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a, **k: None)

    # launch: capture the scheduler and the engine instead of serving.
    def paged_scheduler(**kwargs):
        record.schedulers.append(kwargs)
        return SimpleNamespace(kind="paged", **kwargs)

    def profile_paged_kv_pages(**kwargs):
        record.profiled.append(kwargs)
        return 96

    monkeypatch.setattr(launch, "PagedSchedulerAdapter", paged_scheduler)
    monkeypatch.setattr(launch, "profile_paged_kv_pages", profile_paged_kv_pages)
    monkeypatch.setattr(
        launch, "run", lambda engine, **kwargs: record.served.append((engine, kwargs))
    )

    # bench_offline: one short prompt, then stop right after dispatch.
    monkeypatch.setattr(
        bench_offline,
        "load_inputs",
        lambda *a, **k: ([torch.ones((1, 10), dtype=torch.long)], ["p"], ["q"], [0]),
    )

    def stop_at_warmup(*args, **kwargs):
        raise _Dispatched

    monkeypatch.setattr(bench_offline, "warmup_runner", stop_at_warmup)
    record.tmp_path = tmp_path
    return record


def _launch(wiring, model, *flags):
    wiring.model_config = CONFIGS[model]()
    args = launch.build_parser().parse_args(["launch", "--model", "m", *flags])
    launch.launch(args)
    assert len(wiring.runners) == 1
    return wiring.runners[0]


def _bench(wiring, model, *flags):
    wiring.model_config = CONFIGS[model]()
    args = launch.build_parser().parse_args(
        [
            "bench_offline", "--model", "m", "--dataset", "data.jsonl",
            "--output-dir", str(wiring.tmp_path), *flags,
        ]
    )
    with pytest.raises(_Dispatched):
        bench_offline.bench_offline(args)
    assert len(wiring.runners) == 1
    return wiring.runners[0]


SERVE_DISPATCH = [
    ("gemma", [], "DiffusionGemmaRunner"),
    ("gemma", ["--attention-backend", "flashinfer"], "DiffusionGemmaFlashInferRunner"),
    ("gemma", ["--use-decode-cuda-graph"], "DiffusionGemmaFlashInferRunner"),
    ("nemotron", [], "NemotronFlashInferDiffusionRunner"),
    ("nemotron", ["--attention-backend", "sdpa"], "NemotronDiffusionRunner"),
    ("nemotron", ["--attention-backend", "fa4"], "NemotronFA4DiffusionRunner"),
    (
        "nemotron",
        ["--attention-backend", "sdpa", "--parallel-decoding", "self_speculation"],
        "NemotronSelfSpecRunner",
    ),
    (
        "nemotron",
        ["--attention-backend", "fa4", "--parallel-decoding", "self_speculation",
         "--scheduler-policy", "paged"],
        "NemotronSelfSpecPagedRunner",
    ),
    (
        "nemotron",
        ["--parallel-decoding", "self_speculation"],
        "NemotronFlashInferSelfSpecRunner",
    ),
]


@pytest.mark.parametrize(
    "model,flags,expected", SERVE_DISPATCH, ids=[f"{m}-{' '.join(f) or 'default'}" for m, f, _ in SERVE_DISPATCH]
)
def test_launch_selects_runner(wiring, model, flags, expected):
    runner = _launch(wiring, model, *flags)
    assert type(runner).__name__ == expected
    assert runner.draft_adapter_loaded == ("SelfSpec" in expected)
    assert len(wiring.served) == 1


BENCH_DISPATCH = [
    ("gemma", [], "DiffusionGemmaRunner"),
    ("gemma", ["--attention-backend", "flashinfer"], "DiffusionGemmaFlashInferRunner"),
    ("nemotron", [], "NemotronFlashInferDiffusionRunner"),
    ("nemotron", ["--attention-backend", "sdpa"], "NemotronDiffusionRunner"),
    (
        "nemotron",
        ["--attention-backend", "sdpa", "--parallel-decoding", "self_speculation"],
        "NemotronSelfSpecRunner",
    ),
]


@pytest.mark.parametrize(
    "model,flags,expected", BENCH_DISPATCH, ids=[f"{m}-{' '.join(f) or 'default'}" for m, f, _ in BENCH_DISPATCH]
)
def test_bench_offline_selects_runner(wiring, model, flags, expected):
    runner = _bench(wiring, model, *flags)
    assert type(runner).__name__ == expected
    assert runner.draft_adapter_loaded == ("SelfSpec" in expected)


# --------------------------------------------------------------------------
# What surrounds the runner in `launch`
# --------------------------------------------------------------------------


def test_launch_default_scheduler_without_paged_policy(wiring):
    runner = _launch(wiring, "nemotron")
    engine, serve_kwargs = wiring.served[0]
    assert isinstance(engine.scheduler, DefaultSchedulerAdapter)
    assert serve_kwargs["runner_config"] is runner.runner_config
    assert wiring.schedulers == [] and wiring.profiled == []


def test_launch_paged_policy_profiles_kv_and_builds_paged_scheduler(wiring):
    runner = _launch(wiring, "nemotron", "--scheduler-policy", "paged", "--max-num-seqs", "4")
    engine, _ = wiring.served[0]
    assert [call["page_size"] for call in wiring.profiled] == [64]
    assert wiring.profiled[0]["runner"] is runner
    assert runner.server_args.scheduler_num_device_pages == 96
    assert wiring.schedulers == [
        {
            "max_batch_size": 4,
            "max_scheduled_tokens": 512,
            "page_size": 64,
            "num_device_pages": 96,
            "max_model_len": 2048,
            "full_prompt_prefill": True,
        }
    ]
    assert engine.scheduler.kind == "paged"


def test_launch_sdpa_falls_back_to_dense_kv(wiring):
    runner = _launch(wiring, "nemotron", "--attention-backend", "sdpa")
    assert runner.runner_config.kv_cache_layout == "dense"
    assert runner.runner_config.page_size is None


def test_launch_fa4_defaults_page_size_to_block_length(wiring):
    runner = _launch(wiring, "nemotron", "--attention-backend", "fa4", "--block-length", "32")
    assert runner.runner_config.kv_cache_layout == "paged"
    assert runner.runner_config.page_size == 32


def test_launch_gemma_uses_canvas_length_and_chat_template(wiring):
    runner = _launch(wiring, "gemma")
    assert runner.server_args.sampling_defaults is None
    assert runner.server_args.speculative_context_margin == 0
    assert runner.server_args.generation_block_size == 32
    assert runner.server_args.apply_template is True
    assert runner.runner_config.attention_backend == "sdpa"
    assert runner.runner_config.kv_cache_layout == "dense"


def test_launch_nemotron_sets_ids_sampling_defaults_and_template(wiring):
    runner = _launch(wiring, "nemotron", "--temperature", "0.7", "--seed", "3")
    config = runner.runner_config
    assert config.mask_id == 100
    assert config.temperature == 0.7
    assert config.seed == 3
    assert config.nemotron_prefill_chunk_size == 1024
    assert runner.server_args.sampling_defaults == {"temperature": 0.7, "seed": 3}
    assert runner.server_args.speculative_context_margin == 0
    assert runner.server_args.apply_template is True


def test_launch_nemotron_self_speculation_paged(wiring):
    runner = _launch(
        wiring, "nemotron",
        "--attention-backend", "fa4", "--parallel-decoding", "self_speculation",
        "--scheduler-policy", "paged",
    )
    assert runner.draft_adapter_loaded
    assert runner.server_args.speculative_context_margin == 64
    assert [s["full_prompt_prefill"] for s in wiring.schedulers] == [True]


@pytest.mark.parametrize(
    "model,flags,error,match",
    [
        ("gemma", ["--scheduler-policy", "paged", "--attention-backend", "flashinfer"],
         RuntimeError, "does not support scheduler_policy='paged'"),
        ("gemma", ["--attention-backend", "fa4"], ValueError, "supports LLaDA 2.x only"),
        ("gemma", ["--use-cuda-graph"], ValueError, "decode CUDA graphs only"),
        ("nemotron", ["--scheduler-policy", "paged", "--page-size", "32"],
         ValueError, "page_size == block_length"),
        ("nemotron", ["--attention-backend", "sdpa", "--use-decode-cuda-graph"],
         ValueError, "decode CUDA graphs exist only"),
    ],
)
def test_launch_rejects_unsupported_combinations(wiring, model, flags, error, match):
    with pytest.raises(error, match=match):
        _launch(wiring, model, *flags)
    assert wiring.runners == []


# --------------------------------------------------------------------------
# Where `bench_offline` deliberately differs from `launch`
# --------------------------------------------------------------------------


def test_bench_gemma_forces_block_length_to_canvas_length(wiring):
    runner = _bench(wiring, "gemma", "--block-length", "64")
    assert runner.runner_config.block_length == 32
    assert runner.runner_config.canvas_length == 32
    assert runner.runner_config.kv_cache_layout == "dense"


def test_launch_gemma_keeps_block_length(wiring):
    runner = _launch(wiring, "gemma", "--block-length", "64")
    assert runner.runner_config.block_length == 64
    assert runner.server_args.generation_block_size == 32

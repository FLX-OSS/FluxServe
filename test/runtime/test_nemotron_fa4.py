"""Paged FA4 plumbing and the Nemotron paged block loop.

The kernel itself needs a Hopper/Blackwell GPU, so what runs here is
everything around it: that per-call causality reaches the kernel arguments,
that the architecture gate is a hook rather than a hardcoded model name, that
the KV cache is shaped from the config's head_dim, and that rows in one batch
advance between denoising and committing independently.
"""

from types import SimpleNamespace

import pytest
import torch

from fluxserve.backend.execution.decoders.nemotron import (
    load_thinking_budget,
)

from fluxserve.backend.execution.forward_batch_info import ForwardBatch
from fluxserve.backend.execution.runners.fa4_diffusion import FA4DiffusionRunner
from fluxserve.backend.execution.runners.nemotron_diffusion import (
    NemotronBlockBudgetExceeded,
)
from fluxserve.backend.execution.runners.nemotron_fa4 import (
    COMMIT,
    DENOISE,
    DONE,
    NemotronFA4DiffusionRunner,
)
from fluxserve.backend.layers.attention.base import AttentionForwardConfig
from fluxserve.backend.layers.attention.fa4 import FA4PagedAttention
from fluxserve.backend.layers.attention.metadata import (
    build_block_diffusion_paged_metadata,
)

MASK_ID = 7
EOS_ID = 3
VOCAB = 16
BLOCK = 4


# --------------------------------------------------------------------------
# Per-call causality
# --------------------------------------------------------------------------


def paged_metadata(*, causal=False, phase="decode"):
    return build_block_diffusion_paged_metadata(
        phase=phase,
        q_offsets=torch.tensor([32]),
        q_lens=torch.tensor([16]),
        page_table=torch.tensor([[0, 1, 2, 3]]),
        max_input_len=16,
        block_length=16,
        page_size=16,
        causal=causal,
    )


def test_metadata_causality_defaults_to_the_existing_behaviour():
    without = build_block_diffusion_paged_metadata(
        phase="decode",
        q_offsets=torch.tensor([32]),
        q_lens=torch.tensor([16]),
        page_table=torch.tensor([[0, 1, 2, 3]]),
        max_input_len=16,
        block_length=16,
        page_size=16,
    )
    assert without.causal is False
    assert paged_metadata(causal=True).causal is True
    # Causality must not disturb the rest of the plan.
    explicit = paged_metadata(causal=True)
    assert explicit.kv_lens.tolist() == without.kv_lens.tolist()
    assert explicit.slot_mapping.tolist() == without.slot_mapping.tolist()
    assert explicit.qo_indptr.tolist() == without.qo_indptr.tolist()


@pytest.mark.parametrize("causal", [False, True])
def test_causality_reaches_the_kernel_arguments(causal):
    captured = {}

    def fake_kernel(q, k_cache, v_cache, **kwargs):
        captured.update(kwargs)
        return q

    config = AttentionForwardConfig(
        layer_id=0, num_heads=2, num_kv_heads=1, head_dim=8,
        num_key_value_groups=2, scale=1.0,
    )
    adapter = FA4PagedAttention(config, kernel=fake_kernel)
    metadata = paged_metadata(causal=causal)
    q = torch.zeros(1, 2, 16, 8, dtype=torch.bfloat16)
    k = torch.zeros(1, 1, 16, 8, dtype=torch.bfloat16)
    cache = (
        torch.zeros(4, 16, 1, 8, dtype=torch.bfloat16),
        torch.zeros(4, 16, 1, 8, dtype=torch.bfloat16),
    )
    batch = ForwardBatch(paged_attention_metadata=metadata)
    assert adapter.can_run(q, k, k, cache, None, batch)
    adapter.forward(q, k, k, cache, metadata)
    assert captured["causal"] is causal


# --------------------------------------------------------------------------
# Architecture gate
# --------------------------------------------------------------------------


def test_architecture_gate_is_a_hook_not_a_hardcoded_name():
    llada = SimpleNamespace(architectures=["LLaDA2MoeModelLM"], model_type="llada2_moe")
    nemotron = SimpleNamespace(
        architectures=["NemotronLabsDiffusionModel"],
        model_type="nemotron_labs_diffusion",
    )
    FA4DiffusionRunner._validate_architecture(llada)
    NemotronFA4DiffusionRunner._validate_architecture(nemotron)

    with pytest.raises(ValueError, match="LLaDA 2.x only"):
        FA4DiffusionRunner._validate_architecture(nemotron)
    with pytest.raises(ValueError, match="Nemotron-Labs-Diffusion only"):
        NemotronFA4DiffusionRunner._validate_architecture(llada)


# --------------------------------------------------------------------------
# Cache geometry
# --------------------------------------------------------------------------


def allocator_config(**overrides):
    values = dict(
        num_hidden_layers=2,
        num_key_value_heads=8,
        num_attention_heads=32,
        hidden_size=5120,
        head_dim=128,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_kv_cache_follows_the_config_head_dim_with_the_division_as_fallback(
    monkeypatch,
):
    from fluxserve.backend.execution.runners import block_diffusion
    from fluxserve.backend.execution.runners.block_diffusion import (
        BlockDiffusionRunner,
    )

    # The allocator asks for the attention TP size, which only exists inside a
    # served process; this test is about the head-dim lookup, not sharding.
    monkeypatch.setattr(block_diffusion, "get_attention_tp_size", lambda: 1)
    runner = object.__new__(BlockDiffusionRunner)
    runner.model = SimpleNamespace(model=SimpleNamespace(config=allocator_config()))
    runner.runner_config = SimpleNamespace(kv_cache_layout="dense", page_size=None)
    runner.max_length = 64
    runner.device = "cpu"
    cache = runner.allocate_kv_cache(1)
    # Nemotron: 128, not hidden_size // num_attention_heads == 160.
    assert cache.shape[-1] == 128

    runner.model = SimpleNamespace(
        model=SimpleNamespace(config=allocator_config(head_dim=None))
    )
    assert runner.allocate_kv_cache(1).shape[-1] == 160


# --------------------------------------------------------------------------
# Paged block loop
# --------------------------------------------------------------------------


def stub_config():
    return SimpleNamespace(
        num_hidden_layers=2, num_key_value_heads=2, head_dim=8,
        hidden_size=32, num_attention_heads=4, vocab_size=VOCAB,
    )


def logits_for(rows, batch=1, vocab=VOCAB):
    out = torch.zeros(batch, len(rows), vocab)
    for position, (token_id, probability) in enumerate(rows):
        others = (1.0 - probability) / (vocab - 1)
        out[:, position] = torch.log(torch.full((vocab,), others))
        out[:, position, token_id] = torch.log(torch.tensor(probability))
    return out


_paged_runner_cls = NemotronFA4DiffusionRunner


@pytest.fixture(autouse=True, params=["fa4", "flashinfer"])
def paged_backend(request, monkeypatch):
    from fluxserve.backend.execution.runners.nemotron_flashinfer import (
        NemotronFlashInferDiffusionRunner,
    )
    monkeypatch.setattr(
        __import__(__name__), "_paged_runner_cls",
        NemotronFA4DiffusionRunner if request.param == "fa4"
        else NemotronFlashInferDiffusionRunner,
    )


def make_paged_runner(script, *, steps=0, early_stop=True):
    runner = object.__new__(_paged_runner_cls)
    runner.model = SimpleNamespace(model=SimpleNamespace(config=stub_config()))
    runner.device = "cpu"
    runner.runner_config = SimpleNamespace(
        threshold=0.9, mask_id=MASK_ID, eos_id=EOS_ID, eos_ids=(EOS_ID,),
        block_length=BLOCK, steps=steps, gen_length=BLOCK,
    )
    runner.init_decoder()
    runner.block_length = BLOCK
    runner.max_denoise_steps = steps if steps > 0 else BLOCK
    runner.max_length = 256
    runner.num_forwards = 0
    runner.early_stop = early_stop
    runner.last_stats = []
    runner.thinking_budget = load_thinking_budget(runner.runner_config)
    runner.launches = []
    runner.allocate_kv_cache = lambda batch_size: None

    def paged_forward(*, seq_ids, tokens, q_offsets, causal, is_prefill,
                      max_input_len, q_lens=None, positions=None):
        runner.launches.append({
            "rows": seq_ids.tolist(),
            "causal": causal,
            "prefill": is_prefill,
            "tokens": tokens.tolist(),
            "offsets": q_offsets.tolist(),
            "q_lens": None if q_lens is None else q_lens.tolist(),
        })
        return script(len(runner.launches) - 1, seq_ids, tokens, max_input_len)

    runner._paged_forward = paged_forward
    return runner


def test_prefill_is_causal_and_seeds_position_zero():
    def script(index, seq_ids, tokens, length):
        if index == 0:
            return logits_for([(9, 0.99), (9, 0.99), (1, 0.99)],
                              batch=len(seq_ids))
        return logits_for([(2, 0.99), (4, 0.99), (5, 0.99), (6, 0.99)],
                          batch=len(seq_ids))

    runner = make_paged_runner(script)
    runner.generate(torch.tensor([[9, 9, 9]]), prompt_lengths=[3],
                    generation_lengths=[BLOCK])

    prefill = runner.launches[0]
    assert prefill["prefill"] is True and prefill["causal"] is True
    denoise = runner.launches[1]
    assert denoise["causal"] is False
    assert denoise["tokens"][0][0] == 1, "seed from the prefill's last logit"
    assert denoise["tokens"][0][1:] == [MASK_ID] * (BLOCK - 1)
    assert denoise["offsets"] == [3], "block starts at the committed prefix"


def test_commit_launch_is_causal_and_follows_the_denoise_launches():
    def script(index, seq_ids, tokens, length):
        if index == 0:
            return logits_for([(9, 0.99), (1, 0.99)], batch=len(seq_ids))
        return logits_for([(2, 0.99), (4, 0.99), (5, 0.99), (6, 0.99)],
                          batch=len(seq_ids))

    runner = make_paged_runner(script)
    runner.generate(torch.tensor([[9, 9]]), prompt_lengths=[2],
                    generation_lengths=[BLOCK])

    assert [(item["causal"], item["prefill"]) for item in runner.launches] == [
        (True, True),    # prefill
        (False, False),  # denoise
        (True, False),   # commit
    ]
    stats = runner.last_stats[0]
    assert (stats["denoise_calls"], stats["commit_calls"]) == (1, 1)


def test_rows_advance_independently_between_denoise_and_commit():
    """A row that resolves first must not pay for its neighbour's steps."""

    def script(index, seq_ids, tokens, length):
        if index == 0:
            return logits_for([(9, 0.99), (1, 0.99)], batch=len(seq_ids))
        out = torch.empty(len(seq_ids), BLOCK, VOCAB)
        for position, row in enumerate(seq_ids.tolist()):
            if row == 0:
                # Row 0 resolves the whole block in one confident step.
                out[position] = logits_for(
                    [(2, 0.99), (4, 0.99), (5, 0.99), (6, 0.99)]
                )[0]
            else:
                # Row 1 is below the threshold, so one position per step.
                out[position] = logits_for(
                    [(2, 0.4), (4, 0.3), (5, 0.2), (6, 0.1)]
                )[0]
        return out

    runner = make_paged_runner(script)
    prompts = torch.tensor([[9, 9], [9, 9]])
    runner.generate(prompts, prompt_lengths=[2, 2], generation_lengths=[BLOCK, BLOCK])

    phases = [
        ("commit" if item["causal"] else "denoise", tuple(item["rows"]))
        for item in runner.launches[1:]
    ]
    assert phases == [
        ("denoise", (0, 1)),  # both rows start together
        ("denoise", (1,)),    # row 0 is already done denoising
        ("commit", (0,)),
        ("denoise", (1,)),
        ("commit", (1,)),
    ]
    assert runner.last_stats[0]["denoise_calls"] == 1
    assert runner.last_stats[1]["denoise_calls"] == 3
    assert [stats["commit_calls"] for stats in runner.last_stats] == [1, 1]


def test_distinct_prompt_lengths_get_one_prefill_launch_each():
    def script(index, seq_ids, tokens, length):
        if index < 2:
            return logits_for([(9, 0.99)] * (length - 1) + [(1, 0.99)],
                              batch=len(seq_ids))
        return logits_for([(2, 0.99), (4, 0.99), (5, 0.99), (6, 0.99)],
                          batch=len(seq_ids))

    runner = make_paged_runner(script)
    prompts = torch.tensor([[9, 9, 9], [9, 9, MASK_ID]])
    runner.generate(prompts, prompt_lengths=[3, 2], generation_lengths=[BLOCK, BLOCK])

    prefills = [item for item in runner.launches if item["prefill"]]
    assert [item["rows"] for item in prefills] == [[1], [0]], "grouped by length"
    assert all(item["causal"] for item in prefills)
    # Blocks then start at each row's own prefix.
    first_denoise = next(item for item in runner.launches if not item["prefill"])
    assert sorted(first_denoise["offsets"]) == [2, 3]


def test_eos_seed_commits_without_a_denoise_launch():
    def script(index, seq_ids, tokens, length):
        if index == 0:
            return logits_for([(9, 0.99), (EOS_ID, 0.99)], batch=len(seq_ids))
        return logits_for([(2, 0.99)] * BLOCK, batch=len(seq_ids))

    runner = make_paged_runner(script)
    runner.generate(torch.tensor([[9, 9]]), prompt_lengths=[2],
                    generation_lengths=[2 * BLOCK])

    assert [item["causal"] for item in runner.launches] == [True, True]
    stats = runner.last_stats[0]
    assert stats["denoise_calls"] == 0
    assert stats["commit_calls"] == 1
    assert stats["denoise_per_block"] == [0]


def test_eager_launch_when_no_graph_is_captured():
    runner = make_paged_runner(lambda *a: logits_for([(1, 0.99)] * BLOCK))
    runner.nemotron_graph_runner = None
    assert runner._graph_replay(
        seq_ids=torch.tensor([0]),
        tokens=torch.zeros(1, BLOCK, dtype=torch.long),
        positions=torch.zeros(1, BLOCK, dtype=torch.long),
        causal=False,
    ) is None


def test_row_states_are_named_constants():
    assert (DENOISE, COMMIT, DONE) == ("denoise", "commit", "done")


def test_paged_block_budget_is_enforced_per_row():
    def script(index, seq_ids, tokens, length):
        if index == 0:
            return logits_for([(9, 0.99), (1, 0.99)], batch=len(seq_ids))
        # Always predicts the mask token, so the block never resolves.
        return logits_for([(MASK_ID, 0.99)] * BLOCK, batch=len(seq_ids))

    runner = make_paged_runner(script, steps=2)
    with pytest.raises(NemotronBlockBudgetExceeded, match="request row 0"):
        runner.generate(torch.tensor([[9, 9]]), prompt_lengths=[2],
                        generation_lengths=[BLOCK])


# --------------------------------------------------------------------------
# Nemotron paged metadata: one task per request, arbitrary offsets
# --------------------------------------------------------------------------


def test_llada_builder_rejects_the_offsets_nemotron_actually_uses():
    """Why a local builder exists, stated as a test rather than a comment."""
    # A 34-token prompt is not a whole number of 32-token blocks, and the
    # first generated block therefore starts at 34.
    with pytest.raises(ValueError, match="block-aligned query lengths"):
        build_block_diffusion_paged_metadata(
            phase="prefill",
            q_offsets=torch.tensor([0]),
            q_lens=torch.tensor([34]),
            page_table=torch.tensor([[0, 1, 2]]),
            max_input_len=34,
            block_length=32,
            page_size=32,
        )
    with pytest.raises(ValueError, match="block-aligned query offsets"):
        build_block_diffusion_paged_metadata(
            phase="decode",
            q_offsets=torch.tensor([34]),
            q_lens=torch.tensor([32]),
            page_table=torch.tensor([[0, 1, 2]]),
            max_input_len=32,
            block_length=32,
            page_size=32,
        )


def test_unaligned_causal_prefill_maps_one_task_across_pages():
    from fluxserve.backend.execution.runners.nemotron_fa4 import (
        build_nemotron_paged_metadata,
    )

    metadata = build_nemotron_paged_metadata(
        phase="prefill",
        q_offsets=torch.tensor([0]),
        q_lens=torch.tensor([34]),
        page_table=torch.tensor([[5, 6, 7]]),
        max_input_len=34,
        block_length=32,
        page_size=32,
        causal=True,
    )
    assert metadata.causal is True
    assert metadata.q_lens_cpu == (34,)
    assert metadata.kv_lens.tolist() == [34]
    assert metadata.qo_indptr.tolist() == [0, 34], "one task, not two blocks"
    assert metadata.max_q_len == 34
    # Positions 0..33 land in physical pages 5 then 6.
    expected = [5 * 32 + offset for offset in range(32)] + [6 * 32, 6 * 32 + 1]
    assert metadata.slot_mapping.tolist() == expected


def test_block_at_an_unaligned_offset_spans_two_pages():
    from fluxserve.backend.execution.runners.nemotron_fa4 import (
        build_nemotron_paged_metadata,
    )

    metadata = build_nemotron_paged_metadata(
        phase="decode",
        q_offsets=torch.tensor([34]),
        q_lens=torch.tensor([32]),
        page_table=torch.tensor([[5, 6, 7]]),
        max_input_len=32,
        block_length=32,
        page_size=32,
        causal=False,
    )
    assert metadata.causal is False
    assert metadata.kv_lens.tolist() == [66], "prefix plus the whole block"
    expected = [6 * 32 + offset for offset in range(2, 32)] + [7 * 32, 7 * 32 + 1]
    assert metadata.slot_mapping.tolist() == expected
    assert metadata.q_token_indices.tolist() == list(range(32))


def test_mixed_length_prefill_packs_each_request_once():
    from fluxserve.backend.execution.runners.nemotron_fa4 import (
        build_nemotron_paged_metadata,
    )

    metadata = build_nemotron_paged_metadata(
        phase="prefill",
        q_offsets=torch.tensor([0, 0]),
        q_lens=torch.tensor([34, 30]),
        page_table=torch.tensor([[0, 1, 2], [3, 4, 5]]),
        max_input_len=34,
        block_length=32,
        page_size=32,
        causal=True,
    )
    assert metadata.qo_indptr.tolist() == [0, 34, 64]
    assert metadata.kv_lens.tolist() == [34, 30]
    # The shorter request is padded in the model input, so the packed gather is
    # not the identity and must skip that padding.
    assert metadata.is_identity_mapping is False
    assert metadata.q_token_indices.tolist() == list(range(34)) + list(
        range(34, 64)
    )


def test_metadata_rejects_a_page_table_that_cannot_cover_the_request():
    from fluxserve.backend.execution.runners.nemotron_fa4 import (
        build_nemotron_paged_metadata,
    )

    with pytest.raises(ValueError, match="needs 3 pages"):
        build_nemotron_paged_metadata(
            phase="decode",
            q_offsets=torch.tensor([34]),
            q_lens=torch.tensor([32]),
            page_table=torch.tensor([[5, 6]]),
            max_input_len=32,
            block_length=32,
            page_size=32,
            causal=False,
        )


# --------------------------------------------------------------------------
# Continuously scheduled paged serving
# --------------------------------------------------------------------------


class FakePlanOp:
    def __init__(self, request_ids, input_lengths, extend_prefix_lens, input_ids,
                 occupied_pages, num_extends):
        self.request_ids = request_ids
        self.input_lengths = input_lengths
        self.extend_prefix_lens = extend_prefix_lens
        self.input_ids = input_ids
        self.occupied_pages = occupied_pages
        self._num_extends = num_extends

    def num_extends(self):
        return self._num_extends


class FakeState:
    def __init__(self, rid, input_ids, max_new_tokens=BLOCK, ignore_eos=False):
        self.rid = rid
        self.input_ids = list(input_ids)
        self.output_ids = []
        self.max_new_tokens = max_new_tokens
        self.ignore_eos = ignore_eos
        self.finished_reason = None
        self.plan_prefill_done = False
        self.current_decode_block = 0

    @property
    def finished(self):
        return self.finished_reason is not None


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(value) for value in ids)


class FakePagedCache:
    page_size = 16

    def __init__(self):
        self.page_tables = None

    def set_page_tables(self, slots, pages):
        self.page_tables = (list(slots), [list(item) for item in pages])


def make_plan_runner(script, *, max_num_seqs=4):
    runner = make_paged_runner(script)
    runner.server_args = SimpleNamespace(
        scheduler_num_device_pages=64, max_num_seqs=max_num_seqs
    )
    runner._paged_request_slots = {}
    runner._request_seeds = {}
    runner.past_key_values = FakePagedCache()
    runner.ensure_paged_kv_cache = lambda *, num_device_pages: None
    return runner


def run_plan(runner, op, states):
    import asyncio

    return asyncio.run(
        runner.execute_paged_forward_plan(op, states, FakeTokenizer())
    )


def test_plan_prefill_is_causal_and_stores_the_seed():
    def script(index, seq_ids, tokens, length):
        return logits_for([(9, 0.99)] * (length - 1) + [(1, 0.99)],
                          batch=len(seq_ids))

    runner = make_plan_runner(script)
    states = {"r0": FakeState("r0", [5] * 34)}
    op = FakePlanOp(["r0"], [34], [0], [5] * 34, [[0, 1, 2]], 1)
    results = run_plan(runner, op, states)

    assert [item.rid for item in results] == ["r0"]
    assert results[0].token_ids == []
    assert states["r0"].plan_prefill_done is True
    assert runner._request_seeds["r0"] == 1
    assert runner.launches[0]["causal"] is True
    assert runner.launches[0]["prefill"] is True


def test_an_intermediate_prefill_chunk_yields_no_seed():
    def script(index, seq_ids, tokens, length):
        return logits_for([(9, 0.99)] * (length - 1) + [(1, 0.99)],
                          batch=len(seq_ids))

    runner = make_plan_runner(script)
    states = {"r0": FakeState("r0", [5] * 70)}
    # Only the first 32 tokens of a 70-token prompt.
    op = FakePlanOp(["r0"], [32], [0], [5] * 32, [[0, 1, 2]], 1)
    run_plan(runner, op, states)
    assert "r0" not in runner._request_seeds

    op2 = FakePlanOp(["r0"], [38], [32], [5] * 38, [[0, 1, 2]], 1)
    run_plan(runner, op2, states)
    assert runner._request_seeds["r0"] == 1


def test_one_plan_call_advances_each_request_by_exactly_one_block():
    def script(index, seq_ids, tokens, length):
        return logits_for([(2, 0.99), (4, 0.99), (5, 0.99), (6, 0.99)],
                          batch=len(seq_ids))

    runner = make_plan_runner(script)
    states = {"r0": FakeState("r0", [5] * 34, max_new_tokens=3 * BLOCK)}
    states["r0"].plan_prefill_done = True
    runner._paged_request_slots["r0"] = 0
    runner._request_seeds["r0"] = 1
    op = FakePlanOp(["r0"], [], [], [], [[0, 1, 2, 3]], 0)
    results = run_plan(runner, op, states)

    assert len(results) == 1
    result = results[0]
    assert result.decode_block_completed is True
    assert result.reserve_tokens == BLOCK
    assert result.finished is False
    assert result.token_ids == [1, 2, 4, 5][:BLOCK] or len(result.token_ids) == BLOCK
    # One denoise launch and one commit launch, nothing more.
    assert [item["causal"] for item in runner.launches] == [False, True]


def test_decode_without_a_seed_fails_instead_of_guessing():
    runner = make_plan_runner(lambda *a: logits_for([(1, 0.99)] * BLOCK))
    states = {"r0": FakeState("r0", [5] * 34)}
    runner._paged_request_slots["r0"] = 0
    op = FakePlanOp(["r0"], [], [], [], [[0, 1]], 0)
    with pytest.raises(RuntimeError, match="without a seed"):
        run_plan(runner, op, states)


def test_finished_requests_release_their_slot_and_seed():
    def script(index, seq_ids, tokens, length):
        return logits_for([(EOS_ID, 0.99)] * BLOCK, batch=len(seq_ids))

    runner = make_plan_runner(script)
    states = {"r0": FakeState("r0", [5] * 34, max_new_tokens=3 * BLOCK)}
    runner._paged_request_slots["r0"] = 0
    runner._request_seeds["r0"] = 1
    op = FakePlanOp(["r0"], [], [], [], [[0, 1, 2, 3]], 0)
    results = run_plan(runner, op, states)

    assert results[0].finished is True
    assert results[0].finish_reason == "stop"
    assert "r0" not in runner._paged_request_slots
    assert "r0" not in runner._request_seeds, "page reuse must not inherit a seed"


def test_a_failing_plan_releases_every_slot_it_claimed():
    def script(index, seq_ids, tokens, length):
        raise RuntimeError("kernel exploded")

    runner = make_plan_runner(script)
    states = {"r0": FakeState("r0", [5] * 34)}
    op = FakePlanOp(["r0"], [34], [0], [5] * 34, [[0, 1, 2]], 1)
    with pytest.raises(RuntimeError, match="kernel exploded"):
        run_plan(runner, op, states)
    assert runner._paged_request_slots == {}


def test_scheduler_prefills_the_whole_prompt_only_when_asked():
    from fluxserve.backend.engine.scheduler_adapter import PagedSchedulerAdapter

    state = SimpleNamespace(
        rid="r0", input_ids=[1] * 70,
        aligned_prefill_length=lambda page: (70 // page) * page,
    )
    captured = []

    class Spec:
        pass

    adapter = object.__new__(PagedSchedulerAdapter)
    adapter.page_size = 32
    adapter.max_model_len = 4096
    adapter._active = set()
    adapter._request_spec_cls = Spec
    adapter._scheduler = SimpleNamespace(
        submit_requests=lambda specs: captured.extend(
            spec.prefill_length for spec in specs
        )
    )

    adapter.full_prompt_prefill = False
    adapter.submit([state])
    assert captured == [64], "LLaDA keeps the block-aligned floor"

    adapter._active.clear()
    adapter.full_prompt_prefill = True
    adapter.submit([state])
    assert captured == [64, 70]


# --------------------------------------------------------------------------
# Decode graph bookkeeping (capture itself needs a GPU)
# --------------------------------------------------------------------------


def graph_runner(buckets=(1, 2, 4)):
    from fluxserve.backend.execution.nemotron_cuda_graph_runner import (
        NemotronCudaGraphRunner,
    )

    return NemotronCudaGraphRunner(buckets)


def test_buckets_round_up_and_reject_oversize_batches():
    runner = graph_runner()
    assert runner.bucket_for(1) == 1
    assert runner.bucket_for(2) == 2
    assert runner.bucket_for(3) == 4
    assert runner.bucket_for(4) == 4
    with pytest.raises(ValueError, match="no Nemotron decode graph bucket"):
        runner.bucket_for(5)
    with pytest.raises(ValueError):
        graph_runner(buckets=(0,))


def test_can_replay_requires_a_captured_variant_for_that_causality():
    runner = graph_runner()
    fake = SimpleNamespace(past_key_values=object(), block_length=BLOCK)
    # Nothing captured yet.
    assert not runner.can_replay(fake, batch_size=1, length=BLOCK, causal=False)

    runner.cache = fake.past_key_values
    runner.entries[(1, False)] = object()
    assert runner.can_replay(fake, batch_size=1, length=BLOCK, causal=False)
    # The commit variant is a separate capture: the kernel specialises on the
    # flag, so a denoise graph cannot serve a commit.
    assert not runner.can_replay(fake, batch_size=1, length=BLOCK, causal=True)
    # Wrong token shape, oversize batch, or a different KV allocation.
    assert not runner.can_replay(fake, batch_size=1, length=BLOCK + 1, causal=False)
    assert not runner.can_replay(fake, batch_size=99, length=BLOCK, causal=False)
    assert not runner.can_replay(
        SimpleNamespace(past_key_values=object(), block_length=BLOCK),
        batch_size=1, length=BLOCK, causal=False,
    )


def test_stats_separate_the_two_variants():
    runner = graph_runner()
    runner.entries[(1, False)] = object()
    runner.entries[(1, True)] = object()
    runner.entries[(2, False)] = object()
    stats = runner.stats()
    assert stats["denoise_capture_count"] == 2
    assert stats["commit_capture_count"] == 1
    assert stats["decode_capture_count"] == 3

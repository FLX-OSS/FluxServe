"""Paged, batched self-speculation: rollback and per-row divergence.

What separates this from the dense runner is that rows accept different numbers
of tokens per iteration, so their committed prefixes drift apart immediately. A
block-synchronous loop would be wrong, and the launch partition is the thing
worth pinning down.
"""

from types import SimpleNamespace

import pytest
import torch

from fluxserve.backend.execution.runners.nemotron_selfspec_paged import (
    DONE,
    DRAFT,
    VERIFY,
    NemotronSelfSpecPagedRunner,
    SpecRow,
)
from test_nemotron_diffusion import BLOCK, EOS_ID, MASK_ID, VOCAB, stub_config
from test_nemotron_fa4 import logits_for


_paged_runner_cls = NemotronSelfSpecPagedRunner


@pytest.fixture(autouse=True, params=["fa4", "flashinfer"])
def paged_backend(request, monkeypatch):
    from fluxserve.backend.execution.runners.nemotron_flashinfer import (
        NemotronFlashInferSelfSpecRunner,
    )
    monkeypatch.setattr(
        __import__(__name__), "_paged_runner_cls",
        NemotronSelfSpecPagedRunner if request.param == "fa4"
        else NemotronFlashInferSelfSpecRunner,
    )


def make_runner(script, *, draft_threshold=0.0):
    from fluxserve.backend.execution.decoders.nemotron import (
    load_thinking_budget,
        NemotronThresholdDecoder,
    )

    runner = object.__new__(_paged_runner_cls)
    runner.model = SimpleNamespace(model=SimpleNamespace(config=stub_config()))
    runner.device = "cpu"
    runner.runner_config = SimpleNamespace(
        threshold=0.9, mask_id=MASK_ID, eos_id=EOS_ID, eos_ids=(EOS_ID,),
        block_length=BLOCK, steps=0, gen_length=BLOCK,
    )
    runner.init_decoder()
    runner.block_length = BLOCK
    runner.max_denoise_steps = BLOCK
    runner.max_length = 256
    runner.num_forwards = 0
    runner.early_stop = True
    runner.draft_threshold = draft_threshold
    runner.draft_decoder = NemotronThresholdDecoder(
        threshold=draft_threshold, mask_id=MASK_ID, eos_ids=(EOS_ID,), draft=True,
    )
    runner.lora = None
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
            "offsets": q_offsets.tolist(),
            "tokens": tokens.tolist(),
        })
        return script(len(runner.launches) - 1, seq_ids, tokens, max_input_len)

    runner._paged_forward = paged_forward
    return runner


def phases(runner):
    return [
        ("prefill" if item["prefill"] else ("verify" if item["causal"] else "draft"),
         tuple(item["rows"]))
        for item in runner.launches
    ]


def test_one_iteration_is_prefill_then_draft_then_verify():
    def script(index, seq_ids, tokens, length):
        if index == 0:
            return logits_for([(9, 0.99)] * (length - 1) + [(1, 0.99)],
                              batch=len(seq_ids))
        if index == 1:
            return logits_for([(9, 0.9), (10, 0.9), (11, 0.9), (12, 0.9)],
                              batch=len(seq_ids))
        return logits_for([(10, 0.9), (11, 0.9), (12, 0.9), (13, 0.9)],
                          batch=len(seq_ids))

    runner = make_runner(script)
    output = runner.generate(torch.tensor([[9, 9]]), prompt_lengths=[2],
                             generation_lengths=[BLOCK])
    assert phases(runner) == [("prefill", (0,)), ("draft", (0,)), ("verify", (0,))]
    stats = runner.last_stats[0]
    assert (stats["draft_calls"], stats["verify_calls"]) == (1, 1)
    assert stats["accepted_per_iteration"] == [BLOCK]
    assert output[0, 2:].tolist()[:BLOCK] == [1, 10, 11, 12]


def test_the_prefix_advances_by_the_accepted_length_not_the_block():
    """Rollback on the paged path, observed through the next q_offset."""
    def script(index, seq_ids, tokens, length):
        if index == 0:
            return logits_for([(9, 0.99), (1, 0.99)], batch=len(seq_ids))
        if index % 2 == 1:
            return logits_for([(9, 0.9), (10, 0.9), (14, 0.9), (14, 0.9)],
                              batch=len(seq_ids))
        return logits_for([(10, 0.9), (13, 0.9), (13, 0.9), (13, 0.9)],
                          batch=len(seq_ids))

    runner = make_runner(script)
    runner.generate(torch.tensor([[9, 9]]), prompt_lengths=[2],
                    generation_lengths=[3 * BLOCK])

    drafts = [item for item in runner.launches if not item["prefill"]
              and not item["causal"]]
    # Two tokens accepted per iteration, so offsets step by 2, not by 4.
    assert [item["offsets"][0] for item in drafts[:3]] == [2, 4, 6]
    assert runner.last_stats[0]["accepted_per_iteration"][:3] == [2, 2, 2]


def test_rows_with_different_acceptance_lengths_advance_independently():
    def script(index, seq_ids, tokens, length):
        if index == 0:
            return logits_for([(9, 0.99), (1, 0.99)], batch=len(seq_ids))
        out = torch.empty(len(seq_ids), BLOCK, VOCAB)
        for position, row in enumerate(seq_ids.tolist()):
            if index % 2 == 1:  # draft
                out[position] = logits_for(
                    [(9, 0.9), (10, 0.9), (11, 0.9), (12, 0.9)]
                    if row == 0
                    else [(9, 0.9), (10, 0.9), (14, 0.9), (14, 0.9)]
                )[0]
            else:  # verify
                out[position] = logits_for(
                    [(10, 0.9), (11, 0.9), (12, 0.9), (13, 0.9)]
                    if row == 0
                    else [(10, 0.9), (13, 0.9), (13, 0.9), (13, 0.9)]
                )[0]
        return out

    runner = make_runner(script)
    prompts = torch.tensor([[9, 9], [9, 9]])
    runner.generate(prompts, prompt_lengths=[2, 2],
                    generation_lengths=[BLOCK, BLOCK])

    # Row 0 accepts the whole block and finishes in one iteration; row 1
    # accepts two and needs more, so later launches carry only row 1.
    accepted = [stats["accepted_per_iteration"] for stats in runner.last_stats]
    assert accepted[0] == [BLOCK]
    assert accepted[1][0] == 2 and len(accepted[1]) > 1
    later = [item["rows"] for item in runner.launches if not item["prefill"]][2:]
    assert all(rows == [1] for rows in later), (
        "a finished row must not be relaunched alongside an unfinished one"
    )


def test_prefixes_that_have_drifted_apart_share_one_launch():
    """One task per request is what lets a single launch carry both rows."""
    runner = make_runner(lambda *a: logits_for([(1, 0.9)] * BLOCK, batch=2))
    rows = [
        SpecRow(index=0, seq_id=0, prompt_length=2, generation_length=BLOCK,
                prefix_length=5,
                block=torch.zeros(BLOCK, dtype=torch.long), seed=1),
        SpecRow(index=1, seq_id=1, prompt_length=2, generation_length=BLOCK,
                prefix_length=11,
                block=torch.zeros(BLOCK, dtype=torch.long), seed=1),
    ]
    runner._spec_launch(rows, causal=False)
    assert runner.launches[0]["offsets"] == [5, 11]
    assert runner.launches[0]["rows"] == [0, 1]


def test_eos_in_the_accepted_tokens_finishes_that_row_only():
    def script(index, seq_ids, tokens, length):
        if index == 0:
            return logits_for([(9, 0.99), (1, 0.99)], batch=len(seq_ids))
        out = torch.empty(len(seq_ids), BLOCK, VOCAB)
        for position, row in enumerate(seq_ids.tolist()):
            if index % 2 == 1:
                out[position] = logits_for(
                    [(9, 0.9), (10, 0.9), (11, 0.9), (12, 0.9)]
                )[0]
            else:
                out[position] = logits_for(
                    [(10, 0.9), (EOS_ID, 0.9), (12, 0.9), (13, 0.9)]
                    if row == 0
                    else [(10, 0.9), (11, 0.9), (12, 0.9), (13, 0.9)]
                )[0]
        return out

    runner = make_runner(script)
    output = runner.generate(torch.tensor([[9, 9], [9, 9]]),
                             prompt_lengths=[2, 2],
                             generation_lengths=[2 * BLOCK, 2 * BLOCK])
    first = [t for t in output[0, 2:].tolist() if t != MASK_ID]
    assert first == [1, 10, EOS_ID]
    assert runner.last_stats[0]["iterations"] == 1
    assert runner.last_stats[1]["iterations"] >= 1


def test_decode_graphs_are_refused_with_a_reason():
    """A draft and a verify launch share a shape but differ in causality, and
    the accepted length varies, so a captured graph needs its own design."""
    config = SimpleNamespace(
        attention_backend="fa4", kv_cache_layout="paged",
        enable_prefill_cuda_graph=False, enable_decode_cuda_graph=True,
        page_size=16, block_length=16,
    )
    with pytest.raises(ValueError, match="eager"):
        NemotronSelfSpecPagedRunner(
            SimpleNamespace(), SimpleNamespace(), config, "cuda"
        )


def test_normalization_routes_self_speculation_by_backend():
    from fluxserve.backend.model_loader.nemotron import normalize_nemotron_args
    from test_nemotron_model import checkpoint_config, serve_args

    dense = serve_args(parallel_decoding="self_speculation")
    assert normalize_nemotron_args(dense, checkpoint_config()) is True
    assert dense.kv_cache_layout == "dense"

    paged = serve_args(parallel_decoding="self_speculation",
                       attention_backend="fa4", kv_cache_layout="paged")
    assert normalize_nemotron_args(paged, checkpoint_config()) is True

    # Continuous scheduling is supported on this path; see the plan tests below.
    scheduled = serve_args(parallel_decoding="self_speculation",
                           attention_backend="fa4", kv_cache_layout="paged",
                           scheduler_policy="paged")
    assert normalize_nemotron_args(scheduled, checkpoint_config()) is True


def test_row_states_are_named_constants():
    assert (DRAFT, VERIFY, DONE) == ("draft", "verify", "done")


# --------------------------------------------------------------------------
# Continuously scheduled paged self-speculation
# --------------------------------------------------------------------------


def make_plan_runner(script):
    from test_nemotron_fa4 import FakePagedCache

    runner = make_runner(script)
    runner.server_args = SimpleNamespace(
        scheduler_num_device_pages=64, max_num_seqs=4
    )
    runner._paged_request_slots = {}
    runner._request_seeds = {}
    runner._request_prefix = {}
    runner.past_key_values = FakePagedCache()
    runner.ensure_paged_kv_cache = lambda *, num_device_pages: None
    return runner


def run_plan(runner, op, states):
    import asyncio

    from test_nemotron_fa4 import FakeTokenizer

    return asyncio.run(
        runner.execute_paged_forward_plan(op, states, FakeTokenizer())
    )


def test_prefill_records_both_the_seed_and_the_prefix():
    from test_nemotron_fa4 import FakePlanOp, FakeState

    def script(index, seq_ids, tokens, length):
        return logits_for([(9, 0.99)] * (length - 1) + [(1, 0.99)],
                          batch=len(seq_ids))

    runner = make_plan_runner(script)
    states = {"r0": FakeState("r0", [5] * 34)}
    op = FakePlanOp(["r0"], [34], [0], [5] * 34, [[0, 1, 2]], 1)
    run_plan(runner, op, states)

    assert runner._request_seeds["r0"] == 1
    assert runner._request_prefix["r0"] == 34, (
        "the prefix cannot come from a block counter: an iteration advances it "
        "by the accepted length"
    )


def test_one_plan_call_is_one_speculation_iteration():
    from test_nemotron_fa4 import FakePlanOp, FakeState

    def script(index, seq_ids, tokens, length):
        if index % 2 == 0:  # draft
            return logits_for([(9, 0.9), (10, 0.9), (11, 0.9), (12, 0.9)],
                              batch=len(seq_ids))
        return logits_for([(10, 0.9), (11, 0.9), (12, 0.9), (13, 0.9)],
                          batch=len(seq_ids))

    runner = make_plan_runner(script)
    states = {"r0": FakeState("r0", [5] * 34, max_new_tokens=3 * BLOCK)}
    runner._paged_request_slots["r0"] = 0
    runner._request_seeds["r0"] = 1
    runner._request_prefix["r0"] = 34
    op = FakePlanOp(["r0"], [], [], [], [[0, 1, 2, 3]], 0)
    results = run_plan(runner, op, states)

    # Exactly one draft launch and one verify launch, then the call returns.
    assert [item["causal"] for item in runner.launches] == [False, True]
    assert len(results) == 1
    result = results[0]
    assert result.reserve_tokens == BLOCK, "one speculative block for next time"
    assert result.decode_block_completed is False, (
        "a speculation iteration is not a fixed-stride block"
    )
    # The whole block was accepted, so the prefix advanced by four.
    assert runner._request_prefix["r0"] == 34 + BLOCK
    assert result.token_ids == [1, 10, 11, 12, 13]


def test_the_prefix_advances_by_the_accepted_length_across_plan_calls():
    """The reason this runner keeps its own prefix rather than a block counter."""
    from test_nemotron_fa4 import FakePlanOp, FakeState

    def script(index, seq_ids, tokens, length):
        if index % 2 == 0:
            return logits_for([(9, 0.9), (10, 0.9), (14, 0.9), (14, 0.9)],
                              batch=len(seq_ids))
        return logits_for([(10, 0.9), (13, 0.9), (13, 0.9), (13, 0.9)],
                          batch=len(seq_ids))

    runner = make_plan_runner(script)
    states = {"r0": FakeState("r0", [5] * 34, max_new_tokens=4 * BLOCK)}
    runner._paged_request_slots["r0"] = 0
    runner._request_seeds["r0"] = 1
    runner._request_prefix["r0"] = 34

    prefixes = []
    for _ in range(3):
        op = FakePlanOp(["r0"], [], [], [], [[0, 1, 2, 3, 4, 5]], 0)
        results = run_plan(runner, op, states)
        states["r0"].output_ids.extend(results[0].token_ids)
        prefixes.append(runner._request_prefix["r0"])

    # Two tokens accepted per iteration, so the prefix steps by two, not four.
    assert prefixes == [36, 38, 40]
    offsets = [item["offsets"][0] for item in runner.launches if not item["causal"]]
    assert offsets == [34, 36, 38]


def test_speculating_without_a_recorded_prefix_fails_loudly():
    from test_nemotron_fa4 import FakePlanOp, FakeState

    runner = make_plan_runner(lambda *a: logits_for([(1, 0.9)] * BLOCK))
    states = {"r0": FakeState("r0", [5] * 34)}
    runner._paged_request_slots["r0"] = 0
    runner._request_seeds["r0"] = 1  # seed but no prefix
    op = FakePlanOp(["r0"], [], [], [], [[0, 1]], 0)
    with pytest.raises(RuntimeError, match="seed and\\s+prefix"):
        run_plan(runner, op, states)


def test_a_finished_request_releases_its_seed_and_prefix():
    from test_nemotron_fa4 import FakePlanOp, FakeState

    def script(index, seq_ids, tokens, length):
        if index % 2 == 0:
            return logits_for([(9, 0.9), (10, 0.9), (11, 0.9), (12, 0.9)],
                              batch=len(seq_ids))
        return logits_for([(EOS_ID, 0.9), (11, 0.9), (12, 0.9), (13, 0.9)],
                          batch=len(seq_ids))

    runner = make_plan_runner(script)
    states = {"r0": FakeState("r0", [5] * 34, max_new_tokens=3 * BLOCK)}
    runner._paged_request_slots["r0"] = 0
    runner._request_seeds["r0"] = 1
    runner._request_prefix["r0"] = 34
    op = FakePlanOp(["r0"], [], [], [], [[0, 1, 2, 3]], 0)
    results = run_plan(runner, op, states)

    assert results[0].finished is True
    assert "r0" not in runner._request_seeds
    assert "r0" not in runner._request_prefix, (
        "a reused page slot must not inherit the previous request's prefix"
    )


def test_paged_scheduling_is_now_accepted_for_self_speculation():
    from fluxserve.backend.model_loader.nemotron import normalize_nemotron_args
    from test_nemotron_model import checkpoint_config, serve_args

    args = serve_args(
        parallel_decoding="self_speculation",
        attention_backend="fa4",
        kv_cache_layout="paged",
        scheduler_policy="paged",
    )
    assert normalize_nemotron_args(args, checkpoint_config()) is True

    # The dense path serves one request at a time, so continuous scheduling on
    # it is refused. The generic paged-backend check reports it first.
    with pytest.raises(ValueError, match="attention-backend fa4"):
        normalize_nemotron_args(
            serve_args(parallel_decoding="self_speculation",
                       scheduler_policy="paged"),
            checkpoint_config(),
        )

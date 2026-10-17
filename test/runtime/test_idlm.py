"""I-DLM: Introspective Strided Decoding contract tests.

These run on CPU. The runner tests use a causal stub model that keeps a
position-indexed KV cache: every forward writes its block at its offset, and
each position's prediction reads only the cache up to that position. A MASK
input drafts the true continuation except at positions chosen to be wrong, so
specs are both accepted and rejected. Greedy ISD must then reproduce greedy
autoregressive decoding exactly; a wrong prefix advance, a rollback that keeps
a rejected spec, or a misplaced block would all change the output.
"""

import asyncio
import math
from types import SimpleNamespace

import pytest
import torch

from fluxserve.backend.execution.decoders.idlm import (
    ISDDecoder,
    ISDSampling,
    make_sampling,
    sample_top_k_top_p,
    validate_sampling_params,
)
from fluxserve.backend.execution.runners.idlm.fa4 import (
    DEFAULT_GEN_BLOCK_SIZE,
    IDLMPagedRunner,
)
from fluxserve.backend.models.idlm import (
    check_idlm_config,
    idlm_rope_parameters,
    idlm_weight_plan,
    is_idlm_config,
)

VOCAB = 32
EOS_ID = 30
MASK_ID = 31


@pytest.fixture(autouse=True)
def native_silu_on_cpu(monkeypatch):
    """The tiny models run on CPU; the fused kernel only takes CUDA tensors."""
    monkeypatch.setattr("fluxserve.backend.layers.activation._is_cuda", False)


def peaked(tokens, vocab=VOCAB, high=12.0):
    out = torch.zeros(len(tokens), vocab)
    for row, token in enumerate(tokens):
        out[row, token] = high
    return out


def greedy():
    return make_sampling(SimpleNamespace(mask_id=MASK_ID), {})


def decoder(n=4):
    return ISDDecoder(gen_block_size=n, mask_id=MASK_ID, eos_ids=(EOS_ID,))


# --------------------------------------------------------------------------
# Block layout and greedy resolution
# --------------------------------------------------------------------------


def test_block_is_anchor_specs_then_masks():
    dec = decoder(4)
    assert dec.width == 7
    assert dec.block(5, []) == [5] + [MASK_ID] * 6
    assert dec.block(5, [1, 2, 3]) == [5, 1, 2, 3] + [MASK_ID] * 3
    with pytest.raises(ValueError):
        dec.block(5, [1])


def test_cold_start_emits_the_bonus_and_drafts_from_the_masks():
    dec = decoder(4)
    logits = peaked([8, 9, 10, 11, 1, 1, 1]).unsqueeze(0)
    (out,) = dec.resolve(logits, [[]], [None], [greedy()])
    assert out.emitted == [8]
    assert out.advance == 1, "only the anchor's KV is committed"
    assert (out.next_anchor, out.next_specs) == (8, [9, 10, 11])
    assert not out.rejected


def test_verify_accepting_every_spec_emits_n_tokens():
    dec = decoder(4)
    # Offsets 0..2 confirm specs 1,2,3; offset 3 is the bonus; 4..6 draft.
    logits = peaked([1, 2, 3, 4, 5, 6, 7]).unsqueeze(0)
    (out,) = dec.resolve(logits, [[1, 2, 3]], [None], [greedy()])
    assert out.emitted == [1, 2, 3, 4]
    assert out.advance == 4
    assert (out.next_anchor, out.next_specs) == (4, [5, 6, 7])
    assert out.accepted == 3


@pytest.mark.parametrize("reject_at", [0, 1, 2])
def test_first_rejection_emits_the_correction_and_goes_cold(reject_at):
    dec = decoder(4)
    specs = [1, 2, 3]
    predicted = [1, 2, 3, 4, 5, 6, 7]
    predicted[reject_at] = 20
    logits = peaked(predicted).unsqueeze(0)
    (out,) = dec.resolve(logits, [specs], [None], [greedy()])
    assert out.emitted == specs[:reject_at] + [20]
    assert out.advance == 1 + reject_at
    assert out.next_anchor == 20 and out.next_specs == []
    assert out.rejected and out.accepted == reject_at


def test_gen_block_size_one_is_autoregressive_decoding():
    dec = decoder(1)
    assert dec.width == 1 and dec.num_specs == 0
    (out,) = dec.resolve(peaked([6]).unsqueeze(0), [[]], [None], [greedy()])
    assert (out.emitted, out.advance, out.next_specs) == ([6], 1, [])


def test_the_mask_token_is_never_emitted_or_drafted():
    dec = decoder(2)
    logits = peaked([MASK_ID, MASK_ID, MASK_ID]).unsqueeze(0)
    logits[0, :, 4] = 6.0
    (out,) = dec.resolve(logits, [[]], [None], [greedy()])
    assert out.emitted == [4] and out.next_specs == [4]


def test_rows_resolve_independently_in_one_batch():
    dec = decoder(3)
    rows = [
        ([], peaked([1, 2, 3, 9, 9])),
        ([5, 6], peaked([5, 6, 7, 8, 9])),
        ([5, 6], peaked([5, 1, 7, 8, 9])),
    ]
    batched = dec.resolve(
        torch.stack([logits for _, logits in rows]),
        [specs for specs, _ in rows], [None] * 3, [greedy()] * 3,
    )
    for (specs, logits), out in zip(rows, batched):
        (alone,) = dec.resolve(logits.unsqueeze(0), [specs], [None], [greedy()])
        assert out == alone


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def test_joint_top_k_top_p_keeps_the_shorter_prefix():
    probs = torch.tensor([0.5, 0.3, 0.1, 0.05, 0.05])
    uniform = torch.arange(2000) / 2000  # every kept token is drawn
    logits = probs.log().expand(len(uniform), -1)

    def survivors(top_k, top_p):
        k, p = ISDSampling(temperature=1.0, top_k=top_k, top_p=top_p).bounds(len(probs))
        return len(set(sample_top_k_top_p(logits, k, p, uniform, limit=k).tolist()))

    assert survivors(-1, 1.0) == 5
    assert survivors(-1, 0.7) == 2, "the token that crosses top_p is kept"
    assert survivors(3, 0.7) == 2
    assert survivors(1, 0.9) == 1
    assert survivors(4, 1.0) == 4
    assert survivors(-1, 1e-6) == 1, "the most likely token always survives"


def test_sampling_parameters():
    validate_sampling_params({"temperature": 1.0, "top_k": 50, "top_p": 0.95, "seed": 3})
    for bad in ({"temperature": -1.0}, {"top_k": -2}, {"top_p": 0.0}, {"top_p": 1.5},
                {"seed": -1}, {"frequency_penalty": 0.5}, {"top_k": True}):
        with pytest.raises(ValueError):
            validate_sampling_params(bad)
    sampling = make_sampling(
        SimpleNamespace(temperature=0.6, top_k=20, top_p=0.95, seed=1, mask_id=MASK_ID),
        {"temperature": 1.0, "max_tokens": 9, "ignore_eos": True, "top_k": 0},
    )
    assert (sampling.temperature, sampling.top_k, sampling.top_p, sampling.seed) == (1.0, -1, 0.95, 1)
    assert sampling.suppress_id == MASK_ID


def test_the_request_family_selects_the_validator():
    from fluxserve.backend.execution.decoders.common import sampling_validator

    sampled = {"temperature": 1.0, "top_k": 50, "top_p": 0.95}
    sampling_validator(SimpleNamespace(sampling_family="idlm"))(sampled)
    with pytest.raises(ValueError):
        sampling_validator(SimpleNamespace(sampling_family=None))(sampled)


def sampled(seed, temperature=1.0, **kwargs):
    return ISDSampling(temperature=temperature, seed=seed, suppress_id=MASK_ID, **kwargs)


def test_identical_draft_and_anchor_distributions_always_accept():
    dec = decoder(3)
    torch.manual_seed(0)
    logits = torch.randn(1, 5, VOCAB)
    logits[..., MASK_ID] = -100.0
    specs = [int(logits[0, 0].argmax()), int(logits[0, 1].argmax())]
    # q = p: the ratio is exactly one for every spec.
    draft = logits[0, :2].clone()
    draft[:, MASK_ID] = -torch.inf
    for seed in range(20):
        (out,) = dec.resolve(logits, [specs], [draft], [sampled(seed)])
        assert not out.rejected and out.accepted == 2


def test_a_rejection_never_resamples_the_overconfident_draft():
    dec = decoder(2)
    rejected = 0
    for seed in range(200):
        logits = torch.zeros(1, 3, VOCAB)  # p uniform over the vocabulary
        draft = torch.full((1, VOCAB), -20.0)
        draft[0, 7] = 20.0                 # q almost entirely on the spec
        (out,) = dec.resolve(logits, [[7]], [draft], [sampled(seed)])
        if out.rejected:
            rejected += 1
            assert out.emitted[-1] != 7, "max(0, p - q) is zero at the spec"
    # p(7) / q(7) is about 1/31, so almost every seed rejects.
    assert rejected > 180


def test_sampled_steps_are_reproducible_per_seed():
    dec = decoder(3)
    torch.manual_seed(1)
    logits = torch.randn(1, 5, VOCAB)
    draft = torch.randn(2, VOCAB)
    first = dec.resolve(logits, [[1, 2]], [draft], [sampled(5, top_k=8, top_p=0.9)])
    again = dec.resolve(logits, [[1, 2]], [draft], [sampled(5, top_k=8, top_p=0.9)])
    assert first[0].emitted == again[0].emitted
    assert first[0].next_specs == again[0].next_specs


def test_mixed_greedy_and_sampled_rows():
    dec = decoder(2)
    torch.manual_seed(2)
    logits = torch.randn(2, 3, VOCAB)
    out = dec.resolve(logits, [[], []], [None, None], [greedy(), sampled(3)])
    (solo_greedy,) = dec.resolve(logits[:1], [[]], [None], [greedy()])
    (solo_sampled,) = dec.resolve(logits[1:], [[]], [None], [sampled(3)])
    assert out[0] == solo_greedy
    assert out[1].emitted == solo_sampled.emitted
    assert out[1].next_draft_logits.shape == (1, VOCAB)


def test_batched_sampled_rows_match_each_row_alone():
    """Rows of every kind in one step decide exactly as they would alone."""
    dec = decoder(4)
    torch.manual_seed(3)
    logits = torch.randn(6, 7, VOCAB) * 3
    specs = [[], [1, 2, 3], [4, 5, 6], [], [7, 8, 9], [10, 11, 12]]
    drafts = [torch.randn(3, VOCAB) * 3 if row else None for row in specs]
    # Fresh samplings per call: a generator is consumed as it draws.
    makers = [
        lambda: sampled(1),
        lambda: sampled(2, top_k=5),
        lambda: sampled(3, temperature=0.7, top_p=0.8),
        greedy,
        lambda: sampled(4, top_k=3, top_p=0.5),
        lambda: sampled(5, temperature=1.3),
    ]
    together = dec.resolve(logits, specs, drafts, [make() for make in makers])
    for b, make in enumerate(makers):
        (alone,) = dec.resolve(logits[b : b + 1], [specs[b]], [drafts[b]], [make()])
        mine = together[b]
        assert (mine.emitted, mine.advance, mine.next_anchor, mine.next_specs,
                mine.accepted, mine.rejected) == (
            alone.emitted, alone.advance, alone.next_anchor, alone.next_specs,
            alone.accepted, alone.rejected)
        if alone.next_draft_logits is None:
            assert mine.next_draft_logits is None
        else:
            assert torch.equal(mine.next_draft_logits, alone.next_draft_logits)


def frequencies(tokens, vocab=VOCAB):
    counts = torch.bincount(torch.tensor(tokens), minlength=vocab).double()
    return counts / counts.sum()


SUPPORT = torch.tensor([0.4, 0.25, 0.15, 0.1, 0.1])


def support_logits(probs=SUPPORT, vocab=VOCAB):
    logits = torch.full((vocab,), -torch.inf)
    logits[: len(probs)] = probs.log()
    return logits


@pytest.mark.parametrize("temperature, top_k, top_p, expected", [
    # Joint filter: top-p keeps 0.4, 0.25, 0.15 (0.8 > 0.75 before the 4th).
    (1.0, 4, 0.75, [0.5, 0.3125, 0.1875, 0.0, 0.0]),
    # T = 0.5 squares the probabilities; no filter.
    (0.5, -1, 1.0, (SUPPORT**2 / (SUPPORT**2).sum()).tolist()),
])
def test_sampled_bonus_tokens_follow_the_filtered_distribution(temperature, top_k, top_p, expected):
    dec = decoder(2)
    rows = 4000
    logits = support_logits().expand(rows, 3, VOCAB).clone()
    samplings = [sampled(seed, temperature=temperature, top_k=top_k, top_p=top_p)
                 for seed in range(rows)]
    outcomes = dec.resolve(logits, [[]] * rows, [None] * rows, samplings)
    observed = frequencies([out.emitted[-1] for out in outcomes])
    for token, probability in enumerate(expected):
        tolerance = 4 * math.sqrt(max(probability * (1 - probability), 1e-12) / rows)
        assert abs(float(observed[token]) - probability) <= max(tolerance, 1e-12), token
    assert float(observed[len(expected):].sum()) == 0.0


def test_spec_acceptance_and_residual_corrections_follow_p_and_q():
    """Accept with min(1, p/q); a rejection draws from max(0, p - q)."""
    dec = decoder(2)
    rows = 4000
    p = torch.tensor([0.1, 0.2, 0.3, 0.2, 0.2])
    q = torch.tensor([0.5, 0.1, 0.1, 0.2, 0.1])
    logits = torch.stack([support_logits(p)] * 3).expand(rows, 3, VOCAB).clone()
    draft = support_logits(q).unsqueeze(0)
    outcomes = dec.resolve(
        logits, [[0]] * rows, [draft] * rows, [sampled(seed) for seed in range(rows)]
    )
    accepted = sum(out.accepted for out in outcomes) / rows
    assert abs(accepted - 0.2) <= 4 * math.sqrt(0.2 * 0.8 / rows)
    corrections = [out.emitted[-1] for out in outcomes if out.rejected]
    observed = frequencies(corrections)
    for token, probability in enumerate([0.0, 0.25, 0.5, 0.0, 0.25]):
        tolerance = 4 * math.sqrt(max(probability * (1 - probability), 1e-12) / len(corrections))
        assert abs(float(observed[token]) - probability) <= max(tolerance, 1e-12), token


def test_the_prefill_seed_uses_the_same_filtered_sampler():
    logits = support_logits()
    seeds = [int(sampled(seed, top_p=0.75).categorical(logits)) for seed in range(3000)]
    observed = frequencies(seeds)
    for token, probability in enumerate([0.5, 0.3125, 0.1875]):
        assert abs(float(observed[token]) - probability) <= 4 * math.sqrt(
            probability * (1 - probability) / 3000
        )
    assert float(observed[3:].sum()) == 0.0


def test_inverse_cdf_never_draws_a_zero_weight_token():
    from fluxserve.backend.execution.decoders.idlm import inverse_cdf

    weights = torch.tensor([[0.0, 1.0, 0.0, 3.0, 0.0]])
    for uniform, expected in ((0.0, 1), (0.2499, 1), (0.25, 3), (0.9999999, 3)):
        assert int(inverse_cdf(weights, torch.tensor([uniform]))[0]) == expected


# --------------------------------------------------------------------------
# Runner loop against a causal stub
# --------------------------------------------------------------------------


def ar_rule(context, stop_at):
    """The stub's language: deterministic next token for a clean context."""
    if len(context) >= stop_at:
        return EOS_ID
    return (3 * context[-1] + 5 * len(context)) % 29 + 1


class CausalStub:
    """A causal LM over a position-indexed cache, with a drafter that is right
    except where ``wrong(position)`` says so."""

    def __init__(self, *, stop_at=10**9, wrong=lambda position: position % 3 == 0,
                 peak=12.0):
        self.stop_at = stop_at
        self.wrong = wrong
        self.peak = peak
        self.cache: dict[int, dict[int, int]] = {}
        self.launches = []

    def roll(self, context, upto):
        context = list(context)
        while len(context) <= upto:
            context.append(ar_rule(context, self.stop_at))
        return context

    def forward(self, *, seq_ids, tokens, q_offsets, causal, is_prefill,
                max_input_len, q_lens=None, positions=None):
        assert causal, "I-DLM is causal in every phase"
        self.launches.append({
            "rows": seq_ids.tolist(), "prefill": is_prefill,
            "offsets": q_offsets.tolist(), "tokens": tokens.tolist(),
        })
        out = torch.zeros(len(seq_ids), tokens.shape[1], VOCAB)
        for b, seq in enumerate(seq_ids.tolist()):
            cache = self.cache.setdefault(seq, {})
            offset = int(q_offsets[b])
            length = int(q_lens[b]) if q_lens is not None else tokens.shape[1]
            for j in range(length):
                cache[offset + j] = int(tokens[b, j])
            for j in range(length):
                position = offset + j
                context = [cache[p] for p in range(position + 1)]
                if MASK_ID in context:
                    clean = context[: context.index(MASK_ID)]
                    target = self.roll(clean, position + 1)[position + 1]
                    if self.wrong(position + 1) and target != EOS_ID:
                        target = target % 29 + 1
                else:
                    target = ar_rule(context, self.stop_at)
                out[b, j, target] = self.peak
        return out


def make_runner(stub, *, n=4, page=16, gen_length=64):
    runner = object.__new__(IDLMPagedRunner)
    runner.model = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(
        num_hidden_layers=1, num_key_value_heads=1, head_dim=8,
        hidden_size=8, num_attention_heads=1, vocab_size=VOCAB,
    )))
    runner.device = "cpu"
    runner.runner_config = SimpleNamespace(
        mask_id=MASK_ID, eos_id=EOS_ID, eos_ids=(EOS_ID,), block_length=page,
        page_size=page, idlm_gen_block_size=n, gen_length=gen_length,
        temperature=0.0, seed=0, top_k=-1, top_p=1.0,
    )
    runner.init_decoder()
    runner.block_length = page
    runner.max_length = 1024
    runner.num_forwards = 0
    runner.early_stop = True
    runner.last_stats = []
    runner.nemotron_graph_runner = None
    runner._isd_rows = {}
    runner.allocate_kv_cache = lambda batch_size: None
    runner._paged_forward = stub.forward
    return runner


def ar_reference(prompt, length, stop_at):
    context, out = list(prompt), []
    while len(out) < length:
        token = ar_rule(context, stop_at)
        out.append(token)
        context.append(token)
        if token == EOS_ID:
            break
    return out


def generate(runner, prompts, length):
    width = max(len(p) for p in prompts)
    batch = torch.full((len(prompts), width), MASK_ID, dtype=torch.long)
    for i, prompt in enumerate(prompts):
        batch[i, : len(prompt)] = torch.tensor(prompt)
    out = runner.generate(
        batch, prompt_lengths=[len(p) for p in prompts],
        generation_lengths=[length] * len(prompts),
    )
    rows = []
    for row in out.tolist():
        tokens = row[width:]
        rows.append([t for t in tokens if t != MASK_ID])
    return rows


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_greedy_isd_reproduces_autoregressive_decoding(n):
    stub = CausalStub()
    runner = make_runner(stub, n=n)
    prompts = [[4, 9, 2], [7], [11, 3, 3, 5, 1]]
    out = generate(runner, prompts, 40)
    for prompt, row in zip(prompts, out):
        assert row == ar_reference(prompt, 40, stub.stop_at)
    for stats in runner.last_stats:
        assert 1.0 <= stats["tokens_per_forward"] <= n
        if n > 1:
            assert stats["rejections"] > 0 and stats["accepted_specs"] > 0


def test_sampled_isd_carries_draft_logits_between_steps():
    """T > 0 through the runner: q for a verify is the previous step's draft.

    The stub is peaked hard enough that sampling is deterministic in practice,
    so the output is still the autoregressive one; what this pins down is that
    the stored draft rows reach the next step's p/q test in the right shape.
    """
    stub = CausalStub(peak=40.0)
    runner = make_runner(stub, n=4)
    prompts = [[4, 9, 2], [7]]
    width = max(len(p) for p in prompts)
    batch = torch.full((2, width), MASK_ID, dtype=torch.long)
    for i, prompt in enumerate(prompts):
        batch[i, : len(prompt)] = torch.tensor(prompt)
    out = runner.generate(
        batch, prompt_lengths=[3, 1], generation_lengths=[30, 30],
        sampling_params=[{"temperature": 1.0, "seed": 11, "top_k": 5, "top_p": 0.9},
                         {"temperature": 0.7, "seed": 12}],
    )
    for i, prompt in enumerate(prompts):
        row = [t for t in out[i, width:].tolist() if t != MASK_ID]
        assert row == ar_reference(prompt, 30, stub.stop_at)
    for stats in runner.last_stats:
        assert stats["verify_steps"] > 0 and stats["rejections"] > 0


def test_a_perfect_drafter_emits_n_tokens_per_verify_step():
    stub = CausalStub(wrong=lambda position: False)
    runner = make_runner(stub, n=4)
    generate(runner, [[4, 9, 2]], 41)
    stats = runner.last_stats[0]
    assert stats["rejections"] == 0
    # The cold start emits the seed and its bonus; every later step emits N.
    assert stats["tokens_per_step"][0] == 2
    assert set(stats["tokens_per_step"][1:-1]) == {4}


def test_every_step_starts_at_the_committed_prefix():
    stub = CausalStub()
    runner = make_runner(stub, n=4)
    generate(runner, [[4, 9, 2]], 30)
    decode = [item for item in stub.launches if not item["prefill"]]
    offsets = [item["offsets"][0] for item in decode]
    assert offsets[0] == 3, "the first block starts after the whole prompt"
    for before, after, item in zip(offsets, offsets[1:], decode):
        specs = [t for t in item["tokens"][0][1:] if t != MASK_ID]
        assert 1 <= after - before <= 1 + len(specs)


def test_decode_stats_reach_the_metrics_endpoint():
    from fluxserve.backend.engine.executor import BlockDiffusionExecutor

    runner = make_runner(CausalStub(), n=4)
    generate(runner, [[4, 9, 2], [7]], 30)
    stats = runner.decode_stats()
    rows = runner.last_stats
    assert stats["idlm_row_steps"] == sum(row["steps"] for row in rows)
    # Every committed token except the two prefill seeds.
    assert stats["idlm_step_tokens"] == sum(sum(row["tokens_per_step"]) for row in rows) - 2
    assert stats["idlm_accepted_specs"] == sum(row["accepted_specs"] for row in rows)
    assert 1.0 < stats["idlm_tokens_per_step"] <= 4
    assert 0.0 < stats["idlm_spec_acceptance"] < 1.0

    executor = object.__new__(BlockDiffusionExecutor)
    executor.runner = runner
    merged = executor.runner_stats()
    assert merged["num_forwards"] == runner.num_forwards
    assert merged["idlm_tokens_per_step"] == stats["idlm_tokens_per_step"]


def test_eos_inside_a_step_ends_the_row():
    stub = CausalStub(stop_at=9, wrong=lambda position: False)
    runner = make_runner(stub, n=4)
    (row,) = generate(runner, [[4, 9, 2]], 40)
    assert row == ar_reference([4, 9, 2], 40, 9)
    assert row[-1] == EOS_ID


def test_a_one_token_budget_needs_no_decode_forward():
    stub = CausalStub()
    runner = make_runner(stub, n=4)
    (row,) = generate(runner, [[4, 9, 2]], 1)
    assert row == ar_reference([4, 9, 2], 1, stub.stop_at)
    assert all(item["prefill"] for item in stub.launches)


def test_a_wider_forward_than_a_page_is_rejected():
    from fluxserve.cli.bench_offline import normalize_idlm_args

    args = serve_args(idlm_gen_block_size=17, page_size=32, block_length=32)
    with pytest.raises(ValueError, match="wider than page_size"):
        normalize_idlm_args(args, idlm_config())


# --------------------------------------------------------------------------
# Continuously scheduled serving
# --------------------------------------------------------------------------


def make_plan_runner(stub, **kwargs):
    from test_nemotron_fa4 import FakePagedCache

    runner = make_runner(stub, **kwargs)
    runner.server_args = SimpleNamespace(scheduler_num_device_pages=64, max_num_seqs=4)
    runner._paged_request_slots = {}
    runner._request_seeds = {}
    runner._request_sampling = {}
    runner.past_key_values = FakePagedCache()
    runner.ensure_paged_kv_cache = lambda *, num_device_pages: None
    return runner


def serve_online(runner, prompt, max_new_tokens, *, ignore_eos=False):
    from test_nemotron_fa4 import FakePlanOp, FakeState, FakeTokenizer

    state = FakeState("r0", prompt, max_new_tokens=max_new_tokens, ignore_eos=ignore_eos)
    state.sampling_params = {}
    states = {"r0": state}
    prefill = FakePlanOp(["r0"], [len(prompt)], [0], list(prompt), [[1, 2, 3]], 1)
    (result,) = asyncio.run(runner.execute_paged_forward_plan(prefill, states, FakeTokenizer()))
    assert result.token_ids == [] and not result.finished
    results = []
    for _ in range(4 * max_new_tokens + 4):
        prefix = runner._isd_rows["r0"].prefix
        decode = FakePlanOp(["r0"], [], [], [], [[1, 2, 3, 4]], 0)
        (result,) = asyncio.run(
            runner.execute_paged_forward_plan(decode, states, FakeTokenizer())
        )
        results.append(result)
        state.output_ids.extend(result.token_ids)
        if result.finished:
            break
        assert result.reserve_tokens == runner._isd_rows["r0"].prefix - prefix, (
            "only the committed advance needs new capacity"
        )
        assert result.decode_block_completed is False
    return state, results


def test_online_serving_streams_the_autoregressive_tokens():
    stub = CausalStub()
    runner = make_plan_runner(stub, n=4)
    state, results = serve_online(runner, [4, 9, 2], 25)
    assert state.output_ids == ar_reference([4, 9, 2], 25, stub.stop_at)
    assert results[-1].finish_reason == "length"
    decode = [item for item in stub.launches if not item["prefill"]]
    assert len(decode) == len(results), "one plan call is one ISD step"
    assert "r0" not in runner._isd_rows and "r0" not in runner._paged_request_slots
    stats = runner.decode_stats()
    assert stats["idlm_active_rows"] == 0 and stats["idlm_slots_in_use"] == 0


def test_aborting_mid_stream_releases_the_row_and_its_slot():
    from test_nemotron_fa4 import FakePlanOp, FakeState, FakeTokenizer

    runner = make_plan_runner(CausalStub(), n=4)
    state = FakeState("r0", [4, 9, 2], max_new_tokens=40)
    state.sampling_params = {}
    states = {"r0": state}
    prefill = FakePlanOp(["r0"], [3], [0], [4, 9, 2], [[1, 2, 3]], 1)
    asyncio.run(runner.execute_paged_forward_plan(prefill, states, FakeTokenizer()))
    decode = FakePlanOp(["r0"], [], [], [], [[1, 2, 3, 4]], 0)
    (result,) = asyncio.run(runner.execute_paged_forward_plan(decode, states, FakeTokenizer()))
    assert not result.finished
    stats = runner.decode_stats()
    assert stats["idlm_active_rows"] == 1 and stats["idlm_slots_in_use"] == 1

    # What AsyncLLM.abort calls once the client disconnects.
    runner.release_paged_requests(["r0"])
    stats = runner.decode_stats()
    assert stats["idlm_active_rows"] == 0 and stats["idlm_slots_in_use"] == 0
    assert "r0" not in runner._request_seeds and "r0" not in runner._request_sampling


def test_online_serving_stops_on_eos_without_emitting_it():
    stub = CausalStub(stop_at=12)
    runner = make_plan_runner(stub, n=4)
    state, results = serve_online(runner, [4, 9, 2], 40)
    expected = ar_reference([4, 9, 2], 40, 12)
    assert expected[-1] == EOS_ID
    assert state.output_ids == expected[:-1]
    assert results[-1].finish_reason == "stop"


def test_reservation_keeps_two_pages_past_the_prefix_in_the_real_scheduler():
    """Drive flux_scheduler with ISD's advance pattern and check capacity."""
    pytest.importorskip("flux_scheduler")
    from fluxserve.backend.engine.scheduler_adapter import PagedSchedulerAdapter

    page, width, max_model_len, prompt_len = 16, 7, 256, 37
    margin = 2 * page
    max_new = max_model_len - prompt_len - margin
    adapter = PagedSchedulerAdapter(
        max_batch_size=1, max_scheduled_tokens=256, page_size=page,
        num_device_pages=64, max_model_len=max_model_len, full_prompt_prefill=True,
    )
    adapter.submit([SimpleNamespace(rid="r0", input_ids=list(range(1, prompt_len + 1)))])
    advances = [1, 4, 2, 1, 4, 4, 3, 1, 4]
    prefix, emitted, step = prompt_len, 0, 0
    while True:
        plan = adapter.next_plan()
        (op,) = list(plan.forward)
        pages = list(op.occupied_pages[0])
        assert len(pages) <= math.ceil(max_model_len / page)
        if op.num_extends():
            assert list(op.input_lengths) == [prompt_len]
            adapter.advance_forward()
            continue
        assert len(pages) * page >= prefix + width, "the step's writes must fit"
        advance = advances[step % len(advances)]
        tokens = advance + (1 if step == 0 else 0)
        step += 1
        prefix += advance
        emitted += tokens
        if emitted >= max_new:
            adapter.advance_forward(finished_ids=["r0"])
            break
        adapter.advance_forward(
            token_results={"r0": [5] * tokens}, reserve_tokens={"r0": advance}
        )
    assert step > 10


# --------------------------------------------------------------------------
# Checkpoint recognition, configuration and weights
# --------------------------------------------------------------------------


def idlm_config(**overrides):
    values = dict(
        architectures=["SDARForCausalLM"], model_type="sdar", use_regular_causal=True,
        mask_token_id=151669, vocab_size=151936, hidden_size=4096, intermediate_size=12288,
        num_hidden_layers=36, num_attention_heads=32, num_key_value_heads=8, head_dim=128,
        max_position_embeddings=40960, tie_word_embeddings=False, rms_norm_eps=1e-6,
        rope_theta=1000000, rope_scaling=None, hidden_act="silu", attention_bias=False,
        use_sliding_window=False, block_size=3,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def serve_args(**overrides):
    values = dict(
        attention_backend="fa4", kv_cache_layout="paged", flashinfer_prefill_mode="paged",
        flashinfer_cache_mode="paged", page_size=32, block_length=32,
        parallel_decoding="threshold", parallel_decoding_explicit=False,
        idlm_gen_block_size=None, temperature=0.0, seed=None, top_k=-1, top_p=1.0,
        nemotron_prefill_chunk_size=1024, max_model_len=8192, use_prefill_cuda_graph=False,
        use_cuda_graph=False, max_thinking_tokens=None, end_think_token_id=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_sdar_checkpoints_are_recognized_and_plain_sdar_is_refused():
    assert is_idlm_config(idlm_config())
    assert is_idlm_config(idlm_config(architectures=None))
    assert not is_idlm_config(idlm_config(architectures=["Qwen3ForCausalLM"], model_type="qwen3"))
    check_idlm_config(idlm_config())
    with pytest.raises(ValueError, match="use_regular_causal"):
        check_idlm_config(idlm_config(use_regular_causal=False))
    with pytest.raises(ValueError, match="mask_token_id"):
        check_idlm_config(idlm_config(mask_token_id=None))


def test_rope_theta_is_read_from_standardized_parameters():
    config = idlm_config(rope_parameters={"rope_type": "default", "rope_theta": 5e5})
    assert idlm_rope_parameters(config) == (5e5, {"rope_type": "default", "rope_theta": 5e5})
    assert idlm_rope_parameters(idlm_config()) == (1e6, None)


def test_normalization_defaults_to_isd():
    from fluxserve.cli.bench_offline import normalize_idlm_args

    args = serve_args()
    assert normalize_idlm_args(args, idlm_config()) is True
    assert args.parallel_decoding == "isd"
    assert args.idlm_gen_block_size == DEFAULT_GEN_BLOCK_SIZE


@pytest.mark.parametrize("overrides, message", [
    ({"parallel_decoding": "threshold", "parallel_decoding_explicit": True}, "isd"),
    ({"attention_backend": "sdpa"}, "paged attention"),
    ({"kv_cache_layout": "dense"}, "paged"),
    ({"use_prefill_cuda_graph": True}, "decode-cuda-graph"),
    ({"max_thinking_tokens": 64}, "thinking"),
    ({"top_p": 0.0}, "top_p"),
    ({"idlm_gen_block_size": 0}, ">= 1"),
])
def test_normalization_rejects_unsupported_settings(overrides, message):
    from fluxserve.cli.bench_offline import normalize_idlm_args

    with pytest.raises(ValueError, match=message):
        normalize_idlm_args(serve_args(**overrides), idlm_config())


def test_runner_config_takes_the_checkpoint_ids():
    from fluxserve.backend.execution.forward_batch_info import RunnerConfig
    from fluxserve.cli.bench_offline import apply_idlm_runner_config

    config = RunnerConfig(eos_ids=(151645, 151643))
    apply_idlm_runner_config(config, idlm_config(), serve_args(top_k=50, top_p=0.95, temperature=1.0))
    assert (config.mask_id, config.eos_id, tuple(config.eos_ids)) == (151669, 151645, (151645, 151643))
    assert (config.temperature, config.top_k, config.top_p) == (1.0, 50, 0.95)
    assert config.idlm_gen_block_size == DEFAULT_GEN_BLOCK_SIZE


def test_weight_plan_covers_a_qwen3_layer():
    plan = idlm_weight_plan(idlm_config(num_hidden_layers=1))
    assert len(plan) == 3 + 11
    assert plan["model.layers.0.self_attn.k_proj.weight"] == (
        "model.layers.0.self_attn.qkv_proj.weight", "k", (1024, 4096)
    )
    assert plan["model.layers.0.self_attn.q_norm.weight"][2] == (128,)
    tied = idlm_weight_plan(idlm_config(num_hidden_layers=1, tie_word_embeddings=True))
    assert "lm_head.weight" not in tied


def test_weight_plan_matches_the_released_checkpoint():
    """Header-only comparison with I-DLM-8B, when it is in the local cache."""
    import json

    from fluxserve.backend.model_loader.loader import read_safetensors_header

    try:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            "yifanyu/I-DLM-8B", allow_patterns=["*.json"], local_files_only=True
        )
    except Exception:
        pytest.skip("yifanyu/I-DLM-8B is not in the local Hugging Face cache")
    import os

    with open(os.path.join(snapshot, "config.json")) as handle:
        config = SimpleNamespace(**json.load(handle))
    with open(os.path.join(snapshot, "model.safetensors.index.json")) as handle:
        shards = sorted(set(json.load(handle)["weight_map"].values()))
    paths = [os.path.join(snapshot, shard) for shard in shards]
    if not all(os.path.isfile(path) for path in paths):
        pytest.skip("I-DLM-8B weights are not fully cached")
    header = {}
    for path in paths:
        header.update(read_safetensors_header(path))
    plan = idlm_weight_plan(config)
    assert set(header) == set(plan)
    for name, (_, _, shape) in plan.items():
        assert tuple(header[name]["shape"]) == shape, name


# --------------------------------------------------------------------------
# R-ISD: the gated LoRA
# --------------------------------------------------------------------------

TINY_MASK = 60
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def tiny_idlm_config(**overrides):
    return idlm_config(**{
        "num_hidden_layers": 2, "hidden_size": 32, "intermediate_size": 48,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
        "vocab_size": 64, "mask_token_id": TINY_MASK, "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0, **overrides,
    })


def tiny_model(seed=0):
    from fluxserve.backend.models.idlm import IDLMForCausalLM

    torch.manual_seed(seed)
    model = IDLMForCausalLM(tiny_idlm_config()).eval()
    for _, parameter in model.named_parameters():
        if parameter.dim() == 1:
            torch.nn.init.ones_(parameter)
        else:
            torch.nn.init.normal_(parameter, std=0.5)
    return model


def lora_shapes(config):
    hidden, inter = config.hidden_size, config.intermediate_size
    q, kv = config.num_attention_heads * config.head_dim, config.num_key_value_heads * config.head_dim
    return {"q_proj": (hidden, q), "k_proj": (hidden, kv), "v_proj": (hidden, kv),
            "o_proj": (q, hidden), "gate_proj": (hidden, inter), "up_proj": (hidden, inter),
            "down_proj": (inter, hidden)}


def random_adapter(config, rank=4, seed=1):
    torch.manual_seed(seed)
    tensors = {}
    for layer in range(config.num_hidden_layers):
        for target, (in_dim, out_dim) in lora_shapes(config).items():
            block = "self_attn" if target in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
            prefix = f"base_model.model.model.layers.{layer}.{block}.{target}"
            tensors[prefix + ".lora_A.weight"] = torch.randn(rank, in_dim)
            tensors[prefix + ".lora_B.weight"] = torch.randn(out_dim, rank)
    return tensors


def write_adapter(directory, config, rank=4, **config_overrides):
    import json

    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    tensors = random_adapter(config, rank)
    save_file(tensors, str(directory / "adapter_model.safetensors"))
    adapter_config = {"peft_type": "LORA", "r": rank, "lora_alpha": 2 * rank,
                      "target_modules": list(TARGETS), "bias": "none", "use_dora": False}
    adapter_config.update(config_overrides)
    (directory / "adapter_config.json").write_text(json.dumps(adapter_config))
    (directory / "added_tokens.json").write_text(json.dumps({"<|MASK|>": TINY_MASK}))
    return tensors


def causal(length, batch=1):
    return torch.tril(torch.ones(length, length, dtype=torch.bool)).unsqueeze(0).expand(batch, -1, -1)


def test_the_residual_is_gated_per_token():
    from fluxserve.backend.models.idlm import IDLMGatedLoRA

    torch.manual_seed(3)
    a, b = torch.randn(4, 8), torch.randn(6, 4)
    lora = IDLMGatedLoRA(mask_id=9, scaling=2.0, layers=[{"o_proj": (a, b)}])
    ids = torch.tensor([[1, 9, 2, 9]])
    x = torch.randn(1, 4, 8)
    base = torch.randn(1, 4, 6)
    lora.set_gate(ids, x.dtype)
    out = lora.apply(0, "o", x, base.clone(), (6,))
    expected = base + 2.0 * (ids == 9).unsqueeze(-1) * (x @ a.t() @ b.t())
    assert torch.allclose(out, expected, atol=1e-5)
    assert torch.equal(out[0, 0], base[0, 0]) and torch.equal(out[0, 2], base[0, 2])


def test_fused_slices_receive_their_own_expansion():
    from fluxserve.backend.models.idlm import IDLMGatedLoRA

    torch.manual_seed(4)
    weights = {name: (torch.randn(2, 8), torch.randn(size, 2))
               for name, size in (("q_proj", 6), ("k_proj", 3), ("v_proj", 3))}
    lora = IDLMGatedLoRA(mask_id=0, scaling=0.5, layers=[weights])
    x = torch.randn(1, 2, 8)
    lora.set_gate(torch.zeros(1, 2, dtype=torch.long), x.dtype)  # every token gated on
    out = lora.apply(0, "qkv", x, torch.zeros(1, 2, 12), (6, 3, 3))
    expected = torch.cat([0.5 * x @ a.t() @ b.t() for a, b in weights.values()], dim=-1)
    assert torch.allclose(out, expected, atol=1e-5)


def test_clean_positions_are_exactly_the_base_model(tmp_path):
    from fluxserve.backend.model_loader.loader import load_idlm_lora

    model = tiny_model()
    ids = torch.tensor([[5, 9, 2, 11, TINY_MASK, TINY_MASK, TINY_MASK]])
    base = model(input_ids=ids, attention_mask=causal(7), use_cache=False).logits

    write_adapter(tmp_path / "adapter", model.config)
    lora = load_idlm_lora(model, tmp_path / "adapter", mask_id=TINY_MASK)
    inactive = model(input_ids=ids, attention_mask=causal(7), use_cache=False).logits
    assert torch.equal(inactive, base), "an attached but inactive adapter changes nothing"

    lora.active = True
    gated = model(input_ids=ids, attention_mask=causal(7), use_cache=False).logits
    # Causal attention: positions before the first MASK never see the residual.
    assert torch.equal(gated[0, :4], base[0, :4])
    assert not torch.allclose(gated[0, 4:], base[0, 4:])

    # Without MASK inputs an active adapter is a no-op: prefill and verify
    # positions are exactly the base model.
    clean = torch.tensor([[5, 9, 2, 11, 3, 4, 7]])
    active = model(input_ids=clean, attention_mask=causal(7), use_cache=False).logits
    lora.active = False
    plain = model(input_ids=clean, attention_mask=causal(7), use_cache=False).logits
    assert torch.equal(active, plain)


def test_adapter_scaling_and_validation(tmp_path):
    from fluxserve.backend.model_loader.loader import load_idlm_lora

    model = tiny_model()
    write_adapter(tmp_path / "plain", model.config, rank=4)
    assert load_idlm_lora(model, tmp_path / "plain", mask_id=TINY_MASK).scaling == 2.0
    write_adapter(tmp_path / "rslora", model.config, rank=4, use_rslora=True)
    assert load_idlm_lora(model, tmp_path / "rslora", mask_id=TINY_MASK).scaling == 4.0

    write_adapter(tmp_path / "dora", model.config, use_dora=True)
    with pytest.raises(ValueError, match="use_dora"):
        load_idlm_lora(model, tmp_path / "dora", mask_id=TINY_MASK)
    write_adapter(tmp_path / "embed", model.config, target_modules=["q_proj", "embed_tokens"])
    with pytest.raises(ValueError, match="subset"):
        load_idlm_lora(model, tmp_path / "embed", mask_id=TINY_MASK)

    from safetensors.torch import load_file, save_file

    tensors = write_adapter(tmp_path / "short", model.config)
    tensors.pop("base_model.model.model.layers.1.mlp.up_proj.lora_B.weight")
    save_file(tensors, str(tmp_path / "short" / "adapter_model.safetensors"))
    with pytest.raises(ValueError, match="missing 1"):
        load_idlm_lora(model, tmp_path / "short", mask_id=TINY_MASK)

    tensors = load_file(str(tmp_path / "plain" / "adapter_model.safetensors"))
    tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"] = torch.randn(4, 31)
    write_adapter(tmp_path / "shape", model.config)
    save_file(tensors, str(tmp_path / "shape" / "adapter_model.safetensors"))
    with pytest.raises(ValueError, match="shape mismatch"):
        load_idlm_lora(model, tmp_path / "shape", mask_id=TINY_MASK)


def test_normalization_accepts_qwen3_only_as_an_adapter_base(tmp_path):
    from fluxserve.cli.bench_offline import normalize_idlm_args

    qwen3 = idlm_config(architectures=["Qwen3ForCausalLM"], model_type="qwen3",
                        use_regular_causal=None, mask_token_id=None, vocab_size=64)
    with pytest.raises(ValueError, match="--idlm-lora"):
        normalize_idlm_args(serve_args(), qwen3)

    write_adapter(tmp_path / "adapter", tiny_idlm_config())
    args = serve_args(idlm_lora=str(tmp_path / "adapter"), tp_size=1)
    assert normalize_idlm_args(args, qwen3) is True
    assert args.idlm_mask_id == TINY_MASK

    with pytest.raises(ValueError, match="tp-size 1"):
        normalize_idlm_args(serve_args(idlm_lora=str(tmp_path / "adapter"), tp_size=2), qwen3)
    with pytest.raises(ValueError, match="differs"):
        normalize_idlm_args(
            serve_args(idlm_lora=str(tmp_path / "adapter"), tp_size=1),
            idlm_config(vocab_size=64, mask_token_id=61),
        )
    # A plain I-DLM checkpoint needs no adapter and takes its own MASK id.
    plain = serve_args()
    assert normalize_idlm_args(plain, idlm_config()) is True
    assert plain.idlm_mask_id == 151669


def test_the_adapter_is_active_for_decode_steps_only():
    stub = CausalStub()
    runner = make_runner(stub, n=4)
    runner.lora = SimpleNamespace(active=False)
    seen = []
    forward = stub.forward

    def recording(**kwargs):
        seen.append((kwargs["is_prefill"], runner.lora.active))
        return forward(**kwargs)

    runner._paged_forward = recording
    generate(runner, [[4, 9, 2]], 12)
    assert seen[0] == (True, False), "the prompt's KV must be the base model's"
    assert all(active for prefill, active in seen[1:] if not prefill)
    assert runner.lora.active is False


# --------------------------------------------------------------------------
# Parity with transformers' Qwen3 (dense causal attention)
# --------------------------------------------------------------------------


def hf_qwen3(config, seed=0):
    transformers = pytest.importorskip("transformers")
    hf_config = transformers.Qwen3Config(
        vocab_size=config.vocab_size, hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads, head_dim=config.head_dim,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps, rope_theta=config.rope_theta,
        tie_word_embeddings=False, attention_bias=False, use_sliding_window=False,
    )
    hf_config._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = transformers.Qwen3ForCausalLM(hf_config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dim() == 1:
                # Non-unit norms, so a misplaced q/k norm cannot cancel out.
                parameter.uniform_(0.5, 1.5)
            else:
                parameter.normal_(std=0.3)
    return model


def loaded_from(reference, config):
    from fluxserve.backend.models.idlm import IDLMForCausalLM

    model = IDLMForCausalLM(config).eval()
    consumed, unexpected = model.load_weights(reference.state_dict().items())
    assert not unexpected
    assert consumed == set(idlm_weight_plan(config))
    return model


def test_the_model_matches_transformers_qwen3():
    config = tiny_idlm_config()
    reference = hf_qwen3(config)
    model = loaded_from(reference, config)
    ids = torch.randint(0, TINY_MASK, (2, 11), generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        ours = model(input_ids=ids, attention_mask=causal(11, batch=2), use_cache=False).logits
        ref = reference(ids).logits
    torch.testing.assert_close(ours, ref, atol=1e-4, rtol=1e-4)


def test_gated_positions_run_the_merged_adapter(tmp_path):
    from fluxserve.backend.model_loader.loader import load_idlm_lora

    config = tiny_idlm_config()
    reference = hf_qwen3(config)
    model = loaded_from(reference, config)
    tensors = write_adapter(tmp_path / "adapter", config, rank=4)
    lora = load_idlm_lora(model, tmp_path / "adapter", mask_id=TINY_MASK)
    lora.active = True
    # PEFT: W' = W + (alpha / r) * B @ A, here alpha = 2r.
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            key = "base_model.model." + name.removesuffix(".weight")
            if key + ".lora_A.weight" in tensors:
                parameter += 2.0 * tensors[key + ".lora_B.weight"] @ tensors[key + ".lora_A.weight"]
    # An all-MASK input gates every position, so the whole forward is the
    # merged model's.
    ids = torch.full((1, 6), TINY_MASK)
    with torch.no_grad():
        ours = model(input_ids=ids, attention_mask=causal(6), use_cache=False).logits
        ref = reference(ids).logits
    scale = ref.abs().max()
    torch.testing.assert_close(ours / scale, ref / scale, atol=1e-5, rtol=1e-4)

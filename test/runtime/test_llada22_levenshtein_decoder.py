# Copyright (c) 2026 FLUX-OSS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""CPU tests for LLaDA2.2 Levenshtein joint decoding.

The parity tests drive the checkpoint's own ``_joint_decode_block`` (bound to
a stub with a scripted forward) and the FluxServe decoder loop with identical
state->logits scripts, and require identical final tokens. Scripts are
constructed so the reference's random anti-loop position choice is either
never triggered or forced (single changed position), keeping both sides
deterministic.
"""

import importlib
import os
import shutil
import sys
import types

import pytest
import torch

from fluxserve.backend.execution.decoders import (
    LevenshteinJointDecoder,
    load_decoder,
)
from fluxserve.backend.execution.decoders.levenshtein import (
    apply_edit_operations,
    m2t_schedule_need,
)
from fluxserve.backend.execution.forward_batch_info import RunnerConfig

VOCAB = 20
MASK_ID = 12
EOS_ID = 13
DEL_ID = 14
SPL_ID = 15



# --------------------------------------------------------------- helpers


def make_logits_row(cand, prob):
    """[L, V] float32 logits with softmax(cand)=prob per position."""
    L = len(cand)
    prob_t = torch.as_tensor(prob, dtype=torch.float64).expand(L)
    rest = (1.0 - prob_t) / (VOCAB - 1)
    logits = torch.log(rest).unsqueeze(-1).repeat(1, VOCAB)
    cand_t = torch.as_tensor(cand, dtype=torch.long).unsqueeze(-1)
    logits.scatter_(-1, cand_t, torch.log(prob_t).unsqueeze(-1))
    return logits.to(torch.float32)


class FakeTokenArray:
    def __init__(self, data):
        self.data = data


def make_decoder(**overrides):
    kwargs = dict(
        threshold=0.5,
        editing_threshold=0.0,
        temperature=0,
        mask_id=MASK_ID,
        eos_id=EOS_ID,
        delete_token_id=DEL_ID,
        split_token_id=SPL_ID,
        steps=8,
        max_post_steps=16,
        max_steps_per_block=100,
    )
    kwargs.update(overrides)
    return LevenshteinJointDecoder(**kwargs)


def run_flux_loop(decoder, full, block_start, block_length, prompt_len, script,
                  max_iters=200):
    """Drive the runner's decode-loop contract for one row on CPU."""
    x = FakeTokenArray(full.clone().unsqueeze(0))
    row_state = decoder.make_row_state(1, block_length, "cpu")
    block_start_t = torch.tensor([block_start])
    seq_ids = torch.tensor([0])
    prompt_t = torch.tensor([prompt_len])
    offsets = torch.arange(block_length) + block_start
    iters = 0
    while True:
        block = x.data[0, offsets].clone()
        logits = script(block.tolist()).unsqueeze(0)
        decoder.batch_decode(
            logits,
            block_start_t,
            x,
            block_length,
            prompt_lengths=prompt_t,
            row_state=row_state,
            seq_ids=seq_ids,
        )
        after = x.data[0, offsets]
        had_mask = bool((block == MASK_ID).any())
        changed = bool((after != block).any())
        iters += 1
        if (not had_mask) and (not changed):
            return x.data[0], iters, row_state
        assert iters < max_iters, "flux loop did not terminate"


# --------------------------------------------------------------- reference


# The `reference_module` fixture lives in conftest.py: it resolves the
# checkpoint's own modeling code from FLUXSERVE_LLADA22_REF_DIR or the local
# Hugging Face cache, and skips when neither is available.


def run_reference_loop(module, full, block_start, block_length, script,
                       steps, threshold, editing_threshold, max_post_steps,
                       max_steps_per_block=100):
    """Run the checkpoint's _joint_decode_block with a scripted forward."""
    lm = module.LLaDA2MoeModelLM
    stub = types.SimpleNamespace()
    counter = {"n": 0}

    def forward(window, attention_mask=None, position_ids=None):
        counter["n"] += 1
        block = window[0, block_start : block_start + block_length]
        block_logits = script(block.tolist())
        logits = torch.zeros(1, window.shape[1], VOCAB)
        logits[0, block_start : block_start + block_length] = block_logits
        return types.SimpleNamespace(logits=logits)

    stub.forward = forward
    for name in ("_diffusion_sample", "_resample_to_escape_loop"):
        stub.__dict__[name] = types.MethodType(getattr(lm, name), stub)
    for name in (
        "_get_num_transfer_tokens",
        "_apply_edit_operations_with_tracking",
        "_top_k_logits",
        "_top_p_logits",
    ):
        stub.__dict__[name] = getattr(lm, name)

    x = full.clone().unsqueeze(0)
    lm._joint_decode_block(
        stub,
        x,
        block_start,
        block_start + block_length,
        None,
        None,
        temperature=0.0,
        top_k=None,
        top_p=None,
        steps=steps,
        threshold=threshold,
        editing_threshold=editing_threshold,
        max_post_steps=max_post_steps,
        mask_id=MASK_ID,
        delete_token_id=DEL_ID,
        split_token_id=SPL_ID,
        max_steps_per_block=max_steps_per_block,
    )
    return x[0], counter["n"]


def assert_parity(reference_module, full, block_start, block_length, prompt_len,
                  script, *, steps=8, threshold=0.5, editing_threshold=0.0,
                  max_post_steps=16):
    ref_tokens, ref_fwd = run_reference_loop(
        reference_module, full, block_start, block_length, script,
        steps, threshold, editing_threshold, max_post_steps,
    )
    decoder = make_decoder(
        steps=steps,
        threshold=threshold,
        editing_threshold=editing_threshold,
        max_post_steps=max_post_steps,
    )
    flux_tokens, flux_iters, _ = run_flux_loop(
        decoder, full, block_start, block_length, prompt_len, script
    )
    assert flux_tokens.tolist() == ref_tokens.tolist()
    # FluxServe may spend one extra stability pass (budget-terminated blocks);
    # never more.
    assert ref_fwd <= flux_iters <= ref_fwd + 1
    return flux_tokens


# --------------------------------------------------------------- unit tests


class TestApplyEditOperations:
    def test_delete_shifts_and_pads(self):
        tokens, tracking = apply_edit_operations(
            [1, DEL_ID, 3, 4], [1, 2, 3, 4], [False, False, True, False],
            MASK_ID, 4, DEL_ID, SPL_ID,
        )
        assert tokens == [1, 3, 4, MASK_ID]
        assert tracking == [False, True, False, False]  # pad mask non-original

    def test_split_restores_old_token_and_truncates(self):
        tokens, tracking = apply_edit_operations(
            [1, SPL_ID, 3, 4], [1, 2, 3, 4], [False, True, False, False],
            MASK_ID, 4, DEL_ID, SPL_ID,
        )
        # SPLIT at 1 -> [mask, old_token 2]; block truncated to length 4.
        assert tokens == [1, MASK_ID, 2, 3]
        assert tracking == [False, False, True, False]

    def test_no_ops_is_identity(self):
        tokens, tracking = apply_edit_operations(
            [1, 2, 3, 4], [9, 9, 9, 9], [True, False, True, False],
            MASK_ID, 4, DEL_ID, SPL_ID,
        )
        assert tokens == [1, 2, 3, 4]
        assert tracking == [True, False, True, False]

    def test_delete_and_split_together(self):
        tokens, _ = apply_edit_operations(
            [DEL_ID, 2, SPL_ID, 4], [1, 2, 3, 4], [False] * 4,
            MASK_ID, 4, DEL_ID, SPL_ID,
        )
        # DEL drops pos 0; SPL expands pos 2 into [mask, 3].
        assert tokens == [2, MASK_ID, 3, 4]


class TestSchedule:
    def test_uniform_spread(self):
        # 10 masks over 4 steps -> 3, 3, 2, 2.
        assert [m2t_schedule_need(10, 4, s) for s in range(4)] == [3, 3, 2, 2]

    def test_after_steps_all_masks(self):
        assert m2t_schedule_need(10, 4, 4) == 10


class TestLoopBehavior:
    BLOCK = 8

    def test_plain_fill_terminates_without_edits(self):
        full = torch.tensor([1, 2] + [MASK_ID] * self.BLOCK)
        target = [3, 4, 5, 6, 7, 8, 9, 10]

        def script(block):
            cand = [t if t != MASK_ID else target[i] for i, t in enumerate(block)]
            return make_logits_row(cand, 0.95)

        decoder = make_decoder()
        out, iters, _ = run_flux_loop(decoder, full, 2, self.BLOCK, 2, script)
        assert out[2:].tolist() == target
        assert iters == 2

    def test_prompt_positions_protected(self):
        # Unaligned prompt: block starts at 0, prompt covers positions 0-1.
        full = torch.tensor([1, 2] + [MASK_ID] * 6)
        target = [1, 2, 5, 6, 7, 8, 9, 10]

        def script(block):
            cand = [9] * 2 + [
                t if t != MASK_ID else target[i + 2]
                for i, t in enumerate(block[2:])
            ]
            return make_logits_row(cand, 0.95)

        decoder = make_decoder()
        out, _, _ = run_flux_loop(decoder, full, 0, 8, 2, script)
        assert out[:2].tolist() == [1, 2]  # never rewritten to 9

    def test_oscillating_edit_terminates_via_budget(self):
        max_post_steps = 3
        full = torch.tensor([MASK_ID] * 4)
        base = [1, 2, 3, 4]

        def script(block):
            if MASK_ID in block:
                cand = [t if t != MASK_ID else base[i] for i, t in enumerate(block)]
            else:
                # rotate position 0 through distinct values forever
                cand = list(block)
                cand[0] = 5 + (cand[0] - 5 + 1) % 8 if cand[0] >= 5 else 5
            return make_logits_row(cand, 0.95)

        decoder = make_decoder(max_post_steps=max_post_steps, steps=4)
        out, iters, state = run_flux_loop(decoder, full, 0, 4, 0, script)
        assert (out != MASK_ID).all()
        assert bool(state.finalized[0])
        # fill + max_post_steps edits (final round is the last) + stable pass
        assert iters == 1 + max_post_steps + 1

    def test_fully_prompt_block_finishes_immediately(self):
        full = torch.tensor([1, 2, 3, 4])

        def script(block):
            return make_logits_row([9, 9, 9, 9], 0.99)

        decoder = make_decoder()
        out, iters, _ = run_flux_loop(decoder, full, 0, 4, 4, script)
        assert out.tolist() == [1, 2, 3, 4]
        assert iters == 1

    def test_step_cap_force_resolves(self):
        # Masks whose confidence never clears the threshold and whose
        # schedule floor is exhausted... the schedule floor always forces
        # progress, so instead pin an SPL/DEL churn: model always wants to
        # split position 0 after fill. max_steps_per_block force-finalizes.
        full = torch.tensor([MASK_ID] * 4)
        base = [1, 2, 3, 4]

        def script(block):
            if MASK_ID in block:
                cand = [t if t != MASK_ID else base[i] for i, t in enumerate(block)]
            else:
                cand = list(block)
                cand[0] = SPL_ID
            return make_logits_row(cand, 0.95)

        decoder = make_decoder(max_post_steps=50, max_steps_per_block=10, steps=4)
        out, iters, _ = run_flux_loop(decoder, full, 0, 4, 0, script)
        assert (out != MASK_ID).all()
        assert (out != SPL_ID).all() and (out != DEL_ID).all()
        assert iters <= decoder.max_block_iters


class TestFactory:
    def test_registered(self):
        cfg = RunnerConfig(
            parallel_decoding="levenshtein_joint",
            threshold=0.5,
            editing_threshold=0.0,
            block_length=32,
        )
        decoder = load_decoder(cfg)
        assert isinstance(decoder, LevenshteinJointDecoder)
        assert decoder.steps == 32  # steps=0 falls back to block_length
        assert decoder.delete_token_id == 156930
        assert decoder.split_token_id == 156931

    def test_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            make_decoder(temperature=0.7)


# --------------------------------------------------------------- parity


class TestReferenceParity:
    BLOCK = 8

    def test_plain_fill(self, reference_module):
        full = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8] + [MASK_ID] * self.BLOCK)
        target = [3, 4, 5, 6, 7, 8, 9, 10]

        def script(block):
            cand = [t if t != MASK_ID else target[i] for i, t in enumerate(block)]
            return make_logits_row(cand, 0.95)

        out = assert_parity(
            reference_module, full, self.BLOCK, self.BLOCK, self.BLOCK, script
        )
        assert out[self.BLOCK :].tolist() == target

    def test_gradual_fill_uses_schedule(self, reference_module):
        # Confidence below threshold: only the schedule floor moves masks, so
        # the trajectory exercises per-step num_need on both sides.
        full = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8] + [MASK_ID] * self.BLOCK)
        target = [3, 4, 5, 6, 7, 8, 9, 10]

        def script(block):
            cand = [t if t != MASK_ID else target[i] for i, t in enumerate(block)]
            probs = [0.30 + 0.01 * i for i in range(len(cand))]
            L = len(cand)
            prob_t = torch.tensor(probs, dtype=torch.float64)
            rest = (1.0 - prob_t) / (VOCAB - 1)
            logits = torch.log(rest).unsqueeze(-1).repeat(1, VOCAB)
            logits.scatter_(
                -1,
                torch.tensor(cand).unsqueeze(-1),
                torch.log(prob_t).unsqueeze(-1),
            )
            return logits.to(torch.float32)

        out = assert_parity(
            reference_module, full, self.BLOCK, self.BLOCK, self.BLOCK, script,
            steps=4, threshold=0.5,
        )
        assert out[self.BLOCK :].tolist() == target

    def test_t2t_edit(self, reference_module):
        full = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8] + [MASK_ID] * self.BLOCK)
        target = [3, 4, 5, 6, 7, 8, 9, 10]
        edited = [3, 4, 11, 6, 7, 8, 9, 10]

        def script(block):
            if block == target:
                cand = list(edited)
            elif MASK_ID in block:
                cand = [t if t != MASK_ID else target[i] for i, t in enumerate(block)]
            else:
                cand = list(block)
            return make_logits_row(cand, 0.95)

        out = assert_parity(
            reference_module, full, self.BLOCK, self.BLOCK, self.BLOCK, script,
            editing_threshold=0.5,
        )
        assert out[self.BLOCK :].tolist() == edited

    def test_delete_edit(self, reference_module):
        full = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8] + [MASK_ID] * self.BLOCK)
        target = [3, 4, 5, 6, 7, 8, 9, 10]
        after_delete = [3, 4, 6, 7, 8, 9, 10]  # position 2 removed

        def script(block):
            if block == target:
                cand = list(target)
                cand[2] = DEL_ID
            elif MASK_ID in block:
                # fill any mask (original or the delete-pad) deterministically
                cand = [
                    t if t != MASK_ID else (target[i] if block != after_delete + [MASK_ID] else 11)
                    for i, t in enumerate(block)
                ]
            else:
                cand = list(block)
            return make_logits_row(cand, 0.95)

        out = assert_parity(
            reference_module, full, self.BLOCK, self.BLOCK, self.BLOCK, script,
            editing_threshold=0.5,
        )
        assert out[self.BLOCK :].tolist() == after_delete + [11]

    def test_split_edit(self, reference_module):
        full = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8] + [MASK_ID] * self.BLOCK)
        target = [3, 4, 5, 6, 7, 8, 9, 10]
        # SPLIT at position 2: [3, 4, mask, 5, 6, 7, 8, 9] then mask -> 11.
        after_split_filled = [3, 4, 11, 5, 6, 7, 8, 9]

        def script(block):
            if block == target:
                cand = list(target)
                cand[2] = SPL_ID
            elif MASK_ID in block:
                filled = [3, 4, MASK_ID, 5, 6, 7, 8, 9]
                if block == filled:
                    cand = list(after_split_filled)
                else:
                    cand = [
                        t if t != MASK_ID else target[i] for i, t in enumerate(block)
                    ]
            else:
                cand = list(block)
            return make_logits_row(cand, 0.95)

        out = assert_parity(
            reference_module, full, self.BLOCK, self.BLOCK, self.BLOCK, script,
            editing_threshold=0.5,
        )
        assert out[self.BLOCK :].tolist() == after_split_filled

    def test_budget_termination_rotating_edits(self, reference_module):
        # Distinct states each iteration (no anti-loop), terminated by the
        # post-step budget on both sides.
        full = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8] + [MASK_ID] * self.BLOCK)
        base = [3, 4, 5, 6, 7, 8, 9, 10]

        def script(block):
            if MASK_ID in block:
                cand = [t if t != MASK_ID else base[i] for i, t in enumerate(block)]
            else:
                cand = list(block)
                cur = cand[0]
                cand[0] = 16 + ((cur - 16 + 1) % 4) if cur >= 16 else 16
            return make_logits_row(cand, 0.95)

        assert_parity(
            reference_module, full, self.BLOCK, self.BLOCK, self.BLOCK, script,
            max_post_steps=3, editing_threshold=0.5,
        )

    def test_max_post_steps_zero(self, reference_module):
        full = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8] + [MASK_ID] * self.BLOCK)
        target = [3, 4, 5, 6, 7, 8, 9, 10]

        def script(block):
            cand = [t if t != MASK_ID else target[i] for i, t in enumerate(block)]
            return make_logits_row(cand, 0.95)

        out = assert_parity(
            reference_module, full, self.BLOCK, self.BLOCK, self.BLOCK, script,
            max_post_steps=0,
        )
        assert out[self.BLOCK :].tolist() == target


@pytest.mark.parametrize('reserved', [DEL_ID, SPL_ID])
def test_literal_prompt_edit_tokens_survive_scan(reserved):
    decoder = make_decoder(max_post_steps=2)
    full = torch.tensor([reserved, 2, MASK_ID, MASK_ID])

    def script(block):
        # Attempt to rewrite the prompt as well as split generated content.
        return make_logits_row([DEL_ID, SPL_ID, 3, 4], 0.99)

    out, _, _ = run_flux_loop(decoder, full, 0, 4, 2, script)
    assert out.tolist() == [reserved, 2, 3, 4]


@pytest.mark.parametrize('op', [DEL_ID, SPL_ID])
def test_prompt_prefix_fixed_during_generated_edits(op):
    decoder = make_decoder(max_post_steps=3)
    full = torch.tensor([1, 2, MASK_ID, MASK_ID])
    calls = 0

    def script(block):
        nonlocal calls
        calls += 1
        candidates = [DEL_ID, SPL_ID, 3, 4]
        if calls == 2:
            candidates[2] = op
        return make_logits_row(candidates, 0.99)

    out, _, _ = run_flux_loop(decoder, full, 0, 4, 2, script)
    assert out[:2].tolist() == [1, 2]
    assert not any(t in (MASK_ID, DEL_ID, SPL_ID) for t in out[2:].tolist())


def test_reordered_subbatches_match_independent_rows_and_reset():
    decoder = make_decoder(max_post_steps=3, max_steps_per_block=8)
    state = decoder.make_row_state(5, 4, 'cpu')
    rows = {
        3: torch.tensor([1, 2, MASK_ID, MASK_ID, MASK_ID, MASK_ID, MASK_ID, MASK_ID]),
        1: torch.tensor([5, 6, 7, 8, MASK_ID, MASK_ID, MASK_ID, MASK_ID]),
    }
    starts = {3: 0, 1: 4}
    prompts = {3: 2, 1: 4}

    def script(block):
        if MASK_ID in block:
            return make_logits_row([3 if t == MASK_ID else t for t in block], .99)
        return make_logits_row([4 if t == 3 else 3 if t == 4 else t for t in block], .99)

    expected = {
        row: run_flux_loop(decoder, full, starts[row], 4, prompts[row], script)[0]
        for row, full in rows.items()
    }
    pending = [3, 1]
    for iteration in range(decoder.max_block_iters):
        selected = list(reversed(pending)) if iteration % 2 else pending[:1]
        if not selected:
            break
        x = FakeTokenArray(torch.stack([rows[r] for r in selected]))
        blocks = [rows[r][starts[r]:starts[r]+4].clone() for r in selected]
        decoder.batch_decode(
            torch.stack([script(b.tolist()) for b in blocks]),
            torch.tensor([starts[r] for r in selected]), x, 4,
            prompt_lengths=torch.tensor([prompts[r] for r in selected]),
            row_state=state, seq_ids=torch.tensor(selected),
        )
        for local, r in enumerate(selected):
            rows[r] = x.data[local].clone()
            after = rows[r][starts[r]:starts[r]+4]
            if not (blocks[local] == MASK_ID).any() and torch.equal(after, blocks[local]):
                pending.remove(r)
    assert not pending
    for r in rows:
        assert torch.equal(rows[r], expected[r])
    # Reuse slot 3 for its next block, retaining the other row's state.
    other_steps = state.step_id[1].clone()
    x = FakeTokenArray(rows[3].unsqueeze(0))
    decoder.batch_decode(
        script([MASK_ID]*4).unsqueeze(0), torch.tensor([4]), x, 4,
        prompt_lengths=torch.tensor([2]), row_state=state, seq_ids=torch.tensor([3]),
    )
    assert state.block_start[3] == 4
    assert state.step_id[3] == 1
    assert state.post_steps[3] == 0
    assert not state.finalized[3]
    assert state.step_id[1] == other_steps
    # The block-state history reset with the new block and holds only this
    # iteration's pre-edit state.
    assert state.seen_count[3] == 1
    assert len(state.seen_states(3)) == 1


def test_decisions_and_row_state_follow_source_rank(monkeypatch):
    """Replay rank-0 broadcasts into a rank with deliberately different logits."""
    import fluxserve.backend.execution.decoders.levenshtein as lev

    decoder = make_decoder(max_post_steps=4, max_steps_per_block=8)
    states = [decoder.make_row_state(1, 4, 'cpu') for _ in range(2)]
    tokens = [FakeTokenArray(torch.full((1, 4), MASK_ID)) for _ in range(2)]
    communications = []
    for step in range(6):
        communications.clear()
        source_logits = make_logits_row([3 if step % 2 == 0 else 4]*4, .99).unsqueeze(0)
        other_logits = make_logits_row([SPL_ID]*4, .20).unsqueeze(0)
        # Different second-choice candidates also exercise anti-loop resampling.
        for rank, logits in enumerate([source_logits, other_logits]):
            if rank == 0:
                monkeypatch.setattr(lev, 'broadcast_if_needed', lambda t: communications.append(t.clone()))
            else:
                def replay(t):
                    source = communications.pop(0)
                    assert source.shape == t.shape
                    t.copy_(source)
                monkeypatch.setattr(lev, 'broadcast_if_needed', replay)
            decoder.batch_decode(
                logits, torch.tensor([0]), tokens[rank], 4,
                prompt_lengths=torch.tensor([0]), row_state=states[rank], seq_ids=torch.tensor([0]),
            )
        assert not communications
        assert torch.equal(tokens[0].data, tokens[1].data)
        for field in ('is_original_mask', 'initial_mask_count', 'step_id', 'post_steps', 'finalized', 'block_start', 'seen_count'):
            assert torch.equal(getattr(states[0], field), getattr(states[1], field)), field
        assert torch.equal(states[0].seen_hashes, states[1].seen_hashes)


def test_debug_checks_accept_preserved_prompt_control_tokens(monkeypatch):
    monkeypatch.setenv('FLUXSERVE_DEBUG_LLADA22', '1')
    decoder = make_decoder(max_post_steps=0)
    out, _, _ = run_flux_loop(
        decoder, torch.tensor([DEL_ID, SPL_ID, MASK_ID, MASK_ID]), 0, 4, 2,
        lambda block: make_logits_row([1, 2, 3, 4], .99),
    )
    assert out.tolist() == [DEL_ID, SPL_ID, 3, 4]

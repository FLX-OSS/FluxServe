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

"""CPU unit tests for LLaDA2.1 joint threshold (M2T + T2T) decoding."""

import pytest
import torch

from fluxserve.backend.execution.decoders import (
    JointThresholdDecoder,
    load_decoder,
)
from fluxserve.backend.execution.decoders.joint_threshold import (
    joint_threshold_update,
)
from fluxserve.backend.execution.forward_batch_info import RunnerConfig
from fluxserve.backend.execution.runners.utils import DecodeEditBudget

VOCAB = 16
MASK_ID = 12
EOS_ID = 13


def make_logits(x_block, top_tokens, top_prob):
    """Logits whose per-position softmax gives ``top_tokens`` prob ``top_prob``.

    ``top_prob`` may be a scalar or a [B, L] tensor of probabilities in
    (1/VOCAB, 1). The remaining mass is spread over the other tokens.
    """
    B, L = x_block.shape
    top_prob = torch.as_tensor(top_prob, dtype=torch.float64).expand(B, L)
    rest = (1.0 - top_prob) / (VOCAB - 1)
    logits = torch.log(rest).unsqueeze(-1).repeat(1, 1, VOCAB)
    logits.scatter_(-1, top_tokens.unsqueeze(-1), torch.log(top_prob).unsqueeze(-1))
    return logits.to(torch.float32)


def no_prompt(B, L):
    return torch.zeros(B, L, dtype=torch.bool)


def all_edit(B):
    return torch.ones(B, dtype=torch.bool)


class TestJointThresholdUpdate:
    def test_high_confidence_m2t(self):
        x = torch.tensor([[MASK_ID, MASK_ID, 3, 4]])
        cand = torch.tensor([[5, 6, 3, 4]])
        logits = make_logits(x, cand, 0.95)
        out, m2t, t2t = joint_threshold_update(
            logits, x, MASK_ID, 0.7, 0.5, no_prompt(1, 4), all_edit(1)
        )
        assert out.tolist() == [[5, 6, 3, 4]]
        assert m2t.tolist() == [[True, True, False, False]]
        assert not t2t.any()

    def test_progress_guarantee_below_threshold(self):
        # No mask clears the threshold: the single highest-confidence masked
        # position must still transfer (the clamp path).
        x = torch.tensor([[MASK_ID, MASK_ID, MASK_ID, 4]])
        cand = torch.tensor([[5, 6, 7, 4]])
        probs = torch.tensor([[0.30, 0.40, 0.35, 0.90]])
        logits = make_logits(x, cand, probs)
        out, m2t, _ = joint_threshold_update(
            logits, x, MASK_ID, 0.7, 0.5, no_prompt(1, 4), all_edit(1)
        )
        assert m2t.tolist() == [[False, True, False, False]]
        assert out.tolist() == [[MASK_ID, 6, MASK_ID, 4]]

    def test_t2t_replacement_above_editing_threshold(self):
        x = torch.tensor([[1, 2, 3, 4]])
        cand = torch.tensor([[1, 9, 3, 4]])  # position 1 wants to change
        logits = make_logits(x, cand, 0.8)
        out, m2t, t2t = joint_threshold_update(
            logits, x, MASK_ID, 0.7, 0.5, no_prompt(1, 4), all_edit(1)
        )
        assert not m2t.any()
        assert t2t.tolist() == [[False, True, False, False]]
        assert out.tolist() == [[1, 9, 3, 4]]

    def test_t2t_blocked_below_editing_threshold(self):
        x = torch.tensor([[1, 2, 3, 4]])
        cand = torch.tensor([[1, 9, 3, 4]])
        logits = make_logits(x, cand, 0.4)
        out, _, t2t = joint_threshold_update(
            logits, x, MASK_ID, 0.7, 0.5, no_prompt(1, 4), all_edit(1)
        )
        assert not t2t.any()
        assert out.tolist() == x.tolist()

    def test_unchanged_candidates_are_not_edits(self):
        # Candidate equals current token everywhere: no T2T transfer even at
        # editing_threshold=0, so the stability predicate can fire.
        x = torch.tensor([[1, 2, 3, 4]])
        logits = make_logits(x, x, 0.99)
        out, m2t, t2t = joint_threshold_update(
            logits, x, MASK_ID, 0.7, 0.0, no_prompt(1, 4), all_edit(1)
        )
        assert not m2t.any()
        assert not t2t.any()
        assert out.tolist() == x.tolist()

    def test_mask_id_never_written(self):
        # The unsuppressed argmax IS mask_id at every position, for both a
        # masked position and an editable resolved position. Neither path may
        # write mask_id.
        x = torch.tensor([[MASK_ID, 2, 3, 4]])
        cand = torch.full_like(x, MASK_ID)
        logits = make_logits(x, cand, 0.99)
        out, _, _ = joint_threshold_update(
            logits, x, MASK_ID, 0.0, 0.0, no_prompt(1, 4), all_edit(1)
        )
        # The masked position resolves to the runner-up token, never mask_id.
        assert out[0, 0] != MASK_ID
        # Resolved positions keep or change value but never become masks.
        assert (out[0, 1:] != MASK_ID).all()

    def test_prompt_positions_never_edited(self):
        # Unaligned prompt: first two positions of the block are prompt.
        x = torch.tensor([[1, 2, MASK_ID, 4]])
        cand = torch.tensor([[9, 9, 9, 9]])
        logits = make_logits(x, cand, 0.99)
        prompt = torch.tensor([[True, True, False, False]])
        out, _, t2t = joint_threshold_update(
            logits, x, MASK_ID, 0.7, 0.0, prompt, all_edit(1)
        )
        assert out[0, 0] == 1 and out[0, 1] == 2
        assert out[0, 2] == 9  # M2T still fills the mask
        assert out[0, 3] == 9  # non-prompt resolved token is editable
        assert not t2t[0, :2].any()

    def test_rows_complete_independently(self):
        # Row 0 has masks, row 1 is fully resolved with agreeing candidates.
        # Row 1 must not be forced to update by row 0's progress guarantee.
        x = torch.tensor([[MASK_ID, 2], [3, 4]])
        cand = torch.tensor([[5, 2], [3, 4]])
        probs = torch.tensor([[0.30, 0.90], [0.90, 0.90]])
        logits = make_logits(x, cand, probs)
        out, m2t, t2t = joint_threshold_update(
            logits, x, MASK_ID, 0.7, 0.5, no_prompt(2, 2), all_edit(2)
        )
        assert out[0].tolist() == [5, 2]
        assert out[1].tolist() == [3, 4]
        assert not m2t[1].any() and not t2t[1].any()

    def test_allow_edit_false_disables_t2t_keeps_m2t(self):
        x = torch.tensor([[MASK_ID, 2, 3, 4]])
        cand = torch.tensor([[5, 9, 9, 9]])
        logits = make_logits(x, cand, 0.99)
        allow = torch.tensor([False])
        out, m2t, t2t = joint_threshold_update(
            logits, x, MASK_ID, 0.7, 0.0, no_prompt(1, 4), allow
        )
        assert out.tolist() == [[5, 2, 3, 4]]
        assert m2t[0, 0]
        assert not t2t.any()

    def test_deterministic(self):
        x = torch.tensor([[MASK_ID, 2, 3, 4]])
        cand = torch.tensor([[5, 9, 3, 4]])
        logits = make_logits(x, cand, 0.9)
        args = (x, MASK_ID, 0.7, 0.5, no_prompt(1, 4), all_edit(1))
        out1, _, _ = joint_threshold_update(logits.clone(), *args)
        out2, _, _ = joint_threshold_update(logits.clone(), *args)
        assert torch.equal(out1, out2)


class FakeTokenArray:
    def __init__(self, data):
        self.data = data


class TestBatchDecode:
    def test_batch_decode_mutates_in_place(self):
        block_length = 4
        data = torch.tensor(
            [
                [1, 2, MASK_ID, MASK_ID, MASK_ID, MASK_ID, 0, 0],
                [3, 4, 5, 6, MASK_ID, MASK_ID, MASK_ID, MASK_ID],
            ]
        )
        x = FakeTokenArray(data.clone())
        block_start = torch.tensor([0, 4])
        prompt_lengths = torch.tensor([2, 4])
        blocks = torch.stack(
            [data[0, 0:4], data[1, 4:8]]
        )
        cand = torch.tensor([[9, 9, 7, 8], [9, 9, 7, 8]])
        logits = make_logits(blocks, cand, 0.95)
        decoder = JointThresholdDecoder(
            threshold=0.7,
            editing_threshold=0.5,
            temperature=0,
            mask_id=MASK_ID,
            eos_id=EOS_ID,
        )
        decoder.batch_decode(
            logits,
            block_start,
            x,
            block_length,
            prompt_lengths=prompt_lengths,
        )
        # Row 0: positions 0-1 are prompt (unchanged); masks filled.
        assert x.data[0, :4].tolist() == [1, 2, 7, 8]
        # Row 1: active block is 4:8, all masks filled with candidates.
        assert x.data[1, 4:8].tolist() == [9, 9, 7, 8]
        # Outside the active block nothing changes.
        assert x.data[1, :4].tolist() == [3, 4, 5, 6]

    def test_batch_decode_requires_prompt_lengths(self):
        decoder = JointThresholdDecoder(
            threshold=0.7,
            editing_threshold=0.5,
            temperature=0,
            mask_id=MASK_ID,
            eos_id=EOS_ID,
        )
        x = FakeTokenArray(torch.full((1, 4), MASK_ID))
        logits = torch.zeros(1, 4, VOCAB)
        with pytest.raises(ValueError, match="prompt_lengths"):
            decoder.batch_decode(logits, torch.tensor([0]), x, 4)

    def test_temperature_nonzero_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            JointThresholdDecoder(
                threshold=0.7,
                editing_threshold=0.5,
                temperature=0.5,
                mask_id=MASK_ID,
                eos_id=EOS_ID,
            )


class TestFactory:
    def test_joint_threshold_registered(self):
        config = RunnerConfig(
            parallel_decoding="joint_threshold",
            threshold=0.7,
            editing_threshold=0.5,
        )
        decoder = load_decoder(config)
        assert isinstance(decoder, JointThresholdDecoder)
        assert decoder.threshold == 0.7
        assert decoder.editing_threshold == 0.5

    def test_unknown_decoder_rejected(self):
        config = RunnerConfig(parallel_decoding="threshold")
        config.parallel_decoding = "no_such_decoder"
        with pytest.raises(ValueError, match="no_such_decoder"):
            load_decoder(config)

    def test_num_to_transfer_above_one_rejected(self):
        config = RunnerConfig(
            parallel_decoding="joint_threshold",
            threshold=0.7,
        )
        config.num_to_transfer = 2
        with pytest.raises(ValueError, match="num_to_transfer"):
            load_decoder(config)


class TestRunnerConfigValidation:
    def test_editing_threshold_range(self):
        with pytest.raises(ValueError, match="editing_threshold"):
            RunnerConfig(editing_threshold=1.5)

    def test_max_post_steps_non_negative(self):
        with pytest.raises(ValueError, match="max_post_steps"):
            RunnerConfig(max_post_steps=-1)

    def test_joint_threshold_threshold_range(self):
        with pytest.raises(ValueError, match="threshold"):
            RunnerConfig(parallel_decoding="joint_threshold", threshold=1.2)


class TestBlockLoopSimulation:
    """Simulate the runner's per-block decode loop without a model.

    Mirrors the `_decode_selected_batch` sequence: gather pre-update block,
    decode with allow_edit from the budget, re-gather, apply the
    `(~had_mask) & (~changed)` completion predicate, update the budget.
    """

    BLOCK = 4

    def _run_loop(self, data, prompt_lengths, logits_fn, max_post_steps):
        decoder = JointThresholdDecoder(
            threshold=0.7,
            editing_threshold=0.5,
            temperature=0,
            mask_id=MASK_ID,
            eos_id=EOS_ID,
        )
        x = FakeTokenArray(data.clone())
        B = data.shape[0]
        block_start = torch.zeros(B, dtype=torch.long)
        seq_ids = torch.arange(B)
        budget = DecodeEditBudget(B, max_post_steps, self.BLOCK, "cpu")
        offsets = torch.arange(self.BLOCK).unsqueeze(0) + block_start.unsqueeze(1)
        iters = 0
        while True:
            decoding_block = torch.gather(x.data, 1, offsets)
            logits = logits_fn(decoding_block, iters)
            decoder.batch_decode(
                logits,
                block_start,
                x,
                self.BLOCK,
                prompt_lengths=prompt_lengths,
                allow_edit=budget.allow_edit(seq_ids),
            )
            after = torch.gather(x.data, 1, offsets)
            had_mask = (decoding_block == MASK_ID).any(dim=1)
            changed = (after != decoding_block).any(dim=1)
            block_finished = (~had_mask) & (~changed)
            budget.update(seq_ids, had_mask, changed, block_finished)
            iters += 1
            if bool(block_finished.all()):
                # KV-commit invariant: the finishing iteration's pre-update
                # tokens are the final tokens.
                assert torch.equal(decoding_block, after)
                return iters, x.data
            assert iters < 100, "loop did not terminate"

    def test_stable_model_finishes_in_two_iterations(self):
        data = torch.tensor([[MASK_ID, MASK_ID, MASK_ID, MASK_ID]])
        target = torch.tensor([[1, 2, 3, 4]])

        def logits_fn(block, _):
            cand = torch.where(block == MASK_ID, target, block)
            return make_logits(block, cand, 0.95)

        iters, out = self._run_loop(data, torch.tensor([0]), logits_fn, 16)
        assert out[:, : self.BLOCK].tolist() == target.tolist()
        assert iters == 2  # one fill pass + one stable pass

    def test_oscillating_edit_terminates_via_budget(self):
        # After the masks resolve, the model flips position 0 between two
        # tokens forever. The budget must clear allow_edit and finish.
        max_post_steps = 3
        data = torch.tensor([[MASK_ID, MASK_ID, MASK_ID, MASK_ID]])
        base = torch.tensor([[1, 2, 3, 4]])

        def logits_fn(block, it):
            cand = torch.where(block == MASK_ID, base, block)
            if not (block == MASK_ID).any():
                cand = cand.clone()
                cand[0, 0] = 5 if block[0, 0] != 5 else 6
            return make_logits(block, cand, 0.95)

        iters, out = self._run_loop(data, torch.tensor([0]), logits_fn, max_post_steps)
        # 1 fill + max_post_steps edit iterations + 1 stable pass.
        assert iters == 1 + max_post_steps + 1
        assert (out[:, : self.BLOCK] != MASK_ID).all()

    def test_rows_finish_independently(self):
        # Row 0 resolves immediately; row 1 keeps editing for a while. Row 0
        # must be finished (and unchanged) from iteration 2 onward.
        data = torch.tensor(
            [
                [MASK_ID, MASK_ID, MASK_ID, MASK_ID],
                [MASK_ID, MASK_ID, MASK_ID, MASK_ID],
            ]
        )
        base = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        row0_after_fill = {}

        def logits_fn(block, it):
            cand = torch.where(block == MASK_ID, base, block)
            if not (block[1] == MASK_ID).any() and it < 4:
                cand = cand.clone()
                cand[1, 0] = 9 if block[1, 0] != 9 else 5
            if it >= 1:
                row0_after_fill.setdefault(it, block[0].clone())
            return make_logits(block, cand, 0.95)

        iters, out = self._run_loop(data, torch.tensor([0, 0]), logits_fn, 16)
        assert out[0, : self.BLOCK].tolist() == [1, 2, 3, 4]
        # Row 0's block never changed after its fill pass.
        states = list(row0_after_fill.values())
        assert all(torch.equal(states[0], s) for s in states)


class TestDecodeEditBudget:
    def test_post_steps_only_count_mask_free_changes(self):
        budget = DecodeEditBudget(4, 2, 8, "cpu")
        ids = torch.tensor([0, 1, 2])
        had_mask = torch.tensor([True, False, False])
        changed = torch.tensor([True, True, False])
        finished = torch.tensor([False, False, True])
        budget.update(ids, had_mask, changed, finished)
        assert budget.post_steps.tolist() == [0, 1, 0, 0]

    def test_finished_rows_reset(self):
        budget = DecodeEditBudget(2, 2, 8, "cpu")
        ids = torch.tensor([0])
        budget.update(
            ids,
            torch.tensor([False]),
            torch.tensor([True]),
            torch.tensor([False]),
        )
        assert budget.post_steps.tolist() == [1, 0]
        budget.update(
            ids,
            torch.tensor([False]),
            torch.tensor([False]),
            torch.tensor([True]),
        )
        assert budget.post_steps.tolist() == [0, 0]
        assert budget.block_iters.tolist() == [0, 0]

    def test_allow_edit_flips_at_budget(self):
        budget = DecodeEditBudget(1, 2, 8, "cpu")
        ids = torch.tensor([0])
        assert budget.allow_edit(ids).tolist() == [True]
        for _ in range(2):
            budget.update(
                ids,
                torch.tensor([False]),
                torch.tensor([True]),
                torch.tensor([False]),
            )
        assert budget.allow_edit(ids).tolist() == [False]

    def test_stuck_row_raises(self):
        budget = DecodeEditBudget(1, 1, 2, "cpu")
        ids = torch.tensor([0])
        with pytest.raises(RuntimeError, match="did not finish"):
            for _ in range(budget.max_block_iters + 1):
                budget.update(
                    ids,
                    torch.tensor([True]),
                    torch.tensor([False]),
                    torch.tensor([False]),
                )

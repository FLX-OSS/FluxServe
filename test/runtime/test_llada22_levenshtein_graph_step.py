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

"""LLaDA2.2 fused decode-graph tail.

The decode CUDA graph captures the levenshtein_joint iteration tail together
with the model forward, so the tail must be a fixed-shape tensor program that
(a) agrees exactly with the eager ``batch_decode`` path, (b) leaves padding
rows inert, and (c) carries its per-row state in and out through buffers
rather than mutating runner-owned tensors inside the capture.

These are CPU tests of that contract. Actual graph capture, replay and TP4/EP4
collectives still need GPU validation.
"""

import itertools
import random

import pytest
import torch

from fluxserve.backend.execution.decoders.levenshtein import (
    GRAPH_STATE_FIELDS,
    LevenshteinJointDecoder,
    apply_edit_operations,
    apply_edit_operations_batched,
    block_seen,
)

VOCAB = 20
MASK_ID = 12
EOS_ID = 13
DEL_ID = 14
SPL_ID = 15

STATE_FIELDS = ("is_original_mask", "initial_mask_count", "step_id",
                "post_steps", "finalized", "block_start", "seen_count",
                "seen_blocks")


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
        max_steps_per_block=20,
    )
    kwargs.update(overrides)
    return LevenshteinJointDecoder(**kwargs)


class FakeTokenArray:
    def __init__(self, data):
        self.data = data


def random_logits(rng, batch, block_length):
    """Logits with enough spread that thresholds and top-3 both matter."""
    logits = torch.zeros(batch, block_length, VOCAB)
    for b, i in itertools.product(range(batch), range(block_length)):
        for v in range(VOCAB):
            logits[b, i, v] = rng.uniform(-6.0, 6.0)
    return logits


def eager_step(decoder, state, x_full, block_start, prompt_lengths, logits,
               seq_ids, block_length):
    """One eager batch_decode iteration plus the runner's block predicates."""
    x = FakeTokenArray(x_full)
    offsets = torch.arange(block_length).unsqueeze(0) + block_start.unsqueeze(1)
    before = torch.gather(x.data, 1, offsets)
    decoder.batch_decode(
        logits, block_start, x, block_length,
        prompt_lengths=prompt_lengths, row_state=state, seq_ids=seq_ids,
    )
    after = torch.gather(x.data, 1, offsets)
    had_mask = (before == decoder.mask_id).any(dim=1)
    changed = (after != before).any(dim=1)
    return after, (had_mask, changed, (~had_mask) & (~changed))


def fused_step(decoder, state, x_block, block_start, prompt_positions, logits,
               seq_ids, block_length):
    """The same iteration through the fused-graph entry points."""
    extra = decoder.graph_inputs(
        state, seq_ids, block_start, x_block, block_length
    )
    allow_edit = torch.ones(x_block.shape[0], dtype=torch.bool)
    new_block, had_mask, changed, finished, outputs = decoder.graph_step(
        logits, x_block, prompt_positions, allow_edit, **extra
    )
    decoder.commit_row_state(state, seq_ids, outputs)
    return new_block, (had_mask, changed, finished)


# --------------------------------------------------- batched edit operations


class TestApplyEditOperationsBatched:
    """The vectorized edit scan must match the scalar reference port."""

    def scalar(self, blk, old, orig, prompt_len, block_length):
        return apply_edit_operations(
            blk.tolist(), old.tolist(), orig.tolist(), MASK_ID, block_length,
            DEL_ID, SPL_ID, prompt_length=prompt_len,
        )

    @pytest.mark.parametrize("seed", range(12))
    def test_matches_scalar_oracle_on_random_blocks(self, seed):
        rng = random.Random(seed)
        block_length = 6
        batch = 4
        blk = torch.empty(batch, block_length, dtype=torch.long)
        old = torch.empty(batch, block_length, dtype=torch.long)
        orig = torch.zeros(batch, block_length, dtype=torch.bool)
        prompt_lens = []
        for b in range(batch):
            prompt_lens.append(rng.randint(0, 2))
            for i in range(block_length):
                # Heavily weighted towards edit tokens so truncation, padding
                # and shifting all occur across the sweep.
                blk[b, i] = rng.choice(
                    [DEL_ID, SPL_ID, MASK_ID, 1, 2, 3, DEL_ID, SPL_ID]
                )
                old[b, i] = rng.choice([MASK_ID, 1, 2, 3])
                orig[b, i] = rng.random() < 0.4
        prompt_positions = (
            torch.arange(block_length).unsqueeze(0)
            < torch.tensor(prompt_lens).unsqueeze(1)
        )

        tokens, tracking = apply_edit_operations_batched(
            blk, old, orig, prompt_positions, MASK_ID, DEL_ID, SPL_ID
        )
        for b in range(batch):
            want_tok, want_trk = self.scalar(
                blk[b], old[b], orig[b], prompt_lens[b], block_length
            )
            assert tokens[b].tolist() == want_tok, b
            assert tracking[b].tolist() == want_trk, b

    def test_all_delete_pads_with_masks(self):
        blk = torch.full((1, 4), DEL_ID)
        tokens, tracking = apply_edit_operations_batched(
            blk, blk, torch.ones(1, 4, dtype=torch.bool),
            torch.zeros(1, 4, dtype=torch.bool), MASK_ID, DEL_ID, SPL_ID,
        )
        assert tokens[0].tolist() == [MASK_ID] * 4
        assert tracking[0].tolist() == [False] * 4

    def test_split_truncates_at_block_length(self):
        blk = torch.tensor([[SPL_ID, SPL_ID, 7, 8]])
        old = torch.tensor([[1, 2, 7, 8]])
        tokens, _ = apply_edit_operations_batched(
            blk, old, torch.zeros(1, 4, dtype=torch.bool),
            torch.zeros(1, 4, dtype=torch.bool), MASK_ID, DEL_ID, SPL_ID,
        )
        assert tokens[0].tolist() == [MASK_ID, 1, MASK_ID, 2]

    def test_prompt_prefix_keeps_literal_edit_tokens(self):
        blk = torch.tensor([[DEL_ID, SPL_ID, DEL_ID, 5]])
        prompt_positions = torch.tensor([[True, True, False, False]])
        tokens, tracking = apply_edit_operations_batched(
            blk, blk, torch.ones(1, 4, dtype=torch.bool), prompt_positions,
            MASK_ID, DEL_ID, SPL_ID,
        )
        assert tokens[0, :2].tolist() == [DEL_ID, SPL_ID]
        assert tracking[0, :2].tolist() == [False, False]
        # Only the generated DELETE is consumed.
        assert tokens[0, 2:].tolist() == [5, MASK_ID]


# ------------------------------------------------------------ exact history


class TestBlockHistory:
    def test_exact_lookup_rejects_colliding_weighted_sums(self):
        original = torch.full((1, 32), 100, dtype=torch.long)
        different = original.clone()
        different[0, :4] = torch.tensor([101, 99, 99, 101])
        history = original.unsqueeze(1).clone()
        assert not block_seen(different, history, torch.tensor([1])).item()
        assert block_seen(original, history, torch.tensor([1])).item()
        assert not block_seen(original, history, torch.tensor([0])).item()

    def test_history_is_row_local(self):
        blocks = torch.tensor([[1, 2], [3, 4]])
        history = torch.tensor([[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
        assert block_seen(blocks, history, torch.tensor([1, 1])).tolist() == [True, False]


@pytest.mark.parametrize("seed", range(8))
def test_graph_step_matches_batch_decode(seed):
    """The captured tail must reproduce the eager path token for token.

    Both paths are driven over a multi-iteration block so the row state --
    schedule position, post-step budget, tracking, finalization and the
    anti-loop history -- has to agree at every step, not just on the first.
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)
    block_length, batch = 6, 3
    eager_decoder = make_decoder()
    fused_decoder = make_decoder()
    eager_state = eager_decoder.make_row_state(batch, block_length, "cpu")
    fused_state = fused_decoder.make_row_state(batch, block_length, "cpu")

    prompt_lengths = torch.tensor([0, 2, 1])
    block_start = torch.zeros(batch, dtype=torch.long)
    seq_ids = torch.arange(batch)
    offsets = torch.arange(block_length).unsqueeze(0) + block_start.unsqueeze(1)
    prompt_positions = offsets < prompt_lengths.unsqueeze(1)

    x_full = torch.full((batch, block_length), MASK_ID, dtype=torch.long)
    x_full[1, :2] = torch.tensor([3, 4])
    x_full[2, :1] = torch.tensor([5])
    eager_tokens = x_full.clone()
    fused_block = x_full.clone()

    for _ in range(10):
        logits = random_logits(rng, batch, block_length)
        eager_after, eager_pred = eager_step(
            eager_decoder, eager_state, eager_tokens, block_start,
            prompt_lengths, logits.clone(), seq_ids, block_length,
        )
        fused_after, fused_pred = fused_step(
            fused_decoder, fused_state, fused_block, block_start,
            prompt_positions, logits.clone(), seq_ids, block_length,
        )
        assert torch.equal(eager_after, fused_after)
        for got, want in zip(fused_pred, eager_pred, strict=True):
            assert torch.equal(got, want)
        for field in STATE_FIELDS:
            assert torch.equal(
                getattr(eager_state, field), getattr(fused_state, field)
            ), field
        eager_tokens = eager_after.clone()
        fused_block = fused_after.clone()


def test_graph_inputs_perform_new_block_reset():
    """The block-entry reset is data-dependent, so it must run before replay."""
    decoder = make_decoder()
    state = decoder.make_row_state(1, 4, "cpu")
    seq_ids = torch.tensor([0])
    state.step_id[0] = 7
    state.post_steps[0] = 3
    state.seen_count[0] = 5
    state.block_start[0] = 0

    x_block = torch.tensor([[MASK_ID, MASK_ID, 1, 2]])
    extra = decoder.graph_inputs(state, seq_ids, torch.tensor([4]), x_block, 4)

    assert int(state.block_start[0]) == 4
    assert int(state.step_id[0]) == 0
    assert int(state.post_steps[0]) == 0
    assert int(state.seen_count[0]) == 0
    assert int(state.initial_mask_count[0]) == 2
    assert extra["initial_mask_count"].tolist() == [2]
    assert extra["step_id"].tolist() == [0]
    assert set(extra) == set(GRAPH_STATE_FIELDS)


# ------------------------------------------------------------ padding rows


def test_graph_extra_buffers_are_inert_for_padding_rows():
    """Padding rows carry finalized=True, which is what makes them harmless."""
    decoder = make_decoder()
    buffers = decoder.graph_extra_buffers(4, 6, "cpu")
    assert set(buffers) == set(GRAPH_STATE_FIELDS)
    assert bool(buffers["finalized"].all())
    assert buffers["is_original_mask"].shape == (4, 6)
    assert buffers["seen_blocks"].shape == (4, decoder.max_block_iters, 6)

    padding = decoder.graph_extra_padding()
    assert set(padding) == set(GRAPH_STATE_FIELDS)
    assert padding["finalized"] is True
    assert all(padding[f] == 0 for f in GRAPH_STATE_FIELDS if f != "finalized")


def replay_padded(decoder, state, x_block, block_start, prompt_positions,
                  logits, seq_ids, block_length, bucket):
    """Simulate FlashInferCudaGraphRunner.replay_decode for one iteration.

    Mirrors the real sequence: the row-state inputs are gathered for the real
    rows only, the static buffers are reset to the decoder's padding values
    and overwritten in their leading rows, the step runs at the captured
    bucket size, and only the leading rows are committed.
    """
    rows = x_block.shape[0]
    extra = decoder.graph_inputs(
        state, seq_ids, block_start, x_block, block_length
    )
    buffers = decoder.graph_extra_buffers(bucket, block_length, "cpu")
    padding = decoder.graph_extra_padding()
    for name, buffer in buffers.items():
        buffer.fill_(padding[name])
        buffer[:rows].copy_(extra[name])

    padded_block = torch.full(
        (bucket, block_length), decoder.mask_id, dtype=torch.long
    )
    padded_block[:rows] = x_block
    padded_prompt = torch.ones(bucket, block_length, dtype=torch.bool)
    padded_prompt[:rows] = prompt_positions
    padded_logits = torch.zeros(bucket, block_length, VOCAB)
    padded_logits[:rows] = logits
    allow_edit = torch.zeros(bucket, dtype=torch.bool)

    new_block, had_mask, changed, finished, outputs = decoder.graph_step(
        padded_logits, padded_block, padded_prompt, allow_edit, **buffers
    )
    decoder.commit_row_state(
        state, seq_ids, {k: v[:rows] for k, v in outputs.items()}
    )
    return (
        new_block,
        (had_mask, changed, finished),
        {k: v[rows:] for k, v in outputs.items()},
    )


def test_padding_rows_neither_write_nor_disturb_real_rows():
    """A padded batch must produce the real row's unpadded result exactly.

    This is the property the graph runner relies on when it rounds a batch up
    to a captured bucket size.
    """
    torch.manual_seed(0)
    rng = random.Random(0)
    block_length = 5
    solo = make_decoder()
    padded = make_decoder()
    solo_state = solo.make_row_state(1, block_length, "cpu")
    padded_state = padded.make_row_state(1, block_length, "cpu")

    real_block = torch.full((1, block_length), MASK_ID, dtype=torch.long)
    padded_block = real_block.clone()
    prompt_positions = torch.zeros(1, block_length, dtype=torch.bool)
    block_start = torch.zeros(1, dtype=torch.long)
    seq_ids = torch.arange(1)

    for _ in range(6):
        logits = random_logits(rng, 1, block_length)
        solo_out, solo_pred = fused_step(
            solo, solo_state, real_block, block_start, prompt_positions,
            logits.clone(), seq_ids, block_length,
        )
        padded_out, padded_pred, pad_state = replay_padded(
            padded, padded_state, padded_block, block_start, prompt_positions,
            logits.clone(), seq_ids, block_length, bucket=4,
        )

        assert torch.equal(padded_out[:1], solo_out)
        for got, want in zip(padded_pred, solo_pred, strict=True):
            assert torch.equal(got[:1], want)
        # Padding rows stay masked and advance no counter.
        assert torch.equal(
            padded_out[1:], torch.full((3, block_length), MASK_ID)
        )
        assert bool(pad_state["finalized"].all())
        assert bool((pad_state["step_id"] == 0).all())
        assert bool((pad_state["post_steps"] == 0).all())
        assert not bool(pad_state["history_append"].any())
        for field in STATE_FIELDS:
            assert torch.equal(
                getattr(solo_state, field), getattr(padded_state, field)
            ), field

        real_block = solo_out.clone()
        padded_block = padded_out[:1].clone()


def test_decoder_declares_the_fused_graph_contract():
    decoder = make_decoder()
    assert decoder.graph_fused_step is True
    assert decoder.graph_extra_state is True
    # The history must be able to hold one entry per iteration the runner
    # will allow for a block, otherwise the anti-loop check silently degrades.
    assert decoder.max_block_iters == decoder.max_steps_per_block + 1
    state = decoder.make_row_state(2, 4, "cpu")
    assert state.history == decoder.max_block_iters


@pytest.mark.parametrize("final_round", [False, True])
def test_greedy_and_final_round_ties_match_argmax(final_round):
    decoder = make_decoder(steps=1, max_post_steps=0 if final_round else 16)
    block = torch.full((1, 4), MASK_ID)
    state = decoder.make_row_state(1, 4, "cpu")
    logits = torch.full((1, 4, VOCAB), -10.)
    logits[..., 0] = logits[..., 1] = 10.
    if final_round:
        logits[..., DEL_ID] = 12.
        logits[..., SPL_ID] = 11.
    after, _ = fused_step(
        decoder, state, block, torch.tensor([0]), torch.zeros_like(block, dtype=torch.bool),
        logits, torch.tensor([0]), 4,
    )
    assert after.tolist() == [[0, 0, 0, 0]]


def test_escape_alternative_ties_match_argmax():
    decoder = make_decoder()
    state = decoder.make_row_state(1, 4, "cpu")
    ids, starts = torch.tensor([0]), torch.tensor([0])
    decoder.graph_inputs(state, ids, starts, torch.full((1, 4), MASK_ID), 4)
    state.seen_blocks[0, 0] = 0
    state.seen_count[0] = 1
    logits = torch.full((1, 4, VOCAB), -10.)
    logits[..., 0] = 12.
    logits[..., 2] = logits[..., 3] = 10.
    after, _ = fused_step(
        decoder, state, torch.ones((1, 4), dtype=torch.long), starts,
        torch.zeros((1, 4), dtype=torch.bool), logits, ids, 4,
    )
    assert after.tolist() == [[2, 0, 0, 0]]


def test_colliding_history_does_not_trigger_escape():
    decoder = make_decoder(mask_id=12, delete_token_id=14, split_token_id=15)
    state = decoder.make_row_state(1, 32, "cpu")
    ids, starts = torch.tensor([0]), torch.tensor([0])
    masks = torch.full((1, 32), MASK_ID)
    decoder.graph_inputs(state, ids, starts, masks, 32)
    state.seen_blocks[0, 0] = 100
    state.seen_count[0] = 1
    candidates = torch.full((1, 32), 100)
    candidates[0, :4] = torch.tensor([101, 99, 99, 101])
    logits = torch.full((1, 32, 128), -10.)
    logits.scatter_(-1, candidates.unsqueeze(-1), 10.)
    after, _ = fused_step(
        decoder, state, masks, starts, torch.zeros_like(masks, dtype=torch.bool),
        logits, ids, 32,
    )
    assert torch.equal(after, candidates)
    assert torch.equal(state.seen_blocks[0, 1], candidates[0])

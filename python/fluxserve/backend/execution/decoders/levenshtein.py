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

"""LLaDA2.2 Levenshtein joint decoding (M2T + T2T + DELETE/SPLIT edit ops).

Reference: ``LLaDA2MoeModelLM._joint_decode_block`` in the
``inclusionAI/LLaDA2.2-flash`` checkpoint's ``modeling_llada2_moe.py``.
Temperature 0 only. Documented deviations from the reference:

- the ``mask_id`` logit is suppressed before every argmax (same deliberate
  deviation as the 2.1 joint decoder; the reference does not enforce
  no-remask on writes);
- the anti-loop escape picks the *lowest-confidence* changed position instead
  of a random one, so it is deterministic and rank-consistent (the reference
  uses ``torch.randint``; at temperature 0 the replacement token itself is
  the same greedy argmax in both);
- a row that hits ``max_steps_per_block`` is force-resolved like a final
  round instead of terminating with residual masks (the reference warns and
  returns masks, which a serving engine cannot stream);
- literal DELETE/SPLIT tokens inside the protected prompt prefix are kept
  unchanged, while the reference warns and consumes them.

The block stays fixed-length throughout (DELETE shifts left and pads masks;
SPLIT expands in place and truncates), so the runner-visible sequence length,
KV pages, and scheduling are untouched.

The whole per-iteration tail is one fixed-shape tensor program
(:func:`levenshtein_graph_step`) with no data-dependent control flow, no
host synchronisation, and no collectives. Both entry points use it: eager
``batch_decode`` and, via :meth:`LevenshteinJointDecoder.graph_step`, the
decode CUDA graph, which captures it together with the model forward and the
lm_head. Rank consistency comes from broadcasting the step's *outputs*
(tokens plus every row-state field) before they are committed, which is
strictly stronger than broadcasting intermediate decisions: whatever the
ranks computed locally, the committed state is the source rank's.
"""

import hashlib
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from fluxserve.backend.execution.decoders.base import ParallelDecoder
from fluxserve.backend.execution.decoders.utils import (
    broadcast_if_needed,
    normalize_eos_ids,
)

# Row-state fields carried through the fused decode graph, in commit order.
# Inputs are read by the step, outputs are scattered back into the runner's
# LevenshteinRowState afterwards.
GRAPH_STATE_FIELDS = (
    "is_original_mask",
    "initial_mask_count",
    "step_id",
    "post_steps",
    "finalized",
    "seen_blocks",
    "seen_count",
)


def apply_edit_operations(
    block_tokens,
    old_block_tokens,
    is_original_mask,
    mask_id,
    block_length,
    delete_token_id,
    split_token_id,
    prompt_length=0,
):
    """Consume DELETE/SPLIT tokens in one row's block, tracking original masks.

    Direct port of the reference ``_apply_edit_operations_with_tracking``:
    DELETE drops the position; SPLIT expands to ``[mask, old_token]`` (the
    pre-write token is restored after the inserted mask); the result is
    truncated or right-padded with masks back to ``block_length``. Padding
    and SPLIT-inserted masks are tracked as non-original.

    Token/tracking arguments are plain Python lists. ``prompt_length`` keeps
    that prefix opaque to edits; returns ``(tokens, tracking)``.

    Kept as the scalar oracle the batched implementation
    (:func:`apply_edit_operations_batched`) is tested against; the decode
    paths use the batched one.
    """
    result_tokens = []
    result_tracking = []
    for i, token in enumerate(block_tokens):
        if i < prompt_length:
            result_tokens.append(token)
            result_tracking.append(False)
            continue
        if token == delete_token_id:
            continue
        if token == split_token_id:
            old_token = old_block_tokens[i] if i < len(old_block_tokens) else mask_id
            snapshot = (
                is_original_mask[i] if i < len(is_original_mask) else False
            )
            result_tokens.extend([mask_id, old_token])
            result_tracking.extend([False, snapshot])
        else:
            snapshot = (
                is_original_mask[i] if i < len(is_original_mask) else False
            )
            result_tokens.append(token)
            result_tracking.append(snapshot)

    if len(result_tokens) > block_length:
        result_tokens = result_tokens[:block_length]
        result_tracking = result_tracking[:block_length]
    elif len(result_tokens) < block_length:
        pad = block_length - len(result_tokens)
        result_tokens.extend([mask_id] * pad)
        result_tracking.extend([False] * pad)
    return result_tokens, result_tracking


def apply_edit_operations_batched(
    blk,
    old,
    is_original_mask,
    prompt_positions,
    mask_id,
    delete_token_id,
    split_token_id,
):
    """Batched, fixed-shape ``apply_edit_operations`` over ``[B, L]`` blocks.

    Each position contributes 0 tokens (DELETE), 2 (SPLIT -> ``[mask,
    old]``) or 1 (everything else, prompt positions included), so the
    destination offsets are an exclusive cumulative sum of the contribution
    counts and the whole rewrite is two scatters into a ``2L + 1`` staging
    buffer. The final slot is a sink for suppressed writes -- ``scatter_``
    has no write mask, so dropped positions are pointed at it and the slot
    is discarded with the truncation to ``block_length``.

    Returns ``(tokens, tracking)``, both ``[B, L]``.
    """
    B, L = blk.shape
    device = blk.device
    generated = ~prompt_positions
    is_del = generated & (blk == delete_token_id)
    is_spl = generated & (blk == split_token_id)

    zeros = torch.zeros_like(blk)
    counts = torch.where(is_del, zeros, torch.where(is_spl, zeros + 2, zeros + 1))
    start = torch.cumsum(counts, dim=1) - counts
    total = counts.sum(dim=1)

    width = 2 * L + 1
    sink = width - 1
    tokens = blk.new_full((B, width), mask_id)
    tracking = torch.zeros((B, width), dtype=torch.bool, device=device)
    sink_idx = torch.full_like(start, sink)

    # Primary contribution: the token itself, or the mask that SPLIT inserts
    # ahead of the restored old token. Prompt positions never carry original
    # -mask tracking, matching the scalar oracle.
    primary_idx = torch.where(is_del, sink_idx, start)
    primary_tok = torch.where(is_spl, torch.full_like(blk, mask_id), blk)
    primary_trk = is_original_mask & (~is_spl) & generated
    tokens.scatter_(1, primary_idx, primary_tok)
    tracking.scatter_(1, primary_idx, primary_trk)

    # Secondary contribution: SPLIT's restored pre-write token, which keeps
    # the tracking snapshot of the position it came from.
    secondary_idx = torch.where(is_spl, (start + 1).clamp(max=sink), sink_idx)
    tokens.scatter_(1, secondary_idx, old)
    tracking.scatter_(1, secondary_idx, is_original_mask)

    out_tokens = tokens[:, :L]
    out_tracking = tracking[:, :L]
    pad = torch.arange(L, device=device).unsqueeze(0) >= total.unsqueeze(1)
    out_tokens = torch.where(pad, torch.full_like(out_tokens, mask_id), out_tokens)
    return out_tokens, out_tracking & (~pad)


def m2t_schedule_need(initial_mask_count: int, steps: int, step_id: int) -> int:
    """Per-step M2T floor: the reference spreads ``initial_mask_count`` masks
    uniformly over ``steps`` steps (``_get_num_transfer_tokens``)."""
    if step_id >= steps:
        return initial_mask_count  # caller treats this as "all masks"
    base = initial_mask_count // steps
    remainder = initial_mask_count % steps
    return base + (1 if step_id < remainder else 0)


def block_seen(blk, seen_blocks, seen_count):
    """Exact, fixed-shape history lookup over pre-edit block token sequences."""
    slots = torch.arange(seen_blocks.shape[1], device=seen_blocks.device)
    live = slots.unsqueeze(0) < seen_count.unsqueeze(1)
    same = (seen_blocks == blk.unsqueeze(1)).all(dim=2)
    return (same & live).any(dim=1)


def levenshtein_graph_step(
    logits,
    x_block,
    prompt_positions,
    is_original_mask,
    initial_mask_count,
    step_id,
    post_steps,
    finalized,
    seen_blocks,
    seen_count,
    mask_id,
    delete_token_id,
    split_token_id,
    threshold,
    editing_threshold,
    steps,
    max_post_steps,
    max_steps_per_block,
    escape_iters=5,
):
    """One LLaDA2.2 refinement iteration as a single fixed-shape tensor program.

    Mirrors the reference ``_joint_decode_block`` body: top-of-iteration mask
    accounting, M2T with the transfer-schedule floor, T2T, the final round's
    DELETE/SPLIT suppression, the anti-loop escape, and the edit-op scan --
    with every per-row branch of the scalar loop turned into a ``torch.where``
    so the whole thing has no data-dependent control flow, no ``.item()`` and
    no collectives, and can therefore be captured in a CUDA graph.

    ``logits`` is mutated: the ``mask_id`` column is set to ``-inf``. Rows with
    ``finalized=True`` are inert -- they apply no write, run no edit scan and
    advance no counter -- which is also what makes CUDA-graph padding rows
    safe.

    Returns ``(new_block, state, predicates)`` where ``state`` is the updated
    row state as a dict over :data:`GRAPH_STATE_FIELDS` plus ``block_result``
    and ``history_append``, and ``predicates`` is
    ``(had_mask, changed, block_finished)``.
    """
    # ---- top-of-iteration accounting, all from the pre-update block.
    mask_index = (x_block == mask_id) & (~prompt_positions)
    original_cnt = (mask_index & is_original_mask).sum(dim=1)
    new_cnt = mask_index.sum(dim=1) - original_cnt

    active = ~finalized
    stepped_post = torch.where(
        original_cnt == 0, post_steps + 1, torch.zeros_like(post_steps)
    )
    post_steps = torch.where(active, stepped_post, post_steps)

    # Greedy argmax consistently selects the lowest token ID on ties, matching
    # the reference. topk(..., 3) does not provide that tie ordering.
    logits[..., mask_id] = -float("inf")
    x0 = torch.argmax(logits, dim=-1)
    top1 = x0
    x0_p = torch.squeeze(
        torch.gather(
            F.softmax(logits.to(torch.float32), dim=-1),
            dim=-1,
            index=torch.unsqueeze(x0, -1),
        ),
        -1,
    )
    neg_inf = torch.full_like(x0_p, -float("inf"))

    # ---- M2T with the per-step schedule floor (strict > threshold, matching
    # the 2.2 reference). The scalar loop's three cases become selects:
    # past the schedule take every mask; else take the above-threshold set if
    # it already meets the floor; else take the floor's worth of the most
    # confident masks.
    mask_conf = torch.where(mask_index, x0_p, neg_inf)
    high_conf = (mask_conf > threshold) & mask_index
    schedule = initial_mask_count // steps + (
        step_id < (initial_mask_count % steps)
    ).long()
    num_need = schedule + new_cnt
    take = torch.minimum(num_need, mask_index.sum(dim=1)).clamp(min=0)
    # Stable sorting makes schedule ties deterministic by position. The
    # reference topk fallback does not specify an order for tied positions.
    order = torch.argsort(mask_conf, dim=1, descending=True, stable=True)
    rank = torch.argsort(order, dim=1)  # inverse permutation: position -> rank
    by_rank = rank < take.unsqueeze(1)
    m2t = torch.where(
        (high_conf.sum(dim=1) >= num_need).unsqueeze(1), high_conf, by_rank
    )
    m2t = torch.where((step_id >= steps).unsqueeze(1), mask_index, m2t)
    m2t = m2t & active.unsqueeze(1) & mask_index.any(dim=1, keepdim=True)

    # ---- T2T (strict > editing_threshold, candidate differs).
    editable = (~mask_index) & (~prompt_positions)
    t2t = (torch.where(editable, x0_p, neg_inf) > editing_threshold) & editable & (
        x0 != x_block
    )

    # ---- final round: budget spent (or step cap imminent). On it, all
    # remaining masks are resolved and DELETE/SPLIT are suppressed so the
    # block terminates fully resolved.
    if max_post_steps > 0:
        final_round = post_steps >= max_post_steps
    else:
        remaining_original = mask_index & is_original_mask
        final_round = remaining_original.any(dim=1) & (
            (remaining_original & ~m2t).sum(dim=1) == 0
        )
    final_round = (final_round | (step_id >= max_steps_per_block - 1)) & active
    m2t = torch.where(final_round.unsqueeze(1), mask_index, m2t)

    # Compute the escape alternative with the greedy token suppressed, then
    # restore it before selecting the final-round candidate without edit IDs.
    # These reductions preserve argmax tie handling without sorting the vocab.
    top1_values = logits.gather(-1, top1.unsqueeze(-1))
    logits.scatter_(-1, top1.unsqueeze(-1), -float("inf"))
    top2 = torch.argmax(logits, dim=-1)
    logits.scatter_(-1, top1.unsqueeze(-1), top1_values)
    delete_values = logits[..., delete_token_id].clone()
    split_values = logits[..., split_token_id].clone()
    logits[..., delete_token_id] = -float("inf")
    logits[..., split_token_id] = -float("inf")
    alt_x0 = torch.argmax(logits, dim=-1)
    logits[..., delete_token_id] = delete_values
    logits[..., split_token_id] = split_values
    is_edit_token = (x0 == delete_token_id) | (x0 == split_token_id)
    x0 = torch.where(
        final_round.unsqueeze(1) & (m2t | t2t) & is_edit_token, alt_x0, x0
    )

    # ---- apply writes (finalized rows apply nothing: their next pass is the
    # runner's stability detection).
    write = (m2t | t2t) & active.unsqueeze(1)
    blk = torch.where(write, x0, x_block)

    # ---- anti-loop escape (skipped on final rounds): while the pre-edit
    # block state repeats one already seen, resample the lowest-confidence
    # changed position, suppressing the token currently sitting there. The
    # reference's bounded retry loop is unrolled, each round gated on the
    # rows that are still repeating.
    escapable = active & (~final_round)
    repeating = block_seen(blk, seen_blocks, seen_count)
    for _ in range(escape_iters):
        changed_pos = blk != x_block
        pos = torch.where(
            changed_pos, x0_p, torch.full_like(x0_p, float("inf"))
        ).argmin(dim=1, keepdim=True)
        resample = (repeating & escapable & changed_pos.any(dim=1)).unsqueeze(1)
        current = blk.gather(1, pos)
        replacement = torch.where(
            current == top1.gather(1, pos), top2.gather(1, pos), top1.gather(1, pos)
        )
        blk = blk.scatter(1, pos, torch.where(resample, replacement, current))
        repeating = block_seen(blk, seen_blocks, seen_count)

    # ---- edit-op consumption. The history records the pre-edit state, as in
    # the reference.
    edited, tracking = apply_edit_operations_batched(
        blk,
        x_block,
        is_original_mask,
        prompt_positions,
        mask_id,
        delete_token_id,
        split_token_id,
    )
    new_block = torch.where(active.unsqueeze(1), edited, blk)
    is_original_mask = torch.where(
        active.unsqueeze(1), tracking, is_original_mask
    )

    state = {
        "is_original_mask": is_original_mask,
        "initial_mask_count": initial_mask_count,
        "step_id": step_id + active.long(),
        "post_steps": post_steps,
        "finalized": finalized | final_round,
        "seen_blocks": seen_blocks,
        "seen_count": seen_count,
        "block_result": blk,
        "history_append": active,
    }
    had_mask = (x_block == mask_id).any(dim=1)
    changed = (new_block != x_block).any(dim=1)
    return new_block, state, (had_mask, changed, (~had_mask) & (~changed))


class LevenshteinRowState:
    """Loop-local per-row decoding state, indexed by global seq_id.

    Owned by the runner's decode loop (never by ``RequestState``), same
    ownership split as ``DecodeEditBudget`` for 2.1. The decoder
    self-initializes a row whenever its ``block_start`` changes, so the
    runner only has to construct this once per decode loop.

    ``seen_blocks``/``seen_count`` are the device-resident equivalent of
    the reference's per-block set of exact token tuples: ``history`` slots
    hold the block tokens recorded so far this block, which is enough because
    a block can never run more iterations than that.
    """

    def __init__(self, num_rows: int, block_length: int, device, history: int = 1):
        self.block_length = int(block_length)
        self.is_original_mask = torch.zeros(
            num_rows, block_length, dtype=torch.bool, device=device
        )
        self.initial_mask_count = torch.zeros(
            num_rows, dtype=torch.long, device=device
        )
        self.step_id = torch.zeros(num_rows, dtype=torch.long, device=device)
        self.post_steps = torch.zeros(num_rows, dtype=torch.long, device=device)
        self.finalized = torch.zeros(num_rows, dtype=torch.bool, device=device)
        self.block_start = torch.full(
            (num_rows,), -1, dtype=torch.long, device=device
        )
        self.history = int(history)
        self.seen_blocks = torch.zeros(
            num_rows, self.history, block_length, dtype=torch.long, device=device
        )
        self.seen_count = torch.zeros(num_rows, dtype=torch.long, device=device)

    def seen_states(self, row):
        """Exact block token tuples recorded for ``row`` this block (diagnostics/tests)."""
        count = int(self.seen_count[row])
        return {tuple(tokens) for tokens in self.seen_blocks[row, :count].tolist()}


class LevenshteinJointDecoder(ParallelDecoder):
    """LLaDA2.2 joint M2T + T2T + Levenshtein (DELETE/SPLIT) decoding."""

    # Signals the runners to pass prompt_lengths / row_state / seq_ids.
    needs_row_state = True
    # The iteration tail is a fixed-shape tensor program, so the decode CUDA
    # graph can capture it together with the model forward. Unlike the 2.1
    # joint decoder it also carries per-row state across iterations, which
    # rides in and out of the graph as extra buffers rather than being
    # mutated in place inside the capture.
    graph_fused_step = True
    graph_extra_state = True

    def __init__(
        self,
        threshold,
        editing_threshold,
        temperature=0,
        mask_id=156895,
        eos_id=156892,
        eos_ids=None,
        delete_token_id=156930,
        split_token_id=156931,
        steps=32,
        max_post_steps=16,
        max_steps_per_block=1000,
    ):
        if temperature not in (0, 0.0):
            raise ValueError(
                "LevenshteinJointDecoder only supports temperature=0; "
                f"got {temperature!r}."
            )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold!r}")
        if not 0.0 <= editing_threshold <= 1.0:
            raise ValueError(
                f"editing_threshold must be in [0, 1], got {editing_threshold!r}"
            )
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps!r}")
        if max_post_steps < 0:
            raise ValueError(
                f"max_post_steps must be non-negative, got {max_post_steps!r}"
            )
        if max_steps_per_block < 2:
            raise ValueError(
                f"max_steps_per_block must be >= 2, got {max_steps_per_block!r}"
            )
        super().__init__(temperature, mask_id=mask_id)
        self.threshold = threshold
        self.editing_threshold = editing_threshold
        self.eos_ids = normalize_eos_ids(eos_ids if eos_ids is not None else eos_id)
        self.eos_id = self.eos_ids[0]
        self.delete_token_id = delete_token_id
        self.split_token_id = split_token_id
        self.steps = steps
        self.max_post_steps = max_post_steps
        self.max_steps_per_block = max_steps_per_block
        self.debug_checks = os.environ.get("FLUXSERVE_DEBUG_LLADA22", "0") == "1"

    def make_row_state(self, num_rows, block_length, device):
        return LevenshteinRowState(
            num_rows, block_length, device, history=self.max_block_iters
        )

    @property
    def max_block_iters(self) -> int:
        # A row is force-finalized at step max_steps_per_block - 1 and needs
        # one further no-op pass for the runner's stability predicate.
        return self.max_steps_per_block + 1

    def _begin_block_rows(self, state, seq_ids, block_start, x_block):
        """Reset per-row state for rows entering a new block.

        Branchless: an ``if new_block.any()`` here would sync the device
        every decode iteration, and in the fused path this runs immediately
        before a graph replay.
        """
        new_block = state.block_start[seq_ids] != block_start
        keep = new_block.unsqueeze(1)
        is_mask = x_block == self.mask_id
        count = is_mask.sum(dim=1)
        state.is_original_mask[seq_ids] = torch.where(
            keep, is_mask, state.is_original_mask[seq_ids]
        )
        state.initial_mask_count[seq_ids] = torch.where(
            new_block, count, state.initial_mask_count[seq_ids]
        )
        zero = torch.zeros_like(count)
        state.step_id[seq_ids] = torch.where(
            new_block, zero, state.step_id[seq_ids]
        )
        state.post_steps[seq_ids] = torch.where(
            new_block, zero, state.post_steps[seq_ids]
        )
        # A fully prompt/context block has nothing to decode: reference
        # returns immediately; here the row is born finalized so T2T can
        # never touch it and the runner finishes it on the first pass.
        state.finalized[seq_ids] = torch.where(
            new_block, count == 0, state.finalized[seq_ids]
        )
        state.block_start[seq_ids] = torch.where(
            new_block, block_start, state.block_start[seq_ids]
        )
        state.seen_count[seq_ids] = torch.where(
            new_block, zero, state.seen_count[seq_ids]
        )

    def _step_kwargs(self, state, seq_ids):
        """Row-state slice consumed by :func:`levenshtein_graph_step`."""
        gathered = {
            field: getattr(state, field)[seq_ids] for field in GRAPH_STATE_FIELDS
        }
        return gathered

    def _run_step(self, logits, x_block, prompt_positions, step_kwargs):
        return levenshtein_graph_step(
            logits,
            x_block,
            prompt_positions,
            mask_id=self.mask_id,
            delete_token_id=self.delete_token_id,
            split_token_id=self.split_token_id,
            threshold=self.threshold,
            editing_threshold=self.editing_threshold,
            steps=self.steps,
            max_post_steps=self.max_post_steps,
            max_steps_per_block=self.max_steps_per_block,
            **step_kwargs,
        )

    def commit_row_state(self, state, seq_ids, step_state):
        """Scatter one step's row state back into the runner's row state.

        ``block_result`` is appended to the row's history for the rows that
        actually ran a refinement pass, which is the device-side form of the
        reference's ``seen_block_results.add(...)``.
        """
        for field in ("is_original_mask", "step_id", "post_steps", "finalized"):
            getattr(state, field)[seq_ids] = step_state[field]
        slot = step_state["seen_count"].clamp(max=state.history - 1)
        append = step_state["history_append"]
        state.seen_blocks[seq_ids, slot] = torch.where(
            append.unsqueeze(1),
            step_state["block_result"],
            state.seen_blocks[seq_ids, slot],
        )
        state.seen_count[seq_ids] = (
            step_state["seen_count"] + append.long()
        ).clamp(max=state.history)

    # ---- CUDA-graph fused tail -------------------------------------------

    def graph_extra_buffers(self, batch_size, block_length, device):
        """Static input buffers for the row state carried through the graph.

        Capture-time values only steer data, never shapes. ``finalized`` is
        the one that matters at replay: padding rows keep ``True``, which
        makes every write, counter and edit scan in the step inert for them.
        """
        state = LevenshteinRowState(
            batch_size, block_length, device, history=self.max_block_iters
        )
        state.finalized.fill_(True)
        return {field: getattr(state, field) for field in GRAPH_STATE_FIELDS}

    def graph_extra_padding(self):
        """Value each extra buffer is reset to before real rows are copied in."""
        return {field: (True if field == "finalized" else 0) for field in GRAPH_STATE_FIELDS}

    def graph_inputs(self, state, seq_ids, block_start, x_block, block_length):
        """Row-state inputs for one graph replay.

        Also performs the new-block reset, which is data-dependent on
        ``block_start`` and therefore stays outside the capture.
        """
        self._begin_block_rows(state, seq_ids, block_start, x_block)
        return {field: getattr(state, field)[seq_ids] for field in GRAPH_STATE_FIELDS}

    def graph_step(self, logits, x_block, prompt_positions, allow_edit, **row_state):
        """Graph-capturable decode-iteration tail.

        ``allow_edit`` is part of the shared fused-tail signature and unused
        here: 2.2 bounds editing with ``post_steps``/``max_post_steps`` in the
        row state rather than with the 2.1 per-row edit budget.
        """
        step_kwargs = dict(row_state)
        new_block, state, predicates = self._run_step(
            logits, x_block, prompt_positions, step_kwargs
        )
        had_mask, changed, block_finished = predicates
        # seen_blocks/seen_count pass through unchanged; the append happens
        # eagerly in commit_row_state, so they are not graph outputs.
        outputs = {
            field: state[field]
            for field in (
                "is_original_mask",
                "step_id",
                "post_steps",
                "finalized",
                "block_result",
                "history_append",
            )
        }
        outputs["seen_count"] = state["seen_count"]
        return new_block, had_mask, changed, block_finished, outputs

    # ---- eager path -------------------------------------------------------

    def batch_decode(
        self,
        logits,
        block_start,
        x,
        block_length,
        prompt_lengths=None,
        row_state=None,
        seq_ids=None,
    ):
        """One refinement iteration over the active blocks of selected rows.

        Same in-place contract as the other decoders. ``row_state`` is the
        runner-owned ``LevenshteinRowState`` and ``seq_ids`` the global row
        ids used to index it; ``prompt_lengths`` is the absolute prompt
        boundary per selected row.

        Runs the same tensor program the CUDA graph captures, then broadcasts
        its outputs so every rank commits the source rank's tokens *and* row
        state.
        """
        if prompt_lengths is None or row_state is None or seq_ids is None:
            raise ValueError(
                "LevenshteinJointDecoder.batch_decode requires prompt_lengths, "
                "row_state, and seq_ids from the runner."
            )
        B, T = x.data.shape
        device = x.data.device

        offset = torch.arange(block_length, device=device).unsqueeze(
            0
        ) + block_start.unsqueeze(1)
        gather_idx = offset.clamp(max=T - 1)
        x_block = torch.gather(x.data, 1, gather_idx)
        prompt_positions = offset < prompt_lengths.unsqueeze(1)

        self._begin_block_rows(row_state, seq_ids, block_start, x_block)
        new_block, state, _ = self._run_step(
            logits,
            x_block,
            prompt_positions,
            self._step_kwargs(row_state, seq_ids),
        )

        # Broadcasting the step's outputs (rather than its intermediate
        # decisions) is what keeps ranks consistent: tokens, tracking and
        # every counter come from the source rank even if a near-threshold
        # comparison went the other way locally.
        broadcast_if_needed(new_block)
        for field in ("is_original_mask", "step_id", "post_steps", "finalized"):
            broadcast_if_needed(state[field])
        broadcast_if_needed(state["block_result"])
        broadcast_if_needed(state["history_append"])
        self.commit_row_state(row_state, seq_ids, state)

        x_flat = x.data.view(-1)
        flat_idx = gather_idx + torch.arange(B, device=device).unsqueeze(1) * T
        x_flat[flat_idx] = new_block

        broadcast_if_needed(x.data)
        if self.debug_checks:
            self._check_state(
                row_state, seq_ids, x_block, new_block, prompt_positions
            )

    def _check_state(self, state, seq_ids, before, after, prompt_positions):
        if not torch.equal(before[prompt_positions], after[prompt_positions]):
            raise RuntimeError("LLaDA2.2 edit changed prompt tokens")
        generated = ~prompt_positions
        control = (after == self.delete_token_id) | (after == self.split_token_id)
        if bool((control & generated).any()):
            raise RuntimeError("LLaDA2.2 left an unconsumed edit token")
        if bool(((after == self.mask_id) & generated & state.finalized[seq_ids, None]).any()):
            raise RuntimeError("LLaDA2.2 finalized a block with residual masks")
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            # Diagnostic only: include host-side tracking/history as well as tokens.
            payload = [after.tolist()]
            for field in ("is_original_mask", "step_id", "post_steps", "finalized", "block_start", "seen_count"):
                payload.append(getattr(state, field)[seq_ids].tolist())
            payload.append(
                [sorted(state.seen_states(r)) for r in seq_ids.tolist()]
            )
            digest = hashlib.sha256(repr(payload).encode()).digest()[:8]
            fingerprint = torch.tensor(
                [int.from_bytes(digest, "little") & ((1 << 63) - 1)],
                dtype=torch.long, device=after.device,
            )
            peers = [torch.empty_like(fingerprint) for _ in range(dist.get_world_size())]
            dist.all_gather(peers, fingerprint)
            if any(not torch.equal(fingerprint, peer) for peer in peers):
                raise RuntimeError("LLaDA2.2 decode state differs across ranks")

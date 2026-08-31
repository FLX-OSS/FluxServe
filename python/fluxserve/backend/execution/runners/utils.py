import torch


class DecodeEditBudget:
    """Loop-local per-row state for LLaDA2.1 joint (editing) decoding.

    Indexed by global seq_id, never by position within a sub-batch, so it
    survives CUDA-graph batch decomposition. ``post_steps`` counts mask-free
    iterations that still changed the block (T2T edits); once it reaches
    ``max_post_steps`` the row's T2T is disabled via ``allow_edit`` and the
    next iteration finishes the block with pre-update == post-update tokens,
    keeping the committed KV consistent with the final tokens.

    ``block_iters`` bounds the per-block iteration count
    (``block_length + max_post_steps + 1``); exceeding it means the decode
    loop is not converging and is reported instead of spinning forever. The
    stuck check reads GPU state, so it runs only every ``max_block_iters``
    calls to ``update`` — a per-call check would force a GPU->CPU sync on
    every decode iteration. Detection is therefore delayed by at most one
    check interval; a row can never exceed the budget without eventually
    tripping a periodic check, because a stuck row keeps the decode loop
    (and this counter) running.
    """

    def __init__(
        self,
        num_rows: int,
        max_post_steps: int,
        block_length: int,
        device,
        max_block_iters_override: int | None = None,
    ):
        self.max_post_steps = int(max_post_steps)
        self.block_length = int(block_length)
        self._max_block_iters_override = max_block_iters_override
        self.post_steps = torch.zeros(num_rows, dtype=torch.long, device=device)
        self.block_iters = torch.zeros(num_rows, dtype=torch.long, device=device)
        self._update_calls = 0

    @property
    def max_block_iters(self) -> int:
        # levenshtein_joint decoding is bounded by max_steps_per_block, not by
        # the mask count; the runner passes an override in that case.
        if self._max_block_iters_override is not None:
            return int(self._max_block_iters_override)
        return self.block_length + self.max_post_steps + 1

    def allow_edit(self, seq_ids: torch.Tensor) -> torch.Tensor:
        return self.post_steps[seq_ids] < self.max_post_steps

    def update(
        self,
        seq_ids: torch.Tensor,
        had_mask: torch.Tensor,
        changed: torch.Tensor,
        block_finished: torch.Tensor,
    ) -> None:
        self.post_steps[seq_ids] += ((~had_mask) & changed).long()
        self.block_iters[seq_ids] += 1
        finished_ids = seq_ids[block_finished]
        self.post_steps[finished_ids] = 0
        self.block_iters[finished_ids] = 0
        # A row that finishes on the last budgeted iteration was reset above;
        # any row still counting at the full budget is genuinely stuck. The
        # check syncs the GPU, so run it only periodically (see class doc).
        self._update_calls += 1
        if self._update_calls % self.max_block_iters == 0:
            over = self.block_iters >= self.max_block_iters
            if bool(over.any()):
                stuck = torch.nonzero(over, as_tuple=True)[0]
                raise RuntimeError(
                    "decode block did not finish within "
                    f"block_length + max_post_steps + 1 = {self.max_block_iters} "
                    f"iterations for seq_ids {stuck.tolist()} "
                    f"(post_steps={self.post_steps[stuck].tolist()})."
                )


def align_exp2(x: torch.Tensor | int):
    if isinstance(x, torch.Tensor):
        assert x.ndim == 0
        x = int(x.item())
    assert x >= 0
    shift = 0 if x == 0 else x.bit_length()
    return 1 << shift


def gather_blocks(x: torch.Tensor, idx: torch.Tensor, block_length: int) -> torch.Tensor:
    offsets = torch.arange(block_length, device=x.device).unsqueeze(0)
    indices = idx.unsqueeze(1) + offsets
    return torch.gather(x, dim=1, index=indices)


def select_batch_sequences_by_mask_number(x, valid_flag, mask_id, batch_size):
    cand_idx = torch.nonzero(valid_flag, as_tuple=False).squeeze(1)
    _, sorted_order = torch.sort(
        -((x.data[cand_idx] == mask_id).sum(dim=1)),
        stable=True,
    )
    return cand_idx[sorted_order[:batch_size]]


def select_batch_sequences_by_order(x, valid_flag, mask_id, batch_size):
    del x, mask_id
    return torch.nonzero(valid_flag, as_tuple=True)[0][:batch_size]


select_prefilling_batch_sequences = select_batch_sequences_by_mask_number
select_decoding_batch_sequences = select_batch_sequences_by_mask_number

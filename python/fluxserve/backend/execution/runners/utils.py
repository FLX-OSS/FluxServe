import torch


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

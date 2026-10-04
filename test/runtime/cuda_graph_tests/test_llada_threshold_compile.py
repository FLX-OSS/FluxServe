"""Correctness checks for a test-only compiled LLaDA threshold decode tail."""

import pytest
import torch

from fluxserve.backend.execution.decoders.threshold import ThresholdParallelDecoder
from threshold_tail_probe import threshold_tail


MASK_ID = 28
EOS_ID = 29
BLOCK = 8
VOCAB = 32


class TokenArray:
    def __init__(self, data):
        self.data = data


def reference(logits, block, threshold):
    decoder = ThresholdParallelDecoder(
        temperature=0, threshold=threshold, mask_id=MASK_ID, eos_id=EOS_ID
    )
    tokens = TokenArray(block.clone())
    decoder.batch_decode(
        logits.clone(),
        torch.zeros(block.shape[0], dtype=torch.long, device=block.device),
        tokens,
        block.shape[1],
    )
    updated = tokens.data
    had_mask = (block == MASK_ID).any(dim=1)
    changed = (updated != block).any(dim=1)
    return updated, had_mask, changed, ~had_mask & ~changed


def cases(device):
    generator = torch.Generator(device=device).manual_seed(1234)
    random_logits = torch.randn(4, BLOCK, VOCAB, device=device, generator=generator)
    random_block = torch.randint(
        0, VOCAB - 4, (4, BLOCK), device=device, generator=generator
    )
    random_block[0, ::2] = MASK_ID
    random_block[1] = MASK_ID
    random_block[2, 0] = MASK_ID
    yield random_logits, random_block, 0.9

    # Force ties, a predicted mask token, and an entirely resolved row.
    tied = torch.zeros(3, BLOCK, VOCAB, device=device)
    tied[0, :, MASK_ID] = 20
    tied[1, :, 4] = 20
    tied[2, :, 3] = 20
    blocks = torch.full((3, BLOCK), MASK_ID, device=device, dtype=torch.long)
    blocks[2] = torch.arange(BLOCK, device=device)
    yield tied, blocks, 0.9

    # Make the leading softmax probability straddle the configured threshold.
    boundary = torch.zeros(1, BLOCK, VOCAB, device=device)
    boundary[0, :, 5] = torch.tensor(
        [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7], device=device
    )
    yield boundary, torch.full((1, BLOCK), MASK_ID, device=device), 0.5


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_threshold_tail_matches_production_decoder(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    compiled = (
        torch.compile(threshold_tail, fullgraph=True, dynamic=False)
        if device == "cuda"
        else None
    )
    with torch.inference_mode():
        for logits, block, threshold in cases(device):
            expected = reference(logits, block, threshold)
            eager = threshold_tail(logits, block, MASK_ID, threshold)
            candidates = (eager, compiled(logits, block, MASK_ID, threshold)) if compiled else (eager,)
            for actual in candidates:
                for result, target in zip(actual, expected, strict=True):
                    torch.testing.assert_close(result, target, rtol=0, atol=0)

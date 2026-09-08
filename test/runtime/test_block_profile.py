import json

import pytest
import torch

from fluxserve.backend.execution.block_profile import BlockProfileCollector
from fluxserve.backend.execution.decoders.threshold import ThresholdParallelDecoder


class FakeTokenArray:
    def __init__(self, data):
        self.data = data


def test_threshold_step_stats_cover_only_transferred_tokens():
    mask_id = 7
    decoder = ThresholdParallelDecoder(
        temperature=0,
        threshold=0.7,
        mask_id=mask_id,
        eos_id=6,
    )
    decoder.profile_block_metrics = True
    x = FakeTokenArray(torch.tensor([[mask_id, mask_id, 2, 3]]))
    logits = torch.zeros(1, 4, 8)
    logits[0, 0, 1] = 4.0
    logits[0, 1, 2] = 2.0

    expected = torch.softmax(logits.float(), dim=-1)
    stats = decoder.batch_decode(logits, torch.tensor([0]), x, 4)

    assert stats.transferred.tolist() == [1]
    assert stats.remaining_masks.tolist() == [1]
    assert stats.confidence_min.item() == pytest.approx(expected[0, 0, 1].item())
    assert stats.confidence_sum.item() == pytest.approx(expected[0, 0, 1].item())
    assert stats.confidence_max.item() == pytest.approx(expected[0, 0, 1].item())
    assert stats.masked_count.tolist() == [2]
    assert stats.masked_confidence_sum.item() == pytest.approx(
        expected[0, 0, 1].item() + expected[0, 1, 2].item()
    )
    assert stats.masked_confidence_quantiles[0, 1].item() == pytest.approx(
        (expected[0, 0, 1].item() + expected[0, 1, 2].item()) / 2
    )
    assert stats.readiness_at_or_above.tolist() == [1]
    assert stats.deficit_count.tolist() == [1]
    assert stats.deficit_sum.item() == pytest.approx(
        0.7 - expected[0, 1, 2].item()
    )
    expected_margin = expected.topk(2, dim=-1).values
    expected_margin = expected_margin[..., 0] - expected_margin[..., 1]
    assert stats.margin_sum.item() == pytest.approx(
        expected_margin[0, :2].sum().item()
    )


def test_threshold_stats_are_disabled_by_default():
    decoder = ThresholdParallelDecoder(
        temperature=0, threshold=0.7, mask_id=7, eos_id=6
    )
    x = FakeTokenArray(torch.tensor([[7, 1]]))
    assert decoder.batch_decode(torch.zeros(1, 2, 8), torch.tensor([0]), x, 2) is None


def test_threshold_step_stats_include_progress_fallback():
    mask_id = 7
    decoder = ThresholdParallelDecoder(
        temperature=0,
        threshold=0.99,
        mask_id=mask_id,
        eos_id=6,
    )
    decoder.profile_block_metrics = True
    x = FakeTokenArray(torch.full((1, 4), mask_id))
    logits = torch.zeros(1, 4, 8)
    logits[0, 2, 3] = 0.1

    stats = decoder.batch_decode(logits, torch.tensor([0]), x, 4)

    assert stats.transferred.tolist() == [1]
    assert stats.remaining_masks.tolist() == [3]


def test_mask_prediction_has_zero_transferable_confidence():
    decoder = ThresholdParallelDecoder(
        temperature=0, threshold=0.95, mask_id=7, eos_id=6
    )
    decoder.profile_block_metrics = True
    x = FakeTokenArray(torch.tensor([[7]]))
    logits = torch.zeros(1, 1, 8)
    logits[0, 0, 7] = 10

    stats = decoder.batch_decode(logits, torch.tensor([0]), x, 1)

    assert stats.transferred.tolist() == [0]
    assert stats.remaining_masks.tolist() == [1]
    assert stats.masked_confidence_sum.tolist() == [0.0]
    assert stats.readiness_at_or_above.tolist() == [0]
    assert stats.deficit_count.tolist() == [1]
    assert stats.deficit_sum.item() == pytest.approx(0.95)


def test_collector_groups_interleaved_rows_and_commit_forward():
    collector = BlockProfileCollector(batch_size=2, block_length=4, total_length=12)
    decoder = ThresholdParallelDecoder(
        temperature=0, threshold=0.5, mask_id=7, eos_id=6
    )
    decoder.profile_block_metrics = True
    data = torch.zeros(2, 12, dtype=torch.long)
    data[0, 8:12] = 7
    data[1, 4:8] = 7
    x = FakeTokenArray(data)
    logits = torch.zeros(2, 4, 8)
    logits[..., 0] = 5
    stats = decoder.batch_decode(logits, torch.tensor([8, 4]), x, 4)

    collector.record(
        torch.tensor([1, 0]),
        torch.tensor([8, 4]),
        stats,
        torch.tensor([False, False]),
    )
    commit_x = FakeTokenArray(x.data[1:2])
    commit_stats = decoder.batch_decode(
        torch.zeros(1, 4, 8), torch.tensor([4]), commit_x, 4
    )
    collector.record(
        torch.tensor([0]),
        torch.tensor([4]),
        commit_stats,
        torch.tensor([True]),
    )

    profiles = collector.finalize()

    assert profiles[0][0]["block_start"] == 4
    assert profiles[0][0]["forward_count"] == 2
    assert profiles[0][0]["tokens_generated"] == 4
    assert profiles[0][0]["iterations"][0]["confidence"] is not None
    commit = profiles[0][0]["iterations"][1]
    assert commit["confidence"] is None
    assert commit["is_commit_forward"] is True
    assert profiles[1][0]["block_start"] == 8
    json.dumps(profiles)


def test_collector_tracks_top1_flips_and_fallback_streaks():
    decoder = ThresholdParallelDecoder(
        temperature=0, threshold=0.99, mask_id=7, eos_id=6
    )
    decoder.profile_block_metrics = True
    collector = BlockProfileCollector(batch_size=1, block_length=4, total_length=4)
    x = FakeTokenArray(torch.full((1, 4), 7))

    first_logits = torch.zeros(1, 4, 8)
    for position, token in enumerate((0, 1, 2, 3)):
        first_logits[0, position, token] = 0.4 - position * 0.05
    first = decoder.batch_decode(first_logits, torch.tensor([0]), x, 4)
    collector.record(torch.tensor([0]), torch.tensor([0]), first, torch.tensor([False]))

    second_logits = torch.zeros(1, 4, 8)
    for position, token in enumerate((4, 5, 6, 0)):
        second_logits[0, position, token] = 0.5 - position * 0.05
    second = decoder.batch_decode(second_logits, torch.tensor([0]), x, 4)
    collector.record(torch.tensor([0]), torch.tensor([0]), second, torch.tensor([False]))

    block = collector.finalize()[0][0]
    first_stability = block["iterations"][0]["prediction_stability"]
    second_stability = block["iterations"][1]["prediction_stability"]
    assert first_stability["comparable"] == 0
    assert first_stability["flip_rate"] is None
    assert second_stability["comparable"] == 3
    assert second_stability["flipped"] == 3
    assert second_stability["flip_rate"] == 1.0
    assert block["fallback_forward_count"] == 2
    assert block["longest_fallback_streak"] == 2
    assert [item["fallback_streak"] for item in block["iterations"]] == [1, 2]

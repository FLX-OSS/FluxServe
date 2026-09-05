from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ThresholdDecodeStepStats:
    transferred: torch.Tensor
    confidence_sum: torch.Tensor
    confidence_min: torch.Tensor
    confidence_max: torch.Tensor
    remaining_masks: torch.Tensor
    masked_count: torch.Tensor
    masked_confidence_sum: torch.Tensor
    masked_confidence_min: torch.Tensor
    masked_confidence_quantiles: torch.Tensor
    masked_confidence_max: torch.Tensor
    threshold: torch.Tensor
    readiness_at_or_above: torch.Tensor
    readiness_within_001_below: torch.Tensor
    readiness_within_005_below: torch.Tensor
    deficit_count: torch.Tensor
    deficit_sum: torch.Tensor
    deficit_max: torch.Tensor
    margin_sum: torch.Tensor
    margin_quantiles: torch.Tensor
    candidate_top1: torch.Tensor
    unresolved_before: torch.Tensor
    unresolved_after: torch.Tensor


class BlockProfileCollector:
    """Accumulate compact GPU tensors and materialize once per generation."""

    def __init__(self, batch_size: int, block_length: int, total_length: int):
        self.batch_size = int(batch_size)
        self.block_length = int(block_length)
        self.total_length = int(total_length)
        self._steps = []
        self._previous_top1 = None

    def record(self, seq_ids, block_starts, stats, block_finished) -> None:
        if self._previous_top1 is None:
            self._previous_top1 = torch.full(
                (self.batch_size, self.total_length),
                -1,
                dtype=stats.candidate_top1.dtype,
                device=stats.candidate_top1.device,
            )
        offsets = block_starts.unsqueeze(1) + torch.arange(
            self.block_length, device=block_starts.device
        ).unsqueeze(0)
        previous = self._previous_top1[seq_ids.unsqueeze(1), offsets]
        comparable_mask = stats.unresolved_before & (previous >= 0)
        comparable = comparable_mask.sum(dim=1)
        flipped = (comparable_mask & (previous != stats.candidate_top1)).sum(dim=1)
        stable = comparable - flipped
        self._previous_top1[seq_ids.unsqueeze(1), offsets] = torch.where(
            stats.unresolved_after,
            stats.candidate_top1,
            torch.full_like(stats.candidate_top1, -1),
        )

        compact = (
            seq_ids,
            block_starts,
            stats.transferred,
            stats.confidence_sum,
            stats.confidence_min,
            stats.confidence_max,
            stats.remaining_masks,
            block_finished,
            stats.masked_count,
            stats.masked_confidence_sum,
            stats.masked_confidence_min,
            stats.masked_confidence_quantiles,
            stats.masked_confidence_max,
            stats.threshold,
            stats.readiness_at_or_above,
            stats.readiness_within_001_below,
            stats.readiness_within_005_below,
            stats.deficit_count,
            stats.deficit_sum,
            stats.deficit_max,
            stats.margin_sum,
            stats.margin_quantiles,
            comparable,
            stable,
            flipped,
        )
        self._steps.append(tuple(value.detach().clone() for value in compact))

    def finalize(self) -> list[list[dict]]:
        rows: list[dict[int, list[dict]]] = [dict() for _ in range(self.batch_size)]
        for step in self._steps:
            (
                seq_ids,
                block_starts,
                transferred,
                confidence_sum,
                confidence_min,
                confidence_max,
                remaining_masks,
                block_finished,
                masked_count,
                masked_confidence_sum,
                masked_confidence_min,
                masked_confidence_quantiles,
                masked_confidence_max,
                threshold,
                readiness_at_or_above,
                readiness_within_001_below,
                readiness_within_005_below,
                deficit_count,
                deficit_sum,
                deficit_max,
                margin_sum,
                margin_quantiles,
                comparable,
                stable,
                flipped,
            ) = [value.cpu().tolist() for value in step]
            for index, seq_id in enumerate(seq_ids):
                transfer_count = int(transferred[index])
                confidence = None
                if transfer_count:
                    confidence = {
                        "min": float(confidence_min[index]),
                        "mean": float(confidence_sum[index]) / transfer_count,
                        "max": float(confidence_max[index]),
                    }

                candidate_count = int(masked_count[index])
                masked_confidence = None
                readiness = None
                deficit = None
                margin = None
                stability = None
                if candidate_count:
                    confidence_q = masked_confidence_quantiles[index]
                    margin_q = margin_quantiles[index]
                    masked_confidence = {
                        "count": candidate_count,
                        "min": float(masked_confidence_min[index]),
                        "p10": float(confidence_q[0]),
                        "median": float(confidence_q[1]),
                        "mean": float(masked_confidence_sum[index]) / candidate_count,
                        "p90": float(confidence_q[2]),
                        "max": float(masked_confidence_max[index]),
                    }
                    readiness = {
                        "threshold": float(threshold[index]),
                        "at_or_above": int(readiness_at_or_above[index]),
                        "within_0_01_below": int(readiness_within_001_below[index]),
                        "within_0_05_below": int(readiness_within_005_below[index]),
                    }
                    below_count = int(deficit_count[index])
                    deficit = {
                        "count": below_count,
                        "sum": float(deficit_sum[index]),
                        "mean": (
                            float(deficit_sum[index]) / below_count
                            if below_count
                            else 0.0
                        ),
                        "max": float(deficit_max[index]),
                    }
                    margin = {
                        "p10": float(margin_q[0]),
                        "median": float(margin_q[1]),
                        "mean": float(margin_sum[index]) / candidate_count,
                        "p90": float(margin_q[2]),
                    }
                    comparable_count = int(comparable[index])
                    stability = {
                        "comparable": comparable_count,
                        "stable": int(stable[index]),
                        "flipped": int(flipped[index]),
                        "flip_rate": (
                            float(flipped[index]) / comparable_count
                            if comparable_count
                            else None
                        ),
                    }

                is_fallback = bool(
                    transfer_count and not int(readiness_at_or_above[index])
                )
                block = rows[int(seq_id)].setdefault(int(block_starts[index]), [])
                block.append(
                    {
                        "forward_index": len(block),
                        "tokens_generated": transfer_count,
                        "remaining_masks": int(remaining_masks[index]),
                        "confidence": confidence,
                        "masked_confidence": masked_confidence,
                        "threshold_readiness": readiness,
                        "confidence_deficit": deficit,
                        "top2_margin": margin,
                        "prediction_stability": stability,
                        "is_fallback_forward": is_fallback,
                        "is_commit_forward": bool(block_finished[index]),
                    }
                )

        result = []
        for row in rows:
            blocks = []
            for block_index, block_start in enumerate(sorted(row)):
                iterations = row[block_start]
                fallback_streak = 0
                longest_fallback_streak = 0
                for iteration in iterations:
                    if iteration["is_fallback_forward"]:
                        fallback_streak += 1
                        longest_fallback_streak = max(
                            longest_fallback_streak, fallback_streak
                        )
                    else:
                        fallback_streak = 0
                    iteration["fallback_streak"] = fallback_streak
                blocks.append(
                    {
                        "block_index": block_index,
                        "block_start": block_start,
                        "forward_count": len(iterations),
                        "tokens_generated": sum(
                            item["tokens_generated"] for item in iterations
                        ),
                        "fallback_forward_count": sum(
                            item["is_fallback_forward"] for item in iterations
                        ),
                        "single_token_forward_count": sum(
                            item["tokens_generated"] == 1 for item in iterations
                        ),
                        "longest_fallback_streak": longest_fallback_streak,
                        "iterations": iterations,
                    }
                )
            result.append(blocks)
        return result

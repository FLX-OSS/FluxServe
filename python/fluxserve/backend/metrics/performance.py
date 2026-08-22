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

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Sequence


@dataclass(frozen=True)
class BatchPerformanceMetrics:
    batch_token_number: int
    batch_token_numbers: list[int]
    fps: float
    tps: float
    tpf: float
    sample_time: float
    nfe: int


@dataclass(frozen=True)
class DecodeBlockMetric:
    sample_index: int
    block_index: int
    block_start: int
    block_end: int
    num_forwards: int
    latency_s: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "sample_index": self.sample_index,
            "block_index": self.block_index,
            "block_start": self.block_start,
            "block_end": self.block_end,
            "num_forwards": self.num_forwards,
            "latency_s": self.latency_s,
        }


def _eos_mask(tokens: Any, eos_id: int | Sequence[int]):
    eos_ids = (eos_id,) if isinstance(eos_id, int) else tuple(eos_id)
    if not eos_ids:
        raise ValueError("at least one EOS token ID is required")
    mask = tokens == int(eos_ids[0])
    for stop_id in eos_ids[1:]:
        mask |= tokens == int(stop_id)
    return mask


def count_completion_tokens(
    row: Any,
    input_len: int,
    eos_id: int | Sequence[int],
    mask_id: int,
) -> int:
    generated = row[input_len:]
    eos_indices = _eos_mask(generated, eos_id).nonzero(as_tuple=True)[0]
    if eos_indices.numel() > 0:
        return int(eos_indices[0].item()) + 1
    return int((generated != mask_id).sum().item())


def record_batch_performance_metrics(
    batch_info: Any,
    out: Any,
    sorted_input_ids: Sequence[Any],
    start_idx: int,
    nfe: int,
    sample_time: float,
    eos_id: int | Sequence[int],
    mask_id: int,
) -> BatchPerformanceMetrics:
    batch_token_number = 0
    batch_token_numbers = []
    for j in range(out.shape[0]):
        token_number = count_completion_tokens(
            out[j],
            sorted_input_ids[start_idx + j].shape[1],
            eos_id,
            mask_id,
        )
        batch_token_number += token_number
        batch_token_numbers.append(token_number)
        batch_info.token_numbers.append(token_number)

    fps = nfe / sample_time if sample_time > 0 else 0.0
    tps = batch_token_number / sample_time if sample_time > 0 else 0.0
    batch_tpfs = []
    for token_number in batch_token_numbers:
        tpf = token_number / nfe if nfe else 0.0
        batch_tpfs.append(tpf)
        batch_info.tpfs.append(tpf)
        batch_info.tpss.append(tps)
        batch_info.fpss.append(fps)

    batch_info.total_forward += nfe
    batch_info.total_time += sample_time
    batch_info.total_token += batch_token_number

    return BatchPerformanceMetrics(
        batch_token_number=batch_token_number,
        batch_token_numbers=batch_token_numbers,
        fps=fps,
        tps=tps,
        tpf=batch_tpfs[-1] if batch_tpfs else 0.0,
        sample_time=sample_time,
        nfe=nfe,
    )


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(float(value) for value in values)
    rank = (len(sorted_values) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize_decode_block_metrics(
    metrics: Sequence[DecodeBlockMetric],
) -> dict[str, float | int]:
    forwards = [metric.num_forwards for metric in metrics]
    latencies = [metric.latency_s for metric in metrics]
    return {
        "total_decode_blocks": len(metrics),
        "total_decode_forwards": sum(forwards),
        "block_forwards_avg": mean(forwards) if forwards else 0.0,
        "block_forwards_max": max(forwards) if forwards else 0,
        "block_latency_avg_s": mean(latencies) if latencies else 0.0,
        "block_latency_p50_s": percentile(latencies, 0.50),
        "block_latency_p95_s": percentile(latencies, 0.95),
        "block_latency_max_s": max(latencies) if latencies else 0.0,
    }

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


def _parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _parse_dtype(value: str) -> torch.dtype:
    choices = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return choices[value.lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            f"unsupported dtype {value!r}; expected one of {sorted(choices)}"
        ) from exc


def _parse_index_dtype(value: str) -> torch.dtype:
    choices = {
        "i32": torch.int32,
        "int32": torch.int32,
        "i64": torch.int64,
        "int64": torch.int64,
    }
    try:
        return choices[value.lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            f"unsupported index dtype {value!r}; expected one of {sorted(choices)}"
        ) from exc


def _make_indptr(lengths: tuple[int, ...]) -> tuple[int, ...]:
    values = [0]
    for length in lengths:
        values.append(values[-1] + int(length))
    return tuple(values)


def _page_counts(lengths: tuple[int, ...], page_size: int) -> tuple[int, ...]:
    return tuple((int(length) + page_size - 1) // page_size for length in lengths)


def _last_page_lens(lengths: tuple[int, ...], page_size: int) -> tuple[int, ...]:
    return tuple(((int(length) - 1) % page_size) + 1 for length in lengths)


def _make_paged_kv_metadata(
    lengths: tuple[int, ...],
    page_size: int,
    index_dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    page_indptr = _make_indptr(_page_counts(lengths, page_size))
    num_pages = page_indptr[-1]
    return (
        torch.tensor(page_indptr, dtype=index_dtype, device=device),
        torch.arange(num_pages, dtype=index_dtype, device=device),
        torch.tensor(_last_page_lens(lengths, page_size), dtype=index_dtype, device=device),
    )


def _make_paged_kv_cache(
    num_pages: int,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    kv_layout: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kv_layout == "NHD":
        cache = torch.randn(
            num_pages,
            2,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
        return cache.unbind(1)
    if kv_layout == "HND":
        cache = torch.randn(
            num_pages,
            2,
            num_kv_heads,
            page_size,
            head_dim,
            dtype=dtype,
            device=device,
        )
        return cache.unbind(1)
    raise ValueError(f"unsupported kv_layout {kv_layout!r}")


def _materialize_one(
    pages: torch.Tensor,
    page_indices: torch.Tensor,
    length: int,
    page_size: int,
    kv_layout: str,
) -> torch.Tensor:
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    num_pages = (length + page_size - 1) // page_size
    selected = pages[page_indices[:num_pages].long()]
    if kv_layout == "NHD":
        dense = selected.reshape(-1, selected.shape[-2], selected.shape[-1])
        return dense[:length].permute(1, 0, 2).contiguous()
    if kv_layout == "HND":
        dense = selected.permute(1, 0, 2, 3).reshape(selected.shape[1], -1, selected.shape[-1])
        return dense[:, :length, :].contiguous()
    raise ValueError(f"unsupported kv_layout {kv_layout!r}")


def _materialize_paged_kv(
    paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    lengths: tuple[int, ...],
    page_size: int,
    kv_layout: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    k_pages, v_pages = paged_kv_cache
    dense_k = []
    dense_v = []
    for batch_idx, length in enumerate(lengths):
        start = int(kv_indptr[batch_idx].item())
        end = int(kv_indptr[batch_idx + 1].item())
        page_indices = kv_indices[start:end]
        dense_k.append(_materialize_one(k_pages, page_indices, int(length), page_size, kv_layout))
        dense_v.append(_materialize_one(v_pages, page_indices, int(length), page_size, kv_layout))
    return dense_k, dense_v


@dataclass(frozen=True)
class BlockPrefillPlan:
    qo_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]
    source_seq_ids: tuple[int, ...]
    q_starts: tuple[int, ...]
    source_page_offsets: tuple[int, ...]
    source_page_counts: tuple[int, ...]


@dataclass(frozen=True)
class DecodePlan:
    qo_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]
    source_seq_ids: tuple[int, ...]
    q_starts: tuple[int, ...]
    source_page_offsets: tuple[int, ...]
    source_page_counts: tuple[int, ...]


def _make_block_prefill_plan(lengths: tuple[int, ...], block_length: int, page_size: int) -> BlockPrefillPlan:
    qo_lens = []
    kv_lens = []
    source_seq_ids = []
    q_starts = []
    source_page_offsets = []
    source_page_counts = _page_counts(lengths, page_size)
    page_offsets = _make_indptr(source_page_counts)
    max_steps = max((int(length) + block_length - 1) // block_length for length in lengths)
    for step in range(max_steps):
        q_start = step * block_length
        for seq_id, seq_len in enumerate(lengths):
            if q_start >= int(seq_len):
                continue
            q_len = min(block_length, int(seq_len) - q_start)
            qo_lens.append(q_len)
            kv_lens.append(q_start + q_len)
            source_seq_ids.append(seq_id)
            q_starts.append(q_start)
            source_page_offsets.append(page_offsets[seq_id])
    return BlockPrefillPlan(
        qo_lens=tuple(qo_lens),
        kv_lens=tuple(kv_lens),
        source_seq_ids=tuple(source_seq_ids),
        q_starts=tuple(q_starts),
        source_page_offsets=tuple(source_page_offsets),
        source_page_counts=tuple(source_page_counts),
    )


def _make_decode_plan(
    kv_lens: tuple[int, ...],
    page_size: int,
    block_length: int,
) -> DecodePlan:
    source_page_offsets = []
    source_page_counts = _page_counts(kv_lens, page_size)
    page_offsets = _make_indptr(source_page_counts)
    for seq_id in range(len(kv_lens)):
        source_page_offsets.append(page_offsets[seq_id])
    return DecodePlan(
        qo_lens=tuple(block_length for _ in kv_lens),
        kv_lens=tuple(int(length) for length in kv_lens),
        source_seq_ids=tuple(range(len(kv_lens))),
        q_starts=tuple(0 for _ in kv_lens),
        source_page_offsets=tuple(source_page_offsets),
        source_page_counts=tuple(source_page_counts),
    )


def _make_shared_kv_metadata(
    plan: BlockPrefillPlan | DecodePlan,
    page_size: int,
    index_dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kv_indptr = [0]
    kv_indices = []
    for kv_len, page_offset in zip(plan.kv_lens, plan.source_page_offsets):
        page_count = (int(kv_len) + page_size - 1) // page_size
        kv_indices.extend(range(int(page_offset), int(page_offset) + page_count))
        kv_indptr.append(len(kv_indices))
    return (
        torch.tensor(kv_indptr, dtype=index_dtype, device=device),
        torch.tensor(kv_indices, dtype=index_dtype, device=device),
        torch.tensor(_last_page_lens(plan.kv_lens, page_size), dtype=index_dtype, device=device),
    )


def _make_virtual_q(
    q_by_seq: torch.Tensor,
    plan: BlockPrefillPlan | DecodePlan,
) -> torch.Tensor:
    chunks = []
    for seq_id, q_start, q_len in zip(plan.source_seq_ids, plan.q_starts, plan.qo_lens):
        chunks.append(q_by_seq[int(seq_id), int(q_start) : int(q_start) + int(q_len)])
    return torch.cat(chunks, dim=0).contiguous()


def _repeat_kv_for_gqa(k: torch.Tensor, v: torch.Tensor, num_q_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    if num_q_heads == k.shape[0]:
        return k, v
    repeat = num_q_heads // k.shape[0]
    return k.repeat_interleave(repeat, dim=0), v.repeat_interleave(repeat, dim=0)


def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: Optional[float],
) -> torch.Tensor:
    # FluxServe DenseAttention uses [B, H, T, D] and is_causal=False.
    q_b = q.unsqueeze(0).transpose(1, 2)
    k_b = k.unsqueeze(0)
    v_b = v.unsqueeze(0)
    k_b, v_b = _repeat_kv_for_gqa(k_b[0], v_b[0], q_b.shape[1])
    output = F.scaled_dot_product_attention(
        q_b,
        k_b.unsqueeze(0),
        v_b.unsqueeze(0),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=sm_scale,
    )
    return output.squeeze(0).transpose(0, 1).contiguous()


def _run_sdpa_prefill_reference(
    q_by_seq: torch.Tensor,
    dense_k_by_seq: list[torch.Tensor],
    dense_v_by_seq: list[torch.Tensor],
    plan: BlockPrefillPlan,
    sm_scale: Optional[float],
) -> torch.Tensor:
    outputs = []
    for seq_id, q_start, q_len, kv_len in zip(
        plan.source_seq_ids, plan.q_starts, plan.qo_lens, plan.kv_lens
    ):
        seq_id = int(seq_id)
        q = q_by_seq[seq_id, int(q_start) : int(q_start) + int(q_len)]
        k = dense_k_by_seq[seq_id][:, : int(kv_len), :]
        v = dense_v_by_seq[seq_id][:, : int(kv_len), :]
        outputs.append(_sdpa(q, k, v, sm_scale))
    return torch.cat(outputs, dim=0).contiguous()


def _run_sdpa_decode_reference(
    q_by_seq: torch.Tensor,
    dense_k_by_seq: list[torch.Tensor],
    dense_v_by_seq: list[torch.Tensor],
    plan: DecodePlan,
    sm_scale: Optional[float],
) -> torch.Tensor:
    outputs = []
    for seq_id, q_start, q_len in zip(plan.source_seq_ids, plan.q_starts, plan.qo_lens):
        seq_id = int(seq_id)
        q = q_by_seq[seq_id, int(q_start) : int(q_start) + int(q_len)]
        outputs.append(_sdpa(q, dense_k_by_seq[seq_id], dense_v_by_seq[seq_id], sm_scale))
    return torch.cat(outputs, dim=0).contiguous()


def _run_chunked_sdpa_reference(
    q_p_by_seq: torch.Tensor,
    dense_k_p: list[torch.Tensor],
    dense_v_p: list[torch.Tensor],
    q_d_by_seq: torch.Tensor,
    dense_k_d: list[torch.Tensor],
    dense_v_d: list[torch.Tensor],
    decode_plan: DecodePlan,
    block_length: int,
    sm_scale: Optional[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_steps = max((k.shape[1] + block_length - 1) // block_length for k in dense_k_p)

    prefill_outputs = []
    decode_outputs = []
    first_decode_outputs = []
    for step in range(num_steps):
        for seq_id, seq_len in enumerate(k.shape[1] for k in dense_k_p):
            q_start = step * block_length
            if q_start >= seq_len:
                continue
            q_len = min(block_length, seq_len - q_start)
            kv_len = q_start + q_len
            q = q_p_by_seq[seq_id, q_start : q_start + q_len]
            k = dense_k_p[seq_id][:, :kv_len, :]
            v = dense_v_p[seq_id][:, :kv_len, :]
            prefill_outputs.append(_sdpa(q, k, v, sm_scale))

        step_decode_outputs = []
        for seq_id in decode_plan.source_seq_ids:
            seq_id = int(seq_id)
            q_len = int(decode_plan.qo_lens[seq_id])
            out = _sdpa(
                q_d_by_seq[seq_id, :q_len],
                dense_k_d[seq_id],
                dense_v_d[seq_id],
                sm_scale,
            )
            decode_outputs.append(out)
            step_decode_outputs.append(out)
        if step == 0:
            first_decode_outputs = step_decode_outputs

    prefill = torch.cat(prefill_outputs, dim=0).contiguous()
    decode_all = torch.cat(decode_outputs, dim=0).contiguous()
    decode_first = torch.cat(first_decode_outputs, dim=0).contiguous()
    return prefill, decode_all, decode_first


def _event_time_ms(fn, warmup: int, iters: int):
    output = None
    for _ in range(warmup):
        output = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        output = fn()
    end.record()
    end.synchronize()
    assert output is not None
    return float(start.elapsed_time(end)) / iters, output


def _tensor_shapes(value) -> object:
    if isinstance(value, torch.Tensor):
        return tuple(int(dim) for dim in value.shape)
    if isinstance(value, (tuple, list)):
        return [_tensor_shapes(item) for item in value]
    return str(type(value))


def _split_pod_output(output):
    if isinstance(output, (tuple, list)) and len(output) == 2:
        return output[0], output[1]
    raise TypeError(f"expected BatchPOD output tuple/list of length 2, got {_tensor_shapes(output)}")


def _error_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    abs_err = (actual_f - expected_f).abs()
    rel_err = abs_err / expected_f.abs().clamp_min(1e-6)
    return {
        "max_abs_err": float(abs_err.max().item()),
        "mean_abs_err": float(abs_err.mean().item()),
        "max_rel_err": float(rel_err.max().item()),
        "mean_rel_err": float(rel_err.mean().item()),
    }


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, rtol: float, atol: float) -> dict[str, object]:
    stats = _error_stats(actual, expected)
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        passed = True
        message = ""
    except AssertionError as exc:
        passed = False
        message = str(exc).splitlines()[0]
    return {
        "name": name,
        "passed": passed,
        "rtol": rtol,
        "atol": atol,
        **stats,
        "message": message,
    }


def _validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if args.num_q_heads % args.num_kv_heads != 0:
        raise ValueError(
            "num_q_heads must be divisible by num_kv_heads for grouped-query attention."
        )
    if args.page_size <= 0:
        raise ValueError(f"page_size must be positive, got {args.page_size}.")
    if args.block_length <= 0:
        raise ValueError(f"block_length must be positive, got {args.block_length}.")
    if args.workspace_mb <= 0:
        raise ValueError(f"workspace_mb must be positive, got {args.workspace_mb}.")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup must be >= 0 and iters must be > 0.")
    if any(length <= 0 for length in args.prefill_seq_lens):
        raise ValueError(f"prefill_seq_lens must be positive, got {args.prefill_seq_lens}.")
    if any(length <= 0 for length in args.decode_kv_lens):
        raise ValueError(f"decode_kv_lens must be positive, got {args.decode_kv_lens}.")
    if any(length < args.block_length for length in args.decode_kv_lens):
        raise ValueError(
            "Each decode_kv_len must be >= block_length because every decode request "
            f"uses q_len=block_length. got kv={args.decode_kv_lens}, block_length={args.block_length}."
        )

    if args.decode_q_lens is None:
        decode_q_lens = tuple(args.block_length for _ in args.decode_kv_lens)
    else:
        decode_q_lens = args.decode_q_lens
    if len(decode_q_lens) != len(args.decode_kv_lens):
        raise ValueError(
            "decode_q_lens must have the same batch size as decode_kv_lens: "
            f"got {decode_q_lens} vs {args.decode_kv_lens}."
        )
    if any(length <= 0 for length in decode_q_lens):
        raise ValueError(f"decode_q_lens must be positive, got {decode_q_lens}.")
    if any(q_len > kv_len for q_len, kv_len in zip(decode_q_lens, args.decode_kv_lens)):
        raise ValueError(
            "Each decode_q_len must be <= the matching decode_kv_len: "
            f"got q={decode_q_lens}, kv={args.decode_kv_lens}."
        )
    if any(length != args.block_length for length in decode_q_lens):
        raise ValueError(
            "This benchmark uses exactly one block per decode request; "
            f"each decode_q_len must equal block_length={args.block_length}. "
            f"got {decode_q_lens}."
        )
    return decode_q_lens


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    decode_q_lens = _validate_args(args)

    import flashinfer

    torch.manual_seed(args.seed)
    if args.device is not None:
        torch.cuda.set_device(args.device)
    device = torch.device("cuda", torch.cuda.current_device())

    prefill_plan = _make_block_prefill_plan(
        args.prefill_seq_lens, args.block_length, args.page_size
    )
    decode_plan = _make_decode_plan(args.decode_kv_lens, args.page_size, args.block_length)

    unified_kv_lens = tuple(args.prefill_seq_lens) + tuple(args.decode_kv_lens)
    unified_page_counts = _page_counts(unified_kv_lens, args.page_size)
    unified_page_offsets = _make_indptr(unified_page_counts)
    pod_qo_lens = list(prefill_plan.qo_lens)
    pod_kv_lens = list(prefill_plan.kv_lens)
    pod_source_page_offsets = list(prefill_plan.source_page_offsets)
    for decode_idx, kv_len in enumerate(args.decode_kv_lens):
        source_seq_id = len(args.prefill_seq_lens) + decode_idx
        pod_qo_lens.append(args.block_length)
        pod_kv_lens.append(int(kv_len))
        pod_source_page_offsets.append(unified_page_offsets[source_seq_id])
    pod_prefill_plan = BlockPrefillPlan(
        qo_lens=tuple(pod_qo_lens),
        kv_lens=tuple(pod_kv_lens),
        source_seq_ids=tuple(0 for _ in pod_qo_lens),
        q_starts=tuple(0 for _ in pod_qo_lens),
        source_page_offsets=tuple(pod_source_page_offsets),
        source_page_counts=unified_page_counts,
    )
    qo_indptr_p = torch.tensor(_make_indptr(pod_prefill_plan.qo_lens), dtype=args.index_dtype, device=device)
    base_kv_indptr, base_kv_indices, _ = _make_paged_kv_metadata(
        unified_kv_lens, args.page_size, args.index_dtype, device
    )
    kv_indptr_p, kv_indices_p, last_page_len_p = _make_shared_kv_metadata(
        pod_prefill_plan, args.page_size, args.index_dtype, device
    )
    qo_indptr_d = torch.tensor((0,), dtype=args.index_dtype, device=device)
    kv_indptr_d = torch.tensor((0,), dtype=args.index_dtype, device=device)
    kv_indices_d = torch.empty(0, dtype=args.index_dtype, device=device)
    last_page_len_d = torch.empty(0, dtype=args.index_dtype, device=device)

    max_prefill_len = max(args.prefill_seq_lens)
    q_p_by_seq = torch.randn(
        len(args.prefill_seq_lens),
        max_prefill_len,
        args.num_q_heads,
        args.head_dim,
        dtype=args.dtype,
        device=device,
    )
    q_prefill = _make_virtual_q(q_p_by_seq, prefill_plan)
    max_decode_q_len = args.block_length
    q_d_by_seq = torch.randn(
        len(decode_q_lens),
        max_decode_q_len,
        args.num_q_heads,
        args.head_dim,
        dtype=args.dtype,
        device=device,
    )
    q_decode = q_d_by_seq.reshape(-1, args.num_q_heads, args.head_dim).contiguous()
    q_p = torch.cat((q_prefill, q_decode), dim=0).contiguous()
    prefill_output_tokens = int(q_prefill.shape[0])
    paged_kv_cache_p = _make_paged_kv_cache(
        int(base_kv_indptr[-1]),
        args.page_size,
        args.num_kv_heads,
        args.head_dim,
        args.dtype,
        device,
        args.kv_layout,
    )
    paged_kv_cache_d = _make_paged_kv_cache(
        0,
        args.page_size,
        args.num_kv_heads,
        args.head_dim,
        args.dtype,
        device,
        args.kv_layout,
    )
    q_d = torch.empty(0, args.num_q_heads, args.head_dim, dtype=args.dtype, device=device)

    workspace = torch.empty(
        args.workspace_mb * 1024 * 1024,
        dtype=torch.uint8,
        device=device,
    )
    wrapper = flashinfer.BatchPODWithPagedKVCacheWrapper(workspace, kv_layout=args.kv_layout)

    sm_scale = args.sm_scale if args.sm_scale is not None else 1.0 / math.sqrt(args.head_dim)
    plan_start = time.perf_counter()
    wrapper.plan(
        qo_indptr_p,
        kv_indptr_p,
        kv_indices_p,
        last_page_len_p,
        qo_indptr_d,
        kv_indptr_d,
        kv_indices_d,
        last_page_len_d,
        num_qo_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        page_size=args.page_size,
        pos_encoding_mode=args.pos_encoding_mode,
        window_left=args.window_left,
        q_data_type=args.dtype,
        kv_data_type=args.dtype,
        sm_scale=args.sm_scale,
    )
    torch.cuda.synchronize()
    plan_wall_ms = (time.perf_counter() - plan_start) * 1000.0
    plan_info_p = tuple(int(x) for x in wrapper._plan_info_p)
    plan_info_d = tuple(int(x) for x in wrapper._plan_info_d)

    def run_flashinfer():
        output = wrapper.run(
            q_p,
            paged_kv_cache_p,
            q_d,
            paged_kv_cache_d,
            causal_p=False,
            return_lse=False,
            use_fp16_qk_reduction=args.use_fp16_qk_reduction,
            enable_pdl=args.enable_pdl,
        )
        pod_prefill_output, _ = _split_pod_output(output)
        return (
            pod_prefill_output[:prefill_output_tokens],
            pod_prefill_output[prefill_output_tokens:],
        )

    flashinfer_ms, flashinfer_output = _event_time_ms(run_flashinfer, args.warmup, args.iters)

    correctness = None
    sdpa_ms = None
    sdpa_output = None
    if args.compare_correctness or args.benchmark_reference:
        dense_k_all, dense_v_all = _materialize_paged_kv(
            paged_kv_cache_p,
            base_kv_indptr,
            base_kv_indices,
            unified_kv_lens,
            args.page_size,
            args.kv_layout,
        )
        dense_k_p = dense_k_all[: len(args.prefill_seq_lens)]
        dense_v_p = dense_v_all[: len(args.prefill_seq_lens)]
        dense_k_d = dense_k_all[len(args.prefill_seq_lens) :]
        dense_v_d = dense_v_all[len(args.prefill_seq_lens) :]

        def run_sdpa_reference():
            return _run_chunked_sdpa_reference(
                q_p_by_seq,
                dense_k_p,
                dense_v_p,
                q_d_by_seq,
                dense_k_d,
                dense_v_d,
                decode_plan,
                args.block_length,
                sm_scale,
            )

        if args.benchmark_reference:
            sdpa_ms, sdpa_output = _event_time_ms(run_sdpa_reference, args.warmup, args.iters)
        else:
            sdpa_output = run_sdpa_reference()
            torch.cuda.synchronize()

        if args.compare_correctness:
            actual_p, actual_d = _split_pod_output(flashinfer_output)
            expected_p, _, expected_d = sdpa_output
            prefill_check = _assert_close("prefill", actual_p, expected_p, args.rtol, args.atol)
            decode_check = _assert_close("decode_vs_first_sdpa_step", actual_d, expected_d, args.rtol, args.atol)
            correctness = {
                "passed": bool(prefill_check["passed"] and decode_check["passed"]),
                "checks": [prefill_check, decode_check],
            }
            if args.fail_on_mismatch and not correctness["passed"]:
                raise AssertionError(json.dumps(correctness, indent=2, sort_keys=True))

    try:
        flashinfer_version = importlib.metadata.version("flashinfer-python")
    except importlib.metadata.PackageNotFoundError:
        flashinfer_version = getattr(flashinfer, "__version__", "unknown")

    num_sdpa_steps = max(
        (int(length) + args.block_length - 1) // args.block_length
        for length in args.prefill_seq_lens
    )
    result = {
        "benchmark": "chunked_sdpa_vs_one_shot_flashinfer_pod",
        "modes": ["chunked_sdpa", "flashinfer_pod"],
        "flashinfer_version": flashinfer_version,
        "device": torch.cuda.get_device_name(device),
        "fluxserve_exact": args.fluxserve_exact,
        "num_sdpa_steps": num_sdpa_steps,
        "sdpa_reference_schedule": "chunked_prefill_plus_decode_each_step",
        "flashinfer_schedule": "one_shot_pod",
        "prefill_seq_lens": args.prefill_seq_lens,
        "prefill_virtual_q_lens": prefill_plan.qo_lens,
        "prefill_virtual_kv_lens": prefill_plan.kv_lens,
        "decode_kv_lens": args.decode_kv_lens,
        "decode_q_lens": decode_q_lens,
        "decode_q_len": args.block_length,
        "decode_chunk_size": None,
        "sdpa_decode_repeats": num_sdpa_steps,
        "flashinfer_decode_repeats": 1,
        "block_length": args.block_length,
        "prefill_batch_size": len(args.prefill_seq_lens),
        "prefill_virtual_batch_size": len(prefill_plan.qo_lens),
        "decode_batch_size": len(args.decode_kv_lens),
        "prefill_tokens": sum(args.prefill_seq_lens),
        "prefill_virtual_query_tokens": sum(prefill_plan.qo_lens),
        "flashinfer_decode_query_tokens": sum(decode_q_lens),
        "chunked_sdpa_decode_query_tokens": sum(decode_q_lens) * num_sdpa_steps,
        "decode_kv_tokens": sum(args.decode_kv_lens),
        "num_q_heads": args.num_q_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "page_size": args.page_size,
        "kv_layout": args.kv_layout,
        "dtype": str(args.dtype).replace("torch.", ""),
        "index_dtype": str(args.index_dtype).replace("torch.", ""),
        "pos_encoding_mode": args.pos_encoding_mode,
        "window_left": args.window_left,
        "causal_prefill": False,
        "return_lse": False,
        "sm_scale": sm_scale,
        "workspace_mb": args.workspace_mb,
        "warmup": args.warmup,
        "iters": args.iters,
        "plan_wall_ms": plan_wall_ms,
        "plan_info_p": plan_info_p,
        "plan_info_d": plan_info_d,
        "plan_debug": {
            "prefill_padded_batch_size": plan_info_p[0],
            "prefill_cta_tile_q": plan_info_p[3],
            "prefill_split_kv": bool(plan_info_p[14]),
            "decode_padded_batch_size": plan_info_d[0],
            "decode_cta_tile_q": plan_info_d[3],
            "decode_split_kv": bool(plan_info_d[14]),
        },
        "flashinfer_total_ms": flashinfer_ms,
        "chunked_sdpa_total_ms": sdpa_ms,
        "sdpa_reference_total_ms": sdpa_ms,
        "speedup_vs_chunked_sdpa": (sdpa_ms / flashinfer_ms) if sdpa_ms is not None else None,
        "speedup_vs_sdpa": (sdpa_ms / flashinfer_ms) if sdpa_ms is not None else None,
        "correctness": correctness,
        "flashinfer_output_shape": _tensor_shapes(flashinfer_output),
        "sdpa_output_shape": _tensor_shapes(sdpa_output) if sdpa_output is not None else None,
    }
    # Backward-compatible field for consumers expecting the original key.
    result["total_ms"] = result["flashinfer_total_ms"]
    result["output_shape"] = result["flashinfer_output_shape"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare chunked SDPA against one-shot FlashInfer "
            "BatchPODWithPagedKVCacheWrapper for block-diffusion attention."
        )
    )
    parser.add_argument("--prefill-seq-lens", type=_parse_int_list, default=(256, 512, 1024, 2048))
    parser.add_argument("--decode-kv-lens", type=_parse_int_list, default=(256, 512, 1024, 2048))
    parser.add_argument(
        "--decode-q-lens",
        type=_parse_int_list,
        default=None,
        help="Compatibility option; if set, every value must equal --block-length.",
    )
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--num-q-heads", type=int, default=16) 
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--kv-layout", choices=("NHD", "HND"), default="NHD")
    parser.add_argument("--dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--index-dtype", type=_parse_index_dtype, default=torch.int32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--workspace-mb", type=int, default=128)
    parser.add_argument("--pos-encoding-mode", default="NONE", choices=("NONE", "ROPE_LLAMA", "ALIBI"))
    parser.add_argument("--window-left", type=int, default=-1)
    parser.add_argument("--sm-scale", type=float, default=None)
    parser.add_argument("--use-fp16-qk-reduction", action="store_true")
    parser.add_argument("--enable-pdl", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--causal-prefill",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compatibility option from the old benchmark; ignored because FluxServe prefill is block-causal.",
    )
    parser.add_argument(
        "--return-lse",
        action="store_true",
        help="Compatibility option from the old benchmark; ignored for correctness comparison.",
    )
    parser.add_argument(
        "--fluxserve-exact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Validate FluxServe block-diffusion shape constraints. Prefill is represented "
            "as virtual query blocks with prefix KV; decode uses one block per request."
        ),
    )
    parser.add_argument("--compare-correctness", action="store_true")
    parser.add_argument("--benchmark-reference", action="store_true")
    parser.add_argument("--fail-on-mismatch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()

    result = run_benchmark(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Nemotron decode CUDA graphs on the FlashInfer backend.

Capture needs a GPU, so what is checked here is the bookkeeping a correct replay
depends on: that the state is planned exactly once with split-KV disabled, and
that each replay rebuilds the CSR page lists from the KV lengths it was handed --
compacting rows of differing page counts with device ops alone, since a host
readback would cost a synchronisation per block step.
"""

from types import SimpleNamespace

import pytest
import torch

from fluxserve.backend.execution.nemotron_cuda_graph_runner import (
    NemotronCudaGraphRunner,
)
from fluxserve.backend.layers.attention.flashinfer_token import (
    FlashInferTokenPagedGraphState,
)
from fluxserve.backend.layers.attention.metadata import PagedAttentionMetadata

BLOCK = 32
PAGES_PER_SEQUENCE = 4


class FakeWrapper:
    """Record plans and hand back a mutable plan layout, as FlashInfer does."""

    def __init__(self, workspace, *, kv_layout, backend, use_cuda_graph=False,
                 qo_indptr_buf=None, paged_kv_indptr_buf=None,
                 paged_kv_indices_buf=None, paged_kv_last_page_len_buf=None):
        assert kv_layout == "NHD" and backend == "fa2"
        assert use_cuda_graph, "graph state must ask for the cuda graph wrapper"
        for buffer in (qo_indptr_buf, paged_kv_indptr_buf,
                       paged_kv_indices_buf, paged_kv_last_page_len_buf):
            assert buffer is not None, "graph state must pass persistent buffers"
        self.buffers = (qo_indptr_buf, paged_kv_indptr_buf,
                        paged_kv_indices_buf, paged_kv_last_page_len_buf)
        self.plans = []
        self.runs = 0
        self._plan_info = [0, 1, 2]

    def plan(self, qo_indptr, kv_indptr, indices, last_page_len, **options):
        self.plans.append((qo_indptr.tolist(), kv_indptr.tolist(),
                           indices.tolist(), last_page_len.tolist(), options))
        assert options.get("disable_split_kv") is True, (
            "one plan can only serve every replay when the schedule ignores "
            "the KV lengths"
        )

    def run(self, q, cache):
        self.runs += 1
        return torch.zeros_like(q)


def graph_state(batch_size=2, device="cpu"):
    return FlashInferTokenPagedGraphState(
        device, batch_size=batch_size, pages_per_sequence=PAGES_PER_SEQUENCE,
        wrapper_factory=FakeWrapper,
    )


def metadata_for(batch_size=2, *, kv_lens=(BLOCK, BLOCK), causal=False,
                 device="cpu"):
    pages = torch.arange(
        batch_size * PAGES_PER_SEQUENCE, dtype=torch.int32, device=device
    ).view(batch_size, PAGES_PER_SEQUENCE)
    return PagedAttentionMetadata(
        phase="decode",
        batch_size=batch_size,
        max_input_len=BLOCK,
        block_length=BLOCK,
        page_size=BLOCK,
        q_lens_cpu=(BLOCK,) * batch_size,
        q_offsets_cpu=(0,) * batch_size,
        q_token_indices=torch.arange(batch_size * BLOCK, device=device),
        qo_indptr=torch.arange(
            batch_size + 1, dtype=torch.int32, device=device
        ) * BLOCK,
        kv_lens=torch.tensor(kv_lens, dtype=torch.int32, device=device),
        page_table=pages,
        slot_mapping=torch.arange(batch_size * BLOCK, device=device),
        max_q_len=BLOCK,
        max_kv_len=BLOCK * PAGES_PER_SEQUENCE,
        causal=causal,
        backend="flashinfer",
    )


def layer_config(num_heads=4, num_kv_heads=2, head_dim=8, scale=0.125):
    return SimpleNamespace(num_heads=num_heads, num_kv_heads=num_kv_heads,
                           head_dim=head_dim, scale=scale)


def warm(state, metadata, *, batch_size=2):
    """One unfrozen run, which is what teaches the state its dtypes."""
    config = layer_config()
    q = torch.zeros(batch_size * BLOCK, config.num_heads, config.head_dim)
    cache = (torch.zeros(8, BLOCK, config.num_kv_heads, config.head_dim),) * 2
    state.run(q, cache, metadata, config)
    return state


# --------------------------------------------------------------------------
# One plan, then device-side refreshes
# --------------------------------------------------------------------------


def test_the_state_is_planned_once_at_the_widest_layout():
    state = warm(graph_state(), metadata_for())
    assert state.planned
    qo_indptr, kv_indptr, indices, last_page_len, options = state.wrapper.plans[-1]
    # Every row claims its full page span, so no later replay can ask for more.
    assert kv_indptr == [0, PAGES_PER_SEQUENCE, 2 * PAGES_PER_SEQUENCE]
    assert indices == list(range(2 * PAGES_PER_SEQUENCE))
    assert last_page_len == [BLOCK, BLOCK]
    assert qo_indptr == [0, BLOCK, 2 * BLOCK]
    assert options["disable_split_kv"] is True

    # Further runs never plan again, whatever the KV lengths do.
    warm(state, metadata_for(kv_lens=(BLOCK * 3, BLOCK * 2)))
    assert len(state.wrapper.plans) == 1
    assert state.wrapper.runs == 2


def test_refresh_compacts_rows_of_different_page_counts():
    # Row 0 spans three pages with a one-token tail, row 1 spans one full page.
    metadata = metadata_for(kv_lens=(BLOCK * 2 + 1, BLOCK))
    state = warm(graph_state(), metadata)
    state.refresh(metadata)
    assert state.kv_indptr_buf.tolist() == [0, 3, 4]
    assert state.last_page_len_buf.tolist() == [1, BLOCK]
    # CSR order: row 0's first three pages, then row 1's first page.
    assert state.kv_indices_buf[:4].tolist() == [0, 1, 2, PAGES_PER_SEQUENCE]


def test_refresh_sends_unowned_pages_to_the_trash_slot():
    metadata = metadata_for(kv_lens=(BLOCK, BLOCK))
    state = warm(graph_state(), metadata)
    state.refresh(metadata)
    # One page each: everything past index 2 is untouched by the read window.
    assert state.kv_indptr_buf.tolist() == [0, 1, 2]
    assert state.kv_indices_buf[:2].tolist() == [0, PAGES_PER_SEQUENCE]
    # The pages neither row owns landed in the slot past the page list, which
    # FlashInfer never reads because kv_indptr stops before it.
    assert state.kv_indices_buf.numel() == 2 * PAGES_PER_SEQUENCE + 1
    assert state._trash_slot == 2 * PAGES_PER_SEQUENCE


def test_refresh_is_idempotent_and_tracks_growth():
    metadata = metadata_for(kv_lens=(BLOCK, BLOCK))
    state = warm(graph_state(), metadata)
    state.refresh(metadata)
    first = state.kv_indices_buf[:2].tolist()
    state.refresh(metadata)
    assert state.kv_indices_buf[:2].tolist() == first

    # A block later, row 0 reaches into its second page.
    metadata.kv_lens.copy_(torch.tensor([BLOCK + 1, BLOCK], dtype=torch.int32))
    state.refresh(metadata)
    assert state.kv_indptr_buf.tolist() == [0, 2, 3]
    assert state.kv_indices_buf[:3].tolist() == [0, 1, PAGES_PER_SEQUENCE]
    assert state.last_page_len_buf.tolist() == [1, BLOCK]
    assert len(state.wrapper.plans) == 1, "growth must not trigger a re-plan"


def test_freezing_or_refreshing_before_a_warmup_run_is_refused():
    state = graph_state()
    with pytest.raises(RuntimeError, match="unplanned state"):
        state.freeze()
    with pytest.raises(RuntimeError, match="after planning"):
        state.refresh(metadata_for())


def test_index_buffer_spans_every_page_a_row_can_reach_plus_the_trash_slot():
    state = graph_state(batch_size=3)
    assert state.kv_indices_buf.numel() == 3 * PAGES_PER_SEQUENCE + 1
    assert state.qo_indptr_buf.numel() == 4
    assert state.last_page_len_buf.numel() == 3
    with pytest.raises(ValueError):
        FlashInferTokenPagedGraphState(
            "cpu", batch_size=0, pages_per_sequence=PAGES_PER_SEQUENCE,
            wrapper_factory=FakeWrapper,
        )


# --------------------------------------------------------------------------
# Backend selection and replay
# --------------------------------------------------------------------------


def test_runner_accepts_both_paged_backends_and_nothing_else():
    assert NemotronCudaGraphRunner((1, 2), backend="fa4").backend == "fa4"
    assert NemotronCudaGraphRunner(
        (1, 2), backend="flashinfer"
    ).backend == "flashinfer"
    with pytest.raises(ValueError, match="fa4"):
        NemotronCudaGraphRunner((1, 2), backend="sdpa")


class StubGraph:
    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


def replay_fixture(*, batch_size=2):
    runner = NemotronCudaGraphRunner((batch_size,), backend="flashinfer")
    metadata = metadata_for(batch_size=batch_size)
    state = warm(graph_state(batch_size=batch_size), metadata,
                 batch_size=batch_size)
    state.freeze()
    cache = SimpleNamespace(
        device=torch.device("cpu"),
        page_size=BLOCK,
        page_table=torch.arange(
            8 * PAGES_PER_SEQUENCE, dtype=torch.int32
        ).view(8, PAGES_PER_SEQUENCE),
        pages_per_sequence=PAGES_PER_SEQUENCE,
        max_length=BLOCK * PAGES_PER_SEQUENCE,
    )
    runner.cache = cache
    graph = StubGraph()
    runner.entries[(batch_size, False)] = SimpleNamespace(
        graph=graph,
        input_ids=torch.zeros(batch_size, BLOCK, dtype=torch.long),
        position_ids=torch.zeros(batch_size, BLOCK, dtype=torch.long),
        metadata=metadata,
        dummy_pages=torch.zeros(batch_size, dtype=torch.int32),
        logits=torch.zeros(batch_size, BLOCK, 7),
        seed=None,
        state=state,
    )
    model_runner = SimpleNamespace(
        past_key_values=cache, block_length=BLOCK,
        decoder=SimpleNamespace(mask_id=5),
    )
    return runner, model_runner, graph, state


def test_replay_refreshes_the_buffers_then_replays_without_planning():
    runner, model_runner, graph, state = replay_fixture()
    plans_before = len(state.wrapper.plans)
    result = runner.replay(
        model_runner,
        input_ids=torch.zeros(2, BLOCK, dtype=torch.long),
        position_ids=torch.arange(BLOCK).repeat(2, 1),
        seq_ids=torch.tensor([1, 3]),
        causal=False,
    )
    assert result is not None
    assert len(state.wrapper.plans) == plans_before, "a replay must not re-plan"
    # The replay set kv_lens from the positions, so the buffers describe one
    # full block per row.
    assert state.kv_indptr_buf.tolist() == [0, 1, 2]
    assert state.last_page_len_buf.tolist() == [BLOCK, BLOCK]
    assert graph.replays == 1
    assert runner.replay_count == 1
    assert runner.fallback_count == 0


def test_fa4_replay_touches_no_flashinfer_state():
    runner, model_runner, graph, state = replay_fixture()
    runner.backend = "fa4"
    entry = runner.entries[(2, False)]
    entry.state = None
    plans_before = len(state.wrapper.plans)
    result = runner.replay(
        model_runner,
        input_ids=torch.zeros(2, BLOCK, dtype=torch.long),
        position_ids=torch.arange(BLOCK).repeat(2, 1),
        seq_ids=torch.tensor([1, 3]),
        causal=False,
    )
    assert result is not None and graph.replays == 1
    assert len(state.wrapper.plans) == plans_before


# --------------------------------------------------------------------------
# What the FlashInfer runner will and will not accept
# --------------------------------------------------------------------------


def decode_graph_config(**overrides):
    config = SimpleNamespace(
        enable_prefill_cuda_graph=False,
        enable_decode_cuda_graph=True,
        decode_cuda_graph_mode="padded",
        page_size=BLOCK,
        block_length=BLOCK,
        cuda_graph_capture_batch_sizes=(1, 2, 4, 8),
        supported_batch_sizes=(1, 2, 4, 8),
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def validate(config, **server):
    from fluxserve.backend.execution.runners.nemotron_flashinfer import (
        NemotronFlashInferDiffusionRunner,
    )

    args = SimpleNamespace(
        dp_size=1, pp_size=1, tp_size=1, ep_size=1, max_num_seqs=8,
    )
    for name, value in server.items():
        setattr(args, name, value)
    return NemotronFlashInferDiffusionRunner._validate_decode_graph_config(
        config, {"server_args": args}, ()
    )


def test_a_documented_decode_graph_config_is_accepted():
    validate(decode_graph_config())


@pytest.mark.parametrize("config_kwargs, server_kwargs, message", [
    ({"decode_cuda_graph_mode": "decomposed"}, {}, "padded"),
    ({"page_size": BLOCK * 2}, {}, "page_size == block_length"),
    ({"cuda_graph_capture_batch_sizes": (1, 2)}, {}, "cover max_num_seqs"),
    ({}, {"dp_size": 2}, "DP/PP=1"),
    ({}, {"ep_size": 2}, "TP=EP"),
])
def test_decode_graph_gates_match_fa4(config_kwargs, server_kwargs, message):
    with pytest.raises(ValueError, match=message):
        validate(decode_graph_config(**config_kwargs), **server_kwargs)

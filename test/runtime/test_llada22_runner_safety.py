from types import SimpleNamespace

import pytest
import torch

from fluxserve.backend.execution.runners.block_diffusion import BlockDiffusionRunner
from fluxserve.backend.execution.runners.utils import generated_eos_hit
from fluxserve.backend.execution.decoders.levenshtein import LevenshteinJointDecoder


@pytest.mark.parametrize('eos', [13, 16])
def test_eos_only_in_generated_committed_region(eos):
    tokens = torch.tensor([
        [eos, 1, 2, 3, 4, 5],  # prompt EOS
        [1, 2, eos, 3, 4, 5],  # generated EOS
        [1, 2, 3, 4, eos, 5],  # future / uncommitted EOS
    ])
    assert generated_eos_hit(
        tokens, (13, 16), torch.tensor([2, 2, 2]), torch.tensor([4, 4, 4])
    ).tolist() == [False, True, False]


def test_editing_selection_does_not_deprioritize_mask_free_rows():
    runner = BlockDiffusionRunner.__new__(BlockDiffusionRunner)
    runner.decoder = SimpleNamespace(needs_row_state=True)
    x = SimpleNamespace(data=torch.tensor([[1, 2], [12, 12], [12, 12]]))
    valid = torch.tensor([True, False, True])
    assert runner._select_decode_sequences(x, valid, 12, 1).tolist() == [0]
    runner.decoder = SimpleNamespace()  # existing decoders keep mask-count order
    assert runner._select_decode_sequences(x, valid, 12, 1).tolist() == [2]


def test_runner_owns_global_row_state_and_levenshtein_iteration_cap():
    runner = BlockDiffusionRunner.__new__(BlockDiffusionRunner)
    runner.device = 'cpu'
    runner.block_length = 4
    runner.runner_config = SimpleNamespace(max_post_steps=2)
    runner.decoder = LevenshteinJointDecoder(.5, .0, max_steps_per_block=9)
    budget, state = runner._make_decode_loop_state(5)
    assert budget.max_block_iters == 10
    ids = torch.tensor([4, 1])
    kwargs = runner._decoder_editing_kwargs(ids, torch.tensor([1, 2, 3, 4, 5]), budget, state)
    assert kwargs['row_state'] is state
    assert kwargs['seq_ids'].tolist() == [4, 1]
    assert kwargs['prompt_lengths'].tolist() == [5, 2]


@pytest.mark.parametrize('runner_kind', ['dense', 'flashinfer'])
@pytest.mark.parametrize('eos', [13, 16])
def test_runner_commits_final_input_and_ignores_prompt_eos(runner_kind, eos):
    """Exercise real runner iterations with a scripted model and KV sink."""
    from fluxserve.backend.execution.runners.flashinfer_diffusion import FlashInferDiffusionRunner

    class Tokens:
        def __init__(self, data):
            self.data = data

        def select_seqs(self, ids):
            return Tokens(self.data[ids].clone())

        def __getitem__(self, ids):
            return self.data[ids]

        def __setitem__(self, ids, value):
            self.data[ids] = value

    cls = BlockDiffusionRunner if runner_kind == 'dense' else FlashInferDiffusionRunner
    runner = cls.__new__(cls)
    runner.device = 'cpu'
    runner.block_length = 4
    runner.max_length = 8
    runner.num_forwards = 0
    runner.early_stop = True
    runner.runner_config = SimpleNamespace(max_post_steps=20, max_cache_length_align=8)
    runner.decoder = LevenshteinJointDecoder(
        .5, .0, mask_id=12, eos_id=13, eos_ids=(13, 16),
        delete_token_id=14, split_token_id=15,
        steps=4, max_post_steps=20, max_steps_per_block=3,
    )
    runner.past_key_values = torch.zeros(1, 2, 1, 1, 8, 1)
    runner.flashinfer_graph_runner = None
    runner._use_flashinfer_paged_cache = lambda: False
    runner._make_forward_batch = lambda *a, **kw: None
    runner._make_decode_forward_batch = lambda *a, **kw: None
    committed = []

    def model(tokens, **kw):
        logits = torch.full((*tokens.shape, 20), -10.)
        # Always propose SPLIT with a plain-token fallback; the hard cap
        # must force resolve and then spend one stability forward.
        logits[..., 15] = 10.
        logits[..., 3] = 9.
        return SimpleNamespace(logits=logits, input_tokens=tokens.clone())

    runner.model = model

    def commit(output, ids, starts, finished, *args):
        for local in finished.nonzero(as_tuple=True)[0].tolist():
            committed.append((int(starts[ids[local]]), output.input_tokens[local].clone()))

    runner._update_finished_kv_cache = commit
    x = Tokens(torch.tensor([[eos, 2, 12, 12, 12, 12, 12, 12]]))
    starts = torch.tensor([0])
    prompts = torch.tensor([2])
    if runner_kind == 'dense':
        runner._decode_batches(x, starts, 8, None, 1, 1, prompt_lengths=prompts)
    else:
        budget, state = runner._make_decode_loop_state(1)
        for _ in range(2 * runner.decoder.max_block_iters):
            if starts[0] >= 8:
                break
            runner._decode_selected_batch(
                x, torch.tensor([0]), starts, 8, 8, None, 1,
                prompt_lengths=prompts, edit_budget=budget, row_state=state,
            )
    assert starts.tolist() == [8]
    assert [start for start, _ in committed] == [0, 4]
    for start, final_input in committed:
        assert torch.equal(final_input, x.data[0, start:start+4])
    assert x.data.tolist() == [[eos, 2, 3, 3, 3, 3, 3, 3]]
    assert runner.num_forwards == 2 * runner.decoder.max_block_iters


@pytest.mark.parametrize('eos', [13, 16])
@pytest.mark.parametrize('ignore_eos', [False, True])
@pytest.mark.parametrize('transient', [False, True])
def test_online_publishes_only_stable_block(eos, ignore_eos, transient):
    from fluxserve.backend.engine.request import RequestState
    from fluxserve.backend.execution.runners.flashinfer_diffusion import FlashInferDiffusionRunner
    from fluxserve.backend.execution.runners.utils import gather_blocks

    runner = FlashInferDiffusionRunner.__new__(FlashInferDiffusionRunner)
    runner.device = 'cpu'
    runner.block_length = 4
    runner.server_args = SimpleNamespace(max_num_seqs=5)
    runner.runner_config = SimpleNamespace(max_post_steps=10)
    runner.model = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(num_hidden_layers=1)))
    runner.decoder = LevenshteinJointDecoder(
        .5, .0, mask_id=12, eos_id=13, eos_ids=(13, 16),
        delete_token_id=14, split_token_id=15, max_post_steps=10,
    )
    request = RequestState(rid='r', input_ids=[eos, 2], max_new_tokens=8, ignore_eos=ignore_eos)
    calls = 0

    def decode(x, ids, starts, *args, prompt_lengths, edit_budget, row_state):
        nonlocal calls
        assert request.output_ids == []  # no intermediate block is published
        selected = x.select_seqs(ids)
        before = gather_blocks(selected.data, starts[ids], 4)
        target = [eos, 2, 3, 4 if transient and calls > 0 else eos]
        logits = torch.full((1, 4, 20), -10.)
        logits.scatter_(-1, torch.tensor(target).reshape(1, 4, 1), 10.)
        runner.decoder.batch_decode(
            logits, starts[ids], selected, 4,
            **runner._decoder_editing_kwargs(ids, prompt_lengths, edit_budget, row_state),
        )
        after = gather_blocks(selected.data, starts[ids], 4)
        finished = (~(before == 12).any(dim=1)) & (~(before != after).any(dim=1))
        starts[ids] += finished.long() * 4
        x[ids] = selected.data
        calls += 1

    runner._decode_selected_batch = decode
    results = runner._execute_paged_decode(
        SimpleNamespace(request_ids=['r']), 0, [3], {'r': request},
        SimpleNamespace(decode=lambda ids, **kw: ' '.join(map(str, ids))),
    )
    assert len(results) == 1
    result = results[0]
    assert result.decode_block_completed
    assert calls == (3 if transient else 2)
    assert result.token_ids == ([3, 4] if transient else [3, eos] if ignore_eos else [3])
    assert result.finished == (not transient and not ignore_eos)
    assert result.finish_reason == ('stop' if result.finished else None)

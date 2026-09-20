# LLaDA2.2 integration with the FA4 upstream

## Local integration

The local `llada2.2-fa4-rebase` branch replays all five LLaDA2.2 commits
onto upstream `b6b9d7b` (FA4 support). The original `llada2.2` branch remains
at `1ed5f8d`. No remote branch or pull request was updated.

The following documents retain their complete original-branch contents:

- `docs/serving/llada2.1.md`
- `docs/serving/llada2.2.md`
- `dev-notes/llada2.1-model-support-development-guide.md`
- `dev-notes/llada2.2-model-support-development-guide.md`

Their FlashInfer-specific graph instructions remain the previously documented
path. This integration adds FA4 decoder-state plumbing; it does not establish
device support or replace those instructions with validated FA4 commands.

## Compatibility fixes

- Allocate, refresh, and pad Levenshtein state buffers in FA4 decode graphs.
- Transport the fifth graph-step result (mutable decoder state), broadcast it
  along with tokens and completion predicates, and commit it outside replay.
- Preserve per-request offline decode limits and the structural decoder's
  selection order when DELETE/SPLIT changes the number of masks.
- Check both checkpoint EOS IDs only in committed generated positions and
  apply the same stop set when publishing online output.

CPU regressions cover eager and fused-tail iteration to the hard cap, the
final forward's input tokens, prompt protection, per-request limits, refreshed
state and inert padding, simulated rank synchronization before state commit,
and stable online publication for both EOS IDs and `ignore_eos` settings.
The replay tests use CPU adapters; they do not execute CUDA graphs or FA4.

## Offline validation

Validation used the existing `fluxserve` Conda environment, Python's local
`python/` source path, and PyTorch `2.8.0+cu129`. CUDA was unavailable.

```bash
export OMP_NUM_THREADS=2
export PYTHONPATH=python
export FLUXSERVE_LLADA22_REF_DIR=/path/to/LLaDA2.2-flash/snapshots/2e48107b9ee9c015d18b6269088388d64d6c1289
python -m pytest -q
```

Result: **322 passed, 16 skipped**. The reference directory contained the
checkpoint configuration and modeling code at the revision above.
`git diff --check` passed. A direct Git comparison confirmed that all four
preserved documents are identical to `llada2.2`.

## Pending device acceptance

Before claiming FA4 support for LLaDA2.2, run real-weight eager and padded
CUDA-graph decoding, changing live batch sizes between replays. Verify final
KV contents, graph replay counts, both EOS IDs, DELETE/SPLIT behavior, and
TP/EP rank consistency. Run the existing quality evaluation against the same
checkpoint revision and retain the legacy LLaDA2.0/2.1 regression results.
No GPU correctness, quality, or performance result is claimed here.

# Nemotron 3B / 8B / 14B validation

All four harnesses accept `--model` and `--revision`. Official model IDs default
to the immutable revision recorded under `../data/nemotron_checkpoints/`.
Use a separate output directory for each model and revision. Comparisons reject
artifacts from different checkpoints.

CPU regressions (no weights or network required):

```bash
python -m pytest test/runtime test/ci_system -q
```

The following commands require a GPU, checkpoint storage and the matching
reference/runtime dependencies. Run from the repository root. Select `3B`, `8B`
or `14B`; the scripts share input tokens, not expected answers.

```bash
model_size=3B
model_id="nvidia/Nemotron-Labs-Diffusion-${model_size}"
result_dir=".ci-artifacts/nemotron-${model_size}"

# Reference AR versus FluxServe: logits, KV cache, RoPE and greedy tokens.
python test/runtime/integration/nemotron_ar_harness.py \
  --model "$model_id" --output "$result_dir/ar"

# Reference diffusion, dense, FA4 and FlashInfer use separate processes.
for lane in reference dense fa4 flashinfer selfspec_fa4 selfspec_flashinfer; do
  python test/runtime/integration/nemotron_diffusion_harness.py \
    --model "$model_id" --mode "$lane" --output "$result_dir/diffusion"
done
python test/runtime/integration/nemotron_diffusion_harness.py \
  --model "$model_id" --mode compare --require-flashinfer \
  --output "$result_dir/diffusion"

# Offline baseline, continuously scheduled serving, and decode CUDA graphs.
python test/runtime/integration/nemotron_online_harness.py \
  --model "$model_id" --mode offline --output "$result_dir/online"
python test/runtime/integration/nemotron_online_harness.py \
  --model "$model_id" --mode serve --output "$result_dir/online"
python test/runtime/integration/nemotron_online_harness.py \
  --model "$model_id" --mode serve --graphs --output "$result_dir/online"
python test/runtime/integration/nemotron_online_harness.py \
  --model "$model_id" --mode compare --output "$result_dir/online"

# YaRN and query scaling beyond the original context window.
python test/runtime/integration/nemotron_long_context_harness.py \
  --model "$model_id" --output "$result_dir/long-context"
```

The diffusion harness also accepts `--lora` and `--thinking-budget`. The online
harness accepts `--backend flashinfer`, `--decoding self_speculation`,
`--thinking-budget`, and `--extension-fixtures`; use its `--help` for lane labels
and matching offline baselines.

Each size has one GSM8K CI recipe under `test/ci/1N1G/eval/`: threshold
decoding on FA4 with decode CUDA graphs at TP=1, the configuration whose
accuracy has been measured. The three recipes run when a PR targeting `main`
gets the `ci:run-1N1G` label, and on subsequent PR updates while that label
remains. A push to `nemotron` without a labeled PR does not trigger these GPU
jobs.

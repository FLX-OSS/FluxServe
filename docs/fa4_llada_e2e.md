# LLaDA2.1-mini FA4 end-to-end validation

Validated on Delta AI GH200, job 3107293, using the cached official
`inclusionAI/LLaDA2.1-mini` checkpoint (snapshot
`20e64e2ad21644d0e5248586ed9c942cdd45de0f`). The model runs unquantized BF16,
TP/DP/EP/PP=1, block/page=64, eager execution. Standalone FA4 is 4.0.0b29;
the runtime also needs the project `flux-kernel/python` and native
`flux-scheduler` installation. Use the project's container and mounted model
cache, with those dependencies available on PYTHONPATH.

## Implementation changes

The FA4 runner now handles `HierarchyDecoder`, which exposes single-request
`decode()` instead of `batch_decode()`. It applies the decoder independently
to each row while retaining batched model forwards. Threshold decoding keeps
its existing batched API. Online prefill/decode execute under `no_grad`.

The final unmasked forward of each block is intentionally retained: it refreshes
the cached K/V from the final tokens before the next block starts. A two-row
regression test covers this transition with both decoders. The implementation
uses FluxServe's own model, scheduler, cache and decoder interfaces.

## Reproduce offline generation

```bash
python test/runtime/integration/run_fa4_llada_e2e.py \
  --model inclusionAI/LLaDA2.1-mini \
  --output /tmp/fa4_llada_e2e.json
```

This loads the model once and shares identical weights between the FA4 runner
and the existing dense SDPA runner. The checkpoint chat template is used.
Three task checks are required: exact `4`, exact `Paris`, and the complete
integer sequence 1 through 40. The long output exceeds 64 tokens. Decode
position traces verify progression across blocks, including final KV refresh.
`early_stop=False` deliberately exercises later blocks; this is not an EOS
test (EOS is tested separately through HTTP).

The validator also checks every real FA4 attention invocation along the
generation trajectory against independent FP32 math SDPA and BF16 flash SDPA
on the same Q/K/V. Acceptance requires finite output, FP32 relative L2 <=1%,
and max/L2 errors no greater than twice the BF16 SDPA control error plus 1e-5.
These are numerical accuracy bounds, not a whole-model logit identity claim.

The completed run checked **620 FA4 calls**. Maximum FP32-reference relative
L2 was **0.00187054 (0.1871%)**, with BF16 SDPA control **0.00187054**.
Maximum absolute error was 0.230049 on the real, non-unit-scale activations.
The raw run record is `test/runtime/integration/profiles/fa4_llada_e2e.json`;
it is a generated local artifact, not required to run the tests.

A fixed elementwise FP32 `atol=rtol=.02` was unsuitable for the real activations:
at a failing case FA4 and BF16 SDPA both had max error 0.1569519 and relative
L2 approximately 0.001762. Cancellation can give large relative errors at a
few near-zero output elements. The validator reports measured maximum absolute
and relative L2 errors, and retains the BF16 control measurement.

Cross-backend token identity is recorded for each task. `4` and `Paris` match
exactly; counting differs by a comma near 39/40. `--require-token-match`
additionally enforces exact identity and therefore is stricter than the task
correctness test. The separate existing random-input prefill-logits test is
unchanged and still fails; no threshold was relaxed in that test.

## Reproduce real HTTP tests

```bash
FLUXSERVE_RUN_FA4_E2E=1 python -m pytest -s -q \
  test/runtime/integration/test_fa4_llada_server.py
```

The test starts and stops a real FluxServe HTTP server for each of threshold
and hierarchy decoding. It exercises the native paged scheduler with two
concurrent requests (different prompt lengths), then reused slots/pages for
multi-block counting and a length-limited request. It uses `--apply-template`
to match the model chat template. Without this flag, the legacy renderer uses
different input tokens; in the observed counting request it stopped before 40.

Observed HTTP outputs for both decoders:

| Request | Output | Finish reason |
|---|---|---|
| 2 + 2 | `4` | stop |
| Capital of France, with context | `Paris` | stop |
| Count 1 through 40 | Complete sequence `1, 2, …, 39, 40` | stop |
| Count upwards, max_tokens=8 | `1, 2, 3,` | length |

The HTTP regression suite passed 2 tests; the CPU metadata/adapter/selector
and runner regression suite passed 17 tests. This is functional validation
of these specific tasks, not a general reasoning-quality benchmark or proof
of parity on every LLaDA2.x variant. Full-runner FP16, distributed execution,
online CUDA Graph, streaming/disconnect stress and broader quality evaluation
remain outside this validation.

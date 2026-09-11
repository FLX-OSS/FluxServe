# Architecture

FluxServe separates request handling, scheduling, and GPU execution. Its primary packages are the Python serving runtime, the native Flux Scheduler, and Flux Kernel.

## Request lifecycle

1. An HTTP request arrives at the serving runtime.
2. The runtime tokenizes the input and submits request state to the scheduler.
3. The paged scheduler creates execution plans for prefill and active generation blocks.
4. Model runners execute attention and decoding on the GPU and update the KV cache.
5. The runtime returns generated text through the completion endpoint.

## Serving runtime

The [HTTP entrypoint](../python/fluxserve/backend/entrypoints/http_server.py) provides health checks and completion endpoints, including `/v1/completions` and `/v1/chat/completions`. The [CLI](../python/fluxserve/cli.py) configures model loading, scheduler policy, cache layout, decoding, and distributed execution.

## Block-level scheduling

The [scheduler adapter](../python/fluxserve/backend/engine/scheduler_adapter.py) connects Python request state to the native scheduler. The paged execution path manages cache pages and schedules variable-length prefill and generation work.

For the documented paged configuration, use `--scheduler-policy paged`, `--kv-cache-layout paged`, and `--attention-backend flashinfer` together.

## Attention and decoding

Diffusion generation resolves multiple token positions in an active block. Block-causal attention allows the active block to use the prompt and previously committed blocks.

The threshold decoder is used in the LLaDA2.0 guides. LLaDA2.1 also supports the opt-in `joint_threshold` decoder, which can edit already resolved tokens inside the active block. See the [LLaDA2.1 guide](serving/llada2.1.md) for its constraints. LLaDA2.2 adds the opt-in `levenshtein_joint` decoder, which additionally consumes `DELETE`/`SPLIT` edit tokens inside the fixed-length block; see the [LLaDA2.2 guide](serving/llada2.2.md).

## Distributed execution

FluxServe launches local workers for multi-GPU configurations. Tensor parallelism distributes model computation, expert parallelism distributes MoE experts, and data parallelism distributes request work.

Start with the [four-GPU LLaDA2.0-flash configuration](serving/llada2-flash.md), which uses `TP=4`, `EP=4`, and `DP=1`. GPU memory requirements depend on the checkpoint, cache capacity, context length, and concurrency.

## Measure your workload

Use the [benchmarking guide](guides/benchmark.md) to evaluate your model and serving configuration. Results depend on the hardware, dataset, request rate, and decoding settings.

# FluxServe

FluxServe is a lightweight serving engine for diffusion language models. It combines block-level scheduling, attention kernels, and distributed execution to serve autoregressive (AR) diffusion models on NVIDIA GPUs.

## Start here

- [Quickstart](guides/quickstart.md): launch a server and send your first request.
- [Docker installation](guides/getting_started.md): build the CUDA environment and install FluxServe.
- [Architecture](architecture.md): understand how requests move through the engine.
- [Benchmarking](guides/benchmark.md): measure offline and online performance.

## Core capabilities

### Block attention

Block-causal attention supports variable-length prefill and block decoding. The paged FlashInfer path reuses committed KV cache entries; CUDA graph execution is available through explicit launch options.

### Dynamic scheduling

The native C++ scheduler manages requests at block granularity. The Python runtime consumes execution plans and coordinates model execution and cache updates.

### Multi-GPU execution

Tensor, data, and expert parallelism support larger serving configurations. Use a documented combination for your model and hardware; the parallelism sizes are not independently interchangeable.

## Deployment guides

- [LLaDA2.0-mini on one GPU](serving/llada2-mini.md)
- [LLaDA2.0-flash on four GPUs](serving/llada2-flash.md)
- [LLaDA2.1 decoding presets](serving/llada2.1.md)
- [LLaDA2.2-flash on four GPUs](serving/llada2.2.md)

## Project

FluxServe is developed by [FLX-OSS](https://github.com/FLX-OSS). Source code is available on [GitHub](https://github.com/FLX-OSS/FluxServe) under the [MIT license](../LICENSE).

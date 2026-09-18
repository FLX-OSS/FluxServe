<p align="center">
  <img src="./assets/logo.png" alt="FluxServe" style="width: 640px; object-fit: contain;" />
</p>


## About
**FluxServe** is a lighweight and high-performance serving engine for diffusion langauge models. It is designed and implemented to deliver low-latency and high-throughput inference for autoregressive (AR) diffusion models across different setups, ranging from single GPU batched inference to multi-GPU distributed serving.

Its core features include:

- **Native Block-Causal Attention**: Provides efficient flashinfer runtime with a block-casual attention mechanism suitable for AR diffusion in real-world scenarios, including *varlen prefill* and *varlen block-deocde* with CUDA graph support.
- **Dynamic Request Scheduler**: Provides scheduler with low-overhead C++ control plane and Python execution plane with fine-grained block-level request management suitable for block diffusion models.
- **Efficient Multi-GPU Serving**: Provides tensor paralllel (TP), data parallel (DP) and expert parallel (EP) support for large-scale models.

## News
- 2026/9/21: FluxServe v0.1 is officially open-sourced. We provide support for LLaDA 2.0/2.1 and Diffusion-Gemma models.



## [Getting Started](docs/guides/getting_started.md)

## [Development Roadmap](docs/roadmap.md)

## Performance Results
<img src="./assets/figures/result.png" alt="FluxServe vs. SGLang-dLLM on LLaDA-2.0-mini/flash" width="960px" margin="10px"></img>


## Acknowledgments
We learned the system design and reused code from the following projects: [vllm](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [TokenSpeed](https://github.com/lightseekorg/tokenspeed), [dInfer](https://github.com/inclusionAI/dInfer), [FlashInfer](https://github.com/flashinfer-ai/flashinfer/pull/2722), and [Flash-Attention](https://github.com/dao-ailab/flash-attention).

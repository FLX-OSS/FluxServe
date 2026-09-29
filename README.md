<p align="center">
  <img src="./assets/logo.png" alt="FluxServe" style="width: 640px; object-fit: contain;" />
</p>

## News
- [2026/09] 🦩 FluxServe is announced — a flexible and high-performance inference engine for diffusion language models. [[blog](https://flx-oss.github.io/blog/introducing-fluxserve/)]


## About
**FluxServe** is a lighweight and high-performance serving engine for diffusion langauge models. It is designed and implemented to deliver low-latency and high-throughput inference for autoregressive (AR) diffusion models across different setups, ranging from single GPU batched inference to multi-GPU distributed serving.

Its core features include:

- **Native Block-Causal Attention**: Provides efficient attention runtime with a block-casual attention mechanism suitable for AR diffusion in real-world scenarios, including *varlen prefill* and *varlen block-deocde* with Flashinfer and FA4 support.
- **Dynamic Request Scheduler**: Provides scheduler with low-overhead C++ control plane and Python execution plane with fine-grained block-level request management suitable for block diffusion models.
- **Open Model Support**: Provides native support for [LLaDA2.X](https://github.com/inclusionAI/LLaDA2.X), [Diffusion-Gemma](https://huggingface.co/google/diffusiongemma-26B-A4B-it), and [Nemotron-Labs-Diffusion](https://huggingface.co/collections/nvidia/nemotron-labs-diffusion).
- **Efficient Multi-GPU Serving**: Provides tensor paralllel (TP), data parallel (DP) and expert parallel (EP) support for large-scale models.

## [Getting Started](docs/guides/getting_started.md)

## Performance Results
<img src="./assets/figures/online_throughput.png" alt="FluxServe vs. SGLang on LLaDA-2.0/2.1" width="800px" margin="10px"></img>

## Acknowledgments
We learned the system design and reused code from the following projects: [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [TokenSpeed](https://github.com/lightseekorg/tokenspeed), [dInfer](https://github.com/inclusionAI/dInfer), [FlashInfer](https://github.com/flashinfer-ai/flashinfer/pull/2722), and [Flash-Attention](https://github.com/dao-ailab/flash-attention).


## Citation

```bibtex
@misc{fluxserve2026,
  author = {{FluxServe Team}},
  title = {FluxServe: A Flexible and High-Performance Inference Engine for Open Diffusion Language Models},
  year = {2026},
  month = {September},
  howpublished = {\url{https://github.com/FLX-OSS/FluxServe}}
}
```
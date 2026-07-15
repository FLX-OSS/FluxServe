<p align="center">
  <img src="./assets/logo.png" alt="FluxServe" style="width: 640px; object-fit: contain;" />
</p>

--------------------------------------------------------------------------------

## About
**FluxServe** is a lighweight and high-performance serving engine for diffusion langauge models. 
It is designed and implemented to deliver low-latency and high-throughput inference for AR diffusion models across different setups, ranging from single GPU batched inference to multi-GPU distributed serving.

Its core features include:

- **Block Causal Attention**: Provides efficient attention runtime with a block-casual attention mechanism suitable for block-diffusion LLMs in real-world workloads, including *varlen prefill* and *varlen deocde*.
- **Dynamic Scheduler**: Provides scheduler with efficient C++ control plane and Python execution plane with fine-grained block-level request management suitable for block diffusion models.


## [Getting Started](docs/guides/getting_started.md)


## [Development Roadmap](docs/roadmap.md)

## Performance Results
Detailed benchmark guides can be found [here](docs/guides/benchmark.md).
### Batched Inference 
<table align="center">
  <tr>
    <td>
      <img src="./assets/figures/fig1.png" style="width: 600px; object-fit: contain;" />
    </td>
  </tr>
</table>

### Online Serving

## Acknowledgments
We learned the system design and reused code from the following projects: [vllm](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [TokenSpeed](https://github.com/lightseekorg/tokenspeed), and [dInfer](https://github.com/inclusionAI/dInfer).

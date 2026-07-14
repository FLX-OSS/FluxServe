<p align="center">
  <img src="./assets/logo.png" alt="FluxServe" style="width: 720px; object-fit: contain;" />
</p>

## About
**FluxServe** is a lighweight and high-performance serving engine for diffusion langauge models. 
It is designed and implemented to deliver low-latency and high-throughput inference for AR diffusion models across different setups, ranging from single GPU batched inference to multi-GPU distributed serving.
Its core features include:

- **Block Mix Attention**: Provides efficient attention runtime with a hybrid attention mechanism for different workloads, including *varlen prefill*, *varlen deocde*, and *varlen prefill/decode mix*.
- **Dynamic Scheduler**: Provides scheduler with efficient C++ control plane and Python execution plane with fine-grained block-level request management suitable for block diffusion models.

<!-- ## TO-DO-LISTS

- [ ] Add Batched Inference Results
  - [ ] Add Diffuflex Baseline
- [ ] Flux-Engine
- [ ] Flux-Scheduler
  - [ ] Add Block-level Scheduling Support
  - [ ] Prefix Cache
  - [ ] Add Goodput/Throughput results
- [ ] Final Code Reorg -->

## [Getting Started](docs/guides/getting_started.md)


## [Development Roadmap](docs/roadmap.md)

## Performance Results

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
We learned the system design and reused code from the following projects: [SGLang](https://github.com/sgl-project/sglang), [TokenSpeed](https://github.com/lightseekorg/tokenspeed), and [dInfer](https://github.com/inclusionAI/dInfer).

# Preview checklist

Historical development checklist; use the [public documentation](../docs/index.md) for current setup and serving instructions.

## Functions

### Model Support
- [x] LLaDA2.X
- [x] DiffusionGemma


### Engine Support
- [x] Kernel
    - [x] Attention
        - [x] Torch-SDPA
        - [x] Torch-Flex
        - [x] Flashinfer-Paged
        - [x] Flash-Attn-4
    - [x] MoE Fused Kernels
        - [x] sgl-kernel
    - [x] Flux-Kernels
        - [x] RMSNorm
    - [x] CUDA Graph
- [x] Distributed Execution
    - [x] TP + EP (TP=EP)
    - [x] DP + EP (DP=EP)
- [x] Flux-Scheduler
    - [x] FlashInfer + Paged KV

### Benchmark Support
- [x] Offline Benchmark Scripts
- [x] Online Benchmark Scripts


### Env Support
- [x] Dockerfile build
- [x] UV build (Deprecated)


## Eval Support
- [x] [EvalScope-Perf](https://github.com/modelscope/evalscope)
- [x] [EvalScope-Acc](https://github.com/modelscope/evalscope)



### PR Support
- [x] CI/CD Suites



## Performance Results
- [x] Offline Benchmark
- [x] Online Benchmark

## Documentation
- [x] Getting Started
- [x] Env Setup (Docker)
- [x] Benchmark Guide
- [x] Bug Report
- [ ] Contribution Guide
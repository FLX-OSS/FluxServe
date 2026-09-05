## Functions

### Model Support
- [x] LLaDA2.0
  - [x] [LLaDA2.0-mini](https://huggingface.co/inclusionAI/LLaDA2.0-mini)
  - [x] [LLaDA2.0-flash](https://huggingface.co/inclusionAI/LLaDA2.0-flash)
- [x] LLaDA2.1
  - [x] [LLaDA2.1-mini](https://huggingface.co/inclusionAI/LLaDA2.1-mini)
  - [x] [LLaDA2.1-flash](https://huggingface.co/inclusionAI/LLaDA2.1-flash)
- [ ] LLaDA2.2
  - [ ] [LLaDA2.2-flash](https://huggingface.co/inclusionAI/LLaDA2.2-flash)
- [x] DiffusionGemma


### Engine Support
- [x] Kernel
    - [x] Attention
        - [x] Torch-SDPA
        - [x] Torch-Flex
        - [x] Flashinfer-Paged
    - [x] Quantizaion
        - [ ] FP8
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
- [ ] CI/CD Suites (WIP)

## Docker Support
- [ ] Docker Build Files


## Performance Results
- [x] Offline Benchmark
- [x] Online Benchmark

## Documentation
- [x] Getting Started
- [x] Env Setup (Docker)
- [x] Benchmark Guide
- [x] Bug Report
- [x] Contribution Guide
- [ ] Github Pages
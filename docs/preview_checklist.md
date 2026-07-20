## Functions

### Model Support
- [x] LLaDA2.0
  - [x] [LLaDA2.0-mini](https://huggingface.co/inclusionAI/LLaDA2.0-mini)
  - [x] [LLaDA2.0-flash](https://huggingface.co/inclusionAI/LLaDA2.0-flash)
  - [ ] [LLaDA2.0-flash-fp8](https://huggingface.co/thnkinbtfly/llada2.0-flash-fp8): Full FP8 Support

### Engine Support
- [x] Kernel
    - [ ] Attention
        - [ ] Torch-SDPA
        - [ ] Torch-Flex
        - [ ] Flashinfer-Ragged
        - [ ] Flashinfer-Paged
    - [ ] MoE Fused Kernels
        - [ ] sgl-kernel
    - [x] Flux-Kernels
        - [x] RMSNorm
- [ ] Distributed Execution
    - [x] TP + EP (TP=EP)
    - [ ] DP + EP (DP=EP)
- [x] Flux-Scheduler
    - [x] FlashInfer + Paged KV

### Benchmark Support
- [x] Offline Benchmark Scripts
- [x] Online Benchmark Scripts


### Env Support
- [ ] Dockerfile build
    - [ ] RTX Pro 4000
    - [ ] GH200
- [ ] UV build

### Debugging Support
- [ ] CI/CD


## Performance Results
- [ ] Offline Benchmark
- [ ] Online Benchmark

## Documentation
- [x] Getting Started
- [x] Env Setup (Docker)
- [x] Benchmark Guide
- [ ] Bug Report
- [ ] Contribution Guide
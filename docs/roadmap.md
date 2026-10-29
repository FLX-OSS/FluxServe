# Development planning notes
- 
### Model Support
- [ ] [Nemotron-Labs-Diffusion](https://huggingface.co/collections/nvidia/nemotron-labs-diffusion) (WIP)
- [x] [I-DLM](https://huggingface.co/collections/yifanyu/introspective-diffusion-language-models-i-dlm-69d82ab6948499be9c1f81fb) (8B and 32B, ISD decoding; lossless R-ISD with the gated LoRA on Qwen3-8B)

### Kernel Support
- [ ] FA4 for Diffusion-Gemma

### Quantization Support
- [ ] FP8 (WIP)
- [ ] NVFP4

### Hardware Support
- [ ] NVIDIA Blackwell Support
- [ ] AMD GPU ROCm

### Engine Support
- [ ] Model Parallelism
  - [ ] Pipeline Parallelism
- [ ] Cache Optimization
  - [ ] Prefix Cache (WIP)
- [ ] MoE Optimization
  - [ ] DeepEP
  - [ ] EPLB
- [ ] Model Gateway
  - [ ] gRPC

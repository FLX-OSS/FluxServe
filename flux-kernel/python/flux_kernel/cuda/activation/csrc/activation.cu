/*
 * Copyright (c) 2024 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0.
 * Adapted from SGL Kernel 0.3.15 (8c9670375).
 */
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <flashinfer/activation.cuh>
#include <torch/all.h>
#include <torch/library.h>

template <typename T>
__device__ __forceinline__ float silu(const float& x) {
  return x / (1.0f + expf(-x));
}

torch::Tensor silu_and_mul(torch::Tensor out, torch::Tensor input) {
  TORCH_CHECK(input.is_cuda() && out.is_cuda(), "silu_and_mul requires CUDA tensors");
  TORCH_CHECK(input.is_contiguous() && out.is_contiguous(), "tensors must be contiguous");
  TORCH_CHECK(input.scalar_type() == out.scalar_type(), "input/out dtype mismatch");
  TORCH_CHECK(input.size(-1) == 2 * out.size(-1), "invalid output width");
  const int d = input.size(-1) / 2;
  const int64_t tokens = input.numel() / input.size(-1);
  if (tokens == 0) return out;

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard guard(device_of(input));
#define LAUNCH(TYPE)                                                                       \
  do {                                                                                     \
    constexpr uint32_t vec_size = 16 / sizeof(TYPE);                                       \
    dim3 block(std::min(static_cast<uint32_t>(d / vec_size), 1024U));                       \
    flashinfer::activation::act_and_mul_kernel<TYPE, silu<TYPE>><<<tokens, block, 0, stream>>>( \
        static_cast<TYPE*>(out.data_ptr()), static_cast<TYPE*>(input.data_ptr()), d);        \
  } while (0)
  if (input.scalar_type() == at::kHalf) LAUNCH(half);
  else if (input.scalar_type() == at::kBFloat16) LAUNCH(__nv_bfloat16);
  else if (input.scalar_type() == at::kFloat) LAUNCH(float);
  else TORCH_CHECK(false, "unsupported dtype");
#undef LAUNCH
  return out;
}

TORCH_LIBRARY_FRAGMENT(flux_kernel, m) {
  m.def("silu_and_mul(Tensor(a!) out, Tensor input) -> Tensor(a!)", &silu_and_mul);
}

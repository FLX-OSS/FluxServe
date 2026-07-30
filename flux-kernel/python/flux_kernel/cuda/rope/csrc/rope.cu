/*
 * Copyright (c) 2023 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0.
 * Adapted from SGL Kernel 0.3.15 (8c9670375).
 */
#include <ATen/cuda/CUDAContext.h>
#include <flashinfer/pos_enc.cuh>
#include <torch/all.h>
#include <torch/library.h>

template <typename T>
void launch_rope(torch::Tensor positions, torch::Tensor query, torch::Tensor key,
                 torch::Tensor query_out, torch::Tensor key_out, int head_size,
                 torch::Tensor cache, bool interleave, cudaStream_t stream) {
  const uint32_t tokens = positions.numel();
  const uint32_t q_heads = query.numel() / tokens / head_size;
  const uint32_t k_heads = key.numel() / tokens / head_size;
  const uint32_t rotary_dim = cache.size(1);
  auto status = flashinfer::BatchQKApplyRotaryPosIdsCosSinCache(
      static_cast<T*>(query.data_ptr()), static_cast<T*>(key.data_ptr()),
      static_cast<T*>(query_out.data_ptr()), static_cast<T*>(key_out.data_ptr()),
      cache.data_ptr<float>(), positions.data_ptr<int64_t>(), tokens, q_heads, k_heads,
      rotary_dim, head_size, query.stride(0), head_size, key.stride(0), head_size,
      query_out.stride(0), head_size, key_out.stride(0), head_size, interleave, stream);
  TORCH_CHECK(status == cudaSuccess, "cached RoPE launch failed: ", cudaGetErrorString(status));
}

void apply_rope(torch::Tensor positions, torch::Tensor query, torch::Tensor key,
                torch::Tensor query_out, torch::Tensor key_out, int64_t head_size,
                torch::Tensor cache, bool neox) {
  TORCH_CHECK(positions.is_cuda() && query.is_cuda() && key.is_cuda(), "RoPE requires CUDA tensors");
  TORCH_CHECK(positions.scalar_type() == at::kLong && cache.scalar_type() == at::kFloat,
              "invalid positions/cache dtype");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query/key dtype mismatch");
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const bool interleave = !neox;
  if (query.scalar_type() == at::kHalf)
    launch_rope<half>(positions, query, key, query_out, key_out, head_size, cache, interleave, stream);
  else if (query.scalar_type() == at::kBFloat16)
    launch_rope<__nv_bfloat16>(positions, query, key, query_out, key_out, head_size, cache, interleave, stream);
  else TORCH_CHECK(false, "cached RoPE supports float16 and bfloat16");
}

TORCH_LIBRARY_FRAGMENT(flux_kernel, m) {
  m.def("apply_rope(Tensor positions, Tensor query, Tensor key, Tensor(a!) query_out, "
        "Tensor(b!) key_out, int head_size, Tensor cache, bool is_neox) -> ()", &apply_rope);
}

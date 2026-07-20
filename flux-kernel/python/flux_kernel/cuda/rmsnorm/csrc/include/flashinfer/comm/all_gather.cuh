// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef FLASHINFER_SIMPLE_ALLGATHER
#define FLASHINFER_SIMPLE_ALLGATHER

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cuda/std/optional>
#include <tuple>
#include <type_traits>

#include "../exception.h"
#include "../logging.h"
#include "../utils.cuh"
#include "../vec_dtypes.cuh"
#include "trtllm_reducescatter_fusion.cuh"
#include "trtllm_allreduce_fusion.cuh"

namespace flashinfer {
    namespace simple_all_gather {

        template<typename T>
        struct AllGatherParams {
            int nranks;
            int rank;
            int hidden_dim;
            int hidden_dim_per_rank;
            int num_tokens;
            int max_num_tokens;
            void** workspace;
            void* allgather_in;
            void* allgather_out;
            int comm_size;
        };

        namespace cg = cooperative_groups;

        template <typename T, int NRanks, int VEC_SIZE>
        __global__ void clean_previous_buffer(AllGatherParams<T> params){
            vec_t<T, VEC_SIZE> clear_vec;
            clear_vec.fill(flashinfer::trtllm_reducescatter_fusion::utils::neg_zero_v<T>);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
            cudaGridDependencySynchronize();
            // Launch comm buffer since the buffer this kernel clears
            // is not currently used
            cudaTriggerProgrammaticLaunchCompletion();
#endif
            int block_idx = blockIdx.x;
            int num_threads_per_block = blockDim.x;
            int access_id = block_idx * num_threads_per_block + threadIdx.x;
            int access_stride = num_threads_per_block * gridDim.x;
            // whole buffer size
            int clear_access = (params.max_num_tokens * params.hidden_dim) / VEC_SIZE;
            int buf_size = params.hidden_dim * params.max_num_tokens * 2;
            int* flag_ptr = &reinterpret_cast<int*>(params.workspace[NRanks * 3])[2];
            int flag_value = *flag_ptr;
            int clear_offset = (flag_value + 2) % 3;

            uint8_t* clear_buf = reinterpret_cast<uint8_t*>(params.workspace[2 * NRanks + params.rank]) + clear_offset * buf_size;
            for (int idx = access_id; idx < clear_access; idx += access_stride) {
                clear_vec.store(reinterpret_cast<T*>(clear_buf) + idx * VEC_SIZE);
            }
        }

        template <typename T, int NRanks, bool TriggerCompletionAtEnd = true>
        __global__ void simple_allgather_hidden(AllGatherParams<T> params) {
            static constexpr int VEC_SIZE = 16 / sizeof(T);
            static_assert(VEC_SIZE == 8);
            // Each cluster handles one token
            // Each block handles one part of hidden_dim one token (Since hidden_dim may be huge as vocab size)
            cg::grid_group grid = cg::this_grid();
            cg::cluster_group cluster = cg::this_cluster();

            int token_id = grid.cluster_rank();
            // access_id in partial hidden_size of each rank
            int access_id_in_token = cluster.thread_rank();
            int token_stride = grid.num_clusters();
            int access_per_token = (params.hidden_dim_per_rank + VEC_SIZE - 1) / VEC_SIZE;
            int access_id_local = token_id * params.hidden_dim_per_rank / VEC_SIZE + access_id_in_token;
            int access_local_stride = token_stride * params.hidden_dim_per_rank / VEC_SIZE;
            int tot_access_local = (params.comm_size + VEC_SIZE - 1) / VEC_SIZE;
            flashinfer::trtllm_reducescatter_fusion::RLamportComm<NRanks> comm(params.workspace, params.rank);
            for (int tidx = token_id; tidx < params.num_tokens; tidx += token_stride) {
                for (int vec_id = cluster.thread_rank(); vec_id < access_per_token; vec_id += cluster.num_threads()) {
                    vec_t<T, VEC_SIZE> val;
                    int local_access_id = tidx * access_per_token + vec_id;
                    val.load(reinterpret_cast<T*>(params.allgather_in) + local_access_id * VEC_SIZE);
                    flashinfer::trtllm_reducescatter_fusion::utils::remove_neg_zero<T, VEC_SIZE>(val);
                    for (int r = 0; r < NRanks; ++r) {
                        // Push local data to other ranks
                        val.store(reinterpret_cast<T*>(comm.data_bufs[r]) +
                            (params.rank * tot_access_local + local_access_id) * VEC_SIZE);
                    }
                }
            }

            for (int tidx = token_id; tidx < params.num_tokens; tidx += token_stride) {
                for (int vec_id = cluster.thread_rank(); vec_id < access_per_token; vec_id += cluster.num_threads()) {
                    vec_t<T, VEC_SIZE> vals[NRanks];
                    bool done = false;
                    int partial_access_id = tidx * access_per_token + vec_id;
                    while (!done) {
                        done = true;
#pragma unroll
                        for (int r = 0; r < NRanks; ++r) {
                            // LDG.128 from local rank
                            vals[r].load_global_volatile(reinterpret_cast<T*>(comm.data_bufs[params.rank]) +
                                (r * tot_access_local + partial_access_id) * VEC_SIZE);
                            done &= !flashinfer::trtllm_reducescatter_fusion::utils::has_neg_zero<T, VEC_SIZE>(vals[r]);
                        }
                    }
                    for (int r = 0; r < NRanks; ++r) {
                        int token_offset = tidx * params.hidden_dim / VEC_SIZE;
                        int rank_offset = r * params.hidden_dim_per_rank / VEC_SIZE;
                        int global_access_id = token_offset + rank_offset + vec_id;
                        vals[r].store(reinterpret_cast<T*>(params.allgather_out) + global_access_id * VEC_SIZE);

                    }
                }
            }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
            // wait previous kernel to finish at the end of this kernel
            // because previous kernel clears the buffer not used by this
            // round
            cudaGridDependencySynchronize();
            cudaTriggerProgrammaticLaunchCompletion();
#endif
            // Update next buffer
            comm.update(params.comm_size * NRanks);
        }

        template <typename T, int NRanks, int VEC_SIZE, bool TriggerCompletionAtEnd = true>
        __global__ void simple_allgather_hidden_sync(AllGatherParams<T> params) {
            // Each cluster handles one token
            // Each block handles one part of hidden_dim one token (Since hidden_dim may be huge as vocab size)
            cg::grid_group grid = cg::this_grid();
            cg::cluster_group cluster = cg::this_cluster();

            int token_id = grid.cluster_rank();
            // access_id in partial hidden_size of each rank
            int access_id_in_token = cluster.thread_rank();
            int token_stride = grid.num_clusters();
            int access_per_token = (params.hidden_dim_per_rank + VEC_SIZE - 1) / VEC_SIZE;
            int access_id_local = token_id * params.hidden_dim_per_rank / VEC_SIZE + access_id_in_token;
            int access_local_stride = token_stride * params.hidden_dim_per_rank / VEC_SIZE;
            int tot_access_local = (params.comm_size + VEC_SIZE - 1) / VEC_SIZE;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
            cudaGridDependencySynchronize();
            if constexpr (!TriggerCompletionAtEnd) {
                cudaTriggerProgrammaticLaunchCompletion();
            }
#endif
            using SyncComm = flashinfer::trtllm_allreduce_fusion::SyncComm<NRanks>;
            using Barrier = flashinfer::trtllm_allreduce_fusion::Barrier<NRanks>;

            SyncComm comm(params.workspace);
            Barrier barrier(params.rank, comm);

            barrier.sync();
            for (int tidx = token_id; tidx < params.num_tokens; tidx += token_stride) {
                for (int vec_id = cluster.thread_rank(); vec_id < access_per_token; vec_id += cluster.num_threads()) {
                    vec_t<T, VEC_SIZE> val;
                    int local_access_id = tidx * access_per_token + vec_id;
                    val.load(reinterpret_cast<T*>(params.allgather_in) + local_access_id * VEC_SIZE);
                    for (int r = 0; r < NRanks; ++r) {
                        // Push local data to other ranks
                        val.store(reinterpret_cast<T*>(comm.comm_bufs[r]) +
                            (params.rank * tot_access_local + local_access_id) * VEC_SIZE);
                    }
                }
            }

            barrier.sync();
            for (int tidx = token_id; tidx < params.num_tokens; tidx += token_stride) {
                for (int vec_id = cluster.thread_rank(); vec_id < access_per_token; vec_id += cluster.num_threads()) {
                    vec_t<T, VEC_SIZE> vals[NRanks];
                    bool done = false;
                    int partial_access_id = tidx * access_per_token + vec_id;
                    for (int r = 0; r < NRanks; ++r) {
                        //vals[r].load_global_volatile(reinterpret_cast<T*>(comm.comm_bufs[params.rank]) +
                        vals[r].load(reinterpret_cast<T*>(comm.comm_bufs[params.rank]) +
                            (r * tot_access_local + partial_access_id) * VEC_SIZE);
                        int token_offset = tidx * params.hidden_dim / VEC_SIZE;
                        int rank_offset = r * params.hidden_dim_per_rank / VEC_SIZE;
                        int global_access_id = token_offset + rank_offset + vec_id;
                        vals[r].store(reinterpret_cast<T*>(params.allgather_out) + global_access_id * VEC_SIZE);

                    }
                }
            }
            comm.update(barrier.m_flag_value);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
            if constexpr (TriggerCompletionAtEnd) {
                cudaTriggerProgrammaticLaunchCompletion();
            }
#endif
        }
    }
}

#endif  // FLASHINFER_TRTLLM_ALLGATHER_FUSION_CUH_

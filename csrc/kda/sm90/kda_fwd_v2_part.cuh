// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

// Phase 2.2 Step G — warp-partition K2 kernel.
//
// Parallelize independent matmul pairs by splitting the 8 warps into two
// groups of 4. Steps 1/2 and 4/5 are pairwise independent (different inputs
// and outputs), so we can compute them in parallel.
//
// Layout vs cu_wmma (backend=2):
//   cu_wmma:  [step1 with 8 warps] sync [step2 with 8 warps] sync ...
//   cu_part:  [step1 with warps 0-3 || step2 with warps 4-7] sync
//             [step4 with warps 0-3 || step5 with warps 4-7] sync
// Steps 3 (elementwise) and step 7 (state update) keep all 8 warps.
//
// Trade-off: per-matmul tile count per warp doubles (4 → 8 tiles when only
// 4 warps participate), but we get two matmuls' worth of work in roughly
// one matmul's wallclock time, ASSUMING the tensor-core throughput isn't
// already saturated.

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <mma.h>

namespace kda::sm90::v2::part_impl {

namespace nv = nvcuda;

constexpr int kBT = 64;
constexpr int kK = 128;
constexpr int kV = 128;
constexpr int kBlockThreads = 256;
constexpr int kWarps = 8;

struct SharedStoragePart {
    float state_fp32[kK * kV];                       // 64KB
    __nv_bfloat16 state_bf16[kK * kV];                // 32KB
    __nv_bfloat16 ws_qd[kBT * kK];                    // 16KB
    __nv_bfloat16 ws_kd[kBT * kK];                    // 16KB
    __nv_bfloat16 ws_kr[kBT * kK];                    // 16KB
    float ws_gt[kK];                                    // 0.5KB
    __nv_bfloat16 ws_mqk[kBT * kBT];                  // 8KB
    __nv_bfloat16 ws_inv[kBT * kBT];                  // 8KB
    __nv_bfloat16 v[kBT * kV];                         // 16KB
    float beta[kBT];                                    // 0.25KB
    __nv_bfloat16 bufA[kBT * kV];                      // 16KB
    __nv_bfloat16 bufB[kBT * kV];                      // 16KB
    float warp_scratch[kWarps * 16 * 16];              // 8KB
};
// 216.75 KB

// Partitioned WMMA gemm: only warps in [warp_offset, warp_offset + num_warps)
// participate. Caller is responsible for syncthreads after.
__device__ inline void
gemm_wmma_bf16_part(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int N, int KK, float* warp_scratch_for_warp,
    int warp_offset, int num_warps) {
    int warp_id_global = threadIdx.x / 32;
    int local_warp = warp_id_global - warp_offset;
    int lane_id = threadIdx.x % 32;
    if (local_warp < 0 || local_warp >= num_warps) return;

    int M_tiles = M / 16, N_tiles = N / 16, K_tiles = KK / 16;
    int total_tiles = M_tiles * N_tiles;
    for (int tile_idx = local_warp; tile_idx < total_tiles; tile_idx += num_warps) {
        int m_tile = tile_idx / N_tiles;
        int n_tile = tile_idx % N_tiles;
        nv::wmma::fragment<nv::wmma::accumulator, 16, 16, 16, float> c_frag;
        nv::wmma::fill_fragment(c_frag, 0.0f);
        for (int k_tile = 0; k_tile < K_tiles; ++k_tile) {
            nv::wmma::fragment<nv::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nv::wmma::row_major> a_frag;
            nv::wmma::fragment<nv::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nv::wmma::row_major> b_frag;
            nv::wmma::load_matrix_sync(a_frag, A + (m_tile * 16) * KK + k_tile * 16, KK);
            nv::wmma::load_matrix_sync(b_frag, B + (k_tile * 16) * N + n_tile * 16, N);
            nv::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        nv::wmma::store_matrix_sync(warp_scratch_for_warp, c_frag, 16, nv::wmma::mem_row_major);
        __syncwarp();
        for (int i = lane_id; i < 16 * 16; i += 32) {
            int row = i / 16, col = i % 16;
            C[(m_tile * 16 + row) * N + (n_tile * 16 + col)] = __float2bfloat16(warp_scratch_for_warp[i]);
        }
        __syncwarp();
    }
}

__device__ inline void
state_update_wmma_part(
    float* state_fp32, const float* ws_gt,
    const __nv_bfloat16* ws_kr, const __nv_bfloat16* v_new,
    float* warp_scratch_for_warp,
    int warp_offset, int num_warps) {
    int warp_id_global = threadIdx.x / 32;
    int local_warp = warp_id_global - warp_offset;
    int lane_id = threadIdx.x % 32;
    if (local_warp < 0 || local_warp >= num_warps) return;

    int M_tiles = kK / 16, N_tiles = kV / 16, K_tiles = kBT / 16;
    int total_tiles = M_tiles * N_tiles;
    for (int tile_idx = local_warp; tile_idx < total_tiles; tile_idx += num_warps) {
        int m_tile = tile_idx / N_tiles;
        int n_tile = tile_idx % N_tiles;
        nv::wmma::fragment<nv::wmma::accumulator, 16, 16, 16, float> c_frag;
        nv::wmma::fill_fragment(c_frag, 0.0f);
        for (int k_tile = 0; k_tile < K_tiles; ++k_tile) {
            nv::wmma::fragment<nv::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nv::wmma::col_major> a_frag;
            nv::wmma::fragment<nv::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nv::wmma::row_major> b_frag;
            nv::wmma::load_matrix_sync(a_frag, ws_kr + (k_tile * 16) * kK + m_tile * 16, kK);
            nv::wmma::load_matrix_sync(b_frag, v_new + (k_tile * 16) * kV + n_tile * 16, kV);
            nv::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        nv::wmma::store_matrix_sync(warp_scratch_for_warp, c_frag, 16, nv::wmma::mem_row_major);
        __syncwarp();
        for (int i = lane_id; i < 16 * 16; i += 32) {
            int row = i / 16, col = i % 16;
            int g_k = m_tile * 16 + row, g_v = n_tile * 16 + col;
            state_fp32[g_k * kV + g_v] = ws_gt[g_k] * state_fp32[g_k * kV + g_v]
                                          + warp_scratch_for_warp[i];
        }
        __syncwarp();
    }
}

template <typename T>
__device__ inline void
copy_gmem_to_smem(const T* src, T* dst, int count) {
    int tid = threadIdx.x;
    for (int i = tid; i < count; i += kBlockThreads) dst[i] = src[i];
}

__global__ void
kda_fwd_v2_part_kernel(
    const __nv_bfloat16* __restrict__ ws_qd_ptr,
    const __nv_bfloat16* __restrict__ ws_kd_ptr,
    const __nv_bfloat16* __restrict__ ws_kr_ptr,
    const float* __restrict__ ws_gt_ptr,
    const __nv_bfloat16* __restrict__ ws_mqk_ptr,
    const __nv_bfloat16* __restrict__ ws_inv_ptr,
    const __nv_bfloat16* __restrict__ v_ptr,
    const float* __restrict__ beta_ptr,
    const float* __restrict__ initial_state_ptr,
    __nv_bfloat16* __restrict__ o_ptr,
    float* __restrict__ final_state_ptr,
    const int32_t* __restrict__ cu_seqlens,
    const int32_t* __restrict__ chunk_offsets,
    int H, int N) {
    int n = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / 32;

    int bos = cu_seqlens[n];
    int eos = cu_seqlens[n + 1];
    int T_seq = eos - bos;
    int NT = (T_seq + kBT - 1) / kBT;
    int chunk_offset = chunk_offsets[n];

    extern __shared__ char smem_raw[];
    SharedStoragePart& s = *reinterpret_cast<SharedStoragePart*>(smem_raw);

    float* my_warp_scratch = s.warp_scratch + warp_id * 256;

    // Initial state load (transposed)
    if (initial_state_ptr != nullptr) {
        const float* init_base = initial_state_ptr + (n * H + h) * kV * kK;
        for (int idx = tid; idx < kK * kV; idx += kBlockThreads) {
            int k = idx / kV;
            int v_ = idx % kV;
            s.state_fp32[k * kV + v_] = init_base[v_ * kK + k];
        }
    } else {
        for (int idx = tid; idx < kK * kV; idx += kBlockThreads) s.state_fp32[idx] = 0.0f;
    }
    __syncthreads();

    for (int c = 0; c < NT; ++c) {
        int chunk_idx = chunk_offset + c;
        int chunk_start = c * kBT;
        int actual_len = min(kBT, T_seq - chunk_start);

        const __nv_bfloat16* qd_g = ws_qd_ptr  + ((chunk_idx) * H + h) * kBT * kK;
        const __nv_bfloat16* kd_g = ws_kd_ptr  + ((chunk_idx) * H + h) * kBT * kK;
        const __nv_bfloat16* kr_g = ws_kr_ptr  + ((chunk_idx) * H + h) * kBT * kK;
        const float* gt_g          = ws_gt_ptr  + ((chunk_idx) * H + h) * kK;
        const __nv_bfloat16* mqk_g = ws_mqk_ptr + ((chunk_idx) * H + h) * kBT * kBT;
        const __nv_bfloat16* inv_g = ws_inv_ptr + ((chunk_idx) * H + h) * kBT * kBT;
        const __nv_bfloat16* v_g   = v_ptr      + (bos + chunk_start) * H * kV + h * kV;
        const float* beta_g        = beta_ptr   + (bos + chunk_start) * H + h;

        copy_gmem_to_smem<__nv_bfloat16>(qd_g, s.ws_qd, kBT * kK);
        copy_gmem_to_smem<__nv_bfloat16>(kd_g, s.ws_kd, kBT * kK);
        copy_gmem_to_smem<__nv_bfloat16>(kr_g, s.ws_kr, kBT * kK);
        copy_gmem_to_smem<float>(gt_g, s.ws_gt, kK);
        copy_gmem_to_smem<__nv_bfloat16>(mqk_g, s.ws_mqk, kBT * kBT);
        copy_gmem_to_smem<__nv_bfloat16>(inv_g, s.ws_inv, kBT * kBT);
        for (int idx = tid; idx < kBT * kV; idx += kBlockThreads) {
            int t = idx / kV;
            int v_ = idx % kV;
            s.v[idx] = (t < actual_len) ? v_g[t * H * kV + v_] : __float2bfloat16(0.0f);
        }
        for (int idx = tid; idx < kBT; idx += kBlockThreads) {
            s.beta[idx] = (idx < actual_len) ? beta_g[idx * H] : 0.0f;
        }
        // Refresh state_bf16 view
        for (int idx = tid; idx < kK * kV; idx += kBlockThreads) {
            s.state_bf16[idx] = __float2bfloat16(s.state_fp32[idx]);
        }
        __syncthreads();

        // === Parallel pair: Step 1 (warps 0-3) || Step 2 (warps 4-7) ===
        // Step 1: bufA = ws_kd @ state    (warps 0-3)
        gemm_wmma_bf16_part(s.ws_kd, s.state_bf16, s.bufA,
                             kBT, kV, kK, my_warp_scratch,
                             /*warp_offset=*/0, /*num_warps=*/4);
        // Step 2: bufB = ws_inv @ v       (warps 4-7)
        gemm_wmma_bf16_part(s.ws_inv, s.v, s.bufB,
                             kBT, kV, kBT, my_warp_scratch,
                             /*warp_offset=*/4, /*num_warps=*/4);
        __syncthreads();

        // Step 3: bufA = (bufB - bufA) * beta — all warps
        for (int idx = tid; idx < kBT * kV; idx += kBlockThreads) {
            int t = idx / kV;
            float vp = static_cast<float>(s.bufB[idx]);
            float vd = static_cast<float>(s.bufA[idx]);
            s.bufA[idx] = __float2bfloat16((vp - vd) * s.beta[t]);
        }
        __syncthreads();

        // === Parallel pair: Step 4 (warps 0-3) || Step 5 (warps 4-7) ===
        // Step 4: bufB = ws_mqk @ v_new    (warps 0-3)
        gemm_wmma_bf16_part(s.ws_mqk, s.bufA, s.bufB,
                             kBT, kV, kBT, my_warp_scratch,
                             /*warp_offset=*/0, /*num_warps=*/4);
        // Step 5: state_bf16 region holds o_inter   (warps 4-7)
        gemm_wmma_bf16_part(s.ws_qd, s.state_bf16,
                             reinterpret_cast<__nv_bfloat16*>(s.state_bf16),
                             kBT, kV, kK, my_warp_scratch,
                             /*warp_offset=*/4, /*num_warps=*/4);
        __syncthreads();

        // Step 6: o = o_inter + o_intra → store gmem
        __nv_bfloat16* o_g = o_ptr + (bos + chunk_start) * H * kV + h * kV;
        const __nv_bfloat16* o_inter_view = reinterpret_cast<const __nv_bfloat16*>(s.state_bf16);
        for (int idx = tid; idx < kBT * kV; idx += kBlockThreads) {
            int t = idx / kV;
            int v_ = idx % kV;
            if (t < actual_len) {
                float v_inter = static_cast<float>(o_inter_view[idx]);
                float v_intra = static_cast<float>(s.bufB[idx]);
                o_g[t * H * kV + v_] = __float2bfloat16(v_inter + v_intra);
            }
        }
        __syncthreads();

        // Step 7: state update — all 8 warps participate
        state_update_wmma_part(s.state_fp32, s.ws_gt, s.ws_kr, s.bufA,
                                my_warp_scratch,
                                /*warp_offset=*/0, /*num_warps=*/kWarps);
        __syncthreads();
    }

    if (final_state_ptr != nullptr) {
        float* fin_base = final_state_ptr + (n * H + h) * kV * kK;
        for (int idx = tid; idx < kK * kV; idx += kBlockThreads) {
            int k = idx / kV;
            int v_ = idx % kV;
            fin_base[v_ * kK + k] = s.state_fp32[k * kV + v_];
        }
    }
}

inline void
launch_kda_fwd_v2_part(
    const __nv_bfloat16* ws_qd, const __nv_bfloat16* ws_kd,
    const __nv_bfloat16* ws_kr, const float* ws_gt,
    const __nv_bfloat16* ws_mqk, const __nv_bfloat16* ws_inv,
    const __nv_bfloat16* v, const float* beta,
    const float* initial_state,
    __nv_bfloat16* o, float* final_state,
    const int32_t* cu_seqlens, const int32_t* chunk_offsets,
    int N, int H, cudaStream_t stream) {
    constexpr size_t smem_bytes = sizeof(SharedStoragePart);
    static_assert(smem_bytes <= 228 * 1024, "SharedStoragePart too large");

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(
            kda_fwd_v2_part_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes);
        attr_set = true;
    }

    dim3 grid(N, H, 1);
    dim3 block(kBlockThreads);
    kda_fwd_v2_part_kernel<<<grid, block, smem_bytes, stream>>>(
        ws_qd, ws_kd, ws_kr, ws_gt, ws_mqk, ws_inv,
        v, beta, initial_state,
        o, final_state, cu_seqlens, chunk_offsets,
        H, N);
}

}  // namespace kda::sm90::v2::part_impl

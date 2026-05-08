// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

// Phase 2.2 Step E (partial) — warp-specialized C++ K2 kernel foundation.
//
// Architecture:
//   8 warps per block. Warp 0 dedicated to GMEM→SMEM async loads (cp.async).
//   Warps 1-7 do WMMA matmuls. Synchronization via __syncthreads() at chunk
//   boundary (after load + state update phases).
//
//   This is a minimal first cut. NOT yet:
//   - Multi-stage pipeline (currently single-stage; cp.async issues load,
//     then __syncthreads acts as wait. No overlap across chunks.)
//   - Independent producer/consumer barriers (LdSt and Math share __syncthreads)
//   - WGMMA (still using WMMA m16n16k16)
//   - Register-resident state (state still in SMEM)
//
//   What this DOES validate:
//   - cp.async-based GMEM→SMEM loads (vs per-thread copy) for the workspace
//   - 7-warp Math team (matmul work-share over 16x16 tiles, 1 fewer warp than
//     WMMA backend's 8 since LdSt is dedicated)
//   - Foundation for future multi-stage pipelining

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>

namespace kda::sm90::v2::wsp_impl {

namespace nv = nvcuda;

constexpr int kBT = 64;
constexpr int kK = 128;
constexpr int kV = 128;
constexpr int kBlockThreads = 256;
constexpr int kWarps = 8;
constexpr int kLdStWarpId = 0;
constexpr int kMathWarpFirst = 1;
constexpr int kMathWarps = 7;

struct SharedStorageWsp {
    float state_fp32[kK * kV];                     // 64KB persistent
    __nv_bfloat16 state_bf16[kK * kV];              // 32KB matmul view
    __nv_bfloat16 ws_qd[kBT * kK];                  // 16KB
    __nv_bfloat16 ws_kd[kBT * kK];                  // 16KB
    __nv_bfloat16 ws_kr[kBT * kK];                  // 16KB
    float ws_gt[kK];                                  // 0.5KB
    __nv_bfloat16 ws_mqk[kBT * kBT];                // 8KB
    __nv_bfloat16 ws_inv[kBT * kBT];                // 8KB
    __nv_bfloat16 v[kBT * kV];                       // 16KB
    float beta[kBT];                                  // 0.25KB
    __nv_bfloat16 bufA[kBT * kV];                    // 16KB
    __nv_bfloat16 bufB[kBT * kV];                    // 16KB
    float warp_scratch[kWarps * 16 * 16];            // 8KB
};
// Same as WmmaCompact: 216.75KB

// cp.async helper: issue async copy GMEM → SMEM, 16-byte transactions.
// Returns immediately; synchronize via __pipeline_commit() + __pipeline_wait_prior.
__device__ inline void
cp_async_bytes_16B(void* smem_dst, const void* gmem_src, int n_bytes) {
    int tid = threadIdx.x;
    int chunks = n_bytes / 16;
    char* dst_b = static_cast<char*>(smem_dst);
    const char* src_b = static_cast<const char*>(gmem_src);
    // Each thread issues some chunks. Stride by num threads in this warp range.
    constexpr int kThreadsForLdst = 32;  // single warp issues
    int thread_in_warp = tid % 32;
    for (int i = thread_in_warp; i < chunks; i += kThreadsForLdst) {
        __pipeline_memcpy_async(dst_b + i * 16, src_b + i * 16, 16);
    }
}

__device__ inline void
gemm_wmma_bf16(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int N, int KK, float* warp_scratch_for_warp,
    int math_warp_id) {
    int lane_id = threadIdx.x % 32;
    int M_tiles = M / 16;
    int N_tiles = N / 16;
    int K_tiles = KK / 16;
    int total_tiles = M_tiles * N_tiles;

    for (int tile_idx = math_warp_id; tile_idx < total_tiles; tile_idx += kMathWarps) {
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
            int row = i / 16;
            int col = i % 16;
            int c_row = m_tile * 16 + row;
            int c_col = n_tile * 16 + col;
            C[c_row * N + c_col] = __float2bfloat16(warp_scratch_for_warp[i]);
        }
        __syncwarp();
    }
}

__device__ inline void
state_update_wmma(
    float* state_fp32, const float* ws_gt,
    const __nv_bfloat16* ws_kr, const __nv_bfloat16* v_new,
    float* warp_scratch_for_warp, int math_warp_id) {
    int lane_id = threadIdx.x % 32;
    int M_tiles = kK / 16;
    int N_tiles = kV / 16;
    int K_tiles = kBT / 16;
    int total_tiles = M_tiles * N_tiles;

    for (int tile_idx = math_warp_id; tile_idx < total_tiles; tile_idx += kMathWarps) {
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
            int row = i / 16;
            int col = i % 16;
            int g_k = m_tile * 16 + row;
            int g_v = n_tile * 16 + col;
            state_fp32[g_k * kV + g_v] = ws_gt[g_k] * state_fp32[g_k * kV + g_v]
                                          + warp_scratch_for_warp[i];
        }
        __syncwarp();
    }
}

__global__ void
kda_fwd_v2_wsp_kernel(
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
    int lane_id = tid % 32;

    int bos = cu_seqlens[n];
    int eos = cu_seqlens[n + 1];
    int T_seq = eos - bos;
    int NT = (T_seq + kBT - 1) / kBT;
    int chunk_offset = chunk_offsets[n];

    extern __shared__ char smem_raw[];
    SharedStorageWsp& s = *reinterpret_cast<SharedStorageWsp*>(smem_raw);

    bool is_ldst = (warp_id == kLdStWarpId);
    bool is_math = (warp_id >= kMathWarpFirst);
    int math_warp_id = warp_id - kMathWarpFirst;  // 0..6 for math warps
    float* my_warp_scratch = s.warp_scratch + warp_id * 256;

    // Load initial state — all warps cooperate
    if (initial_state_ptr != nullptr) {
        const float* init_base = initial_state_ptr + (n * H + h) * kV * kK;
        for (int idx = tid; idx < kK * kV; idx += kBlockThreads) {
            int k = idx / kV;
            int v_ = idx % kV;
            s.state_fp32[k * kV + v_] = init_base[v_ * kK + k];
        }
    } else {
        for (int idx = tid; idx < kK * kV; idx += kBlockThreads) {
            s.state_fp32[idx] = 0.0f;
        }
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

        // Phase A: LdSt warp issues cp.async for workspace tensors;
        //          Math warps refresh state_bf16 from state_fp32 in parallel.
        if (is_ldst) {
            cp_async_bytes_16B(s.ws_qd, qd_g, kBT * kK * 2);
            cp_async_bytes_16B(s.ws_kd, kd_g, kBT * kK * 2);
            cp_async_bytes_16B(s.ws_kr, kr_g, kBT * kK * 2);
            cp_async_bytes_16B(s.ws_gt, gt_g, kK * 4);
            cp_async_bytes_16B(s.ws_mqk, mqk_g, kBT * kBT * 2);
            cp_async_bytes_16B(s.ws_inv, inv_g, kBT * kBT * 2);
            // v needs strided load (different head's v interleaved); use per-thread copy
            // (cp.async needs contiguous source; strided not directly supported here).
            // Skip cp.async for v and beta — fall back to per-thread.
            __pipeline_commit();
        }
        if (is_math) {
            // Refresh state_bf16 view for matmul reads
            for (int idx = math_warp_id * 32 + lane_id; idx < kK * kV;
                 idx += kMathWarps * 32) {
                s.state_bf16[idx] = __float2bfloat16(s.state_fp32[idx]);
            }
        }
        // All warps: copy v and beta (per-thread, since strided)
        for (int idx = tid; idx < kBT * kV; idx += kBlockThreads) {
            int t = idx / kV;
            int v_ = idx % kV;
            s.v[idx] = (t < actual_len) ? v_g[t * H * kV + v_] : __float2bfloat16(0.0f);
        }
        for (int idx = tid; idx < kBT; idx += kBlockThreads) {
            s.beta[idx] = (idx < actual_len) ? beta_g[idx * H] : 0.0f;
        }

        // Wait for cp.async to complete on the LdSt warp's side.
        if (is_ldst) {
            __pipeline_wait_prior(0);
        }
        __syncthreads();

        // Phase B: Math warps compute. LdSt warp idles (or could load next chunk
        // here in a multi-stage variant; current single-stage version idles).
        if (is_math) {
            // Step 1: bufA = ws_kd @ state_bf16
            gemm_wmma_bf16(s.ws_kd, s.state_bf16, s.bufA, kBT, kV, kK,
                           my_warp_scratch, math_warp_id);
        }
        __syncthreads();

        if (is_math) {
            // Step 2: bufB = ws_inv @ v
            gemm_wmma_bf16(s.ws_inv, s.v, s.bufB, kBT, kV, kBT,
                           my_warp_scratch, math_warp_id);
        }
        __syncthreads();

        // Step 3: bufA = (bufB - bufA) * beta — all warps participate
        for (int idx = tid; idx < kBT * kV; idx += kBlockThreads) {
            int t = idx / kV;
            float vp = static_cast<float>(s.bufB[idx]);
            float vd = static_cast<float>(s.bufA[idx]);
            s.bufA[idx] = __float2bfloat16((vp - vd) * s.beta[t]);
        }
        __syncthreads();

        if (is_math) {
            // Step 4: bufB = ws_mqk @ v_new (v_new = bufA)
            gemm_wmma_bf16(s.ws_mqk, s.bufA, s.bufB, kBT, kV, kBT,
                           my_warp_scratch, math_warp_id);
        }
        __syncthreads();

        if (is_math) {
            // Step 5: o_inter = ws_qd @ state_bf16 → write to first half of state_bf16 SMEM
            gemm_wmma_bf16(s.ws_qd, s.state_bf16,
                           reinterpret_cast<__nv_bfloat16*>(s.state_bf16),
                           kBT, kV, kK, my_warp_scratch, math_warp_id);
        }
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

        if (is_math) {
            // Step 7: state update (math warps only)
            state_update_wmma(s.state_fp32, s.ws_gt, s.ws_kr, s.bufA,
                              my_warp_scratch, math_warp_id);
        }
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
launch_kda_fwd_v2_wsp(
    const __nv_bfloat16* ws_qd, const __nv_bfloat16* ws_kd,
    const __nv_bfloat16* ws_kr, const float* ws_gt,
    const __nv_bfloat16* ws_mqk, const __nv_bfloat16* ws_inv,
    const __nv_bfloat16* v, const float* beta,
    const float* initial_state,
    __nv_bfloat16* o, float* final_state,
    const int32_t* cu_seqlens, const int32_t* chunk_offsets,
    int N, int H, cudaStream_t stream) {
    constexpr size_t smem_bytes = sizeof(SharedStorageWsp);
    static_assert(smem_bytes <= 228 * 1024, "SharedStorageWsp too large");

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(
            kda_fwd_v2_wsp_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes);
        attr_set = true;
    }

    dim3 grid(N, H, 1);
    dim3 block(kBlockThreads);
    kda_fwd_v2_wsp_kernel<<<grid, block, smem_bytes, stream>>>(
        ws_qd, ws_kd, ws_kr, ws_gt, ws_mqk, ws_inv,
        v, beta, initial_state,
        o, final_state, cu_seqlens, chunk_offsets,
        H, N);
}

}  // namespace kda::sm90::v2::wsp_impl

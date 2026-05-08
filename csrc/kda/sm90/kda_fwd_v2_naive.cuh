// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

// Phase 2.2 Step C — naive but CORRECT K2 kernel.
//
// Math (per FlashKDA K2 contract, see kda_fwd_v2_stub.cuh):
//   For each (seq n, head h), state h ∈ [K, V] fp32 in SMEM. Sequential c loop.
//     v_pred[BT,V]  = ws_kd[BT,K]   @ state[K,V]
//     v_proxy[BT,V] = ws_inv[BT,BT] @ v[BT,V]
//     v_new[BT,V]   = (v_proxy - v_pred) * sigmoid_beta[BT]   (per-row scale)
//     o_inter[BT,V] = ws_qd[BT,K]   @ state[K,V]
//     o_intra[BT,V] = ws_mqk[BT,BT] @ v_new[BT,V]
//     o[BT,V]       = o_inter + o_intra
//     state[K,V]    = diag(ws_gt[K]) * state[K,V] + ws_kr^T[K,BT] @ v_new[BT,V]
//
// Approach: single block per (seq, head). 256 threads. All workspace tiles +
// intermediates kept in SMEM. Naive thread-block matmul (each thread covers
// some output elements; inner loop reduces over K dim with fp32 accumulator).
// Slow but correct — replace with CUTLASS warp-specialized version later.

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

namespace kda::sm90::v2 {

constexpr int kBT = 64;
constexpr int kK = 128;
constexpr int kV = 128;
constexpr int kBlockThreads = 256;

// Naive thread-block matmul: C[M,N] = A[M,KK] @ B[KK,N]
// Inputs / output all in SMEM. Output is fp32 (caller can cast if needed).
template <int M, int N, int KK, typename A_T, typename B_T>
__device__ inline void
gemm_smem_fp32(const A_T* A, const B_T* B, float* C) {
    constexpr int total = M * N;
    int tid = threadIdx.x;
    for (int idx = tid; idx < total; idx += kBlockThreads) {
        int m = idx / N;
        int n = idx % N;
        float acc = 0.0f;
        #pragma unroll 8
        for (int k_iter = 0; k_iter < KK; ++k_iter) {
            float a = static_cast<float>(A[m * KK + k_iter]);
            float b = static_cast<float>(B[k_iter * N + n]);
            acc += a * b;
        }
        C[idx] = acc;
    }
}

// Same but C is bf16 (cast from fp32 acc).
template <int M, int N, int KK, typename A_T, typename B_T>
__device__ inline void
gemm_smem_bf16(const A_T* A, const B_T* B, __nv_bfloat16* C) {
    constexpr int total = M * N;
    int tid = threadIdx.x;
    for (int idx = tid; idx < total; idx += kBlockThreads) {
        int m = idx / N;
        int n = idx % N;
        float acc = 0.0f;
        #pragma unroll 8
        for (int k_iter = 0; k_iter < KK; ++k_iter) {
            float a = static_cast<float>(A[m * KK + k_iter]);
            float b = static_cast<float>(B[k_iter * N + n]);
            acc += a * b;
        }
        C[idx] = __float2bfloat16(acc);
    }
}

// Naive A^T @ B for state update: C[K,V] += A^T[K,BT] @ B[BT,V] where A is [BT,K] in SMEM.
// Combined with diag scale: C[k,v] = scale[k] * C_prev[k,v] + sum_t A[t,k] * B[t,v]
template <int K_, int V_, int BT_>
__device__ inline void
state_update(float* state, const float* gt, const __nv_bfloat16* kr_BT_K, const __nv_bfloat16* v_new_BT_V) {
    constexpr int total = K_ * V_;
    int tid = threadIdx.x;
    for (int idx = tid; idx < total; idx += kBlockThreads) {
        int k = idx / V_;
        int v_ = idx % V_;
        float acc = 0.0f;
        #pragma unroll 8
        for (int t = 0; t < BT_; ++t) {
            float a = static_cast<float>(kr_BT_K[t * K_ + k]);
            float b = static_cast<float>(v_new_BT_V[t * V_ + v_]);
            acc += a * b;
        }
        state[idx] = gt[k] * state[idx] + acc;
    }
}

// SMEM layout for one block:
struct SharedStorage {
    // Persistent across all chunks (must be alive for whole kernel)
    float state[kK * kV];         // 64KB

    // Per-chunk workspace tiles (loaded fresh each chunk)
    __nv_bfloat16 ws_qd[kBT * kK];   // 16KB
    __nv_bfloat16 ws_kd[kBT * kK];   // 16KB
    __nv_bfloat16 ws_kr[kBT * kK];   // 16KB
    float ws_gt[kK];                  // 0.5KB
    __nv_bfloat16 ws_mqk[kBT * kBT]; // 8KB
    __nv_bfloat16 ws_inv[kBT * kBT]; // 8KB
    __nv_bfloat16 v[kBT * kV];        // 16KB
    float beta[kBT];                  // 0.25KB

    // Intermediates (produced and consumed within one chunk)
    __nv_bfloat16 v_pred[kBT * kV];  // 16KB
    __nv_bfloat16 v_proxy[kBT * kV]; // 16KB
    __nv_bfloat16 v_new[kBT * kV];   // 16KB
    __nv_bfloat16 o_inter[kBT * kV]; // 16KB
    __nv_bfloat16 o_intra[kBT * kV]; // 16KB
};
// Total: 64 + 16*4 + 0.5 + 8*2 + 16 + 0.25 + 16*5 = 240.75 KB.
// HOPPER allows up to 228KB per CTA with cudaFuncAttributeMaxDynamicSharedMemorySize.
// We're slightly over → squeeze some out by reusing v_pred buffer for v_new (lifetimes are
// disjoint after step 3). See struct revision below if this matters.

// Reduced-SMEM variant: alias v_pred/v_proxy/v_new/o_inter/o_intra in unions where possible.
// Lifetime analysis:
//   v_pred:  steps 1..3
//   v_proxy: steps 2..3
//   v_new:   steps 3..7 (whole rest of chunk)
//   o_inter: step 5..6
//   o_intra: step 4..6
// At any time, the maximum simultaneously-alive intermediates is 3 (e.g., step 4: v_new, ws_mqk, o_intra)
struct SharedStorageCompact {
    float state[kK * kV];                    // 64KB persistent

    // Workspace tiles (pure inputs, always re-loaded per chunk)
    __nv_bfloat16 ws_qd[kBT * kK];           // 16KB
    __nv_bfloat16 ws_kd[kBT * kK];           // 16KB
    __nv_bfloat16 ws_kr[kBT * kK];           // 16KB
    float ws_gt[kK];                          // 0.5KB
    __nv_bfloat16 ws_mqk[kBT * kBT];         // 8KB
    __nv_bfloat16 ws_inv[kBT * kBT];         // 8KB
    __nv_bfloat16 v[kBT * kV];                // 16KB
    float beta[kBT];                          // 0.25KB

    // Intermediates: 3 buffers, recycled.
    // - buf0: v_pred (step 1) → o_inter (step 5) → reused
    // - buf1: v_proxy (step 2) → o_intra (step 4) → reused
    // - buf2: v_new (step 3..7), persistent within chunk
    __nv_bfloat16 buf0[kBT * kV];            // 16KB
    __nv_bfloat16 buf1[kBT * kV];            // 16KB
    __nv_bfloat16 buf2[kBT * kV];            // 16KB
};
// Total: 64 + 16*4 + 0.5 + 8*2 + 16 + 0.25 + 16*3 = 208.75 KB. Fits in 228KB. ✓

template <typename T>
__device__ inline void
copy_gmem_to_smem(const T* src, T* dst, int count) {
    int tid = threadIdx.x;
    for (int i = tid; i < count; i += kBlockThreads) {
        dst[i] = src[i];
    }
}

template <typename T>
__device__ inline void
copy_smem_to_gmem(const T* src, T* dst, int count) {
    int tid = threadIdx.x;
    for (int i = tid; i < count; i += kBlockThreads) {
        dst[i] = src[i];
    }
}

// Main kernel.
// Grid: (N, H, 1)   N = num_seqs, H = num_heads
// Block: kBlockThreads (256)
__global__ void
kda_fwd_v2_naive_kernel(
    const __nv_bfloat16* __restrict__ ws_qd_ptr,    // [total_NT, H, BT, K] bf16
    const __nv_bfloat16* __restrict__ ws_kd_ptr,    // same
    const __nv_bfloat16* __restrict__ ws_kr_ptr,    // same
    const float* __restrict__ ws_gt_ptr,             // [total_NT, H, K] fp32
    const __nv_bfloat16* __restrict__ ws_mqk_ptr,   // [total_NT, H, BT, BT] bf16
    const __nv_bfloat16* __restrict__ ws_inv_ptr,   // same
    const __nv_bfloat16* __restrict__ v_ptr,         // [packed_seq, H, V] bf16
    const float* __restrict__ beta_ptr,              // [packed_seq, H] fp32 (post-sigmoid)
    const float* __restrict__ initial_state_ptr,    // [N, H, V, K] fp32 transposed, or nullptr
    __nv_bfloat16* __restrict__ o_ptr,               // [packed_seq, H, V] bf16
    float* __restrict__ final_state_ptr,             // [N, H, V, K] fp32 transposed
    const int32_t* __restrict__ cu_seqlens,          // [N+1] int32
    const int32_t* __restrict__ chunk_offsets,       // [N+1] int32, prefix sum of NT per seq
    int H,
    int N) {
    int n = blockIdx.x;  // sequence index
    int h = blockIdx.y;  // head index
    int tid = threadIdx.x;

    int bos = cu_seqlens[n];
    int eos = cu_seqlens[n + 1];
    int T_seq = eos - bos;
    int NT = (T_seq + kBT - 1) / kBT;
    int chunk_offset = chunk_offsets[n];   // first chunk index in workspace for this seq

    extern __shared__ char smem_raw[];
    SharedStorageCompact& s = *reinterpret_cast<SharedStorageCompact*>(smem_raw);

    // Load initial_state (or zero) into smem state.
    // initial_state has layout [N, H, V, K] (transposed, FlashKDA convention).
    // Our SMEM state is [K, V]. Load with transpose.
    if (initial_state_ptr != nullptr) {
        const float* init_base = initial_state_ptr + (n * H + h) * kV * kK;
        // init_base[v_idx * K + k_idx] is the (V, K) element.
        // We want state[k_idx, v_idx] = state[k_idx * V + v_idx].
        for (int idx = tid; idx < kK * kV; idx += kBlockThreads) {
            int k = idx / kV;
            int v_ = idx % kV;
            s.state[k * kV + v_] = init_base[v_ * kK + k];
        }
    } else {
        for (int idx = tid; idx < kK * kV; idx += kBlockThreads) {
            s.state[idx] = 0.0f;
        }
    }
    __syncthreads();

    // Per-chunk loop
    for (int c = 0; c < NT; ++c) {
        int chunk_idx = chunk_offset + c;
        int chunk_start = c * kBT;
        int actual_len = min(kBT, T_seq - chunk_start);

        // Load workspace tiles for this chunk × head.
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
        // v: stride along T is H*V (other heads in between)
        for (int idx = tid; idx < kBT * kV; idx += kBlockThreads) {
            int t = idx / kV;
            int v_ = idx % kV;
            if (t < actual_len) {
                s.v[idx] = v_g[t * H * kV + v_];
            } else {
                s.v[idx] = __float2bfloat16(0.0f);
            }
        }
        // beta: stride along T is H
        for (int idx = tid; idx < kBT; idx += kBlockThreads) {
            if (idx < actual_len) {
                s.beta[idx] = beta_g[idx * H];
            } else {
                s.beta[idx] = 0.0f;
            }
        }
        __syncthreads();

        // Step 1: v_pred = ws_kd @ state    [BT, K] @ [K, V] -> [BT, V]
        gemm_smem_bf16<kBT, kV, kK, __nv_bfloat16, float>(s.ws_kd, s.state, s.buf0);  // buf0 = v_pred
        __syncthreads();

        // Step 2: v_proxy = ws_inv @ v      [BT, BT] @ [BT, V] -> [BT, V]
        gemm_smem_bf16<kBT, kV, kBT, __nv_bfloat16, __nv_bfloat16>(s.ws_inv, s.v, s.buf1);  // buf1 = v_proxy
        __syncthreads();

        // Step 3: v_new = (v_proxy - v_pred) * beta   [BT, V] elementwise
        for (int idx = tid; idx < kBT * kV; idx += kBlockThreads) {
            int t = idx / kV;
            float vp = static_cast<float>(s.buf1[idx]);   // v_proxy
            float vd = static_cast<float>(s.buf0[idx]);   // v_pred
            s.buf2[idx] = __float2bfloat16((vp - vd) * s.beta[t]);
        }
        __syncthreads();
        // buf2 = v_new (stays alive for rest of chunk)

        // Step 4: o_intra = ws_mqk @ v_new   [BT, BT] @ [BT, V] -> [BT, V]
        // Reuse buf1 (v_proxy free after step 3) for o_intra.
        gemm_smem_bf16<kBT, kV, kBT, __nv_bfloat16, __nv_bfloat16>(s.ws_mqk, s.buf2, s.buf1);  // buf1 = o_intra
        __syncthreads();

        // Step 5: o_inter = ws_qd @ state   [BT, K] @ [K, V] -> [BT, V]
        // Reuse buf0 (v_pred free) for o_inter.
        gemm_smem_bf16<kBT, kV, kK, __nv_bfloat16, float>(s.ws_qd, s.state, s.buf0);  // buf0 = o_inter
        __syncthreads();

        // Step 6: o = o_inter + o_intra, store to gmem
        __nv_bfloat16* o_g = o_ptr + (bos + chunk_start) * H * kV + h * kV;
        for (int idx = tid; idx < kBT * kV; idx += kBlockThreads) {
            int t = idx / kV;
            int v_ = idx % kV;
            if (t < actual_len) {
                float v_inter = static_cast<float>(s.buf0[idx]);
                float v_intra = static_cast<float>(s.buf1[idx]);
                o_g[t * H * kV + v_] = __float2bfloat16(v_inter + v_intra);
            }
        }
        __syncthreads();

        // Step 7: state = diag(ws_gt) * state + ws_kr^T @ v_new
        state_update<kK, kV, kBT>(s.state, s.ws_gt, s.ws_kr, s.buf2);
        __syncthreads();
    }

    // Store final state with transpose [K, V] -> [V, K]
    if (final_state_ptr != nullptr) {
        float* fin_base = final_state_ptr + (n * H + h) * kV * kK;
        for (int idx = tid; idx < kK * kV; idx += kBlockThreads) {
            int k = idx / kV;
            int v_ = idx % kV;
            fin_base[v_ * kK + k] = s.state[k * kV + v_];
        }
    }
}

inline void
launch_kda_fwd_v2_naive(
    const __nv_bfloat16* ws_qd,
    const __nv_bfloat16* ws_kd,
    const __nv_bfloat16* ws_kr,
    const float* ws_gt,
    const __nv_bfloat16* ws_mqk,
    const __nv_bfloat16* ws_inv,
    const __nv_bfloat16* v,
    const float* beta,
    const float* initial_state,
    __nv_bfloat16* o,
    float* final_state,
    const int32_t* cu_seqlens,
    const int32_t* chunk_offsets,
    int N,
    int H,
    cudaStream_t stream) {
    constexpr size_t smem_bytes = sizeof(SharedStorageCompact);
    static_assert(smem_bytes <= 228 * 1024, "SharedStorageCompact too large for SM90 max SMEM");

    // Enable max dynamic SMEM (up to 228KB on SM90).
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(
            kda_fwd_v2_naive_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes);
        attr_set = true;
    }

    dim3 grid(N, H, 1);
    dim3 block(kBlockThreads);
    kda_fwd_v2_naive_kernel<<<grid, block, smem_bytes, stream>>>(
        ws_qd, ws_kd, ws_kr, ws_gt, ws_mqk, ws_inv,
        v, beta, initial_state,
        o, final_state, cu_seqlens, chunk_offsets,
        H, N);
}

}  // namespace kda::sm90::v2

// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

// Phase 2.2 Step H-2 — raw mma.sync m16n8k16 K2 with fragment-aware bf16 store.
//
// Eliminates the per-WMMA-tile SMEM scratch round-trip by using documented
// mma.sync m16n8k16 fragment layout. Per-thread fp32 accumulator elements are
// cast to bf16 in registers and written directly to bf16 SMEM at the
// fragment-aware (row, col) positions.
//
// mma.sync m16n8k16 layout (PTX ISA 8.0+, sm_80+):
//   For lane t in warp, with g = t/4, s = t%4:
//     A[16][16] (row-major source), 8 bf16 per thread:
//       a[0] = A[g+0, 2s+0]  a[1] = A[g+0, 2s+1]
//       a[2] = A[g+8, 2s+0]  a[3] = A[g+8, 2s+1]
//       a[4] = A[g+0, 2s+8]  a[5] = A[g+0, 2s+9]
//       a[6] = A[g+8, 2s+8]  a[7] = A[g+8, 2s+9]
//     B[16][8] (col-major source — for our row-major SMEM, strided access):
//       b[0] = B[2s+0, g]    b[1] = B[2s+1, g]
//       b[2] = B[2s+8, g]    b[3] = B[2s+9, g]
//     C[16][8] (output, fp32), 4 fp32 per thread:
//       c[0] = C[g+0, 2s+0]  c[1] = C[g+0, 2s+1]
//       c[2] = C[g+8, 2s+0]  c[3] = C[g+8, 2s+1]
//
// For our [BT=64, V=128, K=128] matmul: output decomposed into m16n8 tiles
// → 4 M tiles × 16 N tiles = 64 output tiles → 8 per warp.

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

namespace kda::sm90::v2::frag_impl {

constexpr int kBT = 64;
constexpr int kK = 128;
constexpr int kV = 128;
constexpr int kBlockThreads = 256;
constexpr int kWarps = 8;

struct SharedStorageFrag {
    float state_fp32[kK * kV];                     // 64KB
    __nv_bfloat16 state_bf16[kK * kV];              // 32KB
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
};
// 208.75KB — even smaller than cu_wmma's 216.75KB (no warp_scratch needed!)

// ============================================================================
// Raw mma.sync m16n8k16 helper
// ============================================================================
__device__ __forceinline__ void
mma_m16n8k16_bf16_f32(
    float& c0, float& c1, float& c2, float& c3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1));
}

// Pack two bf16 into one uint32_t (little-endian: low 16 bits = first elem).
__device__ __forceinline__ uint32_t
pack_bf16x2(__nv_bfloat16 a, __nv_bfloat16 b) {
    uint32_t out;
    __nv_bfloat16* p = reinterpret_cast<__nv_bfloat16*>(&out);
    p[0] = a;
    p[1] = b;
    return out;
}

// Load A fragment (row-major SMEM, m16k16 tile, 8 bf16 per thread → 4 b32).
//   smem_A points to start of the m16k16 tile.
//   ld is the leading dimension (stride between rows).
__device__ __forceinline__ void
load_a_frag(uint32_t a[4], const __nv_bfloat16* smem_A, int ld) {
    int t = threadIdx.x % 32;
    int g = t / 4;
    int s = t % 4;
    a[0] = pack_bf16x2(smem_A[(g + 0) * ld + 2 * s + 0],
                        smem_A[(g + 0) * ld + 2 * s + 1]);
    a[1] = pack_bf16x2(smem_A[(g + 8) * ld + 2 * s + 0],
                        smem_A[(g + 8) * ld + 2 * s + 1]);
    a[2] = pack_bf16x2(smem_A[(g + 0) * ld + 2 * s + 8],
                        smem_A[(g + 0) * ld + 2 * s + 9]);
    a[3] = pack_bf16x2(smem_A[(g + 8) * ld + 2 * s + 8],
                        smem_A[(g + 8) * ld + 2 * s + 9]);
}

// Load B fragment (col-major access of row-major SMEM, k16n8 tile, 4 bf16 per thread).
//   smem_B points to start of the k16n8 tile (in row-major SMEM, ld = N_full).
//   ld is the leading dimension of the FULL B matrix's row-major layout.
//   nt_off is the column offset (within the SMEM B) of the n8 tile we're reading.
__device__ __forceinline__ void
load_b_frag(uint32_t b[2], const __nv_bfloat16* smem_B_row_start, int ld) {
    int t = threadIdx.x % 32;
    int g = t / 4;
    int s = t % 4;
    // For col-major B[k][n] read of row-major SMEM B[K][N] (ld=N):
    //   B[k, g] = smem_B[k * ld + g]
    b[0] = pack_bf16x2(smem_B_row_start[(2 * s + 0) * ld + g],
                        smem_B_row_start[(2 * s + 1) * ld + g]);
    b[1] = pack_bf16x2(smem_B_row_start[(2 * s + 8) * ld + g],
                        smem_B_row_start[(2 * s + 9) * ld + g]);
}

// Store C fragment (m16n8 tile, 4 fp32 per thread) directly to bf16 SMEM at
// fragment-aware positions. Skips SMEM scratch + syncwarp + readback.
//   smem_C points to start of full C matrix in row-major SMEM (ld = N_full).
//   m_off, n_off are the (m, n) offsets within smem_C where to write the 16x8 tile.
__device__ __forceinline__ void
store_c_frag(__nv_bfloat16* smem_C, int ld_full,
              int m_off, int n_off,
              float c0, float c1, float c2, float c3) {
    int t = threadIdx.x % 32;
    int g = t / 4;
    int s = t % 4;
    smem_C[(m_off + g + 0) * ld_full + (n_off + 2 * s + 0)] = __float2bfloat16(c0);
    smem_C[(m_off + g + 0) * ld_full + (n_off + 2 * s + 1)] = __float2bfloat16(c1);
    smem_C[(m_off + g + 8) * ld_full + (n_off + 2 * s + 0)] = __float2bfloat16(c2);
    smem_C[(m_off + g + 8) * ld_full + (n_off + 2 * s + 1)] = __float2bfloat16(c3);
}

// Same C fragment store, but also supports masking rows beyond actual_len and
// writing to gmem with strided H heads (for final o output).
__device__ __forceinline__ void
store_c_frag_to_gmem_strided(
    __nv_bfloat16* gmem_o, int H, int kV_,
    int m_off, int n_off,
    float c0, float c1, float c2, float c3,
    int actual_len, float add0, float add1, float add2, float add3) {
    int t = threadIdx.x % 32;
    int g = t / 4;
    int s = t % 4;
    int rows[2] = {m_off + g + 0, m_off + g + 8};
    int cols[2] = {n_off + 2 * s + 0, n_off + 2 * s + 1};
    float vals[2][2] = {{c0 + add0, c1 + add1}, {c2 + add2, c3 + add3}};
    #pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        if (rows[rr] < actual_len) {
            #pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
                gmem_o[rows[rr] * H * kV_ + cols[cc]] = __float2bfloat16(vals[rr][cc]);
            }
        }
    }
}

// Standard gemm with raw mma.sync, fragment-aware bf16 store (no SMEM scratch).
// Computes C[M, N] = A[M, KK] @ B[KK, N], all in row-major SMEM.
// 8 warps share work over (M/16) × (N/8) output tiles.
__device__ inline void
gemm_mma_bf16(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int N, int KK) {
    int warp_id = threadIdx.x / 32;
    int M_tiles = M / 16;
    int N_tiles = N / 8;        // m16n8k16 → N tile is 8
    int K_tiles = KK / 16;
    int total_tiles = M_tiles * N_tiles;
    for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += kWarps) {
        int m_tile = tile_idx / N_tiles;
        int n_tile = tile_idx % N_tiles;
        float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
        for (int k_tile = 0; k_tile < K_tiles; ++k_tile) {
            uint32_t a[4], b[2];
            load_a_frag(a, A + (m_tile * 16) * KK + k_tile * 16, KK);
            load_b_frag(b, B + (k_tile * 16) * N + n_tile * 8, N);
            mma_m16n8k16_bf16_f32(c0, c1, c2, c3, a[0], a[1], a[2], a[3], b[0], b[1]);
        }
        store_c_frag(C, N, m_tile * 16, n_tile * 8, c0, c1, c2, c3);
    }
}

// State update: state[k, v] = ws_gt[k] * state[k, v] + (ws_kr^T @ v_new)[k, v]
// Uses raw mma + fragment-aware update.
//   ws_kr is [BT, K] row-major; we need ws_kr^T[K, BT] for the matmul.
//   For row.col mma, A is row-major and B is col-major. We want:
//     C[K, V] = ws_kr^T[K, BT] @ v_new[BT, V]
//   So A = ws_kr^T (read col-major from row-major ws_kr SMEM is actually rows of ws_kr...)
//   Easier: read ws_kr as row-major B (k=BT contraction), and v_new as col-major B?
//   Simpler: use mma with A = ws_kr (row-major), B = v_new (col-major), and
//     interpret the matmul output as (ws_kr) @ v_new = [BT, V] → not what we want.
//
// Let me re-derive. We want C[k, v] = sum_t ws_kr[t, k] * v_new[t, v].
// This is C = ws_kr^T @ v_new, which in mma terms with row.col layout:
//   - A = ws_kr^T (M=K=128, K_inner=BT=64)
//   - B = v_new (K_inner=BT=64, N=V=128)
//   - C = [128, 128]
// To express A = ws_kr^T from row-major ws_kr SMEM, we read it as col-major.
// In mma.sync.row.col: A is loaded row-major from SMEM. So we need ws_kr^T to
// be in row-major SMEM. But it's not — we have ws_kr (=  ws_kr^T transposed).
//
// Trick: swap A and B roles:
//   C[k, v] = sum_t ws_kr[t, k] * v_new[t, v]
//          = sum_t v_new^T[v, t] * ws_kr[t, k]   (if we transpose accumulator to [V, K])
//          = (v_new^T @ ws_kr)[v, k]
// → C^T = v_new^T @ ws_kr where C^T = state^T[V, K]
// So transposed matmul: A = v_new^T, B = ws_kr (row.col).
// A = v_new^T (M=V, K_inner=BT). For row-major v_new SMEM we read col-major.
// B = ws_kr (K_inner=BT, N=K). For row-major ws_kr SMEM we read col-major.
// row.col means A row-major, B col-major. So A = v_new^T needs row-major SMEM.
//
// Hmm. Easier: just load from ws_kr with col-major access pattern (treat row-major
// as if col-major), load v_new with col-major access, and compute C transposed.
// Let me just use a SEPARATE state_update function that handles the layout via
// per-element math (no fancy mma) for now — it's only one matmul per chunk.
__device__ inline void
state_update_naive(
    float* state_fp32, const float* ws_gt,
    const __nv_bfloat16* ws_kr, const __nv_bfloat16* v_new) {
    int tid = threadIdx.x;
    // state[k, v] = ws_gt[k] * state[k, v] + sum_t ws_kr[t, k] * v_new[t, v]
    for (int idx = tid; idx < kK * kV; idx += kBlockThreads) {
        int k = idx / kV;
        int v_ = idx % kV;
        float acc = 0.0f;
        #pragma unroll 8
        for (int t = 0; t < kBT; ++t) {
            acc += static_cast<float>(ws_kr[t * kK + k])
                 * static_cast<float>(v_new[t * kV + v_]);
        }
        state_fp32[idx] = ws_gt[k] * state_fp32[idx] + acc;
    }
}

template <typename T>
__device__ inline void
copy_gmem_to_smem(const T* src, T* dst, int count) {
    int tid = threadIdx.x;
    for (int i = tid; i < count; i += kBlockThreads) dst[i] = src[i];
}

__global__ void
kda_fwd_v2_frag_kernel(
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

    int bos = cu_seqlens[n];
    int eos = cu_seqlens[n + 1];
    int T_seq = eos - bos;
    int NT = (T_seq + kBT - 1) / kBT;
    int chunk_offset = chunk_offsets[n];

    extern __shared__ char smem_raw[];
    SharedStorageFrag& s = *reinterpret_cast<SharedStorageFrag*>(smem_raw);

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
        for (int idx = tid; idx < kK * kV; idx += kBlockThreads) {
            s.state_bf16[idx] = __float2bfloat16(s.state_fp32[idx]);
        }
        __syncthreads();

        // Step 1: bufA = ws_kd @ state
        gemm_mma_bf16(s.ws_kd, s.state_bf16, s.bufA, kBT, kV, kK);
        __syncthreads();

        // Step 2: bufB = ws_inv @ v
        gemm_mma_bf16(s.ws_inv, s.v, s.bufB, kBT, kV, kBT);
        __syncthreads();

        // Step 3: bufA = (bufB - bufA) * beta
        for (int idx = tid; idx < kBT * kV; idx += kBlockThreads) {
            int t = idx / kV;
            float vp = static_cast<float>(s.bufB[idx]);
            float vd = static_cast<float>(s.bufA[idx]);
            s.bufA[idx] = __float2bfloat16((vp - vd) * s.beta[t]);
        }
        __syncthreads();

        // Step 4: bufB = ws_mqk @ v_new
        gemm_mma_bf16(s.ws_mqk, s.bufA, s.bufB, kBT, kV, kBT);
        __syncthreads();

        // Step 5: o_inter = ws_qd @ state → into state_bf16 region (dead now)
        gemm_mma_bf16(s.ws_qd, s.state_bf16,
                      reinterpret_cast<__nv_bfloat16*>(s.state_bf16),
                      kBT, kV, kK);
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

        // Step 7: state update — naive for now (only one matmul per chunk, less hot)
        state_update_naive(s.state_fp32, s.ws_gt, s.ws_kr, s.bufA);
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
launch_kda_fwd_v2_frag(
    const __nv_bfloat16* ws_qd, const __nv_bfloat16* ws_kd,
    const __nv_bfloat16* ws_kr, const float* ws_gt,
    const __nv_bfloat16* ws_mqk, const __nv_bfloat16* ws_inv,
    const __nv_bfloat16* v, const float* beta,
    const float* initial_state,
    __nv_bfloat16* o, float* final_state,
    const int32_t* cu_seqlens, const int32_t* chunk_offsets,
    int N, int H, cudaStream_t stream) {
    constexpr size_t smem_bytes = sizeof(SharedStorageFrag);
    static_assert(smem_bytes <= 228 * 1024, "SharedStorageFrag too large");

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(
            kda_fwd_v2_frag_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes);
        attr_set = true;
    }

    dim3 grid(N, H, 1);
    dim3 block(kBlockThreads);
    kda_fwd_v2_frag_kernel<<<grid, block, smem_bytes, stream>>>(
        ws_qd, ws_kd, ws_kr, ws_gt, ws_mqk, ws_inv,
        v, beta, initial_state,
        o, final_state, cu_seqlens, chunk_offsets,
        H, N);
}

}  // namespace kda::sm90::v2::frag_impl

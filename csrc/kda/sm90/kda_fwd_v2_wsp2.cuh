// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

// Phase 2.2 Step F — multi-stage warp-specialized C++ K2 kernel.
//
// Adds 2-stage SMEM ping-pong for the workspace tensors. LdSt warp loads
// chunk c+1 into stage[(c+1)%2] while Math warps compute on stage[c%2].
// This OVERLAPS GMEM workspace load with on-chip compute across chunks,
// the missing piece in the single-stage wsp impl.
//
// To fit 2-stage SMEM in 228KB budget, we dropped state_fp32 (64KB) and
// keep state in bf16 only — same precision compromise as FlashKDA.
// Math accumulators for state update are still fp32 in registers, but
// state is stored as bf16 between chunks.
//
// SMEM allocation:
//   state_bf16:           32KB persistent
//   ws_*  × 2 stages:    160KB (qd, kd, kr each 16KB; mqk, inv each 8KB; v 16KB)
//   ws_gt, beta single:    1KB (small, no benefit from doubling)
//   bufA (intermediate):  16KB
//   warp_scratch:          8KB
//   ───────────────────  ─────
//   TOTAL                217KB  (under 228KB max)

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>

namespace kda::sm90::v2::wsp2_impl {

namespace nv = nvcuda;

constexpr int kBT = 64;
constexpr int kK = 128;
constexpr int kV = 128;
constexpr int kBlockThreads = 256;
constexpr int kWarps = 8;
constexpr int kStages = 2;
constexpr int kLdStWarpId = 0;
constexpr int kMathWarpFirst = 1;
constexpr int kMathWarps = 7;

struct SharedStorageWsp2 {
    __nv_bfloat16 state[kK * kV];                 // 32KB persistent (bf16)

    // 2-stage ping-pong for per-chunk workspace
    __nv_bfloat16 ws_qd[kStages][kBT * kK];        // 32KB
    __nv_bfloat16 ws_kd[kStages][kBT * kK];        // 32KB
    __nv_bfloat16 ws_kr[kStages][kBT * kK];        // 32KB
    __nv_bfloat16 ws_mqk[kStages][kBT * kBT];      // 16KB
    __nv_bfloat16 ws_inv[kStages][kBT * kBT];      // 16KB
    __nv_bfloat16 v[kStages][kBT * kV];             // 32KB

    // Single-stage small tensors (loaded fresh each chunk, no doubling worth it)
    float ws_gt[kK];                                // 0.5KB
    float beta[kBT];                                // 0.25KB

    // Intermediate (single, recycled within a chunk)
    __nv_bfloat16 bufA[kBT * kV];                   // 16KB

    // Per-warp scratch for fp32 acc → bf16 cast
    float warp_scratch[kWarps * 16 * 16];           // 8KB
};

__device__ inline void
cp_async_bytes_16B(void* smem_dst, const void* gmem_src, int n_bytes) {
    int chunks = n_bytes / 16;
    char* dst_b = static_cast<char*>(smem_dst);
    const char* src_b = static_cast<const char*>(gmem_src);
    int thread_in_warp = threadIdx.x % 32;
    for (int i = thread_in_warp; i < chunks; i += 32) {
        __pipeline_memcpy_async(dst_b + i * 16, src_b + i * 16, 16);
    }
}

__device__ inline void
gemm_wmma_bf16(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int N, int KK, float* warp_scratch_for_warp,
    int math_warp_id) {
    int lane_id = threadIdx.x % 32;
    int M_tiles = M / 16, N_tiles = N / 16, K_tiles = KK / 16;
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
            int row = i / 16, col = i % 16;
            int c_row = m_tile * 16 + row, c_col = n_tile * 16 + col;
            C[c_row * N + c_col] = __float2bfloat16(warp_scratch_for_warp[i]);
        }
        __syncwarp();
    }
}

// State update for bf16 state: read bf16 state, multiply by gt (fp32),
// add fp32 acc from ws_kr^T @ v_new, cast back to bf16.
__device__ inline void
state_update_bf16(
    __nv_bfloat16* state_bf16, const float* ws_gt,
    const __nv_bfloat16* ws_kr, const __nv_bfloat16* v_new,
    float* warp_scratch_for_warp, int math_warp_id) {
    int lane_id = threadIdx.x % 32;
    int M_tiles = kK / 16, N_tiles = kV / 16, K_tiles = kBT / 16;
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
            int row = i / 16, col = i % 16;
            int g_k = m_tile * 16 + row, g_v = n_tile * 16 + col;
            float old = static_cast<float>(state_bf16[g_k * kV + g_v]);
            float new_val = ws_gt[g_k] * old + warp_scratch_for_warp[i];
            state_bf16[g_k * kV + g_v] = __float2bfloat16(new_val);
        }
        __syncwarp();
    }
}

// Issue cp.async loads for chunk c into stage[stage_idx]. Caller must
// __pipeline_commit() afterwards.
__device__ inline void
issue_chunk_loads(
    SharedStorageWsp2& s, int stage_idx,
    const __nv_bfloat16* qd_g, const __nv_bfloat16* kd_g,
    const __nv_bfloat16* kr_g, const __nv_bfloat16* mqk_g,
    const __nv_bfloat16* inv_g, const __nv_bfloat16* v_g_strided,
    int actual_len, int H) {
    cp_async_bytes_16B(s.ws_qd[stage_idx],  qd_g,  kBT * kK * 2);
    cp_async_bytes_16B(s.ws_kd[stage_idx],  kd_g,  kBT * kK * 2);
    cp_async_bytes_16B(s.ws_kr[stage_idx],  kr_g,  kBT * kK * 2);
    cp_async_bytes_16B(s.ws_mqk[stage_idx], mqk_g, kBT * kBT * 2);
    cp_async_bytes_16B(s.ws_inv[stage_idx], inv_g, kBT * kBT * 2);
    // v: strided in GMEM (other heads interleaved), can't use 16B cp.async directly.
    // Fall back to per-thread loads INSIDE the warp doing this. Single warp = 32 threads
    // needs to load BT*V = 8192 bf16 = 16KB → 512 elements per thread. Slow but OK.
    int tid_in_warp = threadIdx.x % 32;
    for (int idx = tid_in_warp; idx < kBT * kV; idx += 32) {
        int t = idx / kV;
        int v_ = idx % kV;
        s.v[stage_idx][idx] = (t < actual_len) ? v_g_strided[t * H * kV + v_]
                                                : __float2bfloat16(0.0f);
    }
}

__global__ void
kda_fwd_v2_wsp2_kernel(
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
    SharedStorageWsp2& s = *reinterpret_cast<SharedStorageWsp2*>(smem_raw);

    bool is_ldst = (warp_id == kLdStWarpId);
    bool is_math = (warp_id >= kMathWarpFirst);
    int math_warp_id = warp_id - kMathWarpFirst;
    float* my_warp_scratch = s.warp_scratch + warp_id * 256;

    // Load initial state (transposed [N, H, V, K] → SMEM bf16 [K, V])
    if (initial_state_ptr != nullptr) {
        const float* init_base = initial_state_ptr + (n * H + h) * kV * kK;
        for (int idx = tid; idx < kK * kV; idx += kBlockThreads) {
            int k = idx / kV;
            int v_ = idx % kV;
            s.state[k * kV + v_] = __float2bfloat16(init_base[v_ * kK + k]);
        }
    } else {
        for (int idx = tid; idx < kK * kV; idx += kBlockThreads) {
            s.state[idx] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    auto chunk_ptr = [&](int chunk_idx, int dummy) {
        return chunk_idx;  // placeholder; we just inline the offset arithmetic below
    };

    // Helper: compute GMEM base ptrs for chunk c
    auto get_chunk_ptrs = [&](int c) {
        int ci = chunk_offset + c;
        int chunk_start = c * kBT;
        struct Ptrs {
            const __nv_bfloat16 *qd, *kd, *kr, *mqk, *inv, *v;
            const float *gt, *beta;
            int actual_len;
        } p;
        p.qd  = ws_qd_ptr  + (ci * H + h) * kBT * kK;
        p.kd  = ws_kd_ptr  + (ci * H + h) * kBT * kK;
        p.kr  = ws_kr_ptr  + (ci * H + h) * kBT * kK;
        p.gt  = ws_gt_ptr  + (ci * H + h) * kK;
        p.mqk = ws_mqk_ptr + (ci * H + h) * kBT * kBT;
        p.inv = ws_inv_ptr + (ci * H + h) * kBT * kBT;
        p.v   = v_ptr      + (bos + chunk_start) * H * kV + h * kV;
        p.beta = beta_ptr  + (bos + chunk_start) * H + h;
        p.actual_len = min(kBT, T_seq - chunk_start);
        return p;
    };

    // Pre-load chunk 0 into stage 0
    {
        auto p = get_chunk_ptrs(0);
        if (is_ldst) {
            issue_chunk_loads(s, 0, p.qd, p.kd, p.kr, p.mqk, p.inv, p.v, p.actual_len, H);
            __pipeline_commit();
        }
        // Small tensors loaded by all threads (single-stage)
        for (int idx = tid; idx < kK; idx += kBlockThreads) s.ws_gt[idx] = p.gt[idx];
        for (int idx = tid; idx < kBT; idx += kBlockThreads)
            s.beta[idx] = (idx < p.actual_len) ? p.beta[idx * H] : 0.0f;
        if (is_ldst) __pipeline_wait_prior(0);
        __syncthreads();
    }

    for (int c = 0; c < NT; ++c) {
        int curr_stage = c % kStages;
        int next_stage = (c + 1) % kStages;

        // LdSt warp: issue cp.async for chunk c+1 into next_stage
        if (c + 1 < NT && is_ldst) {
            auto p_next = get_chunk_ptrs(c + 1);
            issue_chunk_loads(s, next_stage, p_next.qd, p_next.kd, p_next.kr,
                               p_next.mqk, p_next.inv, p_next.v, p_next.actual_len, H);
            __pipeline_commit();
        }

        // Math warps: refresh state_bf16 view? NOT needed - state IS bf16 now.
        // Compute on chunk c using stage[curr_stage]
        if (is_math) {
            // Step 1: bufA = ws_kd @ state
            gemm_wmma_bf16(s.ws_kd[curr_stage], s.state, s.bufA, kBT, kV, kK,
                           my_warp_scratch, math_warp_id);
        }
        __syncthreads();

        if (is_math) {
            // Step 2: state region (first 16KB) = ws_inv @ v   (reuse state for v_proxy)
            // state region is 32KB total; bf16 [BT, V] = 16KB fits in first half.
            // We only need state_bf16 PRESERVED for step 5 (ws_qd @ state).
            // ⚠ Step 2 OVERWRITING state would break step 5.
            // → Use a different temp. We can use ws_kd[curr_stage] (16KB, freed after step 1).
            gemm_wmma_bf16(s.ws_inv[curr_stage], s.v[curr_stage],
                           reinterpret_cast<__nv_bfloat16*>(s.ws_kd[curr_stage]),
                           kBT, kV, kBT, my_warp_scratch, math_warp_id);
        }
        __syncthreads();

        // Step 3: bufA = (v_proxy_in_ws_kd - bufA) * beta
        for (int idx = tid; idx < kBT * kV; idx += kBlockThreads) {
            int t = idx / kV;
            __nv_bfloat16* v_proxy = reinterpret_cast<__nv_bfloat16*>(s.ws_kd[curr_stage]);
            float vp = static_cast<float>(v_proxy[idx]);
            float vd = static_cast<float>(s.bufA[idx]);
            s.bufA[idx] = __float2bfloat16((vp - vd) * s.beta[t]);
        }
        __syncthreads();
        // bufA = v_new (lives till state update)

        if (is_math) {
            // Step 4: o_intra in ws_kd region (overwrite v_proxy which is dead)
            gemm_wmma_bf16(s.ws_mqk[curr_stage], s.bufA,
                           reinterpret_cast<__nv_bfloat16*>(s.ws_kd[curr_stage]),
                           kBT, kV, kBT, my_warp_scratch, math_warp_id);
        }
        __syncthreads();
        // ws_kd region holds o_intra (first 16KB)

        if (is_math) {
            // Step 5: o_inter in ws_inv region (16KB; reuse since inv is dead after step 2)
            gemm_wmma_bf16(s.ws_qd[curr_stage], s.state,
                           reinterpret_cast<__nv_bfloat16*>(s.ws_inv[curr_stage]),
                           kBT, kV, kK, my_warp_scratch, math_warp_id);
        }
        __syncthreads();

        // Step 6: o = o_inter + o_intra → store to gmem
        __nv_bfloat16* o_g = o_ptr + (bos + c * kBT) * H * kV + h * kV;
        const __nv_bfloat16* o_intra = reinterpret_cast<const __nv_bfloat16*>(s.ws_kd[curr_stage]);
        const __nv_bfloat16* o_inter = reinterpret_cast<const __nv_bfloat16*>(s.ws_inv[curr_stage]);
        int actual_len = min(kBT, T_seq - c * kBT);
        for (int idx = tid; idx < kBT * kV; idx += kBlockThreads) {
            int t = idx / kV;
            int v_ = idx % kV;
            if (t < actual_len) {
                float v_inter = static_cast<float>(o_inter[idx]);
                float v_intra = static_cast<float>(o_intra[idx]);
                o_g[t * H * kV + v_] = __float2bfloat16(v_inter + v_intra);
            }
        }
        __syncthreads();

        if (is_math) {
            // Step 7: state update (in-place, bf16)
            state_update_bf16(s.state, s.ws_gt, s.ws_kr[curr_stage], s.bufA,
                              my_warp_scratch, math_warp_id);
        }

        // Wait for next chunk's cp.async (issued at start of this iteration)
        if (c + 1 < NT) {
            if (is_ldst) __pipeline_wait_prior(0);
            // Also need to load next chunk's small tensors (ws_gt, beta) — single-stage
            // ⚠ This OVERWRITES current chunk's small tensors. But we're done with them
            // (state update doesn't need ws_gt? Wait it does — step 7 uses ws_gt).
            // Order: state update FIRST (uses curr_stage ws_kr + bufA + ws_gt), THEN
            // load small tensors for next chunk. The state_update is in 'is_math' above,
            // so we need to sync between state_update and small-tensor reload.
        }
        __syncthreads();

        // Reload small tensors for next chunk (after step 7 done)
        if (c + 1 < NT) {
            auto p_next = get_chunk_ptrs(c + 1);
            for (int idx = tid; idx < kK; idx += kBlockThreads) s.ws_gt[idx] = p_next.gt[idx];
            for (int idx = tid; idx < kBT; idx += kBlockThreads)
                s.beta[idx] = (idx < p_next.actual_len) ? p_next.beta[idx * H] : 0.0f;
            __syncthreads();
        }
    }

    // Store final state (cast bf16 → fp32 → transposed [N, H, V, K])
    if (final_state_ptr != nullptr) {
        float* fin_base = final_state_ptr + (n * H + h) * kV * kK;
        for (int idx = tid; idx < kK * kV; idx += kBlockThreads) {
            int k = idx / kV;
            int v_ = idx % kV;
            fin_base[v_ * kK + k] = static_cast<float>(s.state[k * kV + v_]);
        }
    }
}

inline void
launch_kda_fwd_v2_wsp2(
    const __nv_bfloat16* ws_qd, const __nv_bfloat16* ws_kd,
    const __nv_bfloat16* ws_kr, const float* ws_gt,
    const __nv_bfloat16* ws_mqk, const __nv_bfloat16* ws_inv,
    const __nv_bfloat16* v, const float* beta,
    const float* initial_state,
    __nv_bfloat16* o, float* final_state,
    const int32_t* cu_seqlens, const int32_t* chunk_offsets,
    int N, int H, cudaStream_t stream) {
    constexpr size_t smem_bytes = sizeof(SharedStorageWsp2);
    static_assert(smem_bytes <= 228 * 1024, "SharedStorageWsp2 too large");

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(
            kda_fwd_v2_wsp2_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes);
        attr_set = true;
    }

    dim3 grid(N, H, 1);
    dim3 block(kBlockThreads);
    kda_fwd_v2_wsp2_kernel<<<grid, block, smem_bytes, stream>>>(
        ws_qd, ws_kd, ws_kr, ws_gt, ws_mqk, ws_inv,
        v, beta, initial_state,
        o, final_state, cu_seqlens, chunk_offsets,
        H, N);
}

}  // namespace kda::sm90::v2::wsp2_impl

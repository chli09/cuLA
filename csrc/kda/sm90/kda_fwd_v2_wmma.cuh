// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

// Phase 2.2 Step D — WMMA-based K2 kernel using Tensor Cores.
//
// Same K2 math as kda_fwd_v2_naive.cuh, but matmuls go through Tensor Cores
// via the WMMA API (m16n16k16 bf16 inputs, fp32 accumulator). 8 warps × 256
// threads per block, parallel over output 16×16 tiles.
//
// State is kept fp32 in SMEM for precision, but a bf16 view is materialized
// once per chunk (in dedicated state_bf16 SMEM) so WMMA can read it. State
// updates happen in fp32 and the bf16 view is refreshed at the next chunk.

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <mma.h>

namespace kda::sm90::v2::wmma_impl {

namespace nv = nvcuda;

constexpr int kBT = 64;
constexpr int kK = 128;
constexpr int kV = 128;
constexpr int kBlockThreads = 256;
constexpr int kWarps = 8;

// SMEM storage. Lifetime-aliased buffers as in naive impl + a dedicated
// state_bf16 (32KB) used for matmul reads. State_fp32 stays as truth.
struct SharedStorageWmma {
    // 64KB persistent, fp32 truth state across chunks
    float state_fp32[kK * kV];

    // 32KB bf16 view of state, refreshed once per chunk for matmul reads
    __nv_bfloat16 state_bf16[kK * kV];

    // Per-chunk workspace tiles (loaded fresh each chunk)
    __nv_bfloat16 ws_qd[kBT * kK];   // 16KB
    __nv_bfloat16 ws_kd[kBT * kK];   // 16KB
    __nv_bfloat16 ws_kr[kBT * kK];   // 16KB
    float ws_gt[kK];                  // 0.5KB
    __nv_bfloat16 ws_mqk[kBT * kBT]; // 8KB
    __nv_bfloat16 ws_inv[kBT * kBT]; // 8KB
    __nv_bfloat16 v[kBT * kV];        // 16KB
    float beta[kBT];                  // 0.25KB

    // Reusable bf16 buffers for intermediates: 3 × 16KB = 48KB
    __nv_bfloat16 buf0[kBT * kV];
    __nv_bfloat16 buf1[kBT * kV];
    __nv_bfloat16 buf2[kBT * kV];

    // Per-warp fp32 scratch for WMMA accumulator → bf16 cast (8 warps × 1KB)
    float warp_scratch[kWarps * 16 * 16];
};
// Total: 64 + 32 + 16*4 + 0.5 + 8*2 + 16 + 0.25 + 16*3 + 8 = 248.75 KB
// EXCEEDS 228KB! Need to drop some intermediate. We can drop buf2 by keeping
// v_new in fp32 form via state_bf16 buffer (recycled), but simpler: drop one
// of buf0/buf1 and ping-pong more aggressively. See revised below.

// Revised: drop one intermediate (buf0 reused more), recover 16KB
struct SharedStorageWmmaCompact {
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
    float warp_scratch[kWarps * 16 * 16];            // 8KB
};
// Total: 64 + 32 + 16*3 + 0.5 + 8*2 + 16 + 0.25 + 16*2 + 8 = 216.75 KB. Fits.

// Buffer reuse plan (within one chunk):
//   bufA: v_pred (step 1) → o_inter (step 5) → reused
//   bufB: v_proxy (step 2) → v_new (step 3) → o_intra (step 4) → ... wait
//          v_new is needed in steps 4, 5 (o_intra), 7 (state update). Conflict.
//
// Adjusted: use bufA for v_new (lives steps 3..7), bufB ping-pongs:
//   bufB step 1: v_pred  → bufA step 3 reads bufB → ...
//   Actually let's just have:
//     bufA = v_pred (step 1) → v_new (step 3) → ... (lives 3..7)
//     bufB = v_proxy (step 2) → o_intra (step 4) → o_inter (step 5)
//   At step 3: v_pred no longer needed after producing v_new. Compute v_new in
//     register-temp from bufA(=v_pred) - bufB(=v_proxy), write to bufA.
//   At step 4: bufA = v_new (ready); ws_mqk @ v_new produces o_intra in bufB
//     (overwriting v_proxy which is dead).
//   At step 5: ws_qd @ state produces o_inter. We need a temp for o_inter that
//     doesn't clobber v_new (bufA) or o_intra (bufB). HMMMM no buffers left.

// OK alternative: do o_inter and o_intra summation into a single intermediate.
// Compute o_intra into bufB (step 4). Then in step 5, compute o_inter += bufB
// pointwise into output gmem directly (skip storing o_inter to SMEM).

__device__ inline void
gemm_wmma_bf16(
    const __nv_bfloat16* A,  // [M, KK] row-major in SMEM
    const __nv_bfloat16* B,  // [KK, N] row-major in SMEM
    __nv_bfloat16* C,        // [M, N] row-major in SMEM
    int M, int N, int KK,
    float* warp_scratch_for_warp  // [16*16] fp32 scratch
) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int M_tiles = M / 16;
    int N_tiles = N / 16;
    int K_tiles = KK / 16;
    int total_tiles = M_tiles * N_tiles;

    for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += kWarps) {
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

        // Store fp32 acc to per-warp scratch, then cast to bf16 and write to C.
        nv::wmma::store_matrix_sync(warp_scratch_for_warp, c_frag, 16, nv::wmma::mem_row_major);
        __syncwarp();

        // Cast scratch (fp32 [16, 16]) to C tile (bf16 [16, 16] at offset (m_tile, n_tile))
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

// State update: state[k, v] = ws_gt[k] * state[k, v] + sum_t ws_kr[t, k] * v_new[t, v]
// Use WMMA for the sum_t part: compute ws_kr^T @ v_new = [K, V], then add into state.
// ws_kr is [BT, K] in SMEM row-major. Transpose semantically: ws_kr^T[K, BT].
// For WMMA matrix_a [K, BT] row-major → leading dim BT. But ws_kr is laid out [BT, K]
// row-major. If we want to use it as [K, BT] row-major, we need data at
// flat[k * BT + t]. But it's stored at flat[t * K + k] (BT-stride is K, K-stride is 1).
// → matrix_a column_major works: load_matrix_sync with col_major treats ptr layout as
//   [k, t] with k-stride 1 (innermost), t-stride = ld. Setting ld = K reads ws_kr[t, k]
//   correctly as A^T[k, t]. ✓
__device__ inline void
state_update_wmma(
    float* state_fp32,
    const float* ws_gt,
    const __nv_bfloat16* ws_kr,    // [BT, K] row-major
    const __nv_bfloat16* v_new,    // [BT, V] row-major
    float* warp_scratch_for_warp) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int M_tiles = kK / 16;   // 8
    int N_tiles = kV / 16;   // 8
    int K_tiles = kBT / 16;  // 4
    int total_tiles = M_tiles * N_tiles;  // 64 tiles, 8 per warp

    for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += kWarps) {
        int m_tile = tile_idx / N_tiles;  // K-axis tile
        int n_tile = tile_idx % N_tiles;  // V-axis tile

        nv::wmma::fragment<nv::wmma::accumulator, 16, 16, 16, float> c_frag;
        nv::wmma::fill_fragment(c_frag, 0.0f);

        for (int k_tile = 0; k_tile < K_tiles; ++k_tile) {
            // A = ws_kr^T[K, BT], read column-major from ws_kr stored as [BT, K] row-major
            nv::wmma::fragment<nv::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nv::wmma::col_major> a_frag;
            // B = v_new[BT, V] row-major
            nv::wmma::fragment<nv::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nv::wmma::row_major> b_frag;

            // ws_kr^T tile [m_tile*16:m_tile*16+16, k_tile*16:k_tile*16+16] (K x BT view)
            // = ws_kr[k_tile*16:k_tile*16+16, m_tile*16:m_tile*16+16] (BT x K original)
            // Read ws_kr starting at row k_tile*16, col m_tile*16, ld = K (BT-stride for col-major view)
            nv::wmma::load_matrix_sync(a_frag, ws_kr + (k_tile * 16) * kK + m_tile * 16, kK);
            nv::wmma::load_matrix_sync(b_frag, v_new + (k_tile * 16) * kV + n_tile * 16, kV);
            nv::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // c_frag holds ws_kr^T @ v_new for this [16, 16] block of state[K, V].
        // state[k, v] = ws_gt[k] * state[k, v] + c_frag[k_local, v_local]
        nv::wmma::store_matrix_sync(warp_scratch_for_warp, c_frag, 16, nv::wmma::mem_row_major);
        __syncwarp();

        for (int i = lane_id; i < 16 * 16; i += 32) {
            int row = i / 16;       // local k row in tile
            int col = i % 16;       // local v col in tile
            int g_k = m_tile * 16 + row;
            int g_v = n_tile * 16 + col;
            float new_val = ws_gt[g_k] * state_fp32[g_k * kV + g_v] + warp_scratch_for_warp[i];
            state_fp32[g_k * kV + g_v] = new_val;
        }
        __syncwarp();
    }
}

template <typename T>
__device__ inline void
copy_gmem_to_smem(const T* src, T* dst, int count) {
    int tid = threadIdx.x;
    for (int i = tid; i < count; i += kBlockThreads) {
        dst[i] = src[i];
    }
}

__global__ void
kda_fwd_v2_wmma_kernel(
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
    int H,
    int N) {
    int n = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = threadIdx.x / 32;

    int bos = cu_seqlens[n];
    int eos = cu_seqlens[n + 1];
    int T_seq = eos - bos;
    int NT = (T_seq + kBT - 1) / kBT;
    int chunk_offset = chunk_offsets[n];

    extern __shared__ char smem_raw[];
    SharedStorageWmmaCompact& s = *reinterpret_cast<SharedStorageWmmaCompact*>(smem_raw);

    float* my_warp_scratch = s.warp_scratch + warp_id * 256;

    // Load initial_state (transposed [N, H, V, K] → SMEM [K, V])
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

        // Refresh state_bf16 view from state_fp32
        for (int idx = tid; idx < kK * kV; idx += kBlockThreads) {
            s.state_bf16[idx] = __float2bfloat16(s.state_fp32[idx]);
        }
        __syncthreads();

        // Step 1: bufA = ws_kd @ state_bf16    [BT, K] @ [K, V] -> [BT, V]
        gemm_wmma_bf16(s.ws_kd, s.state_bf16, s.bufA, kBT, kV, kK, my_warp_scratch);
        __syncthreads();

        // Step 2: bufB = ws_inv @ v   [BT, BT] @ [BT, V] -> [BT, V]
        gemm_wmma_bf16(s.ws_inv, s.v, s.bufB, kBT, kV, kBT, my_warp_scratch);
        __syncthreads();

        // Step 3: bufA = (bufB - bufA) * beta   (v_new = (v_proxy - v_pred) * beta)
        for (int idx = tid; idx < kBT * kV; idx += kBlockThreads) {
            int t = idx / kV;
            float vp = static_cast<float>(s.bufB[idx]);
            float vd = static_cast<float>(s.bufA[idx]);
            s.bufA[idx] = __float2bfloat16((vp - vd) * s.beta[t]);
        }
        __syncthreads();
        // bufA now = v_new (lives till state update at end of chunk)

        // Step 4: bufB = ws_mqk @ v_new   [BT, BT] @ [BT, V] -> [BT, V] (overwrite v_proxy)
        gemm_wmma_bf16(s.ws_mqk, s.bufA, s.bufB, kBT, kV, kBT, my_warp_scratch);
        __syncthreads();
        // bufB = o_intra

        // Step 5: state_bf16 already set; compute o_inter = ws_qd @ state into per-warp scratch
        // and add o_intra (bufB) → write directly to gmem o.
        // Implementation: re-use the state_bf16 read; call gemm_wmma_bf16 to produce o_inter in
        // ... wait we need a buffer for o_inter that's not bufA (v_new alive) or bufB (o_intra).
        // Solution: write o_inter into scratch via WMMA and combine pointwise inline using
        // the result + bufB[i] → store gmem.
        //
        // For simplicity: compute o_inter into bufB (overwrite o_intra), but first save
        // o_intra as: o[i] = o_intra[i] + ... after gemm produces o_inter, we'd lose o_intra.
        //
        // Alternative: use state_bf16 SMEM as temp output for o_inter (state_bf16 no longer
        // needed after step 5; we'll refresh next chunk). 32KB is plenty for [BT, V] = 16KB bf16.
        gemm_wmma_bf16(s.ws_qd, s.state_bf16, reinterpret_cast<__nv_bfloat16*>(s.state_bf16),
                       kBT, kV, kK, my_warp_scratch);
        // ⚠ Above writes into state_bf16 (32KB fp32 / 64KB bf16 if interpreted, but SMEM is
        // 32KB bf16). [BT, V] = 64*128 = 8192 bf16 = 16KB → fits within state_bf16 (32KB).
        __syncthreads();
        // Now state_bf16 (first 16KB) = o_inter. bufB = o_intra. bufA = v_new (preserved).

        // Step 6: o = o_inter + o_intra → store to gmem
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

        // Step 7: state_fp32 = diag(ws_gt) * state_fp32 + ws_kr^T @ v_new
        state_update_wmma(s.state_fp32, s.ws_gt, s.ws_kr, s.bufA, my_warp_scratch);
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
launch_kda_fwd_v2_wmma(
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
    constexpr size_t smem_bytes = sizeof(SharedStorageWmmaCompact);
    static_assert(smem_bytes <= 228 * 1024, "SharedStorageWmmaCompact too large for SM90 max SMEM");

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(
            kda_fwd_v2_wmma_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes);
        attr_set = true;
    }

    dim3 grid(N, H, 1);
    dim3 block(kBlockThreads);
    kda_fwd_v2_wmma_kernel<<<grid, block, smem_bytes, stream>>>(
        ws_qd, ws_kd, ws_kr, ws_gt, ws_mqk, ws_inv,
        v, beta, initial_state,
        o, final_state, cu_seqlens, chunk_offsets,
        H, N);
}

}  // namespace kda::sm90::v2::wmma_impl

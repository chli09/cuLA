// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

namespace kda::sm90::v2 {

// Phase 2.2 Step B — minimal C++ K2 stub.
//
// CONTRACT (the math K2 must implement, per FlashKDA's K2; this stub does NOT
// implement it yet — placeholder only):
//
// For each (seq, head) sequentially over chunks c = 0 .. NT-1:
//     v_pred  = ws_kd[c]  @ state         // [BT, K] @ [K, V] = [BT, V]
//     v_proxy = ws_inv[c] @ v[c]          // [BT, BT] @ [BT, V]
//     v_new   = (v_proxy - v_pred) * beta_post_INV_or_pre  (see refs/notes)
//     o_chunk = ws_qd[c] @ state + ws_mqk[c] @ v_new
//     state   = diag(ws_gt[c]) @ state + ws_kr[c]^T @ v_new
//     write o_chunk to o[c]
// final_state = state
//
// Workspace shapes (produced by Triton K1 in cula/kda/hopper_k1.py):
//     ws_qd  : [total_NT, H, BT=64, K=128]  bf16
//     ws_kd  : same                          bf16
//     ws_kr  : same                          bf16
//     ws_gt  : [total_NT, H, K]              fp32
//     ws_mqk : [total_NT, H, BT, BT]         bf16  (lower-tri populated)
//     ws_inv : [total_NT, H, BT, BT]         bf16
//
// Other I/O:
//     v             : [packed_seq, H, V] bf16
//     beta          : [packed_seq, H]    fp32 (post-sigmoid)
//     initial_state : [N, H, V, K]       fp32  (transposed layout, FlashKDA convention)
//                     or nullptr for zero start
//     o             : [packed_seq, H, V] bf16  (output)
//     final_state   : [N, H, V, K]       fp32  (output, transposed)
//     cu_seqlens    : [N+1] int32
//
// STUB BEHAVIOR (this file): writes zeros to `o` and `final_state`. The Python
// API still matches FlashKDA via the Python K2-ref fallback while this stub
// gets fleshed out.

// Stub kernel: fill output with zeros so we can validate the launch pipeline
// without producing meaningful results.
__global__ void
kda_fwd_v2_stub_zero_kernel(__nv_bfloat16* out, int64_t total_elements) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        out[idx] = __float2bfloat16(0.0f);
    }
}

__global__ void
kda_fwd_v2_stub_zero_fp32_kernel(float* out, int64_t total_elements) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        out[idx] = 0.0f;
    }
}

// Launch stub: output = zeros, final_state = zeros.
inline void
launch_kda_fwd_v2_stub(
    __nv_bfloat16* o_ptr,
    float* final_state_ptr,
    int64_t o_numel,
    int64_t state_numel,
    cudaStream_t stream) {
    constexpr int kThreads = 256;
    int64_t blocks_o = (o_numel + kThreads - 1) / kThreads;
    int64_t blocks_s = (state_numel + kThreads - 1) / kThreads;
    if (blocks_o > 0) {
        kda_fwd_v2_stub_zero_kernel<<<blocks_o, kThreads, 0, stream>>>(o_ptr, o_numel);
    }
    if (blocks_s > 0 && final_state_ptr != nullptr) {
        kda_fwd_v2_stub_zero_fp32_kernel<<<blocks_s, kThreads, 0, stream>>>(final_state_ptr, state_numel);
    }
}

}  // namespace kda::sm90::v2

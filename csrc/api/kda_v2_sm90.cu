// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

// Phase 2.2 Step B — kda_fwd_v2 PyTorch entry point + dispatch to the stub kernel.
// Real K2 math will replace the stub in a follow-up change.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include "kda/sm90/kda_fwd_v2_stub.cuh"

using OptionalTensor = std::optional<torch::Tensor>;

std::tuple<torch::Tensor, torch::Tensor>
kda_fwd_v2(
    OptionalTensor output_,
    OptionalTensor output_state_,
    torch::Tensor const& v,
    torch::Tensor const& beta,
    torch::Tensor const& ws_qd,
    torch::Tensor const& ws_kd,
    torch::Tensor const& ws_kr,
    torch::Tensor const& ws_gt,
    torch::Tensor const& ws_mqk,
    torch::Tensor const& ws_inv,
    OptionalTensor initial_state_,
    torch::Tensor const& cu_seqlens,
    int64_t chunk_size) {
    // Shape assumptions (mirror cuLA Hopper conventions):
    //   v: [packed_seq, H, V] bf16
    //   beta: [packed_seq, H] fp32 (post-sigmoid)
    //   ws_*: as documented in kda_fwd_v2_stub.cuh
    //   initial_state: [N, H, V, K] fp32 transposed, or absent
    //   cu_seqlens: [N+1] int32

    TORCH_CHECK(v.dim() == 3, "v must be [packed_seq, H, V]");
    TORCH_CHECK(v.dtype() == torch::kBFloat16, "v must be bfloat16");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");

    auto packed_seq = v.size(0);
    auto num_heads  = v.size(1);
    auto v_dim      = v.size(2);
    auto k_dim      = ws_qd.size(-1);
    auto num_seqs   = cu_seqlens.size(0) - 1;

    TORCH_CHECK(ws_qd.dtype() == torch::kBFloat16, "ws_qd must be bfloat16");
    TORCH_CHECK(ws_kd.dtype() == torch::kBFloat16, "ws_kd must be bfloat16");
    TORCH_CHECK(ws_kr.dtype() == torch::kBFloat16, "ws_kr must be bfloat16");
    TORCH_CHECK(ws_gt.dtype() == torch::kFloat32,  "ws_gt must be float32");
    TORCH_CHECK(ws_mqk.dtype() == torch::kBFloat16, "ws_mqk must be bfloat16");
    TORCH_CHECK(ws_inv.dtype() == torch::kBFloat16, "ws_inv must be bfloat16");
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32, "cu_seqlens must be int32");

    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(v.device());
    auto opts_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(v.device());

    torch::Tensor output = output_.has_value()
        ? output_.value()
        : torch::empty({packed_seq, num_heads, v_dim}, opts_bf16);
    torch::Tensor output_state = output_state_.has_value()
        ? output_state_.value()
        : torch::empty({num_seqs, num_heads, v_dim, k_dim}, opts_fp32);

    auto stream = at::cuda::getCurrentCUDAStream();

    // STUB: just zero out the outputs. Replace with real K2 math in a follow-up.
    kda::sm90::v2::launch_kda_fwd_v2_stub(
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        output_state.data_ptr<float>(),
        output.numel(),
        output_state.numel(),
        stream);

    return {output, output_state};
}

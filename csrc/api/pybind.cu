// Copyright 2025-2026 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/nn/functional.h>
#include <torch/python.h>

#if defined(CULA_SM100_ENABLED) || defined(CULA_SM103_ENABLED)
void
ChunkKDAFwdIntra(
    at::Tensor q,
    at::Tensor k,
    at::Tensor g,
    at::Tensor beta,
    at::Tensor cu_seqlens,
    at::Tensor chunk_indices,
    at::Tensor Aqk_out,
    at::Tensor Akk_out,
    at::Tensor tile_counter,
    float scale,
    int chunk_size,
    bool use_tf32_inverse,
    bool unified_gref);
void
ChunkKDAFwdRecompWU(
    at::Tensor k,
    at::Tensor v,
    at::Tensor beta,
    at::Tensor A,
    at::Tensor g,
    at::Tensor cu_seqlens,
    at::Tensor chunk_indices,
    at::Tensor w_out,
    at::Tensor u_out,
    at::Tensor kg_out,
    int chunk_size,
    std::optional<at::Tensor> q,
    std::optional<at::Tensor> qg_out);
#endif

#if defined(CULA_SM90A_ENABLED)
std::tuple<torch::Tensor, std::optional<torch::Tensor>>
kda_fwd_prefill(
    std::optional<torch::Tensor> output_,
    std::optional<torch::Tensor> output_state_,
    torch::Tensor const& q,
    torch::Tensor const& k,
    torch::Tensor const& v,
    std::optional<torch::Tensor> input_state_,
    std::optional<torch::Tensor> alpha_,
    std::optional<torch::Tensor> beta_,
    torch::Tensor const& cu_seqlens,
    torch::Tensor workspace_buffer,
    float scale,
    bool output_final_state,
    bool safe_gate);

int64_t flashkda_get_workspace_size(int64_t T_total, int64_t H, int64_t N);

void flashkda_fwd_prefill(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor g,
    torch::Tensor beta,
    double scale,
    torch::Tensor out,
    torch::Tensor workspace,
    torch::Tensor A_log,
    torch::Tensor dt_bias,
    double lower_bound,
    std::optional<torch::Tensor> initial_state,
    std::optional<torch::Tensor> final_state,
    std::optional<torch::Tensor> cu_seqlens);
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "cuLA";
#if defined(CULA_SM100_ENABLED) || defined(CULA_SM103_ENABLED)
    m.def("chunk_kda_fwd_intra_cuda", &ChunkKDAFwdIntra);
    m.def("recompute_w_u_cuda", &ChunkKDAFwdRecompWU);
#endif
#if defined(CULA_SM90A_ENABLED)
    m.def("kda_fwd_prefill", &kda_fwd_prefill);
    m.def(
        "flashkda_fwd_prefill",
        &flashkda_fwd_prefill,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("g"),
        py::arg("beta"),
        py::arg("scale"),
        py::arg("out"),
        py::arg("workspace"),
        py::arg("A_log"),
        py::arg("dt_bias"),
        py::arg("lower_bound"),
        py::arg("initial_state") = py::none(),
        py::arg("final_state") = py::none(),
        py::arg("cu_seqlens") = py::none());
    m.def(
        "flashkda_get_workspace_size",
        &flashkda_get_workspace_size,
        py::arg("T_total"),
        py::arg("H"),
        py::arg("N") = 1);
#endif
}

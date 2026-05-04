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

// Dispatch function only — does NOT include the .cuh to avoid
// implicit instantiation of all kernel variants in one TU.
// Each SafeGate variant is explicitly instantiated in its own .cu file.

#include <cute/numeric/numeric_types.hpp>
#include <cutlass/arch/arch.h>

namespace kda::sm90 {

using namespace cute;

// Forward declaration of the per-variant launcher (defined in .cuh, instantiated in separate TUs)
template <
    bool NeedsBeta,
    bool NeedsAlpha,
    bool InitStateFromInput,
    bool SafeGate,
    int NumSegments,
    typename ArchTag,
    typename TO,
    typename TQKV,
    typename TState,
    typename TBeta = float>
void
launch_kda_fwd_prefill_kernel_gbai(
    cudaStream_t stream,
    TO* output,
    TState* output_state,
    TQKV const* q,
    TQKV const* k,
    TQKV const* v,
    TState const* input_state,
    float const* alpha,
    TBeta const* beta,
    int32_t const* cu_seqlens,
    uint8_t* workspace_buffer,
    int32_t num_seqs,
    int32_t num_heads,
    int32_t head_size,
    int64_t total_seqlen,
    float scale,
    int32_t sm_count);

template <
    typename ArchTag,  // TODO: hide this
    typename TO,
    typename TQKV,
    typename TState,
    typename TBeta>
void
launch_kda_fwd_prefill_kernel(
    cudaStream_t stream,
    TO* output,
    TState* output_state,
    TQKV const* q,
    TQKV const* k,
    TQKV const* v,
    TState const* input_state,
    float const* alpha,
    TBeta const* beta,
    int32_t const* cu_seqlens,
    uint8_t* workspace_buffer,
    int32_t num_seqs,
    int32_t num_heads,
    int32_t head_size,
    int64_t total_seqlen,
    float scale,
    bool safe_gate,
    int32_t num_segments,
    int32_t sm_count) {
    bool needs_beta = beta != nullptr;
    bool needs_alpha = alpha != nullptr;
    bool init_state = input_state != nullptr;

#define LAUNCH_NSEG(NSEG, needs_beta, needs_alpha, init_state, safe_gate)                              \
    launch_kda_fwd_prefill_kernel_gbai<needs_beta, needs_alpha, init_state, safe_gate, NSEG, ArchTag>( \
        stream,                                                                                        \
        output,                                                                                        \
        output_state,                                                                                  \
        q,                                                                                             \
        k,                                                                                             \
        v,                                                                                             \
        input_state,                                                                                   \
        alpha,                                                                                         \
        beta,                                                                                          \
        cu_seqlens,                                                                                    \
        workspace_buffer,                                                                              \
        num_seqs,                                                                                      \
        num_heads,                                                                                     \
        head_size,                                                                                     \
        total_seqlen,                                                                                  \
        scale,                                                                                         \
        sm_count)

#define LAUNCH(needs_beta, needs_alpha, init_state, safe_gate)                                                                  \
    do {                                                                                                                        \
        if (num_segments == 1) {                                                                                                \
            LAUNCH_NSEG(1, needs_beta, needs_alpha, init_state, safe_gate);                                                     \
        } else if (num_segments == 2) {                                                                                         \
            LAUNCH_NSEG(2, needs_beta, needs_alpha, init_state, safe_gate);                                                     \
        } else if (num_segments == 4) {                                                                                         \
            LAUNCH_NSEG(4, needs_beta, needs_alpha, init_state, safe_gate);                                                     \
        } else if (num_segments == 8) {                                                                                         \
            LAUNCH_NSEG(8, needs_beta, needs_alpha, init_state, safe_gate);                                                     \
        } else if (num_segments == 16) {                                                                                        \
            LAUNCH_NSEG(16, needs_beta, needs_alpha, init_state, safe_gate);                                                    \
        } else if (num_segments == 32) {                                                                                        \
            LAUNCH_NSEG(32, needs_beta, needs_alpha, init_state, safe_gate);                                                    \
        } else {                                                                                                                \
            throw std::runtime_error("unsupported num_segments (only {1, 2, 4, 8, 16, 32} compiled): " + std::to_string(num_segments)); \
        }                                                                                                                       \
    } while (0)

    if (init_state) {
        if (needs_beta && needs_alpha && safe_gate) {
            LAUNCH(true, true, true, true);
        } else {
            throw std::runtime_error("unreachable");
        }
    } else {
        if (needs_beta && needs_alpha && safe_gate) {
            LAUNCH(true, true, false, true);
        } else {
            throw std::runtime_error("unreachable");
        }
    }

#undef LAUNCH
#undef LAUNCH_NSEG
}

using bf16 = cute::bfloat16_t;

// TBeta=float (default)
template void
launch_kda_fwd_prefill_kernel<cutlass::arch::Sm90, bf16, bf16, float, float>(
    cudaStream_t stream,
    bf16* output,
    float* state,
    bf16 const* q,
    bf16 const* k,
    bf16 const* v,
    float const* input_state,
    float const* alpha,
    float const* beta,
    int32_t const* cu_seqlens,
    uint8_t* workspace_buffer,
    int32_t num_seqs,
    int32_t num_heads,
    int32_t head_size,
    int64_t total_seqlen,
    float scale,
    bool safe_gate,
    int32_t num_segments,
    int32_t sm_count);

// TBeta=bf16
template void
launch_kda_fwd_prefill_kernel<cutlass::arch::Sm90, bf16, bf16, float, bf16>(
    cudaStream_t stream,
    bf16* output,
    float* state,
    bf16 const* q,
    bf16 const* k,
    bf16 const* v,
    float const* input_state,
    float const* alpha,
    bf16 const* beta,
    int32_t const* cu_seqlens,
    uint8_t* workspace_buffer,
    int32_t num_seqs,
    int32_t num_heads,
    int32_t head_size,
    int64_t total_seqlen,
    float scale,
    bool safe_gate,
    int32_t num_segments,
    int32_t sm_count);

}  // namespace kda::sm90

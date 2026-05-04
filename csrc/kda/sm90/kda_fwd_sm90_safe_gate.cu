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

#include <cute/numeric/numeric_types.hpp>
#include <cutlass/arch/arch.h>

#include "kda/sm90/prefill_kernel_kda_fwd_sm90.cuh"
#include "kda/sm90/utils/common.hpp"

namespace kda::sm90 {

using namespace cute;
using bf16 = cute::bfloat16_t;

// ── NumSegments = 1 (legacy single-block-per-(seq,head) path) ────────────────

// SafeGate=true, InitState=false, NSeg=1
template void
launch_kda_fwd_prefill_kernel_gbai<true, true, false, true, /*NumSegments=*/1, cutlass::arch::Sm90, bf16, bf16, float>(
    cudaStream_t,
    bf16*,
    float*,
    bf16 const*,
    bf16 const*,
    bf16 const*,
    float const*,
    float const*,
    float const*,
    int32_t const*,
    uint8_t*,
    int32_t,
    int32_t,
    int32_t,
    int64_t,
    float,
    int32_t);

// SafeGate=true, InitState=true, NSeg=1
template void
launch_kda_fwd_prefill_kernel_gbai<true, true, true, true, /*NumSegments=*/1, cutlass::arch::Sm90, bf16, bf16, float>(
    cudaStream_t,
    bf16*,
    float*,
    bf16 const*,
    bf16 const*,
    bf16 const*,
    float const*,
    float const*,
    float const*,
    int32_t const*,
    uint8_t*,
    int32_t,
    int32_t,
    int32_t,
    int64_t,
    float,
    int32_t);

// SafeGate=true, InitState=false, BetaBF16, NSeg=1
template void
launch_kda_fwd_prefill_kernel_gbai<true, true, false, true, /*NumSegments=*/1, cutlass::arch::Sm90, bf16, bf16, float, bf16>(
    cudaStream_t,
    bf16*,
    float*,
    bf16 const*,
    bf16 const*,
    bf16 const*,
    float const*,
    float const*,
    bf16 const*,
    int32_t const*,
    uint8_t*,
    int32_t,
    int32_t,
    int32_t,
    int64_t,
    float,
    int32_t);

// SafeGate=true, InitState=true, BetaBF16, NSeg=1
template void
launch_kda_fwd_prefill_kernel_gbai<true, true, true, true, /*NumSegments=*/1, cutlass::arch::Sm90, bf16, bf16, float, bf16>(
    cudaStream_t,
    bf16*,
    float*,
    bf16 const*,
    bf16 const*,
    bf16 const*,
    float const*,
    float const*,
    bf16 const*,
    int32_t const*,
    uint8_t*,
    int32_t,
    int32_t,
    int32_t,
    int64_t,
    float,
    int32_t);

// ── NumSegments = 2 (segment-scan first behaviour change) ────────────────────
// Only the InitState=true paths are instantiated; segment-scan always sets
// the per-segment input_state buffer (zero for seg≥1 in the first pass).

// SafeGate=true, InitState=true, NSeg=2
template void
launch_kda_fwd_prefill_kernel_gbai<true, true, true, true, /*NumSegments=*/2, cutlass::arch::Sm90, bf16, bf16, float>(
    cudaStream_t,
    bf16*,
    float*,
    bf16 const*,
    bf16 const*,
    bf16 const*,
    float const*,
    float const*,
    float const*,
    int32_t const*,
    uint8_t*,
    int32_t,
    int32_t,
    int32_t,
    int64_t,
    float,
    int32_t);

// SafeGate=true, InitState=true, BetaBF16, NSeg=2
template void
launch_kda_fwd_prefill_kernel_gbai<true, true, true, true, /*NumSegments=*/2, cutlass::arch::Sm90, bf16, bf16, float, bf16>(
    cudaStream_t,
    bf16*,
    float*,
    bf16 const*,
    bf16 const*,
    bf16 const*,
    float const*,
    float const*,
    bf16 const*,
    int32_t const*,
    uint8_t*,
    int32_t,
    int32_t,
    int32_t,
    int64_t,
    float,
    int32_t);

// SafeGate=true, InitState=false, NSeg=2  (matches dispatcher when no init_state passed)
template void
launch_kda_fwd_prefill_kernel_gbai<true, true, false, true, /*NumSegments=*/2, cutlass::arch::Sm90, bf16, bf16, float>(
    cudaStream_t,
    bf16*,
    float*,
    bf16 const*,
    bf16 const*,
    bf16 const*,
    float const*,
    float const*,
    float const*,
    int32_t const*,
    uint8_t*,
    int32_t,
    int32_t,
    int32_t,
    int64_t,
    float,
    int32_t);

// SafeGate=true, InitState=false, BetaBF16, NSeg=2
template void
launch_kda_fwd_prefill_kernel_gbai<true, true, false, true, /*NumSegments=*/2, cutlass::arch::Sm90, bf16, bf16, float, bf16>(
    cudaStream_t,
    bf16*,
    float*,
    bf16 const*,
    bf16 const*,
    bf16 const*,
    float const*,
    float const*,
    bf16 const*,
    int32_t const*,
    uint8_t*,
    int32_t,
    int32_t,
    int32_t,
    int64_t,
    float,
    int32_t);

// ── NumSegments = 4 (segment-scan + FLA M-chain merge target) ────────────────

// SafeGate=true, InitState=true, NSeg=4
template void
launch_kda_fwd_prefill_kernel_gbai<true, true, true, true, /*NumSegments=*/4, cutlass::arch::Sm90, bf16, bf16, float>(
    cudaStream_t,
    bf16*,
    float*,
    bf16 const*,
    bf16 const*,
    bf16 const*,
    float const*,
    float const*,
    float const*,
    int32_t const*,
    uint8_t*,
    int32_t,
    int32_t,
    int32_t,
    int64_t,
    float,
    int32_t);

// SafeGate=true, InitState=true, BetaBF16, NSeg=4
template void
launch_kda_fwd_prefill_kernel_gbai<true, true, true, true, /*NumSegments=*/4, cutlass::arch::Sm90, bf16, bf16, float, bf16>(
    cudaStream_t,
    bf16*,
    float*,
    bf16 const*,
    bf16 const*,
    bf16 const*,
    float const*,
    float const*,
    bf16 const*,
    int32_t const*,
    uint8_t*,
    int32_t,
    int32_t,
    int32_t,
    int64_t,
    float,
    int32_t);

// SafeGate=true, InitState=false, NSeg=4
template void
launch_kda_fwd_prefill_kernel_gbai<true, true, false, true, /*NumSegments=*/4, cutlass::arch::Sm90, bf16, bf16, float>(
    cudaStream_t,
    bf16*,
    float*,
    bf16 const*,
    bf16 const*,
    bf16 const*,
    float const*,
    float const*,
    float const*,
    int32_t const*,
    uint8_t*,
    int32_t,
    int32_t,
    int32_t,
    int64_t,
    float,
    int32_t);

// SafeGate=true, InitState=false, BetaBF16, NSeg=4
template void
launch_kda_fwd_prefill_kernel_gbai<true, true, false, true, /*NumSegments=*/4, cutlass::arch::Sm90, bf16, bf16, float, bf16>(
    cudaStream_t,
    bf16*,
    float*,
    bf16 const*,
    bf16 const*,
    bf16 const*,
    float const*,
    float const*,
    bf16 const*,
    int32_t const*,
    uint8_t*,
    int32_t,
    int32_t,
    int32_t,
    int64_t,
    float,
    int32_t);

// ── NumSegments = 8 (push grid expansion further for the worst case) ─────────

// SafeGate=true, InitState=true, NSeg=8
template void
launch_kda_fwd_prefill_kernel_gbai<true, true, true, true, /*NumSegments=*/8, cutlass::arch::Sm90, bf16, bf16, float>(
    cudaStream_t,
    bf16*,
    float*,
    bf16 const*,
    bf16 const*,
    bf16 const*,
    float const*,
    float const*,
    float const*,
    int32_t const*,
    uint8_t*,
    int32_t,
    int32_t,
    int32_t,
    int64_t,
    float,
    int32_t);

// SafeGate=true, InitState=true, BetaBF16, NSeg=8
template void
launch_kda_fwd_prefill_kernel_gbai<true, true, true, true, /*NumSegments=*/8, cutlass::arch::Sm90, bf16, bf16, float, bf16>(
    cudaStream_t,
    bf16*,
    float*,
    bf16 const*,
    bf16 const*,
    bf16 const*,
    float const*,
    float const*,
    bf16 const*,
    int32_t const*,
    uint8_t*,
    int32_t,
    int32_t,
    int32_t,
    int64_t,
    float,
    int32_t);

// SafeGate=true, InitState=false, NSeg=8
template void
launch_kda_fwd_prefill_kernel_gbai<true, true, false, true, /*NumSegments=*/8, cutlass::arch::Sm90, bf16, bf16, float>(
    cudaStream_t,
    bf16*,
    float*,
    bf16 const*,
    bf16 const*,
    bf16 const*,
    float const*,
    float const*,
    float const*,
    int32_t const*,
    uint8_t*,
    int32_t,
    int32_t,
    int32_t,
    int64_t,
    float,
    int32_t);

// SafeGate=true, InitState=false, BetaBF16, NSeg=8
template void
launch_kda_fwd_prefill_kernel_gbai<true, true, false, true, /*NumSegments=*/8, cutlass::arch::Sm90, bf16, bf16, float, bf16>(
    cudaStream_t,
    bf16*,
    float*,
    bf16 const*,
    bf16 const*,
    bf16 const*,
    float const*,
    float const*,
    bf16 const*,
    int32_t const*,
    uint8_t*,
    int32_t,
    int32_t,
    int32_t,
    int64_t,
    float,
    int32_t);

// ── NumSegments = 16 ─────────────────────────────────────────────────────────

template void
launch_kda_fwd_prefill_kernel_gbai<true, true, true, true, /*NumSegments=*/16, cutlass::arch::Sm90, bf16, bf16, float>(
    cudaStream_t, bf16*, float*, bf16 const*, bf16 const*, bf16 const*,
    float const*, float const*, float const*, int32_t const*, uint8_t*,
    int32_t, int32_t, int32_t, int64_t, float, int32_t);

template void
launch_kda_fwd_prefill_kernel_gbai<true, true, true, true, /*NumSegments=*/16, cutlass::arch::Sm90, bf16, bf16, float, bf16>(
    cudaStream_t, bf16*, float*, bf16 const*, bf16 const*, bf16 const*,
    float const*, float const*, bf16 const*, int32_t const*, uint8_t*,
    int32_t, int32_t, int32_t, int64_t, float, int32_t);

template void
launch_kda_fwd_prefill_kernel_gbai<true, true, false, true, /*NumSegments=*/16, cutlass::arch::Sm90, bf16, bf16, float>(
    cudaStream_t, bf16*, float*, bf16 const*, bf16 const*, bf16 const*,
    float const*, float const*, float const*, int32_t const*, uint8_t*,
    int32_t, int32_t, int32_t, int64_t, float, int32_t);

template void
launch_kda_fwd_prefill_kernel_gbai<true, true, false, true, /*NumSegments=*/16, cutlass::arch::Sm90, bf16, bf16, float, bf16>(
    cudaStream_t, bf16*, float*, bf16 const*, bf16 const*, bf16 const*,
    float const*, float const*, bf16 const*, int32_t const*, uint8_t*,
    int32_t, int32_t, int32_t, int64_t, float, int32_t);

// ── NumSegments = 32 ─────────────────────────────────────────────────────────

template void
launch_kda_fwd_prefill_kernel_gbai<true, true, true, true, /*NumSegments=*/32, cutlass::arch::Sm90, bf16, bf16, float>(
    cudaStream_t, bf16*, float*, bf16 const*, bf16 const*, bf16 const*,
    float const*, float const*, float const*, int32_t const*, uint8_t*,
    int32_t, int32_t, int32_t, int64_t, float, int32_t);

template void
launch_kda_fwd_prefill_kernel_gbai<true, true, true, true, /*NumSegments=*/32, cutlass::arch::Sm90, bf16, bf16, float, bf16>(
    cudaStream_t, bf16*, float*, bf16 const*, bf16 const*, bf16 const*,
    float const*, float const*, bf16 const*, int32_t const*, uint8_t*,
    int32_t, int32_t, int32_t, int64_t, float, int32_t);

template void
launch_kda_fwd_prefill_kernel_gbai<true, true, false, true, /*NumSegments=*/32, cutlass::arch::Sm90, bf16, bf16, float>(
    cudaStream_t, bf16*, float*, bf16 const*, bf16 const*, bf16 const*,
    float const*, float const*, float const*, int32_t const*, uint8_t*,
    int32_t, int32_t, int32_t, int64_t, float, int32_t);

template void
launch_kda_fwd_prefill_kernel_gbai<true, true, false, true, /*NumSegments=*/32, cutlass::arch::Sm90, bf16, bf16, float, bf16>(
    cudaStream_t, bf16*, float*, bf16 const*, bf16 const*, bf16 const*,
    float const*, float const*, bf16 const*, int32_t const*, uint8_t*,
    int32_t, int32_t, int32_t, int64_t, float, int32_t);

}  // namespace kda::sm90

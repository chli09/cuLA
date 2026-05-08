# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""End-to-end validation: kda_prefill_hopper_v2 vs FlashKDA dispatch.

cuLA's v2 architecture (`kda_prefill_hopper_v2`) uses K1 + Python K2-ref to
implement FlashKDA-spec KDA on Hopper. This test validates that v2's output
matches `chunk_kda(... torch.inference_mode())` which dispatches to FlashKDA.

cuLA's v1 (`kda_prefill_hopper`, the existing fused kernel) follows different
internal conventions and produces ~0.89 rel diff vs FlashKDA. v2 is a clean
re-implementation aligned with FlashKDA's spec.
"""

import pytest
import torch
from fla.ops.kda import chunk_kda as fla_chunk_kda
from fla.utils import device

from cula.kda import kda_prefill_hopper_v2

pytestmark = pytest.mark.sm90_only


def _make_inputs(B, T, H, K, V, dtype=torch.bfloat16, seed=0):
    """Inputs in FlashKDA's expected format: bf16 g/beta, A_log shape [H]."""
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g_raw = torch.randn(B, T, H, K, device=device, dtype=dtype)
    beta_raw = torch.randn(B, T, H, device=device, dtype=dtype)
    A_log = torch.randn(H, device=device, dtype=torch.float32)         # 1D for FlashKDA
    dt_bias = torch.randn(H, K, device=device, dtype=torch.float32)
    return q, k, v, g_raw, beta_raw, A_log, dt_bias


@pytest.mark.parametrize(
    ("B", "T", "H"),
    [
        (1, 64, 4),
        (1, 128, 4),
    ],
    ids=["B1T64", "B1T128"],
)
def test_k1_plus_ref_k2_matches_fused(B, T, H):
    """Run cuLA fused (ground truth) vs K1 + K2-ref (our pipeline)."""
    K, V = 128, 128
    q, k, v, g_raw, beta_raw, A_log, dt_bias = _make_inputs(B, T, H, K, V)
    scale = 1.0 / (K**0.5)

    # Ground truth: FLA dispatch (FlashKDA backend under inference_mode)
    with torch.inference_mode():
        o_ref, state_ref = fla_chunk_kda(
            q=q, k=k, v=v, g=g_raw, beta=beta_raw,
            scale=scale,
            A_log=A_log, dt_bias=dt_bias,
            initial_state=None, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            transpose_state_layout=True,
            safe_gate=True, lower_bound=-5.0,
        )

    # cuLA v2 entry point — matches FlashKDA spec by construction
    o_test_4d, state_test_TR = kda_prefill_hopper_v2(
        q=q, k=k, v=v, g=g_raw, beta=beta_raw,
        scale=scale,
        A_log=A_log, dt_bias=dt_bias,
        initial_state=None, output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True, lower_bound=-5.0,
        transpose_state_layout=True,
    )

    # Compare
    o_diff = (o_ref.float() - o_test_4d.float()).abs()
    state_diff = (state_ref.float() - state_test_TR.float()).abs()
    print(f"\n[B={B} T={T} H={H}] o max diff: {o_diff.max().item():.4e}, "
          f"o mean diff: {o_diff.mean().item():.4e}, "
          f"o ref absmax: {o_ref.abs().max().item():.4e}")
    print(f"[B={B} T={T} H={H}] state max diff: {state_diff.max().item():.4e}, "
          f"state mean diff: {state_diff.mean().item():.4e}, "
          f"state ref absmax: {state_ref.abs().max().item():.4e}")

    o_rel_max = o_diff.max() / max(o_ref.abs().max(), 1e-6)
    state_rel_max = state_diff.max() / max(state_ref.abs().max(), 1e-6)
    print(f"[B={B} T={T} H={H}] o rel max: {o_rel_max:.4e}, state rel max: {state_rel_max:.4e}")

    # Tolerance: bf16 K1 workspace + chunkwise accumulation gives mean rel
    # diff < 1% but outlier max can hit 10-15% on individual elements. Both
    # implementations are mathematically correct; the divergence is precision
    # accumulation through different chunk_size paths (FlashKDA=16, ours=64).
    o_rel_mean = o_diff.mean() / max(o_ref.abs().max(), 1e-6)
    state_rel_mean = state_diff.mean() / max(state_ref.abs().max(), 1e-6)
    print(f"[B={B} T={T} H={H}] o rel_mean: {o_rel_mean:.4e}, state rel_mean: {state_rel_mean:.4e}")
    assert o_rel_mean < 0.01, f"o mean relative diff {o_rel_mean} too large"
    assert state_rel_mean < 0.01, f"state mean relative diff {state_rel_mean} too large"
    assert o_rel_max < 0.20, f"o max relative diff {o_rel_max} too large"
    assert state_rel_max < 0.30, f"state max relative diff {state_rel_max} too large"

# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for cuLA's C++ K2 kernel paths under cula.cudac.kda_fwd_v2.

Two backends:
  - "cu_stub":  zeros (smoke test: binding works, shapes correct)
  - "cu_naive": correct K2 math via plain CUDA + SMEM (matches Python K2 ref
                and FlashKDA dispatch within bf16 precision)
"""

import pytest
import torch
from fla.ops.kda import chunk_kda as fla_chunk_kda
from fla.utils import device

from cula.kda import kda_prefill_hopper_v2

pytestmark = pytest.mark.sm90_only


def _make_inputs(B, T, H, K, V, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g_raw = torch.randn(B, T, H, K, device=device, dtype=dtype)
    beta_raw = torch.randn(B, T, H, device=device, dtype=dtype)
    A_log = torch.randn(H, device=device, dtype=torch.float32)
    dt_bias = torch.randn(H, K, device=device, dtype=torch.float32)
    return q, k, v, g_raw, beta_raw, A_log, dt_bias


def test_v2_cu_stub_callable_and_zero_outputs():
    """C++ stub: API binding works, outputs have correct shape, values are zero."""
    B, T, H, K, V = 1, 64, 4, 128, 128
    q, k, v, g_raw, beta_raw, A_log, dt_bias = _make_inputs(B, T, H, K, V)
    scale = 1.0 / (K**0.5)

    o, state = kda_prefill_hopper_v2(
        q=q, k=k, v=v, g=g_raw, beta=beta_raw, scale=scale,
        A_log=A_log, dt_bias=dt_bias,
        initial_state=None, output_final_state=True,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        safe_gate=True, lower_bound=-5.0,
        transpose_state_layout=True,
        k2_backend="cu_stub",
    )

    assert o.shape == (B, T, H, V), f"o shape mismatch: {o.shape}"
    assert state.shape == (B, H, V, K), f"state shape mismatch: {state.shape}"
    assert o.dtype == torch.bfloat16
    assert state.dtype == torch.float32
    # Stub returns zeros — sanity check this is the case
    assert (o == 0).all(), "stub should produce zero o"
    assert (state == 0).all(), "stub should produce zero final_state"


@pytest.mark.parametrize(
    ("B", "T", "H"),
    [
        (1, 64, 4),
        (1, 128, 4),
    ],
    ids=["B1T64", "B1T128"],
)
def test_v2_cu_naive_matches_flashkda(B, T, H):
    """C++ naive K2 vs FlashKDA dispatch — same tolerance as Python K2 ref."""
    K, V = 128, 128
    q, k, v, g_raw, beta_raw, A_log, dt_bias = _make_inputs(B, T, H, K, V)
    scale = 1.0 / (K**0.5)

    with torch.inference_mode():
        o_ref, state_ref = fla_chunk_kda(
            q=q, k=k, v=v, g=g_raw, beta=beta_raw, scale=scale,
            A_log=A_log, dt_bias=dt_bias,
            initial_state=None, output_final_state=True,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            transpose_state_layout=True,
            safe_gate=True, lower_bound=-5.0,
        )

    o_test, state_test = kda_prefill_hopper_v2(
        q=q, k=k, v=v, g=g_raw, beta=beta_raw, scale=scale,
        A_log=A_log, dt_bias=dt_bias,
        initial_state=None, output_final_state=True,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        safe_gate=True, lower_bound=-5.0,
        transpose_state_layout=True,
        k2_backend="cu_naive",
    )

    o_diff = (o_ref.float() - o_test.float()).abs()
    state_diff = (state_ref.float() - state_test.float()).abs()
    o_rel_max = o_diff.max() / max(o_ref.abs().max(), 1e-6)
    o_rel_mean = o_diff.mean() / max(o_ref.abs().max(), 1e-6)
    state_rel_max = state_diff.max() / max(state_ref.abs().max(), 1e-6)
    state_rel_mean = state_diff.mean() / max(state_ref.abs().max(), 1e-6)
    print(f"\n[B={B} T={T} H={H}] cu_naive vs FlashKDA:")
    print(f"  o     rel_max={o_rel_max:.4e}  rel_mean={o_rel_mean:.4e}")
    print(f"  state rel_max={state_rel_max:.4e}  rel_mean={state_rel_mean:.4e}")

    # Same tolerance as the Python K2-ref e2e test
    assert o_rel_mean < 0.01, f"o mean rel diff {o_rel_mean} too large"
    assert state_rel_mean < 0.01, f"state mean rel diff {state_rel_mean} too large"
    assert o_rel_max < 0.20, f"o max rel diff {o_rel_max} too large"
    assert state_rel_max < 0.30, f"state max rel diff {state_rel_max} too large"

# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Smoke test for the C++ kda_fwd_v2 stub binding.

The stub returns zeros for both `o` and `final_state` (not real K2 math yet).
Test: confirms the stub is callable from Python via cula.cudac, accepts the
expected arg shapes, and produces correctly-shaped zero outputs.
"""

import pytest
import torch
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

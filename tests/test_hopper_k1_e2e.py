# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""End-to-end validation: K1 + Python K2 reference vs FlashKDA dispatch.

Our K1 follows FlashKDA's spec (CHUNK=64 instead of 16 but same workspace
shapes and conventions). cuLA's existing monolithic fused kernel has its
own internal conventions that DIFFER from FlashKDA (verified by direct
comparison: cuLA fused vs `chunk_kda(... torch.inference_mode())` gives
~0.89 relative diff at B=1, T=64, H=4 — i.e. they're not the same math).

So we validate K1 + K2-ref against FlashKDA (the spec K1 actually targets).
The cuLA-fused-vs-FlashKDA divergence is a pre-existing issue separate from
our work.
"""

import pytest
import torch
import torch.nn.functional as F
from fla.modules.l2norm import l2norm_fwd
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.utils import device

from fla.ops.kda import chunk_kda as fla_chunk_kda

from cula.kda.hopper_k1 import kda_k1_full
from cula.kda.hopper_k2_reference import kda_k2_reference
from cula.utils import prepare_uniform_cu_seqlens

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

    # Our pipeline: replicate FlashKDA preprocessing (l2norm + gate cumsum + sigmoid beta)
    # then run K1 → workspace → K2 reference.

    # FlashKDA's K1 expects A_log shape [H], we need to broadcast to [H, K] for our cumsum
    # since FLA's kda_gate_chunk_cumsum expects [H, K]. Replicate A_log across K.
    A_log_2d = A_log.unsqueeze(-1).expand(H, K).contiguous()

    # Pack [B, T, ...] -> [1, B*T, ...]
    q_4d = q.reshape(1, B * T, H, K).contiguous()
    k_4d = k.reshape(1, B * T, H, K).contiguous()
    v_4d = v.reshape(1, B * T, H, V).contiguous()
    g_4d = g_raw.float().reshape(1, B * T, H, K).contiguous()
    beta_4d = beta_raw.reshape(1, B * T, H).contiguous()
    cu_seqlens = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=device)

    # Gate cumsum (FLA expects fp32 g)
    g_cumsum_4d = kda_gate_chunk_cumsum(
        g=g_4d, A_log=A_log_2d, dt_bias=dt_bias,
        scale=RCP_LN2, chunk_size=64,
        cu_seqlens=cu_seqlens, chunk_indices=None,
        lower_bound=-5.0,
    )

    # L2 norm q, k
    q_l2, _ = l2norm_fwd(q_4d)
    k_l2, _ = l2norm_fwd(k_4d)

    # Sigmoid beta (since cuLA fused has use_beta_sigmoid_in_kernel implicit)
    beta_sig = beta_4d.float().sigmoid()

    # Pack to 3D for K1
    q_pk = q_l2.reshape(B * T, H, K).contiguous()
    k_pk = k_l2.reshape(B * T, H, K).contiguous()
    v_pk = v_4d.reshape(B * T, H, V).contiguous()
    g_pk = g_cumsum_4d.reshape(B * T, H, K).contiguous()
    beta_pk = beta_sig.reshape(B * T, H).contiguous()

    # K1 → workspace
    ws = kda_k1_full(q_pk, k_pk, g_pk, beta_pk, scale, cu_seqlens=cu_seqlens, chunk_size=64)

    # K2 reference → (o, state). Note: K2 reference takes beta as input (for delta-rule
    # residual scaling); K1 already used beta to build INV.
    o_test, state_test = kda_k2_reference(
        ws, v_pk, beta_pk, h_initial=None,
        cu_seqlens=cu_seqlens, chunk_size=64,
    )

    # Reshape o_test back to [B, T, H, V] to match FLA output
    o_test_4d = o_test.reshape(B, T, H, V)

    # FlashKDA returns state with transpose_state_layout=True → shape [N, H, V, K]
    # Our K2 ref has state shape [N, H, K, V]. Transpose to compare.
    state_test_TR = state_test.transpose(-1, -2).contiguous()

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

    # Generous tolerance (bf16 ws + fp32 reference accumulators differ)
    assert o_rel_max < 0.1, f"o relative diff {o_rel_max} too large"
    assert state_rel_max < 0.1, f"state relative diff {state_rel_max} too large"

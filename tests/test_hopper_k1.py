# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for cula.kda.hopper_k1 — Hopper K1 preprocessing kernel."""

import pytest
import torch
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.utils import assert_close, device

from cula.kda.hopper_k1 import kda_k1_decay_apply, kda_k1_decay_apply_reference

pytestmark = pytest.mark.sm90_only


def _make_inputs(B, T, H, K, dtype=torch.bfloat16, seed=0):
    """Build (q, k, g) in cuLA Hopper's expected post-preprocess form.

    - q/k l2-normed (cheap reproduction here, not the FLA Triton variant)
    - g produced via kda_gate_chunk_cumsum with safe_gate range
    """
    torch.manual_seed(seed)

    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    g_raw = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
    A_log = torch.randn(H, K, device=device, dtype=torch.float32)
    dt_bias = torch.randn(H, K, device=device, dtype=torch.float32)
    beta = torch.randn(B, T, H, device=device, dtype=torch.float32).sigmoid()

    # L2 norm q/k along the last dim (matches l2norm_fwd's effect)
    q = torch.nn.functional.normalize(q.float(), dim=-1).to(dtype)
    k = torch.nn.functional.normalize(k.float(), dim=-1).to(dtype)

    # Pack to [packed_seq=B*T, H, K]
    q_packed = q.reshape(B * T, H, K).contiguous()
    k_packed = k.reshape(B * T, H, K).contiguous()
    g_packed = g_raw.reshape(B * T, H, K).contiguous()
    beta_packed = beta.reshape(B * T, H).contiguous()

    cu_seqlens = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=device)

    # Chunk-cumsum of g (exactly what hopper_fused_fwd does pre-K1)
    g_cumsum = kda_gate_chunk_cumsum(
        g=g_packed,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=RCP_LN2,
        chunk_size=64,
        cu_seqlens=cu_seqlens,
        chunk_indices=None,
        lower_bound=-5.0,
    )
    return q_packed, k_packed, g_cumsum, beta_packed, cu_seqlens


@pytest.mark.parametrize(
    ("B", "T", "H", "K"),
    [
        (1, 64, 4, 128),     # 1 chunk, smallest
        (1, 128, 4, 128),    # 2 chunks
        (1, 1024, 8, 128),   # multi-chunk medium
        (2, 512, 4, 128),    # multi-seq (uniform-len fast path)
    ],
    ids=["B1T64", "B1T128", "B1T1024H8", "B2T512"],
)
def test_decay_apply_matches_reference(B, T, H, K):
    """Triton kernel vs pure-PyTorch reference, bit-exact-ish for decay apply."""
    scale = 1.0 / (K**0.5)
    q, k, g, beta, cu_seqlens = _make_inputs(B, T, H, K)

    ws_qd_t, ws_kd_t, ws_kr_t, ws_gt_t = kda_k1_decay_apply(
        q, k, g, scale, cu_seqlens=cu_seqlens, chunk_size=64
    )
    ws_qd_r, ws_kd_r, ws_kr_r, ws_gt_r = kda_k1_decay_apply_reference(
        q, k, g, scale, cu_seqlens=cu_seqlens, chunk_size=64
    )

    # bf16 outputs: small numerical tolerance
    assert_close("ws_qd", ws_qd_t, ws_qd_r, ratio=2e-3)
    assert_close("ws_kd", ws_kd_t, ws_kd_r, ratio=2e-3)
    assert_close("ws_kr", ws_kr_t, ws_kr_r, ratio=2e-3)
    # ws_gt is fp32: tighter tolerance
    assert_close("ws_gt", ws_gt_t, ws_gt_r, ratio=1e-5)


def test_decay_apply_varlen():
    """Variable-length: 3 sequences with different lengths in one packed tensor."""
    H, K = 4, 128
    seq_lens = [200, 130, 64]  # not all chunk-aligned
    T_total = sum(seq_lens)

    cu_seqlens = torch.tensor(
        [0] + list(torch.tensor(seq_lens).cumsum(0).tolist()),
        dtype=torch.int32,
        device=device,
    )

    # Build a single packed input
    q, k, g, beta, _ = _make_inputs(1, T_total, H, K)

    scale = 1.0 / (K**0.5)
    ws_qd_t, ws_kd_t, ws_kr_t, ws_gt_t = kda_k1_decay_apply(
        q, k, g, scale, cu_seqlens=cu_seqlens, chunk_size=64
    )
    ws_qd_r, ws_kd_r, ws_kr_r, ws_gt_r = kda_k1_decay_apply_reference(
        q, k, g, scale, cu_seqlens=cu_seqlens, chunk_size=64
    )

    assert_close("ws_qd_varlen", ws_qd_t, ws_qd_r, ratio=2e-3)
    assert_close("ws_kd_varlen", ws_kd_t, ws_kd_r, ratio=2e-3)
    assert_close("ws_kr_varlen", ws_kr_t, ws_kr_r, ratio=2e-3)
    assert_close("ws_gt_varlen", ws_gt_t, ws_gt_r, ratio=1e-5)


def test_decay_apply_zero_fill_tail():
    """Last chunk of an uneven sequence must zero-fill rows beyond actual_len."""
    B, T, H, K = 1, 70, 4, 128  # NT=2, last chunk has only 6 valid tokens
    scale = 1.0 / (K**0.5)

    q, k, g, _, cu_seqlens = _make_inputs(B, T, H, K)
    ws_qd_t, ws_kd_t, ws_kr_t, _ = kda_k1_decay_apply(
        q, k, g, scale, cu_seqlens=cu_seqlens, chunk_size=64
    )

    # Last chunk (index 1) should have rows 6..63 all-zero
    tail_qd = ws_qd_t[1, :, 6:, :]
    tail_kd = ws_kd_t[1, :, 6:, :]
    tail_kr = ws_kr_t[1, :, 6:, :]
    assert tail_qd.abs().max().item() == 0.0, "ws_qd tail not zero-filled"
    assert tail_kd.abs().max().item() == 0.0, "ws_kd tail not zero-filled"
    assert tail_kr.abs().max().item() == 0.0, "ws_kr tail not zero-filled"

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

from cula.kda.hopper_k1 import (
    kda_k1_decay_apply,
    kda_k1_decay_apply_reference,
    kda_k1_inv,
    kda_k1_inv_reference,
    kda_k1_mqk,
    kda_k1_mqk_reference,
)

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

    # Pack [B, T, ...] -> [1, B*T, ...] (matches hopper_fused_fwd's flatten step)
    q_4d = q.reshape(1, B * T, H, K).contiguous()
    k_4d = k.reshape(1, B * T, H, K).contiguous()
    g_4d = g_raw.reshape(1, B * T, H, K).contiguous()
    beta_4d = beta.reshape(1, B * T, H).contiguous()

    cu_seqlens = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=device)

    # Chunk-cumsum of g (exactly what hopper_fused_fwd does pre-K1)
    g_cumsum_4d = kda_gate_chunk_cumsum(
        g=g_4d,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=RCP_LN2,
        chunk_size=64,
        cu_seqlens=cu_seqlens,
        chunk_indices=None,
        lower_bound=-5.0,
    )

    # Now flatten to 3D [packed_seq=B*T, H, K] for K1 kernel
    q_packed = q_4d.reshape(B * T, H, K).contiguous()
    k_packed = k_4d.reshape(B * T, H, K).contiguous()
    g_packed = g_cumsum_4d.reshape(B * T, H, K).contiguous()
    beta_packed = beta_4d.reshape(B * T, H).contiguous()
    return q_packed, k_packed, g_packed, beta_packed, cu_seqlens


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
    """Variable-length: 3 sequences in one packed tensor.

    Seq lens are chunk-size (64) aligned. FLA's kda_gate_chunk_cumsum runs
    fixed-size 64-token cumsum chunks irrespective of cu_seqlens, so the
    cu_seqlens entries it consumes have to be 64-aligned in real usage.
    cuLA's hopper_fused_fwd Python orchestrator only packs uniform-len
    sequences, so this is the regime that matters for K1's contract.
    """
    H, K = 4, 128
    seq_lens = [192, 128, 64]  # all multiples of chunk_size=64
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


@pytest.mark.parametrize(
    ("B", "T", "H", "K"),
    [
        (1, 64, 4, 128),
        (1, 128, 4, 128),
        (1, 1024, 4, 128),  # H=4 to keep test fast (reference is O(BT^2 * K))
    ],
    ids=["B1T64", "B1T128", "B1T1024"],
)
def test_mqk_matches_reference(B, T, H, K):
    """Phase 1.2: sub-chunked Mqk with anchor-decay trick, no exp2(-g) overflow."""
    scale = 1.0 / (K**0.5)
    q, k, g, _, cu_seqlens = _make_inputs(B, T, H, K)

    ws_mqk_t = kda_k1_mqk(q, k, g, scale, cu_seqlens=cu_seqlens, chunk_size=64)
    ws_mqk_r = kda_k1_mqk_reference(q, k, g, scale, cu_seqlens=cu_seqlens, chunk_size=64)

    # bf16 outputs from MMA: relax tolerance
    assert_close("ws_mqk", ws_mqk_t, ws_mqk_r, ratio=5e-3)

    # Sanity: above-diagonal must be zero
    o_arr = torch.arange(64, device=device)
    upper = (o_arr[:, None] < o_arr[None, :]).expand_as(ws_mqk_t)
    assert (ws_mqk_t[upper] == 0).all(), "ws_mqk non-zero above diagonal"


@pytest.mark.parametrize(
    ("B", "T", "H", "K"),
    [
        (1, 64, 4, 128),
        (1, 128, 4, 128),
    ],
    ids=["B1T64", "B1T128"],
)
def test_inv_matches_reference(B, T, H, K):
    """Phase 1.3: ws_inv = (I - tril_strict(beta·k·k^T·decay))^-1 via block LU."""
    q, k, g, beta, cu_seqlens = _make_inputs(B, T, H, K)

    ws_inv_t = kda_k1_inv(k, g, beta, cu_seqlens=cu_seqlens, chunk_size=64)
    ws_inv_r = kda_k1_inv_reference(k, g, beta, cu_seqlens=cu_seqlens, chunk_size=64)

    # Generous tolerance: bf16 MMA chains compound rounding error
    assert_close("ws_inv", ws_inv_t, ws_inv_r, ratio=1e-2)


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

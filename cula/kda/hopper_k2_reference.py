# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Python eager K2 reference for FlashKDA-style architectural restructure.

This is the SPEC the cuLA fused C++ kernel needs to satisfy after the K1+K2
split. K2 consumes the workspace produced by K1 (kda_k1_full) and produces
the same (o, final_state) that the existing monolithic fused kernel produces.

Per-chunk math (verified against FlashKDA's csrc/smxx/fwd_kernel2.cuh):
    Phase 1: v_pred = ws_kd · h            # k_decayed @ h_pre, predicts v
             out    = ws_qd · h            # Q·h_pre
    Phase 3: v_err  = (v - v_pred) · beta  # delta-rule residual, beta-scaled per row
             v_new  = ws_inv · v_err       # solve triangular system for chunk-wise v_new
    Phase 4: o      = out + ws_mqk · v_new # q·h + intra-attn · v_new (NOT raw v!)
    Phase 5: h      = diag(ws_gt) · h      # full-chunk gate decay
                      + ws_kr^T · v_new    # state update with future-corrected k

Note beta convention: input beta here is POST-sigmoid (in [0, 1]). FlashKDA's
K2 sigmoid's inside the kernel; our K1 also assumes pre-sigmoid'd beta.

Slow (Python eager, per chunk × head loop) but correct — used to validate K1.
"""

import torch
from fla.ops.utils import prepare_chunk_indices

from cula.utils import prepare_uniform_cu_seqlens


def kda_k2_reference(
    ws: dict,
    v: torch.Tensor,
    beta: torch.Tensor,
    h_initial: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
):
    """Eager K2: take K1's workspace + v + h0, produce (o, h_final).

    Args:
        ws: dict from kda_k1_full with keys ws_qd, ws_kd, ws_kr, ws_gt, ws_mqk, ws_inv.
        v: [packed_seq, H, V] bf16
        h_initial: [N, H, K, V] fp32 or None (zero start).
        cu_seqlens: [N+1] int32. If None, B=1 single-seq.
        chunk_indices: [total_NT, 2] int32.
        chunk_size: BT, must be 64.

    Returns:
        o: [packed_seq, H, V] bf16
        h_final: [N, H, K, V] fp32
    """
    BT = chunk_size
    packed_seq, H, V = v.shape
    K = ws["ws_qd"].shape[-1]
    device = v.device

    if cu_seqlens is None:
        cu_seqlens = prepare_uniform_cu_seqlens(1, packed_seq, device, torch.int32)
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

    cs = cu_seqlens.cpu().tolist()
    N = len(cs) - 1

    if h_initial is None:
        h_state = torch.zeros((N, H, K, V), dtype=torch.float32, device=device)
    else:
        h_state = h_initial.clone().to(torch.float32)

    o = torch.zeros_like(v)

    ci = chunk_indices.cpu().tolist()
    for chunk_idx in range(len(ci)):
        i_n, i_t = ci[chunk_idx]
        bos, eos = cs[i_n], cs[i_n + 1]
        T_seq = eos - bos
        t_start = i_t * BT
        t_end = min(t_start + BT, T_seq)
        actual_len = t_end - t_start

        for h_idx in range(H):
            state = h_state[i_n, h_idx]  # [K, V] fp32

            wqd = ws["ws_qd"][chunk_idx, h_idx, :actual_len].to(torch.float32)
            wkd = ws["ws_kd"][chunk_idx, h_idx, :actual_len].to(torch.float32)
            wkr = ws["ws_kr"][chunk_idx, h_idx, :actual_len].to(torch.float32)
            wgt = ws["ws_gt"][chunk_idx, h_idx]                                # [K]
            wmqk = ws["ws_mqk"][chunk_idx, h_idx, :actual_len, :actual_len].to(torch.float32)
            winv = ws["ws_inv"][chunk_idx, h_idx, :actual_len, :actual_len].to(torch.float32)

            v_chunk = v[bos + t_start : bos + t_end, h_idx].to(torch.float32)  # [actual, V]
            beta_chunk = beta[bos + t_start : bos + t_end, h_idx].to(torch.float32)  # [actual]

            # Phase 1: predict v from current state, compute Q·h
            v_pred = wkd @ state             # [actual, V]
            o_inter = wqd @ state            # [actual, V]

            # Phase 3: delta-rule residual + INV solve
            v_err = (v_chunk - v_pred) * beta_chunk[:, None]  # [actual, V]
            v_new = winv @ v_err                              # [actual, V]

            # Phase 4: o = q·h + Mqk · v_new   (Mqk applied to v_new, not raw v)
            o_intra = wmqk @ v_new           # [actual, V]
            o_chunk = (o_intra + o_inter).to(v.dtype)
            o[bos + t_start : bos + t_end, h_idx] = o_chunk

            # Phase 5: state update
            new_state = wgt[:, None] * state + wkr.T @ v_new
            h_state[i_n, h_idx] = new_state

    return o, h_state

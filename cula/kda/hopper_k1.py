# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Hopper K1 kernel — preprocessing extracted from the SM90 fused kernel.

Issue #11 architectural restructure (FlashKDA-style K1 + K2 split). Phase 1.1
extracts only the per-chunk decay apply, producing ws_qd / ws_kd / ws_kr / ws_gt.

Subsequent phases will fold in the intra-MMA (Mqk) and the delta-rule inverse
(ws_inv) so K2 can drop those steps from its T-loop.

The kernel runs chunk-parallel: grid = (total_NT, H), one block per
(chunk, head). cuLA's existing fused kernel currently does this work serially
inside its T-loop; extracting it exposes NT new parallelism that helps the
small-B/H regime.
"""

import torch
import triton
import triton.language as tl
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2

from cula.utils import prepare_uniform_cu_seqlens


@triton.jit
def kda_k1_decay_apply_kernel(
    q_ptr,         # [packed_seq, H, K] bf16, post-l2norm
    k_ptr,         # [packed_seq, H, K] bf16, post-l2norm
    g_ptr,         # [packed_seq, H, K] fp32, post-chunk-local cumsum (RCP_LN2 scaled)
    ws_qd_ptr,     # [total_NT, H, BT, K] bf16  ← q · exp2(g) · scale
    ws_kd_ptr,     # [total_NT, H, BT, K] bf16  ← k · exp2(g)
    ws_kr_ptr,     # [total_NT, H, BT, K] bf16  ← k · exp2(g_total - g)
    ws_gt_ptr,     # [total_NT, H, K]     fp32  ← exp2(g_total)  (matches FlashKDA)
    cu_seqlens,    # [N+1] int32
    chunk_indices, # [total_NT, 2] int32 — (i_n, i_t)
    scale,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    """One block per (chunk, head). Each block writes a [BT, K] slab to ws_*."""
    pid_t = tl.program_id(0).to(tl.int32)
    pid_h = tl.program_id(1).to(tl.int32)

    # Resolve global chunk index → (i_n, i_t)
    i_n = tl.load(chunk_indices + pid_t * 2).to(tl.int32)
    i_t_local = tl.load(chunk_indices + pid_t * 2 + 1).to(tl.int32)
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T_seq = eos - bos

    chunk_start_local = i_t_local * BT  # token offset within sequence
    if chunk_start_local >= T_seq:
        return  # excess CTA (varlen tail tile that doesn't exist)

    actual_len = tl.minimum(BT, T_seq - chunk_start_local)
    last_token_global = bos + chunk_start_local + actual_len - 1

    o_t = tl.arange(0, BT)
    m_t = o_t < actual_len  # row mask

    # Base pointers (head h of sequence i_n at chunk start)
    q_base = q_ptr + (bos * H + pid_h) * K
    k_base = k_ptr + (bos * H + pid_h) * K
    g_base = g_ptr + (bos * H + pid_h) * K

    # Workspace base (chunk pid_t, head pid_h)
    ws_qd_base = ws_qd_ptr + (pid_t * H + pid_h) * BT * K
    ws_kd_base = ws_kd_ptr + (pid_t * H + pid_h) * BT * K
    ws_kr_base = ws_kr_ptr + (pid_t * H + pid_h) * BT * K
    ws_gt_base = ws_gt_ptr + (pid_t * H + pid_h) * K

    # g at last valid token of chunk: shape [K]
    g_last_global_offset = (last_token_global * H + pid_h) * K

    for i_k in range(tl.cdiv(K, BK)):
        offsets_k = i_k * BK + tl.arange(0, BK)
        m_k = offsets_k < K

        # Load g_total (cumsum at chunk's last valid token), shape [BK]
        b_gt_log = tl.load(
            g_ptr + g_last_global_offset + offsets_k,
            mask=m_k,
            other=0.0,
        ).to(tl.float32)

        # Load q, k, g tile, shape [BT, BK] — contiguous along K, strided along T by H*K
        p_q = tl.make_block_ptr(
            q_base,
            shape=(T_seq, K),
            strides=(H * K, 1),
            offsets=(chunk_start_local, i_k * BK),
            block_shape=(BT, BK),
            order=(1, 0),
        )
        p_k = tl.make_block_ptr(
            k_base,
            shape=(T_seq, K),
            strides=(H * K, 1),
            offsets=(chunk_start_local, i_k * BK),
            block_shape=(BT, BK),
            order=(1, 0),
        )
        p_g = tl.make_block_ptr(
            g_base,
            shape=(T_seq, K),
            strides=(H * K, 1),
            offsets=(chunk_start_local, i_k * BK),
            block_shape=(BT, BK),
            order=(1, 0),
        )
        b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)

        # Decayed variants (exp is base-2: g already RCP_LN2-scaled by the cumsum upstream)
        b_eg = exp2(b_g)             # [BT, BK]
        b_egtmg = exp2(b_gt_log[None, :] - b_g)  # [BT, BK]

        b_qd = b_q * b_eg * scale
        b_kd = b_k * b_eg
        b_kr = b_k * b_egtmg

        # Zero-fill rows beyond actual_len so K2 reads zeros for padding
        b_qd = tl.where(m_t[:, None], b_qd, 0.0)
        b_kd = tl.where(m_t[:, None], b_kd, 0.0)
        b_kr = tl.where(m_t[:, None], b_kr, 0.0)

        # Store to workspace [BT, BK]
        p_ws_qd = tl.make_block_ptr(
            ws_qd_base,
            shape=(BT, K),
            strides=(K, 1),
            offsets=(0, i_k * BK),
            block_shape=(BT, BK),
            order=(1, 0),
        )
        p_ws_kd = tl.make_block_ptr(
            ws_kd_base,
            shape=(BT, K),
            strides=(K, 1),
            offsets=(0, i_k * BK),
            block_shape=(BT, BK),
            order=(1, 0),
        )
        p_ws_kr = tl.make_block_ptr(
            ws_kr_base,
            shape=(BT, K),
            strides=(K, 1),
            offsets=(0, i_k * BK),
            block_shape=(BT, BK),
            order=(1, 0),
        )
        tl.store(p_ws_qd, b_qd.to(ws_qd_ptr.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_ws_kd, b_kd.to(ws_kd_ptr.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_ws_kr, b_kr.to(ws_kr_ptr.dtype.element_ty), boundary_check=(0, 1))

        # Store ws_gt = exp2(g_total) — shape [BK]
        b_gt = exp2(b_gt_log)
        tl.store(ws_gt_base + offsets_k, b_gt, mask=m_k)


def kda_k1_decay_apply(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
):
    """Phase 1.1 — extract per-chunk decay apply from cuLA Hopper fused kernel.

    Args:
        q: [packed_seq, H, K] bf16 (post-l2norm).
        k: [packed_seq, H, K] bf16 (post-l2norm).
        g: [packed_seq, H, K] fp32, already chunk-local cumsum + RCP_LN2 scaled.
        scale: float, attention scale (1/sqrt(K) typically).
        cu_seqlens: [N+1] int32. If None, derived from q.shape[0] / batch.
        chunk_indices: [total_NT, 2] int32 (i_n, i_t). If None, computed.
        chunk_size: chunk length (BT, must be 64 for cuLA Hopper).

    Returns:
        ws_qd: [total_NT, H, BT, K] bf16
        ws_kd: [total_NT, H, BT, K] bf16
        ws_kr: [total_NT, H, BT, K] bf16
        ws_gt: [total_NT, H, K]     fp32  (= exp2(g_total))
    """
    assert chunk_size == 64, f"cuLA Hopper requires chunk_size=64, got {chunk_size}"
    assert q.dim() == 3, f"q must be [packed_seq, H, K], got shape {q.shape}"
    assert q.shape == k.shape, f"q/k shape mismatch: {q.shape} vs {k.shape}"
    assert g.shape == q.shape, f"g shape mismatch: {g.shape} vs {q.shape}"

    packed_seq, H, K = q.shape
    BT = chunk_size

    is_varlen = cu_seqlens is not None
    if not is_varlen:
        cu_seqlens = prepare_uniform_cu_seqlens(1, packed_seq, q.device, torch.int32)

    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    total_NT = chunk_indices.shape[0]

    ws_qd = torch.empty((total_NT, H, BT, K), dtype=q.dtype, device=q.device)
    ws_kd = torch.empty_like(ws_qd)
    ws_kr = torch.empty_like(ws_qd)
    ws_gt = torch.empty((total_NT, H, K), dtype=torch.float32, device=q.device)

    # BK chosen to match FLA's chunk_intra; K=128 gives 4 K-tiles of 32
    BK = 32 if K >= 32 else K

    grid = (total_NT, H, 1)
    kda_k1_decay_apply_kernel[grid](
        q,
        k,
        g,
        ws_qd,
        ws_kd,
        ws_kr,
        ws_gt,
        cu_seqlens,
        chunk_indices,
        scale,
        H=H,
        K=K,
        BT=BT,
        BK=BK,
    )

    return ws_qd, ws_kd, ws_kr, ws_gt


def kda_k1_decay_apply_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
):
    """Pure-PyTorch reference for kda_k1_decay_apply, for unit testing.

    Mirrors the kernel exactly: per-chunk decay apply, zero-fill tail rows.
    """
    BT = chunk_size
    packed_seq, H, K = q.shape

    if cu_seqlens is None:
        cu_seqlens = prepare_uniform_cu_seqlens(1, packed_seq, q.device, torch.int32)

    cs = cu_seqlens.cpu().tolist()
    N = len(cs) - 1

    chunks = []
    for i_n in range(N):
        bos, eos = cs[i_n], cs[i_n + 1]
        T_seq = eos - bos
        n_chunks = (T_seq + BT - 1) // BT
        for i_t in range(n_chunks):
            t_start = i_t * BT
            t_end = min(t_start + BT, T_seq)
            chunks.append((i_n, i_t, bos, t_start, t_end))

    total_NT = len(chunks)

    ws_qd = torch.zeros((total_NT, H, BT, K), dtype=q.dtype, device=q.device)
    ws_kd = torch.zeros_like(ws_qd)
    ws_kr = torch.zeros_like(ws_qd)
    ws_gt = torch.zeros((total_NT, H, K), dtype=torch.float32, device=q.device)

    for chunk_idx, (i_n, i_t, bos, t_start, t_end) in enumerate(chunks):
        seq_q = q[bos + t_start : bos + t_end]      # [actual_len, H, K]
        seq_k = k[bos + t_start : bos + t_end]
        seq_g = g[bos + t_start : bos + t_end].to(torch.float32)
        actual_len = t_end - t_start

        # g_total = g at last valid token (cumsum already applied chunk-locally)
        g_total_log = seq_g[-1]                      # [H, K]
        e_g = torch.exp2(seq_g)                      # [actual_len, H, K]
        e_gtmg = torch.exp2(g_total_log[None] - seq_g)

        qd = seq_q.to(torch.float32) * e_g * scale
        kd = seq_k.to(torch.float32) * e_g
        kr = seq_k.to(torch.float32) * e_gtmg

        # [actual_len, H, K] → [H, actual_len, K]
        ws_qd[chunk_idx, :, :actual_len] = qd.permute(1, 0, 2).to(q.dtype)
        ws_kd[chunk_idx, :, :actual_len] = kd.permute(1, 0, 2).to(q.dtype)
        ws_kr[chunk_idx, :, :actual_len] = kr.permute(1, 0, 2).to(q.dtype)
        ws_gt[chunk_idx] = torch.exp2(g_total_log)

    return ws_qd, ws_kd, ws_kr, ws_gt

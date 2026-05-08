# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Hopper KDA forward — v2 architecture: FlashKDA-style K1 + K2 split.

`kda_prefill_hopper_v2` is the v2 entry point for cuLA Hopper KDA. It implements
a FlashKDA-spec architecture:

    1. Preprocess (l2norm + gate cumsum + sigmoid beta)
    2. K1 (chunk-parallel preprocessing) — Triton kernel, 3 sub-kernels:
         decay-apply, sub-chunked Mqk, sub-chunked INV
    3. K2 (per-chunk recurrence + o + state update) — currently Python eager
         reference; to be replaced by a fast C++ kernel in a future change

vs the existing `kda_prefill_hopper`:
- Different math semantics: this v2 matches FlashKDA's output (verified e2e in
  tests/test_hopper_k1_e2e.py); the existing v1 has its own internal conventions
  that diverge from FlashKDA by ~0.89 rel diff.
- Different chunk parallelism: K1 is chunk-parallel (helps small B/H), K2 is
  per-(seq, head) like v1.

Performance: K2 is currently Python eager (slow). The architecture is validated
end-to-end vs FlashKDA dispatch but does not yet deliver speedup over v1. Replace
the Python K2 with a CUTLASS C++ K2 to realize the architectural win.
"""

import torch
from einops import rearrange
from fla.modules.l2norm import l2norm_fwd
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils.constant import RCP_LN2

import cula.cudac as cula_cuda
from cula.kda.hopper_k1 import kda_k1_full
from cula.kda.hopper_k2_reference import kda_k2_reference
from cula.utils import assert_hopper, prepare_uniform_cu_seqlens


def kda_prefill_hopper_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    safe_gate: bool = True,
    lower_bound: float = -5.0,
    cu_seqlens: torch.IntTensor | None = None,
    chunk_indices: torch.IntTensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    transpose_state_layout: bool = True,
    k2_backend: str = "python_ref",
):
    """v2 KDA prefill on Hopper, FlashKDA-spec.

    Args mirror `kda_prefill_hopper` (the v1 entry). Unlike v1, this matches
    FlashKDA's numerical output by construction (K1 outputs follow FlashKDA's
    K1 spec, K2-ref math follows FlashKDA's K2 source).

    Returns:
        o: [B, T, H, V] bf16
        final_state: [N, H, V, K] (transposed) if output_final_state else None
    """
    assert_hopper(q.device)
    assert use_qk_l2norm_in_kernel and use_gate_in_kernel and safe_gate, (
        "v2 currently requires use_qk_l2norm_in_kernel=use_gate_in_kernel=safe_gate=True "
        "(matches FlashKDA's only-supported flag combination)."
    )
    assert A_log is not None and dt_bias is not None, "v2 requires A_log and dt_bias"
    assert q.shape[-1] == 128 and v.shape[-1] == 128, "v2 currently requires K=V=128"
    assert q.shape[-2] == k.shape[-2] == v.shape[-2], "K/V must share H"

    chunk_size = 64
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = 1.0 / (K**0.5)

    if cu_seqlens is None:
        cu_seqlens = prepare_uniform_cu_seqlens(B, T, q.device, torch.int32)

    # Pack [B, T, ...] -> [1, B*T, ...] (matches v1 hopper_fused_fwd's flatten)
    q_4d = q.reshape(1, B * T, H, K).contiguous() if B != 1 else q
    k_4d = k.reshape(1, B * T, H, K).contiguous() if B != 1 else k
    v_4d = v.reshape(1, B * T, H, V).contiguous() if B != 1 else v
    g_4d = g.reshape(1, B * T, H, K).contiguous() if B != 1 else g
    beta_4d = beta.reshape(1, B * T, H).contiguous() if B != 1 else beta

    # Preprocessing: gate cumsum (with safe-gate sigmoid form), l2norm q/k.
    # IMPORTANT: A_log must be 1D shape [H]. FLA's gate kernel does
    # `tl.load(A_log + i_h)` (flat 1D). Broadcasting to [H, K] would silently
    # mis-load values for heads beyond 0.
    g_cumsum_4d = kda_gate_chunk_cumsum(
        g=g_4d.float(),
        A_log=A_log,
        dt_bias=dt_bias,
        scale=RCP_LN2,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        lower_bound=lower_bound,
    )
    q_l2, _ = l2norm_fwd(q_4d)
    k_l2, _ = l2norm_fwd(k_4d)
    beta_sig = beta_4d.float().sigmoid()

    # Pack to 3D [packed_seq, H, K] for K1
    packed_seq = B * T
    q_pk = q_l2.reshape(packed_seq, H, K).contiguous()
    k_pk = k_l2.reshape(packed_seq, H, K).contiguous()
    v_pk = v_4d.reshape(packed_seq, H, V).contiguous()
    g_pk = g_cumsum_4d.reshape(packed_seq, H, K).contiguous()
    beta_pk = beta_sig.reshape(packed_seq, H).contiguous()

    # K1: produces 6 workspace tensors
    ws = kda_k1_full(q_pk, k_pk, g_pk, beta_pk, scale, cu_seqlens=cu_seqlens, chunk_size=chunk_size)

    # K2: per-chunk recurrence + o + state update.
    # Two backends:
    #   - "python_ref" (default): slow but matches FlashKDA spec (validated e2e)
    #   - "cu_stub": calls the C++ stub kernel — currently returns zeros
    #     (placeholder; real K2 math TODO).
    if k2_backend == "python_ref":
        o_3d, h_state = kda_k2_reference(
            ws, v_pk, beta_pk,
            h_initial=initial_state,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )
        # h_state shape [N, H, K, V]. FlashKDA / FLA convention with
        # transpose_state_layout=True wants [N, H, V, K]. Match.
        if transpose_state_layout:
            h_state = h_state.transpose(-1, -2).contiguous()
    elif k2_backend in ("cu_stub", "cu_naive", "cu_wmma", "cu_wsp", "cu_wsp2", "cu_part"):
        # C++ K2 path (state in [N, H, V, K] transposed layout, FlashKDA convention).
        backend_id = {"cu_stub": 0, "cu_naive": 1, "cu_wmma": 2, "cu_wsp": 3,
                      "cu_wsp2": 4, "cu_part": 5}[k2_backend]
        # chunk_offsets[n] = first chunk index in workspace for sequence n.
        # Computed as prefix sum of NT_per_seq.
        cu_seqlens_cpu = cu_seqlens.to("cpu", non_blocking=False).int()
        nt_per_seq = []
        for i in range(cu_seqlens_cpu.numel() - 1):
            t_seq = (cu_seqlens_cpu[i + 1] - cu_seqlens_cpu[i]).item()
            nt_per_seq.append((t_seq + chunk_size - 1) // chunk_size)
        offsets = [0]
        for nt in nt_per_seq:
            offsets.append(offsets[-1] + nt)
        chunk_offsets = torch.tensor(offsets, dtype=torch.int32, device=cu_seqlens.device)

        o_3d, h_state = cula_cuda.kda_fwd_v2(
            None,
            None,
            v_pk,
            beta_pk,
            ws["ws_qd"],
            ws["ws_kd"],
            ws["ws_kr"],
            ws["ws_gt"],
            ws["ws_mqk"],
            ws["ws_inv"],
            initial_state,
            cu_seqlens,
            chunk_offsets,
            chunk_size,
            backend_id,
        )
        # Output already produces [N, H, V, K] state. If caller doesn't want transposed,
        # flip back.
        if not transpose_state_layout:
            h_state = h_state.transpose(-1, -2).contiguous()
    else:
        raise ValueError(f"Unknown k2_backend: {k2_backend!r}")

    # Reshape o back to [B, T, H, V]
    o_4d = o_3d.reshape(B, T, H, V)

    final_state = h_state if output_final_state else None
    return o_4d, final_state

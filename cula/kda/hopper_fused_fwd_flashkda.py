# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hopper (SM90) KDA forward — FlashKDA two-kernel pipeline.

This is a direct port of FlashKDA (https://github.com/MoonshotAI/FlashKDA) into
cuLA. It runs alongside ``cula.kda.kda_prefill_hopper`` (the existing fused
single-kernel path) and is selected explicitly via ``kda_prefill_hopper_flashkda``.

Key differences vs ``kda_prefill_hopper``:
- ``CHUNK = 16`` (vs 64 in the existing fused kernel)
- Two-kernel pipeline (prepare + recurrence) sharing a GMEM workspace
- L2-norm of q/k and gate activation/cumsum are *always* fused into K1 — the
  ``use_qk_l2norm_in_kernel`` and ``use_gate_in_kernel`` flags must be ``True``.

State layout (matches the existing ``kda_prefill_hopper`` SM90 path): both
``initial_state`` and ``final_state`` are ``[N, H, V, K]`` (K-last). Caller is
responsible for the transpose if working in BHKV convention.
"""

import torch
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

import cula.cudac as cula_cuda
from cula.utils import _get_cache_buf, assert_hopper


class HopperFlashKDAFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        use_beta_sigmoid_in_kernel: bool,
        lower_bound: float,
        cu_seqlens: torch.IntTensor | None,
    ):
        B, T_seq, H, D = q.shape
        T_total = B * T_seq

        # Beta convention reconciliation: FlashKDA K1 always applies sigmoid
        # internally. cuLA's chunk_kda convention is post-sigmoid beta in (0,1).
        # When use_beta_sigmoid_in_kernel=False, take logit so K1's sigmoid recovers it.
        if not use_beta_sigmoid_in_kernel:
            beta = torch.logit(beta.float().clamp(1e-6, 1 - 1e-6)).to(torch.bfloat16)
        elif beta.dtype != torch.bfloat16:
            beta = beta.to(torch.bfloat16)

        # cu_seqlens dtype: cuLA standard is int32, FlashKDA expects int64.
        if cu_seqlens is not None and cu_seqlens.dtype != torch.int64:
            cu_seqlens = cu_seqlens.to(torch.int64)

        # FlashKDA expects dt_bias as [H, K] (cuLA's API stores it flat as [H*K]).
        if dt_bias.dim() == 1:
            dt_bias = dt_bias.view(H, D)

        N = (cu_seqlens.numel() - 1) if cu_seqlens is not None else B

        # FlashKDA writes 'out' in place — we allocate it.
        out = torch.empty_like(q)

        # State: caller provides [N, H, V, K] (K-last) directly; pass through.
        # FlashKDA requires matching dtypes when both initial_state and final_state are set.
        final_state = None
        if output_final_state:
            state_dtype = initial_state.dtype if initial_state is not None else torch.float32
            final_state = torch.empty(N, H, D, D, dtype=state_dtype, device=q.device)

        # Workspace (cached + grow-on-demand to avoid alloc per call).
        ws_size = cula_cuda.flashkda_get_workspace_size(int(T_total), int(H), int(N))
        workspace = _get_cache_buf("flashkda_fwd_workspace", int(ws_size), q.device)
        # _get_cache_buf may return a buffer larger than requested; slice to exact size
        # so the C++ contiguity check sees the right view (uint8 contiguous slice is fine).
        workspace = workspace[:ws_size]

        cula_cuda.flashkda_fwd_prefill(
            q,
            k,
            v,
            g,
            beta,
            float(scale),
            out,
            workspace,
            A_log,
            dt_bias,
            float(lower_bound),
            initial_state,
            final_state,
            cu_seqlens,
        )

        return out, final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        raise NotImplementedError("Backward pass is not implemented yet for FlashKDA path.")


@torch.compiler.disable
def kda_prefill_hopper_flashkda(
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
    use_beta_sigmoid_in_kernel: bool = False,
    safe_gate: bool = True,
    lower_bound: float | None = None,
    cu_seqlens: torch.IntTensor | None = None,
    chunk_indices: torch.IntTensor | None = None,
    **kwargs,
):
    r"""
    Hopper (SM90) KDA forward prefill via FlashKDA's two-kernel pipeline.

    API matches :func:`cula.kda.kda_prefill_hopper` for drop-in comparison, with
    the constraints noted in the module docstring.

    Args:
        q, k, v: ``[B, T, H, D]`` bf16. ``D`` must be 128.
        g: pre-activation gate ``[B, T, H, D]`` bf16 (raw — K1 applies the
            sigmoid + lower_bound + cumsum internally).
        beta: ``[B, T, H]`` bf16 or fp32. When ``use_beta_sigmoid_in_kernel=False``
            (default, matching :func:`cula.kda.kda_prefill_hopper`) beta is taken
            as already-sigmoided values in ``(0, 1)`` and a logit is applied in
            the wrapper before passing to FlashKDA's K1 (which always sigmoids
            internally). When ``True``, beta is treated as raw pre-activation
            logits and passed straight through.
        scale: defaults to ``1 / sqrt(D)`` if ``None``.
        initial_state: ``[N, H, V, K]`` (K-last) bf16 or fp32, or ``None``. Same
            convention as :func:`cula.kda.kda_prefill_hopper`.
        output_final_state: if True, returns ``final_state`` in ``[N, H, V, K]``.
        use_qk_l2norm_in_kernel: must be True (FlashKDA K1 always l2-normalizes).
        use_gate_in_kernel: must be True (FlashKDA K1 always activates the gate).
        safe_gate: must be True. ``lower_bound`` is required and must be in ``[-5, 0)``.
        cu_seqlens: optional int32/int64 ``[N+1]``; varlen requires ``B == 1``.
        chunk_indices: ignored (kept for signature parity).

    Required ``kwargs``:
        A_log: fp32 ``[H]``.
        dt_bias: fp32 ``[H * K]`` (flat) or ``[H, K]``.
    """
    assert_hopper()
    assert use_qk_l2norm_in_kernel, "FlashKDA path always l2-normalizes inside the kernel."
    assert use_gate_in_kernel, "FlashKDA path always applies gate activation inside the kernel."
    assert safe_gate, "FlashKDA path requires safe_gate=True."
    if lower_bound is None or not (-5 <= lower_bound < 0):
        raise ValueError(f"`lower_bound` must be set and in [-5, 0), got {lower_bound}.")

    if "A_log" not in kwargs:
        raise ValueError("A_log must be provided.")
    A_log = kwargs["A_log"]
    dt_bias = kwargs.get("dt_bias")
    if dt_bias is None:
        raise ValueError("dt_bias must be provided.")

    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
            " Please flatten variable-length inputs before processing."
        )
    if cu_seqlens is not None and initial_state is not None:
        if initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"initial_state.shape[0] ({initial_state.shape[0]}) must equal "
                f"len(cu_seqlens) - 1 ({len(cu_seqlens) - 1})."
            )

    assert q.shape == k.shape == g.shape, "q, k, g must have the same shape."
    assert beta.shape == q.shape[:3], "beta must be of shape (B, T, H)."
    assert v.shape == (*q.shape[:3], v.shape[-1]), "v must be of shape (B, T, H, head_dim)."
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16, "q, k, v must be in bfloat16."
    assert q.shape[-1] == k.shape[-1] == v.shape[-1] == 128, "Currently only head dim 128 is supported."

    if scale is None:
        scale = q.shape[-1] ** -0.5

    o, final_state = HopperFlashKDAFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        scale,
        initial_state,
        output_final_state,
        use_beta_sigmoid_in_kernel,
        lower_bound,
        cu_seqlens,
    )
    return o, final_state

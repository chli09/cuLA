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


import torch
from einops import rearrange
from fla.modules.l2norm import l2norm_fwd
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

import cula.cudac as cula_cuda
from cula.utils import _get_cache_buf, assert_hopper, get_device_sm_count, prepare_uniform_cu_seqlens

CHUNK_SIZE = 64


def _kda_hopper_prep(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    *,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    safe_gate: bool,
    lower_bound: float | None,
    cu_seqlens: torch.IntTensor | None,
    chunk_indices: torch.IntTensor | None,
    chunk_size: int = CHUNK_SIZE,
) -> dict:
    """Run the Triton-side prep stack (cumsum + l2norm + batch flatten + packed reshape).

    Returns a dict of packed tensors ready for the cuLA C++ kernel call. The dict
    is the only thing the orchestrator needs to call ``_kda_hopper_call_kernel``
    multiple times without redoing this work — it's the foundation of the
    pass-dedup optimisation for segment-scan.
    """
    assert q.shape[-2] == v.shape[-2] == k.shape[-2], "Number of heads must be the same for q, k, v."
    batch_size, seq_len, num_heads, head_dim = q.shape
    head_dim_v = v.shape[-1]

    if cu_seqlens is None:
        cu_seqlens = prepare_uniform_cu_seqlens(batch_size, seq_len, q.device, torch.int32)

    # set batch size to 1 after handling cu_seqlens
    if batch_size != 1:
        q, k, v, g, beta = map(lambda x: rearrange(x, "b t ... -> 1 (b t) ..."), (q, k, v, g, beta))

    # gate preprocessing
    if use_gate_in_kernel:
        if safe_gate:
            assert lower_bound is not None, "lower_bound must be set when use safe_gate"
        g = kda_gate_chunk_cumsum(
            g=g, A_log=A_log, dt_bias=dt_bias, scale=RCP_LN2, chunk_size=chunk_size,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, lower_bound=lower_bound,
        )
    else:
        g = chunk_local_cumsum(
            g=g, chunk_size=chunk_size, scale=RCP_LN2,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        )

    if use_qk_l2norm_in_kernel:
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)

    # reshape to packed [T, H, D] for the C++ kernel
    packed_seq = batch_size * seq_len
    q = q.reshape(packed_seq, num_heads, head_dim).contiguous()
    k = k.reshape(packed_seq, num_heads, head_dim).contiguous()
    v = v.reshape(packed_seq, num_heads, head_dim_v).contiguous()
    g = g.reshape(packed_seq, num_heads, head_dim).contiguous()
    beta = beta.reshape(packed_seq, num_heads).contiguous()

    sm_count = get_device_sm_count(q.device)
    workspace_buffer = _get_cache_buf("hopper_kda_fwd_workspace", sm_count * 128, q.device)

    return {
        "q": q, "k": k, "v": v, "g": g, "beta": beta,
        "cu_seqlens": cu_seqlens, "workspace_buffer": workspace_buffer,
        "batch_size": batch_size, "seq_len": seq_len,
        "num_heads": num_heads, "head_dim": head_dim, "head_dim_v": head_dim_v,
        "num_seqs": cu_seqlens.shape[0] - 1,
    }


def _kda_hopper_call_kernel(
    prep: dict,
    *,
    scale: float,
    safe_gate: bool,
    num_segments: int,
    initial_state: torch.Tensor | None = None,
    output_state: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
):
    """Single C++ kernel invocation against pre-prepared packed tensors.

    initial_state semantics:
      - num_segments == 1 : 4D ``[N_seq, H, K, V]`` or None
      - num_segments  > 1 : either 4D (auto-expanded with seg 0 = init, seg ≥ 1 = 0)
                            or 5D ``[N_seq, N_seg, H, K, V]`` (used as-is — the
                            orchestrator's pass-2 path)

    output_state: optional pre-allocated output buffer. None lets the C++ side
    auto-allocate the appropriate shape.
    """
    num_seqs = prep["num_seqs"]
    num_heads = prep["num_heads"]
    head_dim = prep["head_dim"]
    head_dim_v = prep["head_dim_v"]
    device = prep["q"].device

    if num_segments > 1:
        seg_shape = (num_seqs, num_segments, num_heads, head_dim, head_dim_v)
        if initial_state is not None and initial_state.dim() == 5:
            assert tuple(initial_state.shape) == seg_shape, (
                f"5D initial_state must be {seg_shape}, got {tuple(initial_state.shape)}"
            )
            seg_input_state = initial_state.contiguous()
        else:
            seg_input_state = torch.zeros(seg_shape, dtype=torch.float32, device=device)
            if initial_state is not None:
                seg_input_state[:, 0, ...] = initial_state
        if output_state is None:
            output_state = torch.zeros(seg_shape, dtype=torch.float32, device=device)
    else:
        seg_input_state = initial_state

    o, final_state = cula_cuda.kda_fwd_prefill(
        None, output_state,
        prep["q"], prep["k"], prep["v"],
        seg_input_state,
        prep["g"],
        prep["beta"],
        prep["cu_seqlens"], prep["workspace_buffer"],
        scale, safe_gate, num_segments,
    )
    o = rearrange(o, "(b t) h d -> b t h d", b=prep["batch_size"])
    if out_dtype is not None:
        o = o.to(out_dtype)
    return o, final_state


class HopperChunkKDAFunction(torch.autograd.Function):
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
        initial_state: torch.Tensor,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        cu_seqlens: torch.IntTensor | None = None,
        chunk_indices: torch.IntTensor | None = None,
        num_segments: int = 1,
    ):
        out_dtype = q.dtype
        prep = _kda_hopper_prep(
            q, k, v, g, beta, A_log, dt_bias,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            safe_gate=safe_gate, lower_bound=lower_bound,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        )
        return _kda_hopper_call_kernel(
            prep, scale=scale, safe_gate=safe_gate, num_segments=num_segments,
            initial_state=initial_state, out_dtype=out_dtype,
        )

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        raise NotImplementedError("Backward pass is not implemented yet.")


@torch.compiler.disable
def cula_kda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: torch.IntTensor | None = None,
    chunk_indices: torch.IntTensor | None = None,
    num_segments: int = 1,
    **kwargs,
):
    r"""
    Hopper (SM90) fully-fused KDA forward prefill using CUTLASS TMA warp-specialized kernel.

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`.
        scale (Optional[float]):
            Scale factor for the KDA attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2norm to the q,k tensor internally. Default: `False`.
        use_gate_in_kernel (bool):
            Whether to compute the log-space KDA decay internally. Default: `False`.
        safe_gate (bool):
            Whether the kernel can assume the input gate values `g` are in a safe range.
            When `True`, the kernel can use M=16 TensorCore acceleration.
            The safe range is approximately [-5, 0). Default: `False`.
        lower_bound (Optional[float]):
            Lower bound for the forget gate activation function. Default: `None`.
        cu_seqlens (torch.IntTensor):
            Cumulative sequence lengths of shape `[N+1]`, int32.
        chunk_indices (torch.IntTensor):
            Chunk indices for variable-length training.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.
    """
    assert_hopper()
    assert safe_gate, "Only support safe_gate=True."
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if initial_state is not None:
        assert initial_state.dtype == torch.float32, "initial_state must be in float32."

    A_log, dt_bias = None, None
    if use_gate_in_kernel:
        assert "A_log" in kwargs, "A_log must be provided when use_gate_in_kernel=True."
        A_log, dt_bias = kwargs["A_log"], kwargs.get("dt_bias")
        if safe_gate:
            if lower_bound is None:
                raise ValueError("`lower_bound` must be specified when `safe_gate=True` and `use_gate_in_kernel=True`.")
            if not (-5 <= lower_bound < 0):
                raise ValueError(f"`lower_bound` must be in the safe range [-5, 0), got {lower_bound}.")

    assert q.shape == k.shape == g.shape, "q, k, g must have the same shape."
    assert beta.shape == q.shape[:3], "beta must be of shape (batch size, seq len, num of head)."
    assert v.shape == (*q.shape[:3], v.shape[-1]), "v must be of shape (batch size, seq len, num of head, head dim)."
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16, "q, k, v must be in bfloat16."
    assert beta.dtype == torch.bfloat16 or beta.dtype == torch.float32, "beta must be in bfloat16 or float32."
    assert q.shape[-1] == k.shape[-1] == v.shape[-1] == 128, "Currently we only support head dim of 128 for KDA"
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = HopperChunkKDAFunction.apply(
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
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        safe_gate,
        lower_bound,
        cu_seqlens,
        chunk_indices,
        num_segments,
    )
    return o, final_state


@torch.compiler.disable
def cula_kda_segment_scan_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: torch.IntTensor | None = None,
    chunk_indices: torch.IntTensor | None = None,
    num_segments: int = 2,
    **kwargs,
):
    r"""
    ChunkWiseParallel segment-scan KDA prefill (issue #11 Option C1).

    Splits each (seq, head)'s T-axis into ``num_segments`` parallel work
    items, expanding the kernel grid from B*H to B*H*N_seg. Targets the
    small-B/H + long-T regime where the legacy single-pass kernel is grid-
    underfilled (e.g. B=1, H=4 → grid 4, vs 132 SMs on GH200).

    Two-pass implementation (for num_segments == 2):

      Pass 1: input_state = [user_init, 0] per (seq, head).
              Kernel emits h_seg[:, 0] = M_0 @ user_init + h_ext_0
                                       (= correct h after seg 0)
                           h_seg[:, 1] = h_ext_1
                                       (= seg-1's local h_ext, init=0)
              o_pass1[seg=0] is correct, o_pass1[seg=1] is wrong.

      Pass 2: input_state = [user_init, h_seg[:, 0]] per (seq, head).
              Now seg 1 sees the correct prefixed h_initial. o_pass2 is
              correct for both segments.

      Final state = h_pass2[:, -1, ...].

    Note (num_segments ≥ 3): requires computing per-segment transition
    matrices M and chaining them; planned via FLA's
    `pre_process_fwd_kernel_merged` and `merge_fwd_bwd_kernel`. Not yet
    implemented — only num_segments == 2 is supported here.

    Args / behavior otherwise identical to ``cula_kda_prefill``.
    """
    if num_segments == 1:
        return cula_kda_prefill(
            q=q, k=k, v=v, g=g, beta=beta,
            scale=scale, initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            safe_gate=safe_gate, lower_bound=lower_bound,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            num_segments=1, **kwargs,
        )
    if num_segments != 2:
        raise NotImplementedError(
            f"cula_kda_segment_scan_prefill currently supports num_segments ∈ {{1, 2}}, "
            f"got {num_segments}. N≥3 needs the FLA M-chain merge step."
        )

    # ── Validation (mirrors cula_kda_prefill's checks) ───────────────────
    assert_hopper()
    assert safe_gate, "Only support safe_gate=True."
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
        )
    if initial_state is not None:
        assert initial_state.dtype == torch.float32, "initial_state must be in float32."
    A_log, dt_bias = None, None
    if use_gate_in_kernel:
        assert "A_log" in kwargs, "A_log must be provided when use_gate_in_kernel=True."
        A_log, dt_bias = kwargs["A_log"], kwargs.get("dt_bias")
        if safe_gate and lower_bound is None:
            raise ValueError("`lower_bound` must be specified when `safe_gate=True` and `use_gate_in_kernel=True`.")
    assert q.shape == k.shape == g.shape, "q, k, g must have the same shape."
    assert beta.shape == q.shape[:3]
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16
    assert q.shape[-1] == k.shape[-1] == v.shape[-1] == 128
    if scale is None:
        scale = q.shape[-1] ** -0.5

    # ── Pass dedup: prep stack runs ONCE (cumsum + l2norm + reshape) ─────
    # Without this, calling cula_kda_prefill twice would re-run all 3 Triton
    # launches per pass — confirmed empirically on H100 to leave N_seg=2 at
    # ratio 0.99x of N_seg=1 because prep duplication offsets grid expansion.
    out_dtype = q.dtype
    prep = _kda_hopper_prep(
        q, k, v, g, beta, A_log, dt_bias,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate, lower_bound=lower_bound,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )

    # ── Pass 1: seg 0 sees real init, seg 1 sees zero (emits h_ext) ──────
    # 4D initial_state → call_kernel expands to [N_seq, 2, H, K, V]
    _o_pass1, h_seg = _kda_hopper_call_kernel(
        prep, scale=scale, safe_gate=safe_gate, num_segments=2,
        initial_state=initial_state, out_dtype=out_dtype,
    )
    # h_seg[:, 0, ...] = M_0 @ user_init + h_ext_0  (correct h after seg 0)
    # h_seg[:, 1, ...] = h_ext_1                    (seg 1's local; replaced below)

    # ── Pass 2: rebuild input_state with seg 1 ← h after seg 0 ───────────
    pass2_init = torch.zeros_like(h_seg)
    if initial_state is not None:
        pass2_init[:, 0, ...] = initial_state
    pass2_init[:, 1, ...] = h_seg[:, 0, ...]

    o, h_final_seg = _kda_hopper_call_kernel(
        prep, scale=scale, safe_gate=safe_gate, num_segments=2,
        initial_state=pass2_init, out_dtype=out_dtype,
    )
    final_state = h_final_seg[:, -1, ...] if output_final_state else None
    return o, final_state

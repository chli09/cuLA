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
import triton
from einops import rearrange
from fla.modules.l2norm import l2norm_fwd
from fla.ops.cp.chunk_delta_h import merge_fwd_bwd_kernel, pre_process_fwd_kernel_merged
from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

import cula.cudac as cula_cuda
from cula.utils import _get_cache_buf, assert_hopper, get_device_sm_count, prepare_uniform_cu_seqlens

CHUNK_SIZE = 64

# Valid C++ template instantiations for num_segments (see kda_fwd_sm90_safe_gate.cu).
_VALID_N_SEGS = (1, 2, 4, 8, 16, 32)


def auto_num_segments(
    num_seqs: int,
    num_heads: int,
    seq_len: int,
    sm_count: int,
    chunk_size: int = CHUNK_SIZE,
) -> int:
    """Pick ``num_segments`` to bring the kernel grid close to ``sm_count``.

    The single-pass Hopper-fused kernel launches with grid ``= num_seqs * num_heads``,
    which underfills the SM array for small B/H. Segment-scan multiplies grid by N
    along the T axis (grid = num_seqs * num_heads * N). This heuristic picks the
    smallest N ∈ {1, 2, 4, 8, 16, 32} whose grid covers ``sm_count`` while
    satisfying the kernel's structural constraints.

    Constraints applied (in order):
      1. N ∈ {1, 2, 4, 8, 16, 32} — matches the C++ template instantiations.
      2. seq_len % N == 0 — uniform-segment requirement.
      3. seq_len // N ≥ 2 * chunk_size — at least 2 chunks per segment;
         shorter segments make the per-segment overhead (FLA intra + fused-M +
         merge + 2nd cuLA pass) dominate.
      4. seq_len ≥ 64 * chunk_size (= 4096) — segment-scan has a ~1 ms
         fixed-cost floor (prep + FLA intra + fused-M + merge + 2nd cuLA pass).
         Calibrated empirically on H100 NVL: at T = 2048 with N ∈ {8, 16},
         segment-scan regresses 28–31 % vs N = 1 even at small grid (B = 1,
         H = 4..16). At T = 4096 it crosses break-even and starts winning.
         Returns N = 1 here regardless of base_grid.
      5. 3 * base_grid ≥ sm_count — when the legacy grid already fills at
         least 1/3 of the SMs, segment-scan's fixed cost (~1 ms) eats more
         than the grid-doubling/quadrupling can recover. Empirical
         calibration: grid=64 + N=2 on H100 (132 SMs) regresses 10–20% vs
         N=1, even though base_grid*N hits sm_count. Returns N = 1 here.

    Returns the chosen N. N == 1 means "fall back to single-pass cula_kda_prefill".
    """
    base_grid = num_seqs * num_heads
    if 3 * base_grid >= sm_count:
        return 1                                         # ≥ 1/3-filled; segment-scan overhead > grid-expansion gain
    if seq_len < 64 * chunk_size:
        return 1                                         # below segment-scan break-even (~4096)
    desired = sm_count / base_grid                       # target multiplier
    # Round to nearest power of 2, clamped to valid set.
    # log2(desired): N=1 if desired<1.5, N=2 if 1.5≤desired<3, etc.
    import math
    cand = 1 << max(0, min(5, round(math.log2(desired))))
    # Tighten by T-divisibility / per-segment-chunk requirements.
    while cand > 1 and (seq_len % cand != 0 or seq_len // cand < 2 * chunk_size):
        cand //= 2
    return cand


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
    emit_transition: bool = False,
):
    """Single C++ kernel invocation against pre-prepared packed tensors.

    initial_state semantics:
      - num_segments == 1 : 4D ``[N_seq, H, K, V]`` or None
      - num_segments  > 1 : either 4D (auto-expanded with seg 0 = init, seg ≥ 1 = 0)
                            or 5D ``[N_seq, N_seg, H, K, V]`` (used as-is — the
                            orchestrator's pass-2 path)

    output_state: optional pre-allocated output buffer. None lets the C++ side
    auto-allocate the appropriate shape.

    emit_transition: when True, the kernel additionally emits the per-segment
    transition matrix M ∈ R^{K, K} (fp32) to a returned tensor of shape
    [N_seq, num_segments, H, K, K]. Layer 2 path. Returned as the third tuple
    element (None when emit_transition=False).
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

    o, final_state, output_M = cula_cuda.kda_fwd_prefill(
        None, output_state,
        prep["q"], prep["k"], prep["v"],
        seg_input_state,
        prep["g"],
        prep["beta"],
        prep["cu_seqlens"], prep["workspace_buffer"],
        scale, safe_gate, num_segments,
        emit_transition, None,
    )
    o = rearrange(o, "(b t) h d -> b t h d", b=prep["batch_size"])
    if out_dtype is not None:
        o = o.to(out_dtype)
    if emit_transition:
        return o, final_state, output_M
    return o, final_state


def _kda_extract_all_segment_ag_hm(
    kg_full: torch.Tensor,    # [B=1, T_packed, H, K] bf16
    u_full: torch.Tensor,     # [B=1, T_packed, H, V] bf16
    w_full: torch.Tensor,     # [B=1, T_packed, H, K] bf16
    gk_full: torch.Tensor,    # [B=1, T_packed, H, K] fp32
    num_segments: int,
    num_orig_seqs: int = 1,
    *,
    chunk_size: int = CHUNK_SIZE,
) -> torch.Tensor:
    """Compute ALL (num_orig_seqs × num_segments) per-orig-seq segment
    (h_ext, M) in ONE Triton launch via FLA's pre_process_fwd_kernel_merged
    (MULTI_SEQS=True path).

    Multi-orig-seq layout (num_orig_seqs > 1, e.g. B=2 input):
      cu_seqlens = [0, T_seg, 2*T_seg, ..., num_orig_seqs*num_segments * T_seg]
      where T_seg = T_orig / num_segments  (T_orig = T_packed / num_orig_seqs).
      Each fused-M block processes one (orig_seq, segment) pair → no segment
      crosses an orig_seq boundary.

    Returns ag_hm ∈ [num_orig_seqs * num_segments, H, K, K+V] fp32, indexed as
    ag_hm[orig_seq * num_segments + seg, ...]. For each (orig_seq, seg):
        ag_hm[..., :V]     = h_ext  (with init=0, FLA's WY-decomp result)
        ag_hm[..., V:V+K]  = M      (transition matrix)
    """
    assert kg_full.shape[0] == 1, "internal: prep tensors are flattened to B=1"
    _B, T_packed, H, K = kg_full.shape
    V = u_full.shape[-1]
    assert T_packed % (num_orig_seqs * num_segments) == 0, (
        f"T_packed={T_packed} must be divisible by num_orig_seqs*num_segments="
        f"{num_orig_seqs * num_segments}"
    )
    T_seg = T_packed // (num_orig_seqs * num_segments)
    total_segs = num_orig_seqs * num_segments

    BK = triton.next_power_of_2(K)
    BLOCK_SIZE = 32 if K <= 64 else 64
    cu = torch.arange(total_segs + 1, dtype=torch.int32, device=kg_full.device) * T_seg
    hm = kg_full.new_zeros(total_segs, H, K, V + K, dtype=torch.float32)
    grid = (
        triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE),
        H,
        total_segs,
    )
    pre_process_fwd_kernel_merged[grid](
        k=kg_full, v=u_full, w=w_full,
        g=None, gk=gk_full,
        hm=hm,
        cu_seqlens=cu,
        T=T_seg, H=H, K=K, V=V,
        BT=chunk_size, BK1=BK,
        USE_EXP2=True,
        BLOCK_SIZE=BLOCK_SIZE,
        MULTI_SEQS=True,
    )
    return hm  # [num_orig_seqs * num_segments, H, K, K+V]


def _kda_merge_segment_chain_via_fla(
    ag_hm: torch.Tensor,            # [num_orig_seqs * num_segments, H, K, K+V]
    initial_state: torch.Tensor | None,  # [num_orig_seqs, H, K, V] or None
    num_segments: int,
    num_orig_seqs: int,
    H: int,
    K: int,
    V: int,
    device: torch.device,
    skip_first_seg: bool = False,
) -> torch.Tensor:
    """FLA's merge_fwd_bwd_kernel chains the affine maps for each orig seq's
    N segments into per-(orig_seq, segment) h_initial. Multi-orig-seq aware.

    Args:
        skip_first_seg: when True, each orig seq's chain starts at its
            local ag_hm[1] (skip ag_hm[0] of that orig seq) and uses
            ``initial_state[orig_seq]`` as h0. This is the cuLA-consistent
            path used by the orchestrator (cuLA pass 1 already produced
            pass2_init[:, 1] = h_seg_pass1[:, 0] cuLA-faithfully).

    Returns ``initial_states_merge`` ∈ [num_orig_seqs * num_writes, H, K, V]
    where num_writes = num_segments - (2 if skip_first_seg else 1).
    Layout: orig_seq s's writes occupy slots [s*num_writes, (s+1)*num_writes).

    Implementation note: FLA's ``merge_fwd_bwd_kernel`` requires *gap-less*
    seq packing — for adjacent seqs i and i+1, ``seq_offsets[i+1]`` is BOTH
    seq i's exclusive end AND seq i+1's inclusive start, so they must be the
    same value. cuLA's natural ``ag_hm`` layout has seq i occupying slots
    ``[i*N, (i+1)*N)`` contiguously. With ``skip_first_seg=True``, the
    orchestrator wants seq i to skip its own seg 0 (slot ``i*N``) — but the
    kernel layout does not allow per-seq gaps. To remove the off-by-one
    (``ss_end[i]`` would otherwise overshoot into seq i+1's seg 0), we
    **pre-pack** ag_hm to exclude seg 0 of every orig seq before calling
    the kernel. The repacked tensor has shape
    ``[num_orig_seqs * (N-1), H, K, V+K]`` and seq i occupies slots
    ``[i*(N-1), (i+1)*(N-1))`` — gap-less.
    """
    total_segs = num_orig_seqs * num_segments
    assert ag_hm.shape[0] == total_segs and ag_hm.shape[3] == K + V

    if skip_first_seg and num_orig_seqs > 1:
        # Pre-pack: drop seg 0 of each orig_seq so seq boundaries are gap-less.
        # Required because FLA's merge kernel uses a single seq_offsets array
        # where seq i's end == seq i+1's start (no gap allowed). With
        # num_orig_seqs > 1 and skip_first_seg=True, the natural cuLA layout
        # leaves a gap of 1 slot at each seq boundary, which the kernel cannot
        # express — so seq i's iteration overruns into seq i+1's seg 0,
        # corrupting numerics.
        # For num_orig_seqs == 1 we keep the original zero-copy path: seg 0 is
        # at index 0 and ss_offset=1 simply skips it; there's no next seq to
        # collide with.
        ag_hm_view = ag_hm.view(num_orig_seqs, num_segments, H, K, V + K)
        ag_hm_eff = ag_hm_view[:, 1:, ...].contiguous().view(
            num_orig_seqs * (num_segments - 1), H, K, V + K
        )
        seg_per_seq_eff = num_segments - 1
        ss_offset_eff = 0
        # Each seq runs N-1 iterations; the final iteration is not stored
        # (it's the post-final-seg state, == the seq's final h, not pass2 input).
        num_writes_per_seq = (num_segments - 1) - 1  # = N - 2
    else:
        # Single-seq or no-skip: use ag_hm as-is, no copy.
        ag_hm_eff = ag_hm
        seg_per_seq_eff = num_segments
        ss_offset_eff = 1 if skip_first_seg else 0
        num_writes_per_seq = num_segments - 1 - ss_offset_eff
    assert num_writes_per_seq >= 1, "merge kernel needs at least one write target"

    # Build seq_offsets / init_offsets / h0_seq_ids over the (possibly
    # pre-packed) ag_hm_eff. Layout:
    #   - num_orig_seqs > 1 + skip_first_seg: ag_hm_eff is pre-packed, seq i
    #     occupies [i*(N-1), (i+1)*(N-1)) gap-less. ss_offset_eff = 0.
    #   - num_orig_seqs == 1: ag_hm_eff is the original ag_hm; seq 0 occupies
    #     [0, N), and ss_offset_eff in {0, 1} chooses whether to skip seg 0.
    seq_offsets_list = []
    init_offsets_list = []
    h0_seq_ids_list = []
    for s in range(num_orig_seqs):
        seq_offsets_list.append(s * seg_per_seq_eff + ss_offset_eff)
        init_offsets_list.append(s * num_writes_per_seq)
        h0_seq_ids_list.append(s)
    seq_offsets_list.append(num_orig_seqs * seg_per_seq_eff)  # end sentinel
    init_offsets_list.append(num_orig_seqs * num_writes_per_seq)  # end sentinel

    seq_offsets = torch.tensor(seq_offsets_list, dtype=torch.int32, device=device)
    init_offsets = torch.tensor(init_offsets_list, dtype=torch.int32, device=device)
    h0_seq_ids = torch.tensor(h0_seq_ids_list, dtype=torch.int32, device=device)

    initial_states_merge = ag_hm.new_empty(
        num_orig_seqs * num_writes_per_seq, H, K, V, dtype=torch.float32,
    )
    BK = triton.next_power_of_2(K)

    def grid(meta):
        return (triton.cdiv(V, meta['BV']), num_orig_seqs, H)

    merge_fwd_bwd_kernel[grid](
        h=initial_states_merge,
        ag_hm=ag_hm_eff,
        pre_or_post_num_ranks=num_orig_seqs,
        rank=0,
        seq_offsets=seq_offsets,
        init_offsets=init_offsets,
        h0_seq_ids=h0_seq_ids,
        h0=initial_state,
        H=H, K=K, V=V,
        BK=BK,
        FORWARD=True,
        INTRACARD_MODE=True,
        NUM_SEQ_ENTRIES=num_orig_seqs,
    )
    return initial_states_merge  # [num_orig_seqs * num_writes_per_seq, H, K, V]


def _kda_compute_segment_initial_states(
    prep: dict,
    h_seg_pass1: torch.Tensor,  # [N_seq, N_seg, H, K, V] from cuLA pass 1
    initial_state: torch.Tensor | None,
    num_segments: int,
    safe_gate: bool,
    scale: float,
    chunk_size: int = CHUNK_SIZE,
) -> torch.Tensor:
    """Build the per-segment h_initial array for cuLA pass 2.

    Hybrid design (cuLA-consistent first step + FLA M-chain for the tail):
        pass2_init[:, 0]      = user_init         (seg 0 starts from user h0)
        pass2_init[:, 1]      = h_seg_pass1[:, 0]  (cuLA-consistent: M_0 @ h0 + h_ext_0
                                                    computed by cuLA pass 1)
        pass2_init[:, 2..N-1] = FLA merge kernel, starting from h_seg_pass1[:, 0],
                                chaining ag_hm[1..N-1] = (h_ext, M) from FLA fused-M

    Why hybrid: an earlier full-FLA chain (skip cuLA pass 1, merge from ag_hm[0])
    gave a ~1.6× speedup but breaks pytest at init_random because FLA's
    M_0 / h_ext_0 differ slightly from cuLA's, and the M_0 @ h0 product
    amplifies that difference proportional to ‖h0‖. Keeping pass 1 to bootstrap
    pass2_init[:, 1] cuLA-consistently fixes the init_random failure.

    Pipeline (per call):
      1. FLA chunk_kda_fwd_intra ~95μs           (recover w, u, kg)
      2. _kda_extract_all_segment_ag_hm ~40μs    (fused-M, all N segs in 1 launch)
      3. _kda_merge_segment_chain_via_fla ~50μs  (1 launch, replaces host fold)
      4. zero-fill + index assignment            (host, ~negligible)

    For num_segments == 2, the merge is trivial (no M needed, no FLA intra
    call needed) — pass2_init[:, 1] = h_seg_pass1[:, 0] suffices.
    """
    num_seqs = prep["num_seqs"]
    num_heads = prep["num_heads"]
    head_dim = prep["head_dim"]
    head_dim_v = prep["head_dim_v"]
    device = prep["q"].device

    pass2_init = torch.zeros(
        (num_seqs, num_segments, num_heads, head_dim, head_dim_v),
        dtype=torch.float32, device=device,
    )
    if initial_state is not None:
        pass2_init[:, 0, ...] = initial_state
    pass2_init[:, 1, ...] = h_seg_pass1[:, 0, ...]  # cuLA-consistent

    if num_segments == 2:
        return pass2_init  # trivial case

    # FLA chunk_kda_fwd_intra: prep tensors are packed [packed_seq, H, D] with B=1.
    q4 = prep["q"].unsqueeze(0)
    k4 = prep["k"].unsqueeze(0)
    v4 = prep["v"].unsqueeze(0)
    g4 = prep["g"].unsqueeze(0)
    beta4 = prep["beta"].unsqueeze(0)
    w_full, u_full, _qg, kg_full, _Aqk, _Akk = chunk_kda_fwd_intra(
        q=q4, k=k4, v=v4, gk=g4, beta=beta4,
        scale=scale, chunk_size=chunk_size, safe_gate=safe_gate,
    )

    T_packed = q4.shape[1]
    assert T_packed % (num_seqs * num_segments) == 0, (
        f"non-uniform segment length not yet supported "
        f"(packed_seq={T_packed}, num_seqs={num_seqs}, num_segments={num_segments})"
    )

    # ag_hm[num_seqs * num_segments, H, K, K+V] — fused-M for ALL (orig_seq, seg) pairs.
    ag_hm = _kda_extract_all_segment_ag_hm(
        kg_full=kg_full, u_full=u_full, w_full=w_full, gk_full=g4,
        num_segments=num_segments, num_orig_seqs=num_seqs, chunk_size=chunk_size,
    )

    # ★ Overwrite FLA's h_ext (for seg ≥ 1, every orig seq) with cuLA's h_ext
    # from h_seg_pass1[s, k] (= M_k @ 0 + h_ext_k, since cuLA pass 1 ran segs
    # ≥ 1 with init=0). Keeps the merge chain cuLA-consistent on h_ext while
    # using FLA-derived M.
    ag_hm_view = ag_hm.view(num_seqs, num_segments, num_heads, head_dim, head_dim_v + head_dim)
    ag_hm_view[:, 1:, :, :, :head_dim_v] = h_seg_pass1[:, 1:, :, :, :]

    # FLA merge kernel: per orig seq, skip its ag_hm[seg=0], use h_seg_pass1[s, 0]
    # as h0. Output: initial_states_merge[s * (N_seg-2) + k] = h_initial for
    # orig_seq s's segment k+2.
    initial_states_merge = _kda_merge_segment_chain_via_fla(
        ag_hm=ag_hm,
        initial_state=h_seg_pass1[:, 0, ...].contiguous(),  # [num_seqs, H, K, V]
        num_segments=num_segments,
        num_orig_seqs=num_seqs,
        H=num_heads, K=head_dim, V=head_dim_v,
        device=device,
        skip_first_seg=True,
    )  # [num_seqs * (N_seg - 2), H, K, V]

    # Reshape to [num_seqs, N_seg-2, H, K, V] and assign to pass2_init[:, 2:].
    pass2_init[:, 2:, ...] = initial_states_merge.view(
        num_seqs, num_segments - 2, num_heads, head_dim, head_dim_v,
    )
    return pass2_init


def _kda_compute_segment_initial_states_full_fla(
    prep: dict,
    initial_state: torch.Tensor | None,
    num_segments: int,
    safe_gate: bool,
    scale: float,
    chunk_size: int = CHUNK_SIZE,
) -> torch.Tensor:
    """Build per-segment h_initial via full FLA path — no cuLA Pass 1 dependency.

    This is the FLA-CP-style ~1.x-pass approach (issue #11 PR4 / Step 1).
    Skips cuLA Pass 1 entirely; derives (h_ext, M) for ALL segments from FLA's
    pre_process_fwd_kernel_merged, then chains them from user_init via
    merge_fwd_bwd_kernel to produce pass2_init for every segment.

    Pipeline (replaces cuLA Pass 1 + the hybrid bootstrap in
    _kda_compute_segment_initial_states):
      1. FLA chunk_kda_fwd_intra ~95μs        recover (w, u, kg)
      2. _kda_extract_all_segment_ag_hm ~40μs fused-M for all N segments
      3. _kda_merge_segment_chain_via_fla ~50μs  chain from user_init
                                               (skip_first_seg=False)

    Caveat: a previous attempt at this path broke pytest at ``init_random``
    tolerance — FLA's M_0 / h_ext_0 differ slightly from cuLA's (likely due
    to bf16 cast ordering inside the WY-decomp), and the M_0 @ h_initial
    product amplifies that gap proportional to ‖h_initial‖. The root cause
    has not been diagnosed; this function exists to provide a clean entry
    point to (a) reproduce the failure, (b) bisect to a single tensor diff,
    (c) fix at source if it's a real bug, or (d) loosen tolerance if it's a
    well-bounded numerical drift. See the ``use_full_fla_path`` flag on
    :func:`cula_kda_segment_scan_prefill`.
    """
    num_seqs = prep["num_seqs"]
    num_heads = prep["num_heads"]
    head_dim = prep["head_dim"]
    head_dim_v = prep["head_dim_v"]
    device = prep["q"].device

    pass2_init = torch.zeros(
        (num_seqs, num_segments, num_heads, head_dim, head_dim_v),
        dtype=torch.float32, device=device,
    )
    if initial_state is not None:
        pass2_init[:, 0, ...] = initial_state

    # Step 1: FLA chunk_kda_fwd_intra → recover (w, u, kg).
    q4 = prep["q"].unsqueeze(0)
    k4 = prep["k"].unsqueeze(0)
    v4 = prep["v"].unsqueeze(0)
    g4 = prep["g"].unsqueeze(0)
    beta4 = prep["beta"].unsqueeze(0)
    w_full, u_full, _qg, kg_full, _Aqk, _Akk = chunk_kda_fwd_intra(
        q=q4, k=k4, v=v4, gk=g4, beta=beta4,
        scale=scale, chunk_size=chunk_size, safe_gate=safe_gate,
    )

    T_packed = q4.shape[1]
    assert T_packed % (num_seqs * num_segments) == 0, (
        f"non-uniform segment length not yet supported "
        f"(packed_seq={T_packed}, num_seqs={num_seqs}, num_segments={num_segments})"
    )

    # Step 2: FLA fused-M kernel → ag_hm[num_seqs * num_segments, H, K, K+V].
    ag_hm = _kda_extract_all_segment_ag_hm(
        kg_full=kg_full, u_full=u_full, w_full=w_full, gk_full=g4,
        num_segments=num_segments, num_orig_seqs=num_seqs, chunk_size=chunk_size,
    )

    # Step 3: FLA merge — chain ALL segments from user_init.
    # skip_first_seg=False so each orig seq runs ag_hm[0..N-1] from h0=user_init,
    # producing pass2_init[:, 1..N-1] for that seq.
    if initial_state is None:
        h0 = torch.zeros(
            (num_seqs, num_heads, head_dim, head_dim_v),
            dtype=torch.float32, device=device,
        )
    else:
        h0 = initial_state.contiguous()

    initial_states_merge = _kda_merge_segment_chain_via_fla(
        ag_hm=ag_hm,
        initial_state=h0,
        num_segments=num_segments,
        num_orig_seqs=num_seqs,
        H=num_heads, K=head_dim, V=head_dim_v,
        device=device,
        skip_first_seg=False,
    )  # [num_seqs * (N_seg - 1), H, K, V]

    # Reshape to [num_seqs, N_seg-1, H, K, V] and assign to pass2_init[:, 1:].
    pass2_init[:, 1:, ...] = initial_states_merge.view(
        num_seqs, num_segments - 1, num_heads, head_dim, head_dim_v,
    )
    return pass2_init


def _kda_compute_segment_initial_states_layer2(
    prep: dict,
    h_seg_pass1: torch.Tensor,    # [N_seq, N_seg, H, K, V] from cuLA pass 1
    M_pass1: torch.Tensor,        # [N_seq, N_seg, H, K, K] from cuLA pass 1 (Layer 2)
    initial_state: torch.Tensor | None,
    num_segments: int,
) -> torch.Tensor:
    """Build per-segment h_initial using cuLA-emitted (h_ext, M) — Layer 2 path.

    This is the Layer-2 counterpart to ``_kda_compute_segment_initial_states``
    (hybrid 2-pass) and ``_kda_compute_segment_initial_states_full_fla``
    (FLA-pre_process). It uses the M_seg matrix that cuLA Pass 1 emits when
    ``emit_transition=True`` to compute pass2_init via direct chain:

        pass2_init[seq, 0]    = user_init
        pass2_init[seq, k+1]  = M_seg[k] @ pass2_init[seq, k] + h_ext[seq, k]

    where h_ext[seq, k] = h_seg_pass1[seq, k] for k ≥ 1 (Pass 1 ran segs ≥ 1
    with init=0, so h_seg_pass1[seq, k] = M_k @ 0 + h_ext_k = h_ext_k).
    For seg 0, h_seg_pass1[seq, 0] is already the corrected init for seg 1
    (= M_0 @ user_init + h_ext_0).

    Pipeline:
      1. cuLA Pass 1 runs ONCE with emit_transition=True
                                 → produces (h_seg_pass1, M_pass1) per segment
      2. This function chains them on host side via simple matmul loop
      3. cuLA Pass 2 runs with the corrected pass2_init

    Total: 2 cuLA Pass calls + tiny host loop + 0 FLA Triton kernels.
    Estimated cost ~570μs (vs ~1.12ms for current hybrid 2-pass).

    ★ NOT YET FUNCTIONAL — depends on Layer 2 step 2b being complete in the
    kernel. As of 2026-05-05, the cuLA kernel emits M = product(diag(decay))
    only, missing the kg^T·w correction term. Calling this function with the
    current kernel produces incorrect pass2_init (M underflows to ~0 for
    realistic KDA gates, collapsing to init_zero behaviour).

    Once kernel step 2b lands and M is correct, this function should pass the
    same test_kda_segment_scan tolerance as the hybrid 2-pass path.
    """
    num_seqs = prep["num_seqs"]
    num_heads = prep["num_heads"]
    head_dim = prep["head_dim"]
    head_dim_v = prep["head_dim_v"]
    device = prep["q"].device

    pass2_init = torch.zeros(
        (num_seqs, num_segments, num_heads, head_dim, head_dim_v),
        dtype=torch.float32, device=device,
    )
    if initial_state is not None:
        pass2_init[:, 0, ...] = initial_state

    # Seg 1: bootstrap from pass 1's h_seg_pass1[:, 0] which is already
    # M_0·user_init + h_ext_0 (cuLA-self-consistent).
    pass2_init[:, 1, ...] = h_seg_pass1[:, 0, ...]

    if num_segments == 2:
        return pass2_init

    # Segs 2..N-1: chain via cuLA's emitted M.
    # pass2_init[:, k+1] = M_pass1[:, k] · pass2_init[:, k] + h_seg_pass1[:, k]
    # for k = 1..N-2.
    for k in range(1, num_segments - 1):
        # M_pass1[:, k] shape: [num_seqs, H, K, K]
        # pass2_init[:, k]:   [num_seqs, H, K, V]
        # Result:             [num_seqs, H, K, V]
        M_at_k = M_pass1[:, k, ...]                      # [num_seqs, H, K, K]
        prev = pass2_init[:, k, ...]                     # [num_seqs, H, K, V]
        # Use einsum for clarity: out[s, h, i, v] = sum_j M[s, h, i, j] · prev[s, h, j, v]
        chained = torch.einsum("shij,shjv->shiv", M_at_k, prev)
        pass2_init[:, k + 1, ...] = chained + h_seg_pass1[:, k, ...]

    return pass2_init


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
    num_segments: int | str = 2,
    use_full_fla_path: bool = False,
    **kwargs,
):
    r"""
    ChunkWiseParallel segment-scan KDA prefill (issue #11 Option C1).

    Splits each (seq, head)'s T-axis into ``num_segments`` parallel work
    items, expanding the kernel grid from B*H to B*H*N_seg. Targets the
    small-B/H + long-T regime where the legacy single-pass kernel is grid-
    underfilled (e.g. B=1, H=4 → grid 4, vs 132 SMs on GH200).

    ``num_segments`` accepts an integer in ``{1, 2, 4, 8, 16, 32}``, or the
    string ``"auto"`` to let :func:`auto_num_segments` pick the value that
    best saturates the SM array given (num_seqs, num_heads, seq_len) and
    the device's SM count.

    ``use_full_fla_path`` selects the orchestration strategy when
    ``num_segments >= 2``:

    - ``False`` (default): 2-pass hybrid. cuLA Pass 1 produces the
      cuLA-consistent ``h_seg_pass1[:, 0] = M_0 @ user_init + h_ext_0``;
      FLA's merge kernel handles segs ≥ 2. Bit-exact-stable across init
      shapes (used by the existing pytest matrix).
    - ``True``: full-FLA ~1.x-pass. Skip cuLA Pass 1; derive (h_ext, M)
      for ALL segments via FLA pre_process + merge, then run the cuLA
      fused kernel ONCE with corrected init. Saves ~250 μs vs the 2-pass
      hybrid, but a past attempt broke ``init_random`` tolerance — root
      cause not yet diagnosed. Use as an opt-in for benchmarking and
      diagnosis.

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
    # Resolve "auto" sentinel before any validation. We need num_seqs / seq_len /
    # num_heads — derive from raw inputs without doing the full prep yet.
    if isinstance(num_segments, str):
        if num_segments != "auto":
            raise ValueError(
                f"num_segments must be int or the literal string 'auto', got {num_segments!r}."
            )
        # num_seqs: from cu_seqlens if varlen, else B (q.shape[0]); num_heads: q.shape[2].
        # T_per_seq: q.shape[1] for fixed; for varlen we require uniform (matches the
        # T_packed % (num_seqs * num_segments) constraint deeper in this function).
        if cu_seqlens is None:
            num_seqs_resolved = q.shape[0]
            T_per_seq = q.shape[1]
        else:
            # cu_seqlens shape [N+1]; segments must be uniform for segment-scan.
            num_seqs_resolved = int(cu_seqlens.shape[0]) - 1
            T_packed = q.shape[1]                         # q is (1, T_packed, ...) for varlen
            if T_packed % num_seqs_resolved != 0:
                # Non-uniform varlen — segment-scan can't apply. Fall back to N=1.
                num_segments = 1
                T_per_seq = T_packed
            else:
                T_per_seq = T_packed // num_seqs_resolved
        num_heads_resolved = q.shape[2]
        if isinstance(num_segments, str):                 # still "auto"
            sm_count = get_device_sm_count(q.device)
            num_segments = auto_num_segments(
                num_seqs=num_seqs_resolved,
                num_heads=num_heads_resolved,
                seq_len=T_per_seq,
                sm_count=sm_count,
                chunk_size=CHUNK_SIZE,
            )

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
    if num_segments not in _VALID_N_SEGS:
        raise NotImplementedError(
            f"cula_kda_segment_scan_prefill currently supports num_segments ∈ {set(_VALID_N_SEGS)} "
            f"(matching the C++ template instantiations), got {num_segments}."
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

    # ── Two implementation paths, selected by ``use_full_fla_path`` ─────
    #
    # 2-pass hybrid (default, use_full_fla_path=False):
    #   Pass 1 keeps cuLA-consistent h_seg_pass1[:, 0] = M_0 @ user_init + h_ext_0
    #   for the first chain step. FLA merge handles segs 2..N-1.
    #   Pipeline: prep + cuLA Pass 1 + (FLA intra + fused-M + merge for N≥3)
    #             + cuLA Pass 2.
    #   Cost: ~1.12 ms in the worst-case shape.
    #
    # Full-FLA ~1.x-pass (use_full_fla_path=True):
    #   Skip cuLA Pass 1; derive (h_ext, M) for ALL segments from FLA's
    #   pre_process_fwd_kernel_merged, chain from user_init, then run cuLA
    #   fused kernel ONCE with corrected per-segment init.
    #   Pipeline: prep + FLA intra + fused-M + merge + cuLA forward (1×).
    #   Estimated cost: ~700 μs (saves ~250 μs vs the 2-pass hybrid).
    #   Caveat: a previous attempt at this path broke pytest at init_random
    #   tolerance — root cause not yet diagnosed. Use with care.
    out_dtype = q.dtype
    prep = _kda_hopper_prep(
        q, k, v, g, beta, A_log, dt_bias,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate, lower_bound=lower_bound,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )

    if use_full_fla_path:
        # Skip cuLA Pass 1 — build pass2_init via the full FLA chain.
        pass2_init = _kda_compute_segment_initial_states_full_fla(
            prep=prep, initial_state=initial_state,
            num_segments=num_segments, safe_gate=safe_gate, scale=scale,
        )
    else:
        # 2-pass hybrid: cuLA Pass 1 → bootstrap pass2_init[:, 1] cuLA-consistently.
        _o_pass1, h_seg = _kda_hopper_call_kernel(
            prep, scale=scale, safe_gate=safe_gate, num_segments=num_segments,
            initial_state=initial_state, out_dtype=out_dtype,
        )
        pass2_init = _kda_compute_segment_initial_states(
            prep=prep, h_seg_pass1=h_seg, initial_state=initial_state,
            num_segments=num_segments, safe_gate=safe_gate, scale=scale,
        )

    # Single forward (= old "Pass 2") with corrected per-segment init.
    o, h_final_seg = _kda_hopper_call_kernel(
        prep, scale=scale, safe_gate=safe_gate, num_segments=num_segments,
        initial_state=pass2_init, out_dtype=out_dtype,
    )
    final_state = h_final_seg[:, -1, ...] if output_final_state else None
    return o, final_state

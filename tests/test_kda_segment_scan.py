# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Verify cula_kda_segment_scan_prefill (issue #11 PR2, Option C1) produces the
# same (o, final_state) as the legacy single-pass cula_kda_prefill within fp32
# tolerance. Today only num_segments == 2 is implemented; the test extends
# trivially when ≥ 3 lands.

import pytest
import torch
import torch.nn.functional as F
from fla.utils import assert_close, device

from cula.kda.hopper_fused_fwd import (
    cula_kda_prefill,
    cula_kda_segment_scan_prefill,
)

pytestmark = pytest.mark.sm90_only


# T must be divisible by chunk_size * num_segments. Largest tested num_segments
# is 32 → for those, require T % (64 * 32) = T % 2048 == 0.
# Each shape is filtered to only the N values that satisfy this divisibility.
SHAPES = [
    pytest.param(1, 4, 256, id="B1-H4-T256"),       # only fits N≤4
    pytest.param(1, 4, 1024, id="B1-H4-T1024"),     # only fits N≤16
    pytest.param(1, 4, 8192, id="B1-H4-T8192"),     # PR1 worst-case shape, fits all N
    pytest.param(1, 8, 2048, id="B1-H8-T2048"),     # fits N≤32
    pytest.param(2, 4, 1024, id="B2-H4-T1024"),     # B>1 path
]


@pytest.mark.parametrize("num_segments", [2, 4, 8, 16, 32], ids=["N2", "N4", "N8", "N16", "N32"])
@pytest.mark.parametrize("B,H,T", SHAPES)
@pytest.mark.parametrize("with_init_state", [False, True], ids=["init_zero", "init_random"])
@pytest.mark.parametrize("use_full_fla_path", [False, True], ids=["hybrid_2pass", "full_fla_1pass"])
def test_segment_scan_n2_matches_single_pass(
    B: int, H: int, T: int, with_init_state: bool, num_segments: int, use_full_fla_path: bool,
):
    """Segment-scan orchestrator must match single-pass on (o, final_state)
    for each supported num_segments value, on both the 2-pass hybrid and the
    full-FLA ~1.x-pass paths."""
    D = 128
    chunk_size = 64
    if T % (chunk_size * num_segments) != 0:
        pytest.skip(f"T={T} not divisible by chunk_size*N_seg={chunk_size * num_segments}")

    torch.manual_seed(0)
    q = torch.rand(B, T, H, D, dtype=torch.bfloat16, device=device)
    k = torch.rand(B, T, H, D, dtype=torch.bfloat16, device=device)
    v = torch.rand(B, T, H, D, dtype=torch.bfloat16, device=device)
    # raw gate input — kernel will apply softplus + scale + cumsum internally
    g_raw = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(H, dtype=torch.float, device=device)
    dt_bias = torch.randn(H * D, dtype=torch.float, device=device)
    beta = torch.rand(B, T, H, dtype=torch.float32, device=device).sigmoid().to(torch.bfloat16)

    initial_state = None
    if with_init_state:
        # transposed-state layout? No — this entry takes [N, H, K, V] not transposed.
        initial_state = torch.randn(B, H, D, D, dtype=torch.float32, device=device)

    common_kw = dict(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g_raw.clone(),
        beta=beta.clone(),
        scale=D ** -0.5,
        initial_state=(initial_state.clone() if initial_state is not None else None),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-2.0,
        A_log=A_log.clone(),
        dt_bias=dt_bias.clone(),
    )

    # Reference: single-pass fused kernel (the unchanged path; bit-identical to before C2-2)
    o_ref, ht_ref = cula_kda_prefill(**{**common_kw, "num_segments": 1})

    # Under test: segment-scan orchestrator on the requested path
    o_seg, ht_seg = cula_kda_segment_scan_prefill(
        **{**common_kw, "num_segments": num_segments, "use_full_fla_path": use_full_fla_path},
    )

    # Tolerance — same as the existing test_kda_fused_fwd checks.
    # For num_segments >= 3 we use FLA's M kernel which uses FLA-recomputed
    # (w, u, kg). FLA's WY-decomposition output may differ slightly from cuLA's
    # internal (w, u, kg) — bf16 ops in different orders. Allow a slightly looser
    # tolerance for N >= 3 (still well within the existing 0.005 budget).
    tol = 0.005
    assert_close(f"o (segment_scan N={num_segments} vs single_pass)", o_ref, o_seg, tol)
    assert_close(f"final_state (segment_scan N={num_segments} vs single_pass)", ht_ref, ht_seg, tol)


@pytest.mark.parametrize("B,H,T", [pytest.param(1, 4, 1024, id="B1-H4-T1024")])
def test_segment_scan_n1_passthrough(B: int, H: int, T: int):
    """num_segments=1 must short-circuit to cula_kda_prefill exactly (bit-identical)."""
    D = 128
    torch.manual_seed(0)
    q = torch.rand(B, T, H, D, dtype=torch.bfloat16, device=device)
    k = torch.rand(B, T, H, D, dtype=torch.bfloat16, device=device)
    v = torch.rand(B, T, H, D, dtype=torch.bfloat16, device=device)
    g_raw = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(H, dtype=torch.float, device=device)
    dt_bias = torch.randn(H * D, dtype=torch.float, device=device)
    beta = torch.rand(B, T, H, dtype=torch.float32, device=device).sigmoid().to(torch.bfloat16)

    common_kw = dict(
        q=q, k=k, v=v, g=g_raw, beta=beta,
        scale=D ** -0.5,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-2.0,
        A_log=A_log.clone(),
        dt_bias=dt_bias.clone(),
    )
    o1, ht1 = cula_kda_prefill(**common_kw)
    o2, ht2 = cula_kda_segment_scan_prefill(**common_kw, num_segments=1)
    # Should be bit-exact (passthrough does not modify any tensor)
    assert torch.equal(o1, o2), "num_segments=1 path must be bit-identical to cula_kda_prefill"
    if ht1 is not None:
        assert torch.equal(ht1, ht2), "final_state must be bit-identical for num_segments=1"

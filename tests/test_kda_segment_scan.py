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


# T must be divisible by chunk_size * num_segments (chunk_size = 64, N_seg = 2 → T % 128 == 0).
# Each shape exercises a different small-B/H regime that's the focus of issue #11.
SHAPES = [
    pytest.param(1, 4, 256, id="B1-H4-T256"),       # smallest sane segment (T/N_seg = 128 = 2 chunks)
    pytest.param(1, 4, 1024, id="B1-H4-T1024"),     # mid
    pytest.param(1, 4, 8192, id="B1-H4-T8192"),     # PR1 worst-case shape (cuLA loses 0.52x to FLA)
    pytest.param(1, 8, 2048, id="B1-H8-T2048"),     # slightly more heads
    pytest.param(2, 4, 1024, id="B2-H4-T1024"),     # B>1 path (internal batch flatten)
]


@pytest.mark.parametrize("B,H,T", SHAPES)
@pytest.mark.parametrize("with_init_state", [False, True], ids=["init_zero", "init_random"])
def test_segment_scan_n2_matches_single_pass(B: int, H: int, T: int, with_init_state: bool):
    """N_seg=2 segment-scan orchestrator must match single-pass on (o, final_state)."""
    D = 128
    chunk_size = 64
    assert T % (chunk_size * 2) == 0, f"T={T} not divisible by chunk_size*N_seg=128"

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

    # Under test: 2-pass segment-scan orchestrator
    o_seg, ht_seg = cula_kda_segment_scan_prefill(**{**common_kw, "num_segments": 2})

    # Tolerance — same as the existing test_kda_fused_fwd checks.
    assert_close("o (segment_scan vs single_pass)", o_ref, o_seg, 0.005)
    assert_close("final_state (segment_scan vs single_pass)", ht_ref, ht_seg, 0.005)


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

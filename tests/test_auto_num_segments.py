# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Pure-Python tests for auto_num_segments heuristic.

No GPU needed — these test the dispatch logic only. Kept in a separate file so
they don't inherit the SM90 skip mark from test_kda_segment_scan.py.
"""

import pytest

from cula.kda.hopper_fused_fwd import auto_num_segments


@pytest.mark.parametrize(
    "num_seqs,num_heads,seq_len,sm_count,expected_n",
    [
        # Half-saturated or above → N=1 (segment-scan overhead > gain).
        (4, 64, 16384, 132, 1),                          # grid=256, way over
        (8, 16, 16384, 132, 1),                          # grid=128, equal to sm_count
        (4, 16, 8192, 132, 1),                           # grid=64, just below half-sat → still N=1
        (8, 8,  8192, 132, 1),                           # grid=64, same
        (1, 64, 16384, 132, 1),                          # grid=64
        # Just under 1/3-saturated: grid=32 (e.g. 4*8=32). 3*32=96 < 132 → pass guard.
        # 132/32=4.1 → log2≈2 → N=4, 8192/4=2048 ≥ 128 ✓.
        (4, 8, 8192, 132, 4),
        # Below segment-scan break-even (T < 32 * chunk_size = 2048) → N=1.
        # Empirical: legacy fused at T ≤ 1024 wins ≥ 3× over FLA, so segment-scan's
        # ~1 ms fixed cost would regress these.
        (1, 4, 64, 132, 1),
        (1, 4, 256, 132, 1),
        (1, 4, 512, 132, 1),
        (1, 4, 1024, 132, 1),
        # T = 2048 < 64*chunk_size break-even → N=1 short-circuit (changed from N=16
        # after empirical evidence: T=2048 segment-scan regresses 28-31% vs N=1).
        (1, 4, 2048, 132, 1),
        # T = exactly 4096 = 64*chunk_size → considered. Desired = 132/4 = 33 → N=32.
        # 4096/32=128 = 2*chunk_size ✓ → expect 32.
        (1, 4, 4096, 132, 32),
        # Issue #11 sweet-spot Zone B: small grid + long T → N picked to saturate SMs.
        # B=1 H=4 (grid=4), T=4096 → desired = 132/4 = 33 → log2≈5 → N=32, 4096/32=128 ≥ 128 ✓.
        (1, 4, 4096, 132, 32),
        (1, 4, 8192, 132, 32),
        (1, 4, 16384, 132, 32),
        # B=2 H=8 (grid=16), T=8192 → desired=132/16=8.25 → log2≈3 → N=8, 8192/8=1024 ≥ 128 ✓.
        (2, 8, 8192, 132, 8),
        # T=5000, base_grid=4: ≥ 4096 break-even ✓. Desired=33 → N=32; 5000%32=8 fails.
        # 16: 5000%16=8 fails. 8: 5000%8=0 ✓, 5000/8=625 ≥ 128 ✓ → expect 8.
        (1, 4, 5000, 132, 8),
    ],
    ids=[
        "saturated-4x64",
        "saturated-8x16",
        "above-1/3-sat-grid64-4x16",
        "above-1/3-sat-grid64-8x8",
        "above-1/3-sat-grid64-1x64",
        "just-under-1/3-sat-grid32",
        "tiny-T-64",
        "tiny-T-256",
        "short-T-512",
        "short-T-1024",
        "T-2048-now-short-circuits",
        "boundary-T-4096-N32",
        "B1H4-T4096-N32",
        "B1H4-T8192-N32",
        "B1H4-T16384-N32",
        "B2H8-T8192-N8",
        "non-divisible-T5000-drops",
    ],
)
def test_auto_num_segments_heuristic(
    num_seqs: int, num_heads: int, seq_len: int, sm_count: int, expected_n: int
):
    """auto_num_segments() should obey: clamp to {1,2,4,8,16,32}, T%N==0,
    seq_len/N >= 2*chunk_size, and short-circuit T<4*chunk_size to N=1."""
    got = auto_num_segments(num_seqs, num_heads, seq_len, sm_count, chunk_size=64)
    assert got == expected_n, (
        f"auto_num_segments({num_seqs}, {num_heads}, {seq_len}, sm={sm_count}) "
        f"= {got}, expected {expected_n}"
    )


def test_auto_num_segments_returns_valid_set():
    """Every returned N must be in {1, 2, 4, 8, 16, 32}."""
    valid = {1, 2, 4, 8, 16, 32}
    import random
    random.seed(0)
    for _ in range(200):
        ns = random.randint(1, 32)
        h = random.choice([1, 4, 8, 16, 32, 64])
        t = random.choice([64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
        sm = random.choice([108, 132, 144, 148])
        n = auto_num_segments(ns, h, t, sm, chunk_size=64)
        assert n in valid, f"got {n} for (ns={ns}, h={h}, t={t}, sm={sm})"

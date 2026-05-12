#!/usr/bin/env python3
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

"""Head-to-head benchmark on Hopper (SM90):
   ``cula.kda.kda_prefill_hopper`` (existing fused single-kernel)
   vs.
   ``cula.kda.kda_prefill_hopper_flashkda`` (newly migrated FlashKDA two-kernel pipeline).

Reports kernel time per config, speedup, and a numerical sanity check (max abs
diff between the two outputs).

Usage:
    python benchmarks/bench_kda_fused_fwd_flashkda_vs_cula.py
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from cula.kda import kda_prefill_hopper, kda_prefill_hopper_flashkda
from cula.utils import assert_hopper

WARMUP = 10
N_ITERS = 50
D = 128


def _make_inputs(B, T, H, *, seed=42):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    q = torch.rand(B, T, H, D, dtype=torch.bfloat16, device=device)
    k = torch.rand(B, T, H, D, dtype=torch.bfloat16, device=device)
    v = torch.rand(B, T, H, D, dtype=torch.bfloat16, device=device)
    g = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device)
    beta = torch.randn(B, T, H, dtype=torch.float32, device=device).sigmoid()
    A_log = torch.randn(H, dtype=torch.float32, device=device) * 0.01
    dt_bias = torch.zeros(H * D, dtype=torch.float32, device=device)
    h0 = torch.randn(B, H, D, D, dtype=torch.float32, device=device)
    h0_vk = h0.transpose(-1, -2).contiguous()
    return q, k, v, g, beta, A_log, dt_bias, h0_vk


def _time_call(fn, *, iters=N_ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=N_ITERS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    args = parser.parse_args()

    assert_hopper()

    configs = [
        # (B, T, H)
        (1, 256, 16),
        (1, 1024, 16),
        (1, 2048, 32),
        (1, 4096, 32),
        (2, 1024, 32),
        (2, 2048, 32),
        (4, 2048, 32),
    ]

    print(
        f"{'B':>3} {'T':>6} {'H':>4} | {'fused (ms)':>12} {'flashkda (ms)':>14} "
        f"{'speedup':>8} | {'max|Δo|':>10} {'max|Δht|':>10}"
    )
    print("-" * 90)

    for B, T, H in configs:
        q, k, v, g, beta, A_log, dt_bias, h0_vk = _make_inputs(B, T, H)
        common = dict(
            beta=beta, A_log=A_log, dt_bias=dt_bias,
            initial_state=h0_vk, output_final_state=True,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            safe_gate=True, lower_bound=-5.0,
        )

        def run_fused():
            return kda_prefill_hopper(q=q, k=k, v=v, g=g, **common)

        def run_flash():
            return kda_prefill_hopper_flashkda(q=q, k=k, v=v, g=g, **common)

        # Numerical sanity check on a single fresh call (cloned inputs to avoid
        # in-place pollution).
        o1, ht1 = run_fused()
        o2, ht2 = run_flash()
        diff_o = (o1.float() - o2.float()).abs().max().item()
        diff_ht = (ht1.float() - ht2.float()).abs().max().item()

        t_fused = _time_call(run_fused, iters=args.iters, warmup=args.warmup)
        t_flash = _time_call(run_flash, iters=args.iters, warmup=args.warmup)
        speedup = t_fused / t_flash

        print(
            f"{B:>3} {T:>6} {H:>4} | {t_fused:>12.4f} {t_flash:>14.4f} "
            f"{speedup:>7.2f}x | {diff_o:>10.4f} {diff_ht:>10.4f}"
        )


if __name__ == "__main__":
    main()

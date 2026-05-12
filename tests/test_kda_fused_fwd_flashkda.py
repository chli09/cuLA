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

"""Smoke tests for the FlashKDA-port SM90 path.

Compares ``cula.kda.kda_prefill_hopper_flashkda`` (newly migrated FlashKDA two-kernel
pipeline) against ``cula.kda.kda_prefill_hopper`` (existing fused single kernel) on
identical inputs. Tolerances are loose because the two kernels use different
chunk sizes and accumulation orders.
"""

import pytest
import torch
import torch.nn.functional as F
from fla.utils import assert_close, device

from cula.kda import kda_prefill_hopper, kda_prefill_hopper_flashkda

pytestmark = pytest.mark.sm90_only


def _make_inputs(B, T, H, D, *, beta_dtype=torch.float32, seed=42):
    torch.manual_seed(seed)
    q = torch.rand(B, T, H, D, dtype=torch.bfloat16, device=device)
    k = torch.rand(B, T, H, D, dtype=torch.bfloat16, device=device)
    v = torch.rand(B, T, H, D, dtype=torch.bfloat16, device=device)
    # Raw pre-activation gate, bf16 (FlashKDA K1 applies sigmoid + lower_bound + cumsum)
    g = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device)
    # Pre-sigmoid beta (both paths apply sigmoid internally when fed sigmoid'd input
    # the existing test sigmoids before passing — match that convention here too)
    beta = torch.randn(B, T, H, dtype=torch.float32, device=device).sigmoid().to(beta_dtype)
    A_log = (torch.randn(H, dtype=torch.float32, device=device) * 0.01)
    dt_bias = torch.zeros(H * D, dtype=torch.float32, device=device)
    h0 = torch.randn(B, H, D, D, dtype=torch.float32, device=device)
    h0_vk = h0.transpose(-1, -2).contiguous()  # both SM90 paths use BHVK
    return q, k, v, g, beta, A_log, dt_bias, h0_vk


@pytest.mark.parametrize(
    ("B", "T", "H", "D"),
    [
        pytest.param(1, 64, 1, 128, id="B1-T64-H1"),
        pytest.param(2, 256, 4, 128, id="B2-T256-H4"),
        pytest.param(2, 1024, 8, 128, id="B2-T1024-H8"),
        pytest.param(1, 2048, 16, 128, id="B1-T2048-H16"),
    ],
)
def test_flashkda_vs_fused_fwd(B, T, H, D):
    q, k, v, g, beta, A_log, dt_bias, h0_vk = _make_inputs(B, T, H, D)

    common_kwargs = dict(
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=h0_vk.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
    )

    o_fused, ht_fused = kda_prefill_hopper(
        q=q.clone(), k=k.clone(), v=v.clone(), g=g.clone(), **common_kwargs
    )
    o_flash, ht_flash = kda_prefill_hopper_flashkda(
        q=q.clone(), k=k.clone(), v=v.clone(), g=g.clone(), **common_kwargs
    )

    # Loose tolerance — two kernels with different chunk size + accumulation order.
    assert_close("o", o_fused, o_flash, 0.01)
    assert_close("ht", ht_fused, ht_flash, 0.01)


def test_flashkda_no_initial_state():
    """No initial_state path — final_state should still be produced."""
    B, T, H, D = 1, 256, 2, 128
    q, k, v, g, beta, A_log, dt_bias, _ = _make_inputs(B, T, H, D)

    o, ht = kda_prefill_hopper_flashkda(
        q=q, k=k, v=v, g=g, beta=beta,
        A_log=A_log, dt_bias=dt_bias,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
    )
    assert o.shape == (B, T, H, D)
    assert ht is not None
    assert ht.shape == (B, H, D, D)


def test_flashkda_no_final_state():
    """output_final_state=False path — final_state should be None."""
    B, T, H, D = 1, 256, 2, 128
    q, k, v, g, beta, A_log, dt_bias, h0_vk = _make_inputs(B, T, H, D)

    o, ht = kda_prefill_hopper_flashkda(
        q=q, k=k, v=v, g=g, beta=beta,
        A_log=A_log, dt_bias=dt_bias,
        initial_state=h0_vk,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
    )
    assert o.shape == (B, T, H, D)
    assert ht is None


def test_flashkda_varlen():
    """Varlen path: B=1, packed sequences with cu_seqlens."""
    H, D = 4, 128
    seq_lens = [64, 128, 96]
    T_total = sum(seq_lens)
    N = len(seq_lens)
    cu_seqlens = torch.tensor([0, *list(_cumsum(seq_lens))], dtype=torch.int32, device=device)

    q, k, v, g, beta, A_log, dt_bias, _ = _make_inputs(1, T_total, H, D)
    h0_vk = torch.randn(N, H, D, D, dtype=torch.float32, device=device).transpose(-1, -2).contiguous()

    o, ht = kda_prefill_hopper_flashkda(
        q=q, k=k, v=v, g=g, beta=beta,
        A_log=A_log, dt_bias=dt_bias,
        initial_state=h0_vk,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        cu_seqlens=cu_seqlens,
    )
    assert o.shape == (1, T_total, H, D)
    assert ht.shape == (N, H, D, D)


def _cumsum(xs):
    s = 0
    for x in xs:
        s += x
        yield s

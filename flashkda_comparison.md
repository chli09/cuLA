# FlashKDA / Kimi 官方实现 — 对比与战略分析

**Last updated:** 2026-05-07
**Scope:** 此文档专门记录对 Kimi 官方 FlashKDA 实现的拆解、与 cuLA 各路径的架构对比，以及由此对 issue #11 优化方向的影响。
**关系:** 与 `issue11_insight.md` 互补。后者是 issue #11 全周期"分析—决策"的活页本；此文件聚焦于"FlashKDA 是什么、跟我们什么关系、决定我们走哪条路"。

---

## 0. 关键事实（结论先行）

1. **FlashKDA 已经是 FLA `chunk_kda` 的默认 backend**（FLA PR #852）。用户在 `torch.inference_mode()` 下调 `fla.ops.kda.chunk_kda` 实际上跑的是 FlashKDA C++，不是 Triton。
2. **Repo:** `github.com/MoonshotAI/FlashKDA`。Deep-dive 博客：`docs/20260420-flashkda-v1-deep-dive.md`。
3. **架构:** 2-kernel (K1 chunk-parallel preprocessing + K2 head-parallel recurrence)。**CHUNK = 16**。
4. **数值优化:** fp16 求逆 (Neumann 级数)、bf16 state on-chip、`tanh.approx` 实现 sigmoid、`ex2.approx.ftz` 实现指数。
5. **硬件:** SM90+，CHUNK=16 也兼容 SM80 mma.sync。
6. **H20 实测:** 比 `fla_chunk_kda` (整合前的 Triton 版) 快 1.85–2.31×，比 `fla_chunk_gdn` 快 1.17–1.43×。
7. **fwd-only**（没有 backward 实现）。

---

## 1. FlashKDA 详细流程

### K1 (token/chunk-parallel)
**Grid:** `(N · H · NT, 1, 1)` —— NT 直接进 grid 维。

每个 block 处理一个 chunk (16 tokens)，全部在 SMEM 内：

```
1. TMA load q[16,128], k[16,128], beta[16], g_bf16[16,128], dt_bias[128]
2. L2 norm (q, k):  warp-shuffle 归约，结果原位写回 SMEM
3. 融合 gate activation + cumsum:
     g[t,d] = cumsum_t( gate_scale · sigmoid_tanh_approx( a_log_exp · (g_bf16[t,d] + dt_bias[d]) ) )
     g_total[d] = 整段 cumsum 总和
   (另一半 warps 做 k 的 tail zero-fill)
4. Decay apply (256 threads):
     q_decayed  = q · exp2(g) · scale
     k_decayed  = k · exp2(g)
     k_inv      = k · exp2(-g)
     k_restored = k · exp2(g_total - g)
5. 并行两个 16×16 MMA (各 1 warp):
     L_fp16 = MMA_bf16xbf16→fp16(k_decayed, k_inv)
     Mqk    = MMA_bf16xbf16→bf16(q_decayed, k_inv)
6. tril 掩码 + INV = I - L (融合, 同 thread 同元素):
     L_fp16(i,j>=i) = 0
     L_fp16(i,j<i)  *= sigmoid(beta[i])
     INV_fp16(i,j) = (i==j ? 1 : 0) - L_fp16(i,j)
7. INV = (I - L)^-1 via Neumann 级数 (1 warp, 全 fp16)
8. TMA store: ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk
```

**Workspace shapes:**
- `ws_kd, ws_qd, ws_kr ∈ [N·total_tiles, 16, 128] bf16`
- `ws_gt ∈ [N·total_tiles, 128] fp32`
- `ws_inv, ws_mqk ∈ [N·total_tiles, 16, 16] bf16`

### K2 (head-parallel)
**Grid:** `(N, H)` —— 没有 NT，chunk 在 block 内串行。

每个 block 持 `state_acc ∈ SMEM bf16 [128, 128]`，跨整个 NT loop 不出片：

```
1. TMA load h0 → state_acc (bf16; fp32 输入走单独转换路径)
2. Warp specialization (LdSt / MMA / Out):
   for t = 0..NT-1:    ← 串行 chunk loop
       LdSt warps: 流水线 TMA load chunk t 的 (v, beta, ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk)
       MMA warps:
           a. v_pred = ws_kd · state_acc^T          (W·h, MOVM_T 寄存器内转置)
           b. v_proxy = INV · v                      (16×16 fp16 inv 应用到 bf16 v)
           c. v_new = v_proxy - v_pred               (即 u - W·h)
           d. o = Mqk · v + ws_qd · state_acc        (intra + inter Q·h, 同步算)
           e. state_acc *= diag(ws_gt)               (gate 衰减, 全片上)
           f. state_acc += ws_kd^T · v_new           (state 更新, 仍 SMEM)
       Out warps: TMA store o
3. TMA store final_state from state_acc
```

**关键性质:**
- `state_acc` 从 K2 入口 TMA load 一次，到出口 TMA store 一次，**整个 NT loop 期间不出 SMEM**
- Workspace 提供 K2 所需的所有"per-chunk 独立"数据，K2 自己不再算 intra-MMA 或求逆

---

## 2. 跟 cuLA Hopper fused (`kda_prefill_hopper`) 的对比

**当前 cuLA Hopper 流程 (4 launch):**

```
Launch 1 (Triton):  cumsum
Launch 2 (Triton):  l2norm(q)
Launch 3 (Triton):  l2norm(k)
Launch 4 (cuLA C++ kda_fwd_prefill):
   for chunk in 0..NT-1:           ← 串行
       ① decay apply
       ② A_qk = q·k^T,  A_kk = k·k^T          ← intra-attn 分数
       ③ L_inv 求逆 (tf32 LU 或 Neumann)        ← delta-rule 求逆
       ④ W, U = L_inv·k, L_inv·v
       ⑤ v_new = u - h·W                       ← 用 h
       ⑥ h = decay·h + k^T·v_new                ← 更新 h
       ⑦ o = tril(A_qk)·v_new + q·h            ← 用 h
```

**逐操作对照表:**

| 操作 | cuLA fused 在哪 | FlashKDA 在哪 |
|---|---|---|
| g 的 cumsum | Launch 1 (Triton) | **K1** (融进去) |
| L2 norm | Launch 2, 3 (Triton) | **K1** (融进去) |
| decay apply | Launch 4 内 | K1 内 |
| **intra-attn 分数 (A_qk / Mqk)** | **Launch 4 内 (chunk loop 里)** | **K1 内 (chunk-parallel) ★** |
| **delta-rule 求逆 (INV)** | **Launch 4 内 (chunk loop 里)** | **K1 内 (chunk-parallel) ★** |
| 算 v_new | Launch 4 内 (用 h) | K2 内 (用 h) |
| 更新 h | Launch 4 内 | K2 内 |
| 算 o | Launch 4 内 | K2 内 |

### 关键差异 —— ★ 行
**cuLA fused 把"per-chunk 独立、不依赖 h"的工作（intra-MMA + 求逆）放在串行 T-loop 里，跟 h 更新搅在一起。FlashKDA 把它们抽出到 K1，独立 chunk-parallel 算。**

Kimi deep-dive 原话：
> "Early prototypes used a single fused kernel. The token-parallel work in K1 was bottlenecked by the much lower parallelism of the recurrence in K2, leaving a large fraction of the SMs idle. Splitting the pipeline into two kernels yielded at least **15%** end-to-end speedup."

cuLA Hopper fused 现在就是 Kimi 描述的"early single fused kernel"。**Kimi 自己走过这条路，然后通过 K1 抽离拿到 +15%。**

### 其它差异 (非架构)

| | cuLA fused | FlashKDA |
|---|---|---|
| Launch 数 | 4 | 2 |
| CHUNK | 64 | **16** |
| 求逆精度 | tf32 LU | **fp16 + Neumann 级数** |
| state 数据类型 | fp32 register | **bf16 SMEM** |
| h 在哪 | register | SMEM |
| h 物化到 GMEM | 否 | 否 |

注意 **`h 物化` 这一行两边都是"否"** —— FlashKDA 跟 cuLA fused 在"h 不出片"这个核心权衡上一致。

---

## 3. 跟 cuLA SM100 modular (`chunk_kda` Blackwell 路径) 的对比

cuLA SM100 modular 走的是 FLA chunk mode 风格 (h-kernel 串行 + o-kernel chunk-parallel)，跟 FlashKDA 选了**相反**的 trade-off：

```
                cuLA SM100 modular              FlashKDA
切在哪          h ↔ o 之间                     preprocessing ↔ recurrence 之间
h 物化吗        是 (NT × 64KB fp32 全程在 GMEM) 否 (state_acc 在 SMEM bf16)
o 的并行轴      chunk-parallel (NT in grid)    chunk-serial (在 K2 内 NT-loop)
小 B/H 长 T 时  o-kernel grid 撑得起来          K2 grid 还是 N·H, 撑不起来
典型 regime 时  付了 h GMEM 流量代价            省了 h GMEM 流量, 赢
```

**结果:** H20 实测 FlashKDA 快 ≈ 2× over `fla_chunk_kda`。"h 不出片"这条权衡在典型 regime 下赢的更多。

---

## 4a. ★ GH200 3-way baseline (cuLA vs FlashKDA vs FLA Triton) — 2026-05-07

测于 Nautilus GH200 pod，B=1 H=变化 T=8192 (issue #11 worst-case 区域)。详见 `cuLA-profiling/2026-05-07_three_way_baseline_gh200.md`。

| B | T | H | cuLA fused | FlashKDA | FLA Triton | 赢家 |
|---|---|---|-----------:|---------:|-----------:|------|
| 1 |  8192 |  **4** | 1.491 ms | 0.847 ms | **0.761 ms** | **Triton** ★ |
| 1 |  8192 | **16** | 1.571 ms | 0.929 ms | **0.768 ms** | **Triton** ★ |
| 1 |  8192 | 64 | 1.780 ms | **1.239 ms** | 1.868 ms | FlashKDA |
| 1 |  8192 | 96 | 1.944 ms | **1.454 ms** | 2.829 ms | FlashKDA |
| 8 |  2048 | 16 | 0.661 ms | **0.436 ms** | 0.994 ms | FlashKDA |

**核心发现 (在前几轮分析里没明文意识到的)：**

1. **FlashKDA 在 H≤16 上输给 FLA Triton**。Kimi 没测这个 regime（H20 bench 全是 H=64/96），所以他们的 1.85-2.31× 加速宣称在小 B/H 上不成立。

2. **issue #11 worst case 的真实对手是 Triton，不是 FlashKDA**。cuLA = 1.49ms vs Triton = 0.76ms = **1.96× 落后**（之前 PR1 估的 0.46× 是 vs Triton，跟这里一致）。

3. **存在一个"全空白"的优化窗口**：在 B=1 H=4 上击败 Triton。FlashKDA 没做、Triton 自己也没专门做。架构上需要 **FlashKDA-style K1（chunk-parallel preprocessing） + FLA chunk-mode o-kernel（chunk-parallel o）的组合**——neither 现有方案有这个。

4. cuLA Hopper fused 在所有 config 上都垫底——**架构问题，不是低层调优问题**。

## 4b. H20 Benchmarks (FlashKDA vs fla_chunk_kda) — 来自 Kimi BENCHMARK_H20.md

来源: `BENCHMARK_H20.md`, 2026-04-22 generated。
Settings: warmup=30, iters=200, repeats=5。

| 配置 | flash_kda (ms) | fla_chunk_kda (ms) | 加速 |
|---|---|---|---|
| T=8192, H=96, Fixed | 2.62 | 4.84 | 1.85× |
| T=8192, H=96, Varlen [1300,547,2048,963,271,3063] | 2.34 | 4.83 | 2.06× |
| T=8192, H=96, Varlen 1024×8 | 2.04 | 4.67 | 2.29× |
| T=8192, H=64, Fixed | 1.62 | 3.17 | 1.95× |
| T=8192, H=64, Varlen 多段 | 1.71 | 3.26 | 1.91× |
| T=8192, H=64, Varlen 1024×8 | 1.40 | 3.22 | 2.31× |

**注意:**
- 这里的 `fla_chunk_kda` baseline 应该是 FlashKDA 整合**之前**的版本（即 Triton 实现）
- **Kimi 没测 H=4 等小 H 配置** —— 他们的目标 regime 不在小 B/H
- **Kimi 也没测 varlen 边界很碎的情况** (除了 1024×8 这种均匀的)

---

## 5. 一个判别标准: "merge" 在不在

不同优化方案是否需要"affine merge h"是判断它属于哪类的快速标准：

```
                   并行什么          需要 merge       什么时候 merge
─────────────────────────────────────────────────────────────────
cuLA SM90 fused    啥都不并行 (B·H)    否            ─
cuLA SM100         o 并行 (NT in       否            ─
modular            grid), h 串行
FlashKDA          K1 chunk-parallel,  否            ─
                   K2 内全串行
C1 segment-scan   递推本身分段并行     是！          pass 1 后 merge,
                                                    再 pass 2 算 o
"chunk-parallel    递推完全 NT 路并行   是！          每层 scan 后
scan h-kernel"     + Blelloch 树                    都要合并
(谁都没做)
```

**结论:** **merge = 把"敢动 chunk-serial 递推"那个动作的代价**。FlashKDA / cuLA modular / cuLA fused 都不动递推串行性，都不需要 merge。C1 是唯一动了的。

---

## 6. 战略影响：FlashKDA 的存在改变了 issue #11 的 framing

### 6.1 待验证: PR1 baseline 跑的到底是 Triton 还是 FlashKDA

如果 FlashKDA 已经装在测试环境（`pip list | grep flash-kda`），那么 PR1 测出来的 `fla_chunk_kda` 0.46× 实际上是 **0.46 / 1.95 ≈ 0.24×** 相对于 FlashKDA，**不是** 相对于 Triton。这把 issue #11 的目标线整体下移了一截。

**Action:** 在下一次远程 GPU run 时跑一次：
```bash
pip list | grep -i flash
python -c "from fla.ops.kda import chunk_kda; import inspect; print(inspect.getsourcefile(chunk_kda))"
```

### 6.2 maintainer 的预期目标

issue #11 的 maintainer (`icavan`, `yzhangcs`) 是否预期 cuLA Hopper 应该追平/超过 FlashKDA？还是只要追平 `fla_chunk_kda` Triton baseline？

**Action:** 在 issue 上提问 / 看历史讨论 / 看 RFC 走向。

### 6.3 cuLA Hopper 优化方向重新洗牌

| 方案 | 跟 FlashKDA 的关系 | 工程量 | issue #11 适配性 |
|---|---|---|---|
| C1 segment-scan (PR3 已实现) | 在 cuLA fused 架构上加并行；fused 架构本身已被 FlashKDA 证明非最优 | 小 (~200 行 Python) | 局限：天花板被 fused 架构限死 |
| K1+K2 split (FlashKDA 风格) | 追平 FlashKDA 架构层；但 CHUNK=64 不能改 → 仍弱于 FlashKDA 数值优化层 | 中-大 (~1500-3000 行) | 单做：跟 FlashKDA 追平不超越 |
| **K1+K2 split + K2 segment-scan** | **K1 抽出 + K2 内段间并行 = FlashKDA 没做的方向** | 大 | **唯一可能在小 B/H 长 T 上超过 FlashKDA 的路径** |
| Option F (FLA chunk mode 风格) | 走 cuLA SM100 modular 同款架构 | 大 | 已被 FlashKDA H20 数据证明在典型 regime 上输 ~2× |

---

## 7. cuLA Hopper K1+K2 重构方案 (推荐 PR3 候选)

### 7.1 抽到新 K1 的内容

```
当前 cuLA Launch 1-3 (Triton)         → K1 内融合
当前 cuLA Launch 4 内的:
  decay apply                          → K1
  A_qk (intra MMA 分数)                → K1 (变成 Mqk)
  A_kk + L 求逆 (delta-rule)           → K1 (变成 INV)
  W = L·k, U = L·v                     → K1 内或 K2 现场算 (FlashKDA 选择 K2 现场算)

K1 输出 workspace:
  ws_kd, ws_qd, ws_kr ∈ [..., 64, 128] bf16   ← cuLA CHUNK=64, 不是 16
  ws_gt ∈ [..., 128] fp32
  ws_inv, ws_mqk ∈ [..., 64, 64] bf16
```

### 7.2 留在改造后 K2 的内容

```
load h0 → state (寄存器或 SMEM)
for chunk in 0..NT-1:
   load ws_*[chunk] 进 SMEM
   v_new = ws_inv·v - ws_kd·state^T
   o = ws_mqk·v + ws_qd·state
   state *= diag(ws_gt)
   state += ws_kd^T·v_new
store final_state
```

### 7.3 实现选项

**K1 实现路径:**
- 选项 A: CUTLASS C++ 写新 SM90 kernel (最贴 FlashKDA 风格) — ~1000-1500 行
- 选项 B: CuTe DSL 写 (跟 cuLA `chunk_delta_h.py` / `fwd_o.py` 同语言风格) — ~500-800 行
- 选项 C: 移植 SM100 intra kernel (`csrc/kda/sm100/kda_fwd_intra_*`) 到 SM90 — UMMA→WGMMA, 输出格式调整 — ~500-800 行

**K2 重构路径:**
- 改 `csrc/kda/sm90/collective/mainloop_kda_fwd.hpp` 删除 T-loop 内的 intra-MMA + 求逆
- 加 workspace TMA load tile
- ~600-1000 行 refactor

**Python orchestrator:**
- `cula/kda/hopper_fused_fwd.py` 加 workspace 分配 + 两 launch 序列
- ~100 行

### 7.4 预期收益

- **典型 regime (H=64+):** +15–30% (跟 Kimi deep-dive 的 +15% 报告对齐)
- **小 B/H worst case (B=1, H=4):** 更显著 — K1 grid `B·H·NT = 4·128 = 512` blocks 直接喂满 132 SM
- **不会追平 FlashKDA**：CHUNK=64 / tf32 inverse / fp32 state 的劣势仍在

### 7.5 后续 PR 候选

- **PR4 (差异化):** 在 K2 上加 segment-scan，把 K2 的 chunk-serial 瓶颈也击穿 — 这是 FlashKDA 没做的方向，可能在小 B/H 长 T 上超过 FlashKDA。已有的 `feat/issue-11-segment-scan` / `feat/issue-11-cula-emit-transition` 分支的 segment-scan 工作可复用思路（虽然代码可能要重写以适配新的 K2 接口）。
- **PR5 (数值优化):** fp16 inverse + bf16 state — 各自独立的小 PR

---

## 8. 已有的工作和资产

### 已实现并验证 (本地分支保存)
- `feat/issue-11-small-bhs-bench` — PR1 benchmark (1 commit)
- `feat/issue-11-segment-scan` — segment-scan 早期 (3 commits)
- `feat/issue-11-adaptive-num-segments` — 中期 (含 N_seg ∈ {4,8,16,32} dispatch)
- `feat/issue-11-cula-emit-transition` — 最新 (含 emit-transition 准备工作)

### 已有的实测数据
- PR1 baseline: `cuLA-profiling/2026-05-02_fused_fwd_small_sweep.log`
- nsys 时间线: `cuLA-profiling/2026-05-02_fused_fwd_repr3.{nsys-rep,md}`
- KCP associativity bit-exact 验证: `cuLA-profiling/2026-05-03_kcp_associativity_check.{py,log}`
- segment-scan PR2/PR3 实测: 1.39× over baseline on B=1, H=4, T=8192 (已验证)

---

## 9. Open Questions / 下一步

1. **PR1 baseline 的真实对手是谁** — Triton 还是 FlashKDA？(待远程 GPU 验证)
2. **maintainer 对 #11 的目标线** — 追 Triton 还是追 FlashKDA？(待 issue 沟通)
3. **K1 该用什么实现**（CUTLASS C++ vs CuTe DSL vs 移植 SM100）— 选定后才能给准确工程量
4. **CHUNK=64 在 K1 内的代价** — FlashKDA K1 的 16×16 inverse + Neumann 在 64×64 上还可行吗？还是要回退 LU？
5. **K1+K2 合 segment-scan 的工程顺序** — 先 K1+K2 split 再 segment-scan，还是直接组合做？

---

## 10. 参考资料

- FlashKDA repo: `https://github.com/MoonshotAI/FlashKDA`
- Deep-dive blog (in repo): `docs/20260420-flashkda-v1-deep-dive.md`
- H20 benchmarks: `BENCHMARK_H20.md`
- FLA integration PR: `https://github.com/fla-org/flash-linear-attention/pull/852`
- 相关 cuLA 文件:
  - `csrc/kda/sm90/collective/mainloop_kda_fwd.hpp` — 当前 fused mainloop (要改的)
  - `csrc/kda/sm90/kernel/kernel_kda_fwd.hpp` — warp specialization 框架
  - `cula/kda/hopper_fused_fwd.py` — Python orchestrator
  - `csrc/kda/sm100/kda_fwd_intra_*` — SM100 intra kernel (可参考/移植)
  - `cula/ops/chunk_delta_h.py` — SM100 modular h-kernel (CuTe DSL)
  - `cula/ops/fwd_o.py` — SM100 modular o-kernel (CuTe DSL)

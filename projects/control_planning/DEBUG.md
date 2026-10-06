# 控制与规划教学代码库调试与测试教程（DEBUG）

> 零基础基准撰写。姊妹文档：[TUTORIAL.md](TUTORIAL.md)（代码导读）｜
> [METRICS.md](METRICS.md)（指标健康值——本文变量表的健康值与它保持一致）｜
> [.vscode/SETUP.md](../../.vscode/SETUP.md)（VS Code 配置总教程）。
> 规范出处：[CONSTRAINTS.md §4.5](../../CONSTRAINTS.md)。

---

## 1. 环境与两种测试

**解释器**：本项目纯 numpy，两套解释器均验证通过（53/53）——

| 解释器 | numpy | 全套件耗时（实测，逐文件直跑） |
|--------|-------|--------------------|
| conda `llm_env`（`~/anaconda3/envs/llm_env/bin/python`） | 2.2.6 | ~33 s |
| 系统 `python3` | 1.26.4 | ~39 s |

任选其一即可；VS Code 默认指向 `llm_env`（见 SETUP.md §2）。最慢的是
`tests/test_ppo_lite.py`（~20 s：训练收敛 + 策略梯度中心差分对照各训练/采样一次），
其余文件均 ≤ 6 s；全套件远低于 90 s 预算。逐文件实测（llm_env）：

```bash
cd /home/dzxu/RoboTwinTutorial/projects/control_planning

python tests/test_metrics.py    #  7/7 tests passed.   ~0.1 s  （统一指标）
python tests/test_ilqr.py       #  7/7 tests passed.   ~0.2 s  （iLQR + Riccati 交叉验证）
python tests/test_osc_arm.py    #  7/7 tests passed.   ~0.3 s  （2R 臂 + 阻抗控制）
python tests/test_cbf.py        #  6/6 tests passed.   ~1.0 s  （CBF 安全滤波）
python tests/test_mpc_cem.py    #  6/6 tests passed.   ~4.4 s  （CEM-MPC）
python tests/test_rrt.py        #  7/7 tests passed.   ~5.6 s  （RRT / RRT*）
python tests/test_mpnet.py      #  7/7 tests passed.   ~1.4 s  （MPNet 采样偏置；系统 python3 ~4.2 s）
python tests/test_ppo_lite.py   #  6/6 tests passed.  ~19.9 s  （微型 PPO）
```

### 1.1 单点测试（改了一个模块 → 用它）

**方式 A：直跑过滤**（每个测试文件的 `__main__` 块都内置，推荐）：

```bash
cd /home/dzxu/RoboTwinTutorial/projects/control_planning   # 必须在本目录（原因见 §1.4）

python tests/test_cbf.py projection   # 只跑名字含 "projection" 的测试
# 预期输出：
# PASS test_closed_form_projection_is_minimal_modification
# 1/1 tests passed.
python tests/test_ilqr.py riccati     # 前缀更宽：两个 riccati 测试都命中
# PASS test_lqr_special_case_matches_riccati
# PASS test_riccati_fixed_point_matches_finite_horizon
# 2/2 tests passed.
python tests/test_cbf.py zzz          # 子串无任何匹配：列出全部可用测试名再退出（exit 1）
python tests/test_rrt.py              # 不带参数 = 该文件全部测试
```

**方式 B：pytest 过滤**：

```bash
cd /home/dzxu/RoboTwinTutorial/projects/control_planning
pytest tests/test_ppo_lite.py::test_gae_matches_manual_kstep_weighting -v  # 精确到函数
pytest tests/test_rrt.py -k rrt_star -v                                    # -k 子串等价于方式 A
```

### 1.2 全局测试（提交前 / 合并前 → 用它）

```bash
# 仓库根目录：
cd /home/dzxu/RoboTwinTutorial
python -m pytest projects/control_planning/tests -v     # 预期：53 passed
# 或在 projects/control_planning 下逐文件直跑（不依赖 pytest）：
cd projects/control_planning
for f in tests/test_*.py; do python3 "$f"; done
```

### 1.3 何时用哪个（速查）

| 场景 | 用法 |
|------|------|
| 改了某个模块（如 `mpc_cem.py`） | 单点：`python tests/test_mpc_cem.py`；`mpc_cem` 复用 `ilqr` 的动力学与 `rrt` 的走廊场景，故再跑 `tests/test_ilqr.py` |
| 改了 `metrics.py`（被全部模块的测试断言取用） | 先 `tests/test_metrics.py`，再全局 |
| 提交前 | 全局 pytest + VS Code Testing 侧栏全绿 |
| 怀疑 numpy 版本差异 | 两个解释器各跑一遍全局；两套解释器的 53 个测试结果与打印数字当前逐位一致（含 iLQR 的 11 轮收敛与 mpnet 的 86.0/109.4 迭代数）。历史上 iLQR 的收敛轮数出现过跨解释器 ±3 轮的浮动（浮点求和顺序差异，最终 J 一致）——若再现此类现象，断言不要绑迭代数，绑单调性与收敛标志 |

### 1.4 常见启动失败

- `ModuleNotFoundError: No module named 'rrt'`（或 ilqr / cbf …）——工作目录不在
  `projects/control_planning`，或用 `python3 /path/to/script.py` 从别处执行
  （`sys.path[0]` 是脚本目录而非 cwd）。tests/ 下的文件自带
  `sys.path.insert`，照抄即可。
- VS Code 里 import 标红线 —— 从仓库根打开窗口后 Reload（SETUP.md §5）。

---

## 2. VS Code 断点调试

`F5` → 顶部下拉选配置（control_planning 的配置注册于
[.vscode/launch.json](../../.vscode/launch.json)，条目见 §5.6 的注册信息）：

| 配置名（建议） | 用途 | 用法 |
|------|------|------|
| `control_planning: 核心入口（demo.py）` | 调试模式跑四段冒烟演示，`cwd` = projects/control_planning | 断点打在任意被调代码处即停 |
| `control_planning: 全局测试（pytest 全套件）` | 以调试模式跑整套 pytest，`cwd` = 仓库根 | 断点打在任意被测代码处即停 |
| `control_planning: 单点测试（当前文件 + 测试名过滤）` | 调试**当前打开的** tests/test_*.py | F5 后弹输入框填测试名子串（如 `projection`、`riccati`；留空 = 全部） |
| `control_planning: 单点示例｜CBF 投影（test_cbf.py projection）` | 现成的单点示例 | 直接 F5，无需输入 |

`justMyCode` 默认 `true`（只在项目代码内停）；要单步进 numpy 内部（如
`np.linalg.cholesky`）时在 launch.json 中改为 `false`。终端里等价的直跑调试：
`python -m debugpy --wait-for-client tests/test_cbf.py projection`。

### 推荐断点位置（函数级，行号为撰写时的参考值）

| 断点位置 | 在这里看什么 |
|----------|--------------|
| `rrt.py::_plan_core`（~L364，steer 内联行之后） | 每轮的 `q_rand / q_new / r_n`；RRT* 分支里 `cand` 是否为空（半径收缩边界，变量表 R-2） |
| `rrt.py::segments_free_batch`（~L164 `pts = ...`） | 批量检查点阵 `(k, c, 2)` 与 `free` 掩码——穿隧争议时先看这里 |
| `ilqr.py::ilqr_solve` 反向扫（~L421 `q_uu_reg = ...`） | `q_uu` 的最小特征值与 `mu` 的 ×10 触发（变量表 I-2） |
| `ilqr.py::ilqr_solve` 接受分支（~L468 `ratio = ...`） | `decrease / expected`（增益比）、`alpha` 的减半轨迹（变量表 I-3/I-4） |
| `mpc_cem.py::cem_plan` 精英更新（~L236 `mean = elites.mean(axis=0)`） | 精英均值/方差、`sigma` 触底 `sigma_min`（变量表 M-1/M-2） |
| `cbf.py::cbf_filter` POCS 循环（~L213 `for i in range(A.shape[0])`） | 每行的违反量、投影步长、收敛轮数（变量表 C-3） |
| `osc_arm.py::dls_ik`（~L244 `dq = J.T @ ...`） | `J @ J.T + λ²I` 的条件数、步长范数（变量表 O-2） |
| `ppo_lite.py::ppo_update` 掩码（~L452 `masked = ...`） | 截断掩码的命中比例、`coef` 置零样本（变量表 P-3） |
| `ppo_lite.py::compute_gae`（~L296 `adv_next = ...`） | `delta`（TD 残差）与反向累积的 `adv_next`（变量表 P-6） |
| `mpnet_lite.py::mse_loss_grad` 反向（~L454 `g = diff / ...`） | 残差 `diff` 与四个梯度块——中心差分对照失效时先看这里（变量表 N-1） |
| `mpnet_lite.py::plan_rrt_biased` 采样分支（~L649） | 偏置判定、条件节点（goal-nearest）与建议点抖动后的 `q_rand`（变量表 N-2/N-3） |

---

## 3. 重点观察变量表（核心章节）

约定：**形状**为断点处的 numpy 形状；**健康值**为基准场景（测试/demo 定种子）
实测；**异常信号**出现即有 bug 或配置错误。分组编号
R(RRT)/I(iLQR)/M(MPC-CEM)/C(CBF)/O(OSC 臂)/P(PPO)/N(MPNet)。

### R：rrt（断点 `rrt.py`）

| # | 断点 / 来源 | 变量 | 含义（形状） | 健康值 | 异常信号 |
|---|-------------|------|--------------|--------|----------|
| R-1 | `gamma_constant` 返回值 | `gamma` | Theorem 38 下界的 2 倍放大常数（标量） | 基准走廊场景 **17.944**（μ(C_free) = 100 − 5π） | 远小于 17.9 → bounds/障碍传入错误（γ 随自由面积缩小） |
| R-2 | `_plan_core` RRT* 分支 | `cand` | 收缩半径内候选父下标 (k,) | n ≤ 3000 时非空（含最近邻）；实测 n=3000 时 r_n ≈ 0.57 | 频繁为空 → (3.7) 的 min{·,η} 退化参数被改坏；空时走最近邻回退分支属正常 |
| R-3 | `_plan_core` 目标判定 | `costs_buf[goal_idx]` | 目标节点代价 (3.5)（标量，m） | RRT* 3000 迭代 seed0 = **10.988**、6000 迭代 = **10.741**；同 seed RRT = 12.500 | RRT* > RRT → 重布线的子树代价传播被跳过（见 §4 案例 1） |
| R-4 | `_plan_core` 返回 | `res.n_iters` | 首解到达轮数（int） | 走廊 seed0 RRT = **70**；空图 bias=0.0 → 170、bias=0.3 → 49 | 不随 goal_bias 改善 → 采样消耗顺序变了（RRT 与 RRT* 的对照随即失效） |

### I：ilqr（断点 `ilqr.py::ilqr_solve`）

| # | 断点 / 来源 | 变量 | 含义（形状） | 健康值 | 异常信号 |
|---|-------------|------|--------------|--------|----------|
| I-1 | 每次迭代 | `costs` | 逐迭代总代价 (iters+1,) | 到达任务 30600 → **6.331**（约 10 轮收敛）；单调不升 | 回升 → 接受判据坏；卡在初值 → 雅可比/值函数更新错 |
| I-2 | 反向扫 | `mu` / `mu_hist` | 正则化系数 (4.14)（标量，list） | 良性问题从 1.0 持续 ×0.5 → 1e-9 下界；实测全程不升 | 单调涨到 mu_max → Q_uu 无下降方向（查代价 Hessian 符号约定） |
| I-3 | 接受分支 | `ratio`（入 `ratio_hist`） | 实际/预期下降比（标量） | 初期 ~0.29–0.45（μ 阻尼的保守性），收敛处 → **1.0±0.1**（实测 0.446→1.021） | 后期仍 ≪1 → ΔV 用了未阻尼 Q_uu（Tassa 口径是阻尼后的）；>1.3 → 线搜索失效 |
| I-4 | 前向回滚 | `alpha` | 线搜索步长（标量，1, 1/2, …） | 二次问题上恒为 1 | 连续砍到 1/64 仍拒 → 当前线性化点失效（检查 `ReachCost` 导数） |

### M：mpc_cem（断点 `mpc_cem.py`）

| # | 断点 / 来源 | 变量 | 含义（形状） | 健康值 | 异常信号 |
|---|-------------|------|--------------|--------|----------|
| M-1 | `cem_plan` 逐轮 | `elite_cost_mean` | 精英集平均代价 (n_iters,) | 实测（原点出发 seed0）221.8 → 159.9 逐轮下降 | 不降 → n_elite 过大（接近 n_samples）或 sigma_min 卡死 |
| M-2 | `cem_plan` 逐轮 | `sigma_trace` | 精英方差迹的逐时刻均值 (n_iters,) | 实测 1.90 → 0.76 收缩；趋势单调 | 反复弹回 sigma0 → 采样箱 `u_limit` 截断了精英（半数样本贴壁） |
| M-3 | `cem_mpc_rollout` 逐周期 | `plans` | 各周期 `plan.best_cost`（list[float]） | 起步 ~O(10²)，接近目标后 → O(1)–O(10) | 中途跳升 → 执行噪声把状态推进罚区（查 `min_dist`） |
| M-4 | 仿真返回 | `res.min_dist` | 全程最小障碍净距（标量，m） | **+0.58**（seeds 3/7/11/21 实测 0.575–0.585） | < 0 = 碰撞：先查 `w_obs/r_safe` 是否被调小（默认 2000/0.6） |
| M-5 | 仿真返回 | `res.final_pos_err` | 终端位置误差（标量，m） | 实测 0.030–0.131（< 到达容差 0.35） | > 0.35 → 温启动链断了（`warm_mean` 未平移）或 horizon 过短 |

### C：cbf（断点 `cbf.py`）

| # | 断点 / 来源 | 变量 | 含义（形状） | 健康值 | 异常信号 |
|---|-------------|------|--------------|--------|----------|
| C-1 | `simulate_filtered`（filtered=True） | `min_h` | 全程最小屏障值（标量，m） | 直冲场景实测 **−5.4e-16**（门限 ≥ −1e-6） | 持续负且扩大 → (7.9) 约束行拼装错（A/b 符号）或 POCS 未收敛 |
| C-2 | `cbf_filter` 返回 `info["viol_after"]` | 投影残余 | POCS 后最大约束违反（标量） | 实测 2.2e-16（多约束同时活跃 ~2e-9） | > 1e-6 → `n_proj_iters` 不够或 `‖L_g h‖ = 0`（相对阶失效，见 §4 案例 3） |
| C-3 | POCS 循环 | `viol`（逐行） | 当前违反量 (k,) | 首轮后减半以上，3 轮内触零 | 振荡不收敛 → 行冲突（可行集为空）：约束互相矛盾，查 `alpha/kappa` |
| C-4 | `cbf_filter` 返回 `info["du_norm"]` | 修改量范数（标量） | 安全时 = 0（最小侵入）；活跃时 = 到半空间的距离 | 深违反场景实测 9.75（u_des=[2,0]、h=−0.2） | 安全时 du_norm > 0 → 违反判定（`A @ u_des + b`）符号错 |

### O：osc_arm（断点 `osc_arm.py`）

| # | 断点 / 来源 | 变量 | 含义（形状） | 健康值 | 异常信号 |
|---|-------------|------|--------------|--------|----------|
| O-1 | `dls_ik` 返回 | `err` / `err_hist` | 终端位置误差与逐迭代历史（m） | 实测 **2.8e-10 @ 6 迭代**（门限 1e-4） | 停在 0.1 量级不降 → 目标不可达或 λ 过大（阻尼吃掉步长） |
| O-2 | `dls_ik` 步长行 | `J @ J.T` 条件数 | (2,2) 的 cond | 良态位形 ~33（q=(0.5,0.8) 实测 32.96）；奇异伸直位形 **inf** | inf 时 DLS 仍应步长有界——若 NaN → `damping=0` 或除零保护被删（§4 案例 3） |
| O-3 | `simulate_impedance` 返回 | `x_traj[-1]` | 末端稳态位置 (2,)（m） | 0.06/−0.08 rad 偏移释放 4 s 后误差 **3.96e-07 m** | 稳态偏差 ~K⁻¹F_ext 量级且不消 → 重力补偿没开（`gravity_comp=True`） |
| O-4 | `mass_matrix` | `eigvalsh(M)` | 惯性阵特征值 (2,) | 全场景 > 0（q=(0.5,0.8) 实测 [0.090, 2.099]） | 出现非正 → 质量参数为负或闭式下标错 |

### P：ppo_lite（断点 `ppo_lite.py`）

| # | 断点 / 来源 | 变量 | 含义（形状） | 健康值 | 异常信号 |
|---|-------------|------|--------------|--------|----------|
| P-1 | `train` 返回 `history` | 逐轮平均回报 (n_iters,) | 训练健康度主仪表 | seed0 实测 **−39.75 → −9.53**（幅度 4.2×；手工 PD 律 −8.9 为天花板参照） | 平台高于 −15 → 查学习率/epochs；发散 → lr 过大 |
| P-2 | `ppo_update` 的 `kl_hist` | approx_kl 代理（逐 epoch） | 新旧行为 KL（标量） | 单轮 0.01–0.08；`target_kl=0.05` 触发早停（实测首轮 0.072 即停） | 单轮 > 0.5 → 学习率过大（数据被"走飞"） |
| P-3 | `ppo_update` 掩码行 | `masked` / `info["clip_frac"]` | 截断掩码 (S,) / 越界占比 | 更新后占比应随步长增大：lr=0.05 实测 0.52、lr=0.3 实测 **0.93** | 恒 0 → 优势全零（GAE/价值坏）；恒 1 → lr 失控 |
| P-4 | 任意次更新后 | `params.log_std` | 探索标准差 (2,)（σ = exp(log_std)） | `lr_log_std=0`（train 默认）下恒 0.5；放开学习则单调漂移（实测 0.5 → 0.70，无收益） | 不冻 σ 还想收敛 → 方差噪声支配 log_std 梯度，属已知现象非 bug |
| P-5 | `ppo_update` 的 `vf_hist` | 价值损失 (逐 epoch) | 0.5·(V−ret)² 均值 | 训练 20 轮后 ~0.34 → 收敛 ~**0.01** | 不降 → 特征/目标错（查 `value_features` 与 GAE `returns`） |
| P-6 | `test_policy_gradient_matches_central_difference` 打印 | FD/解析方向导数比 | 公共随机数中心差分 vs (8.7) 估计 | 实测 **0.68 / 0.84**（8k 条轨迹的采样噪声 ~15%；符号一致） | 比值差整数量级（如 1/T 倍）→ 归一化口径错（见 §4 案例 2） |

### N：mpnet_lite（断点 `mpnet_lite.py`）

| # | 断点 / 来源 | 变量 | 含义（形状） | 健康值 | 异常信号 |
|---|-------------|------|--------------|--------|----------|
| N-1 | `train_mpnet` 返回 `history` | 蒸馏损失曲线 (9.2)（epochs+1 = 601,） | 训练健康度主仪表 | seed0 实测 **2.6111 → 0.1082**（40 环境 × 600 轮，逐位双环境一致） | 不降 → 数据管线/反向传错（先跑 `tests/test_mpnet.py gradient` 的中心差分对照）；回升 → lr 过大 |
| N-2 | `test_proposals_free_rate_beats_uniform` 打印 | 建议点自由率 vs 均匀基线 | 网络避障方向质量（标量对） | **0.938 vs 0.868**（gap +0.070；4 个未见环境、77 个建议点） | gap ≤ 0 → 推理侧 `k` 与训练不一致，或条件点分布/编码归一化被改 |
| N-3 | `test_biased_rrt_needs_fewer_iters` 打印 | 偏置/均匀平均迭代数（标量对） | 采样偏置的实际收益 | **86.0 vs 109.4（ratio 0.786）**，逐 (环境, seed) 配对 8/8 偏置臂不劣 | ratio ≥ 1 → 建议点撞墙率升高（查 `jitter`/`k`/专家数据量）；两臂逐位相等 → 偏置分支未生效（误传 `p_bias=0`） |

---

## 4. 常见调试场景（症状 → 断点 → 看什么 → 结论）

### 案例 1：RRT* 的代价反而比 RRT 高

- **症状**：`test_rrt_star_cost_not_worse_than_rrt_same_seed` 失败；demo ① 打印
  RRT* cost > RRT cost。
- **断点**：`rrt.py::_plan_core` 重布线后的代价传播段（`stack = [i_v]` 的 while 循环）。
- **看什么**：改接父节点（`parent[i_v] = i_new`）之后，`costs_buf[i_v]` 的整棵子树
  是否被逐级更新（R-3）。只更新 `i_v` 本身、不传播后代，是教科书实现最常见
  的简化 bug——后续选父/重布线会基于过期的代价比较，(3.6) 的单调性随之破坏。
- **结论**：代价必须按 (3.5) 的递归定义沿 children 表逐级重算；树的
  `parent/costs` 任何改动都应触发传播。

### 案例 2：策略梯度与中心差分差一个整数倍

- **症状**：`test_policy_gradient_matches_central_difference` 的比值不在
  [0.5, 1.5]（本库的历史 bug 实测比值 ≈ 1/T = 1/10~1/20）。
- **断点**：`ppo_lite.py::policy_gradient_estimate` 的归一化行（`grad_b /= n_traj`）。
- **看什么**：除数是**回合数**还是**总样本数**。(8.7) 的期望是对轨迹取的
  （Σ_τ Σ_t (…) / n_τ），除以 n_samples 会引入整整 T = n_steps 倍的缩小；
  PPO 代理目标（`ppo_update`）是逐样本均值口径，两者**本来就不同**，不要互相"对齐"。
- **结论**：对照 FD 时确认被测函数是 (8.7) 口径；另注意 FD 必须在远离动作
  截断 u_limit 的参数点做（截断使 J(θ) 出现拐点，差分给出无意义的
  大数——本测试特意把策略缩放 0.5 倍就是这个原因）。

### 案例 3：CBF 滤波在障碍附近"失效"（h 仍然变负）

- **症状**：`test_filtered_keeps_h_above_tolerance` 失败，min_h 明显为负；
  或 `viol_after` 不收敛。
- **断点**：`cbf.py::barrier_affine` 与 `cbf_filter` 的 POCS 循环。
- **看什么**：① `‖L_g h‖`（`info["A"]` 的行范数）是否接近 0——以位置为
  h、加速度为输入时 ḣ 不含 u（相对阶 2，教程 07.2 ⑤ 的失效边界），任何
  滤波器都无能为力；本库的 h 已按综述 §IV 口径提升到速度级（κ vᵀn 项），
  若删掉该项即复现失效。② `viol` 是否振荡——两行约束方向冲突时可行集
  可能为空（障碍夹缝小于可刹车距离），此时应放松 `alpha`（更早减速）。
- **结论**：先验证 h 的相对阶，再查约束行拼装 (7.7)，最后才是迭代参数。

### 案例 4：iLQR 在两个解释器下迭代数（曾经）不同

- **症状**：系统 python3 与 llm_env 打印的收敛轮数差 2–3 轮（本库历史上
  实测过 11 vs 14；当前代码两环境逐位一致，均为 11 轮）。
- **断点**：`ilqr.py::ilqr_solve` 的收敛判定行（`rel < tol`）。
- **看什么**：`costs` 末几位——两个 numpy 版本的浮点求和顺序差异使
  相对下降在 1e-6 阈值附近的穿越轮次不同；最终 J 与轨迹一致（demo 打印
  J 收敛值 6.331 两边相同）。
- **结论**：这是浮点非确定性而非实现错误；对迭代数敏感的断言不要写进
  测试（本项目只断言单调性与收敛标志）。

---

## 5. 测试怎么写

新测试放在 `tests/test_<模块名>.py`，遵循本套件既有惯例：

1. **命名 `test_<行为>`**——名字说清"验证什么行为"，因为单点过滤按子串匹配
   （§1.1），名字是过滤键。范例：`test_rrt_star_cost_not_worse_than_rrt_same_seed`、
   `test_filtered_keeps_h_above_tolerance`。
2. **deterministic seed**——一切随机性走 `np.random.default_rng(seed)`（场景、
   噪声、采样各给独立种子），使打印的数字与断言阈值可精确复现；禁止
   `np.random.seed`/全局随机态。
3. **断言阈值旁注明理由**——照套件惯例（"实测依据：……"注释）写清期望值
   的量级来源（如 PPO 回报比的 1.5× 阈值、CBF 的 −1e-6 离散容差），让后来者
   能区分"阈值松了"与"实现错了"。
4. **文件尾部带单点过滤主入口**——照抄 `tests/test_rrt.py` 的 `__main__` 块：

   ```python
   if __name__ == "__main__":
       # 单点测试：python tests/test_X.py <测试名子串>；不带参数 = 全部测试。
       pat = sys.argv[1] if len(sys.argv) > 1 else ""
       fns = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and pat in k]
       if not fns:
           print(f"没有匹配 '{pat}' 的测试；可用测试：")
           for k in sorted(globals()):
               if k.startswith("test_"):
                   print(" ", k)
           sys.exit(1)
       failed = 0
       for fn in fns:
           try:
               fn()
               print(f"PASS {fn.__name__}")
           except Exception:
               failed += 1
               print(f"FAIL {fn.__name__}")
               traceback.print_exc()
       print(f"\n{len(fns) - failed}/{len(fns)} tests passed.")
       sys.exit(1 if failed else 0)
   ```

   （文件头部还需 `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`
   使从任意 cwd 直跑可用，见 §1.4。）

5. **运行时长预算 < 15 s / 文件**——当前最慢的是 `tests/test_ppo_lite.py`
   （~20 s，其中训练与 FD 对照各占一半）；采样类测试通过减少回合数而非删
   断言来控制时长。指标断言从 `metrics.py` 取实现（[METRICS.md](METRICS.md)），
   勿在测试里重写公式。
6. **写完自查**：`python tests/test_<新文件>.py` 全绿 + 该模块断点处的观察量
   （§3 变量表）落在健康值内 + 若新增了被监视变量，同步更新本文件的变量表
   （CONSTRAINTS §4.5 质量要求）。

### 5.6 VS Code 注册信息（供 launch.json 维护者）

- **核心入口**：`projects/control_planning/demo.py`，建议 `cwd` =
  `${workspaceFolder}/projects/control_planning`，解释器 = conda `llm_env`。
- **测试目录**：`projects/control_planning/tests/`（pytest 发现路径加
  `projects/control_planning`，或以仓库根为 cwd 用
  `python -m pytest projects/control_planning/tests -v`）。
- **单点过滤真实命中示例**（已实测）：
  `python tests/test_cbf.py projection` → 1/1
  （`test_closed_form_projection_is_minimal_modification`）；
  `python tests/test_ilqr.py riccati` → 2/2（两个 riccati 测试）；
  `python tests/test_ppo_lite.py gae` → 1/1（`test_gae_matches_manual_kstep_weighting`）。

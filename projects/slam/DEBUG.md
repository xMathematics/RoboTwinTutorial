# SLAM 教学代码库调试与测试教程（DEBUG）

> 零基础基准撰写。姊妹文档：[TUTORIAL.md](TUTORIAL.md)（代码导读）｜
> [METRICS.md](METRICS.md)（指标健康值——本文变量表的健康值与它保持一致）｜
> [.vscode/SETUP.md](../../.vscode/SETUP.md)（VS Code 配置总教程）。
> 规范出处：[CONSTRAINTS.md §4.5](../../CONSTRAINTS.md)。

---

## 1. 环境与两种测试

**解释器**：本项目纯 numpy，两套解释器均验证通过（80/80）——

| 解释器 | numpy | 全套件耗时（实测） |
|--------|-------|--------------------|
| conda `llm_env`（`~/anaconda3/envs/llm_env/bin/python`） | 2.2.6 | ~9.5 s |
| 系统 `python3` | 1.26.4 | ~16.5 s |

任选其一即可；VS Code 默认指向 `llm_env`（见 SETUP.md §2）。唯一与版本相关的
容差注记见 §1.3 的"droidlite 交叉验证"。

### 1.1 单点测试（改了一个模块 → 用它）

**方式 A：直跑过滤**（每个测试文件的 `__main__` 块都内置，推荐）：

```bash
cd /home/dzxu/RoboTwinTutorial/projects/slam     # 必须在本目录（原因见 §1.4）

python tests/test_fastslam.py gate       # 只跑名字含 "gate" 的测试
# 预期输出：
# PASS test_mahalanobis_gate_rejects_outlier
# PASS test_nearest_neighbor_gating_beats_dead_reckoning
# PASS test_range_bearing_jacobian_matches_finite_differences
# 3/3 tests passed.
python tests/test_fastslam.py gat        # 子串写错时：列出全部可用测试名再退出（exit 1）
python tests/test_vins.py                # 不带参数 = 该文件全部测试
```

**方式 B：pytest 过滤**：

```bash
cd /home/dzxu/RoboTwinTutorial/projects/slam
pytest tests/test_vins.py::test_vi_bundle_recovers_metric_scale -v   # 精确到函数
pytest tests/test_fastslam.py -k gate -v                             # -k 子串等价于方式 A
```

### 1.2 全局测试（提交前 / 合并前 → 用它）

```bash
# 仓库根目录：
cd /home/dzxu/RoboTwinTutorial
python -m pytest projects/slam/tests -v      # 预期：80 passed
# 或在 projects/slam 下逐文件直跑（不依赖 pytest）：
cd projects/slam
for f in tests/test_*.py; do python3 "$f"; done
```

### 1.3 何时用哪个（速查）

| 场景 | 用法 |
|------|------|
| 改了某个模块（如 `photoba/photoba.py`） | 单点：`python tests/test_photoba.py`；再跑强依赖它的 `tests/test_droidlite.py`（复用 direct 场景合成）与 `tests/test_direct.py` |
| 改了 `core/`（被 8 个模块复用） | 先 `tests/test_core_slam.py`，再全局 |
| 提交前 | 全局 pytest + VS Code Testing 侧栏全绿 |
| 怀疑 numpy 版本差异 | 两个解释器各跑一遍全局；唯一已知敏感点：`tests/test_droidlite.py` 的 Schur/稠密交叉验证容差取 1e-5（numpy 2.x 与 1.26 的线性代数差 ~1e-6，实测 2.17e-6；真实实现错误通常 > 1e-2，阈值仍能区分） |

### 1.4 常见启动失败

- `ModuleNotFoundError: No module named 'core'` —— 工作目录不在 `projects/slam`，
  或用 `python3 /path/to/script.py` 从别处执行（`sys.path[0]` 是脚本目录而非
  cwd）。tests/ 下的文件自带 `sys.path.insert`，照抄即可。
- VS Code 里 import 标红线 —— 从仓库根打开窗口后 Reload（SETUP.md §5）。

---

## 2. VS Code 断点调试

`F5` → 顶部下拉选配置（4 条 SLAM 配置定义于
[.vscode/launch.json](../../.vscode/launch.json)）：

| 配置名（与 launch.json 逐字一致） | 用途 | 用法 |
|------|------|------|
| `SLAM: 全局测试（pytest 全套件）` | 以调试模式跑整套 pytest，`cwd` = 仓库根 | 断点打在任意被测代码处即停 |
| `SLAM: 单点测试（当前文件 + 测试名过滤）` | 调试**当前打开的** tests/test_*.py | F5 后弹输入框填测试名子串（如 `hartley`、`gate`；留空 = 全部） |
| `SLAM: 单点示例｜FastSLAM 门控关联（test_fastslam.py nn）` | 现成的单点示例：fastslam 最近邻门控端到端 | 直接 F5，无需输入 |
| `SLAM: VI 尺度恢复测试（test_vins.py）` | 现成的单点示例：vins 全文件（尺度恢复 + 纯视觉对照） | 直接 F5 |

`justMyCode` 默认 `true`（只在项目代码内停）；要单步进 numpy 内部（如
`np.linalg.svd`）时在 launch.json 中改为 `false`。终端里等价的直跑调试：
`python -m debugpy --wait-for-client tests/test_fastslam.py gate`。

### 推荐断点位置（函数级，行号为撰写时的参考值）

| 断点位置 | 在这里看什么 |
|----------|--------------|
| `core/lie.py::so3_exp`（~L84，`theta = norm(phi)` 一行） | 小角度分支 `< _SMALL` 是否被触发；三项闭式的 `sin(theta)/theta` 系数 |
| `core/solver.py::gauss_newton`（~L119 接受/拒绝分支） | `lam` 的 ×10 / ÷10 调度、`improvement` 符号（变量表 C-1/C-2） |
| `fastslam/fastslam.py::FastSLAM2D._update_nearest`（~L643 `d2 = ...`） | 每粒子对每路标的平方马氏距离矩阵、`j_star`、`gated_in`（变量表 F-1～F-3） |
| `fastslam/fastslam.py::FastSLAM2D._ekf_update_slots`（~L694 `gain = ...`） | `nu`（新息）、`gain`（卡尔曼增益）、`log_weights` 的衰减量 |
| `preint/preint.py::Preintegration._propagate_covariance`（~L285 `cov = A @ ...`） | 9x9 递推一步的 A/B 块与 `self.cov` 对角增长 |
| `preint/preint.py::Preintegration.correct`（~L326） | `j_bias @ db` 修正量级应为 O(δb²)（变量表 P-3） |
| `vins/vins.py::ImuFactor.residual`（~L187 `return ...`） | 白化 (15,) 残差逐分量；范数应 ~χ₁₅（变量表 V-1） |
| `vins/vins.py::ViBundle.solve`（~L498 接受分支） | `cost` 单调下降、`lam` 调度、`dx` 范数（变量表 V-2/V-3） |
| `photoba/photoba.py::PhotometricBA.solve`（~L612–648，线搜索与 λ 更新） | `alpha`、`ratio`（增益比）、`lam` 的 Nielsen 更新、`_rho_lo/_rho_hi` 箱裁剪（变量表 B-1～B-4） |
| `loam2d/loam2d.py::_solve_point_to_line`（~L455 `cond = np.linalg.cond(...)`） | 条件数与 `healthy` 判定（变量表 L-1/L-2） |
| `droidlite/droidlite.py::DenseBA._schur_step`（~L443 `C = np.einsum(...)`） | 深度对角块 `C`、消元后的 `dxi/drho`（变量表 D-3） |
| `droidlite/droidlite.py::DenseBA.solve`（~L531 逐轮循环尾） | `pose_rmse` 单调性（变量表 D-1） |

---

## 3. 重点观察变量表（核心章节）

约定：**形状**为断点处的 numpy 形状；**健康值**为基准场景（测试定种子）实测；
**异常信号**出现即有 bug 或配置错误。分组编号 F(astslam)/P(reint)/V(ins)/
B(A 光度)/L(oam)/D(roid)/E(pipolar)/C(ore)。

### F：fastslam（断点 `fastslam/fastslam.py`）

| # | 断点 / 来源 | 变量 | 含义（形状） | 健康值 | 异常信号 |
|---|-------------|------|--------------|--------|----------|
| F-1 | `_update_nearest` ~L643 | `d2` | 每粒子到每已观测路标的平方马氏距离 (N, K) | 合法重观测 ≤ ~20（卡方(2)+位姿采样误差）；不同路标 ≥ ~44（间距/σ)² | 大量落在 20–44 灰区 → 噪声参数与仿真器不一致 |
| F-2 | `_update_nearest` ~L646–651 | `j_star` / `d_min` / `gated_in` | ML 关联槽位 (N,) / 最小 d² (N,) / 门控掩码 (N,) | `gated_in.mean()` 随步数上升至接近 1 | 全 False → 全部初始化新槽（门限过紧或位姿发散） |
| F-3 | 构造参数 | `mahalanobis_gate`（`GATE_DEFAULT = 30.0`） | d² 门限（标量） | 30（双尺度：合法 ~20 与异路标 ~44 之间） | 误用 `GATE_CHI2_2DOF_95 = 5.991` → 每次拒绝复制一个路标（论文式 (12) 后自述"门限需仔细斟酌"） |
| F-4 | 任意步后 | `slam.weights()` 的 max | 归一化权重最大值（标量） | ≤ 0.5；重采样后 = 1/N = 0.02 | max > 0.5 → **权重塌缩**（单粒子主导） |
| F-5 | 任意步后 | `slam.effective_sample_size()` | N_eff = 1/Σw²（标量） | 均匀 = N = 50；基准跑完 ~27；低于 `0.3N` 触发重采样 | = 1 → 完全退化；长期贴 0.3N 附近震荡 → 重采样过频（选择噪声） |
| F-6 | 任意步后 | `slam.n_resamples` | 累计系统重采样次数（int） | 基准 60 步 ~35–36 | 0 → 测量信息没进权重（似然未折入）；数百 → 运动噪声设置过大 |
| F-7 | `_ekf_update_slots` ~L687 | `nu` | 测量新息 (n, 2) | 与观测噪声同量级（~0.15 m, ~0.05 rad） | 系统性大且不收敛 → 关联错或位姿错 |
| F-8 | 端到端（`test_fastslam.py` 打印） | pose / traj / map RMSE | 位置误差（标量，m） | known 模式 0.113 / 0.221 / 0.201 m；航位推算基线 1.571 m（≈ 1/14） | 断言要求 < 基线 1/3；接近基线 → 滤波没起作用 |

### P：preint（断点 `preint/preint.py`）

| # | 断点 / 来源 | 变量 | 含义（形状） | 健康值 | 异常信号 |
|---|-------------|------|--------------|--------|----------|
| P-1 | `__init__` 递推循环 ~L240 | `self.cov` | Σ = Cov[δφ; δv; δp] (9, 9)，强制对称 | 对角随积分时长增长且与 Monte-Carlo 经验 std 的相对偏差 **11.3%**（门限 30%，`tests/test_preint.py` 实测 0.113） | > 30% → A/B 块下标错（Eq.(46)-(47) 递推坏）；非对称 → 丢了对称化 |
| P-2 | `__init__` 结束 | `self.j_bias` | 偏置雅可比 (9, 6) | 与完整重积分的中心差分相对差 < 1e-7（实测 5.35e-11） | 某列恒 0 → 对应偏置块未传播 |
| P-3 | `correct` ~L326 | 返回值 vs 在 `b+δb` 处重积分 | 一阶修正残差（标量，rad / m/s / m） | \|δb^g\|≈0.03, \|δb^a\|≈0.10 时 R 2.4e-5 / v 3.5e-4 / p 1.1e-4（< 1e-3）；δb 减半残差 ≈ 1/4（O(δb²)） | 残差随 δb 线性 → `j_bias` 错；残差 > 1e-3 → δb 太大超出一阶有效域 |
| P-4 | `predict` ~L355 | 返回 (R_j, v_j, p_j) | 零噪声状态预测 | 与同批采样的世界系离散积分精确一致（实测差 ~1e-16） | 有差 → 重力符号 / `-½gΔt²` 项错（(10.5)） |

### V：vins（断点 `vins/vins.py`）

| # | 断点 / 来源 | 变量 | 含义（形状） | 健康值 | 异常信号 |
|---|-------------|------|--------------|--------|----------|
| V-1 | `ImuFactor.residual` ~L187 | 返回值 `(15,)` 的范数 | 真值处白化 IMU 残差（χ₁₅） | 期望 E‖e‖ ≈ 3.7，基准实测 [3.5, 3.5, 3.2]；**门限 8.0**（≈ 2 倍期望） | > 8 → 模型失配（测量模型/预积分/白化坏）；真值处就不小 → 查 `Preintegration` |
| V-2 | `ViBundle.solve` ~L498 | `cost` | 鲁棒（白化）代价（标量） | 单调下降至 ~1.9e2（基准 17 轮收敛） | 上升/震荡 → 回缩或雅可比坏；卡死不降 → 查 `imu_weight` |
| V-3 | `ViBundle.solve` ~L501 | `lam` | LM 阻尼（标量，Marquardt 对角缩放） | 初值 1e-6；接受步 ÷10、拒绝步 ×10 | 单调涨到 1e14 → H 不含下降方向（尺度规范自由 → 看案例 2） |
| V-4 | `solve` 参数 | `imu_weight` | IMU 因子全局权重（标量） | 1.0（紧耦合） | 0 = 纯视觉退化运行：尺度**必然**不可恢复（这是设计好的对照，不是 bug） |
| V-5 | 测试 `_interframe_ratios`（`tests/test_vins.py`） | 逐对尺度比 | ‖p_j−p_i‖_est / ‖p_j−p_i‖_gt（list） | 紧耦合恢复后 **0.94–1.04**（初值全 0.5） | 纯视觉 0.38–0.42 = 尺度塌缩的活体样本（教程 §10.3）；紧耦合下仍 < 0.9 → IMU 因子失效 |
| V-6 | 求解后 | `res.states[*].b_g / .b_a` | 零偏估计（(K, 3) 各） | b_g 均值 ≈ 真值 [0.01, -0.008, 0.005]；**b_a 在本教学场景基本不可观**（打印值远离真值属预期，勿当 bug） | b_g 发散 → 陀螺噪声/预积分坏 |
| V-7 | 端到端 | `ate`（align_se3 后） | ATE（标量，m） | 2.36 cm（门限 < 0.1 m） | 对齐后仍大 → 不是 gauge 问题，是真错误 |

### B：photoba（断点 `photoba/photoba.py::PhotometricBA.solve`）

| # | 断点 / 来源 | 变量 | 含义（形状） | 健康值 | 异常信号 |
|---|-------------|------|--------------|--------|----------|
| B-1 | ~L612–646 | `alpha` | 回溯线搜索步长（标量，1, 1/2, 1/4, ...） | 接受步多为 1 或 1/2 | 连续砍到 1/2^16 → 当前线性化点失效（初值太差/雅可比错） |
| B-2 | ~L642–643 | `ratio` / `lam` | Nielsen 增益比（实际/预测下降） / LM 阻尼 | ratio ≈ 1 → λ 按 `max(1/3, 1-(2ρ-1)³)` 放松；**被线搜索砍短的步（ratio << 1）λ 反升**——这道闸门是历史发散 bug 的修复（见 §4 案例 3） | λ 一路 ×10 到 1e12 → 无可接受步，检查目标函数/规范锚点 |
| B-3 | ~L606–610 | `self._rho_lo` / `self._rho_hi` | 逆深度物理箱 (N,) 各 | `0.25 ρ0` / `4 ρ0`（宽箱：优化实际只需百分之几修正） | ρ 贴箱壁 → 初值深度错得离谱或该点弱纹理（无信息） |
| B-4 | 端到端（`test_photoba.py` 打印） | `b_err`（`max|exposure_b - 真值|`） | 曝光偏移恢复误差（标量，灰度） | **< 0.6**（实测 3.8e-1；受渲染底限主导）；`a_err` < 1e-2（实测 3.2e-3） | b_err > 1 → 曝光没参与优化（`optimize_exposure=False`？）或亮度规范被破坏 |
| B-5 | 端到端 | `result.cost` | 最终加权限代价（标量，灰度²） | ~6.9（基准窗口） | ~5.8e4 → 曝光关闭的对照模式（亮度失配未白化，几何也跟着错 ~22 倍） |

### L：loam2d（断点 `loam2d/loam2d.py::_solve_point_to_line`）

| # | 断点 / 来源 | 变量 | 含义（形状） | 健康值 | 异常信号 |
|---|-------------|------|--------------|--------|----------|
| L-1 | ~L455 | `cond`（即 `RegistrationResult.condition_number`） | 终端 `J^T J` 条件数（标量） | 良态矩形房间 **~9.6–11.6**（实测 9.6 / 11.5 / 11.6） | **inf** = 某方向完全不可观（圆形房间原地旋转：旋转方向曲率趋零）；> 1e8 → healthy=False |
| L-2 | ~L458 | `healthy` | 可观测性健康标志（bool） | True（cond ≤ max_condition 且 `n_residuals ≥ 3`） | False 被正确检出是"特性"（退化被显式暴露而非静默失败）；全流程healthy=False → 场景无特征或对应门限 `max_corr_dist` 太小 |
| L-3 | 端到端（`test_loam2d.py` 打印） | odom / map 终端 t-err | 漂移（标量，m） | odom **17.2 mm**，map **2.6 mm**（≈ 1/6，mapping 每 5 帧重新锚定） | map 链不优于 odom 链 → scan-to-map 对应（协方差线性门 `min_linearity`）失效 |

### D：droidlite（断点 `droidlite/droidlite.py`）

| # | 断点 / 来源 | 变量 | 含义（形状） | 健康值 | 异常信号 |
|---|-------------|------|--------------|--------|----------|
| D-1 | `DenseBA.solve` ~L531 | `res.pose_rmse` | 逐轮位姿 RMSE (R+1,)，6 维 se3_log 范数 | **单调不升**：0.064 → 5.2e-5（增益 ~1225x；depth 0.0038 → 6.1e-5，~63x） | 回升 → LM 步被拒 / 光流噪声调度（`flow_decay`）坏；平台不降 → 视差不足 |
| D-2 | `_solve_ba_increment` ~L483 | `lam` | 内层 LM 阻尼 | 接受 ÷10、拒绝 ×10（上限 1e10） | 频繁拒绝 → 光流测量与几何矛盾 |
| D-3 | `_schur_step` ~L443 | `C` | 深度块（对角）(N,) = 列范数平方 + λ | 全部 > 0 | 有 0/negative → 对应列全零（该像素无任何有效观测，`valid` 掩码漏了） |
| D-4 | 交叉验证（`tests/test_droidlite.py` 打印） | Schur vs 稠密解差 | `max|Schur - dense|`（标量） | 单步 ~1e-10；完整求解 ρ 差 9.1e-7、位姿差 4.3e-7（**< 1e-5**，numpy 版本敏感点见 §1.3） | > 1e-2 → 消元实现错（Eq.(5)） |

### E：epipolar / direct（评估量在测试里）

| # | 断点 / 来源 | 变量 | 含义（形状） | 健康值 | 异常信号 |
|---|-------------|------|--------------|--------|----------|
| E-1 | `tests/test_epipolar.py::_rel_error` | `err_e` / `err_f` | E/F 相对 Frobenius 误差（吸收最优尺度；标量） | 无噪声数据 **~1e-14**（实测 1.3e-14 / 5.7e-17） | > 1e-6 → Hartley 归一化缺失/顺序错、或本质流形投影（式 5.6）被跳过 |
| E-2 | `decompose_E` 手性计数 | `in_front` | 正深度点数（int） | 4 候选中唯一最大者 = 真实运动 | 并列 → 退化场景（共面/纯旋转），需换场景 |
| E-3 | `tests/test_direct.py` 打印 | 直接法位姿误差 | trans / rot（标量） | 2.1e-4 m / 5.6e-5 rad | 停在 ~0.05 m（初值量级）不降 → 图像梯度因子为 0（弱纹理）或 `guard` 把像素裁光 |

### C：core 求解器（所有优化模块共用）

| # | 断点 / 来源 | 变量 | 含义（形状） | 健康值 | 异常信号 |
|---|-------------|------|--------------|--------|----------|
| C-1 | `core/solver.py::gauss_newton` ~L121/127 | `lam` | LM 阻尼（标量） | 拒绝 ×10（上限 1e12）、接受 ÷10 | 涨到上限退出 → 残差函数在步内非有限或无下降方向 |
| C-2 | 同上 | `improvement` | 代价下降量（标量） | 接受步 > 0；`< tol` 收敛退出 | 需为负才被接受——若残差/雅可比符号约定不一致（如"观测-预测"与"预测-观测"混用），GN 会在平移与旋转上互相打架（`epipolar.reprojection_jacobian` docstring 的显式警告） |

---

## 4. 常见调试场景（症状 → 断点 → 看什么 → 结论）

### 案例 1：FastSLAM 路标重复（一个真值路标占了两个槽位）

- **症状**：`test_nearest_neighbor_gating_beats_dead_reckoning` 失败；断点处
  `slam.observed.sum(axis=1)` 超过真值路标数 20；`map_estimate()` 出现两行指向
  同一真值路标。
- **断点**：`fastslam/fastslam.py::FastSLAM2D._update_nearest`（L643 `d2` 一行）。
- **看什么**：合法重观测的 `d2` 分布（F-1）与 `self.mahalanobis_gate`（F-3）的
  相对位置。门控最近邻的 d² 有**两个尺度**：合法重观测 = 卡方(2) + 位姿采样
  误差，基准噪声下 ≤ ~20；不同路标之间 (间距/σ)² ≥ (2 m / 0.3 m)² ≈ 44。
  若把门限设成教科书值 `GATE_CHI2_2DOF_95 = 5.991`，95% 分位数没算上"提议
  位姿本身的误差"，百分之几的好测量被拒 → 每次拒绝初始化一个重复路标。
- **结论**：门限必须隔开两个尺度——用 `GATE_DEFAULT = 30.0`；同时确认滤波器
  的 `motion_noise_std / obs_noise_std` 与仿真器一致（`tests/test_fastslam.py`
  顶部的共享常数）。

### 案例 2：VINS 尺度不对（轨迹"大小"不是米制）

- **症状**：`test_vi_bundle_recovers_metric_scale` 的尺度比断言失败，或自跑
  `ViBundle.solve` 后轨迹形状对、整体大小错。
- **断点**：`vins/vins.py::ViBundle.solve` 入口（L476 `lam = ...` 一行）+
  `tests/test_vins.py::_interframe_ratios`。
- **看什么**：① `imu_weight` 是否为 0（V-4）——为 0 时纯视觉代价对 `x → s·x`
  严格不变，尺度方向**无梯度**，实测比值漂到 0.38–0.42（初值 0.5），这不是
  bug 而是尺度不可观的活体演示；② `imu_weight > 0` 时看 `ImuFactor.residual`
  的范数（V-1）：真值处应 ~3.7、门限 8——若爆表，问题在预积分或测量模型而不
  是优化器；③ `lam` 是否一路涨到 1e14（V-3）——H 无下降方向。
- **结论**：尺度只能来自 IMU 因子（教程 §10.3 第 3 步）。健康的紧耦合运行
  逐对尺度比落在 0.94–1.04、ATE ~2.4 cm；纯视觉 0.38–0.42 属预期对照。

### 案例 3：photoba 发散（代价震荡 / 逆深度飞出物理范围）

- **症状**：`PhotometricBA.solve` 代价不降反升或剧烈震荡；`inverse_depths`
  出现荒谬值；`result.converged = False`。
- **断点**：`photoba/photoba.py::PhotometricBA.solve` 的接受分支
  （L642–643 `ratio = ...` / `lam = ...` 两行）与线搜索（L612 `alpha = 1.0`）。
- **看什么**：① `alpha`（B-1）：线搜索是否总把步砍到 1/2^k —— 是则当前
  线性化很不可信；② `ratio`（增益比，B-2）：被大幅砍短（alpha << 1）的步
  其实际下降远小于二次模型预测，ratio << 1；③ `lam` 的走向。
- **结论**：本模块 docstring 与 solve 内注释记录了这段历史：**修复前的实现
  对"被线搜索砍短的接受步"仍执行 λ ← λ/10 的放松**，弱观测的逆深度方向在
  离最优点尚远时反复迈出野步 → 发散。修复（Nielsen 增益比信赖域更新）后，
  λ 只按二次模型预测得有多好来放松：`lam = lam * max(1/3, 1-(2*ratio-1)**3)`
  ——alpha 小（ratio << 1）的步反而**提高** λ；只有被拒步才 ×10。调试同类
  LM 实现时，先确认这道闸门存在，再看逆深度箱（B-3）与曝光初始化
  （`_initialize_exposure`：亮度失配白化后联合阶段才从几何量级残差起步）。

---

## 5. 测试怎么写

新测试放在 `tests/test_<模块名>.py`，遵循本套件既有惯例：

1. **命名 `test_<行为>`**——名字说清"验证什么行为"，因为单点过滤按子串匹配
   （§1.1），名字是过滤键。范例：`test_mahalanobis_gate_rejects_outlier`、
   `test_vi_bundle_recovers_metric_scale`。
2. **deterministic seed**——一切随机性走
   `np.random.default_rng(seed)`（场景、噪声、子采样各给独立种子），使打印
   的数字与断言阈值可精确复现；禁止 `np.random.seed`/全局随机态。
3. **断言阈值旁注明理由**——照套件惯例（"门限理由：……"注释）写清期望值
   的量级来源，让后来者能区分"阈值松了"与"实现错了"。
4. **文件尾部带单点过滤主入口**——照抄 `tests/test_core_slam.py` 的
   `__main__` 块：

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

5. **运行时长预算 < 15 s / 文件**——当前最慢的是 `tests/test_droidlite.py`
   （~6.5 s）；Monte-Carlo 类测试（如 `test_covariance_matches_monte_carlo`
   的 200 次运行）通过减少采样数而非删断言来控制时长。指标断言从
   `metrics.py` 取实现（[METRICS.md](METRICS.md)），勿在测试里重写公式。
6. **写完自查**：`python tests/test_<新文件>.py` 全绿 + 该模块断点处的观察量
   （§3 变量表）落在健康值内 + 若新增了被监视变量，同步更新本文件的变量表
   （CONSTRAINTS §4.5 质量要求）。

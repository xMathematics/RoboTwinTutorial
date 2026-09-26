# SLAM 统一测评指标教程（METRICS）

> 零基础基准撰写：不假设读者有机器人学背景，术语首现给中文通俗解释。
> 代码：[metrics.py](metrics.py)（全部指标的唯一定义处）｜ 测试：[tests/test_metrics.py](tests/test_metrics.py)
> 章节互引：[tutorials/slam 总览](../../tutorials/slam/OVERVIEW.md) ｜ 第 02 章（旋转）｜ 第 05/09 章（尺度与 Sim(3)）｜ [第 10 章 建图与系统实战](../../tutorials/slam/10_建图与系统实战.md)

## 0. 为什么需要统一的指标

**① 问题场景**：本项目的 SLAM 代码库有 fastslam、vins、loam2d、droidlite、photoba 等多个模块，每个模块的测试里都要回答同一个问题——"估计出来的轨迹/地图到底准不准？"。如果每个模块各自手写一个 RMSE 函数、各自决定"要不要先对齐再算"，就会出现两个后果：同一数字在不同模块含义不同（有的含对齐、有的不含），无法横向比较；且新模块的测试会继续复制粘贴旧实现，错误随之繁殖。

**② 解决方法**：所有指标只在一个地方定义——`projects/slam/metrics.py` 的 5 个纯函数；各模块测试中的指标断言从这里取用，模块测试里的 `_position_rmse`、`vins.align_se3` 等是同一公式的就地实现（历史遗留，逐步统一到本模块）。

**③ 为什么这些指标够用**：一条轨迹的"准"可以拆成四个互相独立的侧面——**全局位置**（ATE）、**朝向**（旋转误差）、**尺度**（尺度比）、**局部运动**（RPE）。fastslam 验证滤波后端（教程第 07 章）、vins 验证视觉惯性紧耦合（第 10 章）、loam2d 验证激光里程计与建图（第 10 章 LOAM 部分）、droidlite 验证学习式稠密 BA（第 10 章末前沿导读），它们的主张恰好各落在一两个侧面上——这套指标足以支撑各模块的全部断言。

**通俗概念铺垫**（后文反复出现）：

- **RMSE（均方根误差，Root Mean Square Error）**：先对每个点的误差平方求平均，再开根号。平方放大了大误差、开根号后回到原单位（米），所以"一个 0.1 m 的 RMSE"直观对应"典型位置差了 10 厘米"。
- **规范自由度（gauge freedom）**：SLAM 问题本身"钉不死"的全局摆位。比如纯单目视觉里，把整条轨迹和地图**整体**旋转、平移、缩放，所有图像观测完全不变（[第 10 章 §10.3](../../tutorials/slam/10_建图与系统实战.md) 的可观性论证）——估计器选哪个摆位都"没错"。评估时必须先把这些自由度对齐掉，否则会把与算法好坏无关的摆位差记成误差。

### 指标速查表

| 指标 | 一句话作用 | 代码函数（`metrics.py`） | 单位 | 公式 |
|------|-----------|--------------------------|------|------|
| Umeyama 相似对齐 | 把估计轨迹摆到真值坐标系（评估前置步骤） | `umeyama_align(src, dst, with_scale)` | m / rad / 无量纲 | (M.1) |
| 绝对轨迹误差 ATE RMSE | 全局位置精度（漂移的综合后果） | `ate_rmse(traj_est, traj_gt, align)` | m | (M.2) |
| 旋转误差 | 两个姿态差多少角度 | `rotation_error_deg(R_est, R_gt)` | deg | (M.3) |
| 尺度比 | 估计轨迹的"大小"对不对（单目尺度可观性） | `scale_ratio(lengths_est, lengths_gt)` | 无量纲 | (M.4) |
| 相对位姿误差 RPE（平移） | 局部运动是否平滑准确（与 ATE 互补） | `rpe_translation(traj_est, traj_gt, delta)` | m | (M.5) |

统一运行方式（详见各节"如何运行"）：

```bash
cd projects/slam
python3 tests/test_metrics.py          # 全部指标测试（13 个）
python3 tests/test_metrics.py umeyama  # 单点：只跑名字含 umeyama 的测试
pytest tests/test_metrics.py -v        # pytest 等价跑法
```

---

## 1. Umeyama 相似对齐 —— `umeyama_align`

### 1.1 作用

衡量/解决什么：它本身不是"分数"，而是 ATE 的**前置步骤**——在比较估计轨迹与真值之前，求出最优的旋转 $R$、平移 $t$、（可选）尺度 $s$，把估计轨迹整体摆到真值坐标系里。没有它，第 0 节说的规范自由度会让所有 ATE 数字失去意义。凡是"估计结果与真值只差一个整体摆位"的场合都靠它：轨迹对齐、地图点云对齐、把 vins 的估计 landmarks 摆回真值系。

### 1.2 如何计算

问题：给两组一一对应的点 $\{p_i\}$（估计）与 $\{q_i\}$（真值），求

$$dst \approx s\,R\,src + t,\qquad (R\in SO(3)) \tag{M.1}$$

使 $\sum_i \|q_i - sRp_i - t\|^2$ 最小。闭式解四步（代码位置：`metrics.py::umeyama_align`，每步都有行内中文注释）：

1. **中心化**：减去各自质心 $\bar p,\bar q$，把平移从问题中解耦（对 $t$ 求导置零得 $t = \bar q - sR\bar p$）；
2. **互协方差**：$H = \sum_i p_i' q_i'^{\top}$（$p'$ 为中心化后的点；$3\times 3$ 矩阵，浓缩了两组点的"形状相关性"）；
3. **SVD 分解** $H = U\Lambda V^{\top}$ 后取 $R = V\,\mathrm{diag}(1,1,d)\,U^{\top}$，其中 $d = \mathrm{sign}\,\det(VU^{\top})$——这就是**防反射校正**：SVD 给出的无约束最优解可能是 det $=-1$ 的镜像（反射不是刚体运动），校正把符号翻转吸收到最小奇异值方向；
4. **尺度**（`with_scale=True` 时）：$s = \dfrac{\lambda_1+\lambda_2+d\,\lambda_3}{\frac{1}{N}\sum_i\|p_i'\|^2}$（对 $s$ 求导置零），`with_scale=False` 时固定 $s=1$。

文献出处：S. Umeyama, *"Least-squares estimation of transformation parameters between two point patterns"*, IEEE TPAMI 13(4), 1991（§III 闭式解；防反射即其 det 校正）。理论背景（旋转、对数映射、SO(3)）见[第 02 章](../../tutorials/slam/02_三维刚体运动旋转与位姿.md)与 `core/lie.py`；gauge 与 Sim(3) 的动机见第 09 章（回环）与[第 10 章 §10.3](../../tutorials/slam/10_建图与系统实战.md)。

### 1.3 健康值范围

| 场景 | 健康表现 | 异常信号 |
|------|----------|----------|
| 已知 $R,t,s$ 的合成点集 | 三参数精确恢复（误差 < 1e-9），残差 < 1e-12 | 恢复失败 ⇒ 实现有 bug |
| 恢复出的 $R$ | det $=+1$、正交性 $\|RR^\top - I\|<10^{-10}$ | det $<0$ ⇒ 防反射校正被跳过，解是"作弊的镜像" |
| 镜像（反射）点集输入 | det $=+1$ 强制保持，残差**显性变大** | 残差≈0 且 det<0 ⇒ 反射被静默"完美拟合" |

本项目实测（`tests/test_metrics.py`，确定性 seed=7）：`test_umeyama_rigid_recovery_without_scale` 与 `test_umeyama_with_scale_recovery` 恢复误差 < 1e-9；`test_umeyama_reflection_guard` 实测未校正解 det $=-1$、残差 < 1e-6，校正后 det $=+1$、残差 ≈ 0.02 m（旋转无法表示反射，误差必须显性化）。另见 `vins/vins.py::align_se3`——同一定式的就地实现（VIO 尺度可观，故其 ATE 用 `with_scale=False` 的 SE(3) 对齐，避免对齐吃掉尺度误差）。

### 1.4 如何运行

```bash
cd projects/slam
python3 tests/test_metrics.py umeyama   # 4 个 umeyama 测试：恢复/尺度/朴素交叉/防反射
# 预期输出（节选）：
# PASS test_umeyama_rigid_recovery_without_scale
# PASS test_umeyama_with_scale_recovery
# PASS test_umeyama_naive_implementation_cross_check
# PASS test_umeyama_reflection_guard
# 4/4 tests passed.
```

在自己模块的测试中使用：

```python
import numpy as np
from metrics import umeyama_align

R, t, s = umeyama_align(points_est, points_gt, with_scale=False)
aligned = s * (points_est @ R.T) + t        # (N, 3)：与 R @ est.T + t 等价的向量化写法
```

---

## 2. 绝对轨迹误差 ATE RMSE —— `ate_rmse`

### 2.1 作用

衡量什么：估计轨迹与真值轨迹的**全局位置**差多远——漂移（前端每步的小误差被后端没压住、随时间累积）的综合后果。对应各论文"我们的系统在 X 数据集上 ATE = Y 米"的主张，也是本项目各模块测试断言（如"FastSLAM 必须明显优于航位推算"）所比较的主数字。对齐后它评估的是**估计轨迹的形状**，7 自由度 gauge 已消去（见 1.1 节）。

### 2.2 如何计算

先按 (M.1) 做带尺度的 Umeyama 对齐，再取逐点误差的均方根：

$$\mathrm{ATE}_{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\left\|\,\big(sRp_i^{est} + t\big) - p_i^{gt}\,\right\|^2} \tag{M.2}$$

代码位置：`metrics.py::ate_rmse`（内部调用 `umeyama_align(..., with_scale=True)`）。文献出处：J. Sturm, N. Engelhard, F. Endres, W. Burgard, D. Cremers, *"A Benchmark for the Evaluation of RGB-D SLAM Systems"*, IROS 2012, §IV-A（ATE 的标准定义，含尺度对齐以兼容单目系统）。gauge 自由度为何剩 4 维（单目+IMU）或 7 维（纯单目）：[第 10 章 §10.3 第 3 步](../../tutorials/slam/10_建图与系统实战.md)的可观性论证；纯单目尺度不确定性的来源：第 05 章三角化。

**align 参数怎么选**：`align=True`（默认）——不确定坐标系时永远用它；`align=False`——仅在两条轨迹已确认同坐标系时用（如对比同一 gauge 下的两个消融版本），它会把全局摆位差原样计入误差。

### 2.3 健康值范围（本项目实测）

| 模块（测试） | 实测 ATE / 位置 RMSE | 对照基线 | 说明 |
|--------------|----------------------|----------|------|
| fastslam（`tests/test_fastslam.py`，`[known]` 打印） | **位姿 RMSE 0.113 m**（轨迹 0.221 m） | 航位推算 **1.571 m** | 回环+滤波把误差压到基线的 ~7%；断言要求 < 基线 1/3 |
| vins 紧耦合（`tests/test_vins.py`，`[vi-bundle]`） | **ATE RMSE 2.36 cm** | 初值为错误尺度（比值 0.5） | IMU 把尺度钉住后的米制精度 |
| loam2d 里程计（`tests/test_loam2d.py`，`[dual-rate]`） | 末帧 t-err **17.2 mm** | —— | 纯帧间配准的累积漂移 |
| loam2d 建图（同上） | 末帧 t-err **2.6 mm** | 里程计 17.2 mm | 扫描配准到地图后漂移回灌，≈ 1/6 |
| droidlite（`tests/test_droidlite.py`，`[recurrent]`） | 位姿 RMSE **≈ 5.2e-5**（第 6 轮打印 0.00005） | 初始 0.064 | 学习式稠密 BA 迭代收敛，~1200 倍增益 |

异常信号：**对齐后的 ATE 仍随时间近似线性增长** ⇒ 漂移未被后端/回环抑制（对照 loam2d 里程计 vs 建图的 6 倍差距，看你的后端有没有起作用）；**对齐后 ATE 依然巨大** ⇒ 不是 gauge 问题而是真错误（跟踪丢失、数据关联错、外点未被鲁棒核压制）。

### 2.4 如何运行

```bash
cd projects/slam
python3 tests/test_metrics.py ate
# 预期输出（节选，数值确定性复现）：
# [ate] identity: aligned 1.94e-15 m, unaligned 0.00e+00 m
# [ate] drift traj: aligned 0.0588 m, unaligned 0.2140 m
# 3/3 tests passed.
```

模块测试中的使用示例（vins 式）：

```python
from metrics import ate_rmse, umeyama_align

ate = ate_rmse(p_est, p_gt)              # 默认 align=True，单位 m
assert ate < 0.1, ate                    # 断言：厘米级
```

---

## 3. 旋转误差 —— `rotation_error_deg`

### 3.1 作用

衡量什么：两个**姿态**（朝向）差多少度。位置对不等于朝向对——机器人导航里朝向偏差会让控制器朝错误方向走。它是 (M.2) 只看位置、不看朝向的补集；loam2d、vins 等输出完整位姿 (4, 4) 的模块用它断言朝向精度。文献出处：转角即 SO(3) 上的测地距离，定义见[第 02 章](../../tutorials/slam/02_三维刚体运动旋转与位姿.md)（对数映射；`core/lie.py` 的实现对应视觉 SLAM 十四讲 Eq. 4.19-4.22）。

### 3.2 如何计算

$$e_{rot} = \left\|\log\!\left(R_{est}^{\top} R_{gt}\right)\right\|\cdot\frac{180}{\pi} \tag{M.3}$$

先求**相对旋转** $R_{rel} = R_{est}^{\top} R_{gt}$（"从估计姿态转到真值姿态还需要转多少"），再取对数映射回到旋转矢量，其范数就是转角。代码位置：`metrics.py::rotation_error_deg`，复用 `core/lie.py::so3_log`（不重写旋转数学——全库位姿数学只有 `core.lie` 一处出处）。取值范围 $[0°, 180°]$（SO(3) 测地距离的上限是 $\pi$）。

### 3.3 健康值范围（本项目实测）

| 模块（测试） | 实测旋转误差 | 说明 |
|--------------|--------------|------|
| loam2d（`[dual-rate]` 打印 th-err） | 里程计 0.00119 rad ≈ **0.068°**；建图 0.00085 rad ≈ **0.049°** | 2D 激光 SLAM 的典型末帧朝向精度 |
| 合成测试（`tests/test_metrics.py`） | 1° 输入 → 1.0000000000°（±1e-9）；30° 输入 → 30.0000000000° | 解析值交叉验证 |

异常信号：旋转误差接近 **180°** ⇒ 朝向解反了（对极几何的两义性未消解、回环误接受导致地图翻转）；**位置 ATE 小但旋转误差大** ⇒ 前端在"原地打转"，平移方向系统性偏差。

### 3.4 如何运行

```bash
cd projects/slam
python3 tests/test_metrics.py rot
# 预期输出（节选）：
# [rot] 1 deg -> 1.0000000000 deg, 30 deg -> 30.0000000000 deg
# 2/2 tests passed.
```

模块测试中的使用示例（本函数只接受 (3, 3) 旋转矩阵，不接 (4, 4) 位姿）：

```python
from metrics import rotation_error_deg

err_deg = rotation_error_deg(T_est[:3, :3], T_gt[:3, :3])   # 取旋转块，单位 deg
assert err_deg < 0.1, err_deg                                # 0.1° 以内
```

---

## 4. 尺度比 —— `scale_ratio`

### 4.1 作用

衡量什么：估计轨迹的"**大小**"是真是假——把相邻帧距离（步长）加起来求比值 $\hat s$，就是整条轨迹尺度的单一估计。它是单目系统**尺度可观性**的直接探针：纯单目视觉的轨迹带任意尺度（第 05 章三角化的尺度不确定性；gauge 7 维中的一维），加入 IMU 后尺度才被加计读数钉住（[第 10 章 §10.3 第 3 步](../../tutorials/slam/10_建图与系统实战.md)的论证：缩放轨迹会放大 IMU 残差，方程组不再自洽）。vins 模块"IMU 把错误尺度初值 0.5 拉回 1"的核心主张，就是靠这个指标验证的。

### 4.2 如何计算

$$\hat s \;=\; \frac{\sum_{i=1}^{N-1}\left\|p_{i+1}^{est} - p_i^{est}\right\|}{\sum_{i=1}^{N-1}\left\|p_{i+1}^{gt} - p_i^{gt}\right\|} \tag{M.4}$$

即**路程之比**（相邻帧距离和之比）。代码位置：`metrics.py::scale_ratio`，输入是两串步长 `(N-1,)`（由 `np.linalg.norm(np.diff(traj, axis=0), axis=1)` 从轨迹算出），不是轨迹本身——这样"比值"与"步长怎么算"解耦。文献出处：尺度可观性见 VINS-Mono（[第 10 章主读论文](../../tutorials/slam/10_建图与系统实战.md)，§V 可观性分析）；尺度不确定性的理论来源见第 05 章。vins 测试里的逐对版本（每对关键帧一个比值，而非全程一个）见 `tests/test_vins.py` 的 `_interframe_ratios`。

### 4.3 健康值范围（本项目实测）

| 场景（测试） | 实测尺度比 | 结论 |
|--------------|-----------|------|
| vins 单目+IMU（`[vi-bundle]` est. scale ratios） | 逐对比值 0.94 – 1.04（初值 0.5） | IMU 把尺度钉住 ⇒ 健康 |
| vins 纯视觉（`[pure-visual]`，IMU 权重清零） | 逐对比值 **0.38 – 0.42**（打印 [0.395 0.389 0.399 0.380 0.395 0.416]） | 尺度不可观 ⇒ 尺度塌缩的活体样本 |
| 合成缩放轨迹（`tests/test_metrics.py`） | 0.62 倍输入 → 比值 0.620000（±1e-12） | 解析值交叉验证 |

**异常信号：$|\hat s - 1| > 10\%$** ⇒ 尺度异常：单目+IMU 系统多半是 IMU 因子失效（外点、偏置发散、退化运动激励不足），纯单目系统则属正常（尺度本就不可观，此时评估应配合 Sim(3) 对齐而非把它当错误）。$\hat s$ 随时间单调漂移 ⇒ 加计偏置估计发散。

### 4.4 如何运行

```bash
cd projects/slam
python3 tests/test_metrics.py scale
# 预期输出（节选）：
# [scale] 0.62x scaled traj -> ratio 0.620000
# 2/2 tests passed.
python3 tests/test_vins.py pure_visual   # 实测对照：纯视觉尺度塌缩到 0.38-0.42
```

模块测试中的使用示例：

```python
import numpy as np
from metrics import scale_ratio

step_est = np.linalg.norm(np.diff(traj_est, axis=0), axis=1)  # (N-1,) 步长，m
step_gt  = np.linalg.norm(np.diff(traj_gt,  axis=0), axis=1)
ratio = scale_ratio(step_est, step_gt)
assert abs(ratio - 1.0) < 0.1, ratio     # METRICS.md 的 10% 异常阈值
```

---

## 5. 相对位姿误差 RPE（平移分量）—— `rpe_translation`

### 5.1 作用

衡量什么：**局部运动**准不准——估计轨迹在间隔 `delta` 帧内的相对位移与真值差多少。它与 ATE 刻意互补：ATE 管"终点相对起点的绝对位置"（全局一致性），RPE 管"每一步/每几步走得多准"（局部平滑性）。一条每步 2 mm 漂移的轨迹，200 步后绝对位置差 ~0.26 m（ATE 差），但它每一步的相对运动几乎完美（RPE 好）——这两个数字讲的是两个不同的故事，单一 ATE 会把它们混在一起。文献出处：J. Sturm et al., IROS 2012, §IV-B（RPE 定义在 SE(3) 上，含旋转+平移两个分量；本函数取平移分量，因为 (N, 3) 轨迹只含位置）。

### 5.2 如何计算

对每个间隔 $\delta$ 的帧对 $(i,\ i+\delta)$，取相对位移之差再求均方根：

$$\mathrm{RPE}_{t}(\delta) = \sqrt{\frac{1}{N-\delta}\sum_{i=1}^{N-\delta}\left\|\,\big(p_{i+\delta}^{gt} - p_i^{gt}\big) - \big(p_{i+\delta}^{est} - p_i^{est}\big)\,\right\|^2} \tag{M.5}$$

代码位置：`metrics.py::rpe_translation`。**无需对齐**：相对量对刚体规范自由度不变（整体平移/旋转在差分中相消）——这是它与 ATE 的第二处本质区别。`delta=1` 是逐帧局部误差；`delta` 越大，长程漂移越多地渗入 RPE，指标越接近 ATE 的视角。需要旋转分量时，对 (4, 4) 位姿序列用 `core/lie.py::se3_log` 自行扩展（本模块只覆盖平移分量）。

### 5.3 健康值范围（本项目实测，`tests/test_metrics.py` 确定性输出）

| 场景（测试构造） | 实测 rpe(d=1) | 实测 ATE（不对齐） | 结论 |
|------------------|---------------|--------------------|------|
| 只差全局平移 30 m（纯 gauge） | **≈ 0**（1.8e-15 m） | 33.06 m | gauge 差不该记成局部误差 |
| 每步 2.3 mm 线性漂移 × 200 步 | **2.29 mm** | **263.6 mm** | rpe/ate ≈ 1/115：局部好、全局差 |
| 逐帧 10 mm 零均值抖动 | **24.8 mm** | 17.1 mm | rpe > ate：局部抖、全局均值良好 |

异常信号：**rpe 与 ate 同量级且都大** ⇒ 问题在局部（前端跟踪抖动、特征外点），而不是漂移——优化方向应是前端/鲁棒核，而非后端或回环；**rpe 小但 ate 大** ⇒ 前端健康、漂移来自后端累积——优化方向应是回环/图优化。这组对照是排障的第一刀。

### 5.4 如何运行

```bash
cd projects/slam
python3 tests/test_metrics.py rpe
# 预期输出（节选，数值确定性复现）：
# [rpe] noisy traj: rpe(d=1) 0.02479 m, rpe(d=3) 0.02429 m
# [rpe] shift  : rpe 1.8e-15 m vs ate 33.06 m
# [rpe] drift  : rpe 2.29 mm vs ate 263.6 mm
# [rpe] jitter : rpe 24.81 mm vs ate 17.05 mm
# 2/2 tests passed.
```

模块测试中的使用示例：

```python
from metrics import rpe_translation

rpe1 = rpe_translation(traj_est, traj_gt, delta=1)   # 逐帧局部误差，m
rpe5 = rpe_translation(traj_est, traj_gt, delta=5)   # 5 帧尺度
assert rpe1 < 0.02                                   # 局部平滑性断言
```

---

## 6. ATE 还是 RPE？选择建议

| 你想回答的问题 | 用哪个 | 理由 |
|----------------|--------|------|
| 系统最终定位准不准（导航/建图能不能用） | **ATE**（align=True） | 用户关心的是绝对位置，漂移的后果全部体现在这里 |
| 论文对标 / 与其他系统横向比较 | **ATE** | Sturm et al. 2012 的标准口径，可比性最强 |
| 前端（里程计）单体的质量 | **RPE**（delta=1） | 相对量天然免疫 gauge 与累积漂移，只考局部 |
| 排障：误差出在前端还是后端 | **ATE 与 RPE 同看** | rpe 小 + ate 大 → 后端/回环；两者都大 → 前端（见 5.3） |
| 单目系统尺度是否健康 | **尺度比**（+ SE(3) 对齐的 ATE） | Sim(3) 对齐会吃掉尺度误差，必须让尺度比单独说话 |
| 断言"后端压住了漂移"（如 fastslam vs 航位推算） | **ATE 比值** | 0.113 m vs 1.571 m（≈ 1/14）是后端价值的直接证据 |

一句话记忆：**ATE 回答"最后错到哪儿了"，RPE 回答"每一步走得稳不稳"**——前者是全局一致性指标，后者是局部平滑性指标；二者与旋转误差、尺度比组成四面体，缺一角就会让某类错误（镜像、尺度塌缩、局部抖动）在评估里隐形。所有公式的唯一定义在 [metrics.py](metrics.py)，回归测试在 [tests/test_metrics.py](tests/test_metrics.py)（13 个测试，全部确定性）。

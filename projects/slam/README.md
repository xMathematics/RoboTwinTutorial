# projects/slam — SLAM 论文教学实现

把 [papers/slam/](../../papers/slam/README.md) 中经典论文的**核心算法**写成能跑的教学代码：
纯 NumPy、无重依赖、每个模块带确定性测试，公式与
[tutorials/slam/](../../tutorials/slam/README.md) 十章教程逐式对应。

**零基础入门请先读 [TUTORIAL.md](TUTORIAL.md)**（每个模块的最小可运行示例 + 数据结构 + 流水线图）；
调试与单点/全局测试用法见 [DEBUG.md](DEBUG.md)；评估口径见 [METRICS.md](METRICS.md)。

## 快速开始

```bash
# ① 激活环境（环境清单见根目录 environment.yml；纯 numpy，任意 numpy>=1.26 的环境亦可）
conda activate llm_env

# ② 依赖校验：本包仅依赖 numpy（无 torch/scipy）
python -c "import numpy; print('numpy', numpy.__version__)"        # 预期: numpy 2.2.6

# ③ 运行核心库测试（李群李代数/相机/求解器，11 个测试）
python tests/test_core_slam.py
```

- **全局测试**：`python -m pytest tests/ -v`（80 个测试，llm_env 约 9.5s）
- **单点测试**：`python tests/test_fastslam.py gate`（子串过滤，无匹配会列出可用测试名）
- VS Code 调试配置见 [.vscode/launch.json](../../.vscode/launch.json)，用法见 [.vscode/SETUP.md](../../.vscode/SETUP.md)

## 模块总览（论文 × 模块 × 教程章）

| 模块 | 论文 | 教程章 | 核心内容 | 测试 |
|------|------|--------|---------|------|
| [core/](core/) | 工具层（—） | 02–05、08 | SO(3)/SE(3) 李群李代数（Rodrigues、BCH、左右雅可比）、针孔相机、GN/LM + Huber | 11 |
| [fastslam/](fastslam/) | FastSLAM (AAAI'02) | 07 | 2D Rao-Blackwellized 粒子滤波：位姿粒子 × 路标 EKF、马氏门控、系统重采样 | 7 |
| [epipolar/](epipolar/) | ORB-SLAM 前端 | 05 | 归一化八点法 → E 分解（手性检验）→ DLT 三角化 → PnP（流形 G-N） | 7 |
| [bowloop/](bowloop/) | ORB-SLAM 回环 | 09 | 视觉词袋（k-means++/TF-IDF/L1 评分）+ 倒排索引 + SE(2) 位姿图优化 | 9 |
| [direct/](direct/) | LSD-SLAM | 06 | 半稠密直接法：梯度像素筛选 → 光度残差 → SE(3) 流形 G-N（含可微双线性采样） | 7 |
| [photoba/](photoba/) | DSO | 06、08 | 滑窗光度 BA：仿射曝光 (a,b) + 逆深度 + Nielsen 增益比 + 逆深度物理箱 | 4 |
| [loam2d/](loam2d/) | LOAM (RSS'14) | 10（扩展） | 2D 教学版：曲率特征（边缘/平面）→ 点到线配准 → 双速率 odometry/mapping | 7 |
| [preint/](preint/) | IMU 预积分 (T-RO'17) | 10 | ΔR̃/Δṽ/Δp̃ 递推、Σ 协方差线性传播、偏置一阶修正（−½gΔt² 约定） | 6 |
| [vins/](vins/) | VINS-Mono (RAM'18) | 10 | 紧耦合 VI 因子图（IMU 因子 + 重投影因子）+ LM 流形求解 + 尺度恢复 | 4 |
| [droidlite/](droidlite/) | DROID-SLAM (NeurIPS'21) | 10（扩展） | 递归稠密 BA 结构演示（GT 光流替代学习网络）：对应场 ↔ Schur 补 BA 交替 | 5 |
| [metrics.py](metrics.py) | 评估层（—） | [METRICS.md](METRICS.md) | Umeyama 对齐 ATE、旋转误差、尺度比、RPE（纯函数，公式 M.1–M.5） | 13 |

## 运行示例（FastSLAM，30 秒上手）

```python
from fastslam import FastSLAM2D, simulate_rectangle
import numpy as np

result = simulate_rectangle(seed=7)                      # 矩形轨迹 + 20 路标
slam = FastSLAM2D(n_particles=50, n_landmarks=result.gt_landmarks.shape[0], seed=7)
slam.initialize(result.gt_trajectory[0])                 # 已知起点
traj = []
for t in range(len(result.odometry)):
    k = result.n_obs[t]                                  # 只取有效观测（其余为 NaN 填充）
    slam.step(result.odometry[t], result.observations[t, :k], result.associations[t, :k])
    traj.append(slam.pose_estimate()[:2])                # 加权平均位姿 (x, y)
traj = np.array(traj)
gt = result.gt_trajectory[1:, :2]
rmse = float(np.sqrt(np.mean(np.sum((traj - gt) ** 2, axis=1))))
print(round(rmse, 3), "m")   # 实测 0.221 m；纯航位推算 ~1.57 m
```

各模块健康值与指标口径见 [METRICS.md](METRICS.md)；全部 11 段最小示例见
[TUTORIAL.md](TUTORIAL.md) 第 3 节（已实跑验证）。

## 保真度声明（哪些论文没有对应代码）

本项目是**教学实现**：只实现各论文支撑教程推导的最小核心算法，不是系统复现。
以下论文未建独立模块，理由与替代如下：

| 论文 | 不实现的原因 | 教学替代 |
|------|-------------|---------|
| ORB-SLAM 2 / 3 | 完整系统（多线程、Atlas、VI）工程量远超教学范围 | `epipolar/` + `bowloop/` 覆盖其前端与回环核心；ORB-SLAM3 架构对应表见教程第 10 章 |
| SLAM 权威综述（Cadena 2016） | 综述无算法 | 教程第 01 章导读 |
| NeRF-SLAM / GS-SLAM / SplaTAM / MonoGS | 需 GPU 神经渲染器 | 教程第 10 章 §10.4 路线表；3DGS 见 [3D 重建教程第 08 章](../../tutorials/3d_reconstruction/08_3D高斯泼溅.md) |
| MASt3R-SLAM / VGGT-GS SLAM | 需 MASt3R/VGGT 基础模型权重 | 同上，"基础模型 + SLAM"路线在教程第 10 章导读 |

## 目录结构

```
projects/slam/
├── README.md            # 本文件
├── TUTORIAL.md          # 零基础代码导读（必读入口）
├── DEBUG.md             # 调试与测试教程（40 条重点观察变量表）
├── METRICS.md           # 测评指标教程（ATE/RPE/尺度比 …）
├── metrics.py           # 统一测评模块
├── core|fastslam|epipolar|bowloop|direct|photoba|loam2d|preint|vins|droidlite/
│   └── *.py             # 各模块（中文注释含论文式号与教程式号）
└── tests/test_*.py      # 11 个测试文件，全部支持单点过滤
```

## 规范

遵循根目录 [CONSTRAINTS.md](../../CONSTRAINTS.md)：§4.3 中文注释（变量含义/作用/流水线）、
§4.4 TUTORIAL、§4.5 DEBUG、§4.7 METRICS。代码改动后请运行全局测试并同步四份文档。

# SLAM 论文精读 — 总索引

对 [papers/slam/](../../../papers/slam/README.md) 收录的 **18 篇论文逐一精读**：
每篇一份文档，包含论文信息、问题动机、方法总览、**关键公式推导**（符号表 + 逐式推导、
标注论文原文式号、每步注明依据）、实验解读、局限影响，以及与本项目
[十章教程](../README.md)和 [projects/slam/ 代码](../../../projects/slam/README.md)的双向对照。

**精读文档均基于本地 PDF 原文撰写**——式中号、数据集、实验数字逐页核对，
凡与二手描述不符处一律以原文为准（部分文档含对原论文排印错误的勘误注）。

## 经典论文（11 篇）

| 精读 | 论文（venue） | 一句话内核 | 教程章 | 代码 |
|------|--------------|-----------|--------|------|
| [FastSLAM](FastSLAM_AAAI2002.md) | FastSLAM（AAAI 2002） | Rao-Blackwell 粒子滤波：位姿粒子 × 独立路标 EKF | [07](../07_后端-i滤波与增量估计.md) | `fastslam/` |
| [PTAM](PTAM_ISMAR2007.md) | PTAM（ISMAR 2007） | 跟踪/建图双线程 + 关键帧 BA（现代 SLAM 架构雏形） | [05](../05_视觉里程计-i特征点法.md)、[08](../08_后端-ii图优化与-ba.md) | `epipolar/` |
| [ORB-SLAM](ORB-SLAM_TRO2015.md) | ORB-SLAM（T-RO 2015） | 特征点法集大成：三线程 + 共视图 + 本质图回环 | [05](../05_视觉里程计-i特征点法.md)、[08](../08_后端-ii图优化与-ba.md)、[09](../09_回环检测.md) | `epipolar/`、`bowloop/` |
| [ORB-SLAM2](ORB-SLAM2_TRO2016.md) | ORB-SLAM2（T-RO 2017） | 双目/RGB-D + 稠密地图输出 | [08](../08_后端-ii图优化与-ba.md)、[10](../10_建图与系统实战.md) | —— |
| [ORB-SLAM3](ORB-SLAM3_TRO2021.md) | ORB-SLAM3（T-RO 2021） | VI 紧耦合 + Atlas 多地图，实战基线 | [10](../10_建图与系统实战.md) | `preint/`、`epipolar/`、`bowloop/` |
| [SLAM 综述](SLAM-Survey_TRO2016.md) | Past, Present, Future of SLAM（T-RO 2016） | 统一概率框架 + 鲁棒感知 + 长期运行 | [01](../01_SLAM问题定义与全景.md) | —— |
| [LSD-SLAM](LSD-SLAM_ECCV2014.md) | LSD-SLAM（ECCV 2014） | 大规模半稠密直接法 + Sim(3) 位姿图 | [06](../06_视觉里程计-ii直接法.md) | `direct/` |
| [DSO](DSO_PAMI2018.md) | DSO（PAMI 2018） | 光度 BA：滑窗 + 仿射曝光 + 边缘化 | [06](../06_视觉里程计-ii直接法.md)、[08](../08_后端-ii图优化与-ba.md) | `photoba/` |
| [LOAM](LOAM_RSS2014.md) | LOAM（RSS 2014） | 激光：曲率特征 + 运动补偿 + 双速率 | [10](../10_建图与系统实战.md)（扩展） | `loam2d/` |
| [IMU 预积分](IMU-Preintegration_TRO2017.md) | On-Manifold Preintegration（T-RO 2017） | 预积分递推/协方差传播/偏置雅可比完整推导 | [10](../10_建图与系统实战.md) §10.2 | `preint/`、`vins/` |
| [VINS-Mono](VINS-Mono_RAM2018.md) | VINS-Mono（RAM 2018） | 紧耦合滑窗 VIO：边缘化 + 4-DOF 回环校正 | [10](../10_建图与系统实战.md) §10.3 | `vins/` |

## 前沿论文（7 篇）

| 精读 | 论文（venue） | 一句话内核 | 教程章 | 路线 |
|------|--------------|-----------|--------|------|
| [DROID-SLAM](DROID-SLAM_NeurIPS2021.md) | DROID-SLAM（NeurIPS 2021） | 学习光流 + 递归稠密 BA（可微 SLAM 里程碑） | [10](../10_建图与系统实战.md)（扩展） | 端到端可微 |
| [NeRF-SLAM](NeRF-SLAM_ICRA2023.md) | NeRF-SLAM（ICRA 2023） | VIO 先验 + 分层神经隐式场稠密建图 | [10](../10_建图与系统实战.md)、[NeRF 教程 03](../../nerf/03_数学原理与渲染方程.md) | 神经隐式 |
| [GS-SLAM](GS-SLAM_CVPR2024.md) | GS-SLAM（CVPR 2024 Highlight） | 3DGS 作为地图：可微渲染驱动跟踪与建图 | [10](../10_建图与系统实战.md)、[3D 重建 08](../../3d_reconstruction/08_3D高斯泼溅.md) | 3DGS |
| [SplaTAM](SplaTAM_CVPR2024.md) | SplaTAM（CVPR 2024） | 3DGS 当显式点云地图：深度渲染直接配准 | [10](../10_建图与系统实战.md)、[3D 重建 08](../../3d_reconstruction/08_3D高斯泼溅.md) | 3DGS |
| [MonoGS](MonoGS_CVPR2024.md) | Gaussian Splatting SLAM（CVPR 2024） | 首个单目实时 3DGS SLAM | [10](../10_建图与系统实战.md)、[3D 重建 08](../../3d_reconstruction/08_3D高斯泼溅.md) | 3DGS |
| [MASt3R-SLAM](MASt3R-SLAM_CVPR2025.md) | MASt3R-SLAM（CVPR 2025） | pointmap 先验 + 免标定增量式重建 | [10](../10_建图与系统实战.md)、[3D 重建 09](../../3d_reconstruction/09_哈希编码与前馈重建.md) | 基础模型先验 |
| [VGGT-GS SLAM](VGGT-GS-SLAM_arXiv2026.md) | VGGT-GS SLAM（arXiv 2026） | VGGT 前馈先验替代逐对推理的最新路线 | [10](../10_建图与系统实战.md)、[3D 重建 09](../../3d_reconstruction/09_哈希编码与前馈重建.md) | 基础模型先验 |

## 建议阅读顺序

- **按教程主线**（推荐）：先读 [十章教程](../README.md) 对应章，再读该章映射的论文精读；
  依赖链为 02→05→08→09→10（详见 [OVERVIEW](../OVERVIEW.md)）。
- **按历史脉络**：FastSLAM → PTAM → ORB 三部曲 → 直接法（LSD/DSO）→ VIO（预积分/VINS/ORB-SLAM3）
  → 前沿（DROID → NeRF-SLAM → 3DGS 三家 → 前馈先验）。
- **按公式难度**：入门（PTAM/ORB-SLAM）→ 进阶（FastSLAM/DSO/LSD）→ 困难
  （IMU 预积分/VINS-Mono/DROID）。

## 与教程、代码的分工

- **教程（../01–10 章）**：按概念组织，五步法推导，面向"学会原理"；
- **精读（本目录）**：按论文组织，逐篇核对原文式号与实验，面向"读懂论文"；
- **代码（projects/slam/）**：11 个模块的教学实现，公式 ↔ 函数一一对应（见各模块 docstring 与
  [DEBUG.md](../../../projects/slam/DEBUG.md) 变量表）。

三者的公式编号可能不同（各章独立编号、论文按原文编号），所有交叉引用均标注来源，回引前已核实。

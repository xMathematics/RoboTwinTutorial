# 3D 重建 — 主题总览（论文 / 教程 / 代码）

> 论文库：[papers/3d_reconstruction/](../../papers/3d_reconstruction/README.md)（经典 7 + 前沿 5 + 源码包）｜ 教程：[10 章](README.md)（架构已定）｜ 代码：规划中

## 章节与论文对应

| 章 | 教程 | 主读论文（本地 PDF） | 教材 / 工具支撑 |
|----|------|---------------------|----------------|
| 01 | [3D 重建问题与表示全景](01_3D重建问题与表示全景.md) | [MapAnything 前馈度量重建源码包](../../papers/3d_reconstruction/latex/arXiv-2509.13414_MapAnything_source.tar.gz) | —— |
| 02 | [多视图几何与 SfM](02_多视图几何与SfM.md) | [COLMAP-SfM (CVPR'16)](../../papers/3d_reconstruction/classics/COLMAP-SfM_CVPR2016_SchoenbergerFischer.pdf) | [SLAM 教程 04–08 章](../slam/README.md) |
| 03 | [多视图立体重建 MVS](03_多视图立体重建MVS.md) | [COLMAP-MVS (ECCV'16)](../../papers/3d_reconstruction/classics/COLMAP-MVS_ECCV2016_SchoenbergerFischer.pdf)、[MVSNet (ECCV'18)](../../papers/3d_reconstruction/classics/arXiv-1804.02505_MVSNet.pdf) | —— |
| 04 | [RGB-D 融合与 TSDF](04_RGB-D融合与TSDF.md) | KinnectFusion（[延伸阅读](../../papers/3d_reconstruction/README.md)）、BundleFusion（延伸阅读） | Curless & Levoy 1996 |
| 05 | [从体素到网格：表面提取](05_从体素到网格表面提取.md) | [Poisson 重建 (SGP'06)](../../papers/3d_reconstruction/classics/PoissonSurfaceReconstruction_SGP2006_KazhdanBolithoHoppe.pdf) | Marching Cubes（延伸阅读） |
| 06 | [学习式形状表示：SDF 与占据](06_学习式形状表示SDF与占据.md) | [DeepSDF (CVPR'19)](../../papers/3d_reconstruction/classics/arXiv-1901.05103_DeepSDF.pdf)、[Occupancy Networks (CVPR'19)](../../papers/3d_reconstruction/classics/arXiv-1812.03828_OccupancyNetworks.pdf) | —— |
| 07 | [神经隐式表面重建](07_神经隐式表面重建.md) | [NeuS (NeurIPS'21)](../../papers/3d_reconstruction/classics/arXiv-2106.10689_NeuS.pdf)、[Neuralangelo (CVPR'23)](../../papers/3d_reconstruction/frontier/arXiv-2306.03092_Neuralangelo.pdf) | [NeRF 教程 03 章](../nerf/03_数学原理与渲染方程.md) |
| 08 | [3D 高斯泼溅](08_3D高斯泼溅.md) | [3D Gaussian Splatting (SIGGRAPH'23)](../../papers/3d_reconstruction/frontier/arXiv-2308.04079_3D-Gaussian-Splatting.pdf) | EWA splatting (Zwicker 2001) |
| 09 | [哈希编码与前馈重建](09_哈希编码与前馈重建.md) | [Instant-NGP (SIGGRAPH'22)](../../papers/3d_reconstruction/frontier/arXiv-2201.05989_Instant-NGP.pdf)、[DUSt3R (CVPR'24)](../../papers/3d_reconstruction/frontier/arXiv-2312.14132_DUSt3R.pdf)、[MASt3R (ECCV'24)](../../papers/3d_reconstruction/frontier/arXiv-2406.09756_MASt3R.pdf) | [NeRF 教程 04 章](../nerf/04_网络架构与位置编码.md) |
| 10 | [机器人场景中的重建实战与选型](10_机器人场景中的重建实战与选型.md) | [DexGraspNet (NeurIPS'22)](../../papers/robotwin/frontier/arXiv-2210.02697_DexGraspNet.pdf)、GraspNet-1Billion（[延伸阅读](../../papers/3d_reconstruction/README.md)） | [RoboTwin 教程](../robotwin/README.md) |

## 与其他主题的关系

- **SLAM**：重建要用的相机位姿来自 [SLAM 教程](../slam/OVERVIEW.md)（第 05–08 章估计；第 10 章稠密建图与本主题第 04 章是同一件事的两面）；
- **NeRF**：本主题是 NeRF 的"下游"——体渲染 → NeuS 表面化 → 3DGS 提速，一条演进链；
- **RoboTwin**：第 10 章把重建结果接入抓取与仿真（数字孪生），见 [RoboTwin 教程](../robotwin/README.md)。

## 代码状态

`projects/3d_reconstruction/` 尚未建立。规划：TSDF 融合 + Marching Cubes 的教学实现
（纯 NumPy 可运行，衔接 projects/nerf 的数据生成脚本），以及 3DGS 最小可微光栅化器。

# SLAM — 主题总览（论文 / 教程 / 代码）

> 论文库：[papers/slam/](../../papers/slam/README.md)（经典 11 + 前沿 7）｜ 教程：[10 章](README.md)（架构已定）｜ 代码：规划中

## 章节与论文对应

| 章 | 教程 | 主读论文（本地 PDF） | 教材支撑 |
|----|------|---------------------|---------|
| 01 | [SLAM 问题定义与全景](01_SLAM问题定义与全景.md) | [SLAM 权威综述](../../papers/slam/classics/arXiv-1606.05830_SLAM-Survey-Cadena.pdf)、[FastSLAM](../../papers/slam/classics/FastSLAM_AAAI2002_MontemerloThrun.pdf) | 概率机器人学 ch1–2 |
| 02 | [三维刚体运动：旋转与位姿](02_三维刚体运动旋转与位姿.md) | ——（工具章，教材为主） | 视觉 SLAM 十四讲 ch3–4 |
| 03 | [概率状态估计基础](03_概率状态估计基础.md) | [FastSLAM](../../papers/slam/classics/FastSLAM_AAAI2002_MontemerloThrun.pdf)（对比阅读） | 概率机器人学 ch2–3 |
| 04 | [相机模型与特征提取](04_相机模型与特征提取.md) | [ORB-SLAM](../../papers/slam/classics/arXiv-1502.00956_ORB-SLAM.pdf)（特征部分） | 十四讲 ch4–5 |
| 05 | [视觉里程计 I：特征点法](05_视觉里程计-i特征点法.md) | [PTAM](../../papers/slam/classics/PTAM_ISMAR2007_KleinMurray.pdf)、[ORB-SLAM](../../papers/slam/classics/arXiv-1502.00956_ORB-SLAM.pdf) | 十四讲 ch7 |
| 06 | [视觉里程计 II：直接法](06_视觉里程计-ii直接法.md) | [LSD-SLAM](../../papers/slam/classics/LSD-SLAM_ECCV2014_EngelSchopsCremers.pdf)、[DSO](../../papers/slam/classics/arXiv-1607.02565_DSO.pdf) | 十四讲 ch8 |
| 07 | [后端 I：滤波与增量估计](07_后端-i滤波与增量估计.md) | [FastSLAM](../../papers/slam/classics/FastSLAM_AAAI2002_MontemerloThrun.pdf) | 概率机器人学 ch10 |
| 08 | [后端 II：图优化与 BA](08_后端-ii图优化与-ba.md) | [ORB-SLAM2](../../papers/slam/classics/arXiv-1610.06475_ORB-SLAM2.pdf) | 十四讲 ch9–10 |
| 09 | [回环检测](09_回环检测.md) | [ORB-SLAM](../../papers/slam/classics/arXiv-1502.00956_ORB-SLAM.pdf)（词袋部分） | 十四讲 ch11–12 |
| 10 | [建图与系统实战](10_建图与系统实战.md) | [ORB-SLAM3](../../papers/slam/classics/arXiv-2007.11898_ORB-SLAM3.pdf)、[VINS-Mono](../../papers/slam/classics/arXiv-1708.03852_VINS-Mono.pdf)、[IMU 预积分](../../papers/slam/classics/arXiv-1512.02363_IMU-Preintegration.pdf)、[LOAM](../../papers/slam/classics/LOAM_RSS2014_ZhangSingh.pdf) | 十四讲 ch13–14 |
| 扩展 | 前沿导读（第 10 章末） | [DROID-SLAM](../../papers/slam/frontier/arXiv-2108.10869_DROID-SLAM.pdf)、[NeRF-SLAM](../../papers/slam/frontier/arXiv-2210.13641_NeRF-SLAM.pdf)、[GS-SLAM](../../papers/slam/frontier/arXiv-2311.11700_GS-SLAM.pdf)、[SplaTAM](../../papers/slam/frontier/arXiv-2312.02126_SplaTAM.pdf)、[MonoGS](../../papers/slam/frontier/arXiv-2312.06741_MonoGS-Gaussian-Splatting-SLAM.pdf)、[MASt3R-SLAM](../../papers/slam/frontier/arXiv-2412.12392_MASt3R-SLAM.pdf)、[VGGT-GS SLAM](../../papers/slam/frontier/arXiv-2609.19628_VGGT-GS-SLAM.pdf) | —— |

## 与其他主题的关系

- **3D 重建**：第 10 章稠密建图（TSDF）与 [3D 重建教程第 04 章](../3d_reconstruction/04_RGB-D融合与TSDF.md)交汇；
- **NeRF**：神经隐式建图（NeRF-SLAM）需要 [NeRF 教程](../nerf/OVERVIEW.md)的体渲染基础；
- **RoboTwin**：SLAM 是机器人自主性的感知底座，仿真落地方向见 [RoboTwin 教程](../robotwin/README.md)。

## 代码状态

`projects/slam/` 尚未建立。规划：ORB-SLAM3 数据集实战（第 10 章实战产出）跑通后，
收录配置文件与"理论↔代码模块"对应表。教材配套代码见 [slambook2](https://github.com/gaoxiang12/slambook2)。

# papers/slam/ — SLAM 论文库

按 **经典（classics）/ 前沿（frontier）** 分类。经典 = 奠定性工作与当前实战基线；
前沿 = 研究热点（深度学习、神经隐式、3D 高斯、前馈基础模型路线）。
论文与教程章节的对应关系见 [tutorials/slam/](../../tutorials/slam/README.md)。

## 经典论文（classics/）

| 文件 | 论文（年份 / 发表处） | 为什么重要 | 对应教程 |
|------|----------------------|-----------|---------|
| [FastSLAM_AAAI2002](classics/FastSLAM_AAAI2002_MontemerloThrun.pdf) | FastSLAM（2002 / AAAI） | 用 Rao-Blackwellized 粒子滤波分解 SLAM，大地图滤波路线奠基 | 07 |
| [PTAM_ISMAR2007](classics/PTAM_ISMAR2007_KleinMurray.pdf) | PTAM（2007 / ISMAR） | 首创跟踪/建图双线程 + 关键帧，现代 SLAM 架构雏形 | 05、08 |
| [arXiv-1502.00956](classics/arXiv-1502.00956_ORB-SLAM.pdf) | ORB-SLAM（2015 / T-RO） | 特征点法集大成：三线程架构成为行业标准范式 | 全书主线 |
| [arXiv-1610.06475](classics/arXiv-1610.06475_ORB-SLAM2.pdf) | ORB-SLAM2（2016 / T-RO） | 扩展双目 / RGB-D，加入 BA 稠密地图输出 | 10 |
| [arXiv-1407.7530](classics/arXiv-1407.7530_LSD-SLAM.pdf) | LSD-SLAM（2014 / ICCV） | 大规模直接法开山之作，半稠密建图 | 06 |
| [arXiv-1607.02565](classics/arXiv-1607.02565_DSO.pdf) | DSO（2018 / PAMI） | 直接稀疏里程计：光度bundle adjustment，滑动窗口优化典范 | 06、08 |
| [LOAM_RSS2014](classics/LOAM_RSS2014_ZhangSingh.pdf) | LOAM（2014 / RSS） | 激光 SLAM 奠基，高频里程计 + 低频建图的双速率设计 | 10（扩展） |
| [arXiv-1512.02363](classics/arXiv-1512.02363_IMU-Preintegration.pdf) | On-Manifold Preintegration（2017 / T-RO） | IMU 预积分理论：VIO 紧耦合的数学基石，推导范本 | 10 |
| [arXiv-1708.03852](classics/arXiv-1708.03852_VINS-Mono.pdf) | VINS-Mono（2018 / RAM） | 单目 VIO 完整开源系统，预积分的工程实现标杆 | 10 |
| [arXiv-2007.11898](classics/arXiv-2007.11898_ORB-SLAM3.pdf) | ORB-SLAM3（2021 / T-RO） | 视觉 / 视觉惯性 / 多地图统一框架，当前实战对比基线 | 10 |

**延伸阅读（未收录 PDF，需自行获取）**：

- Smith, Self & Cheeseman, *Estimating Uncertain Spatial Relationships in Robotics*（1986/88）——SLAM 概念起点（随机地图 / EKF），[ACM DL](https://dl.acm.org/doi/10.5555/3023712.3023749)
- Forster et al., *SVO: Fast Semi-Direct Monocular Visual Odometry*（2014 / ICRA）——半直接法，无 arXiv 版，[作者页](https://rpg.ifi.uzh.ch/publications.html)
- Thrun & Montemerlo, *The Graph SLAM Algorithm...*（2006 / IJRR）——图优化路线综述
- Gálvez-López & Tardós, *DBoW2*（2012 / T-RO）——回环检测词袋库

## 前沿论文（frontier/）

| 文件 | 论文（年份 / 发表处） | 为什么重要 | 对应教程 |
|------|----------------------|-----------|---------|
| [arXiv-2108.10869](frontier/arXiv-2108.10869_DROID-SLAM.pdf) | DROID-SLAM（2021 / NeurIPS） | 深度学习 SLAM 里程碑：端到端可微递归 BA | 10（扩展） |
| [arXiv-2210.13641](frontier/arXiv-2210.13641_NeRF-SLAM.pdf) | NeRF-SLAM（2023 / ICRA） | 神经隐式表示进入 SLAM，稠密建图新范式 | 10、[NeRF 主题](../nerf/) |
| [arXiv-2311.11700](frontier/arXiv-2311.11700_GS-SLAM.pdf) | GS-SLAM（2024 / CVPR Highlight） | 3D 高斯泼溅实时 SLAM 的代表工作之一 | 10（扩展） |
| [arXiv-2312.02126](frontier/arXiv-2312.02126_SplaTAM.pdf) | SplaTAM（2024 / CVPR） | 3DGS 作为显式地图的稠密 RGB-D SLAM，基准常客 | 10（扩展） |
| [arXiv-2312.06741](frontier/arXiv-2312.06741_MonoGS-Gaussian-Splatting-SLAM.pdf) | Gaussian Splatting SLAM（2024 / CVPR） | 首个实时纯 RGB 的 3DGS SLAM | 10（扩展） |
| [arXiv-2412.12392](frontier/arXiv-2412.12392_MASt3R-SLAM.pdf) | MASt3R-SLAM（2025 / CVPR） | 前馈 3D 重建先验（pointmap）做实时稠密 SLAM，"基础模型 + SLAM"路线代表 | 10（扩展） |
| [arXiv-2609.19628](frontier/arXiv-2609.19628_VGGT-GS-SLAM.pdf) | VGGT-GS SLAM（2026 / arXiv） | 无标定单目前馈先验 + 高斯建图，路线最新进展 | 10（扩展） |

## 使用约定

1. 新增论文放入对应分类子目录，命名：arXiv 论文用 `arXiv-<id>_<短名>.pdf`，非 arXiv 用 `<短名>_<venue><年份>_<一作>.pdf`
2. 收录后在本表加行，注明"为什么重要"与对应教程章节
3. 下载自 arXiv 的版本可能为最新修订版，引用时以期刊 / 会议正式版为准

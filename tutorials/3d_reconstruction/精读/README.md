# 3D 重建论文精读 — 总索引

对 [papers/3d_reconstruction/](../../../papers/3d_reconstruction/README.md) 收录的 **12 篇论文 + 1 个源码包**逐一精读：
每篇一份文档，包含论文信息、问题动机、方法总览、**关键公式推导**（符号表 + 逐式推导、
标注论文原文式号、每步注明依据）、实验解读、局限影响，以及与本项目
[十章教程](../README.md)的双向对照（论文式号 ↔ 教程 (章.序) 映射表）。

**精读文档均基于本地 PDF/LaTeX 原文撰写**——式号、实验口径逐页核对；
MVSNet/NeuS 与本地 LaTeX 源码双重交叉核对；发现的论文排印/口径问题在文中加勘误注。

## 按重建管线排列（13 篇）

### 几何估计：SfM / MVS（位姿 + 稠密深度）

| 精读 | 论文（venue） | 一句话内核 | 教程章 |
|------|--------------|-----------|--------|
| [COLMAP-SfM](COLMAP-SfM_CVPR2016.md) | SfM Revisited（CVPR 2016） | 增量式 SfM 集大成：场景图 + 下一最优视图 + 冗余视图分组 BA | [02](../02_多视图几何与SfM.md) |
| [COLMAP-MVS](COLMAP-MVS_ECCV2016.md) | Pixelwise View Selection（ECCV 2016） | 几何先验 + 光度遮挡的联合视图采样与深度图融合 | [03](../03_多视图立体重建MVS.md) |
| [MVSNet](MVSNet_ECCV2018.md) | MVSNet（ECCV 2018） | 平面扫描代价体 + 3D CNN 正则 + 软 argmin | [03](../03_多视图立体重建MVS.md) |

### RGB-D 融合与表面提取（经典几何路线）

| 精读 | 论文（venue） | 一句话内核 | 教程章 |
|------|--------------|-----------|--------|
| [Poisson 重建](PoissonRecon_SGP2006.md) | Poisson Surface Reconstruction（SGP 2006） | 点云+法向 → 指示函数梯度场 → 全局 Poisson 解 | [05](../05_从体素到网格表面提取.md) |

### 学习式形状表示

| 精读 | 论文（venue） | 一句话内核 | 教程章 |
|------|--------------|-----------|--------|
| [Occupancy Networks](OccupancyNetworks_CVPR2019.md) | ONet（CVPR 2019） | 连续占据隐函数 + Lambert-W 非饱和损失 | [06](../06_学习式形状表示SDF与占据.md) |
| [DeepSDF](DeepSDF_CVPR2019.md) | DeepSDF（CVPR 2019） | 条件 SDF 隐函数 + 自解码器隐空间补全 | [06](../06_学习式形状表示SDF与占据.md) |

### 神经渲染与隐式表面

| 精读 | 论文（venue） | 一句话内核 | 教程章 |
|------|--------------|-----------|--------|
| [NeuS](NeuS_NeurIPS2021.md) | NeuS（NeurIPS 2021） | SDF→体渲染权重的无偏转换（telescoping 闭合形式） | [07](../07_神经隐式表面重建.md) |
| [Neuralangelo](Neuralangelo_CVPR2023.md) | Neuralangelo（CVPR 2023） | 数值梯度平滑 + 哈希分辨率渐进恢复大尺度几何 | [07](../07_神经隐式表面重建.md) §7.4 |
| [Instant-NGP](InstantNGP_SIGGRAPH2022.md) | Instant-NGP（SIGGRAPH 2022） | 多分辨率哈希编码：全层级并行激活，训练提速两个量级 | [09](../09_哈希编码与前馈重建.md) §9.1 |
| [3D Gaussian Splatting](3DGS_SIGGRAPH2023.md) | 3DGS（SIGGRAPH 2023） | 显式各向异性高斯 + 实时可微光栅化 | [08](../08_3D高斯泼溅.md) |

### 前馈基础模型路线（免标定一次推理）

| 精读 | 论文（venue） | 一句话内核 | 教程章 |
|------|--------------|-----------|--------|
| [DUSt3R](DUSt3R_CVPR2024.md) | DUSt3R（CVPR 2024） | pointmap 回归：两视图免标定前馈重建 | [09](../09_哈希编码与前馈重建.md) §9.2 |
| [MASt3R](MASt3R_ECCV2024.md) | MASt3R（ECCV 2024） | pointmap + InfoNCE 匹配头 + 快速互匹配 | [09](../09_哈希编码与前馈重建.md) §9.3 |
| [MapAnything](MapAnything_arXiv2025.md) | MapAnything（3DV 2026） | 交替注意力一次前馈 N 视图度量重建（无需后处理对齐） | [09](../09_哈希编码与前馈重建.md)、[01](../01_3D重建问题与表示全景.md) |

> MapAnything 的本地资料为 [LaTeX 源码包](../../../papers/3d_reconstruction/latex/arXiv-2509.13414_MapAnything/)（arXiv:2509.13414，已被 3DV 2026 录用）。

## 论文 × 代码状态

图例：**✅ 完整实现**｜**🔶 教学代理**（保留论文几何内核、简化工程外壳，简化项逐条写在
模块 docstring 与 [projects/3d_reconstruction/README.md](../../../projects/3d_reconstruction/README.md)
保真度声明）｜**❌ 未实现**。13 篇中无一"完整复现"——项目的定位是教学实现，不追
论文系统规模；TSDF（Curless & Levoy 1996）等教程主线算法对应
[projects/3d_reconstruction](../../../projects/3d_reconstruction/) 的 `tsdf.py`/
`marching.py`/`metrics.py`，不属本表 13 篇。

| 论文 | 状态 | 代码位置与说明 |
|------|------|----------------|
| COLMAP-SfM | ❌ | 增量式 SfM 系统工程（特征匹配 + 光束法平差 + 下一最优视图）不做；其几何核心（对极几何、三角化、PnP、BA）由 [projects/slam](../../../projects/slam/) 的 `epipolar`/`photoba` 等模块按 SLAM 教程 05–08 章覆盖 |
| COLMAP-MVS | 🔶 | [plane_sweep.py](../../../projects/3d_reconstruction/plane_sweep.py)：平面扫描单应 warp (3.2) + NCC (3.4) + 置信度取优；简化掉 patch 级 (深度, 法向) 联合估计、几何一致性视图选择与深度图融合 |
| MVSNet | ❌ | 3D CNN 代价体正则 + 软 argmin 需 GPU 训练；其平面扫描几何内核由 `plane_sweep.py` 代理，置信度用最优/次优代价差近似 P(d) 峰度 |
| Poisson 重建 | 🔶 | [poisson.py](../../../projects/3d_reconstruction/poisson.py)：法向涂抹（原文式 (2)）→ 散度/求解 (5.6) → 等值面 (5.8)；简化为均匀网格 + FFT 周边界频域求解（原文为自适应八叉树 + 多重网格），等值面水平取原文 §4.4 的样本均值 |
| Occupancy Networks | ❌ | 占据隐函数 + Lambert-W 非饱和损失需神经网络训练（GPU 优化器 + 占据真值）；`scene.py` 的解析 SDF 是 (6.1) Eikonal 的精确载体 |
| DeepSDF | ❌ | 条件 SDF 隐函数 + 自解码器需 GPU 训练与距离变换真值；同上以解析 SDF 承载几何语义 |
| NeuS | ❌ | SDF→体渲染权重无偏转换的逐场景优化需 GPU（逐光线查询网络）；体渲染基线见 [projects/nerf](../../../projects/nerf/) |
| Neuralangelo | ❌ | 数值梯度平滑 + 哈希分辨率渐进需 GPU 训练；推导见教程 07 §7.4 与精读 |
| Instant-NGP | ❌ | 多分辨率哈希编码的训练管线需 GPU；几何/渲染语义由本项目的解析模块承载 |
| 3D Gaussian Splatting | 🔶 | [splatting.py](../../../projects/3d_reconstruction/splatting.py)：前向渲染链 (8.4)-(8.10)（EWA 投影、深度排序、alpha 合成）；优化侧（可微光栅化反传、致密化/剪枝、球谐）需自动微分，不做 |
| DUSt3R | ❌ | pointmap 回归需基础模型权重推理；其要替代的手工基线（位姿已知稠密深度）即 `plane_sweep.py` |
| MASt3R | ❌ | pointmap + 匹配头需基础模型权重推理 |
| MapAnything | ❌ | 交替注意力前馈 N 视图度量重建需基础模型权重推理 |


## 建议阅读顺序

- **按教程主线**：[十章教程](../README.md) 的 02→03（几何估计）、04→05（融合提取）、06→07→08→09（学习式演进），每章读完即读对应精读；
- **按历史脉络**：Poisson → COLMAP 双篇 → MVSNet → ONet/DeepSDF → NeuS → Instant-NGP/3DGS → Neuralangelo → DUSt3R/MASt3R → MapAnything；
- **按目标**：想做 SfM/MVS 读前三篇；做抓取/仿真资产读 Poisson+ONet/DeepSDF；做神经渲染读 NeuS/3DGS；做快速三维感知读最后三篇。

## 与教程、SLAM 精读的关系

- **教程（../01–10 章）**：按概念组织、五步法推导（论文式号 ↔ 教程式号的映射表在各精读 §7）；
- **本目录**：按论文组织，逐篇核对原文（多处纠正二手描述与论文口径不符）；
- **[SLAM 精读](../../slam/精读/README.md)**：NeRF-SLAM/GS-SLAM/SplaTAM/MonoGS/MASt3R-SLAM 五篇 3DGS/前馈系 SLAM 的精读在 SLAM 目录，与本目录的第 08/09 章精读互为上下游。

# projects/3d_reconstruction — 3D 重建教学实现

把 [tutorials/3d_reconstruction/](../../tutorials/3d_reconstruction/README.md) 中
**MVS 平面扫描（第 03 章）→ RGB-D 融合（第 04 章）→ 表面提取（第 05 章，MT +
Poisson 代理）→ SDF 表示（第 06 章）→ 3DGS 前向泼溅（第 08 章）** 的核心算法
写成能跑的教学代码：纯 NumPy、无新依赖、自包含（不 import 其他项目）、每个模块带
确定性测试，公式与教程逐式对应（单应 warp (3.1)-(3.2)、NCC (3.4)、截断 (4.3)、
递归融合 (4.8)、棱上插值 (5.1)、梯度法向 (5.2)、Poisson 方程 (5.6)、EWA 投影
(8.6)/(8.8)、alpha 合成 (8.10)）。

**零基础入门请先读 [TUTORIAL.md](TUTORIAL.md)**（每个模块的最小可运行示例 + 数据结构 + 流水线图）；
调试与单点/全局测试用法见 [DEBUG.md](DEBUG.md)；评估口径见 [METRICS.md](METRICS.md)。

## 快速开始

```bash
# ① 激活环境（环境清单见根目录 environment.yml；纯 numpy，任意 numpy>=1.26 的环境亦可）
conda activate llm_env

# ② 依赖校验：本包仅依赖 numpy（无 torch/scipy/opencv）
python -c "import numpy; print('numpy', numpy.__version__)"        # 预期: numpy 2.2.6

# ③ 场景模块测试（SDF 基元/球追踪渲染/位姿闭环，11 个测试）
python tests/test_scene.py

# ④ 核心入口冒烟演示（场景 → 24 视角深度 → TSDF → MT 网格 → 指标，确定性输出）
python demo.py

# ⑤ 三个论文对应模块的最小命令（教学代理，详见 TUTORIAL.md §3.6-3.8）
python tests/test_poisson.py sphere        # Poisson 表面重建：球点云+法向 → 网格（5 个测试）
python tests/test_plane_sweep.py depth     # 平面扫描立体：RGB 多视角 → 深度图（6 个测试）
python tests/test_splatting.py silhouette  # 3DGS 前向泼溅：球面高斯 → 图像（5 个测试）
```

- **全局测试**：`python -m pytest tests/ -q`（55 个测试，系统 python3 实测 ~16.6 s，
  llm_env ~15.5 s）
- **单点测试**：`python tests/test_marching.py normals`（子串过滤，无匹配会列出可用测试名）
- VS Code 调试配置见 [.vscode/launch.json](../../.vscode/launch.json)，用法见 [.vscode/SETUP.md](../../.vscode/SETUP.md)

## 模块总览（教程章 × 模块 × 核心内容 × 测试数）

| 模块 | 教程章 | 核心内容 | 测试 |
|------|--------|---------|------|
| [scene.py](scene.py) | 01（SDF 约定）、04 §4.3（光线投射）、06 §6.1（Eikonal） | 球/盒/平面 SDF 基元（外正内负，(1.13)）、min 并集场景、解析最近点与真值面采样、球追踪（sphere tracing, Hart 1996）深度渲染、轨道双环位姿、σ(z)=c·z² 深度噪声（(4.2)）、世界坐标程序化纹理（棋盘格/值噪声）+ RGB 渲染（供平面扫描） | 11 |
| [tsdf.py](tsdf.py) | **04（RGB-D 融合与 TSDF）** | TSDF 体素网格 + Curless & Levoy 1996 加权积分：离散观测 s≈z(u)−zₓ（(4.1)）、逆方差权重 w∝1/z²（(4.2)）、截断 ±1（(4.3)）、递归融合 (4.8)；三线性插值查询（(4.13)/(1.15)）、截断符号距离、权重掩码提取场、深度→点云 | 8 |
| [marching.py](marching.py) | **05（从体素到网格表面提取）** | Marching Tetrahedra：立方体 6 四面体拆分、16 case 手写表、棱上插值 t=fₐ/(fₐ−f_b)（(5.1)）、全局棱键顶点去重（水密）、梯度法向（(5.2)）、外法向取向统一 | 10 |
| [poisson.py](poisson.py) | **05 §5.3（Poisson 表面重建，压轴）** | 教学代理（Kazhdan et al., SGP 2006）：带法向点云 → 高斯核涂抹向量场（原文式 (2)）→ 散度 (5.6) → **FFT 周边界频域求解 ∇²χ=∇·V**（原文自适应八叉树+多重网格的教学简化）→ γ−χ 交给 marching 提取 (5.8)；指示函数尺度归一 + 样本处等值面水平（原文 §4.4） | 5 |
| [plane_sweep.py](plane_sweep.py) | **03（多视图立体重建 MVS）** | 教学代理（COLMAP-MVS 几何内核 / MVSNet 代价体的手工版）：相对位姿 → 深度假设平面单应 H(d)=K(R_rel−t_rel nᵀ/d)K⁻¹（(3.1)-(3.2)）→ 邻图双线性 warp → NCC 光度一致性（(3.4)，对仿射光照不变 (3.5)）→ 逐像素取优 + 置信度（最优/次优代价差）；COLMAP 的 (深度,法向) 联合估计与视图选择、MVSNet 的 3D CNN 正则不做 | 6 |
| [splatting.py](splatting.py) | **08（3D 高斯泼溅，前向渲染侧）** | 教学代理（Kerbl et al., SIGGRAPH 2023）：球面切平面各向同性 2D 高斯 → EWA 投影（雅可比 (8.6)、协方差传播 Σ'=JWΣWᵀJᵀ (8.8)）→ 深度排序 → 逐像素 alpha 合成（(8.9)-(8.10)，bbox 截断求值）；**优化侧（可微光栅化、致密化/剪枝、球谐）不做** | 5 |
| [metrics.py](metrics.py) | 评测惯例（DTU / Tanks and Temples） | 对称 Chamfer（(M.1)）、DTU accuracy/completion（(M.2)）、F-score@τ（(M.3)）、面积加权表面采样（(M.4)），全部纯函数 | 10 |
| [demo.py](demo.py) | 04 + 05 端到端 | 冒烟主管线：球+盒场景 → 24 视角深度 → TSDF 融合 → MT 网格 → 与解析真值表面点比 Chamfer/Acc/Comp/F | — |

各模块健康值与指标口径见 [METRICS.md](METRICS.md)；全部 8 段最小示例见
[TUTORIAL.md](TUTORIAL.md) 第 3 节（已实跑验证）。

## 运行示例（重建管线，30 秒上手）

```python
import numpy as np
from scene import make_demo_scene, render_dataset
from tsdf import TSDFVolume
from marching import marching_tetrahedra
from metrics import chamfer_distance, f_score, sample_mesh_surface, sample_scene_surface

scene = make_demo_scene()                                 # 球 r=0.35 + 盒半边 0.15
data = render_dataset(scene, n_views=24, seed=0)          # 24 视角深度（σ(z)=0.002·z²）
vol = TSDFVolume(np.full(3, -0.8), np.full(3, 0.8), resolution=64)
for T_wc, depth in zip(data["poses"], data["depths"]):
    vol.integrate(depth, data["K"], T_wc)                 # 逐帧融合（教程 (4.8)）
mesh = marching_tetrahedra(vol.extraction_field(),        # MT 提取（教程 (5.1)/(5.2)）
                           vol.origin + 0.5 * vol.voxel_size, vol.voxel_size)
rec = sample_mesh_surface(mesh.vertices, mesh.triangles, 8192, seed=0)
gt = sample_scene_surface(scene, 8192, band=0.05, seed=0) # 解析真值表面点
print(len(mesh.vertices), len(mesh.triangles),            # 实测 15076 / 29421
      round(chamfer_distance(rec, gt), 6),                # 实测 0.000128 m^2
      round(f_score(rec, gt, 0.02), 3))                   # 实测 0.960
```

单视角深度点云对照（`tsdf.depth_to_point_cloud`）：Chamfer 0.027 m²、F@20mm 0.315
——多视角融合把完整度抬起来（METRICS.md §2 有逐项解读）。三个论文对应模块的
同款对比数字：Poisson vs TSDF 在球场景上的 Chamfer 见 `tests/test_poisson.py`
（0.001 vs 0.002 m²）、plane_sweep 融合 vs 单视角（0.0019 vs 0.018 m²，置信度
过滤后）见 `tests/test_plane_sweep.py`。

## 保真度声明（哪些章节内容没有对应代码）

本项目是**教学实现**：只实现支撑教程推导的最小可算核心，不是系统复现。
三个论文对应模块（`poisson.py` / `plane_sweep.py` / `splatting.py`）均为**教学代理**
——保留论文的几何内核、简化工程外壳（各自 docstring 顶部有逐条简化声明）；
以下教程内容未建模块，理由与替代如下：

| 教程内容 | 不实现的原因 | 教学替代 / 映射 |
|----------|-------------|---------|
| 第 02 章 COLMAP-SfM（增量式重建系统） | 特征提取匹配 + 光束法平差 + 下一最优视图的完整系统工程远超教学范围 | 几何核心（对极几何、三角化、PnP、BA）由 [projects/slam](../slam/) 的 `epipolar`/`photoba` 等模块覆盖（SLAM 教程 05-08 章）；场景图与增量式重建策略见教程原文 |
| 第 03 章 MVS 的工程外壳（COLMAP 的 (深度,法向) 联合 patch 匹配、几何一致性像素级视图选择、深度图融合） | 需要真实多视图图像与大规模系统工程 | **平面扫描几何内核已建**：[plane_sweep.py](plane_sweep.py)（单应 warp (3.2) + NCC (3.4) + 置信度）；视图选择与融合策略见教程 3.2 节，融合入口即本项目的 `tsdf.TSDFVolume` |
| 第 03 章 MVSNet（3D CNN 代价体正则 + 软 argmin） | 需 GPU 训练（ECCV 2018，O(HWD) 代价体 + 3D U-Net） | 可微 warp 的几何同款即 (3.2)；plane_sweep 的最优/次优代价差是软 argmin 置信度（教程 3.3 ⑤(b)）的硬 argmin 手工版；网络侧见教程原文与精读 |
| 第 05 章 Poisson 的工程实现（自适应八叉树 + 多级求解器，Kazhdan et al. 2006 §4） | 稀疏线性系统 + 多重网格的工程量远超教学篇幅 | **几何内核已建**：[poisson.py](poisson.py)（涂抹 (5.5)→散度/求解 (5.6)→提取 (5.8)），求解用 FFT 周边界教学简化，差异逐条见其模块 docstring |
| 第 06 章 DeepSDF / Occupancy Networks 训练 | 需 GPU 优化器与距离变换/占据真值数据 | `scene.py` 的解析 SDF 即 (6.1) Eikonal 的精确载体（`tests/test_scene.py::test_eikonal_property_of_analytic_sdf` 数值验证 ‖∇s‖=1）；网络化部分见教程原文 |
| 第 07 章 NeuS / Neuralangelo | 需 GPU 逐场景优化（逐光线查询神经 SDF + 神经网络正则） | 体渲染基线见 [projects/nerf](../nerf/)；SDF→权重的无偏转换等推导见教程 07 章与精读 |
| 第 08 章 3DGS 的优化侧（可微光栅化反传、自适应致密化/剪枝、球谐视角色） | 需要自动微分框架与 GPU 光栅化 | **前向渲染已建**：[splatting.py](splatting.py)（(8.4)-(8.10) 完整前向链）；训练循环与致密化规则（教程 8.3，论文经验规则）见教程原文 |
| 第 09 章 Instant-NGP / DUSt3R / MASt3R / MapAnything | 需 GPU 训练（哈希编码）或基础模型权重（前馈大模型推理） | 教程各章原文与精读；`plane_sweep.py` 覆盖"前馈重建要替代的手工基线"（位姿已知的稠密深度） |
| 第 10 章 系统实战与选型 | 综述/工程选型，无单一可复现算法 | 教程原文；本项目 demo 即"机器人桌面场景重建"的最小管线示例 |

## 目录结构

```
projects/3d_reconstruction/
├── README.md            # 本文件
├── TUTORIAL.md          # 零基础代码导读（必读入口）
├── DEBUG.md             # 调试与测试教程（重点观察变量表）
├── METRICS.md           # 测评指标教程（Chamfer / Acc / Comp / F-score）
├── metrics.py           # 统一测评模块（公式 (M.1)-(M.4)）
├── scene.py             # SDF 基元 + 球追踪渲染 + 数据集 + 真值采样 + 程序化纹理 RGB
├── tsdf.py              # TSDF 体素网格 + Curless-Levoy 融合 + 查询
├── marching.py          # Marching Tetrahedra 等值面提取
├── poisson.py           # Poisson 表面重建教学代理（05 §5.3，FFT 频域求解）
├── plane_sweep.py       # 平面扫描立体的教学代理（03 章，单应 warp + NCC）
├── splatting.py         # 3DGS 前向泼溅的教学代理（08 章，EWA 投影 + alpha 合成）
├── demo.py              # 核心入口冒烟（VS Code 调试配置指向本文件）
└── tests/test_*.py      # 7 个测试文件，55 个测试，全部支持单点过滤
```

## 规范

遵循根目录 [CONSTRAINTS.md](../../CONSTRAINTS.md)：§4.3 中文注释（变量含义/作用/流水线）、
§4.4 TUTORIAL、§4.5 DEBUG、§4.7 METRICS。代码改动后请运行全局测试并同步四份文档。

# projects/3d_reconstruction — 3D 重建教学实现

把 [tutorials/3d_reconstruction/](../../tutorials/3d_reconstruction/README.md) 中
**RGB-D 融合（第 04 章）→ 表面提取（第 05 章）→ SDF 表示（第 06 章）** 的核心算法
写成能跑的教学代码：纯 NumPy、无新依赖、自包含（不 import 其他项目）、每个模块带
确定性测试，公式与教程逐式对应（截断 (4.3)、递归融合 (4.8)、棱上插值 (5.1)、
梯度法向 (5.2)、Eikonal (6.1)）。

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
```

- **全局测试**：`python -m pytest tests/ -q`（39 个测试，系统 python3 与 llm_env 均 ~1.4 s）
- **单点测试**：`python tests/test_marching.py normals`（子串过滤，无匹配会列出可用测试名）
- VS Code 调试配置见 [.vscode/launch.json](../../.vscode/launch.json)，用法见 [.vscode/SETUP.md](../../.vscode/SETUP.md)

## 模块总览（教程章 × 模块 × 核心内容 × 测试数）

| 模块 | 教程章 | 核心内容 | 测试 |
|------|--------|---------|------|
| [scene.py](scene.py) | 01（SDF 约定）、04 §4.3（光线投射）、06 §6.1（Eikonal） | 球/盒/平面 SDF 基元（外正内负，(1.13)）、min 并集场景、解析最近点与真值面采样、球追踪（sphere tracing, Hart 1996）深度渲染、轨道双环位姿、σ(z)=c·z² 深度噪声（(4.2)） | 11 |
| [tsdf.py](tsdf.py) | **04（RGB-D 融合与 TSDF）** | TSDF 体素网格 + Curless & Levoy 1996 加权积分：离散观测 s≈z(u)−zₓ（(4.1)）、逆方差权重 w∝1/z²（(4.2)）、截断 ±1（(4.3)）、递归融合 (4.8)；三线性插值查询（(4.13)/(1.15)）、截断符号距离、权重掩码提取场、深度→点云 | 8 |
| [marching.py](marching.py) | **05（从体素到网格表面提取）** | Marching Tetrahedra：立方体 6 四面体拆分、16 case 手写表、棱上插值 t=fₐ/(fₐ−f_b)（(5.1)）、全局棱键顶点去重（水密）、梯度法向（(5.2)）、外法向取向统一 | 10 |
| [metrics.py](metrics.py) | 评测惯例（DTU / Tanks and Temples） | 对称 Chamfer（(M.1)）、DTU accuracy/completion（(M.2)）、F-score@τ（(M.3)）、面积加权表面采样（(M.4)），全部纯函数 | 10 |
| [demo.py](demo.py) | 04 + 05 端到端 | 冒烟主管线：球+盒场景 → 24 视角深度 → TSDF 融合 → MT 网格 → 与解析真值表面点比 Chamfer/Acc/Comp/F | — |

各模块健康值与指标口径见 [METRICS.md](METRICS.md)；全部 5 段最小示例见
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
——多视角融合把完整度抬起来（METRICS.md §2 有逐项解读）。

## 保真度声明（哪些章节内容没有对应代码）

本项目是**教学实现**：只实现支撑教程第 04/05/06 章推导的最小可算核心，不是系统复现。
以下教程内容未建模块，理由与替代如下：

| 教程内容 | 不实现的原因 | 教学替代 |
|----------|-------------|---------|
| 第 02/03 章 SfM 与 MVS（COLMAP 类） | 特征匹配 + 光束法平差 + 深度图融合的工程量远超教学范围，且需要真实多视图图像 | 本项目直接从已知位姿的深度出发（第 04 章入口）；MVS 推导见教程原文 |
| 第 05 章 Poisson 表面重建 | 需要稀疏线性系统求解器（八叉树 + 多重网格），纯 numpy 教学实现过长 | MT 覆盖"隐式场 → 网格"的最短路径；MC 三角形约定与歧义分析见教程 5.1–5.2 节，Poisson 推导链 (5.3)–(5.8) 见教程 5.3 节 |
| 第 06 章 DeepSDF / Occupancy Networks 训练 | 需 GPU 优化器与距离变换真值数据 | `scene.py` 的解析 SDF 即 (6.1) Eikonal 的精确载体（`tests/test_scene.py::test_eikonal_property_of_analytic_sdf` 数值验证 ‖∇s‖=1）；网络化部分见教程原文 |
| 第 07 章 NeuS / 第 08 章 3DGS / 第 09 章 哈希编码与前馈重建 | 需 GPU 神经渲染器 / 基础模型权重 | 教程各章原文；[projects/nerf](../nerf/) 覆盖体渲染基线 |
| 第 10 章 系统实战与选型 | 综述/工程选型，无单一可复现算法 | 教程原文；本项目 demo 即"机器人桌面场景重建"的最小管线示例 |

## 目录结构

```
projects/3d_reconstruction/
├── README.md            # 本文件
├── TUTORIAL.md          # 零基础代码导读（必读入口）
├── DEBUG.md             # 调试与测试教程（24 条重点观察变量表）
├── METRICS.md           # 测评指标教程（Chamfer / Acc / Comp / F-score）
├── metrics.py           # 统一测评模块（公式 (M.1)-(M.4)）
├── scene.py             # SDF 基元 + 球追踪渲染 + 数据集 + 真值采样
├── tsdf.py              # TSDF 体素网格 + Curless-Levoy 融合 + 查询
├── marching.py          # Marching Tetrahedra 等值面提取
├── demo.py              # 核心入口冒烟（VS Code 调试配置指向本文件）
└── tests/test_*.py      # 4 个测试文件，39 个测试，全部支持单点过滤
```

## 规范

遵循根目录 [CONSTRAINTS.md](../../CONSTRAINTS.md)：§4.3 中文注释（变量含义/作用/流水线）、
§4.4 TUTORIAL、§4.5 DEBUG、§4.7 METRICS。代码改动后请运行全局测试并同步四份文档。

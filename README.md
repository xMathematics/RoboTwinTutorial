# tutorial — 3D 视觉论文研读工作区

本工作区以 **NeRF 论文精读**为主线，配套中文教程与最小 PyTorch 参考实现；同时保留了原有的 RoboTwin 2.0 学习资料（独立目录）。

---

## 📁 目录结构（已重构）

```
tutorial/
├── README.md               ← 本文件（总览）
├── .vscode/                ← VS Code 配置（conda 环境 llm_env）
│   ├── settings.json       ← Python 解释器 = llm_env
│   ├── launch.json         ← 训练/测试/渲染 调试配置
│   ├── tasks.json          ← 同名任务
│   └── extensions.json     ← 推荐扩展
│
├── nerf/                   ★ NeRF 论文主线
│   ├── README.md           NeRF 子项目总览
│   ├── paper/latex/        论文 LaTeX 源码（原 latex/ 移入）
│   │   └── arxiv_submission.tex  主文件
│   ├── tutorial/           详细中文教程（10 章 + 导航）
│   └── code/               最小可运行 PyTorch 参考实现
│       ├── run_nerf.py     CLI 主入口
│       ├── nerf/           核心模块（编码/模型/采样/渲染/数据/训练）
│       ├── scripts/make_demo_data.py   演示数据生成
│       └── tests/test_core.py          单元测试（8/8 通过）
│
├── robotwin/               RoboTwin 2.0 学习资料（原 doc/ + paper/ 移入，独立保留）
│   ├── tutorial/           原 10 章中文教程
│   └── paper/              原论文 PDF（arXiv:2506.18088）
│
└── documents/              通用文档（预留）
```

---

## 📖 论文分析（一句话）

`nerf/paper/latex/` 下的论文是计算机视觉里程碑之作：
**《NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis》**
（Mildenhall et al., ECCV 2020）。

**核心创新**：把三维场景表示为连续的 **5D 神经辐射场**（MLP：位置+方向 → 颜色+密度），
用**可微体积渲染**从一组带位姿的照片优化出新视角图像。三大关键技术：
① 位置编码（Eq.4）② 分层采样 + 层次采样（coarse/fine 双网络）③ 体积渲染积分（Eq.3）。
完整解读见 [`nerf/tutorial/`](nerf/tutorial/README.md)。

---

## 🚀 快速开始

### 1. 环境（Anaconda）

项目已配置 VS Code 使用 conda 环境 **`llm_env`**（含 torch 2.11.0）。

```bash
conda activate llm_env
pip install -r nerf/code/requirements.txt   # 缺什么装什么
```

### 2. 读教程

从 [`nerf/tutorial/README.md`](nerf/tutorial/README.md) 开始，按 1→10 章顺序学习。

### 3. 跑代码（无需下载大数据集）

```bash
cd nerf/code
# ① 单元测试（8/8）
python tests/test_core.py

# ② 生成演示数据（彩色小球，秒级）
python scripts/make_demo_data.py --out data/demo_scene --n_train 8 --res 32

# ③ 冒烟训练（CPU 80 步 ~5 秒）
python run_nerf.py --config data/demo_scene --mode train --exp demo_smoke --tiny --steps 80

# ④ 评估与渲染
python run_nerf.py --config data/demo_scene --mode test  --exp demo_smoke --ckpt logs/demo_smoke/latest.pt
python run_nerf.py --config data/demo_scene --mode render --exp demo_smoke --ckpt logs/demo_smoke/latest.pt --frames 30
```

> 💡 在 VS Code 中可直接用 `Run and Debug`（调试配置）或任务面板（`Tasks: Run Task`）执行。

---

## 🛠️ 重构说明

原结构问题：`doc/`（RoboTwin 教程）与 `latex/`（NeRF 论文）主题不一致、`code/` 与 `documents/` 为空。
重构后按"项目主题"组织：NeRF 主线独立成 `nerf/`（论文/教程/代码三位一体），
RoboTwin 内容完整保留于 `robotwin/`，并新增 `.vscode/` 环境配置与统一 README。

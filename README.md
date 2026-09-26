# RoboTwinTutorial — 3D 视觉论文研读工作区

按 **论文（papers）/ 教程（tutorials）/ 项目（projects）** 三大类组织的研读工作区，
当前主线为 **NeRF 论文精读**（论文源码 + 中文教程 + PyTorch 参考实现），
并包含 RoboTwin 2.0、SLAM、3D 重建、机器人控制与规划等主题的学习资料。

---

## 📁 目录结构

```
RoboTwinTutorial/
├── README.md               ← 本文件（总览）
├── CONSTRAINTS.md          ← 全局约束规范（文档/代码/提交规范）
├── CHANGELOG.md            ← 变更日志
│
├── papers/                 ★ 论文库（按主题分目录；本地资料，已加入 .gitignore 不上传）
│   ├── README.md           论文索引
│   ├── nerf/latex/         NeRF 论文 LaTeX 完整源码
│   ├── robotwin/           机器人操作论文（RoboTwin 1.0/2.0 + VLA/模仿学习 11 篇）
│   ├── 3d_reconstruction/  3D 重建论文 12 篇 + LaTeX 源码
│   ├── slam/               SLAM 论文 18 篇 + LaTeX 源码
│   └── control_planning/   控制与规划论文 9 篇 + LaTeX 源码
│
├── tutorials/              ★ 教程文档（按主题分目录，中文）
│   ├── README.md           教程总导航
│   ├── nerf/               NeRF 10 章教程 + OVERVIEW 主题总览
│   ├── robotwin/           RoboTwin 2.0 10 章教程
│   ├── slam/               SLAM 10 章教程 + OVERVIEW
│   ├── 3d_reconstruction/  3D 重建（写作中）
│   └── control_planning/   控制与规划（写作中）
│
├── projects/               ★ 代码项目（可运行参考实现）
│   ├── README.md           项目索引与快速开始
│   ├── nerf/               NeRF PyTorch 教学版（8/8 单元测试通过）
│   └── slam/               SLAM 论文教学实现（core/fastslam/epipolar/bowloop/...）
│
├── scripts/                项目管理脚本（auto_commit.sh 自动提交）
├── .github/workflows/      GitHub Actions（每周一 9:00 自动提交）
└── .vscode/                VS Code 配置（conda 环境 llm_env，调试/任务）
```

---

## 📖 主线：NeRF 论文

*NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*（Mildenhall et al., ECCV 2020）。

**核心创新**：把三维场景表示为连续的 **5D 神经辐射场**（MLP：位置+方向 → 颜色+密度），
用**可微体积渲染**从一组带位姿的照片优化出新视角图像。三大关键技术：
① 位置编码（Eq.4）② 分层采样 + 层次采样（coarse/fine 双网络）③ 体积渲染积分（Eq.3）。

- 论文源码：[`papers/nerf/latex/`](papers/nerf/latex/arxiv_submission.tex)
- 中文教程：[`tutorials/nerf/`](tutorials/nerf/README.md)（主题总览见 [OVERVIEW](tutorials/nerf/OVERVIEW.md)）
- 参考实现：[`projects/nerf/`](projects/nerf/README.md)

其他主题：[RoboTwin 2.0 教程](tutorials/robotwin/README.md) ｜ [SLAM](tutorials/slam/README.md) ｜ [3D 重建](tutorials/3d_reconstruction/README.md) ｜ [控制与规划](tutorials/control_planning/README.md)

---

## 🚀 快速开始

### 1. 环境（Anaconda）

项目统一使用 Anaconda 环境 **`llm_env`**，配置清单见根目录 [environment.yml](environment.yml)
（规范见 [CONSTRAINTS.md §4.6](CONSTRAINTS.md)）。

```bash
conda env create -f environment.yml && conda activate llm_env   # 全新创建
# 验证（应打印 torch/numpy/pytest 版本号）：
python -c "import torch, numpy, pytest; print(torch.__version__, numpy.__version__, pytest.__version__)"
pip install -r projects/nerf/requirements.txt   # NeRF 额外依赖（缺什么装什么）
```

各项目的调试/单点测试/全量测试用法见其 `DEBUG.md`，测评指标见其 `METRICS.md`。

### 2. 读教程

从 [`tutorials/nerf/README.md`](tutorials/nerf/README.md) 开始，按 1→10 章顺序学习。

### 3. 跑代码（无需下载大数据集）

```bash
cd projects/nerf
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

## 📋 规范与自动化

- **开发规范**（文档/代码/提交/版本控制）：见 [CONSTRAINTS.md](CONSTRAINTS.md)
- **本地自动提交**：`./scripts/auto_commit.sh`（用法见 [scripts/README.md](scripts/README.md)）
- **GitHub Actions**：每周一 9:00 自动提交有变更的内容（`.github/workflows/auto-commit.yml`），也可在 Actions 页手动触发
- **提交信息**：Conventional Commits 格式，如 `docs(nerf): 添加渲染方程章节`

## 🤝 贡献

新增主题时三处联动：`papers/<主题>/`（论文）→ `tutorials/<主题>/`（教程）→ `projects/<主题>/`（可选代码），并更新三类索引 README。

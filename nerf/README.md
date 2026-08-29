# nerf/ — NeRF 论文项目（主线）

> **论文**：*NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*
> （Mildenhall et al., ECCV 2020）｜ [ECCV 最佳论文提名](https://arxiv.org/abs/2003.08934)

本目录是"论文源码 + 中文教程 + 参考实现"三位一体的 NeRF 学习子项目。

## 目录

```
nerf/
├── paper/latex/    论文 LaTeX 源码（arxiv_submission.tex 为入口）
├── tutorial/       详细中文教程（10 章，从入门到实战）
└── code/           最小可运行 PyTorch 参考实现（含演示数据脚本与测试）
```

## 快速导航

| 想做什么 | 去哪里 |
|---------|--------|
| 了解论文讲了什么 | [`tutorial/README.md`](tutorial/README.md)（先读导航） |
| 推导核心公式 | [`tutorial/03_数学原理与渲染方程.md`](tutorial/03_数学原理与渲染方程.md) |
| 读懂网络结构 | [`tutorial/04_网络架构与位置编码.md`](tutorial/04_网络架构与位置编码.md) |
| 读懂实验与消融 | [`tutorial/06_数据集与实验结果.md`](tutorial/06_数据集与实验结果.md) |
| 跑代码 | [`code/README.md`](code/README.md) |
| 代码与公式对应 | [`tutorial/09_代码逐行精读.md`](tutorial/09_代码逐行精读.md) |

## 论文源码（paper/latex/）

- 主文件：`paper/latex/arxiv_submission.tex`
- 结果表：`resultstable.tex`（主结果）、`ablationstable.tex`（消融）、`synthresults.tex` / `realresults.tex` / `dvoxresults.tex` / `suppresultstable.tex`（图）
- 数学推导：`ndc.tex`（NDC 空间推导）
- 宏定义：`macros.tex`

## 参考实现（code/）

PyTorch 教学版，覆盖论文全部核心机制（位置编码、coarse/fine 双 MLP、分层/层次采样、可微体积渲染、联合训练损失）。
环境为 conda `llm_env`（见 `.vscode/settings.json`），已通过 8/8 单元测试与端到端冒烟训练。

> ⚠️ 教学简化版：仅支持 Blender 合成数据；完整复现请参考官方实现 [bmild/nerf](https://github.com/bmild/nerf)。

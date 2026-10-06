# RoboTwin — 主题总览（论文 / 教程 / 代码）

> 论文库：[papers/robotwin/](../../papers/robotwin/README.md)（基准 2 篇 + 经典 3 篇 + 前沿 6 篇，共 11 篇）｜ 教程：[10 章](README.md) ｜ 论文精读：[11 篇](精读/README.md) ｜ 代码：官方开源（本地暂无教学实现）

## 章节与论文对应

| 章 | 教程 | 主读论文（本地 PDF） | 支撑材料 |
|----|------|---------------------|---------|
| 01 | [具身智能入门](01_具身智能入门.md) | [RT-1 (2022)](../../papers/robotwin/classics/arXiv-2212.06817_RT1.pdf)、[RT-2 (2023)](../../papers/robotwin/frontier/arXiv-2307.15818_RT2.pdf) | Open X-Embodiment（大规模数据范式） |
| 02 | [双臂操作与仿真基础](02_双臂操作与仿真基础.md) | [ACT / ALOHA (2023)](../../papers/robotwin/classics/arXiv-2304.13705_ACT-ALOHA.pdf)、[RoboTwin 1.0 (2024)](../../papers/robotwin/paper/arXiv-2409.02920_RoboTwin1.0.pdf) | 生成式数字孪生概念 |
| 03 | [RoboTwin 2.0 论文精读](03_RoboTwin2.0论文精读.md) | [RoboTwin 2.0 (2025)](../../papers/robotwin/paper/2506.18088v2.pdf)（本章主线） | RoboTwin 1.0（前作对照） |
| 04 | [核心技术详解](04_核心技术详解.md) | RoboTwin 2.0 §3–§4（MLLM 代码生成 / 域随机化 / 体态适配） | 精读笔记 [RoboTwin2.0](精读/RoboTwin2.0_arXiv2025.md) |
| 05 | [数据集与基准](05_数据集与基准.md) | RoboTwin 2.0（RoboTwin-OD 对象库 + 50 任务） | [Open X-Embodiment (2023)](../../papers/robotwin/frontier/arXiv-2310.08864_OpenXEmbodiment.pdf)（跨本体数据基座对照） |
| 06 | [实验结果与解读](06_实验结果与解读.md) | RoboTwin 2.0 §5（四大实验） | 精读笔记中的表格重排 |
| 07 | [环境搭建与实战](07_环境搭建与实战.md) | ——（操作章，见[官方文档](https://robotwin-platform.github.io/doc/usage/)） | —— |
| 08 | [策略训练与部署](08_策略训练与部署.md) | [Diffusion Policy (2023)](../../papers/robotwin/classics/arXiv-2303.04137_DiffusionPolicy.pdf)、[ACT](../../papers/robotwin/classics/arXiv-2304.13705_ACT-ALOHA.pdf)、[RT-1](../../papers/robotwin/classics/arXiv-2212.06817_RT1.pdf)、[OpenVLA (2024)](../../papers/robotwin/frontier/arXiv-2406.09246_OpenVLA.pdf)、[π0 (2024)](../../papers/robotwin/frontier/arXiv-2410.24164_Pi0.pdf)、[RDT-1B (2024)](../../papers/robotwin/frontier/arXiv-2410.07864_RDT-1B.pdf) | 五种策略各配精读笔记 |
| 09 | [进阶研究方向](09_进阶研究方向.md) | [DexGraspNet (2022)](../../papers/robotwin/frontier/arXiv-2210.02697_DexGraspNet.pdf)、RT-2 / π0 / RDT-1B（前沿路线） | GraspNet-1Billion（延伸阅读） |
| 10 | [术语表与FAQ](10_术语表与FAQ.md) | ——（速查手册） | —— |

## 论文谱系（11 篇精读的分组）

- **基准主线（2）**：RoboTwin 1.0 → RoboTwin 2.0——生成式数字孪生到域随机化数据工厂；
- **策略学习经典（3）**：RT-1（大规模数据 + Transformer）、ACT/ALOHA（双臂遥操作 + CVAE）、
  Diffusion Policy（扩散动作生成，RoboTwin 基线）；
- **VLA 与预训练前沿（5）**：RT-2 → OXE（数据基座）→ OpenVLA（开源可微调）→ π0（流匹配）、RDT-1B（双臂扩散）；
- **抓取扩展（1）**：DexGraspNet（灵巧手抓取合成）。

阅读顺序与跨篇衔接见 [精读/README.md](精读/README.md)。

## 与其他主题的关系

- **NeRF / 3D 重建**：RoboTwin 的"生成式数字孪生"需要三维场景资产——重建是孪生的上游，
  见 [3D 重建教程第 10 章](../3d_reconstruction/10_机器人场景中的重建实战与选型.md)（机器人场景选型）；
- **SLAM**：真值位姿估计与数据采集中的传感器标定，见 [SLAM 教程](../slam/OVERVIEW.md)；
- **控制与规划**：策略输出的关节/末端指令如何落到底层运动控制——
  [控制与规划教程第 10 章](../control_planning/10_控制栈实战与robotwin衔接.md)专门讲"控制栈与 RoboTwin 衔接"。

## 代码状态

`projects/robotwin/` 暂未建立：RoboTwin 依赖 CoppeliaSim 仿真器、MLLM API 与较大规模模型，
不适合像 [projects/nerf](../../projects/nerf/README.md)、[projects/slam](../../projects/slam/README.md)
那样做纯 NumPy/PyTorch 教学实现。实操请直接使用
[官方开源实现与文档](https://robotwin-platform.github.io/doc/)（教程第 7、8 章的命令即来自官方文档）。

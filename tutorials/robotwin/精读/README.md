# 机器人操作论文精读 — 总索引

对 [papers/robotwin/](../../../papers/robotwin/README.md) 收录的 **11 篇论文**逐一精读：
每篇一份文档，包含论文信息、问题动机、方法总览、关键公式推导（符号表 + 逐式推导、
标注论文原文式号）、实验解读、局限影响，以及与本项目 [十章教程](../README.md) 的双向对照。

**精读文档均基于本地 PDF 原文撰写**——式号、数据规模、实验口径逐页核对；
凡规格/二手描述与原文不符处一律以原文为准（各文档头部或偏离说明中显式标注）。
注意：RoboTwin 1.0 本地 PDF 为官方 "early version"（v3），内容与后续完整版有差异，精读中已声明。

## 按论文谱系排列（11 篇）

### 基准本体（本教程主线）

| 精读 | 论文（venue） | 一句话内核 | 教程章 |
|------|--------------|-----------|--------|
| [RoboTwin 1.0](RoboTwin1.0_ECCVW2024.md) | RoboTwin（ECCV Workshop 2024 Best Paper） | 双臂基准 + 生成式数字孪生（真实-仿真对齐）起点 | [02](../02_双臂操作与仿真基础.md)、[03](../03_RoboTwin2.0论文精读.md) |
| [RoboTwin 2.0](RoboTwin2.0_arXiv2025.md) | RoboTwin 2.0（arXiv 2025） | 50 任务 × 5 本体、域随机化自动生成 10 万+ 轨迹 | [03](../03_RoboTwin2.0论文精读.md)–[06](../06_实验结果与解读.md) |

### 模仿学习与操作策略经典

| 精读 | 论文（venue） | 一句话内核 | 教程章 |
|------|--------------|-----------|--------|
| [RT-1](RT1_arXiv2022.md) | RT-1（arXiv 2022） | 大规模数据 + Transformer 策略范式开端（130k+ episodes） | [01](../01_具身智能入门.md)、[08](../08_策略训练与部署.md) |
| [Diffusion Policy](DiffusionPolicy_RSS2023.md) | Diffusion Policy（RSS 2023） | 扩散模型生成动作：多峰分布的表达力论证 | [08](../08_策略训练与部署.md) |
| [ACT / ALOHA](ACT_ALOHA_RSS2023.md) | ACT / ALOHA（RSS 2023） | 低成本双臂遥操作 + CVAE 动作分块 + 时间集成 | [02](../02_双臂操作与仿真基础.md)、[08](../08_策略训练与部署.md) |

### VLA 与大规模预训练前沿

| 精读 | 论文（venue） | 一句话内核 | 教程章 |
|------|--------------|-----------|--------|
| [RT-2](RT2_arXiv2023.md) | RT-2（CoRL 2023） | 动作即语言 token：VLA 开创 + 涌现语义推理 | [01](../01_具身智能入门.md)、[09](../09_进阶研究方向.md) |
| [Open X-Embodiment](OpenXEmbodiment_ICRA2024.md) | OXE（ICRA 2024） | 22 本体 1M+ 轨迹的统一数据集与跨本体正迁移 | [05](../05_数据集与基准.md) |
| [OpenVLA](OpenVLA_CoRL2024.md) | OpenVLA（CoRL 2024） | 7B 开源 VLA：LoRA 微调 + 256-bin 动作离散化 | [08](../08_策略训练与部署.md)、[09](../09_进阶研究方向.md) |
| [π0](Pi0_arXiv2024.md) | π0（arXiv 2024） | 流匹配动作专家 + VLM 骨干，50Hz 高频动作块 | [08](../08_策略训练与部署.md) |
| [RDT-1B](RDT1B_ICLR2025.md) | RDT-1B（ICLR 2025） | 1B 扩散 Transformer 策略，统一动作空间预训练 | [08](../08_策略训练与部署.md) |

### 灵巧抓取

| 精读 | 论文（venue） | 一句话内核 | 教程章 |
|------|--------------|-----------|--------|
| [DexGraspNet](DexGraspNet_NeurIPS2022.md) | DexGraspNet（NeurIPS 2022） | 可微力闭合优化合成 1.3M 灵巧手抓取 | [09](../09_进阶研究方向.md)、[3D 重建 10](../../3d_reconstruction/10_机器人场景中的重建实战与选型.md) |

## 论文 × 代码状态（projects/robotwin）

本目录 11 篇论文与本地教学实现 [projects/robotwin/](../../../projects/robotwin/README.md)
（纯 NumPy 迷你基准，52 项测试双环境全绿）的对应关系：

| 论文 | 代码状态 | 对应实现 | 一句话说明 |
|------|----------|----------|-----------|
| [RoboTwin 2.0](RoboTwin2.0_arXiv2025.md) | ✅ | `benchmark / dr / metrics / tasks` 等 | 两条方法论主线——域随机化（none/mild/strong 三档剂量 × 五维）与评测协议（成功率 / 宏平均+最差 / seen−unseen 差距）——的 2D 迷你复现 |
| [RoboTwin 1.0](RoboTwin1.0_ECCVW2024.md) | 🔶 | 同上（基准协议侧） | 双臂基准的"回合环境 + 成功率"骨架已复现；生成式数字孪生（真实-仿真对齐）需要渲染管线，不在范围 |
| [Diffusion Policy](DiffusionPolicy_RSS2023.md) | 🔶 | [diffusion_lite.py](../../../projects/robotwin/diffusion_lite.py) | DDPM 条件去噪生成的最小教学版；简化：2D 轨迹（无图像观测）、3 层 MLP 手写反传（无 Transformer/CNN 骨干）、条件退化为 (起点, 终点)（无 CVAE、无 DDIM 加速） |
| [DexGraspNet](DexGraspNet_NeurIPS2022.md) | 🔶 | [grasp_2d.py](../../../projects/robotwin/grasp_2d.py) | 力闭合判据 + Ferrari–Canny L1 + 候选排序落到 2D 平行夹爪；简化：接触点对抓取（无 28 维灵巧手位姿）、网格候选（无可微能量优化）、解析判据（无 Isaac Gym 校验） |
| [ACT / ALOHA](ACT_ALOHA_RSS2023.md) | ❌ | — | CVAE + Transformer 动作分块的双臂模仿训练，属 GPU 模仿学习工程（教程 08 章）；"生成式策略"思想由 diffusion_lite 代讲 |
| [RT-1](RT1_arXiv2022.md) | ❌ | — | 130k+ 回合的大规模机器人 Transformer，瓶颈在数据与算力而非公式 |
| [RT-2](RT2_arXiv2023.md) | ❌ | — | VLM → 动作 token 的 VLA 基础模型，依赖网络级预训练权重 |
| [Open X-Embodiment](OpenXEmbodiment_ICRA2024.md) | ❌ | — | 22 本体 1M+ 轨迹的数据基座，没有可仿真化的"最小版本" |
| [OpenVLA](OpenVLA_CoRL2024.md) | ❌ | — | 7B 开源 VLA（LoRA 微调），预训练-微调范式超出纯 NumPy 教学范围 |
| [π0](Pi0_arXiv2024.md) | ❌ | — | 流匹配 VLA（VLM 骨干 + 动作专家），同上 |
| [RDT-1B](RDT1B_ICLR2025.md) | ❌ | — | 1B 扩散 Transformer 双臂基础模型，预训练规模不可教学化 |

原则：能落成"1 分钟内跑通的方法论骨架"的判据 / 协议 / 生成式思想
（✅/🔶）均有对应 NumPy 教学实现；依赖大规模数据、预训练权重或 GPU
工程的（❌）只留教程与精读。各模块"与原文的差异"逐条见其模块 docstring。

## 建议阅读顺序

- **按教程主线**：[十章教程](../README.md) 08 章策略训练 ↔ Diffusion Policy/ACT/RT-1 精读；
  09 章前沿方向 ↔ RT-2/OXE/OpenVLA/π0/RDT 精读；
- **按范式演进**：RT-1（离散 token 监督）→ Diffusion Policy（生成式）/ACT（CVAE 分块）→
  RT-2（VLA 开创）→ OXE（数据基座）→ OpenVLA/π0/RDT（开源 VLA/流匹配/扩散三路线）；
- **做仿真基准**：先读 RoboTwin 双基准精读，再按所选 baseline 深入对应策略精读。

## 与教程的关系

教程（../01–10 章）按概念组织、面向"会用 RoboTwin"；本目录按论文组织、面向"读懂原文"。
策略类论文公式多为论文自定记号（部分论文无编号公式，精读中自编式号并显式声明）；
所有回引教程章节号均已核实。

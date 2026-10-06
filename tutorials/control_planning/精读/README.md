# 控制与规划论文精读 — 总索引

对 [papers/control_planning/](../../../papers/control_planning/README.md) 收录的 **9 篇论文**逐一精读：
每篇一份文档，包含论文信息、问题动机、方法总览、**关键公式推导**（符号表 + 逐式推导、
标注论文原文式号/Theorem 号、每步注明依据）、实验解读、局限影响，以及与本项目
[十章教程](../README.md) 的双向对照（论文式号 ↔ 教程 (章.序) 映射表）。

**精读文档均基于本地 PDF 原文逐页撰写**——RRT* 76 页的 Theorem 15–71 与附录、
PPO/SAC 的公式编号全部逐一核对；发现的原论文排印/口径问题（PPO 式 (10)(11) 指数笔误、
SAC v2 版本差异等）在文中加勘误注。

## 按控制栈层次排列（9 篇）

### 运动规划（"去哪、怎么走"）

| 精读 | 论文（venue） | 一句话内核 | 教程章 |
|------|--------------|-----------|--------|
| [RRT](RRT_TR1998.md) | RRT（TR 98-11） | 采样树快速探索：概率完备的单查询规划 | [03](../03_运动规划-i采样式规划.md) |
| [RRT*](RRTstar_IJRR2011.md) | RRT*（IJRR 2011） | 选父 + 重布线：采样规划的渐近最优性（Theorem 34–39） | [03](../03_运动规划-i采样式规划.md) §03.2 |
| [MPNet](MPNet_ICRA2019.md) | MPNet（ICRA 2019） | 神经采样先验 + 经典兜底的混合规划 | [09](../09_前沿学习式规划与腿式控制.md) §09.1 |

### 轨迹优化与模型预测控制（"怎么走最好"）

| 精读 | 论文（venue） | 一句话内核 | 教程章 |
|------|--------------|-----------|--------|
| [Crocoddyl](Crocoddyl_ICRA2020.md) | Crocoddyl（ICRA 2020） | 多接触 OCP 的解析导数 + DDP/FDDP 求解器 | [04](../04_轨迹优化与最优控制.md) §04.2 |
| [PETS](PETS_NeurIPS2018.md) | PETS（NeurIPS 2018） | 概率集成动力学 + CEM 的模型基 RL/MPC | [05](../05_模型预测控制mpc.md) §05.3 |

### 安全与学习式控制（"如何安全、如何自适应"）

| 精读 | 论文（venue） | 一句话内核 | 教程章 |
|------|--------------|-----------|--------|
| [CBF 综述](CBF-Survey_ECCV2019.md) | Control Barrier Functions（ECCV 2019） | 安全滤波统一框架：CBF-QP/CLF-CBF-QP（式 (1)–(26)） | [07](../07_安全控制控制屏障函数.md) |
| [PPO](PPO_arXiv2017.md) | PPO（arXiv 2017） | 截断代理目标：策略梯度的事实标准（式 (1)–(12)） | [08](../08_学习式控制强化学习.md) §08.2 |
| [SAC](SAC_ICML2018.md) | SAC（ICML 2018） | 最大熵 off-policy：软 Bellman 与重参数化（式 (1)–(13)） | [08](../08_学习式控制强化学习.md) §08.3 |
| [Rapid Locomotion](RapidLocomotion_RSS2022.md) | Rapid Locomotion（RSS 2022） | 特权教师 + 隐空间蒸馏的腿式盲控制 | [09](../09_前沿学习式规划与腿式控制.md) §09.2 |

## 论文 × 代码状态（projects/control_planning）

9 篇精读在 [projects/control_planning/](../../../projects/control_planning/README.md)
的落地状态：✅ = 代码覆盖论文核心主张；🔶 = 简化档（保留主算法结构，去工程重件）；
❌ = 未实现（读推导，代码看替代品）。评估指标统一取自该项目的 `metrics.py`。

| 精读 | 状态 | 代码 | 说明 |
|------|------|------|------|
| [RRT](RRT_TR1998.md) | ✅ | [rrt.py](../../../projects/control_planning/rrt.py) | steer、增量碰撞检测、goal bias 全覆盖（教程 (3.3)–(3.4)/(3.11)） |
| [RRT*](RRTstar_IJRR2011.md) | ✅ | [rrt.py](../../../projects/control_planning/rrt.py) | 选父 + 重布线 + 收缩半径 γ(log n/n)^(1/d)（Theorem 38） |
| [PPO](PPO_arXiv2017.md) | ✅ | [ppo_lite.py](../../../projects/control_planning/ppo_lite.py) | GAE + 截断代理目标全结构（手写前向/反向） |
| [CBF 综述](CBF-Survey_ECCV2019.md) | ✅ | [cbf.py](../../../projects/control_planning/cbf.py) | CBF-QP 闭式投影 (7.13) + 多约束 POCS |
| [Crocoddyl](Crocoddyl_ICRA2020.md) | 🔶 | [ilqr.py](../../../projects/control_planning/ilqr.py) | iLQR 是其 DDP 求解器的简化核心；多接触解析导数与 FDDP 不做 |
| [PETS](PETS_NeurIPS2018.md) | 🔶 | [mpc_cem.py](../../../projects/control_planning/mpc_cem.py) | 已知动力学简化档（论文消融的确定性档）；概率集成模型不做 |
| [MPNet](MPNet_ICRA2019.md) | 🔶 | [mpnet_lite.py](../../../projects/control_planning/mpnet_lite.py) | 采样偏置教学代理：粗 SDF 编码替代 Enet、在线少量 rrt 专家替代大规模离线训练；完整两阶段编码器与双向重规划不做 |
| [SAC](SAC_ICML2018.md) | ❌ | —— | 最大熵 / twin-Q 结构未实现，PPO（`ppo_lite.py`）已覆盖学习式控制章节核心；软 Bellman/重参数化推导见精读 |
| [Rapid Locomotion](RapidLocomotion_RSS2022.md) | ❌ | —— | 需 GPU RL（IsaacGym 级大规模并行）与高保真仿真；特权蒸馏推导见教程 09.2 精读 |

（另有 `osc_arm.py` 对应延伸阅读条目 Khatib/Hogan——无 PDF 故无精读，
见下方注。）

## 建议阅读顺序

- **按教程主线**：[十章教程](../README.md) 03 章 ↔ RRT/RRT* 精读 → 04 章 ↔ Crocoddyl →
  05 章 ↔ PETS → 07 章 ↔ CBF → 08 章 ↔ PPO/SAC → 09 章 ↔ MPNet/Rapid Locomotion；
- **按数学难度**：入门（RRT/PETS）→ 进阶（RRT*/CBF/PPO）→ 困难（Crocoddyl 的接触动力学
  解析导数与 FDDP、SAC 的软 Bellman 收缩证明）；
- **做仿真基准**：RoboTwin 策略用 RL 训练时读 PPO；部署安全层读 CBF；腿式/足式读 Rapid Locomotion。

## 与教程、其他主题精读的关系

- **教程（../01–10 章）**：按控制栈层次组织、五步法推导；精读 §7 各有"论文式号 ↔ (章.序)"映射表；
- **[RoboTwin 精读](../../robotwin/精读/README.md)**：RL 训练器（PPO/SAC）与策略范式（π0/RDT）的
  上游理论；sim-to-real 的域随机化与蒸馏在两主题交汇；
- **[SLAM 精读](../../slam/精读/README.md)**：感知给规划喂地图，两者构成机器人全栈（见教程第 01 章）。

> 注：非 arXiv 经典（Hogan 阻抗、Khatib 操作空间、PRM、CHOMP/TrajOpt、最小 snap）为
> papers/control_planning/ 的延伸阅读条目，未收录 PDF 故无精读，相关推导已写入教程第 04/06 章。

# 机器人控制与规划 — 主题总览（论文 / 教程 / 代码）

> 论文库：[papers/control_planning/](../../papers/control_planning/README.md)（经典 3 + 前沿 6）｜ 教程：[10 章](README.md) ｜ 代码：[projects/control_planning/](../../projects/control_planning/README.md)（RRT*/iLQR/MPC/CBF/阻抗/PPO/MPNet 采样偏置已建立）

## 章节与论文对应

| 章 | 教程 | 主读论文（本地 PDF） | 教材 / 工具支撑 |
|----|------|---------------------|----------------|
| 01 | [控制与规划全景](01_控制与规划全景.md) | ——（导读章） | Craig《机器人学导论》 |
| 02 | [运动学控制：关节空间与任务空间](02_运动学控制关节空间与任务空间.md) | ——（教材章） | Craig ch4-6 |
| 03 | [运动规划 I：采样式规划](03_运动规划-i采样式规划.md) | [RRT (TR98)](../../papers/control_planning/classics/RRT_TR1998_LaValle.pdf)、[RRT* (IJRR'11)](../../papers/control_planning/classics/arXiv-1105.1186_RRTstar.pdf) | LaValle《Planning Algorithms》ch5 |
| 04 | [轨迹优化与最优控制](04_轨迹优化与最优控制.md) | [Crocoddyl (ICRA'20)](../../papers/control_planning/frontier/arXiv-1909.04947_Crocoddyl.pdf) | 最小 snap（延伸阅读） |
| 05 | [模型预测控制 MPC](05_模型预测控制mpc.md) | [PETS (NeurIPS'18)](../../papers/control_planning/frontier/arXiv-1805.12114_PETS.pdf) | —— |
| 06 | [操作空间控制与阻抗控制](06_操作空间控制与阻抗控制.md) | Khatib'87、Hogan'85（[延伸阅读](../../papers/control_planning/README.md)） | Siciliano《Robotics》 |
| 07 | [安全控制：控制屏障函数](07_安全控制控制屏障函数.md) | [CBF 综述 (ECCV'19)](../../papers/control_planning/classics/arXiv-1903.11199_CBF-Survey.pdf) | —— |
| 08 | [学习式控制：强化学习](08_学习式控制强化学习.md) | [PPO (2017)](../../papers/control_planning/frontier/arXiv-1707.06347_PPO.pdf)、[SAC (ICML'18)](../../papers/control_planning/frontier/arXiv-1801.01290_SAC.pdf) | —— |
| 09 | [前沿：学习式规划与腿式控制](09_前沿学习式规划与腿式控制.md) | [MPNet (ICRA'19)](../../papers/control_planning/frontier/arXiv-1806.05767_MPNet.pdf)、[Rapid Locomotion (RSS'22)](../../papers/control_planning/frontier/arXiv-2205.02824_RapidLocomotion.pdf) | —— |
| 10 | [控制栈实战：与 RoboTwin 衔接](10_控制栈实战与robotwin衔接.md) | ——（综合章） | [RoboTwin 教程](../robotwin/README.md) |

## 与其他主题的关系

- **SLAM**：控制栈的输入（位姿/地图）来自 [SLAM 教程](../slam/OVERVIEW.md)；第 10 章给"感知→控制"全栈视图；
- **RoboTwin**：策略学习（Diffusion Policy/ACT/VLA）产出高层动作，本章控制栈负责执行层；
  第 09 章的域随机化与第 10 章的 sim-to-real 直接衔接 RoboTwin 的数字孪生管线；
- **3D 重建**：抓取位姿与障碍几何来自 [3D 重建教程](../3d_reconstruction/OVERVIEW.md)。

## 代码状态

`projects/control_planning/` **已建立**（纯 NumPy，53 项测试双环境全绿），
被控对象统一为 2D 双积分器与平面 2R 臂：`rrt.py`（RRT/RRT*，第 03 章）、
`ilqr.py`（iLQR，含与 Riccati 代数解的交叉验证，第 04 章）、`mpc_cem.py`（CEM-MPC，
PETS 的"已知动力学 + CEM"简化档，第 05 章）、`cbf.py`（QP 安全滤波，第 07 章）、
`osc_arm.py`（FK/雅可比/DLS-IK/阻抗控制，第 06 章）、`ppo_lite.py`（微型 PPO，
第 08 章）、`mpnet_lite.py`（MPNet 采样偏置教学代理：粗 SDF 编码 + 手写 MLP +
偏置采样 RRT，第 09 章）+ `metrics.py`（规划成功率/路径长度/轨迹代价）+
`demo.py` 四段冒烟。9 篇精读的逐篇代码状态见
[精读总索引的「论文 × 代码状态」表](./精读/README.md)。
MuJoCo 仿真与学习式腿控不实现（保真度声明见其 README）。

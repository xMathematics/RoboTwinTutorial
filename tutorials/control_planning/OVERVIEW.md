# 机器人控制与规划 — 主题总览（论文 / 教程 / 代码）

> 论文库：[papers/control_planning/](../../papers/control_planning/README.md)（经典 3 + 前沿 6）｜ 教程：[10 章](README.md)（架构已定）｜ 代码：规划中

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

`projects/control_planning/` 尚未建立。规划：2D RRT/RRT*（纯 NumPy）、二连杆
iLQR + 阻抗控制仿真（MuJoCo 可选）、QP-CBF 安全滤波示例——与 projects/slam 的
教学实现同一风格（类型提示 + 论文式号 docstring + 可运行测试）。

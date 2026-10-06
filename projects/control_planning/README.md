# projects/control_planning — 控制与规划教学实现

把 [papers/control_planning/](../../papers/control_planning/README.md) 中经典论文的**核心算法**写成能跑的教学代码：
纯 NumPy、无重依赖、每个模块带确定性测试，公式与
[tutorials/control_planning/](../../tutorials/control_planning/README.md) 十章教程逐式对应。

**零基础入门请先读 [TUTORIAL.md](TUTORIAL.md)**（每个模块的最小可运行示例 + 数据结构 + 流水线图）；
调试与单点/全局测试用法见 [DEBUG.md](DEBUG.md)；评估口径见 [METRICS.md](METRICS.md)。

## 快速开始

```bash
# ① 激活环境（环境清单见根目录 environment.yml；纯 numpy，任意 numpy>=1.26 的环境亦可）
conda activate llm_env

# ② 依赖校验：本包仅依赖 numpy（无 torch/scipy）
python -c "import numpy; print('numpy', numpy.__version__)"        # 预期: numpy 2.2.6

# ③ 运行全套测试（8 个文件 53 个测试，全部确定性）
python tests/test_rrt.py                                           # 预期: 7/7 tests passed.

# ④ 核心入口冒烟演示（规划→优化→MPC→安全滤波四段，确定性输出，约 11 s）
python demo.py
```

- **全局测试**：`python -m pytest projects/control_planning/tests -v`（53 个测试，llm_env 约 33 s / 系统 python3 约 39 s）
- **单点测试**：`python tests/test_cbf.py projection`（子串过滤，无匹配会列出可用测试名）
- VS Code 调试配置见 [.vscode/launch.json](../../.vscode/launch.json)，用法见 [.vscode/SETUP.md](../../.vscode/SETUP.md)

## 模块总览（论文 × 模块 × 教程章）

| 模块 | 论文 | 教程章 | 核心内容 | 测试 |
|------|------|--------|---------|------|
| [rrt.py](rrt.py) | RRT（LaValle TR 98-11）/ RRT*（Karaman & Frazzoli, IJRR 2011） | [03](../../tutorials/control_planning/03_运动规划-i采样式规划.md) | 2D 采样规划：steer (3.3)、motion validation (3.4)、goal bias (3.11)、选父+重布线与收缩半径 γ(log n/n)^(1/d) (3.5)–(3.7)（Theorem 38） | 7 |
| [ilqr.py](ilqr.py) | DDP（Jacobson & Mayne 1970）/ iLQG（Tassa et al., ICRA 2012） | [04](../../tutorials/control_planning/04_轨迹优化与最优控制.md) | 2D 双积分器 iLQR：Q 函数四块 (4.8)–(4.9)、仿射反馈 (4.11)、前向回滚 (4.13)、阻尼正则化 (4.14)；Riccati (5.5)/(5.6) 交叉验证 | 7 |
| [mpc_cem.py](mpc_cem.py) | PETS（Chua et al., NeurIPS 2018）的已知动力学简化档 | [05](../../tutorials/control_planning/05_模型预测控制mpc.md) | 滚动时域 (5.1)/(5.2) + CEM 精英采样与协方差自适应 (5.11)–(5.13)；障碍惩罚 + 目标代价；温启动 | 6 |
| [cbf.py](cbf.py) | CBF 综述（Ames et al., ECC 2019） | [07](../../tutorials/control_planning/07_安全控制控制屏障函数.md) | 圆障碍速度阻尼屏障（综述 §IV 相对阶口径）、CBF-QP (7.9) 闭式投影 (7.13)、多约束 POCS 投影迭代 | 6 |
| [osc_arm.py](osc_arm.py) | Khatib（IEEE JRA 1987）/ Hogan（J-DSC 1985） | [06](../../tutorials/control_planning/06_操作空间控制与阻抗控制.md) | 平面 2R 臂：FK/解析雅可比 (6.3)、M/C/g 动力学 (6.2)、DLS 逆运动学、任务空间阻抗 τ = Jᵀ(Ke + Dė) (6.12)/(6.4) | 7 |
| [ppo_lite.py](ppo_lite.py) | PPO（Schulman et al., 2017）/ GAE（Schulman et al., 2016） | [08](../../tutorials/control_planning/08_学习式控制强化学习.md) | 线性高斯策略手写前向/反向、GAE (8.13)–(8.15)、截断代理目标 (8.12)、点质量到达 20 步短回合 | 6 |
| [mpnet_lite.py](mpnet_lite.py) | MPNet（Qureshi et al., ICRA 2019） | [09 §09.1](../../tutorials/control_planning/09_前沿学习式规划与腿式控制.md) | 学习式采样偏置：粗 SDF 环境编码 + 单隐层 MLP 手写前向/反向，专家蒸馏 (9.2)（在线 rrt 路径）+ 建议点代替均匀采样 (9.1)（复用 rrt 的 steer/碰撞检测） | 7 |
| [metrics.py](metrics.py) | 评估层（—） | [METRICS.md](METRICS.md) | 规划成功率 (M.1) / 路径长度 (M.2) / 轨迹代价 (M.3)——CONSTRAINTS §4.7 点名的三指标基线（纯函数） | 7 |

## 运行示例（规划 → 优化 → 控制 → 安全，一条走廊串起四章）

```python
import numpy as np
from rrt import plan_rrt_star, make_corridor_obstacles
from metrics import path_length, success_rate

obstacles = make_corridor_obstacles()               # 10x10 场地中部带缺口的圆墙
res = plan_rrt_star(np.array([0.5, 0.5]), np.array([9.0, 6.0]),
                    obstacles, (0, 10, 0, 10), seed=0, max_iters=6000)
print(res.success, round(res.cost, 3), round(path_length(res.path), 3))
# True 10.741 10.741   （3000 迭代约 10.99，6000 迭代 10.74——还在随样本缓慢下降）
```

后续三步见 [demo.py](demo.py)（iLQR 平滑同一走廊、CEM-MPC 滚动到达、CBF 滤波对照，
全部确定性、总计约 11 s）；各模块健康值见 [METRICS.md](METRICS.md)，
全部 8 段最小示例见 [TUTORIAL.md](TUTORIAL.md) 第 3 节（已实跑验证）。

## 保真度声明（哪些内容没有对应代码）

本项目是**教学实现**：只实现各论文支撑教程推导的最小核心算法，不是系统复现。
以下内容未建模块，理由与替代如下：

| 不实现的内容 | 不实现的原因 | 教学替代 |
|------|-------------|---------|
| MuJoCo 物理仿真 / 接触动力学 | 需要重依赖与场景资产，超出纯 NumPy 边界 | 被控对象统一为可手推的 2D 双积分器与平面 2R 臂；接触动力学见教程第 06/09 章 |
| 腿式控制（Crocoddyl 多接触、WBC） | 需浮动基动力学与接触求解器 | 教程第 04/09 章推导导读；`ilqr.py` 覆盖同一递推结构在浮点平面系统上的形态 |
| MPNet 完整两阶段实现 | Enet 点云编码器–解码器与 110 工作空间 × 5000 条专家路径的大规模离线训练工程量大 | **`mpnet_lite.py` 教学代理已建**：固定粗 SDF 编码 + 在线少量 rrt 专家 + 采样偏置 RRT（与原文的逐条差异见该模块 docstring） |
| RapidLocomotion（特权教师–学生蒸馏，教程 09.2） | 需 GPU RL（IsaacGym 级大规模并行）与高保真地形/域参数仿真 | 教程第 09 章 09.2 推导导读；教师阶段所用的 PPO 结构见 `ppo_lite.py` |
| SAC（最大熵 off-policy RL） | **未实现**——最大熵 twin-Q 结构不做，PPO 已覆盖学习式控制章节核心 | 精读《SAC》的软 Bellman/重参数化推导 + 教程 08.3；工程实现见 RoboTwin 的 RL 训练器 |
| PETS 的概率集成模型 | bootstrap 集成 × 概率网络的工程量大 | `mpc_cem.py` 取论文消融的确定性档，CEM 与滚动时域机制完整保留 |
| 微分平坦 / 最小 snap（教程 04.3） | 平坦性是四旋翼专属结构 | 教程 04.3 的 QP 推导；本项目的轨迹优化以 iLQR 压轴 |

## 目录结构

```
projects/control_planning/
├── README.md            # 本文件
├── TUTORIAL.md          # 零基础代码导读（必读入口）
├── DEBUG.md             # 调试与测试教程（27 条重点观察变量表）
├── METRICS.md           # 测评指标教程（成功率/路径长度/轨迹代价）
├── metrics.py           # 统一测评模块（公式 M.1–M.3）
├── rrt.py | ilqr.py | mpc_cem.py | cbf.py | osc_arm.py | ppo_lite.py | mpnet_lite.py
│   └── 各算法模块（中文注释含论文式号与教程式号）
├── demo.py              # 核心入口冒烟演示（四段，确定性）
└── tests/test_*.py      # 8 个测试文件 53 个测试，全部支持单点过滤
```

## 规范

遵循根目录 [CONSTRAINTS.md](../../CONSTRAINTS.md)：§4.3 中文注释（变量含义/作用/流水线）、
§4.4 TUTORIAL、§4.5 DEBUG、§4.7 METRICS。代码改动后请运行全局测试并同步四份文档。

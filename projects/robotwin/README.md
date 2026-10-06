# projects/robotwin — RoboTwin 方法论的纯 NumPy 迷你基准（教学实现）

## 保真度声明（先读这个：本项目不是 RoboTwin 的复刻）

[RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin)（Bu et al.,
arXiv:2506.18088）是一个需要 **CoppeliaSim 仿真器 + MLLM 数据生成 + GPU 策略训练**
的大型双臂基准。本项目**不复刻其本体**——纯 NumPy 的 2D 平面世界里没有渲染、
没有语言、没有可微物理，复刻既不可能也无必要。它实现的是论文两条**方法论主线**
的可跑玩具版，让读者在 1 分钟内亲手复现教程第 04/06 章讲的
"DR 剂量 vs 成功率"曲线：

| | 已实现（本项目） | 未实现（留在官方仓库） |
|---|---|---|
| **域随机化** | 五维 DR 采样器：对象尺寸/摩擦/臂长/观测噪声/动作缩放，`none/mild/strong` 三档剂量，与论文五维的结构对照表见 [dr.py](dr.py) 模块注释 | 视觉五维（杂乱/纹理/光照/桌高/语言）——需要渲染与语言管线（论文 §2.2） |
| **基准评测协议** | 每任务 N 回合 rollout、宏平均 + **最差任务**成功率、**seen/unseen 泛化差距**，全确定种子（[benchmark.py](benchmark.py) + [metrics.py](metrics.py)） | 50 任务 × 5 本体 × 100 rollout 的全矩阵（论文 §4.5、附录 L） |
| **本体运动学** | 平面 2 连杆臂解析 FK / 双解 IK（选肘向）/ 解析-数值雅可比互证（[arm.py](arm.py)） | 7-DoF 臂、CuRobo 运动规划、体态感知抓取适配（论文 §2.3） |
| **策略对照** | 两种脚本策略：`calibrated`（信任名义 CAD 模型）vs `robust`（每回合带噪观测自标定）——**无学习**，重点是协议与 DR 演示 | MLLM 专家代码生成、ACT/DP/RDT/π0 等 VLA 策略训练——见[教程第 08 章](../../tutorials/robotwin/08_策略训练与部署.md) |

仿真器安装、数据采集与官方基准的完整实操：**[教程第 07 章](../../tutorials/robotwin/07_环境搭建与实战.md)**
与 [官方仓库](https://github.com/RoboTwin-Platform/RoboTwin)。

零基础入门请先读 **[TUTORIAL.md](TUTORIAL.md)**（每个模块的最小可运行示例 +
数据结构 + 流水线图）；调试与单点/全局测试用法见 **[DEBUG.md](DEBUG.md)**；
指标口径（成功率/宏平均+最差/泛化差距）见 **[METRICS.md](METRICS.md)**。

## 快速开始

```bash
# ① 激活环境（环境清单见根目录 environment.yml；纯 numpy，任意 numpy>=1.26 的环境亦可）
conda activate llm_env

# ② 依赖校验：本包仅依赖 numpy（无 torch/scipy/coppelia）
python -c "import numpy; print('numpy', numpy.__version__)"        # 预期: numpy 2.2.6

# ③ 验证命令：import 全部模块并打印版本
python -c "import arm, tasks, dr, policies, benchmark, metrics; print('robotwin mini-benchmark OK')"
```

```bash
# 全套测试（35 个，确定性，< 6 s）
python -m pytest tests/ -v          # 预期：35 passed
# 核心入口冒烟演示（3 任务 × 3 DR 档 × 40 回合 × 2 策略，确定性输出，~2.5 s）
python demo.py
```

- **单点测试**：`python tests/test_arm.py jacobian`（子串过滤；无匹配会列出可用测试名）
- 两个解释器均已验证：conda `llm_env`（numpy 2.2.6）与系统 python3（numpy 1.26.4），
  35/35 全绿、`demo.py` 输出逐位一致
- VS Code 调试配置见 [.vscode/launch.json](../../.vscode/launch.json)，用法见
  [.vscode/SETUP.md](../../.vscode/SETUP.md)

## 模块总览（论文 × 模块 × 教程章 × 测试数）

| 模块 | 论文对应（arXiv:2506.18088） | 教程章 | 核心内容 | 测试 |
|------|------------------------------|--------|----------|------|
| [arm.py](arm.py) | §2.3 体态适配的 IK 可达性初筛 | [02 章 §2.1](../../tutorials/robotwin/02_双臂操作与仿真基础.md) | 平面 2 连杆解析 FK、双解 IK（选肘向 + 限位）、解析/数值雅可比互证、`BimanualArm2D` 双臂状态 (4,) | 7 |
| [dr.py](dr.py) | §2.2 五维域随机化（附录 C 剂量口径） | [04 章 §4.2](../../tutorials/robotwin/04_核心技术详解.md) | `WorldParams` 世界参数 + `none/mild/strong` 三档剂量采样（strong ⊇ mild 区间守恒）+ 物理界校验/截断 | 6 |
| [tasks.py](tasks.py) | §3.2 / §4.5 任务基元（50 任务的三类代表） | [02 章 §2.2](../../tutorials/robotwin/02_双臂操作与仿真基础.md)、[05 章 §5.5](../../tutorials/robotwin/05_数据集与基准.md) | `reach / push / pick_place` 三种回合环境：统一 `reset(seed, split) → obs`、`step(action) → obs, reward, done, info`；法向接触-摩擦推、夹爪状态机 | 7 |
| [policies.py](policies.py) | §4.3 Table 3 的 clean vs DR 对照（脚本策略版） | [04 章 §4.2.3](../../tutorials/robotwin/04_核心技术详解.md)、[06 章 §6.4](../../tutorials/robotwin/06_实验结果与解读.md) | `calibrated`（名义 CAD 模型）vs `robust`（4 步探测 + 线性最小二乘自标定连杆长/半径） | 5 |
| [benchmark.py](benchmark.py) | §4.5 基准评测协议（每任务 100 rollout） | [05 章 §5.6](../../tutorials/robotwin/05_数据集与基准.md) | (regime × task × split × episode) 全网格驱动，素数混散种子规则写死，逐位可复现 | 4 |
| [metrics.py](metrics.py) | 附录 G（成功率）/ 附录 L（双层报告） | [05 章 §5.6](../../tutorials/robotwin/05_数据集与基准.md)、[06 章 §6.5–6.6](../../tutorials/robotwin/06_实验结果与解读.md) | `success_rate` / `macro_mean_and_worst`（宏平均+最差任务）/ `generalization_gap`（seen−unseen），纯函数 (M.1)–(M.3) | 6 |
| [demo.py](demo.py) | —（方法论冒烟） | [04 章 §4.2.3](../../tutorials/robotwin/04_核心技术详解.md)、[06 章 §6.4](../../tutorials/robotwin/06_实验结果与解读.md) | 3 任务 × 3 DR 档 × 40 回合 × 2 策略 → 成功率表 + 一句话结论（~2.5 s，确定性） | — |

## 运行示例（1 分钟复现"DR 剂量 vs 成功率"）

```python
from policies import CalibratedPolicy, RobustPolicy
from dr import REGIMES
import tasks as tk, benchmark as bm, metrics as mt

for name, cls in (("calibrated", CalibratedPolicy), ("robust", RobustPolicy)):
    records = bm.run(cls(), tasks=tk.TASKS, regimes=REGIMES,
                     episodes_per_task=40, seed0=0)
    table = bm.aggregate(records)
    macro = {r: mt.macro_mean_and_worst({t: table[r][t]["all"] for t in tk.TASKS})[0]
             for r in REGIMES}
    print(name, {k: round(v, 2) for k, v in macro.items()})
# calibrated {'none': 1.0, 'mild': 0.85, 'strong': 0.34}   ← 剂量↑ 成功率↓
# robust     {'none': 0.98, 'mild': 0.92, 'strong': 0.82}  ← 自标定，基本持平
```

完整表格（含逐任务成功率、最差任务、泛化差距）直接跑 `python demo.py`；
输出解读与各档健康值见 [METRICS.md](METRICS.md)，断点观察点见 [DEBUG.md](DEBUG.md)。

## 目录结构

```
projects/robotwin/
├── README.md            # 本文件（保真度声明 + 快速开始）
├── TUTORIAL.md          # 零基础代码导读（必读入口）
├── DEBUG.md             # 调试与测试教程（25 条重点观察变量表）
├── METRICS.md           # 测评指标教程（成功率/宏平均+最差/泛化差距）
├── arm.py               # 平面 2 连杆双臂运动学（FK/IK/雅可比）
├── tasks.py             # 三种双臂任务回合环境
├── dr.py                # 域随机化采样器（三档剂量 × 五维）
├── policies.py          # 两种脚本对照策略（calibrated / robust）
├── benchmark.py         # 评测协议（种子规则写死，确定性）
├── metrics.py           # 统一测评模块（公式 M.1–M.3 唯一定义处）
├── demo.py              # 核心入口冒烟演示
└── tests/test_*.py      # 6 个测试文件 35 个用例，全部支持单点过滤
```

## 规范

遵循根目录 [CONSTRAINTS.md](../../CONSTRAINTS.md)：§4.3 中文注释（变量含义/形状/
单位、流水线、教程小节引用）、§4.4 TUTORIAL、§4.5 DEBUG、§4.6 环境三段式、
§4.7 METRICS。代码改动后请运行全局测试并同步四份文档。
与 [projects/slam](../slam/README.md) 同构：文档四件套、测试过滤模板、
"论文 × 模块 × 教程章 × 测试数"模块表。

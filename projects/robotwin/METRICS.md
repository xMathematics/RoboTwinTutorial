# RoboTwin 迷你基准测评指标教程（METRICS）

> 零基础基准撰写：不假设读者有机器人学背景，术语首现给中文通俗解释。
> 代码：[metrics.py](metrics.py)（全部指标的唯一定义处）｜ 测试：[tests/test_metrics.py](tests/test_metrics.py)
> 章节互引：[tutorials/robotwin](../../tutorials/robotwin/README.md) 第 05 章 §5.6（评测协议）｜ 第 06 章 §6.4–6.6（实验解读）
> 文献出处：RoboTwin 2.0（Bu et al., **arXiv:2506.18088**）附录 G / 附录 L / §4.4–4.5

## 0. 为什么需要统一的指标

**① 问题场景**：本迷你基准每回合产出一个 `info["success"]` 布尔值，3 任务 ×
3 DR 档 × 2 分层（seen/unseen）× 2 策略 = 360 格·回合的 0/1 海洋。如果每个
测试各自手写"平均一下"，就会出现：同一数字在不同脚本里口径不同（有的含
分层、有的不分），且"报哪个数"随写表人心情漂移——教程 §5.6 的原话：
均值丢信息，"全面 60 分"与"一半满分一半零分"看起来一样。

**② 解决方法**：所有指标只在一个地方定义——`projects/robotwin/metrics.py`
的 3 个纯函数（公式编号 (M.1)–(M.3)）；`benchmark.py` / `demo.py` /
`tests/` 全部从这里取口径，不在别处重写公式（`tests/test_metrics.py` 用
朴素实现交叉验证每一个）。

**通俗概念铺垫**：

- **成功率（success rate）**：N 次执行中成功的比例。RoboTwin 2.0 的唯一
  评测指标（100 次 rollout / 任务，本迷你版 40 次/格）。
- **宏平均（macro average）**：先算每任务成功率、再对任务取简单平均——
  每个任务权重相同，不按回合数加权。
- **最差任务成功率（worst-task success）**：成功率最低的那个任务——鲁棒性
  主张应以最差情形检验（EPOpt 思想，Rajeswaran et al., ICLR 2017）。
- **泛化差距（generalization gap）**：seen 配置成功率 − unseen 配置成功率。
  正值 = 外推到没见过的配置就变弱。

### 指标速查表

| 指标 | 一句话作用 | 代码函数（`metrics.py`） | 单位 | 公式 |
|------|-----------|--------------------------|------|------|
| 成功率 | 单格（任务×档位×分层）的基本盘 | `success_rate(results)` | 无量纲 [0,1] | (M.1) |
| 宏平均 + 最差任务 | 跨任务的"档次 + 短板"双层报告 | `macro_mean_and_worst(per_task)` | 无量纲 | (M.2) |
| 泛化差距 | seen − unseen：外推损失的单个数 | `generalization_gap(seen, unseen)` | 无量纲 [−1,1] | (M.3) |

统一运行方式（详见各节"如何运行"）：

```bash
cd projects/robotwin
python tests/test_metrics.py          # 全部指标测试（6 个）
python tests/test_metrics.py naive    # 单点：只跑名字含 naive（交叉验证）的测试
pytest tests/test_metrics.py -v       # pytest 等价跑法
python demo.py                        # 端到端：协议跑分 + 三指标成表
```

---

## 1. 成功率 —— `success_rate`

### 1.1 作用

衡量什么：一个评测格（某任务 × 某 DR 档 × 某分层）内策略完成任务的比例——
整个基准的基本盘，也是 RoboTwin 2.0 **唯一**的评测指标（论文未报时间等
其他指标；教程第 06 章 §6.0 开篇即定义"成功率 = 100 次执行中成功的比例"）。
所有上层指标（宏平均/最差/差距）都是它的加权组合。

### 1.2 如何计算

$$\mathrm{SR} = \frac{1}{N}\sum_{i=1}^{N} s_i, \qquad s_i \in \{0, 1\} \tag{M.1}$$

代码位置：`metrics.py::success_rate`（纯函数；输入 0/1 数组、bool 序列或
已聚合的比率均可，同一口径）。**文献出处**：RoboTwin 2.0（arXiv:2506.18088）
附录 G——任务级成功率 = 对 M 次仿真执行的成功指示变量取平均；教程
[第 05 章 §5.6](../../tutorials/robotwin/05_数据集与基准.md)（"每任务 100 次
rollout、报告成功率"）。本项目的 $s_i$ 即 `benchmark.EpisodeRecord.success`。

### 1.3 健康值范围（本项目实测，N=40/格，seed0=0）

| 格 | 实测成功率 | 结论 |
|----|-----------|------|
| calibrated × none × 三任务 | **1.00 / 1.00 / 1.00** | 标定世界即满分——这是"标定即完美"的锚点 |
| robust × none × 三任务 | 1.00 / 0.95 / 1.00 | 自标定在无噪声档几乎无代价 |
| calibrated × strong × push | **0.17** | 剂量+摩擦双重打击下的最差格 |

异常信号：**none 档任何任务 < 0.9** → 标定世界被污染（查 `dr` 的 none 档
是否仍恒等于名义值、`tasks` 物理是否被改）；**所有格全 0** → 种子/接口断了
（策略没动，见 DEBUG.md §4 案例 1）。

### 1.4 如何运行

```bash
cd projects/robotwin
python tests/test_metrics.py naive        # 与朴素实现交叉验证（3 个测试，50 组随机对照）
python tests/test_tasks.py exact_ik       # 端到端：解析 IK 策略 12/12 seed 成功
```

在自己代码中使用：

```python
from metrics import success_rate
sr = success_rate([r.success for r in records])   # records: benchmark.run 的返回
assert sr >= 0.9                                   # 断言示例
```

---

## 2. 宏平均 + 最差任务 —— `macro_mean_and_worst`

### 2.1 作用

衡量什么：一个 DR 档位下策略的**总体档次**（宏平均）与**最大短板**（最差
任务）。对应论文的双层报告实践：正文给跨任务平均（Table 3/5 的均值列），
附录 L 公开全部 50 任务 × 5 本体逐格成功率，Table 4 把**最差配置**的相对
提升写进摘要——"报均值负责概括，报明细/最差负责可证伪"（教程 §5.6 的
知识点五步论证）。

### 2.2 如何计算

$$\overline{\mathrm{SR}}_{\mathrm{macro}} = \frac{1}{K}\sum_{k=1}^{K} r_k,
\qquad
\mathrm{SR}_{\mathrm{worst}} = \min_{k}\, r_k \tag{M.2}$$

代码位置：`metrics.py::macro_mean_and_worst`（接受 `dict[任务名, 比率]`
或 `(序列, task_names)`；返回 `(macro, worst, worst_task)`，并列最差取输入
顺序第一个，保证报告可复现）。**文献出处**：RoboTwin 2.0 附录 L（逐格
成功率的双层披露）；教程 [第 05 章 §5.6](../../tutorials/robotwin/05_数据集与基准.md)
知识点（"均值不唯一决定分布"的五步）与 [第 06 章 §6.6](../../tutorials/robotwin/06_实验结果与解读.md)
（Table 5 的均值解读法：先看均值定档次，再查最差行找改进点）。

### 2.3 健康值范围（本项目实测，demo 输出）

| 策略 × 档位 | 宏平均 | 最差任务（实测） | 结论 |
|-------------|--------|------------------|------|
| calibrated × none | **1.00** | 1.00（reach，三任务并列满） | 标定世界无短板 |
| calibrated × mild | 0.85 | 0.55（push） | 摩擦维先打掉 push |
| calibrated × strong | **0.34** | 0.17（push） | 剂量↑ 短板更深——这正是要复现的曲线 |
| robust × none / mild / strong | 0.98 / 0.92 / **0.82** | 恒为 push（0.95/0.75/0.45） | 自标定策略的短板也可预期、可解释 |

异常信号：none 档 worst < 0.9 → 某任务的标定世界物理坏了（对照
`test_policies.py::test_calibrated_none_three_tasks_at_least_090`）；最差
任务从 push 变成其他 → 有人改了那个任务的容差或物理。

### 2.4 如何运行

```bash
cd projects/robotwin
python demo.py          # 表中"宏平均 / 最差任务"两行即 (M.2) 的逐档实现
python tests/test_policies.py none   # calibrated/robust 的 none 档断言（≥0.9 / ≥0.85）
```

在自己代码中使用：

```python
import metrics as mt
macro, worst, worst_task = mt.macro_mean_and_worst(
    {"reach": 1.0, "push": 0.55, "pick_place": 1.0})
# (0.85, 0.55, 'push')
```

---

## 3. 泛化差距 —— `generalization_gap`

### 3.1 作用

衡量什么：策略在 **seen**（标定工作空间内部的目标位姿）与 **unseen**
（外围一圈，两档都在物理可达圈内）两套配置上的成功率之差——单一数字刻画
"外推损失"。对应论文的 {seen, unseen} 配置轴：Sim-to-Real 实验里
"Unseen 背景 × 杂乱"恰是最差配置（9.0% vs Seen 背景·干净的 29.5%），摘要的
"+367%" 就定义在这一轴上；教程 §5.6 ④ 引 RT-1 的 seen/unseen 分层报告
为方法论源头。

### 3.2 如何计算

$$\mathrm{Gap} = \frac{1}{K}\sum_{k} r_k^{\mathrm{seen}}
\; - \; \frac{1}{K}\sum_{k} r_k^{\mathrm{unseen}} \tag{M.3}$$

代码位置：`metrics.py::generalization_gap`（seen/unseen 各传每任务成功率
或 0/1 回合指示，长度须一致；空/不齐抛 `ValueError`）。**文献出处**：
RoboTwin 2.0 §4.4 Table 4（四种 {seen, unseen} × {干净, 杂乱} 评测配置）；
教程 [第 05 章 §5.6 ④](../../tutorials/robotwin/05_数据集与基准.md)（RT-1
的分层报告）与 [第 06 章 §6.5](../../tutorials/robotwin/06_实验结果与解读.md)
（"环境越复杂提升越大"的解读）。本项目的配置轴落在**目标位姿范围**
（`tasks.reset(seed, split)`，范围定义见 `SCENE_RANGES`）。

### 3.3 健康值范围（本项目实测，demo 输出）

| 策略 × 档位 | 实测 GAP | 结论 |
|-------------|----------|------|
| calibrated × none | **+0.00** | 模型即世界时"没见过的位置"不存在——差距恰为 0（有测试守恒：`test_benchmark.py::test_generalization_gap_zero_for_calibrated_at_none`） |
| calibrated × strong | **+0.25** | 失准的策略外推失败更严重——差距随剂量放大 |
| robust × 三档 | +0.03 / +0.13 / +0.23 | 自标定策略同样存在温和的外推损失（unseen 目标更远） |

异常信号：**单档轻微负值**（如 mild 的 −0.03）属 40 回合的采样噪声；
**持续显著负值** → seen/unseen 分层采样坏（unseen 反而更容易——查
`SCENE_RANGES` 是否被改动）；calibrated 的 none 档 gap ≠ 0 → 标定世界
被 DR 泄漏污染（`dr.sample("none", ·)` 不再恒等于名义值）。

### 3.4 如何运行

```bash
cd projects/robotwin
python demo.py          # 表中"GAP"一行即 (M.3) 的逐档实现
python tests/test_benchmark.py generalization   # calibrated@none 差距恒为 0 的守恒测试
```

在自己代码中使用：

```python
import metrics as mt
gap = mt.generalization_gap(
    [table[r][t]["seen"] for t in tasks],
    [table[r][t]["unseen"] for t in tasks])
assert gap >= -0.05, gap    # 排除采样噪声后的合理性断言
```

---

## 4. 三个指标怎么配合读（读表法）

| 你想回答的问题 | 看哪个 | 理由 |
|----------------|--------|------|
| 这个策略在标定条件下是否可用 | 成功率（none 档逐任务） | 基本盘；< 0.9 说明技能本身有 bug 而非 DR 之过 |
| DR 剂量加大后策略塌不塌 | 宏平均跨 none/mild/strong 的走向 | calibrated 1.00→0.85→0.34 vs robust 0.98→0.92→0.82——教程 §6.4 曲线的迷你复现 |
| 策略的短板在哪 | 最差任务名 | 本基准恒为 push（摩擦维伤害所有人 + 推送物理的间隙预算上限） |
| 策略会不会"只会做见过的" | 泛化差距 | calibrated 随剂量放大到 +0.25；none 档恒 0 是协议自检 |
| 两解释器结果是否一致 | `diff <(python3 demo.py) <(python demo.py)` | 应无输出——全套确定性的最终检验 |

一句话记忆：**成功率回答"做没做成"，宏平均+最差回答"整体多强、短板多短"，
泛化差距回答"换个没见过的位置还行不行"**——三者合起来正是教程 §5.6 要求
基准报告"均值 + 逐任务/最差明细 + 分层"的完整口径。所有公式的唯一定义在
[metrics.py](metrics.py)，回归测试在 [tests/test_metrics.py](tests/test_metrics.py)
（6 个测试，全部确定性）。

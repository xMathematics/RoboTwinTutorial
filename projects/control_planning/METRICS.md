# 控制与规划统一测评指标教程（METRICS）

> 零基础基准撰写：不假设读者有机器人学背景，术语首现给中文通俗解释。
> 代码：[metrics.py](metrics.py)（全部指标的唯一定义处）｜ 测试：[tests/test_metrics.py](tests/test_metrics.py)
> 章节互引：[tutorials/control_planning 总览](../../tutorials/control_planning/OVERVIEW.md)｜
> 第 03 章（可行路径）｜ 第 04 章（最优控制目标 (4.1)）｜ [CONSTRAINTS.md §4.7](../../CONSTRAINTS.md)（三指标基线的出处）

## 0. 为什么需要统一的指标

**① 问题场景**：本项目的规划/控制代码有 rrt、ilqr、mpc_cem、cbf、ppo_lite 等多个模块，每个模块的测试与 demo 都要回答同一个问题——"规划出来的路径 / 控制出来的轨迹到底好不好？"。如果每个模块各自手写"到没到""撞没撞""费不费劲"的判定，就会出现两个后果：同一数字在不同模块口径不同（有的算碰撞、有的忽略；有的带控制功效、有的只看几何），无法横向比较；且新模块的测试会继续复制粘贴旧实现，错误随之繁殖。

**② 解决方法**：所有指标只在一个地方定义——`projects/control_planning/metrics.py` 的 3 个纯函数；各模块测试与 demo 的指标断言从这里取用（`tests/test_metrics.py` 用朴素显式循环实现逐一交叉验证）。这 3 个正是 [CONSTRAINTS.md §4.7](../../CONSTRAINTS.md) 为 control_planning 点名的基线指标：**规划成功率 / 路径长度 / 轨迹代价**。

**③ 为什么这三个指标够用**：一条轨迹的"好"可以拆成三个互相独立的侧面——**任务完成**（到得了 + 不碰撞：成功率）、**几何质量**（走得多省：路径长度）、**执行代价**（控制花多大力、多平滑：轨迹代价）。rrt 验证可行性与最优性（教程第 03 章）、ilqr 验证平滑与最优控制（第 04 章）、mpc_cem 验证约束下的滚动执行（第 05 章）、cbf 验证安全保持（第 07 章）、ppo_lite 验证学习式控制（第 08 章）——它们的主张恰好各落在一两个侧面上。三者合用即可支撑全部断言；"到达容差 tol" 是唯一的口径参数，各处统一取 0.35 m（demo/测试）或 0.1 m（metrics 自身的解析用例）。

**通俗概念铺垫**（后文反复出现）：

- **点机器人**：把机器人抽象成一个点，"碰撞" = 点进入障碍圆域。教学实现的标准简化；真实机器人的体量折算进障碍半径（r_safe 余量）。
- **控制功效**：$\|u\|^2$ 的时间积分——加速度平方和，能耗/力矩方量的标准代理。控制量翻倍，功效翻四倍，因此优化器天然偏好"轻柔"的轨迹。

### 指标速查表

| 指标 | 一句话作用 | 代码函数（`metrics.py`） | 单位 | 公式 |
|------|-----------|--------------------------|------|------|
| 规划成功率 | 到达**且**全程无碰撞的轨迹占比（复合主张） | `success_rate(trajs, goal, obstacle_list, tol)` | 无量纲 | (M.1) |
| 路径长度 | 几何意义上的路程（折线总长） | `path_length(traj)` | m | (M.2) |
| 轨迹代价 | 控制功效 + 控制平滑（执行侧代价） | `trajectory_cost(traj, u_traj, dt, w_smooth)` | m²/s³（同网格相对比较） | (M.3) |

统一运行方式（详见各节"如何运行"）：

```bash
cd projects/control_planning
python3 tests/test_metrics.py            # 全部指标测试（7 个）
python3 tests/test_metrics.py success    # 单点：只跑名字含 success 的测试
pytest tests/test_metrics.py -v          # pytest 等价跑法
```

---

## 1. 规划成功率 —— `success_rate`

### 1.1 作用

衡量什么：一批轨迹里"**到达目标且全程无碰撞**"的比例。它把任务完成的两半合成一个数字：只看到达会奖励"穿墙直达"的假成功（可行性主张，教程第 03 章 (3.1) 的 $\sigma: [0,1] \to \mathcal{C}_{free}$），只看安全会奖励"原地不动"的假安全（安全集主张，教程第 07 章 (7.1) 的 $h \ge 0$）。对应各论文"成功率达到 X%"式的主张，是本项目 demo ①（RRT/RRT*）与 ③（CEM-MPC）的主数字。

### 1.2 如何计算

$$
\mathrm{SR} \;=\; \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\Big[\;\underbrace{\|p_i(T_i) - p_{goal}\| \le \epsilon}_{\text{到达}}\;\wedge\; \underbrace{\min_{t,\,j} \big(\|p_i(t) - c_j\| - r_j\big) \ge 0}_{\text{无碰撞}}\;\Big] \tag{M.1}
$$

代码位置：`metrics.py::success_rate`（逐条轨迹先判到达再逐障碍扫碰撞，两个条件缺一不可）。文献口径：采样式规划的成功率按 Karaman & Frazzoli (2011, IJRR) 的概率完备性实验惯例（多次规划的成功频数）；"到达 = 终点进入 goal 的 ε-邻域" 即教程第 03 章目标邻域判定的评估版。**穿隧注意**：碰撞只检查采样点（间距由调用方的轨迹密度决定），与教程第 03 章 (3.10) 的 motion validation 漏检口径相同——密集轨迹或保守半径可补偿。

### 1.3 健康值范围（本项目实测）

| 场景（测试/demo） | 实测成功率 | 说明 |
|--------------|-----------|------|
| demo ① RRT / RRT*（走廊障碍图，5 seed） | **1.00 / 1.00** | 中等密度障碍下采样规划应满成功率 |
| demo ③ / `test_mpc_cem`（CEM-MPC，4 seed，执行噪声 σ=0.05） | **1.00**（阈值 ≥ 0.75） | 滚动重解吸收噪声后仍应全到达 |
| `test_success_rate_manual_hit_collide_miss`（命中/碰撞/未达各 1） | 恰 **1/3** | 手工构造的口径锚点 |

异常信号：**规划器在简单场景成功率 < 1** ⇒ 采样参数（max_iters / goal_bias）或碰撞检测被改坏；**带控制的场景成功率随 seed 大幅波动** ⇒ 执行噪声与代价权重不匹配（障碍惩罚 `w_obs` 太软）。

### 1.4 如何运行

```bash
cd projects/control_planning
python3 tests/test_metrics.py success
# 预期输出（节选）：
# [rate] hit/collide/miss -> 0.3333
# 2/2 tests passed.
```

模块测试中的使用示例（mpc_cem 式）：

```python
from metrics import success_rate

sr = success_rate(trajs, goal, obstacle_list, tol=0.35)   # trajs: list[(T, 4) 或 (T, 2)]
assert sr >= 0.75, sr
```

---

## 2. 路径长度 —— `path_length`

### 2.1 作用

衡量什么：路径的**几何路程**——相邻采样点欧氏距离之和。它是路径规划文献的标准对比量（Karaman & Frazzoli 2011 的代价函数取欧氏长度，教程第 03 章 (3.5) 的逐边累加对象），回答"走得多省"：同一任务的可行解可以任意长，最优解长度是 RRT* 渐近最优性演示的直接证据（demo ①：RRT* 的代价逐 seed 低于 RRT 的 12.5–14.0，收敛到 10.4–11.0）。与轨迹代价的分工：path_length 只管几何、与控制无关；"这条轨迹执行起来费不费劲"归 `trajectory_cost`（第 3 节）。

### 2.2 如何计算

$$
L \;=\; \sum_{i=1}^{N-1} \left\| p_{i+1} - p_i \right\| \tag{M.2}
$$

代码位置：`metrics.py::path_length`（`np.diff` 一次性给出所有相邻差分，与朴素循环在 `tests/test_metrics.py` 中交叉验证到 1e-12）。输入允许 (N, 2) 或 (N, 4)（自动取位置分量，方便直接喂双积分器状态轨迹）。

### 2.3 健康值范围（本项目实测）

| 场景（测试/demo） | 实测路径长度 | 说明 |
|--------------|-----------|------|
| demo ① RRT* seed0（走廊绕障，max_iters=6000） | **10.741 m** | 起终点直线距离 ≈ 11.1 m——绕障开销很小（缺口正对角线） |
| demo ① RRT（同 seed，首条可行解） | 12.5–14.0 m（=其代价） | RRT 的次优性直观可见（(3.6) 前的分布） |
| `test_path_length_known_values` | 直线 = N−1、折返 = 2 倍、静止 = 0 | 解析锚点 |

异常信号：**RRT* 长度不随迭代下降** ⇒ 邻域半径公式 (3.7) 或重布线传播被改坏（对照 [DEBUG.md §3 R-1/R-3](DEBUG.md)）；**长度 < 起终点直线距离** ⇒ 不可能（三角不等式），碰撞检测漏了穿隧。

### 2.4 如何运行

```bash
cd projects/control_planning
python3 tests/test_metrics.py path
# 预期输出（节选）：
# 3/3 tests passed.
```

demo ① 直接打印 `path_length(res.path)`；自己模块里的用法：

```python
from metrics import path_length

L = path_length(traj)          # traj: (N, 2) 或 (N, 4)，单位 m
assert L >= straight_line      # 三角不等式下界
```

---

## 3. 轨迹代价 —— `trajectory_cost`

### 3.1 作用

衡量什么：执行一条轨迹要**花多大力、有多平滑**——控制功效 $\int \|u\|^2 \mathrm{d}t$ 加相邻控制差的平滑惩罚。对应教程第 04 章最优控制目标 (4.1) 的离散评估版：阶段代价取 $\|u\|^2$（能耗代理），平滑项把"拐点抖动"（教程 04.1 的问题场景：RRT* 折线直接交给伺服会抖动、费电）显式计价。demo ② 用它量化 iLQR 前后的执行代价差；它与 path_length 互补——几何最短的折线往往执行代价最高。

### 3.2 如何计算

$$
J \;=\; \sum_{i=0}^{N-2} \|u_i\|^2\,\Delta t \;+\; w_{\mathrm{smooth}} \sum_{i=0}^{N-3} \|u_{i+1} - u_i\|^2\,\Delta t \tag{M.3}
$$

代码位置：`metrics.py::trajectory_cost`（要求 `len(u_traj) == N − 1`：控制与状态同网格；`dt` 为采样周期，`w_smooth` 默认 0.1）。与教程 (4.1) 的对应：第一项 = 阶段代价 $\ell$ 取控制功效的矩形法离散；第二项不在 (4.1) 标准形式里，是控制平滑性的显式计价（iLQR 的解天然平滑，该项主要为折线类初值服务）。

### 3.3 健康值范围（本项目实测）

| 场景（测试/demo） | 实测 trajectory_cost | 说明 |
|--------------|-----------|------|
| demo ② iLQR 优化后（走廊任务，dt=0.1，N=60） | **12.48** | 与总代价 J = 6.331 并读：几何 + 功效各有侧重 |
| demo ② 零控制初值 | 0.000 | 不动 → 无功效，属预期（这正是它不可用的原因） |
| `test_trajectory_cost_smoothness_weight` | 常值控制平滑项 = 0；交替控制 18 + 64 | 解析锚点（功效 18 / 平滑 64） |

异常信号：**优化后代价高于初值** ⇒ iLQR 的接受判据坏（对照 [DEBUG.md §3 I-3](DEBUG.md)）；**量纲误读**——功效项 (m²/s⁴)·s 与平滑项同量纲加权，只作**同网格、同参数**下的相对比较，跨任务比较无意义。

### 3.4 如何运行

```bash
cd projects/control_planning
python3 tests/test_metrics.py cost
# 预期输出（节选）：
# [cost] 1.949859 vs naive 1.949859
# 2/2 tests passed.
```

demo ② 直接打印 iLQR 前后的 trajectory_cost；自己模块里的用法：

```python
from metrics import trajectory_cost

cost = trajectory_cost(x_traj, u_traj, dt=0.1, w_smooth=0.1)
assert cost < baseline   # 同网格同参数下比较
```

---

## 4. 三个指标怎么配合使用

| 你想回答的问题 | 用哪个 | 理由 |
|----------------|--------|------|
| 规划器/控制器能不能完成任务 | **success_rate** | 到达与安全是复合主张，单一几何量无法表达 |
| 采样规划的解质量（RRT* vs RRT） | **path_length**（=树代价口径） | 几何最优性是第 03 章的主张，与控制无关 |
| 轨迹优化/MPC/学习式控制的执行品质 | **trajectory_cost**（+ 终端误差） | 功效与平滑是第 04–08 章共同的优化对象 |
| 排障：成功了但很难看 | success_rate = 1 且 path_length 异常大 | 查规划器参数（goal_bias、eta）而非指标本身 |

一句话记忆：**success_rate 回答"能不能用"，path_length 回答"绕了多少路"，trajectory_cost 回答"使了多大劲"**——三者合起来是 CONSTRAINTS §4.7 为本主题点名的完整基线。所有公式的唯一定义在
[metrics.py](metrics.py)，回归测试在 [tests/test_metrics.py](tests/test_metrics.py)（7 个测试，全部确定性）。

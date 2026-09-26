# 论文精读｜流形上的 IMU 预积分（T-RO 2017）

> **PDF**：[papers/slam/classics/arXiv-1512.02363_IMU-Preintegration.pdf](../../../papers/slam/classics/arXiv-1512.02363_IMU-Preintegration.pdf) ｜ **教程**：[第 10 章](../10_建图与系统实战.md) §10.2 ｜ **代码**：[projects/slam/preint/](../../../projects/slam/preint/)（预积分核心）+ [projects/slam/vins/](../../../projects/slam/vins/)（因子包装）

## 1. 论文信息与一句话贡献

- **题目**：On-Manifold Preintegration for Real-Time Visual-Inertial Odometry
- **作者**：Christian Forster（苏黎世大学 UZH）、Luca Carlone、Frank Dellaert（佐治亚理工 Georgia Tech）、Davide Scaramuzza（UZH）
- **发表**：IEEE T-RO 33(1), 2017；arXiv:1512.02363（本仓库 PDF 共 21 页，为期刊扩展版——ICRA 2016 会议版 [6] 的扩充，附完整技术推导；**本精读所有式号以此 PDF 为准**，注意与会议版/二手材料式号不同）。
- **一句话贡献**：把两关键帧之间的上百次 IMU 测量**在 SO(3) 流形上预积分**（preintegration）为一条与初始状态无关的相对运动约束——给出完整的测量模型（式 (38)）、噪声协方差迭代传播（式 (59)–(63)）与免重积分的偏置一阶修正（式 (44)）——并配无结构视觉因子（structureless vision factors，式 (49)–(55)）构成实时 VIO（前端 SVO + 后端 iSAM2，后端 10ms/帧）。

## 2. 问题与动机

**要解决什么**：IMU 以数百 Hz 输出角速度与比力，相机关键帧只有几 Hz——速率差放进同一个因子图，怎么才能既"吃干净"全部 IMU 信息、又不被计算量拖死。

**三条死路**（论文 §VI 开头的论证 + §I）：

1. **逐采样建节点**：每条 IMU 测量都做一次状态节点，因子图规模随 IMU 速率爆炸（"it would require to include states in the factor graph at high rate"）。
2. **世界系积分当约束**（式 (32)）：积分式右端显含 $R_i, \mathbf{v}_i, \mathbf{p}_i$——**初值锚定**：后端每次修正线性化点（局部优化、回环都会做），以 $i$ 为起点的全部积分必须从头重算。这正是 OKVIS（引文 [24]）的做法，论文原话点破："the integration in (32) has to be repeated whenever the linearization point at time $t_i$ changes"。
3. **EKF 滤波融合**：只线性化一次、历史不可撤回；且 VIO 有 4 个不可观方向（3 个全局平移 + 重力方向 yaw），在错误的线性化点上更新会向 yaw 注入虚假信息、破坏一致性（§I 引 [16]；修复需 FEJ [17] 或可观性约束 EKF [16][18]）——教程第 07→08 章论证过的"滤波范式精度上限"。

**预积分的选择**：把积分**重定义为与初值无关的相对量**（式 (33)）——只依赖 IMU 原始测量与偏置估计；IMU 数据到来时增量更新、协方差同步传播（附录 IX-A）、偏置小修正用一阶雅可比（式 (44)）——三条死路同时绕开。

## 3. 方法总览

```
IMU 原始采样 (ω̃, ã) @数百Hz ──▶【预积分（§VI）】
   式(27)-(28) 测量模型（真值+偏置+噪声）
   式(31)-(32) 世界系积分 → 初值锚定问题
   式(33)      变量替换：ΔR/Δv/Δp ≐ 右乘消去初始姿态（偏置分段常数，式(34)）
   式(35)-(37) 噪声分离（一阶）：真值 = 预积分测量 ⊕ 小噪声
   式(38)      预积分测量模型：Δṽ = R_i^T(v_j−v_i−gΔt) + δv …（重力回到模型）
   式(59)-(63)（附录 IX-A）协方差 Σ_ij 逐样本递推 Σ ← AΣA^T + BΣ_ηB^T
   式(44)（附录 IX-B）偏置更新 δb 的一阶修正（免重积分）
                     │
相机关键帧 @2.5Hz ──▶【因子图 MAP（§IV-C + §VI-D + §VII）】
   式(25)-(26) MAP 分解：先验 + Σ IMU 因子 + Σ 视觉因子
   式(45)      IMU 残差 r_Iij（旋转项走 Log 映射）+ 式(46)-(48) 偏置随机游走
   式(49)-(55) 无结构视觉因子：每次 G-N 迭代消掉路标（Schur 补思想）
   式(20)-(21) 流形回缩 + lift-solve-retract（§III-C）→ iSAM2 增量求解
                     │
                     ▼
   VIO 输出：位姿/速度/偏置（4 个不可观方向：全局平移 + yaw）
```

## 4. 关键公式推导（式号按论文 PDF 原文）

### 4.0 符号表

| 符号 | 含义（首次出现处） |
|---|---|
| $R_{WB} \in SO(3)$, ${}_W\mathbf{p}$ | 机体系 $B$ → 世界系 $W$ 的姿态与位置；${}_W\mathbf{v}, {}_W\mathbf{a}$：世界系速度/加速度（§V） |
| ${}_B\boldsymbol{\omega}_{WB}$ | $B$ 相对 $W$ 的角速度，表达在机体系（§V） |
| ${}_W\mathbf{g}$ | 世界系重力矢量（指向下） |
| $\mathbf{b}^g, \mathbf{b}^a$ | 陀螺/加计零偏（bias），慢变（式 (46)） |
| $\boldsymbol{\eta}^g, \boldsymbol{\eta}^a$ | 测量加性白噪声；离散化后记 $\boldsymbol{\eta}^{gd}, \boldsymbol{\eta}^{ad}$（§V） |
| $\tilde{\boldsymbol{\omega}}, \tilde{\mathbf{a}}$ | 带 $\tilde{}$ = 实际测量；$\mathbf{b}_i \doteq \mathbf{b}(t_i)$，$\Delta t_{ij} \doteq \sum_k \Delta t$（§VI） |
| $i, j$ | 相邻两个相机关键帧；$\mathcal{I}_{ij}$：其间全部 IMU 测量（§IV-C） |
| $\Delta R_{ij}, \Delta\mathbf{v}_{ij}, \Delta\mathbf{p}_{ij}$ | 预积分量定义（式 (33)） |
| $\Delta\tilde{R}_{ij}, \Delta\tilde{\mathbf{v}}_{ij}, \Delta\tilde{\mathbf{p}}_{ij}$ | 预积分**测量**（含噪声，式 (35)–(38)） |
| $\boldsymbol{\eta}^\Delta_{ij} = [\delta\boldsymbol{\phi}_{ij}; \delta\mathbf{v}_{ij}; \delta\mathbf{p}_{ij}]$ | 预积分噪声 9 维矢量，$\sim \mathcal{N}(\mathbf{0}, \Sigma_{ij})$（式 (39)） |
| $J_r(\boldsymbol{\phi})$ | SO(3) 右雅可比（式 (8)） |
| $\mathrm{Exp}(\cdot), \mathrm{Log}(\cdot)$ | 指数/对数映射（式 (6) 一带的记号；$\hat{\wedge}$/$\vee$ 为 hat/vee） |
| $\Sigma_\eta$ | 原始 IMU 测量噪声 $\boldsymbol{\eta}^d$ 的 6×6 协方差（式 (62) 下方） |
| $\bar{(\cdot)}$ | 在偏置估计 $\bar{\mathbf{b}}_i$ 处的取值（式 (64)） |

### 4.1 IMU 测量模型（论文式 (27)–(28)）：加计为什么测比力

$$\tilde{\boldsymbol{\omega}}_{WB}(t) = {}_B\boldsymbol{\omega}_{WB}(t) + \mathbf{b}^g(t) + \boldsymbol{\eta}^g(t) \tag{27}$$

$${}_B\tilde{\mathbf{a}}(t) = \mathbf{R}_{WB}^\top(t) \big( {}_W\mathbf{a}(t) - {}_W\mathbf{g} \big) + \mathbf{b}^a(t) + \boldsymbol{\eta}^a(t). \tag{28}$$

**陀螺**直接测角速度（体表中）；**加计**测的是**比力**（specific force），不是运动加速度——推导：牛顿第二定律把机体真实加速度分为引力与非引力两部分，${}_W\mathbf{a} = \mathbf{f}_W + {}_W\mathbf{g}$（$\mathbf{f}_W$ 为非引力合外力产生的加速度）；加计的弹簧-质量结构只感受非引力部分，读数是其在机体系的投影 $\mathbf{f}_B = R_{WB}^\top \mathbf{f}_W = R_{WB}^\top({}_W\mathbf{a} - {}_W\mathbf{g})$——这正是 (28) 的主体。**静止检查**：${}_W\mathbf{a} = 0$ ⟹ $\tilde{\mathbf{a}} = -R_{WB}^\top {}_W\mathbf{g}$ + 偏置——水平静止时读出 $+9.81$ 沿机体"上"方向（即支撑力；`preint.ImuParams` 取 $\mathbf{g} = [0,0,-9.81]$ 指向下，docstring 的静止读数注记与此一致）。**为什么必须如此**：加计无法区分"引力产生的自由落体加速度"与"支撑力"，只能测比力——重力必须在世界系运动学里显式补回（§4.2 的 $+\mathbf{g}$ 项）。噪声与采样率关系（§V 正文）：$\mathrm{Cov}(\boldsymbol{\eta}^{gd}) = \tfrac{1}{\Delta t}\mathrm{Cov}(\boldsymbol{\eta}^g)$——数据手册的连续时间噪声密度换算到逐样本协方差要除以 $\Delta t$（`preint.py` 的 `sigma_g/sqrt(dt)` 与 $\sigma^2\Delta t$ 归一都源于此）。

### 4.2 连续时间运动学与离散积分（论文式 (29)–(31)）

运动学（式 (29)，依据：陀螺定义 ${}_B\boldsymbol{\omega} = R^\top\dot{R}$ 的积分形式与 $\dot{\mathbf{p}} = \mathbf{v}$）：

$$\dot{R}_{WB} = R_{WB}\, {}_B\boldsymbol{\omega}_{WB}^{\wedge}, \qquad {}_W\dot{\mathbf{v}} = {}_W\mathbf{a}, \qquad {}_W\dot{\mathbf{p}} = {}_W\mathbf{v}. \tag{29}$$

在 $[t, t+\Delta t]$ 上积分（区间内 $\mathbf{a}, \boldsymbol{\omega}$ 视为常值），并把 (27)–(28) 反解的真值（$\boldsymbol{\omega} = \tilde{\boldsymbol{\omega}} - \mathbf{b}^g - \boldsymbol{\eta}^g$、$\mathbf{f}_B = \tilde{\mathbf{a}} - \mathbf{b}^a - \boldsymbol{\eta}^{ad}$）代入，得式 (31)：

$$R(t+\Delta t) = R(t)\,\mathrm{Exp}\big( (\tilde{\boldsymbol{\omega}} - \mathbf{b}^g - \boldsymbol{\eta}^{gd})\,\Delta t \big)$$
$$\mathbf{v}(t+\Delta t) = \mathbf{v}(t) + \mathbf{g}\,\Delta t + R(t)\big( \tilde{\mathbf{a}} - \mathbf{b}^a - \boldsymbol{\eta}^{ad} \big)\Delta t$$
$$\mathbf{p}(t+\Delta t) = \mathbf{p}(t) + \mathbf{v}(t)\,\Delta t + \tfrac{1}{2}\mathbf{g}\,\Delta t^2 + \tfrac{1}{2}\,R(t)\big( \tilde{\mathbf{a}} - \mathbf{b}^a - \boldsymbol{\eta}^{ad} \big)\Delta t^2. \tag{31}$$

（对照：教程第 10 章 (10.2) 的连续形式与 (10.3) 的离散形式即 (29)–(31)。）三个推导要点： 旋转按 (29) 是 $\dot{R} = R\boldsymbol{\omega}^\wedge$，解为右乘指数 $R(t)\mathrm{Exp}(\int\boldsymbol{\omega}\,\mathrm{d}\tau)$，常值假设下积分退化为 $\boldsymbol{\omega}\Delta t$——常值直接形式即式 (30)（$\mathrm{Exp}({}_B\boldsymbol{\omega}_{WB}\Delta t)$ 为**精确**解），再代入 (27)–(28) 反解的真值即 (31)； **比力项要左乘 $R(t)$ 回到世界系、重力项 $\mathbf{g}$ 在世界系直接相加**（依据：4.1 的比力推导——$R\,\mathbf{f}_B = R R^\top({}_W\mathbf{a} - {}_W\mathbf{g}) = {}_W\mathbf{a} - \mathbf{g}$，运动学 $\dot{\mathbf{v}} = {}_W\mathbf{a} = R\,\mathbf{f}_B + \mathbf{g}$）； 位置的 $\tfrac{1}{2}\mathbf{g}\Delta t^2$ 与 $\tfrac{1}{2}R(\cdot)\Delta t^2$ 两项都是常加速度二级积分（依据：对 (29) 的 $\dot{\mathbf{v}}$ 再积一次分）。近似代价（论文自注）：区间内 $R(t)$ 视为常值对 (29) 不是精确解，高采样率下误差可忽略；更慢的 IMU 应换高阶数值积分（引 [54–57]）。

### 4.3 世界系多区间积分与初值锚定（论文式 (32)）

对 $k = i, \ldots, j-1$ 迭代 (31)（依据：逐区间链式代入，$R_k$ 累积进后续所有项），得式 (32)：

$$R_j = R_i \prod_{k=i}^{j-1} \mathrm{Exp}\big( (\tilde{\boldsymbol{\omega}}_k - \mathbf{b}^g_k - \boldsymbol{\eta}^{gd}_k)\,\Delta t \big), \qquad
\mathbf{v}_j = \mathbf{v}_i + \mathbf{g}\Delta t_{ij} + \sum_{k=i}^{j-1} R_k \big( \tilde{\mathbf{a}}_k - \mathbf{b}^a_k - \boldsymbol{\eta}^{ad}_k \big)\Delta t,$$
$$\mathbf{p}_j = \mathbf{p}_i + \sum_{k=i}^{j-1} \Big[ \mathbf{v}_k\,\Delta t + \tfrac{1}{2}\mathbf{g}\,\Delta t^2 + \tfrac{1}{2} R_k \big( \tilde{\mathbf{a}}_k - \mathbf{b}^a_k - \boldsymbol{\eta}^{ad}_k \big)\Delta t^2 \Big]. \tag{32}$$

**锚定问题在公式层面的呈现**：每个 $R_k = R_i \cdot (R_i^\top R_k)$ 都传导 $R_i$，速度/位置式又显含 $\mathbf{v}_i, \mathbf{p}_i$——修正 $i$ 处线性化点 ⟹ 全部乘积与求和重算（§VI 开头原话）。

### 4.4 预积分量定义：右乘消去初始姿态（论文式 (33)–(34)）

按论文 §VI-A，跟随 Lupton & Sukkarieh [2] 定义相对增量（式 (33)——注意这是**定义**，不是推导）：

$$\Delta R_{ij} \doteq R_i^\top R_j = \prod_{k=i}^{j-1} \mathrm{Exp}\big( (\tilde{\boldsymbol{\omega}}_k - \mathbf{b}^g_k - \boldsymbol{\eta}^{gd}_k)\,\Delta t \big)$$
$$\Delta\mathbf{v}_{ij} \doteq R_i^\top \big( \mathbf{v}_j - \mathbf{v}_i - \mathbf{g}\,\Delta t_{ij} \big) = \sum_{k=i}^{j-1} \Delta R_{ik} \big( \tilde{\mathbf{a}}_k - \mathbf{b}^a_k - \boldsymbol{\eta}^{ad}_k \big)\,\Delta t$$
$$\Delta\mathbf{p}_{ij} \doteq R_i^\top \Big( \mathbf{p}_j - \mathbf{p}_i - \mathbf{v}_i\,\Delta t_{ij} - \tfrac{1}{2}\mathbf{g}\,\Delta t_{ij}^2 \Big) = \sum_{k=i}^{j-1} \Big[ \Delta\mathbf{v}_{ik}\,\Delta t + \tfrac{1}{2}\,\Delta R_{ik} \big( \tilde{\mathbf{a}}_k - \mathbf{b}^a_k - \boldsymbol{\eta}^{ad}_k \big)\,\Delta t^2 \Big] \tag{33}$$

其中 $\Delta R_{ik} \doteq R_i^\top R_k$、$\Delta\mathbf{v}_{ik} \doteq R_i^\top(\mathbf{v}_k - \mathbf{v}_i - \mathbf{g}\Delta t_{ik})$。**变量替换逐步验证**（把 (32) 代入 (33) 的左端定义）：

- **旋转**：$R_i^\top R_j \xleftarrow{(32)} R_i^\top \cdot R_i \prod_k \mathrm{Exp}(\cdot) = \prod_k \mathrm{Exp}(\cdot)$（依据：矩阵乘法结合律 + $R_i^\top R_i = I$）——$R_i$ 被左乘消去，且因每个新因子从**右侧**乘入，历史累积项永不改变（`preint.py` docstring 的"右乘更新"即此）。
- **速度**：$\mathbf{v}_j - \mathbf{v}_i - \mathbf{g}\Delta t_{ij} \xleftarrow{(32)} \sum_k R_k(\cdot)\Delta t$；左乘 $R_i^\top$ 后逐项 $R_i^\top R_k = \Delta R_{ik}$（依据：(33) 第一行的定义），$R_i$ 消失。
- **位置**：同上代入，且 $\mathbf{p}_i, \mathbf{v}_i, \mathbf{g}$ 项在定义中**被显式减掉**——剩余项正是 $\sum_k [\Delta\mathbf{v}_{ik}\Delta t + \tfrac{1}{2}\Delta R_{ik}(\cdot)\Delta t^2]$（依据：(32) 位置式逐项左乘 $R_i^\top$）。

**结论**（论文原话意译）：$\Delta\mathbf{v}_{ij}, \Delta\mathbf{p}_{ij}$ **不是**物理上的速度/位置变化量，而是"为使右端只依赖 IMU 测量与相对旋转而定义的数学量"——与 $R_i, \mathbf{v}_i, \mathbf{p}_i$ 及重力全部解耦。（符号注：论文取 $\mathbf{g}$ 指向下，故 $\Delta\mathbf{p}$ 定义中是 $-\tfrac{1}{2}\mathbf{g}\Delta t_{ij}^2$、对应教程 (10.4)；若把 $\mathbf{g}$ 定义为"竖直向上的重力补偿矢量"，相应变号——读文献勿混淆。）偏置在 $[i, j]$ 上视为常值（式 (34)：$\mathbf{b}^g_i = \mathbf{b}^g_{i+1} = \cdots = \mathbf{b}^g_{j-1}$，$\mathbf{b}^a$ 同理），此后偏置变化由 §4.8 的一阶修正吸收。

**增量更新的实现顺序**（右乘性质的直接推论，`preint.Preintegration.__init__` 的循环即此）：每来一个新样本 $k$，按

$$\Delta\tilde{\mathbf{p}} \leftarrow \Delta\tilde{\mathbf{p}} + \Delta\tilde{\mathbf{v}}_{ik}\,\Delta t + \tfrac{1}{2}\,\Delta\tilde{R}_{ik}\big( \tilde{\mathbf{a}}_k - \mathbf{b}^a \big)\Delta t^2, \quad \Delta\tilde{\mathbf{v}} \leftarrow \Delta\tilde{\mathbf{v}} + \Delta\tilde{R}_{ik}\big( \tilde{\mathbf{a}}_k - \mathbf{b}^a \big)\Delta t, \quad \Delta\tilde{R} \leftarrow \Delta\tilde{R}\,\mathrm{Exp}\big( (\tilde{\boldsymbol{\omega}}_k - \mathbf{b}^g)\Delta t \big)$$

三步**就地**更新（依据：(33) 右端求和/连乘的末项即上式；顺序不可换——$\Delta\mathbf{p}$ 必须用**更新前**的 $\Delta\tilde{\mathbf{v}}_{ik}$ 与 $\Delta\tilde{R}_{ik}$，否则下标错位）。每个新样本只动末项、历史项永不重算，"随 IMU 数据到来增量更新并缓存"由此落地。

### 4.5 噪声分离与预积分测量模型（论文式 (35)–(38)）

**旋转**（式 (35)）：对 (33) 旋转式的每个乘积因子用一阶近似 (7)（依据：$\mathrm{Exp}(\boldsymbol{\phi} + \delta\boldsymbol{\phi}) \approx \mathrm{Exp}(\boldsymbol{\phi})\,\mathrm{Exp}(J_r(\boldsymbol{\phi})\,\delta\boldsymbol{\phi})$，旋转噪声"小"）把噪声移到因子末尾，再用 (11)（依据：$\mathrm{Exp}(\boldsymbol{\phi})R = R\,\mathrm{Exp}(R^\top\boldsymbol{\phi})$）把各因子噪声逐个搬过部分连乘：

$$\Delta R_{ij} \simeq \prod_{k=i}^{j-1} \mathrm{Exp}\big( (\tilde{\boldsymbol{\omega}}_k - \mathbf{b}^g_i)\,\Delta t \big)\, \mathrm{Exp}\big( -J^k_r\,\boldsymbol{\eta}^{gd}_k\,\Delta t \big) \xrightarrow{\text{(11)}} \Delta\tilde{R}_{ij} \prod_{k=i}^{j-1} \mathrm{Exp}\big( -\Delta\tilde{R}_{k+1,j}^\top J^k_r\,\boldsymbol{\eta}^{gd}_k\,\Delta t \big) \doteq \Delta\tilde{R}_{ij}\,\mathrm{Exp}\big( -\delta\boldsymbol{\phi}_{ij} \big), \tag{35}$$

其中预积分旋转测量 $\Delta\tilde{R}_{ij} \doteq \prod_k \mathrm{Exp}((\tilde{\boldsymbol{\omega}}_k - \mathbf{b}^g_i)\Delta t)$（偏置按 (34) 取 $\mathbf{b}^g_i$）。**速度**（式 (36)）：把 $\Delta R_{ik} = \Delta\tilde{R}_{ik}\,\mathrm{Exp}(-\delta\boldsymbol{\phi}_{ik}) \approx \Delta\tilde{R}_{ik}(I - \delta\boldsymbol{\phi}_{ik}^{\wedge})$（依据：式 (4) 一阶截断 $\mathrm{Exp}(-\delta\boldsymbol{\phi}^{\wedge}) \approx I - \delta\boldsymbol{\phi}^{\wedge}$）代入 (33) 速度式：

$$\Delta\mathbf{v}_{ij} \simeq \sum_{k=i}^{j-1} \Delta\tilde{R}_{ik}\big( I - \delta\boldsymbol{\phi}_{ik}^{\wedge} \big)\big( \tilde{\mathbf{a}}_k - \mathbf{b}^a_i \big)\Delta t - \sum_{k=i}^{j-1} \Delta\tilde{R}_{ik}\,\boldsymbol{\eta}^{ad}_k\,\Delta t$$

丢弃二阶噪声项（依据：$\delta\boldsymbol{\phi} \cdot \boldsymbol{\eta}$ 为二阶小量），第二项合并、第一项用式 (2)（依据：反对称交换 $\mathbf{a}^{\wedge}\mathbf{b} = -\mathbf{b}^{\wedge}\mathbf{a}$，把 $(I - \delta\boldsymbol{\phi}^{\wedge})(\tilde{\mathbf{a}} - \mathbf{b}^a)$ 展开后的 $-\delta\boldsymbol{\phi}^{\wedge}(\tilde{\mathbf{a}} - \mathbf{b}^a)$ 改写为 $(\tilde{\mathbf{a}} - \mathbf{b}^a)^{\wedge}\delta\boldsymbol{\phi}$）整理符号：

$$\Delta\mathbf{v}_{ij} \simeq \underbrace{\sum_{k=i}^{j-1} \Delta\tilde{R}_{ik}\big( \tilde{\mathbf{a}}_k - \mathbf{b}^a_i \big)\Delta t}_{\doteq\ \Delta\tilde{\mathbf{v}}_{ij}} - \underbrace{\sum_{k=i}^{j-1} \Big[ \Delta\tilde{R}_{ik}\big( \tilde{\mathbf{a}}_k - \mathbf{b}^a_i \big)^{\wedge} \delta\boldsymbol{\phi}_{ik} + \Delta\tilde{R}_{ik}\,\boldsymbol{\eta}^{ad}_k \Big]\Delta t}_{\doteq\ \delta\mathbf{v}_{ij}} \;\Longrightarrow\; \Delta\mathbf{v}_{ij} \simeq \Delta\tilde{\mathbf{v}}_{ij} - \delta\mathbf{v}_{ij}. \tag{36}$$

**位置**（式 (37)）：同理代入 (33) 位置式（并利用 $\Delta\mathbf{v}_{ik} = \Delta\tilde{\mathbf{v}}_{ik} - \delta\mathbf{v}_{ik}$）：$\Delta\mathbf{p}_{ij} \simeq \Delta\tilde{\mathbf{p}}_{ij} - \delta\mathbf{p}_{ij}$，其中 $\Delta\tilde{\mathbf{p}}_{ij} \doteq \sum_k \big[ \Delta\tilde{\mathbf{v}}_{ik}\,\Delta t + \tfrac{1}{2}\Delta\tilde{R}_{ik}(\tilde{\mathbf{a}}_k - \mathbf{b}^a_i)\Delta t^2 \big]$。**测量模型**：把 (35)–(37) 代回 (33) 的定义并移项（依据：$\mathrm{Exp}(-\delta\boldsymbol{\phi})^{-1} = \mathrm{Exp}(\delta\boldsymbol{\phi})$ 及线性移项），得式 (38)：

$$\Delta\tilde{R}_{ij} = R_i^\top R_j\,\mathrm{Exp}\big( \delta\boldsymbol{\phi}_{ij} \big), \qquad
\Delta\tilde{\mathbf{v}}_{ij} = R_i^\top \big( \mathbf{v}_j - \mathbf{v}_i - \mathbf{g}\,\Delta t_{ij} \big) + \delta\mathbf{v}_{ij}, \qquad
\Delta\tilde{\mathbf{p}}_{ij} = R_i^\top \Big( \mathbf{p}_j - \mathbf{p}_i - \mathbf{v}_i\,\Delta t_{ij} - \tfrac{1}{2}\mathbf{g}\,\Delta t_{ij}^2 \Big) + \delta\mathbf{p}_{ij}. \tag{38}$$

形式为"复合测量 = 状态的函数 + 小噪声"（重力项回到模型——对照教程 (10.5)），直接充当因子图的测量模型：零噪声下反解即状态预测（`preint.Preintegration.predict`），带噪声下按高斯负对数写残差（§4.9）。

### 4.6 预积分噪声的线性表达（论文式 (39)–(43)）

$\boldsymbol{\eta}^\Delta_{ij} = [\delta\boldsymbol{\phi}_{ij}^\top, \delta\mathbf{v}_{ij}^\top, \delta\mathbf{p}_{ij}^\top]^\top \sim \mathcal{N}(\mathbf{0}_{9\times 1}, \Sigma_{ij})$（式 (39)，一阶近似）。对 (35) 两侧取 Log 变号（式 (41)）：

$$\delta\boldsymbol{\phi}_{ij} = -\mathrm{Log}\Big( \prod_{k=i}^{j-1} \mathrm{Exp}\big( -\Delta\tilde{R}_{k+1,j}^\top\, J^k_r\, \boldsymbol{\eta}^{gd}_k\,\Delta t \big) \Big), \tag{41}$$

再用 (9)（依据：$\mathrm{Log}(\mathrm{Exp}(\boldsymbol{\phi})\mathrm{Exp}(\delta\boldsymbol{\phi})) \approx \boldsymbol{\phi} + J_r^{-1}(\boldsymbol{\phi})\,\delta\boldsymbol{\phi}$）反复展开——噪声小、右雅可比近单位阵（论文原话："recall that $\boldsymbol{\eta}^{gd}_k$ as well as $\delta\boldsymbol{\phi}_{ij}$ are small rotation noises, hence the right Jacobians are close to the identity"），多个 $\mathrm{Exp}$ 的 Log 退化为各自变元的线性组合，得线性表达（式 (42)）：

$$\delta\boldsymbol{\phi}_{ij} \simeq \sum_{k=i}^{j-1} \Delta\tilde{R}_{k+1,j}^\top\, J^k_r\, \boldsymbol{\eta}^{gd}_k\,\Delta t. \tag{42}$$

（即教程 (10.6)。）$\delta\mathbf{v}_{ij}, \delta\mathbf{p}_{ij}$ 是 $\boldsymbol{\eta}^{ad}_k$ 与 $\delta\boldsymbol{\phi}_{ik}$ 的线性组合（式 (43)）：

$$\delta\mathbf{v}_{ij} \simeq \sum_{k=i}^{j-1} \Big[ -\Delta\tilde{R}_{ik}\big( \tilde{\mathbf{a}}_k - \mathbf{b}^a_i \big)^{\wedge} \delta\boldsymbol{\phi}_{ik}\,\Delta t + \Delta\tilde{R}_{ik}\,\boldsymbol{\eta}^{ad}_k\,\Delta t \Big], \tag{43}$$

$\delta\mathbf{p}_{ij}$ 同式 (43) 的位置行（$+\delta\mathbf{v}_{ik}\Delta t - \tfrac{1}{2}\Delta\tilde{R}_{ik}(\tilde{\mathbf{a}}_k - \mathbf{b}^a_i)^{\wedge}\delta\boldsymbol{\phi}_{ik}\Delta t^2 + \tfrac{1}{2}\Delta\tilde{R}_{ik}\boldsymbol{\eta}^{ad}_k\Delta t^2$）。既然 $\boldsymbol{\eta}^\Delta_{ij}$ 是逐样本噪声的线性函数，其协方差可由 $\boldsymbol{\eta}^d$ 的协方差线性传播——这正是附录 IX-A 的出发点。

### 4.7 协方差的迭代传播与 A/B 矩阵（论文式 (59)–(63)，附录 IX-A）

**迭代化**（附录 IX-A 的手法）：把 (42)–(43) 求和的末项 $k = j-1$ 拆出、用 $\Delta\tilde{R}_{k+1,j} = \Delta\tilde{R}_{k+1,j-1}\,\Delta\tilde{R}_{j-1,j}$ 归并（依据：乘积分裂 + 转置逆序），得三个一阶递推（式 (59)–(61)；记 $\tilde{\boldsymbol{\phi}} = (\tilde{\boldsymbol{\omega}} - \mathbf{b}^g)\Delta t$，$\Delta\tilde{R}_{j-1,j} = \mathrm{Exp}(\tilde{\boldsymbol{\phi}})$）：

$$\delta\boldsymbol{\phi}_{ij} = \mathrm{Exp}(\tilde{\boldsymbol{\phi}})^\top\, \delta\boldsymbol{\phi}_{i,j-1} + J^{j-1}_r\,\boldsymbol{\eta}^{gd}_{j-1}\,\Delta t \tag{59}$$
$$\delta\mathbf{v}_{ij} = \delta\mathbf{v}_{i,j-1} - \Delta\tilde{R}_{i,j-1}\big( \tilde{\mathbf{a}}_{j-1} - \mathbf{b}^a_i \big)^{\wedge} \delta\boldsymbol{\phi}_{i,j-1}\,\Delta t + \Delta\tilde{R}_{i,j-1}\,\boldsymbol{\eta}^{ad}_{j-1}\,\Delta t \tag{60}$$
$$\delta\mathbf{p}_{ij} = \delta\mathbf{p}_{i,j-1} + \delta\mathbf{v}_{i,j-1}\,\Delta t - \tfrac{1}{2}\,\Delta\tilde{R}_{i,j-1}\big( \tilde{\mathbf{a}}_{j-1} - \mathbf{b}^a_i \big)^{\wedge} \delta\boldsymbol{\phi}_{i,j-1}\,\Delta t^2 + \tfrac{1}{2}\,\Delta\tilde{R}_{i,j-1}\,\boldsymbol{\eta}^{ad}_{j-1}\,\Delta t^2. \tag{61}$$

（每步依据：$\delta\boldsymbol{\phi}_{i,j-1}$ 的定义式即拆出末项后的和——例如 (59) 中 $\Delta\tilde{R}_{j-1,j}^\top = \mathrm{Exp}(\tilde{\boldsymbol{\phi}})^\top$、$J^{j-1}_r$ 为该点右雅可比。）写成紧凑线性模型（式 (62)）并传播协方差（式 (63)）：

$$\boldsymbol{\eta}^\Delta_{ij} = A_{j-1}\,\boldsymbol{\eta}^\Delta_{i,j-1} + B_{j-1}\,\boldsymbol{\eta}^d_{j-1}, \qquad
\Sigma_{ij} = A_{j-1}\,\Sigma_{i,j-1}\,A_{j-1}^\top + B_{j-1}\,\Sigma_\eta\,B_{j-1}^\top,\quad \Sigma_{ii} = 0_{9\times 9}. \tag{62}\text{–}(63)$$

**A/B 分块逐块写出**（论文正文只给紧凑形式，未印出分块矩阵；以下由 (62) 与 (59)–(61) 逐项比对读出——`preint._propagate_covariance` 的实现即此）：$\boldsymbol{\eta}^d = [\boldsymbol{\eta}^{gd}; \boldsymbol{\eta}^{ad}]$，

$$A = \begin{bmatrix} \mathrm{Exp}(\tilde{\boldsymbol{\phi}})^\top & 0 & 0 \\ -\Delta\tilde{R}_{i,j-1}(\tilde{\mathbf{a}} - \mathbf{b}^a)^{\wedge}\Delta t & I & 0 \\ -\tfrac{1}{2}\Delta\tilde{R}_{i,j-1}(\tilde{\mathbf{a}} - \mathbf{b}^a)^{\wedge}\Delta t^2 & I\,\Delta t & I \end{bmatrix}, \qquad
B = \begin{bmatrix} J_r(\tilde{\boldsymbol{\phi}}) & 0 \\ 0 & \Delta\tilde{R}_{i,j-1} \\ 0 & \tfrac{1}{2}\Delta\tilde{R}_{i,j-1}\,\Delta t \end{bmatrix}$$

（$A$ 各块来源：第一行 = (59)；第二行 = (60) 的 $\delta\boldsymbol{\phi}$ 项与单位阵；第三行 = (61) 的 $\delta\boldsymbol{\phi}$、$\delta\mathbf{v}$ 项与单位阵。$B$ 各块来源：(59) 的 $J_r\boldsymbol{\eta}^{gd}\Delta t$、(60) 的 $\Delta\tilde{R}\boldsymbol{\eta}^{ad}\Delta t$、(61) 的 $\tfrac{1}{2}\Delta\tilde{R}\boldsymbol{\eta}^{ad}\Delta t^2$。约定注：$B$ 各块乘的是**区间积分噪声** $\boldsymbol{\eta}^{gd}\Delta t$ / $\boldsymbol{\eta}^{ad}\Delta t$，其方差 $\sigma^2\Delta t$ 恰是 §4.1 的逐样本方差 $\sigma^2/\Delta t$ 乘以 $(\Delta t)^2$——故 $\Sigma_\eta = \mathrm{diag}(\sigma_g^2\Delta t\, I_3, \sigma_a^2\Delta t\, I_3)$，`preint._propagate_covariance` 按此组装。）结构上与 EKF 预测协方差 $\Sigma^- = A\Sigma A^\top + Q$（教程第 03 章 (3.9)）完全同构、教程 (10.7) 即式 (63)——每个新采样到达只更新 $\Sigma$、无需存储历史测量。（注记：`preint.py` docstring 把 A/B 递推同时标注为"论文 Eq.(46)-(47) / 紧凑形式 (62)-(63)"——本 PDF 中 (46)–(47) 实为偏置随机游走式，协方差递推的正确式号是 (62)–(63)、一阶递推是 (59)–(61)，读代码时以本行为准。）

### 4.8 偏置一阶修正与偏置雅可比（论文式 (44)，附录 IX-B）

后端优化会更新偏置估计 $\bar{\mathbf{b}}_i \to \hat{\mathbf{b}}_i = \bar{\mathbf{b}}_i + \delta\mathbf{b}_i$；重积分代价高，改用一阶泰勒展开（式 (44)）：

$$\Delta\tilde{R}_{ij}(\mathbf{b}^g_i) \simeq \Delta\tilde{R}_{ij}(\bar{\mathbf{b}}^g_i)\,\mathrm{Exp}\Big( \tfrac{\partial\Delta\bar{R}_{ij}}{\partial\mathbf{b}^g}\,\delta\mathbf{b}^g \Big), \qquad
\Delta\tilde{\mathbf{v}}_{ij} \simeq \Delta\bar{\mathbf{v}}_{ij} + \tfrac{\partial\Delta\bar{\mathbf{v}}_{ij}}{\partial\mathbf{b}^g}\delta\mathbf{b}^g + \tfrac{\partial\Delta\bar{\mathbf{v}}_{ij}}{\partial\mathbf{b}^a}\delta\mathbf{b}^a, \qquad \Delta\tilde{\mathbf{p}}_{ij}\ \text{同理}. \tag{44}$$

**$\partial\Delta\bar{R}/\partial\mathbf{b}^g$ 的推导链**（附录 IX-B，逐步）： 写新偏置下的连乘（式 (65)）：$\Delta\tilde{R}_{ij}(\hat{\mathbf{b}}_i) = \prod_k \mathrm{Exp}((\tilde{\boldsymbol{\omega}}_k - \hat{\mathbf{b}}^g_i)\Delta t)$； 代入 $\hat{\mathbf{b}}_i = \bar{\mathbf{b}}_i + \delta\mathbf{b}_i$ 并对每个因子用 (7)（依据：$\delta\mathbf{b}^g_i$ 小）：$\mathrm{Exp}((\tilde{\boldsymbol{\omega}}_k - \bar{\mathbf{b}}^g_i)\Delta t)\,\mathrm{Exp}(-J^k_r\,\delta\mathbf{b}^g_i\,\Delta t)$（式 (66)）； 用 (11) 把各 $\mathrm{Exp}(-J^k_r\delta\mathbf{b}^g_i\Delta t)$ 搬到连乘末尾（依据同 §4.5），整理得（式 (67)–(68)）：

$$\Delta\tilde{R}_{ij}(\hat{\mathbf{b}}_i) = \Delta\bar{R}_{ij}\,\mathrm{Exp}\Big( -\sum_{k=i}^{j-1} \Delta\tilde{R}_{k+1,j}(\bar{\mathbf{b}}_i)^\top\, J^k_r\,\delta\mathbf{b}^g_i\,\Delta t \Big) \;\Longrightarrow\; \frac{\partial\Delta\bar{R}_{ij}}{\partial\mathbf{b}^g} = -\sum_{k=i}^{j-1} \Big[ \Delta\tilde{R}_{k+1,j}(\bar{\mathbf{b}}_i)^\top\, J^k_r\,\Delta t \Big].$$

速度/位置雅可比按 (36)–(37) 的同型推导链式累积——新偏置下的速度测量展开（式 (69)）为

$$\Delta\tilde{\mathbf{v}}_{ij}(\hat{\mathbf{b}}_i) = \sum_{k=i}^{j-1} \Delta\tilde{R}_{ik}(\hat{\mathbf{b}}_i)\big( \tilde{\mathbf{a}}_k - \hat{\mathbf{b}}^a_i \big)\Delta t \;\simeq\; \Delta\bar{\mathbf{v}}_{ij} - \sum_{k=i}^{j-1} \Big[ \Delta\bar{R}_{ik}\big( \tilde{\mathbf{a}}_k - \bar{\mathbf{b}}^a_i \big)^{\wedge} \tfrac{\partial\Delta\bar{R}_{ik}}{\partial\mathbf{b}^g}\,\delta\mathbf{b}^g_i + \Delta\bar{R}_{ik}\,\delta\mathbf{b}^a_i \Big]\Delta t$$

（依据：旋转因子 $\Delta\tilde{R}_{ik}(\hat{\mathbf{b}}) \simeq \Delta\bar{R}_{ik}\mathrm{Exp}\big( \tfrac{\partial\Delta\bar{R}_{ik}}{\partial\mathbf{b}^g}\delta\mathbf{b}^g \big)$ 一阶展开 + 加计偏置直接进入比力项；附录注明"for (a), we used $\Delta\bar{\mathbf{v}}_{ij} = \sum_k \Delta\bar{R}_{ik}(\tilde{\mathbf{a}}_k - \bar{\mathbf{b}}^a_i)\Delta t$"）。汇总附录 IX-B 的雅可比表：

$$\frac{\partial\Delta\bar{R}_{ij}}{\partial\mathbf{b}^g} = -\sum_{k=i}^{j-1} \Big[ \Delta\tilde{R}_{k+1,j}(\bar{\mathbf{b}}_i)^\top J^k_r\,\Delta t \Big], \quad
\frac{\partial\Delta\bar{\mathbf{v}}_{ij}}{\partial\mathbf{b}^a} = -\sum_{k=i}^{j-1} \Delta\bar{R}_{ik}\,\Delta t, \quad
\frac{\partial\Delta\bar{\mathbf{v}}_{ij}}{\partial\mathbf{b}^g} = -\sum_{k=i}^{j-1} \Big( \Delta\bar{R}_{ik}\big( \tilde{\mathbf{a}}_k - \bar{\mathbf{b}}^a_i \big)^{\wedge} \Big)\frac{\partial\Delta\bar{R}_{ik}}{\partial\mathbf{b}^g}\,\Delta t,$$

位置两行 $\tfrac{\partial\Delta\bar{\mathbf{p}}}{\partial\mathbf{b}^{g/a}}$ 再各多一层 $\Delta t / \tfrac{1}{2}\Delta t^2$ 链（$\tfrac{\partial\Delta\bar{\mathbf{p}}_{ij}}{\partial\mathbf{b}^a} = \sum_k \big[ \tfrac{\partial\Delta\bar{\mathbf{v}}_{ik}}{\partial\mathbf{b}^a}\Delta t - \tfrac{1}{2}\Delta\bar{R}_{ik}\Delta t^2 \big]$ 等）——**全部可随新采样增量计算**（附录原话"the Jacobians can be computed incrementally"；`preint._propagate_bias_jacobians` 的 9×6 `j_bias` 即此递推，`correct()` 按式 (44) 施加、$O(1)$ 计算量、修正残差 $O(\delta\mathbf{b}^2)$）。推导结构对照：与 §4.5 的噪声分离是**同一套 (7)+(11) 机制**——把"噪声"换成"偏置增量"重走一遍（附录 IX-B 开头原话："the derivation… is very similar"）。

### 4.9 IMU 残差与因子图 MAP（论文式 (24)–(26)、(45)–(48)、(20)–(21)）

**MAP**（式 (25)–(26)，依据：各测量条件独立，后验按因子分解；负对数把乘积变求和）：

$$\mathcal{X}_k^* = \arg\min_{\mathcal{X}_k} \Big( \|\mathbf{r}_0\|^2_{\Sigma_0} + \sum_{(i,j) \in \mathcal{K}_k} \|\mathbf{r}_{\mathcal{I}_{ij}}\|^2_{\Sigma_{ij}} + \sum_{i \in \mathcal{K}_k} \sum_{l \in \mathcal{C}_i} \|\mathbf{r}_{\mathcal{C}_{il}}\|^2_{\Sigma_\mathcal{C}} \Big). \tag{26}$$

**IMU 残差**（式 (45)——由测量模型 (38) 移项 + 偏置修正 (44) 一并代入；旋转残差必须经 $\mathrm{Log}$ 映射写入向量空间才能被高斯加权，依据：式 (16) 的 SO(3) 负对数似然）：

$$\mathbf{r}_{\Delta R_{ij}} = \mathrm{Log}\Big( \big( \Delta\tilde{R}_{ij}(\bar{\mathbf{b}}^g_i)\,\mathrm{Exp}\big( \tfrac{\partial\Delta\bar{R}_{ij}}{\partial\mathbf{b}^g}\delta\mathbf{b}^g \big) \big)^{\!\top} R_i^\top R_j \Big), \quad
\mathbf{r}_{\Delta\mathbf{v}_{ij}} = R_i^\top(\mathbf{v}_j - \mathbf{v}_i - \mathbf{g}\Delta t_{ij}) - \big( \Delta\tilde{\mathbf{v}}_{ij} + \tfrac{\partial\Delta\bar{\mathbf{v}}_{ij}}{\partial\mathbf{b}}\delta\mathbf{b} \big),$$

$\mathbf{r}_{\Delta\mathbf{p}_{ij}} = R_i^\top\big( \mathbf{p}_j - \mathbf{p}_i - \mathbf{v}_i\,\Delta t_{ij} - \tfrac{1}{2}\mathbf{g}\,\Delta t_{ij}^2 \big) - \big( \Delta\tilde{\mathbf{p}}_{ij} + \tfrac{\partial\Delta\bar{\mathbf{p}}_{ij}}{\partial\mathbf{b}}\,\delta\mathbf{b} \big)$——其中 $\delta\mathbf{b} = \mathbf{b}_i - \bar{\mathbf{b}}_i$ 为优化中产生的偏置修正（"in which we also included the bias updates of Eq. (44)"）；残差方向取"式 (33) 的零噪声状态侧 − 偏置修正后的预积分测量侧"，与教程 (10.10) 一致（残差整体变号不改变马氏范数的最小点，与第 03 章 $\mathbf{z} = h(\mathbf{x}) + \mathbf{v}$ 的"观测 − 预测"约定只是符号之差）。**偏置随机游走**（式 (46)–(48)）：$\dot{\mathbf{b}} = \boldsymbol{\eta}$（Brownian 模型）⟹ $\mathbf{b}_j = \mathbf{b}_i + \boldsymbol{\eta}^d$、$\mathrm{Cov}(\boldsymbol{\eta}^{bd}) = \Delta t_{ij}\,\mathrm{Cov}(\boldsymbol{\eta}^b)$，作为额外因子进 (26)：

$$\|\mathbf{r}_{\mathbf{b}_{ij}}\|^2 = \|\mathbf{b}^g_j - \mathbf{b}^g_i\|^2_{\Sigma^{bgd}} + \|\mathbf{b}^a_j - \mathbf{b}^a_i\|^2_{\Sigma^{bad}}. \tag{48}$$

**流形上的 G-N**（§III-C）：变量在流形上，不能直接加增量（过参数化 + 解不在流形上——与教程第 08 章 (8.8) 的论证同源）；定义回缩 $\mathcal{R}(\cdot)$ 并 lift 重参数化（式 (17)–(18)），本文取（式 (20)–(21)）：

$$\mathcal{R}_R(\boldsymbol{\phi}) = R\,\mathrm{Exp}(\delta\boldsymbol{\phi}), \qquad \mathcal{R}_T(\delta\boldsymbol{\phi}, \delta\mathbf{p}) = \big( R\,\mathrm{Exp}(\delta\boldsymbol{\phi}),\ \mathbf{p} + R\,\delta\mathbf{p} \big), \tag{20}\text{–}(21)$$

按 lift–solve–retract 迭代（残差线性化为式 (52) 的 $\|F_{il}\,\delta T_i + E_{il}\,\delta\rho_l - b_{il}\|^2$ 形态、解正规方程、回缩更新）——与教程第 08 章 (8.5)/(8.9) 的流形 G-N 循环一一对应（约定差异：论文回缩把扰动放在**右侧**且旋转用右扰动；教程 (8.9) 与本仓库 `vins._retract_state` 用左乘 $\mathrm{Exp}(\delta\boldsymbol{\xi}^\wedge)\mathbf{T}$——同一思想的两种约定，雅可比的扰动侧需与回缩一致，论文 (57) 下方对这一自洽性有专门说明）。求解用 iSAM2 增量平滑（Bayes 树），每次更新约 10ms。

### 4.10 无结构视觉因子（论文式 (49)–(55)，简述）

重投影残差（式 (50)）$\mathbf{r}_{\mathcal{C}_{il}} = \mathbf{z}_{il} - \pi(R_i, \mathbf{p}_i, \boldsymbol{\rho}_l)$ 若直接用会把全部路标拖进优化。论文的**无结构**（structureless）做法：每次 G-N 迭代中，对每个路标 $l$ 把其扰动 $\delta\boldsymbol{\rho}_l$ 从二次代价 (53) 中解析消去（式 (54)：$\delta\boldsymbol{\rho}_l = -(E_l^\top E_l)^{-1}E_l^\top(F_l\delta T_{\mathcal{X}(l)} - b_l)$，依据：对 $\delta\boldsymbol{\rho}_l$ 求导置零），代入得只含位姿的小因子集（式 (55)，投影矩阵 $I - E_l(E_l^\top E_l)^{-1}E_l^\top$）——即 Schur 补思想的逐迭代版本（教程第 08 章 (8.13)–(8.15) 的同一消元；区别：标准 BA 回代更新路标线性化点，本文用快速线性三角化从位姿线性化点更新路标）。路标不再持久存在，连接模式随之收缩（Fig. 3）。

## 5. 实验与结果解读

（口径注：论文实验为**仿真 + 自采 VI-Sensor 数据**，未用 EuRoC——EuRoC 评测属后续工作如 VINS-Mono/ORB-SLAM3；以下数字均出自 PDF §VIII。）

- **仿真**（§VIII-A）：半径 3m 圆轨迹 + 正弦垂直运动、全程 120m；每帧 50 个路标观测、$\sigma_{px} = 1$ 像素、焦距 315 像素、关键帧 2.5Hz；IMU 参数（脚注 2）：$\sigma_g = 0.0007\,\mathrm{rad/(s\sqrt{Hz})}$、$\sigma_a = 0.019$、$\sigma_{bg} = 0.0004$、$\sigma_{ba} = 0.012$——`preint.ImuParams` 默认值即取自这里。50 次 Monte-Carlo。
- **精度与效率**：iSAM2 与批量 MAP 精度基本一致（Fig. 7）、每帧更新约 10ms 且近似恒定（Fig. 6）——增量平滑保住了批量的精度与实时的代价。
- **一致性**（Fig. 8）：姿态/位置误差在 3σ 界内；NEES（式 (56)–(58)）经 $\chi^2$ 检验（$\alpha = 2.5\%$）通过。可观性读数：重力方向可观 ⟹ roll/pitch 有界；全局 yaw 与位置不可观、协方差随时间缓慢增长——与 §2 的"4 个不可观方向"一致。
- **偏置估计**（Fig. 10）：估计的陀螺/加计偏置正确跟踪真值（平滑估计的历史曲线特性使然）。
- **一阶偏置修正**（Fig. 11）：对 $|\delta\mathbf{b}| \in [0.04, 0.2]$ 的扰动，式 (44) 相对重积分的误差可忽略（1000 次 MC）——**免重积分的代价在实际偏置幅度下可接受**。
- **对比 Euler 角预积分**（Fig. 12–13）：原方法 [2] 的 Euler 角版本在角速率 1–3 rad/s 时积分误差快速累积（Euler 角积分只精确到一阶），且其噪声传播依赖平台运动——所提方法与运动无关地准确估计测量协方差。
- **真机**（§VIII-B）：单目 VIO（SVO 前端 + iSAM2 后端，前端 3ms/后端 10ms）。室内（动捕真值）精度优于 OKVIS（后者每次线性化点变化都要重做 IMU 积分）；室外对比 Google Tango：绕办公楼一圈的轨迹（首尾同址）端到端误差 **1.5m vs 2.2m**，跨三层楼返回起点 **0.5m vs 1.4m**（两者传感器不同，作者注明只作定性比较）。

## 6. 局限与后续影响

**局限**：

1. **偏置可观测性与一阶修正的有效域**：预积分假设 $[i,j]$ 内偏置常值（式 (34)）、修正线性于 $\delta\mathbf{b}$（式 (44)）——偏置真值大幅跳变或长时间无视觉约束时，一阶近似失效（Fig. 11 的可忽略性只在 $|\delta\mathbf{b}| \lesssim 0.2$ 内验证）；此时只能重积分。
2. **线性化点与一致性**：预积分量本身与线性化点无关（对比 OKVIS 是它的卖点），但 IMU 残差 (45) 的**雅可比**仍在当前估计处取值；滑窗实现里若边缘化先验与雅可比用不同线性化点，会向不可观方向（全局平移 + yaw）注入虚假信息、破坏一致性——论文在 §I 把 FEJ（first-estimates Jacobians，引文 [17]，"ensures that a state is not updated with different linearization points"）作为滤波端的对应修复提及，其平滑路线靠 iSAM2 的重线性化 + §VIII-A 的一致性实验兜底；工程实践（VINS-Mono 系）在滑窗中显式采用 FEJ。（对照：[DSO 精读](./DSO_PAMI2018.md) §4.4 的 FEJ 动机完全同源。）
3. **假设 IMU-相机硬件同步**（脚注 1 用 Kalibr 标定延迟；可选方案是把延迟加为状态）。
4. **噪声参数需标定准确**：$\Sigma_\eta$ 直接决定权重与协方差（§VIII-A 的 NEES 一致性以此为前提）。
5. **纯里程计**：无回环、无全局一致机制（VIO 定位模块，非完整 SLAM）。

**后续影响**：预积分成为视觉-惯性的标准组件——GTSAM 4.0 随论文开源 `ImuFactor` 与无结构视觉因子；VINS-Mono、ORB-SLAM3 的 VI 线程、VI-DSLAM 一系全部沿用"预积分测量 + 协方差递推 + 偏置一阶修正"三件套；"on-manifold"的推导范式也成为流形状态估计的教材级范例（教程第 10 章 10.2 即按本文展开）。

## 7. 与本项目对照

代码：[projects/slam/preint/preint.py](../../../projects/slam/preint/preint.py)（SO(3) 流形预积分核心）+ [projects/slam/vins/vins.py](../../../projects/slam/vins/vins.py)（把预积分装进紧耦合因子图，对应教程 §10.3 的 (10.9)–(10.10)）。逐条对应与**刻意差距**：

- **测量与递推**：`Preintegration.__init__` 右乘递推 = 式 (35)–(37) 中定义的 $\Delta\tilde{R}/\Delta\tilde{\mathbf{v}}/\Delta\tilde{\mathbf{p}}$ 的增量形式（`delta_R`/`delta_v`/`delta_p`，偏置常值 = 式 (34)）；`predict()` = 式 (38) 零噪声反解（重力项出现在这里、不在递推里——docstring 专门强调）；`so3_right_jacobian` = 式 (8) 的 $J_r$ 闭式（经 $J_r(\boldsymbol{\phi}) = J_l(-\boldsymbol{\phi})$ 由 `core.lie.so3_left_jacobian` 得到）。
- **协方差**：`_propagate_covariance` 显式组装 A/B 分块（§4.7）并按 $\Sigma \leftarrow A\Sigma A^\top + B\Sigma_\eta B^\top$ 递推（式 (63)），强制对称防舍入漂移；`ImuParams` 的噪声密度 → $\sigma^2\Delta t$ 归一对应 §V 的采样率换算。
- **偏置修正**：`_propagate_bias_jacobians` = 附录 IX-B 的增量式（9×6 `j_bias`）；`correct()` = 式 (44)。
- **因子包装**（vins）：`ImuFactor.residual` = 式 (45) + 式 (48)（15 维：旋转 Log 残差 + 速度/位置残差 + 偏置随机游走，前 9 维经 $\Sigma_{ij}$ Cholesky 白化）；`ViBundle.solve` = 流形 LM（Marquardt 对角缩放、Huber IRLS 只加在视觉行）。
- **刻意差距**（docstring 有声明）： 路标用全局 XYZ（初始位姿 DLT 三角化）而非论文式 (49)–(55) 的无结构因子； 无边缘化先验（固定 4 关键帧窗口）； 相机-IMU 外参取单位阵、重力矢量假设已知； **IMU 因子雅可比用数值雅可比**（`ImuFactor.jacobian`，中心差分穿过求解器实际使用的回缩 `_retract_state`）——论文附录 IX-C 给出了解析雅可比（式 (70)–(81)），取舍理由（docstring 原文要点）：残差很廉价（预积分已缓存）且窗口只有 4 关键帧，每因子 60 次中心差分毫秒级；免手推分块矩阵的符号记账风险；求导穿过与更新**同一个**回缩，保证是外层 LM 的正确流形雅可比。注意分工：`preint.py` 内部的偏置雅可比是**解析增量式**（附录 IX-B），数值化只发生在 `vins.py` 的残差雅可比层（附录 IX-C 对应物）。
- **健康值**（实测，`python3 tests/test_preint.py` 6/6、`python3 tests/test_vins.py`）：
  - 零噪声一致性：$\Delta\tilde{R}/\Delta\tilde{\mathbf{v}}/\Delta\tilde{\mathbf{p}}$ 与式 (35)–(37) 的双循环直接求值逐项一致（0 / 机器精度级）；测量模型闭合（式 (38) ↔ 世界系离散积分）误差 $\le 10^{-10}$。
  - **MC 协方差**：200 次运行的经验 std 与传播 $\Sigma$ 对角线最大相对偏差 **0.113**（门限 0.30；MC 标准误 $\approx 1/\sqrt{2n} \approx 5\%$ 叠加一阶线性化系统偏差）——式 (59)–(63) 的活体验证。
  - **偏置修正 $O(\delta\mathbf{b}^2)$**：$\delta\mathbf{b}$（$\|\delta\mathbf{b}^g\| = 0.03\,\mathrm{rad/s}$、$\|\delta\mathbf{b}^a\| = 0.10\,\mathrm{m/s^2}$）处残差 R 2.44e-5 rad / v 3.51e-4 m/s / p 1.10e-4 m；$\delta\mathbf{b}$ 减半后 6.10e-6 / 8.77e-5 / 2.75e-5（≈ 1/4，二次收敛）——式 (44) 的实测。
  - **偏置雅可比**：解析增量式对中心有限差分最大相对列误差 **5.35e-11**——附录 IX-B 递推正确性。
  - **尺度恢复**（vins，`[vi-bundle]`）：故意错误尺度初值（比值 0.5）下 IMU 因子把尺度钉回米制——**ATE RMSE 2.36 cm**、逐对尺度比 0.94–1.04；`imu_weight = 0` 的纯视觉对照（`[pure-visual]`）尺度塌缩到 **0.38–0.42** 且不回升——教程 §10.3 第 3 步尺度可观性论证的活体样本（口径见 [METRICS.md](../../../projects/slam/METRICS.md) §2/§4）。

## 配套阅读

- 本文 PDF：[On-Manifold Preintegration（Forster et al., arXiv:1512.02363 / T-RO 2017）](../../../papers/slam/classics/arXiv-1512.02363_IMU-Preintegration.pdf)（本精读所有式号以此 PDF 为准：测量 (27)–(28)、预积分 (33)–(38)、协方差 (59)–(63)、偏置 (44) 与附录 IX-B、解析雅可比附录 IX-C (70)–(81)）
- 姊妹精读：[ORB-SLAM3（T-RO 2021）](./ORB-SLAM3_TRO2021.md)（VI 线程直接内嵌本文的预积分因子与 IMU 初始化）｜[DSO（PAMI 2018）](./DSO_PAMI2018.md)（滑窗 + 边缘化 + FEJ 的纯视觉同构问题）｜[LOAM（RSS 2014）](./LOAM_RSS2014.md)（"高频测量压缩为低频约束"的激光端对偶：双速率 + 运动补偿）
- 教程：[第 10 章](../10_建图与系统实战.md)（§10.2 完整推导 (10.1)–(10.8)、§10.3 因子图包装 (10.9)–(10.11)）｜[第 08 章](../08_后端-ii图优化与-ba.md)（(8.5) 正规方程、(8.9)–(8.10) 流形更新——式 (20)–(21) 的教程版）｜[第 02 章](../02_三维刚体运动旋转与位姿.md)（(2.17) 指数映射、(2.21) BCH——式 (7)/(9) 的教程版）
- 代码：[projects/slam/preint/](../../../projects/slam/preint/)（式号级 docstring 对照）｜[projects/slam/vins/](../../../projects/slam/vins/)（`ImuFactor`/`ReprojectionFactor`/`ViBundle`）｜[projects/slam/METRICS.md](../../../projects/slam/METRICS.md)（ATE/尺度比口径与健康值表）
- 论文原文引注：VINS-Mono（Qin et al., RA-L 2018，[PDF](../../../papers/slam/classics/arXiv-1708.03852_VINS-Mono.pdf)）——预积分 + 边缘化 + 回环的完整系统化；Forster et al. 的 ICRA 2016 会议版为本文 [6]

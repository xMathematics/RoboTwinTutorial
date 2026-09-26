# 论文精读｜Crocoddyl：多接触最优控制框架（ICRA 2020）

> **PDF**：[papers/control_planning/frontier/arXiv-1909.04947_Crocoddyl.pdf](../../../papers/control_planning/frontier/arXiv-1909.04947_Crocoddyl.pdf)（arXiv:1909.04947v2，共 7 页，式号以此为准） ｜ **教程**：[第 04 章](../04_轨迹优化与最优控制.md) §04.2（iLQR/DDP） ｜ **代码**：[loco-3d/crocoddyl](https://github.com/loco-3d/crocoddyl)（C++/Python 开源库，构建于 Pinocchio 之上——论文的稀疏解析导数即该库的核心资产）

## 1. 论文信息与一句话贡献

- **题目**：Crocoddyl: An Efficient and Versatile Framework for Multi-Contact Optimal Control
- **作者**：Carlos Mastalli、Rohan Budhiraja、Wolfgang Merkt、Guilhem Saurel、Bilal Hammoud、Maximilien Naveau、Justin Carpentier、Ludovic Righetti、Sethu Vijayakumar、Nicolas Mansard（LAAS-CNRS、Edinburgh、Oxford、Tübingen、NYU 等）
- **发表**：IEEE ICRA 2020（arXiv:1909.04947v2，2020-03-11）
- **一句话贡献**：给出多接触全身最优控制（optimal control, OC）的完整工程化方案——把接触建模为**完整定常约束**（holonomic scleronomic constraint）从而解析地消去接触力、用 **RNEA 导数 + 运动学导数**拼出接触动力学的稀疏解析雅可比，并提出 **FDDP**（Feasibility-driven DDP）：反向/前向 pass 允许"不可行暖启动"，每轮只按比例 $(1-\alpha)$ 收缩相邻节点间隙——等价于"只含等式约束的直接多重打靶问题的 Newton 法"，但不引入额外决策变量——使四足/人形机器人的跳跃、前空翻等高动态机动的求解进入**毫秒量级**。

## 2. 问题与动机

**要解决什么**（§I）：腿式机器人的全身控制需要在每个控制周期求解多接触 OC——现有框架（作者自引 [1]–[3] 等全身控制器路线）把"中心动力学轨迹优化 + 接触计划 + 全身控制器"分层拼装，但**不能妥善处理飞行相的机器人朝向**（非完整效应使瞬时时不变控制失效）与**角动量调节**；基于 MPC 的近期工作 [15]–[17] 都要迭代构造并求解 LQR 逼近，且普遍依赖**数值或自动微分**——相比稀疏解析导数"不足够高效"[19]；更重要的是它们不显式处理带接触要素的腿式系统几何结构（SE(3) 浮基）。

**两个具体的数值痛点**（§II-B 开头）：要在实时内解 OC，必须同时压住 (a) 高维搜索空间，(b) 动力学的**不稳定性、不连续性与非凸性**（下层优化）。论文的路线：把下层（接触力消去）**解析化**，把上层求解器（DDP）的**全局化**改造好。

**DDP 自身的短板**（§III 引言）：经典 DDP 只能从"动力学一致的轨迹"出发——暖启动必须给控制序列 $\mathbf{U}_0$ 并先做一次前向 pass 生成相容状态（论文脚注 7：给状态初值容易、给超出准静态的对应控制初值难）。"不可行暖启动"（infeasible warm-start，脚注 1：与系统动力学不一致的状态-控制轨迹）在实践中无处不在——这正是 FDDP 的切入点。

## 3. 方法总览

```
┌─ 上层：多接触 OCP（式 1，双层优化 bilevel）────────────────────┐
│   min l_N(x_N) + Σ∫ l(x,u)dt   s.t.  下层接触力学 + 可行集     │
└─→ 下层解析消元（§II-B）：
    接触 = 完整约束 φ(q)=0 → KKT 线性系统（式 2）
    → 接触力 g(x,τ) 与加速度 y(x,τ) 的闭式（式 4，Cholesky rollout）
    → 解析雅可比（式 5，RNEA 导数 + 运动学导数，LDU 逐块求逆）
    → 冲量动力学（式 6，碰撞切换相）
┌─ 上层求解器：FDDP（§III）───────────────────────────────────┐
│   Bellman 值函数 LQ 化（式 7–8）→ 反向 Riccati 给 k_k, K_k     │
│   多重打靶间隙 f̃（式 9）→ SQP 视角（式 10–11）→ KKT 逐区间结构 │
│   （式 12–14）→ 含间隙的非线性 rollout（式 15–16，α-步收缩）    │
│   → 偏转值函数的修正 Riccati（式 17）→ Goldstein 接受准则       │
│   （式 18–20）                                                │
└───────────────────────────────────────────────────────────────┘
```

三条主线：**接触动力学的解析导数**（计算导数是 OC 求解器的主要开销，§II 引言）、**FDDP 的可行性驱动迭代**（间隙从开到合，全局化优于经典 DDP）、**多线程实现**（只并行化导数计算）。求解器对浮动基系统用李群（SE(3)）数值例程正确处理刚体几何——前空翻机动因此可算。

## 4. 关键公式推导（按论文原文式号）

**符号表**：

| 符号 | 含义 |
|------|------|
| $\mathbf{q}, \mathbf{v}$ | 构型点（浮基为 SE(3) 元素）与切向量；状态 $\mathbf{x} = (\mathbf{q},\mathbf{v}) \in \mathcal{X}$ 用 $n_x$ 元组描述，$\dot{\mathbf{x}} \in T_{\mathbf{x}}\mathcal{X}$ 用 $n_{dx}$ 元组描述 |
| $\mathbf{u} = (\boldsymbol{\tau}, \boldsymbol{\lambda}) \in \mathbb{R}^{n_u}$ | 控制 = 力矩指令 $\boldsymbol{\tau}$ + 接触力 $\boldsymbol{\lambda}$ |
| $\mathcal{X}, \mathcal{U}$ | 状态/控制可行集（关节限、摩擦锥、任务与碰撞约束） |
| $\mathbf{M}$ | 关节空间惯量矩阵（joint-space inertia，正定） |
| $\boldsymbol{\tau}_b$ | 广义力偏置（重力、科氏等非惯性项折入后的等效力矩——由 RNEA 输出） |
| $\boldsymbol{\phi}(\mathbf{q}) = \mathbf{0}$，$\mathbf{J}_c = \partial\boldsymbol{\phi}/\partial\mathbf{q}$ | 接触帧位置的完整定常约束与接触雅可比（局部系表示） |
| $\mathbf{a}_0 \in \mathbb{R}^{n_f}$ | 约束空间的期望加速度（Baumgarte 稳定化目标，式 (3)） |
| $\mathbf{M}_c = \mathbf{J}_c\mathbf{M}^{-1}\mathbf{J}_c^\top$ | 操作空间惯量矩阵（operational space inertia） |
| $\mathbf{y}(\mathbf{x},\boldsymbol{\tau}),\ \mathbf{g}(\mathbf{x},\boldsymbol{\tau})$ | 系统加速度与（负号约定下的）接触力，式 (2) 定义 |
| $V_k,\ l_k,\ f_k$ | 值函数、阶段代价与动力学的 LQ 近似（§III-A） |
| $Q_x, Q_u, Q_{xx}, Q_{xu}, Q_{uu}$ | 控制-Hamilton 量的 LQ 近似四块半 |
| $\mathbf{k}_k, \mathbf{K}_k$ | 前馈项与反馈增益 |
| $\bar{\mathbf{f}}_{k+1}$（式 (9) 记 $\tilde{\mathbf{f}}$） | 多重打靶**间隙**（gap/defect）：$f(\mathbf{x}_k,\mathbf{u}_k) - \mathbf{x}_{k+1}$ |
| $\alpha \in (0,1]$，$\Delta J(\alpha)$，$b_1,b_2$ | 步长、期望代价下降模型、Goldstein 参数（$b_1=0.1, b_2=2$） |

### 4.1 多接触 OCP 的双层形式（式 1）

$$
\Big\{ \bar{\mathbf{x}}_0^*, \cdots, \bar{\mathbf{x}}_N^*;\ \mathbf{u}_0^*, \cdots, \mathbf{u}_{N-1}^* \Big\} = \arg\min_{\bar{\mathbf{X}},\mathbf{U}}\ l_N(\mathbf{x}_N) + \sum_{k=0}^{N-1} \int_{t_k}^{t_k+\Delta t_k} l(\mathbf{x},\mathbf{u})\,\mathrm{d}t
$$
$$
\text{s.t.}\quad \dot{\mathbf{v}}, \boldsymbol{\lambda} = \arg\min_{\dot{\mathbf{v}},\boldsymbol{\lambda}} \big\lVert \dot{\mathbf{v}} - \dot{\mathbf{v}}_{free} \big\rVert_{\mathbf{M}}, \qquad \mathbf{x} \in \mathcal{X},\ \mathbf{u} \in \mathcal{U} \tag{1}
$$

- 上层与教程 (4.1) 同构（连续积分 vs 教程的连续泛函写法，离散后即教程 (4.5)）；$\dot{\mathbf{v}}_{free}$ 是**无约束**广义坐标加速度。
- **下层是高斯最小约束原理**（Gauss' principle of least constraint [22]）：真实加速度与接触力是"在 $\mathbf{M}$-加权范数下离自由加速度最近"的那个——物理约束被写成一个极小问题（依据：分析力学的 Gauss 原理）。可行集 $\mathcal{X},\mathcal{U}$ 可归属下层（关节限、摩擦锥）或上层（任务、碰撞）。

### 4.2 接触作为完整约束与闭式消元（式 2–4）

把接触写成帧位置的完整定常约束 $\boldsymbol{\phi}(\mathbf{q}) = \mathbf{0}$，加速度级约束方程由其对时间两次求导得到，并以 $\mathbf{a}_0$ 作为稳定化后的右端。约束力通过 $\mathbf{J}_c^\top\boldsymbol{\lambda}$ 进入动力学——联立得块线性系统（论文式 (2)）：

$$
\begin{bmatrix} \dot{\mathbf{v}} \\ -\boldsymbol{\lambda} \end{bmatrix} = \begin{bmatrix} \mathbf{M} & \mathbf{J}_c^\top \\ \mathbf{J}_c & \mathbf{0} \end{bmatrix}^{-1} \begin{bmatrix} \boldsymbol{\tau}_b \\ -\mathbf{a}_0 \end{bmatrix} = \begin{bmatrix} \mathbf{y}(\mathbf{x},\boldsymbol{\tau}) \\ -\mathbf{g}(\mathbf{x},\boldsymbol{\tau}) \end{bmatrix} \tag{2}
$$

> **符号备注（原文两式的符号出入）**：按式 (2) 的印刷符号展开第二行得 $\mathbf{J}_c\dot{\mathbf{v}} = -\mathbf{a}_0$；而论文随后给出的式 (4) 反代自洽要求 $\mathbf{J}_c\dot{\mathbf{v}} = +\mathbf{a}_0$（与式 (3) 的 Baumgarte 意图一致）。本精读推导以式 (4) 的自洽符号为准，式 (2) 按原文抄录。

**稳定化右端（式 3）**：数值积分会让 $\boldsymbol{\phi}(\mathbf{q}) = \mathbf{0}$ 漂移，取 PD 型修正（依据：Baumgarte 稳定化思想 [23]，把"位置误差 + 速度误差"反馈进加速度目标）：

$$
\mathbf{a}_0 = \mathbf{a}_{\lambda(c)} - \alpha\ \prescript{o}{}{M}_{\lambda(c)}^{ref} \ominus\ \prescript{o}{}{M}_{\lambda(c)} - \beta\, \mathbf{v}_{\lambda(c)} \tag{3}
$$

其中 $\mathbf{v}_{\lambda(c)}, \mathbf{a}_{\lambda(c)}$ 是接触 $\lambda(c)$ 父连杆的空间速度/加速度，$\alpha,\beta$ 是稳定化增益，$\ominus$ 是参考接触位置与当前位置之间的 **SE(3) 逆复合**（相对位形的对数映射，误差的流形正确表示）[24]。

**闭式解（式 4）——推导无跳步**：把式 (2) 的等价块系统写成

$$
\mathbf{M}\dot{\mathbf{v}} = \boldsymbol{\tau}_b + \mathbf{J}_c^\top\boldsymbol{\lambda}, \qquad \mathbf{J}_c\dot{\mathbf{v}} = \mathbf{a}_0
$$

（依据：第一行是含接触力的广义力平衡，第二行是加速度级约束；$\mathbf{g} \equiv \boldsymbol{\lambda}$）。第一步，第一行左乘 $\mathbf{M}^{-1}$（依据：$\mathbf{M} \succ 0$ 可逆）：

$$
\dot{\mathbf{v}} = \mathbf{M}^{-1}\big(\boldsymbol{\tau}_b + \mathbf{J}_c^\top\boldsymbol{\lambda}\big)
$$

第二步，代入第二行（依据：直接代入）：

$$
\mathbf{J}_c\mathbf{M}^{-1}\boldsymbol{\tau}_b + \underbrace{\mathbf{J}_c\mathbf{M}^{-1}\mathbf{J}_c^\top}_{=:\ \mathbf{M}_c}\boldsymbol{\lambda} = \mathbf{a}_0
$$

第三步，当 $\mathbf{J}_c$ 满秩时 $\mathbf{M}_c$ 可逆（依据：$\mathbf{M}^{-1} \succ 0$ 且 $\mathbf{J}_c$ 列满秩 ⇒ $\mathbf{M}_c \succ 0$；这正是论文"unique solution if $\mathbf{J}_c$ is full-rank"的出处），解出接触力并回代：

$$
\mathbf{g}(\mathbf{x},\boldsymbol{\tau}) = \mathbf{M}_c^{-1}\big(\mathbf{a}_0 - \mathbf{J}_c\mathbf{M}^{-1}\boldsymbol{\tau}_b\big), \qquad \mathbf{y}(\mathbf{x},\boldsymbol{\tau}) = \mathbf{M}^{-1}\big(\boldsymbol{\tau}_b + \mathbf{J}_c^\top \mathbf{g}(\mathbf{x},\boldsymbol{\tau})\big) \tag{4}
$$

（自洽检验：$\mathbf{J}_c\mathbf{y} = \mathbf{J}_c\mathbf{M}^{-1}\boldsymbol{\tau}_b + \mathbf{M}_c\mathbf{M}_c^{-1}(\mathbf{a}_0 - \mathbf{J}_c\mathbf{M}^{-1}\boldsymbol{\tau}_b) = \mathbf{a}_0$，约束恢复 ✓。）工程意义：rollout 时**无需求逆整个 KKT 矩阵**——$\mathbf{M}^{-1}$ 与 $\mathbf{M}_c^{-1} = \mathbf{J}_c\mathbf{M}^{-1}\mathbf{J}_c^\top$ 各用一次 Cholesky 分解即可（依据：SPD 矩阵的 Cholesky 可解性）。式 (2) 忽略摩擦锥与关节限，故动力学是纯等式约束，可用**无约束 DDP** [25]；不等式可经惩罚、活动集 [26] 或增广拉格朗日 [27] 加回。

### 4.3 解析导数（式 5）与冲量动力学（式 6）

对式 (2) 做隐函数求导（依据：矩阵方程全微分，$\mathbf{A} = \begin{bmatrix} \mathbf{M} & \mathbf{J}_c^\top \\ \mathbf{J}_c & \mathbf{0} \end{bmatrix}$ 随 $\mathbf{x}$ 变化，矩阵变差项按链式法则折入右端全导数）：

$$
\begin{bmatrix} \delta\dot{\mathbf{v}} \\ -\delta\boldsymbol{\lambda} \end{bmatrix} = -\begin{bmatrix} \mathbf{M} & \mathbf{J}_c^\top \\ \mathbf{J}_c & \mathbf{0} \end{bmatrix}^{-1} \begin{pmatrix} \Big[ \frac{\partial\boldsymbol{\tau}_b}{\partial\mathbf{x}}\,\delta\mathbf{x} + \frac{\partial\boldsymbol{\tau}_b}{\partial\mathbf{u}}\,\delta\mathbf{u} \Big] \\ \Big[ \frac{\partial\mathbf{a}_0}{\partial\mathbf{x}}\,\delta\mathbf{x} + \frac{\partial\mathbf{a}_0}{\partial\mathbf{u}}\,\delta\mathbf{u} \Big] \end{pmatrix} = \begin{bmatrix} \mathbf{y}_{\mathbf{x}} & \mathbf{y}_{\mathbf{u}} \\ -\mathbf{g}_{\mathbf{x}} & -\mathbf{g}_{\mathbf{u}} \end{bmatrix} \begin{bmatrix} \delta\mathbf{x} \\ \delta\mathbf{u} \end{bmatrix} \tag{5}
$$

关键在于这些偏导**全都有解析来源**：$\partial\boldsymbol{\tau}_b/\partial\mathbf{x}, \partial\boldsymbol{\tau}_b/\partial\mathbf{u}$ 是 **RNEA**（Recursive Newton-Euler Algorithm）的导数——$\boldsymbol{\tau}_b$ 本就是 RNEA 的输出；$\partial\mathbf{a}_0/\partial\mathbf{x}, \partial\mathbf{a}_0/\partial\mathbf{u}$ 是帧加速度的**运动学导数**（含式 (3) 中 SE(3) 对数映射的解析导数）[19],[29]。块矩阵用 **LDU 分解**逐块求逆（依据：$\mathbf{A}$ 的 (2,2) 块为 $\mathbf{0}$，LDU 避免整体求逆）。这避免了有限差分需要的 $D$ 次额外模拟与数值噪声——教程 §04.2 ④ 所说"解析导数消掉二阶导数实现痛点"的落地处。

**冲量动力学（式 6）**：接触切换相（非接触→接触 [30]）速度不连续。对冲击瞬间积分动量方程并施加恢复系数定义（依据：冲量-动量定理 $\mathbf{M}(\mathbf{v}^+ - \mathbf{v}^-) = \mathbf{J}_c^\top\boldsymbol{\Lambda}$；恢复系数 $e\in[0,1]$ 规定约束速度反向比例 $\mathbf{J}_c\mathbf{v}^+ = -e\,\mathbf{J}_c\mathbf{v}^-$）：

$$
\begin{bmatrix} \mathbf{v}^+ \\ -\boldsymbol{\Lambda} \end{bmatrix} = \begin{bmatrix} \mathbf{M} & \mathbf{J}_c^\top \\ \mathbf{J}_c & \mathbf{0} \end{bmatrix}^{-1} \begin{bmatrix} \mathbf{M}\mathbf{v}^- \\ -e\,\mathbf{J}_c\mathbf{v}^- \end{bmatrix} \tag{6}
$$

$e = 0$ 为完全非弹性碰撞（接触速度归零）。求逆与导数同样用 Cholesky。这样**接触相与冲量相都是同一个递推里的阶段模型**——教程 §04.2 ④"接触/冲量作为阶段模型进同一递推"。

### 4.4 DDP 值函数与增益（式 7–8）

DDP 在 $(\delta\mathbf{x}_k, \delta\mathbf{u}_k)$ 附近局部逼近最优流（值函数），Bellman 最优性把问题拆成子问题（依据：动态规划原理）：

$$
V_k(\delta\mathbf{x}_k) = \min_{\delta\mathbf{u}_k}\ l_k(\delta\mathbf{x}_k, \delta\mathbf{u}_k) + V_{k+1}\big(f_k(\delta\mathbf{x}_k, \delta\mathbf{u}_k)\big) \tag{7}
$$

其中 $l_k(\cdot), f_k(\cdot)$ 已是代价与动力学的 **LQ 近似**。极小化写成 KKT 矩阵形式：

$$
\delta\mathbf{u}_k^*(\delta\mathbf{x}_k) = \arg\min_{\delta\mathbf{u}_k} \frac{1}{2} \begin{bmatrix} 1 \\ \delta\mathbf{x}_k \\ \delta\mathbf{u}_k \end{bmatrix}^{\!\top} \underbrace{\begin{bmatrix} 0 & Q_{\mathbf{x}_k}^\top & Q_{\mathbf{u}_k}^\top \\ Q_{\mathbf{x}_k} & Q_{\mathbf{x}\mathbf{x}_k} & Q_{\mathbf{x}\mathbf{u}_k} \\ Q_{\mathbf{u}_k} & Q_{\mathbf{x}\mathbf{u}_k}^\top & Q_{\mathbf{u}\mathbf{u}_k} \end{bmatrix}}_{\mathbf{H}\,(\text{论文脚注 3：这就是 KKT 矩阵})} \begin{bmatrix} 1 \\ \delta\mathbf{x}_k \\ \delta\mathbf{u}_k \end{bmatrix} \tag{8}
$$

$Q$ 诸块是控制-Hamilton 量 $\mathsf{H}(\cdot)$ 的 LQ 近似。对 $\delta\mathbf{u}$ 求驻点（依据：二次型梯度公式 $\partial(\mathbf{g}^\top\mathbf{z})/\partial\mathbf{z} = \mathbf{g}$、$\partial(\mathbf{z}^\top\mathbf{H}\mathbf{z})/\partial\mathbf{z} = 2\mathbf{H}\mathbf{z}$；需 $Q_{\mathbf{u}\mathbf{u}} \succ 0$）：

$$
Q_{\mathbf{u}} + Q_{\mathbf{x}\mathbf{u}}\,\delta\mathbf{x}_k + Q_{\mathbf{u}\mathbf{u}}\,\delta\mathbf{u}_k = 0 \ \Longrightarrow\ \delta\mathbf{u}_k^* = \mathbf{k}_k + \mathbf{K}_k\,\delta\mathbf{x}_k,\quad \mathbf{k}_k = -Q_{\mathbf{u}\mathbf{u}}^{-1}Q_{\mathbf{u}},\ \mathbf{K}_k = -Q_{\mathbf{u}\mathbf{u}}^{-1}Q_{\mathbf{x}\mathbf{u}}
$$

（依据：式 (8) 逐次求解即 **Riccati 递推**，论文原文表述；与教程增益式 **(4.11)** 完全一致。）把极小值代回得值函数的梯度/Hessian 递推——即教程 **(4.12)**。注意式 (17) 中 $Q_{\mathbf{x}\mathbf{x}} = l_{\mathbf{x}\mathbf{x}} + \mathbf{f}_{\mathbf{x}}^\top V_{\mathbf{x}\mathbf{x}}\mathbf{f}_{\mathbf{x}}$ **不含** $\sum_j V'_{x,j} f_{\mathbf{x}\mathbf{x},j}$ 型二阶动力学项——论文实践的是 iLQR 型一阶展开（教程 **(4.9)**），教程 **(4.10)** 的 DDP 二阶项是理论形态。

### 4.5 多重打靶间隙与 FDDP（式 9–16）

**间隙定义（式 9）**：多重打靶把中间状态 $\mathbf{x}_k$（shooting nodes）设为额外决策变量、以等式约束闭合间隙（文献中亦称 *defects*）：

$$
\tilde{\mathbf{f}}_{k+1} = f(\mathbf{x}_k, \mathbf{u}_k) - \mathbf{x}_{k+1} \tag{9}
$$

**SQP 视角（式 10–11）**：多重打靶逼近一步 SQP，单次 QP 迭代为

$$
\min_{\delta\mathbf{X},\delta\mathbf{U}}\ l_N(\delta\mathbf{x}_N) + \sum_{k=0}^{N-1} l_k(\delta\mathbf{x}_k, \delta\mathbf{u}_k) \quad \text{s.t.}\quad \delta\mathbf{x}_0 = \tilde{\mathbf{x}}_0,\ \ \delta\mathbf{x}_{k+1} = \mathbf{f}_{\mathbf{x}k}\,\delta\mathbf{x}_k + \mathbf{f}_{\mathbf{u}k}\,\delta\mathbf{u}_k + \bar{\mathbf{f}}_{k+1} \tag{10}
$$

步长更新 $[\mathbf{X}_{i+1}; \mathbf{U}_{i+1}] = [\mathbf{X}_i; \mathbf{U}_i] + \alpha[\delta\mathbf{X}_i; \delta\mathbf{U}_i]$（式 11）。**关键观察（式 12–14）**：对单个打靶区间 $k$ 写 KKT 的一阶必要条件（FONC），$\mathbf{H}_k = \begin{bmatrix} l_{\mathbf{x}\mathbf{x}_k} & l_{\mathbf{x}\mathbf{u}_k} \\ l_{\mathbf{x}\mathbf{u}_k}^\top & l_{\mathbf{u}\mathbf{u}_k} \end{bmatrix}$、$\nabla\Phi_k = [l_{\mathbf{x}_k}; l_{\mathbf{u}_k}]$、$\delta\mathbf{w}_k = [\delta\mathbf{x}_k; \delta\mathbf{u}_k]$：

$$
\mathbf{H}_k\,\delta\mathbf{w}_k + \begin{bmatrix} \nabla\mathbf{g}_k^- & \nabla\mathbf{g}_k^+ \end{bmatrix} \begin{bmatrix} \boldsymbol{\lambda}_k \\ \boldsymbol{\lambda}_{k+1} \end{bmatrix} = -\nabla\Phi_k, \qquad \begin{bmatrix} \mathbf{I} & \\ -\mathbf{f}_{\mathbf{x}_k} & -\mathbf{f}_{\mathbf{u}_k} \end{bmatrix} \delta\mathbf{w}_k = \begin{bmatrix} \bar{\mathbf{f}}_k \\ \bar{\mathbf{f}}_{k+1} \end{bmatrix} \tag{12, 13}
$$

（式 12 是对偶可行性、式 13 是原始可行性；$\nabla\mathbf{g}_k^-$ 选出"上一区间动力学终止于节点 $k$"的 $\mathbf{I}$ 行，$\nabla\mathbf{g}_k^+ = [-\mathbf{f}_{\mathbf{x}_k}^\top; -\mathbf{f}_{\mathbf{u}_k}^\top]$。）联立解鞍点系统得搜索方向：

$$
\begin{bmatrix} \delta\mathbf{w}_k \\ \delta\boldsymbol{\lambda}_k \\ \delta\boldsymbol{\lambda}_{k+1} \end{bmatrix} = \begin{bmatrix} \mathbf{H}_k & \nabla\mathbf{g}_k^- & \nabla\mathbf{g}_k^+ \\ \nabla\mathbf{g}_k^{-\top} & \mathbf{I} & \\ \nabla\mathbf{g}_k^{+\top} & & \mathbf{I} \end{bmatrix}^{-1} \begin{bmatrix} \nabla\Phi_k \\ \bar{\mathbf{f}}_k \\ \bar{\mathbf{f}}_{k+1} \end{bmatrix} \tag{14}
$$

**α-步只把节点 $k$ 的间隙闭合 $(1-\alpha)\bar{\mathbf{f}}_k$，仅全步 $\alpha=1$ 完全闭合**——这是全文的枢纽事实。

**非线性 rollout 维持收缩率（式 15）**：对 rollout 后的间隙做一阶 Taylor 展开（依据：$f(\mathbf{x}_k + \alpha\delta\mathbf{x}_k, \mathbf{u}_k + \alpha\delta\mathbf{u}_k) - (\mathbf{x}_{k+1} + \alpha\delta\mathbf{x}_{k+1}) \approx \bar{\mathbf{f}}_{k+1} + \alpha(\mathbf{f}_{\mathbf{x}}\delta\mathbf{x}_k + \mathbf{f}_{\mathbf{u}}\delta\mathbf{u}_k - \delta\mathbf{x}_{k+1})$，再用式 (10) 的约束 $\delta\mathbf{x}_{k+1} - \mathbf{f}_{\mathbf{x}}\delta\mathbf{x}_k - \mathbf{f}_{\mathbf{u}}\delta\mathbf{u}_k = \bar{\mathbf{f}}_{k+1}$）：

$$
\bar{\mathbf{f}}^{\,i+1}_{k+1} = \bar{\mathbf{f}}^{\,i}_{k+1} - \alpha\big(\delta\mathbf{x}_{k+1} - \mathbf{f}_{\mathbf{x}k}\,\delta\mathbf{x}_k - \mathbf{f}_{\mathbf{u}k}\,\delta\mathbf{u}_k\big) = (1-\alpha)\big(f(\mathbf{x}_k, \mathbf{u}_k) - \mathbf{x}_{k+1}\big) \tag{15}
$$

于是 FDDP 的前向 pass（式 16）：起点偏移未闭合的初始间隙；控制 = 标称 + α 缩放前馈 + 对"rollout 偏离节点"的反馈；rollout 状态扣掉未闭合间隙（依据：式 (15) 的收缩律在递推中的实现）：

$$
\hat{\mathbf{x}}_0 = \tilde{\mathbf{x}}_0 - (1-\alpha)\bar{\mathbf{f}}_0, \qquad \hat{\mathbf{u}}_k = \mathbf{u}_k + \alpha\mathbf{k}_k + \mathbf{K}_k(\hat{\mathbf{x}}_k - \mathbf{x}_k), \qquad \hat{\mathbf{x}}_{k+1} = f_k(\hat{\mathbf{x}}_k, \hat{\mathbf{u}}_k) - (1-\alpha)\bar{\mathbf{f}}_k \tag{16}
$$

$\alpha = 1$ 时所有间隙闭合，FDDP 前向 pass 与经典 DDP **完全一致**。直觉：不可行暖启动时 $\alpha$ 从小值起步——轨迹先"带着间隙"演化（可行性驱动的全局化），间隙渐次闭合后退化为 DDP 的超线性收敛。

### 4.6 修正 Riccati 与 Goldstein 接受准则（式 17–20）

间隙使下一节点的值函数要在"偏转了 $\bar{\mathbf{f}}_{k+1}$ 的状态"处取值。LQ 值函数是二次型，代入 $\delta\mathbf{x} + \bar{\mathbf{f}}$ 展开（依据：$V(\delta\mathbf{x} + \bar{\mathbf{f}}) = V + V_x^\top\bar{\mathbf{f}} + \frac{1}{2}\bar{\mathbf{f}}^\top V_{xx}\bar{\mathbf{f}}$，梯度平移、Hessian 不变）：

$$
Q_{\mathbf{x}_k} = l_{\mathbf{x}_k} + \mathbf{f}_{\mathbf{x}_k}^\top V_{\mathbf{x}_{k+1}}^{+},\quad Q_{\mathbf{u}_k} = l_{\mathbf{u}_k} + \mathbf{f}_{\mathbf{u}_k}^\top V_{\mathbf{x}_{k+1}}^{+},\quad Q_{\mathbf{x}\mathbf{x}_k} = l_{\mathbf{x}\mathbf{x}_k} + \mathbf{f}_{\mathbf{x}_k}^\top V_{\mathbf{x}\mathbf{x}_{k+1}}\mathbf{f}_{\mathbf{x}_k},\quad Q_{\mathbf{x}\mathbf{u}_k} = l_{\mathbf{x}\mathbf{u}_k} + \mathbf{f}_{\mathbf{x}_k}^\top V_{\mathbf{x}\mathbf{x}_{k+1}}\mathbf{f}_{\mathbf{u}_k},\quad Q_{\mathbf{u}\mathbf{u}_k} = l_{\mathbf{u}\mathbf{u}_k} + \mathbf{f}_{\mathbf{u}_k}^\top V_{\mathbf{x}\mathbf{x}_{k+1}}\mathbf{f}_{\mathbf{u}_k} \tag{17}
$$

其中 $V_{\mathbf{x}_{k+1}}^{+} = V_{\mathbf{x}_{k+1}} + V_{\mathbf{x}\mathbf{x}_{k+1}}\bar{\mathbf{f}}_{k+1}$（偏转后的值函数 Jacobian；Hessian 不变）。**期望代价下降**（式 18–19）：

$$
\Delta J(\alpha) = \Delta_1\alpha + \tfrac{1}{2}\Delta_2\alpha^2,\qquad \Delta_1 = \sum_{k=0}^{N-1} \Big[ \mathbf{k}_k^\top Q_{\mathbf{u}_k} + \bar{\mathbf{f}}_k^\top(V_{\mathbf{x}_k} - V_{\mathbf{x}\mathbf{x}_k}\mathbf{x}_k) \Big],\quad \Delta_2 = \sum_{k=0}^{N-1} \Big[ \mathbf{k}_k^\top Q_{\mathbf{u}\mathbf{u}_k}\mathbf{k}_k + \bar{\mathbf{f}}_k^\top(2V_{\mathbf{x}\mathbf{x}_k}\mathbf{x}_k - V_{\mathbf{x}\mathbf{x}_k}\bar{\mathbf{f}}_k) \Big] \tag{18, 19}
$$

（自洽检验：所有间隙为零时 $\bar{\mathbf{f}} = 0$，$\Delta_1 = \sum \mathbf{k}^\top Q_u$、$\Delta_2 = \sum \mathbf{k}^\top Q_{uu}\mathbf{k}$，代入 $\mathbf{k} = -Q_{uu}^{-1}Q_u$ 得 $\Delta J(1) = -\frac{1}{2}\sum Q_u^\top Q_{uu}^{-1} Q_u \le 0$——正是教程 (4.12) 之后的单步代价改善式；论文亦注明此时与 [32]（Tassa 2012）一致。）**接受准则用 Goldstein 而非经典 DDP 的 Armijo**（式 20）——因为不可行迭代中 $\Delta J$ 可能是**上升**方向：

$$
l' - l \le \begin{cases} b_1\,\Delta J(\alpha) & \Delta J(\alpha) \le 0 \\ b_2\,\Delta J(\alpha) & \text{otherwise} \end{cases}, \qquad b_1 = 0.1,\ b_2 = 2 \tag{20}
$$

### 4.7 多线程实现

论文只并行化**导数计算**（rollout 串行），4–8 线程把单次迭代时间降到 1/2–1/4（§IV-C，Fig. 4/5）；计算频率随节点数线性缩放，jump-4f（60 节点，i9-9900K 8 线程）达 **859.6 Hz**——对高频重解式 MPC（教程第 05 章）而言即"每个控制周期解一次 OCP"的实时迭代实践。（用词核对：论文未使用"RTI"这一术语，本文按其实测口径表述。）

## 5. 实验与结果解读

- **步态生成（§IV-A）**：四足 walk/trot/pace/bound 与双足 walking，均由 FDDP 在**毫秒量级、约 12 次迭代**内算得；全部四足步态共用同一套权重与代价（CoM + 落足点跟踪 + 正则化；摆动腿分段线性参考、重罚落足点偏差；冲击相状态正则权重 $w_{hg} = 10$；切换相用冲量模型式 (6) 保证接触速度归零——相比直接罚接触速度收敛更快）。
- **高动态机动（§IV-B）**：跳跃（ANYmal/iCub）与前空翻（Fig. 3）毫秒量级、**12–36 次迭代**；用朴素且不可行的 $\mathbf{X}_0, \mathbf{U}_0$ 暖启动（状态经姿态序列线性插值、控制按准静态假定）；摩擦锥与力矩限被刻意忽略（可经二次惩罚加回）。
- **收敛与时间（§IV-C）**：所有动作在 **10–34 次迭代、总计算时间 < 0.5 s** 内收敛；$\Delta t = 1\times10^{-2}$ s（双足步行 $3\times10^{-2}$ s），节点 60–115，时域 0.6–3 s；最高 **859.6 Hz**（jump-4f，60 节点，i9-9900K）。Fig. 2：简单动作第 1 次迭代即闭合全部间隙；高动态动作 FDDP 让间隙**先开后合**（前 1–2 次迭代保持），闭合后呈超线性收敛——对比经典 DDP 的糟糕全局化（前几次迭代给出不可行 rollout）。walk-4f 支撑相短至 $\Delta t = 2$ ms，收敛率仍与 [33][34] 相当。
- **鲁棒性统计**：每个基准动作在 4 台 Intel PC（i7-6700K/i7-7700K/i9-9900K/i9-9900XE）上各跑 **50,000 次**、扫并行度—— Fig. 5 显示 4–8 线程收益最大。

## 6. 局限与后续影响

- **接触时序必须预定义**：步态与跳跃的接触序列是输入不是决策——"plan 怎么接触"仍留给上层（论文结论明言 feasibility models 处理 contact gain phases）。这正是后来"学习接触时序/步态 + OCP 跟踪"混合路线的切入口（教程第 08/09 章的 RL 混合谱系）。
- **不等式约束缺位**：摩擦锥、力矩限在本篇被忽略或仅以二次惩罚处理；结论给出后续路线——内点法 [26] 或增广拉格朗日 [27]（Crocoddyl 库后续版本的激活/接触约束求解器沿此展开）。
- **局部最优 + LQ 展开**：iLQR 型一阶动力学展开（§4.4），收敛是局部的；全局化靠 FDDP 的间隙调度而非全局保证。
- **正面影响**：FDDP ≡ "只含等式约束的多重打靶 KKT 上的 Newton 法、但无额外决策变量"（避免经典多重打靶对矩阵维数的立方复杂度）——这一等价性是论文的理论贡献；Crocoddyl 库成为腿式 MPC（含 OCS2 之外的常用开源选择）与后续可行性驱动求解器研究的基础设施。

## 7. 与本项目对照

| 论文概念 | 教程位置 | 对照说明 |
|---------|---------|---------|
| 多接触 OCP（式 1） | 第 04 章 (4.1)、(4.5) | 上层结构同构；论文多出双层下层的接触消元 |
| 值函数 LQ + 增益（式 7–8） | (4.6)、(4.8)–(4.9)、(4.11) | 式 (8) 的一阶条件即 (4.11) 的 $\mathbf{k}, \mathbf{K}$ |
| iLQR 型递推（式 17，无二阶项） | (4.9)（对照 (4.10) 的 DDP 二阶差异） | 论文取一阶展开 + 解析导数，回避动力学 Hessian |
| 前向 pass（式 16，$\alpha=1$） | (4.13) | (4.13) 是式 (16) 的无间隙特例 |
| Goldstein 接受（式 20） | (4.13) 后的 Armijo 型接受（定性） | 论文改进：间隙使 $\Delta J$ 可为正，两参数条件更稳 |
| 多线程 + 859.6 Hz（§4.7） | 第 05 章 MPC"滚动重解" | 毫秒级求解使 MPC 每 tick 重解 OCP 可行 |

**与 [RoboTwin 精读](../../robotwin/精读/README.md) 的仿真控制器关系**：RoboTwin 类双臂操作栈里，策略输出的是**关节位置增量**，底层是 MuJoCo position actuator 的隐式 PD 伺服（教程第 10 章 §10.2 (10.1)），并非力矩级 OCP——Crocoddyl 不在 RoboTwin 默认闭环里，而是第 10 章全栈对应表 **L3 轨迹优化** 行的工程选项（力矩级全身控制 / 接触丰富任务）。两者扮演同类"执行兜底"角色但层次不同：Crocoddyl 的 $\mathbf{K}_k$ 是**时变轨迹反馈**（式 8 反向 pass 副产品），RoboTwin 隐式 PD 是**定常伺服刚度**；若双臂任务升级到力矩级柔顺（第 06 章阻抗之外），Crocoddyl 式多接触 OCP 是系统化路线。第 04 章 §04.3 的微分平坦（(4.15)–(4.19)）与本文是"利用结构降维"的两条分支：平坦性适用于四旋翼类系统，接触丰富的腿式/臂式平台走本文的约束消元路线——按系统结构选，不竞争。

## 配套阅读

- 教程：[第 04 章｜轨迹优化与最优控制](../04_轨迹优化与最优控制.md)（(4.5)–(4.14) 的 iLQR 主线即本文 §4.4–4.6 的无间隙特例）｜ [第 05 章｜MPC](../05_模型预测控制mpc.md)（高频重解的消费方）
- 同目录精读：[RRT*（IJRR 2011）](RRTstar_IJRR2011.md)（第 04 章 OCP 的路径输入来源）
- 跨主题：[RoboTwin 精读总索引](../../robotwin/精读/README.md)（仿真栈与底层控制接口）；教程第 10 章 §10.2（位置增量 × 隐式 PD）
- 原始文献链：DDP 源头 Mayne (1966)、Jacobson & Mayne (1970)；iLQG：Tassa, Erez & Todorov (ICRA 2012)（式 18–19 的 $\bar{\mathbf{f}}=0$ 情形即 [32]）；不可行打靶的 Gauss-Newton 族：Giftthaler et al. (IROS 2018)（[20]，FDDP 反向 pass 的出发点）；解析导数：Carpentier & Mansard (RSS 2018)、Pinocchio [29]

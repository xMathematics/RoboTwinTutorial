# 论文精读｜FastSLAM（AAAI 2002）

> **PDF**：[papers/slam/classics/FastSLAM_AAAI2002_MontemerloThrun.pdf](../../../papers/slam/classics/FastSLAM_AAAI2002_MontemerloThrun.pdf) ｜ **教程**：[第 07 章 §7.3](../07_后端-i滤波与增量估计.md) ｜ **代码**：[projects/slam/fastslam/](../../../projects/slam/fastslam/)

## 1. 论文信息与一句话贡献

- **题目**：FastSLAM: A Factored Solution to the Simultaneous Localization and Mapping Problem
- **作者**：Michael Montemerlo、Sebastian Thrun（CMU）；Daphne Koller、Ben Wegbreit（Stanford）
- **发表**：AAAI-02 Proceedings，pp. 593–598（配套 PDF 已收录，页码 593–598 按论文原页）
- **一句话贡献**：把 SLAM 联合后验**精确分解**（论文 Eq.(4)）为"机器人路径分布 × 给定路径时各路标相互独立的条件分布"，用 Rao-Blackwellized 粒子滤波 + 每粒子 $K$ 个 2 维小 EKF 求解，单步复杂度从 EKF-SLAM 的 $O(K^2)$ 降到 $O(M \log K)$（$M$ 为粒子数），实验建到 **50,000 路标**（100 个粒子），远超当时 EKF 系方法的数百路标极限。

## 2. 问题与动机（要解决什么、为什么难、当时方法的瓶颈）

**要解决什么**：SLAM——机器人运动有噪声，观测有噪声，要同时估计"走过的路径"与"路标地图"的联合后验 $p(s^t, \theta \mid z^t, u^t, n^t)$（论文 Eq.(3)）。

**当时方法的瓶颈（EKF-SLAM）**：主流做法（Smith, Self & Cheeseman 1986）把位姿与全部路标塞进一个 $(3 + 2K)$ 维联合高斯，用 EKF 增量维护。论文摘要点出要害：*"Sensor updates require time quadratic in the number of landmarks K … the covariance matrix maintained by the Kalman filters has O(K²) elements, all of which must be updated even if just a single landmark is observed"*——**哪怕本帧只看见 1 个路标，也要改写全部 $O(K^2)$ 个协方差元素**。这一平方代价是结构性的（本文 §4.2 从联合高斯结构推出，与教程第 07 章 (7.6)–(7.7) 一致），把可处理路标数限制在几百个，而真实环境的特征是百万量级。此外单一高斯表达不了回环闭合前位姿的**多峰**后验（教程 §03.2 ③ 的"代价"讨论）。

**为什么难**：朴素的想法"既然观测只涉及一个路标，问题本该是稀疏的"在滤波范式下走不通——教程 §07.2 证明了 EKF-SLAM 一次更新就把互协方差填满、"稀疏性被烧掉"。FastSLAM 的洞察是：**稀疏性不该在"联合分布"里找，而该在"因子分解"里找**——给定完整路径后，各路标的估计本来就是相互独立的小问题（引言：*"knowledge of the robot's path renders the individual landmark measurements independent"*，此观察 Murphy [13] 已提出并用于栅格地图）。

## 3. 方法总览（系统框架/模块划分，文字版框图）

```
输入：里程计 u_t、测量 z_t（距离-方位角）、关联 n_t
  │
  ├─【路径估计：粒子滤波】论文 Eq.(5)-(7)
  │    M 个粒子 {s^{t,[m]}}，每个粒子按运动模型采样新位姿 s_t^[m]（Eq.(6)），
  │    按重要性权重 w_t^[m]（Eq.(7)）重采样
  │
  ├─【路标估计：每粒子 K 个独立 2 维 EKF】论文 Eq.(8)-(10)
  │    粒子 m 携带 {μ_k^[m], Σ_k^[m]}，k=1..K；
  │    本帧被观测路标 n_t 做 EKF 更新（Eq.(9)），其余路标保持不变（Eq.(10)）
  │
  ├─【数据关联：按粒子做 ML + 门限】论文 Eq.(12)
  │    n_t^[m] = argmax p(z_t | s_t^[m], n_t)；最大值低于阈值 α 则地图增广
  │
  └─【高效实现：平衡二叉树 + 写时复制】
       每粒子的高斯集合存为按路标号索引的平衡二叉树，
       新粒子只复制被修改的一条路径 → 单步 O(M log K)
```

关键设计点有二。其一，路标估计**以粒子位姿为条件**（论文：*"Since this estimate is conditioned on the robot pose, the Kalman filters are attached to individual pose particles"*），因此 $M$ 个粒子各带一套私有地图，共 $MK$ 个 2 维 EKF——路标间的耦合被"路径已知"这一条件吸收，EKF 退化成 2 维小问题。其二，不同粒子允许持有**不同的数据关联甚至不同数量的路标**（论文 Data Association 一节），错误关联的粒子会在重采样中被淘汰——EKF-SLAM 一次定死、无法回退的关联决策在这里变成了可竞争的假设。

与 EKF-SLAM 的结构对照（复杂度数字出处见 §4.6）：

| | EKF-SLAM | FastSLAM |
|---|---|---|
| 状态表示 | 一个 $(2K+3)$ 维联合高斯 | $M$ 个路径粒子 × 每粒子 $K$ 个 2 维高斯 |
| 单次测量更新 | 全体均值 + 全部 $O(K^2)$ 协方差元素 | 一个 2×2 高斯 + $M$ 个权重，$O(M \log K)$ |
| 后验形态 | 单峰高斯（多峰被高斯化抹平） | 多峰（粒子集表达回环前的位置假设） |
| 数据关联 | 一次 ML 定死，错误即污染协方差 | 按粒子竞争，错误粒子被重采样淘汰 |

## 4. 关键公式推导（核心章节）

### 4.0 符号表

| 符号 | 含义（首次出现处） |
|---|---|
| $s_t = (x, y, \psi)$ | $t$ 时刻机器人位姿（平面位置 + 朝向），§SLAM Problem Definition |
| $s^t = \{s_1,\dots,s_t\}$ | 路径（上标 $t$ 表示"时间 1 到 t 的集合"，论文约定） |
| $u_t,\ z_t$ | 里程计控制、测量；$z^t, u^t, n^t$ 为对应集合 |
| $\theta_k \in \mathbb{R}^2$ | 第 $k$ 个路标位置，$k = 1..K$；$\theta$ 为全部路标 |
| $n_t$ | 关联变量（correspondence）：$t$ 时刻观测到的路标编号 |
| $m$（上标 $[m]$） | 粒子编号，$m = 1..M$ |
| $\mu_k^{[m]}, \Sigma_k^{[m]}$ | 粒子 $m$ 中第 $k$ 路标高斯的均值/协方差（论文 Eq.(8)） |
| $h(s_t, \theta_{n_t})$ | 测量函数（距离-方位角），论文 Eq.(2) 的均值部分 |
| $\nu,\ H,\ S$ | 新息、测量雅可比、新息协方差（EKF 实现层记号，教程 (3.20)–(3.21)） |
| $d^2$ | 平方马氏距离 $\nu^\top S^{-1} \nu$（§4.5，由 Eq.(12) 的高斯似然导出） |

生成模型两行（论文 Eq.(1)–(2)，即第 03 章 (3.1) 的 SLAM 具体化）：

$$p(s_t \mid u_t, s_{t-1}) \tag{论文 Eq.(1)} \qquad\qquad p(z_t \mid s_t, \theta, n_t) \tag{论文 Eq.(2)}$$

### 4.1 EKF-SLAM 的 O(K²) 从哪来：联合高斯结构的必然

EKF-SLAM 维护联合高斯（教程 (7.1)）：状态 $\mathbf{x} = [\mathbf{x}_v; \mathbf{m}_1; \dots; \mathbf{m}_N]$，协方差分块，位姿-路标互协方差 $\Sigma_{v m_i}$ 与路标块并存。看一次"只观测第 $j$ 个路标"的更新：

**第一步，雅可比只有两块非零。** 观测方程 $\mathbf{z} = h(\mathbf{x}_v, \mathbf{m}_j) + \mathbf{v}$ 只含两个变量，故（依据：对状态向量逐分量求导，未涉及的分量为零；教程 (7.4)）

$$H = \big[\, H_v,\ \mathbf{0},\ \dots,\ H_j,\ \dots,\ \mathbf{0} \,\big].$$

**第二步，增益却处处非零。** EKF 增益 $K = \Sigma^- H^\top S^{-1}$，$S = H \Sigma^- H^\top + R$（依据：教程 (3.15)；$S$ 只取 $\Sigma^-$ 的位姿/路标 $j$ 四个块，即教程 (7.5)）。按块写 $K$ 的路标 $i$ 块（依据：分块矩阵乘法，教程 (7.6)）：

$$K_{m_i} = \big( \Sigma_{m_i v}^{-}\, H_v^\top + \Sigma_{m_i m_j}^{-}\, H_j^\top \big) S^{-1}.$$

对 $i \neq j$：$\Sigma_{m_i m_j}^{-} = \mathbf{0}$（预测不产生路标间相关性，教程 (7.3)），但 $\Sigma_{m_i v}^{-} \neq \mathbf{0}$——预测步已让位姿与**每个**路标的互协方差随 $F_v$ 传播（教程 (7.3) 第二式）——故 $K_{m_i} = \Sigma_{m_i v}^{-} H_v^\top S^{-1} \neq \mathbf{0}$（一般成立）。

**第三步，协方差交叉块被填满。** 协方差更新 $\Sigma = (I - K H)\Sigma^-$（依据：教程 (3.16)）的路标 $(i, j)$ 交叉块（依据：$KH\Sigma^-$ 按块展开，教程 (7.7)）：

$$[\Sigma]_{m_i m_j} = \Sigma_{m_i m_j}^{-} - K_{m_i}\big( H_v \Sigma_{v m_j}^{-} + H_j \Sigma_{m_j m_j}^{-} \big).$$

$i \neq j$、更新前 $\Sigma_{m_i m_j}^{-} = \mathbf{0}$，而右端第二项一般非零 → **一次更新后所有路标两两相关**。于是：均值更新要改 $N{+}1$ 个块、协方差写回要改 $O(N^2)$ 个块、$\Sigma$ 常驻内存 $O(N^2)$——论文摘要"即使只观测一个路标也必须更新全部 $O(K^2)$ 个元素"对应的正是第二、三步。这是**结构**代价：滤波范式在线维护联合分布，必然为"每对变量的相关性"付费（教程 §07.2 进一步从信息矩阵角度论证该稀疏性为何救不回来）。

### 4.2 条件独立与 Rao-Blackwell 分解（论文 Eq.(4)）

**目标**：分解 $p(s^t, \theta \mid z^t, u^t, n^t)$（论文 Eq.(3)）。

**第一步（链式法则拆成路径 × 地图）。** 按条件概率定义（依据：$p(a, b \mid c) = p(a \mid c)\, p(b \mid a, c)$，对任意事件成立）：

$$p(s^t, \theta \mid z^t, u^t, n^t) = p(s^t \mid z^t, u^t, n^t)\; p(\theta \mid s^t, z^t, u^t, n^t).$$

**第二步（给定路径，测量因子按路标分组）。** 写出给定路径后"地图 × 全部测量"的联合（依据：生成模型，论文 Fig. 1——由 Eq.(2)，测量 $z_t$ 给定 $(s_t, \theta_{n_t}, n_t)$ 与其余一切独立；路标位置无任何先验关联，$p(\theta) = \prod_k p(\theta_k)$；$u^t$ 只通过路径 $s^t$ 影响 $(\theta, z^t)$，给定 $s^t$ 后可从条件中省去——马尔可夫性）：

$$p(\theta, z^t \mid s^t, n^t) = \Big( \prod_{k=1}^{K} p(\theta_k) \Big) \prod_{t'=1}^{t} p\big( z_{t'} \mid s_{t'},\, \theta_{n_{t'}},\, n_{t'} \big).$$

**第三步（因子集合不相交 ⟹ 连乘分解）。** 把右端按"因子含哪个 $\theta_k$"分组：第 $k$ 组 = $p(\theta_k)$ 加上所有 $n_{t'} = k$ 的测量因子。每个测量因子**恰含一个** $\theta_k$（依据：Eq.(2) 中 $z_{t'}$ 只依赖 $\theta_{n_{t'}}$），故各组之间无公共因子。固定 $k$、把其余 $\theta_{k'}\ (k' \neq k)$ 视为常数并逐组归一化（依据：贝叶斯定理逐因子应用；分母 $\int p(\theta_k)\prod_{t':n_{t'}=k} p(z_{t'} \mid s_{t'}, \theta_k, n_{t'})\,\mathrm{d}\theta_k$ 是只依赖本组因子的归一化常数）：

$$p(\theta_k \mid s^t, z^t, n^t) \;\propto\; p(\theta_k) \prod_{t':\, n_{t'} = k} p\big( z_{t'} \mid s_{t'},\, \theta_k,\, n_{t'} \big), \qquad p(\theta \mid s^t, z^t, n^t) = \prod_{k=1}^{K} p(\theta_k \mid s^t, z^t, n^t).$$

合并第一、三步（论文原文把两步合成一行）：

$$p(s^t, \theta \mid z^t, u^t, n^t) = p(s^t \mid z^t, u^t, n^t) \prod_{k=1}^{K} p(\theta_k \mid s^t, z^t, u^t, n^t) \tag{论文 Eq.(4)}$$

这正是教程 (7.11)。**要点**：该分解是**精确**的（论文：*"This factorization is exact and always applicable in the SLAM problem, as previously argued in [13]"*）——近似全部来自后面的采样与线性化，分解本身零损失。"给定路径时各路标条件独立"的几何根源即第二步：测量因子是"一位姿-一路标"的局部因子（对照教程 (7.9)：信息矩阵的稀疏模式 = 变量共现）。Rao-Blackwell 定理（教程 (7.12) 全方差公式）保证"解析积分掉 $\theta$、只在路径上采样"不增方差，且给定路径后每个 $p(\theta_k \mid \cdot)$ 是高斯（高斯先验 × 线性化高斯似然 → 高斯后验）。

**为什么 EKF-SLAM 拿不到这条分解**。Eq.(4) 是关于**后验**的恒等式，难点在于在线维护：路径后验 $p(s^t \mid z^t, u^t, n^t)$ 是多峰的（回环前位置假设不唯一），单一高斯装不下；而若把路径也塞进联合高斯（EKF-SLAM 的选择），教程 (7.6)–(7.7) 已证明一次更新就产生路标两两相关、把 §4.2 的条件独立结构破坏掉——分解是精确的，**滤波所维护的高斯近似恰恰不满足它**。FastSLAM 的分工由此而来：多峰的那部分（路径）用采样，条件高斯的那部分（路标）用 EKF。

### 4.3 路标 EKF 更新（论文 Eq.(8)–(10)）

**表示**（论文 Eq.(8)）：每个粒子携带全部路标高斯

$$S_t = \{ s^{t,[m]},\ \mu_1^{[m]}, \Sigma_1^{[m]},\ \dots,\ \mu_K^{[m]}, \Sigma_K^{[m]} \}_m.$$

**更新（论文 Eq.(9)，$n_t = k$ 时）**：

$$p(\theta_k \mid s^t, z^t, u^t, n^t) \;\overset{\text{Bayes}}{\propto}\; p(z_t \mid \theta_k, s_t, z^{t-1}, u^t, n^t)\; p(\theta_k \mid s_t, z^{t-1}, u^t, n^t)$$
$$\overset{\text{Markov}}{=}\; p(z_t \mid \theta_k, s_t, n_t)\; p(\theta_k \mid s^{t-1}, z^{t-1}, u^{t-1}, n^{t-1})$$

依据逐步注解：① Bayes 步——把 $z_t$ 的似然乘到先验上，分母 $p(z_t \mid \cdots)$ 与 $\theta_k$ 无关，吸收进 $\propto$；② Markov 步——左因子由 Eq.(2) 化简（给定 $(\theta_k, s_t, n_t)$ 与更早的历史无关）；右因子是 $t-1$ 时刻该路标的后验：给定路径后，$u^t$、$s_t$ 对**未被本帧测量改动前**的路标估计没有影响（路标静止；$n^t$ 对 $\theta_k$ 的信息在本帧测量里，已归入左因子）。

**未观测的路标（论文 Eq.(10)）**：$n_t \neq k$ 时 $p(\theta_k \mid s^t, z^t, u^t, n^t) = p(\theta_k \mid s^{t-1}, z^{t-1}, u^{t-1}, n^{t-1})$——路标后验原样保持。注意这同时说明 FastSLAM 的路标"预测步"是**恒等映射**：路标不动，也无须对路标传播运动不确定性（这是与普通 KF 的一个结构差别；论文只把 Eq.(9) 交给 EKF 实现）。

**EKF 实现层**（论文：*"implements the update equation (9) using the extended Kalman filter (EKF) … uses a linearized version of the perceptual model"*）。设本帧粒子 $m$ 的位姿采样为 $s_t^{[m]}$，被观测路标 $n_t$ 的当前高斯为 $\mathcal{N}(\mu, \Sigma)$（即右因子，2 维）。距离-方位角测量的雅可比（`fastslam.py: range_bearing_jacobian`，$r = \lVert \theta_k - s_t^{[m],xy}\rVert$）：

$$H = \frac{\partial h}{\partial \theta_k}\bigg|_{\mu,\, s_t^{[m]}} = \begin{bmatrix} \dfrac{\Delta x}{r} & \dfrac{\Delta y}{r} \\[1ex] -\dfrac{\Delta y}{r^2} & \dfrac{\Delta x}{r^2} \end{bmatrix}, \qquad \nu = z_t - h(\mu, s_t^{[m]}), \qquad S = H \Sigma H^\top + R.$$

代入 EKF 测量更新（依据：一阶 Taylor 线性化 $h(\theta_k) \approx h(\mu) + H(\theta_k - \mu)$，同教程 (3.20)；增益/更新公式同教程 (3.21)）：

$$K_n = \Sigma H^\top S^{-1}, \qquad \mu^{+} = \mu + K_n\, \nu, \qquad \Sigma^{+} = (I - K_n H)\, \Sigma.$$

论文并指出（Landmark Location Estimation 末段）：因为路径是**采样**出来的而非高斯化推断，路标后验在"线性化测量模型 + 高斯先验"下**严格**是高斯——运动模型的非线性误差不会进入路标估计（它只污染路径采样，由粒子集表达）。更新代价：每次只动一个 2×2 高斯，$O(1)$；对照 EKF-SLAM 的 $(2K+3)$ 维更新 $O(K^2)$（论文原文对比）。

**首次观测：逆测量模型初始化（地图增广）**。论文的地图增广规则（Data Association 一节：低于阈值 $\alpha$ 则"augmented accordingly"）在实现层需要一个新高斯的初值：从测量反解路标位置（依据：Eq.(2) 的逆，在粒子 $m$ 的位姿 $s_t^{[m]} = (x, y, \psi)$ 处，`fastslam.py: _initialize_slots`）：

$$\mu = \begin{pmatrix} x + r\cos(\psi + \phi) \\ y + r\sin(\psi + \phi) \end{pmatrix}, \qquad \Sigma = J\, R\, J^\top, \qquad J = \frac{\partial (l_x, l_y)}{\partial (r, \phi)} = \begin{bmatrix} \cos(\psi+\phi) & -r\sin(\psi+\phi) \\ \sin(\psi+\phi) & r\cos(\psi+\phi) \end{bmatrix}.$$

依据：测量噪声经逆测量模型的雅可比 $J$ 线性传播（与教程 (3.21) 的协方差更新同一传播法则，只是方向反过来）；$R = \mathrm{diag}(\sigma_r^2, \sigma_\phi^2)$。注意初始化后的新息恒为零，故 Eq.(11) 的似然因子此时退化为归一化常数 $\mathcal{N}(0; S)$——`_initialize_slots` 的权重注释即此。

### 4.4 重要性权重（论文 Eq.(7) 与 Eq.(11)）

**权重的由来**。提议分布（proposal）取运动模型（论文 Eq.(6)）：$s_t^{[m]} \sim p(s_t \mid u_t, s_{t-1}^{[m]})$——**不含当前观测 $z_t$**。重要性采样标准结论：从 $q$ 采样、按"目标分布 / 提议分布"加权，即可无偏近似目标分布（论文 Eq.(7) 引 [10]；教程第 03 章 §03.2 的高斯单峰限制正是改用采样的动机）：

$$w_t^{[m]} \;=\; \frac{\text{target distribution}}{\text{proposal distribution}} \;=\; \frac{p\big(s^{t,[m]} \mid z^t, u^t, n^t\big)}{p\big(s^{t,[m]} \mid z^{t-1}, u^t, n^{t-1}\big)} \tag{论文 Eq.(7)}$$

**化简链（论文 Eq.(11)）**——从 Eq.(7) 到可计算的测量似然，每步依据如下：

$$\frac{p\big(s^{t,[m]} \mid z^t, u^t, n^t\big)}{p\big(s^{t,[m]} \mid z^{t-1}, u^t, n^{t-1}\big)} \;\overset{\text{Bayes}}{=}\; \frac{p\big(z_t, n_t \mid s^{t,[m]}, z^{t-1}, u^t, n^{t-1}\big)}{p\big(z_t, n_t \mid z^{t-1}, u^t, n^{t-1}\big)} \;\propto\; p\big(z_t, n_t \mid s_t^{[m]}, z^{t-1}, u^t, n^{t-1}\big)$$

① Bayes 步：把分子按新观测展开 $p(s^t \mid z^t,\cdots) \propto p(z_t, n_t \mid s^t, \cdots)\, p(s^t \mid z^{t-1}, \cdots)$，分母是归一化常数（与 $s^{t,[m]}$ 无关）。② 消去：分子的第二因子与 Eq.(7) 的分母（提议分布）**精确相消**。③ 路径→当前位姿：由 Eq.(2)，$z_t, n_t$ 给定 $s_t$ 后与更早位姿无关，$s^t$ 缩成 $s_t^{[m]}$。

$$\;\overset{\text{全概率}}{=}\; \int p\big(z_t, n_t \mid \theta, s_t^{[m]}, \cdots\big)\, p\big(\theta \mid s^{t,[m]}, z^{t-1}, u^t, n^{t-1}\big)\, \mathrm{d}\theta \;\overset{\text{Markov}}{=}\; \int p(z_t \mid \theta, s_t, n_t)\, p(n_t \mid \theta, s_t)\; p\big(\theta \mid s^{t-1,[m]}, z^{t-1}, u^{t-1}, n^{t-1}\big)\, \mathrm{d}\theta$$

④ 全概率公式：对路标位置 $\theta$ 边缘化（这一步把"测量该有多意外"折算到当前地图的信念上）。⑤ Markov：左因子由 Eq.(2) 化简；右因子换成 $t-1$ 时刻的路标先验（同 §4.3 的论证）。

$$\;\overset{p(n_t \mid \theta, s_t)\,\text{取均匀}}{\propto}\; \int p(z_t \mid \theta, s_t, n_t)\, p\big(\theta \mid s^{t-1,[m]}, \cdots\big)\, \mathrm{d}\theta \;\overset{\text{EKF}}{\approx}\; \int p\big(z_t \mid \theta, s_t^{[m]}, n_t\big)\, \mathcal{N}\big(\theta;\, \mu_{n_t}^{[m]}, \Sigma_{n_t}^{[m]}\big)\, \mathrm{d}\theta_{n_t} \tag{论文 Eq.(11)}$$

⑥ 论文原文假设 $p(n_t \mid \theta, s_t^{[m]})$ 均匀（SLAM 文献惯例），该因子被吸收进 $\propto$。⑦ EKF 步：$p(\theta \mid \cdots)$ 用粒子 $m$ 中第 $n_t$ 个高斯代入（§4.3 的充分统计量），观测模型线性化（论文：*"EKF makes explicit the use of a linearized approximation to the observation model"*）。最后这个积分**有闭式**（线性高斯下易算）：线性化 $z_t = h(\mu_{n_t}^{[m]}) + H(\theta - \mu_{n_t}^{[m]}) + v$，其中 $\theta - \mu \sim \mathcal{N}(0, \Sigma_{n_t}^{[m]})$、$v \sim \mathcal{N}(0, R)$ 独立（依据：仿射变换下的高斯——均值/协方差按定义直接算，$\mathbb{E}[z_t] = h(\mu)$、$\mathrm{Cov} = H \Sigma H^\top + R$），故

$$\boxed{\; w_t^{[m]} \;=\; p\big(z_t \mid s_t^{[m]}, \Theta, n_t\big) \;=\; \mathcal{N}\big(z_t;\; \hat{z},\ S\big), \qquad \hat{z} = h(\mu_{n_t}^{[m]}, s_t^{[m]}),\quad S = H \Sigma_{n_t}^{[m]} H^\top + R \;}$$

即"新观测在粒子 $m$ 私有地图预测下的似然"。这正是 `fastslam.py` 折入对数权重的量：`log_weights -= 0.5*(d² + log|2πS|)`（`_ekf_update_slots`，依据注释即 Eq.(7)/(11)）。**注意 Eq.(6) 的提议不含 $z_t$**——这个"偷懒"把全部测量信息压进了权重，是 §4.5 门控失配的根源（对照：FastSLAM 2.0 把 $z_t$ 揉进提议分布，正是对这一点的修正，见 §6）。

### 4.5 数据关联与马氏距离门控（论文 Eq.(12)）

论文把关联变量 $n_t$ 的后验按路径边缘化（依据：全概率公式），再逐级化简：

$$p(n_t \mid z^t, u^t) = \int p(n_t \mid s^t, z^t, u^t)\, p(s^t \mid z^t, u^t)\, \mathrm{d}s^t \;\overset{\text{PF}}{\approx}\; \sum_m p(n_t \mid s^{t,[m]}, z^t, u^t) \;\overset{\text{Markov}}{=}\; \sum_m p(n_t \mid s_t^{[m]}, z_t) \;\overset{\text{Bayes}}{\propto}\; \sum_m p\big(z_t \mid s_t^{[m]}, n_t\big) \tag{论文 Eq.(12)}$$

依据：PF 步用加权粒子集近似路径后验（Eq.(7) 的采样集本身就是这个近似）；Markov 步同前（测量只经 $s_t$ 依赖路径）；Bayes 步取均匀先验 $p(n_t \mid s_t)$（论文原文注明，[2]）。**极大似然关联** = 取使 (12) 最大的 $n_t$；论文强调 FastSLAM 的关联是**按粒子**估计的：$n_t^{[m]} = \arg\max_{n_t} p(z_t \mid s_t^{[m]}, n_t)$，不同粒子可持不同关联、甚至不同数量的路标；若 (12) 的最大值（"仔细斟酌所有常数"之后）低于阈值 $\alpha$，则判定为新路标、地图增广。

**马氏距离与 $\chi^2$ 门控**。式 (12) 的每项由 §4.4 已化为高斯 $\mathcal{N}(z_t; \hat{z}_j, S_j)$（对粒子 $m$、候选路标 $j$），其负对数的二次部分即**平方马氏距离**（依据：高斯密度取 $-\log$，$-\log \mathcal{N} = \tfrac12 d^2 + \tfrac12 \log|2\pi S| + \text{const}$，$d^2 = \nu^\top S^{-1} \nu$）：

$$d^2 = \nu^\top S^{-1} \nu, \qquad \nu = z_t - h(\mu_j^{[m]}, s_t^{[m]}), \qquad S_j = H_j \Sigma_j^{[m]} H_j^\top + R.$$

若测量模型与滤波器假设完全一致（2 维测量），$d^2$ 应服从自由度 2 的卡方分布 $\chi^2(2)$：中位数 $2\ln 2 \approx 1.39$、95% 分位 $5.991$（教科书门控值，即 `fastslam.py` 的 `GATE_CHI2_2DOF_95 = 5.991`）。

**实现经验：$d^2$ 实际超出理想 $\chi^2(2)$ 尺度**。原因在提议分布：Eq.(6) 不看 $z_t$，采出的 $s_t^{[m]}$ 携带运动噪声误差，而 $S = H\Sigma H^\top + R$ 只计入了"路标估计不确定 + 测量噪声"，**没有**把这份位姿采样误差算进新息协方差——于是合法重观测的 $d^2$ 系统性偏大。在本仓库基准场景（`simulate_rectangle`，seed=7，$N=50$ 粒子、60 步、真值关联）实测：全程合并的 $d^2$ 中位数约 1.9、95 分位约 8.6，**约 13% 的合法重观测超过教科书门限 5.991**（后半程路标协方差收紧后升至约 17%、95 分位约 9.5）。照抄 5.991 的后果：每次拒真都把一个已知路标复制成新槽位（`projects/slam/DEBUG.md` 案例 1 的"路标重复"）。工程解法是**双尺度门限**（`fastslam.py` 的 `GATE_DEFAULT` docstring 与 DEBUG.md F-1/F-3）：合法重观测 $d^2 = \chi^2(2) + \text{位姿采样误差}$，基准噪声下保持在约 20 以内；不同路标之间 $d^2 \ge (\text{间距}/\sigma)^2 \approx (2\,\mathrm{m}/0.3\,\mathrm{m})^2 \approx 44$——门限取 **30**，落在两尺度之间且两侧都有余量。这与论文在 Eq.(12) 后的自述一致：接受门限 $\alpha$ 需要"对 (12) 中所有常数的仔细考虑"，而不是照抄分位数表。

## 5. 实验与结果解读

论文的实验分两类（Experimental Results 一节）：

- **物理实验（定性验证精度）**：NASA 火星车研究试验床，Pioneer 机器人 + SICK 激光，绕石块直线行驶。仅用 **M = 10 个粒子**建图，与人工标注的路标位置比对，**平均残差地图误差 8.3 厘米**（Fig. 4）。说明：极小粒子数下算法即给出米级场景内厘米级的地图——重采样与逐粒子 EKF 的组合在真实噪声下工作。
- **仿真实验（规模化能力）**：路标数逐步加到 **50,000**，FastSLAM 用 **100 个粒子**成功建图；此时其参数量约为传统 EKF-SLAM 的 **0.3%**（论文口径）——论文明言 5 万路标量级对 EKF 系方法"因计算复杂度不可及"。Fig. 6a/6b 给出误差随 $K$、$M$ 的扫描（柱为 95% 置信区间）：**增大路标数 $K$ 反而轻微降低**地图与位姿误差——路标越多，任一时刻可用的定位约束越多；**增大粒子数 $M$** 同样单调改善两项误差。

**解读**：真正反直觉且有工程价值的是第一条——粒子数与路标数**解耦**（$M$ 不必随 $K$ 增长），根源是 §4.2 的分解：$M$ 只需覆盖"路径"这个低维多峰分布的不确定性，$K$ 维地图信息被 Rao-Blackwell 化地塞进了每粒子的小 EKF。论文同时诚实报告了边界：在某些情形下"准确建图所需的粒子数可能大得不可行"（摘要）——粒子贫化是 §6 的第一局限。

**数据关联的鲁棒性从哪来**。论文 Data Association 一节给出一个对 EKF 系相当尖锐的对比：传统 EKF-SLAM 对每条测量**只做一次**关联决策，错误决策经稠密增益写进整个协方差、无法撤销（*"false data association will make the conventional EKF approach fail catastrophically [2]"*）；FastSLAM 则允许不同粒子携带不同的关联假设同时竞争（§4.5 的按粒子 ML），"关联错误的粒子（在期望上）比猜对的粒子更容易在重采样中消失"。这把关联从"一次性的硬决策"变成"随滤波器演化的软假设"——代价是错误关联在消失前也会复制进部分粒子的地图（§6 局限 3）。

## 6. 局限与后续影响

**局限**（论文自述 + 教程 §7.3 ③）：

1. **重采样导致路径贫化（path depletion）**：重采样会复制高权重粒子、丢弃低权重粒子，粒子承载的**历史路径**随之同质化，早期的路标估计失去多样性来源。论文只以"某些情形 $M$ 需求过大"带过；后续文献将其定名为 path depletion。
2. **提议分布不含当前观测**（§4.4/§4.5 已见）：权重方差大、门控失配。后续工作 **FastSLAM 2.0**（Montemerlo et al., 2003）把 $z_t$ 揉进提议分布（在测量附近迭代优化采样位姿），直接针对这一点——采样位姿更贴近测量后验后，$d^2$ 回到 $\chi^2(2)$ 尺度、门控失配与权重退化同时缓解。
3. **均匀 $p(n_t \mid \theta, s_t)$ 假设**与逐粒子 ML 关联：关联错误会随粒子复制传播（论文 Data Association 一节承认不同粒子的地图可能互相不一致）；kd-tree 索引只是复杂度层面的缓解。
4. **平面点路标假设**：路标为平面 2 维点、关联大部分篇幅假设已知；推广到 3 维、未知 $K$、未知关联只以扩展形式简述。
5. 粒子滤波近似的收敛保证是渐近意义（$M \to \infty$），有限粒子下后验只是近似（论文 Eq.(7) 段落明示）。

**后续影响**：Rao-Blackwellized 粒子滤波成为 SLAM 标准工具之一——同一条分解思路在栅格地图上发展出 GMapping 一系；"条件独立分解 + 解析滤波"的概率结构（论文 Eq.(4)）也是后来平滑方法对比讨论的基准点。在教程的因果链里，FastSLAM 的角色是"证明滤波范式还有一次结构性红利可吃"：红利吃完（$O(M \log K)$），多峰表达能力与粒子贫化的矛盾仍在，第 08 章的图优化范式（保留因子、整体重线性化）才是终局。

## 7. 与本项目对照

- **教程**：[第 07 章 §7.3](../07_后端-i滤波与增量估计.md) 的 (7.11) 即论文 Eq.(4)（本章 §4.2 给了比教程更细的三步推导）；(7.12) 是 Rao-Blackwell 的全方差表述；EKF-SLAM 的 $O(N^2)$ 来源对应教程 (7.6)–(7.7)（本章 §4.1 复现）。EKF 更新公式出处为第 03 章 (3.20)–(3.21)（注意第 03 章 §03.3 是 EKF，不含粒子滤波内容）。
- **代码**：[projects/slam/fastslam/fastslam.py](../../../projects/slam/fastslam/fastslam.py) 的 `FastSLAM2D` 逐步对应论文式号——`_motion_update`（Eq.(6) 采样）、`_update_nearest`（Eq.(12) 门控最近邻：$d^2$ 矩阵、`j_star` ML 选择、门限分流）、`_ekf_update_slots`（Eq.(9) 的 EKF 更新 + Eq.(7)/(11) 的似然折入权重）、`_weight_only_slots`（越门限且无空槽时只罚权重）、`_initialize_slots`（逆测量模型初始化）、`resample` + `effective_sample_size`（$N_{\mathrm{eff}} = 1/\sum_i w_i^2$ 低于 `resample_fraction * N` 触发系统重采样）。门控常数：`GATE_CHI2_2DOF_95 = 5.991`（教科书值，标注为"实际偏紧"）与 `GATE_DEFAULT = 30.0`（双尺度论证见其 docstring）。
- **健康值**（[projects/slam/METRICS.md](../../../projects/slam/METRICS.md) §2.3，来自 `tests/test_fastslam.py` 实测）：known 关联模式**位姿 RMSE 0.113 m**（轨迹 0.221 m）vs 航位推算基线 **1.571 m**（≈ 1/14，断言要求 < 基线 1/3）——后端把漂移压掉一个数量级以上；门控最近邻模式（`test_nearest_neighbor_gating_beats_dead_reckoning`）额外断言 20 个路标全部被正确发现且不产生重复槽位。调试入口见 [projects/slam/DEBUG.md](../../../projects/slam/DEBUG.md) 案例 1（断点 `_update_nearest` 的 `d2` 行）。

## 配套阅读

- 论文：[FastSLAM（AAAI 2002）PDF](../../../papers/slam/classics/FastSLAM_AAAI2002_MontemerloThrun.pdf)（本精读所有式号以此 PDF 为准）
- 教程：[第 03 章](../03_概率状态估计基础.md)（EKF 五公式 (3.17)–(3.21)、条件高斯 (3.10)–(3.11)）｜[第 07 章](../07_后端-i滤波与增量估计.md)（EKF-SLAM 与 Rao-Blackwell）｜[第 08 章](../08_后端-ii图优化与-ba.md)（为何最终转向图优化）
- 教材：Thrun, Burgard & Fox, *Probabilistic Robotics*, ch13（FastSLAM 完整版，含树结构的伪代码）；Dissayanake et al. (2000)（EKF-SLAM 一致性讨论，论文 [2]）
- 延伸：Montemerlo et al., *FastSLAM 2.0*（2003）——把观测揉进提议分布，直接回应 §4.4/§4.5 的两个实现痛点

# 论文精读｜RRT*：采样式最优运动规划（IJRR 2011）

> **PDF**：[papers/control_planning/classics/arXiv-1105.1186_RRTstar.pdf](../../../papers/control_planning/classics/arXiv-1105.1186_RRTstar.pdf)（arXiv:1105.1186 版，共 76 页，定理编号以此为准） ｜ **教程**：[第 03 章](../03_运动规划-i采样式规划.md) §03.2（压轴） ｜ **代码**：论文随文发布作者开源库（原文第 5 页：ares.lids.mit.edu/software/），后并入 OMPL 的 `geometric/PRMstar`、`geometric/RRG`、`geometric/RRTstar` 等类 ｜ **前置**：[RRT 精读](RRT_TR1998.md)（同目录）

## 1. 论文信息与一句话贡献

- **题目**：Sampling-based Algorithms for Optimal Motion Planning
- **作者**：Sertac Karaman、Emilio Frazzoli（MIT）
- **发表**：International Journal of Robotics Research（IJRR）2011；本仓库为 arXiv:1105.1186 预印本（会议先行版 ICRA 2010/WAFR 相关内容在文中以 Karaman & Frazzoli 2010a/b 自引说明）。
- **编号约定（重要）**：论文正文**展示公式一律无编号**，编号单位是 Definition / Assumption / Lemma / Theorem / Algorithm——本精读按原文编号引用（Def 1/14/24/51，Assumption 27，Lemma 25/28/40–50/71，Cor 49，Thm 15/16/29/33–39，Algorithm 3–6，附录 A–G），推导步骤自编 **(S1)–(S8)** 并逐条声明对应原文位置。
- **一句话贡献**：证明 PRM 与 RRT 这类**固定/朴素连接规则**的采样式规划器**不是渐近最优的**（RRT 几乎必然收敛到次优的随机变量，Thm 33），并提出 PRM*/RRG/RRT* 三个算法——把连接半径取为**收缩球半径** $r(n) = \gamma (\log n / n)^{1/d}$——在**只增加常数倍计算开销**的条件下取得渐近最优性（Thm 34/36/38）。

## 2. 问题与动机

**要解决什么**（§1）：可行（feasible）vs 最优（optimal）。PRM/RRT 时代采样式规划的全部理论承诺是**概率完备**——有解以概率 1 找到，但找到的解可以是任意差的。机器人日复一日在同一环境重复规划时，"每次都比可达最优多走一截且方差可观"不可接受。论文的核心诘问（§4.2 开头）：采样式规划器返回的解，样本数 $n \to \infty$ 时能否以概率 1 收敛到最优代价 $c^*$？

**三个具体的理论悬念**（对应 §4 的三个小节）：

1. RRT/PRM 到底差在哪——是"运气不好"还是结构性次优？（Thm 29/33）
2. 如果结构性次优，最小的算法改动是什么？（§3.3 的选父 + 重布线 + 收缩半径）
3. 改动的代价是多少？（§4.3：碰撞检测调用从每轮 $\Theta(n)$ 降到 $O(\log n)$，Table 1）

论文还带出一个方法论贡献：把采样式规划器（去障碍后）**识别为经典随机几何图**（random geometric graph）——固定半径 r-盘图、k-近邻图、在线最近邻图（§2.2, Def 4–13）——从而把渗流（percolation）与连通性的既有阈值结果搬进规划分析。文末（第 36 页）猜想：概率完备 ⟺ 底层随机图渗流；渐近最优 ⟺ 底层随机图连通。

## 3. 方法总览

**问题形式化一瞥**（§2.1，详见 §4.1）：构型空间 $\mathcal{X} = (0,1)^d$，自由空间 $\mathcal{X}_{\text{free}} = \mathrm{cl}(\mathcal{X} \setminus \mathcal{X}_{\text{obs}})$，规划问题 = 三元组 $(\mathcal{X}_{\text{free}}, x_{\text{init}}, \mathcal{X}_{\text{goal}})$；最优问题（Problem 3）= 求可行路径 $\sigma^*$ 使 $c(\sigma^*) = \min\{c(\sigma) : \sigma \text{ feasible}\}$。

**三个新算法 = 同一连接半径的三个投影**（§3.3）：

```
             半径 r(n) = γ (log n / n)^{1/d}（收缩球半径，γ 见 Thm 34/36/38）
                        │
   ┌────────────────────┼─────────────────────┐
   ▼（批量建图）          ▼（增量建图）            ▼（增量建树）
 PRM* (Algorithm 4)    RRG (Algorithm 5)      RRT* (Algorithm 6)
 n 个样本一次撒好        逐样本：最近邻连一条      在 RRG 基础上再加两条规则：
 每点与半径内全部互连     + 半径内全部互连（允许环）  ①选父：多条可行连接只留代价最小的一条
 图查证 Dijkstra/A*     图查证                   ②重布线：新点反过来改接邻域旧点
```

三者关系（§3.3 原文事实）：**同一采样序列下，RRT 的树是 RRG 图的子图**（边集子集、顶点集相同）；RRT* = RRG 去掉冗余边后的形态（"redundant" = 不在任何根到顶点最短路上）；$k$-近邻变体（$k \sim \log n$）对应 Thm 35/37/39。Table 1（第 5 页）总览：RRT* 处理 $O(n\log n)$、查询 $O(n)$、空间 $O(n)$，概率完备 Yes、渐近最优 Yes、单调收敛 Yes——而 RRT 一行的渐近最优栏是 **No**。

## 4. 关键公式推导（原文公式无编号，按 Definition/Theorem/Lemma 引用；步骤编号 (S1)–(S8) 为本精读自加）

### 4.0 符号表

| 符号 | 含义（对应原文位置） |
|---|---|
| $\mathcal{X},\ \mathcal{X}_{\text{obs}},\ \mathcal{X}_{\text{free}}$ | 构型空间 $(0,1)^d$ / 障碍区 / 自由空间（§2.1） |
| $\sigma: [0,1] \to \mathbb{R}^d$ | 有界变差函数；路径需连续且全程落在 $\mathcal{X}_{\text{free}}$（Def 1） |
| $\mathrm{TV}(\sigma)$ | 全变差：$\sup_{\{\tau_i\}} \sum_i \lVert \sigma(\tau_i) - \sigma(\tau_{i-1}) \rVert$（§2.1，路径长度的解析替身） |
| $c: \Sigma \to \mathbb{R}_{\ge 0}$ | 代价泛函：严格正、单调、有界 $c(\sigma) \le k_c \mathrm{TV}(\sigma)$（§2.1） |
| $Y_n^{\mathrm{ALG}}$ | 算法 ALG 第 $n$ 轮返回图中最优路径的代价（§4 开头） |
| $\mu(\cdot),\ \zeta_d$ | Lebesgue 测度 / $d$ 维单位球体积（§4.2.2 开头） |
| $r(n),\ \gamma,\ \eta$ | 连接半径 / 常数 / steer 步长上限（Algorithm 4–6 的 Near 参数） |
| $V_n,\ E_n,\ G_n$ | 第 $n$ 轮的顶点集、边集、图（各算法） |

### 4.1 最优规划问题的形式化（§2.1–§2.2）

**全变差与路径。** 论文先对任意有界变差函数定义

$$
\mathrm{TV}(\sigma) = \sup_{\{n \in \mathbb{N},\ 0 = \tau_0 < \tau_1 < \cdots < \tau_n = 1\}} \sum_{i=1}^{n} \big\lVert \sigma(\tau_i) - \sigma(\tau_{i-1}) \big\rVert
\tag{S1}
$$

（原文 §2.1 展示式，照录；"取遍一切分割的上确界"是弧长的标准定义。）Def 1 分三级：**path**（连续）→ **collision-free path**（$\sigma(\tau) \in \mathcal{X}_{\text{free}}$）→ **feasible path**（再加 $\sigma(0) = x_{\text{init}}$，$\sigma(1) \in \mathrm{cl}(\mathcal{X}_{\text{goal}})$）。Problem 2（可行规划）与 Problem 3（最优规划）：找 $\sigma^*$ 使 $c(\sigma^*) = \min\{c(\sigma) : \sigma \text{ feasible}\}$，无可行解报失败。

**代价三公理**（§2.1）：严格正（$c(\sigma) = 0 \iff \sigma$ 常值）、**单调**（$c(\sigma_1) \le c(\sigma_1 \oplus \sigma_2)$，$\oplus$ 为拼接）、**有界**（$c(\sigma) \le k_c \mathrm{TV}(\sigma)$）。**注意**：任务规格中的 $c(\sigma) = \int_0^1 \lVert \dot\sigma \rVert \mathrm{d}t$ 不是论文的定义——论文用抽象三公理覆盖一切"合理"代价；实验（§5 开头）默认取 $\mathrm{TV}(\sigma)$ 为代价，而 $\mathrm{TV}$ 对光滑路径恰等于 $\int \lVert \dot\sigma\rVert$（依据：弧长公式；$C^1$ 路径时上确界在匀速分割处达到）。即积分泛函是本文框架下的一个实例而非定义本身。

**渐近最优的定义链**（§4.2）：路径的 **strong $\delta$-clearance** = 全程落在 $\delta$-内部 $\mathrm{int}_\delta(\mathcal{X}_{\text{free}}) := \{x : B_{x,\delta} \subseteq \mathcal{X}_{\text{free}}\}$；问题 **robustly feasible** = 存在某 $\delta > 0$ 的强 clearance 可行路径；**weak $\delta$-clearance** = 存在同伦类内一致带强 clearance 的逼近路径（允许贴障碍，Fig. 4 的双球切点例）；robustly optimal 的 $\sigma^*$ = 解 Problem 3 且有 weak clearance、代价沿同伦连续。Def 24：

$$
\mathbb{P}\Big(\Big\{ \limsup_{n \to \infty} Y_n^{\mathrm{ALG}} = c^* \Big\}\Big) = 1
\tag{S2}
$$

（Def 24 原式照录。）配套两条技术假设：Assumption 27（最优轨迹过点集零测——"有限次采样恰好采到最优路径上"的概率为零，Lemma 28）；Lemma 25（$\limsup Y_n$ 是否等于 $c^*$ 是 0-1 律事件）。

### 4.2 三算法伪代码逐行形式化（Algorithm 4–6）

**PRM\***（Algorithm 4）：撒 $n$ 个样本入 $V$，对每个 $v$ 取 $U \leftarrow \mathrm{Near}(G, v, \gamma_{\mathrm{PRM}}(\log n / n)^{1/d}) \setminus \{v\}$，对每个 $u \in U$ 碰撞检测通过则双向连边。常数要求 $\gamma_{\mathrm{PRM}} > \gamma^*_{\mathrm{PRM}} = 2(1 + 1/d)^{1/d} \big(\mu(\mathcal{X}_{\text{free}})/\zeta_d\big)^{1/d}$（§3.3 原文照录）。

**RRG**（Algorithm 5）：每轮 $x_{\text{new}} \leftarrow \mathrm{Steer}(x_{\text{nearest}}, x_{\text{rand}})$ 过碰撞检测后入树，然后对 $X_{\text{near}} \leftarrow \mathrm{Near}(G, x_{\text{new}}, \min\{\gamma_{\mathrm{RRG}}(\log(\mathrm{card}(V))/\mathrm{card}(V))^{1/d}, \eta\})$ 内每个 $x_{\text{near}}$ 都连边（双向，允许环）。$\gamma_{\mathrm{RRG}} > \gamma^*_{\mathrm{RRG}}$，数值同 PRM*。

**RRT\***（Algorithm 6，在 RRG 骨架上插入两段；论文未用 "ChooseParent/Rewire" 函数名，两段以内嵌注释标出——本精读按注释原文标注）：

```text
1  V ← {x_init};  E ← ∅
2  for i = 1 … n:
3    x_rand  ← SampleFree
4    x_nearest ← Nearest(G, x_rand)
5    x_new   ← Steer(x_nearest, x_rand)
6    if ObstacleFree(x_nearest, x_new):
7      X_near ← Near(G, x_new, min{γ_RRT*(log(card V)/card V)^{1/d}, η})
8      V ← V ∪ {x_new}
9      x_min ← x_nearest;  c_min ← Cost(x_nearest) + c(Line(x_nearest, x_new))
10     foreach x_near ∈ X_near:                              # "Connect along a minimum-cost path"（选父）
11       if CollisionFree(x_near, x_new) ∧ Cost(x_near) + c(Line(x_near, x_new)) < c_min:
12         x_min ← x_near;  c_min ← Cost(x_near) + c(Line(x_near, x_new))
13     E ← E ∪ {(x_min, x_new)}
14     foreach x_near ∈ X_near:                              # "Rewire the tree"（重布线）
15       if CollisionFree(x_new, x_near) ∧ Cost(x_new) + c(Line(x_new, x_near)) < Cost(x_near):
16         E ← (E \ {(Parent(x_near), x_near)}) ∪ {(x_new, x_near)}
```

支撑函数（§3.3，Algorithm 6 前一段）：$\mathrm{Line}(x_1, x_2): [0,s] \to \mathcal{X}$ 为直线段路径；$\mathrm{Cost}(v)$ = 根到 $v$ 唯一路径的代价，**加性假设**下

$$
\mathrm{Cost}(v) = \mathrm{Cost}(\mathrm{Parent}(v)) + c\big(\mathrm{Line}(\mathrm{Parent}(v), v)\big), \qquad \mathrm{Cost}(v_0) = 0
\tag{S3}
$$

（原文照录；即教程第 03 章式 (3.5)。）逐行解读：第 9–13 行**选父**——邻域内每个能无碰撞连到 $x_{\text{new}}$ 的候选都参与比价，只留"经它到达 $x_{\text{new}}$ 总代价最小"的一条边（RRT 在此只有固定候选 $x_{\text{nearest}}$，无比价）；第 14–16 行**重布线**——$x_{\text{new}}$ 反向检查邻域旧点，若"经新点到达更便宜"则改接父边（依据：第 15 行严格不等号保证 (S3) 下全树代价单调不降不增，见 4.3 (S4)）。$k$-近邻变体：$k = k_{\mathrm{ALG}} \log n$，常数 $k^*_{\mathrm{PRM}} = k^*_{\mathrm{RRG}} > e(1+1/d)$、$k_{\mathrm{RRT}^*} > 2^{d+1} e(1+1/d)$（§3.3 与 Thm 35/37/39 原文照录）。

### 4.3 RRT 的次优性（Theorem 33；附录 B 骨架）

附录 B 的简化设定（原文声明）：无障碍 $\mathcal{X}_{\text{free}} = (0,1)^d$、$\eta \ge \mathrm{diam} = \sqrt{d}$——"证明 RRT 非渐近最优只需这一个实例"，推广到一般情形只是技术繁琐。

- **(S4) 极限存在。** RRT 每轮"加一顶点一边或不动"，故 $G_i^{\mathrm{RRT}} \subseteq G_n^{\mathrm{RRT}}\ (i \le n)$，$Y_n$ 沿树单调不增、有下界 $0$（依据：单调收敛；Lemma 26 的单调性性质对 RRT 成立）⇒ $\lim_n Y_n^{\mathrm{RRT}} = Y_\infty^{\mathrm{RRT}}$ 几乎必然存在。问题只剩：$Y_\infty$ 是否等于 $c^*$？
- **(S5) 必要条件（Lemma 44）。** 取 $0 < R < \inf_{y \in \mathcal{X}_{\text{goal}}} \lVert y - x_{\text{init}} \rVert$。若 $\lim Y_n = c^*$，则树的**无穷多个分支**（根的第 $k$ 个孩子及其全部后代）必须伸出球 $B(x_{\text{init}}, R)$。依据链：经第 $k$ 个孩子恰好取到最优代价的概率为 0（Assumption 27 + Lemma 28，$\Gamma(x_k) = c^*$ 零测）；若只有前 $K$ 个分支出球，则树内最优代价 $\ge \sup\{\Gamma(x_k) : k \le K,\ \Gamma(x_k) > c^*\} > c^*$——于是 $\{\lim Y_n = c^*\} \subseteq \{\text{无穷多分支出球}\}$。
- **(S6) 分支长度几乎必然有限（Lemma 45–48, Cor 49）。** Lemma 45：$n$ 个均匀样本外独立再采一点，它以概率 $1/n$ 恰以某个 $X_i$ 为最近邻（对称性），到最近邻的期望距离 $n^{-1/d}$（均匀顺序统计量）。由此定义第 $k$ 分支的"首段长度"$\mathcal{L}_k$，Lemma 46 算出

$$
\mathbb{E}[\mathcal{L}_1] = \sum_{i=1}^{\infty} i^{-1 - 1/d} = \mathrm{zeta}(1 + 1/d) < \infty
\tag{S7}
$$

（附录 B 原式照录；依据：单调收敛定理交换求和序——每项 $i^{-1-1/d}$ 是第 $i$ 个样本作为分支首点贡献的期望距离。）$\mathbb{E}[\mathcal{L}_k]$ 单调不增且 $\to 0$；Lemma 47 用马尔可夫型论证 $\mathbb{P}(\sup_{\alpha \ge k} \mathcal{L}_\alpha > \epsilon) \le \mathbb{E}[\mathcal{L}_k]/\epsilon$（依据：$\mathbb{E}[\mathcal{L}_k] \ge \mathbb{E}[\mathcal{L}_{\bar\alpha} \mathbf{1}_{S_\epsilon}] \ge \epsilon\, \mathbb{P}(S_\epsilon)$）；Cor 49 得 $\sup_{\alpha \ge k}\mathcal{L}_\alpha \to 0$ 依概率——**每条分支的极限长度几乎必然有限，且对分支序号一致有界**。
- **(S7) 合拢。** 取 $\epsilon = R$：无穷多分支伸出 $R$-球的概率为 0（依据：每个伸出球的分支长度必 $> R$，与 (S6) 冲突），由 (S5) 的包含关系 $\mathbb{P}(\{\lim Y_n = c^*\}) = 0$；结合 (S4) 极限存在与 Lemma 25 的 0-1 律：

$$
\mathbb{P}\big(\lim_{n \to \infty} Y_n^{\mathrm{RRT}} > c^*\big) = 1
\tag{S8}
$$

（Theorem 33 + 第 28 页推论段原文结论。）**一句话机理**：RRT 的父边一经写下永不修改，第一条伸进目标区的次优路径把树拓扑锁死；多跑几次 = 对随机变量 $Y_\infty^{\mathrm{RRT}}$ 抽样（第 28 页原话），这是"anytime RRT/多次重跑"类方法（Ferguson & Stentz）有效的解释。RRT* 的两条新规则恰好反向拆除该机理：重布线允许改父（拓扑解锁），选父保证新点永远走最便宜入口（代价解锁）。

### 4.4 渐近最优性（Theorem 38；附录 C/G 骨架，收缩球半径的折中推导）

**待证**（Thm 38 原文）：若 $\gamma_{\mathrm{RRT}^*} > \big(2(1 + 1/d)\big)^{1/d}\big(\mu(\mathcal{X}_{\text{free}})/\zeta_d\big)^{1/d}$，则 RRT* 渐近最优。半径

$$
r(n) = \gamma_{\mathrm{RRT}^*} \Big(\frac{\log n}{n}\Big)^{1/d}
\tag{S9}
$$

（Algorithm 6 第 7 行的 $\min\{\cdot, \eta\}$ 内项。）证明分两半，对应半径的两个约束方向——**这正是"收缩球必须恰好卡在 $(\log n / n)^{1/d}$"的两面**：

**(a) 覆盖论据：半径不能缩太快（$\log n$ 因子的去处）。** 四步，全部有原文落点：

1. **clearance 逼近**：robustly optimal 的 $\sigma^*$ 有 weak clearance，Lemma 50 保证存在强 $\delta_n$-clearance 路径序列 $\sigma_n \to \sigma^*$，其中（附录 C.2 原式）$\delta_n = \min\{\delta, \tfrac{1+\theta_1}{2+\theta_1} r_n\}$（依据：闭集 $\mathcal{X}_n = \mathrm{cl}(\mathrm{int}_{\delta_n}(\mathcal{X}_{\text{free}}))$ 递增穷竭自由空间，同伦沿 $\alpha$ 最大推进）。
2. **覆盖球链**：Definition 51 的 $\mathrm{CoveringBalls}(\sigma_n, q_n, \theta_1 q_n)$，$q_n = \delta_n/(1+\theta_1)$——球心沿 $\sigma_n$ 间隔恰 $\theta_1 q_n$ 的等半径球串。
3. **球内样本几何**（本精读补的中间步，依据：两点距离的三角不等式）：任取相邻两球中各一点 $x_m \in B_{n,m}$、$x_{m+1} \in B_{n,m+1}$，
   $\lVert x_{m+1} - x_m \rVert \le \underbrace{q_n}_{x_m \to 球心} + \underbrace{\theta_1 q_n}_{球心 \to 球心} + \underbrace{q_n}_{球心 \to x_{m+1}} = (2 + \theta_1)\, q_n = \frac{(2+\theta_1)\,\delta_n}{1 + \theta_1} \le r_n$
   （最后一步依据：$\delta_n \le \tfrac{1+\theta_1}{2+\theta_1} r_n$ 恒成立——$\min$ 不超过任一支）。即 **Lemma 53 的结论**：相邻球内的样本对天然落在互连半径内；Lemma 54 进一步保证连线段整段无碰撞（依据：两球都在路径的 $\delta_n$-clearance 缓冲内）。这一步解释 $\theta_1$ 的设计：它把"覆盖"与"互连"两个半径需求锁进同一个 $q_n$。
4. **尾概率**：单球体积 $\zeta_d q_n^d$，球内期望样本数（依据：样本独立同分布、均匀于 $\mathcal{X}_{\text{free}}$；$n \ge n_0$ 后 $\delta_n$ 取 min 的第二支，$q_n = r_n/(2+\theta_1)$）

$$
n \cdot \frac{\zeta_d\, q_n^d}{\mu(\mathcal{X}_{\text{free}})} \;=\; \frac{\zeta_d\, \gamma^d}{(2 + \theta_1)^d\, \mu(\mathcal{X}_{\text{free}})} \,\log n \;\xrightarrow{n \to \infty}\; \infty
\tag{S10}
$$

（代入 $q_n^d = r_n^d/(2+\theta_1)^d = \gamma^d \log n / \big(n (2+\theta_1)^d\big)$——这是 $\log n$ 因子的物理意义：**每球样本数随 $n$ 对数增长，而不是停在常数**。）单球为空的概率 $\le (1 - \zeta_d q_n^d/\mu)^{n} \le \exp(-c\,\log n) = n^{-c}$（依据：$(1-x)^n \le e^{-nx}$，$c$ 为依赖 $\gamma, \theta_1, d$ 的正常数），球数 $M_n \asymp c(\sigma_n)/(\theta_1 q_n) = O((n/\log n)^{1/d})$ 的并集界仍 $\to 0$——**whp 每球非空、相邻球有互连对**，于是树内存在一条穿球链，其代价 $\to c(\sigma^*)$（链长与 $\sigma_n$ 的偏差由 BV 收敛引理，Lemma 55/61，控制到零）。

**(b) 连通性论据：$\gamma$ 常数不能太小（下界的去处）。** "每球非空"不够——必须在**相邻球对的相对两侧**各拿到样本才能跨越（Fig. 8 的教训：外球空则没有边穿过内球）。所以 (S10) 的常数也必须够大：$\gamma^d \zeta_d' \log n$ 的系数 $\gamma$ 低于阈值时，"球对内存在可连样本对"的失效概率不再被并集界吸收，球链以不为零的概率断链，树收不到 $c^*$。论文的严格版本（附录 G.1）：引入**标记点过程**——每个样本带出生序号，边只在"序号靠前 → 靠后"且距离 $\le r_n$ 的点对间生成，精确复现 RRT* 的树结构（批量 PRM* 无此约束，故先证 Thm 34 再迁移到 RRT*）；Lemma 71 先给出中间界 $\gamma_{\mathrm{RRT}^*} > 4\,(\mu(\mathcal{X}_{\text{free}})/\zeta_d)^{1/d}$（原式照录：$\gamma_{\mathrm{RRT}^*} > 4\big(\mu/\zeta_d\big)^{1/d} \Rightarrow \mathbb{P}(\liminf A_n) = 1$），最终常数 $\big(2(1+1/d)\big)^{1/d}$ 由附录中更细的两尺度计数收紧——**(教程第 03 章式 (3.7) 的 $\gamma_{\mathrm{RRT}^*}$ 下界即出自此处)**。

**(c) 两个方向合拢。** 半径快于 $n^{-1/d}$（无 $\log n$）⇒ (S10) 停在常数 ⇒ 覆盖失败 ⇒ 非渐近最优（Theorem 20/32 对变半径 $r = \gamma n^{-1/d}$ sPRM 的否定结论）；半径慢于 $(\log n/n)^{1/d}$ ⇒ 每轮碰撞检测与近邻调用不降（见 4.5）且长边代价与真实弧长的 BV 偏差不消失。**收缩速率 $(\log n/n)^{1/d}$ 是覆盖概率与计算开销的唯一平衡点。**

### 4.5 复杂度（§4.3，Lemma 40–43）

- **每轮碰撞检测调用数 $M_n$**：PRM 与 sPRM $\Omega(n)$（Lemma 40/41：半径 $r$ 固定，球内期望 $\propto n$）；RRT 恒为 1；**PRM*/RRG/RRT* 为 $O(\log n)$**（Lemma 42）。证明核心一步（第 30 页原式）：样本落在 $r_n$-内部时

$$
\mathbb{E}\big[M_n^{\mathrm{PRM}^*} \mid A\big] = \frac{\zeta_d\, \gamma_{\mathrm{PRM}}}{\mu\big(\mathrm{int}_{r_n}(\mathcal{X}_{\text{free}})\big)} \log n
\tag{S11}
$$

（依据：球体积 $\zeta_d r_n^d$ 乘以样本密度 $n/\mu$，$r_n^d = \gamma^d \log n / n$——$n$ 与 $\log n/n$ 相消，每轮调用数只剩 $\log n$。）这就是"渐近最优只比概率完备贵常数倍"的精确含义（第 35 页：constant factor increase）。
- **边数与空间**：PRM*/RRG $O(n \log n)$、RRT/RRT* $O(n)$（第 33 页，依据：边数 ≤ 碰撞检测调用数累加）。
- **查询阶段**：从返回图提取最优解，Lemma 43（引 Schrijver）：最短路树 $O(|V|\log|V| + |E|)$；RRT* 是树，回溯即得、$O(n)$（Table 1 的 Query 列）。

## 5. 实验与结果解读（定性表述；维度以论文实况为准，见题注）

**实验设置**（§5 开头）：C 实现，2.66 GHz / 4 GB RAM 单机；代价默认取 $\mathrm{TV}(\sigma)$（即路径长度）。**维度实况与常见转述不同：论文实验覆盖 2、3、4、5 维（PRM*）与 2、5、10 维（RRT*），并无 "R2–R8"**（任务规格此处与原文不符，本节按原文撰写；RRG 无专门实验图，§5 明说实验主体是 RRT* 对 RRT）。

- **PRM* vs k-近邻 PRM（2D，无障碍，Fig. 10）**：$k = 5,7,10,13,15$ 的 $k$-近邻 PRM 全部停在 $1.01\text{–}1.10 \times$ 最优的平台上，且 $k$ 越大平台越低但不消失（结构性次优，呼应 Thm 31）；PRM* 的代价随迭代降到归一化最优值 1。
- **PRM* 跨维度（2/3/4/5 维，Fig. 11）**：单位立方体、中心体积 0.5 的立方障碍、起终点在对角：代价曲线随样本数单调下降，维度越高同代价水平所需样本越多（定性：高维收敛变慢但不改变单调趋优的形态）。
- **RRT* vs RRT（2D，同采样序列，Fig. 12）**：两者顶点集完全相同、只有边不同——250/500/2500/10000/20000 顶点快照里，RRT 的解反复更换同伦类且绕远，RRT* 的解在重布线下持续变直。
- **Monte-Carlo（2D，500 次 × 20000 迭代，Fig. 13/16）**：无障碍时 RRT 的平均代价收敛到约 $\sqrt{2} \times$ 最优（与确定性版本结论一致，LaValle & Kuffner 2009——即 [RRT 精读](RRT_TR1998.md) §5 的 1.3–2.0 倍的极限形态），方差持续可观；RRT* 收敛到最优、方差 $\to 0$。加障碍后 RRT 平均约 $1.5 \times$ 最优：树撞进两条同伦类，走对类的跑赢、走错类的约 $2 \times$——Thm 33 的"次优随机变量"在直方图上肉眼可见。
- **变代价场（Fig. 17）**：把代价场设为高/低/普通三区（2、1/2、1），RRT* 的树表现出与光折射一致的 Snell–Descartes 行为（论文引 Rowe & Alexander 2000 的路径规划版）——抽象代价框架 (4.1) 直接兑现为"非欧代价下的最短路径"。
- **运行时间与高维**（Fig. 18/19/20–22）：RRT*/RRT 运行时间比随迭代收敛到常数倍（定性，有障碍环境比无障碍略高）；5 维（100 次试验）与 10 维实验里 RRT 代价平台与 RRT* 单调趋优的对照保持不变——渐近最优性在 10 维仍有可见收益。

## 6. 局限与后续影响

- **渐近≠快**：Thm 38 只说 $n \to \infty$，不给出有限样本收敛率；实际中 RRT* 常在首解质量上反而吃亏（首解阶段邻域小、重布线收益未兑现）——这直接催生了后续工作。
- **常数依赖未知量**：$\gamma$ 下界含 $\mu(\mathcal{X}_{\text{free}})$（自由空间测度，实践中未知，实现只能取保守值）；$k$-近邻 RRT* 的常数 $2^{d+1}e(1+1/d)$ 随维度指数增长，高维形同虚设；$\theta$ 系数为证明技术参数，工程副本各有取法。
- **论文自留的开放问题**（第 36 页）：渗流 ⟺ 概率完备、连通 ⟺ 渐近最优的猜想未证；EST 等未分析；确定性采样序列未分析。
- **后续谱系（一段）**：Informed-RRT*（Gammell et al. 2014）用已知 $c^*$ 下界的椭圆域聚焦采样，加速收敛但不改渐近性质；BIT*（Batch Informed Trees, Gammell et al. 2015）把"批量 + 启发式排序"结合两族优点；再往后抽样复杂度精化（如 RRT*-Smart、Lean-RRT* 等工程变体）与学习式采样（MPNet，[第 09 章](../09_前沿学习式规划与腿式控制.md)）——共同点是都保留 RRT* 作完备性/最优性兜底，只替换"采样从哪来"这个组件。
- **方法论遗产**：随机几何图视角渗入后续一切采样式规划分析（渗流阈值、连通常数），"把规划器识别为随机图"成为标准动作。

## 7. 与本项目对照

- **教程映射（第 03 章 §03.2，编号回引以教程实际编号为准）**：(S3) ↔ 教程 (3.5)（代价递归）；重布线单调性（第 15 行严格不等号 + (S3) 递归传播）↔ 教程 (3.6)；(S9) 与 $\gamma$ 下界 ↔ 教程 (3.7)（教程注明"常数取自论文 Theorem 38"，已核一致）；(S10) 的 $\log n$ 因子 ↔ 教程 (3.8)；单球空概率尾界 ↔ 教程 (3.9)。教程"同族关系"段引用的 Thm 34/36/38（半径版）与 Thm 35/37/39（k-近邻版）、Thm 29（固定半径 PRM 非最优）、Thm 23（概率完备）、Lemma 42（碰撞检测 $O(\log n)$）、附录 G（Thm 38 证明）——全部与本地 PDF 核对一致（Theorem 36 的 arXiv v1 原文把 $\gamma$ 误印作 $\gamma_{\mathrm{PRM}}$，实指 $\gamma_{\mathrm{RRG}}$，常数值相同）。
- **与 RRT 精读的关系**：[RRT_TR1998.md](RRT_TR1998.md) 的 (R4)–(R6) 是本文 Thm 33/38 的"被证明对象"——1998 年的"概率完备（声称）+ 不远离最优（经验）"在 2011 年被精确化为"概率完备（Thm 16，指数速率）+ 结构性次优（Thm 33）+ 修两行伪代码即渐近最优（Thm 38）"。两份精读合起来是教程第 03 章 03.1–03.2 的论文侧完整闭环。
- **MPNet 的学习偏置如何继承 RRT* 兜底**（[第 09 章](../09_前沿学习式规划与腿式控制.md) §09.1）：MPNet 用神经采样器 $q_{t+1} \sim p_\theta(\cdot \mid q_t, q_{goal}, \mathbf{Z})$（教程 (9.1)）替换本文的均匀 SampleFree——这恰好落在本文框架允许的自由度内（本文分析只假设样本独立同分布于支撑集；神经采样破坏均匀性后 Thm 38 不再直接适用，故 MPNet 不声称神经阶段最优）。其混合模式把神经阶段失败的间隙逐个交回 RRT*（教程 (9.4) 的并集界继承论证）：**每个间隙子问题 robustly feasible ⇒ RRT* 概率完备逐个桥接 ⇒ 全局完备性继承自 RRT***。换言之：学习偏置接管本文 §5 实验里的"收敛速度"，RRT* 的收缩球机制（(S9)）接管"收敛终点"的保证——教程 09.3 表格 L2 行"神经先验 + RRT* 回退"的理论依据即在此。
- **2D 教学实现**：`projects/control_planning/` 尚未建立（与 [RRT 精读](RRT_TR1998.md) §7 同一条备注）——RRT* 的实现增量极小：(R4) 的 steer 不变，只加 Algorithm 6 第 7–16 行的邻域两循环，适合作为该项目的第一个对照实验（RRT vs RRT* 复现 Fig. 13 的曲线形态）。

## 配套阅读

- 同组姊妹篇：[RRT 精读（LaValle, TR 98-11）](RRT_TR1998.md)——被本文"判次优"的对象，其 Voronoi 偏置与增量碰撞检测思想被 RRT* 原样继承。
- 教程：[第 03 章｜采样式规划](../03_运动规划-i采样式规划.md)（(3.5)–(3.9) 的推导即本文 §4.3–4.4 的浓缩版）｜ [第 09 章｜学习式规划](../09_前沿学习式规划与腿式控制.md)（(9.1)–(9.4)：神经采样偏置 + RRT* 完备性继承）。
- 论文延伸：Karaman & Frazzoli, ICRA 2010（会议先行版）；Gammell, Srinivasa, Barfoot, *Informed RRT*\*, IROS 2014 与 *BIT*\*, ICRA 2015（论文库未收录）；Qureshi et al., *MPNet*, ICRA 2019（本地 [papers/control_planning/frontier/arXiv-1806.05767_MPNet.pdf](../../../papers/control_planning/frontier/arXiv-1806.05767_MPNet.pdf)）。
- 主题导航：[tutorials/control_planning/README.md](../README.md) ｜ 论文索引：[papers/control_planning/README.md](../../../papers/control_planning/README.md)

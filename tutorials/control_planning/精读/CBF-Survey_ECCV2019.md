# 论文精读｜CBF 综述：控制屏障函数理论与应用（ECC 2019）

> **PDF**：[papers/control_planning/classics/arXiv-1903.11199_CBF-Survey.pdf](../../../papers/control_planning/classics/arXiv-1903.11199_CBF-Survey.pdf)（arXiv:1903.11199v1，共 12 页） ｜ **教程**：[第 07 章](../07_安全控制控制屏障函数.md)（整章以本综述为主线） ｜ **代码**：综述无官方统一实现——SOS 计算路线见论文引文 [46] SOSTOOLS（www.cds.caltech.edu/sostools），QP 侧以社区开源 CBF-QP 求解库为主 ｜ **前置**：[RRT 精读](RRT_TR1998.md)、[RRT* 精读](RRTstar_IJRR2011.md)（同目录）

## 1. 论文信息与一句话贡献

- **题目**：Control Barrier Functions: Theory and Applications
- **作者**：Aaron D. Ames、Samuel Coogan、Magnus Egerstedt、Germaro Notomista、Koushil Sreenath、Paulo Tabuada（Caltech / Georgia Tech / UC Berkeley / UCLA）
- **发表**：European Control Conference（ECC）2019；本仓库为 arXiv:1903.11199v1（2019-03-27）
- **编号约定（重要）**：正文编号公式为 (1)–(26)，**三个 QP（CBF-QP、CLF-CBF QP、CLF-ECBF QP）与 Nagumo 条件均为无编号展示式**，按名称与小节位置引用；本精读自加步骤编号 **(S1)–(S12)** 并逐条声明对应原文位置。
- **一句话贡献**：把"安全"形式化为**集合不变性**（set invariance），把 Nagumo 边界条件经 class-K 放宽成对控制输入的**逐点仿射约束**——零化控制屏障函数（zeroing control barrier function, CBF）——使安全-临界控制归结为每个控制周期解一个小型二次规划（CBF-QP / CLF-CBF QP，安全性硬约束、稳定性软目标），并给出从执行约束（§III）、高相对阶指数 CBF（§IV）到步行/汽车/平衡/多机器人（§V）的完整应用图谱。

## 2. 问题与动机

**要解决什么**（§I）：任何工程系统都应"安全"（safe），但"安全"需要定义、且需要能设计。Lyapunov 函数主导了稳定性（活性，liveness——"好事情终将发生"）研究六十余年；安全（safety——"坏事情永不发生"）却缺乏同等成熟的控制设计理论。论文在 §I 开篇给出两点时代动因：(i) 自主系统要在未知非结构化环境中运行，安全性质的强制执行变得远比以前困难；(ii) CBF 的近期出现表明，基于 Lyapunov 的控制设计技术可以**整体迁移**到安全考虑上——安全应当被提升到与活性同等的理论成熟度。

**四条历史线索**（§I.A）——论文把现代 CBF 定位为一路收敛的终点：

1. **Nagumo（1942，引文 [4]）**：对 $\dot x = f(x)$、安全集取光滑函数 $h$ 的上水平集且边界正则（$\partial h/\partial x \ne 0$ 于 $\{h=0\}$），集合不变性的充要条件由 $h$ 在边界上的导数给出（见 §4.1，无编号式）。
2. **屏障证书（barrier certificate，2000 年代，引文 [9][10]）**：为验证混合系统而生——找 $B$ 使 $B(x)\le 0$ 于初始集、$B(x)>0$ 于不安全集，则"$\dot B(x) \le 0 \Rightarrow \mathcal{C}$ 不变"。取 $B = -h$ 时该条件恰退化为边界上的 Nagumo 条件。
3. **控制屏障函数（引文 [20]，2014）**：把条件"$\exists\,u$ s.t. $\dot h(x,u) \ge -\alpha(h(x))$"当作**安全 certificates**——首次把不变性条件从闭环推广到控制输入逐点成立。
4. **现代形式（引文 [21]，IEEE TAC 2017）**：条件在整个安全集 $\mathcal{C}$ 上成立（不再只在边界），合优性（safety）与稳定性（stability）可写进**同一个优化问题**。

**安全性与稳定性的对偶**（§I.B）：CLF 之于稳定，正如 CBF 之于安全——论文的贡献之一是指出这两类 certificates 结构相同（都是"对 $u$ 的仿射不等式"），因而可以在一个 QP 里统一仲裁。

## 3. 方法总览

**系统设定**（§II 开头，式 (1)）：全程假设非线性**控制仿射系统**（control-affine system）

$$
\dot x = f(x) + g(x)\,u, \tag{1}
$$

$f, g$ 局部 Lipschitz，$x \in D \subseteq \mathbb{R}^n$，$u \in U \subseteq \mathbb{R}^m$。

**主线结构**（§II→§V 的逻辑链）：

```
安全集 C = {h ≥ 0} (5) ──Nagumo──▶ 边界条件 ḣ ≥ 0（§I.A，无编号）
      │ class-K 放宽（α 扩展 class-K∞）
      ▼
CBF 定义（Def 2 + 式 (7)）：sup_u [L_f h + L_g h·u] ≥ −α(h(x))  于全 D
      ▼ K_cbf(x) (10)：安全控制集（u 的半空间）
      ▼
CBF-QP（§II.C，无编号）：min ‖u − k(x)‖²  s.t. CBF 约束  ──KKT──▶ min-norm 闭式解
      ▼ 并入 CLF 条件 (3)
CLF-CBF QP（§II.C，无编号）：安全硬约束 + 稳定带松弛 δ，Lipschitz 连续（[21]）
      ▼ 相对阶 > 1 时
指数 CBF（§IV，式 (15)–(19)）：对 h 求 r−1 次导恢复仿射性，极点配置设计 K_α
      ▼
应用（§V）：步行踩石 / 自适应巡航+车道保持 / Segway 安全滤波 / 多机器人长时自主
```

两大定理定调（§II.B）：**Theorem 2**（充分性）——若 $h$ 是 CBF 且边界正则，任何取值于 $K_{cbf}(x)$ 的 Lipschitz 控制器使 $\mathcal{C}$ 安全（前向不变），且 $\mathcal{C}$ 渐近稳定；**Theorem 3**（必要性）——合理假设下，凡能使 $\mathcal{C}$ 安全的控制器，都意味着 $h$ 在 $\mathcal{C}$ 上是 CBF：CBF 条件是"最强可能的"安全条件。

## 4. 关键公式推导（式号按原文；步骤编号 (S1)–(S12) 为本精读自加）

### 4.0 符号表

| 符号 | 含义（对应原文位置） |
|---|---|
| $\dot x = f(x)+g(x)u$ | 控制仿射系统（式 (1)），$f,g$ 局部 Lipschitz，$U \subseteq \mathbb{R}^m$ |
| $h: D \to \mathbb{R}$ | 连续可微屏障函数；安全集取其上水平集（式 (5)） |
| $\mathcal{C},\ \partial\mathcal{C},\ \mathrm{Int}(\mathcal{C})$ | $\{h \ge 0\}$ / $\{h = 0\}$ / $\{h > 0\}$（式 (5)） |
| $\alpha: \mathbb{R} \to \mathbb{R}$ | **扩展 class-$K_\infty$ 函数**：严格递增且 $\alpha(0)=0$，定义在全实轴（§II.B） |
| $L_f h = \frac{\partial h}{\partial x}f,\ \ L_g h = \frac{\partial h}{\partial x}g$ | 李导数（Lie derivative；定义见 §4.3） |
| $K_{cbf}(x),\ K_{clf}(x)$ | 使 $\mathcal{C}$ 安全的控制值集（式 (10)）/ 使 CLF 下降的控制值集（式 (4)） |
| $V$ | 控制李雅普诺夫函数（CLF，§II.A） |
| $\gamma,\ \delta$ | CLF 条件中的 class-K 函数 / CLF-CBF QP 的松弛变量（注意：松弛是 $\delta$，不是 $\gamma$） |
| $r,\ \eta_h,\ K_\alpha$ | 相对阶 / 导数链向量（式 (16)）/ ECBF 增益行向量（Def 7） |

### 4.1 集合不变性与 Nagumo 条件（§I.A；教程 (7.1)–(7.3)）

**(S1) 安全集与不变性。** 取连续可微 $h$，$\mathcal{C} = \{x : h(x)\ge 0\}$、$\partial\mathcal{C} = \{h = 0\}$、$\mathrm{Int}(\mathcal{C}) = \{h>0\}$（式 (5)；对应教程 (7.1)）。闭环 $\dot x = f_{cl}(x) = f(x)+g(x)k(x)$（式 (6)）下，**前向不变**（Definition 1，即"安全"；对应教程 (7.2)）：$x(0)\in\mathcal{C} \Rightarrow x(t)\in\mathcal{C}$ 对 $t \in [0,\tau_{max})$。

**(S2) Nagumo 条件（原文 §I.A，无编号展示式）。** 设 $\frac{\partial h}{\partial x}(x) \ne 0$ 对所有 $h(x)=0$ 的 $x$，则

$$
\mathcal{C} \text{ 是不变的} \iff \dot h(x) \ge 0 \quad \forall\, x \in \partial\mathcal{C}.
$$

（依据：论文引 Nagumo 1942 [4]；重发现史注 Bony/Brezis [7][8]。）

**(S3) 光滑情形论证**（教程 07.1 ⑤ 同款，论文用此条件证明 Theorem 2 并在 Remark 5 注明需要正则性）：沿解 $x(t)$ 记 $g(t)=h(x(t))$，链式法则（依据：$h\in C^1$、$x(t)$ 可微）给 $\dot g(t) = \nabla h(x(t))\cdot f_{cl}(x(t))$。若轨迹在 $t^*$ 触边（$g(t^*)=0$）且 $\dot g(t^*)>0$，由 $\dot g$ 连续，存在 $\delta>0$ 使 $g>0$ 于 $[t^*,t^*+\delta]$——穿不到负侧；若边界上处处 $\dot g \ge 0$，每次触边都被"向外压"，负值区域不可达。相切情形（$\dot g(t^*)=0$）单凭一阶论证排除不掉，完整结论需切锥与闭集分析——论文的 Theorem 2 证明正是引用 Nagumo 定理，正则性 $\nabla h \ne 0$ 于边界是该定理的前提（Remark 5）。

### 4.2 零化 CBF：定义，以及为何优于经典 barrier certificate（Def 2 + 式 (7)；Remark 3/4）

**(S4) 定义（Definition 2，条件为式 (7)）。** $h$ 是**控制屏障函数**，若存在扩展 class-$K_\infty$ 函数 $\alpha$ 使

$$
\sup_{u\in U}\big[\, L_f h(x) + L_g h(x)\,u \,\big] \ \ge\ -\alpha\big(h(x)\big) \tag{7}
$$

对所有 $x \in D$ 成立。（对应教程 (7.4) 的存在性形式。注意称谓：式 (7) 即后续文献所称 **zeroing CBF** 的现代形式，"零化"得名于 $h=0$ 时右端 $-\alpha(0)=0$、约束退化为 Nagumo 的 $\dot h \ge 0$；本综述正文未使用 "zeroing" 一词。）

**(S5) 对比一：经典 barrier certificate 过强。** §I.A 的屏障证书取 $B=-h$ 后条件为 $\dot h(x) \ge 0$——但它按原文要求在（相应集合的）**全部状态**上成立。后果：$\dot h \ge 0$ 排除了"以正速率衰减、正常地接近边界"的一切合法行为——在边界附近轨迹只能相切或远离，过度保守。Remark 4 指出把不变条件从边界扩展到全 $\mathcal{C}$ 的想法源自 Aubin 可行性理论的特例 $\dot h \ge -h$（引文 [14]），这正是式 (7) 取 $\alpha(r)=r$ 的情形。

**(S6) 对比二：倒数型 barrier 的"边界爆破"与"出集失效"。** 早期另一支路线（Remark 3，式 (8)–(9)）用**倒数屏障**（reciprocal barrier）$B$：在 $\mathrm{Int}(\mathcal{C})$ 上 $B\ge 0$ 且 $B \to \infty$ 当 $x\to\partial\mathcal{C}$（"barrier"一词的直义，式 (8)），CBF 条件变为

$$
\inf_{u\in U}\big[L_f B(x) + L_g B(x)\,u\big] \ \le\ \alpha\Big(\frac{1}{B(x)}\Big). \tag{9}
$$

两处结构性缺陷（依据均出自论文原文）：
- **近边界自锁**：$x\to\partial\mathcal{C}$ 时 $1/B\to 0^+$，$\alpha(1/B)\to 0$，约束 (9) 收紧为 $\dot B \le 0$ 的极限——越靠近边界可行控制越被挤压（论文说这类函数 "can be more suitable for some applications"，即承认其适用面更窄）；
- **出集后失效**：$B$ 经 $1/B$ 依赖，在 $\mathcal{C}$ 之外（$B\le 0$）条件失去意义。论文原话（Remark 3）：**"typically barrier functions $h$ are preferable since they are well defined outside of $\mathcal{C}$"**——任务书所称的 "relinquishing 行为"（一旦出集就放弃安全）即指此性质；本综述未用 "relinquish" 字样，机制以上述原文呈现。

**(S7) zeroing 的回拉机制。** 式 (7) 对 $h$ 在全 $D$ 上有意义（$h$ 处处有定义），按 $h$ 的符号分三段读（依据：$\alpha$ 严格递增、$\alpha(0)=0$）：
- $h>0$（集内）：$-\alpha(h)<0$，**允许** $\dot h < 0$——可向边界衰减，速率不得快于 $-\alpha(h)$；
- $h=0$（边界）：约束退化为 $\dot h \ge 0$——恰为 Nagumo（"零化"名称来源）；
- $h<0$（集外）：$-\alpha(h)>0$，式 (7) **强制**存在使 $\dot h>0$ 的控制——把状态推回 $\mathcal{C}$（回拉）。
取线性 $\alpha(r)=\gamma r$ 时闭式回拉率为 $h(t) \ge h(0)e^{-\gamma t}$（教程 (7.5) 的积分因子论证）。配合 Remark 6：Theorem 2 附带 $\mathcal{C}$ 渐近稳定——噪声/模型误差把系统推出 $\mathcal{C}$ 时，取值于 $K_{cbf}$ 的控制器会把它拉回来。

### 4.3 控制仿射系统下的线性化：$L_f h + L_g h\,u$（式 (1)、(7)、(10)）

**(S8) 李导数逐项定义与展开。** 对 $C^1$ 函数 $h$ 与向量场 $f,g$（依据：李导数 = $h$ 沿向量场的方向导数）：
$$
L_f h(x) := \frac{\partial h}{\partial x}(x)\, f(x) \in \mathbb{R}, \qquad
L_g h(x) := \frac{\partial h}{\partial x}(x)\, g(x) \in \mathbb{R}^{1\times m}.
$$
沿式 (1)，链式法则给（对应教程 (7.7)）：
$$
\dot h(x,u) = \frac{\partial h}{\partial x}\,\dot x = \underbrace{\frac{\partial h}{\partial x} f(x)}_{L_f h(x)} + \underbrace{\frac{\partial h}{\partial x} g(x)}_{L_g h(x)}\, u \ =\ L_f h(x) + L_g h(x)\,u.
$$
关键后果（§II.C 开头）：$\dot h$ 是 $u$ 的**仿射函数**，于是"存在 $u$ 使式 (7) 成立"⟺"$\sup_{u}(\cdot) \ge -\alpha(h)$"⟺ 安全控制集是半空间（式 (10)，对应教程 (7.8)）：
$$
K_{cbf}(x) = \big\{\, u \in U \ :\ L_f h(x) + L_g h(x)\,u + \alpha(h(x)) \ \ge\ 0 \,\big\}. \tag{10}
$$
**Theorem 2**（原文）：$h$ 是 CBF 且 $\frac{\partial h}{\partial x}\ne 0$ 于 $\partial\mathcal{C}$，则任何取值于 $K_{cbf}(x)$ 的 Lipschitz 连续控制器使 $\mathcal{C}$ 安全，且 $\mathcal{C}$ 在 $D$ 内渐近稳定。**Theorem 3**：反向也成立（$\mathcal{C}$ 紧、边界正则下，安全控制律存在 ⟹ $h|_{\mathcal{C}}$ 是 CBF）——CBF 是必要的。

### 4.4 CBF-QP 与 KKT 解析解（§II.C；教程 (7.9)–(7.13)）

**(S9) QP 的构造。** 给定可能不安全的期望控制器 $u = k(x)$（论文记号；下文写 $u_{des}$ 同义），"最小侵入"（minimally invasive）地修正它——安全条件 (10) 对 $u$ 仿射，故取最小二乘投影（原文 **CBF-QP**，无编号）：

$$
u(x) = \arg\min_{u\in\mathbb{R}^m}\ \tfrac{1}{2}\lVert u - k(x)\rVert^2 \quad \text{s.t.}\quad L_f h(x) + L_g h(x)\,u \ \ge\ -\alpha\big(h(x)\big),
$$

其中假设 $U = \mathbb{R}^m$（对应教程 (7.9)）。原文指出：无输入约束时单不等式约束 QP 有闭式解（按 KKT 条件，引文 [39] 即 Boyd & Vandenberghe《Convex Optimization》），即 **min-norm controller**；闭式思想最早出现在 CLF 文献 [40][37]——**解析式本身综述未展开**，以下为教程 07.2 ⑤ 的完整推导（回引编号）：

**(S10) KKT 推导**（教程 (7.10)–(7.13)；每步依据）：
1. 约束标准化 $c(u) = -L_g h\,u - L_f h - \alpha(h) \le 0$（移项取负），拉格朗日函数（教程 (7.10)）：$\mathcal{L} = \tfrac12\lVert u - u_{des}\rVert^2 + \mu\, c(u)$，$\mu \ge 0$。
2. 平稳性 $\partial\mathcal{L}/\partial u = 0$（凸 QP、约束线性、$L_g h \ne 0$ 时 Slater 成立，KKT 充要；求导用 $\partial(\tfrac12\lVert u-u_d\rVert^2)/\partial u = u-u_d$），得 $u^* = u_{des} + \mu\,(L_g h)^\top$（教程 (7.11)）。
3. 互补松弛 $\mu\, c(u^*) = 0$ 分两支：$u_{des}$ 已安全 ⟹ $\mu = 0$、$u^*=u_{des}$（滤波器不干预）；违反 ⟹ 约束取等，解出
   $\mu = \dfrac{-\alpha(h) - L_f h - L_g h\,u_{des}}{\lVert L_g h\rVert^2} \ge 0$（教程 (7.12)；分母 $>0$ 由 $L_g h \ne 0$，分子在违反支为正）。
4. 合并（教程 (7.13)）：
$$
u^* = u_{des} + \frac{(L_g h)^\top}{\lVert L_g h\rVert^2}\,\max\!\Big(0,\ -\alpha(h) - L_f h - L_g h\,u_{des}\Big).
$$
几何解释：$u^*$ 是 $u_{des}$ 到半空间 (10) 的**正交投影**，修正方向 $(L_g h)^\top = g^\top\nabla h^\top$ 是"提升 $\dot h$ 效率最高"的方向——只动不安全分量，即最小侵入。$\lVert L_g h\rVert = 0$ 时分母为零：控制无法影响 $\dot h$，滤波器失效（相对阶问题，见 §4.6）。

### 4.5 CLF-CBF QP：松弛变量与优先级（式 (2)–(4)；§II.C）

**(S11) 两个 certificates 进同一个 QP。** CLF 条件（§II.A，式 (2)–(4)）：$\exists\,u = k(x)$ s.t. $L_f V(x) + L_g V(x)\,u \le -\gamma(V(x))$（式 (2)–(3)，$\gamma$ 为 class-K 函数），控制值集 $K_{clf}(x)$（式 (4)）；$U=\mathbb{R}^m$ 时 $L_g V = 0 \Rightarrow L_f V \le -\gamma(V) \Rightarrow K_{clf}\ne\emptyset$。**CLF-CBF QP**（原文无编号）：

$$
u(x) = \underset{(u,\delta)\in\mathbb{R}^{m+1}}{\arg\min}\ \tfrac{1}{2}u^\top H(x)\,u + p\,\delta^2 \quad
\text{s.t.}\quad L_f V + L_g V u \le -\gamma(V) + \delta,\qquad L_f h + L_g h\,u \ge -\alpha\big(h(x)\big)
$$

$H(x)$ 为任意逐点正定权阵（$H=I$ 退化到最小范数），$\delta$ 是**稳定性**的松弛变量，被 $p>0$ 惩罚；[21] 证明该控制器 Lipschitz 连续。（对应教程 (7.14)。注意：松弛变量是 $\delta$；$\gamma$ 是 CLF 条件里的 class-K 函数。）

**为什么"安全硬约束、稳定软目标"是唯一自洽的优先级**（综述 §II.C 原话："to ensure the QP has a solution one must relax the condition on stability to guarantee safety"；教程 07.3 ⑤ 展开）：
- **可行性不对称**：安全约束单独可行（式 (7) 即其存在性），稳定约束加 $\delta$ 后平凡可行（$\delta$ 取大即可）——两约束同时满足的 $(u,\delta)$ 必存在；反之若稳定也写硬的，$L_g V$ 与 $L_g h$ 方向冲突时 QP 无解（滤波器死机比性能损失更糟）。
- **失效代价不对称**：暂时放弃稳定 = 收敛变慢/悬停（可恢复，且 Theorem 2 保证出集轨迹被拉回）；违反安全 = 碰撞（不可逆）。
- $\delta^2$ 惩罚使松弛只在真冲突时启用：平时 $\delta \approx 0$，近边界时稳定性自动让位。

### 4.6 高相对阶系统：指数 CBF 概要（§IV，式 (15)–(19)）

**问题**：式 (7)/(10) 要求 $\dot h$ 显含 $u$（相对阶 $r = 1$）。若 $h$ 的相对阶 $r \ge 2$（如以位置为状态、力/加速度为输入的碰撞约束），$u$ 只出现在 $r$ 阶导数里，式 (10) 为空约束、无法设计。

**(S12) ECBF 机制**（逐式）：
1. 相对阶定义（式 (15) 下方）：$L_g L_f^i h(x) = 0$（$i=0,\dots,r-2$）且 $L_g L_f^{r-1}h(x) \ne 0$。记 $r$ 阶导数（式 (15)）：
$$
h^{(r)}(x,u) := L_f^r h(x) + L_g L_f^{r-1}h(x)\,u. \tag{15}
$$
2. 引入新"输入" $\mu := h^{(r)}(x,u)$——由 $L_g L_f^{r-1}h \ne 0$，$\mu$ 虽是标量却可经 $u$ 任意配置（原文 §IV.A）。
3. 导数链堆叠成状态（式 (16)）$\eta_h(x) = [\,h,\ L_f h,\ \dots,\ L_f^{r-1}h\,]^\top$，其动态恰是**积分器链**（式 (17)–(18)）：$\dot\eta_h = F\eta_h + G\mu$，$F$ 为移位伴矩阵、$G$ 取最后一个基向量、$C = [\,1\ 0\ \cdots\ 0\,]$ 使 $h = C\eta_h$。
4. 线性反馈 $\mu = -K_\alpha\eta_h$ 下 $h(x(t)) = C\,e^{(F-GK_\alpha)t}\,\eta_h(x_0)$；**比较引理**（依据：$\mu \ge -K_\alpha\eta_h$ 时与线性系统的逐点比较）给 $h(x(t)) \ge C e^{(F-GK_\alpha)t}\eta_h(x_0) \ge 0$，只要 $h(x_0) \ge 0$。
5. 于是**指数 CBF**（Definition 7，条件为式 (19)）：
$$
\sup_{u\in U}\big[\, L_f^r h(x) + L_g L_f^{r-1}h(x)\,u \,\big] \ \ge\ -K_\alpha\,\eta_h(x) \tag{19}
$$
对所有 $x\in\mathrm{Int}(\mathcal{C})$（且 $h(x_0)\ge 0$）成立。Remark 9：$r=1$ 时 $-K_\alpha\eta_h = -\alpha h\cdot h$，式 (19) 退化为 Definition 2——ECBF 是 CBF 的真推广。
6. **设计**（§IV.B，Proposition 6 / Theorem 7 / Theorem 8）：$K_\alpha$ 取 $F-GK_\alpha$ 的 Hurwitz 且**全实负**极点；定义递归 $\nu_0 = h$、$\nu_i = \nu_{i-1} + p_i\,\nu_{i-1}'$（$p_i>0$），Theorem 7/8 给出极点比条件 $\nu_{i-1}(x_0)/\nu_i(x_0) \ge$ 门限以保证各级水平集嵌套、前向不变性逐级传导。控制器落地为 **CLF-ECBF QP**（原文无编号）：目标同 CLF-CBF QP，约束为 $L_f^r h + L_g L_f^{r-1}h\,u = \mu$ 与 $\mu \ge -K_\alpha\eta_h(x)$。

### 4.7 应用清单：各自的 $h$ 设计（§V）

| 应用（原文小节） | 系统与约束 | $h$ 的设计（原文式号） |
|---|---|---|
| 双足踩石行走（§V.A） | 混合系统（式 (20)：摆动相连续动态 + 触地冲量切换 $x^+=\Delta(x^-)$）；DURUS 机器人仿真；落点须落在落脚石圆内 | 落点到圆心距离的几何条件 $h_1(x) = R_1 - O_1F(x) \ge 0$、$h_2(x) = O_2F(x) - R_2 \ge 0$——位置约束、相对阶 2，用 ECBF：$L_f^2 h_i + L_g L_f h_i\,u = -\alpha_{1,1}h_i - \alpha_{1,2}L_f h_i$ |
| 自适应巡航 ACC（§V.B） | Khepera 小车按独轮车模型（式 (21)，状态含 2D 位置/朝向/纵向速度）；与前车保持时距 | $h_{asr}(x) = D - \tau v_f \ge 0$（$D$ 前车距离、$\tau$ 最小时距、$v_f$ 后车速度；引文 [20] 推导） |
| 车道保持 LK（§V.B） | 横向偏移 $y_{lat}$ 有界 $y_{lat} \le d_{max}$，且横向加速度受限于 $a_{max}$ | $h_{lk}(x) = d_{max} - \mathrm{sign}(y_{lat})y_{lat} - \frac{1}{2}\frac{\dot y_{lat}^2}{a_{max}}$——把"多远能刹住"折进 $h$；速度调节 $v \to v_d$ 等性能目标走 CLF，两者经 CLF-CBF QP 统一 |
| Segway 平衡（§V.C） | 倒立摆直立 $\lvert\phi\rvert < \tfrac{\pi}{2}$，输入电压 $u \in [-15,15]$ V；期望控制为 PD | 初拟 $h_1(\phi) = -\phi + \tfrac{\pi}{12}$、$h_2(\phi) = \phi + \tfrac{\pi}{12}$；因硬实现还需附加速率/位置约束，实际用 Hamilton-Jacobi 可达性在 $75\times75\times75$ 状态网格上定安全集、再多项式回归出解析 $h$，作为**安全滤波器/主动集不变滤波（ASIF）**串在 PD 之后（Fig 4） |
| 多机器人长时自主（§V.D） | $N$ 台移动机器人增广电量状态 $\chi_i = [x_i^\top, E_i]^\top$；电量够到充电站、不过充、互相避碰 | $h_{e,i} = E_i - E_{min} - \rho_i(p(x_i)) \ge 0$（能量余量）、$h_{o,i} = E_{max} - E_i \ge 0$，min 合并（式 (22)）；覆盖代价（式 (23)）的负值作任务屏障；避碰 $h_s = \lVert p(x_i)-p(x_j)\rVert^2 - \Delta^2 \ge 0$；各机器人解式 (25)–(26) 的分布式 QP（带松弛 $\delta$，权重 $\kappa$） |

（注：任务规格提及"简单双臂"，但本综述 §V 的应用清单为步行/汽车/平衡/多机器人四类，无双臂示例——按原文呈现。）

## 5. 实验与结果解读（定性）

- **汽车场景**（§V.B，Fig 3）：Khepera 独轮小车同时执行车道保持 + 自适应速度调节——车道屏障 $h_{lk}$ 的实验曲线与解析预测值贴合，速度调节屏障 $h_{asr}$ 在前车出现时自动压低速度维持固定时距；仿真（不同初始条件与参数）与实验定性一致。
- **Segway 硬件**（§V.C，Fig 5）：CLF-QP 在板载 BeagleBone Black 上求解，平均计算时间约 0.4 ms（原文给出）——QP 安全滤波实时性的直接证据。实验：给系统注入幅值超出 $\tfrac{\pi}{12}$ 约束的正弦扰动，带 CBF 时保持直立、不带时越界；再加踢击（kick）扰动，无滤波器倒下、有滤波器稳定。
- **双足行走**（§V.A，Fig 2）：DURUS 仿真中 ECBF $h_1,h_2$ 在多步行走全程非负（踩石保证），对不同步长/步宽与变化的落点均完成行走。
- **多机器人**（§V.D，Fig 6）：Robatarium 台面上 6 台机器人在时域长于（仿真）电池寿命的覆盖监控任务中，绕开障碍、回充电站充电，同时保持覆盖率约束——式 (25)–(26) 的带松弛 QP 在真实多机系统上闭环运行。
- **整体读法**：实验的论证点不在"性能数字"，而在三件事——QP 毫秒级可解、滤波对任意名义控制器（PD/CLF/任务控制器）即插即用、保证（非负性）在扰动下仍被观测维持。

## 6. 局限与后续影响

**论文自身可见的局限**（并对应教程的工程化表述）：

1. **$h$ 的手工设计**：安全集的函数表示是逐任务的智力劳动——需要正则性（$\nabla h \ne 0$ 于边界）与可导性，设计不当则滤波器失效或退化（教程 07.2 ③ (i)；§III 的 SOS 路线（Proposition 5，式 (13)–(14)）是自动化尝试，但双线性条件需交替迭代，且要求名义控制器解析可知）。
2. **可行性与输入约束**：CBF-QP 推导假设 $U = \mathbb{R}^m$；输入受限时 $K_{cbf}(x)$ 可为空集，QP 不可行即滤波器失效——需要加松弛或退到备份安全集（教程 07.3 ⑤ 第 3 步的注意项）。多约束冲突同样可能不可行，本综述以"逐条进 QP"呈现、未给一般可行性判据。
3. **离散时间实现**：全部理论是连续时间的；数字实现按采样周期检验约束，只是连续条件的离散近似（教程 07.3 ⑤ 明示"小步长下定性成立"）——离散时间 CBF 的严格化属后续文献。
4. **模型误差**：(7.7) 类展开用名义 $f,g$；Theorem 2 的渐近稳定性（Remark 6）给噪声/误差下的实用回拉余量，但形式保证仍是名义模型下的——鲁棒/扰动下安全（input-to-state safety）是后续方向。

**后续影响**（论文 §VI 的展望已被验证）：CBF-QP 成为学习式策略部署的安全层的标准做法（RL/策略网络 + CBF 滤波的叠层范式）；高相对阶/离散时间/学习式 $h$（神经 CBF）、多约束可行性、随机系统 CBF 均成为活跃分支。

## 7. 与本项目对照

**教程第 07 章映射表**（编号已逐一核实）：

| 本综述内容（原文编号） | 教程第 07 章（编号） |
|---|---|
| 安全集 (5)、前向不变（Def 1） | (7.1)、(7.2) |
| Nagumo 条件（§I.A 无编号） | (7.3)（含相切情形留证的诚实说明） |
| CBF 定义（Def 2，式 (7)）；$\alpha$ 扩展 class-$K_\infty$ | (7.4)；线性 $\alpha$ 的指数回拉闭式 (7.5) |
| 控制仿射系统 (1)、李导数展开 | (7.6)、(7.7) |
| 安全控制集 $K_{cbf}$ (10) | (7.8) |
| CBF-QP（§II.C 无编号） | (7.9) |
| min-norm 闭式解（论文只声明"per KKT"） | (7.10)–(7.13)（教程补全的逐步推导） |
| CLF 条件 (2)–(4)、CLF-CBF QP（无编号）、优先级论证 | 07.3 ②–⑤，(7.14) |
| ECBF（§IV，式 (15)–(19)） | 07.2 ⑤"失效边界"指到的延伸（相对阶 > 1 处理） |
| 应用（§V） | 07.2 ④ 的理论依据清单 |

**CBF 在 RoboTwin 策略部署上的安全层用途**（回引教程第 10 章，编号已核实）：第 10 章 10.1 的全栈对应表把"安全滤波（横切所有层）"单列一行，指向第 07 章与本综述 PDF；10.2 的 (10.1) 表明策略输出的是关节位置增量 $q_{cmd} = q + \Delta q$、经 PD 执行器出力矩——安全层的插入点即在 $u_{des} = q_{cmd}$ 之后、执行器之前：以关节限位裕量/自碰撞 SDF 负距离构造 $h$（速度级状态使相对阶为一），按 (7.9)/(7.13) 每周期做一次半空间投影，把不安全的 $\Delta q$ 拉回 $K_{cbf}$。第 10 章"实战产出"第 5 步正是该最小混合栈：RoboTwin 训练的策略外面包一层 CBF-QP 安全滤波，对比有/无滤波的动作轨迹——"经典为骨、学习为肉、CBF 为保险"。约束来自估计的 SDF 时，保证退化为第 01 章 (1.7) 意义上的条件保证（模型/感知误差下留裕量）。

## 配套阅读

- 教程主线：[第 07 章｜安全控制：控制屏障函数](../07_安全控制控制屏障函数.md)（(7.1)–(7.14) 与本综述逐式对应；工程三步法在 07.3 ⑤）；部署侧：[第 10 章｜控制栈实战](../10_控制栈实战与robotwin衔接.md)（10.1 对应表、(10.1) 动作接口、实战产出第 5 步）。
- 同目录精读：[RRT 精读](RRT_TR1998.md)、[RRT* 精读](RRTstar_IJRR2011.md)——L2 层的"几何可行"与本章"动态安全"互补：RRT* 保证路径不穿障（空间维），CBF 保证轨迹不出集（时间维）。
- 跨主题精读：[RoboTwin2.0 精读](../../robotwin/精读/RoboTwin2.0_arXiv2025.md)、[RT1 精读](../../robotwin/精读/RT1_arXiv2022.md)——学习式策略是 CBF 滤波的典型服务对象。
- 论文延伸：§III 的 SOS 自动构造（Proposition 5 + 引文 [44][45]）；引文 [21]（Ames et al., IEEE TAC 2017）为本综述理论主结果的完整证明；引文 [39] Boyd & Vandenberghe《Convex Optimization》ch5（KKT 条件——(7.13) 的依据）。

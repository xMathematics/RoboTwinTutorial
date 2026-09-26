# 论文精读｜RRT：快速探索随机树（TR 98-11，1998）

> **PDF**：[papers/control_planning/classics/RRT_TR1998_LaValle.pdf](../../../papers/control_planning/classics/RRT_TR1998_LaValle.pdf) ｜ **教程**：[第 03 章](../03_运动规划-i采样式规划.md) §03.1 ｜ **代码**：论文实验为作者自写实现、未随报告发布；事实标准参考实现是 OMPL 的 `geometric/RRT` 类（另有作者后续发布的 RRT 软件页，算法同源）

## 1. 论文信息与一句话贡献

- **题目**：Rapidly-Exploring Random Trees: A New Tool for Path Planning
- **作者**：Steven M. LaValle（Iowa State University 计算机系，单作者）
- **发表**：技术报告 TR 98-11，Iowa State University，1998。**本仓库 PDF 为该报告的 4 页精简版**（GNU Ghostscript 老式排版，文本提取乱码，本精读以逐页读图为准，共 4 页）。
- **版本与引用勘误**：[第 03 章](../03_运动规划-i采样式规划.md) ④ 把该 PDF 标注为 "LaValle & Kuffner, *Randomized Kinodynamic Planning*"——实际 PDF 为 **LaValle 单作者**、题名如上；"Randomized Kinodynamic Planning" 是它的会议/期刊扩展版（LaValle & Kuffner，论文自引 [7]，ICRA 1999 / IJRR 2001）。本精读一律以本地 PDF 实况为准。
- **编号声明**：论文正文**没有任何编号公式**，也没有编号定理——叙述性文字 + 一段伪代码（GENERATE_RRT）。本精读自编式号 **(R1)–(R6)** 并逐条声明对应原文位置；凡原文没有的论证（如覆盖引理）都明确标注"非本文内容"。
- **一句话贡献**：提出快速探索随机树（rapidly-exploring random tree，RRT）——一种用**随机采样的输入序列做数值积分**来增量生长的树结构，天生面向非完整约束（nonholonomic）与动力学（kinodynamic）规划，**不要求求解两点边值问题**（two-point boundary value problem），并给出 Voronoi 偏置解释与"顶点分布收敛到采样分布"的论证。

## 2. 问题与动机

**要解决什么**（§1）：高维构型空间中的路径规划，特别是带**非完整约束与动力学**的版本。当时两大主流在这一点上双双失灵：

1. **随机势场法**（randomized potential field，论文引 [2] Barraquand & Latombe）：性能重度依赖人工设计的启发式势函数；有障碍、运动学与动力学约束并存时势函数难以设计；且扩展到一般非完整问题需要逐点求解两点边值问题（把两端状态绑定的非线性控制问题）。
2. **概率路线图**（probabilistic roadmap，PRM，论文引 [1,4]）：在构型空间撒样本、近邻互连建图。连接一对构型 = 解一个非线性控制问题——对非完整/动力学系统，"成千上万条连接"不可行（§1 原话：每条连接 akin to a nonlinear control problem）。

**RRT 的出发点**（§1 末）：设计一种随机数据结构，扩展它靠"施加控制输入把系统略微推向随机选的点"（applying control inputs to drive the system slightly toward randomly-selected points），而不是"点对点连通"——边由**前向积分**生成，方向天然单向，非完整约束不需要额外处理。摘要报告已用于 holonomic、非完整、动力学规划，状态空间最高 12 维。

## 3. 方法总览

```
状态空间 X（度量空间，含障碍区 X_obs 与自由空间 X_free；只能做碰撞检测查询）
   │  状态转移方程（state transition equation）ẋ = f(x, u)   ← 控制论式建模，可编码几乎所有
   │                                                          动力学与运动学模型（§2）
   ▼
GENERATE_RRT(x_init, K, Δt)：循环 K 次
   x_rand  ~ RANDOM_STATE            随机采样（均匀或任意光滑密度）
   x_near  ← NEAREST_NEIGHBOR        树上离 x_rand 最近的顶点（度量 ρ）
   u       ← SELECT_INPUT            选输入，使 x_near 尽量靠近 x_rand 且不碰
   x_new   ← NEW_STATE(x_near,u,Δt)  从 x_near 施加 u 积分 Δt（Euler / Runge-Kutta）
   增量碰撞检测通过 → x_new 入树，边记录输入 u
```

四个要点（§2 原文）：

- **边 = 输入段**：每条边除端点外还**记录输入 u**（`T.add_edge(x_near, x_new, u)`，算法第 8 行注释）——执行时从叶子回放到根即得可执行输入序列，这是 RRT 与 PRM 在"规划—控制接口"上的本质差异。
- **积分而非求解**：对固定时间间隔 Δt 积分 f 得下一状态；Euler 近似 x_new ≈ x + f(x,u)Δt，论文推荐 Runge-Kutta 等高阶格式。
- **增量碰撞检测**：论文点名 Mitchell 的 V-Clip 等增量法——每轮只查**新增边**，树的既有部分已被历史轮次验证（与教程 03.1 (3.4) 的注释同义）。
- **U 有限时可枚举，否则离散化或改用数值优化**求 SELECT_INPUT（§2 对算法第 5 行的注释）。

**读法建议**：先读 §2 的问题设定（X、X_obs、X_free 与"只能查询碰撞"的接口约定）与 GENERATE_RRT 九行伪代码，再按 §4.3（全向归约）→ §4.4（Voronoi 偏置）→ §4.5（完备性论证程度）的顺序过公式，最后读 §3 的性质列表与 §5 的研究问题清单——后者几乎就是此后十年采样式规划的路线图。

**论文 §3 自列的七条性质**（原文序号照录，每条括注本精读对应处）：

1. 扩展强烈偏置于**未探索**区域（§4.4，(R5)）；
2. 顶点分布**逼近采样分布**，行为一致可预期（§4.5，(R6)）；
3. 在非常一般的条件下**概率完备**（§4.5——注意原文对此只声称、未证明）；
4. 算法**简单**，便于性能分析（与 PRM 同享的优点）；
5. **永远连通**且边数最少（$n-1$ 条边 $n$ 个顶点，树上无环、无冗余边）；
6. 可视为**路径规划模块**，可嵌入更大规划系统（如替换随机势场法中的随机游走阶段——§5 的建议）；
7. 构建完整规划算法**不要求**"在两个指定状态间转向"的能力（two-point BVP free）——这是对比随机势场法的关键一条，也是动机的核心。

## 4. 关键公式推导（式号 (R1)–(R6) 为本精读自编，论文原文无编号公式）

### 4.0 符号表

| 符号 | 含义（对应原文位置） |
|---|---|
| $X,\ X_{\text{obs}},\ X_{\text{free}}$ | 状态空间 / 障碍区 / 自由空间 $X_{\text{free}} = \mathrm{cl}(X\setminus X_{\text{obs}})$（§2） |
| $x_{\text{init}},\ x_{\text{goal}}$ | 初始状态与目标状态（目标可为区域，§2 记 $x_{goal} \subset X$） |
| $u \in U_f$ | 控制输入与输入集（§2，状态转移方程中选取） |
| $\Delta t,\ K$ | 积分步长 / 迭代次数（GENERATE_RRT 的参数） |
| $\rho(\cdot,\cdot)$ | 状态空间距离度量（§2 算法注释；全向特例取欧氏度量，§3 开头） |
| $T=(V,E)$ | 树：顶点集 $V \subset X_{\text{free}}$，边集 $E$（每边附一个输入） |

### 4.1 状态转移方程与 NEW_STATE

$$
\dot{x} = f(x, u)
\tag{R1}
$$

（原文：§2 首段状态转移方程。）对固定时间间隔 $\Delta t$ 积分，Euler 离散化：

$$
x_{\text{new}} \approx x_{\text{near}} + f(x_{\text{near}}, u)\,\Delta t
\tag{R2}
$$

（原文：§2 "Using Euler integration, $x_{new} \approx x_f + f(x, u)\Delta t$"；本式即 `NEW_STATE(x_near, u, Δt)` 的实现。依据：一阶 Taylor 展开 $\dot x$ 在 $[t, t+\Delta t]$ 上取左端点值；论文同时说明实际实现应换 Runge-Kutta 高阶格式以减小离散化误差。）注意 (R2) 的语义与教程 (3.3) 的 steer 不同：它是**沿输入方向积分**，不是直线走向采样点——这正是非完整系统能被处理的原因。

### 4.2 GENERATE_RRT 伪代码逐行形式化

论文算法（§2 右栏，逐行照录后逐行形式化）：

```text
GENERATE_RRT(x_init, K, Δt)
1  T.init(x_init)                       # V ← {x_init}, E ← ∅
2  for k = 1 to K do
3    x_rand ← RANDOM_STATE()            # x_rand ∈ X_free（X 有界，均匀或任意光滑密度 p）
4    x_near ← NEAREST_NEIGHBOR(x_rand, T)   # x_near = argmin_{v∈V} ρ(v, x_rand)
5    u      ← SELECT_INPUT(x_rand, x_near)  # 使 x_near 逼近 x_rand 的输入，见 (R3)
6    x_new  ← NEW_STATE(x_near, u, Δt)  # 按 (R2) 积分
7    T.add_vertex(x_new)                # V ← V ∪ {x_new}
8    T.add_edge(x_near, x_new, u)       # E ← E ∪ {(x_near, x_new)}，边记录输入 u
9  Return T
```

第 5 步的 SELECT_INPUT 形式化为（论文文字："selects the input, u, that minimizes the distance from $x_{near}$ to $x_{rand}$, and ensures that the state remains in $X_{free}$"）：

$$
u^* \in \arg\min_{u \in U_f}\ \rho\big(\,\mathrm{NEW\_STATE}(x_{\text{near}}, u, \Delta t),\ x_{\text{rand}}\,\big), \qquad \text{s.t. } \mathrm{NEW\_STATE}(x_{\text{near}}, u, \Delta t) \in X_{\text{free}}
\tag{R3}
$$

（依据：论文对第 5 步的两点文字要求——距离最小 + 留在自由空间——写成约束优化。原文没有给符号化目标函数，(R3) 的写法是本精读的忠实形式化。）任务规格里"扩展函数 extend(E, x_nearest, x_new)"是后续文献（RRT-Connect、Karaman & Frazzoli 2011）打包出的接口名；本版本论文的对应物是**第 4–8 行整段**，且扩展语义是"选输入 + 积分"而非"定向步进"。

两点如实说明：① 论文算法**没有目标判定与提前终止**——它就是"建一棵 K 顶点的树"，如何从中取解由"性质 6：RRT 可视为路径规划模块"承担（Karaman & Frazzoli 2011 第 13 页复述：原始版本"树一出现目标区域内的节点即停"）；工程实现的 goal 检查是后加约定，非本伪代码内容。② 步长语义由 $(u, \Delta t)$ 二元组承担——**没有显式的 ε 步长参数**，速度界 $\sup\|f\|\cdot\Delta t$ 起步长作用（4.3 归约）。

### 4.3 全向特例：从 (R2)–(R3) 归约到教程 (3.3)

§3 开头的特例设定：全向模型 $f(x,u) = u$，$U_f = \{u \in \mathbb{R}^2 \mid \|u\| \le 1\}$，$\rho$ 取欧氏度量。此时 (R3) 的目标函数（依据：(R2) 代入，$\mathrm{NEW\_STATE}$ 线性于 $u$）：

$$
\rho\big(x_{\text{near}} + u\,\Delta t,\ x_{\text{rand}}\big) = \big\lVert (x_{\text{near}} - x_{\text{rand}}) + u\,\Delta t \big\rVert
$$

极小化该范数（依据：范数的平移不变性；$\|u\Delta t\| \le \Delta t$，故最优点在把 $x_{\text{near}}$ 沿 $u\Delta t$ 尽量推向 $x_{\text{rand}}$ 处），得闭式解：

$$
u^* = \frac{x_{\text{rand}} - x_{\text{near}}}{\lVert x_{\text{rand}} - x_{\text{near}} \rVert}, \qquad
x_{\text{new}} = x_{\text{near}} + \min\big(\Delta t,\ \lVert x_{\text{rand}} - x_{\text{near}} \rVert\big)\, \frac{x_{\text{rand}} - x_{\text{near}}}{\lVert x_{\text{rand}} - x_{\text{near}} \rVert}
\tag{R4}
$$

（依据：把 $u^*$ 代回 (R2)；当剩余距离不足 $\Delta t$ 时受 $\|u\|\le 1$ 限制只能走到 $x_{\text{rand}}$——第二式的 $\min$ 结构。）对照教程第 03 章式 (3.3)：$q_{\text{new}} = q_{\text{near}} + \min(\epsilon, \lVert q_{\text{rand}} - q_{\text{near}}\rVert)\frac{q_{\text{rand}}-q_{\text{near}}}{\lVert q_{\text{rand}}-q_{\text{near}}\rVert}$——**取 $\epsilon := \Delta t \cdot u_{\max}$（本例 $u_{\max}=1$ 时 $\epsilon = \Delta t$）两式完全重合**。即教程 (3.3)–(3.4) 是 GENERATE_RRT 在"全向 + 单位速度界"下的特例；论文的一般形态多出的自由度正是 $u$ 的选择空间。

### 4.4 Voronoi 偏置：树为什么被拉向空白区

设 $x_{\text{rand}}$ 均匀分布于有界 $X$。顶点 $v \in V$ 被选为 $x_{\text{near}}$ 当且仅当采样落入其 Voronoi 胞元（Voronoi cell）：

$$
\mathbb{P}\big(x_{\text{near}} = v \mid x_{\text{rand}} \sim \mathrm{Unif}(X)\big) = \frac{\mu\big(\mathcal{V}(v)\big)}{\mu(X)}, \qquad \mathcal{V}(v) = \{x \in X : \rho(v, x) \le \rho(v', x)\ \forall v' \in V\}
\tag{R5}
$$

（依据：算法第 4 行最近邻定义的直接概率重述；$\mu$ 为 Lebesgue 测度。论文原文是文字表述——"vertices with large Voronoi regions are more likely to be selected for expansion"——(R5) 是其忠实形式化。）直觉论证链条：树节点稀疏的方向上胞元必然大（胞元体积 ~ 邻域内节点间距的 $d$ 次方）；胞元大 ⇒ 被选为扩展父点的概率高 ⇒ 新节点落在稀疏方向。论文称大胞元集中在树的"前沿"（frontier），故"an RRT works in the opposite manner [to random walk]——already explored places are not visited"。对照：随机游走/朴素随机树的扩展点集中在已探索区（密度高、胞元小的地方反而反复选中其**周边**），RRT 的偏置方向相反——这是"rapidly-exploring"一词的数学内容。

### 4.5 顶点分布收敛与概率完备性（原文论证程度如实标注）

**论文自身的论证（§3）。** 三个递进的陈述，全部是**论证性描述 + 数值验证，不是定理证明**：

- (R6) **稠密性**：随机点落在树顶点 $\Delta t$-邻域内的概率随迭代趋于 1，此时该随机样本本身就会被加为顶点；"若样本均匀生成，顶点分布趋于均匀，且该结果与初始顶点位置无关；若样本来自光滑密度 $p(x)$，顶点分布趋于 $p(x)$"（§3，即"性质 2：顶点分布逼近采样分布"）。其机制正是 (R5)：任何足够大的空胞元都会以高概率在后续轮次被"命中并填充"。
- **概率完备性**（probabilistic completeness）只以"性质 3"一句话声称："an RRT is probabilistically complete under very general conditions"，依据引到上述稠密性论证与 §3 末"对非完整系统理想度量难定义故用简单度量"的讨论。
- 数值佐证：多次 Chi-Square 检验确认顶点分布的均匀性（§4）。

**覆盖引理概要（非本文内容，后续形式化）。** 论文没有 $\delta$-clearance 覆盖论证；严格版本是后来的工作——本仓库 RRT* PDF 给出两个可直接引用的形态：Theorem 16（RRT 概率完备）与 Theorem 23（RRG/RRT* 同理）。概要三步（即教程 03.1 ⑤ 的骨架，逐步写出以便与原文论证程度对照）：

1. **球链覆盖。** 设问题 robustly feasible：存在可行路径 $\sigma$ 满足 $\sigma(\tau) \in \mathrm{int}_\delta(X_{\text{free}})$ 对一切 $\tau \in [0,1]$（定义见 RRT* PDF 第 17 页）。沿 $\sigma$ 取有限串半径 $\delta/2$、球心间距 $< \delta/2$ 的球 $B_1, \dots, B_M$（依据：$\sigma([0,1])$ 紧致 ⇒ 连续像可被有限子覆盖覆盖；球心取在路径上故每球整个落在 $X_{\text{free}}$ 内）。设 $B_1$ 已含树根 $x_{\text{init}}$。
2. **单球可达。** 考察下一球 $B_{i+1}$：每轮采样独立地以 $p_B = \mu(B_{i+1})/\mu(X_{\text{free}}) > 0$ 落入 $B_{i+1}$（依据：采样均匀、$B_{i+1} \subset X_{\text{free}}$）。落入后由 (R4)（全向：$x_{\text{rand}}$ 本身成为新顶点）或 (R3)（一般：输入集在球内存在指向球心的输入时）在球内留下节点，树到 $B_{i+1}$ 球心的距离不再增大；"最近节点距离按固定比例期望收缩"的定量版本即 Frazzoli–Dale–LaValle 2002 的核心引理（转引见 RRT* PDF §1）。于是"球 $B_{i+1}$ 在 $n$ 轮内仍未被占据"的概率有指数尾：$\mathbb{P} \le (1 - p_B)^{n - n_0} \le e^{-p_B (n - n_0)}$（依据：几何分布尾界与 $(1-x)^n \le e^{-nx}$）。
3. **并集界合拢。** 对 $M-1$ 个未占球逐球应用第 2 步并相乘（依据：采样独立；前一球被占据后，后续球的条件成功率只增不减——树离得更近）。取 $n \ge n_0(\delta, p_B, M)$ 后失败概率 $\le M e^{-p_B(n-n_0)} \to 0$，即 Theorem 16 的 $1 - e^{-an}$ 型结论（$a$ 只依赖 $X_{\text{free}}, X_{\text{goal}}$，与 $n$ 无关——原文常数表述照录）。

**标注：以上三步 1998 年论文只做到了"稠密性直觉"（R6）这一层**——既无 clearance 假设的显式引入，也无速率常数；"有限步找到解"在本文的形态是 §3 末"顶点均匀化 ⇒ $\Delta t$-邻域覆盖全空间 ⇒ 目标区域以概率 1 被伸入"的定性论证。速率常数是 2002 年后才有。

### 4.6 RRT 与 PRM 的结构对比（§3 末的形式化整理）

论文 §3 末给出一组定性对比，整理成表（"原文声称 + 一句依据"）：

| 维度 | RRT（本文） | PRM（引 [1,4]） |
|---|---|---|
| 数据结构 | 树：$|E| = |V| - 1$，恒连通、无环 | 图：为连通可能生成大量冗余边 |
| 边的生成 | 前向积分（单向、无需可逆） | 点对互连（需解两点边值问题才能验证可连性——非完整时不可行） |
| 每轮查询 | 一次最近邻 | $k$-近邻（更贵，§3 原话 more expensive） |
| 采样复用 | 单查询：树只服务当前起终点 | 多查询：路图摊销到多次查询（教程 03.3 的选型分界） |
| 碰撞检测 | 增量检测（只查新边） | 每条候选边独立检测 |

（依据：均出自论文 §3 末段原文；表头为本精读所加。）其中"树 = 最小连通结构"一条在后续分析里分量最重：Karaman & Frazzoli 2011 沿用同一事实推出 RRT 每轮恰 1 次碰撞检测调用、边数 $O(n)$（见 RRT* 精读 §4.6，Lemma 42 与第 33 页边数统计）。

### 4.7 双树与"规划—控制结合"（论文实况）

- **双树**：本 4 页版**没有**双向搜索算法。§5 Research Issues 仅一句方向性提及："one could generate multiple RRTs (for example, one rooted at $x_{init}$ and another rooted at $x_{goal}$)"。完整的双向算法（RRT-Connect：两树交替扩展 +贪婪连接）是 Kuffner & LaValle 2000 (ICRA) 的工作；本仓库 RRT* PDF 第 13 页也把双向变体归于 RDT（LaValle 2006）。**任务规格所问"RRT-Connect 思想是否在此版本"的答案：只有一句话的萌芽，没有算法。**
- **规划与控制结合**：本版本的结合方式就是算法本身——边记录输入 $u$（第 8 行），解 = 一串输入段，执行器直接回放（§1：不需要点对点收敛/边值求解）。任务规格提到的"Part II 轨迹平滑/平滑滤波"**不在本 4 页 PDF 中**（精简版无 Part 结构、无平滑小节）；轨迹平滑的讨论见于完整长版报告与期刊版（LaValle & Kuffner 2001）的后处理章节，本地 PDF 未收录、本精读不展开，仅注明：树输出的原始解是逐段输入轨迹，平滑属后处理，不改变树的构建逻辑。

## 5. 实验与结果解读（定性表述，数字照录原文）

- **2D 全向基线**（§3–§4）：$X = [0,100]\times[0,100]$，$\Delta t = 1$，$x_{\text{init}} = (50,50)$。三帧快照显示树从中心向**四个角**快速伸展（原文强调"虽然构造法简单，得到这种行为并不容易"：朴素随机树与随机游走都会困在已探索区）；顶点均匀性通过多轮 Chi-Square 检验确认；对凸空间中的 2D 问题，"叶到根路径"与直线（点对根的最短连接）的平均长度比在 **1.3–2.0** 之间（原文定位为"generated paths are not far from optimal"的经验证据，非最优性声明——这个比值后来在 Karaman & Frazzoli 2011 的无障碍 2D 实验中被精化为 $\sqrt{2}$ 的极限，两篇论文在无障碍方形场景上直接对话）。
- **均匀性与初值无关性**（§3）：论文特别强调顶点分布收敛到采样分布"与初始顶点位置无关（also confirmed by our experiments!）"——即树的长期统计行为由采样分布主导，而不是由根的位置主导；这是把 RRT 当作"一致的随机覆盖器"使用的依据，也是运动学动力学规划中"简单度量即可"论断的实证支点。
- **运动学动力学例**（§4，Fig. 1 与第 4 页图组）：紧约束 3D 全向问题；**只能前进 + 右转**的车（非完整，输入只有两档增量）；杂乱环境中的不可控小车（三档固定转向增量，连直行都不行——展示 RRT 对"根本不可控"系统也能出轨迹）；5 自由度动力学车模型（Fig. 1 即其 5D RRT 的 2D 投影，解路径加粗高亮）。作者自评实现"neglected many efficiency issues"，但计算性能已令人鼓舞（定性）。相关工作（论文引 [7]，即期刊版）在杂乱 2D/3D 环境用喷气输入规划 hovercraft 与卫星轨迹。
- **与 PRM 的定性对比**（§3 末）：RRT 恒连通且边数最少（$n-1$）；PRM 为保连通可能生成大量冗余边；PRM 需 $k$-近邻查询而 RRT 每轮一次最近邻；碰撞检测是共同瓶颈，RRT 的增量检测允许用"最快可用"的碰撞检测器。作者谨慎地不下的结论：实验上 RRT 似更快，但"难以做出决定性的实验比较"——这份克制正好留给 RRT* 论文的对照实验去回答（见 [RRT* 精读](RRTstar_IJRR2011.md) §5：同采样序列下 RRT* 解代价单调降到最优，代价是常数倍的运行时间）。

## 6. 局限与后续影响

- **解非最优**：论文只声称"不远离最优"（凸空间 1.3–2.0 倍），树生长机制不编码路径质量。后来被 Karaman & Frazzoli (2011) 严格化为 **Theorem 33**：RRT 几乎必然收敛到一个**严格大于最优代价的随机变量**（详见 [RRT* 精读](RRTstar_IJRR2011.md) §4.4）——次优不是实现瑕疵而是结构性质。
- **窄通道难**：均匀采样落入窄通道的概率正比其测度，(R5) 的偏置对"小胞元但必须穿过"的区域无能为力；论文未讨论，成为后续 goal bias（§5 已预告 biasing $x_{rand}$）、桥采样等工作动机。
- **理论留白即议程**（§5 自列）：高效最近邻结构、收敛速率界、解质量界——分别由后续的 k-d 树最近邻、概率完备速率（Frazzoli–Dale–LaValle 2002）、渐近最优（Karaman–Frazzoli 2011）接棒。
- **影响**：RRT/PRM 两条采样式主线自此并立；RRT-Connect（2000）、运动学动力学规划的期刊版（2001）、OMPL 中的默认规划器家族，都从这份 4 页报告生长。

## 7. 与本项目对照

- **教程映射**：第 03 章 §03.1 的 RRT 主循环 (3.3)（steer）与 (3.4)（增量碰撞检测）即本文 GENERATE_RRT 的全向特例（归约见 §4.3，$\epsilon \leftrightarrow \Delta t \cdot u_{\max}$）；(R5) 对应教程"Voronoi 偏置"段；教程 03.1 ⑤ 的覆盖引理概要与本文 (R6) 的差距在 §4.5 已逐条标注。教程 03.3 的 goal bias 与穿隧分析是论文 §5 "biasing $x_{rand}$" 与增量碰撞检测两处提及的工程展开。
- **2D 教学实现**：`projects/control_planning/` **尚未建立**（projects/ 目前只有 nerf 与 slam）——本精读 §4.2 的伪代码可直接作为该实现的主循环骨架：R4 闭式 steer + 增量碰撞检测 + 边记录输入，与 SLAM 侧 `projects/slam/` 的组织方式对齐后补齐。
- **与 SLAM 精读系列的关系**（规划栈 vs 感知栈）：[tutorials/slam/精读/](../../slam/精读/README.md) 的 LOAM/ORB-SLAM/VINS 等解决"我在哪、地图长什么样"（感知栈输入端），本篇与其后的 [RRT* 精读](RRTstar_IJRR2011.md) 解决"知道地图后怎么走"（规划栈决策端）；[3D 重建精读](../../3d_reconstruction/精读/README.md) 的 COLMAP/NeRF/3DGS 提供规划所需的几何地图表示。教程第 01 章的栈视图（感知 → 建图 → 规划 → 控制）把三者串成一条流水线：SLAM 出位姿与占据信息，重建出 $\mathcal{C}_{free}$ 的几何源，RRT 系在隐式 $\mathcal{C}_{free}$ 上出路径——第 03 章开篇"碰撞检测是唯一接口"的约定正是这条流水线的层间协议。

## 配套阅读

- 同组姊妹篇：[RRT*（Karaman & Frazzoli, IJRR 2011）精读](RRTstar_IJRR2011.md)——RRT 次优性的严格化（Theorem 33）与"选父 + 重布线"的渐近最优修正。
- 教程：[第 03 章｜采样式规划](../03_运动规划-i采样式规划.md)（(3.1)–(3.4) 与概率完备性概要）｜ [第 09 章](../09_前沿学习式规划与腿式控制.md)（MPNet 用神经采样偏置替换本文的均匀采样，RRT* 兜底）。
- 论文延伸：Kuffner & LaValle, *RRT-Connect*, ICRA 2000（双向树，论文库未收录）；LaValle & Kuffner, *Randomized Kinodynamic Planning*, IJRR 2001（本文的期刊扩展，含轨迹平滑后处理）；LaValle, *Planning Algorithms*, Cambridge 2006, ch5（免费在线，系统表述）。
- 主题导航：[tutorials/control_planning/README.md](../README.md) ｜ 论文索引：[papers/control_planning/README.md](../../../papers/control_planning/README.md)

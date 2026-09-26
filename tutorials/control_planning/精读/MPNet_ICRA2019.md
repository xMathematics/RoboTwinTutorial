# 论文精读｜MPNet：运动规划网络（ICRA 2019）

> **PDF**：[papers/control_planning/frontier/arXiv-1806.05767_MPNet.pdf](../../../papers/control_planning/frontier/arXiv-1806.05767_MPNet.pdf)（arXiv:1806.05767v2，共 7 页，式号以此为准） ｜ **教程**：[第 09 章](../09_前沿学习式规划与腿式控制.md) §09.1（MPNet） ｜ **代码**：官方项目页 sites.google.com/view/mpnethome（论文脚注 1，含实现参数与视频；PyTorch 实现，另见作者后续开源仓库）

## 1. 论文信息与一句话贡献

- **题目**：Motion Planning Networks
- **作者**：Ahmed H. Qureshi、Anthony Simeonov、Mayur J. Bency、Michael C. Yip（UC San Diego，机械与航空航天工程系）
- **发表**：IEEE ICRA 2019（arXiv:1806.05767v2，2019-02-24）
- **命名核对**：论文正文的在线双向采样-连接启发式自称 **Neural Planner**（Algorithm 2，"incremental bidirectional path generation heuristic"），**全文未出现"XAT"一词**（已检索原文确认）——"XAT"是教程 §09.1 注明的作者开源实现沿用的名字。本精读按论文用 Algorithm 2 / Neural Planner。
- **一句话贡献**：两个网络（编码器 Enet 把障碍**点云**压入隐空间 $\mathbf{Z}$，规划器 Pnet 以 $(\mathbf{Z}, \mathbf{x}_t, \mathbf{x}_T)$ 回归"下一步构型"）+ 在线阶段用 **dropout 随机化**把确定性回归变成隐式采样器，配增量式双向树（Algorithm 2）与两级重规划（神经重规划 / 混合重规划回退到 RRT*）——在保持 RRT* 概率完备性继承的同时，把 2D/3D 点质量、刚体与 **7-DOF Baxter** 规划的平均耗时压到 **1 秒上下**，比 BIT* 至少快 20 倍，且泛化到训练时未见的障碍布局。

## 2. 问题与动机

**要解决什么**（§I）：采样式规划（SMT: RRT [4]、RRT* [5]、P-RRT* [6]、BIT* [9] 等）的计算复杂度随维度指数式增长——自动驾驶等场景要求可扩展、计算高效、实时的规划器，而现有方法的运行时间在高维问题上不可接受（论文实验里 7-DOF 场景 BIT* 平均 3.1 分钟量级，Fig. 1 说明）。

**既有神经路线的两个缺口**（§I–II）：
1. **纯学习无保证**：早期深度网络规划器因训练复杂度高而失败；近年 VIN [12]、imitation [13]–[16]、Lightning [17] 各有局限——VIN 只在玩具问题评估；Lightning 靠查找表缓存旧解，**内存低效且无法泛化到新环境**。
2. **采样分布学习**（如 Ichter et al. [16]）只学采样偏置，框架仍要完整的探索循环。

**为什么"快"必须是毫秒级**（教程 §09.1 ① 的工程动机）：与在线闭环（如第 05 章 MPC 的滚动重解）组合时，每个控制周期都要重规划——经典规划器秒到分钟级的首解时间会让闭环断裂；只有把 L2 压到亚秒级，"规划"才能进入控制回路。这决定了评价口径：论文比的不是渐近最优性，而是**首解时间与成功率**。

**论文的目标形态**（§I）：端到端从点云观测直接生成"起点→终点"的无碰撞路径，同时**不放弃完备性**——"As neural networks do not provide theoretical guarantees, we propose a hybrid algorithm which combines MPNet with any existing classical motion planning algorithm, in our case RRT*"——混合模式在全部测试环境 100% 成功率。这正是教程 §09.1 的"学习先验 + 经典兜底"，也是 §09.3"保证守恒"原则的实例：学习组件放在**采样分布**这个保证最便宜的组件上。

## 3. 方法总览

**问题形式化**（§III）：状态空间 $\mathcal{X} \subset \mathbb{R}^d$；$\mathcal{X}_{obs} \subset \mathcal{X}$、$\mathcal{X}_{free} = \mathcal{X} \setminus \mathcal{X}_{obs}$；$\mathbf{x}_{init} \in \mathcal{X}_{init} \subset \mathcal{X}_{free}$、$\mathbf{x}_{goal} \in \mathcal{X}_{goal} \subset \mathcal{X}_{free}$；解是正标量长度的有序路径 $\tau$，可行 ⟺ $\tau(0) = \mathbf{x}_{init}$、$\tau(|\text{end}|) \in \mathcal{X}_{goal}$ 且完全位于 $\mathcal{X}_{free}$。（与教程第 03 章 (3.1) 的规划问题契约一致。）

**两阶段结构**（§IV，Fig. 2）：

```
离线训练                       在线规划
┌────────────────────┐   ┌───────────────────────────────────┐
│ Enet: x_obs → Z    │   │ Algorithm 2 Neural Planner:        │
│  (重建损失, 式 1)   │   │  双向树 τ^a/τ^b 交替采样–连接       │
│ Pnet: (Z,x_t,x_T)  │ → │  (dropout 随机化 Pnet)             │
│  → x̂_{t+1}         │   │ Algorithm 1: LSC 平滑 → IsFeasible │
│  (MSE 蒸馏, 式 2)   │   │ 失败 → Algorithm 3 Replanning:     │
│ 专家 = RRT* 路径     │   │  神经重规划(固定步) → 不可连段回退   │
└────────────────────┘   │  RRT*（Hybrid Replanning）          │
                         └───────────────────────────────────┘
```

四个辅助过程（§IV-B）：**Steering** $\tau(\delta) = (1-\delta)\mathbf{x}_1 + \delta\mathbf{x}_2,\ \forall\delta\in[0,1]$，小步长离散检查两点间直线是否全在 $\mathcal{X}_{free}$（即教程第 03 章 (3.3)–(3.4) 的 steer + 增量碰撞检测）；**IsFeasible** 逐段检查整条路径；**LSC**（Lazy States Contraction）把可直接相连的非相邻状态对直连、删去中间"懒惰"状态（固定迭代次数的路径平滑，非必需组件，帮助接近最优）；**Replanning** 见 §4.4。

**接口划分的理由**：Enet 只吃**环境**（点云 $\mathbf{x}_{obs}$）、Pnet 只吃**查询**（起终点）加编码 $\mathbf{Z}$——环境编码与查询解耦后，同一工作空间的千万次查询共享一次编码（点质量/刚体实验中编码器参数冻结，§V-B），训练数据里的"环境多样性"被吸收进 $\mathbf{Z}$、"规划技能"被吸收进 Pnet。这一分工是后续学习式规划工作（环境编码 + 查询头）的标准接口形状。

## 4. 关键公式推导（按论文原文式号）

**符号表**：

| 符号 | 含义 |
|------|------|
| $\mathbf{x}_{obs} \in \mathcal{X}_{obs}$ | 障碍**点云**观测（$N_{pc} \times d_w$ 向量，$N_{pc}$ 个点、工作空间维数 $d_w$） |
| $\mathbf{Z} \in \mathbb{R}^m$ | Enet 输出的隐空间编码，维数 $m \in \mathbb{N}$ |
| $\boldsymbol{\theta}^e, \boldsymbol{\theta}^d, \boldsymbol{\theta}$ | Enet 编码器、解码器、Pnet 参数 |
| $\mathbf{x}_t,\ \mathbf{x}_T,\ \hat{\mathbf{x}}_{t+1} \in \mathcal{X}_{free}$ | 当前构型、目标构型（论文记目标为 $\mathbf{x}_T$）、Pnet 预测的下一步构型 |
| $D_{obs}, N_{obs}, \lambda$ | 点云数据集、工作空间数、惩罚系数（式 1） |
| $\tau^* = \{\mathbf{x}_0, \mathbf{x}_1, \cdots, \mathbf{x}_T\}$ | 专家演示路径（可行状态序列，完全位于 $\mathcal{X}_{free}$） |
| $N_p, \hat{N}$ | 训练集路径总数、路径数 × 路径长度（总转移数）（式 2） |
| $p \in [0,1]$ | dropout 丢弃概率（本文取 $p = 0.5$） |
| $\tau^a, \tau^b$ | 从起点/目标生成的双向树路径（Algorithm 2） |
| $\delta \in [0,1]$ | steer 插值参数 |

### 4.1 编码器：点云 → 隐空间（式 1）

Enet 以**编码器–解码器 + 重建损失**训练（论文实测**收缩自编码器**（contractive auto-encoder, CAE，文献 [18] Rifai et al.）能学出"规划所需的不变特征空间"、泛化到未见工作空间）：

$$
L_{\text{AE}}\big(\boldsymbol{\theta}^e, \boldsymbol{\theta}^d\big) = \frac{1}{N_{obs}} \sum_{\mathbf{x} \in D_{obs}} \big\lVert \mathbf{x} - \hat{\mathbf{x}} \big\rVert^2 + \lambda \sum_{ij} \big(\theta^e_{ij}\big)^2 \tag{1}
$$

逐项读：第一项是重建误差——把点云 $\mathbf{x}$ 压进 $\mathbf{Z}$ 再由解码器还原 $\hat{\mathbf{x}}$，迫使 $\mathbf{Z}$ 保留障碍几何（依据：自编码器的信息瓶颈逻辑；$N_{obs}$ 个工作空间取平均，故 $\mathbf{Z}$ 是**逐环境**的编码）；第二项是编码器参数的 $\ell_2$ 惩罚（权重衰减，依据：岭正则化——CAE 原始文献中收缩项的实现形式），鼓励对输入扰动的鲁棒/不变表示——这是"没见过的点云也能编码"泛化性的来源。注意编码对象**只有障碍点云**：$\mathbf{Z} = \mathrm{Enet}(\mathbf{x}_{obs})$；起点/目标不进编码器（见下——规格中"$z = \mathrm{Enc}(\mathbf{x}_{obs}, \mathbf{x}_{goal})$"的说法与原文不符）。结构为 3 层线性层 + PReLU（Baxter 例外：编码器与 Pnet **端到端**联训，不用解码器）。

### 4.2 规划器：确定性回归 + dropout 隐式采样（式 2）

Pnet 是前馈深度网络，**确定性**地预测下一步构型：

$$
\hat{\mathbf{x}}_{t+1} = \mathrm{Pnet}\big((\mathbf{x}_t, \mathbf{x}_T, \mathbf{Z});\ \boldsymbol{\theta}\big) \tag{Pnet}
$$

训练目标：对专家路径（RRT* 生成的可行近优路径 $\tau^*$）做**逐步模仿**——最小化预测下一步与专家下一步的均方误差（MSE）：

$$
L_{\text{Pnet}}(\boldsymbol{\theta}) = \frac{1}{N_p} \sum_j \sum_{i=0}^{\hat{N}-1} \big\lVert \hat{\mathbf{x}}_{j,i+1} - \mathbf{x}_{j,i+1} \big\rVert^2 \tag{2}
$$

（外层对训练集全部 $N_p$ 条路径、内层对全部 $\hat{N}$ 个状态转移取平均。）推导其最优解：MSE 的期望极小化解是**条件均值**，逐步写出——对条件分布 $p(\mathbf{x}_{t+1} \mid \mathbf{x}_t, \mathbf{x}_T, \mathbf{Z})$ 极小化 $\mathbb{E}\lVert \mathbf{x}_{t+1} - c\rVert^2$，对 $c$ 求导置零（依据：$\nabla_c \mathbb{E}\lVert \mathbf{x}_{t+1} - c\rVert^2 = -2\,\mathbb{E}[\mathbf{x}_{t+1} - c \mid \cdot]$，梯度为零即 $\mathbb{E}[\mathbf{x}_{t+1} \mid \cdot] = c$；二阶变分为正保证极小）：

$$
\boldsymbol{\theta}^* \ \longrightarrow\ \mathbb{E}\big[\mathbf{x}_{t+1} \mid \mathbf{x}_t, \mathbf{x}_T, \mathbf{Z}\big] \qquad (\text{教程 (9.2) 的同一论断})
$$

**因此纯 Pnet 不构成分布，也无碰撞保证**——式 (2) 从不惩罚输出落进 $\mathcal{X}_{obs}$。且当专家路径在"从两侧绕行障碍"这类场景**多模态**时，条件均值是两模态的平均——可能落在障碍里（这是 MSE 蒸馏的固有风险，§6 再议）。

**dropout = 隐式采样分布**（§IV-B.2 与 §VII-A）：在线与离线都对每个隐藏层以概率 $p = 0.5$ 丢弃隐单元（末层除外）——每次前向等价于从一族"瘦网络"（thinned network）中随机抽取一个子网络，输出在条件均值附近抖动。形式化：在线采样写成

$$
\mathbf{x}_{t+1} \sim q_{\boldsymbol{\theta}}\big(\cdot \mid \mathbf{x}_t, \mathbf{x}_T, \mathbf{Z}\big) \qquad (\text{教程 (9.1) 的 } p_\theta\text{；论文中无显式高斯 } \mathrm{mean}/\mathrm{cov}\text{ 头，随机性完全由 dropout 注入})
$$

三个作用（§VII-A，Fig. 5）：① 确定性 Pnet 在同一状态会重复输出同一（可能不可行）候选、迭代陷入死循环，随机化让重试逃出局部陷阱；② 重规划每步生成**不同**候选，帮助从失败中恢复；③ 副产品——可直接为采样式规划器生成自适应样本（文献 [26]，即作者后续的 neural RRT* 路线）。

### 4.3 神经规划器：采样–连接循环（Algorithm 2）与主动终止

Algorithm 2（Neural Planner）的形式化——增量式**双向**路径生成：

```text
NeuralPlanner(x_start, x_goal, Z):            # 论文 Algorithm 2
  τ^a ← {x_start};  τ^b ← {x_goal};  τ ← ∅
  for i ← 0 to N:                             # 固定轮数预算 N
    x_new  ← Pnet(Z, τ^a(end), τ^b(end))      # 神经采样（4.2）
    τ^a    ← τ^a ∪ {x_new}                    # 无条件加入：信任后验检查
    Connect ← steerTo(τ^a(end), τ^b(end))     # 确定性碰撞检查（(3.3)–(3.4) 机制）
    if Connect: τ ← concatenate(τ^a, τ^b); return τ
    SWAP(τ^a, τ^b)                            # 换边：第 i 轮扩 τ^a，第 i+1 轮扩 τ^b
  return ∅                                    # 主动终止：预算耗尽报告失败
```

结构要点：**网络提议、几何裁决**——采样步出的候选只有通过 steerTo 的确定性检查才升格为已验证路径段（返回路径的每一段都被检查过，这一"可靠性"是后文完备性论证的前提）；SWAP 使两树相向行进，搜索深度近似减半、贪心而快（论文原文："march towards each other ... greedy and fast"）。三点展开：

- **为什么无条件加入 $\mathbf{x}_{new}$**（第 6 行无碰撞检查）：候选的可信性不靠网络保证，而由下游的 steerTo/IsFeasible 统一裁决——把"生成"与"验证"解耦，网络可以激进，几何检查兜底。这也是教程 §09.1 ③"契约 (3.1) 原样保留"的机制层解释。
- **双向的收益**：若单向生成的路径需要 $T$ 个神经步才能逼近目标，双向各走约一半在中间接通（依据：两树相向、每轮各扩一节点），神经前向调用次数近似减半；且两端同时受 Pnet 引导，中间接通点的选点自由度更大。
- **终止条件核对**：论文是**固定轮数 $N$ 的预算耗尽即返回 $\emptyset$**（Algorithm 2 第 4/12 行），而非"连续 $k$ 次失败触发回退"——回退发生在下游 Replanning 阶段（下节）。

### 4.4 两级重规划：神经重修复与经典回退（Algorithm 1/3）

Algorithm 1（MPNet 主流程）：`Z ← Enet(x_obs)` → `τ ← NeuralPlanner(...)` → 成功则 LSC 平滑 + IsFeasible 验证后返回；否则 `τ_new ← Replanning(τ, Z)` → 再次 LSC + IsFeasible → 仍失败**返回 $\emptyset$**。

Algorithm 3（Replanning）逐对检查粗路径的相邻状态对 $(\boldsymbol{\tau}_i, \boldsymbol{\tau}_{i+1})$：可连则保留；不可连则调 **Replanner**（神经重规划）：对断口两端**递归**地重跑"粗路径 + 细化"——先找粗解、再对新一层路径的不可连对继续调用自己，递归深度（步数）固定以限定计算量。两个变体：

- **Neural Replanning（NR）**：只用上述神经递归，超限未修复则失败。
- **Hybrid Replanning（HR）**：神经重规划跑**固定步数**后做可行性测试；仍不可连的状态对交给**经典规划器（本文为 RRT*）**连接——"It performs the neural replanning for the fixed number of steps. ... the non-connectable states in the new path are then connected using a classical motion planner." 这就是"学习先验 + 经典兜底"的落地形态。

### 4.5 完备性继承的逐步论证（§VII-B + 教程 (9.4)）

论文的完备性陈述是**散文式**的（§VII-B）："The completeness guarantees for the proposed method depends on the underline replanning heuristic. The classical motion planner based replanning methods are presented to guarantee the completeness ... Since we use RRT*, our proposed method inherits the probabilistic completeness of RRTs and RRT* [5] while retaining the computational gains." 未给形式证明。逐步化如下（自证，沿用教程 (9.4) 的四步结构）：

1. **可靠性（soundness，与 $\boldsymbol{\theta}$ 无关）**：Algorithm 1 的两条返回路径都先过 IsFeasible 的逐段确定性检查（steerTo 机制）——网络输出从不被直接采信。故**输出必无碰撞**；网络坏了只会变慢/失败，不会给错答案。
2. **问题分解**：设神经阶段失败后路径上留下不可连的连续状态对集合 $G = \{(\mathbf{x}_i, \mathbf{x}_{i+1})\}$。每对恰是教程 (3.1) 意义下的一个标准规划子问题（起 $\mathbf{x}_i$、终 $\mathbf{x}_{i+1}$、同一 $\mathcal{X}_{free}$）。
3. **子问题可行性继承**：若原问题 robustly feasible（存在 $\delta$-clearance 路径 $\sigma^*$，教程第 03 章 Theorem 23 的前提），则 $\sigma^*$ 的对应子段穿过每个间隙，且 clearance 逐点继承（依据：clearance 沿路径逐点定义）——**每个间隙子问题 robustly feasible**。
4. **RRT* 概率完备 + 并集界**：RRT* 对 robustly feasible 问题概率完备、失败概率随样本数指数衰减（第 03 章 Theorem 23、(3.9) 型尾界）；对全部间隙用并集界：

$$
\mathbb{P}[\text{混合规划器失败}] \le \sum_{i \in G} \mathbb{P}\big[\text{RRT}^* \text{ 未在 } n_i \text{ 步内桥接间隙 } i\big] \xrightarrow{\ \min_i n_i \to \infty\ } 0 \qquad (\text{教程 (9.4)})
$$

**继承的边界**（诚实声明）：概率完备是渐近陈述，不承诺截止时间；且它绑定在"经典规划器参与重规划"的变体（HR）上——纯 NR 变体（神经递归有固定步数上限）不具备此保证，实测约 97% 成功率（§VII-C）。MPNet 的真正贡献不是提供保证，而是**把"从零探索 $\mathcal{X}_{free}$"压缩成"补几个局部间隙"**：>97% 的查询 NR 直接解决（$O(1)$），仅约 3% 需要在短段上跑 RRT*——把 $O(n\log n)$ 的全局问题降为实践中远小于 $O(n\log n)$ 的局部问题（§VII-C：混合重规划最坏 $O(n\log n)$、最好 $O(1)$）。

**复杂度推导**（§VII-C，逐条）：① 神经网络单次前向的浮点运算量只依赖网络结构（层数/宽度固定），与问题规模无关——$O(1)$；Algorithm 1 的 Enet + NeuralPlanner 两步复杂度不超过 $O(1)$。② LSC 是固定迭代数的路径平滑（可行路径长度有限，依据：路径有正标量长度且步长有限），$O(1)$；论文注明其为非必需组件，加入只为接近最优。③ 神经重规划的递归深度固定——$O(1)$。④ 经典回退用 RRT*：树上近邻与重布线操作合计 $O(n\log n)$（$n$ 为树中样本数，第 03 章 Lemma 42 同源结论）。⑤ 故混合重规划最坏 $O(n\log n)$、最好 $O(1)$；且 RRT* 只在整体路径的**短段**上执行，问题规模被全局骨架压缩，实际耗时"远小于 $O(n\log n)$"。

### 4.6 与神经 RRT*/BIT* 启发式采样器的对比定位

- **BIT* [9]**：启发式（启发函数引导的隐随机几何图批量搜索）——启发是**人工设计**的（距离/代价的解析函数），对障碍几何无感知；MPNet 的引导是**从数据学出**的（点云编码感知障碍）。论文 Table I 显示 BIT* 全面优于 Informed-RRT*，故以 BIT* 为主要对照。
- **神经采样分布路线 [16]（Ichter et al.）与 neural RRT* [26]**：学出的采样分布作为经典规划器**内部组件**（替换采样器），框架仍是完整的探索循环；MPNet 更激进——网络直接**生成路径骨架**，经典规划器只做局部修补。谱系上 neural RRT* [26]（Qureshi & Yip, IROS 2018）正是本文 §VII-A "dropout 随机性可为采样规划器生成自适应样本"的落地。
- 用教程 §09.3 的话说：三者的学习组件都在 L2 层的"采样/引导"组件上，差别只在**组件的接口大小**——采样分布 < 路径骨架。

## 5. 实验与结果解读

- **数据（§V-A）**：每类环境（simple 2D、complex 2D、complex 3D、rigid-body）生成 **110 个工作空间**，每个用 RRT* 生成 **5000 条**无碰撞近优路径；训练用 100 个工作空间 × 4000 条。测试集两种：seen-$X_{obs}$（训练见过的 100 个工作空间、各 200 个**未见**起终点）与 unseen-$X_{obs}$（**完全未见**的 10 个工作空间 × 2000 起终点）。Baxter：10 个挑战性仿真环境、各 900 训练路径 + 100 测试路径；障碍点云由 Kinect 深度相机经 PCL 获取——**真实点云输入**。
- **点质量/刚体对比（Table I，§VI）**：MPNet（NR/HR）所有测试用例平均耗时**约 1 秒**且几乎不随维度变化（NR 均值 0.11–0.48 s）；Informed-RRT* 5.4–30.4 s、BIT* 2.6–11.9 s，且两者随维度显著增长。平均而言 MPNet 比 Informed-RRT* 快约 **40 倍**、比 BIT* 快约 **20 倍**（Table I 右列逐环境 17.9–36.1 倍）。unseen-$X_{obs}$ 行与 seen 行耗时相当——**对未见工作空间泛化**（式 (1) 的不变特征空间的实效）。维度最高的 rigid 场景对比最悬殊：NR 0.32–0.33 s vs BIT* 11.1–11.9 s（右列 34.7/36.1 倍）——经典方法的耗时随维度增长、MPNet 持平，正是"计算时间与规划问题维度无关"这一论文主张的数据支点。
- **成功率（§VI）**：HR 与 NR 的平均精度分别为 **100% 与 97%**（五次试验标准差约 0.4%）——神经阶段覆盖绝大多数查询，回退是少数派。
- **一致性（Fig. 4）**：跨 100 个工作空间的逐题耗时曲线显示 MPNet 贴近横轴、几乎无离群尖峰，BIT* 波动剧烈且在刚体场景出现数十秒尖峰——"快"不止体现在均值，更体现在**方差**（对在线闭环，尾延迟比均值更致命，教程第 05 章视角）。
- **7-DOF Baxter（§VI，Fig. 1）**：MPNet 平均约 **1 秒、85% 成功率**；BIT* 平均约 9 秒、**56% 成功率**（以"路径长度在 MPNet 的 40% 范围内"为口径），要达到 MPNet 平均路径长度的 10% 范围需**数分钟**；Fig. 1 说明中给出"BIT* 找到代价可比的可行路径平均需 **3.1 分钟**，MPNet 不到 1 秒"。教程 §09.1 引用的"BIT* 3.1 min vs MPNet <1 s"即出自 Fig. 1 说明。
- **计时口径提示（§V，诚实读表）**：点质量/刚体实验中，Informed-RRT* 与 BIT* 是论文团队的 **Python 实现**，与 MPNet 的 **CPU-time** 对比（实现语言/运行环境不同，倍数宜作定性参考）；Baxter 实验则统一为 **C++ OMPL 的 BIT* vs C++ 的 MPNet**（MoveIt!/ROS），口径更严——"至少 20× 快于 BIT*"的结论在两组实验中方向一致。
- **复杂度（§VII-C）**：神经网络在线执行 $O(1)$；神经重规划固定步数 $O(1)$；RRT* 回退 $O(n\log n)$——HR 最坏 $O(n\log n)$、最好 $O(1)$，且实测 >97% 停留在 $O(1)$ 侧。

## 6. 局限与后续影响

- **训练分布外的场景**：泛化实验限定在"同类箱体障碍、不同布局"（unseen 工作空间）内；障碍几何类型、传感器噪声形态剧变的场景未被验证——编码器的不变特征是相对训练族而言的。**专家数据偏差**：专家全部来自 RRT*（每工作空间 5000 条），学到的"下一步"分布继承 RRT* 的路径风格与覆盖偏好（采样树偏向开阔区域），MSE 回归对**多模态**专家路径还有条件均值式的抹平风险（绕行左侧/右侧各半时条件均值可能落在障碍上—— dropout 随机化缓解但不消除此问题）。
- **纯学习模式不完备**：NR 失败即返回 $\emptyset$；保证完全寄托在经典回退上——最坏情形退化为 RRT* 且白付神经阶段成本（教程 §09.1 ③ 的选型代价）。
- **逐环境训练成本**：Baxter 实验按环境收集数据（每环境 900 条训练路径、10 个环境），换新工作空间需重新采数据/微调（点质量类靠 Enet 的编码器–解码器训练获得跨工作空间泛化，Baxter 是端到端联训、编码器不独立泛化——§V-B 的架构差异暗示了这条边界）。
- **工程边界**：点云表示限制在静态障碍；kinodynamic（带动力学约束的）规划被论文列为开放问题（结论 §VIII：未来工作为 actor-critic 规划 + Fastron 代理碰撞检查 [27][28]、动态环境）。
- **后续影响**：确立"神经先验 + 经典回退 + 完备性继承"的论文范式，直接延续为 neural RRT*/MPNet 后续系列；"dropout 即采样器"的轻量随机化被后续学习式采样工作广泛沿用；教程 §09.1/§09.3 以它为 L2 层"学习先验 + 经典兜底"的标本。

## 7. 与本项目对照

**教程第 09 章 §09.1 映射表**（本章公式 → 教程编号）：

| 论文内容 | 论文位置 | 教程位置 | 说明 |
|---------|---------|---------|------|
| 采样分布（dropout 随机化） | §IV-B.2、§VII-A | (9.1) | 教程的 $p_\theta(\cdot \mid q_t, q_{goal}, \mathbf{Z})$ 即本文 4.2 的 $q_\theta$ |
| MSE 蒸馏损失 | 式 (2) | (9.2) | 逐步模仿专家（RRT*）下一状态 |
| 双向采样–连接循环 | Algorithm 2 | (9.3) | steerTo 线段检查 = (3.3)–(3.4) 机制 |
| 完备性继承 | §VII-B（散文） | (9.4) 四步论证 | 教程把论文的散文陈述形式化为并集界 |
| Hybrid Replanning | §IV-B.7 | §09.1 ②/③、"经典兜底" | 不可连段交 RRT* |

**"学习先验 + 经典兜底"的工程范式（§09.3）**：第 09 章混合范式表的 L2 行——"经典：RRT*（完备 + 渐近最优，第 03 章）｜学习：MPNet 神经采样（09.1）｜混合：神经先验 + RRT* 回退"——本文即该行的原始出处。三条依据在论文中全部可对应：**保证守恒**（学习组件放在采样分布这个保证最便宜的组件，输出的保证由 IsFeasible/RRT* 两个经典组件承载）；**误差局部化**（失败模式被限制在"间隙补不上"，可定位、可回退）；**数据与算力放在收益最大处**（一次性离线蒸馏吃掉典型查询的探索成本）。与第 03 章 (3.11) 的 goal bias 对照：$p_g$ 直连是无信息的固定偏置，MPNet 是"看得到障碍与目标"的自适应偏置——期望等待时间从 $1/p_g$ 轮降到典型情形一两步。

**选型三候选（教程 §09.1 ③ 的论文对应）**：(a) 纯采样（RRT*/BIT*）——有保证但慢，§2 的痛点原样保留；(b) 纯学习（NR 模式）——毫秒级出候选，但不完备也不保证无碰撞（式 (2) 的 MSE 训练从不惩罚输出进障碍），实测 97% 成功率、失败即返回 $\emptyset$；(c) 混合（HR 模式）——保住计算优势的同时实测 100% 成功率；代价是最坏退化为 RRT* 且离线需一次性专家数据。这一对比正是 §09.3"经典为骨、学习为肉"判据的实验证据。

**与 RoboTwin 栈的关系**：RoboTwin 双臂任务的自由空间搬运中，L2 规划若需在线重规划（教程第 10 章全栈表 L2 行），MPNet 式"毫秒级骨架 + RRT* 补隙"正是把第 03 章秒级规划压进控制周期的路线；其点云输入接口与 RoboTwin 的深度观测一致。跨主题精读见 [RoboTwin 精读总索引](../../robotwin/精读/README.md)。

## 配套阅读

- 教程：[第 09 章｜前沿学习式规划与腿式控制](../09_前沿学习式规划与腿式控制.md)（§09.1 全节即本文的教程投影；§09.3 混合范式全景表）｜ [第 03 章｜采样式规划](../03_运动规划-i采样式规划.md)（(3.1) 契约、(3.3)–(3.4) steer 与碰撞检查、(3.7) 收缩半径、(3.9) 失败尾界、(3.11) goal bias、Theorem 23）
- 同目录精读：[RRT*（IJRR 2011）](RRTstar_IJRR2011.md)（Theorem 23/38——完备性继承的法源）｜ [RRT（TR 1998）](RRT_TR1998.md)（steer/增量检查的原始形态）
- 论文内对照：BIT*（Gammell et al., ICRA 2015，[9]）与 Informed-RRT*（IROS 2014，[25]）——Table I 的两个基准；neural RRT*（Qureshi & Yip, IROS 2018，[26]）——本文思想的采样器形态；Ichter et al.（ICRA 2018，[16]）——学习采样分布的先行路线
- 相邻精读：[Crocoddyl（ICRA 2020）](Crocoddyl_ICRA2020.md)——同一教程第 04/09 章"经典为骨、学习为肉"谱系在 L3（轨迹层）的对照面：Crocoddyl 靠解析模型与结构化递推换速度，MPNet 靠数据先验换速度

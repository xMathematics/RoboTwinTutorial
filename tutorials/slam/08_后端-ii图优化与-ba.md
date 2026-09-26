# 第 08 章｜后端 II：图优化与 BA

第 07 章留下两个死结：滤波范式在线维护联合分布，协方差与更新代价随路标数平方增长（(7.7)）；线性化一次即固定，历史信息被"烧掉"，无法回头修正。本章把状态估计换写成**非线性最小二乘**：变量是节点、观测是因子的**因子图**（factor graph），光束法平差（bundle adjustment, BA）是其特例，用 Gauss-Newton / Levenberg-Marquardt 迭代求解。四个知识点层层递进：目标函数的概率语义（08.1）、求解器的完整推导（08.2）、增量为何必须定义在流形上（08.3）、以及让万级路标可解的稀疏 BA 与 Schur 补（08.4，压轴）。本章是全书后端核心，产出的公式将直接被第 09 章位姿图优化与第 10 章系统实战回引。

## 08.1 从滤波到图优化：状态估计即最小二乘

**① 问题场景** 手头有：前端（第 05 章）在关键帧处给出的位姿初值、三角化路标，以及成千上万条"路标 $\mathbf{p}_j$ 在关键帧 $i$ 中被匹配到像素 $\mathbf{z}_{ij}$"。缺的是：让**全部**观测同时一致的位姿与地图——每个观测都有噪声，逐个满足会互相打架。滤波路线（第 07 章）有两道过不去的坎：其一，路标上万时协方差 $O(N^2)$ 不可维护（(7.7)）；其二，滤波把线性化点固定在预测值、每条信息用完即弃——回环检测（第 09 章预告）发现"走回了旧地"后需要回头修正整段历史，滤波框架下无处安放。想要的：一个可反复重线性化、可批量重算、规模可扩展的估计框架。

**② 解决方法** 把每个观测写成残差（residual）$\mathbf{e}_k(\mathbf{x})$——观测与模型预测之差；状态估计 = 非线性最小二乘：找 $\mathbf{x}$ 使全部残差的信息加权平方和最小。变量节点 + 观测因子的二部图即因子图；**同时**优化位姿与路标的重投影误差（reprojection error）最小二乘就是 BA。求解有两种模式：批量 BA（关键帧 + 全部路标一起优化）与滑窗（sliding-window，只优化最近若干关键帧，旧状态边缘化——08.4 末尾回到这点）。

**③ 选型理由** 核心权衡是"重线性化的收益 vs 计算开销"。滤波只线性化一次，线性化误差被永久固化；图优化每次迭代都在**当前最新估计**处重新线性化（08.3 的循环），迭代收敛到局部最优——精度收益与"可回头修正历史"的能力是质的。开销是多次迭代 + 每次解一个大型线性方程组。敢付这个开销有两个前提：初值由前端 VO 提供、已足够靠近真值（第 05 章）；方程组有可利用的稀疏结构、单次求解可控（08.4）。与 07.2 的 SEIF 对比：SEIF 靠近似稀疏化维持稀疏、有信息损失；因子图**无损保留每一条观测**，只在求解时利用稀疏——这是"保留因子"胜过"维护联合分布"的根本原因。

**④ 理论依据** Triggs, McLauchlan, Hartley, Fitzgibbon, *"Bundle Adjustment — A Modern Synthesis"*, ICCV 1999；高翔《视觉 SLAM 十四讲》第 10 讲（后端 1）、第 11 讲（后端 2）；g2o 通用图优化库（Kümmerle et al., *"g2o: A General Framework for Graph Optimization"*, ICRA 2011）。

**⑤ 完整推导**

重投影残差（依据：第 04 章针孔投影模型 $\pi(\cdot)$）：
$$
\mathbf{e}_{ij}\big( \mathbf{T}_i, \mathbf{p}_j \big) = \mathbf{z}_{ij} - \pi\big( \mathbf{T}_i\, \mathbf{p}_j \big)
\tag{8.1}
$$
观测模型：$\mathbf{z}_{ij} = \pi(\mathbf{T}_i \mathbf{p}_j) + \mathbf{v}$，$\mathbf{v} \sim \mathcal{N}(\mathbf{0}, \Sigma_{ij})$（依据：第 03 章 (3.1) 观测方程的具体化）。

最大后验（maximum a posteriori, MAP）：取弱先验（均匀或大方差高斯，对结果影响可忽略），
$$
p(\mathbf{x} \mid \mathbf{z}) \;\propto\; p(\mathbf{z} \mid \mathbf{x}) \;=\; \prod_k \mathcal{N}\big( \mathbf{z}_k;\, h_k(\mathbf{x}),\, \Sigma_k \big)
$$
（依据：贝叶斯定理 + 观测给定状态条件独立——每条匹配独立产生，因子分解同 07.2）。把密度取 $-\log$（依据：$\log$ 单调，$\arg\max$ 变 $\arg\min$）：
$$
-\log \mathcal{N}(\mathbf{z};\, h,\, \Sigma) = \tfrac{1}{2}(\mathbf{z} - h)^\top \Sigma^{-1} (\mathbf{z} - h) + \tfrac{1}{2} \log|\Sigma| + \tfrac{d}{2}\log 2\pi
$$
后两项与 $\mathbf{x}$ 无关，丢弃（依据：不影响 $\arg\min$）；最小二乘惯例再省略 $\tfrac12$（依据：正常数缩放不改变极小点），得目标函数：
$$
\min_{\mathbf{x}} \; \sum_k \big\| \mathbf{e}_k(\mathbf{x}) \big\|_{\Sigma_k}^2 \;=\; \sum_k \mathbf{e}_k^\top \Sigma_k^{-1} \mathbf{e}_k \;=\; \sum_k \mathbf{e}_k^\top \Lambda_k\, \mathbf{e}_k
\tag{8.2}
$$
权重正是**信息矩阵** $\Lambda_k = \Sigma_k^{-1}$。概率语义与前两章完全一致：第 03 章的 KF/EKF 是同一 MAP 的递归解（线性高斯下"后验信息 = 先验信息 + 似然信息"，见 (7.10)）；第 07 章的 $\Lambda = \Sigma^{-1}$ 在这里以"每条观测的权重"身份重现——图优化的目标函数就是"把每条观测的信息加总"的对数概率翻译。区别只在求解方式：滤波沿时间递归维护分布，图优化把全部因子摆开、一次解出，且允许反复重线性化。

## 08.2 Gauss-Newton 与 Levenberg-Marquardt

**① 问题场景** 目标函数 (8.2) 中 $\mathbf{e}_k$ 是位姿与路标的非线性函数（投影 $\pi$ 与旋转都非线性），闭式解不存在。手头有前端给的初值 $\mathbf{x}_0$；想要一个迭代算法：每步给出让代价下降的增量 $\Delta\mathbf{x}$。

**② 解决方法** Gauss-Newton（GN）：在当前估计处把残差一阶展开，解二次近似问题的最优增量；Levenberg-Marquardt（LM）：在系数矩阵上加阻尼 $\lambda I$，并按"实际下降是否达到预测"自适应调节 $\lambda$。g2o 同时提供 GN 与 LM，工程实现普遍选用 LM（ORB-SLAM2 的局部 BA 即基于 g2o 的 LM，见配套 PDF）。

**③ 选型理由** 三个候选对比：(a) 梯度下降——只用一阶信息，窄谷地形锯齿收敛慢；(b) 牛顿法——需要真实 Hessian，其中含残差二阶导项（见 ⑤ 的展开），大规模时计算与存储昂贵，且不加修正可能不正定、迭代可能发散；(c) GN——用 $J^\top \Sigma^{-1} J$ 近似 Hessian，省去残差二阶项（在最优附近残差小时合理），但远离最优或线性化病态时步长不可靠。LM 在 GN（$\lambda \to 0$）与阻尼梯度下降（$\lambda$ 大）之间插值，保证每步线性系统可解、代价下降——工程上几乎是标配。代价：$\lambda$ 需要调节策略（信赖域更新，定性）；误匹配多时"小残差"假设退化，实际系统用鲁棒核函数兜底（第 10 章实战展开）。

**④ 理论依据** Gauss-Newton 方法（Gauss 1809 的归算思想；现代表述见 Nocedal & Wright, *Numerical Optimization*, 2nd ed., ch10）；LM（Levenberg 1944; Marquardt 1963）；信赖域方法（trust region，Nocedal & Wright ch4）；《视觉 SLAM 十四讲》第 6 讲（非线性优化）、第 10 讲（后端应用）。

**⑤ 完整推导**

把全部残差堆叠为 $\mathbf{e}(\mathbf{x}) \in \mathbb{R}^{D}$，各观测信息矩阵拼成块对角 $\Sigma^{-1} = \mathrm{blkdiag}(\Lambda_k)$（依据：$\sum_k \mathbf{e}_k^\top \Lambda_k \mathbf{e}_k = \mathbf{e}^\top \Sigma^{-1} \mathbf{e}$，求和改写为块对角二次型）。记 $F(\mathbf{x}) = \mathbf{e}^\top \Sigma^{-1} \mathbf{e}$。

**残差一阶展开。** 在当前点 $\mathbf{x}$ 处一阶 Taylor 展开（依据：多元函数 Taylor 展开取一阶项）：
$$
\mathbf{e}(\mathbf{x} + \Delta) \;\approx\; \mathbf{e}(\mathbf{x}) + J\Delta, \qquad J = \left.\frac{\partial \mathbf{e}}{\partial \mathbf{x}}\right|_{\mathbf{x}}
\tag{8.3}
$$

**目标函数二阶近似，逐项展开。** 代入（依据：矩阵乘法展开，共四项）：
$$
F(\mathbf{x}+\Delta) \approx (\mathbf{e} + J\Delta)^\top \Sigma^{-1} (\mathbf{e} + J\Delta) = \mathbf{e}^\top \Sigma^{-1} \mathbf{e} + (J\Delta)^\top \Sigma^{-1} \mathbf{e} + \mathbf{e}^\top \Sigma^{-1} (J\Delta) + \Delta^\top J^\top \Sigma^{-1} J\, \Delta
$$
中间两项互为转置且相等（依据：$\Sigma^{-1}$ 对称，标量的转置等于自身），合并得
$$
F(\mathbf{x}+\Delta) \;\approx\; F(\mathbf{x}) + 2\, (J\Delta)^\top \Sigma^{-1} \mathbf{e} \;+\; \Delta^\top \big( J^\top \Sigma^{-1} J \big)\, \Delta
\tag{8.4}
$$
记 $\mathbf{g} = J^\top \Sigma^{-1} \mathbf{e}$，$H = J^\top \Sigma^{-1} J$。关于 $H$ 的含义（依据：对 $F = \mathbf{e}^\top \Sigma^{-1} \mathbf{e}$ 用链式法则求两次导）：真实 Hessian $= J^\top \Sigma^{-1} J + \sum_k \big( \nabla^2 \mathbf{e}_k \big)$-加权项，第二项 $\propto \mathbf{e}_k$ 本身，在最优附近 $\mathbf{e}_k \to \mathbf{0}$ 相对第一项是小量——GN 正是舍弃了这一项（所以 (8.4) 是 $\Delta$ 的二次近似，但不是 $F$ 的完整二阶 Taylor 展开）。

**对 $\Delta$ 求导置零。** 由 (8.4)（依据：矩阵微积分 $\partial(\mathbf{g}^\top \Delta)/\partial\Delta = \mathbf{g}$、$\partial(\Delta^\top H \Delta)/\partial\Delta = 2H\Delta$，$H$ 对称；极小点梯度为零）：
$$
H\, \Delta = -\mathbf{g}, \qquad \text{即} \qquad \big( J^\top \Sigma^{-1} J \big)\, \Delta = -\, J^\top \Sigma^{-1} \mathbf{e}
\tag{8.5}
$$
等价的配方视角（依据：二次函数配方）：
$F(\mathbf{x}+\Delta) \approx F(\mathbf{x}) + \big( \Delta + H^{-1}\mathbf{g} \big)^\top H \big( \Delta + H^{-1}\mathbf{g} \big) - \mathbf{g}^\top H^{-1} \mathbf{g}$，$H \succ 0$ 时最小点 $\Delta^{*} = -H^{-1}\mathbf{g}$。(8.5) 即**正规方程**（normal equations）。

**LM：阻尼与信赖域。** 解
$$
\big( H + \lambda I \big)\, \Delta = -\mathbf{g}
\tag{8.6}
$$
信赖域（trust region）解释：$\lambda$ 大 → $\Delta \approx -\mathbf{g}/\lambda$，小步沿梯度方向（阻尼项主导，近似梯度下降）；$\lambda \to 0$ → 退回 GN。自适应调节（依据：LM 原始文献的比值策略，定性描述）：比较实际下降 $F(\mathbf{x}) - F(\mathbf{x} + \Delta)$ 与线性化模型预测的下降 $2\mathbf{g}^\top \Delta + \Delta^\top H \Delta$；实际下降充分接近预测 → 线性化模型可信 → 减小 $\lambda$（扩大步长）；反之增大 $\lambda$（收缩步长）。

**为何 $H$ 不正定时 LM 仍可解。** 两步。其一，$H \succeq 0$：对任意 $\mathbf{v}$，$\mathbf{v}^\top H \mathbf{v} = (J\mathbf{v})^\top \Sigma^{-1} (J\mathbf{v}) \ge 0$（依据：$\Sigma^{-1} \succ 0$，其二次型非负）。其二，特征值下界：
$$
\mu_{\min}\big( H + \lambda I \big) = \mu_{\min}(H) + \lambda \;\ge\; \lambda \;>\; 0
\tag{8.7}
$$
（依据：对称矩阵的最小特征值即 Rayleigh 商 $\min_{\mathbf{v} \neq \mathbf{0}} \mathbf{v}^\top (H + \lambda I) \mathbf{v} / \|\mathbf{v}\|^2$ 的最小值，单位位移把整个谱平移 $\lambda$。）故 $H + \lambda I \succ 0$：Cholesky 分解存在、线性系统唯一可解——与 $H$ 是否退化无关。且 $\Delta$ 必为下降方向：对 (8.6) 两边左乘 $\Delta^\top$（依据：代数恒等变形）：
$$
\mathbf{g}^\top \Delta = -\, \Delta^\top (H + \lambda I)\, \Delta = -\Big( \underbrace{\Delta^\top H\, \Delta}_{\ge\, 0} + \lambda \|\Delta\|^2 \Big) \;\le\; -\lambda \|\Delta\|^2 \;<\; 0 \quad (\Delta \neq \mathbf{0})
$$
而 $\nabla F = 2\mathbf{g}$（依据：(8.4) 的线性项系数即梯度），故方向导数 $\nabla F^\top \Delta = 2\, \mathbf{g}^\top \Delta < 0$——沿 $\Delta$ 走一阶代价必降（依据：方向导数定义）。

## 08.3 流形上的迭代：增量为何不能直接加

**① 问题场景** 08.2 解出的 $\Delta$ 是欧氏向量空间里的增量，隐含"新状态 = 旧状态 + $\Delta$"。但状态中位姿 $\mathbf{T}_i \in SE(3)$，旋转是带矩阵约束的对象（第 02 章）：直接把矩阵当向量加，得到的东西可能根本不是一个刚体变换——在非法状态上计算重投影，目标函数失去几何意义。手头是向量 $\Delta$，想要一个既保持流形约束、又能复用 08.2 求解器的更新方式。

**② 解决方法** 把增量定义在切空间（tangent space，即李代数 $\mathfrak{se}(3)$）上：优化变量是扰动向量 $\delta\boldsymbol{\xi} \in \mathbb{R}^6$，更新用指数映射左乘 $\mathbf{T} \leftarrow \exp(\Delta\boldsymbol{\xi}^\wedge)\, \mathbf{T}$；雅可比取"残差对扰动量"的导数。

**③ 选型理由** 替代方案对比：(a) 直接加 + 事后 SVD 投影回 $SO(3)$——每步多一次投影，且非法点上的代价已污染目标函数；(b) 四元数参数化 + 归一化——可行，但归一化非线性、雅可比更繁琐；(c) 李代数扰动——指数映射**精确**落在流形上（不靠事后修补），且 $\delta\boldsymbol{\xi} \in \mathbb{R}^6$ 无约束，08.2 的 GN/LM 原样可用。代价：雅可比要多走一层链式法则与 BCH 一阶近似（只做一阶，第 02 章已推导）。

**④ 理论依据** 李群李代数：$SO(3)$ 定义与指数映射（第 02 章 (2.3)、(2.17)）、BCH 一阶近似与左扰动导数（第 02 章 (2.20)-(2.23)、(2.26)）；流形优化的系统表述见 Absil, Mahony, Sepulchre, *Optimization Algorithms on Matrix Manifolds*, Princeton Univ. Press, 2008。

**⑤ 完整推导**

**直接加为什么不行。** 检验正交约束（依据：$SO(3)$ 定义 (2.3)：$R^\top R = I$；矩阵乘法展开）：
$$
(R + \Delta)^\top (R + \Delta) = R^\top R + R^\top \Delta + \Delta^\top R + \Delta^\top \Delta = I + R^\top \Delta + \Delta^\top R + O(\|\Delta\|^2)
\tag{8.8}
$$
一阶项 $R^\top \Delta + \Delta^\top R$ 一般非零（例：$R = I$、$\Delta$ 取非对称阵时该项 $= \Delta + \Delta^\top \neq 0$）——**正交性在一阶就被破坏**（回引第 02 章 (2.3)）。若坚持要一阶项为零，需 $R^\top \Delta$ 反对称（依据：$(R^\top \Delta)^\top = \Delta^\top R$），即 $\Delta = R\, \Omega$、$\Omega^\top = -\Omega$：可行增量方向只剩 3 维反对称阵张成的切空间，且切向量只是"一阶近似方向"，要**精确**回到流形必须过指数映射（依据：第 02 章 (2.14)-(2.17)，$\exp(\boldsymbol{\phi}^\wedge) \in SO(3)$ 有闭式）。位姿同理（$SE(3)$，第 02 章 (2.25)）。

**正确的更新。**
$$
\mathbf{T} \leftarrow \exp\big( \Delta\boldsymbol{\xi}^\wedge \big)\, \mathbf{T}
\tag{8.9}
$$
（依据：$\exp(\Delta\boldsymbol{\xi}^\wedge) \in SE(3)$（第 02 章 (2.25) 的指数映射），群封闭性保证乘积仍 $\in SE(3)$——流形约束**精确**保持，不需事后修补。）

**残差对扰动量的雅可比。** 取重投影残差 $\mathbf{e}(\mathbf{T}) = \mathbf{z} - \pi(\mathbf{T}\mathbf{p})$（世界系路标 $\mathbf{p}$），左扰动参数化 $\tilde{\mathbf{T}} = \exp(\delta\boldsymbol{\xi}^\wedge)\, \mathbf{T}$。链式法则（依据：复合函数求导）：
$$
J_{\xi} = \left.\frac{\partial\, \mathbf{e}(\tilde{\mathbf{T}})}{\partial\, \delta\boldsymbol{\xi}}\right|_{\delta\boldsymbol{\xi} = \mathbf{0}} = -\, \frac{\partial \pi}{\partial \mathbf{q}} \cdot \left.\frac{\partial \big( \exp(\delta\boldsymbol{\xi}^\wedge)\, \mathbf{T}\mathbf{p} \big)}{\partial\, \delta\boldsymbol{\xi}}\right|_{\mathbf{0}}, \qquad \mathbf{q} = \mathbf{T}\mathbf{p}
$$
第二因子正是第 02 章的左扰动导数 (2.26)：
$$
\left.\frac{\partial \big( \exp(\delta\boldsymbol{\xi}^\wedge)\, \mathbf{T}\mathbf{p} \big)}{\partial\, \delta\boldsymbol{\xi}}\right|_{\mathbf{0}} = \Big[\ I \ \Big|\ \ -(\mathbf{q})^\wedge\ \Big]
$$
（依据：(2.26) 的成立链条：BCH 一阶近似 (2.21) 给出 $\exp(\delta\boldsymbol{\xi}^\wedge) \exp(\boldsymbol{\xi}^\wedge) = \exp\big( (\boldsymbol{\xi} + J_l^{-1} \delta\boldsymbol{\xi})^\wedge \big) + O(\|\delta\boldsymbol{\xi}\|^2)$；小扰动下 $J_l^{-1} \approx I - \tfrac12 \boldsymbol{\phi}^\wedge \approx I$，即 (2.23)——扰动被平移到当前点的切空间，不引入交叉项。）合并：
$$
\mathbf{e}\big( \exp(\delta\boldsymbol{\xi}^\wedge)\, \mathbf{T} \big) \;\approx\; \mathbf{e}(\mathbf{T}) + J_\xi\, \delta\boldsymbol{\xi}, \qquad
J_\xi = -\, \frac{\partial \pi}{\partial \mathbf{q}} \Big[\ I \ \Big|\ \ -(\mathbf{q})^\wedge \Big]
\tag{8.10}
$$
（$\partial \pi / \partial \mathbf{q}$ 是投影对空间点的导数，由第 04 章针孔模型直接求导。）注意 (8.9) 与 (8.10) 用的是**同一个**左扰动参数化：更新方向与线性化方向一致，这是流形迭代自洽的前提。

**迭代循环。** (a) 在当前 $\mathbf{T}$ 处按 (8.10) 算 $J_\xi$、组装 (8.5) 的 $H, \mathbf{g}$；(b) 解 $\delta\boldsymbol{\xi}^{*}$（必要时用 LM (8.6)）；(c) 左乘更新 (8.9)；(d) 回到 (a) 重新线性化，直到 $\|\delta\boldsymbol{\xi}\|$ 足够小。第 (d) 步的"重线性化"正是 08.1 ③ 所说收益的落地点：滤波没有这个循环，线性化点一旦选定即不可撤回。

## 08.4 稀疏 BA 与 Schur 补（压轴）

**① 问题场景** 全局 BA 的变量规模：$m$ 个关键帧位姿（各 6 维）+ $n$ 个路标（各 3 维），且 $n \gg m$（路标通常比关键帧多一到两个数量级）。正规方程 (8.5) 的 $H$ 若按稠密矩阵对待，是 $(6m + 3n)$ 维方阵，直接求逆 $O\big( (6m+3n)^3 \big)$——上万路标时不可承受。但 08.1 的因子图早已提示：每条残差只连接 1 个位姿和 1 个路标，$H$ 必有可利用的结构。想要：把求解代价压到主要由小得多的位姿块主导。

**② 解决方法** 按 [位姿 | 路标] 把 $H$ 分块，证明路标-路标块是块对角（⑤）；用 **Schur 补**（Schur complement，又称分块高斯消元）先消掉全部路标增量，得到只含位姿的约简方程（$6m$ 维，远小）；解出 $\Delta\mathbf{c}$ 后回代得 $\Delta\mathbf{p}$。这是 g2o 与 ORB-SLAM2 局部 BA 的标准求解路径。

**③ 选型理由** 复杂度对比（⑤ 推导）：稠密求解 $O\big( (m+n)^3 \big)$（块维口径）vs Schur 补 $O\big( m^3 + n\, m^2 \big)$ + 稀疏处理；$n \gg m$ 时量级优势明显，且约简后的小系统才谈得上配合稀疏 Cholesky（sparse Cholesky）进一步压缩。代价：约简矩阵会引入 fill-in（原本不直接耦合的位姿之间产生耦合块），规模由共视结构决定（定性）——这正是 07.2 "边缘化破坏条件独立"的又一次出现（⑤ 末尾）。与"直接对 $H$ 做稀疏 Cholesky"对比：也可行，但 Schur 补显式利用块结构、先把变量规模砍到位姿量级，是 BA 的标准解（Triggs et al. 1999）。

**④ 理论依据** Schur 补与分块消元（线性代数标准结果；BA 语境的系统处理见 Triggs et al., *"Bundle Adjustment — A Modern Synthesis"*, 1999，分块稀疏求解部分）；实现参照 g2o（Kümmerle et al. 2011）与 ORB-SLAM2（Mur-Artal & Tardós, *IEEE T-RO* 2017，配套 PDF）；《视觉 SLAM 十四讲》第 10 讲（$H$ 稀疏结构）、第 11 讲（位姿图与边缘化）。

**⑤ 完整推导**

**雅可比分块拼接。** 每条残差 $\mathbf{e}_{ij}$ 只含 $\mathbf{T}_i$ 与 $\mathbf{p}_j$ 两个变量（依据：(8.1) 表达式），其对全状态的雅可比行块只有两段非零。按 [位姿列块 | 路标列块] 拼接（记 $\mathbf{c}$ = 全部位姿、$\mathbf{p}$ = 全部路标）：
$$
J = \big[\, J_c \;\;\; J_p \,\big], \qquad J_c \in \mathbb{R}^{D \times 6m}, \quad J_p \in \mathbb{R}^{D \times 3n}
$$

**$H$ 的分块。** 代入 (8.5) 的 $H = J^\top \Sigma^{-1} J$（依据：分块矩阵乘法；$\Sigma^{-1} = \mathrm{blkdiag}(\Lambda_k)$ 对称）：
$$
H = \begin{bmatrix} J_c^\top \Sigma^{-1} J_c & J_c^\top \Sigma^{-1} J_p \\ J_p^\top \Sigma^{-1} J_c & J_p^\top \Sigma^{-1} J_p \end{bmatrix} = \begin{bmatrix} H_{cc} & H_{cp} \\ H_{pc} & H_{pp} \end{bmatrix}
\tag{8.11}
$$

**$H_{pp}$ 是块对角。** 看交叉块元素（依据：矩阵乘法按行块 × 列块求和）：
$$
\big[ H_{pp} \big]_{(j,\, \ell)} = \sum_{k} \big( J_p^{(k)}[j] \big)^\top \Lambda_k\, J_p^{(k)}[\ell], \qquad j \neq \ell
$$
其中 $J_p^{(k)}[j]$ 是第 $k$ 条残差雅可比在路标 $j$ 列块上的片段。每条残差只涉及**一个**路标（依据：因子图结构，(8.1)），故 $J_p^{(k)}[j]$ 与 $J_p^{(k)}[\ell]$ 至少一个恒为零（$j \neq \ell$）→ 求和每一项都为零：
$$
\big[ H_{pp} \big]_{(j, \ell)} = \mathbf{0} \ (j \neq \ell), \qquad H_{pp} = \mathrm{diag}\big( H_{p_1}, \dots, H_{p_n} \big), \qquad H_{p_j} = \sum_{k:\, j(k) = j} \big( J_p^{(k)} \big)^\top \Lambda_k\, J_p^{(k)}
\tag{8.12}
$$
即每个 $3 \times 3$ 对角块是该路标全部观测的信息累加（有有效观测时 $\succ 0$，故可逆）。对照 07.2 (7.9)：$H_{cp}$ 的非零模式 $\iff$ 位姿-路标共视关系——07.2 证明"应当稀疏"的信息结构，在优化侧原样出现。

**Schur 补消元。** 正规方程分块写开（依据：(8.5) + (8.11)；右端 $\mathbf{b} = -J^\top \Sigma^{-1} \mathbf{e}$ 分块为 $\mathbf{b}_c = -J_c^\top \Sigma^{-1} \mathbf{e}$、$\mathbf{b}_p = -J_p^\top \Sigma^{-1} \mathbf{e}$）：
$$
\begin{bmatrix} H_{cc} & H_{cp} \\ H_{pc} & H_{pp} \end{bmatrix} \begin{bmatrix} \Delta\mathbf{c} \\ \Delta\mathbf{p} \end{bmatrix} = \begin{bmatrix} \mathbf{b}_c \\ \mathbf{b}_p \end{bmatrix}
\tag{8.13}
$$
第一步，取第二行解出 $\Delta\mathbf{p}$（依据：分块方程逐行展开；$H_{pp}$ 可逆——(8.12) 块对角且各块正定）：
$$
H_{pc}\, \Delta\mathbf{c} + H_{pp}\, \Delta\mathbf{p} = \mathbf{b}_p \;\Longrightarrow\; \Delta\mathbf{p} = H_{pp}^{-1} \big( \mathbf{b}_p - H_{pc}\, \Delta\mathbf{c} \big)
\tag{8.14}
$$
（$H_{pp}^{-1}$ 逐块求逆，$O(n)$。）第二步，代入第一行（依据：代入 + 分配律 + 移项合并 $\Delta\mathbf{c}$ 项）：
$$
H_{cc}\, \Delta\mathbf{c} + H_{cp} H_{pp}^{-1} \mathbf{b}_p - H_{cp} H_{pp}^{-1} H_{pc}\, \Delta\mathbf{c} = \mathbf{b}_c
$$
$$
\Longrightarrow\;\; \underbrace{\big( H_{cc} - H_{cp}\, H_{pp}^{-1}\, H_{pc} \big)}_{S\ \text{（约简位姿矩阵）}} \Delta\mathbf{c} \;=\; -\, J_c^\top \Sigma^{-1} \mathbf{e} \;+\; H_{cp}\, H_{pp}^{-1}\, J_p^\top \Sigma^{-1} \mathbf{e}
\tag{8.15}
$$
（右端由 $\mathbf{b}_c = -J_c^\top \Sigma^{-1} \mathbf{e}$、$\mathbf{b}_p = -J_p^\top \Sigma^{-1} \mathbf{e}$ 代入整理得到。）$S = H_{cc} - H_{cp} H_{pp}^{-1} H_{pc}$ 即 $H$ 对路标块的 Schur 补（依据：Schur 补定义）。求解流程：解 (8.15)（$6m$ 维）→ 回代 (8.14) 得 $\Delta\mathbf{p}$。

**复杂度对比。** 消元主体是形成 $H_{cp} H_{pp}^{-1} H_{pc}$：$H_{pp}^{-1}$ 逐块 $O(n)$；外积累加沿共视结构进行——每个被 $d_j$ 个位姿观测的路标贡献 $O(d_j^2)$ 个 $6 \times 6$ 块，总计 $O\big( \sum_j d_j^2 \big)$，最坏 $d_j = m$ 时 $O(n\, m^2)$（实际远小于该上界：共视是稀疏的，定性）。约简系统 $6m$ 维稠密 Cholesky 分解 $O(m^3)$（依据：Cholesky 的立方律）；回代 (8.14) 对每个路标常数级。合计：
$$
\text{直接稠密求解}:\ O\big( (m+n)^3 \big) \qquad \text{vs} \qquad \text{Schur 补}:\ O\big( m^3 + n\, m^2 \big) + \text{稀疏处理}
\tag{8.16}
$$
（两式按块维数 $m, n$ 计，标量维数下分别为 $(6m+3n)^3$ 与 $(6m)^3$ 量级，量级结论不变；$n \gg m$ 时后者由 $m^3$ 主导——**位姿数才是 BA 的"硬"规模**。）

**Schur 补即概率边缘化。** 第 03 章 (3.11) 给出条件分布；它的对偶是边缘分布：把联合高斯 $p(\mathbf{c}, \mathbf{p} \mid \mathbf{z})$ 的信息矩阵按 $[\Lambda_{cc}, \Lambda_{cp}; \Lambda_{pc}, \Lambda_{pp}]$ 分块，边缘 $p(\mathbf{c} \mid \mathbf{z}) = \int p(\mathbf{c}, \mathbf{p} \mid \mathbf{z})\, \mathrm{d}\mathbf{p}$ 的协方差为
$$
\Sigma_{\mathbf{c} \mid \mathbf{z}} = \big( \Lambda_{cc} - \Lambda_{cp}\, \Lambda_{pp}^{-1}\, \Lambda_{pc} \big)^{-1}
\tag{8.17}
$$
（依据：分块矩阵求逆公式——联合协方差 $\Sigma = \Lambda^{-1}$ 的 $cc$ 块；(3.11) 条件协方差公式的边缘化版本。）对照 (8.15)：$S$ 的结构与 (8.17) 的括号项完全一致——**对路标做 Schur 补消元 = 在概率上把路标边缘化**，约简系统正是只关于位姿的后验（信息矩阵为 $S$）。这同时解释了 fill-in 的来源：两个不共享路标的位姿在联合后验中"给定公共路标"才独立，边缘化（积分掉公共路标）后经由它产生直接耦合（依据：与 07.2 滤波稠密化同源的机制——求和/积分破坏条件独立结构）。

**从全局 BA 到滑窗与位姿图。** 既然边缘化有代价（fill-in），在线系统就把它当作受控工具使用：滑窗方法（如 DSO，Engel, Koltun, Cremers, ECCV 2016，已收录）只优化最近若干关键帧，把滑出窗口的状态边缘化成先验、注入窗口内变量——(8.17) 正是其数学内核。而第 09 章的位姿图优化（pose graph optimization）走得更远：回环检出后丢开路标、只优化位姿节点——可视为"路标已被边缘化"的极限情形，其目标函数与求解器直接复用本章的 (8.2)、(8.5)、(8.9)。ORB-SLAM2 的后端正是这套组合：跟踪 / 局部建图 / 回环闭合三线程，局部 BA 与全局优化基于 g2o（LM + 稀疏求解），回环后沿本质图（essential graph）做位姿图优化（配套 PDF 系统架构与回环各节）。

## 本章要点

- 图优化目标函数 (8.2)：$\min_{\mathbf{x}} \sum_k \mathbf{e}_k^\top \Lambda_k \mathbf{e}_k$，由高斯负对数似然推出；权重 = 信息矩阵 $\Lambda_k = \Sigma_k^{-1}$，与第 03/07 章的概率语义一致。
- Gauss-Newton (8.5)：残差一阶展开 (8.3) → 代价二次近似 (8.4) → 对 $\Delta$ 求导置零 → $(J^\top \Sigma^{-1} J) \Delta = -J^\top \Sigma^{-1} \mathbf{e}$；$H = J^\top \Sigma^{-1} J$ 是舍弃残差二阶项的 Hessian 近似。
- LM (8.6)：$(H + \lambda I) \Delta = -\mathbf{g}$；$\mu_{\min}(H + \lambda I) \ge \lambda > 0$ (8.7) 保证正定可解，且 $\Delta$ 恒为下降方向。
- 流形更新 (8.9)：$\mathbf{T} \leftarrow \exp(\Delta\boldsymbol{\xi}^\wedge) \mathbf{T}$ 精确保持 $SE(3)$ 约束；直接加会破坏正交性 (8.8)；左扰动雅可比 (8.10) 由链式法则 + BCH 一阶近似（第 02 章 (2.21)、(2.23)、(2.26)）得到。
- $H_{pp}$ 块对角 (8.12)：每条残差只含一个路标 → 路标列块互不耦合；$H_{cp}$ 的稀疏模式 = 共视结构（呼应 07.2 的变量共现图）。
- Schur 补 (8.15)：消路标得 $6m$ 维约简系统 $S = H_{cc} - H_{cp} H_{pp}^{-1} H_{pc}$；复杂度 $O\big( (m+n)^3 \big) \to O\big( m^3 + n\, m^2 \big)$ (8.16)。
- 概率语义 (8.17)：Schur 补 = 边缘化；fill-in 与 07.2 滤波稠密化同源 → 滑窗（DSO）与第 09 章位姿图优化是其受控使用。

## 配套阅读

- 论文：[ORB-SLAM2（Mur-Artal & Tardós, T-RO 2017）](../../papers/slam/classics/arXiv-1610.06475_ORB-SLAM2.pdf)——本章理论（BA、g2o、位姿图优化）在完整系统中的落地；滑窗边缘化的另一范本见 [DSO（Engel et al., 2016）](../../papers/slam/classics/arXiv-1607.02565_DSO.pdf)。
- 相邻章节：[第 02 章（李群李代数工具箱：(2.3)(2.17)(2.21)-(2.23)(2.26)）](./02_三维刚体运动旋转与位姿.md) ｜ [第 03 章（条件/边缘高斯 (3.11)）](./03_概率状态估计基础.md) ｜ [第 07 章（为何换范式）](./07_后端-i滤波与增量估计.md) ｜ [第 09 章（位姿图优化与回环检测）](./09_回环检测.md)
- 教材：Triggs et al., *"Bundle Adjustment — A Modern Synthesis"*, ICCV 1999；高翔《视觉 SLAM 十四讲》第 6 讲（非线性优化）、第 10-11 讲（后端）；Nocedal & Wright, *Numerical Optimization*, ch4 / ch10。
- 主题导航：[tutorials/slam/README.md](./README.md) ｜ 论文索引：[papers/slam/README.md](../../papers/slam/README.md)

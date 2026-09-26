# 论文精读｜Poisson 表面重建（SGP 2006）

> **PDF**：[papers/3d_reconstruction/classics/PoissonSurfaceReconstruction_SGP2006_KazhdanBolithoHoppe.pdf](../../../papers/3d_reconstruction/classics/PoissonSurfaceReconstruction_SGP2006_KazhdanBolithoHoppe.pdf) ｜ **教程**：[第 05 章｜从体素到网格：表面提取](../05_从体素到网格表面提取.md) ｜ **代码**：无（教学实现规划中）

## 1. 论文信息与一句话贡献

- **题目**：Poisson Surface Reconstruction
- **作者**：Michael Kazhdan、Matthew Bolitho（约翰霍普金斯大学）、Hugues Hoppe（微软研究院）
- **发表**：Eurographics Symposium on Geometry Processing（SGP）2006。PDF 全文 10 页（含参考文献与附录 A），**编号公式仅 3 个**：式 (1) 平滑指示函数的梯度（Lemma 内）、式 (2) 向量场、式 (3) 三线性向量场；变分问题、Poisson 方程 $\Delta\tilde\chi=\nabla\cdot\vec V$、矩阵 $L$ 与等值值 $\gamma$ 均为**未编号展示式**，本文按页/节定位。
- **一句话贡献**：把"带法向点云 → 三角网格"表述为一个**空间 Poisson 问题**——把样本法向平滑延拓成向量场、全局求解"梯度最接近该场"的指示函数、再提取等值面；全部点**同时**进入一个稀疏线性系统（八叉树自适应 + 多级求解），时间与空间复杂度**正比于重建模型的规模**，对噪声稳健、输出天然水密。

## 2. 问题与动机

**要解决什么**：从**定向点样本**（oriented points：位置 $s.p$ + 内向法向 $s.\vec N$，来自深度扫描/多视图立体的融合输出）重建水密三角网格。困难：采样不均匀（遮挡区稀疏）、法向带噪、可达性约束使扫描面片残缺。

**已有路线的两难**（§2）：
- **局部方法**（移动最小二乘、体素分块混合）：逐区域拟合、邻域选取与混合权重都是启发式；隐函数只在样本附近受约束，远离样本处出现**离面伪影**（spurious surface sheets）。
- **全局方法**：理想 RBF（径向基函数）全局支撑且不衰减 → 解矩阵**稠密且病态**；本组前作 Kazhdan 2005 的 FFT 谱解法快（附录 A 证明与本文**数学等价**），但要求均匀 3D 网格——$O(r^3\log r)$ 时间、$O(r^3)$ 空间（$r^3$ 为网格分辨率），无法自适应，分辨率每翻一档内存涨 8 倍。

**MVSNet 式的观察在这里同样成立**（对照[第 01 章 (1.2)](../01_3D重建问题与表示全景.md) 的零水平集统一抽象）：指示函数在几乎处处是常数（内部 1、外部 0），**全部信息集中在表面附近**——理想的离散化应该在表面细、别处粗。这就是"自适应八叉树上的 Poisson"的动机：既有全局解的稳健性（一个方程、无分块启发式），又有局部支撑基带来的稀疏性与自适应性。

## 3. 方法总览

```
带法向点云 {s.p, s.Ñ}
   │ ① 定义梯度场：法向用平滑核 F 卷积延拓
   ▼
向量场 V̄ = Σ_s ∫_{P_s} F̃_p(q)·Ñ(p) dp ≃ Σ_s |P_s|·F̃_{s.p}(q)·sÑ     （论文式(2)）
   │ ② 变分问题：V̄ 一般不可积（非无旋）→ 最小二乘 min‖∇χ̃ − V̄‖²
   ▼
Poisson 方程  Δχ̃ = ∇·V̄    （论文 §3 展示式）
   │ ③ 离散化：八叉树节点紧支基 {F_o}，Galerkin 投影 → min‖Lx − v‖²
   │    多级求解（逐深度 CG + 块 Gauss–Seidel）
   ▼
指示函数 χ̃ ∈ 八叉树函数空间
   │ ④ 等值面提取：γ = 样本处 χ̃ 值的平均 → 八叉树适配 Marching Cubes
   ▼
水密三角网格 ∂M = {q : χ̃(q) = γ}
```

三个环节分别回答：数据如何变成场（式 (2)）、场如何变成方程（变分 → Poisson）、方程如何算得快（稀疏 + 多级）。

## 4. 关键公式推导（按论文原文式号）

**符号表**：

| 论文记号 | 含义 | 教程第 05 章对应 |
|---|---|---|
| $\mathcal{M},\ \partial\mathcal{M}$ | 实体模型及其边界表面 | $\Omega,\ S$（5.3 节） |
| $\chi_M$ | 指示函数：$\mathcal{M}$ 内为 1、外为 0 | $\chi$，(5.3) |
| $\tilde\chi$ | 重建的（平滑近似）指示函数 | $\tilde\chi$，(5.5)–(5.6) 的解 |
| $\bar{\mathbf{N}}_{\partial\mathcal{M}}(p)$ | **内法向**（inward normal） | 教程 $\hat{\mathbf{n}}$ 为外法向，二者差一个负号 |
| $F(\cdot)$ / $\bar F_p(q)=\bar F(q-p)$ | 平滑核 / 其平移 | $\bar F$（5.3 ⑤ 第二步） |
| $\vec V,\ \hat{\vec V}$ | 平滑向量场及其离散近似 | $\vec V$（5.3 ⑤ 第二步） |
| $\mathcal{O},\ o,\ D$ | 八叉树、节点、最大深度 | (5.7) 的基函数节点 |
| $F_{o,v}(q)=F\big(\frac{q-o.c}{o.w}\big)\frac{1}{o.w^3}$ | 节点基函数（平移 + 缩放） | — |
| $L,\ \mathbf{x},\ \mathbf{v}$ | Poisson 系统矩阵、系数、右端 | (5.7) 的 $L,\mathbf{v},\mathbf{b}$ |
| $\gamma$ | 等值值 | (5.8) 的 0.5 |

### 4.1 指示函数与其梯度：论文 Lemma 与式 (1)（内法向约定）

指示函数定义（依据：定义；即教程 (5.3)，也是 (1.11) 占据场的 0/1 理想版）：

$$\chi_M(\mathbf{x}) = \begin{cases}1, & \mathbf{x}\in\mathcal{M}\\ 0, & \mathbf{x}\notin\mathcal{M}\end{cases}$$

它在表面处跳变、经典意义不可微。论文不直接对 $\chi_M$ 求导，而是先与平滑核卷积，再证以下 **Lemma**（PDF 第 3 页）：给定实体 $\mathcal{M}$ 与边界 $\partial\mathcal{M}$，$\bar{\mathbf{N}}_{\partial\mathcal{M}}(p)$ 为 $p$ 处的**内法向**，$\bar F_p(q)=\bar F(q-p)$ 为核的平移，则平滑指示函数的梯度等于平滑法向场：

$$\nabla(\chi_M * \bar F)(q_0) = \int_{\partial\mathcal{M}} \bar F_p(q_0)\,\bar{\mathbf{N}}_{\partial\mathcal{M}}(p)\,\mathrm{d}p. \tag{1}$$

**原文证明逐步展开**（取 $x$ 分量，$y,z$ 同理）：

第一步（卷积展开 + 积分域收缩）：
$$\frac{\partial}{\partial x}\Big|_{q_0}(\chi_M * \bar F) = \frac{\partial}{\partial x}\Big|_{q_0}\int_{\mathbb{R}^3}\chi_M(p)\,\bar F(q-p)\,\mathrm{d}p = \frac{\partial}{\partial x}\Big|_{q_0}\int_{\mathcal{M}}\bar F(q-p)\,\mathrm{d}p$$
（依据：卷积定义；$\chi_M$ 在 $\mathcal{M}$ 外为 0，把积分限缩入 $\mathcal{M}$。）

第二步（积分号下求导 + 链式法则换变量）：
$$= \int_{\mathcal{M}}\frac{\partial}{\partial x}\bar F(q-p)\Big|_{q=q_0}\mathrm{d}p = \int_{\mathcal{M}}-\frac{\partial}{\partial p_x}\bar F(q_0-p)\,\mathrm{d}p$$
（依据：$\bar F$ 光滑紧支，被积函数及其导数可积，求导与积分可交换；链式法则中 $x$ 与 $p_x$ 反号产生负号——原文同样注明"$\frac{\partial}{\partial p_x}\bar F(q-p) = -\frac{\partial}{\partial x}\bar F(q-p) $"。）

第三步（凑成散度）：
$$= -\int_{\mathcal{M}}\nabla_p\cdot\big(\bar F(q_0-p),\,0,\,0\big)\,\mathrm{d}p$$
（依据：向量场 $(\bar F,0,0)$ 的散度只有 $x$ 分量贡献 $\frac{\partial \bar F}{\partial p_x}$。）

第四步（散度定理 + 内法向吸收负号）：
$$= \int_{\partial\mathcal{M}}\big(\bar F_p(q_0),\,0,\,0\big)\cdot\bar{\mathbf{N}}_{\partial\mathcal{M}}(p)\,\mathrm{d}p$$
（依据：散度定理 $\int_{\mathcal{M}}\nabla\cdot\mathbf{F}\,\mathrm{d}V=\oint_{\partial\mathcal{M}}\mathbf{F}\cdot\mathbf{n}_{out}\,\mathrm{d}A$；负号被 $\bar{\mathbf{N}}=-\mathbf{n}_{out}$ 吸收。）

三个分量合并即式 (1)。$\blacksquare$

**与教程 (5.4) 的取向等价性**：教程在分布意义下直接算 $\nabla\chi$，用**外法向** $\hat{\mathbf{n}}$ 写成 $\nabla\chi = -\hat{\mathbf{n}}\,\delta_S$（(5.4)）；论文用**内法向** $\bar{\mathbf{N}}=-\hat{\mathbf{n}}$，式 (1) 取核宽 $\to 0$ 的极限即 $\nabla\chi_M = \bar{\mathbf{N}}\,\delta_S$。因 $\bar{\mathbf{N}}=-\hat{\mathbf{n}}$，两式**逐字恒等**——差别只是"谁带负号"。更关键的是：翻转指示函数 $\chi\to 1-\chi$ 时梯度反号、而等值面集合不变：$\{\tilde\chi=0.5\}=\{1-\tilde\chi=0.5\}$——**0.5 等值面对内外翻转不变**，故两种取向的数据进入同一套管线得到同一张网格，法向取向只需在数据内部自洽。

### 4.2 向量场的平滑延拓：为什么卷积而非直接插值（论文式 (2)）

真 $\nabla\chi_M$ 是奇异分布（$\delta_S$ 铺在曲面上），离散点样本无法直接逼近一个测度——"直接把法向插值成场"在数学上不适定。论文把表面分成小补丁 $\mathcal{P}_s$，对式 (1) 的面积分**逐补丁退化**，式 (2)（PDF 第 3 页）：

$$\nabla(\chi_M*\bar F)(q) = \sum_{s\in S}\int_{\mathcal{P}_s}\bar F_p(q)\,\bar{\mathbf{N}}_{\partial\mathcal{M}}(p)\,\mathrm{d}p \;\simeq\; \sum_{s\in S}\big|\mathcal{P}_s\big|\cdot\bar F_{s.p}(q)\cdot s\bar N \;\equiv\; \bar V(q). \tag{2}$$

（依据：补丁内法向近似常值、核在补丁上近似常值，面积分 $\approx$ 面积 × 核值 × 法向。教程 5.3 ⑤ 第二步的 $\vec V(q)\approx\sum_i a_i\bar F_i(q)\mathbf{n}_i$ 即此式，$a_i=|\mathcal{P}_s|$。）

**核宽的两难与折中**（原文 §3 明文讨论）：核太窄——式 (2) 的"核在补丁上常值"近似失效、采样噪声未被平均；太宽——over-smooth 抹平真实细节。折中：**取方差为采样分辨率量级的高斯核**（原文："a Gaussian whose variance is on the order of the sampling resolution"）。八叉树深度 $d$ 的节点对应采样宽度 $2^{-d}$，故核应逼近方差 $\sim 2^{-D}$ 的单位方差高斯（依据：§4.1）。

**"三重积可分离"的实现论证**（§4.1）：为让散度、拉普拉斯算子**稀疏**、且任意点的求值只需邻近节点，核取一维盒式滤波 $B$ 的自卷积（张量积可分）：

$$F(x,y,z) = \big[B(x)B(y)B(z)\big]^{(*n)},\qquad B(t)=\begin{cases}1, & |t|<0.5\\ 0, & \text{else}\end{cases}$$

$n$ 越大越接近高斯；实现取 $n=3$（分段二次逼近单位方差高斯）。支撑为 $[-1.5,1.5]^3$，故任一节点的基函数至多与同深度 $5^3-1=124$ 个其他节点的函数重叠（依据：支撑半宽 1.5 个节点宽度 → 每轴 ±2 共 5 个邻居）——这正是后文矩阵 $L$ 稀疏的定量来源。

### 4.3 变分问题 → Poisson 方程（论文 §3 展示式；教程 (5.5)–(5.6)）

平滑涂抹破坏了 $\nabla\chi_M$ 原本的无旋结构，$\bar V$ 一般**不可积**（not curl-free，原文 §3："V̄ is generally not integrable (i.e. it is not curl-free), so an exact solution does not generally exist"——依据：梯度场必要条件是无旋，$\oint\nabla\chi\cdot\mathrm{d}l=0$）。于是退而求**梯度最接近 $\bar V$ 的标量场**（最小二乘，教程 (5.5)）：

$$\tilde\chi = \arg\min_{\chi'}\int\big\|\nabla\chi' - \vec V\big\|^2\,\mathrm{d}V.$$

一阶变分（极小点的必要条件）：沿任意扰动 $\psi$ 的方向导数为零，

$$0 = \frac{\mathrm{d}}{\mathrm{d}\epsilon}\Big|_{\epsilon=0}\int\big\|\nabla(\tilde\chi+\epsilon\psi)-\vec V\big\|^2\,\mathrm{d}V = 2\int(\nabla\tilde\chi-\vec V)\cdot\nabla\psi\,\mathrm{d}V$$

（依据：平方展开、保留 $\epsilon$ 一次项。）对两项分别分部积分（依据：格林第一恒等式 $\int\nabla u\cdot\nabla\psi = -\int(\Delta u)\psi + \oint\psi\,\partial_n u$；论文在包围盒上求解，边界按自然边界条件处理，边界项不改变方程）：

$$\int(\Delta\tilde\chi)\,\psi\,\mathrm{d}V = \int(\nabla\cdot\vec V)\,\psi\,\mathrm{d}V,\qquad\forall\,\psi.$$

$\psi$ 任意，由变分法基本引理被积函数逐点相等（教程 (5.6)）：

$$\Delta\tilde\chi = \nabla\cdot\vec V,\qquad \Delta\equiv\nabla\cdot\nabla.$$

（原文表述："we apply the divergence operator to form the standard Poisson equation $\Delta\tilde\chi = \nabla\cdot\bar V$"——变分问题经散度算子变成标准 Poisson 问题。）

### 4.4 函数空间基的选择：特征基 vs 全局 RBF vs 紧支多级基

解 Poisson 方程先要选函数空间，论文的论证（§1、§2、§4.1）：

- **拉普拉斯特征基 / 傅里叶基（全局）**：谱方法数学上最干净——附录 A 证明：周期域上 $\Delta u=f$ 经傅里叶变换化为 $-\|\xi\|^2\hat u(\xi)=\hat f(\xi)$，即 $\hat u(\xi)=-\hat f(\xi)/\|\xi\|^2$；用恒等式 $\widehat{\nabla\cdot\vec V}=i\xi\cdot\hat{\vec V}$ 得频域闭式解，**与本组前作 Kazhdan 2005 的谱解法完全一致**。但 FFT 要求**均匀网格**：$O(r^3\log r)$ 时间、$O(r^3)$ 空间，且分辨率翻倍内存 ×8——均匀采样在"指示函数几乎处处是常数"的事实面前是纯粹的浪费（依据：§2）。
- **理想 RBF（全局支撑）**：插值精度高，但全局支撑且不衰减 → 解矩阵**稠密、病态**（依据：§2）；大点云不可行。
- **本文选择：八叉树节点紧支基（多级/小波式）**：$\mathscr{F}_{\mathcal{O},F}=\mathrm{Span}\{F_o\}$，节点函数
$$F_{o,v}(q) = F\Big(\frac{q-o.c}{o.w}\Big)\frac{1}{o.w^3}$$
（$o.c,o.w$ 为节点中心与宽度）。单位积分保持：$\int F_{o,v}\,\mathrm{d}q=\int F(u)\,\mathrm{d}u=1$（依据：换元 $u=\frac{q-o.c}{o.w}$，$\mathrm{d}q=o.w^3\,\mathrm{d}u$）。该空间"具有类似传统小波表示的多分辨率结构"（原文 §4.1）——细节点编码高频、表面附近表示更精确。论证核心：解的**全部高频信息只存在于表面附近**（$\nabla\cdot\vec V$ 紧支在表面邻域），粗网格在别处足够。原文 §4.1 给出基函数须满足的三条件：(1) $\vec V$ 可被 $\{F_o\}$ 的线性和精确高效表示；(2) Poisson 方程的矩阵表示可高效求解；(3) 指示函数的表示在表面附近可精确高效求值——紧支多级基三条全占。

### 4.5 离散化与多级求解（论文式 (3)、矩阵 $L$、$\min\|L\mathbf{x}-\mathbf{v}\|^2$；教程 (5.7)）

**右端向量场的离散化**，式 (3)（PDF 第 4 页）：为子节点精度，不把样本 clamp 到叶心（clamp 误差至多半个采样宽度，§4.1），而用三线性权重把样本分布到 8 个最近深度-$D$ 节点：

$$\hat V(q) = \sum_{s\in S}\sum_{o\in\mathbf{N}_{GB}(p_s)}\alpha_{o,s}\,F_o(q)\,s\bar N$$

（$\mathbf{N}_{GB}(p_s)$ 为 8 个最近节点，$\alpha_{o,s}$ 为三线性插值权重；教程 5.3 ⑤ 第四步"每点按三线性权重分布到 8 个最近节点"即此。）

**从方程到线性系统**（§4.3）：目标是找 $\tilde\chi\in\mathscr{F}_{\mathcal{O},F}$ 使 $\Delta\tilde\chi$ 的投影最接近 $\nabla\cdot\vec V$ 的投影。因 $\{F_o\}$ **非正交**，直接正交投影昂贵；论文改为最小化**投影残差平方和**：

$$\sum_{o\in\mathcal{O}}\big\|\langle\Delta\tilde\chi-\nabla\cdot\vec V,\ F_o\rangle\big\|^2 = \sum_{o\in\mathcal{O}}\big\|\langle\Delta\tilde\chi,F_o\rangle-\langle\nabla\cdot\vec V,F_o\rangle\big\|^2.$$

设 $\tilde\chi=\sum_{o'}x_{o'}F_{o'}$、右端 $v_o=\langle\nabla\cdot\vec V,F_o\rangle$，定义

$$L_{o,o'} = \Big\langle\frac{\partial^2F_{o'}}{\partial x^2},F_o\Big\rangle+\Big\langle\frac{\partial^2F_{o'}}{\partial y^2},F_o\Big\rangle+\Big\langle\frac{\partial^2F_{o'}}{\partial z^2},F_o\Big\rangle,$$

则求解化为 $\min_{\mathbf{x}\in\mathbb{R}^{|\mathcal{O}|}}\|L\mathbf{x}-\mathbf{v}\|^2$（教程 (5.7) 的 $L\mathbf{v}=\mathbf{b}$ 同构，教程取法方程形式）。**稀疏**——基函数紧支（§4.2 的 124 邻居界）；**对称**——$\langle F'',g\rangle=-\langle F',g'\rangle$（分部积分；紧支使边界项为零；原文记作 $F''g=-F'g'$ 的简写）。

**多级求解**（§4.3）：$L$ 有内在多级结构——逐深度 $d$ 取限制 $L_d$、用共轭梯度解深度-$d$ 子问题、把固定深度解投影回全空间更新残差（multigrid 思想：低频误差在粗网格上更易消除；教程 (5.7) 之后的"粗分辨率先解低频、逐级精化"同此）。**内存**：朴素存 $L$ 每列约 125 个非零元、内存约 125 倍于八叉树——改用**块 Gauss–Seidel**：把第 $d$ 层空间分成重叠区域、分块求解再投影回更新残差，块数随深度选取使矩阵规模不超内存阈值。

**复杂度**：时间与空间**正比于重建模型的规模**（摘要）；八叉树每加深一层（分辨率翻倍），时间、内存、输出三角形数均 ×4（依据：§5.3 Table 1）。

### 4.6 等值面提取：等值值 $\gamma$ 取样本均值而非固定 0（论文 §4.4–§4.5；教程 (5.8)）

论文不取固定阈值，而取**样本处 $\tilde\chi$ 值的平均**（§4.4 展示式）：

$$\partial M \equiv \big\{q\in\mathbb{R}^3 \mid \tilde\chi(q)=\gamma\big\},\qquad \gamma = \frac{1}{|S|}\sum_{s\in S}\tilde\chi(s.p).$$

**为什么不用固定 0（论文的专门论证）**：
1. **尺度不变性**——"this choice of isovalue has the property that scaling $\tilde\chi$ does not change the isosurface"（原文 §4.4）。式 (2) 本身带"$\simeq$"：向量场只知道到一个**乘法常数**（$|\mathcal{P}_s|$ 被假设为常数，原文 §4.2 明言"the choice of multiplicative constant does not affect the reconstruction"）。解的整体幅值不定，任何固定阈值都会随解的缩放漂移；样本均值自动定标。
2. **位置贴近性**——选等值值的准则是"使提取曲面尽量贴近输入样本位置"（原文 §4.4 首句）：样本就在表面上，其 $\tilde\chi$ 值聚在跃迁中点附近但（因平滑与最小二乘近似）**不必恰为 0.5**，取均值正好落在样本云的"中面"上。
3. **采样密度偏差**——非均匀采样下未加权均值有偏：稀疏区（遮挡处）与密集区的话语权应不同。§4.5 给出密度加权版：$\gamma=\frac{\sum_s W_{\tilde D}(s.p)\,\tilde\chi(s.p)}{\sum_s W_{\tilde D}(s.p)}$，其中密度估计 $W_{\tilde D}(q)=\sum_s\sum_o\alpha_{o,s}F_o(q)$（Parzen 核密度估计的 splatting 实现），并**同步自适应核宽**（稠密区窄核保细节、稀疏区宽核抗噪；样本深度按 $\mathrm{Depth}(s.p)=\min(D,\tilde D+\log_2(W_{\tilde D}(s.p)/W))$ 选取）。

实践与后续版本（含 2013 Screened Poisson）固定用 0.5——内部 1、外部 0 的跃迁中点（教程 (5.8) 即取 0.5）；样本在表面上，其均值经验上接近 0.5，两种写法在数据自定标意义下等价。

**网格提取**：用 Marching Cubes 的八叉树改造（引用 Lorensen & Cline 1987 及此前的八叉树适配 WG92、SFYC96）——即教程 5.1 的完整流程：棱上零交叉定顶点（教程 (5.1) 的插值）、$\nabla\tilde\chi$ 直接给顶点法向（教程 (5.2)，且这里梯度有解析/全场定义，无需差分）。因八叉树**非一致**引出两处修改（原文 §4.4）：棱上零交叉由**相邻最细节点**计算；叶节点一条棱上出现多个零交叉则细分该节点；细粗相邻面上的裂缝用"细节点等值线段投影到粗面"消除——这是教程 5.2"块间一致性"问题在自适应网格上的化身，全局解的等值面没有 5.2 的符号配置歧义，但细粗过渡仍需显式规则。

### 4.7 隐函数精度的误差论证（无正式定理，报告见文末）

论文**没有**给出"采样密度—重建精度"的正式定理，其精度论证是一条误差界链（均按原文）：
1. **clamp 界**：样本深度为 $D$、采样宽度 $2^{-D}$，clamp 到叶心的误差至多半个采样宽度；三线性权重（式 (3)）进一步给出子节点精度（§4.1–§4.2）。
2. **核宽匹配**：平滑核方差取采样分辨率量级——窄到不过度平滑、宽到补丁积分近似成立（§3、§4.1）。
3. **密度自适应**（§4.5）：核宽随局部采样密度缩放——密集区保锐利特征、稀疏区平滑拟合；密度同时进入等值值加权，抵消采样不均的偏差。
4. **分辨率实验**：dragon 模型在八叉树深度 6/8/10（等效网格 $64^3/256^3/1024^3$）下细节逐级复现（原文 Fig.3）——分辨率每加一档，可表达的细节频率翻倍。

## 5. 实验与结果解读

- **对比设置**（§5.2，均为公开真实扫描）：Stanford bunny（362,000 点，10 张深度图）、Happy Buddha（3,246,069 点，48 扫描）、Forma Urbis Romae 残片、Michelangelo David。
- **vs 计算几何类**（Power Crust、Robust Cocone）：二者输出本身带噪（原文 Fig.4(a)(b)）；Poisson 输出平滑且完整。
- **vs 插值类隐函数**（Fast RBF、MPU）：它们只在样本附近定义隐函数，远离样本处产生离面伪影片，平滑插值又抹掉细节（Fig.4(c)(d)）；Poisson 的解在全空间受方程约束，无此伪影。
- **vs VRIP**（体素距离场融合）：VRIP 在尖锐折痕处出现 "lipping"——其距离函数沿**视线方向**而非表面法向生长；Poisson 的各向同性 Poisson 解不受视向偏置，准确重建佛龛边角与折痕（Fig.5）。但 VRIP 携带**可见性/空间雕刻**信息：Buddha 基座双脚之间无样本，VRIP 能把两脚断开、Poisson 会粘连（Fig.6，作者自陈的 limitation，见 §6）。
- **vs FFT 谱方法**（Kazhdan 2005）：bunny 上两者几乎一致（Fig.4(g)(h)）——同一数学问题的均匀解 vs 自适应解的互相印证；在**非均匀采样**区域（David 左眼凹陷处，只有少量近掠射扫描覆盖）自适应核宽明显更优（Fig.7）。
- **性能**（§5.3 Table 1，dragon）：深度 7/8/9/10 → 时间 6/26/126/633 s、峰值内存 19/75/155/699 MB、三角形 21,000/90,244/374,868/1,516,806——深度每加一层 ×4，印证"复杂度正比于模型规模"。David 头部深度 11：上亿量级采样点、约 1.9 小时、5.2 GB 内存、1.6 亿量级三角形；同等的 FFT 均匀网格需 $2048^3$ 双体素网格、超 100 GB——自适应的价值一目了然。bunny 对比表（Table 2）中 Poisson 约 263 s / 310 MB，比 FFT（125 s / 1684 MB）省一个量级内存，比 VRIP（86 s / 186 MB）慢但解全局。

## 6. 局限与后续影响

**局限**（原文自陈 + 教程 5.3 ③）：其一，**不携带视线/可见性信息**——采集盲区（两脚之间、管道内侧）没有样本约束，全局解会用光滑曲面"脑补"连接，VRIP 的空间雕刻在此类场景反而正确（原文 Fig.6）；其二，**光滑偏置**——最小二乘 + 平滑核天然抹平薄结构与小细节；其三，法向**取向**必须一致（内法向约定，污染右端即污染整个方程）。**后续**：Screened Poisson（Kazhdan & Hoppe, 2013）在变分目标中加入把 $\tilde\chi$ 梯度钉向样本位置的屏蔽项（位置约束），以极小代价缓解光滑偏置与盲区粘连——如今 Open3D/PCL 里的默认 Poisson 重建即此版本。思想更远的回声：把"场由其散度/梯度决定、在紧支多级基上最小二乘"这一套搬到神经隐式表示，就是第 06/07 章的占据场与 SDF 学习。

## 7. 与本项目对照

**与教程第 05 章的公式映射**（已逐条核实双方编号）：

| 论文 | 内容 | 教程第 05 章 |
|---|---|---|
| Lemma + 式 (1) | $\nabla(\chi_M*\bar F)=\int_{\partial\mathcal{M}}\bar F_p\bar{\mathbf{N}}\,\mathrm{d}p$（**内法向**） | (5.3) 指示函数、(5.4) $\nabla\chi=-\mathbf{n}\,\delta_S$（**外法向**；教程已注明取向等价，本文 4.1 补齐证明细节） |
| 式 (2) | 向量场 $\vec V$（核宽两难与折中） | 5.3 ⑤ 第二步 $\vec V\approx\sum_i a_i\bar F_i\mathbf{n}_i$ |
| §3 展示式 | $\min\|\nabla\tilde\chi-\vec V\|^2 \Rightarrow \Delta\tilde\chi=\nabla\cdot\vec V$ | (5.5)–(5.6)（一阶变分 + 分部积分，逐步推导在教程） |
| 式 (3)、$L$、$\min\|L\mathbf{x}-\mathbf{v}\|^2$ | 八叉树离散化、稀疏对称系统、多级求解 | (5.7)（Galerkin 投影 + $L$ 稀疏对称的依据） |
| §4.4 展示式 | $\gamma=$ 样本均值（§4.5 密度加权） | (5.8) 取 0.5（教程注明"原文取样本均值、实践固定 0.5"，本文补齐"为何非 0"的三条论证） |
| §4.4 末段 | 八叉树适配 MC、细粗过渡 | 5.1 MC 流程（(5.1)/(5.2)）与 5.2 块间一致性（全局解消除符号歧义、自适应网格引入细粗过渡的对应讨论） |

**与 Marching Cubes 的分工**（教程 5.1–5.2 → 5.3 的论证主线）：MC 局部贪心、快、可并行，但二义面是结构性风险（5.2）；Poisson 把全部点纳入一个全局方程，块间一致性是**解的性质**——教程 5.3 ③ 的选型结论（机器人要水密碰撞模型选全局路线）正是本文实验的总结。

**与 MVSNet 的管线衔接**：MVSNet 深度图 → 一致性过滤 → 融合点云（带法向）→ 本文重建水密网格——这是机器人场景重建的标准三级管线（[第 01 章 (1.4)](../01_3D重建问题与表示全景.md) 的 ③ 级）。闭环彩蛋：MVSNet 论文制作 DTU 训练真值用的正是本文的后续版本 Screened Poisson——见[MVSNet 精读](./MVSNet_ECCV2018.md) §4.5 与其式 (4) 的说明。第 10 章给完整选型矩阵。

## 配套阅读

- 原论文：[PoissonSurfaceReconstruction_SGP2006_KazhdanBolithoHoppe.pdf](../../../papers/3d_reconstruction/classics/PoissonSurfaceReconstruction_SGP2006_KazhdanBolithoHoppe.pdf)。
- 教程主章节：[第 05 章｜从体素到网格：表面提取](../05_从体素到网格表面提取.md)（(5.1)–(5.8) 全部推导；MC 15-case 与二义面讨论）；等值面抽象：[第 01 章](../01_3D重建问题与表示全景.md)（零水平集 (1.2)、占据场 (1.11)、SDF/TSDF (1.13)–(1.15)）；占据场 $o=0.5$ 的网格提取复用本章管线：[第 06 章](../06_学习式形状表示SDF与占据.md)。
- 姊妹精读：[MVSNet（ECCV 2018）](./MVSNet_ECCV2018.md)（点云的来源；其训练真值制作即 Screened Poisson）。
- 后续脉络（论文库未收录 PDF，文字提及）：Screened Poisson（Kazhdan & Hoppe, TGP 2013）、Kazhdan 2005 谱重建（附录 A 的对照对象）、Marching Cubes（Lorensen & Cline, SIGGRAPH 1987）。

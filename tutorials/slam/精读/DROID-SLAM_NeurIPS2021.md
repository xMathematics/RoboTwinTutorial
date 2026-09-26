# 论文精读｜DROID-SLAM：深挖"可微 BA"的递归视觉 SLAM（NeurIPS 2021）

> **PDF**：[papers/slam/frontier/arXiv-2108.10869_DROID-SLAM.pdf](../../../papers/slam/frontier/arXiv-2108.10869_DROID-SLAM.pdf)（arXiv v2，2022-02-02，15 页含附录，本精读所有式号以此 PDF 为准）｜ **教程**：[第 10 章](../10_建图与系统实战.md)（§10.4 前沿导读的深读扩展）｜ **代码**：[projects/slam/droidlite/](../../../projects/slam/droidlite/)（结构演示，无学习组件）

## 1. 论文信息与一句话贡献

- **题目**：DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras
- **作者**：Zachary Teed、Jia Deng（Princeton University）
- **发表**：NeurIPS 2021（35th Conference on Neural Information Processing Systems）；arXiv:2108.10869
- **一句话贡献**：把 SLAM 后端改造成一个**端到端可微的递归优化器**——用光流网络（RAFT 血统的全对相关体 + ConvGRU）反复预测"对应场修正与置信度"，每轮修正都经一个**可微稠密 BA 层**（对全部帧的 SE(3) 位姿 + 逐像素逆深度做高斯牛顿步，Schur 补求解）映射成位姿/深度更新；单一模型仅在合成数据 TartanAir 单目上训练，即可在 TartanAir / EuRoC / TUM-RGBD / ETH3D 上以大裕度刷新精度与鲁棒性（零失败），并直接泛化到双目与 RGB-D 输入而无需重训练。

## 2. 问题与动机

**要解决什么**（§1）：经典 SLAM 的失效形态——特征跟踪丢失、优化发散、漂移累积。已有学习式方案两条路都没走通（§2 相关工作逐类评述）： 换更好的前端特征/匹配（仍是"学习前端 + 经典后端"的拼接，后端瓶颈不动）； 端到端回归轨迹（§2 "Deep Learning" 段：只做 small-scale reconstruction、几帧到十几帧、缺回环与全局 BA，"their inability to perform large scale reconstruction"）； BA-Net（其引文 [47]）虽把 BA 层放进网络，但深度是**基函数线性组合**（learned basis，"a set of pre-predicted depth maps combined linearly via a small number of coefficients"）且优化特征空间的光度误差；DeepV2D（其引文 [48]）在深度与位姿间交替但更新不是对"每像素深度 + 全部位姿"的联合 dense 优化（§1 对比段）。还有中间路线的对照（§2）：g2o 的可微封装把经典算法变成计算图但"无 trainable parameters，性能受限于所模拟的经典算法"——DROID 与它的差别正在于有**可学习的更新算子**。

**核心设计思想——DROID**（Differentiable Recurrent Optimization-Inspired Design）：既然经典求解器本来就是"迭代 + 每轮重线性化"（[第 08 章](../08_后端-ii图优化与-ba.md) 的 G-N/LM 骨架），就把"迭代"本身交给**学习到的递归更新算子**，把"每步解什么"交给**可微的 BA 层**，端到端训练。两个关键创新（§1 列表，原文语义）： RAFT 只迭代更新两帧的光流，这里迭代更新**任意多帧**的位姿与深度，享有联合全局精化——回环与长轨迹漂移受益； 每次更新经 **Dense BA 层**最大化与光流的几何相容性——几何约束提升精度与鲁棒性，且单目训练的模型直接吃双目/RGB-D。分类学位置（§2 末段）：像直接法一样不做特征预处理（用整幅图像的信息），但优化的是重投影误差（几何）——"borrows the best of both approaches"。

## 3. 方法总览

```
图像 I_i ──特征网络/上下文网络（6 残差块 + 3 次下采样，1/8 分辨率，Fig. 8）──▶ 特征 g_θ(I)
                │
                ▼
【全对相关体 Eq.(1)】帧图边 (i,j)∈E：C_ij^{u₁u₁u₂u₂} = ⟨g_θ(I_i)_{u₁}, g_θ(I_j)_{u₂u₂}⟩
                平均池化 → 4 级相关金字塔；Lookup 算子按当前流取 (r+1)² 邻域
                ▼
【递归更新算子 §3.2，每轮 k】
  ① 对应场 Eq.(3)：p_ij = Π_c(G_ij ∘ Π_c⁻¹(p_i, d_i))   ←用当前位姿/深度重算（Algorithm 1 复位）
  ② 输入 = 相关特征 + 光流(p_ij − p_i) + 上轮 BA 残差流 → ConvGRU（3×3，隐状态 h 反馈）
  ③ 输出 = 修正流 r_ij + 置信度权重 w_ij（两个卷积头，Fig. 9）
      池化隐状态 → 逐像素阻尼 λ（softplus 保正）+ 8×8 逆深度上采样掩码
  ④ 修正对应 p*_ij = r_ij + p_ij
                ▼
【Dense BA 层 Eq.(4)-(5)】min Σ ‖p*_ij − Π_c(G*_j ∘ Π_c⁻¹(p_i, d*_i))‖²_{Σ_ij}, Σ_ij = diag w_ij
  高斯牛顿一步：[[B, E],[Eᵀ, C]][Δξ; Δd] = [v; w]，C 对角 → Schur 补
                ▼
【回缩 Eq.(2)】G^(k+1) = Exp(Δξ^(k)) ∘ G^(k)，d^(k+1) = Δd^(k) + d^(k) → 下一轮
【系统层 §3.4】frontend（关键帧、3 步长邻接边、每新帧 10 轮更新、非关键帧 motion-only BA）
              + backend（对全部关键帧历史周期性全局 BA，Chebyshev 距离建边，RAFT 式省显存）
【双目/RGB-D §3.4 末段】深度仍当变量，目标 (4) 加"实测深度 vs 预测深度平方距离"惩罚项；
              双目 = 同一系统帧数翻倍 + DBA 层内固定左右相机相对位姿 + 跨相机边
```

## 4. 关键公式推导（式号按论文 PDF 原文）

### 4.0 符号表

| 符号 | 含义 |
|---|---|
| $\mathbf{G}_i \in SE(3)$ | 第 $i$ 帧位姿（共 $N_{\mathrm{frames}}$ 个，§3 Representation）；$\circ$ 为群乘法 |
| $\mathbf{d}_i \in \mathbb{R}^{H\times W}$ | 第 $i$ 帧**逐像素逆深度**图（§3 明确 "we are using the inverse depth parameterization"） |
| $\mathcal{E}$ | 帧图（frame-graph）边集：$(i,j)\in\mathcal{E}$ 表示两帧视场重叠共享点（§3，动态重建；相机回到旧区域即加长程边做回环） |
| $g_\theta(I_i)_{u}$ | 特征网络在像素 $u$ 的特征向量（1/8 分辨率，§3.1） |
| $\mathbf{h}^{(k)}$ | ConvGRU 第 $k$ 轮隐状态（§3.2） |
| $\mathbf{p}_{ij},\ \mathbf{p}^*_{ij} = \mathbf{r}_{ij} + \mathbf{p}_{ij}$ | 预测对应场 / GRU 修正后的对应（(3)、§3.2） |
| $\mathbf{r}_{ij},\ \mathbf{w}_{ij} \in \mathbb{R}^{H\times W\times 2}$ | 修正流 / 置信度权重（GRU 两卷积头输出；$\mathbf{w}_{ij}$ 进 $\Sigma_{ij} = \mathrm{diag}\,\mathbf{w}_{ij}$，(4)） |
| $r$ | Lookup 半径（相关体检索邻域，§3.1） |
| $\Pi_c,\ \Pi_c^{-1}$ | 针孔投影 / 反投影（附录 C (7)） |
| $\mathrm{Exp}(\cdot),\ \mathrm{Adj}_{\mathbf{G}}$ | $SE(3)$ 指数映射 / 伴随算子（附录 C (9)–(10)；第 02 章） |

### 4.1 状态与问题形态（§3 Representation）

变量只有两组：位姿 $\{\mathbf{G}_i\}_{i=0}^{N_{\mathrm{frames}}}$ 与逆深度图 $\{\mathbf{d}_i\}_{i=0}^{N_{\mathrm{frames}}}$——**没有显式 3D 点地图**：每个"点"宿主在某一帧的像素上，由 $(\mathbf{p}_i, d_i)$ 经 $\Pi_c^{-1}$ 定义。与 VINS-Mono/DSO 的宿主帧逆深度同思路（见[精读/VINS-Mono](./VINS-Mono_RAM2018.md) §4.3、[精读/DSO](./DSO_PAMI2018.md) §4.5）：远处点的不确定性被逆深度参数压平、无穷远点有良定义的有限参数。差别在于 DSO/VINS 只为稀疏选点存 $\lambda$，DROID 为**每个关键帧每个像素**都存一份——"dense"由此而来（§1 对 BA-Net 的对比：per-pixel depth, without being handicapped by a depth basis）。位姿更新用局部参数化 $e^{\hat{\boldsymbol{\xi}}}\mathbf{G}$（附录 C），即[第 08 章 (8.9)](../08_后端-ii图优化与-ba.md) 的流形回缩——矢量加法会离开 $SE(3)$（(8.8) 的正交性反例）。

### 4.2 全对相关体（式 (1)）与计算量论证

对帧图的每条边 $(i,j)$，取**所有位置对**的特征内积（§3.1 Correlation Pyramid）：

$$C_{ij}^{u_1u_1u_2u_2} = \big\langle g_\theta(I_i)_{u_1},\; g_\theta(I_j)_{u_2u_2} \big\rangle \tag{1}$$

**维度与代价论证**（依据：张量指标计数；$H\times W$ 为 1/8 分辨率特征图，特征维 $D = 128$，Fig. 8）：单条边的相关体是 4 维张量 $\mathbb{R}^{H\times W\times H\times W}$——每个 $(u_1, u_2)$ 位置对一个 $D$ 维内积，计算量 $O(H^2W^2D)$；$N$ 帧全对共 $\binom{N}{2}$ 条边，总量 $O(N^2H^2W^2D)$、存储 $O(N^2H^2W^2)$。**这就是"all-pairs"的准确含义**：不只是帧对全连（$N^2$），每对帧内部还是像素对全连（$H^2W^2$）。两个后果：

- **查询端便宜**：每轮只需按当前光流用 Lookup 算子取局部邻域——$\mathcal{L}_{ij}: \mathbb{R}^{H\times W\times r}\times\mathbb{R}^{H\times W\times r\times 2}\to\mathbb{R}^{H\times W\times(r+1)^2}$（$H\times W$ 网格坐标 + 半径 $r$ 邻域位移 → 每点 $(r{+}1)^2$ 个相关值，双线性插值，对 4 级金字塔各查一次再拼接，§3.1 Correlation Lookup）。
- **存储端爆炸**：论文原话（§3.4 Backend）："Storing the full set of correlation volumes would quickly exceed video memory. Instead, we use the memory efficient implementation proposed in RAFT"（只缓存特征图、响应按需现算）。$N$ 大时帧对项 $O(N^2)$ 是 backend 的根本瓶颈（§6 展开）。

相关体提供的信息是"视觉相似度"（appearance），与几何信息（对应场/光流）互补——§3.2 明说：相关特征"provide information about visual similarity in the neighborhood of $\mathbf{p}_{ij}$"，而对应场有时歧义（repetitive regions），"the flow provides a complementary source of information"。

### 4.3 更新算子：对应场、GRU 反馈、修正流（式 (2)、(3)）

**对应场（式 (3)）**：用当前估计把第 $i$ 帧像素 $(\mathbf{p}_i, d_i)$ 投到第 $j$ 帧：

$$\mathbf{p}_{ij} = \Pi_c\big(\mathbf{G}_{ij} \circ \Pi_c^{-1}(\mathbf{p}_i, \mathbf{d}_i)\big), \qquad \mathbf{G}_{ij} = \mathbf{G}_j \circ \mathbf{G}_i^{-1}, \quad \mathbf{p}_{ij}\in\mathbb{R}^{H\times W\times 2} \tag{3}$$

链式依据：反投影 $\Pi_c^{-1}(\mathbf{p}_i, d_i)$ 把像素 + 逆深度变成第 $i$ 帧系 3D 点（(7)）；相对位姿 $\mathbf{G}_{ij} = \mathbf{G}_j\circ\mathbf{G}_i^{-1}$ 把它搬到第 $j$ 帧系（"from frame $i$ to frame $j$"，附录 C (6) 前文）；再投影回像素。**每轮开始都重算**（Algorithm 1 的 "Reset $\mathbf{p}_{ij}$"）——这是"重线性化点每轮刷新"的对应物（§4.6）。

**GRU 更新（式 (2)）**：更新算子是 3×3 卷积 GRU（Fig. 2），每轮输出位姿与深度增量，并按流形回缩施加：

$$\mathbf{G}^{(k+1)} = \mathrm{Exp}\big(\Delta\boldsymbol{\xi}^{(k)}\big)\circ\mathbf{G}^{(k)}, \qquad \mathbf{d}^{(k+1)} = \Delta\mathbf{d}^{(k)} + \mathbf{d}^{(k)} \tag{2}$$

依据：位姿增量必须走 $SE(3)$ 指数映射——论文 §3.2 "The pose and depth updates are applied to the current depth and pose estimates using retraction on the SE3 manifold and vector addition respectively"；深度是向量场，直接加法。GRU 每轮的**输入**（§3.2 "Inputs" 段逐项）：相关特征（Lookup 输出）、光流 $\mathbf{p}_{ij} - \mathbf{p}_i$（对应场减源像素 = 运动光流），以及**上一轮 BA 解的残差流**——原文 "the residual flow from the previous BA solution is concatenated with the flow, allowing the network to exploit feedback from the previous iteration"：BA 的输出反馈回网络输入，这是定点迭代的闭环。**输出**（Fig. 9 架构）：隐状态 $\mathbf{h}^{(k+1)}$ 经两个卷积头产出修正流 $\mathbf{r}_{ij}$（"a correction term predicted in order to correct errors in the dense correspondence field"，修的是几何对应而非光度）与置信度 $\mathbf{w}_{ij}$；修正对应 $\mathbf{p}^*_{ij} = \mathbf{r}_{ij} + \mathbf{p}_{ij}$。另外把隐状态按源帧池化（"pool the hidden state over all features which share the same source view"）预测**逐像素阻尼因子 $\lambda$**（softplus 保正）与 8×8 逆深度上采样掩码；池化的动机是 ConvGRU 感受野小、需要全局上下文抵抗大移动物体等造成的误对应（§3.2 Update 段；附录 B Fig. 7 左的消融证实其价值）。

### 4.4 Dense BA 层（式 (4)、(5)）：高斯牛顿步的块结构与 Schur 补

**目标（式 (4)）**：整个帧图上的加权重投影代价：

$$\mathbf{E}(\mathbf{G}^*, \mathbf{d}^*) = \sum_{(i,j)\in\mathcal{E}} \Big\| \mathbf{p}^*_{ij} - \Pi_c\big(\mathbf{G}^*_j \circ \Pi_c^{-1}(\mathbf{p}_i, \mathbf{d}^*_i)\big) \Big\|^2_{\Sigma_{ij}}, \qquad \Sigma_{ij} = \mathrm{diag}\,\mathbf{w}_{ij} \tag{4}$$

论文语义（(4) 后原文）："we want an updated pose $\mathbf{G}^*$ and depth $\mathbf{d}^*$ such that reprojected points match the revised correspondence $\mathbf{p}^*_{ij}$ as predicted by the update operator"——BA 层不直接看图像，而是把 GRU 的修正对应当**观测**、把当前位姿/深度当**模型预测**，这正是学习组件与几何层的接口。

**高斯牛顿正规方程的块结构（逐步推导）**。记单条边、单个宿主像素 $u$ 的残差

$$\mathbf{e}_{ij}(u) = \mathbf{p}^*_{ij}(u) - \Pi_c\big(\mathbf{G}_j\circ\Pi_c^{-1}(\mathbf{p}_i, d_i)\big)(u) \;\in\; \mathbb{R}^2$$

（"观测 − 预测"约定；第 08 章 (8.1) 的同款结构，只是 $\mathbf{z}_{ij}$ 换成了 GRU 修正对应）。在当前估计处对增量变量一阶展开（依据：多元函数一阶 Taylor；位姿走切空间 $e^{\hat{\xi}}\mathbf{G}$，深度走矢量加法——附录 C 的局部参数化）：

$$\mathbf{e}_{ij}(u) \approx \mathbf{e}_{ij}^0(u) - J_{\xi_j}\,\Delta\xi_j - J_{\xi_i}\,\Delta\xi_i - J_{d}\,\Delta d_i$$

其中 $J_{\xi_j}, J_{\xi_i}, J_d$ 分别为投影对 $\mathbf{G}_j$、$\mathbf{G}_i$、$d_i$ 的雅可比（附录 C (13)、(14) 给出其解析式，见 §4.7）。加权堆叠全部边的残差、按 $\|\cdot\|^2_{\Sigma_{ij}}$ 白化（依据：马氏范数 = Cholesky 白化后的普通最小二乘，第 08 章 (8.2)），得正规方程 $H\Delta\mathbf{x} = -J^\top\Sigma^{-1}\mathbf{e}$（第 08 章 (8.5)）。**稀疏结构由"每个残差触达哪些变量"决定**（依据：稀疏模式 = 残差-变量依赖图）：

- 每个 $\mathbf{e}_{ij}(u)$ 依赖两个位姿 $\mathbf{G}_i, \mathbf{G}_j$（$2\times 6$ 列）与**恰好一个**深度变量 $d_i$（宿主像素自己的逆深度）。论文原话（§3.2）："Since each term in Eqn. 4 only includes a single depth variable, the Hessian matrix has block diagonal structure."
- 按 $[\Delta\boldsymbol{\xi}\,|\,\Delta\mathbf{d}]$ 分块：位姿-位姿块 $\mathbf{B}$（$6N\times 6N$，$N$=帧数）**稠密**——任意两位姿可经由共享边的残差产生耦合；位姿-深度块 $\mathbf{E}$ 稀疏（每残差行只有 2 个位姿块 × 1 深度列非零）；深度-深度块 $\mathbf{C}$ **对角**——深度变量只出现在自己射线上的残差里，不同深度间无耦合项。方程即 (5)：

$$\begin{bmatrix} \mathbf{B} & \mathbf{E} \\ \mathbf{E}^\top & \mathbf{C} \end{bmatrix}\begin{bmatrix} \Delta\boldsymbol{\xi} \\ \Delta\mathbf{d} \end{bmatrix} = \begin{bmatrix} \mathbf{v} \\ \mathbf{w} \end{bmatrix}, \quad \Delta\boldsymbol{\xi} = \big[\mathbf{B} - \mathbf{E}\mathbf{C}^{-1}\mathbf{E}^\top\big]^{-1}\big(\mathbf{v} - \mathbf{E}\mathbf{C}^{-1}\mathbf{w}\big), \quad \Delta\mathbf{d} = \mathbf{C}^{-1}\big(\mathbf{w} - \mathbf{E}^\top\Delta\boldsymbol{\xi}\big) \tag{5}$$

**Schur 补推导（无跳步）**。第二步（深度行）先解：$\mathbf{E}^\top\Delta\boldsymbol{\xi} + \mathbf{C}\Delta\mathbf{d} = \mathbf{w}\ \Rightarrow\ \Delta\mathbf{d} = \mathbf{C}^{-1}(\mathbf{w} - \mathbf{E}^\top\Delta\boldsymbol{\xi})$（依据：$\mathbf{C}$ 对角且正定，逐元素取倒数——论文："C is diagonal and can be cheaply inverted $\mathbf{C}^{-1} = 1/\mathbf{C}$"）。代入第一步（位姿行）：$\mathbf{B}\Delta\boldsymbol{\xi} + \mathbf{E}\,\mathbf{C}^{-1}(\mathbf{w} - \mathbf{E}^\top\Delta\boldsymbol{\xi}) = \mathbf{v}$，整理：

$$\big(\mathbf{B} - \mathbf{E}\mathbf{C}^{-1}\mathbf{E}^\top\big)\Delta\boldsymbol{\xi} = \mathbf{v} - \mathbf{E}\mathbf{C}^{-1}\mathbf{w}$$

（依据：分块高斯消元/Schur 补定义，[第 08 章 (8.15)](../08_后端-ii图优化与-ba.md) 的同一代数；先解位姿、再回代深度——回代与第 08 章 (8.14) 同型。）**与经典 BA 的角色对照**：第 08 章 (8.15) 消掉的是路标（$n\gg m$ 位姿时划算）；这里消掉的是**逐像素深度**（每帧 $H\times W$ 个，更是海量），保留的是位姿块——消元对象不同、代数完全同构，"Separating pose and depth variables, the system can be solved efficiently using the Schur complement"（§3.2 原话）。工程落地（§3.3/§3.4）：训练时在 PyTorch 里做稠密 BA、借自动微分展开块稀疏结构、再对约简相机块做稀疏 Cholesky；推理时用定制 CUDA 核利用块稀疏性 + 稀疏 Cholesky。**代价结构**（对照第 08 章 (8.16) 的口径）：稠密整体求解 $O((6N{+}M)^3)$ vs Schur 补 $O((6N)^3 + M)$（$M$ = 深度变量数，对角逆与稀疏回代近似线性）——$M\gg 6N$ 时量级优势同样成立。

### 4.5 权重 $\mathbf{w}$ 与阻尼 $\lambda$：学习出来的"鲁棒核"

经典 BA 的测量权重**事前固定**（每观测一个 $\Sigma_k$，[第 08 章 (8.2)](../08_后端-ii图优化与-ba.md)），鲁棒性靠 Huber 等鲁棒核的 IRLS 迭代。DROID 把两者都交给网络、随迭代自适应： 权重 $\mathbf{w}_{ij}$ 由 GRU **逐轮、逐像素、逐通道**预测，以 $\Sigma_{ij} = \mathrm{diag}\,\mathbf{w}_{ij}$ 进 (4)——语义是"GRU 认为这条对应有多可信"（附录 B Fig. 5 可视化：低纹理/遮挡区权重低）； 逐像素阻尼 $\lambda$（池化隐状态 → softplus 保正）是 LM 阻尼的逐像素版：与[第 08 章 (8.6)–(8.7)](../08_后端-ii图优化与-ba.md) 及[控制与规划第 04 章 (4.14)](../../control_planning/04_轨迹优化与最优控制.md) 的阻尼逆同骨架——保证 GN 步可解且方向下降。论文正文只说明 $\lambda$ 由池化隐状态预测、经 softplus 保证为正、作为 BA 的阻尼因子；其落点（加在 (5) 对角块）按开源实现与本仓库 [droidlite.py](../../../projects/slam/droidlite/droidlite.py) docstring 的对照写法（"论文把逐像素置信度阻尼放进 C"）理解。

### 4.6 递归收敛与 iLQR/DDP 的同构性

**收敛主张**（§3.2 原话）："Iterative applications of the update operator produce a sequence of poses and depths, with the expectation of converging to a fixed point $\{\mathbf{G}^{(k)}\}\to\mathbf{G}^*$, $\{\mathbf{d}^{(k)}\}\to\mathbf{d}^*$, reflecting the true reconstruction."——递归的目标是**定点**（重建即为定点解），而非单步最优。支撑该主张的两个机制都在输入端：每轮重算对应场 (3)（几何侧反馈，保证 BA 在最新估计处线性化）与 BA 残差流回灌 GRU（学习侧反馈，§4.3 "Inputs" 段）；本仓库 [droidlite.py](../../../projects/slam/droidlite/droidlite.py) 用噪声几何收缩调度复现前者的收敛效果、并断言逐轮 RMSE 单调不升（DEBUG.md 表 D-1）。

**与 iLQR/DDP 的同构**（[控制与规划第 04 章 §04.2](../../control_planning/04_轨迹优化与最优控制.md)，式号已 Read 核实）逐角色对照：

| 环节 | iLQR/DDP（教程 (4.5)–(4.13)） | DROID（论文 (2)–(5)） |
|---|---|---|
| 线性化点 | 标称轨迹 $(\hat{\mathbf{x}}_i, \hat{\mathbf{u}}_i)$，每轮由 rollout (4.13) 刷新 | 当前 $(\mathbf{G}^{(k)}, \mathbf{d}^{(k)})$，每轮由回缩 (2) 刷新 |
| 结构化求解 | 反向递推 (4.8)–(4.12)（利用动力学链式结构） | BA 层 (4)–(5)（利用"单深度变量"的块稀疏 + Schur） |
| 增量施加 | 前向 rollout (4.13)（带线搜索 $\alpha$） | 回缩 (2)（位姿走 $\mathrm{Exp}$、深度加法） |
| 收敛循环 | "反向递推 → 前向滚动构成一次迭代，循环至代价收敛"（教程原文） | "with the expectation of converging to a fixed point"（论文原文） |

三点同构： 都依赖"反复局部线性化 + 每轮重解"逼近定点/最优——这正是第 08 章 G-N 与 DDP 共享的骨架； BA 层内部又是一次微缩的"线性化-求解"，套层递归（分形结构）； 阻尼（(4.14) 的 $\mu I$ ↔ 逐像素 $\lambda$）同为信赖域逻辑。**一点差异**：iLQR 的增量由值函数二次型解析决定，其动力学要展开到一阶（iLQR）或二阶（DDP，教程 (4.10)）；DROID 的"测量模型"是固定且精确的几何投影（无需 DDP 式二阶项），而**步的方向由学出来的 GRU 提议**——网络隐状态吸收历史 BA 残差（§4.3 的反馈输入），相当于一个可训练的步长/方向提议器。

### 4.7 投影与雅可比（附录 C，式 (6)–(14)；两处排印注）

warp 函数 (6)：

$$\mathbf{p}' = \Pi_c\big(\mathbf{G}_{ij}\cdot\Pi_c^{-1}(\mathbf{p},\mathbf{d})\big), \qquad \mathbf{G}_{ij} = \mathbf{G}_j\circ\mathbf{G}_i^{-1} \tag{6}$$

针孔投影/反投影 (7)：$\Pi_c(\mathbf{X}) = \big(f_x\frac{X}{Z} + c_y,\; f_y\frac{Y}{Z} + c_y\big)$、$\Pi_c^{-1}(\mathbf{p},d) = \big(\frac{p_x - c_x}{f_x},\ \frac{p_y - c_y}{f_y},\ \frac{1}{d}\big)^{\!\top}$，内参 $c = (f_x, f_y, c_x, c_y)$。**排印注一**（高分辨率渲染核对）：$\Pi_c$ 第一分量印作 $+c_y$；按内参定义与反投影第一分量用 $(p_x - c_x)/f_x$，应为 $+c_x$——笔误。投影雅可比 (8)：

$$\frac{\partial\Pi_c(\mathbf{X})}{\partial\mathbf{X}} = \begin{pmatrix} f_x\frac{1}{Z} & 0 & -f_x\frac{X}{Z^2} & 0 \\ 0 & f_y\frac{1}{Z} & -f_y\frac{Y}{Z^2} & 0 \end{pmatrix}, \qquad \frac{\partial\Pi_c^{-1}(\mathbf{p},d)}{\partial d} = \begin{pmatrix}0\\0\\1\end{pmatrix}$$

（依据：对 $X/Z, Y/Z$ 与 $c$ 的偏导逐项求出；齐次坐标第 4 列导数为零。）

位姿雅可比 (9)–(12)：对局部参数化 $e^{\hat{\xi}_i}\mathbf{G}_i$、$e^{\hat{\xi}_j}\mathbf{G}_j$，变换后的齐次点为

$$\mathbf{X}' = \mathrm{Exp}(\xi_j)\circ\mathbf{G}_j\circ\mathbf{G}_i^{-1}\circ\mathrm{Exp}(-\xi_i)\circ\mathbf{X} \tag{9}$$

用伴随算子把 $\xi_i$ 项移到表达式前部（(10)）："using the adjoint operator to move the $\xi_i$ term to the front of the expression"：

$$\mathbf{X}' = \mathrm{Exp}(\xi_j)\circ\mathrm{Exp}\big(-\mathrm{Adj}_{\mathbf{G}_j\mathbf{G}_i^{-1}}\,\xi_i\big)\circ\mathbf{G}_j\circ\mathbf{G}_i^{-1}\circ\mathbf{X} \tag{10}$$

（依据：伴随/共轭恒等式 $\mathrm{Exp}(\xi)\circ G = G\circ\mathrm{Exp}(\mathrm{Adj}_{G^{-1}}\xi)$ 的等价变体；第 02 章的伴随变换。）于是两个位姿的 $4\times 6$ 雅可比为同一矩阵型（齐次坐标下 $\big[\,W'I \;\big|\; -[\mathbf{x}']_\times\,\big]$ 结构，$W',X',Y',Z'$ 为变换后齐次点的分量）：

$$\frac{\partial\mathbf{X}'}{\partial\xi_j} = \begin{pmatrix} W' & 0 & 0 & 0 & Z' & -Y' \\ 0 & W' & 0 & -Z' & 0 & X' \\ 0 & 0 & W' & Y' & -X' & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 \end{pmatrix}, \qquad \frac{\partial\mathbf{X}'}{\partial\xi_i} = -\frac{\partial\mathbf{X}'}{\partial\xi_j}\Big|_{\text{块}}\cdot\mathrm{Adj}_{\mathbf{G}_j\mathbf{G}_i^{-1}} \tag{11}$$

（(12) 的符号与 $\mathrm{Adj}_{\mathbf{G}_j\mathbf{G}_i^{-1}}$ 因子来自 (10) 中 $-\mathrm{Adj}\,\xi_i$ 的链式；负号源于 $\mathrm{Exp}(-\xi_i)$。）链式合成像素雅可比 (13)：$\frac{\partial\mathbf{p}'}{\partial\xi_j} = \frac{\partial\Pi_c(\mathbf{X}')}{\partial\mathbf{X}'}\frac{\partial\mathbf{X}'}{\partial\xi_j}$，$\frac{\partial\mathbf{p}'}{\partial\xi_i}$ 同理（依据：复合函数求导）。深度雅可比 (14) 论文写作

$$\frac{\partial\mathbf{p}'}{\partial d} = \frac{\partial\Pi_c(\mathbf{X}')}{\partial\mathbf{X}'}\,\frac{\partial\mathbf{X}'}{\partial\mathbf{X}}\,\frac{\partial\Pi^{-1}(\mathbf{p},d)}{\partial d} = \frac{\partial\Pi_c(\mathbf{X}')}{\partial\mathbf{X}'}\begin{pmatrix}t_x\\ t_y\\ t_z\\ 1\end{pmatrix}, \quad (t_x,t_y,t_z)=\mathbf{G}_j\circ\mathbf{G}_i^{-1}\ \text{的平移} \tag{14}$$

**排印注二**：链中 $\frac{\partial\Pi^{-1}}{\partial d}$ 按 (8) 印作 $(0,0,1)^\top$，但 (7) 中 $\Pi^{-1}$ 第三分量是 $1/d$、其导数应为 $-1/d^2$；(14) 末端的齐次 4 维列 $(t_x,t_y,t_z,1)$ 与该链也无法闭合（$4\times 4$ 的 $\partial\mathbf{X}'/\partial\mathbf{X} = \mathbf{G}_{ij}$ 乘方向量应给出 $R_{ij}$ 的第 3 列而非平移列）。按 (7) 的参数化从第一性重推（依据：$\mathbf{X}' = R_{ij}\,\mathrm{ray}/d + t_{ij}$、$\partial\mathbf{X}'/\partial d = -R_{ij}\mathbf{e}_3/d^2$，$\mathbf{e}_3 = (0,0,1)^\top$，$\mathrm{ray} = \Pi_c^{-1}(\mathbf{p})\cdot d$ 的归一化射线）：

$$\frac{\partial\mathbf{p}'}{\partial d} \;=\; -\frac{1}{d^2}\,\frac{\partial\Pi_c}{\partial\mathbf{X}'}\,R_{ij}\,\mathbf{e}_3$$

即"投影雅可比 × 相对旋转第三列 ÷ $d^2$"——这正是本仓库 [droidlite.py](../../../projects/slam/droidlite/droidlite.py) `jacobian_matrix` 实现并经有限差分校验的形式（docstring：`dq/drho_i = -(R_k ray_i)/rho_i²`）。读附录时以 (6)(9)(13) 为骨架，两处排印以第一性推导修正——引用附录公式时须自行复核。

### 4.8 学习组件与传统优化层的分工：哪些可微、梯度怎么流

**论文的实际说法**（如实转述，勿引申）： **DBA 层在计算图内**——§3.2 末句原文 "The DBA layer is implemented as part of the computation graph and backpropagation is performed through the layer during training"：训练时梯度**显式穿过高斯牛顿层的展开迭代**；论文全文未提隐函数定理/固定点反传，机制就是展开计算图上的自动微分； **群元素反传在切空间**——§3.3 "we use the LieTorch extension [50] to perform backpropagation in the tangent space of all group elements"（对应第 08 章 (8.9)–(8.10) 的切空间参数化与左扰动雅可比）； **训练/推理的求解器分工**——§3.4：训练用 PyTorch 稠密 BA + 自动微分（利用块稀疏结构），推理用 CUDA 核 + 约简相机块上的稀疏 Cholesky（只为速度，数学相同）； **监督**——§3.3 Supervision：位姿损失 $\mathcal{L}_{pose} = \sum_i\big\|\mathrm{Log}_{SE(3)}(\mathbf{T}_i^{-1}\mathbf{G}_i)\big\|_2$（真值位姿与预测位姿的测地距离，第 02 章对数映射）+ 光流损失（预测流与真值流的平均 L2 距离），**施加在每一轮的输出上、权重随轮数指数增大（$\gamma = 0.9$）**——深监督 + 后期加权，与定点收敛诉求一致； **规范自由度**——§3.3 Removing gauge freedom：单目只能恢复到相似变换，训练时把前两帧位姿钉在真值上消去规范（第一帧去 6 维、第二帧定尺度），因为"gauge freedom still exists during training which poorly impacts the conditioning of the linear system and the stability of the gradients"；推理时 frontend 对前两帧同样处理、其余帧把全部深度当自由变量（§3.4）。

**分工总结**：GRU（学习）负责"下一次对应场应该长什么样、哪些像素可信"；BA 层（固定几何）负责"在几何约束下把提议投影成合法的位姿/深度更新"。可微性把两者焊在一起；消融（附录 B Fig. 7 右，"RAFT + BA"对照）显示**训练时若不用 DBA 层**（先学光流、测试时再上 BA），"the SLAM system is unstable and prone to failure if the DBA is not used during training"——学习组件必须在与部署一致的优化结构内训练。

### 4.9 双目与 RGB-D：一行目标函数扩展（§3.4 末段）

传感器有深度时**深度仍当变量**（"since sensor depth can be noisy and have missing observations"），只在 (4) 上加一项惩罚"实测深度与预测深度的平方距离"——深度测量的角色从"硬约束"降级为"软先验"，噪声与缺失观测天然被处理。双目则是"exact same system described above, with just double the frames"（左右目各算一帧）+ DBA 层内固定左右相机相对位姿 + 帧图中的跨相机边利用双目信息。这就是 §1 "generalization" 主张的机制：泛化不需要新模块，只需要目标函数多一行。

### 4.10 系统层：前端、后端与全局 BA（§3.4）

**Frontend**（新帧处理）：初始化攒 12 帧（保留光流 > 16 px 的前一帧）；新关键帧向 3 步长（3 timesteps）内的邻接帧建边，初始位姿用线性运动模型外推，随后跑 **10 轮**更新算子；非关键帧用 **motion-only BA**（迭代估计它与邻近关键帧间的光流并只优化该帧位姿）恢复全轨迹相机速率输出——与 VINS-Mono 的 motion-only BA（其 §VI-E，约 5 ms）同一工程手法（见[精读/VINS-Mono](./VINS-Mono_RAM2018.md) §3）。冗余帧按平均光流幅值移除、无好候选时删最老帧。**Backend**：对**全部关键帧历史**周期性做全局 BA——每次迭代重建帧图：先用 $N\times N$ 平均光流距离矩阵加时序相邻边，再按距离升序采样边、每选一条边补 Chebyshev 距离 $\|\,(i,j)-(k,l)\,\|_\infty < k$ 的邻域边——**这就是"periodic global BA over all keyframes"**：没有独立的位姿图/回环模块，长程边直接进全局 BA，回环收益由同一优化层吸收（§3.4 Backend 原文语义）。全部关键帧图像上做全 BA（§3.4 末段）。

## 5. 实验与结果解读

- **TartanAir（Table 1，训练同分布）**：单目平均 ATE **0.24 m**（逐序列 0.08/0.05/0.04/0.02/0.01/1.31/0.30/0.07，全部零失败），同表 ORB-SLAM 5.03、DSO 1.92、DeepV2D 4.93；重训的 DeepV2D/TartanVO 也分别差 20×/8× 量级（§4 TartanAir 段的倍数表述）。ECCV 2020 SLAM 竞赛口径（Table 2）：Droid 得分 0.129（mono）/0.047（stereo），领先前三名提交；比依赖 COLMAP 的冠军方案**快 16 倍**、单目/双目误差分别低 62%/60%（§4 TartanAir 段与 §1 High Accuracy 列表）。
- **EuRoC（Table 3，跨数据集泛化）**：单目平均 ATE **2.2 cm**（正文 §4 EuRoC 段原话 "we achieve an average ATE of 2.2cm"；表内 "Ours" 行均值 0.022），比零失败方法低 82%，在与 ORB-SLAM3 双双成功的序列上低 43%；同表 "Ours (odometry only)"（消融行）均值 0.186 m——**全局 BA 的价值一目了然**（同一网络、只去掉全局优化，差约 8 倍）。双目（Table 5）：均值 0.024 m，比 ORB-SLAM3 的 0.084 低 71%。表中最强的学习式对手 D3VO+DSO 只在 11 条序列中的 9 条上评测、并对含同场景的剩余序列做无监督训练（§4 EuRoC 段对其口径的说明）——DROID-SLAM 不用任何 EuRoC 数据。
- **TUM-RGBD（Table 4）**：均值 **0.011 m**（ORB-SLAM2 0.018、ORB-SLAM3 0.041、DSO 0.104、DeepV2D 0.375、DeepFactors 0.225）；零失败方法中比 DeepFactors 低 83%、比 DeepV2D 低 90%（§4 TUM-RGBD 段）。读表注意其脚注：所有方法都只给单目视频，"†" 标 DeepTAM 用 RGB-D、"²" 标 TartanVO 用真值定标相对位姿——横比口径并不完全对齐。
- **ETH3D SLAM（Table 6 / Fig. 4）**：AUC train/test 360.42/307.79 排名第一；RGB-D 32 序列成功 **30** 个（次优 19/32），含全黑序列——鲁棒性主张的直接证据（§1 High Robustness、§4 ETH3D-SLAM 段）。
- **消融（附录 B）**：Fig. 6 左：双目优于单目、全局优化优于纯局部 BA（精度随关键帧数变化的曲线中 **5 关键帧**为加粗的默认选择）；Fig. 7 左：GRU 全局池化有收益；Fig. 7 右：训练不带 DBA 层则系统不稳定、易失败（§4.8 的引文）。
- **实时性/资源（§4 Timing and Memory）**：单卡 RTX-3090 实时，全局 BA 与回环（帧图长程边）在后台线程；EuRoC 以 320×512 隔帧采样跑 20 fps（相机帧率口径），TUM-RGBD 240×320 约 30 fps，TartanAir 因相机运动更快约 8 fps；frontend 只需 8 GB 显存，backend 需存全部图像特征图——TUM-RGBD 单张 1080Ti 可跑，TartanAir/ETH-3D（视频可达 5000 帧）需 24 GB。训练：TartanAir 合成单目、250k 步、batch 4、384×512、7 帧片段、展开 15 轮更新，4×RTX-3090 一周（§4 开头）。

## 6. 局限与后续影响

**局限**（论文自述 + 结构性）： **GPU 依赖**——实时性完全绑定桌面级 GPU，论文自述资源需求是 "the biggest limitation of our system"（§4 Timing and Memory）； **全对相关体的内存 $O(N^2)$**——§4.2 的论证：帧数大时（ETH-3D/TartanAir 可达 5000 帧）相关体存储超显存，只能按需重算（RAFT 式）；backend 的帧图重建（$N\times N$ 距离矩阵）与全局 BA 同样随 $N$ 二次增长；论文给出的出路（自述）："can be drastically reduced by culling redundant computation and more efficient representations"； **实时性边界**——高速运动场景逼近算力上限（TartanAir 8 fps），每轮的稠密对应场/相关体查找与逐像素深度变量是主要开销； **评测口径**——ATE 依赖对齐（§4 开头自述未评测 3D 重构质量，"typically considered in the domain of Multiview Stereo and outside the scope of this work"）； **训练-推理分布差距仍在**——跨数据集精度已超经典系统，但同分布（TartanAir）裕度最大（Table 1 vs Table 3 的对比读法）。

**后续影响**：确立了"可微优化层 + 学习更新算子"的建模范式——后续 GPU SLAM / 深度 SLAM（本仓库 [papers/slam/frontier/](../../../papers/slam/frontier/) 收录的 NeRF-SLAM、MASt3R-SLAM 等）普遍继承"DROID 式 BA 层 + 相关体查找 + GRU 递归"骨架；RAFT 光流 + 可微 BA 的组合也成为视频深度估计与稠密 SLAM 的标准组件。

## 7. 与本项目对照

- **教程章节**：[第 10 章 §10.4](../10_建图与系统实战.md)（前沿导读将 DROID-SLAM 列为可微优化路线代表，droidlite 为其结构演示模块）；[第 08 章](../08_后端-ii图优化与-ba.md)——(8.1) 重投影误差 ↔ (4) 的残差、(8.2)/(8.5) 加权目标与正规方程 ↔ (4)–(5) 的导出、(8.9)/(8.10) 流形更新与切空间雅可比 ↔ (2) 与附录 C、**(8.14)/(8.15) 回代与 Schur 补 ↔ (5) 逐式对应**、(8.16) 复杂度口径（经典 BA 消路标 vs 这里消逐像素深度）、(8.17) 概率语义；[METRICS.md](../../../projects/slam/METRICS.md) §1 的规范自由度讨论 ↔ §3.3 的 gauge 处理。
- **跨领域同构**：递归"重线性化-求解"与[控制与规划第 04 章 §04.2](../../control_planning/04_轨迹优化与最优控制.md) 的 iLQR/DDP 迭代（(4.5)–(4.13)）同构（§4.6 对照表）；逐像素阻尼 $\lambda$ 与 (4.14)/(8.6) 的 LM 阻尼同骨架。
- **代码**：[projects/slam/droidlite/droidlite.py](../../../projects/slam/droidlite/droidlite.py)——首行声明 **"Structural demo, no learned components"**。逐式对应（docstring 自述，已 Read 核实）： 论文 (1) 的光流来源被替换为 `make_sequence` 的合成 GT 光流 + 高斯噪声（结构性替换）； (3) = `DenseBA.predicted_flow`（每轮用当前位姿/深度重算对应场，Algorithm 1 复位）； (4) = `DenseBA.residuals`（修正项 $r_{ij}$ = 当轮光流测量 − 当前预测，故 $p^* = p_i + f^{\mathrm{meas}}$，逐项对应）； (5) = `DenseBA._schur_step`：$\mathbf{B} = J_x^\top W J_x$、$\mathbf{E} = J_x^\top W J_d$、$\mathbf{C} = \mathrm{diag}(J_d^\top W J_d) + \lambda$（对角），先解位姿方程、回代 $\Delta\mathbf{d} = \mathbf{C}^{-1}(\ldots)$； (2) = `DenseBA.solve` 主循环的回缩（位姿 `se3_exp` 左乘、逆深度加法）； Algorithm 1 = `solve` 的 $R$ 轮循环（解析雅可比 `jacobian_matrix`，复用 `direct.projection_jacobian`）。
- **测试驱动**：由 `tests/test_droidlite.py` 直接驱动（docstring 自述）——打印逐轮位姿/逆深度 RMSE 的收敛曲线，并对 Schur 消元与 `core.solver.gauss_newton` 稠密解做受控交叉验证（§7 健康值即来自这些打印与 [DEBUG.md](../../../projects/slam/DEBUG.md) 表 D）。
- **保真度声明与刻意差距**（docstring + 参数默认值，已核实）： **GT 光流替代网络**——学习组件不实现，光流噪声按 `flow_decay^round`（默认 0.5）几何收缩，模拟"GRU 隐状态反馈使光流逐步收敛到定点"（论文 §3.2 原话入 docstring）； **32×32 宿主网格**（`grid=32`，$N = 1024$ 个宿主点，把论文的逐像素稠密逆深度降采样）； **递归 6 轮**（`n_rounds=6`；论文 frontend 每帧 10 轮、训练展开 15 轮）； 权重 $\mathbf{w}_{ij}$ 取全 1（Huber 核可选替代，`huber_delta=None`）； 阻尼取标量 LM $\lambda$ 而非论文的逐像素 $\lambda$； 规范自由度显式锚定 $\mathbf{G}_0 = I$（第 0 帧即世界系/宿主帧），其余位姿以左扰动坐标进向量正规方程（教程 (8.9)–(8.10)），而非论文的阻尼最小二乘隐式固定。**为什么不实现学习组件**：教学目标是让 (3)–(5) 的几何-优化结构在 CPU + numpy 上可跑、可断言、可复现（合成数据确定性种子）；网络组件需要 GPU、训练权重与数据管线，而其作用面（光流质量与置信度）可用带噪 GT 光流受控模拟——结构的正确性（Schur 消元、递归收敛、雅可比）才是本模块要验证的命题。
- **健康值**（实测，以 [DEBUG.md](../../../projects/slam/DEBUG.md) 表 D 为准）：递归收敛单调不升——位姿 RMSE **0.064 → 5.2e-5（增益 ~1225×）**、逆深度 RMSE 0.0038 → 6.1e-5（~63×）；**Schur 补 vs 稠密求解交叉验证**（`use_dense_gn=True` 走 `core.solver.gauss_newton` 解同一正规方程）：单步差 ~1e-10、完整求解逆深度差 9.1e-7、位姿差 4.3e-7（**< 1e-5** 门限；numpy 版本敏感点，超 1e-2 即消元实现错，DEBUG.md §1.3）；深度对角块 $\mathbf{C}$ 全部为正（有零/负值即 `valid` 掩码漏像素）。另见 [METRICS.md](../../../projects/slam/METRICS.md) 健康值表 droidlite 行（第 6 轮 0.00005，初值 0.064）。

## 配套阅读

- 本文 PDF：[DROID-SLAM（Teed & Deng, NeurIPS 2021）](../../../papers/slam/frontier/arXiv-2108.10869_DROID-SLAM.pdf)（本精读所有式号以此 PDF 为准）
- 光流源头：RAFT（Teed & Deng, ECCV 2020，本文引文 [29]）——相关体、Lookup、GRU 更新的血统
- 姊妹精读：[VINS-Mono](./VINS-Mono_RAM2018.md)（逆深度宿主帧参数化与 motion-only BA 的经典对应物）｜[DSO](./DSO_PAMI2018.md)（同为"逆深度 + 滑窗 GN"，把固定权重换成学习版即是本文的思路桥）｜[ORB-SLAM3](./ORB-SLAM3_TRO2021.md)｜[LSD-SLAM](./LSD-SLAM_ECCV2014.md)
- 教程：[第 10 章](../10_建图与系统实战.md)（§10.4 前沿导读）｜[第 08 章](../08_后端-ii图优化与-ba.md)（(8.5)/(8.14)/(8.15)：正规方程、回代、Schur 补——本文 BA 层的全部代数）
- 控制侧对照：[控制与规划第 04 章](../../control_planning/04_轨迹优化与最优控制.md)（§04.2 iLQR/DDP 的"反复线性化"同构）
- 代码：[projects/slam/droidlite/](../../../projects/slam/droidlite/) ｜ [projects/slam/direct/](../../../projects/slam/direct/)（复用的平面场景合成与投影雅可比）｜ [projects/slam/DEBUG.md](../../../projects/slam/DEBUG.md)（表 D：droidlite 断点与交叉验证门限）
- 主题导航：[tutorials/slam/README.md](../README.md) ｜ [OVERVIEW.md](../OVERVIEW.md)

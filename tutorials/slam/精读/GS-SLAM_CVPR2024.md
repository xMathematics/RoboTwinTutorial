# 论文精读｜GS-SLAM：把 3D 高斯泼溅装进稠密视觉 SLAM（CVPR 2024 Highlight）

> **PDF**：[papers/slam/frontier/arXiv-2311.11700_GS-SLAM.pdf](../../../papers/slam/frontier/arXiv-2311.11700_GS-SLAM.pdf)（arXiv v4，2024-04-07，16 页含补充材料，本精读所有式号以此 PDF 为准）｜ **教程**：[第 10 章 §10.4](../10_建图与系统实战.md)（前沿深读扩展）｜ **3DGS 本体**：[3D 重建教程第 08 章](../../3d_reconstruction/08_3D高斯泼溅.md)（式 (8.1)–(8.11)，已核实后回引）｜ **代码**：无（前沿路线说明）

## 1. 论文信息与一句话贡献

- **题目**：GS-SLAM: Dense Visual SLAM with 3D Gaussian Splatting
- **作者**：Chi Yan、Delin Qu、Dan Xu、Bin Zhao、Zhihang Wang、Dong Wang、Xuelong Li（上海 AI Lab / 复旦 / HKUST / 西北工业大学 / TeleAI）
- **发表**：CVPR 2024（Highlight）；arXiv:2311.11700
- **一句话贡献**：**第一个把 3D 高斯泼溅（3DGS）显式辐射场 + 可微泼溅光栅化管线搬进稠密 RGB-D SLAM 的系统**——为解决 3DGS 原管线"假设 SfM 给好初始点云与位姿"的 SLAM 不适配问题，推导了相机位姿的**解析梯度**（走链式法则穿过投影协方差/不透明度/投影中心三个中间变量），设计了**自适应高斯扩展**（按累积不透明度与深度残差增删高斯，重建新观测区域、抑制浮点），配合 coarse-to-fine 跟踪，在 Replica 上平均 ATE 0.50 cm、渲染 386 FPS，运行 8.34 FPS，做到精度—速度的新均衡。

## 2. 问题与动机

**要解决什么**（§1–2）：NeRF 系稠密 SLAM（iMAP、NICE-SLAM、ESLAM、Point-SLAM 等）的共同瓶颈是**逐光线的体渲染**——想出高分辨率图就得付 $O(N_{\text{采样}})$ 次网络/特征查询，于是它们"only render a small set of pixels to reduce optimization time"（§1 原话），后果是稠密地图缺少细节与真实感（"the reconstructed dense maps lacking the richness and intricacy of details"，§2 开头）。3DGS（Kerbl et al., SIGGRAPH 2023，本地 PDF 即本文引文 [13]）用排序 + alpha 混合的**泼溅光栅化**在 1080p 上跑到实时，恰好对症；但把它当 SLAM 地图有三道坎（§2 "3D Gaussian Representation" 段 + §2.1 首段）： 原管线的前提是**已初始化的点云与相机位姿**（SfM 离线产物，"prerequisites of initialized point clouds or camera pose inputs [28]"）——SLAM 里位姿恰恰是待估量； 渲染损失对位姿的梯度在官方实现里靠**数值/自动微分穿过整个光栅化器**，论文要的是**解析的位姿梯度**（"we derive the analytical derivative equations for pose estimation in the Gaussian representation"，§3 开头）； 高斯初始化后不会自动长出新观测区域，还会积累浮点（floaters）——需要一套**面向 SLAM 的扩展/删除策略**（§1 贡献 2："this strategy is essential to extend 3D Gaussian representation to reconstruct the whole scene rather than synthesize a static object in existing methods"）。

**定位**（§2 末段）：MLP-based（iMAP、NICE-SLAM：内存友好但灾难性遗忘）、Hybrid（ESLAM、Vox-Fusion：特征网格 + 解码器）、MLP-free/Explicit（本文所属：直接优化高斯参数，光栅化反传，无需解码网络——Table 4 的 "Decoder param 0 M" 即此卖点的量化）三分类中，本文把 explicit 路线推到 SLAM。

## 3. 方法总览

```
RGB-D 流 {I_i, D_i}，已知内参 K（第 04 章 (4.4)）
   │
   ▼  【建图（关键帧上）§3.2：自适应 3D 高斯扩展】
   ├─ 首帧：均匀采一半像素反投影 → 初始化 M = HW/2 个高斯（式 (7)）
   ├─ 每个关键帧：先渲染 RGB-D → 累积不透明度 T（式 (8) 判据）找"没盖住/深度对不上"
   │    的像素 → 反投影成新高斯；可见视锥内"悬空"高斯按射线-表面距离衰减 opacity（式 (9)）
   ├─ 以 L_c + L_d（式 (6)）优化全部高斯参数
   ▼
   ▼  【跟踪（逐帧）§3.3：可微位姿估计 + coarse-to-fine】
   ├─ 恒速假设初始化（前两帧相对变换外推）
   ├─ coarse：H/2×W/2 稀疏渲染，优化光度损失 L_track（式 (10)）T_c 轮 → 粗位姿 P_c
   ├─ 可靠高斯筛选（式 (12)：|D_i − d_i| ≤ ε，剔除远离表面的噪声高斯）
   ├─ fine：全分辨率渲染 T_f 轮 → 最终位姿 P
   ├─ 关键帧插入：可靠区域比例 + 与最近关键帧差值 > μ_k
   ▼
   ▼  【BA 阶段（后台）§3.3 末：随机选 K 关键帧，联合优化位姿 + 地图】
        L_ba（式 (13)）：前半程只优化地图 S，后半程地图与位姿同调
```

跟踪与建图**并行**（"In the parallel camera tracking phase"，§3.3），与 [PTAM](./PTAM_ISMAR2007.md) 以来的跟踪/建图双线程传统同构；跟踪基于"渲染图像 vs 观测图像"的光度残差——这是把 [直接法](../06_视觉里程计-ii直接法.md) 的光度残差搬进可微渲染域：深度在 RGB-D 版本里通过式 (6) 的 $\mathcal{L}_d$ 进高斯优化、通过式 (12) 的筛选进跟踪，但不直接进跟踪损失（见 §4.5）。

**数据关联的隐式化**（与经典系统对照读）：特征点法的三角化+匹配（[第 05 章](../05_视觉里程计-i特征点法.md)）在这里被"**渲染即关联**"取代——每帧跟踪时，哪些高斯落在哪些像素、各贡献多少权重，由式 (4) 的排序混合一次性给出且**随位姿可微变化**；位姿更新后对应关系自动刷新，无需显式数据关联步骤。代价是对应质量的"真值"不存在：错误高斯（浮点、错位）会持续产生系统性渲染误差，这正是式 (9) 删除步与式 (12) 筛选步存在的理由——SLAM 的致密化不只是 3DGS 致密化的复用，还要承担"地图维护/清洗"的职责。

**相对原版 3DGS 管线（Kerbl et al. [13]）的四处 SLAM 化改造**（贯穿后文推导，先立清单）：

1. **位姿进优化**：原管线位姿由 SfM 给定、不参与训练；本文把 $\mathbf{P}$ 变成变量并推导解析梯度（§4.4，式 (11)）。
2. **地图可增长**：原管线一次性拟合静态场景；本文加"增（式 (8)）/删（式 (9)）"两步，让高斯集合随相机运动长出新区域。
3. **致密化重定义**：原版 clone/split 由位置梯度阈值触发；本文新增判据基于**累积不透明度与深度一致性**（§4.3），因为 SLAM 里"新内容"来自新视点而非训练盲区。
4. **损失换尺**：原版 L1+D-SSIM（[3D 重建教程第 08 章 (8.11)](../../3d_reconstruction/08_3D高斯泼溅.md)）面向新视角合成；本文换为 L1 颜色 + L1 深度（式 (6)），深度是 RGB-D SLAM 的独特监督通道（建图用、BA 用、筛选用）。

## 4. 关键公式推导（式号按论文 PDF 原文 (1)–(13)；补充材料 (14)–(19)）

### 4.0 符号表

| 符号 | 含义 |
|---|---|
| $\mathcal{G} = \{G_i : (\mathbf{X}_i, \Sigma_i, \Lambda_i, \mathcal{Y}_i)\}$ | 3D 高斯集合（式 (1)）：位置 $\mathbf{X}_i\in\mathbb{R}^3$、协方差 $\Sigma_i\in\mathbb{R}^{3\times3}$、不透明度 $\Lambda_i\in\mathbb{R}$、1 度球谐 $\mathcal{Y}_i\in\mathbb{R}^{12}$（3 通道 × 4 系数） |
| $\mathbf{P} = \{R, \mathbf{t}\}$ | 相机位姿（世界→相机）；旋转以 4 维四元数存储（补充材料 §1） |
| $\mathbf{S}\in\mathbb{R}^3,\ R$ | 尺度向量 / 旋转矩阵（式 (2) 的协方差分解） |
| $\mathbf{J}$ | 投影的仿射近似雅可比（式 (3)；即 [3D 重建教程第 08 章 (8.6)](../../3d_reconstruction/08_3D高斯泼溅.md)） |
| $\Sigma',\ \alpha_i,\ \mathbf{c}_i,\ \mathbf{m}_i$ | 2D 投影协方差 / 第 $i$ 高斯的混合权重 / 颜色 / 投影中心（式 (3)(4)） |
| $d_i$ | 第 $i$ 高斯中心在相机系的 $z$ 坐标（深度，式 (5)） |
| $\hat{C}_m,\ \hat{D}_m$ | 像素 $m$ 的渲染颜色 / 深度；$C_m,\ D_m$ 为传感器观测（式 (6)） |
| $T$ | 逐像素**累积不透明度** $T = \sum_{i\in N}\alpha_i\prod_{j=1}^{i-1}(1-\alpha_j)$（式 (8) 前） |
| $\tau_T, \tau_D, \gamma, \eta, \epsilon, \mu_k$ | 判据与超参（式 (8)(9)(12)；$\eta\ll1$） |
| $T_c, T_f$ | coarse / fine 跟踪迭代数（补充材料 Algorithm 1：前 5 轮 coarse） |
| $K$ | BA 随机抽取的关键帧数（式 (13)；勿与内参 $K$ 混淆） |

### 4.1 地图表示：高斯集合与协方差参数化（式 (1)(2)）

场景 = 各向异性 3D 高斯的集合（式 (1)）：$\mathcal{G} = \{G_i : (\mathbf{X}_i, \Sigma_i, \Lambda_i, \mathcal{Y}_i)\,|\, i = 1,...,N\}$。协方差必须对称正定，直接优化 6 个独立元素不可行——按 3DGS 论文（本文引文 [13]）参数化为式 (2)：

$$\Sigma = R\,S\,S^\top R^\top \tag{2}$$

其中 $S\in\mathbb{R}^3$ 为尺度向量、$R\in\mathbb{R}^{3\times3}$ 为旋转（存 4 维四元数）。**合法性证明**（依据：[3D 重建教程第 08 章 (8.2)–(8.3)](../../3d_reconstruction/08_3D高斯泼溅.md)，逐字同构）：对任意 $\mathbf{v}$，$\mathbf{v}^\top\Sigma\mathbf{v} = (SR^\top\mathbf{v})^\top(SR^\top\mathbf{v}) = \|SR^\top\mathbf{v}\|^2 \ge 0$——末式是向量自身内积恒非负（$R$ 正交、$S$ 对角非负），故任意参数取值自动半正定，无需投影回可行域；反之由实对称矩阵谱定理该分解不损失表达力。与原版 3DGS 的两处 SLAM 化裁剪：球谐降为 **1 度**（12 系数，§3.1；原 3DGS 默认 3 度），另有零阶"light 版"（§4.4，省内存）；不透明度 $\Lambda_i$ 与颜色 $\mathcal{Y}_i$ 仍可学习。

### 4.2 泼溅渲染：颜色与深度（式 (3)(4)(5)）

**投影**：位姿 $\mathbf{P} = \{R, \mathbf{t}\}$ 下（记 $\mathbf{P}^{-1}$ 为世界→相机），3D 协方差经局部仿射近似传到屏幕（式 (3)）：

$$\Sigma' = \mathbf{J}\,\mathbf{P}^{-1}\,\Sigma\,\mathbf{P}^{-\top}\,\mathbf{J}^\top \tag{3}$$

（依据：两次协方差线性变换引理 $A\Sigma A^\top$——先 $\mathbf{P}^{-1}$ 把 3D 高斯搬进相机系，再 $\mathbf{J}$（透视投影在中心的仿射近似）压到 2D；推导链见 [3D 重建教程第 08 章 (8.4)–(8.8)](../../3d_reconstruction/08_3D高斯泼溅.md)，$\mathbf{J}$ 的显式矩阵即其 (8.6)，$f_x/Z,\ -f_xX/Z^2$ 等元素与[第 04 章 (4.3)](../04_相机模型与特征提取.md) 的透视除法逐项对应。$\mathbf{J}$ 按投影中心 $\mathbf{m}_i$ 处取值。）颜色按深度排序做 front-to-back alpha 混合（式 (4)）：

$$\hat{C} = \sum_{i\in N} \mathbf{c}_i\,\alpha_i\prod_{j=1}^{i-1}(1-\alpha_j) \tag{4}$$

（依据：[3D 重建教程第 08 章 (8.10)](../../3d_reconstruction/08_3D高斯泼溅.md) 的同构式；$\alpha_i$ = 学习的不透明度 $\Lambda_i$ × 2D 高斯在像素处求值（其 (8.9)），$\prod_{j<i}(1-\alpha_j)$ 是透射。）深度同样混合（式 (5)）：

$$\hat{D} = \sum_{i\in N} d_i\,\alpha_i\prod_{j=1}^{i-1}(1-\alpha_j) \tag{5}$$

其中 $d_i$ 是第 $i$ 高斯中心投影到相机系 $z$ 轴的深度（§3.1："obtained by projecting to z-axis in the camera coordinate"）——**深度不是高斯的期望终止距离，而是"中心的 z 值按混合权重的加权平均"**，与 [NeRF-SLAM 精读](./NeRF-SLAM_ICRA2023.md) 式 (5) 的体渲染深度期望在概念上同型、在权重来源上不同（解析 splat vs 逐点体密度）。

### 4.3 建图损失与自适应扩展（式 (6)–(9)）

关键帧上的损失是两项 L1（式 (6)）：

$$\mathcal{L}_c = \sum_{m=1}^{HW}\big|C_m - \hat{C}_m\big|,\qquad \mathcal{L}_d = \sum_{m=1}^{HW}\big|D_m - \hat{D}_m\big| \tag{6}$$

（$C_m, D_m$ 为传感器观测。**深度先验如何进入高斯优化**：RGB-D 的 $D_m$ 作为逐像素伪真值直接进 $\mathcal{L}_d$，把"渲染深度贴住传感器深度"写进全部可见高斯的梯度——这是 RGB-D 版紧耦合的全部机制；跟踪阶段只用光度项，见 §4.5。）**初始化**（式 (7)）：首帧均匀采一半像素反投影，$M = HW/2$ 个高斯，位置 = 反投影点、颜色 = 像素 RGB 的零阶 SH、不透明度/协方差按预设与点密度（留一半空间给后续致密化，§3.2 原文）：

$$\{G_i = (\mathbf{P}_i, \Sigma_{\mathrm{init}}, \Lambda_{\mathrm{init}}, C_i)\,|\, i = 1,...,M\}, \qquad M = HW/2 \tag{7}$$

（原文此处用 $\mathbf{P}_i$ 表示初始位置，与位姿 $\mathbf{P}$ 记号冲突——按上下文区分。）**Adding 步判据**（式 (8)）：每个关键帧先渲染 RGB-D 并计算逐像素**累积不透明度**

$$T = \sum_{i\in N}\alpha_i\prod_{j=1}^{i-1}(1-\alpha_j)$$

（"所有已终止光线的质量总和"，即式 (4) 权重的无条件和：某像素 $T$ 低 ⟺ 现有高斯几乎没在该像素留下能量。）不可靠像素定义为

$$T < \tau_T \quad\text{or}\quad |D - \hat{D}| > \tau_D \tag{8}$$

——即"**渲染能量盖不住**"（新观测区域）或"**渲染深度与传感器深度对不上**"（遮挡边界/几何错误）两类；把这些像素反投影成新高斯（初始化同式 (7)）。**Delete 步**（式 (9)）：对当前视锥内每个可见高斯，从相机中心 $\mathbf{o}$ 过其位置 $\mathbf{X}_i$ 作射线 $\mathbf{r}(t) = \mathbf{o} + t(\mathbf{X}_i - \mathbf{o})$，找它与像平面的交点像素 $(u,v)$ 及该像素的观测深度 $D$、交点世界坐标 $\mathbf{P}_{uv}$；若

$$D - \mathrm{dist}(\mathbf{X}_i, \mathbf{P}_{uv}) > \gamma \quad\Longrightarrow\quad G_i: \Lambda_i \Rightarrow \eta\Lambda_i\ (\eta \ll 1) \tag{9}$$

则该高斯中心在观测表面**后方**超过 $\gamma$（悬空浮点），把其不透明度压到近零——不物理删除而是"退化"，让后续优化自然淘汰（依据：式 (4)(5) 中 $\alpha_i \to 0$ 使其权重趋零）。**与任务书说法的核对结论**：本文新增判据是式 (8) 的**不透明度/深度残差判据**，不是"有梯度但无高斯覆盖"的梯度判据（梯度阈值致密化是 3DGS 原版 [13] 的 clone/split 规则，论文仅在式 (7) 上下文提"adaptive density control that splits large points"承接它）。

### 4.4 位姿的解析梯度（式 (10)(11)；补充材料 (14)–(19)）

**跟踪目标**（式 (10)）：恒速假设给初值（用前两帧相对变换外推，§3.3），再以渲染-观测颜色 L1 求位姿：

$$\mathcal{L}_{\mathrm{track}} = \sum_{m=1}^{M}\big|C_m - \hat{C}_m\big|_1,\qquad \min_{R,\mathbf{t}} \mathcal{L}_{\mathrm{track}} \tag{10}$$

**链式分解**（式 (11)，逐步重写）。依式 (3)(4)，$\hat{C}$ 依赖 $\mathbf{P}$ 的通道只有三个中间变量：投影协方差 $\Sigma'$、（视角相关）颜色 $\mathbf{c}_i$、投影中心 $\mathbf{m}_i$。对任一高斯 $G_i$ 逐层应用链式法则：

$$\frac{\partial \mathcal{L}_c}{\partial \mathbf{P}} = \frac{\partial \mathcal{L}_c}{\partial \hat{C}}\cdot\frac{\partial \hat{C}}{\partial \mathbf{P}} = \frac{\partial \mathcal{L}_c}{\partial \hat{C}}\Big(\frac{\partial \hat{C}}{\partial \mathbf{c}_i}\frac{\partial \mathbf{c}_i}{\partial \mathbf{P}} + \frac{\partial \hat{C}}{\partial \alpha_i}\frac{\partial \alpha_i}{\partial \mathbf{P}}\Big) = \frac{\partial \mathcal{L}_c}{\partial \hat{C}}\frac{\partial \hat{C}}{\partial \alpha_i}\Big(\underbrace{\frac{\partial \alpha_i}{\partial \Sigma'}\frac{\partial \Sigma'}{\partial \mathbf{P}}}_{\text{经 (3)}} + \underbrace{\frac{\partial \alpha_i}{\partial \mathbf{m}_i}\frac{\partial \mathbf{m}_i}{\partial \mathbf{P}}}_{\text{经投影}}\Big) \tag{11}$$

（第一步依据：$\mathcal{L}_c$ 只通过 $\hat{C}$ 依赖 $\mathbf{P}$；第二步依据：$\hat{C}$ 对 $\mathbf{P}$ 的依赖经 $\mathbf{c}_i$ 与 $\alpha_i$ 两通道（$\mathbf{m}_i$、$\Sigma'$ 都藏在 $\alpha_i$ 里——2D 高斯求值随中心与协方差变化）；第三步依据：$\alpha_i$ 依赖 $\mathbf{P}$ 的两条路径 = 式 (3) 的 $\Sigma'$ 与投影 $\mathbf{m}_i$。）**三个工程化决策**（式 (11) 后原文逐条）： $\partial \hat{C}/\partial\mathbf{c}_i \cdot \partial\mathbf{c}_i/\partial\mathbf{P}$ **可消去**——跟踪实现只用视角无关颜色（1 度 SH 近似常值）； 论文发现 $\partial(\mathbf{K}\mathbf{P}\mathbf{X}_i)/\partial \mathbf{P}_{d_i}$（$\mathbf{K}\mathbf{P}\mathbf{X}_i$ 为齐次投影坐标，$d_i$ 为 $\mathbf{m}_i$ 的 $z$ 坐标）是位姿梯度的**确定性主导项**（$\partial \mathbf{m}_i/\partial(\mathbf{K}\mathbf{P}\mathbf{X}_i)$ 只是透视除法）； $\partial\Sigma'/\partial\mathbf{P}$（穿过式 (3) 的回传）**为效率计被忽略**——保留的正是与经典重投影雅可比同型的部分。**主导项的显式形式**（补充材料式 (15)，本文据 [第 04 章 (4.3)](../04_相机模型与特征提取.md) 重推）：记 $\mathbf{X}_c^i = \mathbf{P}\mathbf{X}_i = (x^c, y^c, z^c)^\top$，透视除法给出 $\mathbf{m}_i = (f_x x^c/z^c + c_x,\ f_y y^c/z^c + c_y)$，对平移（$\partial \mathbf{X}_c^i/\partial\mathbf{t} = I$）：

$$\frac{\partial \mathbf{m}_i}{\partial \mathbf{t}} = \begin{bmatrix} f_x/z^c & 0 & -f_x x^c/(z^c)^2 \\ 0 & f_y/z^c & -f_y y^c/(z^c)^2 \end{bmatrix}$$

（依据：$\partial(u)/\partial x^c = f_x/z^c$、$\partial(u)/\partial z^c = -f_x x^c/(z^c)^2$ 等逐项求导；这正是 [第 05 章 (5.11)](../05_视觉里程计-i特征点法.md) 重投影雅可比的左 $2\times3$ 块、也是 [3D 重建教程第 08 章 (8.6)](../../3d_reconstruction/08_3D高斯泼溅.md) 的 $\mathbf{J}$。）对旋转（补充材料式 (14) 给出四元数→旋转矩阵的显式表 $R(q)$），$\partial\mathbf{X}_c^i/\partial q_\nu = (\partial R/\partial q_\nu)\mathbf{X}_i$（依据：$\mathbf{X}_c = R(q)\mathbf{X}_i + \mathbf{t}$ 对四元数分量求导；$R(q)$ 的元素是 $q$ 的二次函数，故 (15) 的四个导数矩阵线性于 $q$）。**深度反传**（补充材料式 (19)，逐步）：对 $\hat{D} = \sum_{k=1}^n d_k\alpha_k\prod_{j<k}(1-\alpha_j)$，含 $\alpha_i$ 的项有两类——第 $i$ 项本身（贡献 $d_i\prod_{j<i}(1-\alpha_j)$）与所有 $k>i$ 的项（各含因子 $(1-\alpha_i)$，求导贡献 $-d_k\alpha_k\prod_{j<k,\,j\ne i}(1-\alpha_j)$）：

$$\frac{\partial \hat{D}}{\partial \alpha_i} = d_i\prod_{j=1}^{i-1}(1-\alpha_j) - \sum_{k=i+1}^{n} d_k\alpha_k\prod_{\substack{j=1\\ j\ne i}}^{k-1}(1-\alpha_j),\qquad \frac{\partial \hat{D}}{\partial d_i} = \alpha_i\prod_{j=1}^{i-1}(1-\alpha_j)$$

（后者依据：$\hat{D}$ 对 $d_i$ 线性。$n$ 为影响该像素的高斯数。）**与直接法的对照**：[第 06 章 (6.4)–(6.5)](../06_视觉里程计-ii直接法.md) 的光度残差雅可比 $\partial\mathbf{e}_p/\partial\delta\boldsymbol{\xi} = I_{2,x}\mathbf{J}_u + I_{2,y}\mathbf{J}_v$ 依赖图像梯度 $I_{2,x}$；本文把"图像梯度"替换为**穿过渲染图的解析梯度**（高斯混合权重可微、无插值），且深度通道在 RGB-D 建图中直接对齐传感器（式 (6)），鲁棒性来源从"像素梯度"换成"显式几何"。

### 4.5 Coarse-to-fine 跟踪与 BA（式 (12)(13)）

**动机**（§3.3 原话）：全像素优化会让伪影污染跟踪（"artifacts in images can cause drifted camera tracking"）。**Coarse 阶段**：在 $H/2\times W/2$ 均匀采样坐标上渲染 $\hat{I}_c$、按式 (10) 优化 $T_c$ 轮得 $\mathbf{P}_c$（利用图像正则性先吃掉低频对齐）。**Fine 阶段的可靠高斯筛选**（式 (12)）：在 $\mathbf{P}_c$ 下检查可见高斯，投影深度 $d_i$ 与该像素观测深度 $D_i$ 一致者才参与全分辨率渲染：

$$G_{\mathrm{selected}} = \{G_i\,|\,G_i\in\mathcal{G}\ \text{and}\ \mathrm{abs}(D_i - d_i) \le \epsilon\},\qquad \hat{I}_f = \mathcal{F}(u, v, G_{\mathrm{selected}}) \tag{12}$$

（$\mathcal{F}$ 为颜色泼溅渲染函数；$\hat{I}_c, \hat{I}_f$ 只渲染已建图区域——无高斯像素不回传梯度，见补充材料 Fig. 8。）再用 $\hat{I}_f$ 优化 $T_f$ 轮。**关键帧**：按"当前帧可靠区域占比"插入，且与最近关键帧差异超过阈值 $\mu_k$ 时插入（§3.3 末段原文）。**BA**（式 (13)）：随机选 $K$ 个关键帧，前半程只调地图、后半程地图与位姿同调：

$$\mathcal{L}_{\mathrm{ba}} = \frac{1}{K}\sum_{k=1}^{K}\sum_{m=1}^{HW}\Big(\big|D_m - \hat{D}_m\big|_1 + \lambda_m\big|C_m - \hat{C}_m\big|_1\Big),\qquad \min_{R,\mathbf{t},S}\mathcal{L}_{\mathrm{ba}} \tag{13}$$

（与[第 08 章](../08_后端-ii图优化与-ba.md)传统 BA 的本质差异：路标不再是 3D 点而是高斯参数，"重投影误差"换成渲染误差；BA 侧深度+光度全用。）

**跟踪超参**（补充材料 §5，已 Read 核实）：FusedAdam，平移学习率 $2\times10^{-4}$、四元数学习率 $5\times10^{-4}$；光度损失权重 0.8；每帧前 5 次迭代做 coarse 位姿估计、其余迭代用可靠高斯做 fine（Algorithm 1 的 $T_c/T_f$ 之实践版）；残差超过中位数 10 倍的像素被剔除（在线鲁棒核）。

### 4.6 为何跟踪不用深度损失——与 SplaTAM 的关键分歧（分析）

式 (10) 的跟踪损失**只有颜色项**，深度在跟踪端仅以间接方式出现（式 (12) 的筛选与恒速初值之外无深度项）。论文未直接解释，可从机制推敲（分析，非原文）： 若把 $\hat{D}$（式 (5)，高斯中心深度的加权平均）对齐传感器深度，梯度会**推动高斯中心沿 $z$ 平移**去"追"深度图——在建图线程里这正是想要的（式 (6)），在跟踪线程里却与"位姿不变、地图为准"的次序冲突（跟踪时地图刚按上一帧建好，可能尚未收敛）； 复杂纹理处的颜色残差在 $x$–$y$ 平面的约束远比深度残差丰富（深度沿墙面无约束），光度项天然补齐横向自由度——这与 [SplaTAM 精读](./SplaTAM_CVPR2024.md) Table 4 的消融（仅深度跟踪完全失败、仅 RGB 可跟踪）互为印证，只是两家在"谁主导"上做了相反的选择：SplaTAM 以深度为主、颜色减半加权并配 silhouette 门控；GS-SLAM 跟踪纯光度、把深度留给建图与筛选。

## 5. 实验与结果解读

- **Replica 跟踪（Table 1）**：平均 ATE **0.50 cm**（逐场景 0.48/0.53/0.33/0.52/0.41/0.59/0.46/0.70），Point-SLAM 0.54、ESLAM 0.63、CoSLAM 1.00、NICE-SLAM 1.06、Vox-Fusion 3.09；系统速度 **8.34 FPS**，比 0.42 FPS 的 Point-SLAM 快约 20 倍——论文的卖点是"tracking accuracy 与 runtime 的更好折中"（正文称领先次优 0.4 cm，按 Table 1 实为 0.04 cm 量级，两处不一致，以表为准作定性表述：略优但数量级优势在速度）。注：摘要/贡献处写 8.43 FPS、正文与 Table 4 写 8.34 FPS，论文本身两处不一致。
- **TUM-RGBD（Table 2）**：平均 ATE **3.7 cm**（fr1/desk 3.3、fr2/xyz 1.3、fr3/off 6.6），超过 iMAP（6.1）、Vox-Fusion（10.3）、NICE-SLAM（13.3），与 Point-SLAM/ESLAM/CoSLAM（2.0–2.4）同档；**仍明显落后传统法**（ORB-SLAM2 1.0）——论文自述"神经 vSLAM 与传统 SLAM 之间仍有差距，后者跟踪方案更成熟"（§4.2 原话）。
- **Replica 建图（Table 3）**：Depth L1 平均 **1.16 cm**、Precision **74.0%** 均为全场最佳；F1 70.15%（ESLAM 78.29% 更高，论文称 Recall/F1 与次优 CoSLAM 相当）。
- **渲染（Table 6）**：平均 **PSNR 34.27 dB / SSIM 0.975 / LPIPS 0.082**，全指标第一；比次优（CoSLAM 30.24 dB）高 1.52 dB、SSIM +0.027、LPIPS −0.12；渲染 **386.91 FPS**，约为次快 Vox-Fusion（3.88 FPS）的 100 倍。定性看渲染图（Fig. 5）：NICE-SLAM 出现严重伪影与模糊，CoSLAM/ESLAM 图像边界发糊，GS-SLAM 在物体边界与细节结构处明显更真实（§4.3 原话口径）。
- **运行时/内存（Table 4）**：跟踪 11.9 ms×10 迭代、建图 12.8 ms×100 迭代、**8.34 FPS**；**解码器参数 0 M**（无神经网络的纯显式表示），场景表示 198.04 MB（约为 NICE-SLAM 的 4 倍，主因是球谐系数；另提供零阶 SH 的 light 版：fr1/desk 上 40.8→18.8 MB、ATE 3.3→4.3，Table 5）。
- **消融（Table 7/8，Replica #Room0）**：**去掉 Adding** 完全崩溃（"✗"）——3DGS 原版致密化无法在无精确点云输入的实时建图下工作（§4.5 原话），证明"扩展策略是 3DGS-SLAM 的必需品"；**去掉 Delete**：ATE 0.58、Recall 49.32、F1 51.35，全量版 0.48 / 61.29 / 62.89——加删策略合计改善 0.1 ATE、11.97 Recall。**Coarse-to-fine（Table 8）**：仅 coarse ATE 0.91、仅 fine 0.49、完整 coarse-to-fine 0.48 且 PSNR 31.56 最高（精细阶段若不用可靠高斯筛选，伪影拉低建图与渲染）。

| 消融设置（Replica #Room0） | ATE ↓ | Depth L1 ↓ | Recall ↑ | PSNR ↑ |
|---|---|---|---|---|
| w/o add（仅首帧初始化） | 失败 | 失败 | 失败 | 失败 |
| w/o delete | 0.58 | 1.68 | 49.32 | 31.22 |
| **w/ add & delete（全量）** | **0.48** | **1.31** | **61.29** | **31.56** |
| 仅 coarse 跟踪 | 0.91 | 1.48 | 57.54 | 29.13 |
| 仅 fine 跟踪 | 0.49 | 1.39 | 59.18 | 30.84 |
| **coarse-to-fine（全量）** | **0.48** | **1.31** | **61.29** | **31.56** |

（数字按 Table 7/8 转录；"失败"为论文表中的 ✗——不加点云即无法实时建图。）

## 6. 局限与后续影响

**局限**（§5 论文自述 + 补充材料）： **依赖高质量深度**——RGB-D 输入的质量直接限定跟踪与建图上限（§5："its reliance on high-quality depth data may limit its performance"）； **显存**——198 MB 场景表示、4 倍于 NICE-SLAM，球谐系数是大头，大规模场景需要量化/聚类等压缩（§5 展望）； **网格提取不理想**——补充材料 Fig. 10：直接从高斯中心/高斯 marching cubes 提取网格效果不满意，实用方案仍是把深度图喂 TSDF-Fusion； **单目版本缺位**——本文为 RGB-D 系统，纯单目要等后续工作。另按 Table 4 的口径提醒：其 FPS 换来的是每帧 10 次跟踪迭代 × 100 次建图迭代的 GPU 重负载，"实时"以桌面级 RTX 4090 为前提（§4.1 实现细节）。**后续影响**：与 MonoGS（首个 3DGS 单目 SLAM，3 fps）、[SplaTAM](./SplaTAM_CVPR2024.md) 同期共同点燃 3DGS-SLAM 子领域；其"扩展/删除面向 SLAM 的致密化"与"位姿解析梯度"被后续系统普遍继承；渲染速度数量级优势把"SLAM 地图可直接服务渲染任务"变成现实卖点。

## 7. 与本项目对照

- **为什么 3DGS-SLAM 未入 projects/slam**：本仓库 SLAM 模块全部 CPU + numpy、无 torch/CUDA 依赖；而 GS-SLAM 的每一次位姿/地图更新都要跑 CUDA 光栅化内核的前向+反向（论文明言"We extended the existing code for differentiable Gaussian splatting rasterization with additional functionality"，§4.1 实现细节），教学复现的代价与收益不成比例。概念级对应物： [photoba/](../../../projects/slam/photoba/)（光度直接法 BA——式 (10) 的"渲染-观测残差"在无渲染器时的最近亲，雅可比结构对照[第 06 章 (6.4)–(6.5)](../06_视觉里程计-ii直接法.md)）； [loam2d/](../../../projects/slam/loam2d/)（几何残差配准，对照式 (12) 的深度一致性筛选）。
- **教程锚点**：[第 10 章 §10.4 表](../10_建图与系统实战.md)（本文被概括为"3D 高斯做显式辐射场地图，随跟踪结果自适应扩展高斯，RGB-D / 单目"）｜[3D 重建教程第 08 章](../../3d_reconstruction/08_3D高斯泼溅.md)（(8.2) 协方差参数化、(8.8) 投影、(8.10) alpha 混合、致密化 clone/split——本文式 (2)(3)(4) 与其逐式对应）｜[第 04 章 (4.3)–(4.5)](../04_相机模型与特征提取.md)（投影几何）｜[第 05 章 (5.11)](../05_视觉里程计-i特征点法.md)（位姿雅可比的解析原型）。
- **与姊妹精读的选型差异**（详见 [SplaTAM 精读](./SplaTAM_CVPR2024.md) §7 对照表）：本文是**建图驱动**（扩展策略 + 可靠高斯筛选服务于"把地图做全做对"，跟踪用光度残差）；SplaTAM 是**跟踪驱动**（silhouette 门控的稠密深度+光度直接配准）；[MonoGS](../../../papers/slam/frontier/arXiv-2312.06741_MonoGS-Gaussian-Splatting-SLAM.pdf) 则首次把 3DGS 拉进**单目**设定（3 fps，其摘要自称 "the first application of 3D Gaussian Splatting in monocular SLAM"）。三者对"first 3DGS SLAM"的措辞各按其论文：本文称首个 3DGS 稠密 **RGB-D** SLAM（§1 贡献 1），SplaTAM 称首次以显式体积表示做 RGB-D SLAM（其摘要）。

| 维度 | GS-SLAM（本文） | SplaTAM | MonoGS |
|---|---|---|---|
| 输入 | RGB-D | RGB-D | 单目（可扩展 RGB-D） |
| 高斯 | 各向异性、1 度 SH | 各向同性（球对称）、无 SH | 各向异性 + SH |
| 跟踪损失 | 纯光度（式 (10)）+ coarse-to-fine | 深度 + 0.5×光度，silhouette 门控 | 光度 + 深度（渲染与观测） |
| 地图扩展 | 累积不透明度/深度残差判据增删（式 (8)(9)） | silhouette<0.5 等距致密化（其式 (9)） | 致密化 + 漂移控制 |
| 速度（论文口径） | 8.34 FPS，渲染 386 FPS | Replica 约 3 fps 量级（其 Table 6：每帧 1.44 s 建图） | 3 fps |

（表中后两列数字以各自论文为准，此处只给量级；逐项对照见 [SplaTAM 精读](./SplaTAM_CVPR2024.md)。）
- **指标口径**：本文 ATE 为 cm 级 RMSE（TUM/Replica 惯例）；本仓库统一口径见 [METRICS.md §2](../../../projects/slam/METRICS.md)（(M.2)，Umeyama 对齐后 RMSE，Sturm 2012 §IV-A 定义）——注意论文类表格未必说明是否做了尺度/相似对齐，横向比较时先核口径。

## 配套阅读

- 本文 PDF：[GS-SLAM（Yan et al., CVPR 2024 Highlight）](../../../papers/slam/frontier/arXiv-2311.11700_GS-SLAM.pdf)（本精读式号 (1)–(13) 按正文、(14)–(19) 按补充材料）
- 3DGS 本体：[3D 重建教程第 08 章](../../3d_reconstruction/08_3D高斯泼溅.md)（(8.1)–(8.11)：高斯定义、协方差分解、EWA 投影链、alpha 混合、致密化——本文的表示与渲染全部继承于此）｜原论文 PDF [papers/3d_reconstruction/frontier/arXiv-2308.04079_3D-Gaussian-Splatting.pdf](../../../papers/3d_reconstruction/frontier/arXiv-2308.04079_3D-Gaussian-Splatting.pdf)
- 姊妹精读：[SplaTAM](./SplaTAM_CVPR2024.md)（同为 3DGS-SLAM，跟踪驱动的另一极）｜[NeRF-SLAM](./NeRF-SLAM_ICRA2023.md)（前一代隐式场方案）｜[PTAM](./PTAM_ISMAR2007.md)（跟踪/建图双线程架构的源头）
- 教程：[第 10 章](../10_建图与系统实战.md)（§10.4 前沿表）｜[第 06 章](../06_视觉里程计-ii直接法.md)（直接法光度残差——本文跟踪损失的传统前身）｜[第 08 章](../08_后端-ii图优化与-ba.md)（式 (13) BA 的经典原型）｜[第 04 章](../04_相机模型与特征提取.md)
- 代码：[projects/slam/photoba/](../../../projects/slam/photoba/)｜[projects/slam/loam2d/](../../../projects/slam/loam2d/)｜[METRICS.md](../../../projects/slam/METRICS.md)
- 主题导航：[tutorials/slam/README.md](../README.md) ｜ [OVERVIEW.md](../OVERVIEW.md)

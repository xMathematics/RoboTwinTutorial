# 论文精读｜NeRF-SLAM：DROID-SLAM 喂养分层体素 NeRF 的实时稠密单目 SLAM（ICRA 2023）

> **PDF**：[papers/slam/frontier/arXiv-2210.13641_NeRF-SLAM.pdf](../../../papers/slam/frontier/arXiv-2210.13641_NeRF-SLAM.pdf)（arXiv v1，2022-10-24，10 页，本精读所有式号以此 PDF 为准）｜ **教程**：[第 10 章 §10.4](../10_建图与系统实战.md)（稠密建图与前沿的深读扩展）｜ **代码**：无（前沿路线说明；体渲染概念基础见 [projects/nerf/](../../../projects/nerf/)）

## 1. 论文信息与一句话贡献

- **题目**：NeRF-SLAM: Real-Time Dense Monocular SLAM with Neural Radiance Fields
- **作者**：Antoni Rosinol、John J. Leonard、Luca Carlone（Massachusetts Institute of Technology，MIT SPARK 实验室）
- **发表**：ICRA 2023（IEEE Int. Conf. on Robotics and Automation）；arXiv:2210.13641
- **一句话贡献**：论证"**稠密单目 SLAM 恰好提供了实时训练神经辐射场所需的全部信息**"——DROID-SLAM 输出的位姿 $\mathbf{T}$、稠密深度 $\mathbf{D}$ 及其**边缘协方差** $\Sigma_{\mathbf{D}}$，以不确定性加权深度损失注入 Instant-NGP 风格的分层体素辐射场，单张 RTX 2080 Ti 上跟踪 15 fps、建图 10 fps、整体 12 fps（640×480），在 Replica 上以**纯单目输入**取得 Depth L1 4.49 cm / PSNR 41.40 dB，大幅超过 iMAP 与 Nice-SLAM（后两者用真值深度监督也只有 6.95 dB / 34.61 dB）。

## 2. 问题与动机

**要解决什么**（§1）：从随手拍的单目视频实时构建"光度上准确"（photometrically accurate）的 3D 地图。三条困难叠加： 单目无显式深度测量——RGB-D 怕日照与直射光、Lidar 昂贵且重，单目便宜、轻、易标定（§1 第一段）； NeRF 表示能力强但**训练慢**：体渲染要对每条光线采样成百上千个点、逐点查询网络（§1："the costly volumetric rendering necessary to build a NeRF leading to slow reconstructions"），且原始 NeRF 假设位姿已知（SfM/COLMAP 预计算，在线场景不成立）； 无深度监督的辐射场易产生 "floaters"（局部密度团幽灵几何，源于坏初始化、收敛差与局部极小；§1 末段引 Mono-SDF 等结论：加深度监督显著加速收敛、提升几何）。

**核心洞察**（§1 "our insight" 原话）："having a dense monocular SLAM pipeline, that outputs close-to-perfect pose estimates, together with dense depth maps and uncertainty estimates, provides the right information for building neural radiance fields of the scene **on the fly**"。作者前作 σ-Fusion（Rosinol et al., WACV 2022，本文引文 [23]）已证明：**深度的边缘协方差**是经典 TSDF 体积融合的优秀加权信号，能把"纹理弱区域给的坏深度"压下去。本文的地图表示升级（§2.1 末段原话）："we replace the volumetric TSDF for a hierarchical volumetric neural radiance field as our map representation"——用辐射场换掉 TSDF，光度保真与新视角合成随之而来，还允许"位姿与地图同时优化"。

**为什么是这两个模块**（§2.2–2.3 的排除法）： 传统 NeRF/Barf/iMAP 一系用大 MLP，在线推理太慢； Instant-NGP 用哈希分层体素网格把训练压到实时（§2.2 末段），可作建图引擎； 跟踪端可选 Orbeez-SLAM（ORB-SLAM 提供初始位姿，间接式）或 VolBA（直接 RGB 损失），本文选 DROID-SLAM——间接式光流损失的位姿估计更稳（"known to be more robust than direct image alignment"），且深度由光流反推、天然带协方差（§2.3 末段）。

## 3. 方法总览

```
RGB 图像流 I_1, I_2, …（640×480，单目，无深度传感器）
   │
   ▼  【跟踪线程 §3.1：DROID-SLAM（引文 [31]），预训练权重】
   ├─ RAFT 式架构逐对帧算稠密光流 p_ij 与逐测量权重 Σ_p_ij（ConvGRU 核心）
   ├─ 稠密 BA：变量 = 全部关键帧位姿 T ∈ SE(3) + 逐关键帧逐像素逆深度 d
   │    线性化 → 块稀疏 Hessian（式 (1)）→ Schur 补先解位姿 → 回代深度
   │    → 副产品：位姿/深度的边缘协方差 Σ_T、Σ_D（式 (2)，计算法出自 σ-Fusion [23]）
   ├─ 关键帧策略：当前帧与前一关键帧平均光流 > 2.5 px 即新建关键帧（§3.3）
   ▼
位姿 T、深度 D、边缘协方差 Σ_D、RGB 图 ──（滑动窗口 8 关键帧）──▶
   │
   ▼  【建图线程 §3.2：Instant-NGP 式分层体素 NeRF（引文 [17]，改造）】
   ├─ 地图 Θ = 多分辨率哈希体素网格特征 + 小 MLP（"hierarchical volumetric" 所指）
   ├─ 损失 L_M(T,Θ) = L_rgb + λ_D·L_D（式 (3)）；深度残差按 Σ_D 马氏加权（式 (4)）
   ├─ 深度渲染 = 期望光线终止距离（式 (5)(6)）；颜色渲染同原 NeRF（式 (7)）
   └─ 同时优化辐射场参数 Θ 与位姿 T（λ_D = 1.0，§3.2 末）
   ▼
输出：光度地图（任意新视角渲染）+ 几何地图（按不确定性 σ_σ ≤ 1.0 阈值化的点云/网格）
```

**两线程的通信协议**（§3.3）：跟踪线程滑动窗口 **8** 关键帧持续做 BA 重投影误差最小化；建图线程**只增不减**地接收全部关键帧（无滑动窗口，地图信息单调累积）。唯一同步时刻是新关键帧到达，传递量为"位姿 + 图像 + 深度 + 各自边缘协方差"——恰是式 (3) 求梯度所需的全部信息。两线程共享一张 GPU（该设计也允许各占一张，§3.4）。建图线程还负责交互式可视化（§3.3 末句）。

**通信量的量级直觉**（依据：DROID 参数化 + 本文 §3.3）：跟踪线程每秒约 10 个关键帧，每帧传出 6 维位姿 + $H\times W$ 逆深度 + $H\times W$ 方差 + 一幅 RGB 图；与之对照，若走"渲染像素梯度回传跟踪端"的紧耦合接口，则需 $H\times W\times 3$ 量级的双向握手。本文接口是**单向、只增、批处理友好**的——这是它能在共享 GPU 上不打断对方实时运行的结构原因。位姿在两边都出现：跟踪端是主输出，建图端是可被 $\mathcal{L}_M$ 继续精化的变量（§3.2 末段），"信息单向流 + 变量双向优化"由此分工。

## 4. 关键公式推导（式号按论文 PDF 原文 (1)–(7)）

### 4.0 符号表

| 符号 | 含义 |
|---|---|
| $\mathbf{T}$ | 全部相机位姿（$SE(3)$；式 (3) 中亦为建图阶段的优化变量） |
| $\Delta\boldsymbol{\xi}$ | 位姿的李代数增量（$\mathfrak{se}(3)$ 上，式 (1) 前文："delta updates on the lie algebra of the camera poses in SE(3)"） |
| $\mathbf{d}$ / $\Delta\mathbf{d}$ | 逐关键帧**逐像素逆深度**堆叠及其增量（DROID-SLAM 参数化） |
| $\mathbf{H}\in\mathbb{R}^{(c+p)\times(c+p)}$ | BA 的 Hessian；$c$ = 相机维数、$p$ = 点（逆深度）维数 |
| $\mathbf{C},\ \mathbf{E},\ \mathbf{P}$ | 位姿-位姿块（block camera matrix）/ 位姿-深度 off-diagonal 块 / 逆深度对角块 |
| $\mathbf{v},\ \mathbf{w}$ | 位姿 / 深度的残差项（$\mathbf{b} = [\mathbf{v};\mathbf{w}]$） |
| $\mathbf{H}_T = \mathbf{L}\mathbf{L}^\top$ | 约简相机矩阵（Schur 补）及其 Cholesky 下三角因子 |
| $\Sigma_{\mathbf{T}},\ \Sigma_{\mathbf{D}}$ | 位姿 / 稠密深度的边缘协方差（式 (2)） |
| $\mathbf{D},\ \mathbf{D}^*$ | DROID 估计深度图 / 辐射场渲染深度图（式 (4)） |
| $\Theta$ | 辐射场参数（哈希网格特征 + MLP） |
| $\sigma_i,\ \delta_i,\ d_i$ | 采样点 $i$ 的体密度 / 相邻采样间距（$\delta_i = d_{i+1}-d_i$）/ 采样点沿光线的深度（式 (5) 原文 $d$ 双义） |
| $\mathcal{T}_i$ | 累积透射率（式 (6)；NeRF 教程 §3.5 的 $T_i$） |
| $w_i$ | "光线终止在采样点 $i$"的概率质量（本文分析引入；$w_i = \mathcal{T}_i(1-\exp(-\sigma_i\delta_i))$） |
| $\lambda_D$ | 深度损失权重（实验取 1.0，§3.2） |

（记号注：论文以 $\mathbf{d}$ 同时表示逐像素逆深度变量与式 (5) 的采样点深度；本文推导处均按上下文区分，必要处以 $d_i^{(\mathrm{inv})}$ 与 $d_i^{(\mathrm{ray})}$ 心算替换。$\mathbf{G}_i$ 在 DROID 语境是 $SE(3)$ 位姿，与后两篇精读的"3D 高斯 $G_i$"无关。）

### 4.1 跟踪前端 = 稠密 BA 的块结构（式 (1)）

DROID-SLAM 以**逐关键帧逐像素逆深度**参数化结构、以稠密光流为测量。记帧对 $(i,j)$、宿主像素 $\mathbf{u}$ 的残差（"观测 − 预测"约定）：

$$\mathbf{r}_{ij}(\mathbf{u}) = \mathbf{p}^*_{ij}(\mathbf{u}) - \Pi_c\big(\mathbf{G}_j \circ \Pi_c^{-1}(\mathbf{p}_i, d_i)\big)(\mathbf{u}) \in \mathbb{R}^2$$

（符号沿用[精读/DROID-SLAM](./DROID-SLAM_NeurIPS2021.md) §4.4：$\mathbf{p}^*_{ij}$ 为 GRU 修正后的对应场，$\Pi_c/\Pi_c^{-1}$ 为针孔投影/反投影，[第 04 章 (4.3)–(4.5)](../04_相机模型与特征提取.md) 的同款几何。）在当前估计处一阶展开、按测量协方差 $\Sigma_{p_{ij}}$ 加权、堆叠全部帧对与像素，得加权最小二乘正规方程（依据：G-N 线性化，[第 08 章 (8.5)](../08_后端-ii图优化与-ba.md)；完整推导见精读/DROID-SLAM §4.4），按 $[\Delta\boldsymbol{\xi}\,|\,\Delta\mathbf{d}]$ 分块即论文式 (1)：

$$\mathbf{H}\mathbf{x} = \mathbf{b},\quad \text{i.e.}\quad \begin{bmatrix} \mathbf{C} & \mathbf{E} \\ \mathbf{E}^\top & \mathbf{P} \end{bmatrix}\begin{bmatrix} \Delta\boldsymbol{\xi} \\ \Delta\mathbf{d} \end{bmatrix} = \begin{bmatrix} \mathbf{v} \\ \mathbf{w} \end{bmatrix} \tag{1}$$

**块的稀疏性依据**（逐残差数变量）：每条光流残差 $\mathbf{r}_{ij}(\mathbf{u})$ 依赖两个位姿 $\mathbf{G}_i,\mathbf{G}_j$ 与**恰好一个**宿主像素的逆深度 $d_i$ ⟹ 深度-深度块 $\mathbf{P}$ **对角**（"the diagonal matrix corresponding to the inverse depths per pixel per keyframe"）；任意两位姿可经共享帧对的残差耦合 ⟹ $\mathbf{C}$ 稠密（"block camera matrix"）；$\mathbf{E}$ 是连接两者的 off-diagonal 块。求解顺序（§3.1 原话 + DROID 原文）：Schur 补消去海量逆深度 → 位姿步 $\Delta\boldsymbol{\xi} = \Sigma_{\mathbf{T}}(\mathbf{v} - \mathbf{E}\mathbf{P}^{-1}\mathbf{w})$ → 回代 $\Delta\mathbf{d} = \mathbf{P}^{-1}(\mathbf{w} - \mathbf{E}^\top\Delta\boldsymbol{\xi})$——代数即[第 08 章 (8.14)–(8.15)](../08_后端-ii图优化与-ba.md)，与[精读/DROID-SLAM](./DROID-SLAM_NeurIPS2021.md) §4.4 的式 (5) 完全同构。

**因子图视角的澄清**（任务书常见误解的核对结论）：本文**没有** IMU 预积分因子、里程计因子或独立成项的"平滑深度因子"——跟踪前端是**纯视觉**的 DROID-SLAM，单目输入连 IMU 都没有。若借用因子图语言（解释性框架，非论文原话）：节点 = 位姿与逐像素逆深度；边 = 逐对帧**稠密光流测量因子**，残差即上面的 $\mathbf{r}_{ij}(\mathbf{u})$，权重 $\Sigma_{p_{ij}}^{-1}$ 由网络逐像素预测；此外并无其他因子类型。IMU 预积分因子（残差 = 预积分量与状态推算量之差）属于 VINS-Mono/Kimera-VIO 一系（见[精读/IMU-Preintegration](./IMU-Preintegration_TRO2017.md)、[第 10 章 §10.2–10.3](../10_建图与系统实战.md)）；"平滑深度"的角色在本文中由建图端的**不确定性加权深度损失**（式 (4)，§4.3）承担——它是辐射场优化里的概率先验项，不是 VIO 图中的因子。

### 4.2 边缘协方差：深度不确定性的来源（式 (2)）

本文区别于一切 NeRF-SLAM 前作的钥匙，是把 BA 的 Hessian **再利用**为不确定性来源。对高斯牛顿意义下的线性系统（残差已按测量协方差白化），参数协方差 = Hessian 之逆（依据：加权最小二乘 $\mathrm{Cov} = (\mathbf{J}^\top\Sigma^{-1}\mathbf{J})^{-1}$ 的经典结论；论文将其归功于 [23] 的实时计算法："we can efficiently calculate the marginal covariances … as shown in [23]"）。对式 (1) **逐块求逆**（无跳步）：

**第 1 步**（下行解出深度增量）：由第二行方程 $\mathbf{E}^\top\Delta\boldsymbol{\xi} + \mathbf{P}\,\Delta\mathbf{d} = \mathbf{w}$，两侧左乘 $\mathbf{P}^{-1}$（依据：$\mathbf{P}$ 对角且可逆——每个逆深度变量只出现在自己射线上的残差里）：

$$\Delta\mathbf{d} = \mathbf{P}^{-1}(\mathbf{w} - \mathbf{E}^\top\Delta\boldsymbol{\xi})$$

**第 2 步**（代入上行）：$\mathbf{C}\Delta\boldsymbol{\xi} + \mathbf{E}\,\mathbf{P}^{-1}(\mathbf{w} - \mathbf{E}^\top\Delta\boldsymbol{\xi}) = \mathbf{v}$，按 $\Delta\boldsymbol{\xi}$ 归并（依据：分配律）：

$$(\mathbf{C} - \mathbf{E}\mathbf{P}^{-1}\mathbf{E}^\top)\,\Delta\boldsymbol{\xi} = \mathbf{v} - \mathbf{E}\mathbf{P}^{-1}\mathbf{w}$$

记约简矩阵 $\mathbf{H}_T \triangleq \mathbf{C} - \mathbf{E}\mathbf{P}^{-1}\mathbf{E}^\top$（论文："the reduced camera matrix $\mathbf{H}_T$, which does not depend on the depths"——消元后与深度无关），则 $\Delta\boldsymbol{\xi} = \mathbf{H}_T^{-1}(\mathbf{v} - \mathbf{E}\mathbf{P}^{-1}\mathbf{w})$。定义**位姿边缘协方差**

$$\Sigma_{\mathbf{T}} \triangleq \mathbf{H}_T^{-1} = (\mathbf{L}\mathbf{L}^\top)^{-1}$$

（式 (2) 第二行；论文用 Cholesky $\mathbf{H}_T = \mathbf{L}\mathbf{L}^\top$ 表出——依据：$(\mathbf{L}\mathbf{L}^\top)^{-1} = \mathbf{L}^{-\top}\mathbf{L}^{-1}$，两次三角回代即可，避免显式矩阵求逆。）

**第 3 步**（代回深度行、读出协方差块）：$\Delta\mathbf{d} = \mathbf{P}^{-1}\mathbf{w} - \mathbf{P}^{-1}\mathbf{E}^\top\Sigma_{\mathbf{T}}(\mathbf{v} - \mathbf{E}\mathbf{P}^{-1}\mathbf{w})$，对 $\mathbf{v},\mathbf{w}$ 分别归并系数（依据：分配律 + $\Sigma_{\mathbf{T}}\mathbf{E}\mathbf{P}^{-1}$ 的结合）：

$$\Delta\mathbf{d} = \underbrace{\big(-\mathbf{P}^{-1}\mathbf{E}^\top\Sigma_{\mathbf{T}}\big)}_{\mathbf{v}\ \text{的系数块}}\mathbf{v} \;+\; \underbrace{\big(\mathbf{P}^{-1} + \mathbf{P}^{-1}\mathbf{E}^\top\Sigma_{\mathbf{T}}\mathbf{E}\mathbf{P}^{-1}\big)}_{\mathbf{w}\ \text{的系数块}}\mathbf{w}$$

对照 $\begin{bmatrix}\Delta\boldsymbol{\xi}\\ \Delta\mathbf{d}\end{bmatrix} = \mathbf{H}^{-1}\begin{bmatrix}\mathbf{v}\\ \mathbf{w}\end{bmatrix}$ 可读出 $\mathbf{H}^{-1}$ 的对角块：位姿-位姿块 $=\Sigma_{\mathbf{T}}$；深度-深度块 $=\mathbf{P}^{-1} + \mathbf{P}^{-1}\mathbf{E}^\top\Sigma_{\mathbf{T}}\mathbf{E}\mathbf{P}^{-1}$。把 $\mathbf{P}^{-1}$ 写作 $\mathbf{P}^{-\top}$（$\mathbf{P}$ 对角，二者相等）即论文式 (2)：

$$\Sigma_{\mathbf{d}} = \mathbf{P}^{-1} + \mathbf{P}^{-\top}\mathbf{E}^\top\,\Sigma_{\mathbf{T}}\,\mathbf{E}\,\mathbf{P}^{-1},\qquad \Sigma_{\mathbf{T}} = (\mathbf{L}\mathbf{L}^\top)^{-1} \tag{2}$$

**读法**：单个像素的深度不确定性 = 自身测量不确定（$\mathbf{P}^{-1}$ 对角项）+ **经位姿耦合传导**的不确定（$\mathbf{E}^\top\Sigma_{\mathbf{T}}\mathbf{E}$ 项）——位姿越不确定（$\Sigma_{\mathbf{T}}$ 大）或该像素对位姿越敏感（$\mathbf{E}$ 大，即该像素的光流强烈约束位姿时其深度也随位姿抖动），深度越不可信。直觉自检：无纹理像素的光流本就不可靠，其逆深度的极小值浅而平 ⟹ Hessian 对角元小 ⟹ $\mathbf{P}^{-1}$ 大 ⟹ 式 (4) 自动降权。**方差公式的核对结论**：本 arXiv v1 全文无"渲染深度的方差近似式"；式 (4) 所用的 $\Sigma_{\mathbf{D}}$ 全部来自式 (2)——不确定性来自**跟踪几何（BA 边缘协方差）**而非**体渲染积分**。

### 4.3 建图损失与其 MAP 读法（式 (3)(4)(7)）

给定跟踪线程的全部输出，建图端最小化（式 (3)）：

$$\mathcal{L}_M(\mathbf{T},\Theta) = \mathcal{L}_{\mathrm{rgb}}(\mathbf{T},\Theta) + \lambda_D\,\mathcal{L}_D(\mathbf{T},\Theta) \tag{3}$$

**颜色损失**（式 (7)）与原 NeRF 相同：$\mathcal{L}_{\mathrm{rgb}}(\mathbf{T},\Theta) = \|I - I^*(\mathbf{T},\Theta)\|^2$，$I^*$ 为体渲染颜色图（每像素沿光线采样、alpha 合成密度与颜色，§3.2 原文）。**深度损失**（式 (4)）是本文对 Instant-NGP 的唯一结构性改造：

$$\mathcal{L}_D(\mathbf{T},\Theta) = \|D - D^*(\mathbf{T},\Theta)\|^2_{\Sigma_D} \tag{4}$$

$\|\cdot\|_{\Sigma_D}$ 为马氏范数（依据：$\|\mathbf{e}\|^2_{\Sigma} = \mathbf{e}^\top\Sigma^{-1}\mathbf{e}$ 的定义，与[第 08 章 (8.2)](../08_后端-ii图优化与-ba.md) 的加权最小二乘同构）：**每个像素的深度残差先按自己的标准差白化再平方**——不确定深度自动降权。$\lambda_D = 1.0$（§3.2 末："we set $\lambda_D$ to 1.0"）。

**MAP 读法**（解释性推导；论文未明写后验，此为把 (3)–(4) 读回估计理论的桥。依据：独立高斯噪声下 MAP 负对数 = 马氏范数之和；单步后验更新 $\mathrm{bel}(x_t) = \eta\,p(z_t|x_t)\,\mathrm{bel}^-(x_t)$ 见[第 03 章 (3.6)](../03_概率状态估计基础.md)）。设深度残差服从 $\mathcal{N}(0,\Sigma_D)$（跟踪线程已给出协方差，式 (2)），则

$$p(\Theta,\mathbf{T} \mid I, \mathbf{D}) \;\propto\; \underbrace{\exp(-\mathcal{L}_{\mathrm{rgb}})}_{p(I \mid \mathbf{T},\Theta)\ \text{光度似然}} \;\cdot\; \underbrace{\exp(-\mathcal{L}_D)}_{p(\mathbf{D}\mid\mathbf{T},\Theta)\ \text{深度"观测"似然}}$$

即**后验按信息来源分解**：位姿与深度的先验信息由跟踪线程注入（其置信度就是式 (2) 的协方差），辐射场参数 $\Theta$ 只需在光度一致性下拟合场景——这正是"给定跟踪轨迹、逐关键帧优化场"的精确含义。论文的实际机制（§3.2–3.3）：$\mathcal{L}_M$ 对 $\mathbf{T}$ 与 $\Theta$ **同时**求最小（"we can optimize our radiance field's parameters and refine the camera poses simultaneously"）；跟踪线程输出只进不出，建图线程独立收敛。

### 4.4 体渲染深度期望（式 (5)(6)）

渲染深度被定义为**期望光线终止距离**（"expected ray termination distance"，§3.2）。推导（无跳步）：沿光线在深度 $d_i$ 处采样，MLP 给出体密度 $\sigma_i$（3D 坐标 + 哈希特征输入）；单段"碰撞概率"由 Beer–Lambert 定律的一阶求积给出 $\alpha_i = 1-\exp(-\sigma_i\delta_i)$（依据：区间上"至少碰到一个粒子"= 1 − 零碰撞概率 $\exp(-\int\sigma\,\mathrm{d}t)$，离散化取 $\int_{d_i}^{d_{i+1}}\sigma\,\mathrm{d}t \approx \sigma_i\delta_i$；[NeRF 教程第 03 章 §3.5](../../nerf/03_数学原理与渲染方程.md)、NeRF 论文 Eq.3 同款）。累积透射率即式 (6)：

$$\mathcal{T}_i = \exp\Big(-\sum_{j<i}\sigma_j\delta_j\Big) \overset{(a)}{=} \prod_{j<i}\exp(-\sigma_j\delta_j) \overset{(b)}{=} \prod_{j<i}\big(1-\alpha_j\big)$$

（依据： "指数→连乘"； 由 $\alpha_i$ 定义移项。）"光线恰在 $i$ 处终止"的概率质量 = 能到达（$\mathcal{T}_i$）× 在此处被挡（$\alpha_i$）：

$$w_i \triangleq \mathcal{T}_i\big(1-\exp(-\sigma_i\delta_i)\big) = \mathcal{T}_i\,\alpha_i$$

终止位置的期望 = 概率质量 × 位置的加权和，即论文式 (5)：

$$d^* = \sum_i \mathcal{T}_i\big(1-\exp(-\sigma_i\delta_i)\big)\,d_i \;=\; \sum_i w_i\,d_i \;\equiv\; E[t] \tag{5}$$

归一性：$\sum_i w_i = 1 - \mathcal{T}_{n+1} \le 1$（依据：$\prod_j(1-\alpha_j)$ 的部分乘积展开，终止概率质量之和 = 1 − 逃逸质量），故 $d^*$ 是良性凸组合、恒落在采样范围内；逃逸质量对应背景，式 (5) 未加背景深度项（对深度监督无影响——$D$ 总在场景内）。**符号警示**：论文在式 (5) 用 $d_i$ 同时表示"采样点深度"与式 (4) 的"观测深度 $D$"，推导时须区分。

### 4.5 深度监督为何修复 floaters（机制分析）

论文引 Mono-SDF 等工作断言"depth supervision 显著改善辐射场几何"，本文的贡献是指出：**在 SLAM 语境里，深度监督之所以"用得起"，正因为有协方差可依**。机制上（把式 (4)(5) 联立展开）：

$$\frac{\partial \mathcal{L}_D}{\partial \sigma_i} = \frac{\partial}{\partial \sigma_i}\Big\|D(\mathbf{u}) - \underbrace{\textstyle\sum_k w_k(\sigma)\, d_k}_{D^*(\mathbf{u})}\Big\|^2_{\Sigma_D(\mathbf{u})},\qquad \frac{\partial w_k}{\partial \sigma_i} \ne 0\ \ (i \le k)$$

（依据：$w_k = \mathcal{T}_k\alpha_k$ 依赖全部 $\sigma_{j<k}$——式 (6) 的透射连乘；$\Sigma_D(\mathbf{u})$ 逐像素白化，链式法则。）floaters 是"局部密度团靠遮挡近处光线制造光度极小"的产物：没有深度项时 $\mathcal{L}_{\mathrm{rgb}}$ 对这类解不敏感；加上式 (4) 后，floater 处的渲染深度 $D^*$ 与跟踪深度 $D$（连其不确定度 $\Sigma_D$）直接冲突，而 floater 区域恰是光流不可靠、$\Sigma_D$ 大的区域——**加权**保证"压 floater"的同时不把低纹理区的坏深度当真。Fig. 5 的 4 dB/7 cm 差距就是该机制的定量体现。

### 4.6 与 iMAP 的真实差异、关键帧策略（原文核对）

论文对照 iMAP（引文 [26]）的三点实况（§2.2–2.3、§3.2）： **地图表示**：iMAP 用单个大 MLP 表征全场景，在线推理太慢（§2.2："these approaches are too slow for online inference due to their choice of a large MLP as map representation"）；本文用 Instant-NGP 的**哈希式分层体素网格**（"hash-based hierarchical volumetric representation"）——**"hierarchical"指多分辨率体素网格，不是 NeRF 的 coarse/fine 分层采样**；iMAP 的"两倍分辨率"技巧与可见性关键帧采样**不出现在本文任何章节**。 **深度来源**：iMAP/Nice-SLAM 用 RGB-D 实测深度，本文用**光流反推的单目深度**并按式 (2) 加权（§2.3 末段："our depth loss is weighted by the depth's marginal covariance …, and the depth is estimated from optical-flow rather than measured by an RGB-D camera"）。 **优化耦合**：位姿与地图联合优化（对比 Orbeez-SLAM 等先定后图，§2.3）。**关键帧策略**（§3.3 实况）：当前帧与前一关键帧的**平均光流 > 2.5 px** 即新增关键帧（约 10 关键帧/秒）。本文的关键消融是**深度监督的加权方式**（§5 的 Fig. 5–6），不含"渲染像素数"对比。

## 5. 实验与结果解读

- **数据集（§4.1）**：Replica（5 个 office + 3 个 room 场景，iMAP 管线渲染 2000 帧/场景，深度来自真值网格）+ Cube-Diorama（合成、有真值位姿与深度，用于消融）。
- **Replica 主表（Table 1）**：本文（**纯单目、自有深度**）平均 **Depth L1 4.49 cm / PSNR 41.40 dB**，8 场景全表最优或次优。对照组读法：iMAP 用**真值深度**也只到 7.64 cm / 6.95 dB；Nice-SLAM 真值深度版 4.08 cm / 34.61 dB、**无深度版**崩到 14.18 cm / 17.76 dB（"Nice-SLAM's results deteriorate when not using ground-truth depth"）；经典 TSDF-Fusion/σ-Fusion 用本文跟踪模块的位姿与深度融合，约 21–22 cm / 7 dB——几何对但光度差（网格化表示无新视角合成）。正文定性结论（§4.3）：相对 Nice-SLAM，office-1 上 PSNR 最高提升 179%、room-2 上 L1 最高提升 86%（摘要口径 "up to 179% better PSNR and 86% better L1"）。

| 方法（Replica 平均，Table 1 转录） | 深度来源 | Depth L1 ↓ | PSNR ↑ |
|---|---|---|---|
| iMAP | **真值**深度 | 7.64 cm | 6.95 dB |
| Nice-SLAM | **真值**深度 | 4.08 cm | 34.61 dB |
| TSDF-Fusion / σ-Fusion | 本文跟踪模块深度 | 21.88 / 20.10 cm | 7.07 / 7.08 dB |
| Nice-SLAM（无深度行） | 无 | 14.18 cm | 17.76 dB |
| **Ours（本文）** | **DROID 光流深度 + $\Sigma_D$ 加权** | **4.49 cm** | **41.40 dB** |

（TSDF/σ-Fusion 行吃的就是本文跟踪模块的位姿与深度，故其"几何差"主要来自 TSDF 网格化与融合本身而非跟踪质量；Nice-SLAM 无深度行说明隐式场纯靠光度难以自持。）
- **深度损失消融（Fig. 5，Cube-Diorama）**：噪声位姿 + 噪声深度**不加权**直接监督，120 s 后比加权版 PSNR 差约 4 dB、L1 差约 7 cm；按边缘协方差加权后对这些缺陷**免疫**（"our approach is resilient to these depths"），取得最好结果。**信息来源组合（Fig. 4）**：理想情形是真值位姿 + 真值深度（最快最准）；真值位姿 + 无深度 → 收敛但更慢；噪声位姿 + 无深度 → 60 s 内不收敛；噪声位姿 + 噪声深度**加权** → 接近理想情形——即本文主张的"低质量位姿/深度 + 不确定性"组合的效果上限。
- **有无深度消融（Fig. 6）**：无深度 → 光度收敛良好（位姿足够准）但几何不准（500 s 后 L1 ≈ 7.8 cm）；原始不加权深度 → L1 减半（≈4.1 cm）但 PSNR 差约 3 dB（深度噪声伤外观：纹理弱区被强行拉成错误颜色）；加权深度 → PSNR 最佳且 L1 与"用深度"持平——**加权同时拿到了两边的好处**，这是本文方法主张的直接证据。
- **逐场景读表（Table 1）**：本文在 office 系列优势最大（office-1：PSNR 53.44 dB，对比 Nice-SLAM 真值深度版的 25.22 dB——纹理复杂的桌面上物体重建受益于"光度地图"）；room-2 是所有方法共同的困难场景（本文 L1 9.13 cm，Nice-SLAM 真值深度版 8.41 cm），运动模糊样的视角与玻璃面造成深度歧义。iMAP 真值深度版在 office-2 达 14.23 cm——单 MLP 的 catastrophic forgetting 在大场景外推时暴露。定性表述以表为准，避免对单场景数字过度解读。
- **实时性（§4.5）**：整体 12 fps（640×480）；跟踪 15 fps（≈10 关键帧/s）；建图 10 fps；单卡 RTX 2080 Ti、约 11 GB 显存；自定义 CUDA 核 + 跟踪/建图并行。跟踪线程的关键帧速率随运动量浮动（"depending on the amount of motion"），2.5 px 阈值把关键帧密度与场景视差挂钩。

## 6. 局限与后续影响

**局限**（§5 论文自述）：约 11 GB GPU 显存——帧对稠密相关体 + 分层体素网格双重开销，对"低算力机器人"（robots with low compute power）是硬门槛；论文给的两条缓解路：相关体**现算现用**（引文 [10]，免存储整段视频的稠密相关体）、体素体积**流式驻留 CPU**、只把感兴趣区域的滑窗调入 GPU（引文 [15]）。结构性局限：无独立回环/位姿图模块（回环收益由 DROID 帧图的长程边承担）；深度不确定性是高斯近似，单峰协方差表达不了多模态深度歧义。

**后续影响**：确立了"几何 SLAM 喂可微神经地图 + 协方差加权监督"的组合范式；同年起 3DGS（显式原语）迅速取代神经隐式场成为稠密建图的主流表示——[GS-SLAM](./GS-SLAM_CVPR2024.md)、[SplaTAM](./SplaTAM_CVPR2024.md)（见[第 10 章 §10.4 表](../10_建图与系统实战.md)）都继承了"跟踪输出直接监督可微地图"的骨架，把"地图表示"从辐射场换成高斯集合。"边缘协方差当损失权重"的思想可上溯至 σ-Fusion 与经典概率体积融合。

**横向定位**（同期单目→辐射场 SLAM 三条路线，按 §2.3 的自述）：iMAP/Nice-SLAM = "RGB-D 深度 + 隐式场"，本文 = "单目光流深度（带协方差）+ 隐式场"，VolBA/Orbeez-SLAM = "隐式场但位姿来源不同"（VolBA 直接 RGB 损失、Orbeez 依赖 ORB-SLAM 初始位姿）。本文的独特组合点在**不确定性**：唯一把边缘协方差从跟踪端一路传到地图端损失的方案。这个位置也预示了它的局限——协方差的质量完全继承自 DROID，若光流网络在域外场景失效，深度与协方差一起失效，建图端没有任何兜底机制（§5 局限的隐性来源）。

## 7. 与本项目对照

- **为什么 projects/slam 没有 NeRF-SLAM 模块**：其两半都绑定 GPU——跟踪需要 ConvGRU + CUDA 稠密 BA，建图需要哈希网格神经渲染；本仓库 SLAM 模块坚持 CPU + numpy 可跑、可断言、可复现。教学替代：跟踪侧有 [projects/slam/droidlite/](../../../projects/slam/droidlite/)（DROID 稠密 BA 的结构演示，无学习组件：合成 GT 光流 + 32×32 宿主网格 + 6 轮递归 + Schur 补交叉验证，见[精读/DROID-SLAM](./DROID-SLAM_NeurIPS2021.md) §7）。
- **体渲染概念基础在 projects/nerf/**（已 Read 核实）：[nerf/render.py](../../../projects/nerf/nerf/render.py) 的 `volume_render`（NeRF 论文 Eq.3 的离散体渲染——本文式 (5)(6) 的同款 alpha 合成，深度期望即其输出通道之一）与 `render_rays`（对一批光线做 coarse + fine 两趟渲染）；[nerf/sampling.py](../../../projects/nerf/nerf/sampling.py) 的 `stratified_sample` / `hierarchical_sample` / `ray_deltas`。**概念辨析**：`hierarchical_sample` 是 **NeRF 的层次采样**（coarse 提议分布 → fine 按权重采样，[NeRF 教程第 05 章](../../nerf/05_层次采样与训练细节.md)），与本文"hierarchical volumetric"（多分辨率哈希体素网格）是两个不同概念。
- **教程锚点**：[第 10 章 §10.2–10.3](../10_建图与系统实战.md)（若走 VIO 路线，因子图应含 IMU 预积分因子 + 重投影因子——本文无 IMU，二者对照读）｜[第 08 章 (8.5)/(8.14)/(8.15)](../08_后端-ii图优化与-ba.md)（式 (1) 的代数出处）｜[第 03 章 (3.6)](../03_概率状态估计基础.md)（§4.3 MAP 读法依据）｜[METRICS.md §2](../../../projects/slam/METRICS.md)（(M.2) ATE RMSE、Sturm 2012 口径——注意本文实验**未报轨迹误差**，只报深度/光度指标）。
- **与姊妹精读的衔接**：本文证明"隐式场 + 协方差加权深度"可行但表示昂贵；[GS-SLAM](./GS-SLAM_CVPR2024.md) / [SplaTAM](./SplaTAM_CVPR2024.md) 换成显式高斯后，同样吃 RGB-D/单目先验、渲染快两个数量级——三条精读连读即是 2022–2024 稠密 SLAM 地图表示的演化史。

## 配套阅读

- 本文 PDF：[NeRF-SLAM（Rosinol, Leonard, Carlone, ICRA 2023）](../../../papers/slam/frontier/arXiv-2210.13641_NeRF-SLAM.pdf)（本精读所有式号以此 arXiv v1 为准）
- 跟踪前端：[精读/DROID-SLAM](./DROID-SLAM_NeurIPS2021.md)（式 (1) 的完整推导与 Schur 补；本文是其"下游客户"）
- 体渲染源头：[NeRF 教程第 03 章](../../nerf/03_数学原理与渲染方程.md)（§3.2 积分式、§3.5 离散 alpha 合成）｜[NeRF 教程第 05 章](../../nerf/05_层次采样与训练细节.md)（真·层次采样，与本文 "hierarchical" 辨析）
- 地图表示对照：[3D 重建教程第 07 章](../../3d_reconstruction/07_神经隐式表面重建.md)（NeuS/体渲染权重为何撑不起干净几何——本文用深度监督缓解的正是这类问题）｜[第 08 章](../../3d_reconstruction/08_3D高斯泼溅.md)（显式原语的下一站）
- 姊妹精读：[GS-SLAM](./GS-SLAM_CVPR2024.md)｜[SplaTAM](./SplaTAM_CVPR2024.md)（3DGS 取代神经隐式场的后续路线）｜[IMU-Preintegration](./IMU-Preintegration_TRO2017.md)（因子图背景对照：本文没有的 IMU 因子长什么样）｜[VINS-Mono](./VINS-Mono_RAM2018.md)
- 教程：[第 10 章](../10_建图与系统实战.md)（§10.4 前沿表把本文列为"几何 SLAM + 神经隐式建图"代表）｜[第 08 章](../08_后端-ii图优化与-ba.md)｜[第 03 章](../03_概率状态估计基础.md)
- 代码：[projects/nerf/](../../../projects/nerf/)（`volume_render`/`render_rays`/`stratified_sample`/`hierarchical_sample`）｜[projects/slam/droidlite/](../../../projects/slam/droidlite/)｜[METRICS.md](../../../projects/slam/METRICS.md)
- 主题导航：[tutorials/slam/README.md](../README.md) ｜ [OVERVIEW.md](../OVERVIEW.md)

# 论文精读｜Gaussian Splatting SLAM（CVPR 2024）

> **PDF**：[papers/slam/frontier/arXiv-2312.06741_MonoGS-Gaussian-Splatting-SLAM.pdf](../../../papers/slam/frontier/arXiv-2312.06741_MonoGS-Gaussian-Splatting-SLAM.pdf)（arXiv v2，2024-04-14，21 页含补充材料，本精读所有式号以此 PDF 为准）｜ **教程**：[第 10 章](../10_建图与系统实战.md)（§10.4 前沿导读的深读扩展）、[3D 重建第 08 章](../../3d_reconstruction/08_3D高斯泼溅.md)（3DGS 基础）｜ **代码**：无（前沿路线说明；官方实现见论文网站 murai.co.uk/projects/GaussianSplattingSLAM）

## 1. 论文信息与一句话贡献

- **题目**：Gaussian Splatting SLAM（社区惯称 MonoGS）
- **作者**：Hidenobu Matsuki\*、Riku Murai\*、Paul H. J. Kelly、Andrew J. Davison（帝国理工学院 Dyson Robotics Laboratory 与 Software Performance Optimisation Group；\* 共同一作）
- **发表**：CVPR 2024；arXiv:2312.06741
- **一句话贡献**：把 **3D 高斯泼溅（3DGS）** 用作 SLAM 的**唯一**三维表示，建成第一个**单目输入**下近实时运行的 3DGS SLAM——跟踪 = 对 3DGS 渲染图像的**光度残差**做 SE(3) 直接优化（论文首次给出 3DGS 相对相机位姿的 **Lie 群解析雅可比**，主文式 (3)–(6)）；建图 = 关键帧窗口内对高斯参数的联合优化（式 (11)），并引入**各向同性正则**（式 (10)）抑制沿视线方向的拉伸伪影。系统单目实测 3 fps 在线运行，RGB-D 下可扩展，定位与新视角合成均达当时最优水平。

## 2. 问题与动机

**要解决什么**（§1）：离线 3DGS（Kerbl et al., SIGGRAPH 2023，见 [3D 重建第 08 章](../../3d_reconstruction/08_3D高斯泼溅.md)）需要 SfM 给出的精确位姿与稀疏点云，逐场景训练数万步——SLAM 需要的是**在线**、**增量**、**每帧都要与地图交互**的地图表示。论文问的是：3DGS 能否同时承载 SLAM 的全部职能（跟踪、建图、关键帧管理、新视角合成）？

**选 3DGS 的表示层论证**（§1–2）：与四类已有稠密表示对照—— 体素网格/SDF：分辨率与内存绑定，大场景修正需形变机制； 网格：离散拓扑，融合与优化困难； 点云/surfel：最接近，但"几何以离散、不连续的方式表达"； 神经隐式场：逐像素光线步进查询太慢。3DGS 兼得：局部性（高斯可自由增删、形变）、连续光滑的体积表达、以及 CUDA 光栅化带来的可微高速渲染（1080p 下至 200 fps）——**跟踪每帧要跑约 50 次梯度迭代**（§3.1），表示的渲染与求导速度因此直接决定 SLAM 可行性。

**Map-centric 的立场**（§2 相关工作）：稠密 SLAM 二分为 frame-centric（帧间光度/深度误差最小化，各帧各自持有局部几何，如 DSO——见[精读/DSO](./DSO_PAMI2018.md)）与 map-centric（跟踪直接对着一个全局重建地图）。作者选 map-centric：一张统一 3D 地图让跟踪、建图、关键帧管理共用同一表示。**单目场景**的额外挑战：无深度观测、有尺度模糊——这成为后文各向同性正则与在线尺度估计的动机。

## 3. 方法总览

论文 Fig. 2 的系统流水线（文字版）：

```
输入视频（RGB 或 RGB-D）
   │
   ▼【跟踪 Tracking（§3.3.1，式 (7)(8)）】
   只优化当前帧位姿 T_CW：对当前 3DGS 地图渲染，最小化光度残差
   E_pho（单目）或 λ_pho·E_pho + (1−λ_pho)·E_geo（RGB-D，补充材料式 (12)）；
   100 次迭代、位姿更新量 < 1e−4 提前终止；同步优化仿射亮度参数、
   惩罚低不透明度/非边缘像素
   │
   ▼【关键帧管理 Keyframing（§3.3.2，补充材料式 (14)(15)）】
   以高斯共视性（covisibility）判定：IOUcov / OCcov 低于阈值或
   相对平移过大 → 注册关键帧；窗口满后按 OCcov 淘汰旧关键帧
   │
   ▼【建图 Mapping（§3.3.3，式 (10)(11)）】
   窗口 W = W_k ∪ W_r：联合优化窗口内所有关键帧位姿 + 全部高斯参数；
   每次迭代额外随机抽 2 个历史关键帧防遗忘；
   新高斯插入 + 可见性剪枝（单目下用渲染深度反推插入位置）
```

三个组件共享**同一个** 3DGS 地图；渲染既产生定位用的残差，也产生新视角合成输出。地图是纯高斯集合 $\mathcal{G}$：每个高斯带颜色 $c^i$、不透明度 $\alpha^i$、均值 $\boldsymbol\mu_W^i$ 与协方差 $\boldsymbol\Sigma_W^i$（**各向同性**；SH 球谐视角相关颜色默认关闭，消融见补充材料 §9.3）。

## 4. 关键公式推导（论文式号；补充材料以"补充材料式"标注）

### 符号表

| 符号 | 含义（出处） |
|---|---|
| $\mathcal{G},\ \mathcal{G}^i$ | 高斯集合 / 第 $i$ 个高斯（§3.1） |
| $c^i,\ \alpha^i$ | 第 $i$ 个高斯的颜色 / 不透明度（§3.1） |
| $\boldsymbol\mu_W^i,\ \boldsymbol\Sigma_W^i$ | 高斯 $i$ 在世界系的均值与协方差（§3.1；各向同性） |
| $\boldsymbol\mu_C,\ \boldsymbol\Sigma_C$ | 均值/协方差变换到相机系（式 (2) 推导链） |
| $\boldsymbol\mu_I,\ \boldsymbol\Sigma_I$ | 投影到图像平面的 2D 均值/协方差（式 (2)） |
| $\mathbf{T}_{CW}\in SE(3)$ | 相机位姿（世界→相机；沿用全书 $T_{cw}$ 约定） |
| $\mathbf{W},\ \mathbf{J}$ | $\mathbf{T}_{CW}$ 的旋转块 / 投影的线性化雅可比（式 (2)） |
| $I(\mathcal{G},\mathbf{T}_{CW}),\ D(\mathcal{G},\mathbf{T}_{CW})$ | 从位姿 $\mathbf{T}_{CW}$ 渲染的颜色图 / 深度图（式 (7)(8)） |
| $\mathcal{W}_k,\ \mathcal{W}_r,\ \mathcal{W}$ | 关键帧窗口 / 随机历史关键帧 / 建图窗口（§3.3.3） |
| $\mathbf{s}_i,\ \bar{\mathbf{s}}_i$ | 高斯 $i$ 的三轴尺度向量 / 其均值（式 (10)） |
| $\lambda_{pho},\ \lambda_{iso}$ | 光度权重（0.9）与各向同性正则权重（10）（补充材料 §7.1.1） |

### 4.1 渲染模型（式 (1)(2)）：与 3DGS 教程对照

像素 $\mathbf{p}$ 的颜色由按深度排序的 alpha 混合合成（式 (1)）：

$$\mathcal{C}_p = \sum_{i\in\mathcal{N}} c_i\,\alpha_i \prod_{j=1}^{i-1}(1-\alpha_j) \tag{1}$$

与 [3D 重建第 08 章 (8.10)](../../3d_reconstruction/08_3D高斯泼溅.md) 的 $C=\sum_k \mathbf{c}_k\alpha_k\prod_{j<k}(1-\alpha_j)$ 逐项同构（依据：同一个"透射连乘 × 挡光 × 颜色"结构；$\alpha_i$ 由 2D 高斯在像素处求值，教程 (8.9)）。

3D 高斯投影到图像平面（式 (2)）：

$$\boldsymbol\mu_I = \pi(\mathbf{T}_{CW}\cdot\boldsymbol\mu_W), \qquad \boldsymbol\Sigma_I = \mathbf{J}\mathbf{W}\boldsymbol\Sigma_W\mathbf{W}^T\mathbf{J}^T \tag{2}$$

其中 $\pi$ 是透视投影，$\mathbf{J}$ 是投影在 $\boldsymbol\mu_C$ 处的线性化雅可比。推导链与教程 (8.4)–(8.8) 完全一致（依据：EWA splatting 两步近似）： 刚体变换 $\boldsymbol\mu_C = \mathbf{W}\boldsymbol\mu_W + \mathbf{t}$、$\boldsymbol\Sigma_C = \mathbf{W}\boldsymbol\Sigma_W\mathbf{W}^T$（精确变换，协方差传播引理 = 教程 (8.7)）； 视锥局部把 $\pi$ 一阶泰勒展开（仿射近似），协方差再传播一次 $\boldsymbol\Sigma_I = \mathbf{J}\boldsymbol\Sigma_C\mathbf{J}^T$（教程 (8.5)(8.6)(8.8)）。MonoGS 的新问题是把**位姿**也变成优化变量——需要 $\boldsymbol\mu_I, \boldsymbol\Sigma_I$ 对 $\mathbf{T}_{CW}$ 的导数。

### 4.2 位姿的 Lie 群最小雅可比（式 (3)–(6)；论文核心贡献）

链式法则分解（式 (3)(4)）：

$$\frac{\partial\boldsymbol\mu_I}{\partial\mathbf{T}_{CW}} = \frac{\partial\boldsymbol\mu_I}{\partial\boldsymbol\mu_C}\,\frac{\mathcal{D}\boldsymbol\mu_C}{\mathcal{D}\mathbf{T}_{CW}}, \qquad \frac{\partial\boldsymbol\Sigma_I}{\partial\mathbf{T}_{CW}} = \frac{\partial\boldsymbol\Sigma_I}{\partial\boldsymbol\Sigma_C}\frac{\partial\boldsymbol\Sigma_C}{\partial\boldsymbol\mu_C}\frac{\mathcal{D}\boldsymbol\mu_C}{\mathcal{D}\mathbf{T}_{CW}} + \frac{\partial\boldsymbol\Sigma_I}{\partial\mathbf{W}}\frac{\mathcal{D}\mathbf{W}}{\mathcal{D}\mathbf{T}_{CW}} \tag{3,4}$$

（依据：链式法则。注意 $\boldsymbol\Sigma_C = \mathbf{J}\mathbf{W}\boldsymbol\Sigma_W\mathbf{W}^T\mathbf{J}^T$ 中 $\mathbf{J}$ 在 $\boldsymbol\mu_C$ 处取值，故 $\boldsymbol\Sigma_C$ 既经 $\mathbf{J}(\boldsymbol\mu_C)$ 依赖 $\boldsymbol\mu_C$、又经 $\mathbf{W}$ 依赖位姿——两个通路相加。）对流形上的量定义**最小偏导数**（式 (5)）：

$$\frac{\mathcal{D}f(\mathbf{T})}{\mathcal{D}\mathbf{T}} \triangleq \lim_{\tau\to 0} \frac{\mathrm{Log}\big(f(\mathrm{Exp}(\tau)\circ\mathbf{T})\circ f(\mathbf{T})^{-1}\big)}{\tau} \tag{5}$$

（依据：在 $\mathfrak{se}(3)$ 切空间取扰动 $\tau$、经指数映射回到群，再用 Log 拉回向量空间——保证雅可比维数 = 自由度 6，消除冗余参数化。$\circ$ 为群复合。）

**闭式解**（式 (6)；以下按补充材料式 (19)–(24) 逐步重推）。设 $\tau = (\rho, \theta)$（平移、旋转两部分），相机系均值是群作用 $\boldsymbol\mu_C = \mathbf{T}_{CW}\cdot\boldsymbol\mu_W = \mathbf{W}\boldsymbol\mu_W + \mathbf{t}$。左扰动后（依据：群复合的定义，$\mathrm{Exp}(\tau)\circ\mathbf{T}_{CW} = [\mathrm{Exp}(\boldsymbol\theta)\mathbf{W},\ \mathrm{Exp}(\boldsymbol\theta)\mathbf{t} + \boldsymbol\rho]$）：

$$\boldsymbol\mu_C(\tau) = \mathrm{Exp}(\boldsymbol\theta)\,\mathbf{W}\boldsymbol\mu_W + \mathrm{Exp}(\boldsymbol\theta)\,\mathbf{t} + \boldsymbol\rho$$

一阶展开 $\mathrm{Exp}(\boldsymbol\theta) \approx \mathbf{I} + \boldsymbol\theta^\wedge$（依据：指数映射一阶截断），合并两项得

$$\boldsymbol\mu_C(\tau) \approx \boldsymbol\mu_C + \boldsymbol\theta^\wedge\boldsymbol\mu_C + \boldsymbol\rho = \boldsymbol\mu_C + \boldsymbol\rho - \boldsymbol\mu_C^\times\,\boldsymbol\theta$$

最后一步依据：$\boldsymbol\theta^\wedge\boldsymbol\mu_C = \boldsymbol\theta\times\boldsymbol\mu_C = -\boldsymbol\mu_C\times\boldsymbol\theta = -\boldsymbol\mu_C^\times\boldsymbol\theta$（叉积的反对称性；$\times$ 记反对称矩阵）。故平移列导数为 $\mathbf{I}$、旋转列导数为 $-\boldsymbol\mu_C^\times$：

$$\frac{\mathcal{D}\boldsymbol\mu_C}{\mathcal{D}\mathbf{T}_{CW}} = \big[\mathbf{I},\ -\boldsymbol\mu_C^\times\big], \qquad \frac{\mathcal{D}\mathbf{W}}{\mathcal{D}\mathbf{T}_{CW}} = \begin{bmatrix} \mathbf{0} & -\mathbf{W}_{:,1}^\times \\ \mathbf{0} & -\mathbf{W}_{:,2}^\times \\ \mathbf{0} & -\mathbf{W}_{:,3}^\times \end{bmatrix} \tag{6}$$

$\mathbf{W}$ 块的推导（补充材料式 (25)–(34)）：$\mathbf{W}$ 就是 $\mathbf{T}_{CW}$ 的旋转块，平移扰动不影响它（前三列为 $\mathbf{0}$）；对旋转扰动 $\mathcal{D}\mathbf{W}/\mathcal{D}\mathbf{R}_{CW} = \lim_{\theta\to 0}(\mathrm{Exp}(\boldsymbol\theta)\circ\mathbf{W} - \mathbf{W})/\boldsymbol\theta = \lim \boldsymbol\theta^\times\mathbf{W}/\boldsymbol\theta$（依据：同上的一阶展开）。逐分量求（依据：$\partial\boldsymbol\theta^\times/\partial\theta_x = \mathbf{e}_1^\times$，补充材料式 (29)(30)）：$\partial\mathbf{W}/\partial\theta_x = \mathbf{e}_1^\times\mathbf{W} = [\mathbf{0}_{1\times 3}; -\mathbf{W}_{3,:}; \mathbf{W}_{2,:}]$（式 (31)，$\mathbf{W}_{i,:}$ 为第 $i$ 行；式 (32)(33) 对 $y,z$ 同理），按列向量化后水平堆叠得 $\mathcal{D}\mathbf{W}/\mathcal{D}\mathbf{R}_{CW} = [-\mathbf{W}_{:,1}^\times; -\mathbf{W}_{:,2}^\times; -\mathbf{W}_{:,3}^\times]$（式 (34)，$\mathbf{W}_{:,i}$ 为第 $i$ 列；$9\times 3$）。主文式 (6) 是该块的紧凑写法。

**两个记号要点**（避免读错原文）： 扰动顺序为"平移在前、旋转在后"——式 (6) 的列序是 $[\cdot]_{\text{平移}}, [\cdot]_{\text{旋转}}$，与部分文献（旋转在前）相反，比较符号时须先对齐约定； 旋转列出现 $-\boldsymbol\mu_C^\times$ 而非 $+\boldsymbol\mu_C^\times$，是因为扰动取**左乘**（$\mathrm{Exp}(\tau)\circ\mathbf{T}$），一阶项里旋转作用在**已经变换到相机系的点** $\boldsymbol\mu_C$ 上——若用右扰动 $\mathbf{T}\circ\mathrm{Exp}(\tau)$，雅可比将表达在世界系点上，两种写法经伴随变换等价（第 02 章伴随；补充材料 §10 明确采用左扰动定义 (5)）。

**工程意义**：3DGS 的 CUDA 光栅化只为高斯自身参数实现了显式导数，位姿导数原本要靠反向传播"穿过"整个渲染——本文给出解析式后，位姿优化不再依赖自动微分开销（§3.2：跟踪每帧约 50 次迭代，导数计算必须便宜）。这也是论文自列的第二条贡献："novel techniques within the SLAM framework: the analytic Jacobian on Lie group for direct camera pose estimation, isotropic regularisation of the Gaussian shape, and geometric verification"（§1）。

### 4.3 跟踪残差（式 (7)–(9)、补充材料式 (12)）

单目跟踪只优化 $\mathbf{T}_{CW}$，最小化渲染图与观测图的光度残差（式 (7)）：

$$E_{pho} = \big\| I(\mathcal{G}, \mathbf{T}_{CW}) - I \big\|_1 \tag{7}$$

RGB-D 下加几何残差（式 (8)）$E_{geo} = \| D(\mathcal{G}, \mathbf{T}_{CW}) - D \|_1$，联合目标（补充材料式 (12)）：

$$\min_{\mathbf{T}_{CW}\in SE(3)} \lambda_{pho}E_{pho} + (1-\lambda_{pho})E_{geo}, \qquad \lambda_{pho} = 0.9 \tag{12}$$

深度图由渲染管线给出（式 (9)）——与式 (1) 同一 alpha 混合、把颜色换成沿相机射线的高斯均值深度 $\bar{z}_i$：

$$\mathcal{D}_p = \sum_{i\in\mathcal{N}} \bar{z}_i\,\alpha_i \prod_{j=1}^{i-1}(1-\alpha_j) \tag{9}$$

（依据：式 (1) 的逐项替换；$\bar{z}_i$ 为高斯 $i$ 沿该像素射线的深度。）跟踪还同步优化仿射亮度参数（变曝光）并惩罚低不透明度/非边缘像素（§3.3.1）——与 DSO 的光度建模同源（见[精读/DSO](./DSO_PAMI2018.md) §4；本项目 `photoba/` 即该残差-雅可比结构的教学实现）。

### 4.4 各向同性正则与建图目标（式 (10)(11)、补充材料式 (13)）

光栅化对高斯**沿视线方向的厚度没有任何约束**（§3.3.3）——新视角合成可以容忍（视点受控），连续 SLAM 不行：光度信号弱时高斯沿视线被拉长成"伪影簇"，既污染渲染又干扰跟踪（Fig. 3 定性展示）。对策之一是把高斯限制为各向同性：实现上保留三轴尺度参数 $\mathbf{s}_i$，用正则把三轴拉向相等、鼓励球形（式 (10)）：

$$E_{iso} = \sum_{i=1}^{|\mathcal{G}|} \|\mathbf{s}_i - \bar{\mathbf{s}}_i\,\mathbf{1}\|_1 \tag{10}$$

（依据：对尺度偏离其均值的部分取 L1；$\bar{\mathbf{s}}_i$ 为三个尺度分量的均值。）建图在窗口 $\mathcal{W} = \mathcal{W}_k \cup \mathcal{W}_r$ 上联解位姿与高斯（式 (11)）：

$$\min_{\substack{\mathbf{T}_{CW}^{\mathcal{V}}\in SE(3)\\ \mathcal{G}}} \sum_{\mathcal{V}\in\mathcal{W}} E_{pho}^{\mathcal{V}} + \lambda_{iso}E_{iso}, \qquad \text{RGB-D 时每帧再加 } \lambda_{pho}E_{pho}^{\mathcal{V}} + (1-\lambda_{pho})E_{geo}^{\mathcal{V}} \text{ 项（补充材料式 (13)）} \tag{11}$$

（$\lambda_{iso}=10$；每个 $\mathcal{V}$ 的位姿也一并优化——建图不只调高斯，还精化窗口内关键帧位姿。）$\mathcal{W}_r$ 为每次迭代随机抽的 2 个历史关键帧，防止对旧区域"灾难性遗忘"。

### 4.5 关键帧管理判据（式 (14)(15)）：以高斯共视性为原子

关键帧选择的度量对象不是特征点数，而是**两个关键帧能看到的高斯集合的交叠**（$\mathcal{G}_i^v$ 为关键帧 $i$ 中可见高斯集；可见性判定复用光栅化的排序性质：某视角下高斯若尚未累计到 $\alpha \ge 0.5$ 即不可见——遮挡因此"by design"被处理，§3.3.2）：

$$IOU_{cov}(i,j) = \frac{|\mathcal{G}_i^v \cap \mathcal{G}_j^v|}{|\mathcal{G}_i^v \cup \mathcal{G}_j^v|}, \qquad OC_{cov}(i,j) = \frac{|\mathcal{G}_i^v \cap \mathcal{G}_j^v|}{\min(|\mathcal{G}_i^v|, |\mathcal{G}_j^v|)} \tag{14,15}$$

（依据：集合交并比 / 重叠系数的标准定义——OC 相对 IOU 对"一个视角看得少"更宽容。）带迟滞的判定逻辑（补充材料 §7.1.2）：**注册**——当前帧 $i$ 与上一关键帧 $j$ 的 $IOU_{cov}(i,j) < k_{cov}$，或相对平移 $t_{ij} > k_{fm}\hat{D}_i$（$\hat{D}_i$ 为帧 $i$ 的中位深度）→ 注册新关键帧（依据：看得比上帧少很多 = 出现新内容；位移相对深度过大 = 视差足够，两条件都是"加宽基线、增多约束"的代理判据）；**淘汰**——新关键帧 $i$ 注册后，检查窗口内已有帧 $j$ 与**最新**帧的 $OC_{cov}(i,j) < k_{fc}$ → 移除 $j$（依据：内容被新帧覆盖的旧帧留着只费算力）。阈值：Replica $k_{cov}=0.95, k_{fm}=0.04$；TUM $0.90/0.08$；$k_{fc}=0.3$；窗口大小 Replica $|\mathcal{W}_k|=10$、TUM $=8$。消融（Table 3）显示关闭该选择机制使 TUM RGB-D 平均 ATE 3.96 → 8.73 cm——**窗口里放什么帧比窗口多大更关键**。

### 4.6 单目的"深度从哪来"：渲染深度的自举

**新高斯插入**（§3.3.2、补充材料 §7.1.2）：RGB-D 直接反投影深度初始化 $\boldsymbol\mu_W$；单目**没有深度观测，就用当前地图自己渲染的深度**（式 (9)）反推：有渲染深度 $D_p$ 的像素，深度从 $\mathcal{N}(D_p,\ 0.2\sigma_D)$ 采样；无观测区域从 $\mathcal{N}(\hat{D},\ 0.5\sigma_D)$ 采样（$\hat{D}$ 为渲染深度的中位数）。**单目初始化**（补充材料 §8.3，式 (16)(17)）：系统启动时用初始若干帧训练初始地图 $\mathcal{G}_{init}$——单目最小化 $\sum_{\mathcal{V}\in\mathcal{W}} E_{pho}^{\mathcal{V}} + \lambda_{iso}E_{iso}$（式 (16)，无深度项），RGB-D 加几何项（式 (17)）。TUM 单目实验用真值位姿完成系统初始化、此后在线估计尺度（§4.1 Baseline Methods 段；与 DROID-SLAM 单目初始化口径相同的处理，见[精读/DROID-SLAM](./DROID-SLAM_NeurIPS2021.md)）。**勘误注（规格偏差）**：注意单目初始化**不是**"SfM 点云"——SfM 点云是离线 3DGS 的初始化方式，本文恰恰把对 SfM 的依赖列为要克服的第一点（§1）。**剪枝**：最近 3 个关键帧内插入、且未被至少 3 个其他帧观测到的高斯被判几何不稳定而删除（主文 §3.3.2，未给数值）；补充材料另给出"不透明度低于 0.7 即剪枝"——该数值明显异常（疑为排印笔误，常见实现取 0.005 量级），使用时以官方代码为准。

## 5. 实验与结果解读

- **设置**（§4.1）：Intel i9 12900K + RTX 4090；Replica（8 序列，RGB-D）、TUM RGB-D（单目与 RGB-D 各 3 序列）；跟踪精度 = 关键帧 ATE RMSE（单目做 Umeyama 对齐、RGB-D 不做尺度对齐）；渲染指标每第 5 帧算一次、排除训练视角。
- **跟踪（Table 1/2，定性 + 少量代表数字）**：单目 TUM 上平均 ATE 3.96 cm，显著优于同为无深度先验的 DSO（平均 11.0 cm）与 DepthCov-VO，与 DROID-VO 同量级；fr1 序列上超过 ORB-SLAM2（无回环口径）。RGB-D 上平均 1.47 cm，与当时最优的 Point-SLAM（1.54 cm）相当或更优，但 Point-SLAM 依赖传感器深度引导采样。Replica 上平均 0.32 cm 为最优。作者自己定位：仍不及带显式回环的 DROID-SLAM/ORB-SLAM2（约 1.6–1.7 cm）——差距指向回环模块（§4.2）。
- **渲染（Table 5/7）**：Replica RGB-D 上 PSNR 37.50 / SSIM 0.960 / LPIPS 0.070，多数指标最优；**渲染 769 FPS**，比 NICE-SLAM（0.54）、Vox-Fusion（2.17）、Point-SLAM（1.33）快两个数量级以上——光栅化 vs 光线步进/逐点采样的表示级差距。定性对照（Fig. 4）：Point-SLAM 的深度引导随机射线采样在新视角处细节退化（受限于只能沿已有视角的射线外推），本文的光栅化不依赖深度引导采样、可自由渲染新视角；补充材料 Table 5 注明 Point-SLAM 使用真值深度引导，本文不用。
- **内存（Table 4）**：单目 2.6 MB、RGB-D 3.97 MB，远小于 NICE-SLAM（40–140 MB）与 Point-SLAM（30.8 MB）；原因是剪枝只留被良好约束的高斯、且不带 SH。与离线 3DGS 的 300–700 MB 级别相比（补充材料 §9.6.1），"在线 SLAM 只维护被约束的高斯"本身就是内存优势。
- **消融（Table 3、11–13、15）**：去掉 $E_{iso}$：TUM RGB-D 平均 0.59→0.82 cm，Replica 0.58→0.82 cm——**单目下更关键**（弱纹理/弱光度约束处防拉伸）；去掉关键帧选择：3.96→8.73 cm（影响最大）；单目下去掉剪枝：3.96→46.6 cm（随机初始化的高斯若不清理会毁掉跟踪）；开 SH：渲染略好、但高斯地图与显存明显增大、ATE 略差（补充材料 §9.3：SH 会"错误解释"由相机运动引起的非视角方向变化）。
- **收敛域分析（Table 6、Fig. 5）**：固定地图后从训练视野外围采样目标位姿做 1000 次位姿优化，本方法成功率（收敛到 1 cm 内）平均 0.82（带深度训练），高于 Hash Grid SDF（0.14）与 MLP SDF（0.33）——各向同性高斯沿视线形成平滑梯度、收敛域更宽，不像哈希编码可能产生光度冲突。评测协议（补充材料 §8.3）：合成 Replica 训练 67 个视角、方形排布边长 0.5 m，测试视角均匀分布在半径 0.2–1.2 m 的更大范围；测试只平移不改旋转，以目标视角真值颜色做位姿优化（初始地图 $\mathcal{G}_{init}$ 固定，式 (16)/(17) 训练）。这一实验把"表示的几何形态"与"跟踪难易"直接挂钩——是本文论证各向同性正则价值的最有力证据。
- **实时性（Table 9/10、§9.6.2）**：fr3/office 上单目 3.2 FPS、RGB-D 2.5 FPS（多进程端到端口径）；单进程版 RGB-D 为 1.1 FPS（更多建图迭代，Table 10）；摘要口径为单目 3 fps 在线运行——离 30 fps 还远，瓶颈在逐帧 100 次迭代的位姿优化（§12 自述，作者提示二阶优化器是方向）。
- **其他**：自采 RealSense d455 场景展示透明物体（沿杯口摆高斯）与细结构（电线用细长高斯）的表达力（Fig. 7）；EuroC 双目（Machine Hall）上 easy 序列有竞争力、难序列下降（Table 14）。

## 6. 局限与后续影响

**论文自述局限**（§12）： 仅在小尺度室内场景验证，更大真实场景漂移不可避免； 无回环模块——长轨迹一致性缺失； 基准数据集上未达 30 fps 硬实时，二阶优化器是候选改进。另（§5 结论）：高斯不显式表示表面，表面法向等几何量需后处理提取。作者同时给出两条改进线索（§5、补充材料 §9.5）：回环可借鉴 surfel 系（Elastic-Fusion 式形变）；大规模立体输入（EuroC Machine Hall）的初步实验说明瓶颈在回环而非表示本身。

**后续影响**：本文与 GS-SLAM、SplaTAM（同为 CVPR 2024）共同确立"3DGS-SLAM"子领域；后续工作沿"更快/更稳/可回环"推进——如 Photo-SLAM、DROID-Splat（传统跟踪前端 + 高斯建图）、SplaTAM、GOI(RE)-SLAM、HI-SLAM2 等（这些系统在 VGGT-GS SLAM 的相关工作中被明确列为后续谱系，见本目录[精读/VGGT-GS SLAM](./VGGT-GS-SLAM_arXiv2026.md) §2）；本文的 SE(3) 解析位姿雅可比与 $E_{iso}$ 正则被后续系统普遍沿用（VGGT-GS SLAM 的式 (6) 即原样继承，见[精读/VGGT-GS SLAM](./VGGT-GS-SLAM_arXiv2026.md) §4.2）。

## 7. 与本项目对照

- **教程定位**：[第 10 章 §10.4](../10_建图与系统实战.md) 前沿表的 MonoGS 行——"可微渲染同时服务跟踪与建图：位姿与高斯在同一渲染损失下联合优化"。传统骨架（第 08 章 BA、第 09 章回环）在本文中分别退化为"窗口位姿+高斯联合优化"与"（缺失的）回环模块"——正是 §10.4 末段"可微表示嫁接在传统骨架上"论断的实证。
- **与直接法的对照**（互引[精读/LSD-SLAM](./LSD-SLAM_ECCV2014.md)）：跟踪同为"光度残差 + SE(3) 直接优化"（LSD 的 sim(3)/se(3) 图像对齐 vs 本文的式 (7)(12)），但**参考模型不同**：LSD/DSO 存显式（半）稠密深度图并对源像素做重投影比较；MonoGS 用 3DGS **渲染**出整张参考图——深度被隐式编码进高斯位置，"渲染替代显式深度图"。代价：每帧渲染整图（贵）但天然多视角一致；收益：地图可直接出新视角。
- **与 GS-SLAM / SplaTAM 的路线差异**（精读见 [GS-SLAM](./GS-SLAM_CVPR2024.md)、[SplaTAM](./SplaTAM_CVPR2024.md)）：

| 维度 | GS-SLAM | SplaTAM | **MonoGS（本文）** |
|---|---|---|---|
| 输入 | RGB-D 为主、单目可用 | RGB-D | **单目优先**，RGB-D 可扩展 |
| 跟踪信号 | 稠密 RGB-D 跟踪 + 由粗到细高斯优化（VGGT-GS SLAM 相关工作的归纳） | 渲染深度与传感器深度直接配准（教程 §10.4：光度残差跟踪、逐高斯增长） | **单目光度残差**（式 (7)），不依赖任何深度先验 |
| 地图生长 | 随跟踪自适应扩展高斯 | 反投影点 + 各向同性协方差逐点生长 | 共视性关键帧管理 + 插入/剪枝（式 (14)(15)） |
| 特有技术 | 粗到细策略 | 逐高斯点云地图 | **SE(3) 解析雅可比 + 各向同性正则** |

- **代码对照**：本项目 [projects/slam/](../../../projects/slam/README.md) 未实现 3DGS SLAM（需 CUDA 光栅化与 GPU 训练）；最接近的教学模块是 `direct/`（LSD 式半稠密直接对齐，光度残差 + 流形 G-N，与本跟踪的残差-雅可比结构同构）与 `photoba/`（DSO 式滑窗光度 BA，仿射曝光建模与式 (12) 的亮度参数化同源）。阅读建议：先跑 `direct/` 的 `semi_dense_align` 理解"光度残差对位姿的直接优化"，再回头看式 (6) 的解析雅可比——只是把"参考图像"换成了"3DGS 渲染"。

## 配套阅读

- 原论文：上述 PDF；重点 §3.2（雅可比）、补充材料 §7（超参）、§10（雅可比完整推导，式 (19)–(34)）。
- 教程：[3D 重建第 08 章](../../3d_reconstruction/08_3D高斯泼溅.md)（(8.2) 协方差参数化、(8.8) 投影、(8.10) alpha 混合、(8.11) 损失——本文式 (1)(2)(11) 的静态版）；[第 06 章](../06_视觉里程计-ii直接法.md)（直接法光度残差的一般形式）。
- 同路线精读：[LSD-SLAM](./LSD-SLAM_ECCV2014.md)（直接法与 Sim(3) 位姿图）、[DSO](./DSO_PAMI2018.md)（光度 BA 与仿射曝光）、[DROID-SLAM](./DROID-SLAM_NeurIPS2021.md)（学习式稠密 SLAM 的另一极）。
- 同期 3DGS-SLAM（三方对比见 §7）：[GS-SLAM 精读](./GS-SLAM_CVPR2024.md)、[SplaTAM 精读](./SplaTAM_CVPR2024.md)。
- 后续路线：[MASt3R-SLAM](./MASt3R-SLAM_CVPR2025.md)（先验替代几何估计）、[VGGT-GS SLAM](./VGGT-GS-SLAM_arXiv2026.md)（其单目初始化与 $E_{iso}$ 被本文后续论文沿用）。

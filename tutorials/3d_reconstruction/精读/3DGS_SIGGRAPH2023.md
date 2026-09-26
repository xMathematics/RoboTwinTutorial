# 论文精读｜3D Gaussian Splatting（SIGGRAPH 2023）

> **PDF**：[papers/3d_reconstruction/frontier/arXiv-2308.04079_3D-Gaussian-Splatting.pdf](../../../papers/3d_reconstruction/frontier/arXiv-2308.04079_3D-Gaussian-Splatting.pdf) ｜ **教程**：[第 08 章](../08_3D高斯泼溅.md)（整章即 3DGS 推导）｜ **代码**：无（教学实现规划中；可与 [SLAM 精读 DROID 文档](../../slam/精读/DROID-SLAM_NeurIPS2021.md) 的位姿估计前端组合为 3DGS-SLAM 管线，见 §7）
>
> 式号说明：本文所有"论文 Eq.x"均指本地 PDF（arXiv v1，共 14 页：主文 §1–8 + 附录 A–D）排版核对后的编号——主文编号式仅 Eq.1–7，附录 A 另有 Eq.8–11（解析梯度）；"教程 (8.x)"指[教程第 08 章](../08_3D高斯泼溅.md)式号。两者来源不同、分别标注，对应关系汇总见 §7 映射表。

## 1. 论文信息与一句话贡献

- **题目**：3D Gaussian Splatting for Real-Time Radiance Field Rendering
- **作者**：Bernhard Kerbl、Georgios Kopanas、George Dretakis（Inria / Université Côte d'Azur），Thomas Leimkühler（MPI-INF）
- **发表**：SIGGRAPH 2023（ACM Transactions on Graphics, Vol. 42, No. 4, Article 1）；arXiv:2308.04079
- **官方代码**：https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ （论文 §7.1 声明源码开源）
- **一句话贡献**：用**各向异性 3D 高斯**（anisotropic 3D Gaussians）作为辐射场的显式可微表示，配合**交错进行的自适应密度控制**（adaptive density control：克隆/分裂/剪枝）与 **tile 化可微光栅化**（16×16 tile + 全局深度排序 + alpha 混合），达到 Mip-NeRF360 级画质的同时实现首个**实时**（1080p 下 ≥30 fps；本文配置实测 134–160 fps）新视角合成，训练时间与当时最快方法相当（分钟级，约 25–42 分钟）。

## 2. 问题与动机

**要解决什么**（§1）：多视角照片的辐射场新视角合成中，质量与速度被捆绑在"神经表示"里——SOTA 的 Mip-NeRF360 画质最好但要 48 小时训练、同硬件渲染 10 秒/帧；快速方法（InstantNGP 5–10 分钟、Plenoxels 20 分钟）画质差一截。目标：**两者兼得**——SOTA 画质 + 实时渲染 + 有竞争力的训练时间。

**瓶颈的结构性原因**（§1、§2.2–2.3）：NeRF 一系是**连续场 + 逐光线体渲染**——每个像素要沿光线采样查询网络几十上百次、再做数值积分；随机采样本身就把"渲染"变贵。其速度改进（空间数据结构、哈希表、编码）仍离不开逐点查询，用不上 GPU 光栅化硬件管线。

**点基渲染的既有困境**（§2.1、§2.3）：点基 splatting 天然 GPU 友好，但 (a) 点无面积，放大即走样，需"比像素大很多的基元"（splatting 传统）；(b) 点云渲染表面有洞与不连续（hul 问题）；(c) 之前的可微点基方法依赖 MVS 几何或 CNN 特征，误差级联且时间不稳定。

**机会**：NeRF 式体渲染与有序点 alpha 混合共享**同一个图像形成模型**（§2.3，Eq.1 vs Eq.3）——既然渲染积分可以退化为"排序 + 加权求和"，那么完全可以放弃连续场，直接优化一组显式基元：把"体密度积分"换成"解析 2D 高斯求值"，把"逐光线随机采样"换成"逐 tile 光栅化"。

## 3. 方法总览

```
COLMAP/SfM 稀疏点云（带相机位姿，无需 MVS/法向）
        │  以每个点为中心创建 3D 高斯（各向异性协方差 + 不透明度 + SH 颜色）
        ▼
【优化循环】（论文 Fig.2）
   投影（EWA 仿射近似）→ 可微 tile 光栅化 → 渲染图像
        ▲                                        │ 与真值比：L1 + D-SSIM
        │                                        ▼
   自适应密度控制 ←—— 梯度反传到位置/协方差/不透明度/SH 系数
   （每 100 步：克隆/分裂；剪枝；每 3000 步重置 α）
        ▼
优化结束：约 1–5 百万个高斯（§1），1080p 实时渲染（134–160 fps）
```

三个成分缺一不可（§1 贡献列表）：(i) 各向异性 3D 高斯——连续体积辐射场的表达力 + 离散原语的光栅化效率；(ii) 属性优化与密度控制**交错**——结构错误（过/欠重建）靠增删原语修复，属性错误靠梯度修复；(iii) tile 化光栅化——一次全局排序摊销到所有 tile，各向异性 splat、可见性排序、反向传播都吃满 GPU。整个优化跑在 PyTorch + 自定义 CUDA 内核上（§7.1，光栅化是计算瓶颈）。

## 4. 关键公式推导（按论文原文式号）

**符号表**：

| 记号 | 含义 | 教程/精读对应 |
|---|---|---|
| $\mathbf{x},\boldsymbol\mu,\Sigma$ | 空间点、高斯中心（均值）、3D 协方差 | (8.1) 的 $\mathcal{G}$ 记号 |
| $R,\ S,\ \mathbf{q},\mathbf{s}$ | 旋转矩阵、对角缩放、单位四元数、尺度向量 | (8.2) |
| $W,\ \mathbf{t},\ J$ | 视变换（世界→相机）、平移、投影仿射近似的雅可比 | (8.4)(8.6) |
| $\Sigma'$ | 投影后的协方差（论文先给相机系 3×3，取前两行/列得屏幕 2×2） | (8.8) |
| $\alpha_i,\ \mathbf{c}_i,\ T_i$ | 第 $i$ 个有序原语的不透明度、颜色、透射率 | (8.9)(8.10) |
| $\sigma_i,\delta_i$ | NeRF 体密度与采样段长（仅 Eq.1 对照用） | [NeRF 教程第 03 章](../../nerf/03_数学原理与渲染方程.md) §3.5 |
| $\mathcal{L}_1,\mathcal{L}_{\text{D-SSIM}},\lambda$ | 逐像素损失、结构相似度损失、权重（$\lambda=0.2$） | (8.11) |
| $\tau_{pos},\varphi,\epsilon_\alpha$ | 位置梯度阈值（$2\times10^{-4}$）、分裂缩放因子（1.6）、剪枝不透明度阈值 | 教程 8.3 节 |

### 4.1 图像形成模型的统一：体渲染 → 有序点混合（论文 Eq.1–3）

论文 §2.3 先给 NeRF 式体渲染（Eq.1，即 [NeRF 教程第 03 章](../../nerf/03_数学原理与渲染方程.md) §3.5 的离散式）：

$$C=\sum_{i=1}^{N}T_i\bigl(1-\exp(-\sigma_i\delta_i)\bigr)\mathbf{c}_i,\qquad T_i=\exp\Bigl(-\sum_{j=1}^{i-1}\sigma_j\delta_j\Bigr). \tag{1}$$

**改写为 Eq.2**（论文原文："This can be re-written as"），两步：

第一步，把指数和拆成指数积（依据：指数律 $\exp(-\sum_j a_j)=\prod_j\exp(-a_j)$）：

$$T_i=\exp\Bigl(-\sum_{j=1}^{i-1}\sigma_j\delta_j\Bigr)=\prod_{j=1}^{i-1}\exp(-\sigma_j\delta_j).$$

第二步，定义段不透明度 $\alpha_j:=1-\exp(-\sigma_j\delta_j)$，则 $\exp(-\sigma_j\delta_j)=1-\alpha_j$（依据：恒等变形），代入得：

$$C=\sum_{i=1}^{N}T_i\,\alpha_i\,\mathbf{c}_i,\qquad \alpha_i=1-\exp(-\sigma_i\delta_i),\quad T_i=\prod_{j=1}^{i-1}(1-\alpha_j). \tag{2}$$

**Eq.3（点基混合）**：对有序点集 $\mathcal{N}$，把 $\alpha_i$ 的来源从"密度积分"换成"评估带协方差 $\Sigma$ 的 2D 高斯再乘学习的不透明度"（Kopanas et al. 2021），三因子结构不变：

$$C=\sum_{i\in\mathcal{N}}\mathbf{c}_i\,\alpha_i\prod_{j=1}^{i-1}(1-\alpha_j). \tag{3}$$

论文的论点（§2.3 原文）："the image formation model is the same, however the rendering algorithm is very different"——NeRF 是连续表示 + 代价高昂的随机采样去隐式地填满空间；高斯是离散可增删的表示，靠优化位置与不透明度填补空隙。**这一式是全文的合法性基石**：既然成像模型同构，把渲染端从"逐光线积分"换成"排序 + 混合"不改变被优化的目标。

### 4.2 3D 高斯基元与协方差参数化（论文 Eq.4、Eq.6）

以点（均值）$\boldsymbol\mu$ 为中心的世界系高斯（Eq.4；论文把 $\boldsymbol\mu$ 写在式外文字里）：

$$G(\mathbf{x})=e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)}, \tag{4}$$

乘以不透明度 $\alpha$ 参与混合。**为什么 $\Sigma$ 不能直接优化**（§4 论证链）：

- $\Sigma$ 必须对称正定（依据：把 (4) 看作未归一化密度，若存在方向 $\mathbf{v}$ 使 $\mathbf{v}^\top\Sigma^{-1}\mathbf{v}\le 0$，指数沿该方向不衰减，$G$ 非空间局域原语）；
- 梯度法对 $\Sigma$ 的 6 个独立元素做加性更新不保证半正定（半正定锥对加性更新不封闭），更新步可能产生非法协方差（论文原话："update steps and gradients can too easily create invalid covariance matrices"）。

**参数化**（Eq.6）：给定缩放矩阵 $S$ 与旋转矩阵 $R$，

$$\Sigma=R\,S\,S^{\top}R^{\top}, \tag{6}$$

存储为 3D 缩放向量 $\mathbf{s}$（经指数激活保证非负）与四元数 $\mathbf{q}$（归一化到单位长度，对应 $SO(3)$）。半正定性论证（完整版见教程 (8.2)–(8.3)，此处关键一步）：对任意 $\mathbf{v}$，

$$\mathbf{v}^\top\Sigma\,\mathbf{v}=\bigl(SR^{\top}\mathbf{v}\bigr)^\top\bigl(SR^{\top}\mathbf{v}\bigr)=\bigl\|SR^{\top}\mathbf{v}\bigr\|^2\ \ge\ 0$$

（依据：$(AB)^\top=B^\top A^\top$、$S^\top=S$、向量与自身内积非负；$R$ 正交与 $S$ 非负对角保证任意参数取值合法，无需投影）。反之由实对称矩阵谱定理，任意半正定 $\Sigma$ 均可写成该形式——参数化无表达力损失。**梯度端**：附录 A 按链式法则显式推导 $\partial\Sigma'/\partial s$（Eq.8）、$\partial\Sigma'/\partial q$（Eq.9，借四元数→旋转矩阵展开 Eq.10）乃至四元数各分量的解析梯度（Eq.11），绕开自动微分反向图的开销。

### 4.3 视角相关颜色：球谐函数（论文未编号；与 NeRF 位置编码的取舍）

论文 §4：辐射场的方向色由**球谐函数**（Spherical Harmonics, SH）表示（§2.2 称这是standard practice）。论文未给编号式，按 SH 标准定义写出（每通道 4 band、$(l+1)^2$ 合计 16 个可学习系数 $k_{lm}$）：

$$\mathbf{c}(\mathbf{d})=\sum_{l=0}^{3}\sum_{m=-l}^{l}k_{lm}\,Y_l^m(\mathbf{d}),$$

其中 $\mathbf{d}$ 为视角方向、$Y_l^m$ 为实 SH 基。$l=0$ 项即视角无关的基色（diffuse color），高阶项承载视角相关效果（§7.1 用语）。

**与 NeRF 位置编码 $\gamma(\cdot)$ 的对比**（[NeRF 教程第 04 章](../../nerf/04_网络架构与位置编码.md) §4.2，NeRF 论文 Eq.4）：NeRF 把**坐标** $\mathbf{p}$ 经固定正弦基 $\gamma(\mathbf{p})=(\sin 2^0\pi\mathbf{p},\cos 2^0\pi\mathbf{p},\dots)$ 升维后交给 8 层 MLP 解码——基固定、解码靠深层网络。3DGS 的取舍：颜色只依赖**方向** $\mathbf{d}$ 且方向变化光滑（低频为主），一个对固定 SH 基的**线性模型**就够——可学习系数直接挂在每个原语上，**无网络参与渲染**。代价是 SH 对"缺失视角"敏感（§7.1：采集角度有缺口时零阶分量可能被优化出完全错误的值），论文用由低到高的 warm-up（§4.6 末）缓解：先只优化零阶，之后每 1000 步引入一个 band，直到 4 个 band 全部激活。

### 4.4 投影与 tile 光栅化（论文 Eq.5 + §6；EWA splatting 路线）

**目标**：把 3D 协方差 $\Sigma$ 解析地投影成屏幕上的 2D 协方差，避免逐像素光线—椭球求交。

**(a) 局部仿射近似**。世界系高斯先精确刚体变换到相机系（依据：$T_{cw}$ 的定义，此步无近似）：

$$\mathbf{x}_{cam}=W\mathbf{x}+\mathbf{t}. \tag{8.4}$$

透视投影 $\boldsymbol\pi(\mathbf{x}_{cam})=(f_xX/Z+c_x,\ f_yY/Z+c_y)$ 非线性（除以 $Z$）。在**各高斯自己的中心** $\tilde{\mathbf{x}}_{cam}=\boldsymbol\mu_{cam}$ 处一阶泰勒展开（依据：多元函数一阶泰勒公式；视锥局部高阶项可略——这是唯一的近似，误差随高斯增大而增大，§8 亦承认此局限）：

$$\mathbf{u}\approx\boldsymbol\pi(\tilde{\mathbf{x}}_{cam})+J\,(\mathbf{x}_{cam}-\tilde{\mathbf{x}}_{cam}). \tag{8.5}$$

逐元素求雅可比（依据：偏导 + 商的求导法则；完整过程见教程 (8.6)）：

$$J=\begin{bmatrix} f_x/Z & 0 & -f_xX/Z^2\\[2pt] 0 & f_y/Z & -f_yY/Z^2 \end{bmatrix}.$$

**(b) 协方差传播**。引理：仿射映射 $\mathbf{y}=A\mathbf{x}+\mathbf{b}$ 下 $\Sigma_y=A\Sigma_xA^\top$（依据：从定义 $E[(\mathbf{y}-\bar{\mathbf{y}})(\mathbf{y}-\bar{\mathbf{y}})^\top]$ 出发，中心化差恒等式与期望线性，教程 (8.7) 有逐步推导）。应用两次（$A=W$ 得相机系协方差，$A=J$ 得屏幕协方差），合并即论文 Eq.5：

$$\Sigma'=J\,W\,\Sigma\,W^{\top}J^{\top}. \tag{5}$$

一个表述差异需说明：论文按 Zwicker et al. 2001a 的写法，$J$ 是投影变换仿射近似的雅可比、$\Sigma'$ 为 3×3，取前两行/列得到 2×2 屏幕协方差；教程 (8.6)(8.8) 直接用针孔投影的 2×3 雅可比。二者数值等价（依据：屏幕协方差只涉及前两个输出坐标，故只需 $J$ 的前两行，丢第三行/列不改变 2×2 结果）——这正是官方实现中的做法。

**(c) tile 化光栅化与深度排序（性能设计的算法含义，§6）**：屏幕划成 16×16 tile；每个高斯按其 footprint 复制到覆盖 tile，产生 (tile ID, 视空间深度) 复合键，**每帧一次** GPU Radix sort 全局排序（依据：排序一次摊销到所有 tile，避免逐像素排序）；逐 tile 启动线程块，从排好序的区间**前到后**遍历混合（即 (3)），像素饱和（$\alpha$ 累计到 1）即停。算法含义有三：其一，"沿光线采样积分"被"平面上的解析 splat 求值"取代——每像素成本变为常数级混合加法；其二，无隐式空间结构，原语始终留在欧氏空间，不需为远/大高斯设计投影/收缩策略（§5.2 原话）；其三，反向传播需要重建混合序列——论文不缓存逐步透射链，而在前向每像素存"累计不透明度"，反向时从后向前用各点 $\alpha$ 逐项除回去（§6，附数值稳定处理：$\alpha<\epsilon$ 跳过、上式累计封顶 0.99）。

### 4.5 优化目标（论文 Eq.7；D-SSIM 与 SSIM 的定义）

$$\mathcal{L}=(1-\lambda)\,\mathcal{L}_1+\lambda\,\mathcal{L}_{\text{D-SSIM}}, \tag{7}$$

（论文 §5.1；全文实验取 $\lambda=0.2$。）$\mathcal{L}_1=\frac{1}{|\Omega|}\sum_{\mathbf{p}\in\Omega}|\hat{I}(\mathbf{p})-I(\mathbf{p})|$ 为逐像素绝对误差。D-SSIM = **(1 − SSIM) 的均值**（标准定义，论文未给展开式）。SSIM（Wang et al. 2004）在局部高斯窗口内用四类统计量构造：

$$\operatorname{SSIM}(x,y)=\frac{(2\mu_x\mu_y+C_1)\,(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)\,(\sigma_x^2+\sigma_y^2+C_2)},\qquad \mathcal{L}_{\text{D-SSIM}}=\frac{1}{|\Omega|}\sum_{\mathbf{p}}\bigl(1-\operatorname{SSIM}(\hat I_{\mathbf{p}}, I_{\mathbf{p}})\bigr),$$

其中 $\mu_x,\mu_y$ 为窗口亮度均值、$\sigma_x,\sigma_y$ 方差、$\sigma_{xy}$ 协方差（依据：局部矩的定义；$C_1,C_2$ 防小分母稳定项）。分工：$\mathcal{L}_1$ 逐像素稳健、对离群不敏感；D-SSIM 从亮度/对比/结构三因子衡量局部结构差异，补 $\mathcal{L}_1$ 对低频结构不敏感的短板（教程 8.3 ⑤ 同此论证）。其余优化细节（§5.1）：$\alpha$ 用 sigmoid 限制在 $[0,1]$、尺度用指数激活；初始协方差取各向同性、轴长 = 到最近三个点的平均距离；位置学习率用类似 Plenoxels 的指数衰减调度。

### 4.6 自适应密度控制（论文 §5.2 + 附录 B Algorithm 1；经验规则）

优化与结构修改**交错**：每 **100 步**做一次致密化（densify）与剪枝，均为经验规则（论文明确以工程阈值给出，非推导；教程 8.3 ④ 同此声明）。

- **梯度判据**：统计各高斯视空间位置梯度的平均幅值，超过阈值 $\tau_{pos}=0.0002$ 者为候选（依据：大梯度 ⟺ 该处渲染与真像仍有差、移动原语可降损，即欠重建/过重建区域的**一阶探测器**）。
- **克隆 vs 分裂（大/小判据）**：候选中原语**小**（尺度 $<\tau_s$）说明欠重建——**克隆**一份、沿位置梯度方向平移；候选中原语**大**（尺度 $>\tau_s$，对应高方差过重建区）说明一个大 splat 盖住了本该分开的结构——**分裂**为两个、尺度除以 $\varphi=1.6$，新位置按原 3D 高斯作为 PDF 采样。分裂同时控制总量：删除原语、只净增一（"conservative total volume"）。
- **不透明度剪枝**：$\alpha<\epsilon_\alpha$（$\alpha$ 阈值）或原语在屏幕/世界空间过大者移除（依据：$\alpha$ 过小者对 (3) 贡献趋零却占显存与计算）。
- **每 3000 步重置 $\alpha$**：论文原话——"An effective way to moderate the increase in the number of Gaussians is to set the $\alpha$ value close to zero every $N=3000$ iterations"。原因（§5.2）：无结构化的原语易在输入相机附近"堆积"出高密度漂浮片（贴相机挡视线即消除误差的局部极小），重置迫使几何重建退回到位置/尺度，同时暴露可剪枝的漂浮片。脚注特别声明：这里的"高斯密度"（原语数量）与 NeRF 文献的体密度 $\sigma$ 无关。

附录 B 的 Algorithm 1 给出完整伪代码：`i % iRemovalIteration == 0` 时剪枝（$\alpha<\epsilon_\alpha$ 或过大）、`i % iDensifyIteration == 0` 且 $\nabla_{\boldsymbol\mu}\mathcal{L}>\tau_p$ 时按 $\|S\|>\tau_s$ 分裂否则克隆——与上述规则一一对应。

## 5. 实验与结果解读

**设置**（§7.1–7.2）：13 个真实场景 = Mip-NeRF360 数据集全部 9 个 + Tanks&Temples 2 个（Truck/Train）+ Deep Blending 2 个（DrJohnson/Playroom），另有合成 Blender 数据集；A6000 GPU；所有场景同一套超参；每 8 帧取 1 帧作测试。

**真实场景主表（Table 1，已逐项核对）**：

| 方法 | M360 SSIM/PSNR/LPIPS | 训练 | FPS |
|---|---|---|---|
| Plenoxels | 0.626 / 23.08 / 0.463 | 25m49s | 6.79 |
| InstantNGP-Base | 0.671 / 25.30 / 0.371 | 5m37s | 11.7 |
| Mip-NeRF360（原文数字†） | 0.792 / 27.69 / 0.237 | 48h | 0.06 |
| **Ours-7K** | 0.770 / 25.60 / 0.279 | 6m25s | 160 |
| **Ours-30K** | 0.815 / 27.21 / 0.214 | 41m33s | 134 |

读法（定性优先）：30K 配置以约 40 分钟训练把 Mip-NeRF360 的 **48 小时**压到分钟级，LPIPS（感知指标）**更好**（0.214 vs 0.237），PSNR 略低（27.21 vs 27.69，差约 0.5 dB）——换取 **134 vs 0.06 fps**（三个数据集上 Ours-30K 均为 134–154 fps，7K 配置 160–197 fps，合成场景 180–300 fps）。速度量级差异的本质在 §4.4：光栅化混合 vs 逐光线网络积分。作者自跑 Mip-NeRF360 的全数据集均值为 PSNR 27.58 / SSIM 0.790 / LPIPS 0.240（附录 D），结论不变。内存：训练后场景数百 MB（734MB），训练中峰值可超 20GB（§8）。

**合成场景（Table 2）**：从 **10 万个均匀随机点**（无 SfM）初始化，Ours-30K 平均 PSNR 33.32 dB，与 Mip-NeRF（33.09）、Point-NeRF（33.30）相当或略优——因为合成场景有干净背景包围盒，随机点即可覆盖。注意这**不与**真实场景的消融矛盾（见下）。

**消融（Table 3，Truck/Garden/Bicycle 三场景 5K/30K 平均，已逐项核对）**：

| 配置 | Avg-5K | Avg-30K |
|---|---|---|
| Random Init（随机初始化） | 19.17 | 20.42 |
| No-Split（去分裂） | 21.50 | 23.90 |
| No-Clone（去克隆） | 23.35 | 25.91 |
| Isotropic（各向同性协方差） | 23.56 | 25.23 |
| No-SH（去球谐） | 23.48 | 25.35 |
| **Full** | **23.90** | **26.05** |

- **SfM 初始化 vs 随机初始化**：真实无界场景上随机初始化掉 4.7–5.6 dB（19.17/20.42 vs 23.90/26.05）。但论文强调（§7.3）：随机初始化也**不会彻底失败**（均匀立方体采样 ×3 场景包围盒），只是退化主要发生在背景且产生更多优化删不掉的漂浮片；SfM 点云的价值 = 给出"哪里有东西"的先验。合成场景（Table 2）随机初始化够用，因场景有界且背景干净。
- **致密化策略**：**分裂**对背景重建关键（No-Split Avg-30K 23.90 vs Full 26.05，掉 2.15 dB，其中 Garden 从 27.70 掉到 26.11——论文 Fig.8 的过重建可视化）；**克隆**带来更快收敛、对薄结构关键（No-Clone Avg-5K 23.35 vs Full 23.90，Bicycle 轮辐等细小结构在 3K 步即可见退化，Fig.8 自行车行）。
- **各向异性**：Isotropic 平均掉 0.8 dB（26.05→25.23），但可视化（Fig.10）显示更大差异在"与表面对齐的能力"——各向异性允许高斯压扁贴合表面。
- **SH**：No-SH 掉 0.7 dB，因为视角相关效应（反光等）无法补偿。

## 6. 局限与后续影响

**论文自述局限**（§7.4）：

- 观测不足区域有伪影（各方法共有，Mip-NeRF360 亦然）；
- 各向异性高斯可能描述出"拉长的斑点状"（splotchy）高斯；
- 偶发 popping 伪影：视点变化时大高斯的可见性突然切换/深度排序突变（论文提到 guard band 剔除与抗混叠是未来工作）；
- 无正则化，加入或可同时解决未观测区与 popping；
- 大场景需调低位置学习率；
- 内存：训练峰值可超 20GB，渲染需数百 MB 存模型 + 30–50MB 光栅化临时内存（§8）。

**结构性局限（教程视角）**：原语数量爆炸（1–5 百万/场景，§1）；几何是"体积软原语"——最优解趋向把最薄轴压扁贴表面，但**深度/法向查询与网格提取需后处理**（教程 8.4 的"扁平化"论证）。

**后续影响**：直接催生 3DGS-SLAM/重建系（§7）与两条修路方向：(a) **表面化**——2DGS（投影为 2D 高斯、有向平面原语）与 SuGaR（正则化使高斯贴面再提取网格）专治"表面质量"；(b) **抗混叠**——Mip-Splatting 处理 Eq.5 仿射近似在尺度变化下的走样。这些正是 §7.4 自述局限的社区回应。

## 7. 与本项目对照

**教程第 08 章映射表（已 Read 教程核实编号）**：

| 论文 | 教程第 08 章 | 内容 |
|---|---|---|
| Eq.4 | (8.1) | 3D 高斯基元 $G(\mathbf{x})=\exp(-\frac12(\mathbf{x}-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu))$ |
| Eq.6 | (8.2)(8.3) | $\Sigma=RSS^\top R^\top$ 及半正定性 $\|SR^\top\mathbf{v}\|^2\ge0$ |
| Eq.5 | (8.4)–(8.8) | EWA 投影链：刚体变换 → 一阶泰勒 → 雅可比 → 协方差传播 → $\Sigma'=JW\Sigma W^\top J^\top$ |
| Eq.3 | (8.9)(8.10) | 解析 2D 高斯求值 $\alpha_k$ 与排序 alpha 混合 $C=\sum_k\mathbf{c}_k\alpha_k\prod_{j<k}(1-\alpha_j)$ |
| Eq.1–2 | [NeRF 教程第 03 章](../../nerf/03_数学原理与渲染方程.md) §3.5 | 体渲染离散式（对照物） |
| Eq.7 | (8.11) | $\mathcal{L}=(1-\lambda)\mathcal{L}_1+\lambda\mathcal{L}_{\text{D-SSIM}}$，$\lambda=0.2$ |
| §5.2 / Alg.1 | 8.3 节 | $\tau_p(=2\times10^{-4})$ 判据、$\varphi=1.6$、$\epsilon_\alpha$ 剪枝、每 100 步致密化、每 3000 步重置——经验规则 |

教程以"每步注明依据"的方式展开了本精读 §4.4 的全部推导；本精读补充了教程未展开的 Eq.1→2→3 改写链（§4.1）、附录 A 的解析梯度动机（§4.2）、SH 的基函数视角（§4.3）与逐项消融数字（§5）。

**SLAM 地图表示的迁移：NeRF-SLAM → 3DGS-SLAM**。NeRF-SLAM（ICRA 2023，见 [SLAM 精读](../../slam/精读/NeRF-SLAM_ICRA2023.md)）= DROID-SLAM（位姿/深度前端，见 [DROID 精读](../../slam/精读/DROID-SLAM_NeurIPS2021.md)）+ Instant-NGP 地图，受限于逐光线渲染的更新速度；3DGS 的显式原语让地图**可增删、可局部更新、可光栅化**，催生一批 3DGS-SLAM：[GS-SLAM](../../slam/精读/GS-SLAM_CVPR2024.md)（RGB-D，基于面元/高斯的增量地图）、[SplaTAM](../../slam/精读/SplaTAM_CVPR2024.md)（以高斯图为地图、渲染深度做跟踪）、[MonoGS](../../slam/精读/MonoGS_CVPR2024.md)（单目、跟踪与建图共享可微光栅化损失）。迁移的本质：地图表示从"隐式场 + 哈希缓存"换成本精读的显式高斯集合——跟踪端的渲染损失反传（Eq.7 + Eq.3）与建图端的自适应密度控制（§4.6）共用同一条可微管线。机器人实战选型见[第 10 章](../10_机器人场景中的重建实战与选型.md)。

## 配套阅读

- 原论文：[arXiv-2308.04079_3D-Gaussian-Splatting.pdf](../../../papers/3d_reconstruction/frontier/arXiv-2308.04079_3D-Gaussian-Splatting.pdf)（附录 A：解析梯度 Eq.8–11；附录 B：Algorithm 1；附录 C：光栅化排序细节；附录 D：逐场景指标）。
- EWA splatting 理论源头：Zwicker et al., *"EWA Splatting"*, IEEE TVCG 2002（Eq.5 的仿射近似与协方差传播出处）。
- 教程主线：[第 08 章](../08_3D高斯泼溅.md)（本精读的推导骨架）｜ [第 09 章](../09_哈希编码与前馈重建.md)（"快"的另一条路：哈希编码，见其 [Instant-NGP 精读](./InstantNGP_SIGGRAPH2022.md)）｜ [NeRF 教程第 03 章](../../nerf/03_数学原理与渲染方程.md)（Eq.1 的同构出处）。
- 姊妹精读：[Neuralangelo](./Neuralangelo_CVPR2023.md)（隐式路线的"哈希加速"）、[NeuS](./NeuS_NeurIPS2021.md)（表面精度路线，与 §6 表面局限对照）。
- SLAM 侧互引：[NeRF-SLAM](../../slam/精读/NeRF-SLAM_ICRA2023.md) ｜ [GS-SLAM](../../slam/精读/GS-SLAM_CVPR2024.md) ｜ [SplaTAM](../../slam/精读/SplaTAM_CVPR2024.md) ｜ [MonoGS](../../slam/精读/MonoGS_CVPR2024.md)。

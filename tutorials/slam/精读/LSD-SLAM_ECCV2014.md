# 论文精读｜LSD-SLAM：大规模直接法单目 SLAM（ECCV 2014）

> **PDF**：[papers/slam/classics/LSD-SLAM_ECCV2014_EngelSchopsCremers.pdf](../../../papers/slam/classics/LSD-SLAM_ECCV2014_EngelSchopsCremers.pdf) ｜ **教程**：[第 06 章](../06_视觉里程计-ii直接法.md)（直接法）｜ **代码**：[projects/slam/direct/](../../../projects/slam/direct/)（半稠密直接对齐）

## 1. 论文信息与一句话贡献

- **题目**：LSD-SLAM: Large-Scale Direct Monocular SLAM
- **作者**：Jakob Engel、Thomas Schöps、Daniel Cremers（慕尼黑工业大学 TUM）
- **发表**：ECCV 2014（本文精读采用 TUM 官方 16 页版，§1–5 + 参考文献；所有式号以该 PDF 为准）
- **一句话贡献**：把**半稠密直接法**（semi-dense direct image alignment）从纯视觉里程计升级成完整 SLAM——地图是**关键帧位姿图**（每帧附一张带方差的半稠密逆深度图），关键帧间的边用 **Sim(3) 相似变换**表达与估计，从而在 CPU 上实时构建**大尺度、全局一致**的单目地图，并显式吸收与检测尺度漂移。两大技术支柱： 直接 sim(3) 直接图像对齐跟踪； 把深度估计的噪声**概率一致地**传入跟踪的方差归一化残差。

## 2. 问题与动机

**要解决什么**：单目相机的实时 SLAM——同时估计位姿并重建 3D 环境，且地图要**大尺度**（数百米轨迹）、**全局一致**（回环后闭合）。

**当时方法的瓶颈**（论文 §1）：

- **特征点法**把问题拆成"提特征 → 匹配 → 几何估计"两步，*只有符合特征类型的信息才被使用*——直线/曲线边缘（人造环境中占比很大）被整体丢弃；高维特征（[16][6]）与区域特征（[5]）因估计空间庞大而罕用。
- **直接法**在图像灰度上直接优化几何，*用上图像里的全部信息*，弱纹理处精度与鲁棒性更高；但当时的直接法全部是**纯视觉里程计**：变分稠密公式（[24][20][21]）计算量大、需顶级 GPU 才能实时；半稠密滤波公式（[9]，即 Engel 等的 ICCV 2013 半稠密 VO）虽在 CPU 甚至手机上实时（[22]），却"only locally track the motion of the camera and do not build a consistent, global map"。
- **单目的尺度问题**：单目 SLAM 固有尺度模糊，绝对尺度不可观（理论根源即教程 (5.9)）；尺度随时间**漂移**是主要误差来源之一（§3.5 引 [28]）。已有位姿图 SLAM（[14] 的 RGB-D 位姿图）处理刚体误差，keypoint 系的 [23]（Strasdat, RSS 2010）已指出要用 3D 相似变换表达单目尺度漂移——但直接法一侧尚无对应物。

**动机一句话**：直接法有信息优势、位姿图有一致性优势，把两者接起来缺的是三块拼图——sim(3) 直接对齐（尺度感知的跟踪）、概率一致的深度不确定性传播、尺度可漂的位姿图优化。本文一次补齐。

## 3. 方法总览

论文 Fig. 3 给出三组件流水线（文字版）：

```
新图像 I_i ──▶【跟踪 Tracking（§3.3, Eq.(12)-(15)）】
                在当前关键帧 K_j 上做直接 se(3) 图像对齐，
                最小化方差归一化光度残差（含 Huber）→ ξ_{j·i} ∈ se(3)
                     │ (Sect. 3.3)
                     ▼
        【深度图估计 Depth Map Estimation（§3.4）】
          是否成为新关键帧？──由 Eq.(16) 的 dist 阈值判定
           ├─ 否：用该帧做小基线立体比较，滤波精化当前关键帧深度图（引 [9]）
           └─ 是：新建关键帧——投影上一关键帧的点 + 一次空间正则 + 离群剔除，
                  深度图整体缩放到平均逆深度为 1（缩放因子并入 sim(3) 位姿）
                     │ (Sect. 3.4)
                     ▼
        【地图优化 Map Optimization（§3.2, 3.5, 3.6）】
          新关键帧入图：对近邻关键帧做直接 sim(3) 对齐估边 ξ_{j·i} ∈ sim(3)（Eq.(17)-(19)）
          + 回环候选检索（10 个最近帧 + DBoW2 外观候选 [11]）+ 双向跟踪校验（Eq.(20)）
          → 后台持续 sim(3) 位姿图优化（Eq.(21)，g2o [18]）
```

**地图表示**（§3.2）：关键帧 $\mathcal{K}_i = (I_i, D_i, V_i)$——图像 $I_i: \Omega \to \mathbb{R}$、逆深度图 $D_i: \Omega_{D_i} \to \mathbb{R}^+$ 及其方差 $V_i: \Omega_{D_i} \to \mathbb{R}^+$；$D_i, V_i$ 只在梯度足够大的像素子集 $\Omega_{D_i} \subset \Omega$ 上有定义（**半稠密**）。边 $\mathcal{E}_{ji}$ 存相对对齐 $\boldsymbol{\xi}_{ji} \in \mathfrak{sim}(3)$ 及协方差 $\boldsymbol{\Sigma}_{ji}$。**初始化**（§3.1）：第一关键帧取随机深度图 + 大方差 + 足够平移运动，几个关键帧后自然收敛到正确深度构型——单目尺度由此约定（对照教程 (5.9)：绝对尺度是规范自由度，必须由约定或传感器钉住）。

## 4. 关键公式推导（式号按论文 PDF 原文）

### 4.0 符号表

| 符号 | 含义（首次出现处） |
|---|---|
| $\mathbf{G} \in SE(3)$, $\boldsymbol{\xi} \in \mathfrak{se}(3)$ | 刚体变换及其李代数向量（Eq.(1)） |
| $\circ$ | 位姿复合算子：$\boldsymbol{\xi}_{k·i} := \boldsymbol{\xi}_{k·j} \circ \boldsymbol{\xi}_{j·i}$（Eq.(2)），$i\to j\to k$ |
| $\omega(\mathbf{p}, d, \boldsymbol{\xi})$ | 3D 投影 warp：归一化坐标 $\mathbf{p}$ + 逆深度 $d$ → 目标帧 $(u', v', d')$（Eq.(3)） |
| $\mathbf{S} \in \mathrm{Sim}(3)$, $s$ | 相似变换及尺度分量（Eq.(4)），$\boldsymbol{\xi} \in \mathfrak{sim}(3) \cong \mathbb{R}^7$ |
| $I_{ref}, D_{ref}$; $I_i, I_j$ | 参考关键帧图像/深度图；新帧 $I_i$ / 目标关键帧 $I_j$ |
| $r_p,\ r_d$ | 光度残差（Eq.(13)）/ 深度残差（Eq.(18)） |
| $\sigma_I^2,\ V_i(\mathbf{p})$ | 高斯图像噪声方差 / 逆深度图存储的方差（§2.3） |
| $\sigma^2_{r_p},\ \sigma^2_{r_d}$ | 残差的传播方差（Eq.(14)、(19)） |
| $\|\cdot\|_\delta$ | Huber 范数（Eq.(15)），$\delta$ 为核宽度 |
| $\mathbf{J}, \mathbf{W}, \mathbf{r}$ | 残差雅可比、权重矩阵、堆叠残差（Eq.(6)、(8)） |
| $\mathrm{Ad}_{j·k}$ | sim(3) 伴随（adjoint），切空间间搬协方差用（Eq.(20)） |

### 4.1 变换记号与投影 warp（论文 Eq.(1)–(4)）

Eq.(1) 定义 $SE(3)$ 元素 $\mathbf{G} = \begin{pmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0} & 1 \end{pmatrix}$；Eq.(2) 定义复合（依据：群乘法的对数映射表述，$\exp_{se(3)}(\boldsymbol{\xi}_{k·j})\exp_{se(3)}(\boldsymbol{\xi}_{j·i})$ 仍是 $SE(3)$ 元素——群封闭性，教程第 02 章 (2.25) 同一构造）。

**Eq.(3) 的 warp 逐步展开**（论文只给定义，展开是后文一切雅可比的基础）。输入：归一化坐标 $\mathbf{p} = (x, y)^\top$、逆深度 $d$；Eq.(3) 右端的 4 维向量 $(\mathbf{p}^\top/d,\ 1/d,\ 1,\ 1)^\top = (x/d,\ y/d,\ 1/d,\ 1)^\top$ 恰是齐次 3D 点 $(x/d,\ y/d,\ 1/d)^\top$——即**深度 $z = 1/d$ 的反投影点**（依据：反投影 = 归一化方向 × 深度）。左乘 $\exp_{se(3)}(\boldsymbol{\xi}) = \begin{psmallmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0} & 1 \end{psmallmatrix}$，得目标帧相机系坐标（依据：分块矩阵乘法）：

$$\begin{pmatrix} x' \\ y' \\ z' \end{pmatrix} = \frac{1}{d}\,\underbrace{\mathbf{R} \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}}_{=:\ \mathbf{k}} + \mathbf{t} = \begin{pmatrix} k_1/d + t_x \\ k_2/d + t_y \\ k_3/d + t_z \end{pmatrix}.$$

通分提出 $d$（依据：分子分母同乘 $d \neq 0$）：

$$u' = \frac{x'}{z'} = \frac{k_1 + t_x\, d}{k_3 + t_z\, d}, \qquad v' = \frac{y'}{z'} = \frac{k_2 + t_y\, d}{k_3 + t_z\, d}, \qquad d' := \frac{1}{z'} = \frac{d}{k_3 + t_z\, d}. \tag{3'}$$

所以 Eq.(3) 的 $\omega$ 返回三元组 $(u', v', d')$：前两维是目标帧像点，第三维是**目标帧逆深度**——后面 Eq.(18) 的深度残差直接消费它。两个导数（依据：商法则）：

$$\frac{\partial u'}{\partial d} = \frac{t_x k_3 - t_z k_1}{(k_3 + t_z d)^2}, \qquad \frac{\partial d'}{\partial d} = \frac{k_3}{(k_3 + t_z d)^2}. \tag{A}$$

自检：纯旋转（$\mathbf{t} = \mathbf{0}$）时 $\partial u'/\partial d = 0$——深度噪声完全不进光度残差，与论文 Fig. 4(a) 的说明一致（"for pure rotation, depth noise has no effect on the residual noise"）。

Eq.(4) 定义 $\mathrm{Sim}(3)$：$\mathbf{S} = \begin{pmatrix} s\mathbf{R} & \mathbf{t} \\ \mathbf{0} & 1 \end{pmatrix}$，$s \in \mathbb{R}^+$，李代数 $\boldsymbol{\xi} \in \mathbb{R}^7$——比 $SE(3)$ 多一个尺度自由度（教程第 09 章 (9.4) 同一定义）。

### 4.2 加权 Gauss-Newton 直接对齐（论文 Eq.(5)–(11)）

**目标函数**（Eq.(5)，两帧直接对齐的最小二乘）：

$$E(\boldsymbol{\xi}) = \sum_{i} \Big( I_{ref}(\mathbf{p}_i) - I\big(\omega(\mathbf{p}_i, D_{ref}(\mathbf{p}_i), \boldsymbol{\xi})\big) \Big)^2 = \sum_i r_i^2(\boldsymbol{\xi})$$

（依据：假设 i.i.d. 高斯残差时的最大似然——教程第 08 章 (8.2) 的高斯负对数似然在等方差特例下的形态；这就是教程 (6.4) 光度残差的论文原版，$\mathcal{P}$ 取半稠密像素集。）

**G-N 增量**（Eq.(6)-(7)，左复合参数化）。每步求左乘增量 $\delta\boldsymbol{\xi}^{(n)}$：残差在当前估计处一阶展开（依据：多元 Taylor 展开取一阶项，即教程 (8.3)）：

$$\mathbf{r}\big(\exp_{se(3)}(\delta\boldsymbol{\xi}) \circ \boldsymbol{\xi}^{(n)}\big) \approx \mathbf{r}(\boldsymbol{\xi}^{(n)}) + \mathbf{J}\,\delta\boldsymbol{\xi}, \qquad \mathbf{J} = \left.\frac{\partial (\mathbf{r} \circ \exp_{se(3)}(\boldsymbol{\xi}))}{\partial \boldsymbol{\xi}}\right|_{\boldsymbol{\xi} = \mathbf{0}}$$

代入 $\|\mathbf{r} + \mathbf{J}\,\delta\boldsymbol{\xi}\|^2$、对 $\delta\boldsymbol{\xi}$ 求导置零（依据：教程 (8.4)→(8.5) 的二次型求导，$\partial(\Delta^\top H\Delta)/\partial\Delta = 2H\Delta$）：

$$\delta\boldsymbol{\xi}^{(n)} = -(\mathbf{J}^\top \mathbf{J})^{-1} \mathbf{J}^\top \mathbf{r}(\boldsymbol{\xi}^{(n)}) \tag{6}$$

更新用**群复合**而非加法（Eq.(7)：$\boldsymbol{\xi}^{(n+1)} = \delta\boldsymbol{\xi}^{(n)} \circ \boldsymbol{\xi}^{(n)}$；依据：指数映射精确落在流形上，教程 (8.9) 同理——"with a slight abuse of notation" 是论文原话）。

**鲁棒加权**（Eq.(8)-(9)）。对离群（遮挡、反射），改用迭代重加权的 $E(\boldsymbol{\xi}) = \sum_i w_i(\boldsymbol{\xi})\, r_i^2(\boldsymbol{\xi})$（Eq.(8)）：每次迭代冻结 $\mathbf{W} = \mathbf{W}(\boldsymbol{\xi}^{(n)})$（降权大残差），目标当步变成加权最小二乘——把 (6) 的 $\mathbf{J}^\top\mathbf{J}$、$\mathbf{J}^\top\mathbf{r}$ 各插入 $\mathbf{W}$（依据：对 $\sum_i w_i r_i^2$ 重复 (6) 的推导，$w_i$ 视为常数）：

$$\delta\boldsymbol{\xi}^{(n)} = -(\mathbf{J}^\top \mathbf{W}\mathbf{J})^{-1} \mathbf{J}^\top \mathbf{W}\, \mathbf{r}(\boldsymbol{\xi}^{(n)}) \tag{9}$$

**不确定性估计**（Eq.(10)-(11)）。残差近高斯时，$(\mathbf{J}^\top\mathbf{W}\mathbf{J})^{-1}$ 是左乘误差 $\boldsymbol{\xi}^{(n)} = \boldsymbol{\varepsilon} \circ \boldsymbol{\xi}_{true}$、$\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_{\boldsymbol{\xi}})$ 的协方差估计（Eq.(10)；论文第 6 页提醒：实际残差高度相关，$\boldsymbol{\Sigma}_{\boldsymbol{\xi}}$ 只是下界，但保留了"噪声在各自由度间相关性"的信息；且左复合约定须与位姿图实现一致——g2o 默认右复合）。一阶不确定性传播（Eq.(11)，§2.3）：

$$\boldsymbol{\Sigma}_f \approx \mathbf{J}_f\, \boldsymbol{\Sigma}_x\, \mathbf{J}_f^\top$$

（依据：$f(\mathbf{x} + \delta) \approx f(\mathbf{x}) + \mathbf{J}_f\,\delta$ 一阶展开后对 $\mathbb{E}[\delta\delta^\top] = \boldsymbol{\Sigma}_x$ 取期望——全部后续方差公式都由它生出。）

### 4.3 跟踪新帧：方差归一化光度残差（论文 Eq.(12)–(15)）

跟踪是把新帧 $I_i$ 对齐到当前关键帧 $\mathcal{K}_j$，估计相对位姿 $\boldsymbol{\xi}_{j·i} \in \mathfrak{se}(3)$。**方差归一化目标**（Eq.(12)）：

$$E_{r_p}(\boldsymbol{\xi}_{j·i}) = \sum_{\mathbf{p} \in \Omega_{D_i}} \left\| \frac{r_p^2(\mathbf{p}, \boldsymbol{\xi}_{j·i})}{\sigma^2_{r_p}(\mathbf{p}, \boldsymbol{\xi}_{j·i})} \right\|_\delta, \qquad r_p(\mathbf{p}, \boldsymbol{\xi}_{j·i}) := I_i(\mathbf{p}) - I_j\big(\omega(\mathbf{p}, D_i(\mathbf{p}), \boldsymbol{\xi}_{j·i})\big) \tag{12}{,}(13)$$

**Eq.(14) 的推导——残差方差从哪来**。残差 $r_p$ 有两个不确定输入：两帧图像噪声与参考帧逆深度 $d = D_i(\mathbf{p})$（方差 $V_i(\mathbf{p})$，由深度图存储）。

- **图像噪声**：$I_i(\mathbf{p})$ 与 $I_j(\cdot)$ 各自带独立高斯噪声 $\sigma_I^2$（论文假设），和的方差 = 方差和（依据：独立随机变量方差可加）：贡献 $2\sigma_I^2$。
- **深度噪声**：$d$ 扰动 $\delta_d$ 时，像点经 Eq.(3') 移动，采样灰度随之变化。一阶展开（依据：链式法则 + Eq.(11) 的标量形式）：

$$\delta r_p = \frac{\partial I_j}{\partial \mathbf{u}'} \cdot \frac{\partial (u', v')}{\partial d}\, \delta_d = \Big( I_u\, \tfrac{\partial u'}{\partial d} + I_v\, \tfrac{\partial v'}{\partial d} \Big) \delta_d \;\Longrightarrow\; \text{方差贡献} = \Big( \frac{\partial I_j\big(\omega(\mathbf{p}, \boldsymbol{\xi}_{j·i})\big)}{\partial D_i(\mathbf{p})} \Big)^2 V_i(\mathbf{p})$$

其中 $\partial u'/\partial d$ 即 (A)——**图像梯度 × 投影-深度雅可比**的链式分解。两项相加即论文 Eq.(14)：

$$\sigma^2_{r_p}(\mathbf{p}, \boldsymbol{\xi}_{j·i}) := 2\sigma_I^2 + \Big( \frac{\partial I_j\big(\omega(\mathbf{p}, \boldsymbol{\xi}_{j·i})\big)}{\partial D_i(\mathbf{p})} \Big)^2 V_i(\mathbf{p}) \tag{14}$$

**"梯度大才可靠"的双面性**（这是 Eq.(12)-(14) 设计的要点）：

1. **半稠密像素门**：由 4.2 的雅可比链，像素对增量方程的贡献 $\propto \nabla I_j$（教程 (6.5) 的最外层因子）。梯度为零 ⟹ $\mathbf{J}_p = \mathbf{0}$，对优化零贡献却仍带光度噪声——所以 $\Omega_{D_i}$ 只取高梯度像素（教程 06.2 第三步：半稠密 = 梯度像素集，"无梯度 ⟹ 无信息"）。**梯度大才可观测**。
2. **方差归一化的反向调节**：式 (14) 表明，正是让像素"可观测"的那个梯度，也把深度噪声 $V_i(\mathbf{p})$ 放大进残差——新帧里 $\partial I_j/\partial d$ 大的像素，其残差对深度误差最敏感。Eq.(12) 逐像素除以 $\sigma^2_{r_p}$，等价于按 $1/\sigma^2_{r_p}$ 加权（对照 Eq.(8)-(9)：$w_p = 1/\sigma^2_{r_p}$ 的信息权重，教程 (8.2) 的 $\Lambda_k$）：深度图老、方差大的像素自动降权。**梯度大也更脆弱，按方差定价**。论文 Fig. 4 展示了不同运动下的差异：纯旋转时深度噪声不起作用（(A) 第一式为零）、$z$ 平移只影响图像中心的像素、$x$ 平移只影响有 $x$ 方向梯度的像素——RGB-D 方法深度方差近似恒定，单目直接法则必须逐像素区分。
3. **Huber 外点抑制**（Eq.(15)，逐字转录论文；它是经典 Huber $\rho(r) = \tfrac{r^2}{2},\ \delta|r| - \tfrac{\delta^2}{2}$ 的 $1/\delta$ 缩放版，在 $|r| = \delta$ 处同样连续）：

$$\|r^2\|_\delta := \begin{cases} \dfrac{r^2}{2\delta} & \text{if } |r| \le \delta \\[1ex] |r| - \dfrac{\delta}{2} & \text{otherwise.} \end{cases} \tag{15}$$

应用于**归一化后**的残差（Eq.(12) 的外层 $\|\cdot\|_\delta$）。IRLS 实现：Huber 的 IRLS 权重为 $|r| \le \delta$ 时 $w = 1/\sigma^2_{r_p}$、否则 $w = \delta/(|r|\,\sigma^2_{r_p})$（依据：$\rho'(r)/r$，标量 $1/\delta$ 常数并入 $\mathbf{W}$）——即 4.2 的 Eq.(9) 逐像素代入。最小化即"iteratively re-weighted Gauss-Newton（Sec. 2.2）"。

### 4.4 半稠密深度图与关键帧（论文 Eq.(16) + 深度迁移推导）

**关键帧选择**（Eq.(16)）：相机离当前关键帧太远时新建关键帧，判据是运动大小的加权阈值

$$\mathrm{dist}(\boldsymbol{\xi}_{j·i}) := \boldsymbol{\xi}_{j·i}^\top \mathbf{W}\, \boldsymbol{\xi}_{j·i}$$

（依据：$\mathbf{W}$ 对角、给旋转分量不同权重——旋转不产生可三角化的视差，需与平移分别计价。**尺度相对性**：每个关键帧深度图被缩放到平均逆深度为 1，故该阈值自动随当前场景尺度伸缩，"ensures sufficient possibilities for small-baseline stereo comparisons"。）

**深度图创建与反深度的尺度迁移**（论文 §3.4 "Depth Map Creation"：新关键帧深度图由投影上一关键帧的点初始化，做一次空间正则与离群剔除；随后**整图缩放到平均逆深度为 1，缩放因子直接并入 sim(3) 相机位姿**。论文未给该步独立式号，以下推导按其文字与 Eq.(3') 进行）：

- **单点迁移**：上一关键帧的点经跟踪位姿 warp，新帧逆深度就是 (A) 中 $\omega$ 的第三维 $d' = d/(k_3 + t_z d)$。特例纯前向平移（$\mathbf{R} = \mathbf{I}$，$k_3 = 1$）：$d' = d/(1 + t_z d)$——$d$ 的 Möbius 变换；小基线 $t_z d \ll 1$ 时 $d' \approx d(1 - t_z d) \approx d$（依据：$(1+u)^{-1}$ 的一阶展开）。这解释了为何迁移后还要重新正则/剔除：变换对 $d$ 非线性，误差随视差结构重排。
- **整图缩放为何只需改 sim(3) 位姿**：把关键帧 $i$ 的深度图整体乘 $c$（等价于该帧场景坐标除以 $c$，$X \to X/c$）。原 sim(3) 边 $\mathbf{S} = (s\mathbf{R}, \mathbf{t})$ 作用在新表示上：$s\mathbf{R}(X/c) + \mathbf{t} = (s/c)\mathbf{R}X + \mathbf{t}$——**只有尺度分量 $s \to s/c$ 变**，旋转、平移与几何全部不变（依据：标量提出）。所以"缩放深度图 + 把因子吸收进 sim(3) 位姿"是精确操作，不是近似——这正是选 Sim(3) 而非 SE(3) 作边的一个直接工程红利。

**深度图精化（滤波框架）**：非关键帧用于对当前关键帧做大量小基线立体比较。论文正文**未重复**滤波公式，明确沿用 [9]（Engel 等，ICCV 2013 半稠密 VO）的框架，此处按 [9] 的机制补齐并注明出处：每个高梯度像素持 $(d, V)$；每个立体观测的反深度观测值 $d_{obs}$ 由光度残差极小给出，其方差由 Eq.(11) **反用**得到——残差近极小处 $r_p \approx (\partial r_p/\partial d)(d - d_{obs})$，光度噪声 $\sigma_I$ 折算为 $\sigma^2_{d,obs} \approx \sigma_I^2 / (\partial r_p/\partial d)^2$（依据：把 (11) 的传播倒过来解）；随后高斯加权融合（依据：两高斯乘积的均值/方差公式，教程第 03 章 (3.10)–(3.11) 同型）：

$$d_{new} = \frac{V_{obs}\, d + V\, d_{obs}}{V + V_{obs}}, \qquad V_{new} = \frac{V\, V_{obs}}{V + V_{obs}}$$

**收敛性/偏度判据**（依 [9]；LSD-SLAM 论文文字引用为 "as described in [9]"）：观测只有落在当前估计的置信域内才融合（防止遮挡/外点污染单峰高斯假设）；每个假设带 *validity/converged* 标志，观测分布的**偏度**（一阶与二阶矩的偏离）用于检测多峰情形——偏度大说明假设不可信，丢弃而非平均。这与本精读 §4.3 的思想一脉相承：只在"高斯假设可信"的地方融合。

### 4.5 直接 sim(3) 对齐与回环约束（论文 Eq.(17)–(20)）

**总误差**（Eq.(17)）：关键帧-关键帧直接对齐在光度残差外加**深度残差**：

$$E(\boldsymbol{\xi}_{j·i}) := \sum_{\mathbf{p} \in \Omega'_{D_i}} \left\| \frac{r_p^2(\mathbf{p}, \boldsymbol{\xi}_{j·i})}{\sigma^2_{r_p}(\mathbf{p}, \boldsymbol{\xi}_{j·i})} + \frac{r_d^2(\mathbf{p}, \boldsymbol{\xi}_{j·i})}{\sigma^2_{r_d}(\mathbf{p}, \boldsymbol{\xi}_{j·i})} \right\|_\delta, \qquad r_d(\mathbf{p}, \boldsymbol{\xi}_{j·i}) := [\mathbf{p}']_3 - D_j([\mathbf{p}']_{1,2}) \tag{17}{,}(18)$$

其中 $\mathbf{p}' = \omega(\mathbf{p}, D_i(\mathbf{p}), \boldsymbol{\xi}_{j·i})$：$[\mathbf{p}']_3$ 是 warp 输出的目标帧逆深度 $d'$（4.1），$D_j([\mathbf{p}']_{1,2})$ 是关键帧 $j$ 深度图在投影像素处的存储值——$r_d$ 即"几何假设 vs 对方地图"的深度一致性残差。**方差**（Eq.(19)，由 Eq.(11) 对两个不确定输入逐项传播，依据：独立输入方差相加）：

$$\sigma^2_{r_d}(\mathbf{p}, \boldsymbol{\xi}_{j·i}) := V_j([\mathbf{p}']_{1,2}) \Big( \frac{\partial r_d}{\partial D_j([\mathbf{p}']_{1,2})} \Big)^2 + V_i(\mathbf{p}) \Big( \frac{\partial r_d}{\partial D_i(\mathbf{p})} \Big)^2$$

第一项来自 $j$ 方深度图值、第二项来自 $i$ 方深度经 warp 的传播（$\partial d'/\partial d = k_3/(k_3+t_z d)^2$，见 (A)；论文脚注 1：深度图的空间梯度近似取零以省算——即忽略采样位置移动引起的 $D_j$ 变化）。**为什么 sim(3) 跟踪必须加 $r_d$**（论文原话："for tracking on sim(3), the inclusion of the depth error is *required* as the photometric error alone does not constrain the scale"），推导：sim(3) 下 $\mathbf{S} = (s\mathbf{R}, \mathbf{t})$ 作用给 $u' = (s k_1 + t_x d)/(s k_3 + t_z d)$（由 Eq.(3') 把 $\mathbf{R}(x,y,1)^\top$ 换成 $s\mathbf{R}(x,y,1)^\top$），对尺度求导（依据：商法则）得 $\partial u'/\partial s \propto d(t_z k_1 - t_x k_3)$——**基线 $\mathbf{t} \to \mathbf{0}$ 时光度残差对尺度失聪**（视差消失）；而 $r_d$ 经 $\partial d'/\partial s = -d\,k_3/(s k_3 + t_z d)^2 \neq 0$ 在零基线下仍约束尺度。关键帧间小基线正是本系统的常态，故 $r_d$ 不可省。

**双向往返校验**（Eq.(20)，逐字转录；$\mathcal{K}_{j_k}$ 为回环候选）：对每个候选**独立**跟踪出 $\boldsymbol{\xi}_{j_k i}$ 与 $\boldsymbol{\xi}_{i j_k}$（互为逆的两个估计），要求

$$e(\boldsymbol{\xi}_{j_k i}, \boldsymbol{\xi}_{i j_k}) := (\boldsymbol{\xi}_{j_k i} \circ \boldsymbol{\xi}_{i j_k})^T \Big( \boldsymbol{\Sigma}_{j_k i} + \mathrm{Adj}_{j_k i}\, \boldsymbol{\Sigma}_{i j_k}\, \mathrm{Adj}_{j_k i}^T \Big)^{-1} (\boldsymbol{\xi}_{j_k i} \circ \boldsymbol{\xi}_{i j_k}) \tag{20}$$

足够小才入图。推导逻辑：两次独立估计都无偏时，复合 $\boldsymbol{\xi}_{j_k i} \circ \boldsymbol{\xi}_{i j_k} \approx$ 恒等元；其偏差的协方差是两者之和——$\boldsymbol{\Sigma}_{i j_k}$ 定义在与 $\boldsymbol{\xi}_{j_k i}$ 不同的切空间，须经**伴随** $\mathrm{Adj}_{j_k i}$ 搬运（依据：Eq.(11) 穿过复合映射的雅可比即伴随，论文原话 "the adjoint $\mathrm{Adj}_{j_k i}$ is used to transform $\boldsymbol{\Sigma}_{i j_k}$ into the correct tangent space"）；马氏范数即一致性检验统计量（教程第 09 章 09.2 的"几何校验"思想，这里换成概率版）。此外论文 §3.5 给出**扩大收敛半径**的两个手段：ESM 二阶最小化（[3]，不提精度、只扩半径）与极低分辨率由粗到精（起点低至 $20 \times 15$ 像素）——§4.3 实验验证它们只扩大收敛半径、不改变收敛后的精度。

### 4.6 Sim(3) 位姿图优化（论文 Eq.(21)）

**为什么必须 Sim(3)**：每个关键帧深度图被归一到平均逆深度 1（§4.4），关键帧两两之间的真实相对尺度信息只能由**边携带**；单目直接法沿轨迹的尺度漂移表现为相邻段之间比例不一致，SE(3) 的 6 自由度无法表达"整段地图缩放"（教程第 09 章 (9.4)：单目地图可漂自由度是 7 = 3 平移 + 3 旋转 + 1 尺度，引 Strasdat [23] 的结论）；且 Eq.(17) 的深度残差恰好把边的尺度分量与两侧深度图钉在一起——位姿图由此成为尺度漂移的**显式载体与修正通道**。

**目标函数**（Eq.(21)，逐字转录；$\boldsymbol{\xi}_{Wi}$ 为待优化关键帧位姿，边集 $E$）：

$$E(\boldsymbol{\xi}_{W1} \dots \boldsymbol{\xi}_{WN}) := \sum_{(\boldsymbol{\xi}_{j·i}, \boldsymbol{\Sigma}_{j·i}) \in E} (\boldsymbol{\xi}_{j·i} \circ \boldsymbol{\xi}_{W1}^{-1} \circ \boldsymbol{\xi}_{Wj})^T\, \boldsymbol{\Sigma}_{j·i}^{-1}\, (\boldsymbol{\xi}_{j·i} \circ \boldsymbol{\xi}_{W1}^{-1} \circ \boldsymbol{\xi}_{Wj}) \tag{21}$$

读法：每条边把"直接对齐测得的相对变换 $\boldsymbol{\xi}_{j·i}$"与"优化后世界位姿按左复合约定组合出的相对变换"在切空间中作差，按该边协方差 $\boldsymbol{\Sigma}_{j·i}$ 马氏加权；$\boldsymbol{\xi}_{W1}$ 显式出现在残差中，正是"世界系由第一个关键帧钉住"的规范固定（gauge；对照教程第 09 章 (9.7)-(9.8) 的 SE(3) 位姿图目标函数——同一"切空间残差 + 马氏权重"结构，多一维尺度分量，且流形求解器一致：第 08 章 (8.9) 的流形更新 + (8.5) 型正规方程）。论文提醒（第 6 页）：左复合约定须与优化框架一致（g2o 默认右复合，"this has to be taken into account"）。优化在后台持续进行（§3.6），边集含里程计边、近邻 sim(3) 边与通过 Eq.(20) 校验的回环边。

## 5. 实验与结果解读

- **TUM RGB-D 基准定量结果**（Fig. 9，单目方法用第一帧深度图引导初始化）：LSD-SLAM 在 fr2/desk 为 4.52 cm RMSE（116 关键帧）、fr2/xyz 为 1.47 cm（38 关键帧）、两个 TUM 仿真序列 sim/desk 0.04 cm（39）、sim/slownmo 0.35 cm（12）；对比对象为 keypoint 单目 SLAM [15]、直接法 RGB-D [14]、keypoint RGB-D [7]——[14][7] 用了传感器深度而 LSD-SLAM 没有。论文同时提醒：该基准含快速旋转、强运动模糊与 rolling shutter，对单目 SLAM 是跟踪瓶颈（§4.2），故数字应按此背景解读。
- **大轨迹与尺度变化**（定性，§4.1 + Fig. 7/8 + §5）：约 500 m、6 分钟的手持轨迹在大回环前后（Fig. 7）与一条"平均逆深度从小于 20 cm 到大于 10 m"的大尺度变化轨迹（Fig. 8）上，回环后点云几何一致闭合；Fig. 2/3 展示半稠密深度图按方差阈值累积成点云——阈值放宽点云更密但混入更多噪声，**密度与噪声可由方差阈值连续调节**，这是带概率深度的建图特有性质（特征点法地图没有这种"密度旋钮"）。
- **收敛半径研究**（Fig. 10，§4.3）：以恒等初始化把整段序列对齐到固定帧，统计成功率与精度——ESM 与更多金字塔层显著扩大收敛半径；但**一旦收敛，各配置精度几乎无差**（"if tracking converges, it almost always converges to the same minimum"）——非凸直接法里"收敛域"与"收敛精度"是两个独立指标。
- **计算平台**：全程 CPU 实时（摘要与 §5）；对照当时稠密直接法需 GPU（§1）。半稠密 + 关键帧图是"信息量/算力"折中的关键：梯度像素只是全图一小部分，但按 §4.3 的论证已含几乎全部可观测信息。

## 6. 局限与后续影响

**局限**（论文自述 + 结构性分析）：

1. **无全局 BA**：地图优化只是关键帧**位姿图**（Eq.(21)），深度图在关键帧入图后不再参与全局联合优化（"once a keyframe is replaced as tracking reference... it will not be refined further"，§3.1）——路标级几何误差无法像 BA 那样被重线性化吸收，两次回环之间尺度漂移仍会累积（对照教程第 09 章 09.3：位姿图 ≈ 边缘化路标后的等效问题，精度略低于全局 BA）。DSO 的滑窗光度 BA 正是对这一缺口的结构性回应。
2. **光度模型极简**：假设灰度恒常，不建模响应函数、渐晕与曝光时间（对照 DSO 论文 Eq.(2)-(3) 的完整标定）——曝光/自动增益变化只能靠 Huber 与方差归一化"容忍"（教程 06.3 ③ 的"回避"路线），快速曝光变化时易跟踪失败。
3. **对快速运动敏感**：光度误差非凸、收敛半径有限（§3.5、§4.3）；TUM 实验明确列出运动模糊与 rolling shutter 为失效源（§4.2），需靠极粗层金字塔 + ESM 扩半径。
4. **半稠密边界**：弱纹理区域内部（白墙中心）依旧无深度（梯度门的天性，教程 06.1 第五步）；初始化依赖"随机深度 + 大方差"的自举（§3.1），论文自认其收敛性未做系统评估。

**后续影响**： Sim(3) 关键帧位姿图成为单目直接法系统的标准配置； 深度不确定性进跟踪的方差归一化残差被 DSO 继承为光度/几何加权体系； 与 [9] 一起确立"半稠密 = 梯度像素 + 概率深度滤波"的范式；ORB-SLAM 系（特征路线）与 LSD-SLAM/DSO（直接路线）共同构成现代单目 SLAM 的两条主线（姊妹篇 PTAM 精读 §6 的谱系图）。

## 7. 与本项目对照

- **教程章节**：[第 06 章](../06_视觉里程计-ii直接法.md)——(6.4) 光度残差即本文 Eq.(5)/(13)（教程取无方差加权特例）、(6.5) 三因子雅可比 = 本文 §4.3 "图像梯度 × 投影-深度雅可比"的同一条链（第 05 章 (5.11) 提供几何因子）、(6.6)-(6.7) 的仿射亮度处理是 DSO 路线（本文未做，见 §6.2）；[第 08 章](../08_后端-ii图优化与-ba.md)——(8.5)/(8.9) 是 Eq.(6)-(7) 的教程原型、(8.17) 的"Schur 补即边缘化"是理解"位姿图 = 精简后端"的钥匙；[第 09 章](../09_回环检测.md)——(9.4) Sim(3) 定义与 7 自由度论证、(9.7)-(9.8) 位姿图目标函数与 Eq.(21) 同构。
- **代码**：[projects/slam/direct/direct.py](../../../projects/slam/direct/direct.py) 实现了本文**跟踪核心**的教学版：`select_gradient_pixels`（半稠密梯度门，对应 $\Omega_{D_i}$ 的选取逻辑）、`photometric_residuals`（Eq.(13) 的 $r_p$，教程 (6.4)）、`photometric_jacobian` + `projection_jacobian`（(6.5)/(5.11) 三因子链，有限差分验证）、`semi_dense_align`（复用 `core.solver.gauss_newton` 在左扰动坐标上的 G-N 流形迭代，对应 Eq.(6)-(7)）。
- **刻意差距**（教学取舍，均已在教程覆盖理论）： Eq.(12)/(14) 的方差归一化与深度不确定性传播——模块的深度图是固定真值（合成场景无深度滤波可言），方差权重与深度估计框架属系统层，教程 06.3 ②③ 已做概念覆盖，深度滤波的完整实现见上文 §4.4 的推导； sim(3) 跟踪与 Eq.(21) 位姿图——`direct/` 只做 SE(3) 帧间对齐，Sim(3) 数学见教程 (9.4)，位姿图优化的教学实现在 `bowloop` 模块（[ORB-SLAM 精读 §7](./ORB-SLAM_TRO2015.md)）； Huber 核未接入 `semi_dense_align`（教程 (6.4) 为纯最小二乘）；鲁棒核的 IRLS 教学实现在 `core.solver.huber_weights`，被 [photoba](../../../projects/slam/photoba/) 使用（[DSO 精读 §7](./DSO_PAMI2018.md)）。
- **健康值**：`tests/test_direct.py` 的断言是解析恒等式级（雅可比对有限差分），无端到端 ATE 指标——与 [METRICS.md](../../../projects/slam/METRICS.md) §2.3 未单列该模块一致；端到端精度断言由 `photoba` 模块承接（DSO 精读 §7 引用其实测数字）。

## 配套阅读

- 本文 PDF：[LSD-SLAM（Engel, Schöps & Cremers, ECCV 2014）](../../../papers/slam/classics/LSD-SLAM_ECCV2014_EngelSchopsCremers.pdf)（本精读所有式号以此 PDF 为准；深度滤波细节见其引用的 [9] Engel et al., ICCV 2013，本仓库未收录）
- 姊妹精读：[PTAM（ISMAR 2007）](./PTAM_ISMAR2007.md)（§4.5 澄清了"逐像素深度滤波"属于本文而非 PTAM）｜[DSO（PAMI 2018）](./DSO_PAMI2018.md)（把本文的"回避曝光"换成"建模曝光"，滑窗光度 BA）｜[ORB-SLAM（T-RO 2015）](./ORB-SLAM_TRO2015.md)｜[ORB-SLAM2（T-RO 2016）](./ORB-SLAM2_TRO2016.md)｜[FastSLAM（AAAI 2002）](./FastSLAM_AAAI2002.md)
- 教程：[第 05 章](../05_视觉里程计-i特征点法.md)（雅可比同构的来源）｜[第 06 章](../06_视觉里程计-ii直接法.md)｜[第 08 章](../08_后端-ii图优化与-ba.md)｜[第 09 章](../09_回环检测.md)（Sim(3) 与位姿图）
- 代码：[projects/slam/direct/](../../../projects/slam/direct/)（本文跟踪核心）｜[projects/slam/photoba/](../../../projects/slam/photoba/)（DSO 式滑窗，深度参与优化的对照）｜[projects/slam/METRICS.md](../../../projects/slam/METRICS.md)
- 主题导航：[tutorials/slam/README.md](../README.md) ｜ [OVERVIEW.md](../OVERVIEW.md)

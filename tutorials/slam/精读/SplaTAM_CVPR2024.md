# 论文精读｜SplaTAM：把 3D 高斯当点云地图"Splat, Track & Map"（CVPR 2024）

> **PDF**：[papers/slam/frontier/arXiv-2312.02126_SplaTAM.pdf](../../../papers/slam/frontier/arXiv-2312.02126_SplaTAM.pdf)（arXiv v3，2024-04-16，11 页含补充材料，本精读所有式号以此 PDF 为准）｜ **教程**：[第 10 章 §10.4](../10_建图与系统实战.md)（前沿深读扩展）｜ **3DGS 本体**：[3D 重建教程第 08 章](../../3d_reconstruction/08_3D高斯泼溅.md)（已核实后回引）｜ **代码**：无（前沿路线说明）

## 1. 论文信息与一句话贡献

- **题目**：SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM
- **作者**：Nikhil Keetha、Jay Karhade、Krishna Murthy Jatavallabhula、Gengshan Yang、Sebastian Scherer、Deva Ramanan、Jonathon Luiten（CMU / MIT）
- **发表**：CVPR 2024；arXiv:2312.02126（项目页 spla-tam.github.io，开源）
- **一句话贡献**：**首次用显式体积表示——3D 各向同性高斯集合——做稠密 RGB-D SLAM**：把高斯地图当"可微泼溅的点云地图"，每帧循环三步——**Splat**（可微渲染颜色/深度/轮廓三张图）、**Track**（在过饱和轮廓门控下对渲染-观测深度+光度残差做梯度优化位姿）、**Map**（silhouette 与深度残差驱动的致密化 + 逐高斯更新）；四个基准上相机跟踪取得当时 SOTA（ScanNet++ 平均 ATE 1.2 cm，对比 Point-SLAM 343.8 cm 与 ORB-SLAM3 158.2 cm 的失效），渲染 400 FPS（876×584）。

## 2. 问题与动机

**要解决什么**（§1–2）：稠密 SLAM 的地图表示之争。**显式**表示（点/surfel/面片）便于跟踪与编辑，但跟踪"relies crucially on the availability of rich 3D geometric features"（§1），且只可靠解释已观测部分，难以处理未观测/新视角；**隐式/体积**表示（NeRF 系 SLAM）光度质量高、可微，但"computationally inefficient, not easy to edit, do not model spatial geometry explicitly, and suffer from catastrophic forgetting"（§1 原话）——且体积光线采样使优化退化为稀疏像素采样（Point-SLAM 每次只用 ~200/1000 像素，见其运行时对比 §5），光度误差不完整。论文的核心问题（§1 加粗原话）："**How can an explicit volumetric representation benefit the design of a SLAM solution?**"

**答案的四个好处**（§1 列表，原文语义）： **快速渲染与稠密优化**——3DGS 光栅化高达 400 FPS，使"逐像素稠密光度误差"第一次在 SLAM 里可行（对比隐式法的稀疏采样）；为此做了两处改动：去掉视角相关外观、改用**各向同性**高斯； **显式空间延展（map with explicit spatial extend）**——渲染一张 silhouette 掩码即得当前视角的"地图边界"（spatial frontier）：哪里有累积不透明度哪里就是已建图区域，跟踪只在该区域内做，相机运动还能据此判别"已见/未见"； **场景参数直接优化**——高斯有物理 3D 位置/颜色/尺寸，光度损失到参数间的梯度"almost linear (projective)"，相机运动也能拿到直接梯度（隐式/体积表示要穿过非线性网络层）； **结构化地图扩展**——向未见区域加高斯即可扩展地图，同时保持高保真渲染。

**与两族先行者的关系**（§2 "Implicit scene representations"/"Traditional approaches" 段归纳）：对 iMAP→NICE-SLAM→Point-SLAM 这条隐式/神经点云线，SplaTAM 保留其"逐像素稠密光度损失 + experience replay"的经验，把"体渲染"换成"光栅化"——"volumetric ray sampling greatly limits its efficiency, thereby resorting to optimization over a sparse set of pixels, as opposed to per-pixel dense photometric error"（§2 原话）；对 surfel 传统（§2 引 Keller et al. 的 surfel SLAM），差别在"现代可微光栅化让深度不连续处的梯度可用"（其引文 [50]），而 2D surfel 是不连续原语、需要精心的正则防洞。

## 3. 方法总览

```
RGB-D 帧流（已知内参 K、稠密深度；第 t+1 帧）
   │
   ▼  【① Camera Tracking（其式 (7)(8)）】
   ├─ 恒速初始化：E_{t+1} = E_t + (E_t − E_{t−1})（相机中心 + 四元数空间）
   ├─ 用地图 G_{1:t} 渲染 D(p), C(p), S(p)（其式 (4)(2)(5)）
   ├─ 只在 S(p) > 0.99 的"过饱和"像素上，最小化 L1(D) + 0.5·L1(C)（其式 (8)）
   │    ——高斯参数冻结，只有位姿 E_{t+1} 走梯度（Adam）
   ▼
   ▼  【② Gaussian Densification（其式 (9)）】
   │  致密化掩码 M(p) = (S(p) < 0.5) + (D_GT < D(p))·(L1(D) > λ·MDE)，λ=50
   │  命中像素按首帧同款流程新增高斯（中心=反投影、opacity=0.5、半径=D_GT/f）
   ▼
   ▼  【③ Map Update】
   ├─ warm-start：从最近地图出发；选 k 个关键帧（当前帧 + 最近关键帧 + k−2 个
   │    重叠度最高者；重叠 = 当前帧点云落入关键帧视锥的点数）
   ├─ 位姿冻结，优化高斯参数：L1(D) + L1(C) + SSIM(C)；剔除近零不透明/过大高斯
   ▼  回到 ①（首帧特殊：位姿=单位阵、silhouette 为空 → 全图像素初始化高斯）
```

与 [GS-SLAM](./GS-SLAM_CVPR2024.md) 同为 RGB-D + 3DGS，路线气质相反：SplaTAM 是**跟踪驱动**（先把位姿锁死在已有地图上，地图被动生长），GS-SLAM 是**建图驱动**（扩展/清洗地图为主，跟踪用纯光度）。跟踪端在梯度优化下逐像素用全部 ~120 万像素（Table 6 讨论："we differentiably render 3 orders of magnitude more pixels"），这正是"显式体积表示"红利——对照 [LSD-SLAM](./LSD-SLAM_ECCV2014.md)/[直接法](../06_视觉里程计-ii直接法.md) 的梯度像素选取，这里无需选点。

**三步循环的次序约束**（§3 "SLAM System" 原文归纳）：先 Track 后 Densify 后 Map——致密化要用"本帧跟踪好的位姿"来决定新高斯的位置（否则新点带着错位姿进地图），建图又要在新点就位后才开始（否则旧地图把新点当离群梯度拉扯）；首帧是唯一例外（位姿恒为单位阵、silhouette 为空，跳过跟踪、全图初始化）。这种"次序即正确性"的交替结构与 [PTAM](./PTAM_ISMAR2007.md) 以来的跟踪/建图分离一脉相承，但这里两步共享同一个可微渲染器，边界只是"谁被冻结"。

**关键帧与优化窗口**（§3 "Gaussian Map Updating" 原文归纳）：每 $n$ 帧存一个关键帧；每次建图取 $k$ 帧 = 当前帧 + 最近关键帧 + $k-2$ 个**重叠度最高**的更早关键帧（重叠度 = 当前帧深度图反投影点云落入该关键帧视锥的点数）。这个选法服务致密化：新增高斯只可能出现在与当前帧重叠的旧视角里被"看到"，其余关键帧对它们的梯度贡献近零——是"experience replay"思想的几何化版本（对照 NICE-SLAM 系的均匀回放）。

## 4. 关键公式推导（式号按论文 PDF 原文 (1)–(9)；补充材料 Table S1）

### 4.0 符号表

| 符号 | 含义 |
|---|---|
| $\mathcal{G} = \{(\boldsymbol{\mu}_i, r_i, o_i, \mathbf{c}_i)\}$ | 高斯地图：中心 $\boldsymbol{\mu}_i\in\mathbb{R}^3$、半径 $r_i$、不透明度 $o_i\in(0,1]$、视角无关颜色 $\mathbf{c}_i\in\mathbb{R}^3$——每颗高斯**8 个参数** |
| $f_i(\mathbf{x})$ | 各向同性高斯的体积贡献（式 (1)） |
| $E_t$ | 第 $t$ 帧外参（世界→相机，作用于齐次点；跟踪变量） |
| $\mathbf{K}, f$ | 内参矩阵 / 焦距（[第 04 章 (4.4)](../04_相机模型与特征提取.md)） |
| $\mathbf{p}=(u,v)$ | 像素坐标 |
| $\boldsymbol{\mu}^{2\mathrm{D}}, r^{2\mathrm{D}}$ | 高斯 2D 投影的中心 / 半径（式 (3)） |
| $d_i$ | 第 $i$ 高斯中心在相机系深度 $(E_t\boldsymbol{\mu}_i)_z$（式 (3)） |
| $C(\mathbf{p}), D(\mathbf{p}), S(\mathbf{p})$ | 渲染颜色 / 深度 / **silhouette**（式 (2)(4)(5)） |
| $D_{\mathrm{GT}}(\mathbf{p})$ | 传感器深度 |
| $L_1(\cdot)$ | 逐像素 L1 残差（跟踪与建图共用） |
| $\lambda, \mathrm{MDE}$ | 致密化阈值系数（取 50）/ 深度误差中位数（式 (9)） |
| $n$ | 对某像素有贡献的高斯数（排序后 front-to-back） |
| $w_i(\mathbf{p})$ | 像素 $\mathbf{p}$ 上第 $i$ 高斯的混合权重 $= f_i(\mathbf{p})\prod_{j<i}(1-f_j(\mathbf{p}))$（本文分析引入的记号） |

（记号注：论文把 2D 投影半径记作 $r^{2\mathrm{D}}$、把式 (1) 的 3D 半径简写作 $r$；$E_t$ 在论文中是作用在**齐次点**上的外参矩阵 $-$ "the extrinsic matrix capturing the rotation and translation of the camera at frame $t$"，与 [第 04 章 (4.5)](../04_相机模型与特征提取.md) 的 $T_{cw}$ 同义。）

### 4.1 高斯地图表示：球对称的极端简化（式 (1)）

第 $i$ 颗高斯对点 $\mathbf{x}$ 的"影响"定义为（式 (1)）：

$$f_i(\mathbf{x}) = o_i\,\exp\Big(-\frac{\|\mathbf{x}-\boldsymbol{\mu}_i\|^2}{2r^2}\Big) \tag{1}$$

对照 [3D 重建教程第 08 章 (8.1)](../../3d_reconstruction/08_3D高斯泼溅.md) 的各向异性形式 $\exp(-\frac12(\mathbf{x}-\boldsymbol{\mu})^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}))$：把 $\Sigma^{-1}$ 锁成 $\frac{1}{r^2}I$（球对称）并乘上不透明度 $o_i$。**为什么敢这么砍**（§3 "Gaussian Map Representation" + 消融 Table S1）：跟踪靠的是渲染深度/颜色的位置一致性，各向同性已足够表达"以 $\boldsymbol{\mu}$ 为中心的一团表面物质"；代价是丢掉表面方向性，收益是参数骤减（8 个/颗）、优化病态更少、渲染更快——Table S1 显示 SLAM 精度差异"marginal"（各向异性 ATE 0.55 vs 各向同性 0.57，训练视角 PSNR 28.11 vs 27.82），而各向同性只耗 57.5% 内存、93.3% 时间。颜色用视角无关常值（无球谐），因 RGB-D 跟踪主要吃深度（§5 Table 4），视角效应是二阶的。

### 4.2 可微泼溅渲染：颜色、深度、silhouette（式 (2)–(5)）

**2D 投影**（式 (3)）：给定位姿 $E_t$，高斯中心投到像素 $\boldsymbol{\mu}^{2\mathrm{D}} = \mathbf{K}\,E_t\boldsymbol{\mu}/d$，半径投为 $r^{2\mathrm{D}} = fr/d$，其中 $d = (E_t\boldsymbol{\mu})_z$。逐步推导：$E_t\boldsymbol{\mu}$ 是中心在相机系的齐次坐标，$\mathbf{K}\,E_t\boldsymbol{\mu}$ 的齐次除法即 [第 04 章 (4.2)–(4.3)](../04_相机模型与特征提取.md) 的透视投影（$u = f_x X/Z + c_x$ 型）；半径按**相似三角形**：以光轴垂直距离 $d$ 处、物理半径 $r$ 的球，投影直径与焦距平面之比 $r^{2\mathrm{D}}/f = r/d$，即 $r^{2\mathrm{D}} = fr/d$——球对称使投影仍是圆（各向异性则要 EWA 仿射近似，[3D 重建教程第 08 章 (8.5)–(8.8)](../../3d_reconstruction/08_3D高斯泼溅.md)，本文整条协方差投影链被一行相似三角形替代）。**渲染**：把高斯按深度排序后逐像素 alpha 合成，第 $i$ 个高斯的权重为 $w_i(\mathbf{p}) = f_i(\mathbf{p})\prod_{j=1}^{i-1}(1-f_j(\mathbf{p}))$（能到达 × 在此处被挡），三张图：

$$C(\mathbf{p}) = \sum_{i=1}^{n}\mathbf{c}_i\,f_i(\mathbf{p})\prod_{j=1}^{i-1}(1-f_j(\mathbf{p})) \tag{2}$$

$$D(\mathbf{p}) = \sum_{i=1}^{n}d_i\,f_i(\mathbf{p})\prod_{j=1}^{i-1}(1-f_j(\mathbf{p})) \tag{4}$$

$$S(\mathbf{p}) = \sum_{i=1}^{n}f_i(\mathbf{p})\prod_{j=1}^{i-1}(1-f_j(\mathbf{p})) \tag{5}$$

（与 [3D 重建教程第 08 章 (8.10)](../../3d_reconstruction/08_3D高斯泼溅.md) 的 alpha 混合同构，差异在 $\alpha_i$ 的来源：解析 2D 球对称高斯求值 vs 各向异性 (8.9)；与 NeRF 离散渲染的逐项对照见 [NeRF 教程第 03 章 §3.5](../../nerf/03_数学原理与渲染方程.md)。$S(\mathbf{p})$ 即**累积不透明度** $\sum_i w_i$——"该像素接了多少高斯能量"的归一化前指标，是全系统的不确定性代理。）

**深度渲染的语义**（注意与体渲染的差异）：$D(\mathbf{p})$ 是各高斯**中心深度** $d_i$ 的混合加权平均，不是"期望光线终止距离"（对比 [NeRF-SLAM 精读](./NeRF-SLAM_ICRA2023.md) 式 (5)）——因为高斯是表面物质团而非介质密度，中心即几何载体；$\mathrm{RGB\text{-}D}$ 监督（式 (8)(9)）因此直接约束高斯中心的位置，这正是"显式参数、近线性梯度"（§1 好处 3）的来源。归一性：$S(\mathbf{p}) \le 1$ 恒成立（透射连乘的单调性），$S$ 越接近 1 该像素的混合越"完整"。

### 4.3 初始化与恒速先验（式 (6)(7)）

**首帧初始化**：位姿取单位阵；silhouette 为空 ⟹ 全部像素初始化（跟踪步跳过）。每个像素新增一颗高斯：颜色 = 像素色、中心 = 深度反投影点、不透明度 0.5、**半径按"投影后恰为一个像素"反解**（式 (6)）：

$$r = \frac{D_{\mathrm{GT}}}{f} \tag{6}$$

（推导：把 $r^{2\mathrm{D}} = fr/d = 1$（一个像素）反解出 $r = d/f$，取 $d = D_{\mathrm{GT}}$——依据：式 (3) 的相似三角形关系。）**恒速先验**（式 (7)）：新帧位姿初始化为

$$E_{t+1} = E_t + \big(E_t - E_{t-1}\big) \tag{7}$$

（在"相机中心 + 四元数"参数空间做矢量外推，§3 "Camera Tracking" 原文；即一阶运动模型 $\delta_{t} = E_t - E_{t-1}$ 延续——与 [PTAM](./PTAM_ISMAR2007.md) 的恒速模型、[第 05 章](../05_视觉里程计-i特征点法.md) 的运动先验同思路，只是这里的"加法"发生在所选的欧拉参数空间而非 $SE(3)$ 流形，靠后续梯度迭代修正。）

### 4.4 跟踪损失与"过饱和"silhouette 门控（式 (8)）

$$L_t = \sum_{\mathbf{p}}\Big(S(\mathbf{p}) > 0.99\Big)\Big(L_1\big(D(\mathbf{p})\big) + 0.5\,L_1\big(C(\mathbf{p})\big)\Big) \tag{8}$$

（$(\cdot)$ 为指示函数。）三个设计点（式 (8) 后原文逐条）： **深度为主、颜色减半**——经验观察：渲染对位姿的梯度范数 $C(\mathbf{p})\in[0.01,0.03]$、$D(\mathbf{p})\in[0.002,0.006]$，深度信号更小需保满权，颜色降 0.5 防喧宾夺主； **门控 $S(\mathbf{p}) > 0.99$**——只在"被地图高斯几乎完全覆盖"（过饱和，well-optimized）的像素上计算损失：新帧常含地图里**没有**的内容（epistemic uncertainty 语义），拿这些像素对位姿求梯度等于拿幻觉当真值；这是 silhouette 的第一个用途； **L1 的良定义性**——像素无传感器深度时 $L_1$ 取 0，天然处理深度缺失，无需显式掩码。位姿参数冻结高斯、单走 Adam（"keeping the Gaussian parameters fixed"，§3）；**无 coarse-to-fine**（核对结论：全文 grep 无 "coarse"，分辨率单一的稠密直接优化正是卖点）。

**与滤波/非线性最小二乘前端的结构对照**（分析）：式 (7)+(8) 的组合等价于"运动模型预测 + 单帧观测修正"的贝叶斯两步（[第 03 章 (3.6)](../03_概率状态估计基础.md) 的预测-更新骨架），只是"协方差"由 silhouette 门控以集合成员的方式表达（信 = $S>0.99$，不信 = 否），且修正步不是一次 G-N 解而是数十次 Adam 迭代（Table 6：每帧 40 迭代）。没有边缘协方差、没有马氏加权、没有信息矩阵——鲁棒性全部外包给"渲染 vs 观测"的逐像素 L1 与门控掩码；代价是位姿不确定度不量化，无法像 [VINS-Mono](./VINS-Mono_RAM2018.md) 那样做边缘化/滑窗先验传递。

### 4.5 深度残差对位姿扰动的雅可比——与点到面 ICP 的关系（解读推导）

论文不写显式雅可比（走自动微分）；为建立与传统配准的联系，把深度项的一阶结构推出来（**解读性推导，非论文原式**）。取位姿左扰动 $E_t \leftarrow \mathrm{Exp}(\delta\boldsymbol{\xi})E_t$（$\delta\boldsymbol{\xi} = [\mathbf{v};\boldsymbol{\omega}]$，[第 02 章](../02_三维刚体运动旋转与位姿.md) 的 $SE(3)$ 扰动、[第 08 章 (8.9)](../08_后端-ii图优化与-ba.md) 的流形更新），高斯中心的相机系坐标一阶变化为

$$\mathbf{X}_c' = \mathrm{Exp}(\delta\boldsymbol{\xi})\,\mathbf{X}_c \approx \mathbf{X}_c + \boldsymbol{\omega}\times\mathbf{X}_c + \mathbf{v}$$

（依据：小扰动下 $SE(3)$ 指数映射对点的一阶作用，第 02 章刚体运动学。）其深度分量的雅可比：

$$\frac{\partial d_i'}{\partial \delta\boldsymbol{\xi}} = \Big(0,\ 0,\ 1,\ \underbrace{X_y,\ -X_x,\ 0}_{\partial(\boldsymbol{\omega}\times\mathbf{X}_c)_z/\partial\boldsymbol{\omega}}\Big),\qquad (\boldsymbol{\omega}\times\mathbf{X}_c)_z = \omega_x X_y - \omega_y X_x$$

（依据：叉积分量展开。）渲染深度 $D(\mathbf{p}) = \sum_i w_i d_i$ 的位姿梯度含两部分：$\sum_i w_i\,\partial d_i/\partial\delta\boldsymbol{\xi}$（**几何项**：深度加权中心 $\bar{\mathbf{X}}$ 产生的、形如"以相机 $z$ 轴为平面法向的点到面残差"——对照 [精读/LOAM](./LOAM_RSS2014.md) §4.4 的点到面残差 $\mathbf{n}^\top\mathbf{X}$：LOAM 的 $\mathbf{n}$ 是局部平面拟合的真法向，这里被"逐像素深度残差 + 全图联合"隐式替换为相机光轴方向）与 $\sum_i d_i\,\partial w_i/\partial\delta\boldsymbol{\xi}$（**权重项**：对应关系本身随位姿可微更新——固定对应 ICP 没有的项）。两点结论： 每像素深度残差 ≈ 法向取光轴方向的点到面 ICP，全图逐像素求和后横向/旋转自由度由深度图的**形状变化**提供——这正是式 (8) 深度主导可跟踪的原因，也是仅深度在纯旋转/无纹理墙面受限的根源（§5 Table 4 的 86.03 cm 失效）； GS-SLAM 忽略 $\partial\Sigma'/\partial\mathbf{P}$ 的取舍在此同型：主导信息在 $\partial d/\partial\boldsymbol{\xi}$ 与 $\partial\boldsymbol{\mu}^{2\mathrm{D}}/\partial\boldsymbol{\xi}$（投影雅可比，[第 05 章 (5.11)](../05_视觉里程计-i特征点法.md) 的同款），协方差回传是高阶修正。

### 4.6 致密化与建图更新（式 (9)）

**致密化掩码**（式 (9)）——silhouette 的第二个用途：

$$M(\mathbf{p}) = \big(S(\mathbf{p}) < 0.5\big) + \big(D_{\mathrm{GT}}(\mathbf{p}) < D(\mathbf{p})\big)\Big(L_1\big(D(\mathbf{p})\big) > \lambda\,\mathrm{MDE}\Big),\qquad \lambda = 50 \tag{9}$$

两类命中像素（式 (9) 后原文）： $S < 0.5$——地图没盖住（新观测区域，与跟踪门控 $S>0.99$ 恰为互补区间）； $D_{\mathrm{GT}} < D$ 且 $L_1(D) > \lambda\cdot\mathrm{MDE}$——**传感器深度在渲染深度之前**（有新几何"插到"当前地图前面）且误差远超中位数（$\lambda=50$ 经验值）——只敢在"新几何在前"时加点，因为"观测在后"更可能是噪声而非新表面。命中像素按首帧同款流程（式 (6)）新增高斯。**两个门限的非对称性**（分析）：跟踪门控 $S>0.99$ 取"几乎全信"，致密化 $S<0.5$ 取"几乎不信"——中间带（$0.5\le S\le 0.99$）的像素既不参与跟踪也不致密化，对应"刚建、尚未收敛"的区域：跟踪等它收敛（下帧再说），建图等证据充分（Table 5 显示把跟踪门限放宽到 0.5 会付出 5× 代价）。**Map Update**（§3 "Gaussian Map Updating"）：冻结位姿、warm-start 于现有地图、只选 $k$ 个关键帧（当前帧 + 最近关键帧 + $k-2$ 个重叠度最高者——重叠 = 当前帧点云落入该关键帧视锥的点数，每 $n$ 帧存一个关键帧）；损失与跟踪同型但**去掉 silhouette 门控**（建图要照顾所有像素）并加 SSIM 项（[3D 重建教程第 08 章 (8.11)](../../3d_reconstruction/08_3D高斯泼溅.md) 同款正则思想），同时剔除近零不透明/过大高斯（部分沿用 3DGS 的 culling，§3 原文 "as done partly in [14]"）。

### 4.7 RGB-D 依赖与单目缺位（原文核对）

**核对结论**：论文**没有单目变体**——标题即 "Dense RGB-D SLAM"，§6 局限明言"our method requires known camera intrinsics **and dense depth as input**"。机制上深度不可省：首帧等距初始化（式 (6)）与致密化掩码（式 (9)）都以稠密传感器深度为几何来源；跟踪损失的主项也是深度（式 (8)）。单目路线要解决"没有深度时高斯从哪来"，属于 [MonoGS](../../../papers/slam/frontier/arXiv-2312.06741_MonoGS-Gaussian-Splatting-SLAM.pdf) 的贡献范围（其摘要：3 fps 的首个 3DGS 单目 SLAM，可扩展 RGB-D）——与本文互补而非竞争设定。

## 5. 实验与结果解读

- **协议（§4）**：四基准 = ScanNet++（S1/S2 两景 DSLR 采集，帧间位移 ≈ Replica 的 30 帧间隔，唯一有 hold-out 新视角）/ Replica / TUM-RGBD / 原版 ScanNet；基线数字取自 Point-SLAM；结果为 3 个种子平均；Replica 训练视角渲染每 5 帧评一次；ATE = ATE RMSE（cm）。
- **相机跟踪（Table 1）**：**ScanNet++：SplaTAM 1.2 cm**（S1 0.6 / S2 1.9），Point-SLAM 343.8、ORB-SLAM3 158.2——大位移+弱纹理下两者彻底失效（ORB-SLAM3 因无特征反复重初始化），SplaTAM 仍 sub-cm：显式地图直接配准在大运动下的鲁棒性证据。**Replica：0.36 cm**，优于 Point-SLAM 0.52（降低 >30%）、DROID-SLAM 0.38、ESLAM 0.63。**TUM-RGBD：5.48 cm**，优于 Point-SLAM 8.92（降低近 40%）、NICE-SLAM 15.87；但 **ORB-SLAM2 的 1.98 仍大幅领先**——论文承认特征式方法在该基准仍占优（§5 原话）。**原版 ScanNet：11.88 cm**，与 Point-SLAM（12.19）/NICE-SLAM（10.70）同档，无稠密法能进 10 cm。

| 基准（平均 ATE RMSE，cm） | SplaTAM | Point-SLAM | 次优体积式 | 最强传统式 |
|---|---|---|---|---|
| ScanNet++（S1/S2） | **1.2** | 343.8（失效） | ORB-SLAM3 158.2（失效） | — |
| Replica（8 场景） | **0.36** | 0.52 | ESLAM 0.63 | DROID-SLAM 0.38 |
| TUM-RGBD（5 序列） | **5.48** | 8.92 | Vox-Fusion 11.31 | ORB-SLAM2 1.98 |
| 原版 ScanNet | 11.88 | 12.19 | NICE-SLAM 10.70 | — |

（数字按 Table 1 转录；"—"表示该基准无传统法入表。读法：合成/高质量输入（Replica、ScanNet++）上显式直接配准全面领先；传感器质量差的真实数据（TUM/ScanNet）上特征法仍占优。）
- **渲染质量**：**Replica 训练视角（Table 2）**：PSNR 34.11 / SSIM 0.97 / LPIPS 0.10，与 Point-SLAM（35.17）相当——论文特别声明此对比不公平（Point-SLAM 渲染采样用真值深度）且"训练视角渲染本身意义有限"（容量过拟合），只作与旧工作对齐之用；比 Vox-Fusion/NICE-SLAM 高约 10 dB。**ScanNet++ 新视角（Table 3，更有意义的评测）**：PSNR 24.41 / Depth L1 2.07 cm（新视角）、27.98 / 1.28 cm（训练视角）；Point-SLAM 11.91 dB 且**无法渲染深度**——silhouette 驱动的显式地图在新视角下几何外观双优。

| 渲染指标（ScanNet++，Table 3 转录） | 新视角 PSNR/SSIM/LPIPS | 新视角 Depth L1 | 训练视角 PSNR/Depth L1 |
|---|---|---|---|
| Point-SLAM（用真值深度采样） | 11.91 / 0.28 / 0.68 | ✗ | 14.46 / ✗ |
| **SplaTAM** | **24.41 / 0.88 / 0.24** | **2.07 cm** | **27.98 / 1.28 cm** |

（"✗"为论文原表标注——Point-SLAM 不渲染深度；新视角位姿用真值与 SLAM 地图原点对齐，§5 原文。）
- **消融（Table 4/5，Replica #Room0）**：**损失组合（Table 4）**：仅深度 → 跟踪彻底失败（ATE 86.03，深度对 $x$–$y$ 像素面无约束）；仅颜色 → ATE 1.38（>5× 全量版 0.27）且 Depth L1 12.58；全量 0.27 / 0.49 / 32.81——**深度+光度互补**是硬结论。**跟踪三要素（Table 5）**：去掉恒速传播 → ATE 2.95（>10×）；去掉 silhouette 掩码 → 115.80（彻底失败——未建图像素的幻觉梯度）；门限 0.99 → 0.5 → ATE 0.27→1.30（约 5×）——"过饱和"门控是精度来源而非工程细节。**高斯分布（Table S1，ScanNet++ S1）**：各向异性 0.55 ATE vs 各向同性 0.57，PSNR 差 0.29 dB，但各向同性省 42.5% 内存、快 6.7%——为设计选择提供定量辩护。
- **运行时（Table 6，RTX 3080 Ti）**：每帧跟踪 40 迭代 × 25 ms、建图 60 迭代 × 24 ms，共 1.00+1.44 s/帧；每迭代渲染全幅 1200×980（≈1.2M 像素），而 NICE-SLAM/Point-SLAM 只采 ~200（跟踪）/1000（建图）像素——"多 3 个数量级的像素、耗时相当"是光栅化红利的直接度量；轻量版 SplaTAM-S（10/15 迭代、半分辨率致密化）0.19+0.33 s/帧、ATE 0.39，快约 5×、精度轻微下降。

| 运行时（Replica/R0，Table 6 转录） | 跟踪 ms/迭代 | 建图 ms/迭代 | 每帧耗时（跟踪+建图） | ATE |
|---|---|---|---|---|
| NICE-SLAM | 30 | 166 | 1.18 + 2.04 s | 0.97 |
| Point-SLAM | 19 | 30 | 0.76 + 4.50 s | 0.61 |
| **SplaTAM**（40/60 迭代每帧） | 25 | 24 | 1.00 + 1.44 s | **0.27** |
| SplaTAM-S（10/15 迭代每帧） | 19 | 22 | 0.19 + 0.33 s | 0.39 |

（SplaTAM/SplaTAM-S 的每帧迭代数 40/60 与 10/15 按论文 §5 "Runtime Comparison" 原文；NICE-SLAM/Point-SLAM 只给每迭代毫秒与每帧秒数，迭代数未在本文中列出。另注：其他方法每迭代仅用 ~200 像素（跟踪）/1000 像素（建图），"but attempt to cleverly sample these pixels"，SplaTAM 每迭代渲染全幅。）

## 6. 局限与后续影响

**局限**（§6 论文自述）：对**运动模糊、大深度噪声、剧烈旋转**敏感（三类都破坏"渲染-观测逐像素对齐"的前提）；可扩展性——大规模场景需 OpenVDB 式高效结构（§6 原话）；输入前提——**已知内参 + 稠密深度**；跟踪/建图的逐帧梯度迭代未与增量式前端（IMU/特征）融合，帧率离传统 SLAM 的实时线尚远（每帧秒级）。

**结构性代价**（分析，依 §3–5 机制归纳）： **地图无自由位姿图**——所有高斯锚定在世界系，跟踪错了地图跟着错（错误会随致密化固化进几何），没有 [ORB-SLAM3](./ORB-SLAM3_TRO2021.md) 式的回环/位姿图修正机制兜底； **无不确定性量化**（§4.4 对照段）——silhouette 是二值化的"信/不信"，做不了概率融合； **恒速假设的适用域**——ScanNet++ 帧间位移 ≈ Replica 30 帧（§4 实验设置），式 (7) 的线性外推在该 regime 仍有效，但突转/急停场景会直接掉进门控外。**后续影响**：确立"显式高斯 = 可微点云地图"的极简范式（silhouette 当地图边界、等距致密化、恒速+直接配准），被后续 3DGS-SLAM 大量沿用；其 ScanNet++ 新视角基准成为该领域新的标准评测；开源实现（spla-tam.github.io）是该方向最常见的复现基线。

## 7. 与本项目对照

- **与 GS-SLAM 的选型差异表**（同素材、相反重心）：

| 维度 | SplaTAM（本文） | GS-SLAM |
|---|---|---|
| 设计重心 | **跟踪驱动**：位姿逐帧锁进地图，地图被动生长 | **建图驱动**：扩展/清洗地图为主，跟踪纯光度 |
| 高斯 | 各向同性球对称、视角无关颜色、8 参数/颗 | 各向异性 + 1 度 SH（Table S1 证明本文各向同性够用） |
| 跟踪损失 | $L_1(D) + 0.5L_1(C)$，silhouette $S>0.99$ 门控（式 (8)） | 纯光度（其式 (10)）+ coarse-to-fine + 可靠高斯筛选 |
| 地图扩展 | $S<0.5$ 或"新几何在前"等距致密化（式 (9)） | 累积不透明度/深度残差判据 + 悬空衰减（其式 (8)(9)） |
| BA/后端 | 滑动 $k$ 关键帧地图更新（重叠度选帧） | 随机 $K$ 关键帧联合 BA（其式 (13)） |
| 深度的角色 | 跟踪主项 + 初始化/致密化 | 建图损失 + 筛选（不进跟踪损失） |

- **指标口径**：本文的 ATE RMSE 对应本仓库 [METRICS.md §2](../../../projects/slam/METRICS.md) 的 **(M.2)**：$\mathrm{ATE}_{RMSE} = \sqrt{\frac1N\sum_i\|(sRp_i^{est}+t)-p_i^{gt}\|^2}$（Umeyama 相似对齐后，Sturm 2012 §IV-A 定义；`metrics.py::ate_rmse`）。读本文表格时注意：RGB-D 系统的规范自由度只剩 6 维，文献中"是否带尺度对齐"各论文口径不一，横向比较前先核对（METRICS.md §1.2 的对齐讨论）。
- **代码锚点**： [loam2d/](../../../projects/slam/loam2d/)——点到线/点到面残差与 LM 配准（`loam2d.py` 的残差-雅可比实现，见[精读/LOAM](./LOAM_RSS2014.md) §8 对照表）；本文 §4.5 的"深度残差 = 光轴法向点到面"把两者连起来； [photoba/](../../../projects/slam/photoba/)——光度直接法 BA，式 (8) 颜色项的传统对应物； 无 3DGS-SLAM 模块的原因同 [GS-SLAM 精读](./GS-SLAM_CVPR2024.md) §7：每帧 100 次迭代的 CUDA 光栅化前向+反向，超出本项目 CPU/numpy 的边界。

**"跟踪驱动 vs 建图驱动"的工程含义**（综合分析，回指上表）：SplaTAM 把不确定性前置到**空间门控**（哪里可信由 silhouette 决定），GS-SLAM 把不确定性后置到**残差筛选**（哪些高斯可信由深度一致性决定）——前者让跟踪对地图错误鲁棒（错高斯只要不在 $S>0.99$ 区域就不伤位姿），但地图错误一旦固化只能靠 culling；后者让地图质量可控（增删有据），但跟踪吃全部渲染像素的伪影、要靠 coarse-to-fine 补救。两条路线的消融互为镜像：SplaTAM Table 5（去掉 silhouette 门控 ATE 115.80）与 GS-SLAM Table 7（去掉 delete F1 51.35→62.89）各自证明自己那道闸门的必要性。对机器人落地，前者意味着"重定位/回环要另做"（无位姿图），后者意味着"显存与清洗策略决定上限"——这正是[第 10 章 §10.4](../10_建图与系统实战.md) 选型表要把两者并列的原因。
- **教程锚点**：[第 10 章 §10.4 表](../10_建图与系统实战.md)（本文被概括为"以单颗高斯为地图原子（反投影点 + 各向同性协方差），光度残差跟踪、逐高斯增长地图"）｜[3D 重建教程第 08 章](../../3d_reconstruction/08_3D高斯泼溅.md)（式 (1) vs (8.1)、alpha 混合 (8.10)、致密化对照）｜[第 04 章 (4.2)–(4.4)](../04_相机模型与特征提取.md)（投影与反投影）｜[第 02 章](../02_三维刚体运动旋转与位姿.md)/[第 08 章 (8.9)](../08_后端-ii图优化与-ba.md)（$SE(3)$ 扰动）。

## 配套阅读

- 本文 PDF：[SplaTAM（Keetha et al., CVPR 2024）](../../../papers/slam/frontier/arXiv-2312.02126_SplaTAM.pdf)（本精读所有式号以此 arXiv v3 为准）
- 姊妹精读：[GS-SLAM](./GS-SLAM_CVPR2024.md)（同为 3DGS-SLAM 的另一极：建图驱动、各向异性、纯光度跟踪）｜[NeRF-SLAM](./NeRF-SLAM_ICRA2023.md)（隐式场前代）｜[LOAM](./LOAM_RSS2014.md)（点到面残差——本文深度跟踪的经典亲缘）｜[PTAM](./PTAM_ISMAR2007.md)（恒速跟踪先验的源头）｜[DROID-SLAM](./DROID-SLAM_NeurIPS2021.md)（Replica 上 0.38 cm 的可微 BA 对照）
- 教程：[第 10 章](../10_建图与系统实战.md)（§10.4 前沿表）｜[第 06 章](../06_视觉里程计-ii直接法.md)（直接法：光度残差 + 稀疏选点，本文的稠密版对照）｜[第 04 章](../04_相机模型与特征提取.md)｜[第 02 章](../02_三维刚体运动旋转与位姿.md)
- 3DGS 本体：[3D 重建教程第 08 章](../../3d_reconstruction/08_3D高斯泼溅.md)｜原论文 PDF [papers/3d_reconstruction/frontier/arXiv-2308.04079_3D-Gaussian-Splatting.pdf](../../../papers/3d_reconstruction/frontier/arXiv-2308.04079_3D-Gaussian-Splatting.pdf)
- 代码：[projects/slam/loam2d/](../../../projects/slam/loam2d/)｜[projects/slam/photoba/](../../../projects/slam/photoba/)｜[METRICS.md](../../../projects/slam/METRICS.md)（ATE 口径 (M.2)）
- 主题导航：[tutorials/slam/README.md](../README.md) ｜ [OVERVIEW.md](../OVERVIEW.md)

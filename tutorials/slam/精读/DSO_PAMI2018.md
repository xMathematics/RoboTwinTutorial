# 论文精读｜DSO：直接稀疏里程计（PAMI 2018）

> **PDF**：[papers/slam/classics/arXiv-1607.02565_DSO.pdf](../../../papers/slam/classics/arXiv-1607.02565_DSO.pdf) ｜ **教程**：[第 06 章](../06_视觉里程计-ii直接法.md)、[第 08 章](../08_后端-ii图优化与-ba.md) ｜ **代码**：[projects/slam/photoba/](../../../projects/slam/photoba/)（滑窗光度 BA）

## 1. 论文信息与一句话贡献

- **题目**：Direct Sparse Odometry
- **作者**：Jakob Engel（TUM）、Vladlen Koltun（Intel Labs）、Daniel Cremers（TUM）
- **发表**：arXiv:1607.02565（v2, 2016-10，本仓库收录 PDF，17 页，式号以此为准）；会议版 ECCV 2016，期刊版 IEEE TPAMI 2018
- **一句话贡献**：提出**直接 + 稀疏**（direct & sparse）的单目视觉里程计——在滑动窗口内对"全部关键帧位姿 + 相机内参 + 每帧仿射亮度参数 + 点的逆深度"做**联合光度光束平差**（photometric bundle adjustment），并配以**完整光度相机标定**（响应函数、渐晕、曝光时间）——首次把"像素灰度"当作有物理量纲的测量值建模，在 TUM-monoVO / EuRoC / ICL-NUIM 三个数据集上同时超越当时的直接法与间接法。

## 2. 问题与动机

**要解决什么**：单目视频的实时视觉里程计，精度与鲁棒性同时超越特征点法，且不需要 GPU。

**框架坐标**（§1 的 2×2 分类，是读这篇论文的地图）：按"优化什么误差"分**直接/间接**——直接法优化*光度误差*（传感器原始测量， photometric error），间接法优化*几何误差*（预先算好的中间量：角点位置、光流场）；按"用多少数据"分**稀疏/稠密**——稀疏法只用选定独立点集，稠密法用（近乎）全部像素并需要几何先验。四象限：Spar+Indir = PTAM、monoSLAM、ORB-SLAM；Dense+Indir = 场重构类；**Sparse+Direct = 本文**（此前只有 Jin et al. 2003 的 EKF 版本）；Dense+Direct = DTAM、LSD-SLAM。稠密法的几何先验"may well introduce a bias, and thereby reduce rather than increase the long-term, large-scale accuracy"（§1.1）——这是走向稀疏的第一个理由。

**核心动机——把光度非理想性从"噪声"变成"模型"**（§1.1(1)）：角点对光度/几何畸变天然鲁棒，直接法却对**屏上一切非理想性**敏感：自动曝光、gamma/非线性响应、镜头渐晕（vignetting）、去马赛克伪影、rolling shutter。已有直接法（LSD-SLAM、DTAM）靠"假设亮度恒常 + 鲁棒核容忍"绕开这些因素。DSO 的选择是**显式建模图像成像全链路**："direct approach models the full image formation process down to pixel intensities, it greatly benefits from a more precise sensor model"。第二个动机是**点的采样自由**：不依赖角点检测器/描述子，可以从全部图像区域均匀采样——包括边缘与弱强度变化（§4 的消融证明这有真实收益）。

## 3. 方法总览

```
视频帧（含曝光时间 t_i）──▶【光度标定】I_i(x) = G(t_i·V(x)·B_i(x))   (Eq.(2))
                 光度校正：I'_i = G^{-1}(I_i)/V = t_i·B_i             (Eq.(3))
                              │
                              ▼
【前端 Front-End（§3）】
  Step 1 初始帧跟踪：对最近关键帧做多尺度两帧直接对齐（含 RANSAC 式恢复跟踪）
  Step 2 关键帧创建（三判据，加权超阈即建）：
         视场变化——平均光流 f := ((1/n)·Σ_{i=1}^n ‖p − p'‖²)^{1/2}（初始粗跟踪期计算）
         遮挡/出画——去旋转平均光流 f_t（同式但 R = I_3×3；平移才造成遮挡）
         曝光变化——相对亮度因子 a := | log( e^{a_j − a_i}\, t_j\, t_i^{-1} ) |（即两帧有效曝光之比）
         判据：w_f·f + w_ft·f_t + w_a·a > T_Kf（论文默认权重全 1、T_Kf = 1）
         （策略：先多建——每秒 5-10 个关键帧，再由边缘化稀疏化）
  Step 3 关键帧边缘化：保留最新两帧；<5% 点可见者边缘化；
         否则边缘化"距离得分"最大者（Eq.(20)）——先边缘化其中的全部点
  点管理（§3.2）：候选点选取（32×32 块自适应阈值 ḡ+g_th，g_th=7；d×d 均匀分布）
         → 候选跟踪（沿极线的光度离散搜索，得粗深度初值）→ 激活（保持 N_p=2000 均匀分布）
                              │
                              ▼
【后端 Back-End（§2.2–2.3）】滑窗光度 BA：
  E_photo（Eq.(8)）+ E_prior（Eq.(9)）对 {T_i, c, d_p, a_i, b_i} 联合 G-N
  ——FEJ 雅可比（J_geo, J_photo 在 x=0 处取值，Eq.(13)）
  ——边缘化：Schur 补掉最老帧/点 → 二次先验注入剩余变量（Eq.(15)-(19)）
```

**窗口规模**：活跃关键帧至多 $N_f = 7$ 个（论文原文 "we use $N_f = 7$"；新关键帧先创建并优化、再边缘化，故瞬时多一帧）；活跃点固定 $N_p = 2000$；每个点带 8 像素残差模式 $\mathcal{N}_p$（Fig. 4：中心像素 + 稍散开的 8 邻域——对 $N_P > 1$ 的要求是"所有模型参数都良好约束"，且小邻域 SSD 近似等价于"加入一、二阶辐照度导数约束"）。

## 4. 关键公式推导（式号按论文 PDF 原文）

### 4.0 符号表

| 符号 | 含义（首次出现处） |
|---|---|
| $T_i \in SE(3)$, $\boldsymbol{\xi}_i \in \mathfrak{se}(3)$ | 第 $i$ 帧位姿（世界→相机）及其李代数；$\boxplus$：$\boldsymbol{x}_j \boxplus \mathbf{T}_j := e^{\hat{\boldsymbol{\xi}}_j}\mathbf{T}_j$（Eq.(1)） |
| $B_i(\mathbf{x})$ | 像素 $\mathbf{x}$ 接收的**辐照度**（irradiance，物理量） |
| $V(\mathbf{x}) \in (0,1]$ | 镜头衰减图（vignetting），位置相关 |
| $G: \mathbb{R} \to [0, 255]$ | 非线性**响应函数**（response function） |
| $t_i$ | 曝光时间（ms，已知或待估） |
| $I'_i$ | 光度校正图像 $= t_i B_i$（Eq.(3)） |
| $(a_i, b_i)$ | 仿射亮度参数：有效曝光 $t_i e^{a_i}$、偏移 $b_i$（Eq.(4)） |
| $\mathbf{p} \in \Omega$, $d_{\mathbf{p}}$ | 宿主帧（host frame）像素与该点逆深度（§2.2 "Point Dimensionality"） |
| $\mathbf{p}'$ | 点投影到目标帧的像点（Eq.(5)） |
| $w_{\mathbf{p}}$ | 梯度依赖权重（Eq.(7)） |
| $\mathcal{N}_p$ | 8 像素残差模式（Fig. 4） |
| $E_{P_j}, E_{photo}, E_{prior}$ | 单残差能量（Eq.(4)）/总光度能量（Eq.(8)）/先验（Eq.(9)） |
| $\boldsymbol{x} \in \mathfrak{se}(3)^{n} \times \mathbb{R}^m$ | 全部优化变量相对 $\boldsymbol{\zeta}_0$ 的切空间累积增量（§2.3） |
| $\mathbf{H}, \mathbf{b}, \mathbf{W}, \mathbf{r}, \mathbf{J}$ | G-N 系统（Eq.(10)） |

### 4.1 光度相机标定（论文 Eq.(2)–(3)）

**成像模型**（Eq.(2)，每个因子的物理来源）：

$$I_i(\mathbf{x}) = G\big( t_i\, V(\mathbf{x})\, B_i(\mathbf{x}) \big)$$

- $B_i(\mathbf{x})$：场景辐照度经透镜投射到传感器像素上的**真实能量**——我们真正想测的量；
- $V(\mathbf{x})$：镜头渐晕造成的**位置相关衰减**（$\in (0,1]$）——同一辐照度在画面边缘更暗；
- $t_i$：**曝光时间**——乘性缩放（Fig. 3 显示其数据集中变化超过三个数量级、室内外相差 1:50 到 1:10,000 量级）；
- $G$：传感器的**非线性响应**（gamma、ISP 处理），把能量映射到 $[0,255]$ 的灰度。

间接法可以无视这一切（描述子对单调光照近似不变），直接法逐像素比较灰度，必须把链路逆转。**光度校正**（Eq.(3) 的推导）：$G$ 可逆，故对 Eq.(2) 两侧施加 $G^{-1}$，再除以 $V(\mathbf{x})$（依据：逐步代数逆运算）：

$$G^{-1}\big( I_i(\mathbf{x}) \big) = t_i\, V(\mathbf{x})\, B_i(\mathbf{x}) \;\Longrightarrow\; I'_i(\mathbf{x}) := t_i\, B_i(\mathbf{x}) = \frac{G^{-1}\big( I_i(\mathbf{x}) \big)}{V(\mathbf{x})} \tag{3}$$

关键遗留：$I'_i$ 仍与 $t_i$ **成正比**——校正消掉了 $G$ 与 $V$，消不掉曝光；这正是 §4.2 残差里必须出现曝光比值项的原因。（工程注记：$G, V$ 由离线标定流程获得，Fig. 3 展示了 TUM monoVO 数据集的一组标定结果。）

### 4.2 光度残差、梯度权重与完整能量（论文 Eq.(4)–(9)）

**单点残差**（Eq.(4)，逐符号解释）：

$$E_{P_j} := \sum_{\mathbf{p} \in \mathcal{N}_p} w_{\mathbf{p}} \left\| \big( I_j[\mathbf{p}'] - b_j \big) - \frac{t_j\, e^{a_j}}{t_i\, e^{a_i}} \big( I_i[\mathbf{p}] - b_i \big) \right\|_\gamma \tag{4}$$

- $I_i[\mathbf{p}], I_j[\mathbf{p}']$：宿主帧 $i$（点所在帧）与目标帧 $j$ 的**原始**灰度；$\mathbf{p}'$ 由几何投影给出（Eq.(5)-(6)）：

$$\mathbf{p}' = \Pi_c\big( \mathbf{R}\, \Pi_c^{-1}(\mathbf{p}, d_{\mathbf{p}}) + \mathbf{t} \big), \qquad \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0} & 1 \end{bmatrix} := \mathbf{T}_j\, \mathbf{T}_i^{-1} \tag{5}{,}(6)$$

（$\Pi_c^{-1}(\mathbf{p}, d)$：像素 + 逆深度反投影成 3D 点，即教程 (4.6) 的逆；$c$ 为内参向量。）
- **残差的物理推导**：同一物点在两帧的辐照度应相等（光度恒常假设的物理内核）$B_j[\mathbf{p}'] = B_i[\mathbf{p}]$。用 Eq.(3) 把两侧灰度都还原到辐照度：$I'_j[\mathbf{p}'] = t_j B_j[\mathbf{p}']$、$I'_i[\mathbf{p}] = t_i B_i[\mathbf{p}]$，故 $I'_j[\mathbf{p}'] / t_j = I'_i[\mathbf{p}] / t_i$，即 $I'_j[\mathbf{p}'] = (t_j / t_i)\, I'_i[\mathbf{p}]$——**残差必须按曝光比值缩放**。未知响应/残余渐晕的一阶偏差由仿射对 $(a_i, b_i)$ 吸收：增益走对数 $e^{a_i}$（依据：保证恒正、避免乘性噪声引发漂移，论文 §2.2 明说），偏移 $b_i$ 吸收加性项。代入 Eq.(3) 的 $I'$ 并保留 $(b_i)$ 于校正之内，即得 Eq.(4) 的形式。
- $\|\cdot\|_\gamma$：Huber 范数（外点）；$w_{\mathbf{p}}$：梯度依赖权重（下述）。

**梯度权重 Eq.(7) 的推导——"高梯度像素降权"从哪来**。论文的论证：在投影点 $\mathbf{p}'$ 上叠加小的独立**几何**噪声并立即边缘化。设 $\mathbf{p}' \to \mathbf{p}' + \boldsymbol{\varepsilon}$、$\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma_g^2 \mathbf{I})$，残差 $e = I_j[\mathbf{p}']$ 的一阶传播（依据：与 LSD-SLAM 论文 Eq.(11) 同型的一阶不确定性传播，$e(\mathbf{p}'+\boldsymbol{\varepsilon}) \approx e + \nabla I^\top \boldsymbol{\varepsilon}$）：

$$\mathrm{Var}(e) \approx \sigma_I^2 + \nabla I^\top (\sigma_g^2 \mathbf{I})\, \nabla I = \sigma_I^2 + \sigma_g^2 \lVert \nabla I_{\mathbf{p}}(\mathbf{p}) \rVert_2^2$$

总噪声越小权重越大，取 $w_{\mathbf{p}} \propto 1/\mathrm{Var}(e)$、记 $c^2 := \sigma_I^2/\sigma_g^2$（标定得到的图像噪声与几何噪声之比）：

$$w_{\mathbf{p}} := \frac{c^2}{c^2 + \lVert \nabla I_{\mathbf{p}}(\mathbf{p}) \rVert_2^2} \tag{7}$$

**方向辨析**（与半稠密选择门相反，勿混）：Eq.(7) 是**高梯度降权**——梯度越大，同样的几何/离散化噪声放大成越大的光度残差（论证如上）；而"梯度大才可观测"（LSD-SLAM 半稠密门、教程 06.2 第三步）说的是零梯度像素对位姿零贡献。两者不矛盾：前者在**已选中的像素内部**按噪声分配权重，后者决定**哪些像素入选**。DSO 的候选点选取（§3.2，$g_{th}=7$ 的自适应阈值）承担后一角色。

**完整光度能量与先验**（Eq.(8)-(9)）：

$$E_{photo} := \sum_{i \in \mathcal{F}} \sum_{\mathbf{p} \in \mathcal{P}_i} \sum_{j \in \mathrm{obs}(\mathbf{p})} E_{P_j}, \qquad E_{prior} := \sum_{i \in \mathcal{F}} \big( \lambda_a\, a_i^2 + \lambda_b\, b_i^2 \big) \tag{8}{,}(9)$$

Eq.(8)：帧集 $\mathcal{F}$ × 宿主点集 $\mathcal{P}_i$ × 观测帧 $\mathrm{obs}(\mathbf{p})$ 的三重和——每条残差**同时依赖宿主帧与目标帧两个位姿**（经典重投影误差只依赖一个），论文指出这只给 Hessian 的位姿-位姿块加非对角项、**不改变 Schur 补消点后的稀疏模式**（与教程 (8.12) 的块对角论证衔接）。Eq.(9) 先验把仿射亮度参数拉向零： 曝光时间已知时，$(a_i, b_i)$ 只剩残余模型误差，需先验防漂移（依据：论文引 [7] Leutenegger 等的分析——$(a_i, b_i)$ 同时含噪声测量时，无约束 ML 估计会使 $a$ 漂移）； 曝光时间未知时取 $t_i = 1$、$\lambda_a = \lambda_b = 0$，让 $(a_i, b_i)$ 自由吸收未知曝光。这是亮度**规范自由度**（gauge freedom）的钉法之一——由 (8) 可验算：把 $I_i \to \gamma_i I_i + \delta_i$、$(a_i, b_i) \to (a_i + \log\gamma_i,\ \gamma_i b_i + \delta_i)$，残差逐项不变（教程 (6.7) 的完整证明逐字适用），故无先验时 $(a,b)$ 有 2 维整体不可辨识方向。

### 4.3 仿射亮度参数与完整光度标定：何时可以近似

**两种模型的关系**。完整标定路线（§4.1-4.2）：已知 $G, V$，先用 Eq.(3) 把灰度还原为 $I'_i = G^{-1}(I_i)/V$，残差再按曝光比值缩放。仿射路线（论文 §2.2）：把整条逆链"响应逆 + 渐晕除法 + 曝光缩放"合并成对**原始灰度**的一次仿射变换——论文写作 $e^{-a_i}(I_i - b_i)$，与教程 (6.6) 的正向形式 $\frac{I_i - b_i}{t_i e^{a_i}}$ 逐符号相同（取 $t_i \equiv 1$；本仓库 `photoba` 的正向形式 $I' = e^{a}I + b$ 是同一模型的参数代换，见其模块 docstring）。

**何时仿射足以替代完整标定**（推导）。仿射近似 = 在工作灰度区间上把 $G^{-1}$ 线性化、把渐晕常数化。设像素工作点 $I_0$，对 $G^{-1}$ 做一阶 Taylor 展开（依据：一阶展开，余项二阶小量）：

$$G^{-1}(I) \approx G^{-1}(I_0) + \big( G^{-1} \big)'(I_0)\, (I - I_0)$$

代入 Eq.(3)：校正图像是 $I$ 的**仿射函数**（斜率 $(G^{-1})'(I_0)/V$、截距 $G^{-1}(I_0) - I_0 (G^{-1})'(I_0)/V$）；再设短基线下两帧同一物点的 $V(\mathbf{x}) \approx V(\mathbf{x}')$（渐晕图空间缓变），曝光缩放 $t_j/t_i$ 乘性地并入斜率——于是完整标定模型退化为仿射模型，且**一阶精确**。据此读出三个失效条件： 响应强非线性且场景灰度跨大量级（$(G^{-1})'$ 在工作区间内变化显著，一阶展开失效）； 大平移使同一物点从画面中心移到边缘（$V(\mathbf{x}) \neq V(\mathbf{x}')$，"渐晕常数化"失效）； 曝光时间未知但变化剧烈——此时 $a_i$ 需吸收 $\log t_i$，模型仍自洽（Eq.(4) 把 $t_i$ 乘进 $e^{a_i}$），但每帧多一个待估量、且需 Eq.(9) 先验或规范锚定防漂移。实验佐证（Fig. 15）：去掉 $V, G$ 标定仅小幅变差、**完全去掉仿射（亮度恒常假设）最差**——仿射层是性价比最高的一层，完整标定在其上做精修。

**与 photoba 模块的对照**：`photoba` 的合成场景按 $I_i = e^{-a_i}(P_i - b_i)$ 施加曝光（恰为校正模型的逆），曝光真值即仿射的——故其仿射实现对该数据是**精确**模型而非近似，这是它能把 $(a_i, b_i)$ 恢复到 $10^{-3}$ 量级（§7 健康值）的前提；真实相机上 $G$ 强非线性时，同样的仿射层只是 Eq.(3) 的一阶化身。

### 4.4 滑窗优化与 FEJ（论文 Eq.(10)–(14)、Eq.(20)）

**G-N 系统**（Eq.(10)-(13)）。堆叠残差 $\mathbf{r} \in \mathbb{R}^d$、对角权重 $\mathbf{W}$（含 $w_p$ 与 Huber IRLS），正规方程（依据：教程 (8.4)→(8.5) 同一推导）：

$$\mathbf{H} = \mathbf{J}^\top \mathbf{W} \mathbf{J} \quad \text{and} \quad \mathbf{b} = -\mathbf{J}^\top \mathbf{W} \mathbf{r} \tag{10}$$

单条残差（Eq.(11)，$I_i[\mathbf{p}]$ 为宿主帧灰度）：

$$r_k = \big( I_j[\mathbf{p}'(\mathbf{T}_i, \mathbf{T}_j, d_i, \mathbf{c})] - b_j \big) - \frac{t_j e^{a_j}}{t_i e^{a_i}} \big( I_i[\mathbf{p}] - b_i \big) \tag{11}$$

雅可比按变量分块（Eq.(12)-(13)，增量 $\delta$ 定义在切空间，Eq.(1) 的 $\boxplus$）：

$$\mathbf{J}_k = \left[ \underbrace{\frac{\partial I_j}{\partial \mathbf{p}'} \cdot \frac{\partial \mathbf{p}'((\delta \boxplus \boldsymbol{x}) \boxplus \boldsymbol{\zeta}_0)}{\partial \delta_{geo}}}_{\mathbf{J}_I\ \cdot\ \mathbf{J}_{geo}} \;\middle|\; \underbrace{\frac{\partial r_k((\delta \boxplus \boldsymbol{x}) \boxplus \boldsymbol{\zeta}_0)}{\partial \delta_{photo}}}_{\mathbf{J}_{photo}} \right] \tag{13}$$

三段读法：$\mathbf{J}_I = \partial I_j/\partial \mathbf{p}'$ 是图像梯度（教程 (6.5) 最外层因子）；$\mathbf{J}_{geo}$ 是投影对（位姿、逆深度、内参）的几何雅可比（教程 (5.11) 的推广——多出 $\partial\mathbf{p}'/\partial d_{\mathbf{p}}$ 深度列与内参列）；$\mathbf{J}_{photo}$ 对 $(a_i, b_j)$ 等取解析形式（$\partial e/\partial b_j = -1$ 型）。**First Estimate Jacobians（FEJ）**：$\mathbf{J}_{geo}, \mathbf{J}_{photo}$ 一律在 $\boldsymbol{x} = \mathbf{0}$（首次估计点）处取值，$\mathbf{J}_I$ 在当前 $\boldsymbol{x}$ 处取值。**为什么**（论文原文推导要点）：能量存在非线性零空间（本公式中为绝对位姿与尺度）——若各次迭代在不同线性化点取 $\mathbf{J}_{geo}$，数值上不为零的雅可比会向零空间方向注入虚假信息、"slowly corrupts the system"；固定线性化点使零空间结构跨迭代一致。近似误差可忽略，因 $\mathbf{J}_{photo}, \mathbf{J}_{geo}$ 相对增量光滑（$\mathbf{J}_I$ 不光滑但不影响零空间）。更新（Eq.(14)）：$\boldsymbol{x}^{new} \leftarrow \delta + \boldsymbol{x}$——**切空间增量累积**，$\boldsymbol{x} \boxplus \boldsymbol{\zeta}_0$ 才是真实状态；由 FEJ 还可推出乘性更新 $\boldsymbol{x}^{new} \leftarrow \log(\delta \boxplus e^{\boldsymbol{x}})$ 完全等价（论文原文）。注意 DSO **不用 LM 阻尼**："since we never start far-away from the minimum – a Levenberg-Marquad dampening (which slows down convergence) is not required"（初值由前端保证）。

**滑窗与关键帧边缘化**（Eq.(20)，窗口规则见 §3）：窗口保持 $N_f = 7$ 帧活跃；边缘化规则（论文 §3.1 Step 3）—— 永远保留最新两帧 $I_1, I_2$； 在 $I_1$ 中可见点少于 5% 的帧被边缘化； 否则边缘化（排除 $I_1, I_2$ 后）**距离得分**最大者：

$$s(I_i) = \sqrt{d(i, 1)} \sum_{j \in [3, n] \setminus \{i\}} \big( d(i, j) + \epsilon \big)^{-1} \tag{20}$$

$d(i,j)$ 为关键帧间欧氏距离、$\epsilon$ 小常数。读法：$\sum (d(i,j)+\epsilon)^{-1}$ 大 = 该帧离其他帧都近（冗余，删之损失小）；$\sqrt{d(i,1)}$ 大 = 离最新帧远——两者相乘最大化即"删掉离当前最远、又与近邻重复"的帧，"keep active keyframes well-distributed in 3D space, with more keyframes close to the most recent one"（论文原文；对照固定滑最老帧的 fixed-lag 策略，§4.2 实验显示后者显著更差）。

**残差模式 $\mathcal{N}_p$**：每点不是单像素残差，而是 Fig. 4 的 8 像素模式（中心 + 稍散开的 8 邻域中去掉一个角像素）上的加权和（Eq.(4) 的 $\sum_{\mathbf{p} \in \mathcal{N}_p}$）。两点原因： $|\mathcal{N}_p| > 1$ 是"所有模型参数良好约束"的必要条件——单像素残差对内参等参数过约束不足（论文原文要求 $|\mathcal{N}_p| > 1$ when optimizing over only two frames）； 小邻域 SSD 近似等价于"加入一、二阶辐照度导数约束"（论文原话），信息量略增而对运动模糊更稳。8 个像素共享同一 $d_{\mathbf{p}}$ 与同一条几何 warp，只在 $\mathbf{J}_I$ 与 $w_{\mathbf{p}}$ 上逐像素不同——雅可比由此复用同一条 $\mathbf{J}_{geo}$。

**为什么 DSO"稀疏"却不怕弱纹理**：候选点选取（§3.2）不要求角点，只要求块内相对梯度突出（阈值 $g_{th}=7$），故边缘、弱强度变化都能入选——与"必须先被 FAST 检测到"的间接法形成对照；§4 的 Fig. 17 消融（只用 FAST 点会显著变差）是这条设计主张的直接证据。

### 4.5 边缘化：Schur 补掉最老帧 → 二次先验（论文 Eq.(15)–(19)）

**为什么滑窗必须有先验——规范自由度论证**。光度 BA 的能量 (8) 在以下变换下不变（逐一验算）： 世界系整体 $SE(3)$ 摆动（6 维）； 单目尺度缩放（1 维，教程 (5.9)）； 仿射亮度整体变换（2 维，§4.2 末）——合计 10 维零空间。滑窗只优化近几帧，若不加任何约束，$\mathbf{H}$ 沿这些方向奇异、解不唯一；更危险的是窗口滑动时旧观测被丢弃，若没有把"窗外的信息"保留下来，尺度与漂移会失去约束。**边缘化先验正是这份窗外交接的信息**：把滑出变量的全部残差压缩成一个对剩余变量的二次惩罚，同时把零空间"钉"在历史信息的方向上（配合 Eq.(9) 先验与首帧规范锚定）。

**推导**（逐步）。设 $E'$ 为依赖被边缘化变量 $\boldsymbol{x}_\beta$ 的那部分能量，$\boldsymbol{x}_\alpha$ 为保留变量。第一步：在当前估计 $\boldsymbol{\zeta} = \boldsymbol{x} \boxplus \boldsymbol{\zeta}_0$ 处做 G-N 二阶近似（依据：教程 (8.3)-(8.4)——残差一阶展开后二次型展开；注意这是 $\Delta$ 的二次函数、非完整 Taylor）：

$$E'(\boldsymbol{x} \boxplus \boldsymbol{\zeta}_0) \approx 2(\boldsymbol{x} - \boldsymbol{x}_0)^T \mathbf{b} + (\boldsymbol{x} - \boldsymbol{x}_0)^T \mathbf{H} (\boldsymbol{x} - \boldsymbol{x}_0) + c = 2\boldsymbol{x}^T \underbrace{(\mathbf{b} - \mathbf{H}\boldsymbol{x}_0)}_{=:\ \mathbf{b}'} + \boldsymbol{x}^T \mathbf{H} \boldsymbol{x} + \big( \underbrace{c + \boldsymbol{x}_0^T \mathbf{H} \boldsymbol{x}_0 - \boldsymbol{x}_0^T \mathbf{b}}_{=:\ c'} \big) \tag{15}$$

（展开依据：$2(\boldsymbol{x}-\boldsymbol{x}_0)^T\mathbf{b} = 2\boldsymbol{x}^T\mathbf{b} - 2\boldsymbol{x}_0^T\mathbf{b}$，$(\boldsymbol{x}-\boldsymbol{x}_0)^T\mathbf{H}(\boldsymbol{x}-\boldsymbol{x}_0) = \boldsymbol{x}^T\mathbf{H}\boldsymbol{x} - 2\boldsymbol{x}_0^T\mathbf{H}\boldsymbol{x} + \boldsymbol{x}_0^T\mathbf{H}\boldsymbol{x}_0$，$\mathbf{H}$ 对称故 $-2\boldsymbol{x}^T\mathbf{H}\boldsymbol{x}_0$ 并入 $2\boldsymbol{x}^T(\mathbf{b}-\mathbf{H}\boldsymbol{x}_0)$。**勘误注**：论文印出的常数项为 $c + \boldsymbol{x}_0^\top\mathbf{H}\boldsymbol{x}_0 - \boldsymbol{x}_0^\top\mathbf{b}$，按上式逐项展开应为 $-2\boldsymbol{x}_0^\top\mathbf{b}$——印刷少系数 2；$c'$ 反正作为常数被丢弃，不影响任何结果。）

第二步：丢弃 $c, c'$（依据：不含 $\boldsymbol{x}$，不影响极小点——论文原话 "The constants $c, c'$ can be dropped"），按保留/边缘化分块解线性系统（依据：教程 (8.13) 的分块正规方程）：

$$\begin{bmatrix} \mathbf{H}_{\alpha\alpha} & \mathbf{H}_{\alpha\beta} \\ \mathbf{H}_{\beta\alpha} & \mathbf{H}_{\beta\beta} \end{bmatrix} \begin{bmatrix} \boldsymbol{x}_\alpha \\ \boldsymbol{x}_\beta \end{bmatrix} = \begin{bmatrix} \mathbf{b}'_\alpha \\ \mathbf{b}'_\beta \end{bmatrix} \tag{16}$$

第三步：对 $\boldsymbol{x}_\beta$ 施加 **Schur 补**（依据：从 (16) 第二行解出 $\boldsymbol{x}_\beta = \mathbf{H}_{\beta\beta}^{-1}(\mathbf{b}'_\beta - \mathbf{H}_{\beta\alpha}\boldsymbol{x}_\alpha)$ 代入第一行——教程 (8.14)→(8.15) 的同一消元）：

$$\widehat{\mathbf{H}}_{\alpha\alpha} = \mathbf{H}_{\alpha\alpha} - \mathbf{H}_{\alpha\beta} \mathbf{H}_{\beta\beta}^{-1} \mathbf{H}_{\beta\alpha}, \qquad \widehat{\mathbf{b}'}_\alpha = \mathbf{b}'_\alpha - \mathbf{H}_{\alpha\beta} \mathbf{H}_{\beta\beta}^{-1} \mathbf{b}'_\beta \tag{17}{,}(18)$$

第四步：**剩余能量恰是关于 $\boldsymbol{x}_\alpha$ 的二次函数**（Eq.(19)）——这就是注入后续优化的先验因子：

$$E'(\boldsymbol{x}_\alpha \boxplus (\boldsymbol{\zeta}_0)_\alpha) = 2\boldsymbol{x}_\alpha^T \widehat{\mathbf{b}'}_\alpha + \boldsymbol{x}_\alpha^T \widehat{\mathbf{H}}_{\alpha\alpha}\, \boldsymbol{x}_\alpha \tag{19}$$

概率语义：$\widehat{\mathbf{H}}_{\alpha\alpha}$ 正是被边缘化变量积分掉后、$\boldsymbol{x}_\alpha$ 边缘高斯的信息矩阵（对照教程 (8.17)：$\Sigma_{\boldsymbol{c}|\boldsymbol{z}} = (\Lambda_{cc} - \Lambda_{cp}\Lambda_{pp}^{-1}\Lambda_{pc})^{-1}$——Schur 补 = 概率边缘化，因子 2 是论文的能量记号约定）；$\widehat{\mathbf{b}'}_\alpha$ 编码先验的"线性项"（等价于信息形式高斯的自然参数）。**稀疏性保护**（论文 §2.3/§3.1）：边缘化帧 $i$ 时先边缘化其全部宿主点 $\mathcal{P}_i$ 及最近两帧未见过的点，再把该帧上仍活跃点的剩余观测**直接丢弃**——避免 fill-in 破坏 H 稀疏模式；论文自认此法"clearly suboptimal"（实践中约一半残差因此被丢），换来的是可实时求解。**与 FEJ 的配合**：先验 (19) 所在切空间必须对所有后续优化/边缘化保持不变（论文原文要求）——这正是 4.4 FEJ 的作用：若线性化点漂移，先验的零空间方向会被"重新解释"，历史信息反而腐蚀当前估计。

### 4.6 点的逆深度参数化（"Point Dimensionality" 一节）

**一维而非三维**：间接法中点是任意 3D 位置（3 未知量）；直接法中"点"定义为*源像素光线与表面的交点*——像素位置 $\mathbf{p}$ 已固定光线方向，未知量只剩沿射线的深度，**1 维**。论文的论证：间接法把点隐式定义为"（投影进）图像中的极大值位置"，其图像位置与表面位置都未知；直接法的点就是"the point where the source pixel's ray hits the surface"，一个自由度。参数化收益直接体现为优化变量数：$N_p = 2000$ 个点只贡献 2000 维（而非 6000 维），且 Hessian 的点-点块天然块对角（教程 (8.12)）。

**为什么取"逆"深度**（远处不确定性近高斯的论证）：设点在宿主帧像素 $\mathbf{p}$、逆深度 $d$，目标帧经纯横向平移 $t_x$（$\mathbf{R} = \mathbf{I}$，由 Eq.(3') 类型的 warp，$k = (x, y, 1)^\top$）得 $u' = x + t_x d$（依据：$u' = (k_1 + t_x d)/(k_3 + t_z d)$ 取 $k_3 = 1, t_z = 0$），故

$$d = \frac{u' - x}{t_x} \;\Longrightarrow\; \delta d = \frac{\delta u'}{t_x}$$

匹配误差 $\delta u'$ 近似与深度无关的（同方差）高斯 ⟹ **逆深度的误差是等方差高斯，无论远近**（依据：常数的线性变换不改变高斯性）。反观深度 $z = 1/d$：$\delta z = \delta d / d^2 = z^2\, \delta d$（依据：$z = d^{-1}$ 求导）——误差随 $z^2$ 增长，远处点的深度分布重尾、绝非高斯。论文原文："inverse depth parameterization... is better suited to represent uncertainty from stereo-based depth estimation, in particular for far-away points [3]"（[3] = Civera et al. 的逆深度 SLAM）。候选点跟踪（§3.2 Step 2）沿极线的光度离散搜索给 $d$ 供初值与方差，方式与 LSD-SLAM 的深度滤波同源（[LSD-SLAM 精读 §4.4](./LSD-SLAM_ECCV2014.md)）。

## 5. 实验与结果解读

- **三个数据集、三种口径**（§4）：TUM-monoVO（50 段光度标定序列、105 分钟，用大回环后的累计漂移 $e_r$（旋转）、$e_s$（尺度）与 alignment error $e_{align}$ 评估，Fig. 12/14）；EuRoC MAV（11 段，无光度标定与曝光时间——设 $t_i = 1$、$\lambda_a = \lambda_b = 0$，评 Sim(3) 对齐后的平移 RMSE $e_{aEuRoC}$，Fig. 10）；ICL-NUIM（8 段仿真，无需光度校正、$t = 1$，同评 $e_{aEuRoC}$）。协议：各序列正反向跑 5 遍 ×10 次计 500/230/80 次运行，误差画成累积误差图。
- **指标口径与本项目评测的对应**（读表前必查）：$e_{aEuRoC}$ 经 **Sim(3)** 对齐——单目尺度误差被对齐吃掉，须配合尺度类指标单独看（本仓库 [METRICS.md](../../../projects/slam/METRICS.md) §4 的尺度比 + §6"Sim(3) 对齐会吃掉尺度误差"的提醒正是同一陷阱）；$e_{align}$ 在 TUM-monoVO 数据集论文（其引文 [8]）中定义、衡量大回环后的长程漂移（论文对乘性因子按 $\max(e, e^{-1})$ 聚合）——与 [METRICS.md](../../../projects/slam/METRICS.md) §5 的 ATE/RPE 分工同理：回环口径回答"最后错到哪儿了"，不回答"每一步走得稳不稳"。
- **总体结论**（§4.1）：直接稀疏方法在 TUM-monoVO 与 ICL-NUIM 上**精度与鲁棒性双双超过 ORB-SLAM**；EuRoC 上 ORB-SLAM 精度更好但鲁棒性更低——论文归因两条：EuRoC 无光度标定（DSO 的模型优势用不上），且序列含大量小回环/折返，ORB-SLAM 的建图与回环机制天然受益，而 DSO 是纯里程计、永久边缘化离开视场的帧。**对比其他直接法**：LSD-SLAM 与 SVO "consistently fail on most of the sequences"——主因正是它们假设亮度恒常，而真实数据含剧烈曝光变化（这一条是 §4.2 光度标定价值的直接证据）。实时性：笔记本 CPU 实时；主动点/帧更少的降配版（$N_p = 600$、$N_f = 6$、424×320、每关键帧 ≤4 次 G-N 迭代）跑 5 倍实时仍全面优于间接法（Fig. 10/12 虚线）。
- **光度标定消融**（Fig. 15，定性结论）：已知曝光时间对精度影响小；去掉渐晕 $V$ 或响应 $G$ 标定精度/鲁棒性小幅下降（只去渐晕比两者都去略差）；**"亮度恒常"假设（LSD-SLAM/SVO 式）最差**——完全不处理自动曝光。**数据选择消融**（Fig. 16/17/18，定性）：点数 $N_p > 500$ 后收益迅速趋平（点多了主要是 3D 更密而非更准）；$N_f = 7$ 附近最优，fixed-lag（固定滑最老帧）边缘化显著更差（Eq.(20) 距离得分的价值）；梯度阈值 $g_{th} = 7$ 是甜点；**只用 FAST 角点会显著降低性能**——"能采样边缘与弱纹理"是直接法的真实红利。
- **几何噪声 vs 光度噪声**（§4.3，Eq.(21)-(22) 的人工噪声实验）：模拟 rolling shutter 的低频**几何**噪声（$I'_p(\mathbf{x}) = I(\mathbf{x} + N_p(\mathbf{x}))$，Eq.(21)）使 DSO 性能快速退化而 ORB-SLAM 几乎不受影响——间接法的第一步（局部角点检测）天然过滤几何畸变；模拟非均匀模糊的高频**光度**噪声（Eq.(22)）下 DSO 反而略更鲁棒——联合优化光度误差比逐点描述子匹配更能平均掉光度畸变。结论：**直接法怕几何噪声、间接法怕光度噪声**，各自噪声假设决定各自的软肋（$\delta_\eta > 1.5$ 量级的几何噪声即可能使 DSO 所有残差落出线性化有效半径而完全失败）。

## 6. 局限与后续影响

**局限**（论文自述 + 结构性分析）：

1. **纯视觉里程计，无回环**：永久边缘化离开视场的状态，没有全局一致机制——大回环累计漂移无法消除（这是与 ORB-SLAM 在 EuRoC 上互有胜负的结构原因）。后续工作 **LDSO**（Gao & Zhang, 2018）在 DSO 上加入 DBoW2 回环检测与位姿图优化补齐这一缺口。
2. **对几何噪声敏感**：全局快门假设——rolling shutter、内参不准等几何畸变直接进入光度残差且不被建模（§4.3 的实验量化了这一点；论文给出的出路是把 rolling shutter 建进模型或依赖更精确的机器视觉相机）。
3. **光度标定成本**：完整收益依赖离线标定响应函数与渐晕图（无标定时性能向"亮度恒常"路线退化，Fig. 15）。
4. **非凸性更高**：结论一节明说——能量含图像本身，"likely to restrict the use of our model to video processing"（需前端帧管理 + 金字塔 + 足够好的初值伺候；无 IMU/先验时前端策略仍是启发式的，论文自认 keyframe/marginalization 策略 suboptimal-in-practice）。

**后续影响**：滑窗光度 BA + 边缘化先验 + FEJ 的组合成为视觉（惯性）里程计后端的参考设计（VINS-Mono 等滑窗 VIO 直接沿用该结构）；光度标定模型（Eq.(2)-(3)）确立"photometric calibration"子领域；"direct & sparse"填上了 §1 四象限的最后一格，与 LSD-SLAM（direct & dense）共同终结了"直接法只能稠密、只能 GPU"的成见。

## 7. 与本项目对照

- **教程章节**：[第 06 章](../06_视觉里程计-ii直接法.md)——(6.4)/(6.5) 直接法残差与三因子雅可比（Eq.(11)/(13) 的骨架）、§06.3 ⑤ 的 (6.6)-(6.8)：(6.6) 即本文 Eq.(4) 除以 $t_j e^{a_j}$ 的等价写法、**(6.7) 的仿射不变性证明正是 $(a_i, b_i)$ 规范自由度的来源**、(6.8) 即 Eq.(8) 的滑窗光度 BA；[第 08 章](../08_后端-ii图优化与-ba.md)——(8.5) 正规方程 ↔ Eq.(10)、(8.9)-(8.10) 流形更新与左扰动雅可比 ↔ Eq.(1)/(13)、**(8.15)/(8.17) Schur 补与概率边缘化 ↔ Eq.(17)-(19) 的先验推导**（同一代数、同一概率语义）、(8.12) 块对角 ↔ Eq.(8) "消点后稀疏模式不变"的论证。
- **代码**：[projects/slam/photoba/photoba.py](../../../projects/slam/photoba/photoba.py) 是滑窗光度 BA 的教学实现：`PhotometricBA.residual_vector`（(6.6) 的仿射残差，$t_i \equiv 1$ 的 Eq.(4) 参数代换）、`jacobian_matrix`（解析雅可比：位姿块 = $e^{a_j}$ × 图像梯度 × `direct.projection_jacobian`（(5.11)），逆深度列 $\partial\mathbf{p}/\partial\rho = -\mathrm{ray}/\rho^2$，曝光列 $\partial e/\partial a_j = e^{a_j}I_j$、$\partial e/\partial b_j = 1$——对应 Eq.(13) 的 $\mathbf{J}_I \cdot \mathbf{J}_{geo}$ 与 $\mathbf{J}_{photo}$）、`solve`（写开的流形 LM 循环 + `core.solver.huber_weights` 的 Huber IRLS ↔ Eq.(4) 的 $\|\cdot\|_\gamma$，Marquardt 对角阻尼 + Nielsen 增益比 + 回溯线搜索）、逆深度点 `p = ray/ρ`（§4.5 的 1 维点）。
- **刻意差距**（教学取舍，代码 docstring 有声明）： 窗口 3 帧（vs 论文 $N_f = 7$）且点全部 host 在第 0 帧（vs 多宿主帧分配）； **未实现边缘化**——规范自由度由锚点直接钉死（$T_0$ 固定、$(a_0, b_0) = (0, 0)$、0 号点逆深度固定作尺度锚）而非 Eq.(15)-(19) 的先验注入：这正是把 §4.5 的"先验保 gauge"换成"锚点保 gauge"的最小对照实验，两者是同一问题的两种钉法； 无光度标定（$G, V$）——合成场景直接以仿射 $(g_i, o_i)$ 施加曝光，故只有 Eq.(4) 的仿射层有对应物； 不用 FEJ（小窗口稠密正规方程上每轮重线性化无害，FEJ 的价值在窗口大 + 多轮边缘化时才显现）。
- **健康值**（实测，`tests/test_photoba.py`；[METRICS.md](../../../projects/slam/METRICS.md) 健康值表未单列 photoba——其断言是端到端恢复精度，详录于 [DEBUG.md](../../../projects/slam/DEBUG.md) 案例 B-4）：真值恢复 位姿平移误差 **4.11e-3 m**、旋转 **1.30e-3 rad**、点中位误差 7.25e-3 m、曝光 $a$ 误差 **3.23e-3**（< 1e-2）、$b$ 误差 3.76e-1（< 0.6，渲染底限主导）；**关掉曝光优化的对照实验**：位姿误差 8.89e-2 m，劣化约 **22 倍**——仿射亮度参数存在性的活体样本（(6.7) 的数值印证）；Huber 对照：外点场景下位姿误差 1.05e-2 vs 无核 3.28e-2（约 1/3）；雅可比对有限差分最差相对偏差 < 5e-9。

## 配套阅读

- 本文 PDF：[DSO（Engel, Koltun & Cremers, arXiv:1607.02565 / TPAMI 2018）](../../../papers/slam/classics/arXiv-1607.02565_DSO.pdf)（本精读所有式号以此 PDF 为准）
- 姊妹精读：[LSD-SLAM（ECCV 2014）](./LSD-SLAM_ECCV2014.md)（同作者的"回避曝光"前作：半稠密 + Sim(3) 位姿图，本文多篇引文出处）｜[PTAM（ISMAR 2007）](./PTAM_ISMAR2007.md)（关键帧 + BA 架构的源头）｜[ORB-SLAM（T-RO 2015）](./ORB-SLAM_TRO2015.md)｜[ORB-SLAM2（T-RO 2016）](./ORB-SLAM2_TRO2016.md)｜[FastSLAM（AAAI 2002）](./FastSLAM_AAAI2002.md)
- 教程：[第 06 章](../06_视觉里程计-ii直接法.md)（(6.4)-(6.8)：直接法 → 光度模型 → 滑窗光度 BA）｜[第 08 章](../08_后端-ii图优化与-ba.md)（(8.5)/(8.9)-(8.10)/(8.15)/(8.17)：G-N、流形、Schur 补即边缘化）｜[第 05 章](../05_视觉里程计-i特征点法.md)（(5.9) 尺度规范自由度、(5.11) 几何雅可比）｜[第 09 章](../09_回环检测.md)（LDSO 所补的回环机制）
- 代码：[projects/slam/photoba/](../../../projects/slam/photoba/)（本文后端核心）｜[projects/slam/direct/](../../../projects/slam/direct/)（复用的半稠密选取与几何雅可比）｜[projects/slam/METRICS.md](../../../projects/slam/METRICS.md) ｜ [projects/slam/DEBUG.md](../../../projects/slam/DEBUG.md)（photoba 调试断点表）
- 主题导航：[tutorials/slam/README.md](../README.md) ｜ [OVERVIEW.md](../OVERVIEW.md)

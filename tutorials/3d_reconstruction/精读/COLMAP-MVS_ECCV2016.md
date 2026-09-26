# 论文精读｜COLMAP-MVS：非结构化多视图立体的逐像素视图选择（ECCV 2016）

> **PDF**：[papers/3d_reconstruction/classics/COLMAP-MVS_ECCV2016_SchoenbergerFischer.pdf](../../../papers/3d_reconstruction/classics/COLMAP-MVS_ECCV2016_SchoenbergerFischer.pdf) ｜ **教程**：[第 03 章｜多视图立体重建 MVS](../03_多视图立体重建MVS.md) ｜ **代码**：无（教学实现规划中；[projects/slam/photoba/](../../../projects/slam/photoba/) 的光度残差思想同源，见 §7）

## 1. 论文信息与一句话贡献

- **题目**：Pixelwise View Selection for Unstructured Multi-View Stereo
- **作者**：Johannes L. Schönberger（ETH Zürich）、Enliang Zheng（UNC Chapel Hill）、Marc Pollefeys（ETH / Microsoft）、Jan-Michael Frahm（UNC Chapel Hill）
- **发表**：ECCV 2016（本仓库收录 PDF 共 17 页：正文 14 页 + 参考文献，全部式号已逐条对照原文）；开源实现集成于 COLMAP
- **一句话贡献**：把"选哪些源视图来匹配"从**每张图一次的预处理**变成**每个像素一次的在线推断**——在 Zheng et al. [59] 的变分 PatchMatch 框架上，联合估计深度与法向，用几何先验（三角化角 / 分辨率 / 入射角）与光度遮挡指示共同驱动**逐像素视图选择**，再以时域平滑抑制震荡、以多视图几何一致性内嵌过滤，最后在一致像素图上做传递式深度/法向融合——在 Middlebury / Strecha 基准与互联网照片集上同时取得精度、完备性与效率的领先。

## 2. 问题与动机

**要解决什么**：无序照片集（异构相机、分辨率悬殊、视角杂乱、误配准）的稠密重建。稠密像素对应搜索是 MVS 的核心难题，而在非受控环境下，即使几何已知，遮挡、光照差异、分辨率差异也让"这条对应可信吗"成为首要问题。

**视图选择为何是核心**：MVS 天然面临多视图遮挡问题——参考像素只应与**看得见同一表面**的源视图比较（Sec. 1–2）。已有策略的缺陷：
- Kang et al. [29]：每像素贪心选最优视图（算完深度才知道该选谁），内存与计算双贵；
- Strecha et al. [46]：概率可见性模型 + MRF，无光度先验与法向估计，不可行于大规模；
- Goesele et al. [19]、Furukawa et al. [12]：按稀疏点云**预选**源图——但稀疏场景表示不完整，预选在遮挡边界、三角化角、分辨率、入射角剧烈变化的像素处**次优**（Sec. 4.2 论证）；
- Zheng et al. [59]（本文直接基座）：逐像素遮挡推断，但视图采样**纯按颜色相似度**——零基线视图颜色最像却对深度零信息，是退化偏好；用前向平行单应 warp，斜面上有伪影；无融合阶段。

**四项改进由此展开**（Sec. 1 贡献列表）：逐像素法向估计（改进 PatchMatch 采样）、逐像素几何先验视图选择、"时域"视图选择平滑（消除震荡）、多视图几何一致性项 + 基于图像的过滤与融合。

## 3. 方法总览

```
输入：SfM 恢复的标定图像集 X（位姿/内参已知）
   ▼
【第一阶段：逐图初始深度+法向推断（§4.1，Eq.(5)）】 I1 = 3 遍 sweep
   对参考图逐像素（GEM 交替 E/M 步）：
     E 步：forward-backward 消息推断遮挡指示 Z（Eq.(11)–(13)）
     M 步：PatchMatch 传播 + 采样选 (θ, n)（Eq.(6) 假设集；斜平面单应 warp）
     视图采样分布 P_l(m)：q(Z)·q(α)·q(β)·q(κ)（Eq.(7)，几何先验 + 光度遮挡）
   ▼
【第二阶段：坐标下降式几何一致性推断（§4.5，Eq.(10)）】 I2 = 2 遍 sweep
   逐参考图轮换，其余图固定；代价 ξ = 1 − ρ + η·min(ψ, ψ_max)
   ψ = 前向-回投往返重投影误差（跨视图深度/法向一致性）
   ▼
【过滤（Eq.(14)–(15)）】光度支持集 ∩ 几何支持集，|S_l| < s 的像素剔除
【融合（§4.7）】一致像素有向图 → 从最大支持节点递归聚类（深度/法向/重投影三约束）
   → 位置取中位数、法向取均值 → 稠密点云（可上 Poisson 网格化）
```

sweep = 沿行/列四个方向的传播各一次；两阶段均 E/M 交错。

## 4. 关键公式推导（式号按论文 PDF 原文）

### 4.0 符号表

| 符号 | 含义（首次出现处） |
|---|---|
| $L$，$M$ | 参考图像素数；源视图数（Sec. 3） |
| $\boldsymbol{x}_l \in \mathcal{P}^2$，$\boldsymbol{x}_l^m$ | 参考图像素 $l$ 及其在源图 $m$ 的 warp 像素（Sec. 4.1） |
| $\theta_l$，$\boldsymbol{n}_l \in \mathbb{R}^3$, $\|\boldsymbol{n}_l\|=1$ | 像素 $l$ 的深度假设与法向假设（Sec. 4.1） |
| $\boldsymbol{p}_l = \theta_l K^{-1}\boldsymbol{x}_l$ | 像素 $l$ 反投影的三维点（Sec. 4.1） |
| $d_l = \boldsymbol{n}_l^\top \boldsymbol{p}_l$ | 参考图到像素 $l$ 处斜平面的正交距离（Sec. 4.1） |
| $H_l^m = K^m\big(R^m - d_l\,\boldsymbol{t}^m \boldsymbol{n}_l^\top\big)K^{-1}$ | 参考图到源图 $m$ 的（斜）平面诱导单应（Sec. 4.1） |
| $Z_l^m \in \{0,1\}$ | 遮挡指示：源图 $m$ 是否看见 $\boldsymbol{x}_l$ 处表面（Sec. 3） |
| $\rho_l^m \in [-1,1]$ | 参考块与源块的（双边加权）NCC 相似度（Sec. 4.4，Eq.(9)） |
| $\alpha_l^m$，$\beta_l^m$，$\kappa_l^m$ | 三角化角 / 相对分辨率 / 入射角（Sec. 4.2） |
| $P_l(m)$ | 视图采样分布（Sec. 4.2，Eq.(7)） |
| $\lambda_t$，$T$ | 时域平滑系数与总迭代数（Sec. 4.3） |
| $\psi_l^m$，$\xi_l^m$ | 前向-回投往返重投影误差；鲁棒化几何代价（Sec. 4.5，Eq.(10)） |
| $S_l^{pho}$，$S_l^{geo}$，$S_l$ | 光度 / 几何 / 有效支持集（Sec. 4.7，Eq.(14)–(15)） |

### 4.1 基座框架回顾与法向联合估计（Sec. 3–4.1，Eq.(1)–(6)）

**Zheng et al. 的逐像素模型**（Sec. 3，Eq.(1)）：给定遮挡指示与深度，参考块与源块颜色分布满足以残差为中心的高斯、遮挡时退化为均匀分布 $\mathcal{U}$：

$$
P\big(X_l^m \mid Z_l^m, \theta_l\big) = \begin{cases} \frac{1}{\sqrt{2\pi}}\frac{1}{\sigma_\rho}\exp\!\Big(-\frac{(1-\rho_l^m(\theta_l))^2}{2\sigma_\rho^2}\Big) & \text{if } Z_l^m = 1 \\[1ex] \frac{1}{\sqrt{M}}\,\mathcal{U} & \text{if } Z_l^m = 0 \end{cases}
\tag{1}
$$

（依据：把"块相似度 $\rho \to 1$"当作"观测由同一表面产生"的似然——相似度高 ⟹ 残差 $(1-\rho)$ 小 ⟹ 高斯项大；$A = \int_{-1}^{1}\exp(-\frac{(1-\rho)^2}{2\sigma_\rho^2})\,d\rho$ 归一化常数在推断中抵消。）联合似然沿像素行分解（Eq.(2)）：

$$
P(\boldsymbol{X}, \boldsymbol{Z}, \boldsymbol{\theta}) = \prod_{l=1}^{L}\prod_{m=1}^{M}\Big[P\big(Z_l^m \mid Z_{l-1}^m\big)\, P\big(X_l^m \mid Z_l^m, \theta_l\big)\Big]
\tag{2}
$$

（依据：图模型的条件独立结构——空间平滑进状态转移 $P(Z_l^m|Z_{l-1}^m)$，光度证据进 $P(X_l^m|\cdot)$。）[59] 用变分推断（GEM）解 $\boldsymbol{\theta}$；全深度推断（Eq.(3)）在 $M$ 大时算不起（每候选都要算 NCC），故蒙特卡洛子采样（Eq.(4)）：

$$
\hat{\theta}_l^{opt} = \arg\min_{\theta_l^*} \frac{1}{|S|}\sum_{m \in S}\big(1 - \rho_l^m(\theta_l^*)\big)
\tag{4}
$$

（依据：以 $\frac{1}{|S|}$ 代 $P(m)$ 权重——只对按分布 $P_l(m)$ 采出的 $S \subset \{1..M\}$ 个视图算 NCC，代价线性于 $|S|$。）

**本文改进一：深度+法向联合估计（Eq.(5)）**。把每像素一个未知（深度）扩为三个（深度 + 法向两参数），warp 从前向平行单应换成**斜平面单应** $\boldsymbol{x}_l^m = H_l^m \boldsymbol{x}_l$，$H_l^m = K^m(R^m - d_l\,\boldsymbol{t}^m\boldsymbol{n}_l^\top)K^{-1}$：

$$
\big(\hat{\theta}_l^{opt},\ \hat{\boldsymbol{n}}_l^{opt}\big) = \arg\min_{\theta_l^*,\, \boldsymbol{n}_l^*} \frac{1}{|S|}\sum_{m \in S}\big(1 - \rho_l^m(\theta_l^*,\, \boldsymbol{n}_l^*)\big)
\tag{5}
$$

（$H_l^m$ 与 [教程第 03 章 (3.2)](../03_多视图立体重建MVS.md) 完全同构——同一"平面约束消元"推导，仅把固定法向换成逐像素法向 $\boldsymbol{n}_l$、前向平面 $d$ 换成斜平面距离 $d_l = \boldsymbol{n}_l^\top\boldsymbol{p}_l$；教程 3.1 末"变深度时的处理"预告的正是这一步。法向先验取均匀 $P(\boldsymbol{N})$。）

**本文改进二：传播方案（Eq.(6)）**。逐像素三未知理论上要求 PatchMatch 采样数暴涨；本文利用"深度+法向定义局部平面"做**沿表面传播**：当前像素光线与前一像素的局部平面 $(\theta_{l-1}, \boldsymbol{n}_{l-1})$ 求交。该交点深度可显式写出——参考相机系下光线为 $\boldsymbol{X} = \mu K^{-1}\boldsymbol{x}_l$（$K^{-1}\boldsymbol{x}_l$ 第三分量为 1，$\mu$ 即 $z$ 深度；依据：$\boldsymbol{p}_l = \theta_l K^{-1}\boldsymbol{x}_l$ 的定义），前一像素平面满足 $\boldsymbol{n}_{l-1}^\top(\boldsymbol{X} - \boldsymbol{p}_{l-1}) = 0$（依据：点法式平面方程），代入解 $\mu$：

$$
\theta_{l}^{prp} \;=\; \frac{\boldsymbol{n}_{l-1}^\top \boldsymbol{p}_{l-1}}{\boldsymbol{n}_{l-1}^\top K^{-1}\boldsymbol{x}_l} \;=\; \frac{d_{l-1}}{\boldsymbol{n}_{l-1}^\top K^{-1}\boldsymbol{x}_l}
\tag{4.1}
$$

（(4.1) 是论文文字描述 "propagate the depth of the intersection of the ray of the current pixel with the local surface of the previous pixel" 的闭式展开；每步依据：参数化代入 + 一元线性方程求解；分母非零 ⟺ 光线不与平面平行。直觉：正确的深度沿真实表面传播得比随机采样快得多——一阶光滑性。）每个传播步从**假设集**（Eq.(6)，照录原文）中按 Eq.(4) 型准则选优：

$$
\big\{(\theta_{l,t}, \boldsymbol{n}_l),\ (\theta_{l-1,t}^{prp}, \boldsymbol{n}_{l-1}),\ (\theta_l^{rnd}, \boldsymbol{n}_l),\ (\theta_l^{prt}, \boldsymbol{n}_l^{rnd}),\ (\theta_l^{prt}, \boldsymbol{n}_l),\ (\theta_l, \boldsymbol{n}_l^{prt})\big\}
\tag{6}
$$

其中上标 $rnd$ 为随机采样（无偏随机法向依 Galliani et al. [15]），$prt$ 为扰动：$\theta^{prt} = (1 \pm \epsilon)\theta$、$\boldsymbol{n}^{prt} = R_\epsilon\,\boldsymbol{n}$（$R_\epsilon \in SO(3)$ 小角旋转，约束 $\boldsymbol{p}_l^\top\boldsymbol{n}_l^{prt} < 0$ 保证平面在可见侧）。设计依据（论文原文）：当前最优的深度与法向各自可能"已优 / 未优"，把随机/扰动深度与当前法向交叉组合，覆盖"两者之一或均已近优"的全部状态——省去 [5,15] 在整遍 sweep 之间插入的二分法向精化中间步。

### 4.2 逐像素几何先验与联合视图采样分布（Sec. 4.2，Eq.(7)）——本文核心贡献

**为什么逐像素**：预选源图基于稀疏（因而残缺）的场景表示；遮挡边界、三角化角、分辨率、入射角在**同一对图像内部**逐像素剧变（论文 Fig. 2/Fig. 4 可视化）——故先验必须挂在像素上、与遮挡推断 $Z$ 联合。

**先验一：三角化角（Triangulation Prior）**。纯光度采样偏好小基线（颜色最像），但零基线对深度推断零信息——重建点可沿光线任意滑动而不改颜色相似度（论文原文论证）。故对源图 $m$ 计算两视线在 $\boldsymbol{p}_l$ 处的张角：

$$
\alpha_l^m = \cos^{-1}\frac{(\boldsymbol{p}_l - \boldsymbol{c}^m)^\top \boldsymbol{p}_l}{\|\boldsymbol{p}_l - \boldsymbol{c}^m\|_2\,\|\boldsymbol{p}_l\|_2}, \qquad \boldsymbol{c}^m = -(R^m)^\top \boldsymbol{t}^m
\tag{4.2}
$$

（(4.2) 照录原文符号；分子是"相机中心到点"与"光心到点"两向量的内积，除以模长即夹角余弦——$\boldsymbol{c}^m = -(R^m)^\top\boldsymbol{t}^m$ 为源相机光心（依据：世界原点在相机系的坐标为 $-\!R^\top\boldsymbol{t}$，反解即得光心，与 [教程第 02 章 (2.5)](../02_多视图几何与SfM.md) 的光心公式同构、与 SfM 精读式 (3) 的角度判据同源）。）似然取分段形式：

$$
P(\alpha_l^m) = 1 - \frac{\big(\min(\bar{\alpha},\, \alpha_l^m) - \bar{\alpha}\big)^2}{\bar{\alpha}^2}
$$

（依据：$\alpha_l^m \ge \bar{\alpha}$ 时 $\min = \bar{\alpha}$ ⟹ $P = 1$，不施加额外偏好；$\alpha_l^m < \bar{\alpha}$ 时 $P = 1 - \frac{(\alpha_l^m - \bar\alpha)^2}{\bar\alpha^2} \in (0,1)$，角度越小似然越低——正好压制 SfM 精读 4.3 推导的"小三角化角 ⟹ 深度不确定度 $\propto 1/\sin\alpha$"病态。阈值实验值 $\bar{\alpha} = 1^\circ$。）

**先验二：分辨率（Resolution Prior）**。参考/源块面积比 $\beta_l^m = b_l / b_l^m \in \mathbb{R}^+$（近似度量块的大小与形状差），似然取对称折衰：

$$
P(\beta_l^m) = \min\big(\beta_l^m,\ (\beta_l^m)^{-1}\big)
$$

（依据：$\beta = 1$ 即大小形状最匹配 ⟹ $P = 1$；无论过采样还是欠采样，偏离 1 都使 $P < 1$——min 使两侧对称衰减。欠采样本可用自适应重采样处理，但计算更贵，故以先验降权代替。）

**先验三：入射角（Incident Prior）**。法向把源相机位置约束在表面正半空间、视线须迎面。源相机入射角：

$$
\kappa_l^m = \cos^{-1}\frac{\big|-(\boldsymbol{p}_l - \boldsymbol{c}^m)^\top \boldsymbol{n}_l^m\big|}{\|\boldsymbol{p}_l - \boldsymbol{c}^m\|_2\,\|\boldsymbol{n}_l^m\|_2} \in [0, \pi)
$$

几何可见性要求 $0 \le \kappa_l^m < \pi/2$；似然取高斯衰减 $P(\kappa_l^m) = \exp\!\big(-\frac{\kappa_l^m \cdot \kappa_l^m}{2\sigma_\kappa^2}\big)$（$\sigma_\kappa = 45^\circ$）。**为何 $\kappa \ge \pi/2$ 仍留正信念**（论文原文）：推断初期 $\theta_l$、$\boldsymbol{n}_l^m$ 尚不准，几何约束本身不可靠——直接归零会过早杀死可能纠正回来的视图。

**联合采样分布（Eq.(7)）——光度与几何的逐像素联合评分**：

$$
P_l(m) = \frac{q_l(Z_l^m = 1)\; q_l(\alpha_l^m)\; q_l(\beta_l^m)\; q_l(\kappa_l^m)}{\sum_{m=1}^{M} q_l(Z_l^m = 1)\; q_l(\alpha_l^m)\; q_l(\beta_l^m)\; q_l(\kappa_l^m)}
\tag{7}
$$

（$q(\cdot)$ 是变分推断中对真实后验的近似——以最小化 KL 散度为意义；归一化分母只保证 $\sum_m P_l(m) = 1$，各因子本身无需归一，因为只用作采样调制器。）三个显式假设（论文原文自述）：各先验统计独立（简化近似，换来简单模型）；"不遮挡 + 基线足 + 分辨率近 + 迎面"的视图被联合 favore；与 (4) 的蒙特卡洛子采样相乘后，**NCC 只为高分视图计算**——视图选择本身就是算力分配器。

### 4.3 视图选择的时域平滑（Sec. 4.3，Eq.(8)）

行/列交替传播使 $Z_l^m$ 随传播方向**震荡**（条带伪影，Fig. 5）。对策：给图模型加"时域"转移——$Z_{l,t}^m$ 不仅依赖邻接像素 $l-1$，还依赖自身上一轮 $t-1$ 的状态：

$$
P\big(Z_{l,t}^m \mid Z_{l,t-1}^m\big) = \Big(\tfrac{\lambda_t}{1-\lambda_t},\ \tfrac{1-\lambda_t}{\lambda_t}\Big), \qquad \lambda_t = \frac{t}{2T} + 0.5
\tag{4.3}
$$

（(4.3) 中二元组为"保持 / 翻转"概率：$\lambda_t \to 1$ 时保持概率 $\to \infty$ 倍于翻转——强时域惯性；$\lambda_t$ 从 $t=1$ 的 $\approx 0.5$（中性）线性升向 $t = T-1$ 的 $\to 1$（论文原文 "iterations $t=1$ and $t=T-1$ have maximal and minimal influence"），即推断早期允许改判、后期锁定收敛解。）与空间转移联合建模（Eq.(8)，照录原文）：

$$
P\big(Z_{l,t}^m \mid Z_{l-1,t}^m,\, Z_{l,t-1}^m\big) = P\big(Z_{l,t}^m \mid Z_{l-1,t}^m\big)\, P\big(Z_{l,t}^m \mid Z_{l,t-1}^m\big)
\tag{8}
$$

（依据：两个条件独立的马尔可夫链按链式法则分解——空间链与时间链相乘。）

### 4.4 光度一致性：双边加权 NCC（Sec. 4.4，Eq.(9)）

NCC 统计上对高斯噪声最优、且对逐视图光照仿射差异不变——不变性的完整推导（减均值消亮度 $b$、除标准差消对比度 $a$）见 [教程第 03 章 (3.3)–(3.5)](../03_多视图立体重建MVS.md)，此处不重推。本文的新意是**双边加权**（修正 NCC 在深度不连续处模糊的已知缺陷 [23]，思想承自 [5,54]）：参考块 $\boldsymbol{w}_l$（中心 $\boldsymbol{x}_l$）对源块 $\boldsymbol{w}_l^m$（中心 $\boldsymbol{x}_l^m$）计算

$$
\rho_l^m = \frac{\mathrm{cov}_{\boldsymbol{w}}(\boldsymbol{w}_l,\ \boldsymbol{w}_l^m)}{\sqrt{\mathrm{cov}_{\boldsymbol{w}}(\boldsymbol{w}_l,\ \boldsymbol{w}_l)\ \mathrm{cov}_{\boldsymbol{w}}(\boldsymbol{w}_l^m,\ \boldsymbol{w}_l^m)}}, \qquad \mathrm{cov}_{\boldsymbol{w}}(\boldsymbol{x}, \boldsymbol{y}) = E_{\boldsymbol{w}}(\boldsymbol{x}\boldsymbol{y}) - E_{\boldsymbol{w}}(\boldsymbol{x})\,E_{\boldsymbol{w}}(\boldsymbol{y})
\tag{9}
$$

其中加权均值 $E_{\boldsymbol{w}}(\boldsymbol{x}) = \sum_i w_i x_i / \sum_i w_i$，逐像素权重

$$
w_i = \exp\Big(-\frac{\Delta g_i^2}{2\sigma_g^2} - \frac{\Delta x_i^2}{2\sigma_x^2}\Big), \qquad \Delta g_i = |g_i - g_l|,\ \ \Delta x_i = \|\boldsymbol{x}_i - \boldsymbol{x}_l\|
$$

（$w_i$ 是"像素 $i$ 与中心像素同平面"的似然：颜色距/空间距越大权重越小——即双边核。两处 $\sigma_g, \sigma_x$ 为高斯散差。）两个补充事实： 其一，(9) 的协方差两种写法恒等——$\sum_i w_i (x_i - E_w x)(y_i - E_w y) / \sum_i w$ 展开为 $\sum_i w_i x_i y_i/\sum w - E_w(x)E_w(y)$（依据：对"中心化乘积"做双线性展开 $x_iy_i - x_iE_w y - y_iE_w x + E_wxE_wy$，逐项求加权和后中间两项均塌缩为 $\sum w\, E_wxE_wy$——代回即论文的 $E_w(xy) - E_w(x)E_w(y)$ 形式）； 其二，$\rho_l^m$ 是加权内积下的 Pearson 相关系数，由柯西–施瓦茨不等式保证 $\rho_l^m \in [-1, 1]$（依据：加权内积的正定性），故 Eq.(1) 的残差 $(1 - \rho_l^m) \ge 0$ 有界。光度代价仍进入 $1 - \rho$ 型目标（Eq.(5)/(10)），与 [教程 3.2 的 $C(d) = 1 - \mathrm{NCC}(d)$](../03_多视图立体重建MVS.md) 同形。

### 4.5 多视图几何一致性（Sec. 4.5，Eq.(10)）

**动机**：光度代价对粗大离群迟钝——深度大幅变动只引起代价小幅变化；左右一致性检验若只做事后过滤则无法反哺推断。本文把跨视图几何一致性**内嵌**进代价：参考估计 warp 到源图、用源图自身的估计 warp 回来，看往返误差：

$$
\psi_l^m = \big\|\, \boldsymbol{x}_l - H_l^m\, H_l\, \boldsymbol{x}_l \,\big\|
$$

（$H_l$：由参考图估计 $(\theta_l, \boldsymbol{n}_l)$ 构造的正向单应（参考→源，即 4.1 的 $H_l^m$ 在 $m \to$ 参考自身时的形式）；$H_l^m$：由源图估计 $(\theta_l^m, \boldsymbol{n}_l^m)$ 在前向投影 $\boldsymbol{x}_l^m = H_l\boldsymbol{x}_l$ 处**插值**后构造的反向单应（源→参考）。依据：两段单应复合是"参考→源→参考"的往返映射，深度/法向一致时往返闭合 $\psi \to 0$。）源图侧无法顾及遮挡，故鲁棒化截断——几何代价与光度合并：

$$
\xi_l^m = 1 - \rho_l^m + \eta\, \min\big(\psi_l^m,\ \psi_{max}\big), \qquad \eta = 0.5,\ \ \psi_{max} = 3\,\mathrm{px}
$$

最优深度与法向由合并代价选出（Eq.(10)，照录原文）：

$$
\big(\hat{\theta}_l^{opt},\ \hat{\boldsymbol{n}}_l^{opt}\big) = \arg\min_{\theta_l^*,\, \boldsymbol{n}_l^*} \frac{1}{|S|}\sum_{m \in S} \xi_l^m\big(\theta_l^*,\, \boldsymbol{n}_l^*\big)
\tag{10}
$$

（依据：与 Eq.(5) 同构，仅光度残差 $1-\rho$ 换成 $\xi$；$\min(\cdot, \psi_{max})$ 截断使遮挡源图至多贡献常数 $\eta\psi_{max}$，不会拖偏解——鲁棒核思想的几何版，对照 [SLAM 精读 DSO 篇的核函数角色](../../slam/精读/DSO_PAMI2018.md)。）一致性项在似然中建模为 $P(\theta_l, \boldsymbol{n}_l \mid \theta_l^m, \boldsymbol{n}_l^m)$。实现上，几何一致推断无法同时对全部图做（内存），故两阶段坐标下降：第一阶段按 Eq.(5) 逐图初始化（$I_1 = 3$ 遍 sweep）；第二阶段按 Eq.(10) 逐参考图轮换、其余图固定（$I_2 = 2$ 遍）。

### 4.6 整体推断：变分框架与前向-后向消息（Sec. 4.6，Eq.(11)–(13)）

联合似然把 Eq.(8) 的双转移、Eq.(1) 型光度项（斜平面版）与几何一致性项全部乘起来（Sec. 4.6 开头的连乘式）。真后验 $P(\boldsymbol{Z}, \boldsymbol{\theta}, \boldsymbol{N})$ 以 Kronecker-delta 族 $q(\theta, \boldsymbol{n}) = \delta(\theta - \theta_l^*,\, \boldsymbol{n} - \boldsymbol{n}_l^*)$ 近似（依据：变分族限制为逐像素点估计——GEM 的 M 步即 PatchMatch 选优），E 步用前向-后向算法沿像素行/列推遮挡边缘（Eq.(11)–(13)，照录原文）：

$$
q\big(Z_{l,t}^m\big) = \frac{1}{A}\,\overrightarrow{m}\big(Z_{l,t}^m\big)\,\overleftarrow{m}\big(Z_{l,t}^m\big)
\tag{11}
$$

$$
\overrightarrow{m}\big(Z_l^m\big) = P\big(X_l^m \mid Z_l^m, \theta_l, \boldsymbol{n}_l\big) \sum_{Z_{l-1}^m} \overrightarrow{m}\big(Z_{l-1}^m\big)\, P\big(Z_{l,t}^m \mid Z_{l-1,t}^m,\, Z_{l,t-1}^m\big)
\tag{12}
$$

$$
\overleftarrow{m}\big(Z_l^m\big) = \sum_{Z_{l+1}^m} \overleftarrow{m}\big(Z_{l+1}^m\big)\, P\big(X_{l+1}^m \mid Z_{l+1}^m, \theta_{l+1}, \boldsymbol{n}_{l+1}\big)\, P\big(Z_{l,t}^m \mid Z_{l+1,t}^m,\, Z_{l,t-1}^m\big)
\tag{13}
$$

（无信息先验 $\overrightarrow{m}(Z_0^m) = \overleftarrow{m}(Z_{L+1}^m) = 0.5$；$A$ 为归一化常数。每步依据：隐马尔可夫链的边缘推断——(11) 把链在 $l$ 处分成"左证据 × 右证据"，(12)/(13) 分别是按转移概率递推的链式法则展开；两方向消息相乘即融合两侧全部观测。）$q(Z)$ 连同 $q(\alpha), q(\beta), q(\kappa)$ 进 Eq.(7) 的采样分布，M 步按 Eq.(6) 做 PatchMatch——E/M 两步闭环。

### 4.7 过滤与融合（Sec. 4.7，Eq.(14)–(15) 与三条聚类判据）

**支持集**：一个内点观测应同时光度稳定、几何稳定、有多视图支持（论文原文判据，Eq.(14)–(15) 照录）：

$$
S_l^{pho} = \big\{\boldsymbol{x}_l^m \ \big|\ q(Z_l^m) > \bar{q}_Z\big\}
\tag{14}
$$

$$
S_l^{geo} = \big\{\boldsymbol{x}_l^m \ \big|\ q(\alpha_l^m) \ge \bar{q}_\alpha,\ q(\beta_l^m) \ge \bar{q}_\beta,\ q(\kappa_l^m) > \bar{q}_\kappa,\ \psi_l^m < \psi_{max}\big\}
\tag{15}
$$

有效支持 $S_l = \{\boldsymbol{x}_l^m \mid \boldsymbol{x}_l^m \in S_l^{pho} \wedge \boldsymbol{x}_l^m \in S_l^{geo}\}$；$|S_l| < s$ 的像素整点剔除（实验值 $s = 3$，$\bar{q}_Z = 0.5$，$\bar{q}_\alpha = 1$，$\bar{q}_\beta = 0.5$，$\bar{q}_\kappa = P(\kappa = 90^\circ)$）。由于 $q(\cdot)$ 已在推断中算出，过滤几乎零附加代价（论文原文 "at negligible computational cost"）。

**传递式融合**：全部图的支持集拼成一张"一致像素有向图"（节点 = 有足够支持的像素，边 = 参考像素指向其源图对应像素，边携单应变换）。融合 = 在图上找一致像素簇，**递归**执行：
1. 取支持度 $|S|$ 最大的节点作种子，反投影到三维得 $(\boldsymbol{p}_0, \boldsymbol{n}_0)$；
2. 对候选节点 $i$ 依次检验三条判据（论文原文照录）：
   - **深度一致**：种子投影到 $i$ 的图像得深度 $\tilde{\theta}_0$，与 $i$ 自身估计深度 $\theta_i$ 的**相对差**须满足
     $$\frac{\big|\tilde{\theta}_0 - \theta_i\big|}{\tilde{\theta}_0} < \epsilon_\theta$$
     （相对差而非绝对差：允许深度随尺度缩放的容差传递；依据：论文引 [32] 的同款判据）；
   - **法向一致**：$1 - \boldsymbol{n}_0^\top \boldsymbol{n}_i < \epsilon_n$（切平面夹角余弦接近 1——与 [教程 3.2(c) 的法向一致性筛选](../03_多视图立体重建MVS.md)同一逻辑）；
   - **重投影一致**：$\boldsymbol{p}_0$ 在 $i$ 图像中的重投影误差 $\psi_i < \bar{\psi}$（往返闭合的同款判据，见 4.5）；
3. 簇内融合：位置取**中位数** $\hat{\boldsymbol{p}}_i$、法向取均值 $\hat{\boldsymbol{n}}_i$（中位数防大深度不连续处邻域平均的拖影伪影——论文原文 "avoid artifacts when averaging over multiple neighboring pixels at large depth discontinuities"）；
4. 移除已融合节点，回到第 1 步直至图空。输出为着色稠密点云，可接 Poisson 网格化。

## 5. 实验与结果解读

- **实现与参数**：CUDA 实现于 Nvidia Titan X；$\gamma = 0.999$（平均每 1000 像素一次遮挡状态翻转）、$\sigma_\rho = 0.6$、$\bar{\alpha} = 1^\circ$、$\sigma_\kappa = 45^\circ$；两阶段 $I_1 = 3$、$I_2 = 2$ 遍 sweep。
- **组件消融**（South Building：128 张 7 MP，127 张全作源视图，平均每 sweep 50 s）：法向估计提升斜面（地面等）的完备性与精度，且新采样方案收敛 sweep 数与 [59] 相同、仅多约 25% 运行时间（假设更多）；三个几何先验在**同一源图内部**呈逐像素变化的似然，正确降权小基线/低分辨率/遮挡视图；时域平滑消除纯空间项的方向性震荡、稳定视图采样；几何一致性既提完备性又使过滤结果"几乎无离群"。
- **Middlebury 基准**（Dino / Temple，640×480，Full/Ring/Sparse；Full 每视图约 40 s、约 300 源视图）：标准设置下 **Dino Full（并列）与 Dino Sparse 排名第 1**，Temple Full 第 4、Ring 第 8；高分辨率最有利（法向估计需要大 patch）；仅用基础 Poisson 网格化即取得该成绩。
- **Strecha 基准**（按 Hu & Mordohai 协议，报告误差 <2 cm / <10 cm 的像素比例；对手含 PMVS [13]、Gipuma [15]、CMPMVS [26]、Zheng et al. [59] 等）：四项（Fountain 2cm/10cm、Herz-Jesu 2cm/10cm）中三项最优（0.827 / 0.975 / 0.691 / 0.931）；唯一未拿下的一项（Herz-Jesu 2cm）低于 CMPMVS（0.739 vs 0.691）。对 PMVS 与 Gipuma 在四项上均占优（定性：Gipuma 在高分辨率、大视差场景的该项协议下差距明显）。
- **互联网照片集**（Heinly et al. 100M 数据集）：单机 4× Titan X 处理 41K 图像，每视图 70 s（每 GPU 2 线程），稠密重建 4.2 天（另加 SfM 6 天）；GPU 显存上限约 200 源视图（按共享稀疏点数选最连通者）；融合与过滤耗时可忽略。

**指标口径注记**：本文使用"误差小于阈值的像素比例"（Strecha）与 accuracy/completeness 排名（Middlebury），未使用 F-score；"F-score 比较"常见于后续 MVS 综述对本文的转述，不宜回填为本文结论。

## 6. 局限与后续影响

- **局限**：逐像素 PatchMatch + 变分推断的计算仍重（40–70 s/视图）；弱纹理、反射等光度失效面依旧无判别力（NCC 的本性）；法向估计需大 patch，低分辨率不利；融合是点云级而非全局一致面元，网格化需外接 Poisson。
- **后续影响**：确立"视图选择逐像素化 + 几何/光度联合推断"的范式，成为 COLMAP `patch_match_stereo` 的理论基础与开源 MVS 的默认基线；其平面扫描 warp、一致性过滤与深度图融合被 MVSNet 系（学习式代价体）、COLMAP 后续神经渲染数据管线（NeRF/3DGS 的输入重建）直接继承；时域平滑思想与后续循环一致性正则一脉相承。

## 7. 与本项目对照

- **与教程第 03 章的分工**：平面扫描几何（单应 warp）在 [教程 (3.1)–(3.2)](../03_多视图立体重建MVS.md) 已完整推导，本文 4.1 的 $H_l^m$ 只是把固定法向换成逐像素斜平面（同一消元，教程 3.1 末已预告）；NCC 仿射不变性在教程 (3.3)–(3.5)；本文补充的正是教程 3.2 只作"操作注记"的**视图选择理论**（Eq.(7) 的逐像素联合评分）与教程 3.2(c) 只作"前提"的**融合判据**（4.7 三约束）。
- **与 photoba / direct 的思想同源**：本文的光度残差 $(1 - \rho_l^m)$ 与 [projects/slam/photoba/](../../../projects/slam/photoba/photoba.py) 的光度 BA 残差同根——都是"以亮度/块相似度为测量、对位姿与结构联合优化"；差别在组织尺度：photoba 在滑动窗口内对**稀疏逆深度点**做直接法 BA（[SLAM 精读 DSO 篇](../../slam/精读/DSO_PAMI2018.md) 的路线），本文在**全图像素**上做逐点深度推断再融合；教程第 03 章 3.3 ③ "手工光度 + 学习正则 vs 端到端"的选型讨论正以本文为手工一侧的代表。
- **学习式替代（姊妹篇互引）**：MVSNet 把本文的"平面扫描 warp + 多视图代价聚合"搬进可微代价体（教程 (3.6)–(3.11) 的方差聚合与软 argmin），用 3D CNN 学习正则取代本文的 NCC + 手工平滑——弱纹理/反射面的信息真空由数据先验填补，代价是 $O(HWD)$ 显存与训练数据依赖；两者构成"手工最优 vs 学习最优"的经典对照，见本目录姊妹篇精读（MVSNet, ECCV 2018）与 [教程 3.3 节](../03_多视图立体重建MVS.md)。
- **无代码的原因**：MVS 教学实现（平面扫描 + NCC + 一致性过滤 + 融合）依赖 GPU 级并行才有可用性能，CPU 教学版收敛慢、收益密度低；当前仓库的机器人主线（在线状态估计）优先级更高，列为教学实现规划项。若实现，最小路径为：教程 (3.2) 的 warp + (3.4) 的 NCC 打分 + 本文 Eq.(15) 型过滤 + 4.7 三约束聚类，即可复现论文的定性行为。

## 配套阅读

- 教程主线：[第 03 章｜多视图立体重建 MVS](../03_多视图立体重建MVS.md)（平面扫描 (3.1)–(3.2)、NCC 不变性 (3.3)–(3.5)、MVSNet (3.6)–(3.11)）｜[第 04 章｜RGB-D 融合与 TSDF](../04_RGB-D融合与TSDF.md)（深度图融合的另一条线）。
- 姊妹篇：[COLMAP-SfM 精读](./COLMAP-SfM_CVPR2016.md)（本文的稀疏输入从哪来；三角化角–深度不确定性推导在彼篇 4.3，与本文三角化先验互为表里）。
- 跨主题回引：[SLAM 精读 DSO 篇](../../slam/精读/DSO_PAMI2018.md)（光度残差 + 鲁棒核的同源思想）；[SLAM 教程第 05 章](../../slam/05_视觉里程计-i特征点法.md)（投影 (5.10)——warp 与重投影误差的几何基元）。
- 代码：[projects/slam/photoba/](../../../projects/slam/photoba/)（光度残差的滑动窗口实现）；MVS 教学实现规划中。
- 基准：Middlebury 多视图立体评测（vision.middlebury.edu/mview/eval/）；Strecha 高分辨率数据集。

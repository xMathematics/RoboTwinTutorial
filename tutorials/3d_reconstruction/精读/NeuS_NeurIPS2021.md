# 论文精读｜NeuS（NeurIPS 2021）

> **PDF**：[papers/3d_reconstruction/classics/arXiv-2106.10689_NeuS.pdf](../../../papers/3d_reconstruction/classics/arXiv-2106.10689_NeuS.pdf) ｜ **教程**：[第 07 章｜神经隐式表面重建](../07_神经隐式表面重建.md) ｜ **代码**：无（教学实现规划中）
>
> 式号说明：本文所有"论文 Eq. x"均指本地 PDF（arXiv v3，2023-02-01 编译，共 23 页，主文 Eq.1–17）排版核对后的编号，并与仓库内 LaTeX 源码（`papers/3d_reconstruction/latex/arXiv-2106.10689/`，注意目录名无 `_NeuS` 后缀）交叉核对一致；"教程 (7.x)"指教程第 07 章式号，两者来源不同、分别标注。

## 1. 论文信息与一句话贡献

- **题目**：NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction
- **作者**：Peng Wang、Yuan Liu、Taku Komura（香港大学），Lingjie Liu、Christian Theobalt（MPI Informatics），Wenping Wang（Texas A&M）
- **发表**：NeurIPS 2021；arXiv:2106.10689（本仓库 PDF 为 v3）
- **一句话贡献**：把**符号距离函数**（Signed Distance Function, SDF）的零水平集作为表面表示，并重新设计体渲染（volume rendering）中的权重函数——由 SDF 经 logistic 密度分布（S-density）导出不透明度，使权重**在一阶近似下无偏**（unbiased，权重峰对准零水平集）且**感知遮挡**（occlusion-aware）——从而只用带位姿 RGB 图像（掩码可选）就能端到端训出高质量神经隐式表面，在 DTU/BlendedMVS 上超过 IDR、NeRF、UNISURF 等基线。

## 2. 问题与动机

**要解决什么**：从一组带位姿照片重建物体表面 $\mathcal{S}$，要求几何质量（而非仅新视角画质）——机器人查询"某点有无表面、法向朝哪"需要的是干净的表面。

**两条既有路线各有一半答案**（§1）：

- **表面渲染**（surface rendering）路线（DVR、IDR）：每条光线只取**单个**表面交点反传梯度。梯度过于局部，深度突变（abrupt depth changes，如孔洞边缘）处优化陷入坏局部极小；且 IDR 依赖前景掩码监督。论文 Fig.1(b) 实拍：IDR 在深度突变边缘重建失败。
- **体渲染**（volume rendering）路线（NeRF）：沿光线多点采样做 alpha 合成，深度突变处近远表面都提供梯度、优化稳健；但学的是**体密度** $\sigma$——它只为外观服务，从中提取表面噪声明显（Fig.1(b) 下：平面区域网格满是毛刺）。

**关键观察**（§3.1，也是本文最重要的概念贡献）：把"密度 = S 密度场 $\phi_s(f)$"直接塞进**标准**体渲染 $w(t)=T(t)\sigma(t)$（论文 Eq.3），得到的权重虽然感知遮挡，却**有系统性偏差**（bias）——权重峰落在表面交点之前，重建表面存在固有几何误差。NeuS 的药方是改造权重函数本身，使其在一阶近似下无偏（Theorem 1）。

**与同期工作 UNISURF 的分工**（§2）：UNISURF 用占据（occupancy）表示 + 逐步收缩采样区间；NeuS 用 SDF 表示——表面可自然取零水平集提取（论文 Eq.1），实验精度更高（论文 Table 1 w/o mask 均值）。

## 3. 方法总览

```
带位姿 RGB 图像（+ 可选前景掩码）
        │  每次迭代采样一批像素光线 {o_k, v_k}
        ▼
【两阶段采样】64 均匀点 + 4 轮×16 重要性点 = 128 点/光线
   粗阶段用固定 s（32×2^i）的 S-密度做概率，细阶段用可学习 s —— 单网络
        ▼
【SDF MLP】8 隐层×256、Softplus(β=100)、第 4 层跳跃连接 → f(p)（几何初始化同 IDR）
   【颜色 MLP】4 隐层×256，输入 p、视角 v、法向 n=∇f、SDF 特征 → c
        ▼
【NeuS 不透明度】论文 Eq.13：α_i = max((Φ_s(f_i)−Φ_s(f_{i+1}))/Φ_s(f_i), 0)
   （推导链：Eq.10 连续 ρ → Eq.12 段积分 → Eq.13；每步见 §4）
        ▼
【体渲染】论文 Eq.11：Ĉ = Σ T_i α_i c_i，T_i = Π_{j<i}(1−α_j)
        ▼
【损失】论文 Eq.14：L = L_color + λ·L_reg + β·L_mask
   L_reg = Eikonal 罚（Eq.16）维持 ‖∇f‖=1；s（S-密度尺度）作为可学习参数联合优化
        ▼
【表面提取】训练后对 f 的零水平集（Eq.1）做 marching cubes
```

与教程第 07 章的对应：本图第 4 步即教程 (7.4)，第 5 步即 (7.1)/(7.5)，闭合权重 (7.6) 与两条性质（§7.3）是本章推导的落点；Eikonal 罚即 (7.9)，其几何根据是教程第 06 章的 (6.1)。

## 4. 关键公式推导（按论文原文式号）

**符号表**（论文 §3.1 记号；与教程第 07 章一致）：

| 记号 | 含义 | 教程对应 |
|---|---|---|
| $f:\mathbb{R}^3\to\mathbb{R}$ | SDF 网络，外正内负 | 第 06 章 $s(\mathbf{x})$、(6.1) |
| $c:\mathbb{R}^3\times\mathbb{S}^2\to\mathbb{R}^3$ | 颜色网络 | §7.2 ② |
| $\mathbf{p}(t)=\mathbf{o}+t\mathbf{v}$ | 光线（$\mathbf{o}$ 相机中心，$\mathbf{v}$ 单位方向） | 同 |
| $\Phi_s(y)=(1+\mathrm{e}^{-sy})^{-1}$ | Sigmoid / logistic 分布函数 | (7.2) |
| $\phi_s(y)=\Phi_s'(y)$ | S-density：logistic 密度分布 | (7.3) |
| $s$（或 $1/s$） | S-密度尺度参数，可学习 | §7.4 ② |
| $w(t),\,T(t),\,\rho(t)$ | 权重 / 累积透射率 / 不透明密度（opaque density） | (7.5)/(7.8) |
| $\alpha_i,\,T_i,\,\delta_i$ | 离散不透明度 / 透射率 / 段长 | (7.1) |
| $\theta$ | 视角 $\mathbf{v}$ 与外法向 $\mathbf{n}$ 的夹角 | (7.7) 的 $c=\cos\theta$ |
| $t^\ast$ | 光线与零水平集交点参数 | §7.3 (a) |

### 4.1 表示与渲染方程（论文 Eq.1–2）

表面 = SDF 零水平集（Eq.1，PDF 第 3 页）：$\mathcal{S}=\{\mathbf{x}\in\mathbb{R}^3\mid f(\mathbf{x})=0\}$。

像素颜色沿光线累积（Eq.2，PDF 第 4 页）：

$$C(\mathbf{o},\mathbf{v})=\int_0^{+\infty} w(t)\,c\big(\mathbf{p}(t),\mathbf{v}\big)\,\mathrm{d}t.$$

（依据：即 NeRF 的连续体渲染积分，见 [NeRF 教程第 03 章](../../nerf/03_数学原理与渲染方程.md) §3.2、教程 (7.1) 的连续形式；差别只在权重 $w(t)$ 的来源——NeuS 要重新设计它。）论文对 $w(t)$ 提出两条**要求**（Requirements，PDF 第 4 页列表）：

1. **Unbiased**：$w(t)$ 在 $f(\mathbf{p}(t^\ast))=0$ 处取局部极大；
2. **Occlusion-aware**：同 SDF 值的两点 $f(t_0)=f(t_1)$、$t_0<t_1$，有 $w(t_0)>w(t_1)$。

### 4.2 S-density：logistic 密度分布（教程 (7.2)–(7.3) 对照）

取 sigmoid 函数 $\Phi_s(y)=(1+\mathrm{e}^{-sy})^{-1}$，其导数即论文所称 S-density：

$$\phi_s(y)=\Phi_s'(y)=\frac{s\,\mathrm{e}^{-sy}}{\bigl(1+\mathrm{e}^{-sy}\bigr)^2}.$$

推导（依据：链式法则）：$\frac{\mathrm{d}}{\mathrm{d}y}(1+\mathrm{e}^{-sy})^{-1}=-(1+\mathrm{e}^{-sy})^{-2}\cdot(-s\,\mathrm{e}^{-sy})$。它是关于 $y=0$ 对称的钟形（偶函数），峰值 $\phi_s(0)=s/4$。分子分母同乘 $\mathrm{e}^{sy}$（依据：代数恒等变形 $\mathrm{e}^{sy}+2+\mathrm{e}^{-sy}=(\mathrm{e}^{sy/2}+\mathrm{e}^{-sy/2})^2$）得等价形式

$$\phi_s(y)=\frac{s}{4}\,\mathrm{sech}^2\Bigl(\frac{sy}{2}\Bigr),$$

教程 §7.3 (a) 的对称性论证与 $O(1/s)$ 峰移分析用的正是这一形式。论文称 $\phi_s$ 的"标准差"为 $1/s$——严格说 $1/s$ 是 logistic 分布的**尺度参数**（标准差为 $\pi/(\sqrt{3}\,s)$），比例关系不变；$1/s$ 可学习、随训练收敛趋于 0（表面变"硬"）。此式即教程 (7.3)，$\Phi_s$ 为 (7.2)。

### 4.3 两个不行的方案（论文 Eq.3–4）

**Naive 方案（Eq.3）**：沿用标准体渲染 $w(t)=T(t)\sigma(t)$，$T(t)=\exp(-\int_0^t\sigma(u)\mathrm{d}u)$，取 $\sigma(t)=\phi_s(f(\mathbf{p}(t)))$。感知遮挡，但有偏——补充材料 §2.1 的证明可三步复现：

第一步（依据：乘积法则）：$\frac{\mathrm{d}w}{\mathrm{d}t}=\frac{\mathrm{d}T}{\mathrm{d}t}\sigma+T\frac{\mathrm{d}\sigma}{\mathrm{d}t}$，而 $\frac{\mathrm{d}T}{\mathrm{d}t}=-\sigma T$（依据：对 $T=\exp(-\int_0^t\sigma)$ 求导），故 $\frac{\mathrm{d}w}{\mathrm{d}t}=T\bigl(\sigma'-\sigma^2\bigr)$。

第二步（代入一阶近似 $\sigma(t)=\phi_s(f(\mathbf{p}(t)))$）：$\sigma'(t^\ast)=(\nabla f\cdot\mathbf{v})\,\phi_s'(f(\mathbf{p}(t^\ast)))=0$（依据：$f(\mathbf{p}(t^\ast))=0$ 处 $\phi_s'(0)=0$——$\phi_s$ 是偶函数）。

第三步（合并）：$\frac{\mathrm{d}w}{\mathrm{d}t}(t^\ast)=T(t^\ast)\bigl(0-\phi_s(0)^2\bigr)=-T(t^\ast)\phi_s(0)^2<0$（依据：$T>0$、$\phi_s(0)>0$）。导数为负 ⟹ $w$ 在 $t^\ast$ 处单调下降、峰在 $t^\ast$ **之前**（偏相机一侧）——有偏。补充材料 §3 进一步给出：该峰位误差 $\Delta_t=O(s^{-1})$，而 NeuS 权重 $\Delta_t=O(s^{-2})$（二阶收敛）。

**直接构造（Eq.4）**：把 S-密度归一化直接当权重 $w(t)=\phi_s(f(\mathbf{p}(t)))/\int_0^{+\infty}\phi_s(f(\mathbf{p}(u)))\mathrm{d}u$。无偏（分子是单峰对称函数），但不感知遮挡：光线穿透两个表面时 $f$ 有两个零点 ⟹ $w$ 有两个等高峰，颜色被无差别混合（论文原述）。

### 4.4 NeuS 的构造：从理想单平面反解 $\rho$（论文 Eq.5–10）

保留体渲染框架但换密度：定义**不透明密度**（opaque density）$\rho(t)$，令（Eq.5）

$$w(t)=T(t)\,\rho(t),\qquad T(t)=\exp\Bigl(-\int_0^t \rho(u)\,\mathrm{d}u\Bigr).$$

**理想情形反解**（论文"How we derive opaque density ρ"段）：设单表面、平面、离相机无穷远。此时沿光线 $f(\mathbf{p}(t))=-|\cos\theta|\,(t-t^\ast)$（依据：SDF 沿光线的变化率 = 方向与单位法向夹角余弦，平面时为常数；外正内负约定使接近表面时 $f>0$、穿过后 $f<0$），且 $|\cos\theta|$ 为常数。把 Eq.4 的分母逐变量替换（论文 Eq.6 的五行长链，每步依据如下）：

$\int_0^\infty\phi_s(f(\mathbf{p}(u)))\mathrm{d}u\xrightarrow{\ u^\ast=u-t^\ast\ }\int_{-t^\ast}^{\infty}\phi_s(-|\cos\theta|u^\ast)\mathrm{d}u^\ast\xrightarrow{\ \hat u=-|\cos\theta|u^\ast\ }\frac{1}{|\cos\theta|}\int_{-|\cos\theta|t^\ast}^{\infty}\phi_s(\hat u)\,\mathrm{d}\hat u\xrightarrow{\ t^\ast\to\infty\ }\frac{1}{|\cos\theta|}\int_{-\infty}^{\infty}\phi_s=\frac{1}{|\cos\theta|}$

（依据：换元与上下限变换；最后一步用 $\phi_s$ 是密度、全轴积分为 1，且 $t^\ast\to\infty$ 使下限趋于 $-\infty$。）代回 Eq.4 得理想权重 $w(t)=|\cos\theta|\,\phi_s(f(\mathbf{p}(t)))$，于是（Eq.7）

$$T(t)\,\rho(t)=|\cos\theta|\,\phi_s\big(f(\mathbf{p}(t))\big).$$

两个恒等式对接（依据：对 Eq.5 求导 $\frac{\mathrm{d}T}{\mathrm{d}t}=-\rho T$，即 $T\rho=-\frac{\mathrm{d}T}{\mathrm{d}t}$；链式法则 $\frac{\mathrm{d}}{\mathrm{d}t}\Phi_s(f(\mathbf{p}(t)))=\phi_s(f)\cdot f'(t)=\phi_s(f)\cdot(-|\cos\theta|)$，即 $|\cos\theta|\phi_s=-\frac{\mathrm{d}}{\mathrm{d}t}\Phi_s(f)$）：$\frac{\mathrm{d}T}{\mathrm{d}t}=\frac{\mathrm{d}}{\mathrm{d}t}\Phi_s(f(\mathbf{p}(t)))$，两边积分（依据：原函数差一常数；理想情形 $t\to0$ 时 $T(0)=1$ 且表面无穷远使 $\Phi_s(f(\mathbf{p}(0)))\to1$，常数为 0）得（Eq.8）

$$T(t)=\Phi_s\big(f(\mathbf{p}(t))\big).$$

取对数再求导（依据：$\int_0^t\rho\,\mathrm{d}u=-\ln\Phi_s(f(\mathbf{p}(t)))$；对 $t$ 求导，$\frac{\mathrm{d}}{\mathrm{d}t}\ln\Phi_s=\frac{\Phi_s'}{\Phi_s}$）得单平面情形的不透明密度（Eq.9）

$$\rho(t)=\frac{-\frac{\mathrm{d}\Phi_s}{\mathrm{d}t}\big(f(\mathbf{p}(t))\big)}{\Phi_s\big(f(\mathbf{p}(t))\big)}.$$

**推广到多表面**：$f$ 沿光线上升段（离开表面）中 $-\frac{\mathrm{d}\Phi_s}{\mathrm{d}t}<0$，须截断保证 $\rho\ge0$（依据：不透明密度是非负物理量；截断位置见论文 Fig.3 示意）（Eq.10）

$$\rho(t)=\max\left(\frac{-\frac{\mathrm{d}\Phi_s}{\mathrm{d}t}\big(f(\mathbf{p}(t))\big)}{\Phi_s\big(f(\mathbf{p}(t))\big)},\;0\right).$$

### 4.5 离散化：不透明度闭式（论文 Eq.11–13）

采样 $n$ 点 $t_1<\dots<t_n$，离散渲染（Eq.11）：$\hat C=\sum_{i=1}^n T_i\alpha_i c_i$，$T_i=\prod_{j=1}^{i-1}(1-\alpha_j)$；段不透明度（Eq.12）

$$\alpha_i=1-\exp\Bigl(-\int_{t_i}^{t_{i+1}}\rho(t)\,\mathrm{d}t\Bigr).$$

（依据：Max 求积规则，与 NeRF 离散化同构、教程 (7.1)；差别是 $\alpha_i$ 由 $\rho$ 的段积分定义。）用链式法则把 $-\frac{\mathrm{d}\Phi_s}{\mathrm{d}t}=-(\nabla f\cdot\mathbf{v})\,\phi_s(f)$（依据：复合求导）代入，**下降段**（$f$ 单调减、无截断）上换元 $y=f(\mathbf{p}(t))$、$\mathrm{d}y=(\nabla f\cdot\mathbf{v})\mathrm{d}t$：

$\int_{t_i}^{t_{i+1}}\rho\,\mathrm{d}t=\int_{t_i}^{t_{i+1}}\frac{-(\nabla f\cdot\mathbf{v})\phi_s(f)}{\Phi_s(f)}\mathrm{d}t=-\int_{f_i}^{f_{i+1}}\frac{\phi_s(y)}{\Phi_s(y)}\,\mathrm{d}y=-\ln\Phi_s(f_{i+1})+\ln\Phi_s(f_i)$

（末步依据：$\phi_s=\Phi_s'$ ⟹ $\int\phi_s/\Phi_s=\ln\Phi_s$。）代入 Eq.12（依据：$\mathrm{e}^{\ln a}=a$）：

$$\alpha_i=1-\exp\bigl(\ln\Phi_s(f_{i+1})-\ln\Phi_s(f_i)\bigr)=1-\frac{\Phi_s(f_{i+1})}{\Phi_s(f_i)}=\frac{\Phi_s(f_i)-\Phi_s(f_{i+1})}{\Phi_s(f_i)}.$$

**上升段**：$\nabla f\cdot\mathbf{v}>0$ ⟹ Eq.10 截断使 $\rho=0$ ⟹ Eq.12 给 $\alpha_i=0$。两种情形合并即（Eq.13，PDF 第 6 页；补充材料 §1 完整推导；教程 (7.4) 同式）

$$\alpha_i=\max\left(\frac{\Phi_s\big(f(\mathbf{p}(t_i))\big)-\Phi_s\big(f(\mathbf{p}(t_{i+1}))\big)}{\Phi_s\big(f(\mathbf{p}(t_i))\big)},\;0\right).$$

### 4.6 闭合形式权重（教程 (7.5)–(7.6)；论文未单列此式）

设各段 CDF 差均为正（无截断），把 Eq.13 代入 $w_i=T_i\alpha_i$ 做 telescoping（依据：连乘交错相消；此推导为教程第 07 章内容，本 PDF 中无对应编号公式）：

$1-\alpha_j=\frac{\Phi_s(f_{j+1})}{\Phi_s(f_j)}$（通分化简）⟹ $T_i=\prod_{j<i}\frac{\Phi_s(f_{j+1})}{\Phi_s(f_j)}=\frac{\Phi_s(f_i)}{\Phi_s(f_1)}$ ⟹ $w_i=\frac{\Phi_s(f_i)}{\Phi_s(f_1)}\cdot\frac{\Phi_s(f_i)-\Phi_s(f_{i+1})}{\Phi_s(f_i)}=\frac{\Phi_s(f_i)-\Phi_s(f_{i+1})}{\Phi_s(f_1)}$。

首采样点在近平面、相机在物体外，$f_1\gg0$ ⟹ $\Phi_s(f_1)\approx1$，取等号得教程 (7.6)：

$$w_i=\Phi_s\big(f(\mathbf{p}(t_i))\big)-\Phi_s\big(f(\mathbf{p}(t_{i+1}))\big).$$

概率解读：由 $\phi_s=\Phi_s'$，$w_i=\int_{f_{i+1}}^{f_i}\phi_s(y)\mathrm{d}y$（依据：牛顿–莱布尼茨）——权重 = "视线在该段穿过某 SDF 等值层"的离散概率，完全由 $f$ 决定、不再有独立自由度。

### 4.7 两条性质（Theorem 1 与 Requirement 2）

**无偏性（论文 Theorem 1，PDF 第 6 页；证明在补充材料 §2.2）**：设光线自外向内穿过表面、交点 $\mathbf{p}(t^\ast)$，$f$ 在 $[t_l,t_r]\ni t^\ast$ 单调下降，且局部用切平面近似（$\nabla f$ 视为常量）。证明骨架：

第一步：该区间内 $-(\nabla f\cdot\mathbf{v})>0$、$\phi_s>0$、$\Phi_s>0$ ⟹ 截断不生效，$\rho(t)=\frac{-(\nabla f\cdot\mathbf{v})\phi_s(f)}{\Phi_s(f)}$。

第二步：把 $T(t)$ 从 $t_l$ 拆起：$T(t)=T(t_l)\exp(-\int_{t_l}^t\rho)$；用 §4.5 同一原函数得 $\int_{t_l}^{t}\rho\,\mathrm{d}t'=-\ln\Phi_s(f(\mathbf{p}(t)))+\ln\Phi_s(f(\mathbf{p}(t_l)))$（依据：同上，$\int\phi_s/\Phi_s=\ln\Phi_s$），代入得

$$w(t)=\frac{-(\nabla f\cdot\mathbf{v})\,T(t_l)}{\Phi_s(f(\mathbf{p}(t_l)))}\,\phi_s\big(f(\mathbf{p}(t))\big).$$

第三步：一阶近似下 $\nabla f=\mathbf{n}$（依据：Eikonal 方程 $\|\nabla f\|=1$，教程 (6.1)），故 $-(\nabla f\cdot\mathbf{v})=|\cos\theta|$ 在区间内为常数；$T(t_l)/\Phi_s(f(\mathbf{p}(t_l)))$ 与 $t$ 无关。于是 $w(t)\propto\phi_s(f(\mathbf{p}(t)))$——单峰函数在 $f=0$ 即 $t=t^\ast$ 处取局部极大。证明未对相机与 $t_l$ 之间的表面做假设 ⟹ 多表面情形同样成立。

**教程对照（§7.3 (a)）**：教程以连续权重 (7.8) 重做此论证：线性化 $f(\mathbf{p}(t))\approx c\,(t^\ast-t)$（(7.7)，$c=|\nabla f\cdot\mathbf{d}|=\cos\theta$）代入 $\mathrm{sech}^2$ 形式，主项是以 $t^\ast$ 为心的对称钟形；并**补充了论文没有的 $O(1/s)$ 一阶偏差分析**——把 $T$ 的变化计入后峰位前移 $\Delta t=\frac{\operatorname{arsinh}(1/2c)}{s\,c}=O(1/s)$（垂直入射 $c=1$ 时 $\approx0.48/s$），$s\to\infty$ 时消失。这与补充材料 §3 的二阶分析（$\Delta_t=O(s^{-2})$ vs naive 的 $O(s^{-1})$）是同一"有限 $s$ 残余偏差"的两种刻画，量级结论一致。

**遮挡感知**：本 PDF 版本中它是 §3.1 的 Requirement 2（带列表定义）而非编号定理（**与常见转述"NeuS Theorem 2"不符**，见 §7 偏离说明）。论证骨架（直觉图见论文 Fig.3；教程 §7.3 (b) 的推导）：(i) 框架 Eq.5 保留透射 $T$，任意非负 $\rho$ 下 $T$ 单调降，故同 SDF 值的两点近者权重大（Requirement 2 的定义直接满足）；(ii) 量化"穿透衰减"：穿越前层后透射因子 $1-\alpha=\Phi_s(f_{\text{负}})/\Phi_s(f_{\text{正}})$（依据：§4.6 的 $1-\alpha_j$ 化简），而 sigmoid 尾部 $1-\Phi_s(y)=\frac{1}{1+\mathrm{e}^{sy}}\approx\mathrm{e}^{-sy}$（$|sy|$ 大时取主项），故穿透深度 $|f|$ 越大透射按 $\mathrm{e}^{-s|f|}$ 指数压低；$s\to\infty$ 时 $\Phi_s\to$ 阶跃（依据：(7.2) 逐点极限），穿越段 $\alpha\to1$、其后 $T\to0$——远层权重指数衰减到 0。

### 4.8 训练目标（论文 Eq.14–17）

$$\mathcal{L}=\mathcal{L}_{color}+\lambda\,\mathcal{L}_{reg}+\beta\,\mathcal{L}_{mask},\qquad
\mathcal{L}_{color}=\frac{1}{m}\sum_k\mathcal{R}(\hat C_k,C_k),$$

$$\mathcal{L}_{reg}=\frac{1}{nm}\sum_{k,i}\bigl(\|\nabla f(\hat{\mathbf{p}}_{k,i})\|_2-1\bigr)^2,\qquad
\mathcal{L}_{mask}=\mathrm{BCE}(M_k,\hat O_k),\ \ \hat O_k=\sum_i T_{k,i}\alpha_{k,i}.$$

（依据：$\mathcal{R}$ 取 L1（同 IDR，对离群稳健）；$\mathcal{L}_{reg}$ 即 Eikonal 罚——教程 (7.9)，其几何根据是第 06 章 (6.1) $\|\nabla f\|=1$；掩码监督用权重和 $\hat O_k$ 做占据概率，可选。）

**两阶段采样**（§3.2" hierarchical sampling"段；补充材料 §4）：64 均匀 + 4 轮×16 重要性点共 128 个；粗阶段概率用**固定** $s=32\times2^i$ 的 S-密度、细阶段用可学习 $s$——与 NeRF 双网络不同，NeuS 只维护一个网络。**实现细节**（补充材料 §4）：$\alpha_i$ 用"段端点" $\mathbf{q}_i=\mathbf{o}+t_i\mathbf{v}$ 按 Eq.13 计算，颜色 $c_i$ 取段中点 $\mathbf{p}_i=\mathbf{o}+\frac{t_i+t_{i+1}}{2}\mathbf{v}$——前者对应 CDF 差的几何、后者对应密度的采样语义。

## 5. 实验与结果解读

- **DTU**（15 场景，49/64 视图，1600×1200；Table 1，PDF 第 8 页）：Chamfer 距离均值——**w/ mask**：NeuS **0.77** vs IDR 0.90、NeRF 1.54；**w/o mask**：NeuS **0.84** vs UNISURF 1.02、COLMAP+Poisson 1.36、NeRF 1.49（单位与后续工作一致为 mm 量级；即普遍亚毫米到 1 mm 级）。训练 512 rays/batch、300k 迭代，单卡 RTX 2080Ti 约 14 小时（w/ mask）/16 小时（w/o mask）。
- **BlendedMVS**（7 个低分辨率场景，768×576）：定性对比——IDR 在 Stone 场景深度突变处失败（表面渲染的局部优化所致），NeRF 网格噪声明显（密度场缺几何约束），NeuS 表面保真度更高（论文 Fig.4–5）。
- **细结构**（§4.3"Thin structures"）：两个 32 视图细薄物体，深度突变边缘重建准确。
- **消融**（Fig.7）：(a) naive 方案 Chamfer 显著变差（有偏的直接证据）；(b) Eq.4 直接构造出现严重伪影（不感知遮挡）；(c) 去掉 Eikonal 正则或几何初始化，Chamfer 看似持平，但 SDF 预测与真值的 MAE 大幅变差——即**渲染指标掩盖了"f 不再是有效距离函数"**，两项正则不可省。
- 解读：w/o mask 仅比 w/ mask 低 0.07（0.84 vs 0.77），说明"由 SDF 导出的权重"本身携带了足够的空间正则——这是体渲染路线相对表面渲染路线（IDR 必须掩码）的核心卖点。

## 6. 局限与后续影响

**论文自述局限**（§5）：(i) 依赖纹理特征匹配的程度不高，但**无纹理物体**性能仍退化（补充材料给出失败案例）；(ii) 单一尺度参数 $s$ 服务于所有空间位置，无法按局部几何自适应——作者列为未来工作；(iii) 训练算力开销大。此外可从设定直接读出：**逐场景优化慢**（300k 迭代、小时级），继承了 NeRF 范式；颜色网络把光照与反照率混在一起、无材质分解，复杂光照可能把明暗"雕"进几何（定性）。

**后续影响**：与 VolSDF（Yariv et al., NeurIPS 2021，同期）分别从"改权重"与"改密度"两条路把 SDF 接进体渲染（对比见教程 (7.10) 及 §7.3 ③）；UNISURF（占据路线）、Geo-NeuS / HF-NeuS（引入 SfM 点/监督）沿此线改进；**Neuralangelo**（Li et al., CVPR 2023）把 NeuS 的 SDF 体渲染装上多分辨率哈希编码与大窗口数值梯度，同时解决"慢"与"细节"（见本目录[姊妹精读](./Neuralangelo_CVPR2023.md)）；神经隐式表面由此成为机器人场景重建的可选路线之一（教程第 10 章选型讨论）。

## 7. 与本项目对照

教程第 07 章即以 NeuS 为压轴推导的教材化重写，映射如下（左列 = 本 PDF 排版式号）：

| 论文 | 内容 | 教程第 07 章 | 差异注记 |
|---|---|---|---|
| Eq.1 | 零水平集表面 | §7.2 ② | 同义 |
| Eq.2 | 连续渲染方程 | (7.1) 连续形式 | 回引 NeRF 教程 §3.2 |
| Eq.3 | naive：$w=T\sigma$ | §7.2 ③ 选型理由 | 教程定性指出方向性偏差，本文 §4.3 给出补充材料证明 |
| Eq.4 | 归一化 S-density | §7.2 ③ | 同 |
| Eq.5 | $w=T\rho$ | (7.5) | 教程直接从离散合成出发，省去 Eq.6–8 的单平面极限推导（两条等价路线） |
| Eq.9–10 | 连续 $\rho$（带截断） | (7.4) 的连续原型 | 教程直接定义离散 (7.4) |
| Eq.11–12 | $\hat C$、$\alpha_i$ 段积分 | (7.1) | 教程注明 NeuS 的 $\alpha$ 来源与 NeRF 不同 |
| Eq.13 | $\alpha_i$ 闭式 | (7.4) | 同式 |
| （无编号） | 闭合权重 $\Phi_i-\Phi_{i+1}$ | **(7.6)** | 论文未单列，教程由 (7.4)–(7.5) telescoping 推出 |
| Theorem 1 | 无偏性（一阶近似） | §7.3 (a) | 教程补充 $O(1/s)$ 一阶峰移分析；论文补充材料 §3 有 $O(s^{-2})$ 二阶分析 |
| Requirement 2 | 遮挡感知 | §7.3 (b) | **本版无论 "Theorem 2"**，教程以 $\mathrm{e}^{-s\mid f\mid}$ 衰减论证补足 |
| Eq.16 | Eikonal 罚 | (7.9) ← 第 06 章 (6.1) | 同式 |
| （未涉及） | VolSDF 对照 | (7.10)，§7.3 ③ | **本 PDF 未引用/对比 VolSDF**；对照是教程内容 |
| Eq.14–15/17 | 总损失/颜色/掩码 | §7.4 ② 文字 | 教程未编号 |

**与规格说明的三处偏离**（按 PDF 原文处理）：① 遮挡感知在 arXiv v3 中是 Requirement 2 而非"Theorem 2"；② 论文无"§7 VolSDF 对照实验"（VolSDF 未被引用）——VolSDF 对比改按教程 (7.10) 呈现并明确标注来源；③ 闭合权重 $w_i=\Phi_s(f_i)-\Phi_s(f_{i+1})$ 在论文中无编号公式，作为教程 (7.6) 推导给出。另：仓库内 NeuS LaTeX 源码目录名为 `latex/arXiv-2106.10689`（无 `_NeuS` 后缀），与 PDF 内容一致（v3）。

## 配套阅读

- 论文 PDF：[papers/3d_reconstruction/classics/arXiv-2106.10689_NeuS.pdf](../../../papers/3d_reconstruction/classics/arXiv-2106.10689_NeuS.pdf)；LaTeX 源码：[papers/3d_reconstruction/latex/arXiv-2106.10689/](../../../papers/3d_reconstruction/latex/arXiv-2106.10689/)（含补充材料全部证明与实现细节）。
- 姊妹精读：[Neuralangelo（CVPR 2023）](./Neuralangelo_CVPR2023.md)——NeuS 的提速与细节增强版。
- 教程主线：[第 07 章](../07_神经隐式表面重建.md)（本章推导的教材化）；[第 06 章](../06_学习式形状表示SDF与占据.md)（SDF 性质与 Eikonal (6.1)）；[NeRF 教程第 03 章](../../nerf/03_数学原理与渲染方程.md)（体渲染 (7.1) 的出处）。
- 基线对照精读：[COLMAP-SfM](./COLMAP-SfM_CVPR2016.md)、[COLMAP-MVS](./COLMAP-MVS_ECCV2016.md)、[Poisson 重建](./PoissonRecon_SGP2006.md)（Table 1 中 COLMAP+Poisson 基线即"COLMAP 稠密点云 + 屏蔽泊松网格化"）、[MVSNet](./MVSNet_ECCV2018.md)（学习式 MVS 路线对照）。
- 应用落点：[第 10 章｜机器人场景中的重建实战与选型](../10_机器人场景中的重建实战与选型.md)。

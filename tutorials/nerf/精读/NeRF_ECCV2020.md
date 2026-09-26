# 论文精读｜NeRF（ECCV 2020）

> **LaTeX 源码**：[arxiv_submission.tex](../../../papers/nerf/latex/arxiv_submission.tex) ｜ **教程**：[十章](../README.md) ｜ **代码**：projects/nerf/（15 个测试全过：`tests/test_core.py` 8 + `tests/test_metrics.py` 7）
>
> 式号说明：本文所有"论文 Eq. x"均按本地 LaTeX 源码 `papers/nerf/latex/arxiv_submission.tex` 逐一核对——**主文恰好 6 个编号公式**（`\begin{equation}` 无 `\tag`，按出现顺序自动编号），附录 C 的 NDC 推导从 Eq. 7 起编。注意：教程 [05 章 §5.4](../05_层次采样与训练细节.md)、[09 章 §9.6](../09_代码逐行精读.md) 与 `projects/nerf/METRICS.md` 把损失函数写作"论文 Eq. 7"，经源码核对**实为 Eq. 6**（下文 §4.6 有勘误注）。

## 1. 论文信息与一句话贡献

- **题目**：NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis
- **作者**：B. Mildenhall、P. P. Srinivasan、M. Tancik（UC Berkeley，三人同等贡献）、J. T. Barron（Google）、R. Ramamoorthi（UCSD）、R. Ng（Berkeley）
- **发表**：ECCV 2020（最佳论文提名）；arXiv:2003.08934
- **一句话贡献**：把静态场景表示为一个**全连接网络**（MLP，非卷积）$F_\Theta:(\mathbf{x},\mathbf{d})\to(\mathbf{c},\sigma)$——输入 5D 坐标（3D 位置 + 2D 视角方向），输出颜色与体积密度——再用经典**体渲染**（volume rendering）沿相机光线积分成像素；整个渲染过程天然可微，仅凭带位姿 RGB 图像即可端到端优化。两项关键配套技术：**位置编码**（positional encoding，Eq. 4）解决 MLP 学不了高频的问题；**层次采样**（hierarchical sampling，Eq. 5）把采样预算集中到可见内容上。

## 2. 问题与动机

**任务**：新视角合成（novel view synthesis）——给定若干带位姿的照片，渲染任意新视角。

论文 §2 给出两条既有路线各自的死角：

- **神经 3D 形状表示**（SDF/occupancy 隐式表面）：要么需要 3D 真值监督（ShapeNet 类），要么（DVR/SRN 等可微渲染路线）"受限於简单形状、输出过度平滑"——每条光线只取**单深度单颜色**，表达力天然受限（§6.3 对 SRN 的批评）。
- **体素/MPI 类离散体积方法**（NV、LLFF）：用 CNN 预测离散化的 RGBα 网格。分辨率被 $128^3$ 这类网格上限锁死——"渲染更高分辨率图像需要更精细的 3D 采样"，时间与空间复杂度均不可扩展（§2 末段）；LLFF 还要求输入视角视差不超过 64 像素（"sampling guideline"）。

NeRF 的破局点：把**连续**体积藏进 MLP 权重里——存储只需 5 MB（对比 LLFF 单场景 >15 GB），且 MLP 查询点是连续的。但朴素实现有两个致命缺陷（§1 末段，作者原话）：

1. 直接回归 5D 坐标**收敛不到高分辨率表示**（MLP 的谱偏置，spectral bias，见 §4.4）；
2. 每条光线需要**太多采样点**（自由空间与被遮挡区域被反复白查，见 §4.5）。

论文的贡献条目（§1，共 3 条）：5D 辐射场表示；基于经典体渲染的可微渲染 + 层次采样；位置编码。

## 3. 方法总览

```
带位姿图像 → 每次迭代随机抽一批像素光线 r(t)=o+t·d
   │  分层采样 Nc=64 点（Eq. 2）→ coarse MLP → Eq. 3 渲染 Ĉc → Eq. 5 权重 w_i
   │  归一化 w → 分段常数 PDF → 逆变换采样 Nf=128 点
   │  合并 Nc+Nf 点 → fine MLP → Eq. 3 渲染 Ĉf
   ▼  Eq. 6：对 Ĉc 与 Ĉf 同时做光度 MSE
```

**场景表示与"密度为什么只准看位置"**（§3）：网络先对 $\gamma(\mathbf{x})$ 过 8 层全连接（ReLU、256 通道、第 5 层跳跃连接），输出密度 $\sigma$（经 ReLU 保证非负）与 256 维特征；特征再拼上 $\gamma(\mathbf{d})$ 过一层 128 通道 + sigmoid 输出 RGB（附录 Fig. 7）。$\sigma$ 的分支在方向 $\mathbf{d}$ 注入**之前**就分叉，故 $\sigma$ 只能是 $\mathbf{x}$ 的函数——从任意视角看，同一位置的"挡光程度"必须一致（多视图一致，multiview consistency）；而颜色允许依赖方向，才能表达镜面反射等**非朗伯**（non-Lambertian）效果（论文 Fig. 3：Ship 场景两固定点的方向颜色分布；Fig. 4：去掉视角依赖后铲车履带的高光消失）。

## 4. 关键公式推导

### 4.0 符号表（论文 §3–§5 记号）

| 符号 | 含义 | 教程对应 |
|---|---|---|
| $\mathbf{x}=(x,y,z)$，$\mathbf{d}$ | 3D 位置；单位视角方向（实践中用 3D 笛卡尔单位向量） | [03 章 §3.1](../03_数学原理与渲染方程.md) |
| $\mathbf{r}(t)=\mathbf{o}+t\mathbf{d}$，$t_n,t_f$ | 相机光线；近/远界 | 03 章 §3.1 |
| $\sigma(\mathbf{x})$，$\mathbf{c}(\mathbf{x},\mathbf{d})$ | 体积密度；视角相关辐射颜色 | 02 章 |
| $T(t)$，$C(\mathbf{r})$ | 累积透射率；光线期望颜色（Eq. 1） | 03 章 §3.2 |
| $N,\,N_c,\,N_f$ | 采样数；coarse/fine 采样数（64/128） | 03 章 §3.4 / 05 章 §5.2 |
| $\delta_i=t_{i+1}-t_i$，$\alpha_i$ | 相邻采样间距；第 $i$ 段不透明度 | 03 章 §3.5 |
| $w_i,\ \hat{w}_i$ | 合成权重及其归一化（Eq. 5） | 05 章 §5.2 |
| $\gamma,\ L$ | 位置编码；最大频率（$\mathbf{x}$: 10，$\mathbf{d}$: 4） | 04 章 §4.2 |
| $\hat{C}_c,\hat{C}_f,C(\mathbf{r}),\mathcal{R}$ | 粗/细渲染、真值颜色、光线批 | 05 章 §5.4 |

### 4.1 Eq. 1：连续体渲染（从"终止概率"到积分）

源码位置：主文第 1 个 `equation` 环境。原文先给出 $\sigma$ 的物理解读（引 Kajiya & Herzen 1984）："$\sigma(\mathbf{x})$ 可解读为光线在 $\mathbf{x}$ 处**终止于一个无穷小粒子**的微分概率"。把它写成条件概率形式并逐步推出 Eq. 1：

**第 1 步（定义）**：光线已存活到 $t$ 的条件下，在 $[t, t+\mathrm{d}t]$ 内终止的概率正比于该处密度与段长：

$$
P(\text{终止于}[t,t+\mathrm{d}t]\mid \text{到达}t)=\sigma(\mathbf{r}(t))\,\mathrm{d}t \quad(\text{当 }\mathrm{d}t\to 0).
$$

**第 2 步（透射率的微分方程）**：定义 $T(t):=P(\text{从 } t_n \text{ 存活到 } t)$，$T(t_n)=1$。"存活到 $t+\mathrm{d}t$" = "存活到 $t$" **且** "在 $[t,t+\mathrm{d}t]$ 内未终止"（两事件独立，概率相乘）：

$$
T(t+\mathrm{d}t)=T(t)\,\bigl(1-\sigma(\mathbf{r}(t))\,\mathrm{d}t\bigr).
$$

左边按泰勒展开保留一阶项 $T(t+\mathrm{d}t)=T(t)+T'(t)\,\mathrm{d}t$，右边界 $1-\sigma\mathrm{d}t$ 同样只保留一阶（高阶小量 $\mathrm{d}t^2$ 舍去，依据：一阶近似），对比 $\mathrm{d}t$ 的系数得：

$$
\frac{\mathrm{d}T}{\mathrm{d}t}=-\sigma(\mathbf{r}(t))\,T(t),\qquad T(t_n)=1 .
$$

**第 3 步（解 ODE）**：一阶线性齐次方程，分离变量 $\frac{\mathrm{d}T}{T}=-\sigma\,\mathrm{d}t$，两边从 $t_n$ 积分到 $t$（依据：分离变量法；初始条件定常数）：

$$
\ln T(t)=-\int_{t_n}^{t}\sigma(\mathbf{r}(s))\,\mathrm{d}s
\;\Longrightarrow\;
T(t)=\exp\Bigl(-\int_{t_n}^{t}\sigma(\mathbf{r}(s))\,\mathrm{d}s\Bigr).
$$

**第 4 步（期望颜色）**：光线"恰在 $[t,t+\mathrm{d}t]$ 内首次撞上粒子"的概率 = $P(\text{到达 } t)\cdot P(\text{在该段终止}\mid\text{到达 }t)=T(t)\,\sigma(\mathbf{r}(t))\,\mathrm{d}t$（依据：条件概率乘法公式）。撞下来就记下该点颜色 $\mathbf{c}(\mathbf{r}(t),\mathbf{d})$，对所有可能的撞点求和（积分）即得**期望颜色**：

$$
\textbf{(Eq. 1)}\qquad
C(\mathbf{r}) = \int_{t_n}^{t_f}T(t)\,\sigma(\mathbf{r}(t))\,\mathbf{c}(\mathbf{r}(t),\mathbf{d})\,\mathrm{d}t,
\quad
T(t) = \exp\Bigl(-\int_{t_n}^{t}\sigma(\mathbf{r}(s))\,\mathrm{d}s\Bigr).
$$

说明：这是辐射传输方程（radiative transfer equation）只保留**吸收-发射**项的特例——NeRF 把"被挡下来"的光按 $\mathbf{c}$ 重新发出，忽略向内散射；此抽象正是 NeRF 所需要的（它只关心"这条光线看起来的颜色"）。$T(t)$ 单调递减：前面越密，后面贡献越小。

### 4.2 Eq. 2：分层采样（stratified sampling）

连续积分必须数值化。论文否掉**确定性求积**（deterministic quadrature）：MLP 只会在固定离散点被查询，表示分辨率被采样密度锁死。改用分层采样（源码第 2 个 `equation`，`\label{eq:stratified}`）：

$$
\textbf{(Eq. 2)}\qquad
t_i \sim \mathcal{U}\Bigl[\,t_n+\tfrac{i-1}{N}(t_f-t_n),\;\; t_n+\tfrac{i}{N}(t_f-t_n)\Bigr],\quad i=1,\dots,N .
$$

把 $[t_n,t_f]$ 均分为 $N$ 个 bin，每个 bin 内均匀随机取一点。两点依据：

- **无偏性**：每个 bin 恰贡献一个样本，bin 宽 $\Delta=(t_f-t_n)/N$ 相同，样本整体仍均匀覆盖区间，是对 Eq. 1 的无偏估计；且分层保证"每个 bin 都有人代言"，方差不超过同等样本量的朴素蒙特卡洛（分层采样教科书性质：消除了"样本全挤进同一半区间"的层间随机性）。
- **连续性**：训练中每次迭代采样位置都不同——MLP 在**连续位置**上被查询，网络学到的是连续函数而非固定网格上的查表值（论文原话："stratified sampling enables us to represent a continuous scene representation"）。

### 4.3 Eq. 3：离散渲染与 $\alpha_i$ 的由来

源码第 3 个 `equation`（`\label{eqn:render_coarse}`），求积规则取自 Max 1995 综述。推导分四步：

**第 1 步（分段常数近似）**：在每个小区间 $[t_i,t_{i+1}]$ 内取 $\sigma(\mathbf{r}(s))\approx\sigma_i$、$\mathbf{c}\approx\mathbf{c}_i$（依据：零阶/分段常数近似；$\delta_i=t_{i+1}-t_i$）。

**第 2 步（透射率离散化）**：对第 3 步 ODE 的解中的积分做分段：

$$
\int_{t_n}^{t_i}\sigma(\mathbf{r}(s))\,\mathrm{d}s=\sum_{j=1}^{i-1}\sigma_j\,\delta_j
\;\Longrightarrow\;
T_i:=T(t_i)=\exp\Bigl(-\sum_{j=1}^{i-1}\sigma_j\delta_j\Bigr)
\quad(\text{依据：分段常数函数的积分}=段上取值\times段长\text{求和}).
$$

**第 3 步（段内终止概率 → $\alpha_i$）**：光线在第 $i$ 段内终止的概率为 $T(t_i)-T(t_{i+1})$。代入 $T(t_{i+1})=T_i\exp(-\sigma_i\delta_i)$（依据：第 2 步结论多累积一段），分配律展开：

$$
T_i-T_{i+1}=T_i-T_i e^{-\sigma_i\delta_i}=T_i\bigl(1-e^{-\sigma_i\delta_i}\bigr)=:T_i\,\alpha_i,
\qquad
\boxed{\;\alpha_i=1-e^{-\sigma_i\delta_i}\;}
$$

$\alpha_i$ 性质（依据：指数函数单调性与泰勒展开 $e^{-x}=1-x+O(x^2)$）：$\sigma_i,\delta_i\to 0$ 时 $\alpha_i\approx\sigma_i\delta_i$（密度小则近透明）；两者增大则 $\alpha_i\to 1$（近不透明）。

**第 4 步（求和）**：把每段"撞下来的概率 × 颜色"累加，即离散化 Eq. 1：

$$
\textbf{(Eq. 3)}\qquad
\hat{C}(\mathbf{r})=\sum_{i=1}^{N}T_i\bigl(1-e^{-\sigma_i\delta_i}\bigr)\mathbf{c}_i,
\qquad
T_i=\exp\Bigl(-\sum_{j=1}^{i-1}\sigma_j\delta_j\Bigr).
$$

两个补充结论：① 递推形式 $T_{i+1}=T_i(1-\alpha_i)$、$T_1=1$，即"到第 $i$ 点为止的剩余透明度"，全式退化为图形学经典的**由后到前 alpha 合成**（alpha compositing，论文引 Porter-Duff 84）；② 权重和 $\sum_{i=1}^N T_i\alpha_i=\sum_i (T_i-T_{i+1})=T_1-T_{N+1}=1-T_{N+1}\le 1$（裂项相消），差额 $T_{N+1}$ 是"穿过整个场景没撞任何东西"的概率——代码实现里把末段 $\delta$ 补成大数正是为了把这份残余权重吸收掉。

### 4.4 Eq. 4：位置编码与谱偏置

**问题**（§5.1）：神经网络虽是通用函数逼近器，但让 $F_\Theta$ 直接吃低维坐标，渲染结果"难以表达颜色与几何的高频变化"——与 Rahaman et al. 2018 的**谱偏置**（spectral bias）结论一致：深度网络偏向先拟合低频函数。对策（Rahaman 同文给出）：先用高频函数把输入映到高维，再进网络。于是 $F_\Theta=F_\Theta'\circ\gamma$，$\gamma$ 固定不可学习（源码第 4 个 `equation`，`\label{eq:enc}`）：

$$
\textbf{(Eq. 4)}\qquad
\gamma(p)=\Bigl(\sin(2^0\pi p),\ \cos(2^0\pi p),\ \cdots,\ \sin(2^{L-1}\pi p),\ \cos(2^{L-1}\pi p)\Bigr).
$$

- 定义域：$\mathbf{x}$ 各分量归一到 $[-1,1]$，$\mathbf{d}$ 是单位向量各分量天然在 $[-1,1]$；$\gamma:\mathbb{R}\to\mathbb{R}^{2L}$ 对**每个分量独立**施加。实验取 $L=10$（位置，$3\times2\times10=60$ 维）、$L=4$（方向，24 维）。
- **为什么这就解决了谱偏置（表达力一侧，无跳步论证）**：$\gamma(p)$ 恰是 $p$ 的截断傅里叶基。任何只对 $\gamma(p)$ 做**线性**读出的函数 $g(p)=\sum_k a_k\sin(2^k\pi p)+b_k\cos(2^k\pi p)$ 都是各频率的线性组合——网络第一层给高频通道配大权重即可"搬用"现成的高频基，无需靠深层非线性复合从低频输入中"长出"高频。没有 $\gamma$ 时，ReLU 复合函数的频谱能量随复合次数向低频集中（Rahaman 的谱偏置刻画），拟合高频极慢。**优化一侧**的收益由 Rahaman et al. 与本文实验共同背书。
- **证据**：Fig. 4"去掉位置编码"列整体过平滑；消融表（`ablationstable.tex`）Row 2 无 PE 28.77 dB（vs 完整 31.01），Row 7 $L=5$ 30.59，Row 8 $L=15$ 30.81——$2^L$ 超过输入图像最高频率（约 1024）后无收益，故 $L=10$ 饱和。
- 与 Transformer 的位置编码**同名不同purpose**（§5.1 原文辨析）：Transformer 用它给离散 token 注入顺序信息；NeRF 用它把连续坐标升维以便拟合高频函数。

### 4.5 Eq. 5：层次采样与逆变换采样

均匀采样浪费：自由空间与被遮挡区域对渲染零贡献。做法（§5.2，灵感引 Levoy 1990）：同时优化 coarse/fine 两个**独立同构** MLP。先把 coarse 的 Eq. 3 渲染改写成"颜色的加权和"（源码第 5 个 `equation`，`\label{eqn:weights}`）：

$$
\textbf{(Eq. 5)}\qquad
\hat{C}_c(\mathbf{r})=\sum_{i=1}^{N_c} w_i\,\mathbf{c}_i,
\qquad
w_i=T_i\bigl(1-e^{-\sigma_i\delta_i}\bigr)=T_i\,\alpha_i .
$$

即 $w_i$ = "光线撞在第 $i$ 段的概率"，非负且 $\sum_i w_i\le 1$（§4.3 结论 ②）。归一化 $\hat{w}_i=w_i/\sum_{j=1}^{N_c}w_j$ 后得到沿光线的**分段常数 PDF**。**逆变换采样**（inverse transform sampling）的完整推导：

**第 1 步（构造 CDF）**：在离散 bin 上累计：$F_i=\sum_{j\le i}\hat{w}_j$，且 $F_0=0$、$F_{N_c}=1$（依据：PDF 归一化保证总质量为 1）。bin 内部线性插值，$F$ 成为 $[t_n,t_f]$ 上连续单调的分布函数。

**第 2 步（反演）**：抽 $u\sim\mathcal{U}(0,1)$，取满足 $F_{k-1}\le u<F_k$ 的 bin $k$，再在 $[t_k,t_{k+1}]$ 内均匀取一点。

**第 3 步（正确性）**：样本落入 bin $i$ 的概率 $=P\bigl(F_{i-1}\le u<F_i\bigr)=F_i-F_{i-1}=\hat{w}_i$（依据：均匀分布落在区间的概率 = 区间长度 / 总长 $=F_i-F_{i-1}$）。故采样密度正比于渲染贡献——物体表面附近（$w_i$ 大）被采得密。

随后：从该分布抽 $N_f$ 个点，与 coarse 的 $N_c$ 个点**取并集、按深度排序**，用 fine 网络在全部 $N_c+N_f$ 个点上按 Eq. 3 重渲染出 $\hat{C}_f$。实验取 $N_c=64$、$N_f=128$；测试时每光线共 $64+(64+128)=256$ 次查询（附录 A"Rendering Details"）。论文特意辨析（§5.2 末）：这与重要性采样（importance sampling）目标相同，但用法不同——样本被当作**整个积分域的非均匀离散化**，而非对积分的独立概率估计。

### 4.6 Eq. 6：训练目标

源码第 6 个 `equation`（Implementation Details 小节）：

$$
\textbf{(Eq. 6)}\qquad
\mathcal{L}=\sum_{\mathbf{r}\in\mathcal{R}}\Bigl[\bigl\|\hat{C}_c(\mathbf{r})-C(\mathbf{r})\bigr\|_2^2+\bigl\|\hat{C}_f(\mathbf{r})-C(\mathbf{r})\bigr\|_2^2\Bigr],
$$

纯光度损失（photometric loss）：粗细两个渲染都与真值像素算平方误差。**为什么连 $\hat{C}_c$ 也监督**（论文原文回答）：最终渲染虽来自 $\hat{C}_f$，但 coarse 网络的权重分布是 fine 采样的 PDF——不监督它，Eq. 5 的分布会失准、细采样随之失效。训练配置：batch 4096 条光线、Adam（lr $5\times10^{-4}$ 指数衰减至 $5\times10^{-5}$）、100–300k 步（V100 约 1–2 天）；真实场景对 $\sigma$ 加 $\mathcal{N}(0,1)$ 噪声正则、并在 NDC 坐标系下训练（附录 C，从 Eq. 7 起推导：把前向视锥映为 $[-1,1]^3$、深度换为视差）。

> **勘误注**：教程 05 章 §5.4、09 章 §9.6 与 `projects/nerf/METRICS.md` 称损失为"论文 Eq. 7"；按 LaTeX 源码，主文编号公式只有 6 个，损失是 Eq. 6，Eq. 7 起为附录 NDC 推导。三者行文逻辑不受影响，特此对齐。

### 4.7 "稀疏视角"语境下的对比（动机补笔）

LLFF 要求输入视角视差 ≤ 64 像素，而合成数据集视角间视差达 400–500 像素，LLFF 因此频繁估错几何（§6.3）；LLFF 还为每个输入视角存一整套 MPI 体素、多视角间混合渲染导致视频中的不一致伪影。消融 Row 5：NeRF 仅用 25 张图训练得 27.78 dB，仍高于 NV/SRN/LLFF 用满 100 张的成绩——这正是"光度目标 + 多视图一致性"相对"逐视角融合"的鲁棒性优势。

## 5. 实验与结果解读

**数据集**（§6.1）：① Diffuse Synthetic 360°（DeepVoxels）：4 个朗伯物体，$512\times512$，训练 479 / 测试 1000；② Realistic Synthetic 360°（本文自建，后续社区标准基准）：8 个路径追踪渲染的复杂物体（非朗伯材质），$800\times800$，训练 100 / 测试 200，6 场景上半球 + 2 场景全球面；③ Real Forward-Facing：8 个真实前向场景（手持手机），每场景 20–62 张、留 1/8 测试，$1008\times756$；真实数据位姿由 COLMAP 估计。

**主结果**（`resultstable.tex`，PSNR↑/SSIM↑/LPIPS↓）：

| 方法 | Diffuse 360° | Realistic 360° | Real Forward-Facing |
|---|---|---|---|
| SRN | 33.20 / 0.963 / 0.073 | 22.26 / 0.846 / 0.170 | 22.84 / 0.668 / 0.378 |
| NV | 29.62 / 0.929 / 0.099 | 26.05 / 0.893 / 0.160 | —（有界体积假设不适用） |
| LLFF | 34.38 / 0.985 / 0.048 | 24.88 / 0.911 / 0.114 | 24.13 / 0.798 / **0.212** |
| **NeRF** | **40.15 / 0.991 / 0.023** | **31.01 / 0.947 / 0.081** | **26.50 / 0.811** / 0.250 |

**口径注意**：LPIPS 越低越好；附录 B 披露 SRN 只能跑 $512\times512$（其余方法 $800\times800$ / $1008\times752$），跨方法分辨率不完全对齐。唯一的"例外"是真实数据上 LLFF 的 LPIPS（0.212 优于 0.250）——论文的辩护是视频下 NeRF 多视图一致性更好、伪影更少（指标之外的一致性维度）。

**开销**（§6.3）：所有逐场景优化基线 ≥12 小时/场景，NeRF 1–2 天；但 LLFF 每输入图存一套 MPI（单场景 >15 GB），NeRF 仅 5 MB 权重（约 $3000\times$ 压缩，比输入图像还小）；测试渲染约 30 秒/帧（每图 1.5–2 亿次网络查询）。

**消融**（`ablationstable.tex`，8 场景均值，详见教程 [07 章](../07_消融实验解读.md)）：完整模型 31.01/0.947/0.081；去视角依赖 −3.35 dB、去位置编码 −2.24 dB、去层次采样（改均匀 256 点）−0.95 dB——前两者贡献最大；25/50 张输入降至 27.78/29.79 dB（25 张仍胜全部基线 100 张）；$L=5$ 降、$L=15$ 不升（频率饱和）。

## 6. 局限与后续影响

- **训练慢、逐场景优化**（1–2 天/场景，本源码版无法增量）→ 效率路线：[Instant-NGP（SIGGRAPH 2022）](../../3d_reconstruction/精读/InstantNGP_SIGGRAPH2022.md)用多分辨率哈希表 + 网络蒸馏把训练压到秒级；[3DGS（SIGGRAPH 2023）](../../3d_reconstruction/精读/3DGS_SIGGRAPH2023.md)换成显式各向异性高斯 + 光栅化，训练快、可实时渲染，成为新主流。二者精读均含与 NeRF 体渲染的对照推导。
- **无显式表面**：$\sigma$ 是为外观服务的软密度，取 mesh 噪声大 → 表面路线：NeuS 把 SDF 经 logistic 密度接入体渲染权重使表面无偏（[NeuS 精读](../../3d_reconstruction/精读/NeuS_NeurIPS2021.md)，其 §4 对 NeRF 权重 $T_i\alpha_i$ 的偏差分析直接以本文 Eq. 1/3 为靶）；Neuralangelo 补高频细节。
- **静态、固定外观**：动态场景（D-NeRF 类时间条件）、外观/反照率歧义（光照变化下 $\mathbf{c}$ 与几何耦合）均未处理；逆渲染需显式分解材质与光照。
- **依赖已知位姿**：合成数据用真值位姿，真实数据靠 COLMAP；位姿不准即退化。由此衍生两条线：位姿联合优化与 **NeRF-SLAM** 系——把辐射场作为稠密地图、光度渲染作为额外约束（[NeRF-SLAM（ICRA 2023）精读](../../slam/精读/NeRF-SLAM_ICRA2023.md)：iMAP/DROID-SLAM 混合；同目录 MonoGS、SplaTAM 是 3DGS 版后继）。
- **位置编码是手工固定频率**：抗锯齿与自适应频率由 Mip-NeRF（集成位置编码 IPE）等后续工作解决；本文的"频域升维"思想则被几乎所有坐标网络沿用。

## 7. 与本项目对照

**三方映射表**（论文式号 ↔ 教程位置 ↔ 代码函数；均已逐文件核实）：

| 论文 | 教程 | projects/nerf 实现 |
|---|---|---|
| Eq. 1 连续体渲染 | [03 章 §3.2](../03_数学原理与渲染方程.md)（含 $T(t)$ ODE 推导） | `render.py::volume_render`（经 Eq. 3 离散化实现；docstring 引 Eq. 3/Max 1995） |
| Eq. 2 分层采样 | 03 章 §3.4"方式 B" | `sampling.py::stratified_sample`（`bins[:-1] + u*bin_size`） |
| Eq. 3 离散渲染 + $\alpha_i$ | 03 章 §3.5（$\alpha$ 合成等价 + 数值例） | `render.py::volume_render`（`alpha=1-exp(-σδ)`、`cumprod` 右移一位得 $T_i$）；$\delta_i$ 见 `sampling.py::ray_deltas` |
| Eq. 4 位置编码 | [04 章 §4.2](../04_网络架构与位置编码.md)（谱偏置论证） | `encoding.py::positional_encoding`（`2^k·π` 频率几何级数，sin/cos 拼接） |
| Eq. 5 权重 $w_i$ → 逆变换采样 | [05 章 §5.2–5.3](../05_层次采样与训练细节.md)（含 CDF 例） | `render.py::volume_render` 返回 `weights`；`sampling.py::hierarchical_sample`（`cumsum` 建 CDF + `searchsorted` 反演） |
| Eq. 6 损失 | 05 章 §5.4（**教程标"Eq. 7"系笔误**，见 §4.6 勘误注） | `trainer.py::train_nerf`（coarse+fine 双项 MSE） |
| 网络结构（附录 Fig. 7） | 04 章 §4.3（结构图 + skip 连接） | `model.py::NeRF`（`skip_at=4` 即第 5 层；`sigma_head` ReLU / `rgb_head` sigmoid） |
| 光线生成 / NDC | 03 章 §3.1、08 章 | `rays.py::get_rays_np`；`run_nerf.py::render_image`（整图分块渲染） |

- **指标口径**：`nerf/metrics.py::psnr/ssim` 与论文同口径（PSNR/SSIM），LPIPS 未实现（[METRICS.md](../../../projects/nerf/METRICS.md) 说明）。
- **健康值**（METRICS.md §1.3）：`demo_scene` 冒烟训练（80 步 tiny，约 4 秒）PSNR 16–19 dB，仅证明管线通畅；论文级 ~30 dB 需 100–300k 步——**两者训练步数差 3 个数量级，冒烟值不可对标论文值**。
- 环境搭建与复现路径见 [projects/nerf/TUTORIAL.md](../../../projects/nerf/TUTORIAL.md)；代码逐行讲解见教程 [09 章](../09_代码逐行精读.md)。

## 配套阅读

- 教程主线：[02 核心思想](../02_核心思想与表示方法.md) ｜ [03 渲染方程](../03_数学原理与渲染方程.md) ｜ [04 位置编码](../04_网络架构与位置编码.md) ｜ [05 层次采样](../05_层次采样与训练细节.md) ｜ [06 实验结果](../06_数据集与实验结果.md) ｜ [07 消融](../07_消融实验解读.md)
- 相关精读：[3DGS](../../3d_reconstruction/精读/3DGS_SIGGRAPH2023.md)（显式化提速）｜ [InstantNGP](../../3d_reconstruction/精读/InstantNGP_SIGGRAPH2022.md)（哈希加速）｜ [NeuS](../../3d_reconstruction/精读/NeuS_NeurIPS2021.md)（体渲染→表面）｜ [COLMAP-SfM](../../3d_reconstruction/精读/COLMAP-SfM_CVPR2016.md)（位姿来源）｜ [NeRF-SLAM](../../slam/精读/NeRF-SLAM_ICRA2023.md)（机器人落地）
- 论文材料：[结果表](../../../papers/nerf/latex/resultstable.tex) ｜ [消融表](../../../papers/nerf/latex/ablationstable.tex) ｜ [NDC 推导](../../../papers/nerf/latex/ndc.tex) ｜ [实战手册](../../../projects/nerf/TUTORIAL.md)

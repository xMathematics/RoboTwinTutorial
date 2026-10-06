# 论文精读｜Neuralangelo（CVPR 2023）

> **PDF**：[papers/3d_reconstruction/frontier/arXiv-2306.03092_Neuralangelo.pdf](../../../papers/3d_reconstruction/frontier/arXiv-2306.03092_Neuralangelo.pdf) ｜ **教程**：[第 07 章 §7.4](../07_神经隐式表面重建.md)（扩展小节）｜ **代码**：无（教学实现规划中）
>
> 式号说明：本文所有"论文 Eq. x"均指本地 PDF（arXiv 编译版 2023-06-14，共 18 页，主文 Eq.1–10）排版核对后的编号，并与仓库内 LaTeX 源码（`papers/3d_reconstruction/latex/arXiv-2306.03092/`）交叉核对一致；"教程 (7.x)/(9.x)"分别指教程第 07、09 章式号，来源不同、分别标注。

## 1. 论文信息与一句话贡献

- **题目**：Neuralangelo: High-Fidelity Neural Surface Reconstruction
- **作者**：Zhaoshuo Li（NVIDIA/JHU）、Thomas Müller、Alex Evans、Russell H. Taylor、Mathias Unberath、Ming-Yu Liu、Chen-Hsuan Lin（NVIDIA Research + Johns Hopkins）
- **发表**：CVPR 2023；arXiv:2306.03092（本仓库同时收录 [LaTeX 源码](../../../papers/3d_reconstruction/latex/arXiv-2306.03092/)）
- **一句话贡献**：把 NeuS 的 **SDF 体渲染**（继承其不透明度定义，论文 Eq.3）装上 Instant-NGP 式的**多分辨率哈希编码**（论文 Eq.4），并用**大步长数值梯度**（中心差分，论文 Eq.8）替换解析梯度计算表面法向——数值梯度把反传从"局部哈希单元"扩展到"$\boldsymbol\epsilon$ 窗口内的所有哈希条目"，等价于对解析梯度场做尺度 $\epsilon$ 的平滑（论文 Fig.2 论点）；配合"$\epsilon$ 指数退火 + 哈希分辨率逐级激活"的粗到细（coarse-to-fine）日程，在 DTU 与 Tanks and Temples 上同时拿下当时最优的表面精度与新视角画质，且不用掩码/深度/SfM 点等任何辅助数据。

## 2. 问题与动机

**要解决什么**：神经隐式表面（NeuS/VolSDF 一线）精度高但**逐场景优化太慢**——深层 MLP（NeuS 为 8×256）逐点查询，训练小时级；直接换上哈希编码提速又会**毁掉表面质量**。

**矛盾的两难**（§1、§3.2）：

- 慢的根源：NeuS/VolSDF 用深层 MLP 参数化 $f$（NeuS 为 8 隐层 × 256 宽），体渲染逐点查询、训练小时级（[NeuS 精读 §5](./NeuS_NeurIPS2021.md)）。
- 快的代价：哈希编码（Instant-NGP）靠可学习查表存细节、配浅层网络，把新视角合成提速到分钟级；但表面法向 $\nabla f(\mathbf{x})$ 的**解析梯度在哈希单元边界不连续**（§4.3 展开）——Eikonal 损失的反传只到达被采样单元自己的角点条目，相邻单元各自为政，重建表面碎裂、噪声大。
- 直接证据（论文 DTU 消融）：直接组合（AG，analytical gradients）Chamfer 均值 0.88 mm，**甚至差于慢得多的 NeuS（0.84 mm）**——快而不准，等于不可用。

**为什么"平滑"是本质需求**（§3.4 Level of Details 小节）：大尺度平滑结构（平直墙面）横跨大量网格单元，各单元本应给出连贯法向；实验显示平面要到约第 8 级分辨率才被正确预测，仅靠粗层自身单元间的局部连续性不足以拼出大连续表面——必须有一种让**多个单元的条目被同一次法向计算联合优化**的机制。

**约束条件**：不用分割、深度、COLMAP 点云等辅助输入（§4 实现声明），恢复跨尺度细节——从 DTU 小物体到 Tanks and Temples 大型室内外场景。

## 3. 方法总览

```
带位姿 RGB 图像（COLMAP 位姿；无掩码/深度/点云）
        │  沿光线采样 3D 点 x_i
        ▼
【多分辨率哈希编码】论文 Eq.4：γ(x_i) = (γ_1(x_i,₁), …, γ_L(x_i,L)) ∈ R^{cL}
   L=16 级，分辨率 2⁵→2¹¹；粗级初始激活，细级随训练逐级开启
        ▼
【SDF MLP：仅 1 层】→ f(x)（初始化近似球面）      【颜色 MLP：4 层】→ c
        ▼
【SDF 体渲染】论文 Eq.1 + Eq.3（NeuS 不透明度）：Ĉ = Σ w_i c_i，w_i = T_i α_i
        ▼
【数值梯度法向】论文 Eq.8：∇_x f ≈ (f(γ(x+ε_x)) − f(γ(x−ε_x))) / 2ε   （每点 6 次额外 SDF 查询）
   ε 初始化 = 最粗网格尺寸，随训练指数退火 → 粗到细
        ▼
【总损失】论文 Eq.10：L = L_RGB + w_eik·L_eik + w_curv·L_curv
   Eq.5 Eikonal 罚 + Eq.9 曲率正则（离散 Laplacian，复用 Eq.8 的采样点）
        ▼
【表面提取】marching cubes（DTU 512³、TnT 2048³）
```

与教程的对应：中间两步是教程第 07 章 (7.4)–(7.6) 的 SDF 体渲染（§7.4 扩展小节），编码一步是教程第 09 章 (9.1)–(9.3) 的多分辨率哈希；"数值梯度 = 梯度平滑"是 §7.4 末段论点的完整展开。

## 4. 关键公式推导（按论文原文式号）

**符号表**（论文 §3 记号）：

| 记号 | 含义 | 教程/姊妹精读对应 |
|---|---|---|
| $f(\mathbf{x})$ | SDF 网络（哈希编码后的浅 MLP），外正内负 | 第 07 章 $f$ |
| $\Phi_s$ | Sigmoid 函数（logistic CDF），SDF→不透明度的桥 | (7.2) |
| $\hat{\mathbf{c}},\mathbf{c},\mathbf{o},\mathbf{d}$ | 渲染色 / 真值色 / 相机中心 / 视线方向 | (7.1) |
| $w_i=T_i\alpha_i,\ \delta_i$ | 渲染权重、累积透射、段长 | (7.1)/(7.5) |
| $V_l,\ l=1\dots L$ | 第 $l$ 级网格分辨率（本文 16 级） | 第 09 章 $N_l$，(9.1) |
| $\gamma_l(\mathbf{x}_{i,l})\in\mathbb{R}^c$ | 第 $l$ 级插值特征（$c=8$） | (9.3) |
| $\beta=\mathbf{x}_{i,l}-\lfloor\mathbf{x}_{i,l}\rfloor$ | （三）线性插值系数 | (9.3) 的 $w_i$ |
| $\boldsymbol\epsilon_x=[\epsilon,0,0]$、$\epsilon$ | 轴向扰动向量 / 数值梯度步长 | §7.4 末段 $\boldsymbol\epsilon$ |

### 4.1 SDF 体渲染（论文 Eq.1–3；NeuS 的继承）

Riemann 求和渲染（Eq.1）与颜色损失（Eq.2）：

$$\hat{\mathbf{c}}(\mathbf{o},\mathbf{d})=\sum_{i=1}^N w_i\,\mathbf{c}_i,\qquad w_i=T_i\,\alpha_i, \tag{1}$$

$$\mathcal{L}_{\mathrm{RGB}}=\|\hat{\mathbf{c}}-\mathbf{c}\|_1. \tag{2}$$

（依据：Eq.1 即 NeRF 离散 alpha 合成，$\alpha_i=1-\exp(-\sigma_i\delta_i)$、$T_i=\prod_{j=1}^{i-1}(1-\alpha_j)$，见 [NeRF 教程第 03 章](../../nerf/03_数学原理与渲染方程.md) §3.5、教程 (7.1)。）表面由 SDF 零水平集 $\{\mathbf{x}\mid f(\mathbf{x})=0\}$ 定义，不透明度直接沿用 NeuS（Eq.3）：

$$\alpha_i=\max\left(\frac{\Phi_s\big(f(\mathbf{x}_i)\big)-\Phi_s\big(f(\mathbf{x}_{i+1})\big)}{\Phi_s\big(f(\mathbf{x}_i)\big)},\,0\right).$$

（依据：即 NeuS 论文 Eq.13——SDF 的 logistic CDF 归一化下降比例，完整推导见[NeuS 精读 §4.5](./NeuS_NeurIPS2021.md)与教程 (7.4)；闭合权重 $w_i=\Phi_s(f_i)-\Phi_s(f_{i+1})$ 即教程 (7.6)。）

Eq.3 之所以必要，回到 §3.1 前两段的问题设定：纯密度公式（Eq.1 的 $\alpha_i=1-\mathrm{e}^{-\sigma_i\delta_i}$）中 $\sigma$ 无表面语义，"从密度表示提取表面常得到噪声大、不真实的结果"（§3.1 原述，引 VolSDF/NeuS）；Eq.3 把 $\alpha_i$ 锁定在 $f$ 的零水平集附近，渲染损失因此同时约束几何。本节结论：**渲染端 Neuralangelo 零创新，创新全在"梯度怎么算"**。

### 4.2 多分辨率哈希编码（论文 Eq.4；Instant-NGP 的移植）

每级把坐标缩放到该级分辨率 $\mathbf{x}_{i,l}=\mathbf{x}_i\cdot V_l$，特征由网格角点哈希表项经三线性插值得到，各级拼接（Eq.4）：

$$\gamma(\mathbf{x}_i)=\bigl(\gamma_1(\mathbf{x}_{i,1}),\,\dots,\,\gamma_L(\mathbf{x}_{i,L})\bigr)\in\mathbb{R}^{cL}.$$

（依据：多分辨率哈希编码，Müller et al., Instant-NGP, SIGGRAPH 2022——分辨率几何级数排布、空间哈希、插值三层构造的完整推导见教程第 09 章 (9.1)–(9.3)；仓库另有 Instant-NGP 原文 PDF `papers/3d_reconstruction/frontier/arXiv-2201.05989_Instant-NGP.pdf`，尚无独立精读。）

**记号对照**：本式与教程 (9.3) 的 $\mathbf{z}_l(\mathbf{x})$ 同构，仅记号不同——论文 $V_l$ 即教程 (9.1) 的 $N_l$，论文每级特征维 $c$（本文取 8）即 Instant-NGP 的 $F$（原文取 2），插值系数论文记 $\beta$、教程记 $w_i$（勿与渲染权重 $w_i$ 混淆——本文为免歧义一律用 $\beta$）。Neuralangelo 的取值（§4 实现细节）：分辨率 $2^5$ 到 $2^{11}$ 共 16 级、每级哈希表上限 $2^{22}$ 项；DTU 初始激活 4 级、TnT 初始激活 8 级（场景尺度不同），其余级激活前特征置零（附录）。

**为什么选哈希而非稀疏体素**（§3.1 末段论证）：体素特征网格需要八叉树式层级分解控制内存，而层级设计天然让"细层无法纠正粗层的错误表示"（论文引 NGLOD 的性质）；哈希无空间层级、冲突靠梯度平均自动消解——这个"无层级"性质正是后文逐级激活日程能够成立的结构前提。

**一笔内存账**（按 §4 参数核算，帮助理解"用内存换计算"的量级）：16 级 × 至多 $2^{22}$ 项 × 每项 8 维可学习特征，参数量在 $10^8$ 量级、与场景内容无关地**有界**（哈希冲突保证超表不爆内存）；同一容量若用稠密体素直存，$2^{11}$ 分辨率一格一特征就是 $(2^{11})^3\approx10^{10}$ 项——这正是必须层级化或哈希化的原因（同教程第 09 章 09.1 ③ 的估算逻辑）。

### 4.3 解析梯度的局部性：表面为何碎裂（论文 Eq.5–7）

**Eikonal 损失**（Eq.5，PDF 第 3 页）：

$$\mathcal{L}_{\mathrm{eik}}=\frac{1}{N}\sum_{i=1}^N\bigl(\|\nabla f(\mathbf{x}_i)\|_2-1\bigr)^2. \tag{5}$$

（依据：同 NeuS Eq.16、教程 (7.9)，几何根据为第 06 章 (6.1)；端到端训练需对 $\nabla f$ 再求导——double backward。）

注意坐标约定：论文把兴趣区域归一到单位球内（附录），故第 $l$ 级的世界坐标单元尺寸即 $1/V_l$，$\epsilon$ 与各级网格尺寸直接可比——这是"每当 $\epsilon$ 降到某级尺寸就激活该级"这一日程能够用单一标量调度的前提。

**插值与其导数**。单元内插值系数 $\beta=\mathbf{x}_{i,l}-\lfloor\mathbf{x}_{i,l}\rfloor$，特征（Eq.6，PDF 第 4 页）：

$$\gamma_l(\mathbf{x}_{i,l})=\gamma_l\bigl(\lfloor\mathbf{x}_{i,l}\rfloor\bigr)\cdot(1-\beta)+\gamma_l\bigl(\lceil\mathbf{x}_{i,l}\rceil\bigr)\cdot\beta. \tag{6}$$

对位置求导（Eq.7）：$\frac{\partial\beta}{\partial\mathbf{x}_i}=V_l$（依据：$\mathbf{x}_{i,l}=V_l\mathbf{x}_i$ 线性缩放 + 链式法则；取整 $\lfloor\cdot\rfloor,\lceil\cdot\rceil$ 在单元内部为常数、导数为零——它们**不可微**只发生在单元边界处），代入得

$$\frac{\partial\,\gamma_l(\mathbf{x}_{i,l})}{\partial\,\mathbf{x}_i}=\gamma_l\bigl(\lfloor\mathbf{x}_{i,l}\rfloor\bigr)\cdot(-V_l)+\gamma_l\bigl(\lceil\mathbf{x}_{i,l}\rceil\bigr)\cdot V_l. \tag{7}$$

**局部性论断**（论文 §3.2 核心段落）：解析梯度只含**当前单元自己的两个角点条目**。按坐标轴拆开看更清楚（依据：Eq.6 按轴分解为三次一维线性插值）：三维单元有 $2^3=8$ 个角点条目，$\gamma_l$ 对 $\mathbf{x}_i$ 每个分量的偏导只触该分量方向上的 2 个条目、其余 6 个以系数 $(1-\beta)$ 或 $\beta$ 作为**常数**出现（其梯度在本次采样中为零）。当 $\mathbf{x}_i$ 跨过单元边界，$\lfloor\cdot\rfloor,\lceil\cdot\rceil$ 跳变，梯度表达式整体换到另一对条目上——$\gamma$ 对位置的导数场**在空间上不连续**。

**表面碎裂的因果链**（由 Eq.5–7 组装，论文 Fig.2 与 §3.2 文字）：(i) $\mathcal{L}_{\mathrm{eik}}$ 对 $f(\mathbf{x}_i)$ 的反传只落进被采样单元的局部条目 $\gamma_l(\lfloor\mathbf{x}_{i,l}\rfloor),\gamma_l(\lceil\mathbf{x}_{i,l}\rceil)$；(ii) 连续表面（如平墙）横跨大量单元，本应"各单元产出连贯法向"，但只有当这些单元**恰被同时采样**才会被联合优化——并无保证；(iii) 于是各单元只按各自见到的光线独立拟合，法向场在单元边界突变，表面碎成单元尺度的碎块。(iv) 旁证：三线性插值分段线性 ⟹ 特征对位置的二阶解析导数几乎处处为零，曲率正则也必须改用离散 Laplacian（§3.4，为 Eq.9 埋下伏笔）。定量后果见 §5：AG 变体 DTU Chamfer 0.88 mm、TnT F1 仅 0.27。

**teacher-student 路线为何不行**（§3.2 末段）：用 MLP 的平滑性蒸馏法向（Ref-NeRF 一类）仍是解析梯度反传，依旧只到局部条目——病灶在编码不在损失。

### 4.4 数值梯度 = 解析梯度的窗口平滑（论文 Eq.8）

改用中心差分计算法向（Eq.8，PDF 第 4 页）：

$$\nabla_x f(\mathbf{x}_i)=\frac{f\bigl(\gamma(\mathbf{x}_i+\boldsymbol\epsilon_x)\bigr)-f\bigl(\gamma(\mathbf{x}_i-\boldsymbol\epsilon_x)\bigr)}{2\,\epsilon},\qquad\boldsymbol\epsilon_x=[\epsilon,0,0]. \tag{8}$$

三个坐标轴各取一对扰动点，共 6 次额外 SDF 前向。

**与解析梯度的等价性**（论文原述："If the step size … is smaller than the grid size of hash encoding, the numerical gradient would be equivalent to the analytical gradient; otherwise, hash entries of multiple grid cells would participate"）。把等价性按层级拆开陈述（单看第 $l$ 级、一维记号，依据为 Eq.6–7 的构造）：

- **$\epsilon$ 小于第 $l$ 级单元尺寸**（世界坐标下 $1/V_l$）：$\mathbf{x}_i\pm\boldsymbol\epsilon_x$ 落入同一单元，$\lfloor\cdot\rfloor,\lceil\cdot\rceil$ 在两端相同，两处 $f$ 都对同一对角点条目求导 ⟹ 中心差分经过的正是 Eq.7 的解析导数，两级差商退化为同值（依据：单元内 $f$ 由同一组条目的同一线性组合给出）。
- **$\epsilon$ 大于第 $l$ 级单元尺寸**：$\pm\boldsymbol\epsilon_x$ 跨越多个单元，差分把窗口 $[\mathbf{x}_i-\boldsymbol\epsilon_x,\mathbf{x}_i+\boldsymbol\epsilon_x]$ 内**所有被触条目**纳入一次法向计算；对 Eq.8 反传时，更新同时分发给这些条目（依据：链式法则下差分是两条前向路径的公共上游）。

一维迷你例子（自拟，帮助落实上面两条；依据同为 Eq.6–7）：单看某一级，记 $\theta_{\text{左}}=\gamma_l(\lfloor x_{i,l}\rfloor)$、$\theta_{\text{右}}=\gamma_l(\lceil x_{i,l}\rceil)$，则单元内特征是一维线性函数 $\gamma_l=(1-\beta)\,\theta_{\text{左}}+\beta\,\theta_{\text{右}}$、斜率 $(\theta_{\text{右}}-\theta_{\text{左}})V_l$ 在单元内为常数。若 $\epsilon<1/V_l$，两处扰动仍在同一单元、斜率同一：

$$\frac{\gamma_l(x_{i,l}+\epsilon)-\gamma_l(x_{i,l}-\epsilon)}{2\epsilon}=\frac{(\theta_{\text{右}}-\theta_{\text{左}})\,V_l\cdot 2\epsilon}{2\epsilon}=(\theta_{\text{右}}-\theta_{\text{左}})\,V_l,$$

恰等于 Eq.7；取 $\epsilon>1/V_l$ 则 $x_{i,l}-\epsilon$ 的 $\lceil\cdot\rceil$ 与 $x_{i,l}+\epsilon$ 的 $\lfloor\cdot\rfloor$ 已是**别的条目**，上式的"同一斜率"前提不成立，窗口条目入场——等价性自然破坏。

**"平滑"解读**（论文 Fig.2 标题原话："…thus becoming a smoothed version of analytical gradients"）：数值梯度可解释为对解析梯度表达式做尺度 $\epsilon$ 的平滑——相邻单元的法向目标通过共享差分被耦合，边界突变被摊平，大连续表面因此能整体受力。补充一点数值直觉（论文未展开）：中心差分对光滑 $f$ 的截断误差为 $O(\epsilon^2)$——由泰勒展开 $f(\mathbf{x}\pm\epsilon)=f(\mathbf{x})\pm\epsilon f'\pm\frac{\epsilon^2}{2}f''+O(\epsilon^3)$，相减后奇次项消去（依据：两项相加除 $2\epsilon$）；优于前向差分的 $O(\epsilon)$。

### 4.5 粗到细日程表（§3.4；Eq.8 的 $\epsilon$ 与 Eq.4 的 $V$ 双时钟）

数值梯度天然把两个旋钮暴露出来，论文据此设计退火日程：

- **步长 $\epsilon$**：初始化为**最粗**哈希网格尺寸，随训练**指数下降**，逐级对齐各哈希分辨率尺寸（依据：大 $\epsilon$ ⟹ 法向在更大尺度上保持一致、产出连续大表面；小 $\epsilon$ ⟹ 平滑窗口缩小、不再抹掉细节——§3.4"Step size"段）。
- **哈希分辨率 $V$**：不从全级激活——否则细层先在粗优化阶段"学错"、再随 $\epsilon$ 变小"重学"，收敛后重学失败即丢细节（§3.4"Hash grid resolution"段原述）。做法：只开初始粗级集合，**每当 $\epsilon$ 降到某级尺寸就激活该级**（实现上每 5000 迭代开一级，§4；附录：未激活级特征置零）。另加全参数 weight decay 防止单一分辨率主导（§3.4 末句、附录 AdamW $10^{-2}$）。

日程落地参数（§4"Implementation details" + 附录，均 PDF 核对）：

$$\text{分辨率 } V_l:\ 2^5\to 2^{11}\ (16\ \text{级})；\quad \epsilon_0 = V_{\text{最粗级}}\text{ 的单元尺寸}；\quad \text{每 5000 迭代} \nearrow 1\ \text{级}.$$

初始激活级数按场景尺度取 4（DTU）/ 8（TnT）——两个"时钟"（$\epsilon$ 退火、$V$ 激活）以单元尺寸对齐，故退火与激活天然同步；这也是 §4.2 强调"无层级哈希"的结构原因：稀疏体素的层级设计不允许这种"后到细层从零开始"的日程（细层被粗层错误表示锁死，论文 §3.1 末段引 NGLOD 性质）。

（对照：教程 §7.4 末段把这条日程概括为"$\epsilon$ 从最粗网格尺寸指数退火 + 逐级激活更细哈希分辨率 = coarse-to-fine"，与论文一致。）**曲率正则与总损失**（Eq.9–10）：曲率由**离散** Laplacian 计算——采样点与 Eq.8 复用（依据：6 个扰动点已含各轴 $\pm\epsilon$，二阶差分无需新采样；解析 Hessian 因分段线性为零，见 §4.3(iv)）：

$$\mathcal{L}_{\mathrm{curv}}=\frac{1}{N}\sum_{i=1}^N\bigl|\nabla^2 f(\mathbf{x}_i)\bigr|, \tag{9}$$

$$\mathcal{L}=\mathcal{L}_{\mathrm{RGB}}+w_{\mathrm{eik}}\,\mathcal{L}_{\mathrm{eik}}+w_{\mathrm{curv}}\,\mathcal{L}_{\mathrm{curv}},\qquad w_{\mathrm{eik}}=0.1\ \text{（附录）}. \tag{10}$$

$\mathcal{L}_{\mathrm{curv}}$ 是平滑先验但会"冻结拓扑"（保持曲率奇异点不出现，凹形难成形），故前 5k 迭代线性 warmup（§4.5"Topology warmup"；附录）。

**§4 机制小结**（三句话版本）：哈希编码把容量搬进查表（Eq.4）换来速度，但插值的分段线性使解析梯度局部且不连续（Eq.6–7），Eikonal 反传因此只养护被采样的单元（Eq.5）——表面碎裂；中心差分以 $\epsilon$ 为窗把多单元条目纳入同一法向计算（Eq.8），等价于对梯度场平滑；$\epsilon$ 退火 + 分辨率逐级激活（§3.4）让"先大尺度平滑、后小尺度细节"成为日程而非碰运气。

## 5. 实验与结果解读

- **评测协议**（§4"Datasets / Evaluation criteria"）：DTU 真值来自结构光扫描仪、TnT 真值来自 LiDAR；表面指标为 Chamfer 距离与 F1 score（F1 = 在距离阈值下精度/召回的调和均值，对大场景比 Chamfer 更抗离群面——这正是 NeuralWarp"背景造面"在 TnT 上被重罚的原因），画质指标为 PSNR；全程不使用分割/深度等辅助数据。
- **DTU**（15 物体中心场景，49/64 视图；Table 1，mm 刻度）：Chamfer 均值——**NG+P（完整法）0.61**（全场最优）；对照方法 NeuS 0.84、VolSDF 0.86、HF-NeuS 0.77、NeuralWarp 0.68（需 SfM 点）、RegSDF 0.72（需 SfM 点）、NeRF 1.49。PSNR 均值 33.84，同为全场最优（NeRF 30.65 / VolSDF 30.38 / NeuS 29.79）。注意两点：其一，带 $\dagger$ 的两个方法吃到了 SfM 稀疏点云先验仍不及 NG+P，凸显"纯 RGB、无辅助"设定的含金量；其二，PSNR 与 Chamfer 同居榜首，说明几何收益不是"画质换精度"的折中产物。
- **DTU 消融链**（同表下半）：AG 0.88 → AG+P 0.73 → NG 0.65 → NG+P 0.61。解读：**AG（0.88）比 NeuS（0.84）还差**——"哈希 + NeuS 损失"的直接组合是负收益；逐级激活（+P）与数值梯度（NG）各自有正贡献、组合最优，且 NG 单独（0.65）已超过全部对照方法——平滑才是主收益，日程负责把细节还回来。
- **Tanks and Temples**（6 大型室内外场景，263–1107 视图；Table 2）：F1 均值 **0.50**（NG+P）vs NG 0.45、NeuS 0.38、AG+P 0.30、AG 0.27、COLMAP 0.21、NeuralWarp 0.15；PSNR 27.24 vs NeuS 24.58。背景用 NeuS 式附加网络（NeRF++）建模——论文指出 NeuralWarp 一类（VolSDF 式颜色渲染）会给天空/背景"造面"、F1 被离群面拉垮（§4.2）。消融链在 TnT 上同样单调：AG 0.27 → AG+P 0.30 → NG 0.45 → NG+P 0.50——注意 AG+P（0.30）**低于** NeuS（0.38）：只做"逐级激活"而不做数值梯度，仍救不动大场景。
- **定性观察**（Fig.3 与 Fig.6）：DTU Scan 37（金属薄物、剪刀等）与 Scan 24（建筑模型）上 NeuS/NeuralWarp 的薄结构与深度突变边缘明显退化，Neuralangelo 保形更完整；TnT 的 Courthouse 展示大规模颗粒状立面（同一建筑不同立面出现在 teaser 与 Fig.5），说明方法对"细粒度大表面"的规模化能力。
- **LOD 检查**（§4.3）：粗层丢失的树、桌子、自行车架由细层找回——验证"无层级哈希 + 逐级激活"的跨尺度能力；平面约第 8 级才正确（§2 动机的实验支撑）。论文特别强调：这说明**仅靠粗层单元间的局部连续性不足以拼出大连续表面**，反传必须能越过局部单元（即数值梯度的必要性从消融之外再获一证）。
- **消融**（Fig.6）：AG 表面噪声大（叠加逐级激活也救不动）；NG 平滑但牺牲细节；只有 NG+P 兼得。$\mathcal{L}_{\mathrm{curv}}$ 去除后表面出现尖锐跳变；拓扑 warmup 对凹陷区域关键。
- **In-the-wild**（附录 §B）：NVIDIA 园区与 JHU 校园的无人机实拍视频，位姿/内参由 COLMAP 恢复、兴趣区域用官方 Blender 插件在稀疏点云上人工框选，**直接复用 TnT 全套超参**重建出建筑、雕塑、树木等复杂几何——作者以此佐证超参的通用性。
- **开销**（附录速度表，V100）：单次迭代训练 NeuS 0.16 s vs NG 系 0.10–0.12 s（数值梯度比解析梯度慢约 1.2 倍，但 MLP 小得多——SDF 仅 1 层——总时间反超 NeuS）；$128^3$ 表面提取 0.19 s vs 0.08 s。总训练 500k 迭代（附录）。

**训练配置速查**（附录 §A，PDF 核对）：

| 项 | 取值 | 项 | 取值 |
|---|---|---|---|
| 总迭代 | 500k | 优化器 | AdamW（weight decay $10^{-2}$） |
| 学习率 | $10^{-3}$，5k 线性 warmup，300k/400k 各 ×10 | Eikonal 权重 | $w_{\mathrm{eik}}=0.1$ |
| SDF MLP | **1 层**（哈希后） | 颜色 MLP | 4 层 |
| 哈希 | 16 级 $2^5$–$2^{11}$，每项 8 维，每级 $\le 2^{22}$ 项 | 初始激活 | DTU 4 级 / TnT 8 级 |
| batch | DTU 1 / TnT 16 | marching cubes | DTU $512^3$ / TnT $2048^3$ |

解读两则：(i) "SDF MLP 仅 1 层"是哈希路线的标志性配置——容量几乎全在哈希表里（与教程第 09 章 09.1 ③"用内存换计算"完全同构），这也是它单迭代比 8 层 MLP 的 NeuS 更快的原因；(ii) DTU 与 TnT 的差异全部在尺度相关项（初始激活级数、batch、提取分辨率），核心日程不变——附录 in-the-wild 实验直接复用 TnT 配置重建无人机实拍场景，作者以此佐证超参的通用性。

## 6. 局限与后续影响

**可从论文设定直接读出的局限**（论文无独立 Limitations 小节，以下均为其实验设定与附录的事实推论，定性表述）：(i) **训练时长**仍是逐场景 500k 迭代——单迭代 0.1 s 量级折算为小时级/场景，比 NeuS 快但远非"秒级"（教程第 09 章 09.1 ③ 的"范式没有变"论断在此同样成立）；(ii) **依赖预先标定的精确位姿与内参**——全流程由 COLMAP 提供（附录 in-the-wild 段），未做联合位姿优化（BARF 一类），位姿漂移直接进入几何误差；(iii) 兴趣区域限定单位球内、哈希表内存开销大（16 级 × $2^{22}$ 项）；(iv) 曲率正则与日程超参（初始激活级数等）随场景尺度调整（DTU 4 级 vs TnT 8 级）；(v) 数值梯度每点 6 次额外 SDF 前向 + double backward 的训练期开销仍高于纯解析梯度（约 1.2 倍，附录速度表），推理期无差别。

**后续影响**：确立"哈希编码 + SDF 体渲染 + 数值梯度平滑"为大场景高质量神经表面的标准配方；数值梯度课程式退火被后续表面重建工作广泛沿用（coarse-to-fine 课程学习的代表作之一）；官方开源（Neuralangelo Research 仓库）及 Blender 选区插件（论文附录）使其进入实际重建管线；在机器人场景中，它与 3D 高斯路线的取舍见教程第 10 章。SLAM 视角的"在线版问题"——把位姿从预计算变成在线估计、把小时级压到实时——见 [NeRF-SLAM 精读](../../slam/精读/NeRF-SLAM_ICRA2023.md)（同为哈希体素辐射场，但用深度监督替代纯 RGB 的几何收敛）。

## 7. 与本项目对照

教程第 07 章 §7.4 是 Neuralangelo 的扩展小节（约一段），本精读是其完整展开；映射如下（左列 = 本 PDF 排版式号）：

| 论文 | 内容 | 教程对应 | 差异注记 |
|---|---|---|---|
| Eq.1–2 | 体渲染 / RGB 损失 | (7.1)；NeRF 教程 §3.5 | 同构（NeRF 离散合成） |
| Eq.3 | NeuS 不透明度 | (7.4) | 论文明言沿用 Wang et al.；完整推导在 [NeuS 精读 §4.5](./NeuS_NeurIPS2021.md) |
| Eq.4 | 多分辨率哈希编码 | 第 09 章 **(9.1)–(9.3)**（§9.1） | 教程按 Instant-NGP 原文 (Eq.2/3/4) 推导，构造相同 |
| Eq.5 | Eikonal 罚 | **(7.9)** ← 第 06 章 **(6.1)** | 同式（NeuS Eq.16 同款） |
| Eq.6–7 | 插值及其（局部）解析导数 | 第 09 章 (9.3) 的求导补全 | 教程未展开导数，本精读 §4.3 补齐 |
| Eq.8 | 中心差分数值法向 | §7.4 末段公式 | 同式；"为什么碎裂/为什么平滑"的完整论证见本精读 §4.3–4.4 |
| （无编号） | $\epsilon$ 退火 + 逐级激活日程 | §7.4 末段文字 | 论文 §3.4 + §4 实现细节 |
| Eq.9–10 | 曲率正则 / 总损失 | 教程未涉及 | 本精读 §4.5 补充 |

Instant-NGP 尚无独立精读（`tutorials/3d_reconstruction/精读/` 现存 MVSNet、Poisson、COLMAP 双篇与本篇二者），哈希编码的逐式推导以教程第 09 章 §09.1 为准；姊妹精读 [NeuS](./NeuS_NeurIPS2021.md) 提供渲染端 (Eq.1–3) 的全部前置推导。

## 配套阅读

- 论文 PDF：[papers/3d_reconstruction/frontier/arXiv-2306.03092_Neuralangelo.pdf](../../../papers/3d_reconstruction/frontier/arXiv-2306.03092_Neuralangelo.pdf)；LaTeX 源码：[papers/3d_reconstruction/latex/arXiv-2306.03092/](../../../papers/3d_reconstruction/latex/arXiv-2306.03092/)。
- 姊妹精读：[NeuS（NeurIPS 2021）](./NeuS_NeurIPS2021.md)——渲染端基础（Eq.1–3 即其 Eq.11/13 的移植）。
- 教程主线：[第 07 章](../07_神经隐式表面重建.md)（SDF 体渲染 (7.1)–(7.10) 与 §7.4 扩展）；[第 09 章](../09_哈希编码与前馈重建.md)（哈希编码 (9.1)–(9.3) 与"快"的两条路线）；[NeRF 教程第 03 章](../../nerf/03_数学原理与渲染方程.md)（体渲染出处）。
- 位姿来源：[COLMAP-SfM 精读](./COLMAP-SfM_CVPR2016.md)（本文全部实验的相机位姿/内参管线，TnT 上亦是对照基线）。
- 同门对照：[Instant-NGP 原文 PDF](../../../papers/3d_reconstruction/frontier/arXiv-2201.05989_Instant-NGP.pdf)（哈希编码出处；配教程第 09 章阅读）。
- 在线化视角：[NeRF-SLAM 精读](../../slam/精读/NeRF-SLAM_ICRA2023.md)（哈希体素辐射场 + SLAM 位姿/深度，把本文的离线逐场景范式推向实时）。
- 应用落点：[第 10 章｜机器人场景中的重建实战与选型](../10_机器人场景中的重建实战与选型.md)。

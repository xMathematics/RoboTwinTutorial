# 论文精读｜Occupancy Networks（CVPR 2019）

> **PDF**：[papers/3d_reconstruction/classics/arXiv-1812.03828_OccupancyNetworks.pdf](../../../papers/3d_reconstruction/classics/arXiv-1812.03828_OccupancyNetworks.pdf) ｜ **教程**：[第 06 章 §06.3](../06_学习式形状表示SDF与占据.md) ｜ **代码**：无（教学实现规划中；官方实现见论文脚注 4：`github.com/lmescheder/Occupancy-Networks`）

## 1. 论文信息与一句话贡献

- **题目**：Occupancy Networks: Learning 3D Reconstruction in Function Space
- **作者**：Lars Mescheder、Michael Oechsle、Michael Niemeyer（MPI 智能系统研究所 / 图宾根大学）、Sebastian Nowozin（Google AI Berlin）、Andreas Geiger
- **发表**：CVPR 2019；arXiv:1812.03828v2（本地 PDF 为 11 页正文版，**编号公式仅 (1)–(6)，无附录**——本文式号已逐条对照 PDF 排版核实）
- **一句话贡献**：把 3D 物体的表面表示为**深度神经网络分类器的连续决策边界**——学习占据隐函数（occupancy function）$o_\theta(\mathbf{p}, \mathbf{x}) : \mathbb{R}^3 \times \mathcal{X} \to [0,1]$，用最便宜的"内 / 外"二值标签做交叉熵监督，推理时以**多分辨率等值面提取**（Multiresolution IsoSurface Extraction, MISE）由粗到细地用 Marching Cubes 抽出网格——输出**连续、分辨率无关、内存随参数而非分辨率增长**，且无需模板网格、不产生自交，单视图 / 点云 / 低分辨率体素三种输入共用同一接口。

## 2. 问题与动机

**要解决什么**（§1）：学习式 3D 重建缺一个"既计算高效、又内存高效、还能表示任意拓扑高分辨率几何"的输出表示。

**三类既有表示的结构性缺口**（§1–2，对照[第 01 章](../01_3D重建问题与表示全景.md) (1.5)–(1.8) 的谱系）：
- **体素（voxel）**：内存随分辨率**立方增长**，主流方法被锁死在 $128^3$ 以下；多分辨率 / 八叉树等数据自适应结构能缓解，但实现复杂、需多次前向，且仍限于约 $256^3$。
- **点云（point cloud）**：缺连通性，需要额外后处理才能变成网格。
- **网格（mesh）**：基于模板变形的方法固定拓扑；生成自交网格的方法只覆盖简单拓扑；可端到端微分的 Marching Cubes（DMC）仍受底层 3D 网格内存限制。

**关键观察**（§3.1）：以上方法都在**离散输出空间**上做文章，而"物体表面"本身有一个更自然的连续刻画——**占据的二值指示函数的决策边界**。与其回归体素值 / 点坐标 / 模板顶点，不如直接学一个定义在**任意连续查询点**上的分类器：表示的"分辨率"于是与存储解耦（函数可以在任意精度上求值），监督也降到最便宜的二值标签。

**与 TSDF 路线的分工**（§2）：论文明确指出截断符号距离场（TSDF）"比占据难学得多——网络必须推理 3D 空间中的**距离函数**，而不只是分类一点在物体内外"。这是"判别式占据 vs 回归距离"路线分工的原文依据（与本仓库 [DeepSDF 精读](./DeepSDF_CVPR2019.md)构成姊妹对照）。

## 3. 方法总览

```
观测 x（三种模态任选其一）
 ├─ 单张图像 ──▶ ResNet18 图像编码器（ImageNet 预训练）
 ├─ 点云     ──▶ PointNet 编码器
 └─ 体素     ──▶ 3D CNN（VoxNet 型）编码器        （§3.4）
                    │  条件特征（生成式扩展中显式为隐码 z，式 (4)）
                    ▼
【占据网络】f_θ(p, x)：5 个 ResNet 块 + 条件批量归一化（CBN），
            输入查询点 p 与条件特征，输出占据概率 ∈ [0,1]   （式 (2)）
                    │
                    ▼
【训练】在包围盒内（加小 padding）均匀采 K 个点 p_ij，
        用交叉熵分类损失 L 拉向真值占据 o_ij ∈ {0,1}       （式 (3)）
                    │
                    ▼
【推理：MISE】初始 32³ 网格求值 → 标记 f ≥ τ 为占据 →
        相邻两点预测不同的体素记为"活跃" → 八叉树细分 ×8 →
        重复至目标分辨率 → Marching Cubes 提取               （式 (5) + §3.3）
                    │
                    ▼
【后处理】Fast-Quadric 网格简化 → 按式 (6) 用一阶 + 二阶梯度信息
          精修顶点（拉回等值面 + 法向对齐）；全程约 3 s/网格
```

与教程的对应：占据函数定义与编码器条件化 = [第 06 章](../06_学习式形状表示SDF与占据.md) 6.3 ②；交叉熵监督 = 6.3 ⑤ 第一、二步；等值面提取 = 第 05 章 MC 管线在占据场上的重跑（$f = 2o-1$，决策边界 $o=1/2$ 即零水平集，[第 01 章](../01_3D重建问题与表示全景.md) (1.11)、(1.2)）。

## 4. 关键公式推导（按论文原文式号）

**符号表**（沿用论文记号；教程对照见第 06 章 6.3）：

| 论文记号 | 含义 | 教程对应 |
|---|---|---|
| $\mathbf{p} \in \mathbb{R}^3$ | 连续查询点 | $\mathbf{p}$ |
| $\mathbf{x} \in \mathcal{X}$ | 一次观测（图像 / 点云 / 体素） | 条件 $\mathbf{x}$；编码器输出记 $\mathbf{z}$（任务书中"条件 c"） |
| $o(\mathbf{p}) \in \{0,1\}$ | **真值**占据（二值指示函数），$o_{ij} \equiv o(\mathbf{p}_{ij})$ | (1.11) 的离散采样 $b \in \{0,1\}$ |
| $f_\theta : \mathbb{R}^3 \times \mathcal{X} \to [0,1]$ | 占据网络（输出**概率**） | 6.3 ② 的 $o_\theta(\mathbf{p},\mathbf{x})$ |
| $K$ | 每个训练样本随机采的点数（消融取 2048） | 同 |
| $\tau$ | 等值面阈值（唯一超参数，交叉验证；单视图实验取 0.2） | (6.10) 的 $\tau$ |
| $\mathcal{L}(\cdot,\cdot)$ | 交叉熵分类损失 | (6.7) BCE（经 4.3 的修正变体） |

### 4.1 占据函数与占据网络（论文式 (1)(2)）

论文式 (1)（PDF 第 3 页排版核对）——**真值**占据是二值的：

$$o : \mathbb{R}^3 \to \{0,1\} \tag{1}$$

（依据：§3.1"the occupancy function of the 3D object"——物体内部的指示函数，与[第 01 章](../01_3D重建问题与表示全景.md) (1.11) 把占据写成 $o(\mathbf{x}) \in [0,1]$ 的**概率场**约定不同：概率语义是网络一侧的事，见下。）

**概率语义**：网络不输出 0/1，而输出"该点在物体内部"的概率——$f_\theta(\mathbf{p},\mathbf{x})$ 恰是伯努利分布的参数（§3.1："outputs a real number which represents the probability of occupancy"）。式 (1) 的二值真值扮演伯努利试验的结果，式 (2) 的网络输出扮演成功概率——交叉熵损失（4.2）正是这一对极大似然的负对数。

**条件化（论文式 (2)）**：对观测重建必须把 $\mathbf{x}$ 喂进函数。论文用一条"函数式等价"（柯里化的逆）：把"输入 $\mathbf{x}$、输出 $\mathbf{p} \mapsto \mathbb{R}$ 的函数"改写为"输入 $(\mathbf{p},\mathbf{x}) \in \mathbb{R}^3 \times \mathcal{X}$、输出实数的函数"（依据：多参数函数与返回函数的一元函数相互等价——柯里化等价，两条路径对每个固定 $\mathbf{x}$ 给出同一个 $\mathbf{p} \mapsto$ 值的映射），于是：

$$f_\theta : \mathbb{R}^3 \times \mathcal{X} \to [0,1] \tag{2}$$

这条等价把"任意模态"收进同一个函数签名：$\mathcal{X}$ 可以是图像、点云或体素，编码器只负责把 $\mathbf{x}$ 压成条件特征——**三种输入源统一接口的来历**（§3.4：图像用 ResNet18、点云用 PointNet、体素用 3D CNN；无条件生成时用 PointNet 充当式 (4) 的编码器 $g_\psi$；条件特征经条件批量归一化 CBN 注入 5 个 ResNet 块）。

**决策边界即表面**（§3.1）：二分类网络的分类器边界（decision boundary）$\{ \mathbf{p} : f_\theta = \text{阈值} \}$ 隐式表示物体表面——网络学的是分类，"免费"得到一个连续表面。这与 DeepSDF"回归距离、取零水平集"是两条不同的路线（见 §7 对比表）。

### 4.2 交叉熵损失：伯努利极大似然（论文式 (3)）

论文式 (3)（PDF 第 3 页排版核对）：

$$\mathcal{L}_{\mathcal{B}}(\theta) = \frac{1}{|\mathcal{B}|}\sum_{i=1}^{|\mathcal{B}|}\sum_{j=1}^{K} \mathcal{L}\big(f_\theta(\mathbf{p}_{ij}, \mathbf{x}_i),\, o_{ij}\big) \tag{3}$$

$\mathcal{L}$ 是交叉熵分类损失（§3.2 原文）。**逐层推导其显式形式**（论文未展开，教程 (6.6)–(6.7) 的内容）：

第一步（伯努利似然）：单个采样点上"内 / 外"是二元结果，$o_{ij} \in \{0,1\}$；网络给出概率 $o = f_\theta(\mathbf{p}_{ij},\mathbf{x}_i)$，则该点观测到真值的概率为 $o^{o_{ij}} (1-o)^{1-o_{ij}}$（依据：伯努利分布定义——$o_{ij}=1$ 时取 $o$、$o_{ij}=0$ 时取 $1-o$，两式合一）。

第二步（负对数 → 交叉熵）：独立采样的 $K$ 个点联合概率为各点乘积（依据：均匀独立采样），取负对数把乘积变求和（依据：$\log$ 单调且 $\log(ab) = \log a + \log b$）：

$$-\log \prod_j o^{o_{ij}}(1-o)^{1-o_{ij}} = \sum_j \Big[-o_{ij}\log o - (1-o_{ij})\log(1-o)\Big] =: \sum_j \mathrm{BCE}(o, o_{ij})$$

末步等式即二元交叉熵（binary cross entropy, BCE）的定义（依据：展开幂的对数）。最小化 (3) = 极大似然（依据：负对数似然最小化与似然最大化等价），两级平均分别对应批内样本（$1/|\mathcal{B}|$）与每样本 $K$ 个点的内层求和。

**采样方案**：均匀采自包围盒加小 padding 最优（§3.2；§4.6 消融证实，见 §5）——直觉是给空旷区与内部同等的"出镜率"，任何偏置（如内外各半）都在隐性地给模型注入"物体体积是 0.5"的先验（原文消融分析，定性）。

### 4.3 BCE 的饱和问题与 Lambert-W 修正（教程 (6.8)–(6.9)；公式见论文补充材料）

**饱和问题推导**（教程 6.3 ⑤ 第二步）：写 $o = \sigma(z)$（$z$ 为网络末端的 logits，$\sigma$ 为 sigmoid）。BCE 对 logits 的梯度：

$$\frac{\partial\,\mathrm{BCE}}{\partial z} = \frac{\partial\,\mathrm{BCE}}{\partial o}\cdot\frac{\partial o}{\partial z} = \Big(-\frac{b}{o} + \frac{1-b}{1-o}\Big)\cdot o(1-o) = -b(1-o) + (1-b)\,o = o - b$$

（依据：第一步链式法则；第二步 $\sigma'(z) = \sigma(z)(1-\sigma(z))$ 代入展开；第三步合并同类项。）于是对**远离表面的点**：$o$ 饱和于正确端（$b=1, o \to 1$），$o - b \to 0$——该点梯度趋零、退出训练（依据：上式）。后果：训练信号集中在表面附近，空旷区的占据结构欠约束。

**Lambert-W 修正的出处（如实标注）**：本地 PDF 为正文 11 页版，式 (1)–(6) 之外**无编号公式、无附录**；正文对 $\mathcal{L}$ 只写"cross-entropy classification loss"一句。带 Lambert W 函数（Lambert W function）的修正损失出自 **CVPR 2019 补充材料**，本精读**不对补充材料杜撰式号**，采用[教程第 06 章](../06_学习式形状表示SDF与占据.md) (6.8)–(6.9) 的表述（教程已注明出处）：

$$\mathcal{L}(p_i, o) = -\log\big(w_{c_1}(\mathrm{e}^{-o})\big) - \log\big(w_{c_2}(\mathrm{e}^{\,o-1})\big), \qquad w_\alpha(x) := \frac{W(\alpha x)}{\alpha} \tag{教程 (6.8)}$$

$$\log w_\alpha(x) + \alpha\, w_\alpha(x) = \log x \tag{教程 (6.9)}$$

其中 $W(x)\mathrm{e}^{W(x)} = x$ 为标准 Lambert W 函数，$c_1, c_2$ 编码内 / 外两类点的比例（类平衡）。**非饱和性可自证**（教程 6.3 ⑤ 第四步，无跳步复述）：对内支 $L_{\text{in}}(o) = -\log w_{c_2}(\mathrm{e}^{o-1})$，由 (6.9)（取 $x = \mathrm{e}^{o-1}$，$\log x = o-1$）有 $\log w + c_2 w = o - 1$；两边对 $o$ 求导（依据：隐函数求导）得 $(1/w + c_2)\,\mathrm{d}w/\mathrm{d}o = 1$；于是（依据：链式法则 $\mathrm{d}(-\log w)/\mathrm{d}o = -\tfrac{1}{w}\tfrac{\mathrm{d}w}{\mathrm{d}o}$）：

$$\frac{\mathrm{d}L_{\text{in}}}{\mathrm{d}o} = -\frac{1}{w}\cdot\frac{w}{1 + c_2 w} = -\frac{1}{1 + c_2 w}, \qquad \text{同理外支 } \frac{\mathrm{d}L_{\text{out}}}{\mathrm{d}o} = +\frac{1}{1 + c_1 w}$$

两支梯度模恒为 $1/(1+c\,w) \in (0,1)$——**处处有界且不为零**（依据：$w$ 有限），与 BCE 的"错误端发散、正确端经 $\sigma'$ 缩放后消失"形成对照。任务书中 "hierarchical potential functions" 一词在本地正文 PDF 中未出现（正文 "hierarchical" 仅指 MISE 的层级提取），如实以补充材料口径转述。

### 4.4 等值面与占据场 / SDF 的关系（论文式 (5)）

论文式 (5)（PDF 第 4 页排版核对）——提取目标即等值面：

$$\big\{\mathbf{p} \in \mathbb{R}^3 \;\big|\; f_\theta(\mathbf{p}, \mathbf{x}) = \tau\big\} \tag{5}$$

$\tau$ 是"占据网络唯一的超参数"，决定提取表面的"厚度"，实验中在验证集上交叉验证（原文脚注 2；单视图任务 $\tau = 0.2$，§4.2）。

**与 SDF 语言的对齐**（教程 6.3 ②）：把占据场仿射变换为 $g(\mathbf{p}) = 2 f_\theta(\mathbf{p},\mathbf{x}) - 1 \in [-1,1]$，则 $o = \tau$ 的等值面恰是 $g$ 的 $\hat\tau = 2\tau - 1$ 水平集（依据：等值集在可逆仿射下不变，$\{f = \tau\} = \{2f-1 = 2\tau-1\}$）。标准用法 $\tau = 0.5$ 时 $\hat\tau = 0$——占据决策边界与[第 01 章](../01_3D重建问题与表示全景.md) (1.2) 的零水平集抽象完全同构。**与真 SDF 的差别**：$f_\theta$ 无距离语义——$|1 - 2f_\theta|$ 不近似点到表面距离，$\nabla f_\theta$ 的模长也没有"= 1"的保证（对比 Eikonal [教程 (6.1)](../06_学习式形状表示SDF与占据.md)；DeepSDF 精读 4.1）；论文选择这条路线的理由正是"分类比回归距离好学"（§2，TSDF 难学论证）。

**光滑性讨论（如实转述）**：论文正文未对 0.5 等值面的光滑性给出定理式保证；其一阶光滑机制是经验性的——式 (6) 的第二项把 $\nabla f_\theta$ 的**方向**对齐到网格法向（且 §3.3 指出顶点法向可直接由 $\nabla f_\theta$ 反传得到），隐含"等值面附近 $f_\theta$ 沿法向单调变化"的假设（定性）。这与第 07 章 NeuS 用 Eikonal 显式保证 $\nabla f$ 为单位法向（[教程 (7.9)](../07_神经隐式表面重建.md)）形成"无正则 vs 有正则"的对照。

### 4.5 MISE：由粗到细的等值面提取（§3.3；式 (6) 为精修）

**MISE 流水线**（§3.3 + Fig. 2，逐条按原文）：
1. 在初始分辨率（实践中 $32^3$）上对体素化空间求值 $f_\theta(\mathbf{p},\mathbf{x})$；
2. 标记 $f_\theta \ge \tau$ 的格点为占据；
3. 标记"至少两个相邻格点预测不同"的体素为**活跃**（active）——正是当前分辨率下 Marching Cubes 会穿过网格的体素（依据：MC 只在符号变化的棱上产生顶点，[第 05 章](../05_从体素到网格表面提取.md) 5.1 的棱插值 $t = \frac{f_a}{f_a - f_b}$（(5.1)）只在两端异号的棱上执行）；
4. 把活跃体素细分为 8 个子体素（八叉树增量构建），对新引入格点回到第 2 步；
5. 到达目标分辨率后，在最终网格上运行 Marching Cubes——(5.1) 的插值与查表装配在占据场 $g = 2f_\theta - 1$ 上原样适用（回引[第 05 章](../05_从体素到网格表面提取.md) (5.1)–(5.2)）。

**收敛条件**（原文）：初始网格必须包含物体内、外部**每个连通分支**的点，否则漏掉 disconnected 部件；$32^3$ 几乎总是够。**价值**：不经 MISE 就要在高分辨率全网格上稠密求值 $f_\theta$——MISE 只评估活跃路径上的点，把"分辨率"与"求值次数"解耦（依据：§3.3 开头；八叉树做法引 Meagher 1982 / OctNet 等）。

**网格精修（论文式 (6)，PDF 第 4 页排版核对）**：先以 Fast-Quadric-Mesh-Simplification 简化，再从每个面采样点 $\mathbf{p}_k$ 最小化：

$$\sum_{k=1}^{K}\big(f_\theta(\mathbf{p}_k, x) - \tau\big)^2 + \lambda\left\|\frac{\nabla_\mathbf{p} f_\theta(\mathbf{p}_k, x)}{\|\nabla_\mathbf{p} f_\theta(\mathbf{p}_k, x)\|} - n(\mathbf{p}_k)\right\|^2 \tag{6}$$

- **第一项**：把面上的采样点"拉回"等值面（依据：$f_\theta - \tau = 0$ 的平方罚，残差为零当且仅当该点在 (5) 的等值面上）——抵消 MC 的分段线性离散化误差；
- **第二项**：归一化梯度 $\nabla f_\theta / \|\nabla f_\theta\|$ 对齐真网格法向 $n(\mathbf{p}_k)$（依据：等值面的法向即归一化梯度，[第 01 章](../01_3D重建问题与表示全景.md) (1.3) 的推广）——修平滑着色级别的细节；
- $\lambda = 0.01$（原文）。第二项对 $\theta$ 求导要先对 $\mathbf{p}$ 求一次梯度、再对梯度向量求梯度——**二阶梯度**（Hessian 向量积），论文用 Double-Backpropagation 高效实现（依据：§3.3 与 Pearlmutter/Drucker & Le Cun 技巧 [15]）。原文强调：这一步在体素表示上**不可能**——连续函数空间表示的独有红利；顶点法向也由一次反传免费得到。整个推理（MISE + MC + 精修）约 3 s/网格（§3.3）。

### 4.6 生成式扩展（论文式 (4)：VAE 化）

式 (3) 是判别式训练。论文 §3.2 进一步引入编码器 $g_\psi$：输入 $(\mathbf{p}_{ij}, o_{ij})$，输出高斯 $q_\psi(\mathbf{z} \mid \{(\mathbf{p}_{ij}, o_{ij})_{j=1:K}\})$ 的均值 $\mu_\psi$ 与标准差 $\sigma_\psi$，优化生成模型 $p((o_{ij})_{j=1:K} \mid (\mathbf{p}_{ij})_{j=1:K}) = \int p_\ell(\mathbf{z})\, p_\psi(o_{ij}\mid\mathbf{z})\,\mathrm{d}\mathbf{z}$ 的负对数似然的**下界**：

$$\mathcal{L}^{\text{gen}}_{\mathcal{B}}(\theta, \psi) = \frac{1}{|\mathcal{B}|}\sum_{i=1}^{|\mathcal{B}|}\sum_{j=1}^{K} \mathcal{L}\big(f_\theta(\mathbf{p}_{ij}, \mathbf{z}_i),\, o_{ij}\big) + \mathrm{KL}\big(q_\psi(\mathbf{z}\mid(\mathbf{p}_{ij},o_{ij})_{j=1:K})\,\big\|\,p_0(\mathbf{z})\big) \tag{4}$$

推导骨架（依据：变分推断的 ELBO，论文引 Kingma & Welling / Rezende et al. [21, 39, 59]）：负对数似然 $-\log p(o) = -\log\int q_\psi(\mathbf{z})\frac{p(o\mid\mathbf{z})p_\ell(\mathbf{z})}{q_\psi(\mathbf{z})}\mathrm{d}\mathbf{z} \le \mathbb{E}_{q_\psi}\big[-\log p(o\mid\mathbf{z})\big] + \mathrm{KL}(q_\psi\,\|\,p_\ell)$（依据：Jensen 不等式对凹函数 $\log$ 的方向）。期望项蒙特卡洛估计（采样 $\mathbf{z}_i \sim q_\psi$）即第一项，先验 $p_0$ 典型取高斯。这就是 §4.5 无条件生成实验（§4.5）的训练目标；判别式 (3) 与生成式 (4) 共用同一 $f_\theta$——条件接口的又一次复用。

## 5. 实验与结果解读

- **表示能力上界（§4.1，Fig. 3–4）**：把每个训练样本嵌入 512 维隐空间重训（"chair" 子集，4746 个样本、无验证集划分），网络以约 **6M 参数**达到对高分辨率网格的 **IoU 0.89**（均值）——连续表示能"装下"整个类别；同图对比：体素化 IoU 随分辨率上升逼近同值，但体素参数量立方增长、ONet 参数量与分辨率无关（Fig. 4）。IoU 0.89 ≠ 1 也直接量出了**容量上界**（§6）。
- **单视图重建（§4.2，Table 1，ShapeNet 13 类均值）**：ONet **IoU 0.591 最高**（AtlasNet 0.571、3D-R2N2 0.493、Pix2Mesh 0.480）；**法向一致性 0.834 最高**（AtlasNet 0.811、Pix2Mesh 0.772、3D-R2N2 0.695）；Chamfer-$L_1$ 0.215 次于 AtlasNet 的 0.175——而 PSGN / Pixel2Mesh / AtlasNet 均以 Chamfer 为训练目标，ONet 未用之（原文"surprisingly"段）。PSGN / AtlasNet **无法评 IoU**（输出非水密网格）——闭合水密输出本身就是占据路线的结构性优点。定性上（Fig. 5）：3D-R2N2 过度平滑、PSGN 无连通性、Pixel2Mesh 复杂拓扑处漏洞、AtlasNet 自交伪影，ONet 给出闭合且保细节的网格。$\tau=0.2$ 由验证集网格搜索定（对照 3D-R2N2 用其原论文的 0.4 阈值）。真实数据（KITTI、Online Products）仅合成数据训练仍泛化（Fig. 6，定性）。
- **点云补全（§4.3，Table 2，300 点 + $\sigma=0.05$ 高斯噪声）**：ONet IoU 0.778 / Chamfer 0.079 / 法向一致性 0.895 **三项全部最优**（对比 DMC 0.674/0.117/0.848、PSGN、3D-R2N2）；且全面好于单视图设定——点云输入歧义更小（原文解释）。
- **体素超分（§4.4，Table 3）**：从粗 $32^3$ 体素重建高分辨率网格：IoU 0.631→**0.703**、Chamfer 0.136→**0.109**、法向一致性 0.810→**0.879**——连"低分辨率体素"这种最异质的输入也能经统一接口吃下。
- **无条件下生成（§4.5，Fig. 7）**：以式 (4) 在 4 个 ShapeNet 类（car / airplane / sofa / chair）上无监督训练，可采样出可信新形状并可做隐空间插值。
- **消融（§4.6，Table 4）**：均匀采样（2048 点）最优（IoU 0.571）；"内外各半"最差（0.475，隐含"体积 0.5"偏置）；仅表面采样 0.536；点数降到 64 仍可用但变差。结构消融：去掉条件批量归一化伤害最大（0.571→0.522），去掉 ResNet 块次之（→0.559）。

## 6. 局限与后续影响

- **精度受网络容量与平滑性限制**：§4.1 的 0.89 是"全部训练形状"的重建上界——网络只能拟合它装得下的细节；等值面位置由分类边界决定，无距离监督纠正局部偏移（对比回归式路线，DeepSDF 精读 §7）。
- **饱和区学习困难**：朴素 BCE 在远离表面处梯度消失（4.3）；论文的训练依赖补充材料的 Lambert-W 修正，而这一细节极易被复现者忽略——空旷区占据结构欠约束是其后果。
- **无距离语义**：碰撞查询的 $|s|$ 与避障梯度 $\nabla s$ 不可用（[第 01 章](../01_3D重建问题与表示全景.md) ③ 的判据），机器人下游要用距离时仍需回归式表示或事后转换（[第 10 章](../10_机器人场景中的重建实战与选型.md)）。
- **编码器路线的分布依赖**：一次前向的代价是编码器必须与训练观测分布匹配（点数、噪声、模态），换观测形态要重训（与自解码器的取舍见 §7 与 DeepSDF 精读 4.5）。
- **单物体、规范坐标设定**：单位球归一化的 ShapeNet 物体级重建；场景级重建（平移等变、局部特征、大尺度坐标）由**Convolutional Occupancy Networks**（Peng, Niemeyer, Mescheder, Pollefeys, Geiger, ECCV 2020）接棒——ONet 的占据思想 + 卷积特征场的直接后裔。此后神经隐式表示沿"占据 / SDF / 符号距离概率化"分化，第 07 章的 NeuS（[教程](../07_神经隐式表面重建.md)）把 SDF 接入体渲染，其 Eikonal 罚 (7.9) 补上的正是占据路线放弃的"距离微分语义"。

## 7. 与本项目对照

**论文式号 ↔ 教程第 06 章映射**（已逐条核实双方编号）：

| 论文 | 内容 | 教程第 06 章 |
|---|---|---|
| 式 (1) | 真值占据 $o:\mathbb{R}^3\to\{0,1\}$（二值） | (1.11) 占据约定 + 6.3 ⑤ 第一步的 $b \in \{0,1\}$ |
| 式 (2) | 占据网络 $f_\theta:\mathbb{R}^3\times\mathcal{X}\to[0,1]$ | 6.3 ②（教程记 $o_\theta(\mathbf{p},\mathbf{x})$） |
| 式 (3) | 小批交叉熵损失 | (6.6) 伯努利似然 → (6.7) BCE 的推导对象 |
| §3.2 末段 | Lambert-W 修正损失（补充材料） | (6.8)–(6.9)（教程已注明"论文附录 / supplementary"；本精读 4.3 自证非饱和性） |
| 式 (5) + 脚注 2 | 等值面 $\{f_\theta = \tau\}$、$\tau$ 唯一超参 | (6.10)；$\tau$ 取 0.5 ↔ 零水平集（(1.2)、(1.11)） |
| §3.3 | MISE 八叉树 + 末级 Marching Cubes | 第 05 章 (5.1)–(5.2) 管线在 $g=2f_\theta-1$ 上复用 |
| 式 (6) | 等值面 + 法向精修（Double-Backprop） | 第 05 章 (5.2) 法向 = 归一化梯度的"可学习版" |
| §3.2 式 (4) | VAE 生成式扩展 | 教程未展开（超出 6.3 范围），本精读 4.6 补 |

**判别式 vs 回归式：与 DeepSDF 的对照表**（互引姊妹精读 [DeepSDF_CVPR2019.md](./DeepSDF_CVPR2019.md)）：

| 维度 | Occupancy Networks（判别式占据） | DeepSDF（回归式 SDF） |
|---|---|---|
| 输出语义 | 内 / 外概率，$\tau$ 等值面即表面 | 带符号距离，零水平集即表面 |
| 监督信号 | 内外二值标签（任意水密网格可采，最便宜） | 真值 SDF 值（需距离变换，贵；带内才有效） |
| 损失 | 交叉熵（BCE + Lambert-W 修正，见补充材料） | clamp 截断 L1（论文式 (4)；教程 (6.4)） |
| 微分语义 | 无（$\nabla f$ 只当法向用，无范数保证） | 弱（无 Eikonal；带外不可信；NeuS (7.9) 补足） |
| 条件化 | 编码器（ResNet18 / PointNet / 3D CNN），一次前向 | 自解码器：测试时对 $\mathbf{z}$ 做 MAP 优化（其式 (10)） |
| 测试时延 | 快（约 3 s/网格，含精修） | 慢（其 Table 1 报 9.72 s；结论自认 auto-decode 费时） |
| 教程定位 | 6.3（(6.6)–(6.10)） | 6.2（(6.2)–(6.5)） |

选型结论与[第 06 章](../06_学习式形状表示SDF与占据.md) 6.4 三方对比一致：只要网格资产、监督最省 → ONet；要距离 / 法向语义（碰撞、规划）→ DeepSDF 路线 + Eikonal 正则；两者都把"提取"外包给[第 05 章](../05_从体素到网格表面提取.md)的 MC 管线。

## 配套阅读

- 原论文：[arXiv-1812.03828_OccupancyNetworks.pdf](../../../papers/3d_reconstruction/classics/arXiv-1812.03828_OccupancyNetworks.pdf)（正文式 (1)–(6)；Lambert-W 修正与 "hierarchical potential functions" 表述在 CVPR 2019 补充材料，本地未收录）。
- 教程主章节：[第 06 章｜学习式形状表示](../06_学习式形状表示SDF与占据.md)（6.3 节 (6.6)–(6.10) 推导）；提取管线：[第 05 章｜从体素到网格表面提取](../05_从体素到网格表面提取.md)；定义回引：[第 01 章](../01_3D重建问题与表示全景.md)（(1.2)、(1.11)）；后续脉络：[第 07 章](../07_神经隐式表面重建.md)（Eikonal (7.9)）、[第 10 章](../10_机器人场景中的重建实战与选型.md)（选型矩阵）。
- 姊妹精读：[DeepSDF（CVPR 2019）](./DeepSDF_CVPR2019.md)（判别式 vs 回归式的另一极）；同章已有：[MVSNet（ECCV 2018）](./MVSNet_ECCV2018.md)、[Poisson 表面重建（SGP 2006）](./PoissonRecon_SGP2006.md)（占据路线的替代"水密化"前端）。
- 后续脉络（论文库未收录 PDF，文字提及）：Convolutional Occupancy Networks（ECCV 2020）、Deep Level Sets / DMC（CVPR 2019）。

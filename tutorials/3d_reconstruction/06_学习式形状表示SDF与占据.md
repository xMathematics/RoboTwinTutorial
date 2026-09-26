# 第 06 章｜学习式形状表示：SDF 与占据

前几章的重建都是"逐场景计算"：位姿、深度、融合、提取，每一步由几何约束驱动。
但单视图或稀疏扫描是**欠约束**问题——看不见的背面必须"猜"，而手工正则（平滑、对称）猜不准，数据里却藏着形状先验（网络见过几千把椅子，知道椅背后面长什么样）。
本章把形状表示参数化为神经网络，沿两条主线展开：先弄清理想 SDF 的微分性质——**Eikonal 方程**（6.1，编号定为 (6.1)，第 07 章将回引）；再给出两个学习式隐函数的代表作——**DeepSDF**（6.2，自解码器把形状压进隐空间）与 **Occupancy Networks**（6.3，占据概率 + 伯努利监督）；最后做 SDF / 占据 / 体素回归的三方对比（6.4），为第 07 章"从体渲染逼出几何"铺垫。

**记号约定**：SDF $s(\mathbf{x})$，外正内负、零水平集为表面（第 01 章 (1.13)；第 07 章沿用 NeuS 论文记号 $f$，与 $s$ 同义）；占据 $o(\mathbf{x}) \in [0,1]$（(1.11)）；网络参数 $\theta$；形状隐码 $\mathbf{z}$；体素场 $F(\mathbf{x})$（TSDF，(1.14)）。

## 6.1 SDF 的数学性质：Eikonal 方程

**① 问题场景**：现在要让网络 $s_\theta(\mathbf{x})$ 回归 SDF——网络的输出只被损失"拉"向采样点的真值，于是有两个隐患：
(a) 零水平集位置大致对了，但梯度尺度任意（$\|\nabla s_\theta\| \ne 1$），(1.3) 的法向查询与第 07 章的无偏性分析（(7.7) 的斜率解读）全部失效；
(b) 远离采样点处网络可以任意扭曲，"距离"语义丢失。
要把它变成训练约束，先得回答：**理想 SDF 到底满足什么微分方程？**
手头：(1.13) 的距离定义；缺：定义的微分推论；想要：一个可逐点检验、可进损失的方程。

**② 解决方法**：从 (1.13) 出发沿法向线做一阶分析，导出 **Eikonal 方程**（Eikonal equation）：
$$\big\|\nabla s(\mathbf{x})\big\| = 1 \qquad (\text{在可微点处几乎处处成立}). \tag{6.1}$$
训练中把它作为**软罚项**逐点施加——具体形式与训练细节在第 07 章给出（(7.9)，回引不重推），本章负责论证"为什么罚它是对的"。

**③ 选型理由**：为什么用软罚而非硬约束（如输出层归一化强制 $\|\nabla s\| = 1$）：
硬约束限制表达、难以与可微管线耦合，且只在网络参数化能力范围内近似成立；罚项逐点局部、零额外参数、与距离函数的几何意义严格对应（第 07 章 7.4 ③ 的同款分工论证）。
**合法性问题**（罚它会不会错杀正确解）：
(a) 是真 SDF 的**必要**性质——⑤ 证明真 SDF 满足 (6.1)，故罚项不惩罚正确解；
(b) 但**不充分**——(6.1) 只约束梯度的模长，不携带符号与"全局最近邻"信息（满足 (6.1) 的局部解未必是真距离函数），需符号 / 表面监督兜底；
(c) 中轴（medial axis）处距离函数不可微，(6.1) 只能几乎处处成立——罚项在采样点上离散施加，天然回避了测度零的例外集。

**④ 理论依据**：Eikonal 方程之名来自几何光学中的光程方程，作为水平集方法（level set method）的核心方程系统化于 Osher & Sethian (JCP 1988) 与 Sethian 的 *Level Set Methods* 教材；
SDF 定义回引第 01 章 (1.13)、法向—梯度关系回引 (1.3)；作为神经隐式表示训练正则的现代用法见第 07 章（NeuS (7.9)、Neuralangelo 同款）。

**⑤ 完整推导**（每步注明依据）：

**第一步（沿法向线的取值规律）。**
设 $\mathbf{y} \in S$ 为表面上一点，$\hat{\mathbf{n}}$ 为该处单位外法向（(1.3)），考虑法向线的两个半支：
$\gamma_+(t) = \mathbf{y} + t\,\hat{\mathbf{n}}$（向外）与 $\gamma_-(t) = \mathbf{y} - t\,\hat{\mathbf{n}}$（向内）。
在"最近点唯一"的范围内（表面 reach 以内、中轴之外），两条支上都成立
$$s(\gamma_\pm(t)) = \pm\, t \qquad (0 \le t < R)$$
（依据：$\gamma_\pm(t)$ 到表面的最近点是 $\mathbf{y}$，(1.13) 给出带符号距离 = 行走弧长 $t$，符号由内外约定给出；唯一性在该法向段内成立、中轴上失效——定性。）
直观读法：**沿法向每走单位长度，距离值恰好变化 1**——等价地，从场内任意点 $\mathbf{x}_0$（$s(\mathbf{x}_0) = c$）出发沿其法向线行进 $t$，值变为 $c + t$：距离场以"单位速率"沿线变化。

**第二步（链式求导）。**
对 $s(\gamma_\pm(t)) = \pm t$ 关于 $t$ 求导（依据：链式法则）：
$$\frac{\mathrm{d}}{\mathrm{d}t}\,s(\gamma_\pm(t)) = \nabla s(\gamma_\pm(t))\cdot\gamma_\pm'(t) = \pm 1, \qquad \gamma_\pm'(t) = \pm\,\hat{\mathbf{n}}$$
其中 $\gamma_\pm'(t) = \pm\hat{\mathbf{n}}$ 是**单位向量**（依据：$\hat{\mathbf{n}}$ 为单位法向，(1.3)）。
代入化简（$\pm$ 两支给出同一式）：
$$\nabla s \cdot \hat{\mathbf{n}} = 1$$

**第三步（梯度没有切向分量）。**
从 $\gamma_\pm(t)$ 沿切向横移 $\delta\boldsymbol\epsilon$（$\boldsymbol\epsilon \perp \hat{\mathbf{n}}$）：到最近点 $\mathbf{y}$ 的距离为
$$\sqrt{t^2 + \delta^2\|\boldsymbol\epsilon\|^2} = t + O(\delta^2)$$
（依据：勾股定理展开——切向位移对"到 $\mathbf{y}$ 的距离"只有二阶影响；在最近点唯一的邻域内它仍是全局最近点，依据：reach 内最近点映射连续，定性。）
故 $s$ 沿切向一阶不变 ⟹ $\nabla s$ 无切向分量 ⟹ $\nabla s \,\|\, \hat{\mathbf{n}}$（依据：水平集切空间的正交补即梯度方向，(1.3) 的推广）。

**第四步（夹逼收口）。**
由第二、三步，$\nabla s = \hat{\mathbf{n}}$，故 $\|\nabla s\| = 1$。
换一条更规范的收口，分两个不等式：
下界——柯西–施瓦茨给 $\|\nabla s\| \ge |\nabla s \cdot \hat{\mathbf{n}}| = 1$（依据：柯西–施瓦茨不等式 + 第二步）；
上界——$|s(\mathbf{x}) - s(\mathbf{x}')| \le \|\mathbf{x} - \mathbf{x}'\|$（依据：三角不等式，$\big|\,\mathrm{dist}(\mathbf{x}, S) - \mathrm{dist}(\mathbf{x}', S)\big| \le \|\mathbf{x} - \mathbf{x}'\|$），即 $s$ 是 1-Lipschitz 的，在可微点处 $\|\nabla s\| \le 1$（依据：Lipschitz 函数的梯度界；Rademacher 定理保证 Lipschitz 函数几乎处处可微）。
两边夹逼得 (6.1)。

**接回训练。**
(6.1) 是"距离语义"的微分化身：网络若被数据项拉出 $\|\nabla s_\theta\| \ne 1$ 的解，Eikonal 罚把梯度尺度拉回；
同时它钉住 $s_\theta$ 的数值尺度，消除"压扁 $s$、放大某系数"的退化方向（第 07 章 7.4 ⑤ 的作用链）。

## 6.2 DeepSDF：把形状先验压进隐空间

**① 问题场景**：手头：机器人单视角的 RGB-D（或稀疏的部分扫描）——只有物体朝向相机的一半有观测。
缺：背面完全无约束；手工正则（平滑、对称、模板）猜不准——椅腿不光滑、对称不普遍。
想要的：一种"见过大量形状"的表示，从数据先验里**学**出怎么补全。
欠约束的本质：观测只钉住形状流形上的一个低维切片，解空间是一整族形状——需要一个变量来索引"是哪一个"。

**② 解决方法**：**条件隐函数**（conditional implicit function）$s_\theta(\mathbf{x}, \mathbf{z})$：每个形状由低维隐码 $\mathbf{z}$ 索引，形状族为
$$s_\theta(\cdot, \mathbf{z}) \approx s^{(i)}(\cdot), \qquad \mathbf{z}_i \leftrightarrow \text{形状 } X_i \tag{6.2}$$
采用**自解码器**（auto-decoder）方案：不训练编码器，每个训练形状配一个可学习隐码 $\mathbf{z}_i$，与网络参数 $\theta$ 联合优化（重建自己 = 自解码之名；依据：原文 §4.1–4.2"decoder-only architectures"）；
训练同时在隐码空间上维持高斯先验 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \sigma^2 I)$（以正则项形式进入目标，⑤ 第二、三步）。
测试时固定 $\theta$，对观测做最大后验（Maximum a Posteriori, MAP）估计求 $\mathbf{z}$（(6.5)），再查询 $s_\theta(\cdot, \hat{\mathbf{z}})$ 并用第 05 章的 MC 提取网格。
实现骨架（操作步骤，不含理论）：8 层全连接、512 维、ReLU，输出 tanh；$\mathbf{z}$ 与 $\mathbf{x}$ 拼接进各层（原文 §3 参数）。

**③ 选型理由**（本节重点：判别式占据 vs 回归 SDF 的权衡）。两条表示路线对比：
- **回归 SDF**：输出带符号距离，信息最丰富（距离 + 法向 + 内外一体，(1.3) 三问全答）；但 监督要求真值 SDF 值（需对水密网格做距离变换，数据贵）， 距离语义需要 (6.1) 维持。
- **判别式占据**：输出有界概率、BCE 监督只需内外标签（数据便宜），无需 Eikonal（决策边界即表面，无"距离"可退化）；但没有距离语义（6.3 与 6.4 展开）。
DeepSDF 取 SDF 是为了"一次查询回答三问"（第 01 章 ① 的判据）。
**关于 Eikonal 的权衡**：DeepSDF 原文未加 Eikonal 正则，其替代权衡是 **clamp 策略**（⑤ 第一步）——把有效监督限制在表面 $\delta$ 带内，带外不参与损失、退化空间被有意放弃；代价是远离表面处 $s_\theta$ 不可信（定性）。后续工作（如第 07 章的 NeuS）把 (6.1) 显式加为 (7.9)，几何质量提升、换来二阶梯度开销（定性）。
自解码器（而非编码器）的选择：测试时对**任意形态**观测灵活——(6.5) 对样本数量与分布不作要求（依据：原文"valid for samples of arbitrary size and distribution"，梯度可逐样本计算）；代价是测试时要逐例优化 $\mathbf{z}$（与 ONet 的编码器路线对比，6.3 ③）。

**④ 理论依据**：Park, Florence, Straub, Newcombe, Lovegrove, *"DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation"*, CVPR 2019（[本地 PDF](../../papers/3d_reconstruction/classics/arXiv-1901.05103_DeepSDF.pdf)；式 (4) 的 L1 + clamp 损失、§4.2 的先验设定与式 (9)(10) 的 MAP 目标已对照原文核实）；
概率论的贝叶斯法则与高斯负对数。

**⑤ 完整推导**（每步注明依据）：

**第一步（损失与 clamp 的含义）。**
定义截断算子：
$$\mathrm{clamp}(x, \delta) = \min\big(\delta,\ \max(-\delta,\ x)\big) \tag{6.3}$$
（依据：原文式 (4) 的定义。）
DeepSDF 的逐点损失（教程记为平方形式）：
$$\mathcal{L}(\theta, \mathbf{z}) = \sum_{\mathbf{x} \in X} \Big|\,\mathrm{clamp}\big(s_\theta(\mathbf{x}, \mathbf{z}), \delta\big) - \mathrm{clamp}\big(s(\mathbf{x}), \delta\big)\,\Big|^2 \tag{6.4}$$
**事实注记**：原文式 (4) 采用 L1 范数 $|\,\cdot\,|$（$\mathcal{L}(f_\theta(\mathbf{x}), s) = |\mathrm{clamp}(f_\theta(\mathbf{x}), \delta) - \mathrm{clamp}(s, \delta)|$）；本教程写成平方形式，以便与 (6.5) 的二次正则、第 07 章 (7.9) 的平方罚记号统一——clamp 各项的分析对两种范数同样成立，差异只在远处残差的权重衰减速度：L1 对离群采样更稳健（定性）。
各项含义：
真值端 clamp——带外真值只携带"至少 $\delta$ 远"的语义、不携带表面细节信息（与 TSDF 截断 (1.14) 完全同款："截断换鲁棒"，第 01 章 01.3 ③）；
预测端 clamp——带外梯度为零，网络容量集中于表面附近。
$\delta$ 的权衡（依据：原文对 $\delta$ 的说明）：大 $\delta$——每个样本都提供"安全空间"信息，利于快速光线步进；小 $\delta$——容量集中到表面细节（原文实现取 $\delta = 0.1$ 量级）。

**第二步（自解码器的训练目标）。**
对 $N$ 个训练形状联合优化（依据：原文式 (9)）：
$$\arg\min_{\theta, \{\mathbf{z}_i\}} \sum_{i=1}^{N} \Big( \sum_{j} \mathcal{L}\big(f_\theta(\mathbf{z}_i, \mathbf{x}_j),\ s_j\big) + \frac{1}{\sigma^2}\|\mathbf{z}_i\|_2^2 \Big)$$
（依据：重建项 = (6.4) 逐形状求和；正则项 = 隐码先验的代价，其概率解读见第三步；$\sigma^2$ 为先验方差超参。）

**第三步（概率解读：先验与似然）。**
隐码先验取零均值球形高斯（依据：原文 §4.2"zero-mean multivariate Gaussian with spherical covariance $\sigma^2 I$"——自解码器没有编码器，先验即以正则形式"塞进"训练目标）：
$$p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \sigma^2 I)$$
取负对数（依据：多元高斯密度取负对数，展开完成平方；常数含 $\sigma$ 与归一化因子、与优化变量无关）：
$$-\log p(\mathbf{z}) = \frac{\|\mathbf{z}\|^2}{2\sigma^2} + \mathrm{const}$$
似然取损失指数化（依据：原文式 (8)——损失即负对数似然）：
$$p\big(\{s_j\} \mid \mathbf{z}\big) \;\propto\; \exp\Big(-\sum_j \mathcal{L}\big(s_\theta(\mathbf{x}_j, \mathbf{z}),\ s_j\big)\Big) \;\Longrightarrow\; -\log p = \sum_j \mathcal{L} + \mathrm{const}$$

**第四步（测试时补全 = 对隐码的 MAP 推断）。**
观测给定后，按贝叶斯法则（依据：贝叶斯法则）：
$$p(\mathbf{z} \mid \text{观测}) \propto p(\text{观测} \mid \mathbf{z})\,p(\mathbf{z})$$
MAP = 最大化后验 = 最小化负对数后验（依据：取负对数、丢与 $\mathbf{z}$ 无关的常数；第一、三步的两个负对数相加）：
$$\hat{\mathbf{z}} = \arg\min_{\mathbf{z}} \;\; \frac{\|\mathbf{z}\|^2}{2\sigma^2} + \sum_{\mathbf{x} \in \text{观测}} \mathcal{L}\big(s_\theta(\mathbf{x}, \mathbf{z}),\ s(\mathbf{x})\big) \tag{6.5}$$
（即原文式 (10)。）
**直觉**：数据项把 $\hat{\mathbf{z}}$ 拉向"能重建已观测部分"的码，先验项把 $\hat{\mathbf{z}}$ 拉回训练形状流形——**看不见的背面由先验形状族补出**：解出的 $\hat{\mathbf{z}}$ 索引"与观测最吻合的那个已知类形状"，其未观测区域即网络对该形状的先验猜测。"猜"的依据全部来自训练集，这正是 ①"数据里藏着先验"的形式化。

## 6.3 Occupancy Networks：占据概率与 Lambert-W 修正

**① 问题场景**：6.2 末尾留下对比：若下游只需要"哪里有表面"（重建网格给渲染 / 抓取），距离语义未必必需，而 SDF 的监督却贵（真值距离变换）。
能否用**最便宜的监督**——每个点一个"内 / 外"二值标签——学出连续、可提取网格的表示？
手头：带观测（点云 / 单图 / 体素）+ 任一水密网格（可无限采样标签）；缺：连续的内外决策函数；想要：可直接提取网格、且监督信号简单的学习式表示。

**② 解决方法**：Occupancy Networks（ONet）学习**占据概率**（occupancy probability）：
$$o_\theta(\mathbf{p}, \mathbf{x}) \in [0, 1] \quad\Longleftrightarrow\quad f_\theta: \mathbb{R}^3 \times \mathcal{X} \to [0,1]$$
（依据：原文式 (2) 的占据网络 $f_\theta$；$\mathbf{p}$ 为查询点、$\mathbf{x}$ 为观测输入，经编码器 $g_\psi$ 压成条件特征 / 隐码——图像用 ResNet、点云用 PointNet、体素用 3D CNN，原文 §3.4。）
表面 = 占据的等值面 $\{\mathbf{p} : o_\theta(\mathbf{p}, \mathbf{x}) = \tau\}$，用**多分辨率等值面提取**（Multiresolution IsoSurface Extraction, MISE）逐步细分活跃体素、最终以 Marching Cubes 提取（依据：原文 §3.3）——即第 05 章 5.1 的管线在占据场上重跑（回引 (5.1)–(5.2)；占据场取 $f = 2o - 1$，则决策边界 $o = 0.5$ 与零水平集语言完全一致，回引 (1.11) 的约定）。

**③ 选型理由**：与 DeepSDF 对比：占据是**判别式**路线——输出有界、sigmoid 输出天然校准为概率，BCE 监督只要内外标签（任何水密网格即可采样，数据便宜），也无需 Eikonal；代价是无距离语义——碰撞查询的 $|s|$ 与避障梯度 $\nabla s$ 不可用（第 01 章 ③ 的判据），法向只能借 $\nabla o$ 近似、无范数保证。
编码器路线（ONet）与自解码器路线（DeepSDF）互补：前者一次前向、快，但编码器须与训练观测分布匹配；后者对任意观测灵活但要逐例优化 $\mathbf{z}$（6.2 ③）。
与体素回归对比：体素受立方内存墙限制（依据：原文 §2.1 综述——主流体素法只能处理低分辨率），ONet 连续、分辨率无关、内存随参数而非分辨率增长。

**④ 理论依据**：Mescheder, Oechsle, Niemeyer, Nowozin, Geiger, *"Occupancy Networks: Learning 3D Reconstruction in Function Space"*, CVPR 2019（[本地 PDF](../../papers/3d_reconstruction/classics/arXiv-1812.03828_OccupancyNetworks.pdf)；占据函数式 (1)–(2)、交叉熵损失式 (3)、等值面式 (5)、MISE 与最终 Marching Cubes 已对照原文核实）；
伯努利分布的极大似然；Lambert-W 修正出自其**论文补充材料**（本地 PDF 为正文 11 页版、未含附录，公式以 CVPR 2019 supplementary 为准，本节给出可自证的推导）。

**⑤ 完整推导**（每步注明依据）：

**第一步（伯努利似然 → BCE）。**
对采样点 $\mathbf{p}$，"在物体内 / 外"是二元结果，记 $b \in \{0, 1\}$（依据：(1.11) 占据的离散采样）。
网络给出占据概率 $o = o_\theta(\mathbf{p}, \mathbf{z})$，观测的似然为
$$p(b \mid \mathbf{p}, \mathbf{z}) = o^{\,b}\,(1-o)^{\,1-b} \tag{6.6}$$
（依据：伯努利分布定义——$b = 1$ 时取 $o$、$b = 0$ 时取 $1 - o$，两式合一。）
取负对数（依据：极大似然 = 最小化负对数似然；对数把幂变乘积）：
$$-\log p(b \mid \mathbf{p}, \mathbf{z}) = -b\log o - (1 - b)\log(1 - o) =: \mathrm{BCE}(o, b) \tag{6.7}$$
即**二元交叉熵**（binary cross entropy, BCE）。
训练对每个形状随机采 $K$ 个点、最小化平均 BCE（依据：原文式 (3)；采样方案用包围盒内加 padding 的均匀采样，原文 §4 消融）。

**第二步（BCE 的饱和问题）。**
写 $o = \sigma(z)$（$z$ 为网络输出的 logits），BCE 对 $z$ 的梯度（依据：$\sigma' = \sigma(1 - \sigma)$，链式法则后两项合并）：
$$\frac{\partial\, \mathrm{BCE}}{\partial z} = -b\,\frac{\sigma'}{o} + (1 - b)\,\frac{\sigma'}{1 - o} = \sigma(z) - b = o - b$$
于是对**远离表面的点**：$o$ 饱和于 0/1、$o - b \approx 0$，该点对训练的贡献趋零（依据：上式）。
后果：训练信号几乎全部来自表面附近的点，**空旷区域**的占据结构（等值面该往哪里弯）缺少约束，网络学不到空旷区（定性）。

**第三步（Lambert-W 修正：公式与出处）。**
ONet 论文附录用带 **Lambert W 函数**（Lambert W function）的变换修正 BCE（出处：CVPR 2019 supplementary material）：
$$\mathcal{L}(p_i, o) = -\log\big(w_{c_1}(\mathrm{e}^{-o})\big) - \log\big(w_{c_2}(\mathrm{e}^{\,o-1})\big), \qquad w_\alpha(x) := \frac{W(\alpha x)}{\alpha} \tag{6.8}$$
其中 $W$ 为标准 Lambert W 函数（$W(x)\,\mathrm{e}^{W(x)} = x$ 之解），$w_\alpha$ 称广义 Lambert W 函数；等价的隐式定义（依据：由 $w_\alpha$ 满足 $w\,\mathrm{e}^{\alpha w} = x$，两边取对数）：
$$\log w_\alpha(x) + \alpha\, w_\alpha(x) = \log x \tag{6.9}$$
$c_1, c_2$ 为编码**类平衡**的常数（对应训练集中点在物体内 / 外的比例；依据：原文附录表述）。

**第四步（非饱和性，可自证）。**
对内支 $L_{\text{in}}(o) = -\log w_{c_2}(\mathrm{e}^{o-1})$：由 (6.9)，$\log w + c_2 w = o - 1$。
两边对 $o$ 求导（依据：隐函数求导）：
$$(1/w + c_2)\,\frac{\mathrm{d}w}{\mathrm{d}o} = 1$$
于是（依据：链式法则 $-\frac{1}{w}\frac{\mathrm{d}w}{\mathrm{d}o}$）：
$$\frac{\mathrm{d} L_{\text{in}}}{\mathrm{d}o} = -\frac{1}{w}\cdot\frac{w}{1 + c_2 w} = -\frac{1}{1 + c_2 w}, \qquad \text{同理外支 } \frac{\mathrm{d} L_{\text{out}}}{\mathrm{d}o} = +\frac{1}{1 + c_1 w}$$
两支对 $o$ 的梯度模恒为 $1/(1 + c\,w) \in (0, 1)$——**处处有界且不为零**（依据：$w$ 有限）。
对比 BCE 的 $-1/o$ 与 $1/(1-o)$（错误端发散、正确端经 $\sigma'$ 缩放后随 $o - b$ 消失，第二步）。
直觉：修正后的损失把 BCE 的 $\log$ 项换成由隐方程 (6.9) 定义的形状——每个点（无论离表面多远）都持续提供**有界推力**，空旷区域的占据结构由全体样本共同塑造，而非只由表面附近的点决定；$c_1, c_2$ 再把数量悬殊的内 / 外两类点配平（定性）。

**第五步（提取）。**
训练好后取等值面：
$$\big\{\mathbf{p} : o_\theta(\mathbf{p}, \mathbf{z}) = \tau\big\} \tag{6.10}$$
（依据：原文式 (5)；$\tau$ 是决定表面"厚度"的唯一超参数，原文脚注；决策边界语义对应指示函数的跃迁中点，标准用法在 $\tau = 0.5$ 附近。）
提取流程 = MISE 粗到细标记活跃体素 + 最终分辨率上 Marching Cubes（依据：原文 §3.3 与 Fig.2）——**回引第 05 章**：(5.1) 的棱上插值与查表装配在 $o$ 场上原样适用。

## 6.4 SDF vs 占据 vs 体素回归：三方选型（简短对比）

**① 问题场景**：机器人下游任务各异——碰撞要距离、抓取要表面 + 法向、展示只要网格；三种学习式表示怎么选？

**② 解决方法**：按"信息量 / 监督需求 / 提取方式"三轴对照：

| 表示 | 信息量 | 监督信号需求 | 提取方式 |
|------|--------|--------------|----------|
| 回归 SDF（DeepSDF，6.2） | 最富：距离 + 法向（$\nabla s$）+ 内外，(1.3) 三问全答 | 真值 SDF 值（需水密网格距离变换，贵）+ (6.1) 正则 | 零水平集 MC（第 05 章） |
| 占据（ONet，6.3） | 只判内外，无距离语义 | 内 / 外二值标签（任意水密网格可采，最便宜） | $o = \tau$ 等值面 MC（(6.10)） |
| 体素回归 | 逐体素值，分辨率绑定 | 稠密体素化监督（最贵） | 直接查询 / 等值面 |

**③ 选型理由**：信息量与监督成本是一对交换——SDF 的丰富语义要靠贵的真值 + (6.1) 正则"养"出来；占据用最便宜的标签换掉距离语义；体素回归把两者都变贵、还背上立方内存墙（原文 §2.1，定性）。
机器人判据（第 01 章 ①）：碰撞 / 规划选 SDF（距离可查）；只要网格资产选占据（监督最省）；两者都不是体素——分辨率墙。

**④ 理论依据**：三篇论文的公开实验结论，本节只做定性转述（具体数字见论文与第 10 章）。

**⑤ 论证**：三轴排成矩阵即见分野。
沿演进链看：体素 → 连续隐函数（本章）解决分辨率墙；占据 → SDF 补上距离语义；下一步是把"重建"换成"从多视图照片直接优化"——但体渲染的监督既非距离也非标签，而是**照片**：需要把"渲染最优 ⟹ 表面对"设计进表示本身。
这正是第 07 章 NeuS 的出发点：它选择 SDF 作底层表示（零水平集语义清晰、(6.1) 保证 $\nabla s$ 是单位法向、且逐点可微可入体渲染），Eikonal 罚项 (7.9) 即本章 (6.1) 的落地。

## 本章要点

- **Eikonal 方程 (6.1)**：$\|\nabla s\| = 1$（可微点处 a.e.）——沿法向每走单位长度距离值恰变 1（⑤ 第一步）；链式求导（第二步）+ 切向分量消失（第三步）+ Lipschitz / 柯西–施瓦茨夹逼（第四步）。它是真 SDF 的必要非充分性质；第 07 章的 Eikonal 罚 (7.9) 以此为据。
- **DeepSDF（(6.2)–(6.5)）**：条件隐函数 + 自解码器（无编码器、隐码先验 $\mathcal{N}(\mathbf{0}, \sigma^2 I)$）；损失 (6.4) 的 clamp (6.3) 与 TSDF 截断 (1.14) 同款语义——远处值不携带表面细节、容量集中在 $\delta$ 带内；原文用 L1 范数，教程记号取平方（事实注记见 6.2 ⑤）。
- **形状补全 = 对隐码的 MAP 推断 (6.5)**：$\min_{\mathbf{z}} \|\mathbf{z}\|^2/(2\sigma^2) + \sum \mathcal{L}$——数据项重建观测、先验项补出背面；对任意形态观测成立（原文论证）。
- **Occupancy Networks（(6.6)–(6.10)）**：伯努利似然 → BCE (6.7)；BCE 对 logits 的梯度 $= o - b$，远离表面的点饱和后无训练信号；Lambert-W 修正 (6.8)–(6.9)（论文附录）使梯度处处有界非零（第四步自证），$c_1, c_2$ 编码类平衡。
- **提取**：占据等值面 $o = \tau$（(6.10)）经 MISE + Marching Cubes——第 05 章管线在占据场上复用（占据取 $f = 2o - 1$，回引 (1.11)）。
- **三方选型（6.4）**：SDF 信息最富（需距离真值 + Eikonal）、占据监督最省（无距离语义）、体素受内存墙；交换关系 = 信息量 ↔ 监督成本。
- **铺垫第 07 章**：体渲染的监督是照片——NeuS 选 SDF 为底层表示，(6.1) 保证 $\nabla s$ 是单位法向，是 (7.7) 斜率解读与 (7.9) 罚项的共同前提。

## 配套阅读

- DeepSDF 原论文：[../../papers/3d_reconstruction/classics/arXiv-1901.05103_DeepSDF.pdf](../../papers/3d_reconstruction/classics/arXiv-1901.05103_DeepSDF.pdf)（本章 (6.2)–(6.5) 对应其式 (2)(4)(9)(10) 与 §4.2 的概率表述）。
- Occupancy Networks 原论文：[../../papers/3d_reconstruction/classics/arXiv-1812.03828_OccupancyNetworks.pdf](../../papers/3d_reconstruction/classics/arXiv-1812.03828_OccupancyNetworks.pdf)（占据函数与损失对应其式 (1)–(5)；Lambert-W 修正见 CVPR 2019 补充材料）。
- 上一章：[./05_从体素到网格表面提取.md](./05_从体素到网格表面提取.md)（本章 6.2 / 6.3 的网格提取所复用的 MC 与等值面管线）。
- 下一章：[./07_神经隐式表面重建.md](./07_神经隐式表面重建.md)——回引本章 (6.1)：NeuS 把 SDF 接入体渲染，Eikonal 罚 (7.9) 即 (6.1) 的训练化。
- 定义回引：[第 01 章｜3D 重建问题与表示全景](./01_3D重建问题与表示全景.md)（SDF (1.13)、占据 (1.11)、查询三问 (1.3)）。

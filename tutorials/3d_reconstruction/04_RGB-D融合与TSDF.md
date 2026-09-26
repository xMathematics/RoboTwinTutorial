# 第 04 章｜RGB-D 融合与 TSDF

上一章从多视图 RGB 恢复稠密深度；本章换一类输入——RGB-D 相机直接给出深度，但每帧深度图带噪（误差随距离增长）、只覆盖相机当前视野、位姿本身还有误差。把逐帧点云各自独立存储，同一表面会被重复记录且互相冲突。截断符号距离场（Truncated Signed Distance Function, TSDF）把"多帧融合"变成体素网格上的逐点加权平均——常数内存、可流式、可 GPU 并行，自 KinectFusion 起成为实时稠密重建的标准件。本章三步推进：TSDF 的定义与截断宽度 $\mu$ 的两个依据（4.1 节）；加权融合的极大似然推导——本章压轴（4.2 节）；支撑融合的位姿跟踪（点到面 ICP）与查询（光线投射）（4.3 节）。

章节衔接：输入端是第 03 章产出的深度图（(3.12) 的反投影把深度变成点）；位姿来自本 4.3 节的 ICP 或 [SLAM 教程](../slam/README.md)的视觉里程计；产出的 TSDF 场交给第 05 章提取三角网格，选型取舍在第 10 章汇合。

**记号约定**：内参 $K$、位姿 $T_{cw}\in SE(3)$；SDF 记 $s(\mathbf{x})$（表面为 0、外部为正，第 01 章约定）；TSDF 记 $F(\mathbf{x})\in[-1,1]$、截断宽度 $\mu$；单帧观测权重 $w$、体素累计权重 $W(\mathbf{x})$；法向 $\mathbf{n}$；李代数扰动 $\delta\boldsymbol\xi=(\delta\boldsymbol\rho,\ \delta\boldsymbol\phi)$（平移分量在前）、hat 算子 $(\cdot)^\wedge$，均沿用 [SLAM 教程第 02 章](../slam/02_三维刚体运动旋转与位姿.md)。

## 4.1 TSDF 定义与截断宽度

**① 问题场景**：手头：RGB-D 相机沿轨迹采集的一串（彩色，深度）帧，每帧带位姿估计（来自 4.3 节的 ICP 或 SLAM 前端）。缺：一个**单一、全局、可查询**的表面模型——逐帧深度图只在各自视野内有效，重叠区域数值对不上（深度噪声 + 位姿误差）；逐帧独立存点云则同一表面被记录多遍、点数随帧数线性增长，且对"这里的表面到底在哪"没有单一答案。想要：把多帧带噪观测**在线**合并成一个一致的表示，随帧流式更新、不重放历史帧。

**② 解决方法**：把场景空间划分为固定尺寸体素网格，每个体素存两个数——截断符号距离值 $F$ 与权重 $W$。每来一帧：(i) 用该帧位姿把体素中心投到深度图像素；(ii) 沿该像素视线计算体素到表面的符号距离 $s$（(4.1) 的离散观测）；(iii) 在 $|s|\le\mu$ 带内按噪声水平加权并入体素（4.2 节的递归公式）。表面即 TSDF 的零水平集，查询、渲染、ICP 对应点都从它来（4.3 节）。整条管线 = ICP 跟踪 → TSDF 融合 → 光线投射，即 KinectFusion（ISMAR 2011）确立的实时稠密重建框架。

**③ 选型理由**：与两条替代路线对比。(a) 逐帧点云堆叠：不假设表面位置，但同一表面存多份拷贝，任何查询（法向、碰撞、抓取）都要先去重；内存随帧数增长，不可流式压缩。(b) 逐帧三角化再拼接：网格间撕裂，全局一致性无从谈起。TSDF 把"多帧融合"分解为体素级**加权平均**——内存与帧数无关、每帧每体素 $O(1)$ 更新、体素间天然并行，这是它统治实时重建的原因。代价：固定体积网格的内存墙——大场景/高分辨率下体素数爆炸，引出八叉树、哈希内存（Voxblox、BundleFusion 的 hashed allocation）等路线（第 09、10 章）；BundleFusion 在融合层进一步引入稀疏特征约束与全局位姿优化后的表面重融合（surface reintegration），修复累积位姿漂移。

**④ 理论依据**：截断符号距离场的体积融合出自 Curless & Levoy（*"A Volumetric Method for Building Complex Models from Range Images"*, SIGGRAPH 1996，TSDF 概念起点）；实时化与 GPU 实现见 KinectFusion（Newcombe et al., *KinectFusion: Real-Time Dense Surface Mapping and Tracking*, ISMAR 2011；延伸阅读与 BundleFusion 入口见 [论文库 README](../../papers/3d_reconstruction/README.md)）；SDF 符号约定沿用第 01 章。

**⑤ 完整推导**（每步注明依据）：

第一步（SDF 回顾与离散观测）。连续符号距离函数（第 01 章约定，式 (1.x)，编号待该章定稿对齐）：

$$s(\mathbf{x}) = \pm\operatorname{dist}(\mathbf{x},\,S),\qquad s=0\ \text{于表面}\ S,\quad s>0\ \text{在表面外（朝相机侧）},\ s<0\ \text{在表面内}. \tag{4.1}$$

体素网格上 $s(\mathbf{x})$ 的逐体素观测来自深度图：体素中心 $\mathbf{x}$ 经当前帧位姿投到像素 $\mathbf{u}=\pi(T_{cw}\mathbf{x})$（依据：投影公式 [SLAM 教程 (5.10)](../slam/05_视觉里程计-i特征点法.md)），取该像素深度 $z(\mathbf{u})$ 与体素中心沿视线的深度 $z_{\mathbf{x}}$，则

$$s(\mathbf{x}) \approx z(\mathbf{u}) - z_{\mathbf{x}}$$

（依据：视线方向在体素尺度内近似不变，符号距离沿视线度量；Curless–Levoy 原文沿视线做精确投影，离散实现普遍用此近似）。

第二步（为何截断——依据 (a)：可见性）。$s(\mathbf{x})$ 的定义要求"到最近表面的距离"，但传感器只看到**第一层**表面：视线穿过表面之后的真实几何完全不可知——表面背后的 $s$ 按定义要继续取负值，可这个数值没有任何观测支撑；离表面越远的体素，其"距离值"要么无观测、要么反映的是错误表面的距离。**可靠信息只存在于表面附近的窄带内**——这是截断的第一依据：只把 $|s|\le\mu$ 带内的观测写入体素，带外不写入，而不是硬填负值。

第三步（为何截断——依据 (b)：噪声模型）。结构光/双目类深度的视差误差随深度二次传播：由 $z = f b/\delta$（$b$ 基线、$\delta$ 视差，双目三角化的最小情形）对 $\delta$ 求导得 $\mathrm{d}z = -\frac{fb}{\delta^2}\mathrm{d}\delta = -\frac{z^2}{fb}\mathrm{d}\delta$，故

$$\sigma(z) \propto z^2,\qquad w \propto \frac{1}{\sigma^2} \propto \frac{1}{z^2}. \tag{4.2}$$

（依据：视差–深度关系的误差传播；权重取逆方差，其最优性由 4.2 节 (4.6) 的极大似然结论给出。）距离越远 $\sigma$ 越大、$w$ 越小——同一带宽内远距读数本身就更不可信。两条依据合起来给出截断操作：在带内保留线性、带外饱和：

$$F(\mathbf{x}) = \operatorname{clip}\Bigl(\frac{s(\mathbf{x})}{\mu},\,-1,\,1\Bigr) = \max\Bigl(-1,\ \min\bigl(1,\ s(\mathbf{x})/\mu\bigr)\Bigr). \tag{4.3}$$

带内 $F = s/\mu$ 保留符号与相对距离（零点即表面、$|F|$ 越小越贴近表面），带外饱和 $\pm1$——只剩"在内/在外"的粗信息，不参与融合（4.2 节第六步）；$\mu$ 取几个体素尺度的量级（经验值，与传感器噪声水平和体素分辨率匹配）。另注：任务的 $\min(1, s/\mu)$ 即 (4.3) 的正侧。

第四步（带内线性性与法向）。带内 $F=s/\mu$ 且 $s$ 是真距离，故 $\|\nabla F\| = 1/\mu$、方向指向表面（依据：(4.1) 的距离性质求梯度——SDF 梯度模为 1，第 01 章约定）。因此法向可由 $\mathbf{n}(\mathbf{x}) = \nabla F/\|\nabla F\|$ 从 TSDF 差分恢复（4.3 节 ICP 直接使用），零水平集可由插值精确定位（4.3 节光线投射）。这回答了"为什么不直接存 0/1 占据"——带内线性结构是后续一切查询的基础。

## 4.2 加权融合的统计推导（压轴）

**① 问题场景**：4.1 节把单帧观测写进了体素，但真正的融合问题是：同一表面（同一函数值 $s(\mathbf{x})$）在 $k$ 帧中被观测了 $k$ 次，每次噪声水平不同（远帧 $\sigma$ 大、掠射帧更糟）——怎样把 $k$ 个带噪读数合并成**一个**最优估计？简单平均把可信与不可信的观测等权混合，远帧的读数会把近帧拖偏；"后来帧覆盖先前帧"又浪费历史。此外融合必须**流式**：帧按时间到达，内存里不能存全部历史读数，每帧只允许一次体素级的就地更新。

**② 解决方法**：给观测建立独立高斯噪声模型，用极大似然估计（等价于最小方差加权融合）合并 $k$ 帧读数；再证明该批量解可以写成**递归形式**——新读数与历史估计按权重比例平均——即 Curless–Levoy / KinectFusion 的融合公式。带内有效观测（深度有效且 $|f_k|\le\mu$）才参与更新。

**③ 选型理由**：三个候选对比。(a) 简单平均：等价于假设所有帧同方差——与 (4.2) 的 $\sigma\propto z^2$ 矛盾，远帧把近帧拖偏。(b) 卡尔曼式标量更新：单变量情形与递归极大似然完全同构（本节递归式就是它），无额外信息。(c) 鲁棒统计（中位数、Huber）：能抗离群，但 TSDF 的截断（4.1 节）已把离群挡在带宽之外，且逆方差加权每体素 $O(1)$、可递归，鲁棒版需重排序/重加权、破坏流式更新。取标准形式：逆方差加权 + 截断。代价：高斯假设在掠射、多路径回波下偏乐观——靠带宽与权重（$w\propto1/z^2$）在工程上兜底。

**④ 理论依据**：独立高斯观测下的极大似然估计（估计论标准结果）；加权运行平均的体积融合出自 Curless & Levoy（SIGGRAPH 1996），实时化即 KinectFusion 的融合步骤（ISMAR 2011，见 [论文库 README](../../papers/3d_reconstruction/README.md)）。

**⑤ 完整推导**（每步注明依据）：

第一步（观测模型）。固定体素 $\mathbf{x}$，其真实 SDF 值为 $s(\mathbf{x})$，第 $k$ 帧读数建模为

$$f_k(\mathbf{x}) = s(\mathbf{x}) + \epsilon_k,\qquad \epsilon_k\sim\mathcal{N}(0,\ \sigma_k^2),\ \text{各帧独立}. \tag{4.4}$$

（依据：把深度噪声、位姿误差、投影近似总体折叠为加性高斯噪声——中心极限的工程惯例；$\sigma_k$ 按 (4.2) 随该帧距离 $z_k$ 增长；各帧独立，因位姿误差主体来自不同帧的估计过程。）

第二步（似然与负对数似然）。$K$ 帧联合似然 $L(s) = \prod_k \frac{1}{\sqrt{2\pi}\,\sigma_k}\exp\bigl(-\frac{(f_k-s)^2}{2\sigma_k^2}\bigr)$（依据：(4.4) 独立高斯密度连乘）。取负对数（依据：对数单调，最大化似然 = 最小化 NLL；与 $s$ 无关的常数项丢弃）：

$$J(s) = \sum_{k=1}^{K}\frac{\bigl(f_k - s\bigr)^2}{2\,\sigma_k^2}. \tag{4.5}$$

第三步（求极小）。$J$ 是 $s$ 的二次函数、凸（依据：$J''=\sum_k 1/\sigma_k^2>0$，凸函数驻点即全局极小）。令 $\frac{\mathrm{d}J}{\mathrm{d}s}=0$（依据：链式求导，$\frac{\mathrm{d}}{\mathrm{d}s}\frac{(f_k-s)^2}{2\sigma_k^2} = -\frac{f_k-s}{\sigma_k^2}$）：

$$\sum_k\frac{s-f_k}{\sigma_k^2} = 0\ \Longrightarrow\ s\sum_k\frac1{\sigma_k^2} = \sum_k\frac{f_k}{\sigma_k^2},$$

解出

$$\hat s = \frac{\displaystyle\sum_k f_k\big/\sigma_k^2}{\displaystyle\sum_k 1\big/\sigma_k^2}. \tag{4.6}$$

即**逆方差加权平均**：噪声越大的帧权重越小——(4.2) 的 $w\propto1/\sigma^2\propto1/z^2$ 由此获得最优性依据。

第四步（融合的收益）。写成加权和 $\hat s=\sum_k\lambda_k f_k$，$\lambda_k=\frac{1/\sigma_k^2}{\sum_j 1/\sigma_j^2}$（权重和为 1，依据：(4.6) 分子分母同除 $\sum_j 1/\sigma_j^2$）。由独立性：

$$\operatorname{Var}(\hat s) = \sum_k\lambda_k^2\,\sigma_k^2 = \frac{\sum_k \sigma_k^2/\sigma_k^4}{\bigl(\sum_j 1/\sigma_j^2\bigr)^2} = \frac{1}{\sum_k 1/\sigma_k^2}. \tag{4.7}$$

（依据：独立随机变量加权和的方差 = 方差的加权和；$1/\sigma_k^2$ 正是各观测的费雪信息贡献。）估计方差随观测数单调下降——这就是"多帧融合为什么有效"的定量表述。

第五步（递归化——流式更新）。定义单帧权重 $w_k = 1/\sigma_k^2$、累计权重 $W_k = \sum_{j\le k} w_j$、融合估计 $F_k = \hat s$（前 $k$ 帧的 (4.6)）。把 (4.6) 的求和拆成"前 $k-1$ 帧"+"第 $k$ 帧"（依据：加法结合律），并注意前 $k-1$ 帧的部分和恰为 $W_{k-1}F_{k-1}$（依据：$\sum_{j<k} w_j f_j = W_{k-1}F_{k-1}$，即 (4.6) 在 $k-1$ 帧上的形态，归纳假设）：

$$F_k(\mathbf{x}) = \frac{W_{k-1}\,F_{k-1}(\mathbf{x}) + w_k\,f_k(\mathbf{x})}{W_{k-1} + w_k},\qquad W_k = W_{k-1} + w_k. \tag{4.8}$$

（依据：分母 $\sum_{j\le k} w_j = W_{k-1}+w_k$。）式 (4.8) 正是 Curless–Levoy 的加权运行平均、KinectFusion 论文的融合更新式：每帧只需读旧值 $(F_{k-1},W_{k-1})$、写新值 $(F_k,W_k)$——常数内存、逐体素独立、GPU 一线程一体素并行，流式要求完全满足（初值 $W_0=0$，首帧直接写入）。

第六步（未知区域不参与更新）。两种情形对该体素跳过第 $k$ 帧更新（取 $w_k=0$）：其一，深度像素无效（超出量程、无回波）——$f_k$ 不存在；其二，$|f_k|>\mu$——读数落在截断带外，按 4.1 节依据 (a)，带外距离没有可靠观测支撑，且在 (4.3) 中它已饱和于 $\pm1$，写入只污染而不增信息（依据：(4.4) 的高斯模型在带外失效——可见性论证）。于是每体素的 $(F,W)$ 只由"真正看到过这个表面附近"的帧构成。

## 4.3 ICP 位姿跟踪与光线投射

**① 问题场景**：4.2 节把每帧读数 $f_k(\mathbf{x})$ 视为**带已知位姿**的观测——但位姿从哪来？RGB-D 相机的里程计要自己估计：手头是当前帧的深度（与灰度）图和**已融合的 TSDF 模型**，缺的是当前帧的 $T_{cw}$。同时，融合之外的一切用途（渲染、机器人碰撞与抓取查询、ICP 的对应点）都需要"从相机看 TSDF"的查询——把零水平集从体素网格里取出来。两个需求互相咬合：跟踪为融合提供位姿，融合好的模型反过来为跟踪提供参照（frame-to-model）。

**② 解决方法**：跟踪用**点到面 ICP**（point-to-plane Iterative Closest Point）：当前帧深度反投成点，用当前位姿估计投影进模型（projective data association）取对应点与法向，最小化"点到对应切平面"距离的平方和，对位姿的李代数扰动线性化、Gauss–Newton 迭代求解。查询用**光线投射**（raycasting）：从相机光心沿每像素光线在体素中步进，检测 TSDF 符号变化，在变号区段内插值精化表面交点。

**③ 选型理由**：点到面 vs 点到点 ICP：点到点的残差沿两点连线（与表面结构无关），要多次迭代才能"滑"向正确对齐；点到面把残差投影到局部切平面，一次线性化就吸收大部分相对运动——迭代次数显著更少，代价是需要法向（TSDF 梯度免费提供）。ICP（几何残差）vs 直接法（光度残差，[SLAM 教程第 06 章](../slam/06_视觉里程计-ii直接法.md)）：深度空间残差对曝光/光照不敏感（RGB-D 场景光照常不受控），且 RGB-D 天然有几何可用；代价是依赖深度质量与模型初始对齐。对应点用投影匹配而非全局最近邻搜索（kd-tree）：每点 $O(1)$、GPU 友好，代价是对初值要求更准。

**④ 理论依据**：点到面 ICP 残差与线性化（Chen & Medioni, *"Object Modelling by Registration of Multiple Range Images"*, 1992；点云配准标准结果）；投影对应（Blais & Levine, 1995）；SE(3) 扰动模型与雅可比（[SLAM 教程第 02 章](../slam/02_三维刚体运动旋转与位姿.md) §2.5，式 (2.24)–(2.26)）；TSDF 光线投射查询见 Curless & Levoy / KinectFusion。

**⑤ 完整推导**（每步注明依据）：

(a) **点到面 ICP 的线性化**。残差定义：当前帧点 $\mathbf{p}$（世界系，由深度反投影 (3.12) 得到），对应模型表面点 $\mathbf{q}$ 及其法向 $\mathbf{n}$，$T$ 为待估位姿：

$$e = \mathbf{n}^\top\bigl(T\,\mathbf{p} - \mathbf{q}\bigr). \tag{4.9}$$

（依据：点到平面距离——把两点之差投影到法向，只惩罚沿表面法向的错位。）对位姿加左扰动 $T\to\exp(\delta\boldsymbol\xi^\wedge)T$（依据：[SLAM 教程第 02 章] §2.5 的扰动模型；$\boldsymbol\xi$ 平移分量在前，与该教程 (2.25) 一致）。一阶展开：

$$\exp(\delta\boldsymbol\xi^\wedge)\,T\mathbf{p} \approx T\mathbf{p} + \delta\boldsymbol\rho + \delta\boldsymbol\phi\times\bigl(T\mathbf{p}\bigr). \tag{4.10}$$

（依据：SE(3) 左扰动的一阶结构——$\delta\boldsymbol\rho$ 平移点、$\delta\boldsymbol\phi$ 旋转点，即 (2.26)；旋转项由 (2.5) $\mathbf{a}^\wedge\mathbf{b}=\mathbf{a}\times\mathbf{b}$ 化为叉积。）代入 (4.9)（依据：$\mathbf{n}^\top$ 线性；对应点 $\mathbf{q},\mathbf{n}$ 由当前估计的投影确定，一阶近似下不随扰动变化）：

$$e(\delta\boldsymbol\xi) = e_0 + \mathbf{n}^\top\delta\boldsymbol\rho - \mathbf{n}^\top(T\mathbf{p})^\wedge\,\delta\boldsymbol\phi,\qquad e_0 = \mathbf{n}^\top\bigl(T\mathbf{p}-\mathbf{q}\bigr),$$

（依据：旋转项 $\delta\boldsymbol\phi\times(T\mathbf{p}) = -(T\mathbf{p})\times\delta\boldsymbol\phi = -(T\mathbf{p})^\wedge\delta\boldsymbol\phi$，依据：(2.5) 的反交换性。）得 6×1 雅可比（行向量）：

$$J = \Bigl[\ \mathbf{n}^\top\ \Big|\ -\mathbf{n}^\top(T\mathbf{p})^\wedge\ \Bigr] = \Bigl[\ \mathbf{n}^\top\ \Big|\ \bigl((T\mathbf{p})\times\mathbf{n}\bigr)^\top\ \Bigr]. \tag{4.11}$$

（依据：混合积轮换 $\mathbf{n}\cdot\bigl((T\mathbf{p})\times\delta\boldsymbol\phi\bigr) = \delta\boldsymbol\phi\cdot\bigl(\mathbf{n}\times T\mathbf{p}\bigr)$，故 $-\mathbf{n}^\top(T\mathbf{p})^\wedge\delta\boldsymbol\phi = \bigl((T\mathbf{p})\times\mathbf{n}\bigr)\cdot\delta\boldsymbol\phi$。）对全部对应点求和、Gauss–Newton 求解（依据：最小化 $\sum_i\|J_i\delta\boldsymbol\xi + e_{0,i}\|^2$，对 $\delta\boldsymbol\xi$ 求导置零；与 [SLAM 教程 (5.12)](../slam/05_视觉里程计-i特征点法.md) 同一形式）：

$$\Bigl(\sum_i J_i^\top J_i\Bigr)\,\delta\boldsymbol\xi^{\ast} = -\sum_i J_i^\top e_{0,i}. \tag{4.12}$$

6×6 正规方程，解出后左乘更新位姿并迭代（依据：扰动收敛后 $\exp(\delta\boldsymbol\xi^{\ast\wedge})T$ 为新位姿；迭代的原因是对应点随位姿变化——非线性源头）。

**对应点来源（操作步骤，不含理论）**：projective data association——用当前位姿估计把 $\mathbf{p}$ 投回模型参考帧得到像素 $\mathbf{u}$，在该像素处由光线投射（下文）取出模型表面点 $\mathbf{q}$ 与法向 $\mathbf{n}$（法向取 TSDF 梯度方向 $\nabla F/\|\nabla F\|$，中心差分；依据：4.1 节第四步的带内线性性）。

**初值问题**：(4.12) 的解只在初值邻域有效——对应点由投影产生，位姿偏差大时投影对应的根本不是同一表面点，迭代会收敛到错误对齐。这与直接法光度误差非凸、需较好初值（上一帧 + 图像金字塔）是同一机制（[SLAM 教程第 06 章](../slam/06_视觉里程计-ii直接法.md) §6.2 的对比表）：ICP 用上一帧位姿（或匀速模型外推）作初值；帧到模型（frame-to-model）对齐比帧到帧漂移小；跟踪失败的判定是残差/内点比例超阈——此时暂停融合，避免把错误位姿下的观测按 (4.8) 写进模型。

(b) **光线投射与零点精化**。像素 $\tilde{\mathbf{u}}$ 的相机系方向 $\mathbf{d}_{cam}=K^{-1}\tilde{\mathbf{u}}$，世界系光线 $\mathbf{r}(t)=\mathbf{o}+t\,\mathbf{d}$，$\mathbf{d}=R_{wc}\,\mathbf{d}_{cam}$，起点 $\mathbf{o}$ 为相机光心（依据：$T_{cw}$ 的逆给出相机在世界系的位姿；投影模型同 (3.12)）。步进采样 $t_{m+1}=t_m+\Delta t$，$\Delta t$ 取体素尺度以内（依据：三线性插值下 TSDF 沿光线连续、符号变化只发生在穿越零水平集处，步长超过体素对角线可能跨过整个变号区段而漏检）。检测首个变号对：$F(\mathbf{r}(t_{m-1}))>0$ 而 $F(\mathbf{r}(t_m))\le0$。

**三线性插值**：体素内任意点的 $F$ 由 8 个角点值 $F_{ijk}$（$i,j,k\in\{0,1\}$）重构。记点在该体素内的归一化偏移 $\alpha_x,\alpha_y,\alpha_z\in[0,1]$，一维权重 $w_0=1-\alpha$、$w_1=\alpha$：

$$F(\mathbf{x}) = \sum_{i=0}^{1}\sum_{j=0}^{1}\sum_{k=0}^{1} w_i\,w_j\,w_k\,F_{ijk}. \tag{4.13}$$

推导：先沿 $x$ 方向对 4 条棱做一维线性插值 $F_{jk}(x) = (1-\alpha_x)F_{0jk}+\alpha_x F_{1jk}$（依据：一维线性插值定义——两点间线性过渡，一阶泰勒的精确形式）；再对插出的 4 个值沿 $y$ 插值得 2 个、最后沿 $z$ 插值得 1 个（依据：三维张量积网格上多变量插值的标准构造——逐维嵌套）；三层嵌套展开即乘积权和形式 (4.13)（依据：代数展开）。两个自检性质：权重和为 1（$\bigl(\sum_i w_i\bigr)\bigl(\sum_j w_j\bigr)\bigl(\sum_k w_k\bigr)=1$，依据：每维权重和为 1）——$F$ 是角点的凸组合，不会越出角点取值范围；体素边界处 $\alpha\to 0/1$ 时两侧取到相同角点值——跨体素连续（依据：边界上插值退化为复制）。

**零点精化**：变号区段内把 $F$ 沿光线近似为线性（依据：(4.13) 的分段线性结构），解 $F(t_{m-1}) + \frac{F_m-F_{m-1}}{t_m-t_{m-1}}\,(t-t_{m-1}) = 0$（依据：一元一次方程求解）得

$$t^{\ast} = t_{m-1} + \frac{F\bigl(\mathbf{r}(t_{m-1})\bigr)}{F\bigl(\mathbf{r}(t_{m-1})\bigr) - F\bigl(\mathbf{r}(t_m)\bigr)}\,\bigl(t_m-t_{m-1}\bigr), \tag{4.14}$$

表面交点为 $\mathbf{r}(t^{\ast})$，法向取 $\nabla F/\|\nabla F\|$（依据：SDF 梯度 = 单位法向，第 01 章约定；离散用中心差分）。这就是 ICP 对应点、渲染与机器人碰撞查询共用的同一入口。

## 本章要点

- **TSDF**：体素存截断符号距离与权重，$F=\operatorname{clip}(s/\mu,-1,1)$（(4.3)），零水平集即表面；带内线性（$\|\nabla F\|=1/\mu$）支撑法向与零点查询。
- **截断的两个依据**：可见性——表面背后无观测，$s$ 只在带内可信（(4.1)）；噪声模型 $\sigma(z)\propto z^2$、$w\propto1/z^2$（(4.2)）——远距读数不可信。
- **压轴公式**：独立高斯观测的极大似然 = 逆方差加权 $\hat s=\frac{\sum_k f_k/\sigma_k^2}{\sum_k 1/\sigma_k^2}$（(4.6)），估计方差随观测单调下降（(4.7)）；递归形式 (4.8) 即 KinectFusion 融合式——常数内存、流式、GPU 并行。
- **未知区域**：深度无效或 $|f_k|>\mu$ 时取 $w_k=0$ 跳过更新——带外观测是污染而非信息。
- **点到面 ICP**：残差 (4.9) 经左扰动线性化得雅可比 $J=[\mathbf{n}^\top,\ ((T\mathbf{p})\times\mathbf{n})^\top]$（(4.10)–(4.11)），6×6 正规方程 (4.12)；对应点用投影匹配；初值来自上一帧/匀速模型（初值敏感性与直接法同源，[SLAM 教程第 06 章] §6.2）。
- **光线投射**：体素尺度步进 + 变号检测；三线性插值 (4.13)（凸组合、跨体素连续）+ 线性零点 (4.14) 精化——ICP 对应点与下游查询共用入口。
- **内存墙**：固定体积网格是 TSDF 的代价，八叉树/哈希内存（Voxblox、BundleFusion）与全局重融合是后续修复路线（③、第 10 章）。

## 配套阅读

- KinectFusion / BundleFusion 延伸阅读入口：[../../papers/3d_reconstruction/README.md](../../papers/3d_reconstruction/README.md)（Curless & Levoy SIGGRAPH 1996 为 TSDF 概念起点，KinectFusion ISMAR 2011 为实时化，BundleFusion 加全局一致融合）。
- 上一章：[./03_多视图立体重建MVS.md](./03_多视图立体重建MVS.md)（深度图从哪来；本章 $f_k$ 的观测对象即其输出）。
- 下一章：[./05_从体素到网格表面提取.md](./05_从体素到网格表面提取.md)（把 TSDF 零水平集变成三角网格）。
- 扰动模型与雅可比：[SLAM 教程第 02 章](../slam/02_三维刚体运动旋转与位姿.md)（§2.5，(4.10)–(4.11) 的出处）。
- 直接法与初值问题：[SLAM 教程第 06 章](../slam/06_视觉里程计-ii直接法.md)（§6.2，ICP 初值问题的同源机制）。

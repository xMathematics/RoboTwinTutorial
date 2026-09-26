# 第 03 章｜多视图立体重建 MVS

第 02 章（SfM）回答了"相机在哪、稀疏点在哪"，但稀疏点云只落在角点等可匹配的位置——点与点之间的表面在哪里，是本章要回答的问题。多视图立体重建（Multi-View Stereo, MVS）在位姿已知的条件下为**每个像素**恢复深度：传统路线（COLMAP）沿候选深度假设展开匹配——平面扫描（plane sweeping），用光度一致性打分、用几何一致性选视图；学习路线（MVSNet）把同一套几何写成可微的代价体（cost volume），交给 3D 卷积正则、用软 argmin 回归深度。两条路线共享同一个几何内核——平面扫描的深度—单应变换，这是本章的推导主轴。深度图的多帧融合留给第 04 章，本章末尾只交代融合得以成立的几何前提（法向一致性）。

**记号约定**：相机内参 $K$；世界→相机位姿 $T_{cw}\in SE(3)$；参考相机系三维点 $\mathbf{X}$、源相机系坐标 $\mathbf{X}'$；像素齐次坐标 $\tilde{\mathbf{u}}$；深度假设 $d$；平面单位法向 $\mathbf{n}$；单应矩阵 $H$；图像宽高 $W\times H$、深度假设数 $D$、特征通道数 $F$。

## 3.1 平面扫描：从位姿到稠密深度的几何框架

**① 问题场景**：手头：$N$ 张照片 + 第 02 章 SfM 恢复的相机内外参与稀疏点云。缺：稀疏点之间的表面——三角化只在角点上成功，白墙、地板上没有特征可匹配，点云之间存在大片"表面在哪"的空白；且逐像素的立体匹配必须**跨多视图一致**——同一物理表面点在不同视图里观测到的像素不同，深度假设不能在视图之间互相矛盾。想要：参考视图逐像素的深度图，每个像素的深度在全部源视图上站得住。

**② 解决方法**：平面扫描（Collins, 1996）：在参考相机前方取一族候选深度平面（前向平行 $z=d$，$d$ 在 $[d_{min}, d_{max}]$ 内离散采样）。对每个假设 $d$、每个源视图 $i$：由本章 (3.2) 的单应变换把参考像素映到源视图采样，在参考像素的窗口上与各源视图采样值算光度一致性得分（3.2 节 NCC）并跨视图聚合；沿 $d$ 取最优假设——即该像素深度。整个过程对全部像素、全部 $d$ 并行，GPU 友好。

**③ 选型理由**：与两条替代路线对比。(a) 两两立体匹配再拼接：每对视图产出一张深度图，同一像素在不同配对下深度互不相同，融合时要再做一遍仲裁——一致性是事后补的；平面扫描把全部视图按**同一假设**同时打分，跨视图一致性内建于打分本身。(b) 体积占据法（space carving 类）：需要可靠的轮廓/占据证据，分辨率受体素限制，光度信息未直接使用；平面扫描直接在"深度假设"这一更紧的参数化上工作。每个 (视图, 假设) 只需一个 3×3 单应矩阵 + 双线性采样，计算可完全并行；代价是前向平行假设在倾斜表面上引入模型误差——处理方式见本节末（逐像素深度假设，COLMAP 进一步联合估计法向）。

**④ 理论依据**：空间扫描（space sweep）多图像匹配（Collins, *"A Space-Sweep Approach to True Multi-Image Matching"*, CVPR 1996，平面扫描的起点）；平面诱导单应（平面单应变换，Hartley & Zisserman 教材标准结果）；像素级视图选择与深度图融合（Schönberger & Fischer, ECCV 2016，本地 [PDF](../../papers/3d_reconstruction/classics/COLMAP-MVS_ECCV2016_SchoenbergerFischer.pdf)）。

**⑤ 完整推导**（每步注明依据）：

约定设定：参考相机系中记三维点 $\mathbf{X}$，同一物理点在源相机系中的坐标为 $\mathbf{X}'$，二者由相对位姿联系：

$$\mathbf{X}' = R_{rel}\,\mathbf{X} + \mathbf{t}_{rel},$$

（依据：刚体变换定义；$T_{rel}$ 把参考系坐标映到源系坐标，由两相机的 $T_{cw}$ 相乘消去世界系得到，第 02 章。）候选平面以参考相机系表示，取平面单位法向 $\mathbf{n}$ **指向参考相机**（表面外侧，与第 01 章 SDF"外正内负"约定一致），平面方程用点法式

$$\mathbf{n}^\top\mathbf{X} + d = 0,$$

其中 $d>0$：前向平行平面取 $\mathbf{n} = -\mathbf{e}_3$，方程 $-z+d=0$ 即 $z=d$——$d$ 恰是平面沿光轴的深度。

第一步（平面约束改写为比例式）：$\mathbf{n}^\top\mathbf{X}+d=0$ 两边除以 $d$（依据：$d>0$ 非零）并移项：

$$\frac{\mathbf{n}^\top\mathbf{X}}{d} = -1.$$

第二步（造出平移项）：两边乘以平移向量 $\mathbf{t}_{rel}$（依据：标量 $\frac{\mathbf{n}^\top\mathbf{X}}{d}$ 可自由进出数乘）：

$$\mathbf{t}_{rel}\,\frac{\mathbf{n}^\top\mathbf{X}}{d} = -\,\mathbf{t}_{rel}.$$

第三步（代入消元）：把上式代入相对位姿关系（即 $\mathbf{t}_{rel} = -\mathbf{t}_{rel}\frac{\mathbf{n}^\top\mathbf{X}}{d}$）：

$$\mathbf{X}' = R_{rel}\,\mathbf{X} + \mathbf{t}_{rel} = R_{rel}\,\mathbf{X} - \mathbf{t}_{rel}\,\frac{\mathbf{n}^\top\mathbf{X}}{d} = \Bigl(R_{rel} - \frac{\mathbf{t}_{rel}\,\mathbf{n}^\top}{d}\Bigr)\mathbf{X},$$

（依据：第三步代入；矩阵乘法结合律把 $\mathbf{t}_{rel}\,\mathbf{n}^\top\mathbf{X}$ 收拢为 $(\mathbf{t}_{rel}\mathbf{n}^\top)\,\mathbf{X}$，其中 $\mathbf{t}_{rel}\mathbf{n}^\top\in\mathbb{R}^{3\times3}$ 是"平移 × 法向"的秩一外积矩阵。）记

$$\mathbf{X}' = \Bigl(R_{rel} - \frac{\mathbf{t}_{rel}\,\mathbf{n}^\top}{d}\Bigr)\mathbf{X}. \tag{3.1}$$

第四步（投影到像素）：针孔模型下 $\tilde{\mathbf{u}}\sim K\mathbf{X}$、$\tilde{\mathbf{u}}'\sim K\mathbf{X}'$（依据：投影公式 $\mathbf{u}=(f_x\frac{X}{Z}+c_x,\ f_y\frac{Y}{Z}+c_y)$，见 [SLAM 教程 (5.10)](../slam/05_视觉里程计-i特征点法.md)；"$\sim$"表示齐次坐标下成比例）。由 $\mathbf{X}=K^{-1}\tilde{\mathbf{u}}$（齐次意义下成立，依据：$f_x,f_y>0$ 保证 $K$ 可逆）代入 (3.1)：

$$\tilde{\mathbf{u}}' \sim K\Bigl(R_{rel} - \frac{\mathbf{t}_{rel}\,\mathbf{n}^\top}{d}\Bigr)K^{-1}\,\tilde{\mathbf{u}}.$$

故同一候选平面上的像素映射是**单应变换**（homography，3×3 矩阵、作用在齐次像素坐标上）：

$$H(d) = K\Bigl(R_{rel} - \frac{\mathbf{t}_{rel}\,\mathbf{n}^\top}{d}\Bigr)K^{-1},\qquad \tilde{\mathbf{u}}' \sim H(d)\,\tilde{\mathbf{u}}. \tag{3.2}$$

第五步（特例校验）：前向平行 $\mathbf{n}=-\mathbf{e}_3$，平面 $z=d$ 上任一点 $\mathbf{X}=(X,Y,d)^\top$ 满足 $\frac{\mathbf{t}_{rel}\mathbf{n}^\top}{d}\mathbf{X} = \frac{\mathbf{t}_{rel}\cdot(-z)}{d} = -\mathbf{t}_{rel}$（依据：$\mathbf{n}^\top\mathbf{X}=-z$，$z=d$），代回 (3.1) 得 $\mathbf{X}'=R_{rel}\mathbf{X}+\mathbf{t}_{rel}$——与原始刚体变换一致，推导自洽。

**符号约定注记**：若把平面写成 $\mathbf{n}^\top\mathbf{X}=d$（法向指向场景内部），同样的消元给出 $H(d)=K\bigl(R_{rel}+\frac{\mathbf{t}_{rel}\mathbf{n}^\top}{d}\bigr)K^{-1}$——同一平面、两种等价写法，差异只在法向取向。MVSNet 论文 Eq.1（$\mathbf{H}_i(d)=\mathbf{K}_iR_i\bigl(I-\frac{(\mathbf{t}_1-\mathbf{t}_i)\mathbf{n}_1^\top}{d}\bigr)R_1^\top K_1^\top$）与 COLMAP 一系文献均取本章的减号形式（平面用点法式 $\mathbf{n}^\top\mathbf{X}+d=0$）。

**变深度时的处理（前向平行假设失效）**：(3.2) 把"源视图中的补丁形变"建模为"补丁整体贴在深度为 $d$ 的前向平行平面上"。表面倾斜时，同一窗口内各像素的真实深度不同，与单一平面假设矛盾——NCC 峰变钝、深度估计向"抹平倾斜"的方向系统偏差。处理是把假设从"平面 $(\mathbf{n},d)$"退化到"逐像素深度 $d(\mathbf{u})$"：每个像素拥有自己的假设集 $\{d\}$、独立选优（法向仍固定为主光轴，(3.2) 逐像素逐假设可用）；COLMAP 的 patch 匹配进一步把 $(d,\mathbf{n})$ 都作为逐像素假设联合估计（法向初值来自邻域深度梯度），恢复倾斜表面的正确形变。MVSNet（3.3 节）取第一条路（逐像素深度假设），靠学习正则弥补模型误差。

## 3.2 光度一致性与视图选择

**① 问题场景**：平面扫描给了"把源视图按假设 $d$ 搬到参考视图"的机制（(3.2)），但"搬过来像不像"需要一个判据。麻烦有三处：其一，同一表面在不同照片里亮度不同——曝光、增益、白平衡各异，逐像素比灰度会把光照差异误判为深度不对；其二，假设 $d$ 错误时采样到的可能是被遮挡的背景或毫不相干的表面，判据必须能把它们与真实对应区分开；其三，源视图质量参差——视角太偏、被遮挡、基线过大的视图不该参与打分。手头：参考像素窗口 + $N$ 个源视图按 (3.2) 的采样值；缺：一个对光照差异鲁棒、对错误对应可判的相似性度量，以及"该信哪些视图"的规则。

**② 解决方法**：度量取归一化互相关（Normalized Cross-Correlation, NCC）：参考窗口与按 $d$ 形变后的源视图窗口各自减均值、除以标准差后做内积；代价取 $C(d)=1-\mathrm{NCC}(d)$，跨视图聚合后沿 $d$ 选优。视图选择（COLMAP）用几何与光度两级判据筛掉不可信的源视图（本节末操作注记）。

**③ 选型理由**：更朴素的逐像素差——SAD（绝对差和）与 SSD（平方差和）——都对仿射光照变化 $I'=aI+b$ 敏感：$b$ 直接进入差值、$a$ 直接缩放差值，曝光一变，沿 $d$ 的代价排序即乱。NCC 先减均值（消 $b$）、再除以标准差（消 $a$），对正增益完全不变（⑤ 证明）；代价是窗口统计比逐像素差贵，且窗口大小是权衡——窗口大则统计稳，但在深度不连续处把前后景"平均"出虚假匹配。相比 3.3 的学习式度量，NCC 无需训练、行为可解释，但在弱纹理与反射面处同样无判别力——这正是引入学习正则的缺口。

**④ 理论依据**：NCC 是标准模板匹配度量（Lewis, *"Fast Normalized Cross-Correlation"*, 1995）；光照的仿射模型来自朗伯表面线性曝光响应的窗口内近似；COLMAP-MVS 以 NCC 为光度代价并配合几何一致性做像素级视图选择（Schönberger & Fischer, ECCV 2016，本地 [PDF](../../papers/3d_reconstruction/classics/COLMAP-MVS_ECCV2016_SchoenbergerFischer.pdf)）。

**⑤ 完整推导**（每步注明依据）：

光照建模：两视图对同一表面的观测近似满足仿射关系

$$I'(\mathbf{p}) = a\,I(\mathbf{p}) + b,\qquad a>0, \tag{3.3}$$

（依据：朗伯表面辐亮度经曝光/增益的线性响应；$a$ 吸收对比度、$b$ 吸收亮度；窗口内视为常数。）

NCC 定义（窗口 $\Omega$，$|\Omega|$ 为像素数）：

$$\mathrm{NCC}(I,I') = \frac{\sum_{\mathbf{p}\in\Omega}\bigl(I(\mathbf{p})-\bar I\bigr)\bigl(I'(\mathbf{p})-\bar I'\bigr)}{\sqrt{\sum_{\mathbf{p}\in\Omega}\bigl(I(\mathbf{p})-\bar I\bigr)^2}\ \sqrt{\sum_{\mathbf{p}\in\Omega}\bigl(I'(\mathbf{p})-\bar I'\bigr)^2}},\qquad \bar I = \frac{1}{|\Omega|}\sum_{\mathbf{p}\in\Omega}I(\mathbf{p}). \tag{3.4}$$

命题：对任意 $a>0$、$b$，$\mathrm{NCC}(I,\ aI'+b) = \mathrm{NCC}(I,I')$。

第一步（均值穿越仿射变换）：$\overline{aI'+b} = \frac1{|\Omega|}\sum_{\mathbf{p}}(aI'(\mathbf{p})+b) = a\bar I' + b$（依据：求和的线性；常数项 $\sum_{\mathbf{p}}b = |\Omega|b$）。

第二步（中心化消去 $b$）：$\bigl(aI'(\mathbf{p})+b\bigr)-\bigl(a\bar I'+b\bigr) = a\bigl(I'(\mathbf{p})-\bar I'\bigr)$（依据：第一步，$b$ 在相减中消去；提取公因子 $a$）。

第三步（代入分子）：分子 $= \sum_{\mathbf{p}}(I-\bar I)\cdot a\bigl(I'-\bar I'\bigr) = a\sum_{\mathbf{p}}(I-\bar I)(I'-\bar I')$（依据：第二步与求和线性）。

第四步（代入分母）：第二个标准差因子 $= \sqrt{\sum_{\mathbf{p}}a^2\bigl(I'-\bar I'\bigr)^2} = |a|\sqrt{\sum_{\mathbf{p}}\bigl(I'-\bar I'\bigr)^2} = a\sqrt{\cdots}$（依据：第二步；$\sqrt{a^2}=|a|$，$a>0$ 时 $|a|=a$）。

第五步（约分）：

$$\mathrm{NCC}(I,\ aI'+b) = \frac{a\sum_{\mathbf{p}}(I-\bar I)(I'-\bar I')}{\sqrt{\sum_{\mathbf{p}}(I-\bar I)^2}\;a\sqrt{\sum_{\mathbf{p}}(I'-\bar I')^2}} = \mathrm{NCC}(I,I'). \tag{3.5}$$

即：减均值消去 $b$（亮度偏移）、除以标准差消去 $a$（对比度），NCC 只依赖两个窗口的"形状"，对逐视图光照差异完全不变（负增益 $a<0$ 时反号，物理上不出现）。于是沿深度假设比较 $\mathrm{NCC}(d)$ 时，各源视图曝光差异不改变排序：真实深度处所有视图形状对齐、NCC 趋近 1；错误深度处采样错位、形状不符、NCC 显著偏小。这就是"光度一致性"的可计算定义。

**视图选择（操作步骤，不含理论）**：COLMAP 的像素级视图选择分两级——几何一致性：源视图与参考视图的基线夹角在阈值区间内（太小三角化病态、太大遮挡与透视形变严重）、深度为正、重投影误差低于阈值；光度一致性：按候选深度处的 NCC 得分对源视图排序，只保留 top-k 参与代价聚合，深度与法向由保留视图的得分加权融合。**遮挡与外点的代价机制**：观点差异大的源视图即使名义上对着同一表面，实际看到的往往是不同表面（自遮挡、掠射边缘），其 NCC"谁都不像"，在选择步骤被直接跳过——宁可少用视图，也不让被污染的观测进入融合。

## 3.3 MVSNet 与代价体（压轴）

**① 问题场景**：3.1–3.2 的手工管线有两个结构性弱点。其一，NCC 是逐像素、逐假设独立的局部判据：弱纹理（白墙、桌面）处窗口没有可辨的"形状"，NCC 对所有 $d$ 几乎同样平——局部判据无信息可用；反射、透明等非朗伯表面直接破坏 (3.3) 的光照模型。其二，深度平滑、遮挡连续这些全局结构只能靠手工正则（SGM 惩罚、图割能量）硬编码，换场景就要重新调参。手头：$N$ 张带位姿图像，(3.2) 的几何机器全部可用；缺：一个能从数据里学"表面长什么样"的先验、且端到端可训练的深度推断器；想要：弱纹理/反射面也稳、多视图等权参与、还能报告置信度的稠密深度。

**② 解决方法**：MVSNet（Yao et al., ECCV 2018）把平面扫描搬进网络，四步：(1) **特征提取**——各视图经共享权重的 2D CNN 抽取 $F$ 通道特征图（论文：32 通道、下采样 4 倍，像素邻域信息编码进通道描述子，稠密匹配不丢上下文）；(2) **可微单应 warp**——每个深度假设 $d$ 下，把各源视图特征图按 (3.2) warp 到参考相机的前向平行平面，$N$ 个视图堆成 $N$ 个特征体（feature volume）；(3) **代价体聚合与正则**——$N$ 个特征体沿视图方向用方差度量归并成单通道代价体 $C$（⑤(a)），再经多尺度 3D CNN（类 3D U-Net 编码–解码）正则成概率体，沿深度轴 softmax 得 $P(d)$；(4) **软 argmin 回归**——沿深度轴取期望得连续深度图（⑤(b)）；$P(d)$ 同时给出逐像素置信度（离群像素的分布弥散，据此过滤）。

**③ 选型理由（本章重点）**：手工与学习的分歧点在"正则从哪来"。替代 (a)——手工光度 + 手工正则（COLMAP 路线）：光度失效处（弱纹理、反射）NCC 本身无判别力，手工平滑假设（分段常值深度、各向同性）与真实几何无关，只能"猜平"——先验来自设计者而非数据；优点是无需训练、跨域稳。替代 (b)——在规则欧氏体素网格上做 3D 卷积（SurfaceNet 一类）：整场景立方体的开销随尺寸三次方增长，难以放大到开放场景。选择（MVSNet）：代价体建在**参考相机视锥**上（只在可能出现的深度区间采样），3D CNN 沿深度轴聚合上下文——"平滑先验"变成从数据学到的正则，弱纹理处网络靠上下文与数据先验补足 NCC 的信息真空，反射面靠训练分布中的相似外观修复。代价：显存 $O(HWD)$（正则前还有 $N$ 个体积为 $V=\frac{W}{4}\cdot\frac{H}{4}\cdot D\cdot F$ 的特征体），3D 卷积本身昂贵——$D$ 因此受限，需压缩通道（论文 32→8）、控制假设数或由粗到细分层。精度优先、有训练数据的场景值得；跨域开放场景仍是手工路线更稳。

**④ 理论依据**：MVSNet（Yao et al., *"MVSNet: Depth Inference for Unstructured Multi-View Stereo"*, ECCV 2018；本地 [PDF](../../papers/3d_reconstruction/classics/arXiv-1804.02505_MVSNet.pdf)，其 Eq.1 / Eq.2 / Eq.4 即本章 (3.2) / (3.6) / (3.9)–(3.10)）；软 argmin 源自双目深度学习 GCNet（Kendall et al., *"End-to-End Learning of Geometry and Context for Deep Stereo Regression"*, ICCV 2017）；几何骨架是平面扫描（Collins, 1996，3.1 节）。

**⑤ 完整推导**（每步注明依据）：

(a) **方差代价——为何"方差"而非"均值"**。把同一像素在 $N$ 个特征体中的特征记为标量分量 $x_1,\dots,x_N$（多通道时逐分量同理），分解为"共性 + 不一致"：

$$x_i = s + b_i,$$

$s$ 是所有视图共享的成分（真实信号与共同的外观/亮度偏移），$b_i$ 是视图间不一致的成分（各自的偏移与噪声）。两个候选聚合算子（方差分母取 $N-1$ 即无偏样本方差）：

$$\bar x = \frac{1}{N}\sum_i x_i,\qquad \mathrm{Var} = \frac{\sum_i\bigl(x_i-\bar x\bigr)^2}{N-1}. \tag{3.6}$$

**均值无判别力**：$\bar x = s+\bar b$（依据：求和线性）。两个截然不同的情形给出**同一个**均值："各视图完全一致"（$b_i$ 全同）与"视图彼此分歧但均值恰好相同"（$b_i$ 散布而 $\bar b$ 相同）——把均值当聚合特征，一致性信息已经丢失（MVSNet 论文原文：mean "provides no information about the feature differences"）。

**方差只留不一致**：先证恒等式（对任意 $x_i$）：

$$\sum_i\bigl(x_i-\bar x\bigr)^2 = \sum_i x_i^2 - \frac1N\Bigl(\sum_i x_i\Bigr)^2. \tag{3.7}$$

推导：$\sum_i(x_i-\bar x)^2 = \sum_i x_i^2 - 2\bar x\sum_i x_i + N\bar x^2$（依据：平方展开与求和线性）；代入 $\bar x=\frac1N\sum_i x_i$，中间项 $=-\frac2N(\sum_i x_i)^2$、末项 $= \frac1N(\sum_i x_i)^2$（依据：$N\bar x^2 = N\cdot\frac{1}{N^2}(\sum x_i)^2$），合并即 (3.7)。把 $x_i=s+b_i$ 代入（或直接用 $x_i-\bar x=(s+b_i)-(s+\bar b)=b_i-\bar b$）：

$$\mathrm{Var} = \frac1{N-1}\sum_i\bigl(b_i-\bar b\bigr)^2,$$

共性 $s$ **精确消去**（依据：$x_i-\bar x = b_i-\bar b$）——方差只度量视图间的不一致。它与"成对不一致"的关系：

$$\sum_{i<j}\bigl(x_i-x_j\bigr)^2 = N\sum_i\bigl(x_i-\bar x\bigr)^2. \tag{3.8}$$

推导：左端 $= \frac12\sum_{i,j}(x_i-x_j)^2$（依据：对称性——每对 $(i,j)$ 与 $(j,i)$ 贡献相同，$i=j$ 项为零）$= \frac12\sum_{i,j}\bigl[(x_i-\bar x)-(x_j-\bar x)\bigr]^2$（依据：同减 $\bar x$ 差不变）$= \frac12\Bigl[N\sum_i(x_i-\bar x)^2 + N\sum_j(x_j-\bar x)^2 - 2\bigl(\sum_i(x_i-\bar x)\bigr)\bigl(\sum_j(x_j-\bar x)\bigr)\Bigr] = N\sum_i(x_i-\bar x)^2$（依据：偏差和为零 $\sum_i(x_i-\bar x)=0$）。

即方差正比于**平均成对不一致**：候选深度正确 → 各视图打分一致 → 方差小；深度错 → 视图互相矛盾 → 方差大；且对共同偏移 $\bar b$ 完全不敏感——(3.5) 的多视图版。这同时实现论文的设计哲学"所有视图对代价等权贡献、不偏袒参考视图"（方差对视图对称）。注：论文 Eq.2 分母为 $N$（总体方差），(3.6) 取 $N-1$（无偏样本方差），两者差常数因子，只等价于缩放 softmax 温度，不影响深度假设间的排序与峰位。

(b) **软 argmin——把不可微的选择变成可微的期望**。正则后的代价体沿深度轴给出 $C(d)$。最朴素的读法是 $\arg\min_d C(d)$——但它不可微：argmin 是"选下标"运算，输出是离散假设而非 $C$ 的连续函数；$C$ 的微小变化在非临界处不改变选择，$\partial d^{\ast}/\partial C \equiv 0$ 几乎处处成立（依据：分段常值函数的导数性质），梯度链在 $C$ 之后整段断裂，特征提取与单应 warp 无法训练——"端到端"不可能。改造分两步。第一步，softmax 把代价变成分布（负号使"代价小 → 概率大"；依据：softmax 把任意实值集合归一化为正、和为一的分布，等价于温度为 1 的 Gibbs 分布）：

$$p(d) = \frac{\exp\bigl(-C(d)\bigr)}{\sum_{d'}\exp\bigl(-C(d')\bigr)}. \tag{3.9}$$

第二步，取期望代替取下标（依据：离散分布期望的定义）：

$$\hat d = \sum_d d\cdot p(d). \tag{3.10}$$

**可微性**：由 softmax 的梯度（依据：商法则；$\delta_{dd'}$ 为 Kronecker delta）$\frac{\partial p(d)}{\partial C(d')} = -p(d)\bigl(\delta_{dd'}-p(d')\bigr)$，得

$$\frac{\partial\hat d}{\partial C(d')} = \sum_d d\,\frac{\partial p(d)}{\partial C(d')} = -d'\,p(d') + p(d')\sum_d d\,p(d) = p(d')\bigl(\hat d - d'\bigr). \tag{3.11}$$

梯度显式、处处非零——损失对深度的导数经 (3.10)→(3.9)→$C$→单应 warp（(3.2) 配可微双线性采样）→2D 特征 CNN 一路回传。**极限行为**：在 (3.9) 中把代价放大 $\beta$ 倍，$\beta\to\infty$ 时分布集中到 $\arg\min$（依据：指数集中性——最小代价项指数级压倒其余，Laplace 近似），$\hat d\to d^{\ast}$；有限 $\beta$ 时 $\hat d$ 是软化的峰值位置，且因深度假设均匀采样，期望可落在相邻假设**之间**——给出亚假设精度的连续深度（论文 Fig.3：内点像素的概率分布单峰、离群像素弥散——置信度过滤的依据）。

(c) **深度图的去向：融合与法向一致性（简短）**。逐像素深度 $z(\mathbf{u})$ 反投回三维（依据：(3.2) 所用针孔模型的逆，即 [SLAM 教程 (5.10)](../slam/05_视觉里程计-i特征点法.md) 的反演）：

$$\mathbf{X}_{\mathrm{cam}}(\mathbf{u}) = z(\mathbf{u})\,K^{-1}\tilde{\mathbf{u}},\qquad \mathbf{X}_{\mathrm{world}} = T_{wc}\,\mathbf{X}_{\mathrm{cam}}. \tag{3.12}$$

多视图深度图对同一表面的观测是有噪的同一样本，进入第 04 章的 TSDF 加权融合；融合前按**法向一致性**筛选：像素法向由邻域深度梯度估计（方向 $\propto(-\partial_x z, -\partial_y z, 1)$），两个深度观测只有在法向一致（切平面相合）时才描述同一表面元素——法向冲突的观测（深度不连续边缘、互相遮挡的表面）被剔除，否则平均会混叠不同表面、在边缘处产生拖影（依据：加权平均只在"各观测估计同一个量"时无偏，见 (4.6) 的前提；COLMAP 的几何一致性过滤同此逻辑）。

## 本章要点

- **平面扫描几何**：候选平面上的刚体变换与平面约束消元给出单应 $H(d)=K\bigl(R_{rel}-\frac{\mathbf{t}_{rel}\mathbf{n}^\top}{d}\bigr)K^{-1}$（(3.1)–(3.2)），"试一个深度"变成一次 3×3 矩阵乘 + 双线性采样；前向平行假设失效时退化为逐像素深度假设，COLMAP 再联合估计法向。
- **NCC 不变性**：光照仿射模型 $I'=aI+b$（(3.3)）下，减均值消 $b$、除标准差消 $a$，$\mathrm{NCC}(I,aI'+b)=\mathrm{NCC}(I,I')$（(3.4)–(3.5)）——跨视图曝光差异不影响深度排序。
- **视图选择**（操作）：几何（基线夹角/正深度/重投影）+ 光度（NCC 排序 top-k）两级过滤；观点差异大的源视图被跳过，避免遮挡污染。
- **方差代价**：均值把"完全一致"与"均值恰同的分歧"混为一谈；方差经 (3.7)–(3.8) 恰为平均成对不一致、共性 $s$ 精确消去（(3.6)），且对视图对称、等权。
- **软 argmin**：argmin 不可微 → softmax 分布 (3.9) + 期望 (3.10)，梯度 $p(d')(\hat d-d')$（(3.11)）支持端到端回传；$\beta\to\infty$ 退化为 argmin，均匀假设下给出连续深度。
- **正则与显存的交换**：3D CNN 的学习正则补足弱纹理/反射面的光度信息真空，代价是 $O(HWD)$ 代价体显存与 3D 卷积开销（③）。
- **深度图融合的前提**：反投影 (3.12) + 法向一致性筛选，衔接第 04 章 TSDF 融合。

## 配套阅读

- MVSNet 原论文：[../../papers/3d_reconstruction/classics/arXiv-1804.02505_MVSNet.pdf](../../papers/3d_reconstruction/classics/arXiv-1804.02505_MVSNet.pdf)（Eq.1/2/4 与本章 (3.2)/(3.6)/(3.9)–(3.10) 对应）。
- COLMAP-MVS（像素级视图选择）：[../../papers/3d_reconstruction/classics/COLMAP-MVS_ECCV2016_SchoenbergerFischer.pdf](../../papers/3d_reconstruction/classics/COLMAP-MVS_ECCV2016_SchoenbergerFischer.pdf)。
- 上一章：[./02_多视图几何与SfM.md](./02_多视图几何与SfM.md)（位姿与稀疏点从哪来；撰写中）。
- 下一章：[./04_RGB-D融合与TSDF.md](./04_RGB-D融合与TSDF.md)（深度图加权融合与 TSDF，(3.12) 的去向）。
- 投影与三角化工具：[SLAM 教程第 05 章](../slam/05_视觉里程计-i特征点法.md)（针孔投影 (5.10)、三角化 §05.3、PnP §05.4）。

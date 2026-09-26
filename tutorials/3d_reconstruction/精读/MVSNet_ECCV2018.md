# 论文精读｜MVSNet（ECCV 2018）

> **PDF**：[papers/3d_reconstruction/classics/arXiv-1804.02505_MVSNet.pdf](../../../papers/3d_reconstruction/classics/arXiv-1804.02505_MVSNet.pdf) ｜ **教程**：[第 03 章｜多视图立体重建 MVS](../03_多视图立体重建MVS.md) ｜ **代码**：无（教学实现规划中）

## 1. 论文信息与一句话贡献

- **题目**：MVSNet: Depth Inference for Unstructured Multi-View Stereo
- **作者**：Yao Yao、Zixin Luo、Shiwei Li、Long Quan（香港科技大学），Tian Fang（Altizure）
- **发表**：ECCV 2018；arXiv:1804.02505。本仓库同时收录 [LaTeX 源码](../../../papers/3d_reconstruction/latex/arXiv-1804.02505_MVSNet/)，本文式号已与 PDF 排版逐条核对——**全文编号公式共 4 个**：(1) 单应矩阵、(2) 方差代价度量、(3) 深度期望回归、(4) 训练损失。
- **一句话贡献**：首个**端到端**的多视图立体重建（Multi-View Stereo, MVS）深度推断网络——在**参考相机视锥**（camera frustum）而非规则体素网格上，用**可微单应 warp**（differentiable homography warping）把 2D 特征图搬进 3D 代价体（cost volume），以**方差度量**聚合任意 $N$ 个视图、多尺度 3D CNN 正则、**软 argmin**（soft argmin）回归连续深度图——把传统 MVS 的"平面扫描 + 光度一致性 + 手工正则"整条管线变成可学习的整体，在 DTU 上以显著优势刷新完整性，并在 Tanks and Temples 上**免微调**取得当时榜首。

## 2. 问题与动机

**要解决什么**：非受控（unstructured）多视图照片输入下的稠密深度推断——每张参考图出一张深度图，多图融合成点云。手头只有图像与 SfM 恢复的相机参数。

**手工路线的结构性缺口**（§1）：传统 MVS（NCC 光度度量 + SGM/图割类正则）在理想朗伯场景下精度很高，但**低纹理、高光、反射区域**使稠密匹配无信息可用——基准测试的普遍现象是 accuracy 已经很好、**completeness 仍有大缺口**。学习式双目匹配（MCNet/GCNet 一系）虽已超越手工方法，但依赖"图像对预先矫正、逐像素视差"的特殊设定；直接推广到相机几何任意的多视图情形并不平凡。

**已有学习式 MVS 为何不够**（§1、§2）：SurfaceNet 预先构建彩色体素立方体（CVC）、LSM 用可微投影在规则体素网格上分类——共同的病根是**体积表示**：内存随场景尺寸三次方增长，前者靠分而治之、后者只能做低分辨率合成物体，都无法放大到真实大场景。

**MVSNet 的两个关键决定**（§1）：其一，代价体建在**参考相机视锥**上而不是规则欧氏体素上——只在"深度假设"这一更紧的参数化上花内存；其二，把 MVS 解耦为**逐参考图的深度图估计**——一次只解一个小问题，大场景靠逐视点处理天然可扩展。这两条正是内存墙下的破局思路。

## 3. 方法总览

```
N 张图像（共享权重）──▶【2D 特征提取】8 层 CNN，1/4 分辨率、32 通道特征图 {F_i}
                              │
                              ▼
【可微单应 warp】对每个深度假设 d，按论文式(1)的 H_i(d) 把各源特征图
                warp 到参考相机前向平行平面 → N 个特征体 {V_i}（各 W/4 × H/4 × D × 32）
                              │
                              ▼
【方差代价度量】论文式(2)：逐元素方差把 N 个特征体归并为单代价体 C（32 通道）
                              │
                              ▼
【3D 正则化】四尺度 3D UNet（编码器 stride-2 逐级把分辨率减半，32→8 通道压缩，
             末层 1 通道）→ 沿深度轴 softmax → 概率体 P
                              │
                              ▼
【软 argmin】论文式(3)：D = Σ_d d·P(d) → 初始深度图（1/4 分辨率）
             【概率图】最近 4 个深度假设的概率和 → 置信度
                              │
                              ▼
【深度精修】初始深度 + 参考原图 → 残差网络 → 精修深度图（训练损失见式(4)）
                              │
                              ▼
【后处理】光度滤波（概率 < 0.8 剔除）+ 几何滤波（重投影 <1px 且深度相对误差 <1%，
          至少 3 视一致）→ 可见性融合 → 稠密点云
```

与教程第 03 章的对应：第 1、2 步是 (3.2) 单应几何的可微化，第 3 步是 (3.6) 方差代价，第 4、5 步是 (3.9)–(3.11) 软 argmin，后处理是 3.2 节视图选择/一致性过滤思想在概率体上的翻版。

## 4. 关键公式推导（按论文原文式号）

**符号表**（与教程第 03 章记号互相对照）：

| 论文记号 | 含义 | 教程对应 |
|---|---|---|
| $\mathbf{I}_1,\ \{\mathbf{I}_i\}_{i=2}^N$ | 参考图像 / 源图像 | 3.1 节"参考/源视图" |
| $\mathbf{K}_i,\mathbf{R}_i,\mathbf{t}_i$ | 第 $i$ 个特征图对应的内参、旋转、平移 | $K,\ T_{cw}$ |
| $\mathbf{F}_i \in \mathbb{R}^{\frac{W}{4}\times\frac{H}{4}\times 32}$ | 2D 特征图（原图 1/4 分辨率） | 3.3 ②(1) |
| $W,H,D,F$ | 图像宽、高、深度假设数、特征通道数 | 同名 |
| $\mathbf{V}_i(d)$ | 深度 $d$ 平面上第 $i$ 视图的特征体切片 | 3.3 ②(2) |
| $\mathbf{n}_1$ | 参考相机主轴（**指向场景内部**，前向） | 教程 $\mathbf{n}$ 取反（指向相机） |
| $d\in[d_{min},d_{max}]$ | 深度假设（均匀采样） | 同 |
| $\mathbf{C},\mathbf{P}$ | 代价体 / 概率体 | 3.3 ②(3) |
| $V=\frac{W}{4}\cdot\frac{H}{4}\cdot D\cdot F$ | 单个特征体的元素个数 | 3.3 ③ |

### 4.1 相机锥体上的可微单应 warp（论文式 (1)）

论文式 (1)（PDF 第 5 页排版核对）：

$$\mathbf{H}_i(d) = \mathbf{K}_i \cdot \mathbf{R}_i \cdot \Big(\mathbf{I} - \frac{(\mathbf{t}_1 - \mathbf{t}_i)\cdot \mathbf{n}_1^T}{d}\Big)\cdot \mathbf{R}_1^T \cdot \mathbf{K}_1^T. \tag{1}$$

**几何推导**（无跳步；论文把世界系约定在参考相机系下最易读懂，以下按此推导再换算到教程形式）：

约定：世界系 = 参考相机系，则参考相机位姿为恒等（$\mathbf{R}_1=\mathbf{I}$、相机中心在原点，$\mathbf{t}_1=0$），$\mathbf{n}_1=\mathbf{e}_3$（前向主轴）。第 $i$ 个相机的投影为 $\mathbf{x}_i \sim \mathbf{K}_i(\mathbf{R}_i\mathbf{X}_w + \mathbf{t}_i)$（依据：针孔投影 $\mathbf{u}=(f_x\frac{X}{Z}+c_x, f_y\frac{Y}{Z}+c_y)$ 的齐次形式，见 [SLAM 教程 (5.10)](../../slam/05_视觉里程计-i特征点法.md)；"$\sim$"为齐次成比例）。

第一步（平面约束改写为比例式）：前向平行平面是"参考系深度为 $d$"的点集，$\mathbf{n}_1^T\mathbf{X}_w = d$（依据：$\mathbf{n}_1=\mathbf{e}_3$ 时即 $z=d$，$d$ 恰为沿光轴深度）。两边除以 $d$（依据：$d>d_{min}>0$）得 $\frac{\mathbf{n}_1^T\mathbf{X}_w}{d}=1$。

第二步（造出平移项）：把标量 1 乘进平移向量（依据：数乘恒等 $\mathbf{t}_i = \mathbf{t}_i\cdot 1$）：

$$\mathbf{t}_i = \mathbf{t}_i\,\frac{\mathbf{n}_1^T\mathbf{X}_w}{d} = \frac{\mathbf{t}_i\,\mathbf{n}_1^T}{d}\,\mathbf{X}_w$$

（依据：标量 $\frac{\mathbf{n}_1^T\mathbf{X}_w}{d}$ 移入矩阵积时收拢为秩一外积 $\mathbf{t}_i\mathbf{n}_1^T\in\mathbb{R}^{3\times3}$ 作用在 $\mathbf{X}_w$ 上。）

第三步（代入消元）：$\mathbf{X}_i = \mathbf{R}_i\mathbf{X}_w + \mathbf{t}_i = \mathbf{R}_i\mathbf{X}_w + \frac{\mathbf{t}_i\mathbf{n}_1^T}{d}\mathbf{X}_w$。为凑成论文的 $\mathbf{R}_i(\,\cdot\,)$ 形式，把平移写成相机坐标轴下的基线：$\mathbf{t}_i = \mathbf{R}_i\,\tilde{\mathbf{t}}_i$（定义 $\tilde{\mathbf{t}}_i=\mathbf{R}_i^{-1}\mathbf{t}_i$，依据：旋转可逆），则

$$\mathbf{X}_i = \mathbf{R}_i\Big(\mathbf{I} + \frac{\tilde{\mathbf{t}}_i\,\mathbf{n}_1^T}{d}\Big)\mathbf{X}_w.$$

第四步（投影到像素）：$\mathbf{x}_i \sim \mathbf{K}_i\mathbf{X}_i$、$\mathbf{x}_1\sim\mathbf{K}_1\mathbf{X}_w$，故 $\mathbf{X}_w\sim\mathbf{K}_1^{-1}\mathbf{x}_1$（依据：$\mathbf{K}$ 可逆且"$\sim$"吸收尺度），代入得

$$\mathbf{x}_i \sim \underbrace{\mathbf{K}_i\,\mathbf{R}_i\Big(\mathbf{I} - \frac{(\mathbf{t}_1-\mathbf{t}_i)\mathbf{n}_1^T}{d}\Big)\mathbf{K}_1^{-1}}_{\mathbf{H}_i(d)}\ \mathbf{x}_1,$$

其中 $\mathbf{t}_1-\mathbf{t}_i = 0-\tilde{\mathbf{t}}_i$ 的符号来自"基线 = 参考 → 源"的方向定义——与论文式 (1) 的减号一致。

**两个校验**：(a) $i=1$ 时基线为零，式 (1) 给 $\mathbf{H}_1=\mathbf{I}$——与论文原文"the homography for reference feature map $\mathbf{F}_1$ itself is an $3\times3$ identity matrix"一致；(b) 特例 $d\to\infty$（平面退到无穷远）时修正项 $\to 0$，$\mathbf{H}_i\to\mathbf{K}_i\mathbf{R}_i\mathbf{K}_1^{-1}$——纯旋转的单应，自洽。

**与教程 (3.1)–(3.2) 的换算**：一般世界系下记 $\mathbf{R}_{rel}=\mathbf{R}_i\mathbf{R}_1^T$、$\mathbf{t}_{rel}=\mathbf{t}_i-\mathbf{R}_i\mathbf{R}_1^T\mathbf{t}_1$（依据：$\mathbf{X}_i=\mathbf{R}_i\mathbf{R}_1^T(\mathbf{X}_1-\mathbf{t}_1)+\mathbf{t}_i$ 展开合并，即教程第 03 章 (3.1) 的相对位姿消元），教程取平面法向**指向参考相机**（$\mathbf{n}=-\mathbf{n}_1$，点法式 $\mathbf{n}^T\mathbf{X}+d=0$），得

$$\mathbf{H}(d) = \mathbf{K}\Big(\mathbf{R}_{rel}-\frac{\mathbf{t}_{rel}\,\mathbf{n}^T}{d}\Big)\mathbf{K}^{-1}.\qquad\text{（教程 (3.2)）}$$

代入 $\mathbf{n}=-\mathbf{n}_1$ 恰为"+$\mathbf{t}_{rel}\mathbf{n}_1^T/d$"——**论文式 (1) 与教程 (3.2) 是同一几何的两种记法**，差异只在法向取向与位姿的书写约定；官方开源实现按相对位姿形式编码。注记：式 (1) 按字面把 $\mathbf{R}_1^T\mathbf{K}_1^T$ 当作 $\mathbf{K}_1^{-1}$ 用，严格成立需在"参考相机系=世界系 + 归一化坐标"下读它——读论文时按本节的换算理解即可，几何不受影响。

**可微性**：warp 的采样用**双线性插值**（依据：论文 §3.2——"the differentiable bilinear interpolation is used to sample pixels from feature maps"），对像素坐标可导，梯度可从 3D 体一路回传到 2D 特征 CNN——这是"端到端"的枢纽，也是与经典平面扫描（Collins 1996）唯一的操作差异。

### 4.2 特征体的构建与方差代价度量（论文式 (2)）

**特征体**：对每个深度假设 $d$，把 $N$ 张特征图按 (1) warp 到参考视锥平面，堆成 $N$ 个体积为 $V=\frac{W}{4}\cdot\frac{H}{4}\cdot D\cdot F$ 的特征体 $\{\mathbf{V}_i\}_{i=1}^N$（依据：论文 §3.2，$F=32$）。这一步把"图像 + 相机参数"编码成统一的 3D 张量——后续一切操作都是纯卷积。

**方差度量**，论文式 (2)（PDF 第 6 页排版核对；逐元素运算）：

$$\mathbf{C} = \mathcal{M}(\mathbf{V}_1,\cdots,\mathbf{V}_N) = \frac{\sum_{i=1}^N(\mathbf{V}_i - \overline{\mathbf{V}_i})^2}{N} \tag{2}$$

其中 $\overline{\mathbf{V}_i}=\frac1N\sum_i\mathbf{V}_i$。注意分母是 $N$（总体方差）——**教程 (3.6) 取 $N-1$（无偏样本方差），两者差常数因子**，只等价于缩放 softmax 温度，不影响深度假设间的排序与峰位；本精读与教程各自按原文书写。

**为什么方差优于均值**（论文原文论点 + 教程 (3.6)–(3.8) 的证明）：设计哲学是"所有视图对代价**等权**贡献、不偏袒参考视图"（依据：论文 §3.2 引 Hartmann et al.——传统方法以参考图为锚逐对算代价再启发式聚合，方差对视图置换对称，天然等权）。把同一像素位置的特征分量分解为 $x_i = s + b_i$（$s$ 共性、$b_i$ 视图间不一致）：

- **均值无判别力**：$\bar x = s+\bar b$（依据：求和线性）。"各视图完全一致"与"彼此分歧但均值恰好相同"给出同一个 $\bar x$——一致性信息在取均值时丢失。论文原话：mean "provides no information about the feature differences"，因此 Hartmann 等需要前后再接 CNN 层去补推相似性。
- **方差只留不一致**：由恒等式 $\sum_i(x_i-\bar x)^2 = \sum_i x_i^2 - \frac1N(\sum_i x_i)^2$（教程 (3.7)）与 $x_i-\bar x = b_i-\bar b$，共性 $s$ 精确消去；且成对不一致满足 $\sum_{i<j}(x_i-x_j)^2 = N\sum_i(x_i-\bar x)^2$（教程 (3.8)），故式 (2)（分母 $N$）正比于**平均成对不一致**：深度假设正确 → 各视图特征对齐 → $\mathbf{C}$ 小；深度错误 → 视图互相矛盾 → $\mathbf{C}$ 大。
- **消融证实**（§5.3）：把式 (2) 换成均值度量重训，收敛更慢、验证损失更高。

注意式 (2) 逐元素作用于 32 通道特征，所以代价体 $\mathbf{C}$ 仍是 32 通道——这正是下节"第一个 3D 卷积后 32→8"的压缩对象。

### 4.3 多尺度 3D CNN 正则化（成本体分辨率逐级减半）

原始代价体被噪声污染（非朗伯面、遮挡），需要平滑约束——即"正则"。MVSNet 用**四尺度 3D UNet**（编码–解码，encoder-decoder）：编码器以 stride-2 的 3D 卷积逐级把代价体分辨率**减半**（大感受野、低内存地聚合上下文），解码器逐级恢复分辨率并以跨尺度连接融合（依据：论文 §3.2"similar to a 3D version UNet"及 Fig.1 图例的 stride-2 卷积）。为压内存：第一个 3D 卷积后把 32 通道压到 8；每个尺度内卷积从 3 层减到 2 层；末层输出 **1 通道**体。最后**沿深度方向做 softmax**归一化成概率体 $\mathbf{P}$。

与传统正则（SGM 惩罚、图割）的本质区别：平滑先验不是手工设计的能量，而是从训练数据里学出来的 3D 上下文——弱纹理处靠邻域上下文、反射面靠训练分布里的相似外观"补足"光度信息的真空（对照教程 3.3 ③ 的选型论证）。概率体同时服务两件事：逐像素深度（4.4）与置信度（4.5）。

### 4.4 软 argmin：可微的深度回归（论文式 (3)）

最朴素的读法是逐像素 winner-take-all（$\arg\max_d \mathbf{P}(d)$）——但它**不可微**（选下标运算，$\partial d^\ast/\partial\mathbf{P}\equiv 0$ 几乎处处，依据：分段常值函数的导数性质），无法反传训练，也给不出亚像素精度。论文改为沿深度方向取**期望**，式 (3)（PDF 第 7 页排版核对）：

$$\mathbf{D} = \sum_{d=d_{min}}^{d_{max}} d \times \mathbf{P}(d). \tag{3}$$

**与教程 (3.9)–(3.10) 的关系**：教程把 softmax 显式写进软 argmin，$p(d)=\frac{\exp(-C(d))}{\sum_{d'}\exp(-C(d'))}$（(3.9)）、$\hat d=\sum_d d\,p(d)$（(3.10)）；论文则把 softmax 放在 3D 正则的末尾（$\mathbf{P}$ 已归一化），式 (3) 中不再出现指数。记正则网络输出的未归一化分数为 $z(d)$，$\mathbf{P}(d)=\frac{\exp(z(d))}{\sum_{d'}\exp(z(d'))}$，令 $C=-z$ 即与教程 (3.9) 逐字相同——**同一件事的两种记法**。

**可微性**（教程 (3.11) 的结论在论文记号下）：由商法则 $\frac{\partial \mathbf{P}(d)}{\partial z(d')} = \mathbf{P}(d)(\delta_{dd'}-\mathbf{P}(d'))$（依据：softmax 梯度；$\delta$ 为 Kronecker delta），得

$$\frac{\partial \mathbf{D}}{\partial z(d')} = \mathbf{P}(d')\big(\mathbf{D}-d'\big),$$

梯度显式、处处非零——损失对深度的导数经 (3) → softmax → 3D CNN → 式 (2) → 双线性 warp（式 (1)）→ 2D CNN 全程回传。

**连续深度**：深度假设在 $[d_{min},d_{max}]$ 均匀采样，期望可落在相邻假设**之间**——亚假设精度的连续深度（依据：论文 §3.3"the expectation value here is able to produce a continuous depth estimation"）。极限行为：把分数放大 $\beta$ 倍、$\beta\to\infty$ 时分布集中到 $\arg\max$（依据：指数集中性），式 (3) 退化为 winner-take-all。论文 Fig.2(c)：内点像素的概率分布**单峰**、离群像素**弥散**——这是下一节置信度过滤的依据。

### 4.5 概率图、滤波与损失（论文式 (4)）

**概率图（quality）**：定义深度估计质量为"真值落在估计附近小区间"的概率，以**最近 4 个深度假设的概率和**度量（依据：论文 §3.3；标准差/熵等统计量实测无额外收益，且概率和的阈值更好调）。

**两步过滤**（§4.1，判据数字按论文）：
- **光度一致性**：概率 $<0.8$ 的像素判为离群（即保留概率 $\ge 0.8$ 者）；
- **几何一致性**：参考像素 $p_1$ 按深度 $d_1$ 投到源视图得 $p_i$，再按源视图深度 $d_i$ 投回，若 $|p_{reproj}-p_1|<1$ 像素且 $|d_{reproj}-d_1|/d_1<0.01$，则 $d_1$ 通过两视一致检验；**全部深度要求至少 3 视一致**。

**深度精修**：正则的大感受野使深度边界过平滑；把初始深度（预缩放到 $[0,1]$）与参考原图拼成 4 通道输入，过三层 32 通道 2D 卷积 + 一层 1 通道卷积学**深度残差**（末层无 BN/ReLU 以学得负残差），加回初始深度。

**训练损失**，式 (4)（只在有真值标签的像素上）：

$$Loss = \sum_{p\in\mathbf{p}_{valid}} \big\|d(p)-\hat d_i(p)\big\|_1 + \lambda\cdot\big\|d(p)-\hat d_r(p)\big\|_1,\qquad \lambda = 1.0 \tag{4}$$

（$\hat d_i$ 初始估计、$\hat d_r$ 精修估计；真值深度图由 DTU 点云经**Screened Poisson** 重建成网格再渲染得到——见[Poisson 精读](./PoissonRecon_SGP2006.md)与其 2013 屏蔽版。）

### 4.6 显存分析：$O(W\!\cdot\!H\!\cdot\!D)$ 的代价体

单特征体元素数 $V=\frac{W}{4}\cdot\frac{H}{4}\cdot D\cdot F$（依据：式 (2) 上方定义），显存粗算（单位：float32）：

$$\underbrace{N\cdot\tfrac{W}{4}\cdot\tfrac{H}{4}\cdot D\cdot F}_{\text{特征体}} + \underbrace{\tfrac{W}{4}\cdot\tfrac{H}{4}\cdot D\cdot F}_{\text{代价体}} + \underbrace{\text{3D UNet 中间体}}_{\text{各尺度 8/16 通道}}\ \propto\ W\,H\,D.$$

线性于像素数与深度假设数、且带 $F$ 与 $N$ 的乘子——**深度维 $D$ 与 1/4 分辨率是仅有的两根杠杆**。论文的应对：图像须被 32 整除（4 倍下采样 × 4 尺度编解码）；训练裁剪 $640\times512$、$D=256$（11 GB 的 GTX 1080ti 可训）；DTU 评测 $1600\times1184$、Tanks and Temples $1920\times1056$ 均 $D=256$，需 16 GB 的 Tesla P100。$D$ 受限时只能收窄 $[d_{min},d_{max}]$（DTU 训练取 425–935 mm、2 mm 间隔）或降低深度分辨率——这就是"显存墙"的直接形态，也是后续级联方案的出发点（§6）。

## 5. 实验与结果解读

- **DTU（室内，受控）**：训练集 79 scan（$N=3$、$640\times512$、$D=256$，深度 425–935 mm、10 万次迭代）；评测 22 scan（$N=5$、$1600\times1184$、$D=256$）。距离度量（mm，越低越好）：MVSNet **overall 0.462 全场最佳**、completeness 0.527 最佳，accuracy 0.396 逊于 Gipuma 的 0.283——学习正则换来的是"补全"而非"更准"。百分比度量 f-score（<1mm / <2mm）：**75.69 / 80.25 均为最佳**。定性结论（论文 Fig.4）：弱纹理与反射区域点云最完整——正是手工光度度量无判别力的区域。
- **Tanks and Temples（室外，泛化）**：**DTU 模型零微调**直接测试（深度范围与源视图由 OpenMVG 稀疏点确定），在 intermediate 集**2018 年 4 月 18 日前提交中排名第一**（mean f-score 43.48，COLMAP 42.14、Pix4D 43.24）——方差度量对视图数/几何的适应性 + 学习先验的跨域稳健性是主因。
- **运行时间**：单 scan 约 230 s（每视图 4.7 s），比 Gipuma 快约 5 倍、比 COLMAP 快约百倍、比 SurfaceNet 快约百六十倍（依据：论文 §5.4，同机实测）。
- **消融**（验证损失）：视图数 $N=2/3/5$——用 $N=3$ 训练的模型在 $N=5$ 下反而更好（方差度量天然适配任意视图数，均值度量无此性质）；8 层 2D 特征显著优于单层 $7\times7$ 卷积（上下文编码的价值）；方差度量收敛更快、损失更低；深度精修对验证损失影响小，但把 DTU f-score 从 75.58→75.69（<1mm）、79.98→80.25（<2mm）。

## 6. 局限与后续影响

**显存墙**是结构性局限：代价体 $O(W\!\cdot\!H\!\cdot\!D)$ 线性于分辨率与假设数、乘上通道与视图数，$D$ 因此被锁死在 256 量级——大深度范围只能牺牲深度分辨率，重建边界也因正则的大感受野而过平滑（论文用精修网络缓解但不消除）。此外图像尺寸须被 32 整除、深度范围依赖 SfM 稀疏点先验、逐参考图处理使超大场景的耗时随视图数线性增长。

**后续一脉由此展开**：R-MVSNet（Yao et al., CVPR 2019）把深度维的 3D 卷积换成沿 $d$ 的循环展开，内存从 3D 体降到 2D 切片，用时间换空间、首次做到大规模场景的全球尺度重建；CasMVSNet（Gu et al., CVPR 2020）再进一步——由粗到细的**级联代价体**，先在粗分辨率以小 $D$ 定大范围，再逐级在窄区间内加密假设，把"窄深度区间"从手工先验变成算法结构。此后的 Fast-MVSNet、PatchmatchNet、Vis-MVSNet 直至今天的 NeRF/3DGS 时代，代价体 + 深度回归仍是学习式 MVS 的骨架范式。

## 7. 与本项目对照

**与教程第 03 章的公式映射**（已逐条核实双方编号）：

| 论文 | 内容 | 教程第 03 章 |
|---|---|---|
| 式 (1) | 可微单应 $\mathbf{H}_i(d)$ | (3.1)–(3.2)（相对位姿消元；法向取向相反，见 4.1 的换算） |
| 式 (2) | 方差代价（分母 $N$） | (3.6)（分母 $N-1$，差常数因子）＋ (3.7)/(3.8) 的判别力证明 |
| 式 (3) | 深度期望回归 | (3.9)–(3.10)（教程把 softmax 显式写进软 argmin）＋ (3.11) 梯度 |
| §3.2 末 | 沿深度 softmax → $\mathbf{P}$ | (3.9) 的归一化步骤 |
| §4.1 | 概率 <0.8 + 3 视几何一致过滤 | 3.2 节视图选择/一致性过滤（手工判据的概率体翻版） |
| §4.2 | 深度图 → 点云融合 | (3.12) 反投影 + 法向一致性 → 第 04 章 TSDF |

注意：教程 3.3 ④ 与"配套阅读"把软 argmin 标注为"论文 Eq.4"——**实际软 argmin 是式 (3)，式 (4) 是训练损失**（本表按 PDF 原文更正；教程文件未改动）。

**与 COLMAP-MVS 的手工路线对比**（COLMAP 双篇精读暂未收录，此处文字提及）：COLMAP-MVS（Schönberger & Fischer, ECCV 2016，本地 `papers/3d_reconstruction/classics/`）用 NCC 光度代价 + 像素级视图选择 + SGM 类手工正则，无需训练、跨域稳、精度高；MVSNet 把"特征、度量、正则"三件事全部交给数据——弱纹理/反射处补足了 NCC 的信息真空（completeness 大幅领先），代价是显存墙与训练分布依赖。教程 3.3 ③ 的选型论证（精度优先且有训练数据 → 学习路线；开放跨域 → 手工路线）正是这两个系统的对照实验总结。

**项目内的衔接**：深度图经 (3.12) 反投影与一致性过滤后，可交第 04 章 TSDF 融合，或直接以带法向点云进入[Poisson 表面重建精读](./PoissonRecon_SGP2006.md)成网格——MVSNet 论文自己制作 DTU 真值时用的正是 Screened Poisson（Kazhdan 2013），构成一条漂亮的闭环；第 10 章给机器人场景的完整选型矩阵。

## 配套阅读

- 原论文：[arXiv-1804.02505_MVSNet.pdf](../../../papers/3d_reconstruction/classics/arXiv-1804.02505_MVSNet.pdf)；LaTeX 源码（式号交叉核对）：[papers/3d_reconstruction/latex/arXiv-1804.02505_MVSNet/](../../../papers/3d_reconstruction/latex/arXiv-1804.02505_MVSNet/)。
- 教程主章节：[第 03 章｜多视图立体重建 MVS](../03_多视图立体重建MVS.md)（(3.1)–(3.12) 全部推导）；位姿来源：[第 02 章｜多视图几何与 SfM](../02_多视图几何与SfM.md)；深度图去向：[第 04 章｜RGB-D 融合与 TSDF](../04_RGB-D融合与TSDF.md)。
- 姊妹精读：[Poisson 表面重建（SGP 2006）](./PoissonRecon_SGP2006.md)（点云 → 水密网格；亦即 MVSNet 制作训练真值的方法）。
- 针孔投影工具：[SLAM 教程第 05 章](../../slam/05_视觉里程计-i特征点法.md)（投影 (5.10)）。
- 后续脉络（论文库未收录 PDF，文字提及）：R-MVSNet（CVPR 2019）、CasMVSNet（CVPR 2020）、SurfaceNet（ICCV 2017）、GCNet（ICCV 2017，软 argmin 出处）。

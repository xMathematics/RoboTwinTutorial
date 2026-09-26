# 论文精读｜MapAnything：Universal Feed-Forward Metric 3D Reconstruction（arXiv 2025 / 3DV 2026）

> **LaTeX 源码**：[papers/3d_reconstruction/latex/arXiv-2509.13414_MapAnything/](../../../papers/3d_reconstruction/latex/arXiv-2509.13414_MapAnything/root.tex)（root.tex 为主文，text/00–06 为分节，本精读所有式号以源码排版为准：全文编号公式仅 2 个——`eq:MapAnything` 为式 (1)、`eq:training_loss` 为式 (2)，其余为行内式，已逐条核对）｜ **教程**：[第 09 章](../09_哈希编码与前馈重建.md)（"前馈"路线扩展条目）+ [第 01 章](../01_3D重建问题与表示全景.md)（表示全景）｜ **代码**：无（教学实现规划中；官方实现为 Meta 的 map-anything 开源仓库，权重 Apache 2.0）

## 1. 论文信息与一句话贡献

- **题目**：MapAnything: Universal Feed-Forward Metric 3D Reconstruction
- **作者**：Nikhil Keetha、Norman Müller、Johannes Schönberger、Lorenzo Porzi 等（Meta Reality Labs + CMU）
- **发表**：arXiv:2509.13414（2025-09），已被 3DV 2026 录用（源码 root.tex 中 `\confName{3DV}`、`\confYear{2026}`）
- **一句话贡献**：首个**通用（universal）前馈米制 3D 重建骨干**——输入 $N$ 张图像 + **可选**的几何先验（射线方向/位姿/深度/尺度，任意子集任意视图），一次前向输出**因式分解**（factored）的场景表示（单位射线方向 + 沿线深度 + 位姿四元数/平移 + 全局米制尺度因子），端到端统一 12 种以上重建任务配置（未标定 SfM、标定 MVS、单目深度、相机定位、米制深度补全等），不微调即达到或超过各任务专家模型。

## 2. 问题与动机

**前馈重建的"碎片化"现状**（§1、§2 Related Work）：DUSt3R/MASt3R 证明了"一次前向出几何"的威力，但存在四重限制—— **耦合表示**：相机、位姿、几何全部纠缠在 pointmap 里，多视图时还要昂贵的后处理与对称推理去"事后"拆出相机； **输入固定**：只吃图像，机器人明明常有内参/外参/深度先验（IMU、GPS、RGB-D）却用不上； **任务割裂**：每个任务配置各训各的模型（Pow3R 虽引入先验但只支持两张针孔图像、无法条件化米制尺度）； **尺度天生缺失**：损失对尺度归一化，输出"直到未知尺度"（[DUSt3R 精读](./DUSt3R_CVPR2024.md) §4.2）。

**作者的关键洞察**（§1）：把多视图场景几何**因式分解**——不再直接回归"点图集合"，而是回归"深度图 + 局部射线图 + 相机位姿 + 一个米制尺度因子"的乘积分解。好处是三重的：输入输出共用同一套分解（有什么先验就编码什么）；带部分标注（只有 up-to-scale 真值）的数据集都能参训；尺度因子把局部重建"升级"为全局一致的米制框架。机器人场景（论文引文中明确点名）恰是"有部分先验 + 要米制尺度"的典型用户。

## 3. 方法总览

```
N 张图像 ──▶ DINOv2 ViT-G（第 24 层归一化 patch 特征, 1536×H/14×W/14）──┐
可选 射线方向 R̂ ──▶ 浅层卷积编码器（ControlNet 式, pixel-unshuffle 14）──┤
可选 逐像素深度 D̂ ─┘（先分解为 平均深度尺度 + 归一化深度）               ├──逐项 LayerNorm→求和→LayerNorm
可选 位姿（四元数 Q̂ + 平移 T̂ → 平移分解为位姿尺度 + 方向）─▶ 4 层 MLP ┘        │
                                                                    tokens F_E ∈ ℝ^{1536×HW/256}
                                                                    + 可学习 scale token + 参考视图 embedding
                                                                    ▼
                                    16 层交替注意力 transformer（24 头, 维度 1536, MLP 比 4,
                                    由 DINOv2 ViT-G 最后 16 层初始化；无 RoPE）
                                    ▼
   ┌ DPT 头（单头解码 N 视图 tokens）──▶ 单位射线方向 R_i ∥ 上至尺度深度 D̃_i ∥ 掩码 M_i ∥ 世界系点图置信度 C_i
   ├ 平均池化卷积位姿头 ──────────────▶ 四元数 Q_i + 上至尺度平移 T̃_i（均相对视图 1）
   └ scale token ─▶ 2 层 MLP(ReLU) ─▶ 米制尺度因子 m（指数参数化）
```

- **交替注意力**（alternating attention，VGGT 同款）：帧内/帧间注意力逐层交替，使任意数量视图在**同一次前向**内交换信息——多视图一致性由此获得，**不再需要测试时全局对齐**（§4.4）。
- **训练**（§4.3）：13 个高质量数据集（室内/室外/wild），其中 6 个训 Apache 2.0 模型、加 7 个训 CC BY-NC 4.0 模型；新增 MPSD 真实多视图米制数据集（约 72K 场景，元数据开源）；基于共视图的随机游走多视图采样（阈值 25%）。
- **开源**：数据处理/推理/评测/训练全套代码 + 预训练权重（Apache 2.0），定位是"3D/4D 基础模型的可扩展底座"。

## 4. 关键公式推导（核心章节：先给符号表，再逐式推导）

**符号表**（依源码 text/03_method.tex 记法；$\uts{\cdot}$ = 上至尺度量（tilde），$\metric{\cdot}$ = 米制量）：

| 符号 | 含义 |
|---|---|
| $\hat{\mathcal{I}} = (\hat{I}_i)_{i=1}^N$ | $N$ 张输入图像（必须） |
| $\hat{\mathcal{R}}, \hat{\mathcal{Q}}, \hat{\mathcal{T}}, \hat{\mathcal{D}}$ | 可选输入：射线方向、位姿四元数、平移、逐像素射线深度；各自定义在视图下标子集 $S_\text{r}, S_\text{q}, S_\text{t}, S_\text{d} \subseteq [1,N]$ 上 |
| $m \in \mathbb{R}$ | 预测的全局米制尺度因子 |
| $R_i \in \mathbb{R}^{3\times H\times W}$ | 预测的局部射线方向（单位化） |
| $\uts{D}_i \in \mathbb{R}^{1\times H\times W}$ | 上至尺度的射线深度 |
| $\uts{P}_i \in \mathbb{R}^{4\times4}$ | 图像 $\hat{I}_i$ 在视图 1 系下的位姿（四元数 $Q_i$ + 上至尺度平移 $\uts{T}_i$） |
| $\uts{L}_i, \uts{X}_i$ | 上至尺度的局部/世界系点图（由式 (1) 输出复合，见 §4.1） |
| $\hat{X}_i, V_i$ | 真值世界系点图与有效像素掩码 |
| $\hat{z}, \uts{z}, \metric{z}$ | 真值/上至尺度预测/米制预测的归一化尺度因子（§4.2） |
| $C_i$ | 预测的世界系点图置信度图 |
| $f_\mathrm{log}$ | 径向对数压缩 $f_\mathrm{log}: \mathbf{x} \to (\mathbf{x}/\|\mathbf{x}\|)\cdot\log(1+\|\mathbf{x}\|)$ |
| $\mathrm{sg}(\cdot)$ | stop-gradient（截断梯度） |

### 4.1 因式表示的定义与复合推导（式 (1) 及其后行内式）

$$
f_\text{MapAnything}\bigl(\hat{\mathcal{I}}, [\hat{\mathcal{R}}, \hat{\mathcal{Q}}, \hat{\mathcal{T}}, \hat{\mathcal{D}}] \bigr)
= \{m, (R_i, \uts{D}_i, \uts{P}_i)_{i=1}^N \}
\tag{1}
$$

式 (1) 是全文的表示论纲领：网络不直接输出点图，而输出**射线 × 深度 × 位姿 × 尺度**四个因子。随后用三个行内式把它们复合回点图（每步注明依据）：

1. **局部点图**：$\uts{L}_i = R_i \cdot \uts{D}_i \in \mathbb{R}^{3\times H\times W}$（依据：逐像素广播乘——单位方向向量乘以沿线标量深度即得相机系 3D 点；与针孔反投影 $K^{-1}[uD,vD,D]^\top$（[DUSt3R 精读](./DUSt3R_CVPR2024.md) §4.1）逐项对比：$R_i$ 把"内参 $K$"显式化为逐像素方向，故同一表示兼容针孔、鱼眼等任意中心投影相机）。
2. **世界系点图**：$\uts{X}_i = O_i \cdot \uts{L}_i + \uts{T}_i$（依据：$O_i$ 为 $Q_i$ 对应的旋转矩阵，$3\times3$ 旋转 + $3$ 维平移构成刚体变换，作用在逐像素点上；$\uts{T}$ 带波浪号故整体仍是 up-to-scale）。
3. **米制重建**：$\metric{X}_i = m \cdot \uts{X}_i$（依据：单一全局标量把 up-to-scale 空间整体放大到米制；尺度自由度（[教程第 02 章](../02_多视图几何与SfM.md) 式 (2.10) 的 7 自由度等价类中的尺度 1 维）被显式参数化为一个可监督的标量，而不是散落在点图里）。

**为什么分解是关键**（论文消融 tab:ablation_rep 的结论，§5"Insights"）：耦合点图中"标定（内参）×几何（深度）×位姿"互相纠缠，多视图时冗余预测互相拖累（FASt3R/VGGT 各自的冗余分支问题，§2 Related Work 有评述）；分解后射线与深度是**逐视图**量（单头即可）、位姿是**成对无关**量、尺度是**全局单标量**——各自只需最简单的头，且"有先验给先验、没先验全靠猜"对网络只是"哪些输入通道非零"。

### 4.2 输入编码的分解（行内式：尺度解耦与对数变换）

可选几何输入在进网络前先做两件规范化（每步注明依据）：

1. **深度分解**：$\hat{D}_i \to (\hat{z}_{di},\ \hat{D}_i/\hat{z}_{di})$，$\hat{z}_{di} \in \mathbb{R}^+$ 为逐视图平均深度（依据：分离"场景量级"与"形状"；支撑 metric 与 up-to-scale 两种输入共存）。
2. **平移分解**：位姿尺度 $\hat{z}_\text{p} = \frac{1}{|S_t|} \sum_{i\in S_t} \lVert \hat{T}_i \rVert$，输入用归一化平移 $\hat{T}_i/\hat{z}_\text{p}$，且 $\hat{z}_\text{p}$ 作为同一标量喂给所有带平移的帧（依据：平移与尺度纠缠——只有平移模长携带尺度信息，方向不变；旋转平移分开编码还兼容"只有 IMU 姿态、没有平移"的输入组合）。
3. **对数变换**：尺度值跨场景差异巨大（室内 2 m 到室外 200 m），编码前取对数（依据：把跨数量级的正量压到线性可回归的区间——与 §4.3 的 $f_\mathrm{log}$ 同一动机）。

图像特征（DINOv2 ViT-G 第 24 层，1536 维 patch tokens）、射线/归一化深度（浅层卷积编码器，ControlNet 式条件注入）、全局量（4 层 MLP + GeLU）投影到同一 1536 维空间后逐项 LayerNorm、求和、再 LayerNorm（依据：多模态早期融合——论文 Limitations 自述这是可改进的简化）。

### 4.3 损失体系（行内式 + 式 (2)）：归一化不变损失 + 可选米制损失

**与尺度无关的两项**（射线、四元数不受尺度影响，直接回归）：

$$
\mathcal{L}_\text{rays} = \sum_{i=1}^N \bigl\| \hat{R}_i - R_i \bigr\|,
\qquad
\mathcal{L}_\text{rot} = \sum_{i=1}^N \min\bigl(\| \hat{Q}_i - Q_i \|,\ \| {-\hat{Q}_i} - Q_i \|\bigr)
$$

$\mathcal{L}_\text{rot}$ 的推导（论文注明"accounts for the two-to-one mapping of unit quaternions"）：单位四元数 $q$ 与 $-q$ 表示**同一旋转**（$\theta$ 绕轴 $\mathbf{u}$ 与 $-\theta$ 绕 $-\mathbf{u}$ 同余，$q = (\cos\tfrac{\theta}{2}, \mathbf{u}\sin\tfrac{\theta}{2})$ 取负后对应转角 $\theta + 2\pi$——同一旋转，依据：四元数双覆盖 $SU(2)\to SO(3)$）。故真值 $\hat{Q}_i$ 与预测 $Q_i$ 差的正负号无意义，对两个符号取最小即等价于测地角距离的单调代理。

**尺度因子三件套**（沿用 DUSt3R 的 norm 思想，但全 $N$ 视图联合统计）：

$$
\hat{z} = \frac{\bigl\| (\hat{X}_i[V_i])_{i=1}^{N} \bigr\|}{\sum_{i=1}^N V_i},
\qquad
\uts{z} = \frac{\bigl\| (\uts{X}_i[V_i])_{i=1}^{N} \bigr\|}{\sum_{i=1}^N V_i},
\qquad
\metric{z} = m \cdot \mathrm{sg}(\uts{z})
$$

（依据：有效像素上全部 3D 点的**平均模长**——DUSt3R 式 (3) norm 的 $N$ 视图推广；$\metric{z}$ 用预测的 $m$ 乘**截断梯度**的 $\uts{z}$，$\mathrm{sg}$ 使 $\mathcal{L}_\text{scale}$ 的梯度只流向尺度头、不通过平均模长反渗扰动几何——论文原话"to ensure that gradients from the scale loss do not influence the geometry"。）

**归一化不变损失**（先过 $f_\mathrm{log}$，再各自除以 $\hat{z}/\uts{z}$）：

$$
\mathcal{L}_\text{translation} = \sum_{i=1}^N \Bigl\| \hat{T}_i/\hat{z} - \uts{T}_i/\uts{z} \Bigr\|,
\quad
\mathcal{L}_\text{depth} = \sum_{i=1}^N \Bigl\| f_\mathrm{log}(\hat{D}_i/\hat{z}) - f_\mathrm{log}(\uts{D}_i/\uts{z}) \Bigr\|,
\quad
\mathcal{L}_\text{lpm} = \sum_{i=1}^N \Bigl\| f_\mathrm{log}(\hat{L}_i/\hat{z}) - f_\mathrm{log}(\uts{L}_i/\uts{z}) \Bigr\|
$$

**$f_\mathrm{log}$ 的推导**（论文注明"critical to apply losses in log-space"）：$f_\mathrm{log}(\mathbf{x}) = \frac{\mathbf{x}}{\|\mathbf{x}\|}\log(1+\|\mathbf{x}\|)$ 保持方向不变（单位向量因子）、把模长压成 $\log(1+\|\mathbf{x}\|)$（依据：径向坐标变换；$\|\mathbf{x}\|\to 0$ 时 $\log(1+\|\mathbf{x}\|) \approx \|\mathbf{x}\|$ 退化回线性，近处无损；远处大模长的残差被对数压缩，防止单个远点主导损失——对深度/点图这类量程跨数量级的输出尤为关键）。

**置信度加权点图损失**（DUSt3R 式 (4) + $f_\mathrm{log}$）与米制尺度损失：

$$
\mathcal{L}_\text{pointmap} = \sum_{i=1}^N \Bigl( C_i \bigl\| f_\mathrm{log}(\hat{X}_i/\hat{z}) - f_\mathrm{log}(\uts{X}_i/\uts{z}) \bigr\| - \alpha \log C_i \Bigr),
\qquad
\mathcal{L}_\text{scale} = \bigl\| f_\mathrm{log}(\hat{z}) - f_\mathrm{log}(\metric{z}) \bigr\|
$$

（$\mathcal{L}_\text{pointmap}$ 中 $-\alpha\log C$ 防置信度归零逃损失、最优 $C^\ast = \alpha/\ell$ 的推导见 [DUSt3R 精读](./DUSt3R_CVPR2024.md) §4.3，两文同构；注意 $\uts{X}$ 依赖位姿与深度——该项把多视图一致性一并监督。）

**辅助项**：$\mathcal{L}_\text{normal}$（局部点图上的法向损失）与 $\mathcal{L}_\text{GM}$（局部点图对数 z-深度的多尺度梯度匹配损失）**只施于合成数据**（真实数据几何粗噪，依据：监督质量决定损失适用性）；$\mathcal{L}_\text{mask}$（非歧义类掩码的二元交叉熵）。逐像素回归损失**剔除最高的 5%**（剔离群真值）；所有回归损失用自适应鲁棒损失包装（参数 $c = 0.05$、$\alpha = 0.5$——鲁棒核从手工选 Huber/Tukey（[教程第 02 章](../02_多视图几何与SfM.md) 式 (2.15)(2.16)）变为参数可学的连续族）。

**总损失（式 (2)，源码 `eq:training_loss`）**：

$$
\mathcal{L}
= 10 \cdot \mathcal{L}_\text{pointmap}
+ \mathcal{L}_\text{rays}
+ \mathcal{L}_\text{rot}
+ \mathcal{L}_\text{translation}
+ \mathcal{L}_\text{depth}
+ \mathcal{L}_\text{lpm}
+ \mathcal{L}_\text{scale}
+ \mathcal{L}_\text{normal}
+ \mathcal{L}_\text{GM}
+ 0.1 \cdot \mathcal{L}_\text{mask}
\tag{2}
$$

（依据：消融表明上权重全局点图项（$10\times$，多视图一致性的主锚点）、下权重掩码项（$0.1\times$）最有利。）

### 4.4 "通用"的训练法与多视图一致性策略

- **先验即提示（prompt）的训练**：几何输入整体出现概率 0.9（射线/深度/位姿各 0.5，深度选中时稠密与 90% 随机稀疏各半），逐视图出现概率 0.95，且以 0.05 概率对 metric 真值数据**扣下**米制尺度因子（逼网络在"给先验/不给先验"两种模式下都可用）。消融（tab:ablation_uni_training）显示这一概率化训练的**单一**通用模型与按配置特训的定制模型打平——一次训练覆盖 12+ 任务配置。
- **多视图采样**：预计算全场景成对共视（真值深度+位姿的重投影误差校验），训练时按 25% 共视阈值做随机游走，采样"单连通分量"视图组——保证输入视图互相可约束。
- **多视图全局对齐策略：没有，而这正是卖点**（任务规格问"若有：写出"——论文的答案是"无测试时对齐"）：DUSt3R/MASt3R 两两推理后需式 (5) 式的 BA-free 优化（[DUSt3R 精读](./DUSt3R_CVPR2024.md) §4.4）拼成全局一致模型；MapAnything 用交替注意力让 $N$ 视图 tokens 在前向内互看，一致几何一步到位（§2 Related Work："without redundancies or costly post-processing"）；评测中唯一的"对齐"是部分基准协议要求的**中位数尺度对齐**（把 up-to-scale 输出对到真值尺度量级——这不是模型组件，是评测惯例）。

### 4.5 与 DUSt3R / MASt3R / VGGT 的谱系（论文 §2 Related Work 自己的表）

| 方法 | 表示 | 视图 | 输入 | 尺度 | 多视图一致化 |
|---|---|---|---|---|---|
| DUSt3R | 耦合点图 | 2 | 仅图像 | up-to-scale | 成对推理 + 式 (5) 全局优化 |
| MASt3R | 耦合点图 + 局部特征 | 2 | 仅图像 | **metric**（式 (6) 的 $z := \hat z$） | 同上（论文外，代码库/下游） |
| Pow3R | 耦合点图 | 2 | 图像 + 先验（针孔单焦距） | 不可条件化 | BA 变体 |
| MV-DUSt3R+ / VGGT | 耦合点图（VGGT 双分支多量冗余） | N | 仅图像 | up-to-scale | 交替注意力，一次前向 |
| Spann3R / CUT3R / MUSt3R | 隐式记忆状态 | 流式 | 仅图像 | — | 无经典优化，性能尚不及"MASt3R 输出 + 优化" |
| π³ | 去参考系解耦 | N | 仅图像 | up-to-scale | 一次前向（VGGT 微调） |
| **MapAnything** | **因式分解（射线×深度×位姿×尺度）** | **任意 N** | **图像 + 任意几何先验子集** | **metric（可提示可预测）** | **交替注意力，一次前向，零后处理** |

## 5. 实验与结果解读

- **多视图稠密重建**（ETH3D 去畸变版 / ScanNet++ v2 / TartanAirV2-WB，2–100 视图单连通组）：仅图像输入下点图/位姿/深度/射线四项指标（AbsRel、τ@1.03、ATE RMSE、AUC@5）对 VGGT 等前馈基线全面领先，且视图数增大时优势扩大（VGGT 显存溢出处 MapAnything 仍在跑）；加入可选先验后性能再显著抬升；对比同为"先验条件化"的 Pow3R（含其 BA 变体）也胜出。
- **两视图重建与匹配**：稀疏视图重建与图像匹配对前馈 SOTA 取胜；带先验时显著超过仅图像基线与 Pow3R。
- **单视图标定**（ETH3D/ScanNet++v2/TartanAirV2 随机帧，3:1 到 1:2 随机裁剪逼非居中主点）：未专训单图仍达透视标定 SOTA——射线头即"可学习标定器"，且作者据此判断可推广到鱼眼等宽角模型。
- **单目/多视图深度**（RMVD 基准）：未专训单目米制深度仍 SOTA 或可比；仅图像的多视图米制深度超过 MASt3R-BA 与 MUSt3R；带标定/位姿输入时逼近专用模型；ScanNet 上的米制尺度估计弱于 MoGe-2/MVSA（作者归因于基准数据质量，做中位数尺度对齐后表现强）。
- **消融**（§5 Insights）：因式 RDP（rays-depth-pose）表示 + 米制尺度是性能的关键使能（对比耦合点图与 π³ 式解耦均更差）；概率化先验输入训练的通用模型 ≈ 定制模型。

**解读**：MapAnything 的实验设计处处对着"universal"两个字验证——跨任务不微调、跨视图数不重训、先验随到随用且用了就涨点。

## 6. 局限与后续影响

**论文自述局限**（§ Limitations）： 不对几何输入的噪声/不确定性建模（给的先验被默认全信）； 尚不支持无图像视图（如 NVS 目标视图只有相机——架构可扩展）； 迭代推理/测试时算力扩展的潜力未挖掘（与 一体）； 多模态特征在进 transformer 前融合（早期融合的效率与表达天花板）； **可扩展性受"输入像素—输出表示一比一"限制**，大场景需要"按需表示/按需解码"的新参数化； 不建模动态场景与场景流。

**影响与机器人前端潜力**：代码 + Apache 2.0 权重全开源 + 评测协议标准化，作者明确定位为 3D/4D 基础模型的底座；对机器人（[教程第 10 章](../10_机器人场景中的重建实战与选型.md) §10.2 选型矩阵、§10.4 口诀"没标定、要快 → 前馈先出几何与位姿"）的意义： 机器人常有的内参/外参/深度先验从"丢弃"变为"提示"，秒级重建的精度上限被抬高； 米制尺度开箱即得，抓取/避障所需的绝对量纲（第 10 章 §10.1 的 ESDF 距离、§10.3 的抓取闭环）不再依赖后处理定标； 无后处理单次前向 = 可预测的延迟，适合在线闭环。开源权重生态使"前馈前端 + 逐场景精化（哈希加速的 3DGS/NeuS）"配方（[教程第 09 章](../09_哈希编码与前馈重建.md) §9.4）可以直接落地。

## 7. 与本项目对照

**教程第 09 章"前馈"路线的收束——DUSt3R → MASt3R → MapAnything 演进表**：

| | DUSt3R（CVPR 2024） | MASt3R（ECCV 2024） | MapAnything（arXiv 2025） |
|---|---|---|---|
| 表示 | 耦合点图 $X^{1,1}, X^{2,1}$ | 点图 + 稀疏匹配特征 $D^v$ | 因式分解 $R_i \times \uts{D}_i \times \uts{P}_i \times m$ |
| 视图窗口 | 2 | 2 | 任意 $N$ |
| 输入 | 仅图像 | 仅图像 | 图像 + 可选射线/位姿/深度/尺度 |
| 尺度 | up-to-scale | 可 metric | metric（可提示可预测，全局标量） |
| 多视图 | 两两推理 + BA-free 对齐（式 (5)） | 两两推理（对齐交下游） | 交替注意力一次前向，零后处理 |
| 教程位置 | §09.2（(9.5)–(9.8)） | §09.3（匹配头 + (9.9)–(9.11) 兜底） | **第 09 章"扩展"条目主读** |

- **第 01 章衔接**：教程 (1.4) 三级管线中"①几何估计 + ②多视图融合"被前馈路线压进一次网络前向——第 01 章正文已点名"前馈路线（DUSt3R/MASt3R/**MapAnything**）"；MapAnything 把融合级做到 $N$ 视图单步、把估计级做到可注入先验，是三级管线压缩的当前完成态。
- **第 09 章衔接**：(9.6) 的尺度归一化思想在 MapAnything 中升级为"$\uts{z}/\hat{z}$ 归一化不变损失 + 独立米制尺度损失 $\mathcal{L}_\text{scale}$"两截；(9.7)–(9.8) 的置信度加权原样保留于 $\mathcal{L}_\text{pointmap}$；§9.4"前馈给初值、优化精化"中，MapAnything 把"初值"质量推高到多数任务免优化。
- **精读链**：[DUSt3R 精读](./DUSt3R_CVPR2024.md)（表示与损失起点）→ [MASt3R 精读](./MASt3R_ECCV2024.md)（匹配与米制）→ 本篇（$N$ 视图、因式化、通用化）→ [SLAM 精读 MASt3R-SLAM](../../slam/精读/MASt3R-SLAM_CVPR2025.md)（先验装回优化骨架的对照样本：优化派对前馈派的回答）。

## 配套阅读

- LaTeX 源码：[root.tex](../../../papers/3d_reconstruction/latex/arXiv-2509.13414_MapAnything/root.tex)（主文）｜[text/03_method.tex](../../../papers/3d_reconstruction/latex/arXiv-2509.13414_MapAnything/text/03_method.tex)（式 (1)(2) 与全部行内式）｜[text/02_related.tex](../../../papers/3d_reconstruction/latex/arXiv-2509.13414_MapAnything/text/02_related.tex)（谱系评述）｜[text/04_benchmarking.tex](../../../papers/3d_reconstruction/latex/arXiv-2509.13414_MapAnything/text/04_benchmarking.tex)（基准与消融）。
- 教程：[第 09 章](../09_哈希编码与前馈重建.md)（(9.5)–(9.11) 前馈主线）｜[第 01 章 (1.4)](../01_3D重建问题与表示全景.md)（三级管线与前馈的位置）｜[第 10 章 §10.2/10.4](../10_机器人场景中的重建实战与选型.md)（机器人选型与"没标定、要快"口诀）。
- 姊妹精读：[DUSt3R（CVPR 2024）](./DUSt3R_CVPR2024.md)｜[MASt3R（ECCV 2024）](./MASt3R_ECCV2024.md)｜[SLAM 精读 MASt3R-SLAM（CVPR 2025）](../../slam/精读/MASt3R-SLAM_CVPR2025.md)。
- 论文库索引：[papers/3d_reconstruction/README.md](../../../papers/3d_reconstruction/README.md)。

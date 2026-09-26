# 论文精读｜VINS-Mono：单目视觉惯性状态估计（RAM 2018）

> **PDF**：[papers/slam/classics/arXiv-1708.03852_VINS-Mono.pdf](../../../papers/slam/classics/arXiv-1708.03852_VINS-Mono.pdf)（arXiv v2，2017-08-13，17 页，本精读所有式号以此 PDF 为准）｜ **教程**：[第 10 章 §10.2–10.3](../10_建图与系统实战.md) ｜ **代码**：[projects/slam/vins/](../../../projects/slam/vins/)（紧耦合因子图）+ [projects/slam/preint/](../../../projects/slam/preint/)（预积分引擎）

## 1. 论文信息与一句话贡献

- **题目**：VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator
- **作者**：Tong Qin、Peiliang Li、Shaojie Shen（香港科技大学 HKUST Aerial Robotics）
- **发表**：arXiv:1708.03852（本仓库收录 PDF）；期刊版发表于 IEEE Robotics and Automation Magazine（RAM）2018 年第 25 卷第 3 期。**注**：教程与代码中的引用此前误写为 "IEEE T-RO 2018"，现已统一修正为 RAM 2018（预积分理论出处 Forster et al. 为 T-RO 2017，二者勿混淆）。
- **一句话贡献**：提出一套**完整可开源**的单目视觉惯性 SLAM 系统——以"滑动窗口内对全部帧位姿/速度/偏置 + 外参 + 特征逆深度做紧耦合 MAP 优化"为核心，配齐鲁棒初始化（视觉-惯性对齐）、在线外参与 IMU 偏置标定、DBoW2 回环重定位与 **4 自由度位姿图**全局优化四个模块，并在 EuRoC、室内长廊、5.62 km 校园级场景与真机四旋翼闭环控制上验证（贡献清单见论文 §I 末尾列表）。

## 2. 问题与动机

**要解决什么**：单目相机 + 低成本 IMU 是机器人的最小传感器组合（尺寸、功耗、成本），但融合有三个经典难点（§I）： 无法直接测距离，尺度不可观——必须靠 IMU 加计积分补上米制尺度； IMU 处理存在估计器初始化、外参标定、非线性优化三重挑战； 长期漂移需要完整的"回环检测 + 重定位 + 全局优化"链路。

**为什么是紧耦合优化而非滤波**（§II 相关工作）：松耦合（把视觉里程计结果当测量喂给滤波）丢弃了视觉相关性；紧耦合 EKF（MSCKF 系）线性化一次即固定。图优化类 VIO 的分歧在于滑窗大小：固定窗口滑窗法（论文自引 [7][8]，即 VINS 前身）只优化近期状态、把出窗状态边缘化——计算量有界，适合无人机与手机。本文在自家前作上补齐：预积分中的 IMU 偏置修正（(12)）、鲁棒初始化（§V）、4 自由度位姿图（§VIII）与完整开源实现。

**框架速记**：相机 30 Hz 出帧 → KLT 跟踪 + 两点关键帧判据（平均视差、跟踪质量，§IV-A）；IMU 100–200 Hz 读数 → 预积分（§IV-B）；二者进滑窗 MAP（§VI）；回环用 DBoW2（词袋，bag-of-words）+ BRIEF 描述子检索、特征级紧耦合重定位（§VII）；全局位姿图只优化 4 自由度（§VIII）。

## 3. 方法总览

```
【测量预处理 §IV】相机(30Hz): KLT光流跟踪 + 新角点补充 + RANSAC基本矩阵剔外点
                 IMU(100Hz+): 预积分 α,β,γ —— (6)定义、(7)欧拉积分、(9)-(11)协方差/雅可比递推
                              偏置变化 → 一阶修正 (12)，不重新积分
【初始化 §V】视觉滑窗 SfM（五点法恢复相对旋转+up-to-scale平移，PnP 补全，全局 BA）
        → 陀螺偏置标定 (15) → 速度/重力/尺度线性最小二乘 (16)-(20) → 重力 2 自由度精化
【滑窗 VIO §VI】状态 (21)：每帧 [p, v, q, b_a, b_w] + 外参 [p_b^c, q_b^c] + 每特征逆深度 λ
        MAP (22) = 边缘化先验 ‖r_p − H_p χ‖² + Σ‖r_B‖²_P (IMU, (24)) + Σ ρ(‖r_C‖²_P) (视觉, (25), Huber (23))
        边缘化策略 (§VI-D)：次新帧是关键帧→丢最老帧并 Schur 补成先验；否则丢其视觉测量、保留 IMU 链
        伴生机制：motion-only BA（30Hz 相机速率输出，约 5ms）+ IMU 前向传播（IMU 速率反馈）
【重定位 §VII】DBoW2 检索候选 + BRIEF 特征检索对应 + 2D-2D/3D-2D 两级几何剔错
        → 把回环特征观测以额外残差项加进 (22)（公式 (26)），回环帧位姿当常量
【位姿图 §VIII】关键帧出窗即入图：顺序边 (27) + 回环边；4-DOF 残差 (28)、代价 (29)，异步线程
```

## 4. 关键公式推导（式号按论文 PDF 原文）

### 4.0 符号表

| 符号 | 含义 |
|---|---|
| $(\cdot)^w, (\cdot)^b, (\cdot)^c$ | 世界系（z 轴与重力对齐）、体/IMU 系、相机系（§III 帧定义） |
| $R_a^b,\ q_a^b$ | 把 $a$ 系坐标变换到 $b$ 系的旋转（**下标源、上标目标**；依据 §III "rotation from … to …" 与 (3)(25) 的用法一致） |
| $\hat{a}_t, \hat{\omega}_t$ | $t$ 时刻加计/陀螺原始读数；$b_{a_t}, b_{w_t}$ 偏置；$n_a, n_w$ 高斯白噪声 (1) |
| $\Delta t_k$ | 关键帧 $b_k \to b_{k+1}$ 的时间间隔 |
| $\alpha_{b_{k+1}}^{b_k}, \beta_{b_{k+1}}^{b_k}, \gamma_{b_{k+1}}^{b_k}$ | 预积分的位置/速度/旋转增量，在 $b_k$ 系中表达 (6) |
| $\lambda_l$ | 第 $l$ 个特征的**逆深度**（首次观测 $i$ 处，(21)） |
| $[b_1, b_2]$ | 单位球切平面的正交基（视觉残差的 2 维投影基，(25)） |
| $q \otimes p,\ [q]_{xyz}$ | 四元数 Hamilton 乘法 / 取向量部分 (24)；本文用 Hamilton 四元数（§III） |

### 4.1 状态向量：滑窗里到底放了什么（式 (21)）

$$\mathcal{X} = \big[\,x_0, x_1, \cdots, x_n,\; x_c^b,\; \lambda_0, \lambda_1, \cdots, \lambda_m\,\big], \quad x_k = \big[\,\mathbf{p}_{b_k}^w, \mathbf{v}_{b_k}^w, \mathbf{q}_{b_k}^w, \mathbf{b}_a, \mathbf{b}_g\,\big],\ k \in [0,n], \quad x_c^b = \big[\,\mathbf{p}_b^c, \mathbf{q}_b^c\,\big] \tag{21}$$

逐项解释：窗口内 $n{+}1$ 个关键帧，每帧 $x_k$ 含 3 维位置 + 3 维速度 + 4 元数姿态 + 加计/陀螺偏置（**每帧各自的偏置**，帧间靠 (24) 的随机游走残差拉住）；外参 $x_c^b$ 全窗口共享一份——**在线标定**（§I 贡献点）；$\lambda_l$ 是每个特征的 1 维逆深度。注意三点： 偏置进状态、预积分量对偏置一阶展开（(12)），因此偏置可以在滑窗里被优化而不用重新积分； 没有全局位姿图节点——出窗的关键帧交给 §VIII； 特征不存 XYZ 而存逆深度（§4.3 论证）。

### 4.2 IMU 预积分：本文的取舍点（式 (1)–(13)、(24)）

预积分理论本文直接引用 Forster et al.（T-RO 2017，[配套精读式号见教程 §10.2](../10_建图与系统实战.md)），本文重点在**它在图里怎么用**。对应关系（论文式号 ↔ 教程式号）：

| 本文 | 内容 | 教程 |
|---|---|---|
| (1)(2) | 测量模型 $\hat{a}_t = a_t + b_{a_t} + R_t^{tw}g^w + n_a$、偏置随机游走 $\dot{b}_{a_t} = n_{a_t}$ | (10.1) |
| (3)(4) | 世界系积分（初值锚定问题）与 $\Omega(\omega)$ 四元数微分矩阵 | (10.2)–(10.3) |
| (5)(6) | 换参考系到 $b_k$：$R_w^{b_k}p_{b_{k+1}}^w = R_w^{b_k}(p_{b_k}^w + v_{b_k}^w\Delta t_k - \frac{1}{2}g^w\Delta t_k^2) + \alpha_{b_{k+1}}^{b_k}$ 等；$\alpha,\beta,\gamma$ 定义 | (10.4) |
| (9)(10)(11) | 误差状态动力学 $\dot{z}_t^{b_k} = F_t\delta z_t^{b_k} + G_t n_t$、协方差递推 (10)、雅可比递推 (11) | (10.6)–(10.7) |
| (12) | 预积分量对偏置的一阶近似 $\alpha_{b_{k+1}}^{b_k} \approx \hat{\alpha}_{b_{k+1}}^{b_k} + J_{b_a}^\alpha\delta b_{a_k} + J_{b_w}^\alpha\delta b_{w_k}$ | (10.8) |
| (13) | 打包成测量向量 $\hat{z}_{b_{k+1}}^{b_k} = [\hat{\alpha}; \hat{\beta}; \hat{\gamma}; b_{a_{b_k}}; b_{w_{b_k}}]$ 及协方差 $P_{b_{k+1}}^{b_k}$ | — |

**在图中的使用——IMU 残差 (24)**：把 (13) 的测量模型"反着写"成残差：

$$r_{\mathcal{B}}(\hat{z}_{b_{k+1}}^{b_k}, \mathcal{X}) = \begin{bmatrix} R_w^{b_k}\big(p_{b_{k+1}}^w - p_{b_k}^w + \frac{1}{2}g^w\Delta t_k^2 - v_{b_k}^w\Delta t_k\big) - \hat{\alpha}_{b_{k+1}}^{b_k} \\ R_w^{b_k}\big(v_{b_{k+1}}^w + g^w\Delta t_k - v_{b_k}^w\big) - \hat{\beta}_{b_{k+1}}^{b_k} \\ 2\big[q_{b_k}^{w-1} \otimes q_{b_{k+1}}^w \otimes (\hat{\gamma}_{b_{k+1}}^{b_k})^{-1}\big]_{xyz} \\ b_{a_{b_{k+1}}} - b_{a_{b_k}} \\ b_{w_{b_{k+1}}} - b_{w_{b_k}} \end{bmatrix} \tag{24}$$

逐行依据：前两行 = (13) 的测量模型右端减去预积分量（残差 = 模型预测 − 测量；本式取"预测在前"的符号约定）；旋转行 = 相对旋转 $q_w^{b_k}\otimes q_{b_{k+1}}^w$（把 $b_{k+1}$ 系坐标变到 $b_k$ 系，依据 §III 约定）与预积分 $\hat{\gamma}$ 之差经 $\mathrm{Log}$ 提升为 3 维角误差向量——四元数误差到旋转向量的换算因子 2 来自 $[1, \frac{1}{2}\delta\theta]$ 型小量四元数的定义 (8)；后两行 = 偏置随机游走 (2) 的离散化残差（帧间偏置应近似相等）。**（排印注）** (5) 第三行印作 $q_{b_k}^w \otimes q_{b_{k+1}}^w = \gamma_{b_{k+1}}^{b_k}$，漏了 $q_{b_k}^w$ 的逆；按 (24) 的实际用法应为 $q_{b_k}^{w-1} \otimes q_{b_{k+1}}^w$——以 (24) 为准。教程 (10.10) 的 $\mathbf{r}_{\Delta R}, \mathbf{r}_{\Delta v}, \mathbf{r}_{\Delta p}$ 即 (24) 的前三行、偏置随机游走与本文一致的另一写法。

### 4.3 视觉残差：单位球 + 切平面 + 逆深度（式 (25)）

$$r_{\mathcal{C}}(\hat{z}_l^{c_j}, \mathcal{X}) = \big[\mathbf{b}_1, \mathbf{b}_2\big]^T \cdot \Big(\hat{P}_l^{c_j} - \frac{P_l^{c_j}}{\|P_l^{c_j}\|}\Big), \qquad \hat{P}_l^{c_j} = \pi_c^{-1}\Big(\Big[\begin{smallmatrix}\hat{u}_l^{c_j} \\ \hat{v}_l^{c_j}\end{smallmatrix}\Big]\Big) \tag{25}$$

预测项 $P_l^{c_j}$ 的**完整链式**（依据：每步是一次刚体变换 $x^b = R_a^b x^a + p_a^b$，第 02 章）：

$$P_l^{c_j} = R_b^c\Big(R_w^{b_j}\big(R_{b_i}^w\big(R_c^b\,\tfrac{1}{\lambda_l}\,\pi_c^{-1}\big(\Big[\begin{smallmatrix}u_l^{c_i}\\ v_l^{c_i}\end{smallmatrix}\Big]\big) + p_b^c\big) + p_{b_i}^w - p_{b_j}^w\big) - p_b^c\Big)$$

1. $\pi_c^{-1}([u_l^{c_i}, v_l^{c_i}]^T)$：首次观测 $i$ 的像素 → 相机系 $c_i$ 单位射线 $\bar{P}_l^{c_i}$（依据：针孔反投影，$\pi_c^{-1}$ 用内参把像素变回单位向量，§VI-C）。
2. 乘深度 $\frac{1}{\lambda_l}$：射线 → 3D 点（相机系）。
3. $R_c^b(\cdot) + p_b^c$：相机系 → 机体系（**外参 $x_c^b$ 在此处进入残差**，因此 (22) 能在线标定外参）。
4. $R_{b_i}^w(\cdot) + p_{b_i}^w$：$b_i$ 系 → 世界系。
5. 减 $p_{b_j}^w$、乘 $R_w^{b_j}$：世界系 → $b_j$ 系；再 $-p_b^c$、乘 $R_b^c$：机体系 → 相机系 $c_j$。
6. 归一化 $\frac{P_l^{c_j}}{\|P_l^{c_j}\|}$ 投回单位球，与观测反投影 $\hat{P}_l^{c_j}$ 相减得三维误差向量。

**为什么残差长这样**：视觉观测只有 2 自由度，三维误差必须投影到观测射线的切平面上才有意义——$[\mathbf{b}_1, \mathbf{b}_2]^T$ 是切平面正交基（构造同 Algorithm 1 的叉积法，§VI-C + Fig. 6）。单位球建模（而非针孔平面）使残差对广角/鱼眼/全向相机通用（§VI-C 首段）。

**逆深度参数化为什么好**（与 [DSO](./DSO_PAMI2018.md) §4.5 同一论证，互为印证）：三角化深度的条件数随深度增大而恶化——远处点的视差对深度变化极不敏感，深度 $\rho$ 的测量不确定性 $\sigma_\rho \propto \rho^2$ 量级增长，其分布严重非高斯；逆深度 $\lambda = 1/\rho$ 的不确定性近似均匀，对"无穷远"点也有良定义的有限参数（$\lambda \to 0$），使滑窗优化中远处点不至于把数值条件炸掉（依据：第 05 章三角化 + DSO 的同型论证）。VINS 只在**首次观测帧**存一个 $\lambda_l$（宿主帧参数化），后续帧由位姿链预测。

### 4.4 滑窗优化目标：MAP 形式（式 (22)、(23)）

$$\min_{\mathcal{X}} \bigg\{ \|r_p - H_p\mathcal{X}\|^2 + \sum_{k\in\mathcal{B}} \big\| r_{\mathcal{B}}(\hat{z}_{b_{k+1}}^{b_k}, \mathcal{X}) \big\|^2_{P_{b_{k+1}}^{b_k}} + \sum_{(l,j)\in\mathcal{C}} \rho\Big( \big\| r_{\mathcal{C}}(\hat{z}_l^{c_j}, \mathcal{X}) \big\|^2_{P_l^{c_j}} \Big) \bigg\} \tag{22}$$

逐项解释： 第三项视觉残差外层套 **Huber 核** $\rho(s) = s\ (s\le 1);\ 2\sqrt{s} - 1\ (s>1)$ (23)——剔匹配外点； IMU 残差**不加**鲁棒核（论文未对 $\mathcal{B}$ 求和项加 $\rho$，依据：IMU 是自家传感器、无数据关联外点，且预积分协方差 (13) 已正确加权）； 第一项 $\{r_p, H_p\}$ 是边缘化先验（"prior information from marginalization"，§VI 论证），保留出窗信息。概率语义：高斯测量的负对数似然展开后常数项与 $\arg\min$ 无关（[第 08 章 (8.2)](../08_后端-ii图优化与-ba.md) 的推导），马氏范数 $\|\cdot\|^2_P$ 即白化后的最小二乘。求解器：Ceres Solver（§VI-A 末尾）。

### 4.5 边缘化：先验如何构造、FEJ 为什么必要（§VI-D；回引 [第 08 章 (8.15)/(8.17)](../08_后端-ii图优化与-ba.md)、教程 (10.11)）

**两种丢帧情形**（§VI-D + Fig. 7）： 次新帧是关键帧 → 保留它，**边缘化最老帧** $x_0$ 及其全部视觉/IMU 测量，化为先验； 次新帧非关键帧 → 直接**扔掉它的视觉测量**（保持稀疏），但保留连着它的 IMU 预积分、预积分过程继续向下一帧累积。设计意图：窗口内保持**空间上分离**的关键帧，保证足够视差三角化（§VI-D）。

**先验构造（流形上的二次型消元）**。设本轮要把状态分成被边缘化块 $\delta x_m$（出窗帧相关状态）与保留块 $\delta x_r$，流形更新 $\mathcal{X} \leftarrow \mathcal{X}\boxplus\delta\mathcal{X}$（切向量增量，[第 08 章 (8.9)](../08_后端-ii图优化与-ba.md)）。把与 $x_m$ 相关的全部测量（IMU 因子 (24)、触达 $x_m$ 的视觉因子 (25)、以及**既有先验**）在当前线性化点 $x_0$ 处一阶展开：$e(x) \approx e(x_0) + J\,\delta\mathcal{X}$（依据：多元函数一阶 Taylor 展开）。加权代价 $C = \|e_0 + J\delta\mathcal{X}\|^2_\Sigma$ 展开为

$$C = \text{const} + 2\,g^\top \delta\mathcal{X} + \delta\mathcal{X}^\top H\,\delta\mathcal{X}, \qquad H = J^\top\Sigma^{-1}J,\quad g = J^\top\Sigma^{-1}e_0$$

（依据：二次型展开；$H$ 的定义同第 08 章 (8.5) 正规方程矩阵）。按块写 $H = \begin{bmatrix} H_{mm} & H_{mr} \\ H_{rm} & H_{rr}\end{bmatrix}$、$g = [g_m; g_r]$。**边缘化 = 对 $\delta x_m$ 取极小**（概率上即积分掉 $x_m$，与第 08 章 (8.17) 的"Schur 补即概率边缘化"同一条对偶）：令 $\partial C/\partial\delta x_m = 0$（依据：$H_{mm}$ 对称正定，极小点梯度为零），得

$$\delta x_m^\star = -H_{mm}^{-1}\big(H_{mr}\,\delta x_r + g_m\big)$$

代回 $C$ 并逐项合并（依据：代入展开；中间块交叉项 $\delta x_m^\top H_{mm}\delta x_m$ 贡献 $+\delta x_r^\top H_{rm}H_{mm}^{-1}H_{mr}\delta x_r$ 与交叉项 $-2\delta x_r^\top H_{rm}H_{mm}^{-1}H_{mr}\delta x_r$ 相抵后净剩一份，线性项 $2g_r^\top\delta x_r + 2\delta x_r^\top H_{rm}H_{mm}^{-1}g_m - 2\delta x_r^\top H_{rm}H_{mm}^{-1}g_m - 2\delta x_r^\top H_{rm}H_{mm}^{-1}g_m$ 净剩 $-2$）：

$$C_p(\delta x_r) = \text{const} + 2\,b_p^\top \delta x_r + \delta x_r^\top H_p\,\delta x_r, \qquad H_p = H_{rr} - H_{rm}H_{mm}^{-1}H_{mr},\quad b_p = g_r - H_{rm}H_{mm}^{-1}g_m$$

$H_p$ 正是 $H$ 关于 $x_m$ 的 **Schur 补**（依据：分块高斯消元，第 08 章 (8.15) 的同一代数；论文 §VI-D 明说 "Marginalization is carried out using the Schur complement"，引其文献 [39]）。对 $H_p$ 做 Cholesky 分解 $H_p = LL^\top$（依据：对称正定阵可 Cholesky），配方成平方形式：

$$C_p = \big\|\, L^\top\delta x_r + L^{-1}b_p \,\big\|^2 + \text{const} \;\;\equiv\;\; \big\|\, r_p + H_p^{1/2}\,\delta x_r \,\big\|^2$$

（依据：$\|L^\top\delta x_r + L^{-1}b_p\|^2 = \delta x_r^\top LL^\top\delta x_r + 2\delta x_r^\top LL^{-1}b_p + \text{const} = \delta x_r^\top H_p\delta x_r + 2b_p^\top\delta x_r + \text{const}$，用 $LL^{-1} = I$。）取 $\delta x_r = \mathcal{X} - x_0^{(r)}$（保留状态相对边缘化时刻估计的增量）并吸收符号，即得 (22) 的先验项 $\|r_p - H_p\mathcal{X}\|^2$——**论文的 $H_p$ 是平方根信息矩阵的记号复用**，语义为先验因子；教程 (10.11) 写作 $p(x_{\mathrm{remain}}) \propto \exp(-\frac{1}{2}\|r_{\mathcal{P}} + H_{\mathcal{P}}\delta x\|^2)$，是同一构造的概率表述。

**FEJ（First-Estimates Jacobians，首估计雅可比）为什么必要**。论文自己的话（§VI-D）："marginalization results in the early fix of linearization points, which may result in suboptimal estimation results. However, since small drifting is acceptable for VIO, we argue that the negative impact caused by marginalization is not critical."——先验的 $\{r_p, H_p^{1/2}\}$ 在**边缘化时刻一次性计算并冻结**（线性化点"早固定"，即 FEJ）。若不冻结、在后续迭代的当前估计处重算先验雅可比会怎样？一阶论证如下：先验代表的是**已被消费掉的一批非线性测量**，它的 $H$ 是这批测量在 $x_0$ 处的一阶近似。设某变量同时被"冻结先验"与"新线性化的因子"触达，两个因子在**不同**线性化点 $x_a$ 与 $x_b = x_a + \delta$ 处取雅可比，则总信息阵的交叉块为

$$J_1(x_a)^\top J_2(x_b) \approx J_1(x_a)^\top J_2(x_a) + J_1(x_a)^\top \dot{J}_2\,\delta$$

（依据：对 $J_2$ 在 $x_a$ 处做一阶 Taylor 展开；$\dot{J}_2$ 为雅可比对状态的导数。）多出的 $O(\|\delta\|)$ 项没有任何一致的概率解释。特别地，对 VIO 的不可观方向 $\mathbf{n}$（全局 yaw + 全局平移，见 §4.6）本来严格满足 $J\mathbf{n} = 0$（信息为零），混入不同线性化点后 $n^\top H n \approx O(\|\delta\|) \ne 0$——系统在**本应不可观的方向上凭空获得虚假信息**，估计协方差变得过度自信（inconsistent；点估计经规范对齐后可能仍好看，但状态与不确定度不再自洽）。这一效应正是 DSO 显式采用 FEJ（其论文 Eq.(13)，见[精读](./DSO_PAMI2018.md)）与 MSCKF 系可观测性一致性分析的同一根源。VINS 的务实取舍：滑窗内状态漂移小（IMU 把转速/加速度都钉住了），冻结线性化点的次优性可接受（论文引文如上）。

### 4.6 4 自由度不可观与回环后的位姿图（式 (26)–(29)；回引 [第 09 章 (9.7)–(9.8)](../09_回环检测.md)）

**为什么只剩 4 维不可观**（§VIII 开篇）：视觉-惯性系统使横滚/俯仰（重力方向在 (1) 的 $R_t^{tw}g^w$ 中出现）与尺度（加计积分带物理单位，见教程 §10.3 第 3 步的缩放论证）完全可观，漂移只发生在 4 自由度：全局 3 维平移 $(x,y,z)$ + 绕重力方向的偏航 yaw。因此全局优化**不碰**滚转/俯仰，只做 4-DOF 位姿图——这是它与第 09 章 (9.7)–(9.8) 的 6 自由度 $SE(3)$ 位姿图的唯一差别（那里是通用表述，这里是可观测性裁剪后的特例）。

**边与残差**。关键帧出窗即入图，连两种边：顺序边（VIO 相对量）与回环边（重定位给出），都只含 4-DOF 相对量 (27)：

$$\hat{\mathbf{p}}_{ij}^l = \hat{R}_i^{w-1}\big(\hat{\mathbf{p}}_j^w - \hat{\mathbf{p}}_i^w\big), \qquad \hat{\psi}_{ij} = \hat{\psi}_j - \hat{\psi}_i \tag{27}$$

（依据：相对平移 = 把世界系位移左乘 $-\hat{R}_i^w$ 旋进第 $i$ 帧的局部水平系；yaw 差直接相减。）边的残差 (28)：

$$r_{\mathcal{P}}\big(\mathbf{p}_i^w, \psi_i, \mathbf{p}_j^w, \psi_j\big) = \begin{bmatrix} R(\hat{\phi}_i, \hat{\theta}_i, \psi_i)^{-1}\big(\mathbf{p}_j^w - \mathbf{p}_i^w\big) - \hat{\mathbf{p}}_{ij}^l \\ \psi_j - \psi_i - \hat{\psi}_{ij} \end{bmatrix}_2 \tag{28}$$

其中 $\hat{\phi}_i, \hat{\theta}_i$ 是 VIO 给出的、**视为常量**的滚转/俯仰（依据：§VIII-B "obtained directly from monocular VIO"——它们可观且无漂移，不必进优化变量）；$R(\phi,\theta,\psi)$ 为 ZYX 欧拉角构造的旋转，下标 $[\cdot]_2$ 取 $[\hat{\mathbf{p}}_{ij}^l; \hat{\psi}_{ij}]$ 所属的 4 维（平面 $l$ 上 3 维位置 + 1 维 yaw）误差（依据：与 (27) 逐项相减）。全局代价 (29)：

$$\min_{p,\psi}\; \bigg\{ \sum_{(i,j)\in\mathcal{S}} \|\mathbf{r}_{i,j}\|^2 + \sum_{(i,j)\in\mathcal{L}} \rho\big(\|\mathbf{r}_{i,j}\|^2\big) \bigg\} \tag{29}$$

顺序边 $\mathcal{S}$ 不加鲁棒核（VIO 自带剔外点机制），回环边 $\mathcal{L}$ 加 Huber $\rho(\cdot)$ 防错误回环（§VIII-B）。回环特征对应以紧耦合方式先进 (22)（式 (26)：在 (22) 基础上加 $\sum_{(l,v)\in\mathcal{L}}\rho(\|r_{\mathcal{C}}(\hat{z}_l^{v}, \mathcal{X}, \hat{q}_v^w, \hat{p}_v^w)\|^2_{P_l^{c_v}})$，回环帧位姿 $(\hat{q}_v^w, \hat{p}_v^w)$ 从位姿图取值、当常量），重定位验证通过后才由 (29) 做全局校正——两级设计把"错误回环"的代价限制在窗口内（§VII-C、§VIII 开篇）。

### 4.7 初始化：视觉-惯性对齐逐式推导（§V；式 (14)–(20)）

初始化 = 松耦合线性求解：视觉滑窗 SfM 先自举（稳定跟踪 >30 特征、旋转补偿视差 >20 px 的帧对上跑五点法，PnP 补全窗口，再全局 BA，§V-A），得到 **up-to-scale** 的相机位姿 $\{\bar{p}_{c_k}^{c_0}, q_{c_k}^{c_0}\}$（以首帧 $c_0$ 为参考系）；再与 IMU 对齐。

**第 1 步：位姿从相机系搬到机体系（式 (14)）**。已知粗外参 $(p_b^c, q_b^c)$，把 SfM 相机位姿转成 IMU 位姿：

$$q_{b_k}^{c_0} = q_{c_k}^{c_0} \otimes (q_b^c)^{-1}, \qquad s\,\bar{\mathbf{p}}_{b_k}^{c_0} = s\,\bar{\mathbf{p}}_{c_k}^{c_0} - R_{b_k}^{c_0}\,\mathbf{p}_b^c \tag{14}$$

依据：把"$b_k \to c_k$ 再 $c_k \to c_0$"两个旋转复合（$\otimes$ 顺序为 Hamilton 约定下的链式复合）；平移侧，$c_0$ 系下 IMU 位置 = 相机位置 − 旋转后的外参平移（$x^{c_0} = R_{b_k}^{c_0}(x^{b_k} + \cdot)$ 型展开，第 02 章刚体变换），且该式把未知尺度 $s$ 显式留在外面。

**第 2 步：陀螺偏置标定（式 (15)）**。视觉 SfM 给出相邻帧相对旋转（已到机体系 $q_{b_k}^{c_0}, q_{b_{k+1}}^{c_0}$），IMU 预积分给出 $\hat{\gamma}_{b_{k+1}}^{b_k}$。二者应一致，对偏置一阶展开后取模长平方和最小：

$$\min_{\delta b_w} \sum_{k\in\mathcal{B}} \Big\| q_{b_{k+1}}^{c_0-1} \otimes q_{b_k}^{c_0} \otimes \hat{\gamma}_{b_{k+1}}^{b_k} \Big\|^2, \qquad \hat{\gamma}_{b_{k+1}}^{b_k} \approx \hat{\gamma}_{b_{k+1}}^{b_k} \otimes \Big[\begin{smallmatrix} 1 \\ \frac{1}{2} J_{b_w}^\gamma \delta b_w \end{smallmatrix}\Big] \tag{15}$$

推导： 复合 $q_{b_{k+1}}^{c_0-1} \otimes q_{b_k}^{c_0}$ 是"$b_{k+1}$ 系 → $b_k$ 系"的相对旋转（依据：(14) 各位姿同在 $c_0$ 系中，中间系相消，第 02 章位姿复合）； 若 IMU 完美，该复合应等于 $\hat{\gamma}$，残差四元数应为单位元 $[1, 0, 0, 0]^T$； 偏置修正按 (12) 的旋转项进入：$\hat{\gamma} \otimes [1; \frac{1}{2}J_{b_w}^\gamma\delta b_w]$（(12) 的 $J_{b_w}^\gamma$ 即预积分旋转对陀螺偏置的雅可比，教程 (10.8)）； 小角度四元数 $[\cos\frac{\theta}{2}, \sin\frac{\theta}{2}\mathbf{u}] \approx [1, \frac{1}{2}\theta\mathbf{u}]$（依据：$\cos\frac{\theta}{2} \approx 1 - \frac{\theta^2}{8} \approx 1$、$\sin\frac{\theta}{2} \approx \frac{\theta}{2}$，泰勒一阶），于是残差四元数的向量部分线性化为

$$\mathbf{v}\big(q_{b_{k+1}}^{c_0-1} \otimes q_{b_k}^{c_0} \otimes \hat{\gamma}\big) \cdot 1 + \tfrac{1}{2}\,w\big(q_{b_{k+1}}^{c_0-1} \otimes q_{b_k}^{c_0} \otimes \hat{\gamma}\big)\, J_{b_w}^\gamma\,\delta b_w \;\approx\; \mathbf{0}$$

（依据：Hamilton 乘法 $[q_w, q_v] \otimes [1, \frac{1}{2}\delta\theta] = [q_w,\ q_v + \frac{1}{2}q_w\delta\theta]$ 保留 $\delta\theta$ 一阶项，高阶项 $q_v \times \frac{1}{2}\delta\theta$ 略去。）把各帧的 $\frac{1}{2}q_w J_{b_w}^\gamma$ 堆成矩阵、$-\mathbf{v}(\cdot)$ 堆成右端，即线性最小二乘，解出 $\delta b_w$ 并更新 $b_w$，再用新偏置**重新传播**全部预积分量（§V-B-1；偏置只在此步整体重估一次，所以重积分代价可接受）。

**第 3 步：速度/重力/尺度的线性最小二乘（式 (16)–(20)）**。待求状态：

$$\mathcal{X}_I = \big[\mathbf{v}_{b_0}^{b_0}, \mathbf{v}_{b_1}^{b_1}, \cdots, \mathbf{v}_{b_n}^{b_n}, \mathbf{g}^{c_0}, s\big] \tag{16}$$

（每帧一个体系速度 + $c_0$ 系重力 + 一个尺度；注意**不含**偏置——第 2 步已标定，§V-B-2。）把预积分关系 (5) 第一行左乘 $R_{c_0}^{b_k}$、代入 (14)，把世界系量全部搬到 $c_0$ 系（依据：$R_w^{b_k}R_{c_0}^w = R_{c_0}^{b_k}$ 复合律；$p_{b_{k+1}}^w - p_{b_k}^w = R_{c_0}^w\,s(\bar{p}_{b_{k+1}}^{c_0} - \bar{p}_{b_k}^{c_0})$ 由 (14) 差分——外参项相消；$\mathbf{v}_{b_k}^{c_0} = R_{b_k}^{c_0}\mathbf{v}_{b_k}^{b_k}$）得 (17)：

$$\alpha_{b_{k+1}}^{b_k} = R_{c_0}^{b_k}\Big(s\big(\bar{\mathbf{p}}_{b_{k+1}}^{c_0} - \bar{\mathbf{p}}_{b_k}^{c_0}\big) + \tfrac{1}{2}\mathbf{g}^{c_0}\Delta t_k^2 - R_{b_k}^{c_0}\mathbf{v}_{b_k}^{b_k}\Delta t_k\Big), \quad \beta_{b_{k+1}}^{b_k} = R_{c_0}^{b_k}\Big(R_{b_{k+1}}^{c_0}\mathbf{v}_{b_{k+1}}^{b_{k+1}} + \mathbf{g}^{c_0}\Delta t_k - R_{b_k}^{c_0}\mathbf{v}_{b_k}^{b_k}\Big) \tag{17}$$

关键观察：**$\mathcal{X}_I$ 只以一次项出现**——把 (17) 移项（预积分量搬到左边、状态项搬到右边，并把外参平移 $p_b^c$ 按链式补进等号左侧，依据：(14) 与 (17) 的代数整理）得线性测量模型 (18)(19)：

$$\hat{z}_{b_{k+1}}^{b_k} = \begin{bmatrix} \hat{\alpha}_{b_{k+1}}^{b_k} - \mathbf{p}_b^c + R_{c_0}^{b_k}R_{b_{k+1}}^{c_0}\,\mathbf{p}_b^c \\ \hat{\beta}_{b_{k+1}}^{b_k} \end{bmatrix} = H_{b_{k+1}}^{b_k}\,\mathcal{X}_I + \mathbf{n}_{b_{k+1}}^{b_k}, \quad H_{b_{k+1}}^{b_k} = \begin{bmatrix} -I\Delta t_k & 0 & \frac{1}{2}R_{c_0}^{b_k}\Delta t_k^2 & R_{c_0}^{b_k}(\bar{\mathbf{p}}_{c_{k+1}}^{c_0} - \bar{\mathbf{p}}_{c_k}^{c_0}) \\ -I & R_{c_0}^{b_k}R_{b_{k+1}}^{c_0} & R_{c_0}^{b_k}\Delta t_k & 0 \end{bmatrix} \tag{18}$$

（$H$ 的列依次对应 $\mathbf{v}_{b_k}^{b_k}$、$\mathbf{v}_{b_{k+1}}^{b_{k+1}}$、$\mathbf{g}^{c_0}$、$s$；末列用 $s(\bar{p}_{b_{k+1}}^{c_0} - \bar{p}_{b_k}^{c_0}) = s(\bar{p}_{c_{k+1}}^{c_0} - \bar{p}_{c_k}^{c_0})$ 换成 SfM 相机位置差，依据：(14) 差分时外参项相消；$R_{c_0}^{b_k}, R_{b_{k+1}}^{c_0}$ 来自 SfM + 外参 (14)，视为已知。）全窗口堆叠后解 (20)：

$$\min_{\mathcal{X}_I} \sum_{k\in\mathcal{B}} \Big\| \hat{z}_{b_{k+1}}^{b_k} - H_{b_{k+1}}^{b_k}\mathcal{X}_I \Big\|^2 \tag{20}$$

（依据：标准线性最小二乘/正规方程，第 08 章 (8.5) 的线性特例。）这就同时得到全部速度、重力方向（$c_0$ 系）与**米制尺度**。

**第 4 步：重力精化（2 自由度切空间）**。(20) 给出的重力模长不准（视觉-惯性耦合弱），而真模长 $g \approx 9.81\,\mathrm{m/s^2}$ 已知——重力只剩切空间 2 自由度（Fig. 5）：

$$\mathbf{g} = g\cdot\hat{\bar{\mathbf{g}}} + w_1\,\mathbf{b}_1 + w_2\,\mathbf{b}_2$$

（$\hat{\bar{\mathbf{g}}}$ 为当前估计的单位方向，$\mathbf{b}_1 \perp \mathbf{b}_2$ 为切平面基，由 Algorithm 1 的叉积构造：$\hat{\bar{\mathbf{g}}} \times [1,0,0]$ 归一化，退化时换 $[0,0,1]$。）把该参数化代回 (17)–(20)，未知量从 3 维重力降为标量 $w_1, w_2$，与速度/尺度一起重解线性最小二乘，迭代直至 $\hat{\mathbf{g}}$ 收敛（§V-B-3）。**第 5 步**：把重力旋到 z 轴得到世界系定义，所有变量旋转+缩放到位，喂给紧耦合 VIO（§V-B-4）。

## 5. 实验与结果解读

- **EuRoC（§IX-A，与 OKVIS 对比，定性）**：选 MH_03_median、MH_05_difficult、V1_03_difficult（大偏置、光照变化、剧烈运动）三个序列，只用左目。对比协议：丢弃前 150 个输出以对齐真值后比较其余输出（§IX-A）。结论：VIO 部分与 OKVIS 精度"难分伯仲"（论文原话：it is hard to distinguish which one is better），但 VINS-Mono 是**完整系统**——初始化快（V1_03_difficult 得益于专用初始化流程）、带回环后平移误差最小（MH_05_difficult 误差图 Fig. 14）。论文未给 ATE 数值表（以误差随时间/里程曲线呈现），故此处不引具体数字。
- **室内长走廊（§IX-B/C-1）**：约 700 m、10 min 手持环游。OKVIS 的最终漂移在 x/y/z 轴达 $[13.80, -5.26, 7.23]$ m（占轨迹长 2.36%）；VINS-Mono **不带回环**的 VIO 漂移 $[1.547, 2.76, -0.29]$ m，已只占 0.88%；**带回环后**最终漂移 $[-0.032, 0.09, -0.07]$ m，相对轨迹长可忽略（§IX-C-1）。
- **大尺度（§IX-C-2）**：绕 HKUST 校园 5.62 km、1h34min 手持采集，关键帧数据库限 2000；与 Google 地图对齐"almost drift-free"（Fig. 20）。时延（Table I，i7-4790）：特征检测 15 ms @25 Hz、KLT 跟踪 5 ms @25 Hz、滑窗优化 50 ms @10 Hz、回环检测 100 ms、位姿图优化 130 ms。
- **真机闭环（§IX-D）**：四旋翼以 VINS-Mono 输出做位置反馈飞八字（回环关闭），OptiTrack 真值下 61.97 m 轨迹末端漂移 $[0.08, 0.09, 0.13]$ m，即 **0.29%**——定位精度直接支撑闭环控制的实证。
- **手机 AR（§IX-E）**：iPhone7 Plus 上 640×480@30 Hz + 内置 IMU，与 Google Tango 对比；Tango 局部精度更高（专用硬件），但 VINS-Mobile 在开门等挑战场景靠 4-DOF 位姿图消除漂移。

## 6. 局限与后续影响

**局限**（论文自述 + 结构性）： 单目 VIO 理论上 4 自由度不可观，若运动缺激励（匀速直线）尺度会退化——初始化与运行都依赖足够加速度激励（§V 的动机、§VII-C "weakly observable or even degenerate conditions" 的 future work 表述）； 边缘化冻结线性化点（§VI-D 自述 suboptimal，§4.5 的 inconsistency 论证）； 初始化忽略加计偏置（与重力耦合难观测，§V-B 引其文献 [34] 单独分析）； 特征级前端（KLT + 视差关键帧）在纯旋转、弱纹理下仍会退化；论文自列 future work：在线可观测性评估、稠密建图、手机级在线标定。

**后续影响**：与 ORB-SLAM3（[配套精读](./ORB-SLAM3_TRO2021.md)）并列为视觉惯性滑窗优化的两套参考实现；其"预积分因子 + 逆深度视觉因子 + 边缘化先验"的因子图配方（教程 (10.9)）成为开源 VIO 的默认模板（VINS-Fusion、OpenVINS 等后续工作沿此展开）；4-DOF 位姿图思想被广泛沿用为"可观测性裁剪位姿图"的范例。

## 7. 与本项目对照

- **教程章节**：[第 10 章 §10.2](../10_建图与系统实战.md)（(10.1)–(10.8)：预积分推导，Forster 框架）与 **§10.3**（(10.9) MAP = 本文 (22)、(10.10) IMU 因子 = 本文 (24)、(10.11) 边缘化先验 = 本文 §4.5 的构造、第 3 步尺度可观性 = 本文 §V 的动机与 §VIII 的 4-DOF 结论）；[第 08 章](../08_后端-ii图优化与-ba.md)：(8.1) 重投影误差 ↔ (25) 的针孔平面版、(8.5) 正规方程 ↔ §4.5 推导中的 $H, g$、(8.15)/(8.17) Schur 补与概率边缘化 ↔ 边缘化先验、(8.9) 流形更新；[第 09 章](../09_回环检测.md)：(9.7)/(9.8) 6-DOF 位姿图 ↔ 本文 (27)–(29) 的 4-DOF 特例。
- **代码**：[projects/slam/vins/vins.py](../../../projects/slam/vins/vins.py)（教学实现）——`build_vins_bundle` 组装 (10.9) 因子图、`ImuFactor.residual` 实现 (24) 的白化 15 维残差（复用 [preint/](../../../projects/slam/preint/preint.py) 的 `Preintegration` 增量递推与 `correct` 一阶偏置修正，即 (12)）、`ReprojectionFactor` 实现 (8.1) 的归一化平面版（$K = I$）、`ViBundle.solve` 为 SE(3) 流形 LM（Marquardt 对角阻尼 + Huber IRLS，第 08 章 (8.5)–(8.8)）、`simulate_vi_scene` 故意给 0.5 倍错误尺度初值演示尺度恢复、`align_se3`（Kabsch/Umeyama）对齐评估。
- **简化清单**（`vins.py` docstring 自述，已 Read 核实）： 路标为全局 XYZ 点而非逆深度参数化（DLT 三角化）； IMU 因子用**数值**雅可比（中心差分穿过同一回缩）、重投影为解析雅可比； 相机-IMU **外参 = 单位阵**（不在线标定）； **重力假设已知**（不估重力方向）； 固定 **4 关键帧**窗口，(10.11) 边缘化先验未实现（无先验项）。
- **健康值**（实测，以 [METRICS.md](../../../projects/slam/METRICS.md)/[DEBUG.md](../../../projects/slam/DEBUG.md) 为准）：紧耦合 ATE RMSE **2.36 cm**（初值尺度 0.5 被拉回）；逐对尺度比恢复后 **0.94–1.04**；`imu_weight=0` 的纯视觉对照尺度塌缩至 **0.38–0.42**（实测打印 [0.395, 0.389, 0.399, 0.380, 0.395, 0.416]）——尺度可观性论证的活体样本；IMU 白化残差范数在真值处应服从 $\chi_{15}$，期望 $E\|e\| \approx 3.7$（基准实测 [3.5, 3.5, 3.2]），**门限 8.0**（约 2 倍期望，超限即模型失配，DEBUG.md 表 V-1）。

## 配套阅读

- 本文 PDF：[VINS-Mono（Qin, Li & Shen, arXiv:1708.03852 / RAM 2018）](../../../papers/slam/classics/arXiv-1708.03852_VINS-Mono.pdf)
- 预积分源头：[Forster et al., IMU Preintegration（T-RO 2017）](../../../papers/slam/classics/arXiv-1512.02363_IMU-Preintegration.pdf)——本文 (6)–(13) 的完整推导所在，教程 §10.2 逐式对应；其配套精读见 [IMU-Preintegration（T-RO 2017）](./IMU-Preintegration_TRO2017.md)
- 姊妹精读：[DSO](./DSO_PAMI2018.md)（逆深度 + 边缘化 + FEJ 的纯视觉先行者，§4.3/§4.5 与本文互为印证）｜[ORB-SLAM3](./ORB-SLAM3_TRO2021.md)（另一套 VI 紧耦合系统：多地图 + Sim(3) 重定位）｜[ORB-SLAM2](./ORB-SLAM2_TRO2016.md)｜[SLAM 综述](./SLAM-Survey_TRO2016.md)
- 教程：[第 10 章](../10_建图与系统实战.md)（§10.2 预积分、§10.3 紧耦合因子图与尺度可观性）｜[第 08 章](../08_后端-ii图优化与-ba.md)（(8.15)/(8.17) Schur 补即边缘化）｜[第 09 章](../09_回环检测.md)（(9.7)–(9.8) 位姿图）
- 代码：[projects/slam/vins/](../../../projects/slam/vins/) ｜ [projects/slam/preint/](../../../projects/slam/preint/) ｜ [projects/slam/METRICS.md](../../../projects/slam/METRICS.md)（§4 尺度比、§5 ATE）｜ [projects/slam/DEBUG.md](../../../projects/slam/DEBUG.md)（表 V：vins 调试断点与 χ₁₅ 门限）
- 主题导航：[tutorials/slam/README.md](../README.md) ｜ [OVERVIEW.md](../OVERVIEW.md)

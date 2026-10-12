# 第 03 章｜高斯滤波：KF / EKF / EIF

第 02 章证明了贝叶斯滤波的正确性，也留下一个残酷事实：预测步的积分与测量步的乘积，对一般分布**没有闭式解**——精确的贝叶斯滤波不可计算（原书 2.6 节）。本章兑现 02.6 埋下的第一种精确可行情形：**线性高斯系统**。把置信锁进高斯族（单峰、用 $(\mu, \Sigma)$ 两个有限参数表示），预测与更新就都变成矩阵运算，得到卡尔曼滤波 KF；把线性换成"在均值处一阶泰勒展开"，得到扩展卡尔曼滤波 EKF；把"切线近似"换成"确定性 σ 点采样"，得到无迹卡尔曼滤波 UKF；最后把参数从矩 $(\mu,\Sigma)$ 换成对偶的信息形式 $(\Omega,\xi)$，得到信息滤波 IF 及其非线性版 EIF——它与第 11 章 GraphSLAM、第 12 章 SEIF 一脉相承。读完本章你应当能：徒手推出 KF 全部六行（含 Sherman–Morrison 求逆引理）、说出 EKF 两类误差的来源与 UKF 为何更精确、以及解释"为什么 IF 的测量更新是加法"这件事在第 12 章会变得至关重要。

> 原书锚点：第 3 章，书 29–63 页（PDF 47–81 页）。3.2.4 节 KF 推导（书 33–40 页）为全书最长的代数段落，本章**全量展开**；3.4 节 UKF 原书未给推导（仅算法），教程补充其与 EKF 的精度对比论证。

## 03.1 高斯置信与两种参数化（原书 3.1）

**① 问题场景**：第 02 章的 $\mathrm{bel}$ 是任意函数——参数无穷多。想用有限参数表示置信，同时保住"预测/更新后仍是同一族"的封闭性，让递归永不变形。

**② 解决方法**：多元高斯（原书式 3.1，即第 02 章工具 3）：

$$p(x) = \det(2\pi\Sigma)^{-\frac12}\exp\left\{-\tfrac12 (x-\mu)^T\Sigma^{-1}(x-\mu)\right\} \tag{3.1}$$

只用均值 $\mu$（$n$ 维向量）与协方差 $\Sigma$（$n \times n$ 半正定对称阵）。参数个数 $O(n^2)$，与历史长度无关。

**③ 两种参数化的对偶**：除矩参数 $(\mu, \Sigma)$ 外，同一密度还可由**正则参数**（canonical/natural parameterization）$(\Omega, \xi)$ 表示：

$$\Omega \triangleq \Sigma^{-1} \ \text{（信息矩阵 / 精度矩阵）}, \qquad \xi \triangleq \Sigma^{-1}\mu \ \text{（信息向量）}, \qquad \Sigma = \Omega^{-1},\ \mu = \Omega^{-1}\xi. \tag{3.2}$$

两套参数由双射互逆转换，**功能等价但计算性格相反**：一个参数里容易的运算在另一个里困难（03.5 将定量展示）。这对"对偶"概念是第 11、12 章的种子。

**④ 选型理由**：高斯的三点卖点——对线性变换与贝叶斯乘法**封闭**（递归不变形）、参数极简、全线性代数可实时。代价：**单峰**。跟踪问题（后验聚在真值附近的小邻域）天然契合；全局定位这类多假设问题（走廊里三扇门 → 三峰）则严重失配——第 01 章的三扇门实例在高斯世界只能坍缩成一个"平均位置"，这正是第 04 章非参数方法登场的理由。

**⑤ 边界提醒**：高斯的全部高阶矩由一、二阶矩决定（三阶及以上中心矩为零），因此 $(\mu,\Sigma)$ 不是"一阶近似"而是**完备**描述——这是"矩参数"名称的由来，也是高斯世界格外温顺的根源。

## 03.2 卡尔曼滤波：线性高斯世界的精确解（原书 3.2）

### 03.2.1 三条假设

**① 问题场景**：给第 02 章的贝叶斯滤波加什么假设，能把预测/更新变成闭式？

**② 解决方法**（原书 3.2.1）——三条，缺一不可：

1. **线性高斯转移**：$x_t = A_t x_{t-1} + B_t u_t + \varepsilon_t$，$\varepsilon_t \sim \mathcal{N}(0, R_t)$，即
$$p(x_t \mid u_t, x_{t-1}) = \det(2\pi R_t)^{-\frac12}\exp\left\{-\tfrac12 (x_t - A_tx_{t-1} - B_tu_t)^T R_t^{-1} (x_t - A_tx_{t-1} - B_tu_t)\right\} \tag{3.3}$$
（$A_t$：$n \times n$，$B_t$：$n \times m$，$R_t$：过程噪声协方差；原书式 3.2–3.4）
2. **线性高斯测量**：$z_t = C_t x_t + \delta_t$，$\delta_t \sim \mathcal{N}(0, Q_t)$，即
$$p(z_t \mid x_t) = \det(2\pi Q_t)^{-\frac12}\exp\left\{-\tfrac12 (z_t - C_tx_t)^T Q_t^{-1} (z_t - C_tx_t)\right\} \tag{3.4}$$
（$C_t$：$k \times n$，$Q_t$：测量噪声协方差；原书式 3.5–3.6）
3. **高斯初始置信**：$\mathrm{bel}(x_0) = \mathcal{N}(\mu_0, \Sigma_0)$（原书式 3.7）。

**③ 理论依据**：三者共同保证 $\mathrm{bel}(x_t)$ 在**任意时刻**都是高斯（3.2.4 的推导即证明）。直观：高斯过线性变换仍是高斯（预测步是仿射映射 + 加性高斯）；两个高斯的乘积仍是高斯（更新步是似然 × 先验）。递归链条每一步都不逃出高斯族——这就是 03.1 要的封闭性。

### 03.2.2 KF 算法（原书程序 3.1）

```text
Algorithm Kalman_filter( μ_{t-1}, Σ_{t-1}, u_t, z_t ):
    μ̄_t = A_t μ_{t-1} + B_t u_t                                  # 预测：均值
    Σ̄_t = A_t Σ_{t-1} A_t^T + R_t                                # 预测：协方差
    K_t  = Σ̄_t C_t^T (C_t Σ̄_t C_t^T + Q_t)^{-1}                  # 卡尔曼增益
    μ_t  = μ̄_t + K_t (z_t − C_t μ̄_t)                             # 更新：均值（innovation = z_t − C_t μ̄_t）
    Σ_t  = (I − K_t C_t) Σ̄_t                                     # 更新：协方差
    return μ_t, Σ_t
```

五个公式读法：预测均值走转移的确定性版本（把 $x_{t-1}$ 换成均值 $\mu_{t-1}$，噪声期望为零）；预测协方差按二次型传播再叠噪声——**不确定性经运动放大**；增益 $K_t$ 是"新息（innovation，实际测量与预测测量之差 $z_t - C_t\bar\mu_t$）的信任权重"；更新均值沿新息方向修正；更新协方差收缩。复杂度：矩阵求逆 $O(k^{2.4})$（$k$ = 测量维数）加 $O(n^2)$ 乘法（$n$ = 状态维数）；机器人学常见 $k \ll n$（一次测量只涉及少量状态），更新被 $O(n^2)$ 主导。

**增益的直观读法**（对照第 02 章习题 4 的单变量结果）：$Q$ 大（测量噪声大）→ $K$ 小 → 更信预测；$\bar\Sigma$ 大（预测不确定大）→ $K$ 大 → 更信测量；$C$ 的结构决定哪些状态分量被哪些测量分量修正。

### 03.2.3 例证与两个反直觉点（原书 3.2.3，图 3.1）

一维定位：先验高斯 → 一次测量更新后，**后验方差比两个来源（先验、测量）都小**（原书称"可能违反直觉，但这是 KF 信息集成的一般特点"）——独立高斯信息相加，精度（方差倒数）相加，收缩是必然（第 02 章习题 4 的 $\mathcal{N}(1090, 90)$ 已演示：$90 < 100 < 900$）。随后运动使分布变宽，再测量再收窄——KF 就是"收窄-摊宽"的循环，与第 01 章的定性口诀完全吻合，只是这里每一步都精确可算。

### 03.2.4 KF 完整推导（原书 3.2.4 全展开）★

**推导总策略**：一切归约为一个动作——**对二次型指数配平方**。高斯的指数是二次型；两个高斯相乘，指数相加仍是二次型；二次型的"顶点"（一阶导为零处）给出均值、"曲率"（二阶导）给出行列式的逆。积分的处理靠一个分解技巧：把指数拆成"仅含积分变量的部分"（积出常数）与"剩余部分"。

**第 1 部分：预测步（第 2–3 行）**

从第 02 章式 (2.15) 出发，代入两个高斯（原书式 3.8–3.11）：

$$\overline{\mathrm{bel}}(x_t) = \int p(x_t \mid u_t, x_{t-1})\,\mathrm{bel}(x_{t-1})\,dx_{t-1} = \eta \int \exp\{-L_t\}\, dx_{t-1}, \tag{3.5}$$

$$L_t = \underbrace{\tfrac12 (x_t - A_tx_{t-1} - B_tu_t)^T R_t^{-1} (x_t - A_tx_{t-1} - B_tu_t)}_{\text{转移密度的指数}} + \underbrace{\tfrac12 (x_{t-1} - \mu_{t-1})^T \Sigma_{t-1}^{-1} (x_{t-1} - \mu_{t-1})}_{\text{旧置信的指数}}. \tag{3.6}$$

$L_t$ 同时是 $x_{t-1}$ 与 $x_t$ 的二次型。**分解**（原书式 3.12）：找二次型 $L_{1,t}(x_{t-1}, x_t)$，使 $\int \exp\{-L_{1,t}\}dx_{t-1}$ 与 $x_t$ 无关，令 $L_{2,t}(x_t) = L_t - L_{1,t}(x_{t-1}, x_t)$，则积分把 $L_1$ 吞进归一化常数：

$$\overline{\mathrm{bel}}(x_t) = \eta \exp\{-L_{2,t}(x_t)\}. \tag{3.7}$$

如何构造 $L_{1,t}$：把 $L_t$ 看作 $x_{t-1}$ 的二次函数，对其求一、二阶导（原书式 3.16）：

$$\frac{\partial L_t}{\partial x_{t-1}} = -A_t^T R_t^{-1}(x_t - A_tx_{t-1} - B_tu_t) + \Sigma_{t-1}^{-1}(x_{t-1} - \mu_{t-1}), \qquad \frac{\partial^2 L_t}{\partial x_{t-1}^2} = A_t^T R_t^{-1}A_t + \Sigma_{t-1}^{-1}. \tag{3.8}$$

曲率记 $\Phi_t^{-1} \triangleq A_t^T R_t^{-1} A_t + \Sigma_{t-1}^{-1}$（$\Phi_t$ 即二次型在 $x_{t-1}$ 方向的"宽度"）。一阶导为零的解（原书式 3.17–3.18，逐步移项）：

$$\big(A_t^T R_t^{-1}A_t + \Sigma_{t-1}^{-1}\big) x_{t-1} = A_t^T R_t^{-1}(x_t - B_tu_t) + \Sigma_{t-1}^{-1}\mu_{t-1} \ \Longrightarrow\ x_{t-1}^{\min} = \Phi_t\big[A_t^T R_t^{-1}(x_t - B_tu_t) + \Sigma_{t-1}^{-1}\mu_{t-1}\big].$$

于是

$$L_{1,t}(x_{t-1}, x_t) = \tfrac12 \big(x_{t-1} - x_{t-1}^{\min}\big)^T \Phi_t^{-1} \big(x_{t-1} - x_{t-1}^{\min}\big). \tag{3.9}$$

它是 $x_{t-1}$ 的标准二次型，故 $\int \exp\{-L_{1,t}\}dx_{t-1} = \det(2\pi\Phi_t)^{1/2}$——**与 $x_t$ 无关**（原书式 3.20–3.22），分解成立。剩余部分 $L_{2,t} = L_t - L_{1,t}$（原书式 3.24–3.26）：展开后所有含 $x_{t-1}$ 的项**成对抵消**（这是构造 $x_{t-1}^{\min}$ 的必然结果）：

$$L_{2,t}(x_t) = \tfrac12 (x_t - B_tu_t)^T R_t^{-1}(x_t - B_tu_t) + \tfrac12 \mu_{t-1}^T\Sigma_{t-1}^{-1}\mu_{t-1} - \tfrac12 \big[A_t^TR_t^{-1}(x_t - B_tu_t) + \Sigma_{t-1}^{-1}\mu_{t-1}\big]^T \Phi_t \big[A_t^TR_t^{-1}(x_t - B_tu_t) + \Sigma_{t-1}^{-1}\mu_{t-1}\big]. \tag{3.10}$$

$L_{2,t}$ 是 $x_t$ 的二次型 → $\overline{\mathrm{bel}}(x_t)$ 是高斯。求其均值（一阶导为零）与协方差（二阶导的逆）。一阶导（原书式 3.27）：

$$\frac{\partial L_{2,t}}{\partial x_t} = \big[R_t^{-1} - R_t^{-1}A_t\Phi_t A_t^TR_t^{-1}\big](x_t - B_tu_t) - R_t^{-1}A_t\Phi_t\Sigma_{t-1}^{-1}\mu_{t-1}. \tag{3.11}$$

对第一括号用**求逆引理**（原书程序 3.2，又称 Sherman–Morrison 公式）：

> **求逆引理**：对可逆阵 $R$、$Q$ 与适维 $P$：$(R + PQP^T)^{-1} = R^{-1} - R^{-1}P\,(Q^{-1} + P^TR^{-1}P)^{-1}P^TR^{-1}$。
> **证明**（原书给出，此处照录思路）：令 $W = (Q^{-1} + P^TR^{-1}P)^{-1}$，只需验证 $(R^{-1} - R^{-1}PWP^TR^{-1})(R + PQP^T) = I$。展开 $R^{-1}R + R^{-1}PQP^T - R^{-1}PWP^TR^{-1}R - R^{-1}PWP^TR^{-1}PQP^T = I + R^{-1}P[QP^T - WP^T - WP^TR^{-1}PQP^T]$；内部括号 $= QP^T - W[Q^{-1}QP^T + P^TR^{-1}PQP^T] = QP^T - W[Q^{-1} + P^TR^{-1}P]QP^T = QP^T - QP^T = 0$。$\blacksquare$

取 $(R, Q, P) = (R_t,\ \Sigma_{t-1},\ A_t)$：$R_t^{-1} - R_t^{-1}A_t\Phi_tA_t^TR_t^{-1} = (R_t + A_t\Sigma_{t-1}A_t^T)^{-1}$，(3.11) 化为

$$\frac{\partial L_{2,t}}{\partial x_t} = (R_t + A_t\Sigma_{t-1}A_t^T)^{-1}(x_t - B_tu_t) - R_t^{-1}A_t\Phi_t\Sigma_{t-1}^{-1}\mu_{t-1}. \tag{3.12}$$

**验证均值**（比原书式 3.29–3.31 更直接的一条路）：断言极小点在 $x_t = B_tu_t + A_t\mu_{t-1}$。代回 (3.12) 左项得 $(R_t + A_t\Sigma_{t-1}A_t^T)^{-1}A_t\mu_{t-1}$；右项中 $\Phi_t\Sigma_{t-1}^{-1} = (\Sigma_{t-1}^{-1} + A_t^TR_t^{-1}A_t)^{-1}\Sigma_{t-1}^{-1} = \big(\Sigma_{t-1}(\Sigma_{t-1}^{-1} + A_t^TR_t^{-1}A_t)\big)^{-1} = (I + \Sigma_{t-1}A_t^TR_t^{-1}A_t)^{-1}$，故右项为 $R_t^{-1}A_t(I + \Sigma_{t-1}A_t^TR_t^{-1}A_t)^{-1}\mu_{t-1}$。两者相等的条件是

$$(R_t + A_t\Sigma_{t-1}A_t^T)^{-1}A_t = R_t^{-1}A_t\big(I + \Sigma_{t-1}A_t^TR_t^{-1}A_t\big)^{-1} \iff A_t\big(I + \Sigma_{t-1}A_t^TR_t^{-1}A_t\big) = (R_t + A_t\Sigma_{t-1}A_t^T)R_t^{-1}A_t,$$

右边展开 = $A_t + A_t\Sigma_{t-1}A_t^TR_t^{-1}A_t$ = 左边展开。$\blacksquare$ 于是**预测均值**

$$\bar\mu_t = A_t\mu_{t-1} + B_tu_t. \tag{3.13}$$

**协方差**：$L_{2,t}$ 的二阶导（原书式 3.32）$\frac{\partial^2 L_{2,t}}{\partial x_t^2} = (A_t^T R_t^{-1}A_t + \Sigma_{t-1}^{-1})^{-1}$，再用一次求逆引理（同上参数）：

$$\bar\Sigma_t = \big(A_t^T R_t^{-1}A_t + \Sigma_{t-1}^{-1}\big)^{-1} = R_t + A_t\Sigma_{t-1}A_t^T. \tag{3.14}$$

（注意 $\Phi_t$ 与 $\bar\Sigma_t$ 是同一个矩阵的两种身份：前者是 $x_{t-1}$ 方向的条件曲率之逆，后者是预测协方差——求逆引理正是连接"精度域"与"矩域"的桥。）

**第 2 部分：测量更新（第 4–6 行）**

从 $\mathrm{bel}(x_t) = \eta\, p(z_t \mid x_t)\,\overline{\mathrm{bel}}(x_t)$（第 02 章式 2.12）出发，代入两个高斯（原书式 3.33–3.35）：

$$\mathrm{bel}(x_t) = \eta \exp\{-J_t\}, \qquad J_t = \tfrac12 (z_t - C_tx_t)^TQ_t^{-1}(z_t - C_tx_t) + \tfrac12 (x_t - \bar\mu_t)^T\bar\Sigma_t^{-1}(x_t - \bar\mu_t). \tag{3.15}$$

这次没有积分——指数直接对 $x_t$ 求导（原书式 3.36–3.37）：

$$\frac{\partial J_t}{\partial x_t} = -C_t^TQ_t^{-1}(z_t - C_tx_t) + \bar\Sigma_t^{-1}(x_t - \bar\mu_t), \qquad \frac{\partial^2 J_t}{\partial x_t^2} = C_t^TQ_t^{-1}C_t + \bar\Sigma_t^{-1}. \tag{3.16}$$

**后验协方差**（二阶导的逆，原书式 3.38）：

$$\Sigma_t = \big(C_t^TQ_t^{-1}C_t + \bar\Sigma_t^{-1}\big)^{-1}. \tag{3.17}$$

**后验均值**：一阶导为零（原书式 3.39）。对左项做"加减 $C_t\bar\mu_t$"的恒等变形（原书式 3.40）：

$$C_t^TQ_t^{-1}(z_t - C_t\bar\mu_t) - C_t^TQ_t^{-1}C_t(\mu_t - \bar\mu_t) = \big(C_t^TQ_t^{-1}C_t + \bar\Sigma_t^{-1}\big)(\mu_t - \bar\mu_t) = \Sigma_t^{-1}(\mu_t - \bar\mu_t),$$

左乘 $\Sigma_t$（原书式 3.43）得 $\mu_t = \bar\mu_t + \Sigma_tC_t^TQ_t^{-1}(z_t - C_t\bar\mu_t)$。定义**卡尔曼增益**（矩形式）

$$K_t \triangleq \Sigma_t C_t^T Q_t^{-1}, \qquad \mu_t = \bar\mu_t + K_t(z_t - C_t\bar\mu_t). \tag{3.18}$$

**增益的另一种形式**（程序 3.1 第 4 行；原书式 3.45 的代数补全）：$\Sigma_t$ 含 $\bar\Sigma_t^{-1}$，直接算需要 $n \times n$ 求逆；换算成只含 $k \times k$ 求逆的形式。在 (3.18) 的 $K_t$ 右侧乘除同一个因子 $(C_t\bar\Sigma_tC_t^T + Q_t)^{-1}(C_t\bar\Sigma_tC_t^T + Q_t)$：

$$K_t = \Sigma_t\big(C_t^TQ_t^{-1}C_t\bar\Sigma_tC_t^T + C_t^TQ_t^{-1}Q_t\big)(C_t\bar\Sigma_tC_t^T + Q_t)^{-1} = \Sigma_t\big(C_t^TQ_t^{-1}C_t + \bar\Sigma_t^{-1}\big)\bar\Sigma_tC_t^T\,(C_t\bar\Sigma_tC_t^T + Q_t)^{-1},$$

中括号正是 $\Sigma_t^{-1}$（式 3.17），于是

$$K_t = \bar\Sigma_t C_t^T \big(C_t\bar\Sigma_tC_t^T + Q_t\big)^{-1}. \tag{3.19}$$

——只需对 $k \times k$ 的新息协方差求逆，$k \ll n$ 时是巨大的节省。**后验协方差的矩形式**（程序 3.1 第 6 行；原书式 3.46–3.47）：对 (3.17) 用求逆引理（取 $R = \bar\Sigma_t^{-1}$，$P = C_t$，$Q = Q_t$）：

$$\Sigma_t = \bar\Sigma_t - \bar\Sigma_tC_t^T\big(Q_t + C_t\bar\Sigma_tC_t^T\big)^{-1}C_t\bar\Sigma_t = \bar\Sigma_t - K_t\big(C_t\bar\Sigma_tC_t^T + Q_t\big)\big(C_t\bar\Sigma_tC_t^T + Q_t\big)^{-1}C_t\bar\Sigma_t = \big(I - K_tC_t\big)\bar\Sigma_t. \tag{3.20}$$

（优点：避免对 $\Sigma_t$ 的显式 $n \times n$ 求逆——高维状态空间下重要，原书特别强调。）KF 全部六行证毕。$\blacksquare$

## 03.3 扩展卡尔曼滤波：一阶泰勒近似（原书 3.3）

**① 问题场景**：真实机器人几乎不满足线性——常速 $(v, \omega)$ 的运动轨迹是圆弧（非线性 $g$），路标观测是距离/方位角（非线性 $h$）。高斯过**非线性**变换不再高斯（原书图 3.2b：对 $p(x)$ 采 50 万样本过 $g$ 画直方图，形状被扭曲）——(3.3)(3.4) 的前提被打破，贝叶斯滤波无闭式解。

**② 解决方法**：EKF 保留高斯表示，把非线性函数在"当前最可能的状态"处**一阶泰勒展开**成切线，然后原封不动套用 KF 五公式：

$$g(u_t, x_{t-1}) \approx g(u_t, \mu_{t-1}) + G_t(x_{t-1} - \mu_{t-1}), \qquad G_t \triangleq \left.\frac{\partial g}{\partial x_{t-1}}\right|_{u_t, \mu_{t-1}} \ \ (\text{雅可比},\ n\times n), \tag{3.21}$$

$$h(x_t) \approx h(\bar\mu_t) + H_t(x_t - \bar\mu_t), \qquad H_t \triangleq \left.\frac{\partial h}{\partial x_t}\right|_{\bar\mu_t} \ \ (k \times n). \tag{3.22}$$

**展开点为何选均值**：泰勒展开点可以任选，但线性化误差由展开点与"分布实际散开的区域"的失配决定——高斯的最可能状态是均值，分布质量集中在其邻域，故在 $\mu_{t-1}$（预测）与 $\bar\mu_t$（更新）处展开是局部误差最小的选择（原书明示这一逻辑）。

**③ 算法**（原书程序 3.3）：把 KF 五公式中的 $A_t\mu_{t-1} + B_tu_t$ 换成 $g(u_t, \mu_{t-1})$、$C_t\bar\mu_t$ 换成 $h(\bar\mu_t)$、$A_t$ 换成 $G_t$、$C_t$ 换成 $H_t$：

$$\bar\mu_t = g(u_t, \mu_{t-1}), \quad \bar\Sigma_t = G_t\Sigma_{t-1}G_t^T + R_t, \quad K_t = \bar\Sigma_tH_t^T(H_t\bar\Sigma_tH_t^T + Q_t)^{-1}, \quad \mu_t = \bar\mu_t + K_t(z_t - h(\bar\mu_t)), \quad \Sigma_t = (I - K_tH_t)\bar\Sigma_t. \tag{3.23}$$

**④ 推导为何照搬**：把 (3.21)(3.22) 的线性化模型当作"局部成立的线性高斯模型"，02.4/03.2.4 的推导逐行适用（原书 3.3.4 给出对应式 3.55–3.64；预测部分多出一个 $\Phi_t = (G_t^TR_t^{-1}G_t + \Sigma_{t-1}^{-1})^{-1}$ 的中变量，结论同构）。唯一实质差别：线性化点每时刻不同，$G_t, H_t$ 是时变的。

**⑤ 两类误差与实践准则**（原书 3.3.5，图 3.4–3.5）：

1. **不确定性越宽，误差越大**（图 3.4）：展开点固定在均值，分布越宽、"扫过"的 $g$ 区间越长，切线失真越大——蒙特卡罗真均值/方差与线性化结果的偏移随先验宽度增大。结论：**实践中保持估计的小不确定性**（频繁测量、及时校正）。
2. **局部非线性越强，误差越大**（图 3.5）：均值落在 $g$ 弯曲剧烈的区域时切线失真大。
3. EKF 系统性地**低估**传播后的不确定性（切线"抹平"了弯曲带来的扩散）。
4. 复杂度与 KF 同阶 $O(k^{2.4} + n^2)$——EKF 流行的根本原因。千维以上状态空间的应用（本书后续的建图）也能跑。
5. **多峰不可表示**：两个假设的均值之"算术平均"可能是哪个假设都不在的鬼位置。补救：高斯混合 $bel = \sum_i w_i \mathcal{N}(\mu_i, \Sigma_i)$（原书式 3.65），即**多假设 EKF**（MHEKF）——第 07 章多假设跟踪（MHT）的表示基础。

## 03.4 无迹卡尔曼滤波：确定性 σ 点采样（原书 3.4）

**① 问题场景**：EKF 的误差来自"只用切线（一阶）"。能不能不显式求导、又拿到二阶精度？

**② 解决方法——无迹变换**：不用解析线性化，而从当前高斯中取 $2n+1$ 个**确定性** σ 点（sigma points），全部过非线性函数 $g$，再用映射后的点重算均值与协方差（矩匹配）：

$$X^{[0]} = \mu, \quad X^{[i]} = \mu + \big(\sqrt{(n+\lambda)\Sigma}\,\big)_i, \quad X^{[i+n]} = \mu - \big(\sqrt{(n+\lambda)\Sigma}\,\big)_i \quad (i = 1, \dots, n), \tag{3.24}$$

$$\lambda = \alpha^2(n + \kappa) - n; \qquad w_m^{[0]} = \frac{\lambda}{n+\lambda}, \quad w_c^{[0]} = \frac{\lambda}{n+\lambda} + (1 - \alpha^2 + \beta), \quad w_m^{[i]} = w_c^{[i]} = \frac{1}{2(n+\lambda)}. \tag{3.25}$$

（$(\sqrt{(n+\lambda)\Sigma})_i$ 是矩阵平方根的第 $i$ 列，由 Cholesky 分解得到；$\alpha, \kappa$ 控制 σ 点离均值的距离，$\beta$ 编码高阶分布信息、精确高斯下最优取 $\beta = 2$；权重非负和为 1，$w_c^{[0]}$ 可为负。）映射后重算矩（原书式 3.68–3.69）：

$$Y^{[i]} = g(X^{[i]}), \qquad \mu' = \sum_{i=0}^{2n} w_m^{[i]} Y^{[i]}, \qquad \Sigma' = \sum_{i=0}^{2n} w_c^{[i]} \big(Y^{[i]} - \mu'\big)\big(Y^{[i]} - \mu'\big)^T. \tag{3.26}$$

**③ 精度论证**（原书只给结论，此处补理由）：σ 点是对称设计的，其一阶矩、二阶矩恰好还原 $(\mu, \Sigma)$；过 $g$ 后的矩匹配能捕捉到 $g$ 的**二阶**信息（曲率项通过对称点的均值化抵消三阶、保留二阶），而 EKF 只有一阶——原书结论"无迹变换与泰勒展开前两项同精度"即此意（图 3.7–3.8 的蒙特卡罗对比显示 UKF 虚线更贴近真分布）。

**④ 算法结构**（原书程序 3.4，13 行）：预测——旧置信取 σ 点 → 过 $g$ → 重算 $(\bar\mu, \bar\Sigma)$ → **叠加 $R_t$**；更新——用预测高斯重新取 σ 点 → 过 $h$ → 新息协方差 $S_t = \sum w_c (\cdot) + Q_t$（对应 EKF 的 $H\bar\Sigma H^T + Q$）→ **互协方差** $\bar\Sigma_t^{xz} = \sum w_c (X - \bar\mu)(Z - \hat z)^T$（对应 EKF 的 $\bar\Sigma H^T$）→ $K_t = \bar\Sigma^{xz}S_t^{-1}$ → $\mu_t = \bar\mu + K(z - \hat z)$，$\Sigma_t = \bar\Sigma - KS_tK^T$。结构上与 EKF 逐项对应，只是每个"雅可比乘积"换成了"σ 点加权统计"。

**⑤ 选型理由与边界**：

| | EKF | UKF |
|---|---|---|
| 精度 | 一阶（泰勒） | 二阶（σ 点） |
| 需要 $g, h$ 可导？ | 需要（雅可比） | **不需要**（免求导 / derivative-free，$g,h$ 可以是黑箱程序） |
| 复杂度 | $O(k^{2.4}+n^2)$ | 同阶，常数因子略大 |
| 线性系统下 | 精确 | 与 KF 完全一致 |
| 多峰/强非高斯 | 不能 | **同样不能**（仍是单一高斯） |

UKF 与粒子滤波的相似（都用"点集 + 权重"表示分布）与本质区别：σ 点**确定性**、个数 $2n+1$ 固定，样本**随机**、个数可调。近高斯情形 UKF 远比粒子滤波高效；强非高斯（多峰）情形两者立场反转。实践中许多应用 EKF 与 UKF 差异不大（原书原话）——先 EKF 后升级是合理的默认路线。

## 03.5 信息滤波：对偶表示（原书 3.5）

### 03.5.1 正则参数：从指数展开读出 $(\Omega, \xi)$

**① 问题场景**：$(\mu, \Sigma)$ 里"测量更新"难（需求逆），有没有一种参数让乘测量这步变成加法？

**② 解决方法**：把高斯密度指数按 $x$ 的幂次重排（原书式 3.74–3.77 的展开补全）：

$$p(x) \propto \exp\left\{-\tfrac12 (x-\mu)^T\Sigma^{-1}(x-\mu)\right\} = \exp\left\{-\tfrac12 x^T\Sigma^{-1}x + x^T\Sigma^{-1}\mu - \tfrac12 \mu^T\Sigma^{-1}\mu\right\},$$

末项与 $x$ 无关，吸入归一化（原书式 3.76）：

$$p(x) = \eta \exp\left\{-\tfrac12 x^T\,\Omega\,x + \xi^T x\right\}. \tag{3.27}$$

读出二次项系数 $\Omega = \Sigma^{-1}$、一次项系数 $\xi = \Sigma^{-1}\mu$——正则参数就是"指数二次型的系数表"。负对数（原书式 3.78）：

$$-\log p(x) = \text{const} + \tfrac12 x^T\Omega x - \xi^T x. \tag{3.28}$$

一阶导为零给均值 $\mu = \Omega^{-1}\xi$（原书式 3.79）；$\Omega$ 加权的二次型即**马氏距离**（Mahalanobis distance）——第 07 章数据关联的打分函数。$\Omega \succ 0$ 时 $-\log p$ 是凸的距离函数：**高斯 = 信息空间里的一个抛物面**。

### 03.5.2 IF 算法与推导（原书 3.5.2–3.5.3）

**算法**（原书程序 3.5；线性模型同 (3.3)(3.4)）：

```text
Algorithm Information_filter( Ω_{t-1}, ξ_{t-1}, u_t, z_t ):
    D_t  = (A_t Ω_{t-1}^{-1} A_t^T + R_t)^{-1}          # 预测的"信息协方差"（= Σ̄_t）
    ξ̄_t  = D_t (A_t Ω_{t-1} ξ_{t-1} + B_t u_t)          # 预测信息向量
    Ω_t  = C_t^T Q_t^{-1} C_t + D_t                     # 更新：信息矩阵相加！
    ξ_t  = C_t^T Q_t^{-1} z_t + ξ̄_t                     # 更新：信息向量相加！
    return ξ_t, Ω_t
```

**更新步为何是加法**（推导，原书式 3.88–3.92）：更新 $= \eta\,p(z_t|x_t)\,\overline{\mathrm{bel}}(x_t)$，两个高斯指数相加——在正则形式下指数相加就是**系数相加**：

$$\mathrm{bel}(x_t) = \eta \exp\left\{-\tfrac12 x^T\big[\underbrace{C_t^TQ_t^{-1}C_t + \bar\Omega_t}_{\Omega_t}\big]x + x^T\big[\underbrace{C_t^TQ_t^{-1}z_t + \bar\xi_t}_{\xi_t}\big]\right\}. \tag{3.29}$$

每个测量贡献一个秩-$k$ 的信息块 $C^TQ^{-1}C$ 与信息向量 $C^TQ^{-1}z$，**直接累加**——多传感器融合、多机器人分散融合、任意顺序/任意时延地收信息，在正则形式下都是加法（矩形式下 Bayes 乘法的对数版本，IF 把对数运算显式化了）。**预测步为何变难**（原书式 3.86–3.87）：把 KF 预测 (3.13)(3.14) 中的 $\mu, \Sigma$ 用 (3.2) 反换代入：

$$\Omega_t = \big(A_t\Omega_{t-1}^{-1}A_t^T + R_t\big)^{-1}, \qquad \xi_t = \Omega_t\big(A_t\Omega_{t-1}^{-1}\xi_{t-1} + B_tu_t\big). \tag{3.30}$$

——嵌套两个 $n \times n$ 求逆。**复杂度对偶表**：

| 步骤 | KF（矩参数） | IF（正则参数） |
|------|----------------|-----------------|
| 预测 | $O(n^2)$ 乘法，增量友好 | 两个 $n\times n$ 求逆，$O(n^{2.4})$ |
| 测量更新 | $k \times k$ 求逆 + $O(n^2)$ | **加法**，$O(n^2)$，局部信息可直接叠加 |

（"哪个环节频繁"决定选谁：多数机器人在线问题运动频繁 → KF；信息来自分散来源、局部测量 → IF。）

### 03.5.3 EIF：非线性版与代价（原书 3.5.4–3.5.6）

非线性 $x_t = g(u_t, x_{t-1}) + \varepsilon_t$、$z_t = h(x_t) + \delta_t$ 下，线性化方式与 EKF 相同（3.21–3.22）。EIF（原书程序 3.6）的尴尬在于：$g, h$ 需要在**均值**处展开，而正则参数里没有均值——每步都要**恢复状态估计** $\mu = \Omega^{-1}\xi$（程序第 2、5 行）。预测与更新公式（原书式 3.101–3.102、3.106–3.107）：

$$\bar\mu_t = g(u_t, \Omega_{t-1}^{-1}\xi_{t-1}), \qquad \bar\Omega_t = \big(G_t\Omega_{t-1}^{-1}G_t^T + R_t\big)^{-1}, \qquad \bar\xi_t = \bar\Omega_t\,\bar\mu_t; \tag{3.31}$$

$$\Omega_t = \bar\Omega_t + H_t^TQ_t^{-1}H_t, \qquad \xi_t = \bar\xi_t + H_t^TQ_t^{-1}\big[z_t - h(\bar\mu_t) + H_t\bar\mu_t\big]. \tag{3.32}$$

（(3.32) 由把 (3.29) 的 $C_t \to H_t$、$z_t \to z_t - h(\bar\mu_t) + H_t\bar\mu_t$ 直接读出——线性化把测量方程凑回"等效线性 + 等效观测"的形式。）**实践盘点**（原书 3.5.6）：

1. IF 表示"全局不确定"极简单：$\Omega = 0$（对应 $\Sigma = \infty$——矩参数下写不出）；
2. 更新的可加性使 IF 在"信息分散采集、局部叠加"的大型估计问题中天然占优；
3. 多机器人融合：贝叶斯准则在**对数域**是加法，正则参数即对数域参数——任意时延、任意顺序融合；
4. 代价：每次预测/更新都要求逆恢复 $\mu$——高维问题 EKF 通常快于 EIF，这是 EIF 远不如 EKF 流行的主因；
5. **稀疏性只属于 $\Omega$**：$\Omega$ 的非对角零元 ↔ 状态间无条件依赖 ↔ 高斯马尔可夫随机场（Markov random field）的图结构；$\Sigma$ 不具备此性质（近满的 $\Sigma$ 可对应稀疏 $\Omega$，原书习题 6）。大型建图问题中状态交互是局部的 → $\Omega$ 稀疏 → 循环置信传播等高效算法。**这是第 11 章 GraphSLAM 与第 12 章 SEIF 的全部出发点。**

## 03.6 四滤波器总览与选型速查

| | KF | EKF | UKF | IF / EIF |
|---|----|-----|-----|----------|
| 假设 | 线性 + 高斯噪声 + 高斯先验 | 同左，$g,h$ 放宽为非线性（一阶切线） | 同左（σ 点统计线性化） | 同 KF / 同 EKF |
| 线性化 | 无需 | 雅可比 | 免导 | 雅可比 + 每步恢复 $\mu$ |
| 精度 | 精确 | 一阶 | 二阶 | 同 EKF（EIF） |
| 预测 / 更新成本 | $O(n^2)$ / $O(k^{2.4}+n^2)$ | 同 KF | 同 KF（常数略大） | 求逆 / **加法** |
| 多峰 | ✗ | ✗（混合高斯补） | ✗ | ✗ |
| 机器人学角色 | 教科书基线 | **默认工作马**（ch7 定位、ch10 SLAM） | 雅可比难写时替身 | ch12 SEIF、多机器人 |
| 本教程后续章 | 03.2 | 07（EKF 定位）、10（EKF SLAM） | 07.7 | 11、12 |

## 03.7 小结、术语表与书后习题全解

**要点回顾**：

1. 高斯表示 = $O(n^2)$ 参数 + 封闭性（线性变换、贝叶斯乘法），代价是单峰（03.1）；
2. 三条线性高斯假设下贝叶斯滤波有精确闭式解 = KF；推导全靠"二次型配平方 + 求逆引理"，预测均值/协方差与更新五公式逐一验证（03.2.4）；
3. EKF = 在均值处的一阶泰勒；误差随**不确定性宽度**与**局部非线性强度**增大，且系统性低估不确定性；实践准则是"保持估计紧凑"（03.3）；
4. UKF = 确定性 σ 点 + 矩匹配，二阶精度、免求导，线性情形退化为 KF（03.4）；
5. 正则参数是矩参数的对偶：IF 更新为加法（信息可局部累加、多机器人友好）、预测需双重求逆；EIF 的"每步恢复均值"是它不如 EKF 流行的主因；稀疏性只属于 $\Omega$——GraphSLAM/SEIF 的源头（03.5）。

**术语表（本章新增）**：矩参数（moments parameterization）、正则/本质参数（canonical/natural parameterization）、信息矩阵 / 精度矩阵（information / precision matrix）、信息向量（information vector）、线性高斯系统（linear Gaussian system）、卡尔曼增益（Kalman gain）、新息（innovation）、求逆引理 / Sherman–Morrison 公式（inversion lemma）、雅可比矩阵（Jacobian）、线性化（linearization）、无迹变换 / σ 点（unscented transform / sigma points）、免求导滤波器（derivative-free filter）、马氏距离（Mahalanobis distance）、高斯马尔可夫随机场（Gaussian Markov random field）、多假设 EKF（MHEKF）。

**书后习题全解指导**（原书 3.8 节，书 62–63 页）：

**习题 1（匀加速汽车：预测步实操）**——(a) 马尔可夫最小状态：$(x, \dot x)$（下一位置由当前位置与速度决定，缺一不可）。(b) $\Delta t = 1$ 的双积分器：
$$A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}, \qquad B = \begin{pmatrix} 1/2 \\ 1 \end{pmatrix},$$
加速度 $u_t \sim \mathcal{N}(0,1)$ 作为随机控制，预测协方差增长项为 $B\,\mathrm{Var}(u)\,B^T = \begin{pmatrix} 0.25 & 0.5 \\ 0.5 & 1\end{pmatrix}$（即 $R$ 的角色）。(c) $\mu_0 = (0,0)^T$、$\Sigma_0 = 0$ 起步，纯预测 $\Sigma_t = A\Sigma_{t-1}A^T + BB^T$：$\Sigma_1 = \begin{pmatrix} 0.25 & 0.5 \\ 0.5 & 1 \end{pmatrix}$，$\Sigma_2 = \begin{pmatrix} 2.5 & 2 \\ 2 & 2 \end{pmatrix}$，依此递推到 $t=5$（均值恒为零——$E[u]=0$）；(d) 不确定椭圆由 $\Sigma_t$ 的特征方向拉伸（特征值 = 半轴长平方）；(e) $t \to \infty$ 时相关系数**快速收敛到 $\sqrt3/2 \approx 0.866$**（闭式：$A^kB = (k+\tfrac12,\ 1)^T$，故 $\Sigma_{xx} \sim t^3/3$、$\Sigma_{x\dot x} \sim t^2/2$、$\Sigma_{\dot x\dot x} = t$，比值 $\to \tfrac{1/2}{\sqrt{1/3}}$；$t=1$ 时恰为 1——$BB^T$ 秩一、完全相关，随后回落到稳态值 0.866）。结论：位置与速度误差**强正相关**——"位置偏了多半是速度也偏了"，这正是里程计漂移必须靠外部测量纠正的定量注脚。

**习题 2（加测量更新）**——(a) 只测 $x$：$C = (1\ 0)$，$Q = (10)$。(b) 接习题 1 递推得 $\Sigma_5 = \begin{pmatrix} 41.25 & 12.5 \\ 12.5 & 5 \end{pmatrix}$、$\bar\mu_5 = 0$；$z = 5$ 时新息方差 $S = 41.25 + 10 = 51.25$，$K = \Sigma_5C^T/S = (0.805,\ 0.244)^T$，后验
$$\mu_5 = K \cdot 5 = \begin{pmatrix} 4.02 \\ 1.22 \end{pmatrix}, \qquad \Sigma_5^+ = (I - KC)\Sigma_5 = \begin{pmatrix} 8.05 & 2.44 \\ 2.44 & 1.95 \end{pmatrix}.$$
画椭圆对比：速度方向的不确定性也被测量**连带压缩**了（$12.5 \to 2.44$ 的交叉项削弱）——单分量测量通过协方差结构惠及全状态，这是"协方差是整体账本"的具体演示。

**习题 3（用卷积定理推预测步）**——预测积分 (3.5) 是两个高斯密度的卷积；傅里叶域中卷积 ↔ 频谱相乘，高斯的特征函数仍是高斯，指数二次型相加后逆变换即得 $\mathcal{N}(A\mu + Bu,\ A\Sigma A^T + R)$。方法要点：特征函数 $\varphi_X(\omega) = \exp\{j\omega^T\mu - \tfrac12\omega^T\Sigma\omega\}$；$x_{t-1} \to x_t$ 的线性映射把 $\omega$ 作对偶变换 $A^T\omega$。本题是"频域视角"的练习，正文路线（配平方）与它等价。

**习题 4（EKF 线性化到底差多少）**——本教程强烈建议动手做。设定：$x, y$ 精确已知（方差 0）、$\theta \sim \mathcal{N}(0, 0.01)$，机器人向前走 $d_1$：$g(u, x) = (x + d\cos\theta,\ y + d\sin\theta,\ \theta)$。(a) 直觉后验：把 $\theta$ 的分布"画"到圆弧 $(d\cos\theta, d\sin\theta)$ 上——一条**香蕉形/弧形**密度，径向薄、切向长。(b) EKF 预测：雅可比 $G = \begin{pmatrix} 1 & 0 & -d\sin\bar\theta \\ 0 & 1 & d\cos\bar\theta \\ 0 & 0 & 1\end{pmatrix}$，$\bar\theta = 0$ 时 $G\Sigma G^T$ 在 $x\text{-}y$ 平面投影为 $\begin{pmatrix} 0 & 0 \\ 0 & 0.01d^2 \end{pmatrix}$——一条**竖直线段**（$x$ 方向零方差！）。(c)(e) 对比：弧形真后验 vs 直线近似，差异显著且 EKF 完全丢掉 $x$ 方向的展宽；减小初始 $\theta$ 不确定性可改善；反之若**已知 $\theta$、未知 $y$**，则 $g$ 对 $(x,y)$ 恰好线性 → EKF **精确**。本题是"非线性 + 大不确定性 = 灾难；非线性 + 小不确定性 = 可用"的最小可复现实例，直接呼应 03.3 ⑤。

**习题 5（常数项扩展）**——把 (3.2) 写成 $x_t = A_tx_{t-1} + B_tu_t + c_t$：预测均值多加 $c_t$（第 2 行 $\bar\mu_t = A\mu + Bu + c$），协方差与增益不受影响；测量侧同理（$z_t = Cx + d_t$ 时新息改为 $z_t - C\bar\mu_t - d_t$）。练习点：常量只进均值账本。

**习题 6（稀疏信息矩阵也能强相关）**——构造路径图 $P_d$（$d$ 个节点的一条链）的高斯马尔可夫随机场：
$$\Omega = \mathrm{Lap}(P_d) + \varepsilon I = \begin{pmatrix} 1{+}\varepsilon & -1 & & \\ -1 & 2{+}\varepsilon & -1 & \\ & \ddots & \ddots & \ddots \\ & & -1 & 1{+}\varepsilon \end{pmatrix}$$
（对角 = 节点度数 $+\varepsilon$：端点 1、内部 2——**别写成处处 2**，否则行和不为零、常数向量不在零空间，结论失效；本教程数值验证踩过此坑）。$\Omega$ 三对角、$O(d)$ 个非零元，天然稀疏。$\varepsilon = 0$ 时 $\Omega$ 奇异（零空间 = 常数向量——"整体平移"无锚定）；$\varepsilon \to 0^+$ 时谱分解给出 $\Sigma = \Omega^{-1} \approx \mathrm{Lap}^{+} + \tfrac{1}{\varepsilon d}\mathbf{1}\mathbf{1}^T$：公共分量方差 $\sim 1/\varepsilon$ 压倒各向异性部分，任意 $i \ne j$ 的 $\mathrm{Corr}(x_i, x_j) \to 1$（数值验证：$d = 8$、$\varepsilon = 10^{-4}$ 时最小两两相关已达 0.997）。结论：**信息矩阵稀疏 $\ne$ 变量间弱相关**——一条链加一个弱锚就足以让全变量几乎完全相关。这正是第 11/12 章敢于用稀疏信息结构表示大地图的合法性来源（回环 = 链条闭合成环）。

---

**下一篇**：[第 04 章 非参数滤波：直方图与粒子](04_非参数滤波_直方图与粒子.md)——挣脱单峰枷锁：网格化与采样化两条近似路线，重要性采样与重采样的完整推导。

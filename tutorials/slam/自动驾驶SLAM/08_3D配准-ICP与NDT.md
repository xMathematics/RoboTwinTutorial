# 第 8 讲｜3D 配准：ICP 与 NDT

> 对应：书 §7.1–7.2（p.245–266）｜大纲：[00_自动驾驶SLAM技术教程大纲.md](../00_自动驾驶SLAM技术教程大纲.md) §讲 8
> 本章推导产出：**D8-1**（3D ICP 的 SVD 闭式解）、**D8-2**（点到面 ICP 的
> 扰动雅可比）、**D8-3**（NDT 目标函数、加权残差与高斯退化防护）。
> 公式 (8.1)–(8.11)。§7.1 的雷达器件科普部分不设推导（综述豁免）。
> 代码对象：`src/ch7/icp_3d.{h,cc}`（`sad::Icp3d`）、`src/ch7/ndt_3d.{h,cc}`
> （`sad::Ndt3d`）、`src/ch7/test/test_icp.cc`、仿真数据 `gen_simu_data`。

## 8.1 本章定位

第 7 讲在 SE(2) 上把配准做完了；本讲升维到 SE(3)。升维不是简单照搬：
① SVD 闭式解多出**反射检查**（2D 也有但 3D 更隐蔽）；② 残差形式从
点到线/面升级，法向估计变成 3D 邻域的 PCA；③ 出现了激光配准的第二个
大范式——**NDT**，它不找对应点、直接对目标做概率建模，与 ICP 形成
方法论上的对极。第 9 讲的两种里程计（直接法/特征法）分别架在 NDT 与
点到线/面 ICP 上。

## 8.2 学习目标

1. **独立复现 D8-1 全链**：目标函数展开 → 平移消元 → 正交 Procrustes →
   反射检查；
2. 复现 D8-2：左扰动约定下点到面残差的雅可比，并说清为什么配准残差
   用**左**扰动（呼应第 2 讲 D2-3 的选型理由）；
3. 复现 D8-3：从高斯负对数似然写出 NDT 的 GN 目标，解释退化防护与
   邻域插值两个工程项；
4. 用 `test_icp.cc` 与 PCL 对比三种 ICP 变体与 NDT 的精度/耗时。

## 8.3 问题场景（五步法①）

**手头**：同一辆车的两帧 3D 扫描（源 $\mathcal{S}$ 与目标 $\mathcal{T}$，各数万点），
初值位姿 $T_0$（来自 IMU/里程计，误差分米级）。**想要**：把源配准到目标的
相对位姿 $T$，精度厘米级。**缺**：3D 点云没有自然的对应关系（第 7 讲的
最近邻在 3D 更不可靠：密度不均、遮挡边界多），且 SE(3) 的求解不能再靠
拍脑袋的参数化。

## 8.4 核心内容

### 8.4.1 多线激光雷达（书 §7.1，综述豁免）

机械旋转式（§7.1.1）：16/32/64 线绕竖轴 360° 旋转，点云自带 ring/水平角
结构——第 6 讲 (6.1) 模型与距离图直接适用（Velodyne 系列的 Packet 解析见
`tools/pointcloud_convert`）。固态式（§7.1.2，Livox 类）：振镜非重复扫描，
同一区域密度随积分时间增长、视场约 70°、**没有 ring**——特征提取与去畸变
都要改按时间分块处理（第 9 讲 §7.4 的差异点）。配准算法本身与器件解耦，
只要求输入已转换到统一雷达系或 IMU 系的点云（`common/point_types.h`）。

### 8.4.2 D8-1：点到点 ICP 的 SVD 闭式解（书 §7.2.1）

**② 解决方法**：给定对应 $\{(\mathbf{p}_i \in \mathcal{S}, \mathbf{q}_i \in \mathcal{T})\}$
（初值 $T_0$ 下最近邻匹配，`Icp3d::AlignP2P` 内部建目标 Kd 树），解

$$\min_{R\in SO(3),\ \mathbf{t}} \sum_i \|\,R\mathbf{p}_i + \mathbf{t} - \mathbf{q}_i\,\|^2 \tag{8.1}$$

**④ 依据**：正交 Procrustes 问题 + 谱定理。

**⑤ 推导**。第一步（平移消元）：展开 (8.1)，对 $\mathbf{t}$ 求导置零得
$\mathbf{t}^* = \bar{\mathbf{q}} - R\bar{\mathbf{p}}$（$\bar{\cdot}$ 为质心），代回后目标变为
$\sum\|R(\mathbf{p}_i - \bar{\mathbf{p}}) - (\mathbf{q}_i - \bar{\mathbf{q}})\|^2$——**配准只看去质心
点集**。第二步（化为迹问题）：

$$\sum_i \|R\mathbf{p}_i' - \mathbf{q}_i'\|^2 = \sum_i\big(\|\mathbf{p}_i'\|^2 + \|\mathbf{q}_i'\|^2\big) - 2\,\mathrm{tr}\Big(R\,\underbrace{\textstyle\sum_i \mathbf{p}_i'\mathbf{q}_i'^\top}_{H}\Big) \tag{8.2}$$

（$\mathbf{p}' = \mathbf{p} - \bar{\mathbf{p}}$；迹的循环性质把二次型聚成 $\mathrm{tr}(RH)$，
依据：$\|R\mathbf{p}' - \mathbf{q}'\|^2 = \|\mathbf{p}'\|^2 + \|\mathbf{q}'\|^2 - 2\mathbf{q}'^\top R\mathbf{p}'$，
末项求和即 $\mathrm{tr}(RH)$。）第三步（Procrustes）：最大化 $\mathrm{tr}(RH)$
s.t. $R \in SO(3)$。设 H 的 SVD 为 $H = U\Sigma V^\top$，由迹循环性与正交阵
谱范数 ≤ 1 的性质（Frobenius 内积视角），

$$\boxed{\ R^* = V\,\mathrm{diag}\big(1,\ 1,\ \det(VU^\top)\big)\,U^\top, \qquad
\mathbf{t}^* = \bar{\mathbf{q}} - R^*\bar{\mathbf{p}}\ } \tag{8.3}$$

对角阵把 $\det(VU^\top) = -1$（反射解）修正回 $SO(3)$——**没有这一步，
镜像点云会被当成合法解**，对应失败时 (8.3) 恰恰经常落入反射分支。
GN 迭代版（书中同时实现）：残差 $\mathbf{e}_i = R\mathbf{p}_i + \mathbf{t} - \mathbf{q}_i$，
雅可比由 (2.8) 直接给出 $\mathbf{J}_i = \big[\,-(R\mathbf{p}_i)^\wedge,\ I\,\big]_{3\times6}$，
外层循环"最近邻 → GN 一步"交替（同第 7 讲流程）。逐点对应法的天花板
在第 7 讲已分析：收敛域小、切向退化——3D 下由下面两种残差形式缓解。

### 8.4.3 D8-2：点到线与点到面 ICP（书 §7.2.2）

**② 解决方法**：把残差投影到目标局部结构的法向/切向上。点到面残差
（$\mathbf{q}_i$ 处法向 $\mathbf{n}_i$，由第 6 讲 (6.5) 的 3D 版本 PCA 求得）：

$$\mathbf{e}_i = \mathbf{n}_i^\top\big(T\,\mathbf{p}_i - \mathbf{q}_i\big) \tag{8.4}$$

**⑤ 推导（D8-2，左扰动）**。配准残差定义在目标系（世界侧），按 D2-3 的
选型理由取**左扰动** $T \leftarrow \mathrm{Exp}(\delta\boldsymbol\xi^\wedge)T$，$\delta\boldsymbol\xi = (\delta\boldsymbol\theta, \delta\mathbf{t})$：

$$\mathrm{Exp}(\delta\boldsymbol\theta^\wedge)(R\mathbf{p}_i + \mathbf{t})
= R\mathbf{p}_i + \mathbf{t} + \delta\boldsymbol\theta\times(R\mathbf{p}_i) + \delta\mathbf{t}
= T\mathbf{p}_i - (R\mathbf{p}_i)^\wedge\delta\boldsymbol\theta + \delta\mathbf{t}$$

（第二步用 $\delta\boldsymbol\theta\times\mathbf{x} = \delta\boldsymbol\theta^\wedge\mathbf{x} = -\mathbf{x}^\wedge\delta\boldsymbol\theta$。）
代回 (8.4) 取一阶：

$$\mathbf{e}_i(\delta\boldsymbol\xi) \approx \mathbf{e}_i
+ \underbrace{\Big[\ -\mathbf{n}_i^\top(R\mathbf{p}_i)^\wedge\ \ \ \mathbf{n}_i^\top\ \Big]}_{\mathbf{J}_i\ (1\times6)}\delta\boldsymbol\xi
= \mathbf{e}_i + \big[(\mathbf{n}_i\times R\mathbf{p}_i)^\top,\ \mathbf{n}_i^\top\big]\delta\boldsymbol\xi \tag{8.5}$$

（末步用混合积轮换 $\mathbf{n}^\top(\mathbf{x}^\wedge\delta\boldsymbol\theta) = (\mathbf{n}\times\mathbf{x})^\top\delta\boldsymbol\theta$。）
点到线同法：残差 $\mathbf{e}_i = \|(I - \mathbf{d}_i\mathbf{d}_i^\top)(T\mathbf{p}_i - \mathbf{q}_i)\|$
（$\mathbf{d}_i$ 为直线方向，投影阵把沿方向分量去掉），雅可比把 (8.5) 的
$\mathbf{n}_i^\top$ 换成 $(I - \mathbf{d}_i\mathbf{d}_i^\top)$ 型投影——第 7 讲 D7-1
的 3D 完全体。`Icp3d::AlignP2Line / AlignP2Plane` 即 (8.5) 的 GN 实现。

### 8.4.4 D8-3：NDT（书 §7.2.3，`sad::Ndt3d`）

**① 问题场景**：ICP 系每轮都要**逐点找最近邻**（Kd 树查询占大半耗时），
且对应错误直接污染雅可比。
**② 解决方法**：把目标点云**体素化成高斯混合**——每个体素存 $(\boldsymbol\mu_v,
\Sigma_v)$（体内点的均值/协方差，BuildTarget 阶段一次性算好），匹配时不找
对应点，直接问"源点落在目标高斯里的似然多大"。
**③ 选型理由**：对比点到面 ICP——查询从"逐点 kNN"降为"逐点定位体素"
（哈希，第 6 讲 (6.2)）；概率模型天然吸收对应模糊。代价：体素化损失
亚体素分辨率（由邻域插值补救，见下）。

**⑤ 推导（目标函数与 GN）**。源点 $\mathbf{p}_i$ 经位姿变换落入体素 $v(i)$，
该点出现在目标的负对数似然（取该体素高斯的贡献；$\rho$ 为归一化常数）：

$$\phi_i = -\ln\Big(\rho\,\exp\big(-\tfrac12\,\mathbf{d}_i^\top\Sigma_{v(i)}^{-1}\mathbf{d}_i\big)\Big)
\;\doteq\; \tfrac12\,\mathbf{d}_i^\top\Sigma_{v(i)}^{-1}\mathbf{d}_i, \qquad
\mathbf{d}_i = T\mathbf{p}_i - \boldsymbol\mu_{v(i)} \tag{8.6}$$

总目标 $\phi = \sum_i \phi_i$。GN 残差取**马氏距离加权**的 3 维向量
$\bar{\mathbf{e}}_i = L_{v(i)}^{-1}\mathbf{d}_i$（$L$ 为 $\Sigma$ 的 Cholesky 因子，
依据：$\mathbf{d}^\top\Sigma^{-1}\mathbf{d} = \|L^{-1}\mathbf{d}\|^2$），雅可比把 (8.5)
左乘 $L^{-1}$：

$$\mathbf{J}_i = L_{v(i)}^{-1}\big[\,-(R\mathbf{p}_i)^\wedge,\ \ I\,\big] \tag{8.7}$$

**两个工程必备项（书实现与 PCL 的差异点）**：

- **高斯退化防护**：体素内点近共面时 $\Sigma$ 有近零特征值，(8.6) 沿该方向
  权重 $\to\infty$——数值不稳。防护：$L^{-1}$ 前对 $\Sigma$ 做最小特征值截断
  （`Options::min_eigenvalue`，等价于 $\Sigma \leftarrow \Sigma + \epsilon I$）。注意这与
  点到面 ICP 的哲学相反：ICP 靠法向**只留**法向信息，NDT 靠截断**丢弃**
  退化方向——同一个"降维"问题的两种处理；
- **邻域插值**（`Options::nearby_type`）：点落在体素边缘时，相邻体素的
  高斯同样有效——按点在体内分数坐标对 $v$ 与 6 邻域体素线性加权
  （PCL 同款），消除体素边界的目标函数跳变，GN 收敛明显更稳。

**与 PCL 对比（§7.2.4）**：`pcl::NormalDistributionsTransform` 与
`sad::Ndt3d` 同为体素高斯 + GN，书中自实现省去了 PCL 的多轮启发式，
在相同初值/体素下的配准精度相当、耗时更短（基准见 §8.5）；
`sad::Icp3d` 对 `pcl::IterativeClosestPoint` 结论相同。

## 8.5 实践任务

- **操作步骤，不含理论**（豁免声明）：`ch7/test/test_icp.cc`（仿真点云上跑
  三种 ICP）、NDT 单测；`gen_simu_data` 生成可控真值的仿真配准对。
- 收敛域实验（理论回引）：固定一对扫描，初值平移误差从 0.1 m 扫到 2.0 m，
  记录点对点 / 点对面 / NDT 的成功率——预期排序与第 7 讲同构
  （点对点最差、NDT 与点对面接近且更稳）；
- 退化实验：纯走廊数据（沿走廊方向无结构）跑 NDT，观察退化防护的
  必要性；再用 GNSS/IMU 先验补足该方向（第 9 讲松耦合的动机之一）。

## 8.6 易错点与高频疑问

1. **反射解**：(8.3) 的 $\mathrm{diag}(1,1,\det(VU^\top))$ 不是数学装饰——
   对应点质量差时 SVD 常给反射分支，漏检表现为"配准结果的行列式为 −1"
   或点云镜像；
2. **扰动约定错配**：(8.5) 按**左扰动**推导；若实现按右扰动（$T \leftarrow
   T\mathrm{Exp}(\delta\boldsymbol\xi^\wedge)$）雅可比变 $-\mathbf{n}^\top R\,\mathbf{p}_i^\wedge R^\top$
   型——照抄雅可比不核对约定，是配准"越优化越远"的头号原因；
   验收永远做数值差分；
3. **NDT 体素尺寸**：与结构尺度匹配（室外 1–4 m 常用）——过小：单体素点
   太少、$\Sigma$ 退化频发；过大：亚体素分辨率损失主导误差；
4. **协方差直接求逆**：(8.7) 用 Cholesky 而非显式逆（数值稳定）；
5. **以为 NDT 无初值要求**：它只是把"找对应"换成了"定位体素"——初值差
   到点落错体素时同样失败，多分辨率初值（第 7 讲金字塔思想）在 3D
   同样适用。

## 8.7 自检题

1. 完整重推 (8.2)(8.3)：从 (8.1) 出发，写出平移消元、迹合并、Procrustes
   三步的每步依据。
2. 构造 $\det(VU^\top) = -1$ 的最小算例（4 个对应点），验证不加反射修正时
   (8.3) 给出镜像解。
3. 推导点到**线** ICP 的 $1\times6$ 雅可比（提示：(8.5) + 投影阵）。
4. 由 (8.6) 推 (8.7)：写出 Cholesky 加权后的残差与雅可比定义式，
   并解释为什么马氏加权不改变 GN 的收敛点。
5. 解释"NDT 丢弃退化方向、点到面 ICP 只留法向"为何是同一件事的两面
   （提示：走廊沿向无约束方向是什么？两种方法各自把它排除到哪里？）
6. （数值）对 `ndt_3d.cc` 的雅可比做差分；再关掉邻域插值，对比目标函数
   曲面的平滑性与 GN 收敛步数。

## 8.8 本章要点

- 点到点 ICP 有闭式解 (8.3)——质心消元 + Procrustes + 反射检查；
  GN 迭代版与闭式解在同对应下等价；
- 点到面/线 (8.4)(8.5)：残差投影到强约束方向，收敛域与稳定性显著优于
  点对点；雅可比是 (2.8) 的直接应用（左扰动约定）；
- NDT (8.6)–(8.7)：目标体素高斯化，对应关系消失；退化防护与邻域插值
  是从"能跑"到"好用"的两步工程；
- 选型速记：局部地图配准/有法向结构 → 点到面；体素化建图管线、追求
  速度 → NDT；两者在第 9 讲分别成为直接法与松耦合 LIO 的观测引擎。

## 8.9 配套阅读

- 上一讲：[07_2D-SLAM.md](07_2D-SLAM.md)；下一讲：
  [09_3D里程计与松耦合LIO.md](09_3D里程计与松耦合LIO.md)（把配准串成里程计）；
- 十四讲对照：[../十四讲/07_视觉里程计1特征点法.md](../十四讲/07_视觉里程计1特征点法.md)
  §ICP（3D SVD 推导的另一版本）；
- 主题教程：[../06_视觉里程计-ii直接法.md](../06_视觉里程计-ii直接法.md)
  （"直接法"思想的光度版本——本讲 NDT 是其几何版本）；
- 代码：`src/ch7/icp_3d`、`ndt_3d`、`test/test_icp`；PCL 对照模块；
  `projects/slam/` 无 3D ICP 的 numpy 版（第 9 讲 `loam2d/` 提供 2D
  特征式的对照实现）。

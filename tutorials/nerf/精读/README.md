# NeRF 论文精读 — 总索引

本主题的论文库（[papers/nerf/latex/](../../../papers/nerf/latex/arxiv_submission.tex)）收录 NeRF 论文完整 LaTeX 源码，
对应精读一篇：

| 精读 | 论文（venue） | 一句话内核 | 教程 | 代码 |
|------|--------------|-----------|------|------|
| [NeRF](NeRF_ECCV2020.md) | NeRF（ECCV 2020） | 5D 辐射场 + 可微体渲染：新视角合成的隐式表示里程碑 | [十章教程](../README.md) | [projects/nerf](../../../projects/nerf/README.md) |

精读基于论文 **LaTeX 源码**逐式核对（主文恰 6 个编号公式），含：

- Eq.1 体渲染方程的四步物理推导（终止概率 → 透射率 ODE → 分离变量 → 期望）；
- Eq.2/3 分层采样与离散渲染（α_i、T_i 递推、权重和 ≤1 的裂项证明）；
- Eq.4 位置编码（谱偏置的"截断傅里叶基线性读出"表达力论证）；
- Eq.5 层次采样（逆变换采样三步推导 + 正确性证明）；
- **三方映射表**：论文式号 ↔ 教程十章式号 ↔ projects/nerf 函数名（全部核实）；
- 勘误注：训练损失为论文 **Eq. 6**（教程/METRICS 此前误写 Eq. 7，已同步修正）。

## 与其他主题精读的衔接

- **下游演进**：[Instant-NGP](../../3d_reconstruction/精读/InstantNGP_SIGGRAPH2022.md)（哈希提速）、
  [3DGS](../../3d_reconstruction/精读/3DGS_SIGGRAPH2023.md)（显式高斯替代）、
  [NeuS](../../3d_reconstruction/精读/NeuS_NeurIPS2021.md)（体渲染逼出几何）——
  见 [3D 重建精读索引](../../3d_reconstruction/精读/README.md)；
- **SLAM 落地**：[NeRF-SLAM](../../slam/精读/NeRF-SLAM_ICRA2023.md)——神经隐式建图；
- **代码实战**：[projects/nerf/TUTORIAL.md](../../../projects/nerf/TUTORIAL.md)（零基础导读）与
  [DEBUG.md](../../../projects/nerf/DEBUG.md)（调试变量表）。
